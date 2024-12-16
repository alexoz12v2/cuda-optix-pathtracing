module;

// NOLINTBEGIN
#include <array>
#include <string_view>
#include <type_traits>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

#if defined(DMT_DEBUG)
#include <backward.hpp>
#endif

#if defined(DMT_OS_LINUX)
#include <fcntl.h>
#include <linux/kernel-page-flags.h>
#include <sys/mman.h>
#include <unistd.h>

#define PAGE_INFO_IMPL
#include <page-info/page-info.h>

#include <cerrno>
#include <cstdlib>
#elif defined(DMT_OS_WINDOWS)
// Link against `VirtualAlloc2` Manually
// https://learn.microsoft.com/en-us/answers/questions/129563/how-to-use-virtualalloc2-memory-api
#pragma comment(lib, "mincore")
#include <AclAPI.h>
#include <Windows.h>
#include <securitybaseapi.h>
#include <sysinfoapi.h>
#endif
// NOLINTEND

module platform;

namespace dmt
{

#if defined(DMT_OS_LINUX)
inline constexpr uint32_t     howMany4KBsIn1GB = toUnderlying(EPageSize::e1GB) / toUnderlying(EPageSize::e4KB);
static thread_local page_info pageInfoPool[howMany4KBsIn1GB];
static thread_local uint64_t  bitsPool[howMany4KBsIn1GB];
#elif defined(DMT_OS_WINDOWS)
inline constexpr uint32_t                                           tokenPrivilegesBytes = 2048;
static thread_local std::array<unsigned char, tokenPrivilegesBytes> sTokenPrivileges;
using PVirtualAlloc = PVOID (*)(
    HANDLE                  Process,
    PVOID                   BaseAddress,
    SIZE_T                  Size,
    ULONG                   AllocationType,
    ULONG                   PageProtection,
    MEM_EXTENDED_PARAMETER* ExtendedParameters,
    ULONG                   ParameterCount);
static constexpr uint32_t                              sErrorBufferSize = 256;
static thread_local std::array<char, sErrorBufferSize> sErrorBuffer{};
#endif

#if defined(DMT_DEBUG)
void internStringToCurrent(char const* str, uint64_t sz)
{
    // TODO implement table switching mechanism
}
#endif

#if defined(DMT_OS_LINUX)
enum PageAllocationFlags : uint32_t
{
    DMT_OS_LINUX_MMAP_ALLOCATED          = 1 << 0, // Use bit 0
    DMT_OS_LINUX_ALIGNED_ALLOC_ALLOCATED = 1 << 1  // Use bit 1
};

// My own version from page-info which doesn't allocate
static page_info_array getInfoForFirstNInRange(void* start, void* end, page_info* pool, uint64_t* bitmapPool, size_t poolSize)
{
    unsigned psize     = get_page_size();
    char*    startPage = (char*)pagedown(start, psize);
    char*    endPage   = (char*)pagedown((char*)end - 1, psize) + psize;
    size_t   pageCount = start < end ? (endPage - startPage) / psize : 0;
    assert(pageCount == 0 || startPage < endPage);

    if (pageCount == 0)
    {
        return (page_info_array){0, NULL};
    }

    pageCount = std::min(pageCount, poolSize);

    // open the pagemap file
    FILE* pagemapFile = fopen("/proc/self/pagemap", "rb");
    if (!pagemapFile)
        err(EXIT_FAILURE, "failed to open pagemap");

    // seek to the first page
    if (fseek(pagemapFile, static_cast<long>((uintptr_t)startPage / psize * sizeof(uint64_t)), SEEK_SET))
        err(EXIT_FAILURE, "pagemap seek failed");

    assert(bitmapPool);
    size_t bitmapBytes = pageCount * sizeof(uint64_t);
    size_t readc       = fread(bitmapPool, bitmapBytes, 1, pagemapFile);
    if (readc != 1)
        err(EXIT_FAILURE, "unexpected fread(pagemap) return: %zu", readc);

    fclose(pagemapFile);

    FILE* kpageflags_file = NULL;
    enum
    {
        INIT,
        OPEN,
        FAILED
    } file_state = INIT;

    for (size_t page_idx = 0; page_idx < pageCount; page_idx++)
    {
        page_info info = extract_info(bitmapPool[page_idx]);

        if (info.pfn)
        {
            // we got a pfn, try to read /proc/kpageflags

            // open file if not open
            if (file_state == INIT)
            {
                kpageflags_file = fopen("/proc/kpageflags", "rb");
                if (!kpageflags_file)
                {
                    warn("failed to open kpageflags");
                    file_state = FAILED;
                }
                else
                {
                    file_state = OPEN;
                }
            }

            if (file_state == OPEN)
            {
                uint64_t bits;
                if (fseek(kpageflags_file, info.pfn * sizeof(bits), SEEK_SET))
                    err(EXIT_FAILURE, "kpageflags seek failed");
                if ((readc = fread(&bits, sizeof(bits), 1, kpageflags_file)) != 1)
                    err(EXIT_FAILURE, "unexpected fread(kpageflags) return: %zu", readc);
                info.kpageflags_ok = true;
                info.kpageflags    = bits;
            }
        }

        pool[page_idx] = info;
    }

    if (kpageflags_file)
        fclose(kpageflags_file);

    return {pageCount, pool};
}

PageAllocator::PageAllocator(PlatformContext& ctx)
{
    // Paths for huge pages information
    char const* hugePagesPaths[] = {"/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages",
                                    "/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages"};

    // Read huge pages information
    for (char const* path : hugePagesPaths)
    {
        FILE* file = fopen(path, "r");
        if (file)
        {
            int32_t num;
            if (fscanf(file, "%d", &num) == 1)
            {
                if (strstr(path, "2048kB"))
                    m_mmap2MBHugepages = num;
                else if (strstr(path, "1048576kB"))
                    m_mmap1GBHugepages = num;
            }
            fclose(file);
        }
    }

    FILE*   file         = fopen("/proc/meminfo", "r");
    int64_t hugepageSize = -1;
    if (file)
    {
        char line[256];

        // Read each line in /proc/meminfo
#define HUGE_PAGE_SIZE_KEY "Hugepagesize:"
        while (fgets(line, sizeof(line), file))
        {
            if (strncmp(line, HUGE_PAGE_SIZE_KEY, strlen(HUGE_PAGE_SIZE_KEY)) == 0)
            {
                // Extract the hugepage size from the line
                if (sscanf(line, HUGE_PAGE_SIZE_KEY " %ld kB", &hugepageSize) == 1)
                {
                    break;
                }
            }
        }
#undef HUGE_PAGE_SIZE_KEY

        fclose(file);

        if (hugepageSize != -1)
            hugepageSize *= 1024; // Convert from kB to bytes
    }

    // Check if transparent huge pages are enabled

    bool hugePagesEnabled = hugepageSize > 0 && (m_mmap2MBHugepages > 0 || m_mmap1GBHugepages > 0);
    bool use1GB           = hugepageSize == toUnderlying(EPageSize::e1GB);
    bool use2MB           = hugepageSize == toUnderlying(EPageSize::e2MB);
    if (hugePagesEnabled)
    {
        char                     buffer[256];
        static char const* const path           = "/sys/kernel/mm/transparent_hugepage/enabled";
        static char const* const sizePath       = "/sys/kernel/mm/transparent_hugepage/hpage_pmd_size";
        FILE*                    thpEnabledFile = fopen(path, "r");
        FILE*                    thpSizeFile    = fopen(sizePath, "r");
        uint32_t                 size           = 0;

        // get the enabled value
        if (thpEnabledFile && thpSizeFile && fgets(buffer, sizeof(buffer), thpEnabledFile))
        {
            if (strstr(buffer, "[always]") || strstr(buffer, "[madvise]"))
            {
                if (fscanf(thpSizeFile, "%u", &size) == 1)
                {
                    m_thpSize = size;
                }
            }
            else
            {
                ctx.warn("Transparent Huge Pages are not enabled, check \"{}\"", {path});
            }

            fclose(thpEnabledFile);
            fclose(thpSizeFile);
        }
    }

    if (hugePagesEnabled)
    {
        m_hugePageEnabled    = true;
        m_mmapHugeTlbEnabled = true;
        m_enabledPageSize    = use1GB ? EPageSize::e1GB : use2MB ? EPageSize::e2MB : EPageSize::e4KB;
        if (m_enabledPageSize != EPageSize::e4KB && toUnderlying(m_enabledPageSize) == m_thpSize)
        {
            m_thpEnabled = true;
        }
    }
}

PageAllocation PageAllocator::allocatePage(PlatformContext& ctx)
{
    PageAllocation ret{};
    uint32_t const size     = toUnderlying(EPageSize::e4KB);
    uint32_t const pageSize = toUnderlying(m_enabledPageSize); // used as alignment

    if (m_mmapHugeTlbEnabled)
    {
        static constexpr int32_t flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED | MAP_POPULATE | MAP_NONBLOCK |
                                         MAP_STACK | MAP_HUGETLB;
        auto* ptr = reinterpret_cast<unsigned char*>(mmap(nullptr, pageSize, PROT_READ | PROT_WRITE, flags, -1, 0));
        if (ptr != MAP_FAILED)
        {
            ptr[0] = 0; //touch the page to actually allocate it (since there's MAP_LOCKED and no MAP_NORESERVE, it shouldn't be necessary)
            ret.address = ptr;
            ret.count   = 1;
            ret.bits |= DMT_OS_LINUX_MMAP_ALLOCATED;
        }
    }

    if (!ret.address) // Try to use transparent huge pages
    {
        // disable huge zero page, which is the page used when you have a read fault
        // >> echo 0 >/sys/kernel/mm/transparent_hugepage/use_zero_page
        // https://www.kernel.org/doc/html/v5.1/admin-guide/mm/transhuge.html
        // 1. reserve, not effectively allocate, necessary virtual address space
        auto* buf = reinterpret_cast<unsigned char*>(aligned_alloc(pageSize, pageSize));
        if (!buf)
        {
            ctx.error("Could not allocate buffer! (check errno for more details)");
            ret.address = nullptr;
        }
        else
        {
            ret.address = buf;
            ret.bits |= DMT_OS_LINUX_ALIGNED_ALLOC_ALLOCATED;

            // 2. tag the memory as huge page (required for madvise, not for always)
            if (m_thpEnabled)
            {
                if (!madvise(buf, size, MADV_HUGEPAGE))
                {
                    // 3. for each page, touch the first byte to be sure to allocate memory, and, if debug,
                    // check that that's effectively a huge page
                    for (uint32_t i = 0; i < size; i += toUnderlying(m_enabledPageSize))
                    {
                        buf[i] = 0;
                    }
                }
            }
        }
    }

    if (ret.address)
    {
        page_info_array const pinfo    = getInfoForFirstNInRange(ret.address,
                                                              reinterpret_cast<unsigned char*>(ret.address) + size,
                                                              pageInfoPool,
                                                              bitsPool,
                                                              howMany4KBsIn1GB);
        flag_count const      thpCount = get_flag_count(pinfo, KPF_THP); // 0 if not root

        assert(pinfo.num_pages == 1);
        int64_t proposedFrameNum = pinfo.info ? static_cast<int64_t>(pinfo.info[0].pfn) : 0;
        ret.pageNum              = proposedFrameNum != 0 ? proposedFrameNum : -1;
        ret.pageSize             = pinfo.num_pages == 1 ? m_enabledPageSize : EPageSize::e4KB;
        ret.count                = 1;
    }

    // TODO print if allocation failed
    // TODO refactor common bits of functionality/logging among operating systems
#if defined(DMT_DEBUG)
    ctx.trace(
        "Called allocatePage, allocated "
        "at {} page of {} Printing Stacktrace",
        {ret.address, (void*)toUnderlying(ret.pageSize)});
    if (ctx.traceEnabled())
    {
        backward::Printer    p;
        backward::StackTrace st;
        st.load_here();
        p.print(st);
    }
#else
    ctx.trace(
        "Called allocatePage, allocated "
        "at {} page of {}",
        {ret.address, (void*)toUnderlying(ret.pageSize)});
#endif
    return ret;
}

void PageAllocator::deallocatePage(PlatformContext& ctx, PageAllocation& alloc)
{
    if ((alloc.bits & DMT_OS_LINUX_MMAP_ALLOCATED) != 0)
    {
        if (munmap(alloc.address, toUnderlying(alloc.pageSize)))
        {
#if defined(DMT_DEBUG)
            backward::Printer    p;
            backward::StackTrace st;
            st.load_here();
            ctx.error("Couldn't deallocate {}, printing stacktrace:", {alloc.address});
            p.print(st);
#else
            ctx.error("Couldn't deallocate {}", {alloc.address});
#endif
        }
        alloc.address = nullptr;
    }
    else if ((alloc.bits & DMT_OS_LINUX_ALIGNED_ALLOC_ALLOCATED) != 0)
    {
        free(alloc.address);
        alloc.address = nullptr;
    }
    else // TODO add unreachable or something
    {
#if defined(DMT_DEBUG)
        ctx.trace("You called deallocatePage, but nothing done. Printing Stacktrace");
        if (ctx.traceEnabled())
        {
            backward::Printer    p;
            backward::StackTrace st;
            st.load_here();
            p.print(st);
        }
#else
        ctx.trace("You called deallocatePage, but nothing done.");
#endif
    }
#if defined(DMT_DEBUG)
    ctx.trace("Deallocated memory at {} size {}, Printing stacktrace", {alloc.address, toUnderlying(alloc.pageSize)});
    if (ctx.traceEnabled())
    {
        backward::Printer    p;
        backward::StackTrace st;
        st.load_here();
        p.print(st);
    }
#else
    ctx.trace("Deallocated memory at {} size {}", {alloc.address, toUnderlying(alloc.pageSize)});
#endif
}

PageAllocator::~PageAllocator()
{
}

#elif defined(DMT_OS_WINDOWS)

// TODO refactor in utils
//Returns the last Win32 error, in string format. Returns an empty string if there is no error.
static uint32_t getLastErrorAsString(char* buffer, uint32_t maxSize)
{
    //Get the error message ID, if any.
    DWORD errorMessageID = ::GetLastError();
    if (errorMessageID == 0)
    {
        buffer[0] = '\n';
        return 0;
    }
    else
    {

        LPSTR messageBuffer = nullptr;

        //Ask Win32 to give us the string version of that message ID.
        //The parameters we pass in, tell Win32 to create the buffer that holds the message for us (because we don't yet know how long the message string will be).
        size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                     NULL,
                                     errorMessageID,
                                     MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                     (LPSTR)&messageBuffer,
                                     0,
                                     NULL);

        //Copy the error message into a std::string.
        size_t actual = std::min(static_cast<size_t>(maxSize - 1), size);
        std::memcpy(buffer, messageBuffer, actual);
        buffer[actual] = '\0';

        //Free the Win32's string's buffer.
        LocalFree(messageBuffer);
        return actual;
    }
}

// TODO error handing with the Janitor Pattern
// the Process Security Descriptor, the Token -> Close
PageAllocator::PageAllocator(PlatformContext& ctx)
{
    class Janitor
    {
    public:
        Janitor()                              = default;
        Janitor(Janitor const&)                = delete;
        Janitor(Janitor&&) noexcept            = delete;
        Janitor& operator=(Janitor const&)     = delete;
        Janitor& operator=(Janitor&&) noexcept = delete;
        ~Janitor() noexcept
        {
            if (hProcessToken)
                CloseHandle(hProcessToken);
            if (mallocatedMem)
                std::free(mallocatedMem);
            if (pSecDescriptor)
                LocalFree(pSecDescriptor);
        }

        HANDLE               hProcessToken  = nullptr;
        void*                mallocatedMem  = nullptr;
        PSECURITY_DESCRIPTOR pSecDescriptor = nullptr;
    };
    constexpr auto luidComparator = [](LUID const& luid0, LUID const& luid1) -> bool { //
        return luid0.HighPart == luid1.HighPart && luid1.LowPart == luid0.LowPart;
    };

    Janitor janitor;

    // Get some of the system information relevant to memory allocation, eg
    // - when `VirtualAlloc` is called with MEM_RESERVE, the allocation is aligned to the `allocation granularity`
    // - when `VirtualAlloc` it called with MEM_COMMIT, the allocation is aligned to a page boundary
    SYSTEM_INFO sysInfo{};
    GetSystemInfo(&sysInfo);
    m_systemPageSize        = sysInfo.dwPageSize;
    m_allocationGranularity = sysInfo.dwAllocationGranularity;
    assert(m_systemPageSize == toUnderlying(EPageSize::e4KB));

    // Retrieve the LUID associated with the SE_LOCK_MEMORY_NAME = "SeLockMemoryPrivilege"
    LUID seLockMemoryPrivilegeLUID{};
    if (!LookupPrivilegeValue(nullptr /*on the local system*/, SE_LOCK_MEMORY_NAME, &seLockMemoryPrivilegeLUID))
    {
        uint32_t         length = getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        std::string_view view{sErrorBuffer.data(), length};
        ctx.error("Could not retrieve the LUID for the SeLockMemoryPrivilege. Error: {}", {view});
        return;
    }

    // get the pseudo handle (fixed to -1) of the current process (you call a
    // function anyways for compatibility with the future)
    // pseudo handles need not to be `Closehandle`d
    HANDLE hCurrentProc = GetCurrentProcess();

    // Retrieve the user access token associated to the user of the current process.
    // Open it in DesiredAccessMode = TOKEN_ADJUST_PRIVILEDGES, NOT TOKEN_QUERY, such that, if we need
    // we can add some access control entries into it
    // see https://learn.microsoft.com/en-us/windows/win32/secauthz/access-rights-for-access-token-objects
    HANDLE hProcessToken   = nullptr;
    DWORD  tokenAccessMode = TOKEN_QUERY | TOKEN_ADJUST_PRIVILEGES;
    if (!OpenProcessToken(hCurrentProc, tokenAccessMode, &hProcessToken))
    {
        uint32_t         length = getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        std::string_view view{sErrorBuffer.data(), length};
        ctx.error(
            "Couldn't open in TOKEN_ADJUST_PRIVILEDGES "
            "mode the user access token. Error: {}",
            {view});
        // TODO error handling
    }
    janitor.hProcessToken = hProcessToken;

    // iterate over the existing priviledges on the user token, and if you find SE_LOCK_MEMORY_NAME with
    // attribute SE_PRIVILEGE_ENABLED, then you are good to go
    // 1. Get count of bytes for the TokenPrivileges TOKEN_INFORMATION_CLASS for this token handle
    DWORD requiredSize = 0;
    if (!GetTokenInformation(hProcessToken, TokenPrivileges, nullptr, 0, &requiredSize) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER)
    {
        uint32_t         length = getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        std::string_view view{sErrorBuffer.data(), length};
        ctx.error(
            "Could not get the required size {} for the "
            "`TokenPrivileges` Token Information, error: {}",
            {requiredSize, view});
        return;
    }

    // hoping we fit into the statically allocated buffer
    // note: we need to account for an additional LUID_AND_ATTRIBUTE in case the SeLockMemoryPrivilege
    // is not found. `requiredSize + sizeof(LUID_AND_ATTRIBUTES)` should respect the alignment
    // of LUID_AND_ATTRIBUTES because requiredSize is a (multiple of it + DWORD)
    static_assert(alignof(LUID_AND_ATTRIBUTES) == 4);
    void* pTokenPrivilegesInformation = nullptr;
    bool  mallocated                  = requiredSize > tokenPrivilegesBytes - sizeof(LUID_AND_ATTRIBUTES);
    if (mallocated)
    {
        ctx.warn("Allocating on the Heap token information");
        pTokenPrivilegesInformation = std::malloc(requiredSize + sizeof(LUID_AND_ATTRIBUTES));
        if (pTokenPrivilegesInformation)
        {
            janitor.mallocatedMem = pTokenPrivilegesInformation;
        }
    }
    else
    {
        pTokenPrivilegesInformation = sTokenPrivileges.data();
    }

    if (!pTokenPrivilegesInformation)
    {
        ctx.error("Couldn't reserve memory to hold the token information");
        return;
    }

    // 2. actually get the TokenPrivileges TOKEN_INFORMATION_CLASS
    if (!GetTokenInformation(hProcessToken, TokenPrivileges, pTokenPrivilegesInformation, requiredSize, &requiredSize))
    {
        uint32_t         length = getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        std::string_view view{sErrorBuffer.data(), length};
        ctx.error("Could not get the `TokenPrivileges` Token Information, error: {}", {view});
        return;
    }

    // 3. Iterate over the list of TOKEN_PRIVILEGES and find the one with the SeLockMemoryPrivilege LUID
    TOKEN_PRIVILEGES& tokenPrivilegeStruct = *reinterpret_cast<TOKEN_PRIVILEGES*>(&sTokenPrivileges[0]);
    auto*             pPrivilegeLUIDs      = reinterpret_cast<LUID_AND_ATTRIBUTES*>(
        sTokenPrivileges.data() + offsetof(TOKEN_PRIVILEGES, Privileges));
    bool    seLockMemoryPrivilegeEnabled = false;
    int64_t seLockMemoryPrivilegeIndex   = -1;
    for (uint32_t i = 0; i < tokenPrivilegeStruct.PrivilegeCount; ++i)
    {
        LUID  luid       = pPrivilegeLUIDs[i].Luid;
        DWORD attributes = pPrivilegeLUIDs[i].Attributes;
        if (luidComparator(luid, seLockMemoryPrivilegeLUID))
        {
            // possible attributes: E_PRIVILEGE_ENABLED_BY_DEFAULT, SE_PRIVILEGE_ENABLED,
            // SE_PRIVILEGE_REMOVED, SE_PRIVILEGE_USED_FOR_ACCESS
            if ((attributes & SE_PRIVILEGE_ENABLED) != 0)
                seLockMemoryPrivilegeEnabled = true;

            seLockMemoryPrivilegeIndex = i;
            break;
        }
    }

    // If the SeLockMemoryPrivilege is not enabled, then try to enable it
    if (!seLockMemoryPrivilegeEnabled)
    {
        // write into a new entry if we didn't find it at all
        // we are basically preparing the `NewState` parameter for the `AdjustTokenPrivileges`
        if (seLockMemoryPrivilegeIndex < 0)
        {
            // also HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management registry can
            // tell if you disabled it
            ctx.error(
                "SeLockMemoryPrivilege is absent in the current user token. "
                "Try to run as admin, check `secpol.msc` or use cudaAllocHost");
        }
        else
        {
            TOKEN_PRIVILEGES privs; // Assuming ANYSIZE_ARRAY = 1
            privs.PrivilegeCount           = 1;
            privs.Privileges[0].Luid       = seLockMemoryPrivilegeLUID;
            privs.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
            if (!AdjustTokenPrivileges(hProcessToken, false, &privs, sizeof(TOKEN_PRIVILEGES), nullptr, nullptr))
            {
                uint32_t         length = getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
                std::string_view view{sErrorBuffer.data(), length};
                ctx.error("Could not add SeLockMemoryPrivilege to the user token, error: {}", {view});
                return;
            }

            int32_t errCode = GetLastError();
            if (errCode == ERROR_NOT_ALL_ASSIGNED)
            {
                // Sanity check
                PRIVILEGE_SET privilegeSet;
                privilegeSet.PrivilegeCount          = 1;
                privilegeSet.Control                 = PRIVILEGE_SET_ALL_NECESSARY;
                privilegeSet.Privilege[0].Luid       = seLockMemoryPrivilegeLUID;
                privilegeSet.Privilege[0].Attributes = SE_PRIVILEGE_ENABLED;
                BOOL result                          = false;
                if (!PrivilegeCheck(hProcessToken, &privilegeSet, &result))
                {
                    uint32_t         length = getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
                    std::string_view view{sErrorBuffer.data(), length};
                    ctx.error("Call to `PrivilegeCheck` failed, error: {}", {view});
                    return;
                }
                if (!result)
                {
                    ctx.error(
                        "Even though you called `AdjustTokenPrivileges`, the SeLockMemoryPrivilege is still nowhere to "
                        "be found");
                    return;
                }

                seLockMemoryPrivilegeEnabled = true;
            }
            else if (errCode == ERROR_SUCCESS)
            {
                seLockMemoryPrivilegeEnabled = true;
            }
            else 
            {
                ctx.error("`AdjustTokenPrivilege` returned an unexpected error code: {}", {errCode});
            }
        }
    }

    // still not enabled? Fail.
    if (!seLockMemoryPrivilegeEnabled)
    {
        ctx.error("Found SeLockMemoryPrivilege not enabled, hence no large page allocation");
        return;
    }

    // Phase 2: Retrieve the minimum large page the processor supports. if 0, then the processor doesn't support them
    // since we are on x86_64, it should always be different than 0
    size_t minimumPageSize = GetLargePageMinimum();
    if (minimumPageSize == 0)
    {
        ctx.error("For some reason, the current CPU architecture doesn't support large TLB entries");
        return;
    }

    // The size and alignment must be a multiple of the large-page minimum
    if (static_cast<size_t>(toUnderlying(EPageSize::e2MB)) % minimumPageSize != 0)
    {
        ctx.error("Page Size we support (2MB) is not a multiple of the MinimumPageSize, {} B", {minimumPageSize});
        return;
    }

    // ------------ ENABLE LARGE PAGES ------------------------------------------------------------------
    // TODO check miminum page size better
    m_largePageEnabled = true;

    // At this point, we know that we can use Large Page Allocation. We need to decide among 2MB and 1GB
    // - 2MB is allowed by default
    // - 1GB requires 1) support for `VirtualAlloc2`, 2) `PROCESS_VM_OPERATION` Access Right
    // 1. Check whether we support VirtualAlloc2 (Windows 10 and above)
    // to seach for a module specifically, take a look to AddDllDirectory. Usually you can use this
    // only on `LoadLibrary` DLLs. If you didn't allocate the library, don't free it (hence why it is called "Get")
    HMODULE hKernel32Dll = GetModuleHandleA("kernel32.dll");
    if (!hKernel32Dll)
    {
        uint32_t         length = getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        std::string_view view{sErrorBuffer.data(), length};
        ctx.error("Could not load the Kernel32.dll into an HMODULE, error: {}", {view});
        return;
    }

    auto* functionPointer = reinterpret_cast<PVirtualAlloc>(GetProcAddress(hKernel32Dll, "VirtualAlloc2"));
    if (!functionPointer)
    {
        uint32_t         length = getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        std::string_view view{sErrorBuffer.data(), length};
        ctx.error("Could not load the `VirtualAlloc2` from Kernel32.dll, using VirtualAlloc, error: {}", {view});
        return;
    }

    // 2. Look into the DACL of the current process to see whether you have the `PROCESS_VM_OPERATION` access right
    // Docs: https://learn.microsoft.com/en-us/windows/win32/procthread/process-security-and-access-rights
    // 2.1 First retrieve the Process Security Descriptor (to then free With LocalFree)
    PSECURITY_DESCRIPTOR securityDescriptor = nullptr;
    DWORD                status             = GetSecurityInfo(hCurrentProc, // the current process HANDLE
                                   SE_KERNEL_OBJECT, // a process is classified as a kernel object
                                   DACL_SECURITY_INFORMATION, // bits of the info to retrieve. we want process specific (discretionary)
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   &securityDescriptor);
    if (status != ERROR_SUCCESS)
    {
        ctx.error("Could not retrieve the Process Security Descriptor, error code {}", {status});
        return;
    }

    janitor.pSecDescriptor = securityDescriptor;

    // GENERIC_MAPPING = r?w?x?. Each member is an int, ACCESS_MASK, https://learn.microsoft.com/en-us/windows/win32/secauthz/access-mask
    GENERIC_MAPPING genericMapping = {
        PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, // GENERIC_READ (required for QueryWorkingSet)
        PROCESS_VM_WRITE | PROCESS_VM_OPERATION,     // GENERIC_WRITE
        PROCESS_CREATE_THREAD,                       // GENERIC_EXECUTE
        PROCESS_ALL_ACCESS                           // GENERIC_ALL
    };
    ACCESS_MASK outAccessMask = 0;

    std::remove_cvref_t<decltype(*std::declval<LPBOOL>())> bAccessStatus = false;

    // `AccessCheck` function to see whether the process security descriptor has a predefined
    // set of access rights
    if (!AccessCheck(securityDescriptor, // security descriptor against which access is checked
                     hProcessToken,      // impersonation token representing the user attempting the access
                     PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION, // desired access rights
                     &genericMapping,
                     nullptr,
                     0,
                     &outAccessMask,
                     &bAccessStatus))
    {
        uint32_t         length = getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        std::string_view view{sErrorBuffer.data(), length};
        ctx.error("Failed call to AccessCheck, error: {}", {view});
        return;
    }

    if (!bAccessStatus)
    {
        ctx.error("The Process doesn't own the PROCESS_VM_OPERATION access rights, using 2MB large pages");
        return;
    }

    m_largePage1GB = true;
}

enum PageAllocationFlags : uint32_t
{
    DMT_OS_WINDOWS_SMALL_PAGE    = 0,
    DMT_OS_WINDOWS_LARGE_PAGE    = 1 << 0, // Use bit 0
    DMT_OS_WINDOWS_VIRTUALALLOC2 = 1 << 1  // Use bit 1
};

// NOTE: Only Pinned memory can be mapped to CUDA device memory, hence you need to
// check that the page size is not 4KB. Mapping is carried out with cudaHostRegister
// reserve = take virtual address space. commit = when you write to it, it will be backed by physical memory
// TODO see Address Windowing Extension pages (AWE)
PageAllocation PageAllocator::allocatePage(PlatformContext& ctx)
{
    static constexpr uint32_t log4KB = 12u;
    PageAllocation            ret{};
    ret.pageNum               = -1;
    uint32_t  errorLength     = 0;
    DWORD     allocationFlags = MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES;
    DWORD     protectionFlags = PAGE_READWRITE;
    EPageSize pageSize        = EPageSize::e1GB;

    if (m_largePageEnabled)
    {
        if (m_largePage1GB)
        {
            // necessary additional parameter to perform 1GB allocation attempt
            MEM_EXTENDED_PARAMETER extended{};
            extended.Type    = MemExtendedParameterAttributeFlags;
            extended.ULong64 = MEM_EXTENDED_PARAMETER_NONPAGED_HUGE;

            ret.address = VirtualAlloc2( //
                nullptr /*current process*/,
                nullptr /*no base hint*/,
                toUnderlying(pageSize),
                allocationFlags,
                protectionFlags,
                &extended,
                1);
            ret.bits    = DMT_OS_WINDOWS_VIRTUALALLOC2;
            if (!ret.address)
            {
                errorLength = getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
                std::string_view view{sErrorBuffer.data(), errorLength};
                ctx.error("Failed call to `VirtualAlloc2`, trying `VirtualAlloc`, error: {}", {view});
            }
        }

        // If you are allowed to perform 1GB large page allocation, but failed for some
        // reason, fall back to 2MB allocation
        if (!ret.address)
        { // https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc
            pageSize = EPageSize::e2MB;

            // call VirtualAlloc. How is this memory aligned? See this link
            // https://stackoverflow.com/questions/20023446/is-virtualalloc-alignment-consistent-with-size-of-allocation
            ret.address = VirtualAlloc(nullptr, toUnderlying(pageSize), allocationFlags, protectionFlags);
            ret.bits    = DMT_OS_WINDOWS_LARGE_PAGE;
            if (!ret.address)
            {
                errorLength = getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
                std::string_view view{sErrorBuffer.data(), errorLength};
                ctx.error("`VirtualAlloc` with MEM_LARGE_PAGES, trying without it, error: {}", {view});
            }
        }
    }
    else
    {
        allocationFlags &= ~MEM_LARGE_PAGES;
        pageSize    = EPageSize::e4KB;
        ret.address = VirtualAlloc(nullptr, toUnderlying(pageSize), allocationFlags, protectionFlags);
        ret.bits    = DMT_OS_WINDOWS_SMALL_PAGE;
        if (!ret.address)
        {
            errorLength = getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
            std::string_view view{sErrorBuffer.data(), errorLength};
            ctx.error("`VirtualAlloc` failed, couldn't allocate memory, error: {}", {view});
        }
    }

    if (ret.address)
    {
        // touch first byte to make sure committed memory is backed to physical memory
        reinterpret_cast<unsigned char*>(ret.address)[0] = 0;

        // bookkeeping
        ret.count             = 1;
        ret.pageSize          = pageSize;
        bool isLargePageAlloc = (allocationFlags & MEM_LARGE_PAGES) != 0;

        // if you allocated the page with MEM_LARGE_PAGES, then it is locked to memory, and hence, like AWE,
        // is NOT part of the working set of the process.
        // Therefore you can get the physical frame nubmber with `QueryWorkingSetEx`
        // if instead you didn't allocate with MEM_LARGE_PAGES, then the allocated block is part of
        // the working set, and therefore you can use `QueryWorkingSet`
        // (Requires PROCESS_QUERY_INFORMATION and PROCESS_VM_READ access right to the process)
        MEMORY_BASIC_INFORMATION memoryInfo{};
        size_t                   memoryInfoBytes = ret.count * sizeof(MEMORY_BASIC_INFORMATION);

        // VirtualAddress = input, VirtualAttributes = output
        PSAPI_WORKING_SET_EX_INFORMATION input{};
        input.VirtualAddress = ret.address;
        uint32_t inputSize   = sizeof(PSAPI_WORKING_SET_EX_INFORMATION);
        if (!QueryWorkingSetEx(GetCurrentProcess(), &input, inputSize))
        {
            errorLength = getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
            std::string_view view{sErrorBuffer.data(), errorLength};
            ctx.error("Call to `QueryWorkingSetEx` failed, hence cannot check page status, error: {}", {view});
        }
        else if (!input.VirtualAttributes.Valid)
        {
            ctx.error("Call to `QueryWorkingSetEx` succeeded, but for some reason attributes are invalid, {}",
                      {input.VirtualAttributes.Flags});
        }
        else
        {
            // POSSIBLE TODO = use `VirtualLock` to lock a page even if not large
            bool effectivelyLarge = input.VirtualAttributes.LargePage;
            bool locked           = input.VirtualAttributes.Locked;
            if (isLargePageAlloc != effectivelyLarge)
            {
                ctx.error("Allocation should be large? {}. But found value {}", {isLargePageAlloc, effectivelyLarge});
            }

            if (isLargePageAlloc && !locked)
            {
                ctx.error("Allocation should be large, but its not locked");
            }
        }

        size_t numBytes = VirtualQuery(ret.address, &memoryInfo, memoryInfoBytes);
        if (numBytes == 0)
        {
            errorLength = getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
            std::string_view view{sErrorBuffer.data(), errorLength};
            ctx.error("Call to `VirtualQuery` failed, hence cannot acquire virtual page number, error: {}", {view});
        }
        else
        {
            size_t   expectedSize = toUnderlying(ret.pageSize);
            void*    pRegion      = memoryInfo.BaseAddress;
            uint64_t region       = (uint64_t)pRegion;
            uint64_t mask         = toUnderlying(EPageSize::e4KB) - 1;
            if (expectedSize != memoryInfo.RegionSize)
            {
                ctx.error("Expected region at {} size to be {} B but found {} B",
                          {pRegion, expectedSize, memoryInfo.RegionSize});
            }

            if ((memoryInfo.State & MEM_COMMIT) == 0)
            {
                ctx.error("Expected memory region at {} to be committed, but it's not", {pRegion});
            }

            if ((region & mask) != 0)
            {
                ctx.error("Expected memory region at {} to be aligned to a 4KB boundary. It's not.", {pRegion});
            }
            else
            {
                ret.pageNum = region >> log4KB;
            }
        }

#if defined(DMT_DEBUG)
        ctx.trace(
            "Called allocatePage, allocated "
            "at {} page of {} B. Printing Stacktrace",
            {ret.address, (void*)toUnderlying(ret.pageSize)});
        if (ctx.traceEnabled())
        {
            backward::Printer    p;
            backward::StackTrace st;
            st.load_here();
            p.print(st);
        }
#else
        ctx.trace(
            "Called allocatePage, allocated "
            "at {} page of {}",
            {ret.address, (void*)toUnderlying(ret.pageSize)});
#endif
    }
    else
    {
#if defined(DMT_DEBUG)
        ctx.error("Printing stacktrace");
        if (ctx.errorEnabled())
        {
            backward::Printer    p;
            backward::StackTrace st;
            st.load_here();
            p.print(st);
        }
#endif
    }

    return ret;
}

void PageAllocator::deallocatePage(PlatformContext& ctx, PageAllocation& alloc)
{
    // decommit = get out of memory, but leave the virtual address space reserved
    // POSSIBLE TODO. The other type, release, will decommit + free up the virtual address space
    DWORD freeType = MEM_RELEASE;
    if (!VirtualFree(alloc.address, 0 /*size 0 if MEM_RELEASE*/, freeType))
    {
        uint32_t         errorLength = getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        std::string_view view{sErrorBuffer.data(), errorLength};
#if defined(DMT_DEBUG)
        ctx.error("Failed to Free memory at {}, error {}, Printing Stacktrace", {alloc.address, view});
        if (ctx.errorEnabled())
        {
            backward::Printer    p;
            backward::StackTrace st;
            st.load_here();
            p.print(st);
        }
#else
#endif
    }
    else
    {
#if defined(DMT_DEBUG)
        ctx.trace("Deallocated memory at {} size {}, Printing stacktrace", {alloc.address, toUnderlying(alloc.pageSize)});
        if (ctx.traceEnabled())
        {
            backward::Printer    p;
            backward::StackTrace st;
            st.load_here();
            p.print(st);
        }
#else
        ctx.trace("Deallocated memory at {} size {}", {alloc.address, toUnderlying(alloc.pageSize)});
#endif
    }

    alloc.address = nullptr;
}

PageAllocator::~PageAllocator()
{
}

#endif

} // namespace dmt
