module;

// NOLINTBEGIN
#include <array>
#include <bit>
#include <numeric>
#include <string_view>
#include <thread>
#include <type_traits>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

#if defined(DMT_DEBUG)
#include <backward.hpp> // TODO incapsulare nel logger
#include <map>
#include <memory_resource>
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

// StringTable --------------------------------------------------------------------------------------------------------
#if defined(DMT_DEBUG)

// m_pool represents the Debug Memory. If there is a need to store something else
// inside debug memory other than the string table, refactor this class
class StringTable
{
public:
    static constexpr uint32_t MAX_SID_LEN = 256;

    StringTable() : m_pool(opts)
    {
        std::construct_at<decltype(U::stringTable)>(&u.stringTable, &m_pool);
    }
    StringTable(StringTable const&)                = delete;
    StringTable(StringTable&&) noexcept            = delete;
    StringTable& operator=(StringTable const&)     = delete;
    StringTable& operator=(StringTable&&) noexcept = delete;
    ~StringTable() noexcept
    {
        std::destroy_at<decltype(U::stringTable)>(&u.stringTable);
    }

    void intern(sid_t sid, char const* str, uint64_t sz);

    union U
    {
        // clang-format off
        U() { }
        ~U() { }
        // clang-format on
        std::pmr::map<sid_t, std::array<char, MAX_SID_LEN>> stringTable;
    };
    U u;

private:
    static inline std::pmr::pool_options const opts{
        .max_blocks_per_chunk        = 8,
        .largest_required_pool_block = 256,
    };
    std::pmr::synchronized_pool_resource m_pool;
    std::mutex                           m_mtx;
};

static StringTable s_stringTable;

void StringTable::intern(sid_t sid, char const* str, uint64_t sz)
{
    assert(sz < MAX_SID_LEN);
    std::lock_guard lock{m_mtx};

    auto const& [it, wasInserted] = s_stringTable.u.stringTable.try_emplace(sid);
    if (wasInserted)
    {
        std::memcpy(it->second.data(), str, sz);
        it->second[sz] = '\0';
    }
}

void internStringToCurrent(sid_t sid, char const* str, uint64_t sz)
{
    s_stringTable.intern(sid, str, sz);
}
#endif

std::string_view lookupInternedStr(sid_t sid)
{
    using namespace std::string_view_literals;
    static std::string_view empty = "NOT FOUND"sv;
#if defined(DMT_DEBUG)
    auto it = s_stringTable.u.stringTable.find(sid);
    if (it != s_stringTable.u.stringTable.cend())
    {
        std::string_view str{it->second.data()};
        return str;
    }
#endif

    return empty;
}

// PageAllocator ------------------------------------------------------------------------------------------------------

class HooksJanitor
{
public:
    HooksJanitor(PlatformContext& ctx, bool alloc) : m_ctx(ctx), m_alloc(alloc)
    {
    }
    HooksJanitor(HooksJanitor const&)                = delete;
    HooksJanitor(HooksJanitor&&) noexcept            = delete;
    HooksJanitor& operator=(HooksJanitor const&)     = delete;
    HooksJanitor& operator=(HooksJanitor&&) noexcept = delete;
    ~HooksJanitor() noexcept
    {
        if (pHooks && pAlloc)
        {
            if (m_alloc)
            {
                pHooks->allocHook(pHooks->data, m_ctx, *pAlloc);
            }
            else
            {
                pHooks->freeHook(pHooks->data, m_ctx, *pAlloc);
            }
        }
    }

    PageAllocatorHooks* pHooks = nullptr;
    PageAllocation*     pAlloc = nullptr;

private:
    PlatformContext& m_ctx;
    bool             m_alloc;
};

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

PageAllocator::PageAllocator(PlatformContext& ctx, PageAllocatorHooks const& hooks) : PageAllocator(hooks)
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

PageAllocation PageAllocator::allocatePage(PlatformContext& ctx, EPageSize sizeOverride)
{
    PageAllocation ret{};
    uint32_t const size = toUnderlying(EPageSize::e4KB);
    uint32_t const pageSize = std::min(toUnderlying(m_enabledPageSize), toUnderlying(sizeOverride)); // used as alignment

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
    PageAllocation allocCopy = alloc;
    if ((allocCopy.bits & DMT_OS_LINUX_MMAP_ALLOCATED) != 0)
    {
        if (munmap(allocCopy.address, toUnderlying(allocCopy.pageSize) * allocCopy.count))
        {
#if defined(DMT_DEBUG)
            backward::Printer    p;
            backward::StackTrace st;
            st.load_here();
            ctx.error("Couldn't deallocate {}, printing stacktrace:", {allocCopy.address});
            p.print(st);
#else
            ctx.error("Couldn't deallocate {}", {allocCopy.address});
#endif
        }
        allocCopy.address = nullptr;
    }
    else if ((allocCopy.bits & DMT_OS_LINUX_ALIGNED_ALLOC_ALLOCATED) != 0)
    {
        free(allocCopy.address);
        allocCopy.address = nullptr;
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
    ctx.trace("Deallocated memory at {} size {}, Printing stacktrace",
              {allocCopy.address, toUnderlying(allocCopy.pageSize)});
    if (ctx.traceEnabled())
    {
        backward::Printer    p;
        backward::StackTrace st;
        st.load_here();
        p.print(st);
    }
#else
    ctx.trace("Deallocated memory at {} size {}", {allocCopy.address, toUnderlying(allocCopy.pageSize)});
#endif
}

bool PageAllocator::allocate2MB(PlatformContext& ctx, PageAllocation& out)
{
    size_t const size        = toUnderlying(EPageSize::e2MB);
    bool         isLargePage = false;

    out.address = nullptr;

    if (m_mmapHugeTlbEnabled)
    {
        static constexpr int32_t hugeFlags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED | MAP_POPULATE | MAP_NONBLOCK |
                                             MAP_HUGETLB;
        out.address = mmap(nullptr, size, PROT_READ | PROT_WRITE, hugeFlags, -1, 0);
        if (out.address == MAP_FAILED)
        {
            out.address = nullptr;
            ctx.error("`mmap` with MAP_HUGETLB failed, trying without it, error: {}", {strerror(errno)});
        }
        else
        {
            isLargePage  = true;
            out.count    = 1;
            out.pageSize = EPageSize::e2MB;
            out.bits     = DMT_OS_LINUX_MMAP_ALLOCATED | DMT_OS_LINUX_HUGETLB;
        }
    }

    if (!out.address)
    {
        // Fallback to standard pages
        static constexpr int32_t normalFlags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED | MAP_POPULATE;
        out.address                          = mmap(nullptr, size, PROT_READ | PROT_WRITE, normalFlags, -1, 0);
        if (out.address == MAP_FAILED)
        {
            out.address = nullptr;
            ctx.error("`mmap` failed, error: {}", {strerror(errno)});
            return false;
        }
        else
        {
            out.count    = size / toUnderlying(EPageSize::e4KB);
            out.pageSize = EPageSize::e4KB;
            out.bits     = DMT_OS_LINUX_MMAP_ALLOCATED | DMT_OS_LINUX_NORMAL_PAGE;
        }
    }

    if (out.address)
    {
        // Touch memory to ensure allocation
        unsigned char* ptr = static_cast<unsigned char*>(out.address);
        for (size_t i = 0; i < size; i += toUnderlying(out.pageSize))
        {
            ptr[i] = 0;
        }
    }

    // TODO
    // addAllocInfo(ctx, isLargePage, out);

    m_hooks.allocHook(m_hooks.data, ctx, ret);
    return true;
}
PageAllocator::~PageAllocator()
{
}

#elif defined(DMT_OS_WINDOWS)
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
        if (bRevertToSelf)
            RevertToSelf();
        if (hImpersonationToken)
            CloseHandle(hImpersonationToken);
    }

    HANDLE               hProcessToken       = nullptr;
    void*                mallocatedMem       = nullptr;
    PSECURITY_DESCRIPTOR pSecDescriptor      = nullptr;
    bool                 bRevertToSelf       = false;
    HANDLE               hImpersonationToken = nullptr;
};

bool PageAllocator::checkAndAdjustPrivileges(PlatformContext& ctx,
                                             void*            phProcessToken,
                                             void const*      pseLockMemoryPrivilegeLUID,
                                             void*            pData)
{
    HANDLE      hProcessToken             = reinterpret_cast<HANDLE>(phProcessToken);
    Janitor&    janitor                   = *reinterpret_cast<Janitor*>(pData);
    LUID const& seLockMemoryPrivilegeLUID = *reinterpret_cast<LUID const*>(pseLockMemoryPrivilegeLUID);
    // 1. Get count of bytes for the TokenPrivileges TOKEN_INFORMATION_CLASS for this token handle
    DWORD requiredSize = 0;
    if (!GetTokenInformation(hProcessToken, TokenPrivileges, nullptr, 0, &requiredSize) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER)
    {
        uint32_t         length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        std::string_view view{sErrorBuffer.data(), length};
        ctx.error(
            "Could not get the required size {} for the "
            "`TokenPrivileges` Token Information, error: {}",
            {requiredSize, view});
        return false;
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
        return false;
    }

    // 2. actually get the TokenPrivileges TOKEN_INFORMATION_CLASS
    if (!GetTokenInformation(hProcessToken, TokenPrivileges, pTokenPrivilegesInformation, requiredSize, &requiredSize))
    {
        uint32_t         length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        std::string_view view{sErrorBuffer.data(), length};
        ctx.error("Could not get the `TokenPrivileges` Token Information, error: {}", {view});
        return false;
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
        if (win32::luidCompare(luid, seLockMemoryPrivilegeLUID))
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
        return enableLockPrivilege(ctx, hProcessToken, pseLockMemoryPrivilegeLUID, seLockMemoryPrivilegeIndex, pData);
    }

    return true;
}

bool PageAllocator::checkVirtualAlloc2InKernelbaseDll(PlatformContext& ctx)
{
    HMODULE hKernel32Dll = LoadLibraryA("kernelbase.dll");
    if (!hKernel32Dll)
    {
        uint32_t         length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        std::string_view view{sErrorBuffer.data(), length};
        ctx.error("Could not load the kernelbase.dll into an HMODULE, error: {}", {view});
        return false;
    }

    auto* functionPointer = reinterpret_cast<PVirtualAlloc>(GetProcAddress(hKernel32Dll, "VirtualAlloc2"));
    if (!functionPointer)
    {
        uint32_t         length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        std::string_view view{sErrorBuffer.data(), length};
        ctx.error("Could not load the `VirtualAlloc2` from kernelbase.dll, using VirtualAlloc, error: {}", {view});
        ctx.error(
            "If you are unsure whether you support `VirtualAlloc2` "
            "or not, you can check the command `dumpbin /EXPORTS C:\\Windows\\System32\\kernelbase.dll | findstr "
            "VirtualAlloc2`");
        return false;
    }

    return true;
}

bool PageAllocator::enableLockPrivilege(PlatformContext& ctx,
                                        void*            phProcessToken,
                                        void const*      pseLockMemoryPrivilegeLUID,
                                        int64_t          seLockMemoryPrivilegeIndex,
                                        void*            pData)
{
    HANDLE      hProcessToken             = reinterpret_cast<HANDLE>(phProcessToken);
    LUID const& seLockMemoryPrivilegeLUID = *reinterpret_cast<LUID const*>(pseLockMemoryPrivilegeLUID);
    // write into a new entry if we didn't find it at all
    // we are basically preparing the `NewState` parameter for the `AdjustTokenPrivileges`
    if (seLockMemoryPrivilegeIndex < 0)
    {
        // also HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management registry can
        // tell if you disabled it
        ctx.error(
            "SeLockMemoryPrivilege is absent in the current user token. "
            "Try to run as admin, check `secpol.msc` or use cudaAllocHost");
        return false;
    }
    else
    {
        TOKEN_PRIVILEGES privs; // Assuming ANYSIZE_ARRAY = 1
        privs.PrivilegeCount           = 1;
        privs.Privileges[0].Luid       = seLockMemoryPrivilegeLUID;
        privs.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
        if (!AdjustTokenPrivileges(hProcessToken, false, &privs, sizeof(TOKEN_PRIVILEGES), nullptr, nullptr))
        {
            uint32_t         length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
            std::string_view view{sErrorBuffer.data(), length};
            ctx.error("Could not add SeLockMemoryPrivilege to the user token, error: {}", {view});
            return false;
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
                uint32_t         length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
                std::string_view view{sErrorBuffer.data(), length};
                ctx.error("Call to `PrivilegeCheck` failed, error: {}", {view});
                return false;
            }
            if (!result)
            {
                ctx.error(
                    "Even though you called `AdjustTokenPrivileges`, the SeLockMemoryPrivilege is still nowhere to "
                    "be found");
                return false;
            }

            return true;
        }
        else if (errCode == ERROR_SUCCESS)
        {
            return true;
        }
        else
        {
            ctx.error("`AdjustTokenPrivilege` returned an unexpected error code: {}", {errCode});
            return false;
        }
    }
}

void* PageAllocator::createImpersonatingThreadToken(PlatformContext& ctx, void* phProcessToken, void* pData)
{
    HANDLE   hProcessToken       = reinterpret_cast<HANDLE>(phProcessToken);
    Janitor& janitor             = *reinterpret_cast<Janitor*>(pData);
    HANDLE   hImpersonationToken = nullptr;
    if (!DuplicateToken(hProcessToken, SecurityImpersonation, &hImpersonationToken))
    {
        uint32_t         length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        std::string_view view{sErrorBuffer.data(), length};
        ctx.error("Failed call to `OpenThreadToken`, error: {}", {view});
        return INVALID_HANDLE_VALUE;
    }
    janitor.hImpersonationToken = hImpersonationToken;
    HANDLE hThread              = GetCurrentThread();
    if (!SetThreadToken(&hThread, hImpersonationToken))
    {
        uint32_t         length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        std::string_view view{sErrorBuffer.data(), length};
        ctx.error("Failed call to `SetThreadToken`, error: {}", {view});
        return INVALID_HANDLE_VALUE;
    }
    janitor.bRevertToSelf = true;
    return hImpersonationToken;
}

// TODO error handing with the Janitor Pattern
// the Process Security Descriptor, the Token -> Close
PageAllocator::PageAllocator(PlatformContext& ctx, PageAllocatorHooks const& hooks) : PageAllocator(hooks)
{
    Janitor janitor;

    // Get some of the system information relevant to memory allocation, eg
    // - when `VirtualAlloc` is called with MEM_RESERVE, the allocation is aligned to the `allocation granularity`
    // - when `VirtualAlloc` it called with MEM_COMMIT, the allocation is aligned to a page boundary
    SYSTEM_INFO sysInfo{};
    GetSystemInfo(&sysInfo);
    m_systemPageSize        = sysInfo.dwPageSize;
    m_allocationGranularity = sysInfo.dwAllocationGranularity;
    if (m_systemPageSize == 0)
    {
        ctx.error("The current system does not support large pages for some reason");
        return;
    }

    // Retrieve the LUID associated with the SE_LOCK_MEMORY_NAME = "SeLockMemoryPrivilege"
    LUID seLockMemoryPrivilegeLUID{};
    if (!LookupPrivilegeValue(nullptr /*on the local system*/, SE_LOCK_MEMORY_NAME, &seLockMemoryPrivilegeLUID))
    {
        uint32_t         length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
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
    // TOKEN_DUPLICATE and TOKEN_IMPOERSONATION for AccessCheck, as they allow me
    // to duplicate the process token to impersonate the user with a thread token
    // see https://learn.microsoft.com/en-us/windows/win32/secauthz/access-rights-for-access-token-objects
    HANDLE hProcessToken   = nullptr;
    DWORD  tokenAccessMode = TOKEN_QUERY | TOKEN_ADJUST_PRIVILEGES | TOKEN_DUPLICATE | TOKEN_IMPERSONATE;
    if (!OpenProcessToken(hCurrentProc, tokenAccessMode, &hProcessToken))
    {
        uint32_t         length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
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
    bool seLockMemoryPrivilegeEnabled = checkAndAdjustPrivileges(ctx, hProcessToken, &seLockMemoryPrivilegeLUID, (void*)&janitor);

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
    bool bVirtualAlloc2Supported = checkVirtualAlloc2InKernelbaseDll(ctx);
    if (!bVirtualAlloc2Supported)
    {
        return;
    }

    // 2. Look into the DACL of the current process to see whether you have the `PROCESS_VM_OPERATION` access right
    // Docs: https://learn.microsoft.com/en-us/windows/win32/procthread/process-security-and-access-rights
    // 2.1 First retrieve the Process Security Descriptor (to then free With LocalFree)
    PSECURITY_DESCRIPTOR securityDescriptor = nullptr;

    // left comments for future reference
    SECURITY_INFORMATION securityInfo = OWNER_SECURITY_INFORMATION | GROUP_SECURITY_INFORMATION |
                                        DACL_SECURITY_INFORMATION /*| SACL_SECURITY_INFORMATION|
                                        LABEL_SECURITY_INFORMATION | ATTRIBUTE_SECURITY_INFORMATION |
                                        SCOPE_SECURITY_INFORMATION | PROCESS_TRUST_LABEL_SECURITY_INFORMATION |
                                        ACCESS_FILTER_SECURITY_INFORMATION | BACKUP_SECURITY_INFORMATION
                                        */
        ;
    DWORD status = GetSecurityInfo(hCurrentProc,     // the current process HANDLE
                                   SE_KERNEL_OBJECT, // a process is classified as a kernel object
                                   securityInfo, // bits of the info to retrieve. we want process specific (discretionary)
                                   nullptr,      // Owner SID
                                   nullptr,      // Group SID
                                   nullptr,      // DACL
                                   nullptr,      // SACL
                                   &securityDescriptor);

    if (status != ERROR_SUCCESS)
    {
        ctx.error("Could not retrieve the Process Security Descriptor, error code {}", {status});
        return;
    }

    janitor.pSecDescriptor = securityDescriptor;

    if (!IsValidSecurityDescriptor(securityDescriptor))
    {
        ctx.error("The retrieved security descriptor at {} is not valid", {securityDescriptor});
        return;
    }

    // GENERIC_MAPPING = r?w?x?. Each member is an int, ACCESS_MASK, https://learn.microsoft.com/en-us/windows/win32/secauthz/access-mask
    GENERIC_MAPPING genericMapping = {
        PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, // GENERIC_READ
        PROCESS_VM_WRITE | PROCESS_VM_OPERATION,     // GENERIC_WRITE
        PROCESS_CREATE_THREAD,                       // GENERIC_EXECUTE
        PROCESS_ALL_ACCESS                           // GENERIC_ALL
    };

    ACCESS_MASK   outAccessMask = 0;
    DWORD         desiredAccess = PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION;
    BOOL          bAccessStatus = false;
    PRIVILEGE_SET privilegeSet;
    DWORD         privilegeSetSize = sizeof(PRIVILEGE_SET);
    // https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-mapgenericmask
    MapGenericMask(&desiredAccess, &genericMapping);

    // `AccessCheck` function to see whether the process security descriptor has a predefined
    // set of access rights
    // `AccessCheck` requires a *Client Token*, which is a token associated to some entity (local user or client-server), derived from a primary token
    // But `OpenProcessToken` returns a *Primary Token*, which represents the user account under which the process is running
    // To Impersonate a client token from a primary token, use `ImpersonateSelf`
    // source: https://stackoverflow.com/questions/35027524/whats-the-difference-between-a-primary-token-and-an-impersonation-token
    // basically, AccessCheck works with thread tokens, not process tokens, so we need to
    // fetch the association user - process and map it onto the current thread
    // source: book "Programming Windows Security"
    //if (!ImpersonateSelf(SecurityImpersonation))
    //{
    //    uint32_t         length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
    //    std::string_view view{sErrorBuffer.data(), length};
    //    ctx.error("Failed call to `ImpersonateSelf`, error: {}", {view});
    //    return;
    //}
    //janitor.bRevertToSelf = true;
    // source: https://blog.aaronballman.com/2011/08/how-to-check-access-rights/
    HANDLE hImpersonationToken = reinterpret_cast<HANDLE>(
        createImpersonatingThreadToken(ctx, hProcessToken, (void*)&janitor));
    if (hImpersonationToken == INVALID_HANDLE_VALUE)
    {
        return;
    }

    if (!AccessCheck(securityDescriptor,  // security descriptor against which access is checked
                     hImpersonationToken, // impersonation token representing the user attempting the access
                     desiredAccess,       // desired access rights
                     &genericMapping,
                     &privilegeSet,
                     &privilegeSetSize,
                     &outAccessMask,
                     &bAccessStatus))
    {
        uint32_t         length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        std::string_view view{sErrorBuffer.data(), length};
        ctx.error("Failed call to AccessCheck, error: {}", {view});
        return;
    }

    if (!bAccessStatus)
    {
        uint32_t         length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        std::string_view view{sErrorBuffer.data(), length};
        ctx.error(
            "The Process doesn't own the"
            " PROCESS_VM_OPERATION access rights, using 2MB large pages. Error: {}",
            {view});
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

#if defined(DMT_OS_WINDOWS)
// TODO promote to class static protected, shared between windows and linux
void PageAllocator::addAllocInfo(PlatformContext& ctx, bool isLargePageAlloc, PageAllocation& ret)
{
    static constexpr uint32_t log4KB      = 12u;
    uint32_t                  errorLength = 0;
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
        errorLength = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
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
        errorLength = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
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

    m_hooks.allocHook(m_hooks.data, ctx, ret);
}
#endif

// NOTE: Only Pinned memory can be mapped to CUDA device memory, hence you need to
// check that the page size is not 4KB. Mapping is carried out with cudaHostRegister
// reserve = take virtual address space. commit = when you write to it, it will be backed by physical memory
// TODO see Address Windowing Extension pages (AWE)
// TODO integrate MEM_WRITE_WATCH
PageAllocation PageAllocator::allocatePage(PlatformContext& ctx, EPageSize sizeOverride)
{
    HooksJanitor janitor{ctx, true};
    janitor.pHooks = &m_hooks;

    static constexpr uint32_t log4KB = 12u;
    PageAllocation            ret{};
    ret.pageNum               = -1;
    uint32_t  errorLength     = 0;
    DWORD     allocationFlags = MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES;
    DWORD     protectionFlags = PAGE_READWRITE;
    EPageSize pageSize        = sizeOverride;

    if (m_largePageEnabled && pageSize >= EPageSize::e2MB)
    {
        if (m_largePage1GB && sizeOverride == EPageSize::e1GB)
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
                errorLength = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
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
                errorLength = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
                std::string_view view{sErrorBuffer.data(), errorLength};
                ctx.error("`VirtualAlloc` with MEM_LARGE_PAGES, trying without it, error: {}", {view});
            }
        }
    }

    if (!ret.address)
    {
        allocationFlags &= ~MEM_LARGE_PAGES;
        pageSize    = EPageSize::e4KB;
        ret.address = VirtualAlloc(nullptr, toUnderlying(pageSize), allocationFlags, protectionFlags);
        ret.bits    = DMT_OS_WINDOWS_SMALL_PAGE;
        if (!ret.address)
        {
            errorLength = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
            std::string_view view{sErrorBuffer.data(), errorLength};
            ctx.error("`VirtualAlloc` failed, couldn't allocate memory, error: {}", {view});
        }
    }

    if (ret.address)
    {
        janitor.pAlloc = &ret;
        // touch first byte to make sure committed memory is backed to physical memory
        reinterpret_cast<unsigned char*>(ret.address)[0] = 0;

        // bookkeeping
        ret.count             = 1;
        ret.pageSize          = pageSize;
        bool isLargePageAlloc = (allocationFlags & MEM_LARGE_PAGES) != 0;

#if defined(DMT_DEBUG)
        ctx.trace(
            "Called allocatePage, allocated "
            "at {} page of {} B. Printing Stacktrace",
            {ret.address, StrBuf{toUnderlying(ret.pageSize), "0x%zx"}});
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

bool PageAllocator::allocate2MB(PlatformContext& ctx, PageAllocation& out)
{
    HooksJanitor janitor{ctx, true};
    janitor.pHooks = &m_hooks;
    DWORD        allocationFlags = MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES;
    DWORD const  protectionFlags = PAGE_READWRITE;
    size_t const size            = toUnderlying(EPageSize::e2MB);
    out.address                  = nullptr;
    bool     isLargePage         = false;
    uint32_t errorLength         = 0;
    if (m_largePageEnabled)
    {
        out.address = VirtualAlloc(nullptr, size, allocationFlags, protectionFlags);
        if (!out.address)
        {
            errorLength = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
            std::string_view view{sErrorBuffer.data(), errorLength};
            ctx.error("`VirtualAlloc` with MEM_LARGE_PAGES, trying without it, error: {}", {view});
        }
        else
        {
            out.count    = 1;
            out.pageSize = EPageSize::e2MB;
            out.bits     = DMT_OS_WINDOWS_LARGE_PAGE;
            isLargePage  = true;
        }
    }

    if (!out.address)
    {
        allocationFlags &= ~MEM_LARGE_PAGES;
        out.address = VirtualAlloc(nullptr, size, allocationFlags, protectionFlags);
        if (!out.address)
        {
            errorLength = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
            std::string_view view{sErrorBuffer.data(), errorLength};
            ctx.error("`VirtualAlloc` failed, error: {}", {view});
            return false;
        }
        else
        {
            out.count    = num4KBIn2MB;
            out.pageSize = EPageSize::e4KB;
            out.bits     = DMT_OS_WINDOWS_SMALL_PAGE;
        }
    }

    janitor.pAlloc = &out;

    return true;
}

void PageAllocator::deallocatePage(PlatformContext& ctx, PageAllocation& alloc)
{
    // the page allocation might be part of the memory portion itself, we need to copy it
    PageAllocation allocCopy = alloc;

    // decommit = get out of memory, but leave the virtual address space reserved
    // POSSIBLE TODO. The other type, release, will decommit + free up the virtual address space
    DWORD freeType = MEM_RELEASE;
    if (!VirtualFree(alloc.address, 0 /*size 0 if MEM_RELEASE*/, freeType))
    {
        uint32_t         errorLength = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        std::string_view view{sErrorBuffer.data(), errorLength};
#if defined(DMT_DEBUG)
        ctx.error("Failed to Free memory at {}, error {}, Printing Stacktrace", {allocCopy.address, view});
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
        ctx.trace("Deallocated memory at {} size {}, Printing stacktrace",
                  {allocCopy.address, toUnderlying(allocCopy.pageSize)});
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
}

PageAllocator::~PageAllocator()
{
}

#endif // DMT_OS_LINUX, DMT_OS_WINDOWS

void PageAllocator::deallocPage(PlatformContext& ctx, PageAllocation& alloc)
{
    PageAllocation allocCopy = alloc;
    PageAllocator::deallocatePage(ctx, alloc);
    m_hooks.freeHook(m_hooks.data, ctx, allocCopy);
}


AllocatePageForBytesResult PageAllocator::allocatePagesForBytes(
    PlatformContext& ctx,
    size_t           numBytes,
    PageAllocation*  pOut,
    uint32_t         inNum,
    EPageSize        pageSize)
{
    static constexpr uint32_t maxAttempts         = 2048;
    uint32_t                  totalPagesAllocated = 0;
    uint32_t numPages  = static_cast<uint32_t>(ceilDiv(numBytes, static_cast<size_t>(toUnderlying(pageSize))));
    uint32_t allocated = 0;
    uint32_t index     = 0;
    for (uint32_t i = 0; allocated < numBytes && i < maxAttempts; ++i)
    {
        pOut[index] = allocatePage(ctx, pageSize);
        if (!pOut[index].address)
        {
            pageSize = scaleDownPageSize(pageSize);
        }
        else
        {
            ++index;
            allocated += toUnderlying(pageSize);
            ++totalPagesAllocated;
        }
    }

    return {.numBytes = allocated, .numPages = totalPagesAllocated};
}

EPageSize PageAllocator::allocatePagesForBytesQuery(PlatformContext&            ctx,
                                                    size_t                      numBytes,
                                                    uint32_t&                   inOutNum,
                                                    EPageAllocationQueryOptions opts)
{
#if defined(DMT_OS_WINDOWS)
    EPageSize pageSize = m_largePageEnabled ? (m_largePage1GB ? EPageSize::e1GB : EPageSize::e2MB) : EPageSize::e4KB;
#elif defined(DMT_OS_LINUX)
    EPageSize pageSize = m_enabledPageSize;
#else
    size_t pageSize = EPageSize::e4KB;
#endif

    if (pageSize == EPageSize::e1GB && opts != EPageAllocationQueryOptions::eNone)
    {
        pageSize = scaleDownPageSize(pageSize);
    }

    while (toUnderlying(pageSize) > numBytes && pageSize != EPageSize::e4KB)
    {
        pageSize = scaleDownPageSize(pageSize);
    }

    if (opts != EPageAllocationQueryOptions::eForce4KB)
    {
        while (pageSize != EPageSize::e4KB)
        {
            if (checkPageSizeAvailability(ctx, pageSize))
            {
                inOutNum = static_cast<uint32_t>(ceilDiv(numBytes, static_cast<size_t>(toUnderlying(pageSize))));
                return pageSize;
            }

            pageSize = scaleDownPageSize(pageSize);
        }
    }

    inOutNum = static_cast<uint32_t>(ceilDiv(numBytes, static_cast<size_t>(toUnderlying(EPageSize::e4KB))));
    return pageSize;
}

bool PageAllocator::checkPageSizeAvailability(PlatformContext& ctx, EPageSize pageSize)
{ // TODO: don't commit memory or save it into a cache
    ctx.warn("Don't use me");
    // Attempt a small test allocation to verify if the page size is supported
    bool           supported = false;
    PageAllocation testAlloc = allocatePage(ctx, pageSize);
    if (testAlloc.address && testAlloc.pageSize == pageSize)
        supported = true;

    if (testAlloc.address)
        deallocatePage(ctx, testAlloc);

    return supported;
}

// PageAllocatorTracker -----------------------------------------------------------------------------------------------

static void logAndAbort(PlatformContext& ctx, std::string_view str, uint32_t initialNodeNum)
{
    ctx.error("Couldn't commit the first {} nodes into the reserved space", {initialNodeNum});
#if defined(DMT_OS_WINDOWS)
    size_t           errorLength = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
    std::string_view view{sErrorBuffer.data(), errorLength};
    ctx.error("{}, error: {}", {str, view});
#endif
    std::abort();
}

PageAllocationsTracker::PageAllocationsTracker(PlatformContext& ctx, uint32_t pageTrackCapacity, uint32_t allocTrackCapacity)
{
    using namespace std::string_view_literals; 
    // reserve virtual address space the double ended buffer
    m_pageTracking.m_capacity       = pageTrackCapacity;
    m_allocTracking.m_capacity      = allocTrackCapacity;
    m_pageTracking.m_growBackwards  = false;
    m_allocTracking.m_growBackwards = true;
    size_t sysAlign                 = systemAlignment();
    assert(sysAlign >= alignof(PageNode) && sysAlign >= alignof(AllocNode));

    size_t pageFreeListBytes  = initialNodeNum * sizeof(PageNode);
    size_t allocFreeListBytes = initialNodeNum * sizeof(AllocNode);
    size_t allocOffset        = m_allocTracking.m_capacity * sizeof(AllocNode);
    size_t pageOffset         = m_pageTracking.m_capacity * sizeof(PageNode);
    m_bufferBytes             = allocOffset + pageOffset;
    m_base                    = reserveVirtualAddressSpace(m_bufferBytes);
    if (!m_base)
    {
        ctx.error("Couldn't reserve virtual address space for {} bytes", {m_bufferBytes});
        std::abort();
    }

    // commit `initialNodeNum` for the low address free list and the high address free list
    uintptr_t end       = reinterpret_cast<uintptr_t>(m_base) + m_bufferBytes;
    void*     allocBase = alignToBackward(reinterpret_cast<void*>(end - allocFreeListBytes), sysAlign);
    if (!commitPhysicalMemory(m_base, pageFreeListBytes))
    {
        logAndAbort(ctx, "pageFreeList, VirtualAlloc MEM_COMMIT"sv, initialNodeNum);
    }
    if (!commitPhysicalMemory(allocBase, end - reinterpret_cast<uintptr_t>(allocBase)))
    {
        logAndAbort(ctx, "allocFreeList, VirtualAlloc MEM_COMMIT"sv, initialNodeNum);
    }

    // set head and reset each list
    void* pEnd = reinterpret_cast<void*>(end - sizeof(AllocNode));
    std::memset(pEnd, 0, 32);
    m_pageBase  = alignTo(m_base, alignof(PageNode));
    m_allocBase = alignToBackward(pEnd, alignof(AllocNode));

    m_pageTracking.m_freeHead  = reinterpret_cast<PageNode*>(m_pageBase);
    m_allocTracking.m_freeHead = reinterpret_cast<AllocNode*>(m_allocBase);
    m_pageTracking.m_freeSize  = initialNodeNum;
    m_allocTracking.m_freeSize = initialNodeNum;

    m_pageTracking.reset();
    m_allocTracking.reset();
}

template <typename T>
    requires requires(T t) { t.address; }
static bool shouldTrack(PlatformContext& ctx, T const& alloc)
{
    bool ret = alloc.address != nullptr;
    if (!ret)
    {
        ctx.warn("Passed an invalid allocation to the trakcer, nullptr address");
    }
    return ret;
}

void PageAllocationsTracker::track(PlatformContext& ctx, PageAllocation const& alloc)
{
    ctx.trace("tracking allocation at address {}", { alloc.address });
    if (!m_pageTracking.m_freeHead)
    {
        growFreeList(ctx, m_pageTracking, m_pageBase);
    }

    if (shouldTrack(ctx, alloc))
    {
        m_pageTracking.addNode(alloc);
    }
}

void PageAllocationsTracker::untrack(PlatformContext& ctx, PageAllocation const& alloc)
{
    ctx.trace("Untracking allocation at address {}", { alloc.address });
    if (m_pageTracking.removeNode(alloc))
    {
        return;
    }

    ctx.error("Attempted to deallocate a page that was not tracked.");
    std::abort();
}

void PageAllocationsTracker::track(PlatformContext& ctx, AllocationInfo const& alloc)
{
    ctx.trace("tracking allocation at address {}", { alloc.address });
    if (!m_allocTracking.m_freeHead)
    {
        growFreeList(ctx, m_allocTracking, m_allocBase);
    }

    if (shouldTrack(ctx, alloc))
    {
        m_allocTracking.addNode(alloc);
    }
}

void PageAllocationsTracker::untrack(PlatformContext& ctx, AllocationInfo const& alloc)
{
    ctx.trace("Untracking allocation at address {}", { alloc.address });
    if (m_allocTracking.removeNode(alloc))
    {
        return;
    }

    ctx.error("Attempted to deallocate a page that was not tracked.");
    std::abort();
}

void PageAllocationsTracker::claenTransients(PlatformContext& ctx)
{
    ctx.log("Untracking (deallocating tracking data) for all transient allocations");
    m_allocTracking.removeTransientNodes();
}

PageAllocationsTracker::~PageAllocationsTracker() noexcept
{
    PlatformContext::Table       nullTable;
    PlatformContext::InlineTable nullInlineTable;
    PlatformContext              nullCtx{nullptr, &nullTable, nullInlineTable};
    assert(!m_pageTracking.m_occupiedHead && !m_allocTracking.m_occupiedHead &&
           "some allocated memory is outliving the tracker");

    freeVirtualAddressSpace(m_base, m_bufferBytes);
}

// StackAllocator -----------------------------------------------------------------------------------------------------
StackAllocator::StackAllocator(PlatformContext& ctx, PageAllocator& pageAllocator, AllocatorHooks const& hooks)
{
    // try to allocate the first 2MB stack buffer
    PageAllocation alloc;
    if (!pageAllocator.allocate2MB(ctx, alloc))
    {
        ctx.error("Couldn't allocate 2MB buffer for the stack allocator, aborting ...");
        std::abort();
    }

    // create the StackHeader structure in the beginning of the page
    assert(alloc.address == alignTo(alloc.address, alignof(StackHeader)) &&
           "page address isn't "
           "aligned to a stack header size boundary");

    StackHeader& header = *reinterpret_cast<StackHeader*>(alloc.address);
    header.alloc        = alloc;
    header.bp           = reinterpret_cast<uintptr_t>(&header) + sizeof(StackHeader);
    header.sp           = header.bp;
    header.prev         = nullptr;
    header.next         = nullptr;
    header.notUsedFor   = 0;

    // wire the buffer
    m_pFirst = &header;
    m_pLast  = &header;

    // copy hooks
    m_hooks = hooks;
}

void StackAllocator::cleanup(PlatformContext& ctx, PageAllocator& pageAllocator)
{
    std::lock_guard lock{m_mtx};
    m_hooks.cleanTransients(m_hooks.data, ctx);

    // start from last and free all buffers
    for (StackHeader* curr = m_pLast; curr != nullptr;)
    {
        StackHeader* prev = curr->prev;
        ctx.trace("Deallocating buffer at address: {}", {reinterpret_cast<void*>(curr)});
        pageAllocator.deallocPage(ctx, curr->alloc);
        curr = prev;
    }

    m_pFirst = nullptr;
    m_pLast  = nullptr;
    ctx.trace("Cleanup complete. All buffers have been deallocated.");
}

void* StackAllocator::allocate(PlatformContext& ctx, PageAllocator& pageAllocator, size_t size, size_t alignment)
{
    if (size > bufferSize || (alignment & (alignment - 1)) != 0)
    {
        ctx.error("Invalid allocation parameters: size={}, alignment={}", {size, alignment});
        return nullptr;
    }

    std::lock_guard lock{m_mtx};
    assert(m_pFirst);

    // verify that sp is within the bounds [bp, bp+bufferSize)
    // if yes, then verify that, given size and alignment, we can satisfy the allocation
    // if yes again, then compute the aligned pointer, update sp, and return
    // if any of the aforementioned conditions failed, we need to try onto the next buffer the same procedure
    // if there's no next buffer, then call `newBuffer` and try again
    while (true)
    {
        for (StackHeader* curr = m_pLast; curr != nullptr; curr = curr->next)
        {
            uintptr_t alignedSP = alignToAddr(curr->sp, alignment);
            uintptr_t end       = curr->bp + bufferSize;

            if (alignedSP + size <= end)
            {
                void* ptr        = reinterpret_cast<void*>(alignedSP);
                curr->sp         = alignedSP + size;
                curr->notUsedFor = 0;

                // call allocation hooks (TODO create properly AllocationInfo)
                AllocationInfo allocInfo;
                allocInfo.address = ptr;
                allocInfo.size    = size;
                allocInfo.alignment = alignment;
                allocInfo.transient = 1;
                m_hooks.allocHook(m_hooks.data, ctx, allocInfo);

                return ptr;
            }
        }

        if (!newBuffer(ctx, pageAllocator))
        {
            ctx.error("Allocation failed: size={}, alignment={}", {size, alignment});
            return nullptr;
        }
    }

    return nullptr;
}

void StackAllocator::reset(PlatformContext& ctx, PageAllocator& pageAllocator)
{
    std::lock_guard lock{m_mtx};
    assert(m_pFirst);

    m_hooks.cleanTransients(m_hooks.data, ctx);

    for (StackHeader* curr = m_pLast; curr != nullptr;)
    {
        curr->sp          = curr->bp;
        StackHeader* prev = curr->prev;

        // if you didn't use the buffer for a certain number of times, then deallocate it
        if (++curr->notUsedFor > notUsedForThreshold)
        {
            pageAllocator.deallocPage(ctx, curr->alloc);
        }
        curr = prev;
        if (prev)
            prev->next = nullptr; // there's the assumption that you are using buffers in order. how to check?
    }
}

bool StackAllocator::newBuffer(PlatformContext& ctx, PageAllocator& pageAllocator)
{
    assert(m_pLast);
    PageAllocation alloc;
    if (!pageAllocator.allocate2MB(ctx, alloc))
    {
        ctx.error("Failed to allocate new 2MB buffer.");
        return false;
    }

    // Create the new StackHeader at the beginning of the new buffer
    StackHeader& newHeader = *reinterpret_cast<StackHeader*>(alloc.address);
    newHeader.alloc        = alloc;
    newHeader.bp           = reinterpret_cast<uintptr_t>(&newHeader) + sizeof(StackHeader);
    newHeader.sp           = newHeader.bp;
    newHeader.prev         = m_pLast;
    newHeader.next         = nullptr;
    newHeader.notUsedFor   = 0;

    // Update the chain
    m_pLast->next = &newHeader;
    m_pLast       = &newHeader;

    return true;
}

// MultiPoolAllocator -------------------------------------------------------------------------------------------------

static constexpr uint16_t bufferIndex(uint16_t tag)
{ // 7 high bits + 3 low bits
    return tag >> 2;
}

static constexpr uint8_t blockSizeEncoding(uint16_t tag)
{ // 2 least significant bits
    return tag & 0x3;
}

static constexpr TaggedPointer encode(uint16_t bufferIndex, uint8_t blockSizeEncoding, void* ptr)
{
    uint16_t tag = bufferIndex | static_cast<uint16_t>(blockSizeEncoding);
    assert(tag <= 0xFFF);
    return {ptr, tag};
}

MultiPoolAllocator::MultiPoolAllocator(PlatformContext&                    ctx,
                                       PageAllocator&                      pageAllocator,
                                       std::array<uint32_t, numBlockSizes> numBlocksPerPool,
                                       AllocatorHooks const&               hooks) :
m_hooks(hooks)
{
    static_assert(numBlockSizes == 4);
    static constexpr uint16_t max = 256;

    // compute numBytes and counts for each pool
    uint16_t blockSizes[numBlockSizes]{toUnderlying(EBlockSize::e32B),
                                       toUnderlying(EBlockSize::e64B),
                                       toUnderlying(EBlockSize::e128B),
                                       toUnderlying(EBlockSize::e256B)};
    size_t   bytesPerPool[numBlockSizes]{numBlocksPerPool[0] * blockSizes[0],
                                         numBlocksPerPool[1] * blockSizes[1],
                                         numBlocksPerPool[2] * blockSizes[2],
                                         numBlocksPerPool[3] * blockSizes[3]};

    // compute size of the metadata required to associate 1 bit to each block (account for the PageAllocation(24B) + next pointer + uint8_t notUsedFor)
    // modify numBytes and counts such that we can fit metadata + pools into the 2MB buffer
    uint32_t numBitsMetadata = std::reduce(std::begin(numBlocksPerPool), std::end(numBlocksPerPool), 0u, std::plus<>());
    m_numBytesMetadata       = ceilDiv(numBitsMetadata, 8u);
    m_totalSize = std::reduce(std::begin(bytesPerPool), std::end(bytesPerPool), 0ULL, std::plus<>()) + m_numBytesMetadata;
    ctx.log("tring to allocate pool buffer for {} {} {} {}",
            {StrBuf{bytesPerPool[0], "0x%zx"},
             StrBuf{bytesPerPool[1], "0x%zx"},
             StrBuf{bytesPerPool[2], "0x%zx"},
             StrBuf{bytesPerPool[3], "0x%zx"}});
    while (m_totalSize > bufferSize)
    { // reduce the largest pool whose block size is bigger than the difference between total and bufferSize
        size_t residual = m_totalSize - bufferSize;
        for (uint32_t i = numBlockSizes - 1; i < numBlockSizes; --i)
        {
            if (residual > blockSizes[i])
            {
                --numBlocksPerPool[i];
                bytesPerPool[i] -= blockSizes[i];
                m_totalSize -= blockSizes[i];
                --numBitsMetadata;
                m_numBytesMetadata = ceilDiv(numBitsMetadata, 8u);
                break;
            }
        }
    }

    // Adjust the pools to fill up the remaining space but leave room for alignment overhead
    size_t remainingSpace = bufferSize - m_totalSize;
    while (remainingSpace > *std::min_element(blockSizes, blockSizes + numBlockSizes)) // Ensure we leave space for alignment
    {
        bool blockAdded = false;
        for (uint32_t i = 0; i < numBlockSizes; ++i)
        {
            size_t blockSizeWithMetadata = blockSizes[i] + (1 / 8u);        // Block size + 1 bit metadata
            if (remainingSpace > blockSizeWithMetadata + poolBaseAlignment) // Leave room for alignment
            {
                ++numBlocksPerPool[i];
                bytesPerPool[i] += blockSizes[i];
                m_totalSize += blockSizeWithMetadata;
                ++numBitsMetadata;
                m_numBytesMetadata = ceilDiv(numBitsMetadata, 8u);
                remainingSpace     = bufferSize - m_totalSize;
                blockAdded         = true;
                break;
            }
        }
        if (!blockAdded)
            break; // No block could fit into the remaining space
    }

    ctx.log("The possible sizes are {} {} {} {}",
            {StrBuf{bytesPerPool[0], "0x%zx"},
             StrBuf{bytesPerPool[1], "0x%zx"},
             StrBuf{bytesPerPool[2], "0x%zx"},
             StrBuf{bytesPerPool[3], "0x%zx"}});
    ctx.log("Allocating a buffer of {} Bytes, actually using {} Bytes",
            {StrBuf{bufferSize, "0x%zx"}, StrBuf{m_totalSize, "0x%zx"}});

    for (uint32_t i = 0; i != numBlockSizes; ++i)
        m_numBlocksPerPool[i] = m_numBlocksPerPool[i];

    // allocate initial 2MB block
    newBlock(ctx, pageAllocator, &m_firstBuffer);
    m_lastBuffer = m_firstBuffer;
}

void MultiPoolAllocator::newBlock(PlatformContext& ctx, PageAllocator& pageAllocator, BufferHeader** ptr)
{
    PageAllocation pageAlloc;
    if (!pageAllocator.allocate2MB(ctx, pageAlloc))
    {
        ctx.error("Couldn't allocate 2MB buffer for multi pool allocator, aborting...");
        std::abort();
    }
    auto& header = *reinterpret_cast<BufferHeader*>(pageAlloc.address);
    header.alloc = pageAlloc;
    header.next  = nullptr;

    // compute the buffer base
    uintptr_t metadataAddr = reinterpret_cast<uintptr_t>(&header) + sizeof(BufferHeader);
    uintptr_t poolBase     = metadataAddr + m_numBytesMetadata;
    poolBase               = alignToAddr(poolBase, poolBaseAlignment);
    if (size_t alignmentOverhead = poolBase - (metadataAddr + m_numBytesMetadata); alignmentOverhead > 0)
    {
        size_t adjustedTotal = m_totalSize + alignmentOverhead;
        if (adjustedTotal > bufferSize)
        {
            ctx.error("Alignment overhead caused total size to exceed buffer limit.");
            std::abort();
        }
        m_totalSize += alignmentOverhead;
    }

    header.poolBase = poolBase;

    // tag all blocks in the metadata as empty
    std::memset(std::bit_cast<void*>(metadataAddr), 0, poolBase - metadataAddr);

    if (ptr)
    {
        *ptr = reinterpret_cast<BufferHeader*>(pageAlloc.address);
    }
}

void MultiPoolAllocator::cleanup(PlatformContext& ctx, PageAllocator& pageAllocator)
{
    std::lock_guard lock{m_mtx};

    // Traverse the linked list of buffers starting from the first buffer
    BufferHeader* currentBuffer = m_firstBuffer;

    while (currentBuffer)
    {
        // Save the pointer to the next buffer before deallocating the current one
        BufferHeader* nextBuffer = currentBuffer->next;

        // Deallocate the page associated with the current buffer
        pageAllocator.deallocPage(ctx, currentBuffer->alloc);

        // Move to the next buffer in the list
        currentBuffer = nextBuffer;
    }

    // Clear the state of the allocator
    m_firstBuffer      = nullptr;
    m_lastBuffer       = nullptr;
    m_totalSize        = 0;
    m_numBytesMetadata = 0;
    std::fill(std::begin(m_numBlocksPerPool), std::end(m_numBlocksPerPool), 0);

    ctx.log("MultiPoolAllocator cleanup complete. All buffers deallocated.");
}

TaggedPointer MultiPoolAllocator::allocateBlocks(PlatformContext& ctx, PageAllocator& pageAllocator, uint32_t numBlocks, EBlockSize blockSize)
{
    std::lock_guard lock{m_mtx};

    // Determine the block size index
    uint8_t blockSizeIndex = blockSizeEncoding(blockSize);

    BufferHeader* currentBuffer = m_firstBuffer;
    uint16_t      bufferIdx     = 0;

    // Iterate through the buffers to find free blocks
    while (currentBuffer)
    {
        uintptr_t metadataAddr = std::bit_cast<uintptr_t>(currentBuffer) + sizeof(BufferHeader);
        uintptr_t poolBase     = currentBuffer->poolBase;

        // Scan the metadata for a free region of numBlocks
        uint8_t* metadata        = reinterpret_cast<uint8_t*>(metadataAddr);
        uint32_t numBlocksInPool = m_numBlocksPerPool[blockSizeIndex];
        uint16_t blockSizeBytes  = toUnderlying(blockSize);

        for (uint32_t i = 0; i <= numBlocksInPool - numBlocks; ++i)
        {
            // Check if a region of numBlocks is free
            bool isRegionFree = true;
            for (uint32_t j = 0; j < numBlocks; ++j)
            {
                if (metadata[(i + j) / 8] & (1 << ((i + j) % 8))) // not sure this works
                {
                    isRegionFree = false;
                    ++bufferIdx;
                    break;
                }
            }

            if (isRegionFree)
            {
                // Mark the blocks as allocated in metadata
                for (uint32_t j = 0; j < numBlocks; ++j)
                {
                    metadata[(i + j) / 8] |= (1 << ((i + j) % 8)); // not sure this works
                }

                // Calculate the starting address of the allocated blocks
                uintptr_t blockAddr = poolBase + i * blockSizeBytes;

                ctx.log("Allocated {} blocks of size {} at address {}",
                        {numBlocks, blockSizeBytes, StrBuf{blockAddr, "0x%zx"}});

                // TODO perform hooks
                AllocationInfo info;
                info.address = reinterpret_cast<void*>(blockAddr);
                info.size    = blockSizeBytes * numBlocks;
                info.alignment = poolBaseAlignment;
                info.transient = 0;
                m_hooks.allocHook(m_hooks.data, ctx, info);
                return encode(bufferIdx, blockSizeIndex, reinterpret_cast<void*>(blockAddr));
            }
        }

        currentBuffer = currentBuffer->next;
    }

    // No free blocks found, try to allocate a new buffer
    BufferHeader* newBuffer = nullptr;
    newBlock(ctx, pageAllocator, &newBuffer);

    if (!newBuffer)
    {
        ctx.error("Failed to allocate a new buffer for MultiPoolAllocator.");
        return {nullptr, 0};
    }

    // Add the new buffer to the list
    m_lastBuffer->next = newBuffer;
    m_lastBuffer       = newBuffer;

    // Retry allocation
    return allocateBlocks(ctx, pageAllocator, numBlocks, blockSize);
}

void MultiPoolAllocator::freeBlocks(PlatformContext& ctx, PageAllocator& pageAllocator, uint32_t numBlocks, TaggedPointer ptr)
{
    std::lock_guard lock{m_mtx};

    if (ptr == taggedNullptr)
    {
        ctx.error("Attempting to free a null pointer.");
        return;
    }

    // Extract buffer index and block size encoding from the tag
    uint16_t tag          = ptr.tag();
    uint16_t bufferIdx    = bufferIndex(tag);
    uint8_t  blockSizeEnc = blockSizeEncoding(tag);

    if (blockSizeEnc >= numBlockSizes)
    {
        ctx.error("Invalid block size encoding in tagged pointer.");
        return;
    }

    EBlockSize blockSize      = fromEncoding(blockSizeEnc);
    uint16_t   blockSizeBytes = toUnderlying(blockSize);

    // Locate the buffer using the buffer index
    BufferHeader* currentBuffer = m_firstBuffer;
    for (uint16_t i = 0; i < bufferIdx && currentBuffer; ++i)
    {
        currentBuffer = currentBuffer->next;
    }

    if (!currentBuffer)
    {
        ctx.error("Buffer index out of bounds.");
        return;
    }

    uintptr_t metadataAddr = reinterpret_cast<uintptr_t>(currentBuffer) + sizeof(BufferHeader);
    uintptr_t poolBase     = currentBuffer->poolBase;

    // Calculate the block index
    uintptr_t blockAddr = ptr.address();
    if (blockAddr < poolBase || (blockAddr - poolBase) % blockSizeBytes != 0) // not sure this works
    {
        ctx.error("Pointer does not correspond to a valid block in this buffer.");
        return;
    }

    uint32_t blockIndex = static_cast<uint32_t>((blockAddr - poolBase) / blockSizeBytes);

    // Update metadata to mark blocks as free
    uint8_t* metadata = reinterpret_cast<uint8_t*>(metadataAddr);
    for (uint32_t i = 0; i < numBlocks; ++i)
    {
        metadata[(blockIndex + i) / 8] &= ~(1 << ((blockIndex + i) % 8)); // not sure this works
    }

    AllocationInfo info;
    info.address = ptr.pointer();
    m_hooks.freeHook(m_hooks.data, ctx, info);
    ctx.log("Freed {} blocks of size {} at address {}", {numBlocks, blockSizeBytes, StrBuf{blockAddr, "0x%zx"}});
}

} // namespace dmt
