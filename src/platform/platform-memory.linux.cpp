#include "platform-memory.h"
#include "platform-os-utils.linux.h"

#include <fcntl.h>
#include <linux/kernel-page-flags.h>
#include <sys/mman.h>
#include <unistd.h>

#define PAGE_INFO_IMPL
#include <page-info/page-info.h>

#include <cerrno>
#include <cstdlib>

namespace dmt::os {
    void* reserveVirtualAddressSpace(size_t size)
    {
        void* address = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (address == MAP_FAILED)
        {
            return nullptr;
        }
        return address;
    }

    bool commitPhysicalMemory(void* address, size_t size)
    {
        int result = mprotect(address, size, PROT_READ | PROT_WRITE);
        return result == 0;
    }

    void decommitPhysicalMemory(void* pageAddress, size_t pageSize)
    {
        mprotect(pageAddress, pageSize, PROT_NONE);
        madvise(pageAddress, pageSize, MADV_DONTNEED); // Optional: Release physical memory
    }

    bool freeVirtualAddressSpace(void* address, size_t size) // true if success
    {
        return !munmap(address, size);
    }

    void* allocateLockedLargePages(void* address, size_t size, EPageSize pageSize, bool skipAclCheck) { return false; }

    void deallocateLockedMemoryLargePages(void* address, size_t size, EPageSize pageSize) {}
} // namespace dmt::os

namespace dmt {
    inline constexpr uint32_t     howMany4KBsIn1GB = toUnderlying(EPageSize::e1GB) / toUnderlying(EPageSize::e4KB);
    static thread_local page_info pageInfoPool[howMany4KBsIn1GB];
    static thread_local uint64_t  bitsPool[howMany4KBsIn1GB];

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

    PageAllocator::PageAllocator(LoggingContext& ctx, PageAllocatorHooks const& hooks) : PageAllocator(hooks)
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

    PageAllocation PageAllocator::allocatePage(LoggingContext& ctx, EPageSize sizeOverride)
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
        ctx.trace(
            "Called allocatePage, allocated "
            "at {} page of {}",
            {ret.address, (void*)toUnderlying(ret.pageSize)});
        ctx.dbgTraceStackTrace();
        return ret;
    }

    void PageAllocator::deallocatePage(LoggingContext& ctx, PageAllocation& alloc)
    {
        PageAllocation allocCopy = alloc;
        if ((allocCopy.bits & DMT_OS_LINUX_MMAP_ALLOCATED) != 0)
        {
            if (munmap(allocCopy.address, toUnderlying(allocCopy.pageSize) * allocCopy.count))
            {
                ctx.error("Couldn't deallocate {}", {allocCopy.address});
                ctx.dbgErrorStackTrace();
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
            ctx.trace("You called deallocatePage, but nothing done.");
            ctx.dbgTraceStackTrace();
        }

        ctx.trace("Deallocated memory at {} size {}", {allocCopy.address, toUnderlying(allocCopy.pageSize)});
        ctx.dbgTraceStackTrace();
    }

    bool PageAllocator::allocate2MB(LoggingContext& ctx, PageAllocation& out)
    {
        size_t const size        = toUnderlying(EPageSize::e2MB);
        bool         isLargePage = false;

        out.address = nullptr;

        if (m_mmapHugeTlbEnabled)
        {
            static constexpr int32_t hugeFlags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED | MAP_POPULATE |
                                                 MAP_NONBLOCK | MAP_HUGETLB;
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
    PageAllocator::~PageAllocator() {}


} // namespace dmt