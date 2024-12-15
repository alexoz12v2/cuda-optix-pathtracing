module;

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>

#if defined(DMT_OS_LINUX)
#include <fcntl.h>
#include <linux/kernel-page-flags.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdlib>
#elif defined(DMT_OS_WINDOWS)
#endif

module platform;

namespace dmt
{

#if defined(DMT_DEBUG)
void internStringToCurrent(char const* str, uint64_t sz)
{
    // TODO implement table switching mechanism
}
#endif

PageAllocator::PageAllocator(PlatformContext& ctx, EPageSize preference)
{
#if defined(DMT_OS_LINUX)
    // 1. Make sure transparent huge pages are enabled
    char const* path = "/sys/kernel/mm/transparent_hugepage/enabled";
    char        buffer[256];
    FILE*       file = fopen(path, "r");
    if (!file || !fgets(buffer, sizeof(buffer), file))
    {
        ctx.warn("Couldn't read from file {}, using default 4KB page size", {path});
        return;
    }

    fclose(file);
    if (strstr(buffer, "[always]") || strstr(buffer, "[madvise]"))
    {
        m_enabledPageSize = preference;
        if (ctx.logEnabled())
            ctx.log("Using {} as page size", {toUnderlying(preference)});
    }
#elif defined(DMT_OS_WINDOWS)
#endif
}

#define PAGEMAP_PRESENT(ent) (((ent) & (1ull << 63)) != 0)
#define PAGEMAP_PFN(ent) ((ent) & ((1ull << 55) - 1))
static void checkHugePage(void const* ptr, uint32_t pageSize)
{
    int32_t pagemapFd    = open("/proc/self/pagemap", O_RDONLY);
    int32_t kpageflagsFd = open("/proc/kpageflags", O_RDONLY);
    assert(pagemapFd >= 0 && "Couldn't open /proc/self/pagemap");
    assert(kpageflagsFd >= 0 && "Couldn't open /proc/self/pagemap");

    uint64_t ent;
    if (pread(pagemapFd, &ent, sizeof(ent), ((uintptr_t) ptr) / pageSize * 8) != sizeof(ent)) {
        assert(false && "could not read from pagemap\n");
    }
    
    if (!PAGEMAP_PRESENT(ent)) {
        assert(false && "page not present in /proc/self/pagemap, did you allocate it?\n");
    }
    if (!PAGEMAP_PFN(ent)) {
        assert(false && "page frame number not present, run this program as root\n");
    }

    uint64_t flags;
    if (pread(kpageflagsFd, &flags, sizeof(flags), PAGEMAP_PFN(ent) << 3) != sizeof(flags)) {
        assert(false && "could not read from kpageflags\n");
    }

    if (!(flags & (1ull << KPF_THP))) {
        assert(false && "could not allocate huge page\n");
    }

    if (close(pagemapFd) < 0) {
        assert(false && "could not close /proc/self/pagemap: %s");
    }

    if (close(kpageflagsFd) < 0) {
        assert(false && "could not close /proc/kpageflags: %s");
    }}
#undef PAGEMAP_PRESENT
#undef PAGEMAP_PFN

PageAllocation PageAllocator::allocatePages(PlatformContext& ctx, uint32_t numPages)
{
#if defined(DMT_OS_LINUX)
    uint32_t size     = toUnderlying(m_enabledPageSize) * numPages;
    uint32_t pageSize = toUnderlying(m_enabledPageSize); // used as alignment
    // 1. reserve, not effectively allocate, necessary virtual address space
    void* buf = reinterpret_cast<unsigned char*>(aligned_alloc(pageSize, numPages));

    // 2. tag the memory as huge page (required for madvise, not for always)
    if (m_enabledPageSize != EPageSize::e4KB)
    {
        madvise(buf, size, MADV_HUGEPAGE);
        // 3. for each page, touch the first byte to be sure to allocate memory, and, if debug,
        // check that that's effectively a huge page
        unsigned char* ptr = buf;
        for (unsigned char* end = buf + size; ptr < end; ptr += pageSize)
        {
            std::memset(ptr, 0, 1);
            checkHugePage(ptr, pageSize);
        }
    }

#elif defined(DMT_OS_WINDOWS)
#endif
}

PageAllocator::~PageAllocator()
{
}

} // namespace dmt