#include "platform-memory.h"
#include "platform-os-utils.linux.h"
#include <sys/mman.h>
#include <unistd.h>
#include <sys/resource.h>
#include <string.h>
#include <errno.h>
#include <string>
#include <tuple>
#include <iostream>

#include "platform-context.h"

namespace dmt::os {

    static std::string getLastError() { return std::string(strerror(errno)); }

    // Utility to check if a hugepage size is available by looking into sysfs
    static bool hugepageSizeAvailable(size_t pageSize)
    {
        std::string path = "/sys/kernel/mm/hugepages/hugepages-" + std::to_string(pageSize / 1024) + "kB";
        return access(path.c_str(), F_OK) == 0;
    }

    void* reserveVirtualAddressSpace(size_t size)
    {
        void* addr = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        return addr == MAP_FAILED ? nullptr : addr;
    }

    bool commitPhysicalMemory(void* address, size_t size)
    {
        return mprotect(address, size, PROT_READ | PROT_WRITE) == 0;
    }

    void decommitPhysicalMemory(void* address, size_t size)
    {
        // Drop physical pages but keep VA range reserved
        madvise(address, size, MADV_DONTNEED);
    }

    bool freeVirtualAddressSpace(void* address, size_t size) { return munmap(address, size) == 0; }

    void* allocateLockedLargePages(size_t size, EPageSize pageSize, bool skipAclCheck)
    {
        Context ctx;
        size_t  hugePageSize = 0;
        switch (pageSize)
        {
            case EPageSize::e2MB: hugePageSize = 2ULL * 1024 * 1024; break;
            case EPageSize::e1GB: hugePageSize = 1ULL * 1024 * 1024 * 1024; break;
            default: return nullptr;
        }

        if (!hugepageSizeAvailable(hugePageSize))
        {
            if (ctx.isValid() && ctx.isErrorEnabled())
                ctx.error("Hugepage size {} kB not available on this system.", std::make_tuple((hugePageSize / 1024)));
            return nullptr;
        }

        // Align size to huge page
        if (size % hugePageSize != 0)
        {
            if (ctx.isValid() && ctx.isErrorEnabled())
                ctx.error("Size must be multiple of hugepage size", {});
            return nullptr;
        }

        int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB;
        // optional: request explicit size with MAP_HUGE_SHIFT, but non-portable.
        // flags |= (21 << MAP_HUGE_SHIFT); // 2MB = 2^21
        // flags |= (30 << MAP_HUGE_SHIFT); // 1GB = 2^30

        void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, -1, 0);
        if (addr == MAP_FAILED)
        {
            if (ctx.isValid() && ctx.isErrorEnabled())
                ctx.error("mmap failed: {}", std::make_tuple(getLastError()));
            return nullptr;
        }

        if (!skipAclCheck)
        {
            // lock into RAM, requires CAP_IPC_LOCK or RLIMIT_MEMLOCK
            if (mlock(addr, size) != 0)
            {
                if (ctx.isValid() && ctx.isErrorEnabled())
                    ctx.error("mlock failed: {}", std::make_tuple(getLastError()));
                munmap(addr, size);
                return nullptr;
            }
        }

        return addr;
    }

    void deallocateLockedLargePages(void* address, size_t size, [[maybe_unused]] EPageSize pageSize)
    {
        Context ctx;
        // unlock and unmap
        if (munlock(address, size) != 0)
        {
            if (ctx.isValid() && ctx.isErrorEnabled())
                ctx.error("munlock failed: {}", std::make_tuple(getLastError()));
        }
        if (munmap(address, size) != 0)
        {
            if (ctx.isValid() && ctx.isErrorEnabled())
                ctx.error("munmap failed: {}", std::make_tuple(getLastError()));
        }
    }
} // namespace dmt::os
