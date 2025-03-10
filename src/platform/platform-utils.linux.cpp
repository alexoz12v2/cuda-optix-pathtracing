#include "platform-utils.h"
#include "platform-os-utils.linux.h"

#include <unistd.h>

namespace dmt::os {
    uint64_t processId()
    {
        uint64_t    ret = 0;
        pid_t const pid = getpid();
        ret             = static_cast<uint64_t>(pid);
        return ret;
    }

    uint64_t threadId()
    {
        uint64_t ret = 0;
        ret          = static_cast<uint64_t>(gettid());
        return ret;
    }

    void* reserveVirtualAddressSpace(size_t size)
    {
        void* address = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (address == MAP_FAILED)
        {
            return nullptr;
        }
        return address;
    }

    size_t systemAlignment()
    {
        // TODO
        return 0;
    }

    bool commitPhysicalMemory(void* address, size_t size)
    {
        int result = mprotect(address, size, PROT_READ | PROT_WRITE);
        return result == 0;
    }

    bool freeVirtualAddressSpace(void* address, size_t size) // true if success
    {
        return !munmap(address, size);
    }

    void decommitPage(void* pageAddress, size_t pageSize)
    {
        mprotect(pageAddress, pageSize, PROT_NONE);
        madvise(pageAddress, pageSize, MADV_DONTNEED); // Optional: Release physical memory
    }

    void* allocate(size_t _bytes, size_t _align) { return std::aligned_alloc(_bytes, _align); }

    void deallocate(void* ptr, [[maybe_unused]] size_t _bytes, [[maybe_unused]] size_t _align) { std::free(ptr); }

    std::vector<std::pair<std::u8string, std::u8string>> getEnv()
    {
        std::vector<std::pair<std::u8string, std::u8string>> vec;
        return vec;
    }
} // namespace dmt::os
