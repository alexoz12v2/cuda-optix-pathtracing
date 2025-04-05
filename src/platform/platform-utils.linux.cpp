#include "platform-utils.h"
#include "platform-os-utils.linux.h"

#include <unistd.h>

namespace dmt::os {
    // TODO switch to uint32_t
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

    size_t systemAlignment()
    {
        // TODO
        return 0;
    }

    void* allocate(size_t _bytes, size_t _align) { return std::aligned_alloc(_bytes, _align); }

    void deallocate(void* ptr, [[maybe_unused]] size_t _bytes, [[maybe_unused]] size_t _align) { std::free(ptr); }

    std::vector<std::pair<std::u8string, std::u8string>> getEnv()
    {
        std::vector<std::pair<std::u8string, std::u8string>> vec;
        return vec;
    }
} // namespace dmt::os
