#include "platform-utils.h"
#include "platform-os-utils.linux.h"

#include <unistd.h>

namespace dmt::os {
    // TODO switch to uint32_t
    uint32_t processId()
    {
        uint32_t    ret = 0;
        pid_t const pid = getpid();
        ret             = static_cast<uint32_t>(pid);
        return ret;
    }

    uint32_t threadId()
    {
        uint32_t ret = 0;
        ret          = static_cast<uint32_t>(gettid());
        return ret;
    }

    size_t systemAlignment()
    {
        // TODO
        return 0;
    }

    FileStat Path::stat() const
    {
        FileStat info;
        if (!m_valid)
            return info;

        char const* cpath = static_cast<char const*>(m_data);
        struct stat st;
        if (::stat(cpath, &st) != 0)
            return info;

        info.valid        = true;
        info.isDirectory  = S_ISDIR(st.st_mode);
        info.size         = static_cast<uint64_t>(st.st_size);
        info.accessTime   = static_cast<uint64_t>(st.st_atime);
        info.modifiedTime = static_cast<uint64_t>(st.st_mtime);

#ifdef __APPLE__
        info.creationTime = static_cast<uint64_t>(st.st_birthtime);
#else
        info.creationTime = static_cast<uint64_t>(st.st_ctime); // best effort on Linux (time of last touch)
#endif

        return info;
    }

    std::pmr::string readFileContents(Path const& path, std::pmr::memory_resource* resource)
    {
        std::pmr::string str{resource};
        if (!path.isFile())
            return str;

        FileStat stats    = path.stat();
        uint64_t fileSize = stats.size;
        if (fileSize == 0)
            return str;

        str.resize(static_cast<size_t>(fileSize));
        char const* cpath = static_cast<char const*>(path.internalData());
        int         fd    = open(cpath, O_RDONLY);
        if (fd < 0)
            return std::pmr::string{resource};

        ssize_t bytesRead = ::read(fd, str.data(), static_cast<size_t>(fileSize));
        close(fd);

        if (bytesRead < 0 || static_cast<size_t>(bytesRead) != fileSize)
            return std::pmr::string{resource};

        return str;
    }


    void* allocate(size_t _bytes, size_t _align) { return std::aligned_alloc(_bytes, _align); }

    void deallocate(void* ptr, [[maybe_unused]] size_t _bytes, [[maybe_unused]] size_t _align) { std::free(ptr); }

    std::vector<std::pair<std::u8string, std::u8string>> getEnv()
    {
        std::vector<std::pair<std::u8string, std::u8string>> vec;
        return vec;
    }
} // namespace dmt::os
