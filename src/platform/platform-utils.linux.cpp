#include "platform-utils.h"
#include "platform-os-utils.linux.h"

#include <backward.hpp>
#include <fcntl.h>
#include <memory_resource>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>
#include <cstdint>

#include <sys/types.h>
#include <pwd.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/stat.h>

namespace dmt::os {
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

    namespace env {
        // read /proc/self/environ fully into a pmr buffer
        static std::pmr::string readEnviron(std::pmr::memory_resource* resource)
        {
            int fd = open("/proc/self/environ", O_RDONLY);
            if (fd == -1)
                return {};

            // read file
            char             buffer[4096];
            std::pmr::string content(resource);
            ssize_t          n;
            while ((n = read(fd, buffer, sizeof(buffer))) > 0)
            {
                content.append(buffer, n);
            }
            close(fd);
            return content;
        }

        bool set(std::string_view name, std::string_view value, std::pmr::memory_resource* /*resource*/)
        {
            // libc call is the only supported way
            return setenv(std::string(name).c_str(), std::string(value).c_str(), 1) == 0;
        }

        bool remove(std::string_view name, std::pmr::memory_resource* /*resource*/)
        {
            return unsetenv(std::string(name).c_str()) == 0;
        }

        std::pmr::string get(std::string_view name, std::pmr::memory_resource* resource)
        {
            std::pmr::string envBlock = readEnviron(resource);
            if (envBlock.empty())
                return {};

            char const* data = envBlock.data();
            size_t      size = envBlock.size();

            char const* end = data + size;
            char const* p   = data;

            while (p < end)
            {
                char const* eq = (char const*)memchr(p, '=', end - p);
                if (!eq)
                    break;

                auto const keyLen = static_cast<size_t>(eq - p);
                if (keyLen == name.size() && memcmp(p, name.data(), keyLen) == 0)
                {
                    // found match
                    char const* valStart = eq + 1;
                    char const* valEnd   = reinterpret_cast<char const*>(
                        memchr(valStart, '\0', static_cast<size_t>(end - valStart)));

                    if (!valEnd)
                        valEnd = end;
                    return std::pmr::string(valStart, valEnd, resource);
                }

                // skip to next entry (null terminated)
                char const* next = reinterpret_cast<char const*>(memchr(p, '\0', static_cast<size_t>(end - p)));
                if (!next)
                    break;
                p = next + 1;
            }

            return {};
        }
    } // namespace env

    std::vector<std::string> cmdLine()
    {
        int const fd = open("/proc/self/cmdline", O_RDONLY);
        if (fd == -1)
            return {};

        std::string buf{};
        size_t      marker  = 0;
        ssize_t     numRead = 0;

        buf.resize(64);

        while ((numRead = read(fd, buf.data() + marker, 64)) > 0)
        {
            buf.resize(buf.size() + 64);
            marker += static_cast<size_t>(numRead);
        }
        buf.shrink_to_fit();

        std::vector<std::string> args;

        marker = 0;
        while (marker < buf.size())
        {
            size_t pos = buf.find_first_of('\0', marker);
            if (pos != std::string::npos && pos > marker)
            {
                args.emplace_back(buf.data() + marker, buf.data() + pos);
            }
            marker = pos + 1;
        }

        return args;
    }

    // Path ----------------------------------------------------------------------------------------------------------
    static bool exists(char const* path)
    {
        struct stat sb{};
        return stat(path, &sb) == 0 && (S_ISDIR(sb.st_mode) || S_ISREG(sb.st_mode));
    }

    static bool isDir(char const* path)
    {
        struct stat sb{};
        return stat(path, &sb) == 0 && S_ISDIR(sb.st_mode);
    }

    static void normalizePath(char* path)
    {
        char* src = path;
        char* dst = path;

        // absolute or relative handling
        if (*src == '/')
        {
            *dst++ = '/';
            while (*src == '/')
                src++;
        }

        char* stack[PATH_MAX]; // pointers to start of components
        int   depth = 0;

        while (*src)
        {
            char* seg = src;
            while (*src && *src != '/')
                src++;

            int segLen = static_cast<int>(src - seg);
            while (*src == '/')
                src++; // skip multiple slashes

            if (segLen == 0 || (segLen == 1 && seg[0] == '.'))
            {
                // skip "."
                continue;
            }
            if (segLen == 2 && seg[0] == '.' && seg[1] == '.')
            {
                // handle ".."
                if (depth > 0)
                {
                    dst = stack[--depth];
                }
                else
                {
                    // preserve leading ".." if relative
                    if (dst != path && *(dst - 1) != '/')
                        *dst++ = '/';
                    *dst++ = '.';
                    *dst++ = '.';
                }
                continue;
            }

            // copy segment
            if (dst != path && *(dst - 1) != '/')
                *dst++ = '/';
            stack[depth++] = dst;
            for (int i = 0; i < segLen; i++)
                *dst++ = seg[i];
        }

        // remove trailing slash (except root "/")
        if (dst - path > 1 && *(dst - 1) == '/')
            dst--;
        *dst = '\0';
    }


    Path::Path(std::pmr::memory_resource* resource, void* content, uint32_t capacity, uint32_t size) :
    m_resource(resource),
    m_data(content),
    m_capacity(capacity),
    m_dataSize(size),
    m_isDir(isDir(reinterpret_cast<char*>(content))),
    m_valid(content != nullptr && capacity > 0 && size > 0 && exists(reinterpret_cast<char*>(content)))
    {
    }

    Path Path::home(std::pmr::memory_resource* resource)
    {
        if (!resource)
            resource = std::pmr::get_default_resource();

        char const* homeEnv  = getenv("HOME");
        char const* homePath = nullptr;
        if (homeEnv && *homeEnv)
        {
            homePath = homeEnv;
        }
        else
        {
            // Fallback to passwd
            struct passwd* pw = getpwuid(getuid());
            if (pw)
                homePath = pw->pw_dir;
        }

        if (!homePath)
            return invalid(resource);

        size_t len    = strlen(homePath);
        char*  buffer = reinterpret_cast<char*>(resource->allocate(len + 1));
        memcpy(buffer, homePath, len);
        buffer[len] = '\0';

        return Path{resource, buffer, static_cast<uint32_t>(len + 1), static_cast<uint32_t>(len)};
    }

    Path Path::cwd(std::pmr::memory_resource* resource)
    {
        if (!resource)
            resource = std::pmr::get_default_resource();

        char tmp[PATH_MAX];
        if (!getcwd(tmp, sizeof(tmp)))
            return invalid(resource);

        size_t len    = strlen(tmp);
        char*  buffer = reinterpret_cast<char*>(resource->allocate(len + 1));
        memcpy(buffer, tmp, len);
        buffer[len] = '\0';

        return Path{resource, buffer, static_cast<uint32_t>(len + 1), static_cast<uint32_t>(len)};
    }

    Path Path::invalid(std::pmr::memory_resource* resource) { return Path{resource, nullptr, 0, 0}; }

    Path Path::root(char const* diskDesignator, std::pmr::memory_resource* resource)
    {
        if (!resource)
            resource = std::pmr::get_default_resource();

        // On Linux, "root" is always "/"
        (void)diskDesignator; // ignored

        char const* rootPath = "/";
        size_t      len      = 1;
        char*       buffer   = reinterpret_cast<char*>(resource->allocate(len + 1));
        buffer[0]            = '/';
        buffer[1]            = '\0';

        return Path{resource, buffer, static_cast<uint32_t>(len + 1), static_cast<uint32_t>(len)};
    }

    Path Path::executableDir(std::pmr::memory_resource* resource)
    {
        if (!resource)
            resource = std::pmr::get_default_resource();

        char    tmp[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", tmp, sizeof(tmp) - 1);
        if (len <= 0)
            return invalid(resource);

        tmp[len] = '\0';

        // Strip the executable name
        for (ssize_t i = len - 1; i >= 0; --i)
        {
            if (tmp[i] == '/')
            {
                tmp[i] = '\0';
                len    = i;
                break;
            }
        }

        char* buffer = reinterpret_cast<char*>(resource->allocate(PATH_MAX));
        memcpy(buffer, tmp, len);
        buffer[len] = '\0';

        return Path{resource, buffer, static_cast<uint32_t>(len + 1), static_cast<uint32_t>(len)};
    }

    Path Path::fromString(std::string_view str, std::pmr::memory_resource* resource)
    {
        if (!resource)
            resource = std::pmr::get_default_resource();
        if (str.empty())
            return invalid(resource);

        // allocate size + 1 ('\0')
        size_t const len    = str.size();
        char*        buffer = reinterpret_cast<char*>(resource->allocate(PATH_MAX));
        if (!buffer)
            return invalid(resource);

        std::memcpy(buffer, str.data(), len);
        buffer[len] = '\0';

        normalizePath(buffer);

        if (exists(buffer))
            return Path{resource, buffer, PATH_MAX, static_cast<uint32_t>(len + 1)};

        return invalid(resource);
    }

    std::pmr::string Path::toUnderlying(std::pmr::memory_resource* resource) const
    {
        if (!resource)
            resource = m_resource;

        std::pmr::string result{resource};
        if (!m_data || m_dataSize == 0)
            return result;

        // On Linux, the internal data is UTF-8 char*
        result.assign(reinterpret_cast<char const*>(m_data), m_dataSize);
        return result;
    }

    // Copy constructor
    Path::Path(Path const& other) :
    m_resource(other.m_resource),
    m_capacity(other.m_capacity),
    m_dataSize(other.m_dataSize),
    m_isDir(other.m_isDir),
    m_valid(other.m_valid)
    {
        if (other.m_data)
        {
            m_data = m_resource->allocate(m_capacity);
            memcpy(m_data, other.m_data, m_dataSize);
            static_cast<char*>(m_data)[m_dataSize] = '\0';
        }
        else
            m_data = nullptr;
    }

    // Copy assignment
    Path& Path::operator=(Path const& other)
    {
        if (this != &other)
        {
            if (m_data)
                m_resource->deallocate(m_data, m_capacity);

            m_capacity = other.m_capacity;
            m_dataSize = other.m_dataSize;
            m_isDir    = other.m_isDir;
            m_valid    = other.m_valid;
            m_resource = other.m_resource;

            if (other.m_data)
            {
                m_data = m_resource->allocate(m_capacity);
                memcpy(m_data, other.m_data, m_dataSize);
                static_cast<char*>(m_data)[m_dataSize] = '\0';
            }
            else
                m_data = nullptr;
        }
        return *this;
    }

    // Move constructor
    Path::Path(Path&& other) noexcept :
    m_resource(std::exchange(other.m_resource, nullptr)),
    m_data(std::exchange(other.m_data, nullptr)),
    m_capacity(std::exchange(other.m_capacity, 0)),
    m_dataSize(std::exchange(other.m_dataSize, 0)),
    m_isDir(std::exchange(other.m_isDir, false)),
    m_valid(std::exchange(other.m_valid, false))
    {
    }

    // Move assignment
    Path& Path::operator=(Path&& other) noexcept
    {
        if (this != &other)
        {
            if (m_data)
                m_resource->deallocate(m_data, m_capacity);

            m_resource = std::exchange(other.m_resource, nullptr);
            m_data     = std::exchange(other.m_data, nullptr);
            m_capacity = std::exchange(other.m_capacity, 0);
            m_dataSize = std::exchange(other.m_dataSize, 0);
            m_isDir    = std::exchange(other.m_isDir, false);
            m_valid    = std::exchange(other.m_valid, false);
        }
        return *this;
    }

    // Destructor
    Path::~Path() noexcept
    {
        if (m_data)
            m_resource->deallocate(m_data, m_capacity);
    }

    // remove last component in place
    void Path::parent_()
    {
        if (!m_data)
            return;

        char*    path = reinterpret_cast<char*>(m_data);
        uint32_t len  = m_dataSize;

        // remove trailing slashes
        while (len > 1 && path[len - 1] == '/')
        {
            path[--len] = '\0';
        }

        // find last slash
        char* lastSlash = nullptr;
        for (int i = static_cast<int>(len - 1); i >= 0; --i)
        {
            if (path[i] == '/')
            {
                lastSlash = &path[i];
                break;
            }
        }

        if (lastSlash)
        {
            if (lastSlash == path)
            {
                // root "/"
                path[1]    = '\0';
                m_dataSize = 1;
            }
            else
            {
                *lastSlash = '\0';
                m_dataSize = static_cast<uint32_t>(lastSlash - path);
            }
        }

        m_isDir = true;
        m_valid = exists(path);
    }

    Path Path::parent() const
    {
        Path copy(*this);
        copy.parent_();
        return copy;
    }

    // append + normalize
    void Path::operator/=(char const* pathComponent)
    {
        if (!pathComponent || !*pathComponent)
            return;
        char* base = reinterpret_cast<char*>(m_data);

        // compute needed size
        uint32_t baseLen = m_dataSize;
        uint32_t compLen = 0;
        for (char const* p = pathComponent; *p; ++p)
            compLen++;

        uint32_t needLen = static_cast<uint32_t>(baseLen) + 1 + compLen + 1; // slash + comp + null

        if (needLen > m_capacity)
        {
            uint32_t newCap = static_cast<uint32_t>(needLen) + 32;
            char*    newBuf = reinterpret_cast<char*>(m_resource->allocate(newCap));
            if (m_data)
            {
                memcpy(newBuf, m_data, m_dataSize);
                m_resource->deallocate(m_data, m_capacity);
            }
            m_data     = newBuf;
            m_capacity = newCap;
            base       = newBuf;
        }

        // add slash if missing
        if (baseLen > 0 && base[baseLen - 1] != '/')
        {
            base[baseLen++] = '/';
        }

        // copy component
        memcpy(base + baseLen, pathComponent, compLen);
        baseLen += compLen;
        base[baseLen] = '\0';

        // normalize
        normalizePath(base);
        m_dataSize = static_cast<uint32_t>(strlen(base));

        m_valid = exists(base);
        m_isDir = isDir(base);
    }


    Path Path::operator/(char const* pathComponent) const
    {
        Path copy(*this);
        copy /= pathComponent;
        return copy;
    }

    FileStat Path::stat() const
    {
        FileStat info;
        if (!m_valid)
            return info;

        char const* cpath = static_cast<char const*>(m_data);
        struct stat st{};
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

    // Basic File Management -----------------------------------------------------------------------------------------
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

    // Library Loader ------------------------------------------------------------------------------------------------
    void* LibraryLoader::loadLibrary(std::string_view name, bool useSystemPaths, Path const* pathOverride) const
    {
        void* result = nullptr;
        if (name.empty())
            return result;

        if (pathOverride)
        {
            Path const  path    = *pathOverride / name.data();
            char const* pathLib = reinterpret_cast<char const*>(path.internalData());

            result = dlopen(pathLib, RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);
        }
        else if (m_searchPathsLen > 0)
        {
            std::string path;
            for (uint32_t i = 0; i < m_searchPathsLen && !result; ++i)
            {
                path = reinterpret_cast<char const*>(m_searchPaths[i].internalData());
                if (!path.ends_with('/'))
                    path.push_back('/');
                path.append(name);

                result = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);
            }
        }

        if (!result && useSystemPaths)
        {
            result = dlopen(name.data(), RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);
        }

        return result;
    }

    bool LibraryLoader::unloadLibrary(void* library) const
    {
        if (!library)
            return false;

        return dlclose(library) == 0;
    }

    // library utils
    namespace lib {
        void* getFunc(void* library, char const* funcName)
        {
            if (!library)
                return nullptr;
            return dlsym(library, funcName);
        }
    } // namespace lib

    size_t systemAlignment()
    {
        // we need to recover the smallest page size. There's `sysconf(_SC_PAGESIZE)`, but it
        // gives the "default" page size. Good enough. Other sizes are in linux at `/sys/kernel/mm/hugepages` (and transparent)
        return static_cast<size_t>(sysconf(_SC_PAGESIZE));
    }

    void* allocate(size_t _bytes, size_t _align) { return std::aligned_alloc(_align, _bytes); }

    void deallocate(void* ptr, [[maybe_unused]] size_t _bytes, [[maybe_unused]] size_t _align) { std::free(ptr); }

    std::pmr::vector<std::pair<std::pmr::string, std::pmr::string>> getEnv(std::pmr::memory_resource* resource)
    {
        std::pmr::vector<std::pair<std::pmr::string, std::pmr::string>> vec{resource};

        int const fd = open("/proc/self/environ", O_RDONLY);
        if (fd < 0)
            return vec;

        std::vector<uint8_t> buffer;
        buffer.reserve(1024);
        size_t const chunkSize = 4096; // read in chunks
        while (true)                   // until EOF
        {
            size_t const currentSize = buffer.size();
            buffer.resize(buffer.size() + chunkSize);
            ssize_t bytesRead = read(fd, buffer.data() + currentSize, chunkSize);
            if (bytesRead < 0) // error
            {
                close(fd);
                return vec;
            }

            if (bytesRead == 0) // EOF reached
            {
                buffer.resize(currentSize);
                break;
            }

            buffer.resize(currentSize + static_cast<uint64_t>(bytesRead));
        }
        // close proc pseudofile
        close(fd);

        // Now parse the contents of the buffer, which contains the full environ data.
        std::string_view const bufferView(reinterpret_cast<char const*>(buffer.data()), buffer.size());

        size_t currentPos = 0;
        while (currentPos < bufferView.size())
        {
            size_t nextNull = bufferView.find('\0', currentPos);
            if (nextNull == std::string_view::npos)
                nextNull = bufferView.size();

            std::string_view entryView = bufferView.substr(currentPos, nextNull);
            if (!entryView.empty())
            {
                size_t const equalPos = entryView.find('=');
                if (equalPos != std::u8string_view::npos)
                {
                    std::pmr::string key{entryView.substr(0, equalPos), resource};
                    std::pmr::string value(entryView.substr(equalPos + 1), resource);
                    vec.emplace_back(std::move(key), std::move(value));
                }
            }

            currentPos = nextNull + 1;
        }
        return vec;
    }
} // namespace dmt::os
