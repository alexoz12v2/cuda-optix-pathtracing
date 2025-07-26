#include "platform-utils.h"

#include "platform-os-utils.win32.h"
#include "platform-memory.h"

#pragma comment(lib, "mincore")
#pragma comment(lib, "Shlwapi.lib")
#include <AclAPI.h>
#include <Windows.h>
#include <errhandlingapi.h>
#include <fileapi.h>
#include <securitybaseapi.h>
#include <sysinfoapi.h>
#include <Shlwapi.h> // PathFileExistsW. use `dumpbin /EXPORTS C:\Windows\System32\Shlwapi.dll` for more details
#undef max
#undef min

#include <utility>

namespace dmt::os {
    uint32_t processId()
    {
        uint32_t const ret = static_cast<uint32_t>(GetCurrentProcessId());
        return ret;
    }

    uint32_t threadId()
    {
        uint32_t const ret = static_cast<uint32_t>(GetCurrentThreadId());
        return ret;
    }

    namespace env {
        bool set(std::string_view name, std::string_view value, std::pmr::memory_resource* resource)
        {
            static constexpr size_t maxWinLength = 32760;
            bool                    result       = false;
            if (name.length() <= 0 || value.length() > maxWinLength)
                return result;

#if defined(_WIN32)
            // TODO switch to smart pointers maybe?
            std::pmr::wstring const wName  = dmt::os::win32::utf16FromUtf8(name, resource);
            std::pmr::wstring const wValue = dmt::os::win32::utf16FromUtf8(value, resource);
            result                         = SetEnvironmentVariableW(wName.c_str(), wValue.c_str());
#else
    #error "not done yet"
#endif
            return result;
        }

        bool remove(std::string_view name, std::pmr::memory_resource* resource)
        {
            std::pmr::wstring const wName = dmt::os::win32::utf16FromUtf8(name, resource);
            return SetEnvironmentVariableW(wName.c_str(), nullptr);
        }

        std::pmr::string get(std::string_view name, std::pmr::memory_resource* resource)
        {
            std::pmr::wstring const wName = dmt::os::win32::utf16FromUtf8(name, resource);
            std::pmr::string        result{resource};
            if (DWORD numChars = GetEnvironmentVariableW(wName.c_str(), nullptr, 0); numChars > 0)
            {
                wchar_t* wBuffer  = reinterpret_cast<wchar_t*>(resource->allocate((numChars + 1) * sizeof(wchar_t)));
                wBuffer[numChars] = L'\0';
                if (GetEnvironmentVariableW(wName.c_str(), wBuffer, numChars) > 0)
                {
                    result.resize(numChars + 10);
                    // clang-format off
                    if (WideCharToMultiByte(
                        CP_UTF8, 0,
                        wBuffer, numChars,
                        result.data(), static_cast<int32_t>(result.size()),
                        nullptr, nullptr)
                        <= 0)
                    {
                        // clang-format on
                        result.clear();
                    }
                }
            }

            return result;
        }
    } // namespace env

    // Path ----------------------------------------------------------------------------------------------------------
    Path::Path(std::pmr::memory_resource* resource, void* content, uint32_t capacity, uint32_t size) :
    m_resource(resource),
    m_data(content),
    m_capacity(capacity),
    m_dataSize(size),
    m_isDir(true),
    m_valid(PathFileExistsW(reinterpret_cast<wchar_t*>(content)) && capacity > 0 && size > 0)
    {
    }

    static void prependLongPath(wchar_t* target)
    {
        static wchar_t const* longPath = L"\\\\?\\";
        memcpy(target, longPath, 8);
    }

    Path Path::home(std::pmr::memory_resource* resource)
    {
        // assuming home is not a long path
        static constexpr uint32_t capacity = (MAX_PATH + 5) * sizeof(wchar_t);
        wchar_t*                  homePath = reinterpret_cast<wchar_t*>(resource->allocate(capacity));
        homePath[MAX_PATH]                 = L'\0';
        prependLongPath(homePath);
        DWORD len = GetEnvironmentVariableW(L"USERPROFILE", homePath + 4, MAX_PATH);
        if (len > 0)
            return Path{resource, homePath, capacity, static_cast<uint32_t>(len * sizeof(wchar_t) + 8)};
        else
        {
            resource->deallocate(homePath, capacity);
            return invalid(resource);
        }
    }

    Path Path::cwd(std::pmr::memory_resource* resource)
    {
        uint32_t numCharsNeeded = GetCurrentDirectoryW(0, nullptr);
        uint32_t capacity       = (numCharsNeeded + 5) * sizeof(wchar_t);
        wchar_t* path           = reinterpret_cast<wchar_t*>(resource->allocate(capacity));
        uint32_t len            = GetCurrentDirectoryW(numCharsNeeded, path + 4);
        // TODO check that there is not an ucn. if there is, then move the path string to the beginning of
        // the buffer, otherwise prepend the long path namespace
        prependLongPath(path);
        if (len > 0)
            return Path{resource, path, capacity, static_cast<uint32_t>(len * sizeof(wchar_t) + 8)};
        else
        {
            resource->deallocate(path, capacity);
            return invalid(resource);
        }
    }

    Path Path::invalid(std::pmr::memory_resource* resource) { return Path{resource, nullptr, 0, 0}; }

    Path Path::root(char const* diskDesignator, std::pmr::memory_resource* resource)
    {
        static constexpr uint32_t maxDesignatorLen = 64;
        uint32_t                  capacity         = (MAX_PATH + 5) * sizeof(wchar_t);
        wchar_t*                  pathBuffer       = reinterpret_cast<wchar_t*>(resource->allocate(capacity));

        if (diskDesignator[0] == '\\' && diskDesignator[1] == '\\')
        {
            // It's a UNC path, verify that it exists and return it (NOT Long path)
            uint32_t lenDesignator = MultiByteToWideChar(CP_UTF8, 0, diskDesignator, -1, pathBuffer, maxDesignatorLen);
            if (lenDesignator > 0 && PathFileExistsW(pathBuffer))
            {
                uint32_t byteSize = (lenDesignator - 1) * sizeof(wchar_t); // Remove null terminator
                return Path{resource, pathBuffer, capacity, byteSize};
            }
        }
        else if (isalpha(diskDesignator[0]) && diskDesignator[1] == ':')
        {
            // It's a drive letter, prepend long path namespace
            constexpr wchar_t longPrefix[]  = L"\\\\?\\";
            constexpr size_t  longPrefixLen = 4;

            wcsncpy(pathBuffer, longPrefix, longPrefixLen);
            pathBuffer[longPrefixLen]     = static_cast<wchar_t>(toupper(diskDesignator[0])); // Drive letter
            pathBuffer[longPrefixLen + 1] = L':';
            pathBuffer[longPrefixLen + 2] = L'\\';
            pathBuffer[longPrefixLen + 3] = L'\0'; // Null terminator

            uint32_t byteSize = (longPrefixLen + 3) * sizeof(wchar_t); // Correct byte size

            if (GetDriveTypeW(pathBuffer + longPrefixLen) != DRIVE_NO_ROOT_DIR)
                return Path{resource, pathBuffer, capacity, byteSize};
        }

        // Invalid path
        resource->deallocate(pathBuffer, capacity);
        return invalid(resource);
    }

    Path Path::executableDir(std::pmr::memory_resource* resource)
    {
        static constexpr wchar_t longPrefix[]  = L"\\\\?\\";
        static size_t const      longPrefixLen = wcslen(longPrefix);

        // Reserve space for the long path prefix *before* calling GetModuleFileNameW
        DWORD    capacity = (MAX_PATH + longPrefixLen + 1) * sizeof(wchar_t);
        wchar_t* exePath  = reinterpret_cast<wchar_t*>(resource->allocate(capacity));
        DWORD    len      = 0;

        while (true)
        {
            len = GetModuleFileNameW(nullptr, exePath + longPrefixLen, (capacity / sizeof(wchar_t)) - longPrefixLen - 1);
            if (len > 0 && len < (capacity / sizeof(wchar_t)) - longPrefixLen - 1)
            {
                len += longPrefixLen; // Account for the long prefix if added
                exePath[len] = L'\0'; // Ensure null termination
                break;                // Successfully retrieved full path
            }

            DWORD lastErr = GetLastError();
            if (lastErr != ERROR_INSUFFICIENT_BUFFER)
            {
                resource->deallocate(exePath, capacity);
                return invalid(resource); // Failed to retrieve path
            }

            // Increase buffer size and retry
            resource->deallocate(exePath, capacity);
            capacity += 32 * sizeof(wchar_t);
            exePath = reinterpret_cast<wchar_t*>(resource->allocate(capacity));
        }

        // Check if the path already has the long path prefix
        bool hasLongPrefix = wcsncmp(exePath + longPrefixLen, longPrefix, longPrefixLen) == 0;
        if (!hasLongPrefix)
        {
            memcpy(exePath, longPrefix, longPrefixLen * sizeof(wchar_t)); // Prepend "\\?\"
        }
        else
        {
            memmove(exePath, exePath + longPrefixLen, (len + 1) * sizeof(wchar_t)); // Shift back if prefix exists
            len -= longPrefixLen; // Adjust length since we're removing the prefix
        }

        // Remove the executable name (last path component)
        for (wchar_t* p = exePath + len; p >= exePath; --p)
        {
            if (*p == L'\\' || *p == L'/')
            {
                *p  = L'\0';
                len = (p - exePath); // Adjust len to reflect new path size
                break;
            }
        }

        return Path{resource, exePath, capacity, static_cast<uint32_t>(len * sizeof(wchar_t))};
    }

    std::pmr::string Path::toUnderlying(std::pmr::memory_resource* resource) const
    {
        if (resource == nullptr)
            resource = m_resource;

        std::pmr::string buffer{resource};

        // Estimate worst-case required space (UTF-16 -> UTF-8 conversion)
        int32_t estimatedSize = WideCharToMultiByte(CP_UTF8,
                                                    0,
                                                    reinterpret_cast<wchar_t const*>(m_data),
                                                    m_dataSize >> 1,
                                                    nullptr,
                                                    0,
                                                    nullptr,
                                                    nullptr);
        if (estimatedSize > 0)
        {
            buffer.resize(estimatedSize); // Resize first so it manages its own buffer

            // Perform actual conversion into the preallocated buffer
            int32_t actualSize = WideCharToMultiByte(CP_UTF8,
                                                     0,
                                                     reinterpret_cast<wchar_t const*>(m_data),
                                                     m_dataSize >> 1,
                                                     buffer.data(),
                                                     estimatedSize,
                                                     nullptr,
                                                     nullptr);

            if (actualSize <= 0)
                buffer.clear(); // Clear on failure
            else
                buffer.resize(actualSize); // Set final size to actual converted length
        }

        return buffer;
    }

    // Copy Constructor
    Path::Path(Path const& other) :
    m_resource(other.m_resource),
    m_capacity(other.m_capacity),
    m_dataSize(other.m_dataSize),
    m_isDir(other.m_isDir),
    m_valid(other.m_valid)
    {
        m_data = m_resource->allocate(m_capacity);
        memcpy(m_data, other.m_data, m_dataSize);
        memset(static_cast<char*>(m_data) + other.m_dataSize, 0, 2);
    }

    // Copy Assignment
    Path& Path::operator=(Path const& other)
    {
        if (this != &other) // Avoid self-assignment
        {
            void* newData = m_resource->allocate(other.m_capacity);
            memcpy(newData, other.m_data, other.m_dataSize);
            memset(static_cast<char*>(m_data) + other.m_dataSize, 0, 2);

            // Free old memory
            m_resource->deallocate(m_data, m_capacity);

            // Assign new values
            m_data     = newData;
            m_capacity = other.m_capacity;
            m_dataSize = other.m_dataSize;
            m_isDir    = other.m_isDir;
            m_valid    = other.m_valid;
            m_resource = other.m_resource;
        }
        return *this;
    }

    // Move Constructor
    Path::Path(Path&& other) noexcept :
    m_resource(std::exchange(other.m_resource, nullptr)),
    m_data(std::exchange(other.m_data, nullptr)),
    m_capacity(std::exchange(other.m_capacity, 0)),
    m_dataSize(std::exchange(other.m_dataSize, 0)),
    m_isDir(std::exchange(other.m_isDir, false)),
    m_valid(std::exchange(other.m_valid, false))
    {
    }

    // Move Assignment
    Path& Path::operator=(Path&& other) noexcept
    {
        if (this != &other) // Avoid self-assignment
        {
            // Free existing memory
            if (m_data)
                m_resource->deallocate(m_data, m_capacity);

            // Steal resources
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
    Path::~Path()
    {
        if (m_data)
            m_resource->deallocate(m_data, m_capacity);
    }

    // Removes the last component of the path (modifies in place)
    void Path::parent_()
    {
        wchar_t* path = reinterpret_cast<wchar_t*>(m_data);
        int32_t  len  = m_dataSize / sizeof(wchar_t);

        // Handle long path namespace case (\\?\C:\)
        constexpr wchar_t longPrefix[]  = L"\\\\?\\";
        constexpr size_t  longPrefixLen = 4;

        if (len <= longPrefixLen + 2 && wcsncmp(path, longPrefix, longPrefixLen) == 0)
            return; // Don't modify root path (e.g., \\?\C:\)

        // Handle UNC root case (\\Server\Share)
        if (path[0] == L'\\' && path[1] == L'\\' && path[2] != L'?')
        {
            int slashCount = 0;
            for (int i = 2; i < len; ++i)
            {
                if (path[i] == L'\\')
                {
                    ++slashCount;
                    if (slashCount == 2) // Stop at `\\Server\Share`
                        return;
                }
            }
        }

        // General case: strip the last component
        for (int32_t i = len - 1; i >= 0; --i)
        {
            if (path[i] == L'\\' || path[i] == L'/')
            {
                path[i]    = L'\0';               // Null-terminate
                m_dataSize = i * sizeof(wchar_t); // Update byte size
                m_isDir    = true;
                return;
            }
        }
    }

    // Returns a new Path object without the last component
    Path Path::parent() const
    {
        Path copy(*this); // Use copy constructor
        copy.parent_();   // Modify copy
        return copy;      // Return modified copy
    }

    // Append a path component in-place
    void Path::operator/=(char const* pathComponent)
    {
        if (!pathComponent || *pathComponent == '\0')
            return; // Nothing to append

        wchar_t* pathStart = reinterpret_cast<wchar_t*>(m_data);
        size_t   pathLen   = m_dataSize / sizeof(wchar_t);

        // Ensure there is a separator before appending
        bool hasSeparator = (pathStart[pathLen - 1] == L'\\' || pathStart[pathLen - 1] == L'/');

        // Convert input to wide string
        size_t compLen = MultiByteToWideChar(CP_UTF8, 0, pathComponent, -1, nullptr, 0);
        if (compLen == 0)
            return; // Invalid input

        auto compBuffer = dmt::makeUniqueRef<wchar_t[]>(m_resource, compLen);

        MultiByteToWideChar(CP_UTF8, 0, pathComponent, -1, compBuffer.get(), compLen);

        // Compute new size needed
        size_t additionalSize = compLen * sizeof(wchar_t);
        if (!hasSeparator)
            additionalSize += sizeof(wchar_t); // Extra space for separator

        if (m_dataSize + additionalSize > m_capacity)
        {
            // Reallocate memory
            size_t   newCapacity = m_dataSize + additionalSize + 32; // Add extra buffer
            wchar_t* newPath     = reinterpret_cast<wchar_t*>(m_resource->allocate(newCapacity));

            memcpy(newPath, m_data, m_dataSize);
            m_resource->deallocate(m_data, m_capacity);

            m_data     = newPath;
            m_capacity = newCapacity;
        }

        // Append separator if needed
        wchar_t* pathEnd = reinterpret_cast<wchar_t*>(m_data) + (m_dataSize / sizeof(wchar_t));
        if (!hasSeparator)
            *pathEnd++ = L'\\';

        // Append new component
        memcpy(pathEnd, compBuffer.get(), additionalSize);
        m_dataSize += additionalSize - sizeof(wchar_t); // Exclude null terminator

        // Ensure null termination
        pathEnd[compLen - 1] = L'\0';

        // Update validity
        pathStart = reinterpret_cast<wchar_t*>(m_data);
        m_valid   = PathFileExistsW(pathStart);
        if (m_valid)
            m_isDir = PathIsDirectoryW(pathStart);
    }

    // Returns a new Path object with an appended component
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

        wchar_t const*            wpath = static_cast<wchar_t const*>(m_data);
        WIN32_FILE_ATTRIBUTE_DATA data{};
        if (!GetFileAttributesExW(wpath, GetFileExInfoStandard, &data))
            return info;

        info.valid       = true;
        info.isDirectory = (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        info.size        = (static_cast<uint64_t>(data.nFileSizeHigh) << 32) | data.nFileSizeLow;

        ULARGE_INTEGER t;

        t.LowPart       = data.ftLastAccessTime.dwLowDateTime;
        t.HighPart      = data.ftLastAccessTime.dwHighDateTime;
        info.accessTime = (t.QuadPart - 116444736000000000ULL) / 10000000ULL;

        t.LowPart         = data.ftLastWriteTime.dwLowDateTime;
        t.HighPart        = data.ftLastWriteTime.dwHighDateTime;
        info.modifiedTime = (t.QuadPart - 116444736000000000ULL) / 10000000ULL;

        t.LowPart         = data.ftCreationTime.dwLowDateTime;
        t.HighPart        = data.ftCreationTime.dwHighDateTime;
        info.creationTime = (t.QuadPart - 116444736000000000ULL) / 10000000ULL;

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
        wchar_t const* wpath = static_cast<wchar_t const*>(path.internalData());

        HANDLE hFile = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

        if (hFile == INVALID_HANDLE_VALUE)
            return std::pmr::string{resource};

        DWORD bytesRead = 0;
        if (!ReadFile(hFile, str.data(), static_cast<DWORD>(fileSize), &bytesRead, nullptr))
        {
            CloseHandle(hFile);
            return std::pmr::string{resource};
        }

        CloseHandle(hFile);

        return str;
    }


    // Library Loader ------------------------------------------------------------------------------------------------
    void* LibraryLoader::loadLibrary(std::string_view name, bool useSystemPaths, Path const* pathOverride) const
    {
        HMODULE result = nullptr;
        if (name.empty())
            return result;

        if (pathOverride)
        {
            Path const     path     = *pathOverride / name.data();
            wchar_t const* wPathLib = reinterpret_cast<wchar_t const*>(path.internalData());

            result = LoadLibraryW(wPathLib);
        }
        else if (m_searchPathsLen > 0)
        {
            // Compute the maximum path length
            uint32_t maxPathLen = 0;
            for (uint32_t i = 0; i < m_searchPathsLen; ++i)
            {
                maxPathLen = std::max(maxPathLen, m_searchPaths[i].dataLength());
            }

            // Allocate buffer once
            uint32_t          requiredSize = static_cast<uint32_t>(maxPathLen + name.size() + 1);
            std::pmr::wstring fullPath(m_resource);
            fullPath.resize(requiredSize);
            wchar_t* buffer = fullPath.data();

            // convert name to UTF16
            std::pmr::wstring wName   = win32::utf16FromUtf8(name, m_resource);
            uint32_t          nameLen = static_cast<uint32_t>(wName.size());

            // Try each search path
            for (uint32_t i = 0; i < m_searchPathsLen && !result; ++i)
            {
                uint32_t offset = m_searchPaths[i].dataLength();
                std::memcpy(buffer, m_searchPaths[i].internalData(), offset * sizeof(wchar_t));

                assert(offset + nameLen <= requiredSize);

                std::memcpy(buffer + offset, wName.data(), nameLen * sizeof(wchar_t));
                buffer[offset + nameLen] = L'\0'; // Ensure null termination
                result                   = LoadLibraryW(buffer);
            }
        }

        if (!result && useSystemPaths)
        {
            std::pmr::wstring wNameLib = win32::utf16FromUtf8(name, m_resource);
            result                     = LoadLibraryW(wNameLib.c_str());
        }

        return result;
    }

    bool LibraryLoader::unloadLibrary(void* library) const
    {
        if (!library)
            return false; // Nothing to unload

        return FreeLibrary(static_cast<HMODULE>(library)) != 0;
    }

    // library utils
    namespace lib {
        void* getFunc(void* library, char const* funcName)
        {
            if (!library)
                return nullptr;
            return GetProcAddress(static_cast<HMODULE>(library), funcName);
        }
    } // namespace lib

    size_t systemAlignment()
    {
        SYSTEM_INFO sysInfo{};
        GetSystemInfo(&sysInfo);
        return static_cast<size_t>(sysInfo.dwAllocationGranularity);
    }

    void* allocate(size_t _bytes, size_t _align) { return _aligned_malloc(_bytes, _align); }

    void deallocate(void* ptr, [[maybe_unused]] size_t _bytes, [[maybe_unused]] size_t _align) { _aligned_free(ptr); }

    std::pmr::vector<std::pair<std::pmr::string, std::pmr::string>> getEnv(std::pmr::memory_resource* resource)
    {
        std::pmr::vector<std::pair<std::pmr::string, std::pmr::string>> vec{resource};
        vec.reserve(32);

        wchar_t* envStrings = GetEnvironmentStringsW();
        if (!envStrings)
            return vec;

        wchar_t* current = envStrings;
        while (*current)
        {
            std::wstring_view wideEntry{current};
            size_t            pos = wideEntry.find(L'=');
            if (pos != std::wstring_view::npos)
            {
                std::wstring_view wideName  = wideEntry.substr(0, pos);
                std::wstring_view wideValue = wideEntry.substr(pos + 1);

                // TODO document this check
                if (!wideEntry.starts_with(L'='))
                {
                    std::pmr::string name  = win32::utf8FromUtf16(wideName, resource);
                    std::pmr::string value = win32::utf8FromUtf16(wideValue, resource);
                    vec.emplace_back(std::move(name), std::move(value));
                }
            }

            current += wideEntry.length() + 1;
        }
        FreeEnvironmentStringsW(envStrings);

        return vec;
    }

} // namespace dmt::os