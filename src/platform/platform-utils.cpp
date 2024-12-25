module;

#if defined(DMT_OS_WINDOWS)
#pragma comment(lib, "mincore")
#include <AclAPI.h>
#include <Windows.h>
#include <fileapi.h>
#include <securitybaseapi.h>
#include <sysinfoapi.h>
#endif

module platform;

namespace dmt {
    ChunkedFileReader::ChunkedFileReader(std::string_view filePath)
    {
#if defined(DMT_OS_WINDOWS)
        // create file with ascii path only
        CreateFileA(filePath.c_str(),
                    GENERIC_WRITE,
                    FILE_SHARE_READ,
                    nullptr,
                    OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS |
                        FILE_FLAG_POSIX_SEMANTICS,
                    nullptr);
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
    }

    ChunkedFileReader::~ChunkedFileReader() noexcept
    {
#if defined(DMT_OS_WINDOWS)
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
    }
} // namespace dmt
