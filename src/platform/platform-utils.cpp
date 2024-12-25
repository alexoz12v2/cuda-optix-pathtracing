module;

#include <bit>
#include <string_view>

#include <cassert>

#if defined(DMT_OS_WINDOWS)
#pragma comment(lib, "mincore")
#include <AclAPI.h>
#include <Windows.h>
#include <errhandlingapi.h>
#include <fileapi.h>
#include <securitybaseapi.h>
#include <sysinfoapi.h>
#endif

module platform;

namespace dmt {

#if defined(DMT_OS_WINDOWS)
    static constexpr uint32_t                              sErrorBufferSize = 256;
    static thread_local std::array<char, sErrorBufferSize> sErrorBuffer{};
    struct Win32ChunkedFileReader
    {
        HANDLE hFile;
        OVERLAPPED overlapped; // TODO move class in platform or threadpool level and inject memory context or app context
        size_t   fileSize;
        uint32_t chunkSize;
        uint32_t numChunks;
        uint32_t numBytesReadLastTransfer;
    };
    static_assert(sizeof(Win32ChunkedFileReader) == ChunkedFileReader::size &&
                  alignof(Win32ChunkedFileReader) == ChunkedFileReader::alignment);

    static void __stdcall completionRoutine(_In_ DWORD           dwErrorCode,
                                            _In_ DWORD           dwNumberOfBytesTransfered,
                                            _Inout_ LPOVERLAPPED lpOverlapped)
    {
        reinterpret_cast<Win32ChunkedFileReader*>(lpOverlapped->hEvent)->numBytesReadLastTransfer = dwNumberOfBytesTransfered;
    }
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif

    ChunkedFileReader::ChunkedFileReader(PlatformContext& pctx, std::string_view filePath, uint32_t chunkSize)
    {
#if defined(DMT_OS_WINDOWS)
        LARGE_INTEGER fileSize;

        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(&m_data);
        // create file with ascii path only
        data.chunkSize = chunkSize;
        data.hFile     = CreateFileA(filePath.data(),
                                 GENERIC_READ,
                                 FILE_SHARE_READ,
                                 nullptr, // TODO maybe insert process descriptor, when you refactor system and process information
                                 OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING |
                                     FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_POSIX_SEMANTICS,
                                 nullptr);
        if (data.hFile == INVALID_HANDLE_VALUE)
        {
            uint32_t         length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
            std::string_view view{sErrorBuffer.data(), length};
            pctx.error("CreateFileA failed: {}", {view});
            return;
        }
        if (!GetFileSizeEx(data.hFile, &fileSize))
        {
            uint32_t         length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
            std::string_view view{sErrorBuffer.data(), length};
            pctx.error("CreateFileA failed: {}", {view});
            return;
        }
        data.fileSize  = fileSize.QuadPart;
        data.numChunks = static_cast<uint32_t>(ceilDiv(data.fileSize, static_cast<uint64_t>(chunkSize)));

        // from docs: The ReadFileEx function ignores the OVERLAPPED structure's hEvent member. An application is
        // free to use that member for its own purposes in the context of a ReadFileEx call.
        data.overlapped.hEvent = this;
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
    }

    bool ChunkedFileReader::requestChunk(PlatformContext& pctx, void* chunkBuffer, uint32_t chunkNum)
    {
#if defined(DMT_OS_WINDOWS)
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(&m_data);
        assert(chunkNum < data.numChunks);
        size_t offset              = chunkNum * data.chunkSize;
        data.overlapped.Offset     = static_cast<DWORD>(offset & 0x0000'0000'FFFF'FFFFULL);
        data.overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32); // file size > 4GB

        if (!ReadFileEx(data.hFile, chunkBuffer, data.chunkSize, &data.overlapped, completionRoutine))
        {
            uint32_t         length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
            std::string_view view{sErrorBuffer.data(), length};
            pctx.error("CreateFileA failed: {}", {view});
            return false;
        }
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
        return true;
    }

    uint32_t ChunkedFileReader::lastNumBytesRead()
    {
#if defined(DMT_OS_WINDOWS)
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(&m_data);
        return data.numBytesReadLastTransfer;
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
    }

    bool ChunkedFileReader::waitForPendingChunk(PlatformContext& pctx, uint32_t timeoutMillis)
    {
#if defined(DMT_OS_WINDOWS)
        if (DWORD err = GetLastError(); err != ERROR_SUCCESS)
        {
            SetLastError(err);
            uint32_t         length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
            std::string_view view{sErrorBuffer.data(), length};
            pctx.error("Read Operation failed: {}", {view});
            return false;
        }

        return SleepEx(timeoutMillis, true) == WAIT_IO_COMPLETION;

#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
        return true;
    }

    ChunkedFileReader::~ChunkedFileReader() noexcept
    {
#if defined(DMT_OS_WINDOWS)
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(&m_data);
        if (data.hFile && data.hFile != INVALID_HANDLE_VALUE)
        {
            CloseHandle(data.hFile);
            data.hFile = INVALID_HANDLE_VALUE;
        }
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
    }
} // namespace dmt
