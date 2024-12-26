module;

#include <array>
#include <bit>
#include <limits>

#include <cassert>

#if defined(DMT_OS_WINDOWS)
#pragma comment(lib, "mincore")
#include <AclAPI.h>
#include <Windows.h>
#include <errhandlingapi.h>
#include <fileapi.h>
#include <securitybaseapi.h>
#include <sysinfoapi.h>
#undef max
#undef min
#endif

module platform;

namespace dmt {
    inline constexpr uint8_t bufferFree     = 0;
    inline constexpr uint8_t bufferOccupied = 1;
    inline constexpr uint8_t bufferFinished = 2;

#if defined(DMT_OS_WINDOWS)
    inline constexpr uint32_t                              sErrorBufferSize = 256;
    static thread_local std::array<char, sErrorBufferSize> sErrorBuffer{};

    struct ExtraData
    {
        OVERLAPPED overlapped;
        uint32_t   numBytesReadLastTransfer;
        uint32_t   chunkNum;
    };

    struct Win32ChunkedFileReader
    {
        static constexpr uint64_t theMagic = std::numeric_limits<uint64_t>::max();
        struct PData
        {
            static constexpr uint32_t bufferStatusCount = 9;
            static constexpr uint32_t bufferStatusBytes = 2 * bufferStatusCount * sizeof(uint8_t);

            void*    buffer;
            uint64_t magic;
            uint32_t numChunksRead;
            uint8_t  numBuffers;
            uint8_t  bufferStatus[2 * bufferStatusCount]; // 0 = free, 1 = occupied, 2 = finished,
        };
        struct UData
        {
            OVERLAPPED overlapped;
            uint32_t   numBytesReadLastTransfer;
        };
        union U
        {
            PData pData;
            UData uData;
        };
        HANDLE   hFile;
        size_t   fileSize;
        U        u;
        uint32_t chunkSize;
        uint32_t numChunks;
    };
    static_assert(sizeof(Win32ChunkedFileReader) == ChunkedFileReader::size &&
                  alignof(Win32ChunkedFileReader) == ChunkedFileReader::alignment);

    static void __stdcall completionRoutine(_In_ DWORD           dwErrorCode,
                                            _In_ DWORD           dwNumberOfBytesTransfered,
                                            _Inout_ LPOVERLAPPED lpOverlapped)
    {
        auto pt = std::bit_cast<TaggedPointer>(lpOverlapped->hEvent);
        if (pt.tag() == 0x400)
        {
            pt.pointer<Win32ChunkedFileReader>()->u.uData.numBytesReadLastTransfer = dwNumberOfBytesTransfered;
        }
        else
        {
        }
    }

    static bool initFile(PlatformContext& pctx, char const* filePath, Win32ChunkedFileReader& data)
    {
        LARGE_INTEGER fileSize;

        // create file with ascii path only
        data.hFile = CreateFileA(filePath,
                                 GENERIC_READ,
                                 FILE_SHARE_READ,
                                 nullptr, // TODO maybe insert process descriptor, when you refactor system and process information
                                 OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING |
                                     FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_POSIX_SEMANTICS,
                                 nullptr);
        if (data.hFile == INVALID_HANDLE_VALUE)
        {
            uint32_t length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
            StrBuf   view{sErrorBuffer.data(), static_cast<int32_t>(length)};
            pctx.error("CreateFileA failed: {}", {view});
            return false;
        }
        if (!GetFileSizeEx(data.hFile, &fileSize))
        {
            uint32_t length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
            StrBuf   view{sErrorBuffer.data(), static_cast<int32_t>(length)};
            pctx.error("CreateFileA failed: {}", {view});
            return false;
        }
        data.fileSize = fileSize.QuadPart;
        return true;
    }
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif

    ChunkedFileReader::ChunkedFileReader(PlatformContext& pctx, char const* filePath, uint32_t chunkSize)
    {
#if defined(DMT_OS_WINDOWS)
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(&m_data);
        data.chunkSize               = chunkSize;
        if (!initFile(pctx, filePath, data))
        {
            return;
        }
        data.numChunks = static_cast<uint32_t>(ceilDiv(data.fileSize, static_cast<uint64_t>(chunkSize)));

        // from docs: The ReadFileEx function ignores the OVERLAPPED structure's hEvent member. An application is
        // free to use that member for its own purposes in the context of a ReadFileEx call.
        data.u.uData.overlapped.hEvent = std::bit_cast<HANDLE>(TaggedPointer{this, 0x400});
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
    }

    ChunkedFileReader::ChunkedFileReader(PlatformContext& pctx,
                                         char const*      filePath,
                                         uint32_t         chunkSize,
                                         uint8_t          numBuffers,
                                         uintptr_t*       pBuffers)
    {
#if defined(DMT_OS_WINDOWS)
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(&m_data);
        if (numBuffers > maxNumBuffers)
        {
            pctx.error("Exceeded maximum number of buffers for chunked file read");
            data.hFile = INVALID_HANDLE_VALUE;
            return;
        }

        data.chunkSize = chunkSize;
        if (!initFile(pctx, filePath, data))
        {
            return;
        }
        data.numChunks          = static_cast<uint32_t>(ceilDiv(data.fileSize, static_cast<uint64_t>(chunkSize)));
        data.u.pData.magic      = Win32ChunkedFileReader::theMagic;
        data.u.pData.numBuffers = numBuffers;
        std::memset(data.u.pData.bufferStatus, 0, Win32ChunkedFileReader::PData::bufferStatusBytes);

        // for each buffer, initialize offset to metadata and void* to actual data
        data.u.pData.buffer = reinterpret_cast<void*>(pBuffers);
        for (uint64_t i = 0; i < numBuffers; ++i)
        {
            void* ptr     = std::bit_cast<void*>(pBuffers[i]);
            auto* pOffset = std::bit_cast<uint64_t*>(alignToAddr(pBuffers[i] + chunkSize, alignof(uint64_t)));
            auto* pExtra  = std::bit_cast<ExtraData*>(
                alignToAddr(std::bit_cast<uintptr_t>(pOffset) + sizeof(uint64_t), alignof(ExtraData)));
            *pOffset = std::bit_cast<uintptr_t>(pExtra) - std::bit_cast<uintptr_t>(ptr);

            pExtra->overlapped               = {};
            pExtra->overlapped.hEvent        = std::bit_cast<HANDLE>(TaggedPointer{ptr, 0x800});
            pExtra->numBytesReadLastTransfer = 0;
        }
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

        if (data.u.pData.magic == Win32ChunkedFileReader::theMagic)
        {
            pctx.error("invalid state. initialized for multi chunk operator, tried single buffer op");
            return false;
        }

        size_t offset                      = chunkNum * data.chunkSize;
        data.u.uData.overlapped.Offset     = static_cast<DWORD>(offset & 0x0000'0000'FFFF'FFFFULL);
        data.u.uData.overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32); // file size > 4GB

        if (!ReadFileEx(data.hFile, chunkBuffer, data.chunkSize, &data.u.uData.overlapped, completionRoutine))
        {
            uint32_t length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
            StrBuf   view{sErrorBuffer.data(), static_cast<int32_t>(length)};
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
        if (data.u.pData.magic == Win32ChunkedFileReader::theMagic)
        {
            return 0;
        }
        return data.u.uData.numBytesReadLastTransfer;
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
        return 0;
    }

    bool ChunkedFileReader::waitForPendingChunk(PlatformContext& pctx, uint32_t timeoutMillis)
    {
#if defined(DMT_OS_WINDOWS)
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(&m_data);
        if (data.u.pData.magic == Win32ChunkedFileReader::theMagic)
        {
            pctx.error("invalid state. initialized for multi chunk operator, tried single buffer op");
            return false;
        }

        if (DWORD err = GetLastError(); err != ERROR_SUCCESS)
        {
            SetLastError(err);
            uint32_t length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
            StrBuf   view{sErrorBuffer.data(), static_cast<int32_t>(length)};
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

    size_t ChunkedFileReader::computeAlignedChunkSize(size_t chunkSize)
    {
#if defined(DMT_OS_WINDOWS)
        constexpr size_t alignment = alignof(OVERLAPPED); // alignof(OVERLAPPED) == 8
        constexpr size_t extraSize = sizeof(ExtraData);

        // Compute total size needed, aligning the sum to the alignment boundary
        size_t totalSize = sizeof(uint64_t) + chunkSize + extraSize;
        return (totalSize + (alignment - 1)) & ~(alignment - 1); // Align to next multiple of alignment
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
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

    bool ChunkedFileReader::InputIterator::operator==(ChunkedFileReader::EndSentinel const&) const
    {
        return false;
    }

    ChunkedFileReader::ChunkInfo ChunkedFileReader::InputIterator::operator*() const
    {
#if defined(DMT_OS_WINDOWS)
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(m_pData);
        // retrieve the next buffer
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
        return {};
    }

    ChunkedFileReader::InputIterator& ChunkedFileReader::InputIterator::operator++()
    {
#if defined(DMT_OS_WINDOWS)
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(m_pData);
        // if there are any in flight operations (bufferStatus == 1), then return immediately
        // or if in data, the numChunksRead == numChunks, return
        for (uint32_t i = 0; i < Win32ChunkedFileReader::PData::bufferStatusCount)
        {
            uint32_t byteIndex = i >> 1;
            for (uint32_t j = 0; j < 4; ++j)
            {
                uint32_t bufferIndex = i + j;
                uint8_t  status      = (data.u.pData.bufferStatus[byteIndex] >> (j << 1)) & 0b11u;
                switch (status)
                {
                    case bufferFree:
                    {

                        void* ptr     = reinterpret_cast<void**>(data.u.pData.buffer)[bufferIndex];
                        auto* pOffset = std::bit_cast<uint64_t*>(
                            alignToAddr(std::bit_cast<uintptr_t>(ptr) + chunkSize, alignof(uint64_t)));
                        auto* pExtra = std::bit_cast<ExtraData*>(
                            alignToAddr(std::bit_cast<uintptr_t>(pOffset) + sizeof(uint64_t), alignof(ExtraData)));
                        OVERLAPPED* pOverlapped = &pExtra->overlapped;
                        size_t      offset      = m_chunkNum * data.chunkSize;
                        pOverlapped->Offset     = static_cast<DWORD>(offset & 0x0000'0000'FFFF'FFFFULL);
                        pOverlapped->OffsetHigh = static_cast<DWORD>(offset >> 32); // file size > 4GB

                        if (!ReadFileEx(data.hFile, chunkBuffer, data.chunkSize, &data.u.uData.overlapped, completionRoutine))
                        {
                            uint32_t length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
                            StrBuf   view{sErrorBuffer.data(), static_cast<int32_t>(length)};
                            pctx.error("CreateFileA failed: {}", {view});
                            return false;
                        }

                        ++m_chunkNum;
                        data.u.pData.bufferStatus[byteIndex] |= (1u << (j << 1));
                        break;
                    }
                    case bufferOccupied:
                        break;
                    case bufferFinished:
                        return *this;
                        break;
                }
            }
        }
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
        return *this;
    }
} // namespace dmt
