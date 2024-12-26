module;

#include <array>
#include <bit>
#include <concepts>
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

    template <std::integral I>
    static constexpr bool isPowerOfTwoAndGE512(I num)
    {
        // Check if number is greater than or equal to 512
        if (num < 512)
        {
            return false;
        }

        // Check if the number is a power of 2
        // A number is a power of 2 if it has only one bit set
        return (num & (num - 1)) == 0;
    }

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
            static constexpr uint32_t bufferStatusCount = 18;

            void*    buffer;
            uint64_t magic;
            uint32_t numChunksRead;
            uint8_t  numBuffers;
            uint8_t  bufferStatus[bufferStatusCount]; // 0 = free, 1 = occupied, 2 = finished,
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
                  alignof(Win32ChunkedFileReader) <= ChunkedFileReader::alignment);

    static ExtraData* extraFromBuffer(void* buffer, uint32_t chunkSize)
    { // TODO remove offset
        auto* pOffset = std::bit_cast<uint64_t*>(
            alignToAddr(std::bit_cast<uintptr_t>(buffer) + chunkSize, alignof(uint64_t)));
        auto* pExtra = std::bit_cast<ExtraData*>(
            alignToAddr(std::bit_cast<uintptr_t>(pOffset) + sizeof(uint64_t), alignof(ExtraData)));
        return pExtra;
    }

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
            uint16_t tag         = pt.tag() & 0x7FFu;
            uint8_t  i           = static_cast<uint8_t>(tag >> 2);
            uint8_t  j           = static_cast<uint8_t>(tag & 0b11);
            auto*    ptr         = pt.pointer<Win32ChunkedFileReader>();
            uint8_t  bufferIndex = (i << 2) + j;
            assert(bufferIndex < ChunkedFileReader::maxNumBuffers);
            uint8_t byteIndex = i;

            ptr->u.pData.bufferStatus[byteIndex] ^= (bufferOccupied << (j << 1));
            ptr->u.pData.bufferStatus[byteIndex] |= (bufferFinished << (j << 1));
            void* p                          = reinterpret_cast<void**>(ptr->u.pData.buffer)[bufferIndex];
            auto* pExtra                     = extraFromBuffer(p, ptr->chunkSize);
            pExtra->numBytesReadLastTransfer = dwNumberOfBytesTransfered;
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
        if (!isPowerOfTwoAndGE512(chunkSize))
        {
            pctx.error("Invalid Chunk Size. Win32 requires a POT GE 512");
            data.hFile = INVALID_HANDLE_VALUE;
            return;
        }
        if (!initFile(pctx, filePath, data))
        {
            return;
        }

        data.chunkSize = chunkSize;
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
        if (!isPowerOfTwoAndGE512(chunkSize))
        {
            pctx.error("Invalid Chunk Size. Win32 requires a POT GE 512");
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
        std::memset(data.u.pData.bufferStatus, 0, Win32ChunkedFileReader::PData::bufferStatusCount * sizeof(uint8_t));

        // for each buffer, initialize offset to metadata and void* to actual data
        data.u.pData.buffer = reinterpret_cast<void*>(pBuffers);
        for (uint64_t i = 0; i < numBuffers; ++i)
        {
            void* ptr    = std::bit_cast<void*>(pBuffers[i]);
            auto* pExtra = extraFromBuffer(ptr, chunkSize);

            pExtra->overlapped               = {};
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
            if (!CloseHandle(data.hFile))
            {
                assert(false && "error while closing a file");
            }
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
#if defined(DMT_OS_WINDOWS)
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(m_pData);
        if (data.hFile == INVALID_HANDLE_VALUE || data.fileSize == 0)
        {
            return true;
        }

        // see whether all chunks were requested
        bool allChunksRequested = data.numChunks >= m_chunkNum;
        if (!allChunksRequested)
        {
            return false;
        }

        // now scan the metadata so see whether all buffers are actually free
        for (uint8_t i = 0; i < Win32ChunkedFileReader::PData::bufferStatusCount; ++i)
        {
            uint8_t byteIndex = i;
            for (uint8_t j = 0; j < 4; ++j)
            { // since maxNumBuffers is 72, we need 7 bits to store the bufferIndex + 2 bits for j
                uint8_t bufferIndex = (i << 2) + j;
                uint8_t status      = (data.u.pData.bufferStatus[byteIndex] >> (j << 1)) & 0b11u;
                if (status != bufferFree)
                {
                    return false;
                }
            }
        }

        return true;
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
    }

    ChunkInfo ChunkedFileReader::InputIterator::operator*() const
    {
        ChunkInfo ret{};
#if defined(DMT_OS_WINDOWS)
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(m_pData);
        // find the first free buffer. If there's none, `SleepEx` and try again
        while (true)
        {
            for (uint8_t i = 0; i < Win32ChunkedFileReader::PData::bufferStatusCount; ++i)
            {
                uint8_t byteIndex = i;
                for (uint8_t j = 0; j < 4; ++j)
                { // since maxNumBuffers is 72, we need 7 bits to store the bufferIndex + 2 bits for j
                    uint8_t bufferIndex = (i << 2) + j;

                    // TODO better
                    if (bufferIndex >= data.u.pData.numBuffers)
                    {
                        break;
                    }

                    uint8_t status = (data.u.pData.bufferStatus[byteIndex] >> (j << 1)) & 0b11u;
                    if (status == bufferFinished)
                    {
                        // return a chunk info with the current buffer. It is the caller responsibility to
                        // call the `markFree` method to let the reader use the buffer for another chunk
                        void* ptr        = reinterpret_cast<void**>(data.u.pData.buffer)[bufferIndex];
                        auto* pExtra     = extraFromBuffer(ptr, data.chunkSize);
                        ret.buffer       = ptr;
                        ret.numBytesRead = pExtra->numBytesReadLastTransfer;
                        ret.chunkNum     = pExtra->chunkNum;
                        ret.indexData    = (static_cast<uint64_t>(byteIndex) << 3) | (j << 1);
                        return ret;
                    }
                }
            }

            // nothing finished, then wait
            if (SleepEx(INFINITE, true) != WAIT_IO_COMPLETION)
            {
                return {};
            }
        }
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
        return {};
    }

    void ChunkedFileReader::markFree(ChunkInfo const& chunkInfo)
    {
#if defined(DMT_OS_WINDOWS)
        Win32ChunkedFileReader& data      = *reinterpret_cast<Win32ChunkedFileReader*>(&m_data);
        uint32_t                byteIndex = static_cast<uint32_t>(chunkInfo.indexData >> 3);
        uint32_t                shamt     = static_cast<uint32_t>(chunkInfo.indexData & 0b111);
        uint8_t                 status    = (data.u.pData.bufferStatus[byteIndex] >> shamt) & 0b11u;
        assert(status == bufferFinished);
        data.u.pData.bufferStatus[byteIndex] ^= (bufferFinished << shamt);
        data.u.pData.bufferStatus[byteIndex] |= (bufferFree << shamt);
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
    }

    ChunkedFileReader::InputIterator& ChunkedFileReader::InputIterator::operator++()
    {
#if defined(DMT_OS_WINDOWS)
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(m_pData);
        if (m_chunkNum >= data.numChunks)
        {
            return *this;
        }

        // if there are any in flight operations (bufferStatus == 1), then return immediately
        // or if in data, the numChunksRead == numChunks, return
        for (uint8_t i = 0; i < Win32ChunkedFileReader::PData::bufferStatusCount; ++i)
        {
            uint8_t byteIndex = i;
            for (uint8_t j = 0; j < 4; ++j)
            { // since maxNumBuffers is 72, we need 7 bits to store the bufferIndex + 2 bits for j
                uint8_t bufferIndex = (i << 2) + j;
                if (bufferIndex >= data.u.pData.numBuffers)
                {
                    return *this;
                }

                uint16_t tag = (static_cast<uint16_t>(i) << 2) | j;
                assert(tag <= 0x7FFu);
                tag |= 0x800u;
                uint8_t status = (data.u.pData.bufferStatus[byteIndex] >> (j << 1)) & 0b11u;
                switch (status)
                {
                    case bufferFree:
                    {
                        void* ptr     = reinterpret_cast<void**>(data.u.pData.buffer)[bufferIndex];
                        auto* pOffset = std::bit_cast<uint64_t*>(
                            alignToAddr(std::bit_cast<uintptr_t>(ptr) + data.chunkSize, alignof(uint64_t)));
                        auto* pExtra = std::bit_cast<ExtraData*>(
                            alignToAddr(std::bit_cast<uintptr_t>(pOffset) + sizeof(uint64_t), alignof(ExtraData)));
                        OVERLAPPED* pOverlapped = &pExtra->overlapped;
                        size_t      offset      = m_chunkNum * data.chunkSize;
                        pOverlapped->Offset     = static_cast<DWORD>(offset & 0x0000'0000'FFFF'FFFFULL);
                        pOverlapped->OffsetHigh = static_cast<DWORD>(offset >> 32); // file size > 4GB
                        pOverlapped->hEvent     = std::bit_cast<HANDLE>(TaggedPointer{&data, tag});
                        pExtra->chunkNum        = m_chunkNum;

                        if (!ReadFileEx(data.hFile, ptr, data.chunkSize, pOverlapped, completionRoutine))
                        {
                            // TODO handle better
                            uint32_t length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
                            StrBuf   view{sErrorBuffer.data(), static_cast<int32_t>(length)};
                            m_chunkNum = data.numChunks;
                            return *this;
                        }

                        ++m_chunkNum;
                        data.u.pData.bufferStatus[byteIndex] |= (bufferOccupied << (j << 1));

                        if (m_chunkNum >= data.numChunks)
                        {
                            return *this;
                        }

                        break;
                    }
                    case bufferFinished:
                        return *this;
                        break;
                    case bufferOccupied:
                        [[fallthrough]];
                    default:
                        break;
                }
            }
        }

        // if you are still here, it means that all buffers are occupied, meaning you can go on
        return *this;
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
        return *this;
    }

    ChunkedFileReader::operator bool() const
    {
#if defined(DMT_OS_WINDOWS)
        Win32ChunkedFileReader const& data = *reinterpret_cast<Win32ChunkedFileReader const*>(&m_data);
        return data.hFile != INVALID_HANDLE_VALUE;
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
    }

    uint32_t ChunkedFileReader::numChunks() const
    {
#if defined(DMT_OS_WINDOWS)
        Win32ChunkedFileReader const& data = *reinterpret_cast<Win32ChunkedFileReader const*>(&m_data);
        return data.numChunks;
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
    }
} // namespace dmt
