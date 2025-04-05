#include "platform-file.h"
#include "platform-os-utils.win32.h"
#include "platform-context.h"

#include <Windows.h>

static constexpr uint32_t errorBufferSize = 1024;
static thread_local char  s_errorBuffer[errorBufferSize];

static std::string_view getLastWin32Error()
{
    uint32_t const   length = ::dmt::os::win32::getLastErrorAsString(s_errorBuffer, errorBufferSize);
    std::string_view view{s_errorBuffer, length};
    return view;
}

namespace dmt::os {
    // ChunkedFileReader ----------------------------------------------------------------------------------------------
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

    // TODO move to path class, enable long path handling, ...
    static bool initFile(wchar_t const* filePath, Win32ChunkedFileReader& data)
    {
        dmt::Context  ctx;
        LARGE_INTEGER fileSize;

        // create file with ascii path only
        data.hFile = CreateFileW(filePath,
                                 GENERIC_READ,
                                 FILE_SHARE_READ,
                                 nullptr, // TODO maybe insert process descriptor, when you refactor system and process information
                                 OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING |
                                     FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_POSIX_SEMANTICS,
                                 nullptr);
        if (data.hFile == INVALID_HANDLE_VALUE)
        {
            if (ctx.isValid())
                ctx.error("CreateFileA failed: {}", std::make_tuple(getLastWin32Error()));
            return false;
        }
        if (!GetFileSizeEx(data.hFile, &fileSize))
        {
            if (ctx.isValid())
                ctx.error("CreateFileA failed: {}", std::make_tuple(getLastWin32Error()));
            return false;
        }
        data.fileSize = fileSize.QuadPart;
        return true;
    }

    ChunkedFileReader::ChunkedFileReader(char const* filePath, uint32_t chunkSize, std::pmr::memory_resource* resource) :
    m_resource(resource)
    {
        assert(m_resource);
        dmt::Context            ctx;
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(&m_data);
        if (!isPowerOfTwoAndGE512(chunkSize))
        {
            if (ctx.isValid())
                ctx.error("Invalid Chunk Size. Win32 requires a POT GE 512", {});
            data.hFile = INVALID_HANDLE_VALUE;
            return;
        }
        auto wPath = dmt::os::win32::utf16FromUtf8(filePath, m_resource);
        if (!initFile(wPath.c_str(), data))
        {
            if (ctx.isValid())
                ctx.error("Invalid File. Cannot create ChunkedFileReader", {});
            return;
        }

        data.chunkSize = chunkSize;
        data.numChunks = static_cast<uint32_t>(ceilDiv(data.fileSize, static_cast<uint64_t>(chunkSize)));

        // from docs: The ReadFileEx function ignores the OVERLAPPED structure's hEvent member. An application is
        // free to use that member for its own purposes in the context of a ReadFileEx call.
        data.u.uData.overlapped.hEvent = std::bit_cast<HANDLE>(TaggedPointer{this, 0x400});
    }

    ChunkedFileReader::ChunkedFileReader(char const*                filePath,
                                         uint32_t                   chunkSize,
                                         uint8_t                    numBuffers,
                                         uintptr_t*                 pBuffers,
                                         std::pmr::memory_resource* resource) :
    m_resource(resource)
    {
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(&m_data);
        assert(m_resource);
        dmt::Context ctx;
        if (numBuffers > maxNumBuffers)
        {
            if (ctx.isValid())
                ctx.error("Exceeded maximum number of buffers for chunked file read", {});
            data.hFile = INVALID_HANDLE_VALUE;
            return;
        }
        if (!isPowerOfTwoAndGE512(chunkSize))
        {
            if (ctx.isValid())
                ctx.error("Invalid Chunk Size. Win32 requires a POT GE 512", {});
            data.hFile = INVALID_HANDLE_VALUE;
            return;
        }

        data.chunkSize = chunkSize;
        auto wPath     = dmt::os::win32::utf16FromUtf8(filePath, m_resource);
        if (!initFile(wPath.c_str(), data))
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
    }

    bool ChunkedFileReader::requestChunk(void* chunkBuffer, uint32_t chunkNum)
    {
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(&m_data);
        dmt::Context            ctx;
        assert(chunkNum < data.numChunks);

        if (alignTo(chunkBuffer, 8) != chunkBuffer)
        {
            if (ctx.isValid())
                ctx.error("invalid chunk buffer, nned it aligned to a 8 byte boundary", {});
            return false;
        }

        if (data.u.pData.magic == Win32ChunkedFileReader::theMagic)
        {
            if (ctx.isValid())
                ctx.error("invalid state. initialized for multi chunk operator, tried single buffer op", {});
            return false;
        }

        size_t offset                      = chunkNum * data.chunkSize;
        data.u.uData.overlapped.Offset     = static_cast<DWORD>(offset & 0x0000'0000'FFFF'FFFFULL);
        data.u.uData.overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32); // file size > 4GB

        if (!ReadFileEx(data.hFile, chunkBuffer, data.chunkSize, &data.u.uData.overlapped, completionRoutine))
        {
            if (ctx.isValid())
                ctx.error("ReadFileEx failed: {}", std::make_tuple(getLastWin32Error()));
            return false;
        }
        return true;
    }

    uint32_t ChunkedFileReader::lastNumBytesRead()
    {
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(&m_data);
        if (data.u.pData.magic == Win32ChunkedFileReader::theMagic)
        {
            return 0;
        }
        return data.u.uData.numBytesReadLastTransfer;
    }

    bool ChunkedFileReader::waitForPendingChunk()
    {
        uint32_t timeoutMillis;
        timeoutMillis = INFINITE;
        return waitForPendingChunk(timeoutMillis);
    }

    bool ChunkedFileReader::waitForPendingChunk(uint32_t timeoutMillis)
    {
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(&m_data);
        dmt::Context            ctx;
        if (data.u.pData.magic == Win32ChunkedFileReader::theMagic)
        {
            if (ctx.isValid())
                ctx.error("invalid state. initialized for multi chunk operator, tried single buffer op", {});
            return false;
        }

        if (DWORD err = GetLastError(); err != ERROR_SUCCESS && err != ERROR_IO_PENDING)
        {
            SetLastError(err);
            if (ctx.isValid())
                ctx.error("Read Operation failed: {}", std::make_tuple(getLastWin32Error()));
            return false;
        }

        return SleepEx(timeoutMillis, true) == WAIT_IO_COMPLETION;
    }

    size_t ChunkedFileReader::computeAlignedChunkSize(size_t chunkSize)
    {
        constexpr size_t alignment = alignof(OVERLAPPED); // alignof(OVERLAPPED) == 8
        constexpr size_t extraSize = sizeof(ExtraData);

        // Compute total size needed, aligning the sum to the alignment boundary
        size_t totalSize = sizeof(uint64_t) + chunkSize + extraSize;
        return (totalSize + (alignment - 1)) & ~(alignment - 1); // Align to next multiple of alignment
    }

    ChunkedFileReader::~ChunkedFileReader() noexcept
    {
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(&m_data);
        if (data.hFile && data.hFile != INVALID_HANDLE_VALUE)
        {
            if (!CloseHandle(data.hFile))
            {
                assert(false && "error while closing a file");
            }
            data.hFile = INVALID_HANDLE_VALUE;
        }
    }

    bool ChunkedFileReader::InputIterator::operator==(ChunkedFileReader::EndSentinel const&) const
    {
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(m_pData);
        if (data.hFile == INVALID_HANDLE_VALUE || data.fileSize == 0)
        {
            return true;
        }

        // see whether all chunks were requested
        bool allChunksRequested = !inRange();
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
    }

    ChunkInfo ChunkedFileReader::InputIterator::operator*() const
    {
        ChunkInfo               ret{};
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

        return {};
    }

    void ChunkedFileReader::markFree(ChunkInfo const& chunkInfo)
    {
        Win32ChunkedFileReader& data      = *reinterpret_cast<Win32ChunkedFileReader*>(&m_data);
        uint32_t                byteIndex = static_cast<uint32_t>(chunkInfo.indexData >> 3);
        uint32_t                shamt     = static_cast<uint32_t>(chunkInfo.indexData & 0b111);
        uint8_t                 status    = (data.u.pData.bufferStatus[byteIndex] >> shamt) & 0b11u;
        assert(status == bufferFinished);
        data.u.pData.bufferStatus[byteIndex] ^= (bufferFinished << shamt);
        data.u.pData.bufferStatus[byteIndex] |= (bufferFree << shamt);
    }

    ChunkedFileReader::InputIterator& ChunkedFileReader::InputIterator::operator++()
    {
        Win32ChunkedFileReader& data = *reinterpret_cast<Win32ChunkedFileReader*>(m_pData);
        Context                 ctx;
        if (!inRange())
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
                        size_t      offset      = m_current * data.chunkSize;
                        pOverlapped->Offset     = static_cast<DWORD>(offset & 0x0000'0000'FFFF'FFFFULL);
                        pOverlapped->OffsetHigh = static_cast<DWORD>(offset >> 32); // file size > 4GB
                        pOverlapped->hEvent     = std::bit_cast<HANDLE>(TaggedPointer{&data, tag});
                        pExtra->chunkNum        = m_current;

                        if (!ReadFileEx(data.hFile, ptr, data.chunkSize, pOverlapped, completionRoutine))
                        {
                            if (ctx.isValid())
                                ctx.error("ReadFileEx error: {}", std::make_tuple(getLastWin32Error()));

                            m_current = m_chunkNum + m_numChunks;
                            return *this;
                        }

                        ++m_current;
                        data.u.pData.bufferStatus[byteIndex] |= (bufferOccupied << (j << 1));

                        if (!inRange())
                        {
                            return *this;
                        }

                        break;
                    }
                    case bufferFinished: return *this; break;
                    case bufferOccupied: [[fallthrough]];
                    default: break;
                }
            }
        }

        // if you are still here, it means that all buffers are occupied, meaning you can go on
        return *this;
    }

    ChunkedFileReader::operator bool() const
    {
        Win32ChunkedFileReader const& data = *reinterpret_cast<Win32ChunkedFileReader const*>(&m_data);
        return data.hFile != INVALID_HANDLE_VALUE;
    }

    uint32_t ChunkedFileReader::numChunks() const
    {
        Win32ChunkedFileReader const& data = *reinterpret_cast<Win32ChunkedFileReader const*>(&m_data);
        return data.numChunks;
    }

    ChunkedFileReader::InputIterator ChunkedFileReader::Range::begin()
    {
        Win32ChunkedFileReader const& data = *reinterpret_cast<Win32ChunkedFileReader const*>(pData);
        assert(data.numChunks >= chunkNum + numChunks);
        return ++InputIterator(pData, chunkNum, numChunks);
    }
} // namespace dmt::os