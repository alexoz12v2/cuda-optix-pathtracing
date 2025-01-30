#pragma once
#include "platform/platform-macros.h"
#include "platform/platform-logging.h"

#include <iterator>
#include <utility>

#include <cstdint>

namespace dmt {
    struct ChunkInfo
    {
        void*    buffer;
        uint64_t indexData;
        uint32_t numBytesRead;
        uint32_t chunkNum;
    };

} // namespace dmt

namespace dmt::os {
    class DMT_PLATFORM_API alignas(32) ChunkedFileReader
    {
    public:

    private:
        struct DMT_PLATFORM_API EndSentinel
        {
        };
        struct DMT_PLATFORM_API InputIterator
        {
        public:
            using difference_type = std::ptrdiff_t;
            using value_type      = ChunkInfo;

            InputIterator(void* pData, uint32_t chunkedNum, uint32_t numChunks) :
            m_pData(pData),
            m_chunkNum(chunkedNum),
            m_current(chunkedNum),
            m_numChunks(numChunks)
            {
            }

            ChunkInfo operator*() const;

            bool           operator==(EndSentinel const&) const;
            InputIterator& operator++();
            void           operator++(int) { ++*this; }

        private:
            constexpr bool inRange() const { return m_current < m_chunkNum + m_numChunks; }

            void*    m_pData;
            uint32_t m_chunkNum;
            uint32_t m_current;
            uint32_t m_numChunks;
        };
        static_assert(std::input_iterator<InputIterator>);

        struct DMT_PLATFORM_API Range
        {
            constexpr Range(void* pData, uint32_t chunkNum, uint32_t numChunks) :
            pData(pData),
            chunkNum(chunkNum),
            numChunks(numChunks)
            {
            }

            InputIterator begin();

            EndSentinel end() { return {}; }

            void*    pData;
            uint32_t chunkNum;
            uint32_t numChunks;
        };

        friend struct InputIterator;

    public:
        static constexpr uint32_t maxNumBuffers = 72;
        static constexpr uint32_t size          = 64;
        static constexpr uint32_t alignment     = 32;
        ChunkedFileReader(LoggingContext& pctx, char const* filePath, uint32_t chunkSize); // udata mode
        ChunkedFileReader(LoggingContext& pctx,
                          char const*     filePath,
                          uint32_t        chunkSize,
                          uint8_t         numBuffers,
                          uintptr_t*      pBuffers); // pdata mode
        ChunkedFileReader(ChunkedFileReader const&)                = delete;
        ChunkedFileReader(ChunkedFileReader&&) noexcept            = delete;
        ChunkedFileReader& operator=(ChunkedFileReader const&)     = delete;
        ChunkedFileReader& operator=(ChunkedFileReader&&) noexcept = delete;
        ~ChunkedFileReader() noexcept;

        bool     requestChunk(LoggingContext& pctx, void* chunkBuffer, uint32_t chunkNum);
        bool     waitForPendingChunk(LoggingContext& pctx, uint32_t timeoutMillis);
        bool     waitForPendingChunk(LoggingContext& pctx);
        uint32_t lastNumBytesRead();
        void     markFree(ChunkInfo const& chunkInfo);
        uint32_t numChunks() const;
        Range    range(uint32_t chunkNum, uint32_t numChunks) { return Range{&m_data, chunkNum, numChunks}; }

        operator bool() const;

        static size_t computeAlignedChunkSize(size_t chunkSize);

    private:
        alignas(alignment) unsigned char m_data[size];
    };
} // namespace dmt::os