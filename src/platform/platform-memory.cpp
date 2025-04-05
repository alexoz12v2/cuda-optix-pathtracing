#include "platform-memory.h"

#include "platform-context.h"

#include <array>
#include <bit>
#include <numeric>
#include <string_view>
#include <thread>
#include <type_traits>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

#if defined(DMT_DEBUG)
#include <backward.hpp>
#include <map>
#include <memory_resource>
#endif

namespace dmt {
    char const* memoryTagStr(EMemoryTag tag)
    {
        // clang-format off
        static char const* strs[toUnderlying(EMemoryTag::eCount)] {
            "Unknown", "Debug", "Engine", "HashTable", "Buffer", "Blob", "Job", "Queue", "Scene", 
            "AccelerationStructure", "Geometry", "Material", "Textures", "Spectrum"
        };
        // clang-format on
        assert(tag < EMemoryTag::eCount);

        return strs[toUnderlying(tag)];
    }

    // StringTable --------------------------------------------------------------------------------------------------------

    StringTable::StringTable(std::pmr::memory_resource* resource) : m_resource(resource), m_stringTable(resource) {}

    sid_t StringTable::intern(std::string_view str) { return intern(str.data(), str.size()); }

    sid_t StringTable::intern(char const* str, uint64_t sz)
    {
        assert(sz < MAX_SID_LEN);
        std::lock_guard lock{m_mtx};
        sid_t           sid = hashCRC64(str);

        auto const& [it, wasInserted] = m_stringTable.try_emplace(sid);
        if (wasInserted)
        {
            std::memcpy(it->second.data(), str, sz);
            it->second[sz] = '\0';
        }

        return sid;
    }

    std::string_view StringTable::lookup(sid_t sid) const
    {
        using namespace std::string_view_literals;
        std::string_view empty = "NOT FOUND"sv;
        auto             it    = m_stringTable.find(sid);
        if (it != m_stringTable.cend())
        {
            std::string_view str{it->second.data()};
            return str;
        }

        return empty;
    }

    // SyncPoolAllocator --------------------------------------------------------------------------------------------------
    size_t SyncPoolAllocator::bitmapReservedSize() const
    {
        assert(m_blockSize && m_reservedSize >= m_blockSize);
        return ceilDiv(m_reservedSize, static_cast<size_t>(m_blockSize)) >> 2; // 2 bits per block
    }

    size_t SyncPoolAllocator::bitmapCommittedSize() const
    {
        assert(m_blockSize && m_committedSize >= m_blockSize);
        return ceilDiv(m_committedSize, static_cast<size_t>(m_blockSize)) >> 2; // 2 bits per block
    }

    uint32_t SyncPoolAllocator::numBlocks() const
    {
        assert(m_blockSize && m_committedSize >= m_blockSize);
        return m_committedSize / m_blockSize;
    }

    SyncPoolAllocator::S SyncPoolAllocator::bitPairOffsetAndMaskFromBlock(size_t blkIdx) const
    {
        return {.offset = blkIdx >> 2, .shamt = static_cast<uint8_t>((blkIdx & 0b11u) << 1u)};
    }

    SyncPoolAllocator::SyncPoolAllocator(EMemoryTag tag, size_t reservedSize, uint32_t numInitialBlocks, EBlockSize blockSize) :
    m_committedSize(numInitialBlocks == 0 ? 0 : reservedSize / numInitialBlocks),
    m_reservedSize(reservedSize),
    m_bitmap(nullptr),
    m_memory(nullptr),
    m_tag(tag),
    m_blockSize(toUnderlying(blockSize))
    {
        // there is no need to lock the spinlock here, since construction is contained in one thread/process only
        Context ctx;
        if (!m_committedSize)
        {
            if (ctx.isValid())
                ctx.error("{} insufficient for {} blocks", std::make_tuple(reservedSize, numInitialBlocks));
            return;
        }

        size_t const bitmapReservedSz  = bitmapReservedSize();
        size_t const bitmapCommittedSz = bitmapCommittedSize();

        // compute the reservation size and initial commit size for the bitmap
        // reserve and initial commit for memory (assuming block is std::max_align_t from system allocation)
        // reserve and commit for bitmap (0 = free, 1 = occupied)
        m_memory = os::reserveVirtualAddressSpace(m_reservedSize + bitmapReservedSz);
        m_bitmap = reinterpret_cast<unsigned char*>(m_memory) + m_reservedSize;
        if (!m_memory)
        {
            if (ctx.isValid())
                ctx.error("Failed to reserve memory", {});
            return;
        }
        if (!os::commitPhysicalMemory(m_memory, m_committedSize))
        {
            if (ctx.isValid())
                ctx.error("Failed to commit memory for block pool", {});
            os::freeVirtualAddressSpace(m_memory, m_reservedSize);
            m_memory = m_bitmap = nullptr;
            return;
        }
        if (!os::commitPhysicalMemory(m_bitmap, bitmapCommittedSz))
        {
            if (ctx.isValid())
                ctx.error("Failed to commit memory for bitmap", {});
            os::freeVirtualAddressSpace(m_memory, m_reservedSize);
            m_memory = m_bitmap = nullptr;
            return;
        }
    }

    bool SyncPoolAllocator::isValid() const
    {
        std::lock_guard lk{m_mtx};
        return m_committedSize > 0 && m_reservedSize > 0 && m_bitmap && m_memory && isPOT(m_blockSize);
    }

    SyncPoolAllocator::~SyncPoolAllocator() noexcept
    {
        std::lock_guard lk{m_mtx};
        Context         ctx;
        if (m_memory)
        {
            os::decommitPhysicalMemory(m_memory, m_committedSize);
            os::decommitPhysicalMemory(m_bitmap, bitmapCommittedSize());
            os::freeVirtualAddressSpace(m_memory, m_reservedSize + bitmapReservedSize());
            m_memory = nullptr;
            m_bitmap = nullptr;
        }
    }

    void* SyncPoolAllocator::do_allocate(size_t bytes, size_t align)
    {
        std::lock_guard lk{m_mtx};

        // assert that blockSize is multiple of align and that memory is align aligned
        assert(m_blockSize % align == 0);

        // find out number of blocks necessary to satisfy the allocation -> numBlocks
        size_t numBlocksRequired = ceilDiv(bytes, static_cast<size_t>(m_blockSize));

        // search in the currently committed memory (bitmap) for numBlocks free blocks (bitmap[i] = 0)
        // if blocks found, tag all bitmap bits as occupied (2) and the last occupied as (1)
        if (void* ptr = scrollAndTagBlocks(0, numBlocksRequired); ptr)
            return ptr;

        // if you didn't find any free block in the currently committed memory, check that the remainng size to be committed hosts enough additional blocks
        size_t const oldNumBlock      = numBlocks();
        size_t const additionalMemory = numBlocksRequired * m_blockSize;
        size_t const additionalBitmap = numBlocksRequired >> 2;

        if (additionalMemory > m_reservedSize - m_committedSize)
            return nullptr;

        if (!grow(additionalMemory, additionalBitmap))
            return nullptr;

        if (void* ptr = scrollAndTagBlocks(oldNumBlock, numBlocksRequired); ptr)
            return ptr;
    }

    void* SyncPoolAllocator::scrollAndTagBlocks(size_t startBlkIdx, size_t numBlocksRequired)
    {
        size_t   blkStart = startBlkIdx;
        uint32_t blkNum   = 0;

        // search in the currently committed memory (bitmap) for numBlocks free blocks (bitmap[i] = 0)
        for (size_t blkIdx = blkStart; blkIdx < numBlocks(); ++blkIdx)
        {
            uint8_t const state = extractBitPairState(blkIdx);
            if (state == bitmapStateFree)
            {
                // if its the first, set the blkStart
                if (blkNum == 0)
                    blkStart = blkIdx;

                ++blkNum;
                // if you reached the requested size, good
                if (blkNum == numBlocksRequired)
                    break;
            }
            else
                blkNum = 0;
        }

        // if blocks found, tag all bitmap bits as occupied (2) and the last occupied as (1)
        if (blkNum == numBlocksRequired)
        {
            for (size_t blkIdx = blkStart; blkIdx < blkStart + static_cast<size_t>(blkNum) - 1; ++blkIdx)
            {
                setBitPairState(blkIdx, bitmapStateOccupied);
            }
            setBitPairState(blkStart + static_cast<size_t>(blkNum) - 1, bitmapStateLastOccupied);
            return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(m_memory) + blkStart * m_blockSize);
        }

        return nullptr;
    }

    bool SyncPoolAllocator::grow(size_t additionalMemory, size_t additionalBitmap)
    {
        assert(additionalMemory + m_committedSize <= m_reservedSize);
        void* memAddr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(m_memory) + m_committedSize);
        void* bitAddr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(m_memory) + bitmapCommittedSize());
        if (!os::commitPhysicalMemory(memAddr, additionalMemory))
            return false;
        if (!os::commitPhysicalMemory(bitAddr, additionalBitmap))
        {
            os::decommitPhysicalMemory(memAddr, additionalMemory);
            return false;
        }

        m_committedSize += additionalMemory;
        return true;
    }

    uint8_t SyncPoolAllocator::extractBitPairState(size_t blkIdx) const
    {
        auto const [bitmapOffset, bitmapShamt] = bitPairOffsetAndMaskFromBlock(blkIdx);
        uint8_t const state = (reinterpret_cast<uint8_t const*>(m_bitmap)[bitmapOffset] >> bitmapShamt) & bitmapMask;
        return state;
    }

    void SyncPoolAllocator::setBitPairState(size_t blkIdx, uint8_t state)
    {
        static constexpr uint8_t full          = std::numeric_limits<uint8_t>::max();
        auto const [bitmapOffset, bitmapShamt] = bitPairOffsetAndMaskFromBlock(blkIdx);

        // clean bit pair
        reinterpret_cast<uint8_t*>(m_bitmap)[bitmapOffset] |= (full & (bitmapMask << bitmapShamt));
        // set new state
        reinterpret_cast<uint8_t*>(m_bitmap)[bitmapOffset] |= (state << bitmapShamt);
    }

    void SyncPoolAllocator::do_deallocate(void* ptr, size_t bytes, size_t align)
    {
        std::lock_guard lk{m_mtx};
        if (!ptr || !bytes)
            return;

        // we can actually ignore the bytes passed, as we keep track of allocations through the bitmap
        // size_t numBlocksRequired = ceilDiv(bytes, static_cast<size_t>(m_blockSize));

        // if pointer != nullptr, from the pointer, find out the offset from start of the committed memory area
        // if outside, issue a warning and return
        size_t offset = reinterpret_cast<uintptr_t>(ptr) - reinterpret_cast<uintptr_t>(m_memory);
        if (offset > m_committedSize)
            return;

        // if inside, compute the starting index inside the committed bitmap,
        size_t blkIdx = offset / m_blockSize;
        for (/**/; extractBitPairState(blkIdx) != bitmapStateLastOccupied; ++blkIdx)
        {
            setBitPairState(blkIdx, bitmapStateFree);
        }
        setBitPairState(blkIdx, bitmapStateFree);
    }

    bool SyncPoolAllocator::do_is_equal(memory_resource const& _That) const noexcept
    {
        // more constructor, move assignment, copy constructor, copy assignment are all deleted
        return false;
    }
} // namespace dmt
