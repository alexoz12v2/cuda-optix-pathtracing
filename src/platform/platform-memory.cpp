#include "platform-memory.h"

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

    // PageAllocator ------------------------------------------------------------------------------------------------------

    void PageAllocator::deallocPage(LoggingContext& ctx, PageAllocation& alloc)
    {
        PageAllocation allocCopy = alloc;
        PageAllocator::deallocatePage(ctx, alloc);
        m_hooks.freeHook(m_hooks.data, ctx, allocCopy);
    }


    AllocatePageForBytesResult PageAllocator::allocatePagesForBytes(
        LoggingContext& ctx,
        size_t          numBytes,
        PageAllocation* pOut,
        uint32_t        inNum,
        EPageSize       pageSize)
    {
        static constexpr uint32_t maxAttempts         = 2048;
        uint32_t                  totalPagesAllocated = 0;
        uint32_t numPages  = static_cast<uint32_t>(ceilDiv(numBytes, static_cast<size_t>(toUnderlying(pageSize))));
        uint32_t allocated = 0;
        uint32_t index     = 0;
        for (uint32_t i = 0; allocated < numBytes && i < maxAttempts; ++i)
        {
            pOut[index] = allocatePage(ctx, pageSize);
            if (!pOut[index].address)
            {
                pageSize = scaleDownPageSize(pageSize);
            }
            else
            {
                ++index;
                allocated += toUnderlying(pageSize);
                ++totalPagesAllocated;
            }
        }

        return {.numBytes = allocated, .numPages = totalPagesAllocated};
    }

    EPageSize PageAllocator::allocatePagesForBytesQuery(LoggingContext&             ctx,
                                                        size_t                      numBytes,
                                                        uint32_t&                   inOutNum,
                                                        EPageAllocationQueryOptions opts)
    {
#if defined(DMT_OS_WINDOWS)
        EPageSize pageSize = m_largePageEnabled ? (m_largePage1GB ? EPageSize::e1GB : EPageSize::e2MB) : EPageSize::e4KB;
#elif defined(DMT_OS_LINUX)
        EPageSize pageSize = m_enabledPageSize;
#else
        EPageSize pageSize = EPageSize::e4KB;
#endif

        if (pageSize == EPageSize::e1GB && opts != EPageAllocationQueryOptions::eNone)
        {
            pageSize = scaleDownPageSize(pageSize);
        }

        while (toUnderlying(pageSize) > numBytes && pageSize != EPageSize::e4KB)
        {
            pageSize = scaleDownPageSize(pageSize);
        }

        if (opts != EPageAllocationQueryOptions::eForce4KB)
        {
            while (pageSize != EPageSize::e4KB)
            {
                if (checkPageSizeAvailability(ctx, pageSize))
                {
                    inOutNum = static_cast<uint32_t>(ceilDiv(numBytes, static_cast<size_t>(toUnderlying(pageSize))));
                    return pageSize;
                }

                pageSize = scaleDownPageSize(pageSize);
            }
        }

        inOutNum = static_cast<uint32_t>(ceilDiv(numBytes, static_cast<size_t>(toUnderlying(EPageSize::e4KB))));
        return pageSize;
    }

    bool PageAllocator::checkPageSizeAvailability(LoggingContext& ctx, EPageSize pageSize)
    { // TODO: don't commit memory or save it into a cache
        ctx.warn("Don't use me");
        // Attempt a small test allocation to verify if the page size is supported
        bool           supported = false;
        PageAllocation testAlloc = allocatePage(ctx, pageSize);
        if (testAlloc.address && testAlloc.pageSize == pageSize)
            supported = true;

        if (testAlloc.address)
            deallocatePage(ctx, testAlloc);

        return supported;
    }

    // ObjectAllocationSlidingWindow --------------------------------------------------------------------------------------

    ObjectAllocationsSlidingWindow::ObjectAllocationsSlidingWindow(size_t reservedSize) :
    m_reservedSize(reservedSize),
    m_activeIndex(0)
    {
        using HType = ObjectAllocationsSlidingWindow::Header;
        // reserve memory
        void* ptr = os::reserveVirtualAddressSpace(reservedSize);
        if (!ptr)
        {
            std::abort();
        }
        uintptr_t    addr      = std::bit_cast<uintptr_t>(ptr);
        size_t const sliceSize = reservedSize / numBlocks;
        size_t const pageSize  = toUnderlying(EPageSize::e4KB);
        assert(reservedSize % numBlocks == 0);
        assert(pageSize <= sliceSize && sliceSize % pageSize == 0);

        // split the reserved size into `numBlocks` slices and commit first page
        for (uint32_t i = 0; i < numBlocks; ++i)
        {
            uintptr_t headerAddr = addr + i * sliceSize;
            m_blocks[i]          = std::bit_cast<HType*>(headerAddr);
            if (!os::commitPhysicalMemory(m_blocks[i], pageSize))
            {
                std::abort();
            }

            // initialize header data
            HType& header            = *m_blocks[i];
            header.firstNotCommitted = headerAddr + pageSize;
            header.limit             = headerAddr + sliceSize;
            header.size              = 0;
        }
    }

    ObjectAllocationsSlidingWindow::~ObjectAllocationsSlidingWindow() noexcept
    {
        os::freeVirtualAddressSpace(m_blocks[0], m_reservedSize);
    }

    void ObjectAllocationsSlidingWindow::addToCurrent(AllocationInfo const& alloc)
    {
        using HType              = ObjectAllocationsSlidingWindow::Header;
        HType*          pHeader  = m_blocks[m_activeIndex];
        AllocationInfo* pNext    = firstAllocFromHeader(pHeader) + pHeader->size;
        uintptr_t       nextAddr = std::bit_cast<uintptr_t>(pNext);

        // if we are exceeding the limit, assert false in debug, otherwise just return (big address space)
        if (nextAddr >= pHeader->limit)
        {
            assert(false && "Reserve more virtual address space");
            return;
        }

        // if we are exceeding `firstNotCommitted`, commit the next page and update it
        if (nextAddr >= pHeader->firstNotCommitted && !commitBlock())
        {
            assert(false && "Somehow, we couldn't commit a new page of memory");
            return;
        }

        *pNext = alloc;

        ++pHeader->size;
    }

    void ObjectAllocationsSlidingWindow::switchTonext()
    {
        m_activeIndex                 = (m_activeIndex + 1) % numBlocks;
        m_blocks[m_activeIndex]->size = 0;
    }

    bool ObjectAllocationsSlidingWindow::commitBlock()
    {
        using HType    = ObjectAllocationsSlidingWindow::Header;
        HType* pHeader = m_blocks[m_activeIndex];
        size_t sz      = toUnderlying(EPageSize::e4KB);
        bool   b       = os::commitPhysicalMemory(std::bit_cast<void*>(pHeader->firstNotCommitted), sz);
        if (b)
        {
            pHeader->firstNotCommitted += sz;
        }
        return b;
    }

    void ObjectAllocationsSlidingWindow::touchFreeTime(void* address, uint64_t freeTime)
    {
        using HType             = ObjectAllocationsSlidingWindow::Header;
        HType*          pHeader = m_blocks[m_activeIndex];
        AllocationInfo* pBegin  = firstAllocFromHeader(pHeader);
        AllocationInfo* pEnd    = firstAllocFromHeader(pHeader) + pHeader->size;
        for (AllocationInfo* pCurr = pBegin; pCurr != pEnd; ++pCurr)
        {
            if (pCurr->address == address)
            {
                pCurr->freeTime = freeTime;
            }
        }
    }

    // PageAllocatorTracker -----------------------------------------------------------------------------------------------

    static void logAndAbort(LoggingContext& ctx, std::string_view str, uint32_t initialNodeNum)
    {
        ctx.error("Couldn't commit the first {} nodes into the reserved space", {initialNodeNum});
        std::abort();
    }

    PageAllocationsTracker::PageAllocationsTracker(LoggingContext& ctx, uint32_t pageTrackCapacity, uint32_t allocTrackCapacity)
    {
        using namespace std::string_view_literals;
        // reserve virtual address space the double ended buffer
        m_pageTracking.m_capacity       = pageTrackCapacity;
        m_allocTracking.m_capacity      = allocTrackCapacity;
        m_pageTracking.m_growBackwards  = false;
        m_allocTracking.m_growBackwards = true;
        size_t sysAlign                 = os::systemAlignment();
        assert(sysAlign >= alignof(PageNode) && sysAlign >= alignof(AllocNode));

        size_t pageFreeListBytes  = initialNodeNum * sizeof(PageNode);
        size_t allocFreeListBytes = initialNodeNum * sizeof(AllocNode);
        size_t allocOffset        = m_allocTracking.m_capacity * sizeof(AllocNode);
        size_t pageOffset         = m_pageTracking.m_capacity * sizeof(PageNode);
        m_bufferBytes             = allocOffset + pageOffset;
        m_base                    = os::reserveVirtualAddressSpace(m_bufferBytes);
        if (!m_base)
        {
            ctx.error("Couldn't reserve virtual address space for {} bytes", {m_bufferBytes});
            std::abort();
        }

        // commit `initialNodeNum` for the low address free list and the high address free list
        uintptr_t end       = reinterpret_cast<uintptr_t>(m_base) + m_bufferBytes;
        void*     allocBase = alignToBackward(reinterpret_cast<void*>(end - allocFreeListBytes), sysAlign);
        if (!os::commitPhysicalMemory(m_base, pageFreeListBytes))
        {
            logAndAbort(ctx, "pageFreeList, VirtualAlloc MEM_COMMIT"sv, initialNodeNum);
        }
        if (!os::commitPhysicalMemory(allocBase, end - reinterpret_cast<uintptr_t>(allocBase)))
        {
            logAndAbort(ctx, "allocFreeList, VirtualAlloc MEM_COMMIT"sv, initialNodeNum);
        }

        // set head and reset each list
        void* pEnd = reinterpret_cast<void*>(end - sizeof(AllocNode));
        std::memset(pEnd, 0, 32);
        m_pageBase  = alignTo(m_base, alignof(PageNode));
        m_allocBase = alignToBackward(pEnd, alignof(AllocNode));

        m_pageTracking.m_freeHead  = reinterpret_cast<PageNode*>(m_pageBase);
        m_allocTracking.m_freeHead = reinterpret_cast<AllocNode*>(m_allocBase);
        m_pageTracking.m_freeSize  = initialNodeNum;
        m_allocTracking.m_freeSize = initialNodeNum;

        m_pageTracking.reset();
        m_allocTracking.reset();
    }

    template <typename T>
        requires requires(T t) { t.address; }
    static bool shouldTrack(LoggingContext& ctx, T const& alloc)
    {
        bool ret = alloc.address != nullptr;
        if (!ret)
        {
            ctx.warn("Passed an invalid allocation to the trakcer, nullptr address");
        }
        return ret;
    }

    void PageAllocationsTracker::track(LoggingContext& ctx, PageAllocation const& alloc)
    {
        ctx.trace("tracking allocation at address {}", {alloc.address});
        if (!m_pageTracking.m_freeHead)
        {
            growFreeList(ctx, m_pageTracking, m_pageBase);
        }

        if (shouldTrack(ctx, alloc))
        {
            m_pageTracking.addNode(alloc);
        }
    }

    void PageAllocationsTracker::untrack(LoggingContext& ctx, PageAllocation const& alloc)
    {
        ctx.trace("Untracking allocation at address {}", {alloc.address});
        if (m_pageTracking.removeNode(alloc))
        {
            return;
        }

        ctx.error("Attempted to deallocate a page that was not tracked.");
        std::abort();
    }

    void PageAllocationsTracker::track(LoggingContext& ctx, AllocationInfo const& alloc)
    {
        ctx.trace("tracking allocation at address {}", {alloc.address});
        if (!m_allocTracking.m_freeHead)
        {
            growFreeList(ctx, m_allocTracking, m_allocBase);
        }

        if (shouldTrack(ctx, alloc))
        {
            m_allocTracking.addNode(alloc);
            m_slidingWindow.addToCurrent(alloc);
        }
    }

    void PageAllocationsTracker::untrack(LoggingContext& ctx, AllocationInfo const& alloc)
    {
        ctx.trace("Untracking allocation at address {}", {alloc.address});
        if (m_allocTracking.removeNode(alloc))
        {
            uint64_t freeTime = ctx.millisFromStart();
            m_slidingWindow.touchFreeTime(alloc.address, freeTime);
            return;
        }

        ctx.error("Attempted to deallocate a page that was not tracked.");
        std::abort();
    }

    void PageAllocationsTracker::claenTransients(LoggingContext& ctx)
    {
        ctx.log("Untracking (deallocating tracking data) for all transient allocations");
        uint64_t freeTime = ctx.millisFromStart();
        m_allocTracking.forEachTransientNodes(this, freeTime, [](void* self, uint64_t freeTime, AllocationInfo const& alloc) {
            auto& tracker = *reinterpret_cast<PageAllocationsTracker*>(self);
            tracker.m_slidingWindow.touchFreeTime(alloc.address, freeTime);
        });
        m_allocTracking.removeTransientNodes();
    }

    PageAllocationsTracker::~PageAllocationsTracker() noexcept
    {
        assert(!m_pageTracking.m_occupiedHead && !m_allocTracking.m_occupiedHead &&
               "some allocated memory is outliving the tracker");

        os::freeVirtualAddressSpace(m_base, m_bufferBytes);
    }

    // StackAllocator -----------------------------------------------------------------------------------------------------
    StackAllocator::StackAllocator(LoggingContext& ctx, PageAllocator& pageAllocator, AllocatorHooks const& hooks)
    {
        // try to allocate the first 2MB stack buffer
        PageAllocation alloc;
        if (!pageAllocator.allocate2MB(ctx, alloc))
        {
            ctx.error("Couldn't allocate 2MB buffer for the stack allocator, aborting ...");
            std::abort();
        }

        // create the StackHeader structure in the beginning of the page
        assert(alloc.address == alignTo(alloc.address, alignof(StackHeader)) &&
               "page address isn't "
               "aligned to a stack header size boundary");

        StackHeader& header = *reinterpret_cast<StackHeader*>(alloc.address);
        header.alloc        = alloc;
        header.bp           = reinterpret_cast<uintptr_t>(&header) + sizeof(StackHeader);
        header.sp           = header.bp;
        header.prev         = nullptr;
        header.next         = nullptr;
        header.notUsedFor   = 0;

        // wire the buffer
        m_pFirst = &header;
        m_pLast  = &header;

        // copy hooks
        m_hooks = hooks;
    }

    void StackAllocator::cleanup(LoggingContext& ctx, PageAllocator& pageAllocator)
    {
        std::lock_guard lock{m_mtx};
        m_hooks.cleanTransients(m_hooks.data, ctx);

        // start from last and free all buffers
        for (StackHeader* curr = m_pLast; curr != nullptr;)
        {
            StackHeader* prev = curr->prev;
            ctx.trace("Deallocating buffer at address: {}", {reinterpret_cast<void*>(curr)});
            pageAllocator.deallocPage(ctx, curr->alloc);
            curr = prev;
        }

        m_pFirst = nullptr;
        m_pLast  = nullptr;
        ctx.trace("Cleanup complete. All buffers have been deallocated.");
    }

    void* StackAllocator::allocate(LoggingContext& ctx,
                                   PageAllocator&  pageAllocator,
                                   size_t          size,
                                   size_t          alignment,
                                   EMemoryTag      tag,
                                   sid_t           sid)
    {
        if (size > bufferSize || (alignment & (alignment - 1)) != 0)
        {
            ctx.error("Invalid allocation parameters: size={}, alignment={}", {size, alignment});
            return nullptr;
        }

        std::lock_guard lock{m_mtx};
        assert(m_pFirst);

        // verify that sp is within the bounds [bp, bp+bufferSize)
        // if yes, then verify that, given size and alignment, we can satisfy the allocation
        // if yes again, then compute the aligned pointer, update sp, and return
        // if any of the aforementioned conditions failed, we need to try onto the next buffer the same procedure
        // if there's no next buffer, then call `newBuffer` and try again
        while (true)
        {
            uint16_t bufIdx = 0;
            for (StackHeader* curr = m_pLast; curr != nullptr; curr = curr->next)
            {
                uintptr_t alignedSP = alignToAddr(curr->sp, alignment);
                uintptr_t end       = curr->bp + bufferSize;

                if (alignedSP + size <= end)
                {
                    void* ptr        = reinterpret_cast<void*>(alignedSP);
                    curr->sp         = alignedSP + size;
                    curr->notUsedFor = 0;

                    // call allocation hooks (TODO create properly AllocationInfo)
                    AllocationInfo allocInfo;
                    allocInfo.address   = ptr;
                    allocInfo.size      = size;
                    allocInfo.alignment = alignment;
                    allocInfo.transient = 1;
                    allocInfo.tag       = tag;
                    allocInfo.sid       = sid;
                    allocInfo.freeTime  = 0;
                    allocInfo.allocTime = ctx.millisFromStart();
                    m_hooks.allocHook(m_hooks.data, ctx, allocInfo);

                    return ptr;
                }
                ++bufIdx;
            }

            if (bufIdx == std::numeric_limits<uint16_t>::max() - 1)
            {
                ctx.error("Allocation failed: size={}, alignment={}", {size, alignment});
                return nullptr;
            }

            if (!newBuffer(ctx, pageAllocator))
            {
                ctx.error("Allocation failed: size={}, alignment={}", {size, alignment});
                return nullptr;
            }
        }

        return nullptr;
    }

    void StackAllocator::reset(LoggingContext& ctx, PageAllocator& pageAllocator)
    {
        std::lock_guard lock{m_mtx};
        assert(m_pFirst);

        m_hooks.cleanTransients(m_hooks.data, ctx);

        for (StackHeader* curr = m_pLast; curr != nullptr;)
        {
            curr->sp          = curr->bp;
            StackHeader* prev = curr->prev;

            // if you didn't use the buffer for a certain number of times, then deallocate it
            if (++curr->notUsedFor > notUsedForThreshold)
            {
                pageAllocator.deallocPage(ctx, curr->alloc);
            }
            curr = prev;
            if (prev)
                prev->next = nullptr; // there's the assumption that you are using buffers in order. how to check?
        }
    }

    bool StackAllocator::newBuffer(LoggingContext& ctx, PageAllocator& pageAllocator)
    {
        assert(m_pLast);
        PageAllocation alloc;
        if (!pageAllocator.allocate2MB(ctx, alloc))
        {
            ctx.error("Failed to allocate new 2MB buffer.");
            return false;
        }

        // Create the new StackHeader at the beginning of the new buffer
        StackHeader& newHeader = *reinterpret_cast<StackHeader*>(alloc.address);
        newHeader.alloc        = alloc;
        newHeader.bp           = reinterpret_cast<uintptr_t>(&newHeader) + sizeof(StackHeader);
        newHeader.sp           = newHeader.bp;
        newHeader.prev         = m_pLast;
        newHeader.next         = nullptr;
        newHeader.notUsedFor   = 0;

        // Update the chain
        m_pLast->next = &newHeader;
        m_pLast       = &newHeader;

        return true;
    }

    // MultiPoolAllocator -------------------------------------------------------------------------------------------------

    static constexpr TaggedPointer encode(uint16_t bufferIndex, uint8_t blockSizeEncoding, void* ptr)
    {
        uint16_t tag = bufferIndex | static_cast<uint16_t>(blockSizeEncoding);
        assert(tag <= 0xFFF);
        return {ptr, tag};
    }

    MultiPoolAllocator::MultiPoolAllocator(LoggingContext&                     ctx,
                                           PageAllocator&                      pageAllocator,
                                           std::array<uint32_t, numBlockSizes> numBlocksPerPool,
                                           AllocatorHooks const&               hooks) :
    m_hooks(hooks)
    {
        static_assert(numBlockSizes == 4);
        static constexpr uint16_t max = 256;

        // compute numBytes and counts for each pool
        uint16_t blockSizes[numBlockSizes]{toUnderlying(EBlockSize::e32B),
                                           toUnderlying(EBlockSize::e64B),
                                           toUnderlying(EBlockSize::e128B),
                                           toUnderlying(EBlockSize::e256B)};
        size_t   bytesPerPool[numBlockSizes]{numBlocksPerPool[0] * blockSizes[0],
                                             numBlocksPerPool[1] * blockSizes[1],
                                             numBlocksPerPool[2] * blockSizes[2],
                                             numBlocksPerPool[3] * blockSizes[3]};

        // compute size of the metadata required to associate 1 bit to each block (account for the PageAllocation(24B) + next pointer + uint8_t notUsedFor)
        // modify numBytes and counts such that we can fit metadata + pools into the 2MB buffer
        uint32_t numBitsMetadata = std::reduce(std::begin(numBlocksPerPool), std::end(numBlocksPerPool), 0u, std::plus<>());
        m_numBytesMetadata = ceilDiv(numBitsMetadata, 8u);
        m_totalSize = std::reduce(std::begin(bytesPerPool), std::end(bytesPerPool), 0ULL, std::plus<>()) + m_numBytesMetadata;
        ctx.log("tring to allocate pool buffer for {} {} {} {}",
                {StrBuf{bytesPerPool[0], "0x%zx"},
                 StrBuf{bytesPerPool[1], "0x%zx"},
                 StrBuf{bytesPerPool[2], "0x%zx"},
                 StrBuf{bytesPerPool[3], "0x%zx"}});
        while (m_totalSize > bufferSize)
        { // reduce the largest pool whose block size is bigger than the difference between total and bufferSize
            size_t residual = m_totalSize - bufferSize;
            for (uint32_t i = numBlockSizes - 1; i < numBlockSizes; --i)
            {
                if (residual > blockSizes[i])
                {
                    --numBlocksPerPool[i];
                    bytesPerPool[i] -= blockSizes[i];
                    m_totalSize -= blockSizes[i];
                    --numBitsMetadata;
                    m_numBytesMetadata = ceilDiv(numBitsMetadata, 8u);
                    break;
                }
            }
        }

        m_blocksPerPool = numBlocksPerPool;

        // Adjust the pools to fill up the remaining space but leave room for alignment overhead
        size_t remainingSpace = bufferSize - m_totalSize;
        while (remainingSpace > *std::min_element(blockSizes, blockSizes + numBlockSizes)) // Ensure we leave space for alignment
        {
            bool blockAdded = false;
            for (uint32_t i = 0; i < numBlockSizes; ++i)
            {
                size_t blockSizeWithMetadata = blockSizes[i] + (1 / 8u);        // Block size + 1 bit metadata
                if (remainingSpace > blockSizeWithMetadata + poolBaseAlignment) // Leave room for alignment
                {
                    ++numBlocksPerPool[i];
                    bytesPerPool[i] += blockSizes[i];
                    m_totalSize += blockSizeWithMetadata;
                    ++numBitsMetadata;
                    m_numBytesMetadata = ceilDiv(numBitsMetadata, 8u);
                    remainingSpace     = bufferSize - m_totalSize;
                    blockAdded         = true;
                    break;
                }
            }
            if (!blockAdded)
                break; // No block could fit into the remaining space
        }

        ctx.log("The possible sizes are {} {} {} {}",
                {StrBuf{bytesPerPool[0], "0x%zx"},
                 StrBuf{bytesPerPool[1], "0x%zx"},
                 StrBuf{bytesPerPool[2], "0x%zx"},
                 StrBuf{bytesPerPool[3], "0x%zx"}});
        ctx.log("Allocating a buffer of {} Bytes, actually using {} Bytes",
                {StrBuf{bufferSize, "0x%zx"}, StrBuf{m_totalSize, "0x%zx"}});

        for (uint32_t i = 0; i != numBlockSizes; ++i)
            m_numBlocksPerPool[i] = m_numBlocksPerPool[i];

        // allocate initial 2MB block
        newBlock(ctx, pageAllocator, &m_firstBuffer);
        m_lastBuffer = m_firstBuffer;
    }

    void MultiPoolAllocator::newBlock(LoggingContext& ctx, PageAllocator& pageAllocator, BufferHeader** ptr)
    {
        PageAllocation pageAlloc;
        if (!pageAllocator.allocate2MB(ctx, pageAlloc))
        {
            ctx.error("Couldn't allocate 2MB buffer for multi pool allocator, aborting...");
            std::abort();
        }
        auto& header = *reinterpret_cast<BufferHeader*>(pageAlloc.address);
        header.alloc = pageAlloc;
        header.next  = nullptr;

        // compute the buffer base
        uintptr_t metadataAddr = reinterpret_cast<uintptr_t>(&header) + sizeof(BufferHeader);
        uintptr_t poolBase     = metadataAddr + m_numBytesMetadata;
        poolBase               = alignToAddr(poolBase, poolBaseAlignment);
        if (size_t alignmentOverhead = poolBase - (metadataAddr + m_numBytesMetadata); alignmentOverhead > 0)
        {
            size_t adjustedTotal = m_totalSize + alignmentOverhead;
            if (adjustedTotal > bufferSize)
            {
                ctx.error("Alignment overhead caused total size to exceed buffer limit.");
                std::abort();
            }
            m_totalSize += alignmentOverhead;
        }

        header.poolBase = poolBase;

        // tag all blocks in the metadata as empty
        std::memset(std::bit_cast<void*>(metadataAddr), 0, poolBase - metadataAddr);

        if (ptr)
        {
            *ptr = reinterpret_cast<BufferHeader*>(pageAlloc.address);
        }
    }

    void MultiPoolAllocator::cleanup(LoggingContext& ctx, PageAllocator& pageAllocator)
    {
        std::lock_guard lock{m_mtx};

        // Traverse the linked list of buffers starting from the first buffer
        BufferHeader* currentBuffer = m_firstBuffer;

        while (currentBuffer)
        {
            // Save the pointer to the next buffer before deallocating the current one
            BufferHeader* nextBuffer = currentBuffer->next;

            // Deallocate the page associated with the current buffer
            pageAllocator.deallocPage(ctx, currentBuffer->alloc);

            // Move to the next buffer in the list
            currentBuffer = nextBuffer;
        }

        // Clear the state of the allocator
        m_firstBuffer      = nullptr;
        m_lastBuffer       = nullptr;
        m_totalSize        = 0;
        m_numBytesMetadata = 0;
        std::fill(std::begin(m_numBlocksPerPool), std::end(m_numBlocksPerPool), 0);

        ctx.log("MultiPoolAllocator cleanup complete. All buffers deallocated.");
    }

    static constexpr uint8_t flipBits(uint8_t value)
    {
        uint8_t result = 0;
        for (int i = 0; i < 8; i++)
        {
            result |= ((value >> i) & 1) << (7 - i);
        }
        return result;
    }

    TaggedPointer MultiPoolAllocator::allocateBlocks(
        LoggingContext& ctx,
        PageAllocator&  pageAllocator,
        uint32_t        numBlocks,
        EBlockSize      blockSize,
        EMemoryTag      tag,
        sid_t           sid)
    {
        std::lock_guard lock{m_mtx};

        // Determine the block size index
        uint8_t blockSizeIndex = ::dmt::blockSizeEncoding(blockSize);

        BufferHeader* currentBuffer = m_firstBuffer;
        uint16_t      bufferIdx     = 0;
        uint32_t      bitsOffset    = 0;
        for (uint32_t i = 0; i < blockSizeIndex; ++i)
        {
            bitsOffset += m_blocksPerPool[i];
        }

        // Iterate through the buffers to find free blocks
        while (currentBuffer)
        {
            uintptr_t metadataAddr = std::bit_cast<uintptr_t>(currentBuffer) + sizeof(BufferHeader);
            uintptr_t poolBase     = currentBuffer->poolBase;
            for (uint8_t i = 0; i < blockSizeIndex; ++i)
            {
                poolBase += m_blocksPerPool[i] * toUnderlying(fromEncoding(i));
            }

            // Scan the metadata for a free region of numBlocks
            uint8_t* metadata        = reinterpret_cast<uint8_t*>(metadataAddr);
            uint32_t numBlocksInPool = m_numBlocksPerPool[blockSizeIndex];
            uint16_t blockSizeBytes  = toUnderlying(blockSize);

            for (uint32_t i = 0; i <= numBlocksInPool; ++i)
            {
                // Check if a region of numBlocks is free
                bool isRegionFree = true;
                for (uint32_t j = 0; j < numBlocks; ++j)
                {
                    uint32_t byteIndex = (bitsOffset + i + j) / 8;
                    uint8_t  byteMask  = 1u << ((bitsOffset + i + j) % 8);
                    byteMask           = flipBits(byteMask);
                    if (metadata[byteIndex] & byteMask) // not sure this works
                    {
                        isRegionFree = false;
                        break;
                    }
                }

                if (isRegionFree)
                {
                    // Mark the blocks as allocated in metadata
                    for (uint32_t j = 0; j < numBlocks; ++j)
                    {
                        uint32_t byteIndex = (bitsOffset + i + j) / 8;
                        uint8_t  byteMask  = 1u << ((bitsOffset + i + j) % 8);
                        byteMask           = flipBits(byteMask);
                        metadata[byteIndex] |= byteMask; // not sure this works
                    }

                    // Calculate the starting address of the allocated blocks
                    uintptr_t blockAddr = poolBase + i * blockSizeBytes;

                    ctx.log("Allocated {} blocks of size {} at address {}",
                            {numBlocks, blockSizeBytes, StrBuf{blockAddr, "0x%zx"}});

                    // TODO perform hooks
                    AllocationInfo info;
                    info.address   = reinterpret_cast<void*>(blockAddr);
                    info.size      = blockSizeBytes * numBlocks;
                    info.alignment = poolBaseAlignment;
                    info.transient = 0;
                    info.tag       = tag;
                    info.sid       = sid;
                    info.freeTime  = 0;
                    info.allocTime = ctx.millisFromStart();
                    m_hooks.allocHook(m_hooks.data, ctx, info);
                    return encode(bufferIdx, blockSizeIndex, reinterpret_cast<void*>(blockAddr));
                }
            }

            ++bufferIdx;
            currentBuffer = currentBuffer->next;
        }

        // No free blocks found, try to allocate a new buffer
        BufferHeader* newBuffer = nullptr;
        newBlock(ctx, pageAllocator, &newBuffer);

        if (!newBuffer)
        {
            ctx.error("Failed to allocate a new buffer for MultiPoolAllocator.");
            return {nullptr, 0};
        }

        // Add the new buffer to the list
        m_lastBuffer->next = newBuffer;
        m_lastBuffer       = newBuffer;

        // Retry allocation
        return allocateBlocks(ctx, pageAllocator, numBlocks, blockSize, tag, sid);
    }

    void MultiPoolAllocator::freeBlocks(LoggingContext& ctx, PageAllocator& pageAllocator, uint32_t numBlocks, TaggedPointer ptr)
    {
        std::lock_guard lock{m_mtx};

        if (ptr == taggedNullptr)
        {
            ctx.error("Attempting to free a null pointer.");
            return;
        }

        // Extract buffer index and block size encoding from the tag
        uintptr_t     blockAddr       = ptr.address();
        uint16_t      tag             = ptr.tag();
        uint16_t      bufferIdx       = bufferIndex(tag);
        uint8_t       blockSizeEnc    = blockSizeEncoding(tag);
        uint32_t      bitsOffset      = 0;
        BufferHeader* currentBuffer   = m_firstBuffer;
        uintptr_t     metadataAddr    = reinterpret_cast<uintptr_t>(currentBuffer) + sizeof(BufferHeader);
        uintptr_t     poolBase        = currentBuffer->poolBase;
        EBlockSize    blockSize       = fromEncoding(blockSizeEnc);
        uint16_t      blockSizeBytes  = toUnderlying(blockSize);
        uint32_t      metadataBaseIdx = 0;

        // the last buffer index value is used by callers to hold special meaning
        assert(bufferIdx != (std::numeric_limits<uint16_t>::max() >> 6u));

        if (blockSizeEnc >= numBlockSizes)
        {
            ctx.error("Invalid block size encoding in tagged pointer.");
            return;
        }

        for (uint8_t i = 0; i < blockSizeEnc; ++i)
        {
            poolBase += m_blocksPerPool[i] * toUnderlying(fromEncoding(i));
            bitsOffset += m_blocksPerPool[i];
        }

        // add the block size until you go past the tagged address. Then you found the base index
        // for the metadata
        for (uintptr_t poolAddr = poolBase; poolAddr < ptr.address(); poolAddr += blockSizeBytes)
        {
            ++metadataBaseIdx;
        }

        // Locate the buffer using the buffer index
        for (uint16_t i = 0; i < bufferIdx && currentBuffer; ++i)
        {
            currentBuffer = currentBuffer->next;
        }

        if (!currentBuffer)
        {
            ctx.error("Buffer index out of bounds.");
            return;
        }

        // Calculate the block index
        if (blockAddr < poolBase || (blockAddr - poolBase) % blockSizeBytes != 0) // not sure this works
        {
            ctx.error("Pointer does not correspond to a valid block in this buffer.");
            return;
        }

        uint32_t blockIndex = static_cast<uint32_t>((blockAddr - poolBase) / blockSizeBytes);

        // Update metadata to mark blocks as free
        uint8_t* metadata = reinterpret_cast<uint8_t*>(metadataAddr);
        for (uint32_t i = 0; i < numBlocks; ++i)
        {
            uint32_t byteIndex = (bitsOffset + metadataBaseIdx + i) / 8;
            uint8_t  byteMask  = 1u << ((bitsOffset + metadataBaseIdx + i) % 8);
            byteMask           = flipBits(byteMask);
            metadata[byteIndex] &= ~byteMask; // not sure this works
        }

        AllocationInfo info;
        info.address = ptr.pointer();
        m_hooks.freeHook(m_hooks.data, ctx, info);
        ctx.log("Freed {} blocks of size {} at address {}", {numBlocks, blockSizeBytes, StrBuf{blockAddr, "0x%zx"}});
    }

    // MemoryContext ---------------------------------------------------------------------------------------------------------

    MemoryContext::MemoryContext(uint32_t                                   pageTrackCapacity,
                                 uint32_t                                   allocTrackCapacity,
                                 std::array<uint32_t, numBlockSizes> const& numBlocksPerPool) :
    tracker{pctx, pageTrackCapacity, allocTrackCapacity},
    pageHooks{
        .allocHook =
            [](void* data, LoggingContext& ctx, PageAllocation const& alloc) {
                auto& tracker = *reinterpret_cast<PageAllocationsTracker*>(data);
                tracker.track(ctx, alloc);
            },
        .freeHook =
            [](void* data, LoggingContext& ctx, PageAllocation const& alloc) {
                auto& tracker = *reinterpret_cast<PageAllocationsTracker*>(data);
                tracker.untrack(ctx, alloc);
            },
        .data = &tracker,
    },
    allocHooks{
        .allocHook =
            [](void* data, LoggingContext& ctx, AllocationInfo const& alloc) {
                auto& tracker = *reinterpret_cast<PageAllocationsTracker*>(data);
                tracker.track(ctx, alloc);
            },
        .freeHook =
            [](void* data, LoggingContext& ctx, AllocationInfo const& alloc) {
                auto& tracker = *reinterpret_cast<PageAllocationsTracker*>(data);
                tracker.untrack(ctx, alloc);
            },
        .cleanTransients =
            [](void* data, LoggingContext& ctx) {
                auto& tracker = *reinterpret_cast<PageAllocationsTracker*>(data);
                tracker.claenTransients(ctx);
            },
        .data = &tracker,
    },
    pageAllocator{pctx, pageHooks},
    stackAllocator{pctx, pageAllocator, allocHooks},
    multiPoolAllocator{pctx, pageAllocator, numBlocksPerPool, allocHooks}
    {
    }

    // stack methods
    void* MemoryContext::stackAllocate(size_t size, size_t alignment, EMemoryTag tag, sid_t sid)
    {
        return stackAllocator.allocate(pctx, pageAllocator, size, alignment, tag, sid);
    }

    void MemoryContext::stackReset() { stackAllocator.reset(pctx, pageAllocator); }

    // pool methods
    TaggedPointer MemoryContext::poolAllocateBlocks(uint32_t numBlocks, EBlockSize blockSize, EMemoryTag tag, sid_t sid)
    {
        return multiPoolAllocator.allocateBlocks(pctx, pageAllocator, numBlocks, blockSize, tag, sid);
    }

    void MemoryContext::poolFreeBlocks(uint32_t numBlocks, TaggedPointer ptr)
    {
        multiPoolAllocator.freeBlocks(pctx, pageAllocator, numBlocks, ptr);
    }

    // clean up everything
    void MemoryContext::cleanup()
    {
        stackAllocator.cleanup(pctx, pageAllocator);
        multiPoolAllocator.cleanup(pctx, pageAllocator);
    }
} // namespace dmt
