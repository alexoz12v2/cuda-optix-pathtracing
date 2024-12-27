module;

#include "platform-os-utils.h"

#include <atomic>
#include <bit>
#include <functional>
#include <future>
#include <iostream>
#include <queue>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

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
    ThreadPool::ThreadPool(int const size) :
    busyThreads(size),
    m_threads(std::vector<std::thread>(size)),
    m_shutdownRequested(false)
    {
        for (size_t i = 0; i < size; ++i)
        {
            m_threads[i] = std::thread(ThreadWorker(this));
        }
    }

    ThreadPool::~ThreadPool()
    {
        Shutdown();
    }

    void ThreadPool::Shutdown()
    {
        //define a scope for lock
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_shutdownRequested = true;
            m_conditionVariable.notify_all(); //wake-up all threads
        }

        for (size_t i = 0; i < m_threads.size(); ++i)
        {
            if (m_threads[i].joinable())
            {
                m_threads[i].join();
            }
        }
        std::cout << "Derstroyed Threads" << std::endl;
    }


    int ThreadPool::QueueSize()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_queue.size();
    }

    //set the pointer of the pool so the worker can access to the shared queue and mutex
    ThreadPool::ThreadWorker::ThreadWorker(ThreadPool* pool) : m_threadPool(pool)
    {
    }
    //execute immediately starts to execute
    void ThreadPool::ThreadWorker::operator()()
    {
        //acuire the lock of the mutex and given back
        //std::unique_lock<std::mutex> lock(thread_pool->mutex, std::defer_lock);
        std::unique_lock<std::mutex> lock(m_threadPool->m_mutex);
        //continusly run
        while (!m_threadPool->m_shutdownRequested || (m_threadPool->m_shutdownRequested && !m_threadPool->m_queue.empty()))
        {
            m_threadPool->busyThreads--;
            //thrad stops and gives back the mutex until it is woken up again
            //it will check the condition-> true continue, false go back into sleep
            m_threadPool->m_conditionVariable.wait(lock,
                                                   [this] {
                                                       return this->m_threadPool->m_shutdownRequested ||
                                                              !this->m_threadPool->m_queue.empty();
                                                   });
            m_threadPool->busyThreads++;
            //check if there is a task
            if (!this->m_threadPool->m_queue.empty())
            {
                auto func = m_threadPool->m_queue.front();
                m_threadPool->m_queue.pop();
                lock.unlock();
                func();
                lock.lock();
            }
        }
    }

    // SpinLock -------------------------------------------------------------------------------------------------------
    void SpinLock::lock() noexcept
    {
        for (;;)
        {
            // Optimistically assume the lock is free on the first try
            if (!lock_.exchange(true, std::memory_order_acquire))
            {
                return;
            }
            // Wait for lock to be released without generating cache misses
            while (lock_.load(std::memory_order_relaxed))
            {
                // Issue X86 PAUSE or ARM YIELD instruction to reduce contention between
                // hyper-threads
#if defined(DMT_COMPILER_GCC) || defined(DMT_COMPILER_CLANG)
                __builtin_ia32_pause();
#elif defined(DMT_OS_WINDOWS)
                YieldProcessor();
#endif
            }
        }
    }

    bool SpinLock::try_lock() noexcept
    {
        // First do a relaxed load to check if lock is free in order to prevent
        // unnecessary cache misses if someone does while(!try_lock())
        return !lock_.load(std::memory_order_relaxed) && !lock_.exchange(true, std::memory_order_acquire);
    }

    void SpinLock::unlock() noexcept
    {
        lock_.store(false, std::memory_order_release);
    }

    // ThreadPoolV2 ---------------------------------------------------------------------------------------------------
    struct BufferCountPair
    {
        std::unique_ptr<TaggedPointer[]> p;
        uint32_t                         count;
    };

    static void jobWorkerThread(ThreadPoolV2* threadPool)
    {
        Job       copy;
        EJobLayer currentLayer   = EJobLayer::eEmpty;
        bool      otherRemaining = false;

        // forever
        while (true)
        {
            // wait for a job to become available
            {
                std::unique_lock<decltype(threadPool->m_mtx)> lk{threadPool->m_mtx};
                if (threadPool->m_shutdownRequested)
                {
                    break;
                }

                EJobLayer activeLayer = currentLayer;
                threadPool->m_cv.wait(lk,
                                      [&threadPool, &activeLayer]() {
                                          return (threadPool->m_ready && !threadPool->otherLayerActive(activeLayer)) ||
                                                 threadPool->m_shutdownRequested;
                                      });
                if (threadPool->m_shutdownRequested)
                {
                    break;
                }

                // copy the next job
                copy = threadPool->nextJob(otherRemaining, currentLayer);

                if (activeLayer != currentLayer)
                {
                    threadPool->m_activeLayer = currentLayer;
                }
            }

            if (copy.func)
            {
                threadPool->m_jobsInFlight.test_and_set();
                copy.func(copy.data);
                if (!otherRemaining)
                {
                    threadPool->m_jobsInFlight.clear();
                }
            }
        }
    }

    ThreadPoolV2::ThreadPoolV2(MemoryContext& ctx)
    {
        using namespace std::string_view_literals;
        static_assert(numBlocks >= 5);
        static constexpr uint32_t threadBlocks = 3;
        static_assert(threadBlocks > 1);
        std::lock_guard<decltype(m_mtx)> lk{m_mtx};
        m_memoryTrackingSid = ctx.strTable.intern("ThreadPool data (Index block, Job Block, or Thread Block)"sv);

        // allocate the 256 Bytes block for the index
        // TODO: manage allocation of more blocks at a time
        // TODO: sid
        m_pIndex = ctx.poolAllocateBlocks(numBlocks, EBlockSize::e256B, EMemoryTag::eJob, m_memoryTrackingSid);
        if (m_pIndex == taggedNullptr)
        {
            ctx.pctx.error("Couldn't allocate {} index blocks for threadPool", {numBlocks});
            std::abort();
        }

        // initialize index with `ceilDiv((numBlocks - 1), 2)` pointers, the rest taggedNullptr,
        // and all layers as empty
        auto& index = *m_pIndex.pointer<IndexNode256B>();
        index.next  = taggedNullptr;
        for (uint32_t i = 0; i != layerCardinality; ++i)
        {
            index.data.ptrs[i] = taggedNullptr;
        }

        for (uint32_t i = 0; i != layerCardinality; ++i)
        {
            index.data.layers[i] = EJobLayer::eEmpty;
        }

        uint32_t jobBlocks = numBlocks - 1 - threadBlocks;
        uint32_t offset    = blockSz;
        for (uint32_t i = 0; i != jobBlocks; ++i)
        {
            void* ptr = std::bit_cast<void*>(m_pIndex.address() + offset);
            assert(alignTo(ptr, toUnderlying(EBlockSize::e32B)) == ptr);

            // store it in the job index
            index.data.ptrs[i] = TaggedPointer{ptr, nullTag};

            // zero initialize memory of the block
            std::memset(ptr, 0, sizeof(JobNode256B));

            offset += blockSz;
        }

        assert(std::thread::hardware_concurrency() >= 1);
        uint32_t numThreads = std::max(std::thread::hardware_concurrency() - 1u, 1u);
        // TODO hsndle thread construction
        // 1. see how many additional blocks you need to accomodate the `numThreads` and allocate them
        // 2. arrange the blocks in a linked list structure, and store the `activeCount`
        // 3. construct all threads with the entry point

        uint32_t threadOffset       = offset;
        m_pThreads                  = TaggedPointer{std::bit_cast<void*>(m_pIndex.address() + threadOffset), nullTag};
        TaggedPointer ptThreads     = m_pThreads;
        TaggedPointer ptThreadsNext = TaggedPointer{std::bit_cast<void*>(m_pIndex.address() + threadOffset + blockSz),
                                                    nullTag};
        for (uint32_t i = 0; i != threadBlocks; ++i)
        {
            uint8_t numTs = static_cast<uint8_t>(std::min(numThreads, ThreadBlob::numTs));
            numThreads -= numTs;
            prepareThreadNode(ptThreads, ptThreadsNext, numTs);

            offset += blockSz;
            ptThreads     = i < threadBlocks - 1 ? ptThreadsNext : ptThreads;
            ptThreadsNext = i < threadBlocks - 2
                                ? TaggedPointer(std::bit_cast<void*>(ptThreadsNext.address() + blockSz), nullTag)
                                : taggedNullptr;
        }

        if (numThreads != 0)
        {
            uint32_t additionalBlocks = ceilDiv(numThreads, ThreadBlob::numTs);
            uint32_t threadsResidual  = numThreads % ThreadBlob::numTs;
            for (uint32_t i = 0; i < additionalBlocks; ++i)
            {
                uint8_t threadNum = static_cast<uint8_t>(i == additionalBlocks - 1 ? threadsResidual : ThreadBlob::numTs);
                ptThreadsNext = ctx.poolAllocateBlocks(1, EBlockSize::e256B, EMemoryTag::eJob, m_memoryTrackingSid);
                if (ptThreadsNext == taggedNullptr)
                {
                    ctx.pctx.error("Could not allocate additional block of 256 B index {} for the threadpool", {i});
                    std::abort();
                }

                ptThreads.pointer<ThreadNode256B>()->next = ptThreadsNext;
                prepareThreadNode(ptThreadsNext, taggedNullptr, threadNum);
                ptThreads = ptThreadsNext;
            }
        }
    }

    void ThreadPoolV2::prepareThreadNode(TaggedPointer current, TaggedPointer next, uint8_t activeCount)
    {
        auto& threadNode = *current.pointer<ThreadNode256B>();

        threadNode.data.activeCount = activeCount;
        threadNode.next             = next;
        for (uint32_t j = 0; j < activeCount; ++j)
        {
            std::construct_at(&threadNode.data.ts[j].t, jobWorkerThread, this);
        }
    }

    void ThreadPoolV2::kickJobs()
    {
        assert(m_pIndex != taggedNullptr);
        {
            std::lock_guard<decltype(m_mtx)> lk{m_mtx};
            m_ready = true;
        }
        m_cv.notify_one();
    }

    void ThreadPoolV2::pauseJobs()
    {
        assert(m_pIndex != taggedNullptr);
        std::lock_guard<decltype(m_mtx)> lk{m_mtx};
        m_ready = false;
    }

    void ThreadPoolV2::forEachTrueJobIndexBlock(void (*func)(void*, TaggedPointer), void* p)
    {
        TaggedPointer ptIndexNode = m_pIndex;
        auto*         pIndexNode  = ptIndexNode.pointer<IndexNode256B>();
        bool          first       = true;
        while (pIndexNode->next != taggedNullptr)
        {
            if (!first && isTrueTaggedPointer(ptIndexNode))
            {
                func(p, ptIndexNode);
            }
            first = false;

            // 2. account for all job blocks inside the current index block
            for (uint32_t i = 0; i < layerCardinality; ++i)
            {
                TaggedPointer ptJobBlock = pIndexNode->data.ptrs[i];
                while (ptJobBlock != taggedNullptr)
                {
                    if (isTrueTaggedPointer(ptJobBlock))
                    {
                        func(p, ptJobBlock);
                    }
                    ptJobBlock = ptJobBlock.pointer<JobNode256B>()->next;
                }
            }

            ptIndexNode = pIndexNode->next;
            pIndexNode  = ptIndexNode.pointer<IndexNode256B>();
        }
    }

    void ThreadPoolV2::forEachTrueThreadBlock(void (*func)(void*, TaggedPointer), void* p, bool joinAll)
    {
        TaggedPointer ptThreadBlock = m_pThreads;
        while (ptThreadBlock != taggedNullptr)
        {
            auto& threadBlock = *ptThreadBlock.pointer<ThreadNode256B>();
            if (isTrueTaggedPointer(ptThreadBlock))
            {
                func(p, ptThreadBlock);
            }

            if (joinAll)
            {
                for (uint8_t i = 0; i < threadBlock.data.activeCount; ++i)
                {
                    ThreadWrapper& thread = threadBlock.data.ts[i];
                    while (!thread.t.joinable())
                    { // twiddle thumbs
                    }
                    thread.t.join();
                    std::destroy_at(&thread.t);
                }
            }

            ptThreadBlock = threadBlock.next;
        }
    }

    void ThreadPoolV2::cleanup(MemoryContext& ctx)
    {
        // cycle through all Tagged pointers and count the number of true taggeed pointers
        uint32_t       trueTaggedPointerCount = 0;
        constexpr auto incrementCount         = [](void* p, TaggedPointer tp)
        {
            assert(tp.tag() != nullTag);
            auto& cnt = *reinterpret_cast<uint32_t*>(p);
            ++cnt;
        };
        constexpr auto storePtr = [](void* p, TaggedPointer tp)
        {
            auto& pair           = *reinterpret_cast<BufferCountPair*>(p);
            pair.p[pair.count++] = tp;
        };
        constexpr auto nothing = [](void* p, TaggedPointer tp) {};

        {
            std::lock_guard<decltype(m_mtx)> lk{m_mtx};

            // 1. account for all index blocks
            forEachTrueJobIndexBlock(incrementCount, &trueTaggedPointerCount);

            // 3. account for all thread blocks (signal all threads for destruction and join them all in the meantime)
            m_shutdownRequested = true;
        } // lock guard scope

        m_cv.notify_all();
        forEachTrueThreadBlock(nothing, nullptr, true);

        {
            std::lock_guard<decltype(m_mtx)> lk{m_mtx};

            forEachTrueThreadBlock(incrementCount, &trueTaggedPointerCount, false);

            // allocate a unique ptr with the necessary space to hold the true tagged pointers (we don't care
            // about memory here as this should be the shutdown of the application)
            BufferCountPair t{.p = std::make_unique<TaggedPointer[]>(trueTaggedPointerCount), .count = 0};

            // copy all true tagged pointers in the unique pointer buffer
            forEachTrueJobIndexBlock(storePtr, &t);
            forEachTrueThreadBlock(storePtr, &t, false);

            // free all memory
            for (uint32_t i = 0; i < trueTaggedPointerCount; ++i)
            {
                ctx.poolFreeBlocks(1, t.p[i]);
            }

            // free the index block, the only one with `numBlocks` adjacent
            ctx.poolFreeBlocks(numBlocks, m_pIndex);

            // bookkeeping
            m_ready    = false;
            m_pIndex   = taggedNullptr;
            m_pThreads = taggedNullptr;
        } // lock guard scope
    }

    static TaggedPointer tryAlloc(MemoryContext& ctx, EJobLayer layer, sid_t sid)
    {
        TaggedPointer newJobBlock = ctx.poolAllocateBlocks(1, EBlockSize::e256B, EMemoryTag::eJob, sid);
        if (newJobBlock == taggedNullptr)
        {
            ctx.pctx.error("failed to allocate new job block for layer {}", {toUnderlying(layer)});
        }

        return newJobBlock;
    }

    void ThreadPoolV2::addJob(MemoryContext& ctx, Job const& job, EJobLayer layer)
    {
        std::lock_guard<decltype(m_mtx)> lk{m_mtx};
        assert(m_pIndex != taggedNullptr);

        auto* pIndexNode = m_pIndex.pointer<IndexNode256B>();
        // track the next pointer of the last node which was already allocated, such that
        // we can know if an allocation happened
        TaggedPointer lastPtr = taggedNullptr;

        while (pIndexNode != nullptr)
        { // check if the layer is already in the index
            for (uint32_t i = 0; i < layerCardinality; ++i)
            { // check either if layer is in the index or if there's an empty slot
                if (pIndexNode->data.layers[i] == layer)
                { // add job to this layers's job linked list
                    TaggedPointer jobBlockPtr = pIndexNode->data.ptrs[i];
                    auto*         jobNode     = jobBlockPtr.pointer<JobNode256B>();
                    assert(jobNode && "layer not empty, but somehow nullptr job linked list");
                    while (jobNode != nullptr)
                    { // check for space in the current job node
                        for (auto& jobSlot : jobNode->data.jobs)
                        { // if the slot is available, add the job
                            if (jobSlot.func == nullptr)
                            {
                                ++jobBlockPtr.pointer<JobNode256B>()->data.counter;
                                jobSlot = job;
                                ++m_numJobs;
                                return;
                            }
                        }
                        // if all job slots are occupied, move to the next job node
                        // and if this is the last one, try to allocate a new one
                        if (jobNode->next == taggedNullptr)
                        {
                            TaggedPointer newJobBlock = tryAlloc(ctx, layer, m_memoryTrackingSid);

                            jobNode->next = newJobBlock;
                            std::memset(newJobBlock.pointer(), 0, sizeof(JobNode256B));
                        }

                        jobNode = jobNode->next.pointer<JobNode256B>();
                    }
                } // pIndexNode->data.layers[i] == layer
                else if (pIndexNode->data.layers[i] == EJobLayer::eEmpty)
                { // this is a new layer, allocate a new job block
                    pIndexNode->data.layers[i] = layer;
                    TaggedPointer newJobBlock  = pIndexNode->data.ptrs[i] == taggedNullptr
                                                     ? tryAlloc(ctx, layer, m_memoryTrackingSid)
                                                     : pIndexNode->data.ptrs[i];
                    pIndexNode->data.ptrs[i]   = newJobBlock;
                    std::memset(newJobBlock.pointer(), 0, sizeof(JobNode256B));

                    // add the job in the new block
                    auto* newJobNode         = newJobBlock.pointer<JobNode256B>();
                    newJobNode->data.jobs[0] = job;
                    newJobNode->data.counter = 1;
                    ++m_numJobs;
                    return;
                } // pIndexNode->data.layers[i] == EJobLayer::eEmpty
            } // loop over layer cardinality of the current index block

            // if you didn't find any slots from the current index block, go to the next one
            lastPtr    = pIndexNode->next;
            pIndexNode = lastPtr.pointer<IndexNode256B>(); // can be nullptr
        }

        // if we exhausted the index, allocate a new block, for the index...
        TaggedPointer newIndexBlock = ctx.poolAllocateBlocks(1, EBlockSize::e256B, EMemoryTag::eJob, m_memoryTrackingSid);
        if (newIndexBlock == taggedNullptr)
        {
            ctx.pctx.error("Failed to allocate new index block for thread pool");
            std::abort();
        }

        if (lastPtr != taggedNullptr)
        {
            lastPtr.pointer<IndexNode256B>()->next = newIndexBlock;
        }
        else // this shouldn't be hit
        {
            ctx.pctx.warn("You shouldn't be here");
            m_pIndex = newIndexBlock;
        }

        auto* newIndex = newIndexBlock.pointer<IndexNode256B>();
        std::memset(newIndex, 0, sizeof(IndexNode256B));
        newIndex->data.layers[0] = layer;
        for (uint32_t i = 1; i < layerCardinality; ++i)
        {
            newIndex->data.layers[i] = EJobLayer::eEmpty;
        }

        // ... and a new block for the job
        TaggedPointer newJobBlock = tryAlloc(ctx, layer, m_memoryTrackingSid);
        newIndex->data.ptrs[0]    = newJobBlock;
        auto* newJobNode          = newJobBlock.pointer<JobNode256B>();
        std::memset(newJobNode, 0, sizeof(JobNode256B));

        newJobNode->data.jobs[0] = job;
        newJobNode->data.counter = 1;
        ++m_numJobs;
    }

    Job ThreadPoolV2::nextJob(bool& otherJobsRemaining, EJobLayer& outLayer)
    {
        assert(m_pIndex != taggedNullptr);

        // Find the smallest layer with jobs
        TaggedPointer jobBlockPtr = getSmallestLayer(outLayer);
        if (jobBlockPtr == taggedNullptr)
        {
            // No jobs left
            otherJobsRemaining = false;
            outLayer           = EJobLayer::eEmpty;
            m_ready            = false;
            return {};
        }

        auto* jobNode = jobBlockPtr.pointer<JobNode256B>();
        while (jobNode != nullptr)
        {
            for (auto& jobSlot : jobNode->data.jobs)
            {
                if (jobSlot.func != nullptr)
                {
                    Job copy     = jobSlot;
                    jobSlot.func = nullptr;

                    uint64_t value = --jobBlockPtr.pointer<JobNode256B>()->data.counter;
                    --m_numJobs;

                    // Determine if other jobs remain in this layer
                    otherJobsRemaining = (value != 0);
                    return copy;
                }
            }

            jobNode = jobNode->next.pointer<JobNode256B>();
        }

        // If we exhaust the current layer, mark it as empty
        auto* pIndexNode = m_pIndex.pointer<IndexNode256B>();
        while (pIndexNode != nullptr)
        {
            for (uint32_t i = 0; i < layerCardinality; ++i)
            {
                if (pIndexNode->data.layers[i] == outLayer)
                {
                    pIndexNode->data.layers[i] = EJobLayer::eEmpty;
                    break;
                }
            }
            pIndexNode = pIndexNode->next.pointer<IndexNode256B>();
        }

        // No jobs left in the layer
        otherJobsRemaining = false;
        return {};
    }

    TaggedPointer ThreadPoolV2::getSmallestLayer(EJobLayer& outLayer) const
    {
        assert(m_pIndex != taggedNullptr);

        TaggedPointer smallestJobBlock = taggedNullptr;
        EJobLayer     smallestLayer    = EJobLayer::eEmpty;

        auto* pIndexNode = m_pIndex.pointer<IndexNode256B>();
        while (pIndexNode != nullptr)
        {
            for (uint32_t i = 0; i < layerCardinality; ++i)
            {
                EJobLayer currentLayer = pIndexNode->data.layers[i];
                if (currentLayer != EJobLayer::eEmpty &&
                    (smallestLayer == EJobLayer::eEmpty || currentLayer < smallestLayer))
                {
                    smallestLayer    = currentLayer;
                    smallestJobBlock = pIndexNode->data.ptrs[i];
                }
            }
            pIndexNode = pIndexNode->next.pointer<IndexNode256B>();
        }

        outLayer = smallestLayer;
        return smallestJobBlock;
    }

    bool ThreadPoolV2::otherLayerActive(EJobLayer& inOutLayer) const
    {
        if (inOutLayer == m_activeLayer)
            return false;

        inOutLayer = m_activeLayer;
        return m_activeLayer != EJobLayer::eEmpty && m_jobsInFlight.test();
    }

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

    static bool initFile(LoggingContext& pctx, char const* filePath, Win32ChunkedFileReader& data)
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

    ChunkedFileReader::ChunkedFileReader(LoggingContext& pctx, char const* filePath, uint32_t chunkSize)
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

    ChunkedFileReader::ChunkedFileReader(LoggingContext& pctx,
                                         char const*     filePath,
                                         uint32_t        chunkSize,
                                         uint8_t         numBuffers,
                                         uintptr_t*      pBuffers)
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

    bool ChunkedFileReader::requestChunk(LoggingContext& pctx, void* chunkBuffer, uint32_t chunkNum)
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

    bool ChunkedFileReader::waitForPendingChunk(LoggingContext& pctx, uint32_t timeoutMillis)
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
                            // TODO handle better
                            uint32_t length = win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
                            StrBuf   view{sErrorBuffer.data(), static_cast<int32_t>(length)};
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

    ChunkedFileReader::InputIterator ChunkedFileReader::Range::begin()
    {
#if defined(DMT_OS_WINDOWS)
        Win32ChunkedFileReader const& data = *reinterpret_cast<Win32ChunkedFileReader const*>(pData);
        assert(data.numChunks >= chunkNum + numChunks);
#elif defined(DMT_OS_LINUX)
#error "todo"
#else
#error "platform not supported"
#endif
        return ++InputIterator(pData, chunkNum, numChunks);
    }

} // namespace dmt