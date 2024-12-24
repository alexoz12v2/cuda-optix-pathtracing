module;

#include <bit>
#include <functional>
#include <future>
#include <iostream>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include <cassert>

#if defined(DMT_OS_WINDOWS)
#include <windows.h>
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
        Job copy;

        // forever
        while (true)
        {
            // wait for a job to become available
            {
                std::unique_lock<SpinLock> lk{threadPool->m_mtx};
                if (threadPool->m_shutdownRequested)
                {
                    break;
                }

                threadPool->m_cv.wait(lk, [&threadPool]() { return threadPool->m_ready; });
                if (threadPool->m_shutdownRequested)
                {
                    break;
                }

                // copy the next job
                copy = threadPool->nextJob();
            }

            assert(copy.func && "bad function call in job");
            if (copy.func)
            {
                copy.func(copy.data);
            }
        }
    }

    ThreadPoolV2::ThreadPoolV2(PlatformContext&    ctx,
                               PageAllocator&      pageAllocator,
                               MultiPoolAllocator& multiPoolAllocator,
                               StackAllocator&     stackAllocator)
    {
        static_assert(numBlocks >= 5);
        static constexpr uint32_t threadBlocks = 3;
        static_assert(threadBlocks > 1);
        std::lock_guard<SpinLock> lk{m_mtx};

        // allocate the 256 Bytes block for the index
        // TODO: manage allocation of more blocks at a time
        m_pIndex = multiPoolAllocator.allocateBlocks(ctx, pageAllocator, numBlocks, EBlockSize::e256B);
        if (m_pIndex == taggedNullptr)
        {
            ctx.error("Couldn't allocate {} index blocks for threadPool", {numBlocks});
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
            assert(alignTo(ptr, toUnderlying(EBlockSize::e32B)) == 0);

            // store it in the job index
            index.data.ptrs[i] = TaggedPointer{ptr, nullTag};

            // zero initialize memory of the block
            std::memset(ptr, 0, sizeof(JobNode256B));

            offset += blockSz;
        }

#undef max
#undef min
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
                ptThreadsNext = multiPoolAllocator.allocateBlocks(ctx, pageAllocator, 1, EBlockSize::e256B);
                if (ptThreadsNext == taggedNullptr)
                {
                    ctx.error("Could not allocate additional block of 256 B index {} for the threadpool", {i});
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
        std::lock_guard<SpinLock> lk{m_mtx};
        m_ready = true;
        m_cv.notify_one();
    }

    void ThreadPoolV2::pauseJobs()
    {
        assert(m_pIndex != taggedNullptr);
        std::lock_guard<SpinLock> lk{m_mtx};
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

    void ThreadPoolV2::cleanup(PlatformContext& ctx, PageAllocator& pageAllocator, MultiPoolAllocator& multiPoolAllocator)
    {
        // cycle through all Tagged pointers and count the number of true taggeed pointers
        uint32_t       trueTaggedPointerCount = 0;
        constexpr auto incrementCount         = [](void* p, TaggedPointer)
        {
            auto& cnt = *reinterpret_cast<uint32_t*>(p);
            ++cnt;
        };
        constexpr auto storePtr = [](void* p, TaggedPointer tp)
        {
            auto& pair           = *reinterpret_cast<BufferCountPair*>(p);
            pair.p[pair.count++] = tp;
        };

        {
            std::lock_guard<SpinLock> lk{m_mtx};

            // 1. account for all index blocks
            forEachTrueJobIndexBlock(incrementCount, &trueTaggedPointerCount);

            // 3. account for all thread blocks (signal all threads for destruction and join them all in the meantime)
            m_shutdownRequested = true;
        } // lock guard scope

        m_cv.notify_all();

        {
            std::lock_guard<SpinLock> lk{m_mtx};

            forEachTrueThreadBlock(incrementCount, &trueTaggedPointerCount, true);

            // allocate a unique ptr with the necessary space to hold the true tagged pointers (we don't care
            // about memory here as this should be the shutdown of the application)
            BufferCountPair t{.p = std::make_unique<TaggedPointer[]>(trueTaggedPointerCount), .count = 0};

            // copy all true tagged pointers in the unique pointer buffer
            forEachTrueJobIndexBlock(storePtr, &t);
            forEachTrueThreadBlock(storePtr, &t, false);

            // free all memory
            for (uint32_t i = 0; i < trueTaggedPointerCount; ++i)
            {
                multiPoolAllocator.freeBlocks(ctx, pageAllocator, 1, t.p[i]);
            }

            // free the index block, the only one with `numBlocks` adjacent
            multiPoolAllocator.freeBlocks(ctx, pageAllocator, numBlocks, m_pIndex);

            // bookkeeping
            m_ready    = false;
            m_pIndex   = taggedNullptr;
            m_pThreads = taggedNullptr;
        } // lock guard scope
    }

    static TaggedPointer tryAlloc(PlatformContext&    ctx,
                                  PageAllocator&      pageAllocator,
                                  MultiPoolAllocator& multiPoolAllocator,
                                  EJobLayer           layer)
    {
        TaggedPointer newJobBlock = multiPoolAllocator.allocateBlocks(ctx, pageAllocator, 1, EBlockSize::e256B);
        if (newJobBlock == taggedNullptr)
        {
            ctx.error("failed to allocate new job block for layer {}", {toUnderlying(layer)});
        }

        return newJobBlock;
    }

    void ThreadPoolV2::addJob(PlatformContext&    ctx,
                              PageAllocator&      pageAllocator,
                              MultiPoolAllocator& multiPoolAllocator,
                              Job const&          job,
                              EJobLayer           layer)
    {
        std::lock_guard<SpinLock> lk{m_mtx};
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
                                jobSlot = job;
                                ++m_numJobs;
                                return;
                            }
                        }
                        // if all job slots are occupied, move to the next job node
                        // and if this is the last one, try to allocate a new one
                        if (jobNode->next == taggedNullptr)
                        {
                            TaggedPointer newJobBlock = tryAlloc(ctx, pageAllocator, multiPoolAllocator, layer);

                            jobNode->next = newJobBlock;
                            std::memset(newJobBlock.pointer(), 0, sizeof(JobNode256B));
                        }
                    }
                } // pIndexNode->data.layers[i] == layer
                else if (pIndexNode->data.layers[i] == EJobLayer::eEmpty)
                { // this is a new layer, allocate a new job block
                    pIndexNode->data.layers[i] = layer;
                    TaggedPointer newJobBlock  = pIndexNode->data.ptrs[i] == taggedNullptr
                                                     ? tryAlloc(ctx, pageAllocator, multiPoolAllocator, layer)
                                                     : pIndexNode->data.ptrs[i];
                    pIndexNode->data.ptrs[i]   = newJobBlock;
                    std::memset(newJobBlock.pointer(), 0, sizeof(JobNode256B));

                    // add the job in the new block
                    auto* newJobNode         = newJobBlock.pointer<JobNode256B>();
                    newJobNode->data.jobs[0] = job;
                    ++m_numJobs;
                    return;
                } // pIndexNode->data.layers[i] == EJobLayer::eEmpty
            } // loop over layer cardinality of the current index block

            // if you didn't find any slots from the current index block, go to the next one
            lastPtr    = pIndexNode->next;
            pIndexNode = lastPtr.pointer<IndexNode256B>(); // can be nullptr
        }

        // if we exhausted the index, allocate a new block, for the index...
        TaggedPointer newIndexBlock = multiPoolAllocator.allocateBlocks(ctx, pageAllocator, 1, EBlockSize::e256B);
        if (newIndexBlock == taggedNullptr)
        {
            ctx.error("Failed to allocate new index block for thread pool");
            std::abort();
        }

        if (lastPtr != taggedNullptr)
        {
            lastPtr.pointer<IndexNode256B>()->next = newIndexBlock;
        }
        else // this shouldn't be hit
        {
            ctx.warn("You shouldn't be here");
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
        TaggedPointer newJobBlock = tryAlloc(ctx, pageAllocator, multiPoolAllocator, layer);
        newIndex->data.ptrs[0]    = newJobBlock;
        auto* newJobNode          = newJobBlock.pointer<JobNode256B>();
        std::memset(newJobNode, 0, sizeof(JobNode256B));

        newJobNode->data.jobs[0] = job;
        ++m_numJobs;
    }

    Job ThreadPoolV2::nextJob()
    {
        std::lock_guard<SpinLock> lk{m_mtx};
        assert(m_pIndex != taggedNullptr);

        auto* pIndexNode = m_pIndex.pointer<IndexNode256B>();
        while (pIndexNode != nullptr)
        { // for each index node, for each layer in the node different than `eEmpty`
            for (uint32_t i = 0; i < layerCardinality; ++i)
            {
                if (pIndexNode->data.layers[i] != EJobLayer::eEmpty)
                {
                    TaggedPointer jobBlockPtr = pIndexNode->data.ptrs[i];
                    auto*         jobNode     = jobBlockPtr.pointer<JobNode256B>();
                    // iterate the entire linked list to find the first job with non null `func`
                    while (jobNode != nullptr)
                    {
                        for (auto& jobSlot : jobNode->data.jobs)
                        {
                            if (jobSlot.func != nullptr)
                            {
                                Job copy     = jobSlot;
                                jobSlot.func = nullptr;
                                --m_numJobs;

                                // TODO maybe: if this is the last job in the node, free it?
                                return copy;
                            }
                        } // iteration over job slots

                        jobNode = jobNode->next.pointer<JobNode256B>();
                    } // iteration over job nodes

                    // if there are no jobs left in this index eleent, mark it as empty
                    pIndexNode->data.layers[i] = EJobLayer::eEmpty;
                } // pIndexNode->data.layers[i] != EJobLayer::eEmpty
            }

            pIndexNode = pIndexNode->next.pointer<IndexNode256B>();
        }

        // no jobs left
        return {};
    }
} // namespace dmt