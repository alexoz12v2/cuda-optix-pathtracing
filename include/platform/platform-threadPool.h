#pragma once

#include "dmtmacros.h"
// Keep in sync with .cppm
#include <array>
#include <atomic>
#include <condition_variable>
#include <limits>
#include <thread>

#include <compare>

// used by threadpool v1
#include <functional>
#include <future>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

#include <cstdint>

#if defined(DMT_INTERFACE_AS_HEADER)
// Keep in sync with .cppm
#include <platform/platform-memory.h>
#else
import <platform/platform-memory.h>;
#endif

DMT_MODULE_EXPORT dmt {
    // https://rigtorp.se/spinlock/
    // should be usable with lock_guard
    // TODO move implememntation to cpp
    // implements the NamedRequireemnt BasicLockable https://en.cppreference.com/w/cpp/named_req/BasicLockable
    struct SpinLock
    {
        std::atomic<bool> lock_ = {0};

        void lock() noexcept;

        bool try_lock() noexcept;

        void unlock() noexcept;
    };

    using JobSignature = void (*)(uintptr_t data);

    /**
     * Enum containing tagged job priorities. Since Thread Pool based job scheduling
     * Doesn't support a "yield" mechanism, we need to execute jobs in a topologically
     * sorted order. We'll ensure this givin to each job a "layer", guaranteeing that
     * all jobs belonging to layer i are executed before the jobs on layer i + 1
     * Using a typed enum allows us to assign more tags to the same number and rearrange in
     * a centralised way job execution.
     * We'll use the layer as an index into an array of pointers. Say 32 pointers (see `LayerCardinality` of the ThreadPool class)
     */
    enum class EJobLayer : uint32_t
    {
        eDefault = 0,
        eTest0   = 0,
        eTest1   = 1,
        eEmpty   = static_cast<uint32_t>(-1),
    };

    constexpr uint32_t toUnderlying(EJobLayer a)
    {
        return static_cast<uint32_t>(a);
    }

    constexpr std::strong_ordering operator<=>(EJobLayer a, EJobLayer b)
    {
        return toUnderlying(a) <=> toUnderlying(b);
    }

    struct Job
    {
        JobSignature func;
        uintptr_t    data;
    };
    static_assert(sizeof(Job) == 16 && alignof(Job) == 8);

    struct JobBlob
    {
        std::array<Job, 15> jobs;
        uint64_t            counter;
    };
    template struct PoolNode<JobBlob, EBlockSize::e256B>;
    using JobNode256B = PoolNode<JobBlob, EBlockSize::e256B>;

    inline constexpr uint32_t layerCardinality = 20;
    struct IndexArray
    {
        std::array<TaggedPointer, layerCardinality> ptrs;
        std::array<EJobLayer, layerCardinality>     layers;
        unsigned char                               padding[8];
    };
    template struct PoolNode<IndexArray, EBlockSize::e256B>;
    using IndexNode256B = PoolNode<IndexArray, EBlockSize::e256B>;

    /**
     * Class which wraps a thread and ensures that the thread struct size is uniform across the supported platforms
     * clang and gcc have, as of now (gcc 14 and clang 17) sizeof(std::thread) == 8, while msvc 19.40 has sizeof(std::thread) == 16
     */
    struct ThreadWrapper
    {
        static constexpr uint32_t tSize = 16;
        std::thread               t;
#if !defined(DMT_COMPILER_MSVC)
        unsigned char padding[tSize - sizeof(std::thread)];
#endif
    };
    static_assert(sizeof(ThreadWrapper) == ThreadWrapper::tSize && alignof(ThreadWrapper) == 8);
    struct ThreadBlob
    {
        static constexpr uint32_t        numTs = 15;
        std::array<ThreadWrapper, numTs> ts;
        uint8_t                          activeCount;
    };

    template struct PoolNode<ThreadBlob, EBlockSize::e256B>;
    using ThreadNode256B = PoolNode<ThreadBlob, EBlockSize::e256B>;

    class ThreadPoolV2
    {
#undef max
        static constexpr uint16_t nullTag     = 0xFFFU;
        static constexpr uint32_t blockSz     = toUnderlying(EBlockSize::e256B);
        static constexpr uint32_t numBlocks   = 10;
        static constexpr uint32_t num32Blocks = ceilDiv(layerCardinality * static_cast<uint32_t>(sizeof(TaggedPointer)),
                                                        layerCardinality);
        friend void               jobWorkerThread(ThreadPoolV2* threadPool);

    public:
        ThreadPoolV2(PlatformContext&    ctx,
                     PageAllocator&      pageAllocator,
                     MultiPoolAllocator& multiPoolAllocator,
                     StackAllocator&     stackAllocator);
        ThreadPoolV2(ThreadPoolV2 const&)                = delete;
        ThreadPoolV2(ThreadPoolV2&&) noexcept            = delete;
        ThreadPoolV2& operator=(ThreadPoolV2 const&)     = delete;
        ThreadPoolV2& operator=(ThreadPoolV2&&) noexcept = delete;

        void addJob(PlatformContext&    ctx,
                    PageAllocator&      pageAllocator,
                    MultiPoolAllocator& multiPoolAllocator,
                    Job const&          job,
                    EJobLayer           layer);

        void cleanup(PlatformContext& ctx, PageAllocator& pageAllocator, MultiPoolAllocator& multiPoolAllocator);

        void kickJobs();

        void pauseJobs();

        bool otherLayerActive(EJobLayer& layer) const;

    private:
        static constexpr bool isTrueTaggedPointer(TaggedPointer ptr)
        {
            return ptr.tag() != nullTag;
        }

        Job nextJob(bool& otherJobsRemaining, EJobLayer& outLayer);

        void          forEachTrueJobIndexBlock(void (*func)(void*, TaggedPointer), void* p);
        void          forEachTrueThreadBlock(void (*func)(void*, TaggedPointer), void* p, bool joinAll);
        void          prepareThreadNode(TaggedPointer current, TaggedPointer next, uint8_t activeCount);
        TaggedPointer getSmallestLayer(EJobLayer& outLayer) const;

        /**
         * pointer to `num32Blocks` 32B adjacent blocks from the pool allocator
         * when an index is inserted for the first time, it will follow insertion sort order,
         * then you will reuse the index already assigned to it, which can be found by binary search
         * Each pointed object is a linked list of arrays of jobs.
         */
        TaggedPointer m_pIndex;

        /**
         * Pointer to threads running an entry point which wait on the condition variable, wake up, try to steal
         * the lowest layer job from the index, copy its data, mark the place on the index empty, and execute the job
         */
        TaggedPointer m_pThreads;

        mutable std::condition_variable_any m_cv;
        mutable SpinLock                    m_mtx;
        EJobLayer                           m_activeLayer{EJobLayer::eEmpty};
        uint32_t                            m_numJobs           = 0;
        std::atomic_flag                    m_jobsInFlight      = ATOMIC_FLAG_INIT;
        mutable bool                        m_ready             = false;
        mutable bool                        m_shutdownRequested = false;
    };
    static_assert(std::atomic<EJobLayer>::is_always_lock_free);

    class ThreadPool
    {
    public:
        ThreadPool(int const size);

        ~ThreadPool();

        //ThreadPool cannot be copied or assigned
        ThreadPool(ThreadPool const&)            = delete;
        ThreadPool& operator=(ThreadPool const&) = delete;

        //ThreadPool object cannot be moved and move assignment
        ThreadPool(ThreadPool&&)            = delete;
        ThreadPool& operator=(ThreadPool&&) = delete;

        void Shutdown();

        //Allow to the user to add different functions with a arbitrary type and arbitrary arguments
        //f function, args function arguments, detect the return a futere of the detection function type returned
        //trailing return type
        template <typename F, typename... Args>
        auto AddTask(F&& f, Args&&... args) -> std::future<decltype(f(args...))>
        {
            //make a shared_ptr(more effient allocation)
            auto task_ptr = std::make_shared<std::packaged_task<decltype(f(args...))()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...));

            auto wrapper_func = [task_ptr]() { (*task_ptr)(); };

            //scope of lock m_mutex
            //push a new task in queue
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_queue.push(wrapper_func);
                // Wake up one thread if its waiting
                m_conditionVariable.notify_one();
            }

            // Return turn a future
            return task_ptr->get_future();
        }

        int QueueSize();

    private:
        //callable object
        class ThreadWorker
        {
        public:
            //set the pointer of the pool so the worker can access to the shared queue and mutex
            ThreadWorker(ThreadPool* pool);

            //execute immediately starts to execute
            void operator()();

        private:
            ThreadPool* m_threadPool;
        };
        //jobs queue of void funcitions with nothing parameters

    public:
        int busyThreads;

    private:
        mutable std::mutex m_mutex;
        //allows to put threads into a spleeping mode and wake-up
        std::condition_variable m_conditionVariable;

        std::vector<std::thread> m_threads;
        //destroy the threads
        bool m_shutdownRequested;

        std::queue<std::function<void()>> m_queue;
    };
} // namespace dmt