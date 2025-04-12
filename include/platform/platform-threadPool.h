#pragma once

#include "dmtmacros.h"
#include <platform/platform-macros.h>

#include <platform/platform-memory.h>

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
#include <memory_resource>

#include <cstdint>

namespace dmt::os {
    class DMT_PLATFORM_API Thread
    {
    public:
        using ThreadFunc = void (*)(void*); // Raw function pointer for triviality
        struct Internal
        {
            ThreadFunc func;
            void*      data;
        };

        Thread(ThreadFunc func, std::pmr::memory_resource* resource = std::pmr::get_default_resource());
        Thread(Thread const&)                = delete;
        Thread(Thread&&) noexcept            = delete;
        Thread& operator=(Thread const&)     = delete;
        Thread& operator=(Thread&&) noexcept = delete;

        void     start(void* arg = nullptr);
        void     join();
        void     terminate();
        bool     running() const;
        uint32_t id() const;

    private:
#ifdef _WIN32
        void* m_handle = nullptr; // HANDLE is a void* in Windows
#else
        unsigned long m_thread = 0; // pthread_t is an unsigned long (or struct on some platforms)
#endif
        Internal*                  m_internal;
        std::pmr::memory_resource* m_resource;
    };
    static_assert(std::is_trivially_destructible_v<Thread>);
} // namespace dmt::os

namespace dmt {
    using JobSignature = void (*)(uintptr_t data, uint32_t tid);

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

    constexpr std::strong_ordering operator<=>(EJobLayer a, EJobLayer b) { return toUnderlying(a) <=> toUnderlying(b); }

    struct Job
    {
        JobSignature func;
        uintptr_t    data;
    };
    static_assert(sizeof(Job) == 16 && alignof(Job) == 8);

    class ThreadPoolV2
    {
#undef max
        static constexpr uint16_t nullTag   = 0xFFFU;
        static constexpr uint32_t blockSz   = toUnderlying(EBlockSize::e256B);
        static constexpr uint32_t numBlocks = 10;
        friend void               jobWorkerThread(void* that);

    public:
        DMT_PLATFORM_API ThreadPoolV2(uint32_t                   numThreads = std::thread::hardware_concurrency(),
                                      std::pmr::memory_resource* resource   = std::pmr::get_default_resource());
        ThreadPoolV2(ThreadPoolV2 const&)                = delete;
        ThreadPoolV2(ThreadPoolV2&&) noexcept            = delete;
        ThreadPoolV2& operator=(ThreadPoolV2 const&)     = delete;
        ThreadPoolV2& operator=(ThreadPoolV2&&) noexcept = delete;
        DMT_PLATFORM_API ~ThreadPoolV2() noexcept;

        DMT_PLATFORM_API bool addJob(Job const& job, EJobLayer layer);
        DMT_PLATFORM_API void cleanup();
        DMT_PLATFORM_API void kickJobs();
        DMT_PLATFORM_API void pauseJobs();
        DMT_PLATFORM_API bool otherLayerActive(EJobLayer& layer) const;
        DMT_PLATFORM_API bool isValid() const;
        DMT_PLATFORM_API std::pmr::string debugPrintLayerJobs(
            EJobLayer                  layer,
            std::pmr::memory_resource* resource = std::pmr::get_default_resource()) const;


        DMT_PLATFORM_API uint32_t numJobs(EJobLayer layer) const;

    private:
        static constexpr uint32_t layerCardinality = 20;

        struct JobBlob
        {
            static constexpr uint32_t     maxBlobCount = 15;
            std::array<Job, maxBlobCount> jobs;
            uint64_t                      counter;
            JobBlob*                      next;
        };
        static_assert(std::is_standard_layout_v<JobBlob> && std::is_trivial_v<JobBlob>);

        struct IndexArray
        {
            std::array<JobBlob*, layerCardinality>  ptrs;
            std::array<EJobLayer, layerCardinality> layers;
        };
        static_assert(std::is_standard_layout_v<IndexArray>); // allows to use memcpy/memset

        struct ThreadStorage
        {
            ThreadPoolV2* threadPool;
            uint32_t      index;
        };

    private:
        Job              nextJob(bool& otherJobsRemaining, EJobLayer& outLayer);
        std::pmr::string debugPrintLayerJobsUnlocked(
            EJobLayer                  layer,
            std::pmr::memory_resource* resource = std::pmr::get_default_resource()) const;

        std::pmr::memory_resource* m_resource;

        /**
         * pointer to `num32Blocks` 32B adjacent blocks from the pool allocator
         * when an index is inserted for the first time, it will follow insertion sort order,
         * then you will reuse the index already assigned to it, which can be found by binary search
         * Each pointed object is a linked list of arrays of jobs.
         */
        UniqueRef<IndexArray> m_index;

        /**
         * Pointer to threads running an entry point which wait on the condition variable, wake up, try to steal
         * the lowest layer job from the index, copy its data, mark the place on the index empty, and execute the job
         */
        UniqueRef<os::Thread[]> m_threads;

        UniqueRef<ThreadStorage[]> m_threadIndices;

        mutable std::condition_variable_any m_cv;

        mutable std::atomic_flag m_jobsInFlight = ATOMIC_FLAG_INIT;
        EJobLayer                m_activeLayer{EJobLayer::eEmpty};
        uint32_t                 m_numJobs = 0;
        uint32_t                 m_numThreads;

        mutable SpinLock m_mtx;
        mutable bool     m_ready             = false;
        mutable bool     m_shutdownRequested = false;
    };

} // namespace dmt