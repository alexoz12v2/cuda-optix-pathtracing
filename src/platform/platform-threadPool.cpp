#include "platform-threadPool.h"

#include "platform-context.h"

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

namespace dmt {
    static void jobWorkerThread(void* that)
    {
        auto*          storage    = reinterpret_cast<ThreadPoolV2::ThreadStorage*>(that);
        auto*          threadPool = storage->threadPool;
        uint32_t const index      = storage->index;
        Job            copy;
        EJobLayer      currentLayer   = EJobLayer::eEmpty;
        bool           otherRemaining = false;

        // forever
        while (true)
        {
            // Context ctx;
            // wait for a job to become available
            {
                std::unique_lock<decltype(threadPool->m_mtx)> lk{threadPool->m_mtx};
                if (threadPool->m_shutdownRequested)
                {
                    break;
                }

                EJobLayer activeLayer = currentLayer;
                threadPool->m_cv.wait(lk, [&threadPool, &activeLayer]() {
                    return (threadPool->m_ready && !threadPool->otherLayerActive(activeLayer)) ||
                           threadPool->m_shutdownRequested;
                });
                if (threadPool->m_shutdownRequested)
                {
                    break;
                }

                // copy the next job
                copy = threadPool->nextJob(otherRemaining, currentLayer);

                threadPool->m_activeLayer = currentLayer;
            }

            // uncomment to debug
            // static std::mutex mtx;
            if (copy.func)
            {
                //if (ctx.isValid() && ctx.isTraceEnabled())
                //    ctx.trace("number of jobs remaining {}, func: {}",
                //              std::make_tuple(threadPool->numJobs(currentLayer), copy.func));
                //std::lock_guard lk{mtx};
                threadPool->m_jobsInFlight.test_and_set();
                //if (ctx.isValid() && ctx.isTraceEnabled())
                //    ctx.trace("index: {}", std::make_tuple(index));
                copy.func(copy.data, index);
                copy.func = nullptr;
                --threadPool->m_numJobs;
                if (otherRemaining)
                {
                    threadPool->m_cv.notify_all(); // 👈 Wake up another worker if more work is pending
                }
                else
                {
                    threadPool->m_jobsInFlight.clear();
                }
            }
        }
    }

    ThreadPoolV2::ThreadPoolV2(uint32_t numThreads, std::pmr::memory_resource* resource) :
    m_resource(resource),
    m_index(makeUniqueRef<IndexArray>(resource)),
    m_threads(makeUniqueRef<os::Thread[]>(resource, numThreads)),
    m_threadIndices(makeUniqueRef<ThreadStorage[]>(resource, numThreads)),
    m_numThreads(numThreads)
    {
        using namespace std::string_view_literals;
        // TODO: why are we locking at construction? Do threads need to wait construction
        std::lock_guard<decltype(m_mtx)> lk{m_mtx};
        Context                          ctx;
        if (!ctx.isValid())
        {
            m_resource = nullptr;
            return;
        }

        // allocate the 256 Bytes block for the index
        if (!m_index)
        {
            ctx.error("Couldn't allocate {} index blocks for threadPool", std::make_tuple(numBlocks));
            m_resource = nullptr;
            return;
        }

        // initialize first Job blob of each index
        std::ranges::fill(m_index->layers, EJobLayer::eEmpty);
        for (uint32_t i = 0; i != layerCardinality; ++i)
        {
            // TODO maybe: alternative: allocate them all at once?
            m_index->ptrs[i] = reinterpret_cast<JobBlob*>(m_resource->allocate(sizeof(JobBlob)));
            memset(m_index->ptrs[i], 0, sizeof(JobBlob));
            std::construct_at(m_index->ptrs[i]);
            m_index->ptrs[i]->next    = nullptr;
            m_index->ptrs[i]->counter = 0;
        }

        for (uint32_t i = 0; i < m_numThreads; ++i)
        {
            std::construct_at(&m_threads[i], jobWorkerThread, m_resource);
            m_threadIndices[i].index      = i;
            m_threadIndices[i].threadPool = this;
            m_threads[i].start(&m_threadIndices[i]);
        }
    }

    void ThreadPoolV2::kickJobs()
    {
        { // TODO is this scope necessary
            std::lock_guard<decltype(m_mtx)> lk{m_mtx};
            m_ready = true;
        }
        m_cv.notify_all();
    }

    void ThreadPoolV2::pauseJobs()
    {
        std::lock_guard<decltype(m_mtx)> lk{m_mtx};
        m_ready = false;
    }

    void ThreadPoolV2::cleanup()
    {
        {
            std::lock_guard<decltype(m_mtx)> lk{m_mtx};

            m_shutdownRequested = true;
            m_cv.notify_all(); // Wake up all worker threads so they can exit
        }

        // Join all threads
        for (uint32_t i = 0; i < m_numThreads; ++i)
        {
            m_threads[i].join();
        }

        // Free all JobBlobs per layer
        if (m_index)
        {
            for (uint32_t i = 0; i < layerCardinality; ++i)
            {
                JobBlob* blob = m_index->ptrs[i];
                while (blob)
                {
                    JobBlob* next = blob->next;
                    std::destroy_at(blob);
                    m_resource->deallocate(blob, sizeof(JobBlob), alignof(JobBlob));
                    blob = next;
                }

                m_index->ptrs[i]   = nullptr;
                m_index->layers[i] = EJobLayer::eEmpty;
            }
        }

        m_ready             = false;
        m_numJobs           = 0;
        m_activeLayer       = EJobLayer::eEmpty;
        m_shutdownRequested = false;
        m_jobsInFlight.clear();
    }

    ThreadPoolV2::~ThreadPoolV2() noexcept { cleanup(); }

    bool ThreadPoolV2::addJob(Job const& job, EJobLayer layer)
    {
        std::lock_guard<SpinLock> lk{m_mtx};
        if (!m_index)
            return false;

        auto layerIndex             = toUnderlying(layer);
        m_index->layers[layerIndex] = layer;

        JobBlob*& head = m_index->ptrs[layerIndex];
        if (!head)
        {
            head = reinterpret_cast<JobBlob*>(m_resource->allocate(sizeof(JobBlob)));
            if (!head)
                return false;

            memset(head, 0, sizeof(JobBlob));
            std::construct_at(head);
        }

        JobBlob* blob = head;
        while (blob)
        {
            for (uint32_t l = 0; l < JobBlob::maxBlobCount; ++l)
            {
                if (!blob->jobs[l].func) // unused slot
                {
                    blob->jobs[l] = job;
                    ++blob->counter;
                    ++m_numJobs;
                    return true;
                }
            }

            if (!blob->next)
            {
                blob->next = reinterpret_cast<JobBlob*>(m_resource->allocate(sizeof(JobBlob)));
                if (!blob->next)
                    return false;

                memset(blob->next, 0, sizeof(JobBlob));
                std::construct_at(blob->next);
            }

            blob = blob->next;
        }

        return false; // Should never happen
    }

    void ThreadPoolV2::waitForAll(uint32_t sleepMillis)
    {
        while (true)
        {
            std::lock_guard<SpinLock> lk{m_mtx};
            if (m_numJobs == 0)
                return;
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepMillis));
        }
    }

    Job ThreadPoolV2::nextJob(bool& otherJobsRemaining, EJobLayer& outLayer)
    {
        Job result{};
        otherJobsRemaining = false;

        for (uint32_t i = 0; i < layerCardinality; ++i)
        {
            JobBlob* blob = m_index->ptrs[i];
            JobBlob* prev = nullptr;

            while (blob)
            {
                for (size_t j = 0; j < blob->jobs.size(); ++j)
                {
                    if (blob->jobs[j].func != nullptr)
                    {
                        result   = blob->jobs[j];
                        outLayer = m_index->layers[i];

                        blob->jobs[j].func = nullptr;
                        --blob->counter;

                        // Optional: compact blob if needed

                        // Check if other jobs exist
                        for (uint32_t k = 0; k < layerCardinality; ++k)
                        {
                            JobBlob* check = m_index->ptrs[k];
                            while (check)
                            {
                                if (check->counter > 0 || check->next)
                                {
                                    otherJobsRemaining = true;
                                    return result;
                                }
                                check = check->next;
                            }
                        }

                        return result;
                    }
                }

                prev = blob;
                blob = blob->next;
            }
        }

        outLayer = EJobLayer::eEmpty;
        return result;
    }

    bool ThreadPoolV2::otherLayerActive(EJobLayer& layer) const
    {
        if (m_jobsInFlight.test())
        {
            for (uint32_t i = 0; i < layerCardinality; ++i)
            {
                if (m_index->layers[i] < layer && m_index->ptrs[i] != nullptr && m_index->ptrs[i]->counter > 0)
                {
                    layer = m_index->layers[i];
                    return true;
                }
            }
        }
        return false;
    }

    bool ThreadPoolV2::isValid() const { return m_resource != nullptr && m_index != nullptr && m_threads != nullptr; }

    std::pmr::string ThreadPoolV2::debugPrintLayerJobs(EJobLayer layer, std::pmr::memory_resource* resource) const
    {
        std::lock_guard<SpinLock> lk{m_mtx};

        return debugPrintLayerJobsUnlocked(layer, resource);
    }

    DMT_PLATFORM_API uint32_t ThreadPoolV2::numJobs(EJobLayer layer) const
    {
        if (!m_index)
            return 0;

        uint32_t counter = 0;
        for (uint32_t l = 0; l < layerCardinality; ++l)
        {
            if (m_index->layers[l] == layer)
            {
                JobBlob* blob = m_index->ptrs[l];
                while (blob)
                {
                    counter += blob->counter;
                    blob = blob->next;
                }
            }
        }

        return counter;
    }

    std::pmr::string ThreadPoolV2::debugPrintLayerJobsUnlocked(EJobLayer layer, std::pmr::memory_resource* resource) const
    {
        std::pmr::string result(resource);


        int layerIndex = -1;
        for (uint32_t i = 0; i < layerCardinality; ++i)
        {
            if (m_index->layers[i] == layer)
            {
                layerIndex = static_cast<int>(i);
                break;
            }
        }

        if (layerIndex < 0)
        {
            result += "Layer not found.\n";
            return result;
        }

        JobBlob* blob      = m_index->ptrs[layerIndex];
        uint32_t blobIndex = 0;

        while (blob)
        {
            result += std::format("Blob {}: counter = {}\n", blobIndex, blob->counter);

            for (size_t i = 0; i < blob->jobs.size(); ++i)
            {
                Job const& job = blob->jobs[i];
                if (job.func)
                    result += std::format("  [{}] func = {}\n", i, reinterpret_cast<void*>(job.func));
                else
                    result += std::format("  [{}] <null>\n", i);
            }

            blob = blob->next;
            ++blobIndex;
        }

        return result;
    }

} // namespace dmt