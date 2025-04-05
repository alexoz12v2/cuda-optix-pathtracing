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

                if (activeLayer != currentLayer)
                {
                    threadPool->m_activeLayer = currentLayer;
                }
            }

            if (copy.func)
            {
                threadPool->m_jobsInFlight.test_and_set();
                copy.func(copy.data, index);
                if (!otherRemaining)
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
            m_index->ptrs[i]          = reinterpret_cast<JobBlob*>(m_resource->allocate(sizeof(JobBlob)));
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
        m_cv.notify_one();
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

    void ThreadPoolV2::addJob(Job const& job, EJobLayer layer)
    {
        std::lock_guard<SpinLock> lk{m_mtx};
        if (!m_index)
            return;

        // Search for the layer in the current index
        int insertPos   = -1;
        int existingPos = -1;
        for (uint32_t i = 0; i < layerCardinality; ++i)
        {
            if (m_index->layers[i] == layer)
            {
                existingPos = static_cast<int>(i);
                break;
            }

            if (m_index->layers[i] == EJobLayer::eEmpty || m_index->layers[i] > layer)
            {
                insertPos = static_cast<int>(i);
                break;
            }
        }

        // Insert layer if not found
        int layerIndex = 0;
        if (existingPos >= 0)
        {
            layerIndex = existingPos;
        }
        else if (insertPos >= 0)
        {
            // Shift elements to make room
            for (int i = layerCardinality - 1; i > insertPos; --i)
            {
                m_index->layers[i] = m_index->layers[i - 1];
                m_index->ptrs[i]   = m_index->ptrs[i - 1];
            }

            m_index->layers[insertPos] = layer;

            // Allocate new JobBlob
            m_index->ptrs[insertPos] = reinterpret_cast<JobBlob*>(m_resource->allocate(sizeof(JobBlob)));
            std::construct_at(m_index->ptrs[insertPos]);
            m_index->ptrs[insertPos]->counter = 0;
            m_index->ptrs[insertPos]->next    = nullptr;

            layerIndex = insertPos;
        }
        else
        {
            // No space to insert layer
            return;
        }

        JobBlob* blob = m_index->ptrs[layerIndex];
        while (blob->counter >= blob->jobs.size())
        {
            if (!blob->next)
            {
                blob->next = reinterpret_cast<JobBlob*>(m_resource->allocate(sizeof(JobBlob)));
                std::construct_at(blob->next);
                blob->next->counter = 0;
                blob->next->next    = nullptr;
            }
            blob = blob->next;
        }

        // Insert the job
        blob->jobs[blob->counter++] = job;

        ++m_numJobs;
    }

    Job ThreadPoolV2::nextJob(bool& otherJobsRemaining, EJobLayer& outLayer)
    {
        Job result{};
        otherJobsRemaining = false;

        for (uint32_t i = 0; i < layerCardinality; ++i)
        {
            JobBlob* blob = m_index->ptrs[i];
            if (!blob || blob->counter == 0)
                continue;

            // Found a job to return
            result   = blob->jobs[0];
            outLayer = m_index->layers[i];

            // Shift remaining jobs in this blob forward
            for (uint64_t j = 1; j < blob->counter; ++j)
            {
                blob->jobs[j - 1] = blob->jobs[j];
            }

            blob->counter--;

            // Check for any remaining jobs in any layer
            for (uint32_t j = 0; j < layerCardinality; ++j)
            {
                if (m_index->ptrs[j] && m_index->ptrs[j]->counter > 0)
                {
                    otherJobsRemaining = true;
                    break;
                }
            }

            return result;
        }

        // Nothing to do
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

} // namespace dmt