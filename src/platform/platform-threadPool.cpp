module;

#include <functional>
#include <future>
#include <iostream>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

module platform;

namespace dmt
{
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
        m_threadPool->m_conditionVariable
            .wait(lock,
                  [this] { return this->m_threadPool->m_shutdownRequested || !this->m_threadPool->m_queue.empty(); });
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
} // namespace dmt