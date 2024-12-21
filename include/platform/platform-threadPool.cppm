/**
 * @file platform-threadPool.cppm
 * @brief ...
 * 
 * @defgroup platform platform Module
 * @{
 */
module;
#if defined(DMT_INTERFACE_AS_HEADER)
#pragma once
#endif

#include <functional>
#include <future>
#include <iostream>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#if !defined(DMT_INTERFACE_AS_HEADER)
export module platform:threadPool;
#endif

// TODO move this in an header grouping commonly used macros
#if !defined(DMT_INTERFACE_AS_HEADER)
#define DMT_MODULE_EXPORT export
#else
#define DMT_MODULE_EXPORT
#endif

namespace dmt
{
DMT_MODULE_EXPORT class ThreadPool
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
/** @} */