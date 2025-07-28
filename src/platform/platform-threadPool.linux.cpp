#include "platform-threadPool.h"

#include <pthread.h>

Thread::Thread(ThreadFunc func) : function(func), thread(0) {}

void* threadFuncWrapper(void* arg)
{
    auto func = reinterpret_cast<Thread::ThreadFunc>(arg);
    if (func)
        func();
    return nullptr;
}

void Thread::start()
{
    if (pthread_create(reinterpret_cast<pthread_t*>(&thread), nullptr, threadFuncWrapper, reinterpret_cast<void*>(function)) !=
        0)
    {
        // error
    }
}

void Thread::join()
{
    if (thread)
    {
        pthread_join(static_cast<pthread_t>(thread), nullptr);
        thread = 0;
    }
}

void Thread::terminate()
{
    if (thread)
    {
        pthread_cancel(static_cast<pthread_t>(thread));
        thread = 0;
    }
}