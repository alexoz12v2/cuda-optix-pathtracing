#include "platform-threadPool.h"

#include <pthread.h>

namespace dmt::os {
    static void* threadFuncWrapper(void* arg)
    {
        auto* internal = reinterpret_cast<Thread::Internal*>(arg);
        if (!internal)
            return nullptr;

        internal->func(internal->data);
        return nullptr;
    }

    void Thread::start(void* arg)
    {
        m_internal->data = arg;
        if (pthread_create(reinterpret_cast<pthread_t*>(m_thread), nullptr, threadFuncWrapper, reinterpret_cast<void*>(m_internal)))
        {
            // error
        }
    }

    void Thread::join()
    {
        if (m_thread)
        {
            pthread_join(static_cast<pthread_t>(m_thread), nullptr);
            m_thread = 0;
        }
    }

    void Thread::terminate()
    {
        if (m_thread)
        {
            pthread_cancel(static_cast<pthread_t>(m_thread));
            m_thread = 0;
        }
    }
} // namespace dmt::os