#include "platform-threadPool.h"

#include <Windows.h>

namespace dmt::os {
    static DWORD WINAPI threadFuncWrapper(LPVOID arg)
    {
        auto& internal = *reinterpret_cast<Thread::Internal*>(arg);
        internal.func(internal.data);
        return 0;
    }

    uint32_t Thread::id() const { return GetThreadId(m_handle); }

    void Thread::start(void* arg)
    {
        if (!m_internal)
            return;

        DWORD id         = 0;
        m_internal->data = arg;
        m_handle         = CreateThread(nullptr, 0, threadFuncWrapper, m_internal, 0, &id);
    }

    void Thread::join()
    {
        if (m_handle && m_internal)
        {
            WaitForSingleObject(m_handle, INFINITE);
            CloseHandle(m_handle);
            m_handle = nullptr;
        }

        m_resource->deallocate(m_internal, sizeof(Thread::Internal));
    }

    bool Thread::running() const { return m_handle != nullptr; }

    void Thread::terminate()
    {
        if (m_handle)
        {
            TerminateThread(m_handle, 1);
            CloseHandle(m_handle);
            m_handle = nullptr;
        }
    }
} // namespace dmt::os
