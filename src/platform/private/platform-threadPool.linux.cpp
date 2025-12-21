#include "platform-threadPool.h"

#include <pthread.h>
#include <unistd.h>

namespace /*static*/ {
using namespace dmt::os;
void* threadFuncWrapper(void* arg) {
  auto* internal = reinterpret_cast<Thread::Internal*>(arg);
  if (!internal) return nullptr;

  internal->func(internal->data);
  return nullptr;
}

}  // namespace

namespace dmt::os {
void Thread::start(void* arg) {
  m_internal->data = arg;
  pthread_attr_t attr{};
  pthread_attr_init(&attr);
  if (pthread_create(reinterpret_cast<pthread_t*>(&m_thread), &attr,
                     threadFuncWrapper, reinterpret_cast<void*>(m_internal))) {
    // error
  }
  pthread_attr_destroy(&attr);
}

void Thread::join() {
  if (m_thread) {
    pthread_join(static_cast<pthread_t>(m_thread), nullptr);
    m_thread = 0;
  }
}

void Thread::terminate() {
  if (m_thread) {
    pthread_cancel(static_cast<pthread_t>(m_thread));
    m_thread = 0;
  }
}

bool Thread::running() const { return m_thread != 0; }

uint32_t Thread::id() const {
  if (m_thread) {
    return static_cast<uint32_t>(gettid());
  }
  return 0;
}
}  // namespace dmt::os