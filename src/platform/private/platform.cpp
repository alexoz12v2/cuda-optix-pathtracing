#include "platform.h"

#include <cstdint>
#include <cstdlib>
#include <cassert>

namespace dmt {
static std::recursive_mutex s_allocMutex;

void Ctx::init(bool destroyIfExising, std::pmr::memory_resource* resource) {
  {
    std::scoped_lock lock{s_allocMutex};
    if (ctx::cs) {
      if (destroyIfExising)
        destroy();
      else
        return;
    }

    assert(!ctx::cs);
    m_resource = resource;
    ctx::cs = reinterpret_cast<ctx::Contexts*>(
        resource->allocate(sizeof(ctx::Contexts)));
    std::construct_at(ctx::cs);
  }

  int32_t idx = -1;
  ctx::cs->addContext(false, &idx);
  ctx::cs->setActive(idx);
  Context ctx;
  ctx.impl()->addHandler(
      [](dmt::LogHandler& _out) { dmt::createConsoleHandler(_out); });
}

void Ctx::destroy() {
  std::scoped_lock lock{s_allocMutex};
  std::destroy_at(ctx::cs);
  m_resource->deallocate(ctx::cs, sizeof(ctx::Contexts));
  m_resource = nullptr;
}
}  // namespace dmt
