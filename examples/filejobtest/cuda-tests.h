#pragma once

namespace dmt {
    struct AppContext;
    class BaseMemoryResource;
} // namespace dmt

void testBuddyDirectly(dmt::AppContext& actx, dmt::BaseMemoryResource* pMemRes);

void testMemPoolAsyncDirectly(dmt::AppContext& actx, dmt::BaseMemoryResource* pMemRes);
