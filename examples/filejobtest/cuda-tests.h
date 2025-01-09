#pragma once

namespace dmt {
    struct AppContext;
    class BaseMemoryResource;
    class DynaArray;
} // namespace dmt

void testBuddyDirectly(dmt::AppContext& actx, dmt::BaseMemoryResource* pMemRes);

void testMemPoolAsyncDirectly(dmt::AppContext& actx, dmt::BaseMemoryResource* pMemRes);

void testDynaArrayDirectly(dmt::AppContext& actx, dmt::DynaArray& dynaArray);
