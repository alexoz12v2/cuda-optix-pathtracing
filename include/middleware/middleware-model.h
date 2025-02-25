#pragma once

#include "dmtmacros.h"

#if !defined(DMT_PLATFORM_IMPORTED)
#include <platform/platform.h>
#endif

namespace dmt::model {
    using namespace dmt;
    void test(AppContext& ctx);
} // namespace dmt::model

namespace dmt::model::soa {
    using namespace dmt;
}