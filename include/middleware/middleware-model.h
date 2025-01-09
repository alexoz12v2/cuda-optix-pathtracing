#pragma once

#include "dmtmacros.h"

#if defined(DMT_INTERFACE_AS_HEADER)
#include <platform/platform.h>
#else
import platform;
#endif

DMT_MODULE_EXPORT dmt::model {
    using namespace dmt;
    void test(AppContext & ctx);
}

DMT_MODULE_EXPORT dmt::model::soa { using namespace dmt; }