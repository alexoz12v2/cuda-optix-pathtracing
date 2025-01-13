#pragma once

#include "dmtmacros.h"

#include <cstdint>

#if defined(DMT_TESTSHARED_SHARED)

#if defined(DMT_TESTSHARED_EXPORTS)
#define DMT_TESTSHARED_API DMT_API_EXPORT
#else
#define DMT_TESTSHARED_API DMT_API_IMPORT
#endif

#else

#define DMT_TESTSHARED_API

#endif

namespace dmt {

    DMT_TESTSHARED_API uint32_t add(uint32_t a, uint32_t b);

}