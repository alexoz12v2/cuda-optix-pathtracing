#pragma once

#include "dmtmacros.h"

// windows bull*
#undef RGB

namespace dmt {
    struct RGB
    {
        float r, g, b;
    };
} // namespace dmt

#if defined(DMT_CUDAUTILS_IMPL) || defined(DMT_CUDAUTILS_COLOR_IMPL)
#include "cudautils-color.cu"
#endif