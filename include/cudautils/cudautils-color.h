#pragma once

#include "cudautils/cudautils-macro.h"

// windows bull*
#undef RGB

namespace dmt {
    struct DMT_CORE_API RGB
    {
        float r, g, b;
    };
} // namespace dmt

#if defined(DMT_CUDAUTILS_IMPL) || defined(DMT_CUDAUTILS_COLOR_IMPL)
#include "cudautils-color.cu"
#endif