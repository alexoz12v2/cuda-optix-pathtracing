#pragma once

#include "core/core-macros.h"
#include "cudautils/cudautils-vecmath.h"

#if !defined(DMT_ARCH_X86_64)
#error "Support only for AVX2 capable x86_64 CPU"
#endif

#include <immintrin.h>

namespace dmt::arch {
   DMT_CORE_API float hmin_ps(__m128 v);
   DMT_CORE_API float hmax_ps(__m128 v);
   DMT_CORE_API float hmin_ps(__m256 v);
   DMT_CORE_API float hmax_ps(__m256 v);
} // namespace dmt::arch

namespace dmt {
    struct DMT_CORE_API TriangleData
    {
        Point3f v0, v1, v2;
    };
} // namespace dmt