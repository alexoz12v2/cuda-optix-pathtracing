#pragma once

#include "core/core-macros.h"
#include "cudautils/cudautils-vecmath.h"
#include "cudautils/cudautils-transform.h"

#if !defined(DMT_ARCH_X86_64)
    #error "Support only for AVX2 capable x86_64 CPU"
#endif


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

namespace dmt::transforms {
    /**
     * `F/a  0  0       0`
     * `0   -F  0       0`
     * `0    0  n/(f-n) fn/(f-n)`
     * `0    0 -1       0`
     * where
     * - F = 1/(tan(fov/2))
     * - f = far clip plane
     * - n = near clip plane
     * - a = aspect ratio
     * This is used to initialize the screenFromCamera Matrix
     */
    DMT_CORE_API Transform DMT_FASTCALL persp(float fovRadians, float aspectRatio, float near, float far);

    DMT_CORE_API Transform DMT_FASTCALL scale(Vector3f s);

    DMT_CORE_API Transform DMT_FASTCALL translate(Vector3f v);

    /**
     * Build matrix column-wise (camera space → camera-world space)
     * Camera space (left handed):
     * - X: right | Y: up      | Z: forward
     * Camera-World space (right handed):
     * - X: right | Y: forward | Z: up
     * So camera X → world right, Y → world up, Z → world forward
     */
    DMT_CORE_API Transform DMT_FASTCALL cameraWorldFromCamera(Normal3f cameraDirection);

    /**
     * Build matrix column-wise (camera space → world space)
     * Camera space (left handed):
     * - X: right | Y: up      | Z: forward
     * - origin: camera position
     * World space (right handed):
     * - X: right | Y: forward | Z: up
     * - origin: 0,0,0
     * So camera X → world right, Y → world up, Z → world forward
     */
    DMT_CORE_API Transform DMT_FASTCALL worldFromCamera(Normal3f cameraDirection, Point3f cameraPosition);

    DMT_CORE_API Transform DMT_FASTCALL
        cameraFromRaster_Perspective(float fovRadians, float aspectRatio, uint32_t xRes, uint32_t yRes, float focalLength);
} // namespace dmt::transforms