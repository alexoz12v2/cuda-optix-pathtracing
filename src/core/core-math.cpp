#include "core-math.h"

namespace dmt::arch {
    float hmin_ps(__m128 v)
    {
        __m128 shuf = _mm_movehdup_ps(v); // (v1,v1,v3,v3)
        __m128 mins = _mm_min_ps(v, shuf);
        shuf        = _mm_movehl_ps(shuf, mins); // (v2,v3)
        mins        = _mm_min_ss(mins, shuf);
        return _mm_cvtss_f32(mins);
    }

    float hmax_ps(__m128 v)
    {
        __m128 shuf = _mm_movehdup_ps(v);
        __m128 maxs = _mm_max_ps(v, shuf);
        shuf        = _mm_movehl_ps(shuf, maxs);
        maxs        = _mm_max_ss(maxs, shuf);
        return _mm_cvtss_f32(maxs);
    }

    float hmin_ps(__m256 v)
    {
        __m128 low  = _mm256_castps256_ps128(v);   // lower 128
        __m128 high = _mm256_extractf128_ps(v, 1); // upper 128
        __m128 min1 = _mm_min_ps(low, high);
        __m128 shuf = _mm_movehdup_ps(min1);
        __m128 min2 = _mm_min_ps(min1, shuf);
        shuf        = _mm_movehl_ps(shuf, min2);
        min2        = _mm_min_ss(min2, shuf);
        return _mm_cvtss_f32(min2);
    }

    float hmax_ps(__m256 v)
    {
        __m128 low  = _mm256_castps256_ps128(v);
        __m128 high = _mm256_extractf128_ps(v, 1);
        __m128 max1 = _mm_max_ps(low, high);
        __m128 shuf = _mm_movehdup_ps(max1);
        __m128 max2 = _mm_max_ps(max1, shuf);
        shuf        = _mm_movehl_ps(shuf, max2);
        max2        = _mm_max_ss(max2, shuf);
        return _mm_cvtss_f32(max2);
    }
} // namespace dmt::arch

namespace dmt {
    static uint16_t encodeOcta(float f)
    {
        static constexpr float max = std::numeric_limits<uint16_t>::max();
        return static_cast<uint16_t>(fl::round(fl::clamp01((f + 1) / 2) * max));
    }

    OctahedralNorm octaFromNorm(Normal3f n)
    {
        // 1: planar projection
        Vector3f projected = n.asVec() / (fl::abs(n.x) + fl::abs(n.y) + fl::abs(n.z));

        // 2: outfold the downward faces to the external triangles of the quad
        if (projected.z < 0.f)
        {
            float const ox = projected.x;
            float const oy = projected.y;
            projected.x    = (1.f - fl::abs(oy)) * fl::sign(ox);
            projected.y    = (1.f - fl::abs(ox)) * fl::sign(oy);
        }
        // 3: mapping into [0,1]
        return {.x = encodeOcta(projected.x), .y = encodeOcta(projected.y)};
    }


    Normal3f normFromOcta(OctahedralNorm o)
    {
        static constexpr float max = std::numeric_limits<uint16_t>::max();

        // 1: convert back to [-1, 1] range
        Vector2f const f{static_cast<float>(o.x) / max * 2.f - 1.f, static_cast<float>(o.y) / max * 2.f - 1.f};

        // 2: reconstruct z component
        Vector3f n(f.x, f.y, 1.f - fl::abs(f.x) - fl::abs(f.y));

        // 3: fold back the lower hemisphere
        if (n.z < 0.f)
        {
            float const ox = n.x;
            float const oy = n.y;

            n.x = (1.f - fl::abs(oy)) * fl::sign(ox);
            n.y = (1.f - fl::abs(ox)) * fl::sign(oy);
        }

        // 4: normalize
        return Normal3f{n};
    }

    void extractAffineTransform(Matrix4f const& m, float affineTransform[12])
    {
        affineTransform[0] = m.m[0]; // m(0,0)
        affineTransform[1] = m.m[1]; // m(0,1)
        affineTransform[2] = m.m[2]; // m(0,2)

        affineTransform[3] = m.m[4]; // m(1,0)
        affineTransform[4] = m.m[5]; // m(1,1)
        affineTransform[5] = m.m[6]; // m(1,2)

        affineTransform[6] = m.m[8];  // m(2,0)
        affineTransform[7] = m.m[9];  // m(2,1)
        affineTransform[8] = m.m[10]; // m(2,2)

        affineTransform[9]  = m.m[12]; // m(3,0) -> translation x
        affineTransform[10] = m.m[13]; // m(3,1) -> translation y
        affineTransform[11] = m.m[14]; // m(3,2) -> translation z
    }

    Matrix4f matrixFromAffine(float const affineTransform[12])
    {
        // clang-format off
        Matrix4f const m {
            affineTransform[0], affineTransform[1],  affineTransform[2],  0,
            affineTransform[3], affineTransform[4],  affineTransform[5],  0,
            affineTransform[6], affineTransform[7],  affineTransform[8],  0,
            affineTransform[9], affineTransform[10], affineTransform[11], 1
        };
        // clang-format on
        return m;
    }

    Transform transformFromAffine(float const affineTransform[12])
    {
        return Transform{matrixFromAffine(affineTransform)};
    }

} // namespace dmt

namespace dmt::transforms {
    Transform DMT_FASTCALL persp(float fovRadians, float aspectRatio, float near, float far)
    {
        float const focalLen = fl::rcp(tan(fovRadians * 0.5f));

        float const x = focalLen / aspectRatio;
        float const y = -focalLen;

        // OpenGL style [z -> -1, 1]
        float const a = near / (far - near);
        float const b = far * a;

        // Direct3D style [z -> 0, 1] (PBRT)
        // float const a = far / (far - near);
        // float const b = -near * a;

        // clang-format off
        Matrix4f const m{{
            x, 0, 0,  0, // first column
            0, y, 0,  0, // second column
            0, 0, a, -1, // third column
            0, 0, b,  0  // fourth column
        }};
        // clang-format on
        return dmt::Transform{m};
    }

    Transform DMT_FASTCALL scale(Vector3f s)
    {
        // clang-format off
        Matrix4f const m{{
            s[0], 0,    0,    0, // first column
            0,    s[1], 0,    0, // second column
            0,    0,    s[2], 0, // third column
            0,    0,    0,    1  // fourth column
        }};
        // clang-format on
        return dmt::Transform{m};
    }

    Transform DMT_FASTCALL translate(Vector3f v)
    {
        // clang-format off
        Matrix4f const m{{
            1,    0,    0,    0, // first column
            0,    1,    0,    0, // second column
            0,    0,    1,    0, // third column
            v[0], v[1], v[2], 1  // fourth column
        }};
        // clang-format on
        return dmt::Transform{m};
    }

    Transform DMT_FASTCALL cameraWorldFromCamera(Normal3f cameraDirection)
    {
        Normal3f tmpUp{{0, 0, 1}};
        if (absDot(cameraDirection, tmpUp) > 0.99f)
            tmpUp = {{0, 1, 0}}; // handle gimbal lock

        // orthonormal basis
        Normal3f const right = normalFrom(cross(tmpUp, cameraDirection)); // world X
        Normal3f const up    = normalFrom(cross(cameraDirection, right)); // world Z

        // clang-format off
        Matrix4f const m{{
            right.x,  up.x,    cameraDirection.x,  0,
            right.y,  up.y,    cameraDirection.y,  0,
            right.z,  up.z,    cameraDirection.z,  0,
            0,        0,       0,                  1
        }};
        // clang-format on

        return dmt::Transform{m};
    }

    // | Rx Ux Fx Tx |
    // | Ry Uy Fy Ty |
    // | Rz Uz Fz Tz |
    // | 0  0  0  1  |
    Transform DMT_FASTCALL worldFromCamera(Normal3f cameraDirection, Point3f cameraPosition)
    {
        Vector3f const forward = cameraDirection; // +Z in camera space
        Vector3f const worldUp{0.0f, 0.0f, 1.0f}; // World up (0, 0, +Z)

        // Compute right (X) and up (Y) vectors for the camera frame (left handed system)
        Vector3f right = cross(forward, worldUp); // +X in camera space
        Vector3f up    = cross(right, forward);   // +Y in camera space
        right          = normalize(right);
        up             = normalize(up);

        // Column-major matrix for worldFromCamera
        // clang-format off
        Matrix4f const m{{
            right.x,   right.y,   right.z,   0.0f, // Column 0: right
            up.x,      up.y,      up.z,      0.0f, // Column 1: up
            forward.x, forward.y, forward.z, 0.0f, // Column 2: forward
            cameraPosition.x, cameraPosition.y, cameraPosition.z, 1.0f // Column 3: position
        }};
        // clang-format on
        assert(determinant(m) < 0 &&
               "left handed camera space -> right handed world space"
               " should swap handedness");

        return Transform{m};
    }

    Transform DMT_FASTCALL
        cameraFromRaster_Perspective(float fovRadians, float aspectRatio, uint32_t xRes, uint32_t yRes, float focalLength)
    {
        // Compute half width/height of image plane at focal distance
        float const halfHeight = focalLength * tan(0.5f * fovRadians);
        float const halfWidth  = halfHeight * aspectRatio;

        float const pixelSizeX = 2.0f * halfWidth / static_cast<float>(xRes);
        float const pixelSizeY = 2.0f * halfHeight / static_cast<float>(yRes);

        // Translation to move raster origin (top-left) to image center in camera space
        float const tx = -halfWidth + 0.5f * pixelSizeX;
        float const ty = halfHeight - 0.5f * pixelSizeY; // Flip Y

        // clang-format off
        Matrix4f const m {{
            pixelSizeX, 0.0f,       0.0f,        0.0f,
            0.0f,      -pixelSizeY, 0.0f,        0.0f, // flip Y axis
            0.0f,       0.0f,       1.0f,        0.0f,
            tx,         ty,         focalLength, 1.0f
        }};
        // clang-format on

        return dmt::Transform{m};
    }
} // namespace dmt::transforms
