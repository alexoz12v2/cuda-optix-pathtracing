#pragma once

#include "cudautils/cudautils-macro.h"
#include "cudautils/cudautils.h"
#include "cudautils/cudautils-vecmath.h"

#define WARP_SIZE       32
#define CHILDREN        8
#define WARP_STACK_SIZE 64

namespace dmt {

    struct HitOut
    {
        float t;
        int   triIdx;
    };
    struct StackEntry
    {
        int   nodeIdx;
        float tmin;
        bool  isInner;
    };

#if defined(__CUDA_ARCH__)
    // ---- Device triangle intersection (M�ller�Trumbore) ----
    DMT_GPU bool intersect_tri_mt(Ray const& r, float3 const& v0, float3 const& v1, float3 const& v2, float tMax, float& tOut);

    // ---- AABB intersect (per-lane) ----
    DMT_FORCEINLINE DMT_GPU bool aabb_intersect_local(
        Ray const& r,
        float      bxmin,
        float      bxmax,
        float      bymin,
        float      bymax,
        float      bzmin,
        float      bzmax,
        float      tcur,
        float&     tminOut)
    {
        // slabs using precomputed invd
        float tx1  = (bxmin - r.o.x) * r.d_inv.x;
        float tx2  = (bxmax - r.o.x) * r.d_inv.x;
        float tmin = fminf(tx1, tx2);
        float tmax = fmaxf(tx1, tx2);

        float ty1 = (bymin - r.o.y) * r.d_inv.y;
        float ty2 = (bymax - r.o.y) * r.d_inv.y;
        tmin      = fmaxf(tmin, fminf(ty1, ty2));
        tmax      = fminf(tmax, fmaxf(ty1, ty2));

        float tz1 = (bzmin - r.o.z) * r.d_inv.z;
        float tz2 = (bzmax - r.o.z) * r.d_inv.z;
        tmin      = fmaxf(tmin, fminf(tz1, tz2));
        tmax      = fminf(tmax, fmaxf(tz1, tz2));

        if (tmax >= tmin && tmin < tcur)
        {
            tminOut = tmin;
            return true;
        }
        return false;
    }
#endif
}; // namespace dmt

#if defined(DMT_CUDAUTILS_IMPLEMENTATION)
namespace dmt {
    #if defined(__CUDA_ARCH__)

    __forceinline__ __device__ float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

    __forceinline__ __device__ float3 cross3(float3 a, float3 b)
    {
        float3 x{};
        x.x = a.y * b.z - a.z * b.y;
        x.y = a.z * b.x - a.x * b.z;
        x.z = a.x * b.y - a.y * b.x;
        return x;
    }

    DMT_GPU bool intersect_tri_mt(Ray const& r, float3 const& v0, float3 const& v1, float3 const& v2, float tMax, float& tOut)
    {
        float const EPS = 1e-7f;
        float3      e1  = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
        float3      e2  = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
        float3      p   = cross3(to_float3(r.d), e2);
        float       det = dot3(e1, p);
        if (det > -EPS && det < EPS)
            return false; // parallel or nearly
        float invDet = 1.0f / det;

        float3 tvec{r.o.x - v0.x, r.o.y - v0.y, r.o.z - v0.z};
        float  u    = dot3(tvec, p) * invDet;
        if (u < 0.0f || u > 1.0f)
            return false;

        float3 q = cross3(tvec, e1);
        float  v = dot3(to_float3(r.d), q) * invDet;
        if (v < 0.0f || u + v > 1.0f)
            return false;

        float t = dot3(e2, q) * invDet;
        if (t <= EPS || t >= tMax)
            return false;

        tOut = t;
        return true;
    }
    #endif
} // namespace dmt
#endif