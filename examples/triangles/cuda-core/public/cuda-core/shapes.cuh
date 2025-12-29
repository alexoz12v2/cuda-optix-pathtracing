#ifndef DMT_CUDA_CORE_SHAPES_CUH
#define DMT_CUDA_CORE_SHAPES_CUH

#include "cuda-core/types.cuh"
#include "cuda-core/common_math.cuh"

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------
// TODO

HitResult hostIntersectMT(const float3& o, const float3& d, const float3& v0,
                          const float3& v1, const float3& v2);
__device__ HitResult triangleIntersect(float4 x, float4 y, float4 z, Ray ray);

#endif