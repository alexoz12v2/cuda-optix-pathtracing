#ifndef DMT_CUDA_CORE_EXTRA_MATH_CUH
#define DMT_CUDA_CORE_EXTRA_MATH_CUH

#include "cuda-core/types.cuh"
#include "cuda-core/common_math.cuh"
#include "cuda-core/rng.cuh"

__host__ __device__ float lookupTableRead(float const* __restrict__ table,
                                          float x, int32_t size);
__host__ __device__ float lookupTableRead2D(float const* __restrict__ table,
                                            float x, float y, int32_t sizex,
                                            int32_t sizey);

__host__ __device__ Transform worldFromCamera(float3 cameraDirection,
                                              float3 cameraPosition);
__host__ __device__ Transform cameraFromRaster_Perspective(float focalLength,
                                                           float sensorHeight,
                                                           uint32_t xRes,
                                                           uint32_t yRes);

inline __host__ __device__ __forceinline__ float gamma(int32_t n) {
#ifdef __CUDA_ARCH__
  float const f = static_cast<float>(n) * FLT_EPSILON * 0.5f;
#else
  float const f =
      static_cast<float>(n) * std::numeric_limits<float>::epsilon() * 0.5f;
#endif
  return f / (1 - f);
}

inline __host__ __device__ __forceinline__ float3 errorFromTriangleIntersection(
    float u, float v, float3 p0, float3 p1, float3 p2) {
  return gamma(7) * (abs(u * p0) + abs(v * p1) + abs((1 - u - v) * p2));
}

__host__ __device__ __forceinline__ float3 offsetRayOrigin(float3 const p,
                                                           float3 const error,
                                                           float3 const ng,
                                                           float3 const w) {
  // Push along the geometric normal, but flip if ray goes below surface
  float d = dot(abs(ng), error);
  float3 offset = ng * d;

  if (dot(w, ng) < 0.f) offset = -offset;

  float3 po = p + offset;

  // Round away from surface to avoid re-intersection
  po.x = nextafterf(po.x, offset.x > 0 ? INFINITY : -INFINITY);
  po.y = nextafterf(po.y, offset.y > 0 ? INFINITY : -INFINITY);
  po.z = nextafterf(po.z, offset.z > 0 ? INFINITY : -INFINITY);

  return po;
}

// ---------------------------------------------------------------------------
// Next Event Estimation
// ---------------------------------------------------------------------------
__device__ CameraSample getCameraSample(int2 pPixel, DeviceHaltonOwen& rng,
                                        DeviceHaltonOwenParams const& params);
__device__ Ray getCameraRayNonRandom(int2 pixel,
                                     Transform const& cameraFromRaster,
                                     Transform const& renderFromCamera);
__device__ Ray getCameraRay(CameraSample const& cs,
                            Transform const& cameraFromRaster,
                            Transform const& renderFromCamera);

#endif