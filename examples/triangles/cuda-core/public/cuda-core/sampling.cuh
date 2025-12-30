#ifndef DMT_CUDA_CORE_SAMPLING_CUH
#define DMT_CUDA_CORE_SAMPLING_CUH

#include "cuda-core/types.cuh"

__host__ __device__ float sphereLightPDF(float distSqr, float radiusSqr,
                                         float3 n, float3 rayD,
                                         bool hadTransmission);
// probably not needed as we are deleting texture coordinates from sample types
__host__ __device__ float2 mapToSphere(float3 co);
__host__ __device__ bool raySphereIntersect(float3 rayO, float3 rayD,
                                            float tMin, float tMax,
                                            float3 sphereC, float sphereRadius,
                                            float3* isect_p, float* isect_t);
__host__ __device__ float3 sampleUniformCone(float3 const N,
                                             float const one_minus_cos_angle,
                                             float2 const rand,
                                             float* cos_theta, float* pdf,
                                             int* delta);
__host__ __device__ float3 sampleUniformSphere(float2 const rand);
__host__ __device__ float2 sampleUniformDisk(float2 u);
__host__ __device__ float3 sampleCosHemisphere(float3 n, float2 u, float* pdf);
__host__ __device__ float3 sampleUniformHemisphere(float3 n, float2 u,
                                                   float* pdf);
__host__ __device__ float cosHemispherePDF(float3 n, float3 d);

#endif  // DMT_CUDA_CORE_SAMPLING_CUH