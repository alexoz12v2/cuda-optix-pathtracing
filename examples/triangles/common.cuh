#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <iostream>
#include <vector>
#include <vector>

// ---------------------------------------------------------------------------
// Common Types
// ---------------------------------------------------------------------------
struct Ray {
  float o[3];
  float d[3];
};

struct Transform {
  float m[16];
  float mInv[16];

  __host__ __device__ Transform();
  __host__ __device__ Transform(float const* m);
};

struct CameraSample {
  /// point on the film to which the generated ray should carry radiance
  /// (meaning pixelPosition + random offset)
  float2 pFilm;

  /// scale factor that is applied when the ray�s radiance is added to the image
  /// stored by the film; it accounts for the reconstruction filter used to
  /// filter image samples at each pixel
  float filterWeight;
  unsigned _padding;
};
static_assert(sizeof(CameraSample) == 16);

struct DeviceHaltonOwen {};

// sys left hand: z:up+, y:foward+, x:right+
struct DeviceCamera {
  // float focalLength = 20.f;
  // float sensorSize = 36.f;
  float3 dir{0.f, 1.f, 0.f};
  int spp = 4;
  float3 pos{0.f, 0.f, 0.f};
  int width = 128;
  int height = 128;
};

// ---------------------------------------------------------------------------
// Device Types
// ---------------------------------------------------------------------------
struct TriangleSoup {
  // x0_0 x0_1 x0_2 pad | x1_0 ...
  float* xs;
  float* ys;
  float* zs;
  // array of 4-byte booleans. starts at 0, if intersected 1
  // TODO: change this to u,v,t
  int32_t* intersected;
  // count of triangles
  size_t count;
};

struct HitResult {
  int32_t hit;  // 0 = no hit, 1 = hit
};

// ---------------------------------------------------------------------------
// Host Types
// ---------------------------------------------------------------------------
struct HostTriangleSoup {
  std::vector<float> xs, ys, zs;
  std::vector<int32_t> expected;
  size_t count = 0;
};

// ---------------------------------------------------------------------------
// CUDA Helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      std::cerr << "CUDA Error: " << cudaGetErrorName(err) << ": "         \
                << cudaGetErrorString(err) << "\n\tat " << __FILE__ << ':' \
                << __LINE__ << std::endl;                                  \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

struct CudaTimer {
  cudaEvent_t start{}, stop{};

  CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~CudaTimer() noexcept {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void begin(cudaStream_t const stream = 0) const {
    CUDA_CHECK(cudaEventRecord(start, stream));
  }
  float end(cudaStream_t const stream = 0) const {
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

// ---------------------------------------------------------------------------
// Common Math
// ---------------------------------------------------------------------------
__device__ __forceinline__ float3 cross(float3 a, float3 b) {
  return {
      +a.y * b.z - a.z * b.y,
      -a.x * b.z + a.z * b.x,
      +a.y * b.z - a.z * b.y,
  };
}
__device__ __forceinline__ float3 normalize(float3 a) {
  float const invMag = rsqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
  return make_float3(a.x * invMag, a.y * invMag, a.z * invMag);
}
__device__ __forceinline__ float lerp(float a, float b, float t) {
  // (1 - t) a + t b =  a - ta + tb = a + t ( b - a )
  float const _1mt = 1.f - t;
  return _1mt * a + t * b;
}
__device__ __forceinline__ float3 operator*(float3 v, float a) {
  return make_float3(v.x * a, v.y * a, v.z * a);
}
__device__ __forceinline__ float3 operator*(float a, float3 v) {
  return make_float3(v.x * a, v.y * a, v.z * a);
}
__device__ __forceinline__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// ---------------------------------------------------------------------------
// BSDF Bits and Pieces
// ---------------------------------------------------------------------------
__device__ __forceinline__ float3 sampleGGCX_VNDF(float3 wo, float2 u, float ax,
                                                  float ay) {
  // stretch view (anisotropic roughness)
  float3 const V = normalize(make_float3(ax * wo.x, ay * wo.y, wo.z));

  // orthonormal basis (Frame) (azimuthal only)
  float const lensq = V.x * V.x + V.y * V.y;
  float invLen = rsqrtf(lensq + 1e-7f);

  float3 const T1 = make_float3(-V.y * invLen, V.x * invLen, 0.f);
  float3 const T2 = cross(V, T1);

  // sample disk
  float const r = sqrtf(u.x);
  float const phi = 2.f * CUDART_PI_F * u.y;

  float const t1 = r * cosf(phi);

  // blend toward normal
  float const s = 0.5f * (1.f + V.z);
  float const t2 = lerp(sqrtf(fmaxf(0.f, 1.f - t1 * t1)), r * sinf(phi), s);

  // recombine and unstretch
  float3 Nh = t1 * T1 + t2 * T2 + sqrtf(max(0.f, 1.f - t1 * t1 - t2 * t2)) * V;
  Nh = normalize(make_float3(ax * Nh.x, ay * Nh.y, fmaxf(0.f, Nh.z)));
  return Nh;
}

__device__ __forceinline__ float smithG1(float3 v, float ax, float ay) {
  float const tan2 = (v.x * v.x) / (ax * ax) + (v.y * v.y) / (ay * ay);
  float const cos2 = v.z * v.z; // assumes _outward_ normal local space
  return 2.f / (1.f + sqrtf(1.f + tan2 / cos2));
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
__device__ void raygen();
__global__ void raygenKernel(DeviceCamera* d_cam,
                             DeviceHaltonOwen* d_haltonOwen,
                             CameraSample* d_samples);

__device__ HitResult triangleIntersect(float4 x, float4 y, float4 z, Ray ray);
__global__ void triangleIntersectKernel(TriangleSoup soup, Ray ray);
void triangleIntersectTest();
