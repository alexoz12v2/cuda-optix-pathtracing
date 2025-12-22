#pragma once

#include <cuda_runtime.h>

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

struct DeviceHaltonOwen
{

};

// sys left hand: z:up+, y:foward+, x:right+
struct DeviceCamera {
  //float focalLength = 20.f;
  //float sensorSize = 36.f;
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

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
__device__ void raygen();
__global__ void raygenKernel(DeviceCamera* d_cam, DeviceHaltonOwen* d_haltonOwen, CameraSample* d_samples);

__device__ HitResult triangleIntersect(float4 x, float4 y, float4 z, Ray ray);
__global__ void triangleIntersectKernel(TriangleSoup soup, Ray ray);
void triangleIntersectTest();
