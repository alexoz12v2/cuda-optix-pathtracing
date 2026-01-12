#ifndef DMT_CUDA_CORE_TYPES_H
#define DMT_CUDA_CORE_TYPES_H

#include <cuda_runtime.h>
#include <driver_types.h>
#include <math_constants.h>
#include <vector_types.h>
#include <cooperative_groups.h>

#include <cstdint>
#include <iostream>
#include <float.h>
#include <numbers>
#include <vector>

#define DMT_ENABLE_MSE 1

inline constexpr int WARP_SIZE = 32;

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t const err = (call);                                        \
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

struct Ray {
  float3 o;
  float3 d;
};

struct RayTile {
  __device__ __forceinline__ void addRay(Ray const& ray) {
    int const laneId = cooperative_groups::thread_block_tile<32>::thread_rank();
    ox[laneId] = ray.o.x;
    oy[laneId] = ray.o.y;
    oz[laneId] = ray.o.z;
    dx[laneId] = ray.d.x;
    dy[laneId] = ray.d.y;
    dz[laneId] = ray.d.z;
  }

  float ox[WARP_SIZE];
  float oy[WARP_SIZE];
  float oz[WARP_SIZE];
  float dx[WARP_SIZE];
  float dy[WARP_SIZE];
  float dz[WARP_SIZE];
};

inline constexpr int NUM_FILM_DIMENSION = 2;

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

struct DeviceHaltonOwenParams {
  int32_t baseScales[NUM_FILM_DIMENSION];
  int32_t baseExponents[NUM_FILM_DIMENSION];
  int32_t multInvs[NUM_FILM_DIMENSION];
};

// Camera Space (left-handed):       y+ up, z+ forward, x+ right
// World/Render Space (left-handed): z+ up, y+:forward, x+ right
struct DeviceCamera {
  float3 dir{0.f, 1.f, 0.f};
  float3 pos{0.f, 0.f, 0.f};
  int width = 16;
  int height = 16;
  int spp = 2;
  float focalLength = 20.f;
  float sensorSize = 36.f;
};

struct MortonLayout2D {
  uint32_t rows;  // height
  uint32_t cols;  // width
  uint32_t mortonRows;
  uint32_t mortonCols;
  uint64_t mortonCount;
};

struct TriangleSoup {
  // x0_0 x0_1 x0_2 pad | x1_0 ...
  float* xs;
  float* ys;
  float* zs;
  // tri0->matIndex | tri1->matIndex ...
  uint32_t* matId;

  // count of triangles
  size_t count;
};

struct HostTriangleSoup {
  std::vector<float> xs, ys, zs;
  std::vector<int32_t> expected;
  size_t count = 0;
};

struct HitResult {
  __host__ __device__ HitResult()
      : pos{},
        normal{},
        error{},
        hit{},
        matId{},
        t{
#ifdef __CUDA_ARCH__
            CUDART_INF_F
#else
            std::numeric_limits<float>::infinity()
#endif
        } {
  }

  float3 pos;
  float3 normal;
  float3 error;    // intersection error bounds
  int32_t hit;     // 0 = no hit, 1 = hit
  uint32_t matId;  // bsdf index
  float t;
};

#if DMT_ENABLE_MSE
struct HostPinnedOutputBuffer {
  __host__ void allocate(int width, int height) {
    size_t const bytes = width * height * sizeof(float4);
    CUDA_CHECK(cudaMallocHost(&meanPtr, bytes));
    CUDA_CHECK(cudaMallocHost(&m2Ptr, bytes));
  }
  __host__ void free() {
    cudaFree(meanPtr);
    cudaFree(m2Ptr);
  }
  float4* meanPtr;
  float4* m2Ptr;  // array of delta squared
};
struct DeviceOutputBuffer {
  __host__ void allocate(int width, int height) {
#  ifdef DMT_ENABLE_ASSERTS
    assert(!meanPtr && !m2Ptr);
#  endif
    size_t const bytes = width * height * sizeof(float4);
    CUDA_CHECK(cudaMalloc(&meanPtr, bytes));
    CUDA_CHECK(cudaMemset(meanPtr, 0, bytes));
    CUDA_CHECK(cudaMalloc(&m2Ptr, bytes));
    CUDA_CHECK(cudaMemset(m2Ptr, 0, bytes));
  }
  __host__ void free() {
    cudaFree(meanPtr);
    cudaFree(m2Ptr);
  }

  float4* meanPtr;
  float4* m2Ptr;  // array of delta squared
};
#endif

#endif  // DMT_CUDA_CORE_TYPES_H