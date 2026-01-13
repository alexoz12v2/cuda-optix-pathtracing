#ifndef DMT_MEGAKERNEL_KERNELS_CUH
#define DMT_MEGAKERNEL_KERNELS_CUH

#include "cuda-core/types.cuh"
#include "cuda-core/shapes.cuh"
#include "cuda-core/rng.cuh"
#include "cuda-core/common_math.cuh"
#include "cuda-core/light.cuh"
#include "cuda-core/bsdf.cuh"

extern __constant__ float CMEM_cameraFromRaster[32];
extern __constant__ float CMEM_renderFromCamera[32];
extern __constant__ DeviceHaltonOwenParams CMEM_haltonOwenParams;
extern __constant__ int2 CMEM_imageResolution;
extern __constant__ int CMEM_spp;
extern __constant__ MortonLayout2D CMEM_mortonLayout;

__host__ void allocateDeviceConstantMemory(DeviceCamera const& h_camera);
inline __host__ void freeDeviceConstantMemory() {
  freeDeviceGGXEnergyPreservingTables();
}

// Large Kernel Parameters from Volta with R530. Otherwise, it's 4096 KB
// https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/

// Grid-Stride Loop + Occupancy API = profit
// https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
#if DMT_ENABLE_MSE
template <uint32_t BLOCK_SIZE = 64>
  requires((BLOCK_SIZE & 31) == 0)
struct SMEMLayout {
  static int constexpr LOG_NUM_BANKS = 5;  // popcount(32)
  static int constexpr PAD = BLOCK_SIZE >> LOG_NUM_BANKS;

  // Use a static helper to get the instance
  static __device__ __forceinline__ SMEMLayout& get() {
    extern __shared__ char s_mem[];
    return *reinterpret_cast<SMEMLayout*>(s_mem);
  }

  static __device__ __forceinline__ int idxOff() {
    return threadIdx.x + (threadIdx.x >> LOG_NUM_BANKS);
  }

  __device__ __forceinline__ void startSample(float4 const* meanPtr,
                                              float4 const* deltaPtr) {
    int const idx = idxOff();
    float4 const m = *meanPtr;
    float4 const v = *deltaPtr;
    meanX[idx] = m.x;
    delta2X[idx] = v.x;
    meanY[idx] = m.y;
    delta2Y[idx] = v.y;
    meanZ[idx] = m.z;
    delta2Z[idx] = v.z;
    N[idx] = v.w;
  }

  __device__ __forceinline__ void updateSample(float3 L) {
    int const idx = idxOff();
    float3 mean = make_float3(meanX[idx], meanY[idx], meanZ[idx]);
    float3 M2 = make_float3(delta2X[idx], delta2Y[idx], delta2Z[idx]);

    // Welford update
    float const num = ++N[idx];
    float3 const delta = L - mean;
    mean += delta / num;
    float3 const delta2 = L - mean;
    M2 += delta * delta2;

    // write back
    meanX[idx] = mean.x;
    meanY[idx] = mean.y;
    meanZ[idx] = mean.z;

    delta2X[idx] = M2.x;
    delta2Y[idx] = M2.y;
    delta2Z[idx] = M2.z;
  }

  __device__ __forceinline__ void endSample(float4* meanPtr, float4* deltaPtr) {
    int const idx = idxOff();
    (*meanPtr) = make_float4(meanX[idx], meanY[idx], meanZ[idx], 0);
    (*deltaPtr) = make_float4(delta2X[idx], delta2Y[idx], delta2Z[idx], N[idx]);
  }

  float meanX[BLOCK_SIZE + PAD];
  float delta2X[BLOCK_SIZE + PAD];
  float meanY[BLOCK_SIZE + PAD];
  float delta2Y[BLOCK_SIZE + PAD];
  float meanZ[BLOCK_SIZE + PAD];
  float delta2Z[BLOCK_SIZE + PAD];
  float N[BLOCK_SIZE + PAD];  // 1998 byte mark
  uint32_t _padding[BLOCK_SIZE - 7 * PAD];
};
static_assert(sizeof(SMEMLayout<64>) == 2048);
#endif

__global__ void  // __launch_bounds__(/*max threads per block*/ 256,
                 //                   /*min blocks per SM*/ 10)
pathTraceMegakernel(DeviceCamera* d_cam, TriangleSoup d_triSoup,
                    Light const* d_lights, uint32_t const lightCount,
                    Light const* d_infiniteLights,
                    uint32_t const infiniteLightCount, BSDF const* d_bsdf,
                    uint32_t const bsdfCount, uint32_t const sampleOffset,
                    DeviceHaltonOwen* d_haltonOwen,
#if DMT_ENABLE_MSE
                    DeviceOutputBuffer d_out
#else
                    float4* d_outBuffer
#endif
);

#endif
