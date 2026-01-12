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

__global__ void  // __launch_bounds__(/*max threads per block*/ 256,
                 //                   /*min blocks per SM*/ 10)
pathTraceMegakernel(DeviceCamera* d_cam, TriangleSoup d_triSoup,
                    Light const* d_lights, uint32_t const lightCount,
                    Light const* d_infiniteLights,
                    uint32_t const infiniteLightCount, BSDF const* d_bsdf,
                    uint32_t const bsdfCount, uint32_t const sampleOffset,
                    DeviceHaltonOwen* d_haltonOwen, float4* d_outBuffer);

#endif