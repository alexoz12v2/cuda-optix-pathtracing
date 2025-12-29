#ifndef DMT_CUDA_CORE_RNG_CUH
#define DMT_CUDA_CORE_RNG_CUH

#include "cuda-core/types.cuh"

// Or switch to Sobol (easier on GPU)
/// warp-wide struct
struct DeviceHaltonOwen {
  static constexpr int MAX_RESOLUTION = 128;
  __host__ __device__ DeviceHaltonOwenParams computeParams(int width,
                                                           int height);

  __device__ void startPixelSample(DeviceHaltonOwenParams const& params, int2 p,
                                   int32_t sampleIndex, int32_t dim = 0);
  __device__ float get1D(DeviceHaltonOwenParams const& params);
  __device__ float2 get2D(DeviceHaltonOwenParams const& params);
  __device__ float2 getPixel2D(DeviceHaltonOwenParams const& params);

  int haltonIndex[WARP_SIZE];  // pixel
  int dimension[WARP_SIZE];    // general value
};

#endif  // DMT_CUDA_CORE_RNG_CUH