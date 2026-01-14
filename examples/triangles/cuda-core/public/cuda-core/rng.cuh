#ifndef DMT_CUDA_CORE_RNG_CUH
#define DMT_CUDA_CORE_RNG_CUH

#include "common_math.cuh"
#include "cuda-core/types.cuh"

// ---------------------------------------------------------------------------
// Warp Wide Halton Sequence for pixel plane sampling, plus Owen Scrambling
// (digit-wise hash) for repurposing halton state into an RNG
// - requires per sample state, which we are storing int AOSOA format
// - hard to use on wavefront, as this would get stored into path state
// ---------------------------------------------------------------------------

struct DeviceHaltonOwen {
  static constexpr int MAX_RESOLUTION = 128;
  static __host__ __device__ DeviceHaltonOwenParams computeParams(int width,
                                                                  int height);

  __device__ void startPixelSample(DeviceHaltonOwenParams const& params, int2 p,
                                   int32_t sampleIndex, int32_t dim = 0);
  __device__ float get1D(DeviceHaltonOwenParams const& params);
  __device__ float2 get2D(DeviceHaltonOwenParams const& params);
  __device__ float2 getPixel2D(DeviceHaltonOwenParams const& params);

  int haltonIndex[WARP_SIZE];  // pixel
  int dimension[WARP_SIZE];    // general value
};

// ---------------------------------------------------------------------------
// Stateless RNG: PCG Hash
// - Src: https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
// ---------------------------------------------------------------------------
namespace PcgHash {

__device__ __forceinline__ unsigned pcgHash(unsigned seed) {
  uint32_t const state = seed * 747796405u + 2891336453u;
  uint32_t const word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

__device__ __forceinline__ unsigned seedFromClockAnd1DGlobalTid() {
  unsigned const tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned clock, clock_hi;
  asm volatile(
      "mov.u32 %0, %%clock;\n\t"
      "mov.u32 %1, %%clock_hi;"
      : "=r"(clock), "=r"(clock_hi));
  // from cycles' codebase on hashing structs
  unsigned const seed = tid * 33 ^ (clock * 33 ^ clock_hi);
  return seed;
}

template <typename T>
  requires std::is_floating_point_v<T> || std::is_integral_v<T>
__device__ __forceinline__ T get1D() {
  if constexpr (std::is_same_v<T, float>) {
    return uint_to_float01(pcgHash(seedFromClockAnd1DGlobalTid()));
  } else {
    return pcgHash(seedFromClockAnd1DGlobalTid());
  }
}

template <typename T>
  requires std::is_same_v<T, int2> || std::is_same_v<T, float2>
__device__ __forceinline__ T get2D() {
  if constexpr (std::is_same_v<T, float2>) {
    // We need two distinct random numbers.
    // We achieve this by running the hash twice with slightly perturbed inputs.
    // Re-hashing the output of h1 is cheaper
    unsigned const h1 = pcgHash(seedFromClockAnd1DGlobalTid());
    unsigned const h2 = pcgHash(h1 ^ 0X85EBCA6B);
    return make_float2(uint_to_float01(h1), uint_to_float01(h2));
  } else {
    unsigned const h1 = pcgHash(seedFromClockAnd1DGlobalTid());
    unsigned const h2 = pcgHash(h1 ^ 0X85EBCA6B);
    return make_int2(h1, h2);
  }
}

}  // namespace PcgHash

#endif  // DMT_CUDA_CORE_RNG_CUH
