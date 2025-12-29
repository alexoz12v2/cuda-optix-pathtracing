#ifndef DMT_CUDA_CORE_MORTON_CUH
#define DMT_CUDA_CORE_MORTON_CUH

#include "cuda-core/types.cuh"
#include "cuda-core/common_math.cuh"

inline __host__ __device__ __forceinline__ MortonLayout2D
mortonLayout(uint32_t rows, uint32_t cols) {
  MortonLayout2D layout{};
  layout.rows = rows;
  layout.cols = cols;
  layout.mortonRows = nextPow2(rows);
  layout.mortonCols = nextPow2(cols);
  layout.mortonCount =
      (uint64_t)layout.mortonRows * (uint64_t)layout.mortonCols;
  return layout;
}

// clang-format off
inline __host__ __device__ __forceinline__ uint32_t part1by1(uint32_t x) {
  x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  x = (x | (x << 8)) & 0x00ff00ff;  // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x | (x << 4)) & 0x0f0f0f0f;  // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x | (x << 2)) & 0x33333333;  // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x | (x << 1)) & 0x55555555;  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  return x;
}
inline __host__ __device__ __forceinline__ uint32_t compact1By1(uint32_t x) {
  x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  x = (x | (x >> 1)) & 0x33333333;  // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x | (x >> 2)) & 0x0f0f0f0f;  // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x | (x >> 4)) & 0x00ff00ff;  // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x | (x >> 8)) & 0x0000ffff;  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  return x;
}
// clang-format on

// row-major morton 2D -> warp 4 x 8
inline __host__ __device__ __forceinline__ uint32_t encodeMorton2D(uint32_t x,
                                                                   uint32_t y) {
  return (part1by1(y) << 1) | part1by1(x);
}
inline __host__ __device__ __forceinline__ void decodeMorton2D(uint32_t code,
                                                               uint32_t* x,
                                                               uint32_t* y) {
  *x = compact1By1(code);
  *y = compact1By1(code >> 1);
}

#endif  // DMT_CUDA_CORE_MORTON_CUH