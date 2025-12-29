#ifndef DMT_CUDA_CORE_ENCODING_CUH
#define DMT_CUDA_CORE_ENCODING_CUH

// ---------------------------------------------------------------------------
// Octahedral mapping (normal buffer, light direction)
// ---------------------------------------------------------------------------
// TODO: probably __forceinline__?
__host__ __device__ uint32_t octaFromDir(float3 const dir);
__host__ __device__ float3 dirFromOcta(uint32_t const octa);

// ---------------------------------------------------------------------------
// Half Precision floats Portable storage
// ---------------------------------------------------------------------------
// on device code, you can use float16 intrinsics. Host coverts back and forth
// from full 4-byte float
// TODO: probably __forceinline__
__host__ __device__ uint16_t float_to_half_bits(float f);
__host__ __device__ float half_bits_to_float(uint16_t h);

inline __host__ __device__ __forceinline__ float3
half_vec_to_float3(uint16_t const h[3]) {
  return make_float3(half_bits_to_float(h[0]), half_bits_to_float(h[1]),
                     half_bits_to_float(h[2]));
}

#endif