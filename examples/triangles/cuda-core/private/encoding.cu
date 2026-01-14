#include "encoding.cuh"

#include "common_math.cuh"

#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Octahedral mapping (normal buffer, light direction)
// ---------------------------------------------------------------------------

// If you want IEEE-style sign (i.e., sign(0)=0), then this is not correct
__host__ __device__ __forceinline__ float signf(float f) {
  bool const b = signbit(f);
  return b * -1.f + !b * 1.f;
}

__host__ __device__ __forceinline__ uint32_t encodeOctaComponent(float f) {
  static constexpr float max = static_cast<uint16_t>(~0);
  return static_cast<uint32_t>(
      roundf(fmaxf(fminf((f + 1) * 0.5f * max, 1.f), 0.f)));
}

// TODO: is branchless code actually compiling different PTX/SASS instructions?
// Note: PTX/SASS have the capability to issue masked/conditioned instructions
// similar to ARM, so branchless might be only a punch to my eyes
__host__ __device__ uint32_t octaFromDir(float3 const dir) {
  // 1: planar projection: Division by manhattan norm
  auto [x, y, z] = dir / (fabsf(dir.x) + fabsf(dir.y) + fabsf(dir.z));

  // 2: outfold the downward faces to the external triangles of the quad
  bool const flip = z < 0.f;
  x = flip * (1.f - fabsf(y)) * signf(x) + !flip * x;
  y = flip * (1.f - fabsf(x)) * signf(y) + !flip * y;

  // 3: mapping into [0,1]
  return encodeOctaComponent(y) << 16 | encodeOctaComponent(x);
}

__host__ __device__ float3 dirFromOcta(uint32_t const octa) {
  static constexpr float max = static_cast<uint16_t>(~0);

  // 1: convert back to [-1, 1] range
  uint32_t const ox = octa & 0xFFFFu;
  uint32_t const oy = (octa >> 16) & 0xFFFFu;
  float2 const f{static_cast<float>(ox) / max * 2.f - 1.f,
                 static_cast<float>(oy) / max * 2.f - 1.f};

  // 2: reconstruct z component
  float3 n(f.x, f.y, 1.f - fabsf(f.x) - fabsf(f.y));
  float const nx = n.x;
  float const ny = n.y;

  // 3: fold back the lower hemisphere
  bool const flip = n.z < 0.f;
  n.x = flip * (1.f - fabsf(ny)) * signf(nx) + !flip * nx;
  n.y = flip * (1.f - fabsf(nx)) * signf(ny) + !flip * ny;

  // 4: normalize
  return normalize(n);
}

// ---------------------------------------------------------------------------
// Half Precision floats Portable storage
// ---------------------------------------------------------------------------
__host__ __device__ uint16_t float_to_half_bits(float f) {
#if defined(__CUDA_ARCH__)
  static_assert(sizeof(uint16_t) == sizeof(__half));
  // Device: use native instruction
  union {
    uint16_t u;
    __half h;
  } h;
  // don't drop numbers to zero
  // assert(f != 0.f || h.u != 0);
  h.h = __float2half_rn(f);
  return h.u;
#else
  // Host: software conversion
  union {
    float f;
    uint32_t u;
  } v{f};

  uint32_t sign = (v.u >> 16) & 0x8000;
  int32_t exp = ((v.u >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = v.u & 0x007FFFFF;

  // NaN / Inf
  if (exp >= 31) {
    return sign | 0x7C00u | (mant ? 0x200u : 0);
  }

  // Underflow
  if (exp <= 0) {
    if (exp < -10) {
      return sign;  // too small -> signed zero
    }
    mant |= 0x00800000u;
    uint32_t shift = 14 - exp;
    uint32_t rounded = (mant >> shift) + ((mant >> (shift - 1)) & 1);
    return sign | static_cast<uint16_t>(rounded);
  }

  // Normalized
  uint32_t rounded = mant + 0x00001000u;  // round to nearest even
  if (rounded & 0x00800000u) {
    rounded = 0;
    exp += 1;
    if (exp >= 31) {
      return sign | 0x7C00u;  // overflow -> inf
    }
  }

  return sign | static_cast<uint16_t>(exp << 10) |
         static_cast<uint16_t>(rounded >> 13);
#endif
}
__host__ __device__ float half_bits_to_float(uint16_t h) {
#if defined(__CUDA_ARCH__)
  __half t;
  reinterpret_cast<uint16_t&>(t) = h;
  return __half2float(t);
#else
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x03FFu;

  uint32_t out;

  if (exp == 0) {
    if (mant == 0) {
      out = sign;  // zero
    } else {
      // subnormal
      exp = 127 - 15 + 1;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x03FFu;
      out = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    // Inf / NaN
    out = sign | 0x7F800000u | (mant << 13);
  } else {
    // Normalized
    out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }

  union {
    uint32_t u;
    float f;
  } v{out};
  return v.f;
#endif
}
