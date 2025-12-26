#include "common.cuh"

#include <cuda_fp16.h>

#include <cstring>
#include <numbers>

__host__ __device__ Transform::Transform() {
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      mInv[i * 4 + j] = 0.f;
      m[i * 4 + j] = 0.f;
    }
  }
  for (int j = 0; j < 4; ++j) {
    mInv[j * 4 + j] = 1.f;
    m[j * 4 + j] = 1.f;
  }
}

__host__ __device__ Transform::Transform(float const* _m) {
  for (int i = 0; i < 16; ++i) {
    m[i] = _m[i];
  }

  mInv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
            m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];

  mInv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
            m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];

  mInv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
            m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];

  mInv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
             m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

  mInv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
            m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];

  mInv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
            m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];

  mInv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
            m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];

  mInv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
             m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

  mInv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
            m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];

  mInv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
            m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];

  mInv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
             m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];

  mInv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
             m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

  mInv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
            m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];

  mInv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
            m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];

  mInv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
             m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];

  mInv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
             m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

  float det =
      m[0] * mInv[0] + m[1] * mInv[4] + m[2] * mInv[8] + m[3] * mInv[12];

  // assert(det != 0.0f); // non-singular assumption

  float mInvDet = 1.0f / det;
  for (float& v : mInv) {
    v *= mInvDet;
  }
}

__host__ __device__ float3 Transform::applyDirection(float3 v) const {
  float const x = m[0] * v.x + m[4] * v.y + m[8] * v.z;
  float const y = m[1] * v.x + m[5] * v.y + m[9] * v.z;
  float const z = m[2] * v.x + m[6] * v.y + m[10] * v.z;
  return make_float3(x, y, z);
}

// Point transform (w = 1)
__host__ __device__ float3 Transform::apply(float3 p) const {
  float x = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
  float y = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
  float z = m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14];
  float w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];

  if (w != 1.0f && w != 0.0f) {
    float invW = 1.0f / w;
    x *= invW;
    y *= invW;
    z *= invW;
  }

  return make_float3(x, y, z);
}

__host__ __device__ float3 Transform::applyInverse(float3 p) const {
  float x = mInv[0] * p.x + mInv[4] * p.y + mInv[8] * p.z + mInv[12];
  float y = mInv[1] * p.x + mInv[5] * p.y + mInv[9] * p.z + mInv[13];
  float z = mInv[2] * p.x + mInv[6] * p.y + mInv[10] * p.z + mInv[14];
  float w = mInv[3] * p.x + mInv[7] * p.y + mInv[11] * p.z + mInv[15];

  if (w != 1.0f && w != 0.0f) {
    float invW = 1.0f / w;
    x *= invW;
    y *= invW;
    z *= invW;
  }

  return make_float3(x, y, z);
}

__host__ __device__ float3 Transform::applyTranspose(float3 n) const {
  return make_float3(m[0] * n.x + m[1] * n.y + m[2] * n.z,
                     m[4] * n.x + m[5] * n.y + m[6] * n.z,
                     m[8] * n.x + m[9] * n.y + m[10] * n.z);
}

__host__ __device__ float3 Transform::applyInverseTranspose(float3 n) const {
  return make_float3(mInv[0] * n.x + mInv[1] * n.y + mInv[2] * n.z,
                     mInv[4] * n.x + mInv[5] * n.y + mInv[6] * n.z,
                     mInv[8] * n.x + mInv[9] * n.y + mInv[10] * n.z);
}

inline int constexpr NUM_PRIMES = 10;
static int h_primes[NUM_PRIMES] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};

__device__ __constant__ int primes[NUM_PRIMES] = {2,  3,  5,  7,  11,
                                                  13, 17, 19, 23, 29};

namespace {

__host__ __device__ __forceinline__ int64_t multiplicativeInverse(int64_t a,
                                                                  int64_t n) {
  int64_t t = 0, newt = 1;
  int64_t r = n, newr = a;

  while (newr != 0) {
    int64_t q = r / newr;

    int64_t tmp;

    tmp = t - q * newt;
    t = newt;
    newt = tmp;

    tmp = r - q * newr;
    r = newr;
    newr = tmp;
  }

  // If r > 1, inverse does not exist (a and n not coprime)
  // Caller responsibility if modulus is prime

  if (t < 0) t += n;

  return t;
}

__device__ __forceinline__ int32_t inverseRadicalInverse(int32_t inverse,
                                                         int32_t base,
                                                         int32_t nDigits) {
  int32_t index = 0;
  for (int32_t i = 0; i < nDigits; ++i) {
    int32_t const digit = inverse % base;
    inverse /= base;
    index *= base;
    index += digit;
  }
  return index;
}

__device__ __forceinline__ uint32_t mixBits32(uint32_t v) {
  v ^= v >> 16;
  v *= 0x7feb352dU;
  v ^= v >> 15;
  v *= 0x846ca68bU;
  v ^= v >> 16;
  return v;
}

// TODO Warp Uniform Version
__device__ __forceinline__ float radicalInverse(uint32_t primeIndex,
                                                uint32_t index) {
  const uint32_t base = primes[primeIndex];
  const float invBase = __frcp_rn(static_cast<float>(base));
  const uint32_t limit = ~0u / base - base;

  uint32_t result = 0;
  float factor = invBase;

  // TODO is it necessary to sync?
  const cooperative_groups::coalesced_group theGroup =
      cooperative_groups::coalesced_threads();
  while (index && result < limit) {
    uint32_t next = index / base;
    uint32_t digit = index - next * base;

    result = result * base + digit;
    factor *= invBase;
    index = next;
  }
  theGroup.sync();

  return fminf(result * factor, 0.99999994f);
}

__device__ __forceinline__ uint32_t permutationElement(uint32_t digit,
                                                       uint32_t base,
                                                       uint32_t seed) {
  // identical to PBRT's "permute"
  uint32_t w = base - 1;
  w |= w >> 1;
  w |= w >> 2;
  w |= w >> 4;
  w |= w >> 8;
  w |= w >> 16;

  do {
    digit ^= seed;
    digit *= 0xe170893d;
    digit ^= seed >> 16;
    digit ^= (digit & w) >> 4;
    digit ^= seed >> 8;
    digit *= 0x0929eb3f;
    digit ^= seed >> 23;
    digit ^= (digit & w) >> 1;
    digit *= 1 | seed >> 27;
    digit *= 0x6935fa69;
    digit ^= (digit & w) >> 11;
    digit *= 0x74dcb303;
    digit ^= (digit & w) >> 2;
    digit *= 0x9e501cc3;
    digit ^= (digit & w) >> 2;
    digit *= 0xc860a3df;
    digit &= w;
    digit ^= digit >> 5;
  } while (digit >= base);

  return digit;
}

// TODO Warp Uniform version
__device__ __forceinline__ float owenScrambledRadicalInverse32(int primeIndex,
                                                               uint32_t index,
                                                               uint32_t seed) {
  const uint32_t base = primes[primeIndex];
  double invBase = 1.0 / double(base);
  double invBaseM = 1.0;

  uint64_t reversed = 0;

  while (index) {
    uint32_t next = index / base;
    uint32_t digit = index - next * base;

    uint32_t scramble = mixBits32(seed ^ uint32_t(reversed));
    digit = permutationElement(digit, base, scramble);

    reversed = reversed * base + digit;
    invBaseM *= invBase;
    index = next;
  }

  const double result = static_cast<double>(reversed) * invBaseM;
  return static_cast<float>(fmin(result, 0.999999999999));
}

__device__ __forceinline__ float sampleDim(int dimension, int haltonIndex) {
  uint32_t const seed = mixBits32(1u + (static_cast<uint32_t>(dimension) << 4));
  return owenScrambledRadicalInverse32(dimension, haltonIndex, seed);
}

}  // namespace

__host__ __device__ DeviceHaltonOwenParams
DeviceHaltonOwen::computeParams(int width, int height) {
  DeviceHaltonOwenParams params{};
  int const res[2] = {width, height};
  for (int i = 0; i < NUM_FILM_DIMENSION; ++i) {
#ifdef __CUDA_ARCH__
    int32_t const base = primes[i];
#else
    int32_t const base = h_primes[i];
#endif
    params.baseScales[i] = 1;
    params.baseExponents[i] = 0;
    while (params.baseScales[i] < min(res[i], MAX_RESOLUTION)) {
      params.baseScales[i] *= base;
      ++params.baseExponents[i];
    }
  }

  params.multInvs[0] =
      multiplicativeInverse(params.baseScales[1], params.baseScales[0]);
  params.multInvs[1] =
      multiplicativeInverse(params.baseScales[0], params.baseScales[1]);
#ifdef __CUDA_ARCH__
  __syncwarp();
#endif
  return params;
}

__device__ void DeviceHaltonOwen::startPixelSample(
    DeviceHaltonOwenParams const& params, int2 p, int32_t sampleIndex,
    int32_t dim) {
  int const laneId = cooperative_groups::thread_block_tile<32>::thread_rank();

  int const sampleStride = params.baseScales[0] * params.baseScales[1];
  haltonIndex[laneId] = 0;
  int const pm[2] = {p.x % MAX_RESOLUTION, p.y % MAX_RESOLUTION};

  for (int i = 0; i < NUM_FILM_DIMENSION; ++i) {
    // Radical inverse
    int32_t const dimOffset =
        inverseRadicalInverse(pm[i], primes[i], params.baseExponents[i]);

    haltonIndex[laneId] +=
        dimOffset * (sampleStride / params.baseScales[i]) * params.multInvs[i];
  }
  haltonIndex[laneId] %= sampleStride;
  haltonIndex[laneId] += sampleIndex * sampleStride;
  dimension[laneId] = max(dim, NUM_FILM_DIMENSION);
  // TODO global debug printing macro control
#if 0
  printf("   START PIXEL SAMPLE STATE:\n");
  printf("     - haltonIndex: %d\n", haltonIndex[laneId]);
  printf("     - dimension:   %d\n", dimension[laneId]);
#endif
}

__device__ float DeviceHaltonOwen::get1D(DeviceHaltonOwenParams const& params) {
  int const laneId = cooperative_groups::thread_block_tile<32>::thread_rank();

  if (dimension[laneId] > NUM_PRIMES) dimension[laneId] = 2;
  int const dim = dimension[laneId]++;
  float const result = sampleDim(dim, haltonIndex[laneId]);
#if 0
  printf("dim: %d halton index: %d exp: %d result: %f\n", dimension[laneId],
         haltonIndex[laneId],
         params.baseExponents[max(dim, NUM_FILM_DIMENSION - 1)], result);
#endif
  return result;
}

__device__ float2
DeviceHaltonOwen::get2D(DeviceHaltonOwenParams const& params) {
  int const laneId = cooperative_groups::thread_block_tile<32>::thread_rank();

  if (dimension[laneId] + 1 > NUM_PRIMES) dimension[laneId] = 2;
  int const dim = dimension[laneId];
  dimension[laneId] += 2;

  return make_float2(sampleDim(dim, haltonIndex[laneId]),
                     sampleDim(dim + 1, haltonIndex[laneId]));
}

__device__ float2
DeviceHaltonOwen::getPixel2D(DeviceHaltonOwenParams const& params) {
  int const laneId = cooperative_groups::thread_block_tile<32>::thread_rank();

  float2 const result = make_float2(
      radicalInverse(0, haltonIndex[laneId] >> params.baseExponents[0]),
      radicalInverse(1, haltonIndex[laneId] / params.baseScales[1]));
#if 0
  printf("  Sampled: %f %f\n", result.x, result.y);
#endif
  return result;
}

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
  // Device: use native instruction
  return __float2half_rn(f);
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

// ---------------------------------------------------------------------------
// Lights
// ---------------------------------------------------------------------------
namespace {

__host__ __device__ __forceinline__ void packIntensity(Light& l,
                                                       float3 const color,
                                                       ELightType type) {
  l.intensity[0] = float_to_half_bits(color.x);
  l.intensity[1] = float_to_half_bits(color.y);
  l.intensity[2] = float_to_half_bits(color.z);
  l.intensity[3] = static_cast<uint16_t>(type);
}

}  // namespace

__host__ __device__ Light makePointLight(float3 const color,
                                         float3 const position, float radius) {
  Light l{};
  packIntensity(l, color, ELightType::ePoint);
  l.data.point.pos = position;
  l.data.point.radius = float_to_half_bits(radius);
  return l;
}

__host__ __device__ Light makeSpotLight(float3 color, float3 position,
                                        float3 direction, float cosTheta0,
                                        float cosThetaE, float radius) {
  Light l{};
  packIntensity(l, color, ELightType::eSpot);
  l.data.spot.pos = position;
  l.data.spot.direction = octaFromDir(direction);
  l.data.spot.cosTheta0 = float_to_half_bits(cosTheta0);
  l.data.spot.cosThetaE = float_to_half_bits(cosThetaE);
  l.data.spot.radius = float_to_half_bits(radius);
  return l;
}

__host__ __device__ Light makeDirectionalLight(float3 const color,
                                               float3 const direction,
                                               float const oneMinusCosAngle) {
  Light l{};
  packIntensity(l, color, ELightType::eDirectional);
  l.data.dir.direction = octaFromDir(direction);
  l.data.dir.oneMinusCosAngle = float_to_half_bits(oneMinusCosAngle);
  return l;
}

__host__ __device__ Light makeEnvironmentalLight(float3 const color) {
  Light l{};
  packIntensity(l, color, ELightType::eEnv);
  return l;
}

__host__ __device__ float3 evalLight(Light const& light,
                                     LightSample const& ls) {
  float3 Le = light.getIntensity() * ls.factor;
  if (ELightType const type = light.type();
      type == ELightType::ePoint || type == ELightType::eSpot) {
    // quadratic attenuation (TODO Other variances?)
    Le /= (ls.distance * ls.distance);
  }
  return Le;
}

__host__ __device__ float3 evalInfiniteLight(Light const& light, float3 dir,
                                             float* pdf) {
  if (ELightType const type = light.type(); type != ELightType::eEnv) {
    *pdf = 0;
    return make_float3(0, 0, 0);
  }

  float3 const Le = light.getIntensity();
  *pdf = 0.25f * std::numbers::pi_v<float>;
  return Le;
}
