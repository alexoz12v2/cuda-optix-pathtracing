#include "common.cuh"

#include <cstring>

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

// Approx
__device__ __forceinline__ float radicalInverse(uint32_t primeIndex,
                                                uint32_t num, int nDigits) {
  uint32_t base = primes[primeIndex];
  float invBase = __frcp_rn((float)base);

  float result = 0.0f;
  float factor = invBase;

  // Fixed iteration count → warp-uniform
  for (int i = 0; i < nDigits; ++i) {
    uint32_t next = num / base;
    uint32_t digit = num - next * base;

    result += digit * factor;
    factor *= invBase;
    num = next;
  }

  return fminf(result, 0.99999994f);
}

__device__ __forceinline__ uint32_t mixBits32(uint32_t v) {
  v ^= v >> 16;
  v *= 0x7feb352dU;
  v ^= v >> 15;
  v *= 0x846ca68bU;
  v ^= v >> 16;
  return v;
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

__device__ __forceinline__ float owenScrambledRadicalInverse32(int primeIndex,
                                                               uint32_t num,
                                                               uint32_t hash,
                                                               int nDigits) {
  uint32_t base = primes[primeIndex];
  float invBase = __frcp_rn((float)base);

  uint32_t reversed = 0;
  float invBaseM = 1.0f;

  // IMPORTANT: fixed iteration count → warp-uniform
  for (int i = 0; i < nDigits; ++i) {
    uint32_t next = num / base;
    uint32_t digit = num - next * base;

    uint32_t scramble = mixBits32(hash ^ reversed);

    digit = permutationElement(digit, base, scramble);

    reversed = reversed * base + digit;
    invBaseM *= invBase;
    num = next;
  }

  return fminf(reversed * invBaseM, 0.99999994f);
}

__device__ __forceinline__ float sampleDim(int dimension, int haltonIndex,
                                           int baseExponent) {
  return owenScrambledRadicalInverse32(
      dimension, haltonIndex, mixBits32(1 + (dimension << 4)), baseExponent);
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
}

__device__ float DeviceHaltonOwen::get1D(DeviceHaltonOwenParams const& params) {
  int const laneId = cooperative_groups::thread_block_tile<32>::thread_rank();

  dimension[laneId] = max(dimension[laneId], NUM_PRIMES);
  int const dim = dimension[laneId]++;
  return sampleDim(dim, haltonIndex[laneId],
                   params.baseExponents[max(dim, NUM_FILM_DIMENSION - 1)]);
}

__device__ float2
DeviceHaltonOwen::get2D(DeviceHaltonOwenParams const& params) {
  int const laneId = cooperative_groups::thread_block_tile<32>::thread_rank();

  dimension[laneId] = max(dimension[laneId] + 1, NUM_PRIMES);
  int const dim = dimension[laneId];
  dimension[laneId] += 2;

  return make_float2(
      sampleDim(dim, haltonIndex[laneId],
                params.baseExponents[max(dim, NUM_FILM_DIMENSION - 1)]),
      sampleDim(dim + 1, haltonIndex[laneId],
                params.baseExponents[max(dim + 1, NUM_FILM_DIMENSION - 1)]));
}

__device__ float2
DeviceHaltonOwen::getPixel2D(DeviceHaltonOwenParams const& params) {
  int const laneId = cooperative_groups::thread_block_tile<32>::thread_rank();

  return make_float2(
      radicalInverse(0, haltonIndex[laneId] >> params.baseExponents[0],
                     params.baseExponents[0]),
      radicalInverse(1, haltonIndex[laneId] / params.baseScales[1],
                     params.baseExponents[1]));
}