#include "rng.cuh"

// override header's definition (TODO REMOVE)
#undef PRINT
#define PRINT(...)

//------------------------------------------------
// HaltonOwen
//------------------------------------------------

inline int constexpr NUM_PRIMES = 10;
#ifndef __CUDA_ARCH
static int h_primes[NUM_PRIMES] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
#endif

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

__device__ __forceinline__ float radicalInverse(uint32_t primeIndex,
                                                uint32_t index) {
  const uint32_t base = primes[primeIndex];
  const float invBase = __frcp_rn(__uint2float_rn(base));
  float result = 0.0f;

  float invBasePow = invBase;

  // Loop depends on index magnitude, but removed the sync() and limit checks
  // which causes unnecessary overhead.
  while (index > 0) {
    // Fast integer division/modulo
    uint32_t const next = index / base;
    uint32_t const digit = index - next * base;

    // Fused Multiply Add: result += digit * invBasePow
    result = __fmaf_rn(__uint2float_rn(digit), invBasePow, result);

    invBasePow *= invBase;
    index = next;
  }

  // Clamp to avoid 1.0f inclusive
  return fminf(result, 0.99999994f);
}

__device__ __forceinline__ uint32_t permutationElement(uint32_t digit,
                                                       uint32_t base,
                                                       uint32_t seed) {
#if 0  // while this rejection sampling strategy works, it's not device friendly
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
#else
  // Simple randomized shift (Cranley-Patterson rotation on the digit)
  // This is strictly non-divergent and O(1).
  return (digit + seed) % base;
#endif
}

__device__ __forceinline__ float owenScrambledRadicalInverse32(int primeIndex,
                                                               uint32_t index,
                                                               uint32_t seed) {
  const uint32_t base = primes[primeIndex];
  const float invBase = __frcp_rn(__uint2float_rn(base));

  float result = 0.0f;
  float invBasePow = invBase;

  // We use reversed_int just for hash generation, not for accumulation
  // to avoid 64-bit integer requirements.
  uint32_t reversed_hash_input = 0;

  while (index > 0) {
    uint32_t const next = index / base;
    uint32_t const digit = index - next * base;

    // Mix the bits based on depth (reversed) to scramble the seed
    uint32_t const scramble = mixBits32(seed ^ reversed_hash_input);

    // Get permuted digit (branchless)
    uint32_t const permuted_digit = permutationElement(digit, base, scramble);

    // Accumulate directly in float
    result = __fmaf_rn(__uint2float_rn(permuted_digit), invBasePow, result);

    // Update state for next iteration
    // Note: We don't need a full integer reverse, just a changing hash input.
    // Standard accumulation is sufficient for the hash noise.
    reversed_hash_input = reversed_hash_input * base + digit;

    invBasePow *= invBase;
    index = next;
  }

  return fminf(result, 0.99999994f);
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
  // printf("DeviceHaltonOwen::startPixelSample\n");
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

  if (dimension[laneId] >= NUM_PRIMES) dimension[laneId] = 2;
  int const dim = dimension[laneId]++;
  float const result = sampleDim(dim, haltonIndex[laneId]);
  return result;
}

__device__ float2
DeviceHaltonOwen::get2D(DeviceHaltonOwenParams const& params) {
  int const laneId = cooperative_groups::thread_block_tile<32>::thread_rank();

  if (dimension[laneId] + 1 >= NUM_PRIMES) dimension[laneId] = 2;
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
  return result;
}
