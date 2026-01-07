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
  PRINT("      - Requested Halton Owen RNG: pi: %d, i: %u, s: %u | Base: %u\n",
        primeIndex, index, seed, base);
  while (index) {
    uint32_t next = index / base;
    uint32_t digit = index - next * base;

    uint32_t scramble = mixBits32(seed ^ uint32_t(reversed));
    digit = permutationElement(digit, base, scramble);

    reversed = reversed * base + digit;
    invBaseM *= invBase;
    index = next;
    PRINT("         - Halton Owen RNG update: index: %u digit: %u\n", index,
          digit);
  }
  PRINT("      - Halton Owen RNG: finished\n");

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
  // TODO global debug printing macro control
#if 0
  printf(
      "   START PIXEL SAMPLE STATE:\n"
      "     - haltonIndex: %d\n"
      "     - dimension:   %d\n",
      haltonIndex[laneId], dimension[laneId]);
#endif
}

__device__ float DeviceHaltonOwen::get1D(DeviceHaltonOwenParams const& params) {
  int const laneId = cooperative_groups::thread_block_tile<32>::thread_rank();

  if (dimension[laneId] >= NUM_PRIMES) dimension[laneId] = 2;
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
#if 0
  printf("  Sampled: %f %f\n", result.x, result.y);
#endif
  return result;
}
