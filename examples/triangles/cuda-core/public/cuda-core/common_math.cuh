#ifndef DMT_CUDA_CORE_COMMON_MATH_CUH
#define DMT_CUDA_CORE_COMMON_MATH_CUH

#include "cuda-core/debug.cuh"

#include <cooperative_groups.h>
#include <cuda_fp16.h>

#include <cassert>

// ---------------------------------------------------------------------------
__device__ __forceinline__ void pascal_fixed_sleep(uint64_t nanoseconds) {
  uint64_t start;
  // Read the 64-bit nanosecond global timer
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(start));

  uint64_t now = start;
  while (now < start + nanoseconds) {
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(now));
  }
}

__device__ __forceinline__ unsigned getWarpId() {
  unsigned warpId = -1U;
  asm("mov.u32 %0, %%warpid;" : "=r"(warpId));
  return warpId;
}
__device__ __forceinline__ unsigned getLaneId() {
  unsigned laneId = -1U;
  asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
  return laneId;
}

// this is an alternatives to coalesced groups. Does it make a diff? IDK
__device__ __forceinline__ unsigned getCoalescedLaneId(unsigned mask) {
  // 1. Create a mask of all lanes _Lower_ than the current lane
  uint32_t const lower = (1u << getLaneId()) - 1;
  // 2. Count how many active threads are lower than me
  int const rank = __popc(mask & lower);
  return rank;
}

// TODO how to use?
__device__ __forceinline__ float groupedAtomicFetch(float* address) {
  unsigned const fullwarp = __activemask();
#if __CUDA_ARCH__ >= 700
  // 1. Identify all lanes in the wapr hitting same address
  unsigned const mask = __match_any_sync(fullwarp, (uintptr_t)address);
  // 2. Elect a leader for each unique address group
  int const leader = __ffs(mask) - 1;
  int const laneId = threadIdx.x % 32;  // could use %%laneid;
  float val = (float)0;
  // 3. Only leader performs hardware atomic fetch
  if (laneId == leader) {
    val = atomicAdd(address, (float)0);
  }
  // 4. Broadcast the fetched value from the leader to all lanes in the sam
  // group. we use the same mask to ensure shuffle stays within the matched
  // group
  return __shfl_sync(mask, val, leader);
#else
  // loop through all possible groups emulating match with ballot instruction
  unsigned active = fullwarp;
  float result = (float)0;
  while (active > 0) {
    // 1. Pick the first remaining lane as the leader of the round
    int const leader = __ffs(active) - 1;
    // 2. Broadcast the leader's address to se who else matches it
    float* refAddr = (float*)__shfl_sync(active, (uintptr_t)address, leader);
    unsigned match = __ballot_sync(active, address == refAddr);
    // 3. leader performs atomic fetch
    float fetchedVal = (float)0;
    if ((threadIdx.x % 32) == leader) {
      fetchedVal = atomicAdd(refAddr, (float)0);
    }
    // 4. Broadcast value to everyone in the current match group from
    // currently elected leader.
    float const sharedVal = __shfl_sync(active, fetchedVal, leader);
    // 5. if current lane was part of group, then save result
    if (address == refAddr) {
      result = sharedVal;
    }
    // 6. Clear processed lanes
    active &= ~match;
  }
  return result;
#endif
}

// TODO: how can We group atomic operation per lanes sharing address
// __match_any_sync and __match_all_sync perform a broadcast-and-compare
// operation of a variable between threads within a warp. Supported by
// devices of compute capability 7.x or higher.
template <typename T>
  requires((std::is_floating_point_v<T> || std::is_integral_v<T>) &&
           sizeof(T) == 4)
__device__ __forceinline__ void groupedAtomicAdd(T* address, T val) {
  unsigned const fullwarp = __activemask();
#if __CUDA_ARCH__ >= 700
  // 1. find all lanes that have the same address
  unsigned const mask = __match_any_sync(fullwarp, (uintptr_t)address);
  // 2. Identify the leader (lane with lowest ID in mask)
  int const leader = __ffs(mask) - 1;
  int const laneId = threadIdx.x % 32;  // might be %laneid
  // 3. Intra-warp reduction within group
  float res = val;  // start with your value
  for (int i = 0; i < 32; ++i) {
    // if not leader and active
    if (i != leader && (mask & (1 << i))) {
      res += __shfl_sync(mask, val, i);
    }
  }

  // 4. leader does the add
  if (laneId == leader) {
    atomicAdd(address, res);
  }
#else
  // pascal doesn't have match instructions. iterative ballot approach.
  // loop through unique addresses present in the warp one by one
  // 0. mask of lanes which haven't been processed yet
  unsigned active = fullwarp;
  while (active > 0) {
    // 1. Pick a reference address from the first active lane
    int const leader = __ffs(active) - 1;
    auto* const refAddr = (T*)__shfl_sync(active, (uintptr_t)address, leader);
    // 2. Find all other lanes with same address
    unsigned const matching = __ballot_sync(active, address == refAddr);
    // 3. reduction within the matching group
    float res = val;  // start with your value
    for (int i = 16; i > 0; i >>= 1) {
      float const temp = __shfl_down_sync(matching, res, i);
      // if lane is part of match, add value (%laneid might be used here)
      if (matching & (1 << ((threadIdx.x % 32) + i))) {
        res += temp;
      }
    }
    // 4. Leader of matching group performs the atomic
    if ((threadIdx.x % 32) == leader) {
      atomicAdd(refAddr, res);
    }
    // 5. Remove the processed lanes from the active mask
    active &= ~matching;
  }
#endif
}

// TODO __CUDA_ARCH__ >= 700 version
__device__ __forceinline__ int groupedAtomicIncLeaderOnly(int* address) {
  // pascal doesn't have match instructions. iterative ballot approach.
  // loop through unique addresses present in the warp one by one
  // 0. mask of lanes which haven't been processed yet
  unsigned active = __activemask();
  int old = 0;
  while (active > 0) {
    // 1. Pick a reference address from the first active lane
    int const leader = __ffs(active) - 1;
    auto* const refAddr = (int*)__shfl_sync(active, (uintptr_t)address, leader);
    // 2. Find all other lanes with same address
    unsigned const matching = __ballot_sync(active, address == refAddr);
    // 3. Leader of matching group performs the atomic inc
    if ((threadIdx.x % 32) == leader) {
      old = atomicAdd(refAddr, 1);
    }
    // 4. Leader broadcasts result to members of the current matching group
    if (address == refAddr) {
      old = __shfl_sync(active, old, leader);
    }
    // 5. Remove the processed lanes from the active mask
    active &= ~matching;
  }
  return old;
}

__device__ __forceinline__ float uint_to_float01(unsigned int x) {
  // Multiply by 1 / 2^32
  return (float)x * 2.3283064365386963e-10f;
}

struct Transform {
  float m[16];
  float mInv[16];

  __host__ __device__ Transform();
  __host__ __device__ explicit Transform(float const* m);

  // TODO distinguish between point/vector
  __host__ __device__ float3 applyDirection(float3 v) const;

  __host__ __device__ float3 apply(float3 p) const;
  __host__ __device__ float3 applyInverse(float3 p) const;
  __host__ __device__ float3 applyTranspose(float3 n) const;
  __host__ __device__ float3 applyInverseTranspose(float3 n) const;
};

__host__ __device__ __forceinline__ __half flip_fp16(__half x) {
#if defined(__CUDA_ARCH__)
  // GPU optimized path
  return hrcp(x);
#else
  // CPU fallback path
  return __float2half(1.0f / __half2float(x));
#endif
}

__host__ __device__ __forceinline__ int ceilDiv(int num, int den) {
  return (num + den - 1) / den;
}

__host__ __device__ __forceinline__ uint32_t ceilDiv(uint32_t num,
                                                     uint32_t den) {
  return (num + den - 1) / den;
}

// (SEL or CSEL in PTX/SASS assembly) rather than actual branching logic
// TODO check generated PTX:  should be
// and.b32
// shl.b32
// xor.b32
__host__ __device__ __forceinline__ float flipIfOdd(float v, int b) {
  // Flip sign bit if b is odd
  int mask = (b & 1) << 31;
#ifdef __CUDA_ARCH__
  return __int_as_float(__float_as_int(v) ^ mask);
#else
  union {
    unsigned b32;
    float f32;
  } val;
  val.f32 = v;
  val.b32 = val.b32 ^ mask;
  return val.f32;
#endif
}

__host__ __device__ __forceinline__ float3 flipIfOdd(float3 v, int b) {
  return make_float3(flipIfOdd(v.x, b), flipIfOdd(v.y, b), flipIfOdd(v.z, b));
}

__forceinline__ __host__ __device__ float& float3_at(float3& v, int i) {
#ifdef __CUDA_ARCH__
  __builtin_assume(i >= 0 && i < 3);
#endif
#if DMT_ENABLE_ASSERTS
  assert(i >= 0 && i < 3);
#endif
  return reinterpret_cast<float*>(&v.x)[i];
}
__forceinline__ __host__ __device__ float const& float3_at(float3 const& v,
                                                           int i) {
#ifdef __CUDA_ARCH__
  __builtin_assume(i >= 0 && i < 3);
#endif
#if DMT_ENABLE_ASSERTS
  assert(i >= 0 && i < 3);
#endif
  return reinterpret_cast<float const*>(&v.x)[i];
}

__host__ __device__ __forceinline__ float sqrf(float const f) { return f * f; }
__host__ __device__ __forceinline__ float maxComponentValue(float3 v) {
  return fmaxf(v.x, fmaxf(v.y, v.z));
}
__host__ __device__ __forceinline__ float minComponentValue(float3 v) {
  return fminf(v.x, fminf(v.y, v.z));
}
__host__ __device__ __forceinline__ bool isZero(const float3 v) {
  return v.x == 0.f && v.y == 0.f && v.z == 0.f;
}
__host__ __device__ __forceinline__ float3 minv(float3 a, float3 b) {
  return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
__host__ __device__ __forceinline__ float3 maxv(float3 a, float3 b) {
  return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
// threshold on L∾ norm
__host__ __device__ __forceinline__ bool nearZero(const float3 v,
                                                  const float tol) {
  return fabsf(v.x) < tol && fabsf(v.y) < tol && fabsf(v.z) < tol;
}
// small improvement if you know for a fact that your numbers are positive
__host__ __device__ __forceinline__ bool nearZeroPos(const float3 v,
                                                     float tol) {
  return v.x < tol && v.y < tol && v.z < tol;
}
__host__ __device__ __forceinline__ float luminance(const float3 c) {
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}
__host__ __device__ __forceinline__ float length(float3 a) {
  return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}
__host__ __device__ __forceinline__ float3 cross(float3 a, float3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
// TODO safety assert
__host__ __device__ __forceinline__ float3 normalize(float3 a) {
  float const invMag = rsqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
  return make_float3(a.x * invMag, a.y * invMag, a.z * invMag);
}
__host__ __device__ __forceinline__ float lerp(float a, float b, float t) {
  // (1 - t) a + t b =  a - ta + tb = a + t ( b - a )
  float const _1mt = 1.f - t;
  return _1mt * a + t * b;
}
__host__ __device__ __forceinline__ float3 operator/(float3 v, float a) {
  return make_float3(v.x / a, v.y / a, v.z / a);
}
__host__ __device__ __forceinline__ float3 operator/(float3 v, float3 a) {
  return make_float3(v.x / a.x, v.y / a.y, v.z / a.z);
}
__host__ __device__ __forceinline__ float3& operator/=(float3& v, float a) {
  v.x /= a;
  v.y /= a;
  v.z /= a;
  return v;
}
__host__ __device__ __forceinline__ float3& operator/=(float3& v,
                                                       float3 const a) {
  v.x /= a.x;
  v.y /= a.y;
  v.z /= a.z;
  return v;
}
__host__ __device__ __forceinline__ float3 operator*(float3 v, float a) {
  return make_float3(v.x * a, v.y * a, v.z * a);
}
__host__ __device__ __forceinline__ float3& operator*=(float3& v, float3 a) {
  v.x *= a.x;
  v.y *= a.y;
  v.z *= a.z;
  return v;
}
__host__ __device__ __forceinline__ float3& operator*=(float3& v, float a) {
  v.x *= a;
  v.y *= a;
  v.z *= a;
  return v;
}
__host__ __device__ __forceinline__ float3 operator*(float a, float3 v) {
  return make_float3(v.x * a, v.y * a, v.z * a);
}
__host__ __device__ __forceinline__ float2 operator*(float2 v, float a) {
  return make_float2(v.x * a, v.y * a);
}
__host__ __device__ __forceinline__ float2 operator*(float a, float2 v) {
  return make_float2(v.x * a, v.y * a);
}
__host__ __device__ __forceinline__ float2& operator*=(float2& v, float a) {
  v.x = a;
  v.y = a;
  return v;
}
__host__ __device__ __forceinline__ float3 operator*(float3 a, float3 v) {
  return make_float3(v.x * a.x, v.y * a.y, v.z * a.z);
}
__host__ __device__ __forceinline__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ __forceinline__ float3& operator+=(float3& a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}
__host__ __device__ __forceinline__ float4& operator+=(float4& a, float4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}
__host__ __device__ __forceinline__ float3 operator+(float3 a, float b) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}
__host__ __device__ __forceinline__ float3 operator+(float b, float3 a) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}
__host__ __device__ __forceinline__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ __forceinline__ float3 operator-(float3 a, float b) {
  return make_float3(a.x - b, a.y - b, a.z - b);
}
__host__ __device__ __forceinline__ float3 operator-(float a, float3 b) {
  return make_float3(a - b.x, a - b.y, a - b.z);
}
__host__ __device__ __forceinline__ float3 operator-(float3 a) {
  return make_float3(-a.x, -a.y, -a.z);
}
__host__ __device__ __forceinline__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ __forceinline__ float dot(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
}
__host__ __device__ __forceinline__ float safeSqrt(float a) {
  return sqrtf(fmaxf(a, 0.f));
}
__host__ __device__ __forceinline__ float3 sqrt(float3 a) {
  return make_float3(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z));
}
__host__ __device__ __forceinline__ float3 abs(float3 a) {
  return make_float3(fabsf(a.x), fabsf(a.y), fabs(a.z));
}
__host__ __device__ __forceinline__ float average(float3 a) {
  return (a.x + a.y + a.z) / 3.f;
}

__host__ __device__ __forceinline__ float2 operator+(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}
__host__ __device__ __forceinline__ float2 operator-(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}
__host__ __device__ __forceinline__ float3 lerp(float3 a, float3 b, float t) {
  // (1 - t) a + t b =  a - ta + tb = a + t ( b - a )
  float const _1mt = 1.f - t;
  return _1mt * a + t * b;
}
__host__ __device__ __forceinline__ bool allStrictlyPositive(float3 a) {
  return a.x > FLT_MIN && isfinite(a.x) && a.y > FLT_MIN && isfinite(a.y) &&
         a.z > FLT_MIN && isfinite(a.z);
}
__host__ __device__ __forceinline__ bool componentWiseNear(float3 a, float3 b) {
  bool const validNums0 = isfinite(a.x) && isfinite(a.y) && isfinite(a.z);
  bool const validNums1 = isfinite(b.x) && isfinite(b.y) && isfinite(b.z);
  return validNums1 && validNums0 && fabsf(maxComponentValue(a - b)) < 1e-3f;
}

__host__ __device__ __forceinline__ bool is_valid_non_denormal(float x) {
  if (!isfinite(x)) return false;

  // fabsf is device-safe
  float ax = fabsf(x);
  return (ax == 0.0f) || (ax >= FLT_MIN);
}

__host__ __device__ __forceinline__ float sin_sqr_to_one_minus_cos(
    float const s_sq) {
  // Using second-order Taylor expansion at small angles for better accuracy.
  return s_sq > 0.0004f ? 1.0f - safeSqrt(1.0f - s_sq) : 0.5f * s_sq;
}
__host__ __device__ __forceinline__ float safeacos(float v) {
  return acosf(fminf(fmaxf(v, -1.f), 1.f));
}
__host__ __device__ __forceinline__ float length2(float3 v) {
  return dot(v, v);
}
__host__ __device__ __forceinline__ float length2(float2 v) {
  return dot(v, v);
}
__host__ __device__ __forceinline__ void gramSchmidt(float3 n, float3* a,
                                                     float3* b) {
#if DMT_ENABLE_ASSERTS
  assert(a && b && abs(length(n) - 1.f) < 1e-5f);
#endif
  if (fabsf(n.x - n.y) > 1e-3f || fabsf(n.x - n.z) > 1e-3f)
    *a = {n.z - n.y, n.x - n.z, n.y - n.x};  //(1,1,1)x N
  else
    *a = {n.z - n.y, n.x + n.z, -n.y - n.x};  //(-1,1,1)x N

  *a = normalize(*a);
  *b = cross(n, *a);
}
__host__ __device__ __forceinline__ void orthonormalTangent(const float3 n,
                                                            float3 const t,
                                                            float3* a,
                                                            float3* b) {
  *b = normalize(cross(n, t));
  *a = cross(*b, n);
}
__host__ __device__ __forceinline__ float2 cartesianFromPolar(float rho,
                                                              float phi) {
  return {rho * cosf(phi), rho * sinf(phi)};
}

__host__ __device__ __forceinline__ float smoothstep(float x) {
  if (x <= 0.f) return 0.f;
  if (x >= 1.f) return 1.f;
  float const x2 = x * x;
  return 3.f * x2 - 2.f * x2 * x;
}
__host__ __device__ __forceinline__ float smoothstep(float a, float b,
                                                     float x) {
  float const t = fmaxf(fminf((x - a) / (b - a), 0.f), 1.f);

  return smoothstep(t);
}
__host__ __device__ __forceinline__ float sin_from_cos(float const cosine) {
  return safeSqrt(1.f - sqrf(cosine));
}

__host__ __device__ __forceinline__ uint32_t nextPow2(uint32_t x) {
  if (x == 0) return 1;
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}

inline __host__ __device__ __forceinline__ Transform const* arrayAsTransform(
    float const* arr) {
  static_assert(sizeof(Transform) == 32 * sizeof(float) &&
                alignof(Transform) <= 16);
  // 16-byte aligned and at least 32 elements
  return reinterpret_cast<Transform const*>(arr);
}

#endif