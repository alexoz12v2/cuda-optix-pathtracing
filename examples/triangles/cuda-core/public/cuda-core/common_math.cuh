#ifndef DMT_CUDA_CORE_COMMON_MATH_CUH
#define DMT_CUDA_CORE_COMMON_MATH_CUH

#include <cassert>

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

inline __host__ __device__ __forceinline__ int ceilDiv(int num, int den) {
  return (num + den - 1) / den;
}

inline __host__ __device__ __forceinline__ uint32_t ceilDiv(uint32_t num, uint32_t den) {
  return (num + den - 1) / den;
}

// While the reinterpret_cast approach feels more "efficient" because it looks
// like direct memory access, it can actually hinder the compiler's ability to
// optimize registers in a CUDA kernel. The CUDA compiler (nvcc) is highly
// optimized for small, fixed-size switches. Because float3 only has three
// members, the compiler can often resolve this into a simple conditional move
// (SEL or CSEL in PTX/SASS assembly) rather than actual branching logic
// TODO check generated PTX
inline __host__ __device__ __forceinline__ float& float3_at(float3& v, int i) {
  switch (i) {
    case 0:
      return v.x;
    case 1:
      return v.y;
    case 2:
      return v.z;
    default:
      assert(false);
      return v.x;
  }
}
inline __host__ __device__ __forceinline__ float const& float3_at(
    float3 const& v, int i) {
  switch (i) {
    case 0:
      return v.x;
    case 1:
      return v.y;
    case 2:
      return v.z;
    default:
      assert(false);
      return v.x;
  }
}

inline __host__ __device__ __forceinline__ float sqrf(float const f) {
  return f * f;
}
inline __host__ __device__ __forceinline__ float maxComponentValue(float3 v) {
  return fmaxf(v.x, fmaxf(v.y, v.z));
}
inline __host__ __device__ __forceinline__ float minComponentValue(float3 v) {
  return fminf(v.x, fminf(v.y, v.z));
}
inline __host__ __device__ __forceinline__ bool isZero(const float3 v) {
  return v.x == 0.f && v.y == 0.f && v.z == 0.f;
}
inline __host__ __device__ __forceinline__ float3 minv(float3 a, float3 b) {
  return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
inline __host__ __device__ __forceinline__ float3 maxv(float3 a, float3 b) {
  return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
// threshold on L∾ norm
inline __host__ __device__ __forceinline__ bool nearZero(const float3 v,
                                                         const float tol) {
  return fabsf(v.x) < tol && fabsf(v.y) < tol && fabsf(v.z) < tol;
}
// small improvement if you know for a fact that your numbers are positive
inline __host__ __device__ __forceinline__ bool nearZeroPos(const float3 v,
                                                            float tol) {
  return v.x < tol && v.y < tol && v.z < tol;
}
inline __host__ __device__ __forceinline__ float luminance(const float3 c) {
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}
inline __host__ __device__ __forceinline__ float length(float3 a) {
  return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}
inline __host__ __device__ __forceinline__ float3 cross(float3 a, float3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
// TODO safety assert
inline __host__ __device__ __forceinline__ float3 normalize(float3 a) {
  float const invMag = rsqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
  return make_float3(a.x * invMag, a.y * invMag, a.z * invMag);
}
inline __host__ __device__ __forceinline__ float lerp(float a, float b,
                                                      float t) {
  // (1 - t) a + t b =  a - ta + tb = a + t ( b - a )
  float const _1mt = 1.f - t;
  return _1mt * a + t * b;
}
inline __host__ __device__ __forceinline__ float3 operator/(float3 v, float a) {
  return make_float3(v.x / a, v.y / a, v.z / a);
}
inline __host__ __device__ __forceinline__ float3 operator/(float3 v,
                                                            float3 a) {
  return make_float3(v.x / a.x, v.y / a.y, v.z / a.z);
}
inline __host__ __device__ __forceinline__ float3& operator/=(float3& v,
                                                              float a) {
  v.x /= a;
  v.y /= a;
  v.z /= a;
  return v;
}
inline __host__ __device__ __forceinline__ float3& operator/=(float3& v,
                                                              float3 const a) {
  v.x /= a.x;
  v.y /= a.y;
  v.z /= a.z;
  return v;
}
inline __host__ __device__ __forceinline__ float3 operator*(float3 v, float a) {
  return make_float3(v.x * a, v.y * a, v.z * a);
}
inline __host__ __device__ __forceinline__ float3& operator*=(float3& v,
                                                              float3 a) {
  v.x *= a.x;
  v.y *= a.y;
  v.z *= a.z;
  return v;
}
inline __host__ __device__ __forceinline__ float3& operator*=(float3& v,
                                                              float a) {
  v.x *= a;
  v.y *= a;
  v.z *= a;
  return v;
}
inline __host__ __device__ __forceinline__ float3 operator*(float a, float3 v) {
  return make_float3(v.x * a, v.y * a, v.z * a);
}
inline __host__ __device__ __forceinline__ float2 operator*(float2 v, float a) {
  return make_float2(v.x * a, v.y * a);
}
inline __host__ __device__ __forceinline__ float2 operator*(float a, float2 v) {
  return make_float2(v.x * a, v.y * a);
}
inline __host__ __device__ __forceinline__ float2& operator*=(float2& v,
                                                              float a) {
  v.x = a;
  v.y = a;
  return v;
}
inline __host__ __device__ __forceinline__ float3 operator*(float3 a,
                                                            float3 v) {
  return make_float3(v.x * a.x, v.y * a.y, v.z * a.z);
}
inline __host__ __device__ __forceinline__ float3 operator+(float3 a,
                                                            float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ __forceinline__ float3& operator+=(float3& a,
                                                              float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}
inline __host__ __device__ __forceinline__ float3 operator+(float3 a, float b) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ __forceinline__ float3 operator+(float b, float3 a) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ __forceinline__ float3 operator-(float3 a,
                                                            float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ __forceinline__ float3 operator-(float3 a, float b) {
  return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ __forceinline__ float3 operator-(float a, float3 b) {
  return make_float3(a - b.x, a - b.y, a - b.z);
}
inline __host__ __device__ __forceinline__ float3 operator-(float3 a) {
  return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ __forceinline__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ __forceinline__ float dot(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ __forceinline__ float safeSqrt(float a) {
  return sqrtf(fmaxf(a, 0.f));
}
inline __host__ __device__ __forceinline__ float3 sqrt(float3 a) {
  return make_float3(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z));
}
inline __host__ __device__ __forceinline__ float3 abs(float3 a) {
  return make_float3(fabsf(a.x), fabsf(a.y), fabs(a.z));
}
inline __host__ __device__ __forceinline__ float average(float3 a) {
  return (a.x + a.y + a.z) / 3.f;
}

inline __host__ __device__ __forceinline__ float2 operator+(float2 a,
                                                            float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ __forceinline__ float2 operator-(float2 a,
                                                            float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ __forceinline__ float3 lerp(float3 a, float3 b,
                                                       float t) {
  // (1 - t) a + t b =  a - ta + tb = a + t ( b - a )
  float const _1mt = 1.f - t;
  return _1mt * a + t * b;
}
inline __host__ __device__ __forceinline__ bool allStrictlyPositive(float3 a) {
  return a.x > 1e-4f && isfinite(a.x) && a.y > 1e-4f && isfinite(a.y) &&
         a.z > 1e-4f && isfinite(a.z);
}
inline __host__ __device__ __forceinline__ bool componentWiseNear(float3 a,
                                                                  float3 b) {
  bool const validNums0 = isfinite(a.x) && isfinite(a.y) && isfinite(a.z);
  bool const validNums1 = isfinite(b.x) && isfinite(b.y) && isfinite(b.z);
  return validNums1 && validNums0 && fabsf(maxComponentValue(a - b)) < 1e-3f;
}

__host__ __device__ inline bool is_valid_non_denormal(float x) {
  if (!isfinite(x)) return false;

  // fabsf is device-safe
  float ax = fabsf(x);
  return (ax == 0.0f) || (ax >= FLT_MIN);
}

inline __host__ __device__ __forceinline__ float sin_sqr_to_one_minus_cos(
    float const s_sq) {
  // Using second-order Taylor expansion at small angles for better accuracy.
  return s_sq > 0.0004f ? 1.0f - safeSqrt(1.0f - s_sq) : 0.5f * s_sq;
}
inline __host__ __device__ __forceinline__ float safeacos(float v) {
  return acosf(fminf(fmaxf(v, -1.f), 1.f));
}
inline __host__ __device__ __forceinline__ float length2(float3 v) {
  return dot(v, v);
}
inline __host__ __device__ __forceinline__ float length2(float2 v) {
  return dot(v, v);
}
inline __host__ __device__ __forceinline__ void gramSchmidt(float3 n, float3* a,
                                                            float3* b) {
  assert(a && b && abs(length(n) - 1.f) < 1e-5f);
  if (fabsf(n.x - n.y) > 1e-3f || fabsf(n.x - n.z) > 1e-3f)
    *a = {n.z - n.y, n.x - n.z, n.y - n.x};  //(1,1,1)x N
  else
    *a = {n.z - n.y, n.x + n.z, -n.y - n.x};  //(-1,1,1)x N

  *a = normalize(*a);
  *b = cross(n, *a);
}
inline __host__ __device__ __forceinline__ void orthonormalTangent(
    const float3 n, float3 const t, float3* a, float3* b) {
  *b = normalize(cross(n, t));
  *a = cross(*b, n);
}
inline __host__ __device__ __forceinline__ float2
cartesianFromPolar(float rho, float phi) {
  return {rho * cosf(phi), rho * sinf(phi)};
}

inline __host__ __device__ __forceinline__ float smoothstep(float x) {
  if (x <= 0.f) return 0.f;
  if (x >= 1.f) return 1.f;
  float const x2 = x * x;
  return 3.f * x2 - 2.f * x2 * x;
}
inline __host__ __device__ __forceinline__ float smoothstep(float a, float b,
                                                            float x) {
  float const t = fmaxf(fminf((x - a) / (b - a), 0.f), 1.f);

  return smoothstep(t);
}
inline __host__ __device__ __forceinline__ float sin_from_cos(
    float const cosine) {
  return safeSqrt(1.f - sqrf(cosine));
}

inline __host__ __device__ __forceinline__ uint32_t nextPow2(uint32_t x) {
  if (x == 0) return 1;
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}

#endif