#include "dmt/core/math/vec3.h"

namespace dmt::impl {
__device__ Vec3f Vector3Ops<float, CudaTag>::add(Vec3f a, Vec3f b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__device__ Vec3f Vector3Ops<float, CudaTag>::add(Vec3f a, float s) {
  return {a.x + s, a.y + s, a.z + s};
}
__device__ void Vector3Ops<float, CudaTag>::addTo(Vec3f& target, Vec3f src) {
  target.x += src.x;
  target.y += src.y;
  target.z += src.z;
}
__device__ void Vector3Ops<float, CudaTag>::addTo(Vec3f& target, float s) {
  target.x += s;
  target.y += s;
  target.z += s;
}
__device__ Vec3f Vector3Ops<float, CudaTag>::sub(Vec3f a, Vec3f b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__device__ Vec3f Vector3Ops<float, CudaTag>::sub(Vec3f a, float s) {
  return {a.x - s, a.y - s, a.z - s};
}
__device__ void Vector3Ops<float, CudaTag>::subTo(Vec3f& target, Vec3f src) {
  target.x -= src.x;
  target.y -= src.y;
  target.z -= src.z;
}
__device__ void Vector3Ops<float, CudaTag>::subTo(Vec3f& target, float s) {
  target.x -= s;
  target.y -= s;
  target.z -= s;
}
__device__ Vec3f Vector3Ops<float, CudaTag>::mul(Vec3f a, Vec3f b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}
__device__ Vec3f Vector3Ops<float, CudaTag>::mul(Vec3f a, float s) {
  return {a.x * s, a.y * s, a.z * s};
}
__device__ void Vector3Ops<float, CudaTag>::mulTo(Vec3f& target, Vec3f src) {
  target.x *= src.x;
  target.y *= src.y;
  target.z *= src.z;
}
__device__ void Vector3Ops<float, CudaTag>::mulTo(Vec3f& target, float s) {
  target.x *= s;
  target.y *= s;
  target.z *= s;
}
__device__ Vec3f Vector3Ops<float, CudaTag>::div(Vec3f a, Vec3f b) {
  return {a.x / b.x, a.y / b.y, a.z / b.z};
}
__device__ Vec3f Vector3Ops<float, CudaTag>::div(Vec3f a, float s) {
  return {a.x / s, a.y / s, a.z / s};
}
__device__ void Vector3Ops<float, CudaTag>::divTo(Vec3f& target, Vec3f src) {
  target.x /= src.x;
  target.y /= src.y;
  target.z /= src.z;
}
__device__ void Vector3Ops<float, CudaTag>::divTo(Vec3f& target, float s) {
  target.x /= s;
  target.y /= s;
  target.z /= s;
}
__device__ bool Vector3Ops<float, CudaTag>::eq(Vec3f a, Vec3f b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
__device__ bool Vector3Ops<float, CudaTag>::epsilonEq(Vec3f a, Vec3f b,
                                                      float tol) {
  a = abs(sub(a, b));
  tol = fabsf(tol);
  return a.x <= tol && a.y <= tol && a.z <= tol;
}
__device__ Vec3f Vector3Ops<float, CudaTag>::normalize(Vec3f a) {
  float const invMag = rsqrtf(dot(a, a));
  return mul(a, invMag);
}
__device__ Vec3f Vector3Ops<float, CudaTag>::abs(Vec3f a) {
  return {fabsf(a.x), fabsf(a.y), fabsf(a.z)};
}
__device__ Vec3f Vector3Ops<float, CudaTag>::ceil(Vec3f a) {
  return {ceilf(a.x), ceilf(a.y), ceilf(a.z)};
}
__device__ Vec3f Vector3Ops<float, CudaTag>::floor(Vec3f a) {
  return {floorf(a.x), floorf(a.y), floorf(a.z)};
}
__device__ Vec3f Vector3Ops<float, CudaTag>::sqrt(Vec3f a) {
  return {sqrtf(a.x), sqrtf(a.y), sqrtf(a.z)};
}
__device__ Vec3f Vector3Ops<float, CudaTag>::fma(Vec3f mult0, Vec3f mult1,
                                                 Vec3f add) {
  return {
      fmaf(mult0.x, mult1.x, add.x),
      fmaf(mult0.y, mult1.y, add.y),
      fmaf(mult0.z, mult1.z, add.z),
  };
}
__device__ Vec3f Vector3Ops<float, CudaTag>::min(Vec3f a, Vec3f b) {
  return {
      fminf(a.x, b.x),
      fminf(a.y, b.y),
      fminf(a.z, b.z),
  };
}
__device__ Vec3f Vector3Ops<float, CudaTag>::max(Vec3f a, Vec3f s) {
  return {
      fmaxf(a.x, s.x),
      fmaxf(a.y, s.y),
      fmaxf(a.z, s.z),
  };
}
__device__ float Vector3Ops<float, CudaTag>::dot(Vec3f a, Vec3f b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ Vec3i Vector3Ops<int32_t, CudaTag>::add(Vec3i a, Vec3i b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__device__ Vec3i Vector3Ops<int32_t, CudaTag>::add(Vec3i a, int32_t s) {
  return {a.x + s, a.y + s, a.z + s};
}
__device__ void Vector3Ops<int32_t, CudaTag>::addTo(Vec3i& target, Vec3i src) {
  target.x += src.x;
  target.y += src.y;
  target.z += src.z;
}
__device__ void Vector3Ops<int32_t, CudaTag>::addTo(Vec3i& target, int32_t s) {
  target.x += s;
  target.y += s;
  target.z += s;
}
__device__ Vec3i Vector3Ops<int32_t, CudaTag>::sub(Vec3i a, Vec3i b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__device__ Vec3i Vector3Ops<int32_t, CudaTag>::sub(Vec3i a, int32_t s) {
  return {a.x - s, a.y - s, a.z - s};
}
__device__ void Vector3Ops<int32_t, CudaTag>::subTo(Vec3i& target, Vec3i src) {
  target.x -= src.x;
  target.y -= src.y;
  target.z -= src.z;
}
__device__ void Vector3Ops<int32_t, CudaTag>::subTo(Vec3i& target, int32_t s) {
  target.x -= s;
  target.y -= s;
  target.z -= s;
}
__device__ Vec3i Vector3Ops<int32_t, CudaTag>::mul(Vec3i a, Vec3i b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}
__device__ Vec3i Vector3Ops<int32_t, CudaTag>::mul(Vec3i a, int32_t s) {
  return {a.x * s, a.y * s, a.z * s};
}
__device__ void Vector3Ops<int32_t, CudaTag>::mulTo(Vec3i& target, Vec3i src) {
  target.x *= src.x;
  target.y *= src.y;
  target.z *= src.z;
}
__device__ void Vector3Ops<int32_t, CudaTag>::mulTo(Vec3i& target, int32_t s) {
  target.x *= s;
  target.y *= s;
  target.z *= s;
}
__device__ Vec3i Vector3Ops<int32_t, CudaTag>::div(Vec3i a, Vec3i b) {
  return {a.x / b.x, a.y / b.y, a.z / b.z};
}
__device__ Vec3i Vector3Ops<int32_t, CudaTag>::div(Vec3i a, int32_t s) {
  return {a.x / s, a.y / s, a.z / s};
}
__device__ void Vector3Ops<int32_t, CudaTag>::divTo(Vec3i& target, Vec3i src) {
  target.x /= src.x;
  target.y /= src.y;
  target.z /= src.z;
}
__device__ void Vector3Ops<int32_t, CudaTag>::divTo(Vec3i& target, int32_t s) {
  target.x /= s;
  target.y /= s;
  target.z /= s;
}
__device__ bool Vector3Ops<int32_t, CudaTag>::eq(Vec3i a, Vec3i b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
__device__ Vec3i Vector3Ops<int32_t, CudaTag>::abs(Vec3i a) {
  return {::abs(a.x), ::abs(a.y), ::abs(a.z)};
}
__device__ Vec3i Vector3Ops<int32_t, CudaTag>::min(Vec3i a, Vec3i b) {
  return {::min(a.x, b.x), ::min(a.y, b.y), ::min(a.z, b.z)};
}
__device__ Vec3i Vector3Ops<int32_t, CudaTag>::max(Vec3i a, Vec3i s) {
  return {::max(a.x, s.x), ::max(a.y, s.y), ::max(a.z, s.z)};
}
__device__ int32_t Vector3Ops<int32_t, CudaTag>::dot(Vec3i a, Vec3i b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

}  // namespace dmt::impl
