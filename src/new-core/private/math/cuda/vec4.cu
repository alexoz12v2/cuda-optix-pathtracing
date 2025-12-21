#include "dmt/core/math/vec4.h"

namespace dmt::impl {
__device__ Vec4f Vector4Ops<float, CudaTag>::add(Vec4f a, Vec4f b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
__device__ Vec4f Vector4Ops<float, CudaTag>::add(Vec4f a, float s) {
  return {a.x + s, a.y + s, a.z + s, a.w + s};
}
__device__ void Vector4Ops<float, CudaTag>::addTo(Vec4f& target, Vec4f src) {
  target.x += src.x;
  target.y += src.y;
  target.z += src.z;
  target.w += src.w;
}
__device__ void Vector4Ops<float, CudaTag>::addTo(Vec4f& target, float s) {
  target.x += s;
  target.y += s;
  target.z += s;
  target.w += s;
}
__device__ Vec4f Vector4Ops<float, CudaTag>::sub(Vec4f a, Vec4f b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
__device__ Vec4f Vector4Ops<float, CudaTag>::sub(Vec4f a, float s) {
  return {a.x - s, a.y - s, a.z - s, a.w - s};
}
__device__ void Vector4Ops<float, CudaTag>::subTo(Vec4f& target, Vec4f src) {
  target.x -= src.x;
  target.y -= src.y;
  target.z -= src.z;
  target.w -= src.w;
}
__device__ void Vector4Ops<float, CudaTag>::subTo(Vec4f& target, float s) {
  target.x -= s;
  target.y -= s;
  target.z -= s;
  target.w -= s;
}
__device__ Vec4f Vector4Ops<float, CudaTag>::mul(Vec4f a, Vec4f b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}
__device__ Vec4f Vector4Ops<float, CudaTag>::mul(Vec4f a, float s) {
  return {a.x * s, a.y * s, a.z * s, a.w * s};
}
__device__ void Vector4Ops<float, CudaTag>::mulTo(Vec4f& target, Vec4f src) {
  target.x *= src.x;
  target.y *= src.y;
  target.z *= src.z;
  target.w *= src.w;
}
__device__ void Vector4Ops<float, CudaTag>::mulTo(Vec4f& target, float s) {
  target.x *= s;
  target.y *= s;
  target.z *= s;
  target.w *= s;
}
__device__ Vec4f Vector4Ops<float, CudaTag>::div(Vec4f a, Vec4f b) {
  return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}
__device__ Vec4f Vector4Ops<float, CudaTag>::div(Vec4f a, float s) {
  return {a.x / s, a.y / s, a.z / s, a.w / s};
}
__device__ void Vector4Ops<float, CudaTag>::divTo(Vec4f& target, Vec4f src) {
  target.x /= src.x;
  target.y /= src.y;
  target.z /= src.z;
  target.w /= src.w;
}
__device__ void Vector4Ops<float, CudaTag>::divTo(Vec4f& target, float s) {
  target.x /= s;
  target.y /= s;
  target.z /= s;
  target.w /= s;
}
__device__ bool Vector4Ops<float, CudaTag>::eq(Vec4f a, Vec4f b) {
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
__device__ bool Vector4Ops<float, CudaTag>::epsilonEq(Vec4f a, Vec4f b,
                                                      float tol) {
  a = abs(sub(a, b));
  tol = fabsf(tol);
  return a.x <= tol && a.y <= tol && a.z <= tol && a.w <= tol;
}
__device__ Vec4f Vector4Ops<float, CudaTag>::normalize(Vec4f a) {
  float const invMag = rsqrtf(dot(a, a));
  return mul(a, invMag);
}
__device__ Vec4f Vector4Ops<float, CudaTag>::abs(Vec4f a) {
  return {fabsf(a.x), fabsf(a.y), fabsf(a.z), fabsf(a.w)};
}
__device__ Vec4f Vector4Ops<float, CudaTag>::ceil(Vec4f a) {
  return {ceilf(a.x), ceilf(a.y), ceilf(a.z), ceilf(a.w)};
}
__device__ Vec4f Vector4Ops<float, CudaTag>::floor(Vec4f a) {
  return {floorf(a.x), floorf(a.y), floorf(a.z), floorf(a.w)};
}
__device__ Vec4f Vector4Ops<float, CudaTag>::sqrt(Vec4f a) {
  return {sqrtf(a.x), sqrtf(a.y), sqrtf(a.z), sqrtf(a.w)};
}
__device__ Vec4f Vector4Ops<float, CudaTag>::fma(Vec4f mult0, Vec4f mult1,
                                                 Vec4f add) {
  return {
      fmaf(mult0.x, mult1.x, add.x),
      fmaf(mult0.y, mult1.y, add.y),
      fmaf(mult0.z, mult1.z, add.z),
      fmaf(mult0.w, mult1.w, add.w),
  };
}
__device__ Vec4f Vector4Ops<float, CudaTag>::min(Vec4f a, Vec4f b) {
  return {
      fminf(a.x, b.x),
      fminf(a.y, b.y),
      fminf(a.z, b.z),
      fminf(a.w, b.w),
  };
}
__device__ Vec4f Vector4Ops<float, CudaTag>::max(Vec4f a, Vec4f s) {
  return {
      fmaxf(a.x, s.x),
      fmaxf(a.y, s.y),
      fmaxf(a.z, s.z),
      fmaxf(a.w, s.w),
  };
}
__device__ float Vector4Ops<float, CudaTag>::dot(Vec4f a, Vec4f b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ Vec4i Vector4Ops<int32_t, CudaTag>::add(Vec4i a, Vec4i b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
__device__ Vec4i Vector4Ops<int32_t, CudaTag>::add(Vec4i a, int32_t s) {
  return {a.x + s, a.y + s, a.z + s, a.w + s};
}
__device__ void Vector4Ops<int32_t, CudaTag>::addTo(Vec4i& target, Vec4i src) {
  target.x += src.x;
  target.y += src.y;
  target.z += src.z;
  target.w += src.w;
}
__device__ void Vector4Ops<int32_t, CudaTag>::addTo(Vec4i& target, int32_t s) {
  target.x += s;
  target.y += s;
  target.z += s;
  target.w += s;
}
__device__ Vec4i Vector4Ops<int32_t, CudaTag>::sub(Vec4i a, Vec4i b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
__device__ Vec4i Vector4Ops<int32_t, CudaTag>::sub(Vec4i a, int32_t s) {
  return {a.x - s, a.y - s, a.z - s, a.w - s};
}
__device__ void Vector4Ops<int32_t, CudaTag>::subTo(Vec4i& target, Vec4i src) {
  target.x -= src.x;
  target.y -= src.y;
  target.z -= src.z;
  target.w -= src.w;
}
__device__ void Vector4Ops<int32_t, CudaTag>::subTo(Vec4i& target, int32_t s) {
  target.x -= s;
  target.y -= s;
  target.z -= s;
  target.w -= s;
}
__device__ Vec4i Vector4Ops<int32_t, CudaTag>::mul(Vec4i a, Vec4i b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}
__device__ Vec4i Vector4Ops<int32_t, CudaTag>::mul(Vec4i a, int32_t s) {
  return {a.x * s, a.y * s, a.z * s, a.w * s};
}
__device__ void Vector4Ops<int32_t, CudaTag>::mulTo(Vec4i& target, Vec4i src) {
  target.x *= src.x;
  target.y *= src.y;
  target.z *= src.z;
  target.w *= src.w;
}
__device__ void Vector4Ops<int32_t, CudaTag>::mulTo(Vec4i& target, int32_t s) {
  target.x *= s;
  target.y *= s;
  target.z *= s;
  target.w *= s;
}
__device__ Vec4i Vector4Ops<int32_t, CudaTag>::div(Vec4i a, Vec4i b) {
  return {a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
}
__device__ Vec4i Vector4Ops<int32_t, CudaTag>::div(Vec4i a, int32_t s) {
  return {a.x / s, a.y / s, a.z / s, a.w / s};
}
__device__ void Vector4Ops<int32_t, CudaTag>::divTo(Vec4i& target, Vec4i src) {
  target.x /= src.x;
  target.y /= src.y;
  target.z /= src.z;
  target.w /= src.w;
}
__device__ void Vector4Ops<int32_t, CudaTag>::divTo(Vec4i& target, int32_t s) {
  target.x /= s;
  target.y /= s;
  target.z /= s;
  target.w /= s;
}
__device__ bool Vector4Ops<int32_t, CudaTag>::eq(Vec4i a, Vec4i b) {
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
__device__ Vec4i Vector4Ops<int32_t, CudaTag>::abs(Vec4i a) {
  return {::abs(a.x), ::abs(a.y), ::abs(a.z), ::abs(a.w)};
}
__device__ Vec4i Vector4Ops<int32_t, CudaTag>::min(Vec4i a, Vec4i b) {
  return {::min(a.x, b.x), ::min(a.y, b.y), ::min(a.z, b.z), ::min(a.w, b.w)};
}
__device__ Vec4i Vector4Ops<int32_t, CudaTag>::max(Vec4i a, Vec4i s) {
  return {::max(a.x, s.x), ::max(a.y, s.y), ::max(a.z, s.z), ::max(a.w, s.w)};
}
__device__ int32_t Vector4Ops<int32_t, CudaTag>::dot(Vec4i a, Vec4i b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

}  // namespace dmt::impl
