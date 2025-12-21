#include "dmt/core/math/vec2.h"

namespace dmt::impl {
__device__ Vec2f Vector2Ops<float, CudaTag>::add(Vec2f a, Vec2f b) {
  return {a.x + b.x, a.y + b.y};
}
__device__ Vec2f Vector2Ops<float, CudaTag>::add(Vec2f a, float s) {
  return {a.x + s, a.y + s};
}
__device__ void Vector2Ops<float, CudaTag>::addTo(Vec2f& target, Vec2f src) {
  target.x += src.x;
  target.y += src.y;
}
__device__ void Vector2Ops<float, CudaTag>::addTo(Vec2f& target, float s) {
  target.x += s;
  target.y += s;
}
__device__ Vec2f Vector2Ops<float, CudaTag>::sub(Vec2f a, Vec2f b) {
  return {a.x - b.x, a.y - b.y};
}
__device__ Vec2f Vector2Ops<float, CudaTag>::sub(Vec2f a, float s) {
  return {a.x - s, a.y - s};
}
__device__ void Vector2Ops<float, CudaTag>::subTo(Vec2f& target, Vec2f src) {
  target.x -= src.x;
  target.y -= src.y;
}
__device__ void Vector2Ops<float, CudaTag>::subTo(Vec2f& target, float s) {
  target.x -= s;
  target.y -= s;
}
__device__ Vec2f Vector2Ops<float, CudaTag>::mul(Vec2f a, Vec2f b) {
  return {a.x * b.x, a.y * b.y};
}
__device__ Vec2f Vector2Ops<float, CudaTag>::mul(Vec2f a, float s) {
  return {a.x * s, a.y * s};
}
__device__ void Vector2Ops<float, CudaTag>::mulTo(Vec2f& target, Vec2f src) {
  target.x *= src.x;
  target.y *= src.y;
}
__device__ void Vector2Ops<float, CudaTag>::mulTo(Vec2f& target, float s) {
  target.x *= s;
  target.y *= s;
}
__device__ Vec2f Vector2Ops<float, CudaTag>::div(Vec2f a, Vec2f b) {
  return {a.x / b.x, a.y / b.y};
}
__device__ Vec2f Vector2Ops<float, CudaTag>::div(Vec2f a, float s) {
  return {a.x / s, a.y / s};
}
__device__ void Vector2Ops<float, CudaTag>::divTo(Vec2f& target, Vec2f src) {
  target.x /= src.x;
  target.y /= src.y;
}
__device__ void Vector2Ops<float, CudaTag>::divTo(Vec2f& target, float s) {
  target.x /= s;
  target.y /= s;
}
__device__ bool Vector2Ops<float, CudaTag>::eq(Vec2f a, Vec2f b) {
  return a.x == b.x && a.y == b.y;
}
__device__ bool Vector2Ops<float, CudaTag>::epsilonEq(Vec2f a, Vec2f b,
                                                      float tol) {
  a = abs(sub(a, b));
  tol = fabsf(tol);
  return a.x <= tol && a.y <= tol;
}
__device__ Vec2f Vector2Ops<float, CudaTag>::normalize(Vec2f a) {
  float const invMag = rsqrtf(dot(a, a));
  return mul(a, invMag);
}
__device__ Vec2f Vector2Ops<float, CudaTag>::abs(Vec2f a) {
  return {fabsf(a.x), fabsf(a.y)};
}
__device__ Vec2f Vector2Ops<float, CudaTag>::ceil(Vec2f a) {
  return {ceilf(a.x), ceilf(a.y)};
}
__device__ Vec2f Vector2Ops<float, CudaTag>::floor(Vec2f a) {
  return {floorf(a.x), floorf(a.y)};
}
__device__ Vec2f Vector2Ops<float, CudaTag>::sqrt(Vec2f a) {
  return {sqrtf(a.x), sqrtf(a.y)};
}
__device__ Vec2f Vector2Ops<float, CudaTag>::fma(Vec2f mult0, Vec2f mult1,
                                                 Vec2f add) {
  return {
      fmaf(mult0.x, mult1.x, add.x),
      fmaf(mult0.y, mult1.y, add.y),
  };
}
__device__ Vec2f Vector2Ops<float, CudaTag>::min(Vec2f a, Vec2f b) {
  return {
      fminf(a.x, b.x),
      fminf(a.y, b.y),
  };
}
__device__ Vec2f Vector2Ops<float, CudaTag>::max(Vec2f a, Vec2f s) {
  return {
      fmaxf(a.x, s.x),
      fmaxf(a.y, s.y),
  };
}
__device__ float Vector2Ops<float, CudaTag>::dot(Vec2f a, Vec2f b) {
  return a.x * b.x + a.y * b.y;
}

__device__ Vec2i Vector2Ops<int32_t, CudaTag>::add(Vec2i a, Vec2i b) {
  return {a.x + b.x, a.y + b.y};
}
__device__ Vec2i Vector2Ops<int32_t, CudaTag>::add(Vec2i a, int32_t s) {
  return {a.x + s, a.y + s};
}
__device__ void Vector2Ops<int32_t, CudaTag>::addTo(Vec2i& target, Vec2i src) {
  target.x += src.x;
  target.y += src.y;
}
__device__ void Vector2Ops<int32_t, CudaTag>::addTo(Vec2i& target, int32_t s) {
  target.x += s;
  target.y += s;
}
__device__ Vec2i Vector2Ops<int32_t, CudaTag>::sub(Vec2i a, Vec2i b) {
  return {a.x - b.x, a.y - b.y};
}
__device__ Vec2i Vector2Ops<int32_t, CudaTag>::sub(Vec2i a, int32_t s) {
  return {a.x - s, a.y - s};
}
__device__ void Vector2Ops<int32_t, CudaTag>::subTo(Vec2i& target, Vec2i src) {
  target.x -= src.x;
  target.y -= src.y;
}
__device__ void Vector2Ops<int32_t, CudaTag>::subTo(Vec2i& target, int32_t s) {
  target.x -= s;
  target.y -= s;
}
__device__ Vec2i Vector2Ops<int32_t, CudaTag>::mul(Vec2i a, Vec2i b) {
  return {a.x * b.x, a.y * b.y};
}
__device__ Vec2i Vector2Ops<int32_t, CudaTag>::mul(Vec2i a, int32_t s) {
  return {a.x * s, a.y * s};
}
__device__ void Vector2Ops<int32_t, CudaTag>::mulTo(Vec2i& target, Vec2i src) {
  target.x *= src.x;
  target.y *= src.y;
}
__device__ void Vector2Ops<int32_t, CudaTag>::mulTo(Vec2i& target, int32_t s) {
  target.x *= s;
  target.y *= s;
}
__device__ Vec2i Vector2Ops<int32_t, CudaTag>::div(Vec2i a, Vec2i b) {
  return {a.x / b.x, a.y / b.y};
}
__device__ Vec2i Vector2Ops<int32_t, CudaTag>::div(Vec2i a, int32_t s) {
  return {a.x / s, a.y / s};
}
__device__ void Vector2Ops<int32_t, CudaTag>::divTo(Vec2i& target, Vec2i src) {
  target.x /= src.x;
  target.y /= src.y;
}
__device__ void Vector2Ops<int32_t, CudaTag>::divTo(Vec2i& target, int32_t s) {
  target.x /= s;
  target.y /= s;
}
__device__ bool Vector2Ops<int32_t, CudaTag>::eq(Vec2i a, Vec2i b) {
  return a.x == b.x && a.y == b.y;
}
__device__ Vec2i Vector2Ops<int32_t, CudaTag>::abs(Vec2i a) {
  return {::abs(a.x), ::abs(a.y)};
}
__device__ Vec2i Vector2Ops<int32_t, CudaTag>::min(Vec2i a, Vec2i b) {
  return {::min(a.x, b.x), ::min(a.y, b.y)};
}
__device__ Vec2i Vector2Ops<int32_t, CudaTag>::max(Vec2i a, Vec2i s) {
  return {::max(a.x, s.x), ::max(a.y, s.y)};
}
__device__ int32_t Vector2Ops<int32_t, CudaTag>::dot(Vec2i a, Vec2i b) {
  return a.x * b.x + a.y * b.y;
}

}  // namespace dmt::impl
