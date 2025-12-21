#include "dmt/core/math/vec4.h"

#include "_vec_sse.h"

#include <cstdint>
#include <array>

using farr = std::array<float, 4>;
using iarr = std::array<int32_t, 4>;

namespace dmt::impl {

Vec4f Vector4Ops<float, CpuTag>::add(Vec4f a, Vec4f b) {
  Vec4f result{};
  arch::add4_f32(&result.x, &a.x, &b.x);
  return result;
}
Vec4f Vector4Ops<float, CpuTag>::add(Vec4f a, float s) {
  Vec4f result{};
  farr const b{s, s, s, s};
  arch::add4_f32(&result.x, &a.x, b.data());
  return result;
}
void Vector4Ops<float, CpuTag>::addTo(Vec4f& target, Vec4f src) {
  target = add(target, src);
}
void Vector4Ops<float, CpuTag>::addTo(Vec4f& target, float s) {
  target = add(target, s);
}
Vec4f Vector4Ops<float, CpuTag>::sub(Vec4f a, Vec4f b) {
  Vec4f result{};
  arch::sub4_f32(&result.x, &a.x, &b.x);
  return result;
}
Vec4f Vector4Ops<float, CpuTag>::sub(Vec4f a, float s) {
  Vec4f result{};
  farr const b{s, s, s, s};
  arch::sub4_f32(&result.x, &a.x, b.data());
  return result;
}
void Vector4Ops<float, CpuTag>::subTo(Vec4f& target, Vec4f src) {
  target = sub(target, src);
}
void Vector4Ops<float, CpuTag>::subTo(Vec4f& target, float s) {
  target = sub(target, s);
}
Vec4f Vector4Ops<float, CpuTag>::mul(Vec4f a, Vec4f b) {
  Vec4f result{};
  arch::mul4_f32(&result.x, &a.x, &b.x);
  return result;
}
Vec4f Vector4Ops<float, CpuTag>::mul(Vec4f a, float s) {
  Vec4f result{};
  farr const b{s, s, s, s};
  arch::mul4_f32(&result.x, &a.x, b.data());
  return result;
}
void Vector4Ops<float, CpuTag>::mulTo(Vec4f& target, Vec4f src) {
  target = mul(target, src);
}
void Vector4Ops<float, CpuTag>::mulTo(Vec4f& target, float s) {
  target = mul(target, s);
}
Vec4f Vector4Ops<float, CpuTag>::div(Vec4f a, Vec4f b) {
  Vec4f result{};
  arch::div4_f32(&result.x, &a.x, &b.x);
  return result;
}
Vec4f Vector4Ops<float, CpuTag>::div(Vec4f a, float s) {
  Vec4f result{};
  farr const b{s, s, s, s};
  arch::div4_f32(&result.x, &a.x, b.data());
  return result;
}
void Vector4Ops<float, CpuTag>::divTo(Vec4f& target, Vec4f src) {
  target = div(target, src);
}
void Vector4Ops<float, CpuTag>::divTo(Vec4f& target, float s) {
  target = div(target, s);
}
bool Vector4Ops<float, CpuTag>::eq(Vec4f a, Vec4f b) {
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
bool Vector4Ops<float, CpuTag>::epsilonEq(Vec4f a, Vec4f b, float tol) {
  a = abs(sub(a, b));
  tol = fabsf(tol);
  return a.x <= tol && a.y <= tol && a.z <= tol && a.w <= tol;
}
Vec4f Vector4Ops<float, CpuTag>::normalize(Vec4f a) {
  Vec4f result{};
  arch::normalize4_f32(&result.x, &a.x);
  return result;
}
Vec4f Vector4Ops<float, CpuTag>::abs(Vec4f a) {
  Vec4f result{};
  arch::abs4_f32(&result.x, &a.x);
  return result;
}
Vec4f Vector4Ops<float, CpuTag>::ceil(Vec4f a) {
  Vec4f result{};
  arch::ceil4_f32(&result.x, &a.x);
  return result;
}
Vec4f Vector4Ops<float, CpuTag>::floor(Vec4f a) {
  Vec4f result{};
  arch::floor4_f32(&result.x, &a.x);
  return result;
}
Vec4f Vector4Ops<float, CpuTag>::sqrt(Vec4f a) {
  Vec4f result{};
  arch::sqrt4_f32(&result.x, &a.x);
  return result;
}
Vec4f Vector4Ops<float, CpuTag>::fma(Vec4f mult0, Vec4f mult1, Vec4f add) {
  Vec4f result{};
  arch::fma4_f32(&result.x, &mult0.x, &mult1.x, &add.x);
  return result;
}
Vec4f Vector4Ops<float, CpuTag>::min(Vec4f a, Vec4f b) {
  Vec4f result{};
  arch::min4_f32(&result.x, &a.x, &b.x);
  return result;
}
Vec4f Vector4Ops<float, CpuTag>::max(Vec4f a, Vec4f s) {
  Vec4f result{};
  arch::max4_f32(&result.x, &a.x, &s.x);
  return result;
}
float Vector4Ops<float, CpuTag>::dot(Vec4f a, Vec4f b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

Vec4i Vector4Ops<int32_t, CpuTag>::add(Vec4i a, Vec4i b) {
  arch::add4_i32(&a.x, &a.x, &b.x);
  return a;
}
Vec4i Vector4Ops<int32_t, CpuTag>::add(Vec4i a, int32_t s) {
  iarr const b{s, s, s, s};
  arch::add4_i32(&a.x, &a.x, b.data());
  return a;
}
void Vector4Ops<int32_t, CpuTag>::addTo(Vec4i& target, Vec4i src) {
  target = add(target, src);
}
void Vector4Ops<int32_t, CpuTag>::addTo(Vec4i& target, int32_t s) {
  target = add(target, s);
}
Vec4i Vector4Ops<int32_t, CpuTag>::sub(Vec4i a, Vec4i b) {
  arch::sub4_i32(&a.x, &a.x, &b.x);
  return a;
}
Vec4i Vector4Ops<int32_t, CpuTag>::sub(Vec4i a, int32_t s) {
  iarr const b{s, s, s, s};
  arch::sub4_i32(&a.x, &a.x, b.data());
  return a;
}
void Vector4Ops<int32_t, CpuTag>::subTo(Vec4i& target, Vec4i src) {
  target = sub(target, src);
}
void Vector4Ops<int32_t, CpuTag>::subTo(Vec4i& target, int32_t s) {
  target = sub(target, s);
}
Vec4i Vector4Ops<int32_t, CpuTag>::mul(Vec4i a, Vec4i b) {
  arch::mul4_i32(&a.x, &a.x, &b.x);
  return a;
}
Vec4i Vector4Ops<int32_t, CpuTag>::mul(Vec4i a, int32_t s) {
  iarr const b{s, s, s, s};
  arch::mul4_i32(&a.x, &a.x, b.data());
  return a;
}
void Vector4Ops<int32_t, CpuTag>::mulTo(Vec4i& target, Vec4i src) {
  target = mul(target, src);
}
void Vector4Ops<int32_t, CpuTag>::mulTo(Vec4i& target, int32_t s) {
  target = mul(target, s);
}
Vec4i Vector4Ops<int32_t, CpuTag>::div(Vec4i a, Vec4i b) {
  arch::div4_i32(&a.x, &a.x, &b.x);
  return a;
}
Vec4i Vector4Ops<int32_t, CpuTag>::div(Vec4i a, int32_t s) {
  iarr const b{s, s, s, s};
  arch::div4_i32(&a.x, &a.x, b.data());
  return a;
}
void Vector4Ops<int32_t, CpuTag>::divTo(Vec4i& target, Vec4i src) {
  target = div(target, src);
}
void Vector4Ops<int32_t, CpuTag>::divTo(Vec4i& target, int32_t s) {
  target = div(target, s);
}
bool Vector4Ops<int32_t, CpuTag>::eq(Vec4i a, Vec4i b) {
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}
Vec4i Vector4Ops<int32_t, CpuTag>::abs(Vec4i a) {
  arch::abs4_i32(&a.x, &a.x);
  return a;
}
Vec4i Vector4Ops<int32_t, CpuTag>::min(Vec4i a, Vec4i b) {
  arch::min4_i32(&a.x, &a.x, &b.x);
  return a;
}
Vec4i Vector4Ops<int32_t, CpuTag>::max(Vec4i a, Vec4i s) {
  arch::max4_i32(&a.x, &a.x, &s.x);
  return a;
}
int32_t Vector4Ops<int32_t, CpuTag>::dot(Vec4i a, Vec4i b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
}  // namespace dmt::impl