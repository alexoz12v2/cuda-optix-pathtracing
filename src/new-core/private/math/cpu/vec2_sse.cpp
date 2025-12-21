#include "dmt/core/math/vec2.h"

#include "_vec_sse.h"

#include <cstdint>
#include <array>

using farr = std::array<float, 4>;
using iarr = std::array<int32_t, 4>;

namespace dmt::impl {

Vec2f Vector2Ops<float, CpuTag>::add(Vec2f a, Vec2f b) {
  farr a4{a.x, a.y, 0, 0};
  farr const b4{b.x, b.y, 0, 0};
  arch::add4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
Vec2f Vector2Ops<float, CpuTag>::add(Vec2f a, float s) {
  farr a4{a.x, a.y, 0, 0};
  farr const b4{s, s, 0, 0};
  arch::add4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
void Vector2Ops<float, CpuTag>::addTo(Vec2f& target, Vec2f src) {
  target = add(target, src);
}
void Vector2Ops<float, CpuTag>::addTo(Vec2f& target, float s) {
  target = add(target, s);
}
Vec2f Vector2Ops<float, CpuTag>::sub(Vec2f a, Vec2f b) {
  farr a4{a.x, a.y, 0, 0};
  farr const b4{b.x, b.y, 0, 0};
  arch::sub4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
Vec2f Vector2Ops<float, CpuTag>::sub(Vec2f a, float s) {
  farr a4{a.x, a.y, 0, 0};
  farr const b4{s, s, 0, 0};
  arch::sub4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
void Vector2Ops<float, CpuTag>::subTo(Vec2f& target, Vec2f src) {
  target = sub(target, src);
}
void Vector2Ops<float, CpuTag>::subTo(Vec2f& target, float s) {
  target = sub(target, s);
}
Vec2f Vector2Ops<float, CpuTag>::mul(Vec2f a, Vec2f b) {
  farr a4{a.x, a.y, 0, 0};
  farr const b4{b.x, b.y, 0, 0};
  arch::mul4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
Vec2f Vector2Ops<float, CpuTag>::mul(Vec2f a, float s) {
  farr a4{a.x, a.y, 0, 0};
  farr const b4{s, s, 0, 0};
  arch::mul4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
void Vector2Ops<float, CpuTag>::mulTo(Vec2f& target, Vec2f src) {
  target = mul(target, src);
}
void Vector2Ops<float, CpuTag>::mulTo(Vec2f& target, float s) {
  target = mul(target, s);
}
Vec2f Vector2Ops<float, CpuTag>::div(Vec2f a, Vec2f b) {
  farr a4{a.x, a.y, 0, 0};
  farr const b4{b.x, b.y, 0, 0};
  arch::div4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
Vec2f Vector2Ops<float, CpuTag>::div(Vec2f a, float s) {
  farr a4{a.x, a.y, 0, 0};
  farr const b4{s, s, 0, 0};
  arch::div4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
void Vector2Ops<float, CpuTag>::divTo(Vec2f& target, Vec2f src) {
  target = div(target, src);
}
void Vector2Ops<float, CpuTag>::divTo(Vec2f& target, float s) {
  target = div(target, s);
}
bool Vector2Ops<float, CpuTag>::eq(Vec2f a, Vec2f b) {
  return a.x == b.x && a.y == b.y;
}
bool Vector2Ops<float, CpuTag>::epsilonEq(Vec2f a, Vec2f b, float tol) {
  a = abs(sub(a, b));
  tol = fabsf(tol);
  return a.x <= tol && a.y <= tol;
}
Vec2f Vector2Ops<float, CpuTag>::normalize(Vec2f a) {
  farr a4{a.x, a.y, 0, 0};
  arch::normalize4_f32(a4.data(), a4.data());
  return {a4[0], a4[1]};
}
Vec2f Vector2Ops<float, CpuTag>::abs(Vec2f a) {
  farr a4{a.x, a.y, 0, 0};
  arch::abs4_f32(a4.data(), a4.data());
  return {a4[0], a4[1]};
}
Vec2f Vector2Ops<float, CpuTag>::ceil(Vec2f a) {
  farr a4{a.x, a.y, 0, 0};
  arch::ceil4_f32(a4.data(), a4.data());
  return {a4[0], a4[1]};
}
Vec2f Vector2Ops<float, CpuTag>::floor(Vec2f a) {
  farr a4{a.x, a.y, 0, 0};
  arch::floor4_f32(a4.data(), a4.data());
  return {a4[0], a4[1]};
}
Vec2f Vector2Ops<float, CpuTag>::sqrt(Vec2f a) {
  farr a4{a.x, a.y, 0, 0};
  arch::sqrt4_f32(a4.data(), a4.data());
  return {a4[0], a4[1]};
}
Vec2f Vector2Ops<float, CpuTag>::fma(Vec2f mult0, Vec2f mult1, Vec2f add) {
  farr mult04{mult0.x, mult0.y, 0, 0};
  farr const mult14{mult1.x, mult1.y, 0, 0};
  farr const add4{add.x, add.y, 0, 0};
  arch::fma4_f32(mult04.data(), mult04.data(), mult14.data(), add4.data());
  return {mult04[0], mult04[1]};
}
Vec2f Vector2Ops<float, CpuTag>::min(Vec2f a, Vec2f b) {
  farr a4{a.x, a.y, 0, 0};
  farr const b4{b.x, b.y, 0, 0};
  arch::min4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
Vec2f Vector2Ops<float, CpuTag>::max(Vec2f a, Vec2f s) {
  farr a4{a.x, a.y, 0, 0};
  farr const b4{s.x, s.y, 0, 0};
  arch::max4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
float Vector2Ops<float, CpuTag>::dot(Vec2f a, Vec2f b) {
  return a.x * b.x + a.y * b.y;
}

Vec2i Vector2Ops<int32_t, CpuTag>::add(Vec2i a, Vec2i b) {
  iarr a4{a.x, a.y, 0, 0};
  iarr const b4{b.x, b.y, 0, 0};
  arch::add4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
Vec2i Vector2Ops<int32_t, CpuTag>::add(Vec2i a, int32_t s) {
  iarr a4{a.x, a.y, 0, 0};
  iarr const b4{s, s, 0, 0};
  arch::add4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
void Vector2Ops<int32_t, CpuTag>::addTo(Vec2i& target, Vec2i src) {
  target = add(target, src);
}
void Vector2Ops<int32_t, CpuTag>::addTo(Vec2i& target, int32_t s) {
  target = add(target, s);
}
Vec2i Vector2Ops<int32_t, CpuTag>::sub(Vec2i a, Vec2i b) {
  iarr a4{a.x, a.y, 0, 0};
  iarr const b4{b.x, b.y, 0, 0};
  arch::sub4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
Vec2i Vector2Ops<int32_t, CpuTag>::sub(Vec2i a, int32_t s) {
  iarr a4{a.x, a.y, 0, 0};
  iarr const b4{s, s, 0, 0};
  arch::sub4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
void Vector2Ops<int32_t, CpuTag>::subTo(Vec2i& target, Vec2i src) {
  target = sub(target, src);
}
void Vector2Ops<int32_t, CpuTag>::subTo(Vec2i& target, int32_t s) {
  target = sub(target, s);
}
Vec2i Vector2Ops<int32_t, CpuTag>::mul(Vec2i a, Vec2i b) {
  iarr a4{a.x, a.y, 0, 0};
  iarr const b4{b.x, b.y, 0, 0};
  arch::mul4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
Vec2i Vector2Ops<int32_t, CpuTag>::mul(Vec2i a, int32_t s) {
  iarr a4{a.x, a.y, 0, 0};
  iarr const b4{s, s, 0, 0};
  arch::mul4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
void Vector2Ops<int32_t, CpuTag>::mulTo(Vec2i& target, Vec2i src) {
  target = mul(target, src);
}
void Vector2Ops<int32_t, CpuTag>::mulTo(Vec2i& target, int32_t s) {
  target = mul(target, s);
}
Vec2i Vector2Ops<int32_t, CpuTag>::div(Vec2i a, Vec2i b) {
  iarr a4{a.x, a.y, 0, 0};
  iarr const b4{b.x, b.y, 0, 0};
  arch::div4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
Vec2i Vector2Ops<int32_t, CpuTag>::div(Vec2i a, int32_t s) {
  iarr a4{a.x, a.y, 0, 0};
  iarr const b{s, s, 0, 0};
  arch::div4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
void Vector2Ops<int32_t, CpuTag>::divTo(Vec2i& target, Vec2i src) {
  target = div(target, src);
}
void Vector2Ops<int32_t, CpuTag>::divTo(Vec2i& target, int32_t s) {
  target = div(target, s);
}
bool Vector2Ops<int32_t, CpuTag>::eq(Vec2i a, Vec2i b) {
  return a.x == b.x && a.y == b.y;
}
Vec2i Vector2Ops<int32_t, CpuTag>::abs(Vec2i a) {
  iarr a4{a.x, a.y, 0, 0};
  arch::abs4_i32(a4.data(), a4.data());
  return {a4[0], a4[1]};
}
Vec2i Vector2Ops<int32_t, CpuTag>::min(Vec2i a, Vec2i b) {
  iarr a4{a.x, a.y, 0, 0};
  iarr const b4{b.x, b.y, 0, 0};
  arch::min4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
Vec2i Vector2Ops<int32_t, CpuTag>::max(Vec2i a, Vec2i s) {
  iarr a4{a.x, a.y, 0, 0};
  iarr const b4{s.x, s.y, 0, 0};
  arch::max4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1]};
}
int32_t Vector2Ops<int32_t, CpuTag>::dot(Vec2i a, Vec2i b) {
  return a.x * b.x + a.y * b.y;
}
}  // namespace dmt::impl