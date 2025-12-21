#include "dmt/core/math/vec3.h"

#include "_vec_sse.h"

#include <cstdint>
#include <array>

using farr = std::array<float, 4>;
using iarr = std::array<int32_t, 4>;

namespace dmt::impl {

Vec3f Vector3Ops<float, CpuTag>::add(Vec3f a, Vec3f b) {
  farr a4{a.x, a.y, a.z, 0};
  farr const b4{b.x, b.y, b.z, 0};
  arch::add4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
Vec3f Vector3Ops<float, CpuTag>::add(Vec3f a, float s) {
  farr a4{a.x, a.y, a.z, 0};
  farr const b4{s, s, s, 0};
  arch::add4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
void Vector3Ops<float, CpuTag>::addTo(Vec3f& target, Vec3f src) {
  target = add(target, src);
}
void Vector3Ops<float, CpuTag>::addTo(Vec3f& target, float s) {
  target = add(target, s);
}
Vec3f Vector3Ops<float, CpuTag>::sub(Vec3f a, Vec3f b) {
  farr a4{a.x, a.y, a.z, 0};
  farr const b4{b.x, b.y, b.z, 0};
  arch::sub4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
Vec3f Vector3Ops<float, CpuTag>::sub(Vec3f a, float s) {
  farr a4{a.x, a.y, a.z, 0};
  farr const b4{s, s, s, 0};
  arch::sub4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
void Vector3Ops<float, CpuTag>::subTo(Vec3f& target, Vec3f src) {
  target = sub(target, src);
}
void Vector3Ops<float, CpuTag>::subTo(Vec3f& target, float s) {
  target = sub(target, s);
}
Vec3f Vector3Ops<float, CpuTag>::mul(Vec3f a, Vec3f b) {
  farr a4{a.x, a.y, a.z, 0};
  farr const b4{b.x, b.y, b.z, 0};
  arch::mul4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
Vec3f Vector3Ops<float, CpuTag>::mul(Vec3f a, float s) {
  farr a4{a.x, a.y, a.z, 0};
  farr const b4{s, s, s, 0};
  arch::mul4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
void Vector3Ops<float, CpuTag>::mulTo(Vec3f& target, Vec3f src) {
  target = mul(target, src);
}
void Vector3Ops<float, CpuTag>::mulTo(Vec3f& target, float s) {
  target = mul(target, s);
}
Vec3f Vector3Ops<float, CpuTag>::div(Vec3f a, Vec3f b) {
  farr a4{a.x, a.y, a.z, 0};
  farr const b4{b.x, b.y, b.z, 0};
  arch::div4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
Vec3f Vector3Ops<float, CpuTag>::div(Vec3f a, float s) {
  farr a4{a.x, a.y, a.z, 0};
  farr const b4{s, s, s, 0};
  arch::div4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
void Vector3Ops<float, CpuTag>::divTo(Vec3f& target, Vec3f src) {
  target = div(target, src);
}
void Vector3Ops<float, CpuTag>::divTo(Vec3f& target, float s) {
  target = div(target, s);
}
bool Vector3Ops<float, CpuTag>::eq(Vec3f a, Vec3f b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
bool Vector3Ops<float, CpuTag>::epsilonEq(Vec3f a, Vec3f b, float tol) {
  a = abs(sub(a, b));
  tol = fabsf(tol);
  return a.x <= tol && a.y <= tol && a.z <= tol;
}
Vec3f Vector3Ops<float, CpuTag>::normalize(Vec3f a) {
  farr a4{a.x, a.y, a.z, 0};
  arch::normalize4_f32(a4.data(), a4.data());
  return {a4[0], a4[1], a4[2]};
}
Vec3f Vector3Ops<float, CpuTag>::abs(Vec3f a) {
  farr a4{a.x, a.y, a.z, 0};
  arch::abs4_f32(a4.data(), a4.data());
  return {a4[0], a4[1], a4[2]};
}
Vec3f Vector3Ops<float, CpuTag>::ceil(Vec3f a) {
  farr a4{a.x, a.y, a.z, 0};
  arch::ceil4_f32(a4.data(), a4.data());
  return {a4[0], a4[1], a4[2]};
}
Vec3f Vector3Ops<float, CpuTag>::floor(Vec3f a) {
  farr a4{a.x, a.y, a.z, 0};
  arch::floor4_f32(a4.data(), a4.data());
  return {a4[0], a4[1], a4[2]};
}
Vec3f Vector3Ops<float, CpuTag>::sqrt(Vec3f a) {
  farr a4{a.x, a.y, a.z, 0};
  arch::sqrt4_f32(a4.data(), a4.data());
  return {a4[0], a4[1], a4[2]};
}
Vec3f Vector3Ops<float, CpuTag>::fma(Vec3f mult0, Vec3f mult1, Vec3f add) {
  farr mult04{mult0.x, mult0.y, mult0.z, 0};
  farr const mult14{mult1.x, mult1.y, mult1.z, 0};
  farr const add4{add.x, add.y, add.z, 0};
  arch::fma4_f32(mult04.data(), mult04.data(), mult14.data(), add4.data());
  return {mult04[0], mult04[1], mult04[2]};
}
Vec3f Vector3Ops<float, CpuTag>::min(Vec3f a, Vec3f b) {
  farr a4{a.x, a.y, a.z, 0};
  farr const b4{b.x, b.y, b.z, 0};
  arch::min4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
Vec3f Vector3Ops<float, CpuTag>::max(Vec3f a, Vec3f s) {
  farr a4{a.x, a.y, a.z, 0};
  farr const b4{s.x, s.y, s.z, 0};
  arch::max4_f32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
float Vector3Ops<float, CpuTag>::dot(Vec3f a, Vec3f b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3i Vector3Ops<int32_t, CpuTag>::add(Vec3i a, Vec3i b) {
  iarr a4{a.x, a.y, a.z, 0};
  iarr const b4{b.x, b.y, b.z, 0};
  arch::add4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
Vec3i Vector3Ops<int32_t, CpuTag>::add(Vec3i a, int32_t s) {
  iarr a4{a.x, a.y, a.z, 0};
  iarr const b4{s, s, s, 0};
  arch::add4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
void Vector3Ops<int32_t, CpuTag>::addTo(Vec3i& target, Vec3i src) {
  target = add(target, src);
}
void Vector3Ops<int32_t, CpuTag>::addTo(Vec3i& target, int32_t s) {
  target = add(target, s);
}
Vec3i Vector3Ops<int32_t, CpuTag>::sub(Vec3i a, Vec3i b) {
  iarr a4{a.x, a.y, a.z, 0};
  iarr const b4{b.x, b.y, b.z, 0};
  arch::sub4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
Vec3i Vector3Ops<int32_t, CpuTag>::sub(Vec3i a, int32_t s) {
  iarr a4{a.x, a.y, a.z, 0};
  iarr const b4{s, s, s, 0};
  arch::sub4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
void Vector3Ops<int32_t, CpuTag>::subTo(Vec3i& target, Vec3i src) {
  target = sub(target, src);
}
void Vector3Ops<int32_t, CpuTag>::subTo(Vec3i& target, int32_t s) {
  target = sub(target, s);
}
Vec3i Vector3Ops<int32_t, CpuTag>::mul(Vec3i a, Vec3i b) {
  iarr a4{a.x, a.y, a.z, 0};
  iarr const b4{b.x, b.y, b.z, 0};
  arch::mul4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
Vec3i Vector3Ops<int32_t, CpuTag>::mul(Vec3i a, int32_t s) {
  iarr a4{a.x, a.y, a.z, 0};
  iarr const b4{s, s, s, 0};
  arch::mul4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
void Vector3Ops<int32_t, CpuTag>::mulTo(Vec3i& target, Vec3i src) {
  target = mul(target, src);
}
void Vector3Ops<int32_t, CpuTag>::mulTo(Vec3i& target, int32_t s) {
  target = mul(target, s);
}
Vec3i Vector3Ops<int32_t, CpuTag>::div(Vec3i a, Vec3i b) {
  iarr a4{a.x, a.y, a.z, 0};
  iarr const b4{b.x, b.y, b.z, 0};
  arch::div4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
Vec3i Vector3Ops<int32_t, CpuTag>::div(Vec3i a, int32_t s) {
  iarr a4{a.x, a.y, a.z, 0};
  iarr const b{s, s, s, 0};
  arch::div4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
void Vector3Ops<int32_t, CpuTag>::divTo(Vec3i& target, Vec3i src) {
  target = div(target, src);
}
void Vector3Ops<int32_t, CpuTag>::divTo(Vec3i& target, int32_t s) {
  target = div(target, s);
}
bool Vector3Ops<int32_t, CpuTag>::eq(Vec3i a, Vec3i b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}
Vec3i Vector3Ops<int32_t, CpuTag>::abs(Vec3i a) {
  iarr a4{a.x, a.y, a.z, 0};
  arch::abs4_i32(a4.data(), a4.data());
  return {a4[0], a4[1], a4[2]};
}
Vec3i Vector3Ops<int32_t, CpuTag>::min(Vec3i a, Vec3i b) {
  iarr a4{a.x, a.y, a.z, 0};
  iarr const b4{b.x, b.y, b.z, 0};
  arch::min4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
Vec3i Vector3Ops<int32_t, CpuTag>::max(Vec3i a, Vec3i s) {
  iarr a4{a.x, a.y, a.z, 0};
  iarr const b4{s.x, s.y, s.z, 0};
  arch::max4_i32(a4.data(), a4.data(), b4.data());
  return {a4[0], a4[1], a4[2]};
}
int32_t Vector3Ops<int32_t, CpuTag>::dot(Vec3i a, Vec3i b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
}  // namespace dmt::impl
