#pragma once

#include "dmt/core/utils/macros.h"
#include "dmt/core/utils/backend_tags.h"
#include "dmt/core/math/vec_base.h"

#include <cstdint>

namespace dmt {
template <Scalar S>
struct Vec2;

using Vector4f = Vec2<float>;
using Vec2f = Vec2<float>;
using Vector4i = Vec2<int32_t>;
using Vec2i = Vec2<int32_t>;

}  // namespace dmt

// necessary boilerplate for compile time dispatch
namespace dmt::impl {

template <Scalar S>
struct CpuVector2;

template <typename Scalar, typename Backend>
struct Vector2Ops;

template <>
struct Vector2Ops<float, CpuTag> {
  static DMT_CPU Vec2f add(Vec2f a, Vec2f b);
  static DMT_CPU Vec2f add(Vec2f a, float s);
  static DMT_CPU void addTo(Vec2f& target, Vec2f src);
  static DMT_CPU void addTo(Vec2f& target, float s);
  static DMT_CPU Vec2f sub(Vec2f a, Vec2f b);
  static DMT_CPU Vec2f sub(Vec2f a, float s);
  static DMT_CPU void subTo(Vec2f& target, Vec2f src);
  static DMT_CPU void subTo(Vec2f& target, float s);
  static DMT_CPU Vec2f mul(Vec2f a, Vec2f b);
  static DMT_CPU Vec2f mul(Vec2f a, float s);
  static DMT_CPU void mulTo(Vec2f& target, Vec2f src);
  static DMT_CPU void mulTo(Vec2f& target, float s);
  static DMT_CPU Vec2f div(Vec2f a, Vec2f b);
  static DMT_CPU Vec2f div(Vec2f a, float s);
  static DMT_CPU void divTo(Vec2f& target, Vec2f src);
  static DMT_CPU void divTo(Vec2f& target, float s);
  static DMT_CPU bool eq(Vec2f a, Vec2f b);
  static DMT_CPU bool epsilonEq(Vec2f a, Vec2f b, float tol);
  static DMT_CPU Vec2f normalize(Vec2f a);
  static DMT_CPU Vec2f abs(Vec2f a);
  static DMT_CPU Vec2f ceil(Vec2f a);
  static DMT_CPU Vec2f floor(Vec2f a);
  static DMT_CPU Vec2f sqrt(Vec2f a);
  static DMT_CPU Vec2f fma(Vec2f mult0, Vec2f mult1, Vec2f add);
  static DMT_CPU Vec2f min(Vec2f a, Vec2f b);
  static DMT_CPU Vec2f max(Vec2f a, Vec2f s);
  static DMT_CPU float dot(Vec2f a, Vec2f b);
};

template <>
struct Vector2Ops<int32_t, CpuTag> {
  static DMT_CPU Vec2i add(Vec2i a, Vec2i b);
  static DMT_CPU Vec2i add(Vec2i a, int32_t s);
  static DMT_CPU void addTo(Vec2i& target, Vec2i src);
  static DMT_CPU void addTo(Vec2i& target, int32_t s);
  static DMT_CPU Vec2i sub(Vec2i a, Vec2i b);
  static DMT_CPU Vec2i sub(Vec2i a, int32_t s);
  static DMT_CPU void subTo(Vec2i& target, Vec2i src);
  static DMT_CPU void subTo(Vec2i& target, int32_t s);
  static DMT_CPU Vec2i mul(Vec2i a, Vec2i b);
  static DMT_CPU Vec2i mul(Vec2i a, int32_t s);
  static DMT_CPU void mulTo(Vec2i& target, Vec2i src);
  static DMT_CPU void mulTo(Vec2i& target, int32_t s);
  static DMT_CPU Vec2i div(Vec2i a, Vec2i b);
  static DMT_CPU Vec2i div(Vec2i a, int32_t s);
  static DMT_CPU void divTo(Vec2i& target, Vec2i src);
  static DMT_CPU void divTo(Vec2i& target, int32_t s);
  static DMT_CPU bool eq(Vec2i a, Vec2i b);
  static DMT_CPU Vec2i abs(Vec2i a);
  static DMT_CPU Vec2i min(Vec2i a, Vec2i b);
  static DMT_CPU Vec2i max(Vec2i a, Vec2i s);
  static DMT_CPU int32_t dot(Vec2i a, Vec2i b);
};

template <>
struct Vector2Ops<float, CudaTag> {
  static DMT_GPU Vec2f add(Vec2f a, Vec2f b);
  static DMT_GPU Vec2f add(Vec2f a, float s);
  static DMT_GPU void addTo(Vec2f& target, Vec2f src);
  static DMT_GPU void addTo(Vec2f& target, float s);
  static DMT_GPU Vec2f sub(Vec2f a, Vec2f b);
  static DMT_GPU Vec2f sub(Vec2f a, float s);
  static DMT_GPU void subTo(Vec2f& target, Vec2f src);
  static DMT_GPU void subTo(Vec2f& target, float s);
  static DMT_GPU Vec2f mul(Vec2f a, Vec2f b);
  static DMT_GPU Vec2f mul(Vec2f a, float s);
  static DMT_GPU void mulTo(Vec2f& target, Vec2f src);
  static DMT_GPU void mulTo(Vec2f& target, float s);
  static DMT_GPU Vec2f div(Vec2f a, Vec2f b);
  static DMT_GPU Vec2f div(Vec2f a, float s);
  static DMT_GPU void divTo(Vec2f& target, Vec2f src);
  static DMT_GPU void divTo(Vec2f& target, float s);
  static DMT_GPU bool eq(Vec2f a, Vec2f b);
  static DMT_GPU bool epsilonEq(Vec2f a, Vec2f b, float tol);
  static DMT_GPU Vec2f normalize(Vec2f a);
  static DMT_GPU Vec2f abs(Vec2f a);
  static DMT_GPU Vec2f ceil(Vec2f a);
  static DMT_GPU Vec2f floor(Vec2f a);
  static DMT_GPU Vec2f sqrt(Vec2f a);
  static DMT_GPU Vec2f fma(Vec2f mult0, Vec2f mult1, Vec2f add);
  static DMT_GPU Vec2f min(Vec2f a, Vec2f b);
  static DMT_GPU Vec2f max(Vec2f a, Vec2f s);
  static DMT_GPU float dot(Vec2f a, Vec2f b);
};

template <>
struct Vector2Ops<int32_t, CudaTag> {
  static DMT_GPU Vec2i add(Vec2i a, Vec2i b);
  static DMT_GPU Vec2i add(Vec2i a, int32_t s);
  static DMT_GPU void addTo(Vec2i& target, Vec2i src);
  static DMT_GPU void addTo(Vec2i& target, int32_t s);
  static DMT_GPU Vec2i sub(Vec2i a, Vec2i b);
  static DMT_GPU Vec2i sub(Vec2i a, int32_t s);
  static DMT_GPU void subTo(Vec2i& target, Vec2i src);
  static DMT_GPU void subTo(Vec2i& target, int32_t s);
  static DMT_GPU Vec2i mul(Vec2i a, Vec2i b);
  static DMT_GPU Vec2i mul(Vec2i a, int32_t s);
  static DMT_GPU void mulTo(Vec2i& target, Vec2i src);
  static DMT_GPU void mulTo(Vec2i& target, int32_t s);
  static DMT_GPU Vec2i div(Vec2i a, Vec2i b);
  static DMT_GPU Vec2i div(Vec2i a, int32_t s);
  static DMT_GPU void divTo(Vec2i& target, Vec2i src);
  static DMT_GPU void divTo(Vec2i& target, int32_t s);
  static DMT_GPU bool eq(Vec2i a, Vec2i b);
  static DMT_GPU Vec2i abs(Vec2i a);
  static DMT_GPU Vec2i min(Vec2i a, Vec2i b);
  static DMT_GPU Vec2i max(Vec2i a, Vec2i s);
  static DMT_GPU int32_t dot(Vec2i a, Vec2i b);
};

}  // namespace dmt::impl

namespace dmt {

template <Scalar S>
struct Vec2 : VecBase<Vec2<S>> {
  using value_type = S;
  static consteval int32_t numComponents() { return 2; }

  Vec2() = default;
  DMT_CPU_GPU Vec2(S s0, S s1) : x(s0), y(s1) {}

  DMT_CPU_GPU Vec2<S>& operator+=(Vec2<S> other) {
    impl::Vector2Ops<S, DMT_BACKEND_TAG>::addTo(*this, other);
    return *this;
  }
  DMT_CPU_GPU Vec2<S>& operator-=(Vec2<S> other) {
    impl::Vector2Ops<S, DMT_BACKEND_TAG>::subTo(*this, other);
    return *this;
  }
  DMT_CPU_GPU Vec2<S>& operator*=(Vec2<S> other) {
    impl::Vector2Ops<S, DMT_BACKEND_TAG>::mulTo(*this, other);
    return *this;
  }
  DMT_CPU_GPU Vec2<S>& operator/=(Vec2<S> other) {
    impl::Vector2Ops<S, DMT_BACKEND_TAG>::divTo(*this, other);
    return *this;
  }

  S x, y;
};
static_assert(Vector2<Vec2f> && Vector2<Vec2i>);

template <Scalar S>
inline DMT_CPU_GPU Vec2<S> operator+(Vec2<S> a, Vec2<S> b) {
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::add(a, b);
}
template <Scalar S>
inline DMT_CPU_GPU Vec2<S> operator+(Vec2<S> a, S s) {
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::add(a, s);
}
template <Scalar S>
inline DMT_CPU_GPU Vec2<S> operator-(Vec2<S> a, Vec2<S> b) {
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::sub(a, b);
}
template <Scalar S>
inline DMT_CPU_GPU Vec2<S> operator-(Vec2<S> a, S s) {
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::sub(a, s);
}
template <Scalar S>
inline DMT_CPU_GPU Vec2<S> operator*(Vec2<S> a, Vec2<S> b) {
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::mul(a, b);
}
template <Scalar S>
inline DMT_CPU_GPU Vec2<S> operator*(Vec2<S> a, S s) {
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::mul(a, s);
}
template <Scalar S>
inline DMT_CPU_GPU Vec2<S> operator/(Vec2<S> a, Vec2<S> b) {
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::div(a, b);
}
template <Scalar S>
inline DMT_CPU_GPU Vec2<S> operator/(Vec2<S> a, S s) {
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::div(a, s);
}
template <Scalar S>
DMT_CPU_GPU bool operator==(Vec2<S> a, Vec2<S> b) {
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::eq(a, b);
}
template <Scalar S>
DMT_CPU_GPU bool operator!=(Vec2<S> a, Vec2<S> b) {
  return !impl::Vector2Ops<S, DMT_BACKEND_TAG>::eq(a, b);
}
template <Scalar S>
DMT_CPU_GPU Vec2<S> min(Vec2<S> a, Vec2<S> b) {
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::min(a, b);
}
template <Scalar S>
DMT_CPU_GPU Vec2<S> max(Vec2<S> a, Vec2<S> s) {
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::max(a, s);
}
template <Scalar S>
DMT_CPU_GPU S dot(Vec2<S> a, Vec2<S> b) {
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::dot(a, b);
}
template <Scalar S>
DMT_CPU_GPU bool epsilonEqual(Vec2<S> a, Vec2<S> b, S tol)
  requires std::is_floating_point_v<S>
{
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::epsilonEq(a, b, tol);
}
template <Scalar S>
DMT_CPU_GPU Vec2<S> normalize(Vec2<S> a)
  requires std::is_floating_point_v<S>
{
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::normalize(a);
}
template <Scalar S>
DMT_CPU_GPU Vec2<S> abs(Vec2<S> a)
  requires std::is_floating_point_v<S>
{
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::abs(a);
}
template <Scalar S>
DMT_CPU_GPU Vec2<S> ceil(Vec2<S> a)
  requires std::is_floating_point_v<S>
{
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::ceil(a);
}
template <Scalar S>
DMT_CPU_GPU Vec2<S> floor(Vec2<S> a)
  requires std::is_floating_point_v<S>
{
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::floor(a);
}
template <Scalar S>
DMT_CPU_GPU Vec2<S> sqrt(Vec2<S> a)
  requires std::is_floating_point_v<S>
{
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::sqrt(a);
}
template <Scalar S>
DMT_CPU_GPU Vec2<S> fma(Vec2<S> mult0, Vec2<S> mult1, Vec2<S> add)
  requires std::is_floating_point_v<S>
{
  return impl::Vector2Ops<S, DMT_BACKEND_TAG>::fma(mult0, mult1, add);
}

}  // namespace dmt