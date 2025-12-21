#pragma once

#include "dmt/core/utils/macros.h"
#include "dmt/core/utils/backend_tags.h"
#include "dmt/core/math/vec_base.h"

#include <cstdint>

namespace dmt {
template <Scalar S>
struct Vec4;

using Vector4f = Vec4<float>;
using Vec4f = Vec4<float>;
using Vector4i = Vec4<int32_t>;
using Vec4i = Vec4<int32_t>;

}  // namespace dmt

// necessary boilerplate for compile time dispatch
namespace dmt::impl {

template <Scalar S>
struct CpuVector4;

template <typename Scalar, typename Backend>
struct Vector4Ops;

template <>
struct Vector4Ops<float, CpuTag> {
  static DMT_CPU Vec4f add(Vec4f a, Vec4f b);
  static DMT_CPU Vec4f add(Vec4f a, float s);
  static DMT_CPU void addTo(Vec4f& target, Vec4f src);
  static DMT_CPU void addTo(Vec4f& target, float s);
  static DMT_CPU Vec4f sub(Vec4f a, Vec4f b);
  static DMT_CPU Vec4f sub(Vec4f a, float s);
  static DMT_CPU void subTo(Vec4f& target, Vec4f src);
  static DMT_CPU void subTo(Vec4f& target, float s);
  static DMT_CPU Vec4f mul(Vec4f a, Vec4f b);
  static DMT_CPU Vec4f mul(Vec4f a, float s);
  static DMT_CPU void mulTo(Vec4f& target, Vec4f src);
  static DMT_CPU void mulTo(Vec4f& target, float s);
  static DMT_CPU Vec4f div(Vec4f a, Vec4f b);
  static DMT_CPU Vec4f div(Vec4f a, float s);
  static DMT_CPU void divTo(Vec4f& target, Vec4f src);
  static DMT_CPU void divTo(Vec4f& target, float s);
  static DMT_CPU bool eq(Vec4f a, Vec4f b);
  static DMT_CPU bool epsilonEq(Vec4f a, Vec4f b, float tol);
  static DMT_CPU Vec4f normalize(Vec4f a);
  static DMT_CPU Vec4f abs(Vec4f a);
  static DMT_CPU Vec4f ceil(Vec4f a);
  static DMT_CPU Vec4f floor(Vec4f a);
  static DMT_CPU Vec4f sqrt(Vec4f a);
  static DMT_CPU Vec4f fma(Vec4f mult0, Vec4f mult1, Vec4f add);
  static DMT_CPU Vec4f min(Vec4f a, Vec4f b);
  static DMT_CPU Vec4f max(Vec4f a, Vec4f s);
  static DMT_CPU float dot(Vec4f a, Vec4f b);
};

template <>
struct Vector4Ops<int32_t, CpuTag> {
  static DMT_CPU Vec4i add(Vec4i a, Vec4i b);
  static DMT_CPU Vec4i add(Vec4i a, int32_t s);
  static DMT_CPU void addTo(Vec4i& target, Vec4i src);
  static DMT_CPU void addTo(Vec4i& target, int32_t s);
  static DMT_CPU Vec4i sub(Vec4i a, Vec4i b);
  static DMT_CPU Vec4i sub(Vec4i a, int32_t s);
  static DMT_CPU void subTo(Vec4i& target, Vec4i src);
  static DMT_CPU void subTo(Vec4i& target, int32_t s);
  static DMT_CPU Vec4i mul(Vec4i a, Vec4i b);
  static DMT_CPU Vec4i mul(Vec4i a, int32_t s);
  static DMT_CPU void mulTo(Vec4i& target, Vec4i src);
  static DMT_CPU void mulTo(Vec4i& target, int32_t s);
  static DMT_CPU Vec4i div(Vec4i a, Vec4i b);
  static DMT_CPU Vec4i div(Vec4i a, int32_t s);
  static DMT_CPU void divTo(Vec4i& target, Vec4i src);
  static DMT_CPU void divTo(Vec4i& target, int32_t s);
  static DMT_CPU bool eq(Vec4i a, Vec4i b);
  static DMT_CPU Vec4i abs(Vec4i a);
  static DMT_CPU Vec4i min(Vec4i a, Vec4i b);
  static DMT_CPU Vec4i max(Vec4i a, Vec4i s);
  static DMT_CPU int32_t dot(Vec4i a, Vec4i b);
};

template <>
struct Vector4Ops<float, CudaTag> {
  static DMT_GPU Vec4f add(Vec4f a, Vec4f b);
  static DMT_GPU Vec4f add(Vec4f a, float s);
  static DMT_GPU void addTo(Vec4f& target, Vec4f src);
  static DMT_GPU void addTo(Vec4f& target, float s);
  static DMT_GPU Vec4f sub(Vec4f a, Vec4f b);
  static DMT_GPU Vec4f sub(Vec4f a, float s);
  static DMT_GPU void subTo(Vec4f& target, Vec4f src);
  static DMT_GPU void subTo(Vec4f& target, float s);
  static DMT_GPU Vec4f mul(Vec4f a, Vec4f b);
  static DMT_GPU Vec4f mul(Vec4f a, float s);
  static DMT_GPU void mulTo(Vec4f& target, Vec4f src);
  static DMT_GPU void mulTo(Vec4f& target, float s);
  static DMT_GPU Vec4f div(Vec4f a, Vec4f b);
  static DMT_GPU Vec4f div(Vec4f a, float s);
  static DMT_GPU void divTo(Vec4f& target, Vec4f src);
  static DMT_GPU void divTo(Vec4f& target, float s);
  static DMT_GPU bool eq(Vec4f a, Vec4f b);
  static DMT_GPU bool epsilonEq(Vec4f a, Vec4f b, float tol);
  static DMT_GPU Vec4f normalize(Vec4f a);
  static DMT_GPU Vec4f abs(Vec4f a);
  static DMT_GPU Vec4f ceil(Vec4f a);
  static DMT_GPU Vec4f floor(Vec4f a);
  static DMT_GPU Vec4f sqrt(Vec4f a);
  static DMT_GPU Vec4f fma(Vec4f mult0, Vec4f mult1, Vec4f add);
  static DMT_GPU Vec4f min(Vec4f a, Vec4f b);
  static DMT_GPU Vec4f max(Vec4f a, Vec4f s);
  static DMT_GPU float dot(Vec4f a, Vec4f b);
};

template <>
struct Vector4Ops<int32_t, CudaTag> {
  static DMT_GPU Vec4i add(Vec4i a, Vec4i b);
  static DMT_GPU Vec4i add(Vec4i a, int32_t s);
  static DMT_GPU void addTo(Vec4i& target, Vec4i src);
  static DMT_GPU void addTo(Vec4i& target, int32_t s);
  static DMT_GPU Vec4i sub(Vec4i a, Vec4i b);
  static DMT_GPU Vec4i sub(Vec4i a, int32_t s);
  static DMT_GPU void subTo(Vec4i& target, Vec4i src);
  static DMT_GPU void subTo(Vec4i& target, int32_t s);
  static DMT_GPU Vec4i mul(Vec4i a, Vec4i b);
  static DMT_GPU Vec4i mul(Vec4i a, int32_t s);
  static DMT_GPU void mulTo(Vec4i& target, Vec4i src);
  static DMT_GPU void mulTo(Vec4i& target, int32_t s);
  static DMT_GPU Vec4i div(Vec4i a, Vec4i b);
  static DMT_GPU Vec4i div(Vec4i a, int32_t s);
  static DMT_GPU void divTo(Vec4i& target, Vec4i src);
  static DMT_GPU void divTo(Vec4i& target, int32_t s);
  static DMT_GPU bool eq(Vec4i a, Vec4i b);
  static DMT_GPU Vec4i abs(Vec4i a);
  static DMT_GPU Vec4i min(Vec4i a, Vec4i b);
  static DMT_GPU Vec4i max(Vec4i a, Vec4i s);
  static DMT_GPU int32_t dot(Vec4i a, Vec4i b);
};

}  // namespace dmt::impl

namespace dmt {

template <Scalar S>
struct Vec4 : VecBase<Vec4<S>> {
  using value_type = S;
  static consteval int32_t numComponents() { return 4; }

  Vec4() = default;
  DMT_CPU_GPU Vec4(S s0, S s1, S s2, S s3) : x(s0), y(s1), z(s2), w(s3) {}

  DMT_CPU_GPU Vec4<S>& operator+=(Vec4<S> other) {
    impl::Vector4Ops<S, DMT_BACKEND_TAG>::addTo(*this, other);
    return *this;
  }
  DMT_CPU_GPU Vec4<S>& operator-=(Vec4<S> other) {
    impl::Vector4Ops<S, DMT_BACKEND_TAG>::subTo(*this, other);
    return *this;
  }
  DMT_CPU_GPU Vec4<S>& operator*=(Vec4<S> other) {
    impl::Vector4Ops<S, DMT_BACKEND_TAG>::mulTo(*this, other);
    return *this;
  }
  DMT_CPU_GPU Vec4<S>& operator/=(Vec4<S> other) {
    impl::Vector4Ops<S, DMT_BACKEND_TAG>::divTo(*this, other);
    return *this;
  }

  S x, y, z, w;
};
static_assert(Vector4<Vec4f> && Vector4<Vec4i>);

template <Scalar S>
inline DMT_CPU_GPU Vec4<S> operator+(Vec4<S> a, Vec4<S> b) {
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::add(a, b);
}
template <Scalar S>
inline DMT_CPU_GPU Vec4<S> operator+(Vec4<S> a, S s) {
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::add(a, s);
}
template <Scalar S>
inline DMT_CPU_GPU Vec4<S> operator-(Vec4<S> a, Vec4<S> b) {
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::sub(a, b);
}
template <Scalar S>
inline DMT_CPU_GPU Vec4<S> operator-(Vec4<S> a, S s) {
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::sub(a, s);
}
template <Scalar S>
inline DMT_CPU_GPU Vec4<S> operator*(Vec4<S> a, Vec4<S> b) {
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::mul(a, b);
}
template <Scalar S>
inline DMT_CPU_GPU Vec4<S> operator*(Vec4<S> a, S s) {
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::mul(a, s);
}
template <Scalar S>
inline DMT_CPU_GPU Vec4<S> operator/(Vec4<S> a, Vec4<S> b) {
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::div(a, b);
}
template <Scalar S>
inline DMT_CPU_GPU Vec4<S> operator/(Vec4<S> a, S s) {
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::div(a, s);
}
template <Scalar S>
DMT_CPU_GPU bool operator==(Vec4<S> a, Vec4<S> b) {
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::eq(a, b);
}
template <Scalar S>
DMT_CPU_GPU bool operator!=(Vec4<S> a, Vec4<S> b) {
  return !impl::Vector4Ops<S, DMT_BACKEND_TAG>::eq(a, b);
}
template <Scalar S>
DMT_CPU_GPU Vec4<S> min(Vec4<S> a, Vec4<S> b) {
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::min(a, b);
}
template <Scalar S>
DMT_CPU_GPU Vec4<S> max(Vec4<S> a, Vec4<S> s) {
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::max(a, s);
}
template <Scalar S>
DMT_CPU_GPU S dot(Vec4<S> a, Vec4<S> b) {
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::dot(a, b);
}
template <Scalar S>
DMT_CPU_GPU bool epsilonEqual(Vec4<S> a, Vec4<S> b, S tol)
  requires std::is_floating_point_v<S>
{
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::epsilonEq(a, b, tol);
}
template <Scalar S>
DMT_CPU_GPU Vec4<S> normalize(Vec4<S> a)
  requires std::is_floating_point_v<S>
{
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::normalize(a);
}
template <Scalar S>
DMT_CPU_GPU Vec4<S> abs(Vec4<S> a)
  requires std::is_floating_point_v<S>
{
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::abs(a);
}
template <Scalar S>
DMT_CPU_GPU Vec4<S> ceil(Vec4<S> a)
  requires std::is_floating_point_v<S>
{
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::ceil(a);
}
template <Scalar S>
DMT_CPU_GPU Vec4<S> floor(Vec4<S> a)
  requires std::is_floating_point_v<S>
{
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::floor(a);
}
template <Scalar S>
DMT_CPU_GPU Vec4<S> sqrt(Vec4<S> a)
  requires std::is_floating_point_v<S>
{
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::sqrt(a);
}
template <Scalar S>
DMT_CPU_GPU Vec4<S> fma(Vec4<S> mult0, Vec4<S> mult1, Vec4<S> add)
  requires std::is_floating_point_v<S>
{
  return impl::Vector4Ops<S, DMT_BACKEND_TAG>::fma(mult0, mult1, add);
}

}  // namespace dmt