#pragma once

#include "dmt/core/utils/macros.h"
#include "dmt/core/utils/backend_tags.h"
#include "dmt/core/math/vec_base.h"

#include <cstdint>

namespace dmt {
template <Scalar S>
struct Vec3;

using Vector3f = Vec3<float>;
using Vec3f = Vec3<float>;
using Vector3i = Vec3<int32_t>;
using Vec3i = Vec3<int32_t>;

}  // namespace dmt

// necessary boilerplate for compile time dispatch
namespace dmt::impl {

template <Scalar S>
struct CpuVector3;

template <typename Scalar, typename Backend>
struct Vector3Ops;

template <>
struct Vector3Ops<float, CpuTag> {
  static DMT_CPU Vec3f add(Vec3f a, Vec3f b);
  static DMT_CPU Vec3f add(Vec3f a, float s);
  static DMT_CPU void addTo(Vec3f& target, Vec3f src);
  static DMT_CPU void addTo(Vec3f& target, float s);
  static DMT_CPU Vec3f sub(Vec3f a, Vec3f b);
  static DMT_CPU Vec3f sub(Vec3f a, float s);
  static DMT_CPU void subTo(Vec3f& target, Vec3f src);
  static DMT_CPU void subTo(Vec3f& target, float s);
  static DMT_CPU Vec3f mul(Vec3f a, Vec3f b);
  static DMT_CPU Vec3f mul(Vec3f a, float s);
  static DMT_CPU void mulTo(Vec3f& target, Vec3f src);
  static DMT_CPU void mulTo(Vec3f& target, float s);
  static DMT_CPU Vec3f div(Vec3f a, Vec3f b);
  static DMT_CPU Vec3f div(Vec3f a, float s);
  static DMT_CPU void divTo(Vec3f& target, Vec3f src);
  static DMT_CPU void divTo(Vec3f& target, float s);
  static DMT_CPU bool eq(Vec3f a, Vec3f b);
  static DMT_CPU bool epsilonEq(Vec3f a, Vec3f b, float tol);
  static DMT_CPU Vec3f normalize(Vec3f a);
  static DMT_CPU Vec3f abs(Vec3f a);
  static DMT_CPU Vec3f ceil(Vec3f a);
  static DMT_CPU Vec3f floor(Vec3f a);
  static DMT_CPU Vec3f sqrt(Vec3f a);
  static DMT_CPU Vec3f fma(Vec3f mult0, Vec3f mult1, Vec3f add);
  static DMT_CPU Vec3f min(Vec3f a, Vec3f b);
  static DMT_CPU Vec3f max(Vec3f a, Vec3f s);
  static DMT_CPU float dot(Vec3f a, Vec3f b);
};

template <>
struct Vector3Ops<int32_t, CpuTag> {
  static DMT_CPU Vec3i add(Vec3i a, Vec3i b);
  static DMT_CPU Vec3i add(Vec3i a, int32_t s);
  static DMT_CPU void addTo(Vec3i& target, Vec3i src);
  static DMT_CPU void addTo(Vec3i& target, int32_t s);
  static DMT_CPU Vec3i sub(Vec3i a, Vec3i b);
  static DMT_CPU Vec3i sub(Vec3i a, int32_t s);
  static DMT_CPU void subTo(Vec3i& target, Vec3i src);
  static DMT_CPU void subTo(Vec3i& target, int32_t s);
  static DMT_CPU Vec3i mul(Vec3i a, Vec3i b);
  static DMT_CPU Vec3i mul(Vec3i a, int32_t s);
  static DMT_CPU void mulTo(Vec3i& target, Vec3i src);
  static DMT_CPU void mulTo(Vec3i& target, int32_t s);
  static DMT_CPU Vec3i div(Vec3i a, Vec3i b);
  static DMT_CPU Vec3i div(Vec3i a, int32_t s);
  static DMT_CPU void divTo(Vec3i& target, Vec3i src);
  static DMT_CPU void divTo(Vec3i& target, int32_t s);
  static DMT_CPU bool eq(Vec3i a, Vec3i b);
  static DMT_CPU Vec3i abs(Vec3i a);
  static DMT_CPU Vec3i min(Vec3i a, Vec3i b);
  static DMT_CPU Vec3i max(Vec3i a, Vec3i s);
  static DMT_CPU int32_t dot(Vec3i a, Vec3i b);
};

template <>
struct Vector3Ops<float, CudaTag> {
  static DMT_GPU Vec3f add(Vec3f a, Vec3f b);
  static DMT_GPU Vec3f add(Vec3f a, float s);
  static DMT_GPU void addTo(Vec3f& target, Vec3f src);
  static DMT_GPU void addTo(Vec3f& target, float s);
  static DMT_GPU Vec3f sub(Vec3f a, Vec3f b);
  static DMT_GPU Vec3f sub(Vec3f a, float s);
  static DMT_GPU void subTo(Vec3f& target, Vec3f src);
  static DMT_GPU void subTo(Vec3f& target, float s);
  static DMT_GPU Vec3f mul(Vec3f a, Vec3f b);
  static DMT_GPU Vec3f mul(Vec3f a, float s);
  static DMT_GPU void mulTo(Vec3f& target, Vec3f src);
  static DMT_GPU void mulTo(Vec3f& target, float s);
  static DMT_GPU Vec3f div(Vec3f a, Vec3f b);
  static DMT_GPU Vec3f div(Vec3f a, float s);
  static DMT_GPU void divTo(Vec3f& target, Vec3f src);
  static DMT_GPU void divTo(Vec3f& target, float s);
  static DMT_GPU bool eq(Vec3f a, Vec3f b);
  static DMT_GPU bool epsilonEq(Vec3f a, Vec3f b, float tol);
  static DMT_GPU Vec3f normalize(Vec3f a);
  static DMT_GPU Vec3f abs(Vec3f a);
  static DMT_GPU Vec3f ceil(Vec3f a);
  static DMT_GPU Vec3f floor(Vec3f a);
  static DMT_GPU Vec3f sqrt(Vec3f a);
  static DMT_GPU Vec3f fma(Vec3f mult0, Vec3f mult1, Vec3f add);
  static DMT_GPU Vec3f min(Vec3f a, Vec3f b);
  static DMT_GPU Vec3f max(Vec3f a, Vec3f s);
  static DMT_GPU float dot(Vec3f a, Vec3f b);
};

template <>
struct Vector3Ops<int32_t, CudaTag> {
  static DMT_GPU Vec3i add(Vec3i a, Vec3i b);
  static DMT_GPU Vec3i add(Vec3i a, int32_t s);
  static DMT_GPU void addTo(Vec3i& target, Vec3i src);
  static DMT_GPU void addTo(Vec3i& target, int32_t s);
  static DMT_GPU Vec3i sub(Vec3i a, Vec3i b);
  static DMT_GPU Vec3i sub(Vec3i a, int32_t s);
  static DMT_GPU void subTo(Vec3i& target, Vec3i src);
  static DMT_GPU void subTo(Vec3i& target, int32_t s);
  static DMT_GPU Vec3i mul(Vec3i a, Vec3i b);
  static DMT_GPU Vec3i mul(Vec3i a, int32_t s);
  static DMT_GPU void mulTo(Vec3i& target, Vec3i src);
  static DMT_GPU void mulTo(Vec3i& target, int32_t s);
  static DMT_GPU Vec3i div(Vec3i a, Vec3i b);
  static DMT_GPU Vec3i div(Vec3i a, int32_t s);
  static DMT_GPU void divTo(Vec3i& target, Vec3i src);
  static DMT_GPU void divTo(Vec3i& target, int32_t s);
  static DMT_GPU bool eq(Vec3i a, Vec3i b);
  static DMT_GPU Vec3i abs(Vec3i a);
  static DMT_GPU Vec3i min(Vec3i a, Vec3i b);
  static DMT_GPU Vec3i max(Vec3i a, Vec3i s);
  static DMT_GPU int32_t dot(Vec3i a, Vec3i b);
};

}  // namespace dmt::impl

namespace dmt {

template <Scalar S>
struct Vec3 : VecBase<Vec3<S>> {
  using value_type = S;
  static consteval int32_t numComponents() { return 3; }

  Vec3() = default;
  DMT_CPU_GPU Vec3(S s0, S s1, S s2) : x(s0), y(s1), z(s2) {}

  DMT_CPU_GPU Vec3<S>& operator+=(Vec3<S> other) {
    impl::Vector3Ops<S, DMT_BACKEND_TAG>::addTo(*this, other);
    return *this;
  }
  DMT_CPU_GPU Vec3<S>& operator-=(Vec3<S> other) {
    impl::Vector3Ops<S, DMT_BACKEND_TAG>::subTo(*this, other);
    return *this;
  }
  DMT_CPU_GPU Vec3<S>& operator*=(Vec3<S> other) {
    impl::Vector3Ops<S, DMT_BACKEND_TAG>::mulTo(*this, other);
    return *this;
  }
  DMT_CPU_GPU Vec3<S>& operator/=(Vec3<S> other) {
    impl::Vector3Ops<S, DMT_BACKEND_TAG>::divTo(*this, other);
    return *this;
  }

  S x, y, z;
};
static_assert(Vector3<Vec3f> && Vector3<Vec3i>);

template <Scalar S>
inline DMT_CPU_GPU Vec3<S> operator+(Vec3<S> a, Vec3<S> b) {
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::add(a, b);
}
template <Scalar S>
inline DMT_CPU_GPU Vec3<S> operator+(Vec3<S> a, S s) {
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::add(a, s);
}
template <Scalar S>
inline DMT_CPU_GPU Vec3<S> operator-(Vec3<S> a, Vec3<S> b) {
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::sub(a, b);
}
template <Scalar S>
inline DMT_CPU_GPU Vec3<S> operator-(Vec3<S> a, S s) {
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::sub(a, s);
}
template <Scalar S>
inline DMT_CPU_GPU Vec3<S> operator*(Vec3<S> a, Vec3<S> b) {
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::mul(a, b);
}
template <Scalar S>
inline DMT_CPU_GPU Vec3<S> operator*(Vec3<S> a, S s) {
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::mul(a, s);
}
template <Scalar S>
inline DMT_CPU_GPU Vec3<S> operator/(Vec3<S> a, Vec3<S> b) {
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::div(a, b);
}
template <Scalar S>
inline DMT_CPU_GPU Vec3<S> operator/(Vec3<S> a, S s) {
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::div(a, s);
}
template <Scalar S>
DMT_CPU_GPU bool operator==(Vec3<S> a, Vec3<S> b) {
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::eq(a, b);
}
template <Scalar S>
DMT_CPU_GPU bool operator!=(Vec3<S> a, Vec3<S> b) {
  return !impl::Vector3Ops<S, DMT_BACKEND_TAG>::eq(a, b);
}
template <Scalar S>
DMT_CPU_GPU Vec3<S> min(Vec3<S> a, Vec3<S> b) {
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::min(a, b);
}
template <Scalar S>
DMT_CPU_GPU Vec3<S> max(Vec3<S> a, Vec3<S> s) {
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::max(a, s);
}
template <Scalar S>
DMT_CPU_GPU S dot(Vec3<S> a, Vec3<S> b) {
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::dot(a, b);
}
template <Scalar S>
DMT_CPU_GPU bool epsilonEqual(Vec3<S> a, Vec3<S> b, S tol)
  requires std::is_floating_point_v<S>
{
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::epsilonEq(a, b, tol);
}
template <Scalar S>
DMT_CPU_GPU Vec3<S> normalize(Vec3<S> a)
  requires std::is_floating_point_v<S>
{
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::normalize(a);
}
template <Scalar S>
DMT_CPU_GPU Vec3<S> abs(Vec3<S> a)
  requires std::is_floating_point_v<S>
{
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::abs(a);
}
template <Scalar S>
DMT_CPU_GPU Vec3<S> ceil(Vec3<S> a)
  requires std::is_floating_point_v<S>
{
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::ceil(a);
}
template <Scalar S>
DMT_CPU_GPU Vec3<S> floor(Vec3<S> a)
  requires std::is_floating_point_v<S>
{
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::floor(a);
}
template <Scalar S>
DMT_CPU_GPU Vec3<S> sqrt(Vec3<S> a)
  requires std::is_floating_point_v<S>
{
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::sqrt(a);
}
template <Scalar S>
DMT_CPU_GPU Vec3<S> fma(Vec3<S> mult0, Vec3<S> mult1, Vec3<S> add)
  requires std::is_floating_point_v<S>
{
  return impl::Vector3Ops<S, DMT_BACKEND_TAG>::fma(mult0, mult1, add);
}

}  // namespace dmt