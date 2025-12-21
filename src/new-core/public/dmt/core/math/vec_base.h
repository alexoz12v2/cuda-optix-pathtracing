#pragma once

#include "dmt/core/utils/macros.h"

#include <cstdint>
#include <cmath>
#include <concepts>
#include <type_traits>

/// \file vec_base.h
/// contains basic concepts and base classes to implement a low-dimensional
/// mathematical vector

namespace dmt {

template <typename T>
concept Vector2 = requires(T t) {
  requires std::is_trivial_v<T> && std::is_standard_layout_v<T>;
  typename T::value_type;
  { T::numComponents() } -> std::same_as<int32_t>;
  { t[0] } -> std::same_as<typename T::value_type&>;
  { t.x } -> std::same_as<typename T::value_type&>;
  { t.y } -> std::same_as<typename T::value_type&>;
  requires sizeof(T) == 2 * sizeof(typename T::value_type);
};

template <typename T>
concept Vector3 = requires(T t) {
  requires std::is_trivial_v<T> && std::is_standard_layout_v<T>;
  typename T::value_type;
  { T::numComponents() } -> std::same_as<int32_t>;
  { t[0] } -> std::same_as<typename T::value_type&>;
  { t.x } -> std::same_as<typename T::value_type&>;
  { t.y } -> std::same_as<typename T::value_type&>;
  { t.z } -> std::same_as<typename T::value_type&>;
  requires sizeof(T) == 3 * sizeof(typename T::value_type);
};

template <typename T>
concept Vector4 = requires(T t) {
  requires std::is_trivial_v<T> && std::is_standard_layout_v<T>;
  typename T::value_type;
  { T::numComponents() } -> std::same_as<int32_t>;
  { t[0] } -> std::same_as<typename T::value_type&>;
  { t.x } -> std::same_as<typename T::value_type&>;
  { t.y } -> std::same_as<typename T::value_type&>;
  { t.z } -> std::same_as<typename T::value_type&>;
  { t.w } -> std::same_as<typename T::value_type&>;
  requires sizeof(T) == 4 * sizeof(typename T::value_type);
};

template <typename T>
concept Scalar = std::integral<T> || std::floating_point<T>;

template <typename T>
concept Vector = Vector2<T> || Vector3<T> || Vector4<T>;

template <typename Derived>
struct VecBase {
  // note: unsafe cast. Don't mix up floats and integers, they wor't convert
  template <Scalar T>
  DMT_CPU_GPU std::remove_cvref_t<std::remove_all_extents_t<T>>& ref() {
    return *ptr<T>();
  }
  template <Scalar T>
  DMT_CPU_GPU std::remove_cvref_t<std::remove_all_extents_t<T>> const& ref()
      const {
    return *ptr<T>();
  }

  template <Scalar T>
  DMT_CPU_GPU std::remove_cvref_t<std::remove_all_extents_t<T>>* ptr() {
    return reinterpret_cast<std::remove_cvref_t<std::remove_all_extents_t<T>>*>(
        &static_cast<Derived*>(this)->x);
  }
  template <Scalar T>
  DMT_CPU_GPU std::remove_cvref_t<std::remove_all_extents_t<T>> const* ptr()
      const {
    return reinterpret_cast<std::remove_cvref_t<std::remove_all_extents_t<T>>*>(
        &static_cast<Derived const*>(this)->x);
  }

  DMT_CPU_GPU auto& operator[](int32_t idx) {
    return *(ptr<typename Derived::value_type>() + idx);
  }
  DMT_CPU_GPU auto const& operator[](int32_t idx) const {
    return *(ptr<typename Derived::value_type>() + idx);
  }
};

template <Vector T, typename F>
  requires std::is_invocable_r_v<typename T::value_type, F,
                                 typename T::value_type>
inline DMT_CPU T map(T vec, F&& mapFunc) {
  T result = vec;
  result.x = mapFunc(result.x);
  result.y = mapFunc(result.y);
  result.z = mapFunc(result.z);
  return result;
}

template <Vector T>
DMT_CPU inline T bcast(typename T::value_type t) {
  T ret;
  for (int32_t i = 0; i < T::numComponents(); ++i) ret[i] = t;
  return ret;
}

template <Vector T>
DMT_CPU inline T operator*(T v, typename T::value_type k) {
  T ret;
  for (int32_t i = 0; i < T::numComponents(); ++i) ret[i] = k * v[i];
  return ret;
}

template <Vector T>
DMT_CPU inline T operator*(typename T::value_type k, T v) {
  T ret;
  for (int32_t i = 0; i < T::numComponents(); ++i) ret[i] = k * v[i];
  return ret;
}

template <Vector T>
DMT_CPU inline T operator/(T v, typename T::value_type k) {
  T ret;
  for (int32_t i = 0; i < T::numComponents(); ++i) ret[i] = v[i] / k;
  return ret;
}

template <Vector T>
DMT_CPU inline T operator/(typename T::value_type k, T v) {
  T ret;
  for (int32_t i = 0; i < T::numComponents(); ++i) ret[i] = k / v[i];
  return ret;
}

template <Vector T>
DMT_CPU inline T& operator*=(T& v, typename T::value_type k) {
  for (int32_t i = 0; i < T::numComponents(); ++i) v[i] *= k;
  return v;
}
template <Vector T>
DMT_CPU inline T& operator/=(T& v, typename T::value_type k) {
  for (int32_t i = 0; i < T::numComponents(); ++i) v[i] /= k;
  return v;
}

template <Vector T>
DMT_CPU inline int32_t maxComponentIndex(T v) {
  int32_t max = 0;
  for (int32_t i = 1; i < T::numComponents(); ++i)
    if (v[max] < v[i]) max = i;

  return max;
}

template <Vector T>
DMT_CPU inline T::value_type maxComponent(T v) {
  int32_t max = 0;
  for (int32_t i = 1; i < T::numComponents(); ++i)
    if (v[max] < v[i]) max = i;

  return v[max];
}

template <Vector T>
DMT_CPU inline int32_t minComponentIndex(T v) {
  int32_t min = 0;
  for (int32_t i = 1; i < T::numComponents(); ++i)
    if (v[min] > v[i]) min = i;

  return min;
}

template <Vector T>
DMT_CPU inline T::value_type minComponent(T v) {
  int32_t min = 0;
  for (int32_t i = 1; i < T::numComponents(); ++i)
    if (v[min] > v[i]) min = i;

  return v[min];
}

template <Vector T>
  requires(T::numComponents() == 2)
DMT_CPU inline T permute(T v, int32_t i0, int32_t i1) {
  T ret;
  ret[0] = v[i0];
  ret[1] = v[i1];
  return ret;
}

template <Vector T>
  requires(T::numComponents() == 3)
DMT_CPU inline T permute(T v, int32_t i0, int32_t i1, int32_t i2) {
  T ret;
  ret[0] = v[i0];
  ret[1] = v[i1];
  ret[2] = v[i2];
  return ret;
}

template <Vector T>
  requires(T::numComponents() == 4)
DMT_CPU inline T permute(T v, int32_t i0, int32_t i1, int32_t i2, int32_t i3) {
  T ret;
  ret[0] = v[i0];
  ret[1] = v[i1];
  ret[2] = v[i2];
  ret[3] = v[i3];
  return ret;
}

template <Vector T>
DMT_CPU inline T::value_type hprod(T v) {
  typename T::value_type ret = v[0];
  for (int32_t i = 1; i < T::numComponents(); ++i) ret *= v[i];
  return ret;
}

template <Vector T>
  requires std::is_floating_point_v<typename T::value_type>
DMT_CPU inline bool hasNaN(T v) {
  for (int32_t i = 0; i < T::numComponents(); ++i) {
    if (std::isnan(v[i])) return false;
  }
  return true;
}

}  // namespace dmt
