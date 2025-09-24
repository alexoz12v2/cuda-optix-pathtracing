#pragma once

#include "cudautils/cudautils-macro.h"

#if defined(__NVCC__)
    #pragma nv_diag_suppress 20012         // both eigen and glm
    #pragma nv_diag_suppress 3012          // glm
    #define diag_suppress nv_diag_suppress // eigen uses old syntax?
#endif
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/common.hpp>
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/vec3.hpp>   // Vec3f
#include <glm/vec4.hpp>   // Vec4f
#include <glm/ext/quaternion_float.hpp>
#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/ext/matrix_transform.hpp>  // glm::translate, glm::rotate, glm::scale
#include <glm/ext/scalar_constants.hpp>  // glm::pi
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/matrix_decompose.hpp> // glm::decompose
#include <glm/gtx/norm.hpp>             // glm::length2

#if !defined(__NVCC__) && !defined(__CUDA_ARCH__)
    #include <Eigen/Dense>
#endif
#if defined(__NVCC__)
    #pragma nv_diag_default 20012
    #pragma nv_diag_default 3012
#endif
#undef diag_suppress

#if !defined(__NVCC__) && !defined(__CUDA_ARCH__)
    #include <platform/platform-utils.h>
    #include <bit>
#endif

#include "cudautils/cudautils-float.h"

#include <array>

#include <cassert>
#include <cstdint>
#include <cmath>

#if defined(DMT_ARCH_X86_64) && !defined(__CUDA_ARCH__)
    #include <immintrin.h>
#endif

#if defined(__CUDA_ARCH__)
    #include <cstddef>
    #include <cuda/std/type_traits>
#endif

#if defined(DMT_OS_WINDOWS)
    #pragma push_macro("near")
    #undef near
#endif

namespace dmt {
    // Vector Types ---------------------------------------------------------------------------------------------------
#if __cplusplus >= 202002L
    // clang-format off
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

    template <typename T, typename = void>
    struct has_normalized : std::false_type {};

    template <typename T>
    struct has_normalized<T, std::void_t<typename T::normalized>> : std::true_type {};

    template <typename T>
    concept VectorScalable = Vector<T> && !has_normalized<T>::value;

    template <typename T>
    concept VectorNormalized = Vector<T> && has_normalized<T>::value;
    // clang-format on
#endif

#if __cplusplus >= 202002L
    template <Scalar S>
#else
    template <typename S>
#endif
    struct Tuple2
    {
        DMT_CPU_GPU static constexpr Tuple2<S> zero() { return {.x = static_cast<S>(0), .y = static_cast<S>(0)}; }
        DMT_CPU_GPU static constexpr Tuple2<S> s(S s) { return {.x = static_cast<S>(s), .y = static_cast<S>(s)}; }

        using value_type = S;
#if __cplusplus >= 202002L
        static consteval int32_t numComponents() { return 2; }
#else
        static constexpr int32_t numComponents() { return 2; }
#endif
        DMT_CPU_GPU S& operator[](int32_t i)
        {
            assert(i >= 0 && i < 2);
#if defined(__CUDA_ARCH__)
            return *(reinterpret_cast<S*>(this) + i);
#else
            return *(std::bit_cast<S*>(this) + i);
#endif
        }
        DMT_CPU_GPU S const& operator[](int32_t i) const
        {
            assert(i >= 0 && i < 2);
#if defined(__CUDA_ARCH__)
            return *(reinterpret_cast<S const*>(this) + i);
#else
            return *(std::bit_cast<S const*>(this) + i);
#endif
        }
        S x, y;
    };

#if __cplusplus >= 202002L
    template <Scalar S>
#else
    template <typename S>
#endif
    struct Tuple3
    {
        DMT_CPU_GPU static constexpr Tuple3<S> zero()
        {
            return {static_cast<S>(0), static_cast<S>(0), static_cast<S>(0)};
        }
        DMT_CPU_GPU static constexpr Tuple3<S> xAxis()
        {
            return {static_cast<S>(1), static_cast<S>(0), static_cast<S>(0)};
        }
        DMT_CPU_GPU static constexpr Tuple3<S> yAxis()
        {
            return {static_cast<S>(0), static_cast<S>(1), static_cast<S>(0)};
        }
        DMT_CPU_GPU static constexpr Tuple3<S> zAxis()
        {
            return {static_cast<S>(0), static_cast<S>(0), static_cast<S>(1)};
        }
        DMT_CPU_GPU static constexpr Tuple3<S> one()
        {
            return {static_cast<S>(1), static_cast<S>(1), static_cast<S>(1)};
        }
        DMT_CPU_GPU static constexpr Tuple3<S> s(S s)
        {
            return {static_cast<S>(s), static_cast<S>(s), static_cast<S>(s)};
        }

        using value_type = S;
#if __cplusplus >= 202002L
        static consteval int32_t numComponents() { return 3; }
#else
        static constexpr int32_t numComponents() { return 3; }
#endif
        DMT_CPU_GPU S& operator[](int32_t i)
        {
            assert(i >= 0 && i < 3);
#if defined(__CUDA_ARCH__)
            return *(reinterpret_cast<S*>(this) + i);
#else
            return *(std::bit_cast<S*>(this) + i);
#endif
        }
        DMT_CPU_GPU S const& operator[](int32_t i) const
        {
            assert(i >= 0 && i < 3);
#if defined(__CUDA_ARCH__)
            return *(reinterpret_cast<S const*>(this) + i);
#else
            return *(std::bit_cast<S const*>(this) + i);
#endif
        }
        S x, y, z;
    };

#if __cplusplus >= 202002L
    template <Scalar S>
#else
    template <typename S>
#endif
    struct Tuple4
    {
        DMT_CPU_GPU static constexpr Tuple4<S> zero()
        {
            return {.x = static_cast<S>(0), .y = static_cast<S>(0), .z = static_cast<S>(0), .w = static_cast<S>(0)};
        }
        DMT_CPU_GPU static constexpr Tuple4<S> quatIdentity()
        {
            return {.x = static_cast<S>(1), .y = static_cast<S>(0), .z = static_cast<S>(0), .w = static_cast<S>(0)};
        }
        DMT_CPU_GPU static constexpr Tuple4<S> s(S s)
        {
            return {.x = static_cast<S>(s), .y = static_cast<S>(s), .z = static_cast<S>(s)};
        }

        using value_type = S;
#if __cplusplus >= 202002L
        static consteval int32_t numComponents() { return 4; }
#else
        static constexpr int32_t numComponents() { return 4; }
#endif
        DMT_CPU_GPU S& operator[](int32_t i)
        {
            assert(i >= 0 && i < 4);
#if defined(__CUDA_ARCH__)
            return *(reinterpret_cast<S*>(this) + i);
#else
            return *(std::bit_cast<S*>(this) + i);
#endif
        }
        DMT_CPU_GPU S const& operator[](int32_t i) const
        {
            assert(i >= 0 && i < 4);
#if defined(__CUDA_ARCH__)
            return *(reinterpret_cast<S const*>(this) + i);
#else
            return *(std::bit_cast<S const*>(this) + i);
#endif
        }
        S x, y, z, w;
    };

    // clang-format off
    using Tuple2f = Tuple2<float>;
    using Tuple2i = Tuple2<int32_t>;
    using Tuple3f = Tuple3<float>;
    using Tuple3i = Tuple3<int32_t>;
    using Tuple4f = Tuple4<float>;
    using Tuple4i = Tuple4<int32_t>;

    struct Normalized { struct normalized { }; };

    struct DMT_CORE_API Vector2i : public Tuple2i { Vector2i() = default; DMT_CPU_GPU Vector2i(Tuple2i t) : Tuple2i(t) {} DMT_CPU_GPU Vector2i(int32_t x, int32_t y) : Tuple2i{x, y} {} };
    struct DMT_CORE_API Vector2f : public Tuple2f { Vector2f() = default; DMT_CPU_GPU Vector2f(Tuple2f t) : Tuple2f(t) {} DMT_CPU_GPU Vector2f(float x, float y) : Tuple2f{x, y} {} };
    struct DMT_CORE_API Vector3i : public Tuple3i { Vector3i() = default; DMT_CPU_GPU Vector3i(Tuple3i t) : Tuple3i(t) {} DMT_CPU_GPU Vector3i(int32_t x, int32_t y, int32_t z) : Tuple3i{x, y, z} {} };
    struct DMT_CORE_API Vector3f : public Tuple3f { Vector3f() = default; DMT_CPU_GPU Vector3f(Tuple3f t) : Tuple3f(t) {} DMT_CPU_GPU Vector3f(float x, float y, float z) : Tuple3f{x, y, z} {} };
    struct DMT_CORE_API Vector4i : public Tuple4i { Vector4i() = default; DMT_CPU_GPU Vector4i(Tuple4i t) : Tuple4i(t) {} DMT_CPU_GPU Vector4i(int32_t x, int32_t y, int32_t z, int32_t w) : Tuple4i{x, y, z, w} {} };
    struct DMT_CORE_API Vector4f : public Tuple4f { Vector4f() = default; DMT_CPU_GPU Vector4f(Tuple4f t) : Tuple4f(t) {} DMT_CPU_GPU Vector4f(float x, float y, float z, float w) : Tuple4f{x, y, z, w} {} };

    struct DMT_CORE_API Point2i : public Tuple2i { Point2i() = default; DMT_CPU_GPU Point2i(Tuple2i t) : Tuple2i(t) {} explicit DMT_CPU_GPU operator Vector2i(); DMT_CPU_GPU Point2i(int32_t x, int32_t y) : Tuple2i{x, y} {} };
    struct DMT_CORE_API Point2f : public Tuple2f { Point2f() = default; DMT_CPU_GPU Point2f(Tuple2f t) : Tuple2f(t) {} explicit DMT_CPU_GPU operator Vector2f(); DMT_CPU_GPU Point2f(float x, float y) : Tuple2f{x, y} {} };
    struct DMT_CORE_API Point3i : public Tuple3i { Point3i() = default; DMT_CPU_GPU Point3i(Tuple3i t) : Tuple3i(t) {} explicit DMT_CPU_GPU operator Vector3i(); DMT_CPU_GPU Point3i(int32_t x, int32_t y, int32_t z) : Tuple3i{x, y, z} {} };
    struct DMT_CORE_API Point3f : public Tuple3f { Point3f() = default; DMT_CPU_GPU Point3f(Tuple3f t) : Tuple3f(t) {} explicit DMT_CPU_GPU operator Vector3f(); DMT_CPU_GPU Point3f(float x, float y, float z) : Tuple3f{x, y, z} {} };
    struct DMT_CORE_API Point4i : public Tuple4i { Point4i() = default; DMT_CPU_GPU Point4i(Tuple4i t) : Tuple4i(t) {} explicit DMT_CPU_GPU operator Vector4i(); DMT_CPU_GPU Point4i(int32_t x, int32_t y, int32_t z, int32_t w) : Tuple4i{x, y, z, w} {} };
    struct DMT_CORE_API Point4f : public Tuple4f { Point4f() = default; DMT_CPU_GPU Point4f(Tuple4f t) : Tuple4f(t) {} explicit DMT_CPU_GPU operator Vector4f(); DMT_CPU_GPU Point4f(float x, float y, float z, float w) : Tuple4f{x, y, z, w} {} };

    // https://eater.net/quaternions
    // https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    struct DMT_CORE_API Quaternion : public Tuple4f { Quaternion() = default; DMT_CPU_GPU Quaternion(Tuple4f t) : Tuple4f(t) {} DMT_CPU_GPU Quaternion(float x, float y, float z, float w) : Tuple4f{x, y, z, w } {}};
    // clang-format on

    DMT_CORE_API DMT_CPU_GPU Tuple2f normalize(Tuple2f v);
    DMT_CORE_API DMT_CPU_GPU Tuple3f normalize(Tuple3f v);
    DMT_CORE_API DMT_CPU_GPU Tuple4f normalize(Tuple4f v);

    struct DMT_CORE_API Normal2f : public Tuple2f, public Normalized
    {
        Normal2f() = default;
        DMT_CPU_GPU                  Normal2f(Tuple2f t) : Tuple2f(normalize(t)) {}
        DMT_CPU_GPU                  Normal2f(float x, float y) : Normal2f(Tuple2f{x, y}) {}
        DMT_CPU_GPU inline Vector2f& asVec()
        {
#if defined(__CUDA_ARCH__)
            return *reinterpret_cast<Vector2f*>(this);
#else
            return *std::bit_cast<Vector2f*>(this);
#endif
        }
        DMT_CPU_GPU inline Vector2f const& asVec() const
        {
#if defined(__CUDA_ARCH__)
            return *reinterpret_cast<Vector2f const*>(this);
#else
            return *std::bit_cast<Vector2f const*>(this);
#endif
        }
    };

    struct DMT_CORE_API Normal3f : public Tuple3f, public Normalized
    {
        Normal3f() = default;
        DMT_CPU_GPU                  Normal3f(Tuple3f t) : Tuple3f(normalize(t)) {}
        DMT_CPU_GPU                  Normal3f(float x, float y, float z) : Normal3f(Tuple3f{x, y, z}) {}
        DMT_CPU_GPU inline Vector3f& asVec()
        {
#if defined(__CUDA_ARCH__)
            return *reinterpret_cast<Vector3f*>(this);
#else
            return *std::bit_cast<Vector3f*>(this);
#endif
        }
        DMT_CPU_GPU inline Vector3f const& asVec() const
        {
#if defined(__CUDA_ARCH__)
            return *reinterpret_cast<Vector3f const*>(this);
#else
            return *std::bit_cast<Vector3f const*>(this);
#endif
        }
    };

    DMT_CORE_API DMT_CPU_GPU Normal2f normalFrom(Vector2f v);
    DMT_CORE_API DMT_CPU_GPU Normal3f normalFrom(Vector3f v);

    /**
     * Triplet of orthonormal vectors, representing a coordinate system
     * Note: Because of their nature, the 3x3 Matrix representing the transformation from World Space to The Frame's Space
     * is a orthonormal matrix, meaning its inverse is equal to its transpose. Since normals are applied the inverse transpose of a given matrix
     * the staring orthonormal matrix is already its own inverse transpose
     */
    struct DMT_CORE_API Frame
    {
        Frame() = default;
        DMT_CPU_GPU              Frame(Normal3f x, Normal3f y, Normal3f z);
        DMT_CPU_GPU static Frame fromXZ(Normal3f x, Normal3f z);
        DMT_CPU_GPU static Frame fromXY(Normal3f x, Normal3f y);
        DMT_CPU_GPU static Frame fromZ(Normal3f z);

        DMT_CPU_GPU static inline Frame fromXZ(Vector3f x, Vector3f z) { return fromXZ(normalFrom(x), normalFrom(z)); }
        DMT_CPU_GPU static inline Frame fromXY(Vector3f x, Vector3f y) { return fromXY(normalFrom(x), normalFrom(y)); }
        DMT_CPU_GPU static inline Frame fromZ(Vector3f z) { return fromZ(normalFrom(z)); }

        DMT_CPU_GPU Vector3f toLocal(Vector3f v) const;
        DMT_CPU_GPU Normal3f toLocal(Normal3f n) const;

        DMT_CPU_GPU Vector3f fromLocal(Vector3f v) const;
        DMT_CPU_GPU Normal3f fromLocal(Normal3f n) const;

        Normal3f xAxis{{1, 0, 0}}, yAxis{{0, 1, 0}}, zAxis{{0, 0, 1}};
    };
    // TODO compressed normal, half floats


    // Vector Types: Fundamental Operations ---------------------------------------------------------------------------
    DMT_CORE_API DMT_CPU_GPU bool operator==(Point2i a, Point2i b);
    DMT_CORE_API DMT_CPU_GPU bool operator==(Point3i a, Point3i b);
    DMT_CORE_API DMT_CPU_GPU bool operator==(Point4i a, Point4i b);

    DMT_CORE_API DMT_CPU_GPU Point2i    operator+(Point2i a, Vector2i b);
    DMT_CORE_API DMT_CPU_GPU Point2f    operator+(Point2f a, Vector2f b);
    DMT_CORE_API DMT_CPU_GPU Point3i    operator+(Point3i a, Vector3i b);
    DMT_CORE_API DMT_CPU_GPU Point3f    operator+(Point3f a, Vector3f b);
    DMT_CORE_API DMT_CPU_GPU Point4i    operator+(Point4i a, Vector4i b);
    DMT_CORE_API DMT_CPU_GPU Point4f    operator+(Point4f a, Vector4f b);
    DMT_CORE_API DMT_CPU_GPU Vector2i   operator+(Vector2i a, Vector2i b);
    DMT_CORE_API DMT_CPU_GPU Vector2f   operator+(Vector2f a, Vector2f b);
    DMT_CORE_API DMT_CPU_GPU Vector3i   operator+(Vector3i a, Vector3i b);
    DMT_CORE_API DMT_CPU_GPU Vector3f   operator+(Vector3f a, Vector3f b);
    DMT_CORE_API DMT_CPU_GPU Vector4i   operator+(Vector4i a, Vector4i b);
    DMT_CORE_API DMT_CPU_GPU Vector4f   operator+(Vector4f a, Vector4f b);
    DMT_CORE_API DMT_CPU_GPU Normal2f   operator+(Normal2f a, Normal2f b);
    DMT_CORE_API DMT_CPU_GPU Normal3f   operator+(Normal3f a, Normal3f b);
    DMT_CORE_API DMT_CPU_GPU Quaternion operator+(Quaternion a, Quaternion b);

    DMT_CORE_API DMT_CPU_GPU Vector2i   operator-(Point2i a, Point2i b);
    DMT_CORE_API DMT_CPU_GPU Vector2f   operator-(Point2f a, Point2f b);
    DMT_CORE_API DMT_CPU_GPU Vector3i   operator-(Point3i a, Point3i b);
    DMT_CORE_API DMT_CPU_GPU Vector3f   operator-(Point3f a, Point3f b);
    DMT_CORE_API DMT_CPU_GPU Vector4i   operator-(Point4i a, Point4i b);
    DMT_CORE_API DMT_CPU_GPU Vector4f   operator-(Point4f a, Point4f b);
    DMT_CORE_API DMT_CPU_GPU Vector2i   operator-(Point2i a, Vector2i b);
    DMT_CORE_API DMT_CPU_GPU Vector2f   operator-(Point2f a, Vector2f b);
    DMT_CORE_API DMT_CPU_GPU Vector3i   operator-(Point3i a, Vector3i b);
    DMT_CORE_API DMT_CPU_GPU Vector3f   operator-(Point3f a, Vector3f b);
    DMT_CORE_API DMT_CPU_GPU Vector4i   operator-(Point4i a, Vector4i b);
    DMT_CORE_API DMT_CPU_GPU Vector4f   operator-(Point4f a, Vector4f b);
    DMT_CORE_API DMT_CPU_GPU Vector2i   operator-(Vector2i a, Vector2i b);
    DMT_CORE_API DMT_CPU_GPU Vector2f   operator-(Vector2f a, Vector2f b);
    DMT_CORE_API DMT_CPU_GPU Vector3i   operator-(Vector3i a, Vector3i b);
    DMT_CORE_API DMT_CPU_GPU Vector3f   operator-(Vector3f a, Vector3f b);
    DMT_CORE_API DMT_CPU_GPU Vector4i   operator-(Vector4i a, Vector4i b);
    DMT_CORE_API DMT_CPU_GPU Vector4f   operator-(Vector4f a, Vector4f b);
    DMT_CORE_API DMT_CPU_GPU Normal2f   operator-(Normal2f a, Normal2f b);
    DMT_CORE_API DMT_CPU_GPU Normal3f   operator-(Normal3f a, Normal3f b);
    DMT_CORE_API DMT_CPU_GPU Quaternion operator-(Quaternion a, Quaternion b);

    DMT_CORE_API DMT_CPU_GPU Vector2i   operator-(Point2i v);
    DMT_CORE_API DMT_CPU_GPU Vector2f   operator-(Point2f v);
    DMT_CORE_API DMT_CPU_GPU Vector3i   operator-(Point3i v);
    DMT_CORE_API DMT_CPU_GPU Vector3f   operator-(Point3f v);
    DMT_CORE_API DMT_CPU_GPU Vector4i   operator-(Point4i v);
    DMT_CORE_API DMT_CPU_GPU Vector4f   operator-(Point4f v);
    DMT_CORE_API DMT_CPU_GPU Vector2i   operator-(Vector2i v);
    DMT_CORE_API DMT_CPU_GPU Vector2f   operator-(Vector2f v);
    DMT_CORE_API DMT_CPU_GPU Vector3i   operator-(Vector3i v);
    DMT_CORE_API DMT_CPU_GPU Vector3f   operator-(Vector3f v);
    DMT_CORE_API DMT_CPU_GPU Vector4i   operator-(Vector4i v);
    DMT_CORE_API DMT_CPU_GPU Vector4f   operator-(Vector4f v);
    DMT_CORE_API DMT_CPU_GPU Normal2f   operator-(Normal2f v);
    DMT_CORE_API DMT_CPU_GPU Normal3f   operator-(Normal3f v);
    DMT_CORE_API DMT_CPU_GPU Quaternion operator-(Quaternion q);

    DMT_CORE_API DMT_CPU_GPU Vector2i   operator*(Vector2i a, Vector2i b);
    DMT_CORE_API DMT_CPU_GPU Vector2f   operator*(Vector2f a, Vector2f b);
    DMT_CORE_API DMT_CPU_GPU Vector3i   operator*(Vector3i a, Vector3i b);
    DMT_CORE_API DMT_CPU_GPU Vector3f   operator*(Vector3f a, Vector3f b);
    DMT_CORE_API DMT_CPU_GPU Vector4i   operator*(Vector4i a, Vector4i b);
    DMT_CORE_API DMT_CPU_GPU Vector4f   operator*(Vector4f a, Vector4f b);
    DMT_CORE_API DMT_CPU_GPU Quaternion operator*(Quaternion a, Quaternion b);

    DMT_CORE_API DMT_CPU_GPU Vector2i operator/(Vector2i a, Vector2i b);
    DMT_CORE_API DMT_CPU_GPU Vector2f operator/(Vector2f a, Vector2f b);
    DMT_CORE_API DMT_CPU_GPU Vector3i operator/(Vector3i a, Vector3i b);
    DMT_CORE_API DMT_CPU_GPU Vector3f operator/(Vector3f a, Vector3f b);
    DMT_CORE_API DMT_CPU_GPU Vector4i operator/(Vector4i a, Vector4i b);
    DMT_CORE_API DMT_CPU_GPU Vector4f operator/(Vector4f a, Vector4f b);

    DMT_CORE_API DMT_CPU_GPU Point2i&    operator+=(Point2i& a, Vector2i b);
    DMT_CORE_API DMT_CPU_GPU Point2f&    operator+=(Point2f& a, Vector2f b);
    DMT_CORE_API DMT_CPU_GPU Point3i&    operator+=(Point3i& a, Vector3i b);
    DMT_CORE_API DMT_CPU_GPU Point3f&    operator+=(Point3f& a, Vector3f b);
    DMT_CORE_API DMT_CPU_GPU Point4i&    operator+=(Point4i& a, Vector4i b);
    DMT_CORE_API DMT_CPU_GPU Point4f&    operator+=(Point4f& a, Vector4f b);
    DMT_CORE_API DMT_CPU_GPU Vector2i&   operator+=(Vector2i& a, Vector2i b);
    DMT_CORE_API DMT_CPU_GPU Vector2f&   operator+=(Vector2f& a, Vector2f b);
    DMT_CORE_API DMT_CPU_GPU Vector3i&   operator+=(Vector3i& a, Vector3i b);
    DMT_CORE_API DMT_CPU_GPU Vector3f&   operator+=(Vector3f& a, Vector3f b);
    DMT_CORE_API DMT_CPU_GPU Vector4i&   operator+=(Vector4i& a, Vector4i b);
    DMT_CORE_API DMT_CPU_GPU Vector4f&   operator+=(Vector4f& a, Vector4f b);
    DMT_CORE_API DMT_CPU_GPU Normal2f&   operator+=(Normal2f& a, Normal2f b);
    DMT_CORE_API DMT_CPU_GPU Normal3f&   operator+=(Normal3f& a, Normal3f b);
    DMT_CORE_API DMT_CPU_GPU Quaternion& operator+=(Quaternion& a, Quaternion b);

    DMT_CORE_API DMT_CPU_GPU Vector2i&   operator-=(Vector2i& a, Vector2i b);
    DMT_CORE_API DMT_CPU_GPU Vector2f&   operator-=(Vector2f& a, Vector2f b);
    DMT_CORE_API DMT_CPU_GPU Vector3i&   operator-=(Vector3i& a, Vector3i b);
    DMT_CORE_API DMT_CPU_GPU Vector3f&   operator-=(Vector3f& a, Vector3f b);
    DMT_CORE_API DMT_CPU_GPU Vector4i&   operator-=(Vector4i& a, Vector4i b);
    DMT_CORE_API DMT_CPU_GPU Vector4f&   operator-=(Vector4f& a, Vector4f b);
    DMT_CORE_API DMT_CPU_GPU Normal2f&   operator-=(Normal2f& a, Normal2f b);
    DMT_CORE_API DMT_CPU_GPU Normal3f&   operator-=(Normal3f& a, Normal3f b);
    DMT_CORE_API DMT_CPU_GPU Quaternion& operator-=(Quaternion& a, Quaternion b);

    DMT_CORE_API DMT_CPU_GPU Vector2i&   operator*=(Vector2i& a, Vector2i b);
    DMT_CORE_API DMT_CPU_GPU Vector2f&   operator*=(Vector2f& a, Vector2f b);
    DMT_CORE_API DMT_CPU_GPU Vector3i&   operator*=(Vector3i& a, Vector3i b);
    DMT_CORE_API DMT_CPU_GPU Vector3f&   operator*=(Vector3f& a, Vector3f b);
    DMT_CORE_API DMT_CPU_GPU Vector4i&   operator*=(Vector4i& a, Vector4i b);
    DMT_CORE_API DMT_CPU_GPU Vector4f&   operator*=(Vector4f& a, Vector4f b);
    DMT_CORE_API DMT_CPU_GPU Quaternion& operator*=(Quaternion& a, Quaternion b);

    DMT_CORE_API DMT_CPU_GPU Vector2i& operator/=(Vector2i& a, Vector2i b);
    DMT_CORE_API DMT_CPU_GPU Vector2f& operator/=(Vector2f& a, Vector2f b);
    DMT_CORE_API DMT_CPU_GPU Vector3i& operator/=(Vector3i& a, Vector3i b);
    DMT_CORE_API DMT_CPU_GPU Vector3f& operator/=(Vector3f& a, Vector3f b);
    DMT_CORE_API DMT_CPU_GPU Vector4i& operator/=(Vector4i& a, Vector4i b);
    DMT_CORE_API DMT_CPU_GPU Vector4f& operator/=(Vector4f& a, Vector4f b);


    // Vector Types: Common Inline Methods ----------------------------------------------------------------------------
#if __cplusplus >= 202002L
    template <Vector T>
#else
    template <typename T>
#endif
    DMT_CPU_GPU inline T bcast(typename T::value_type t)
    {
        T ret;
        for (int32_t i = 0; i < T::numComponents(); ++i)
            ret[i] = t;
        return ret;
    }

#if __cplusplus >= 202002L
    template <VectorScalable T>
#else
    template <typename T>
#endif
    DMT_CPU_GPU inline T operator*(T v, typename T::value_type k)
    {
        T ret;
        for (int32_t i = 0; i < T::numComponents(); ++i)
            ret[i] = k * v[i];
        return ret;
    }
#if __cplusplus >= 202002L
    template <VectorScalable T>
#else
    template <typename T>
#endif
    DMT_CPU_GPU inline T operator*(typename T::value_type k, T v)
    {
        T ret;
        for (int32_t i = 0; i < T::numComponents(); ++i)
            ret[i] = k * v[i];
        return ret;
    }
#if __cplusplus >= 202002L
    template <VectorScalable T>
#else
    template <typename T>
#endif
    DMT_CPU_GPU inline T operator/(T v, typename T::value_type k)
    {
        T ret;
        for (int32_t i = 0; i < T::numComponents(); ++i)
            ret[i] = v[i] / k;
        return ret;
    }
#if __cplusplus >= 202002L
    template <VectorScalable T>
#else
    template <typename T>
#endif
    DMT_CPU_GPU inline T operator/(typename T::value_type k, T v)
    {
        T ret;
        for (int32_t i = 0; i < T::numComponents(); ++i)
            ret[i] = k / v[i];
        return ret;
    }

#if __cplusplus >= 202002L
    template <VectorScalable T>
#else
    template <typename T>
#endif
    DMT_CPU_GPU inline T& operator*=(T& v, typename T::value_type k)
    {
        for (int32_t i = 0; i < T::numComponents(); ++i)
            v[i] *= k;
        return v;
    }
#if __cplusplus >= 202002L
    template <VectorScalable T>
#else
    template <typename T>
#endif
    DMT_CPU_GPU inline T& operator/=(T& v, typename T::value_type k)
    {
        for (int32_t i = 0; i < T::numComponents(); ++i)
            v[i] /= k;
        return v;
    }

    // Vector Types: Component Operations -----------------------------------------------------------------------------

#if __cplusplus >= 202002L
    template <Vector T>
#else
    template <typename T>
#endif
    DMT_CPU_GPU inline int32_t maxComponentIndex(T v)
    {
        int32_t max = 0;
        for (int32_t i = 1; i < T::numComponents(); ++i)
            if (v[max] < v[i])
                max = i;

        return max;
    }

#if __cplusplus >= 202002L
    template <Vector T>
#else
    template <typename T>
#endif
    DMT_CPU_GPU inline typename T::value_type maxComponent(T v)
    {
        int32_t max = 0;
        for (int32_t i = 1; i < T::numComponents(); ++i)
            if (v[max] < v[i])
                max = i;

        return v[max];
    }

#if __cplusplus >= 202002L
    template <Vector T>
#else
    template <typename T>
#endif
    DMT_CPU_GPU inline int32_t minComponentIndex(T v)
    {
        int32_t min = 0;
        for (int32_t i = 1; i < T::numComponents(); ++i)
            if (v[min] > v[i])
                min = i;

        return min;
    }

#if __cplusplus >= 202002L
    template <Vector T>
#else
    template <typename T>
#endif
    DMT_CPU_GPU inline typename T::value_type minComponent(T v)
    {
        int32_t min = 0;
        for (int32_t i = 1; i < T::numComponents(); ++i)
            if (v[min] > v[i])
                min = i;

        return v[min];
    }

#if __cplusplus >= 202002L
    template <Vector T>
        requires(T::numComponents() == 2)
#else
    template <typename T>
#endif
    DMT_CPU_GPU inline T permute(T v, int32_t i0, int32_t i1)
    {
        T ret;
        ret[0] = v[i0];
        ret[1] = v[i1];
        return ret;
    }
#if __cplusplus >= 202002L
    template <Vector T>
        requires(T::numComponents() == 3)
#else
    template <typename T>
#endif
    DMT_CPU_GPU inline T permute(T v, int32_t i0, int32_t i1, int32_t i2)
    {
        T ret;
        ret[0] = v[i0];
        ret[1] = v[i1];
        ret[2] = v[i2];
        return ret;
    }
#if __cplusplus >= 202002L
    template <Vector T>
        requires(T::numComponents() == 4)
#else
    template <typename T>
#endif
    DMT_CPU_GPU inline T permute(T v, int32_t i0, int32_t i1, int32_t i2, int32_t i3)
    {
        T ret;
        ret[0] = v[i0];
        ret[1] = v[i1];
        ret[2] = v[i2];
        ret[3] = v[i3];
        return ret;
    }

#if __cplusplus >= 202002L
    template <Vector T>
#else
    template <typename T>
#endif
    DMT_CPU_GPU inline typename T::value_type hprod(T v)
    {
        typename T::value_type ret = v[0];
        for (int32_t i = 1; i < T::numComponents(); ++i)
            ret *= v[i];
        return ret;
    }

#if __cplusplus >= 202002L
    template <Vector T>
        requires std::is_floating_point_v<typename T::value_type>
#else
    template <typename T>
#endif
    DMT_CPU_GPU inline bool hasNaN(T v)
    {
        for (int32_t i = 0; i < T::numComponents(); ++i)
        {
#if defined(__CUDA_ARCH__)
            if (isnan(v[i]))
#else
            if (std::isnan(v[i]))
#endif
                return false;
        }
        return true;
    }

    // Vector Types: Generic Tuple Operations -------------------------------------------------------------------------
    // If any of these are used to initialize a Normal, it has to be manually normalized
    DMT_CORE_API DMT_CPU_GPU Tuple2f abs(Tuple2f v);
    DMT_CORE_API DMT_CPU_GPU Tuple2i abs(Tuple2i v);
    DMT_CORE_API DMT_CPU_GPU Tuple3f abs(Tuple3f v);
    DMT_CORE_API DMT_CPU_GPU Tuple3i abs(Tuple3i v);
    DMT_CORE_API DMT_CPU_GPU Tuple4f abs(Tuple4f v);
    DMT_CORE_API DMT_CPU_GPU Tuple4i abs(Tuple4i v);

    DMT_CORE_API DMT_CPU_GPU Tuple2f ceil(Tuple2f v);
    DMT_CORE_API DMT_CPU_GPU Tuple3f ceil(Tuple3f v);
    DMT_CORE_API DMT_CPU_GPU Tuple4f ceil(Tuple4f v);

    DMT_CORE_API DMT_CPU_GPU Tuple2f floor(Tuple2f v);
    DMT_CORE_API DMT_CPU_GPU Tuple3f floor(Tuple3f v);
    DMT_CORE_API DMT_CPU_GPU Tuple4f floor(Tuple4f v);

    DMT_CORE_API DMT_CPU_GPU Tuple2f sqrt(Tuple2f v);
    DMT_CORE_API DMT_CPU_GPU Tuple3f sqrt(Tuple3f v);
    DMT_CORE_API DMT_CPU_GPU Tuple4f sqrt(Tuple4f v);

    DMT_CORE_API DMT_CPU_GPU Tuple2f lerp(float t, Tuple2f zero, Tuple2f one);
    DMT_CORE_API DMT_CPU_GPU Tuple3f lerp(float t, Tuple3f zero, Tuple3f one);
    DMT_CORE_API DMT_CPU_GPU Tuple4f lerp(float t, Tuple4f zero, Tuple4f one);
    //to move in a math clas
    DMT_CPU_GPU inline float lerp(float x, float a, float b) { return (1 - x) * a + x * b; }

    DMT_CORE_API DMT_CPU_GPU Tuple2f fma(Tuple2f mult0, Tuple2f mult1, Tuple2f add);
    DMT_CORE_API DMT_CPU_GPU Tuple2i fma(Tuple2i mult0, Tuple2i mult1, Tuple2i add);
    DMT_CORE_API DMT_CPU_GPU Tuple3f fma(Tuple3f mult0, Tuple3f mult1, Tuple3f add);
    DMT_CORE_API DMT_CPU_GPU Tuple3i fma(Tuple3i mult0, Tuple3i mult1, Tuple3i add);
    DMT_CORE_API DMT_CPU_GPU Tuple4f fma(Tuple4f mult0, Tuple4f mult1, Tuple4f add);
    DMT_CORE_API DMT_CPU_GPU Tuple4i fma(Tuple4i mult0, Tuple4i mult1, Tuple4i add);

    DMT_CORE_API DMT_CPU_GPU Tuple2f min(Tuple2f a, Tuple2f b);
    DMT_CORE_API DMT_CPU_GPU Tuple2i min(Tuple2i a, Tuple2i b);
    DMT_CORE_API DMT_CPU_GPU Tuple3f min(Tuple3f a, Tuple3f b);
    DMT_CORE_API DMT_CPU_GPU Tuple3i min(Tuple3i a, Tuple3i b);
    DMT_CORE_API DMT_CPU_GPU Tuple4f min(Tuple4f a, Tuple4f b);
    DMT_CORE_API DMT_CPU_GPU Tuple4i min(Tuple4i a, Tuple4i b);

    DMT_CORE_API DMT_CPU_GPU Tuple2f max(Tuple2f a, Tuple2f b);
    DMT_CORE_API DMT_CPU_GPU Tuple2i max(Tuple2i a, Tuple2i b);
    DMT_CORE_API DMT_CPU_GPU Tuple3f max(Tuple3f a, Tuple3f b);
    DMT_CORE_API DMT_CPU_GPU Tuple3i max(Tuple3i a, Tuple3i b);
    DMT_CORE_API DMT_CPU_GPU Tuple4f max(Tuple4f a, Tuple4f b);
    DMT_CORE_API DMT_CPU_GPU Tuple4i max(Tuple4i a, Tuple4i b);

    DMT_CORE_API DMT_CPU_GPU bool near(Tuple2f a, Tuple2f b, float tolerance = fl::eqTol());
    DMT_CORE_API DMT_CPU_GPU bool near(Tuple2i a, Tuple2i b);
    DMT_CORE_API DMT_CPU_GPU bool near(Tuple3f a, Tuple3f b, float tolerance = fl::eqTol());
    DMT_CORE_API DMT_CPU_GPU bool near(Tuple3i a, Tuple3i b);
    DMT_CORE_API DMT_CPU_GPU bool near(Tuple4f a, Tuple4f b, float tolerance = fl::eqTol());
    DMT_CORE_API DMT_CPU_GPU bool near(Tuple4i a, Tuple4i b);

    DMT_CORE_API DMT_CPU_GPU Tuple2f::value_type dot(Tuple2f a, Tuple2f b);
    DMT_CORE_API DMT_CPU_GPU Tuple3f::value_type dot(Tuple3f a, Tuple3f b);
    DMT_CORE_API DMT_CPU_GPU Tuple4f::value_type dot(Tuple4f a, Tuple4f b);

    DMT_CORE_API DMT_CPU_GPU Tuple2f::value_type absDot(Tuple2f a, Tuple2f b);
    DMT_CORE_API DMT_CPU_GPU Tuple3f::value_type absDot(Tuple3f a, Tuple3f b);
    DMT_CORE_API DMT_CPU_GPU Tuple4f::value_type absDot(Tuple4f a, Tuple4f b);

    DMT_CORE_API DMT_CPU_GPU Tuple3f cross(Tuple3f a, Tuple3f b);

    DMT_CORE_API DMT_CPU_GPU Tuple2f::value_type normL2(Tuple2f v);
    DMT_CORE_API DMT_CPU_GPU Tuple3f::value_type normL2(Tuple3f v);
    DMT_CORE_API DMT_CPU_GPU Tuple4f::value_type normL2(Tuple4f v);

    DMT_CORE_API DMT_CPU_GPU Tuple2f operator>(Tuple2f v0, Tuple2f v1);
    DMT_CORE_API DMT_CPU_GPU Tuple3f operator>(Tuple3f v0, Tuple3f v1);
    DMT_CORE_API DMT_CPU_GPU Tuple2f operator>(Tuple2f v0, Tuple2f v1);
    DMT_CORE_API DMT_CPU_GPU Tuple3f operator<(Tuple3f v0, Tuple3f v1);
    DMT_CORE_API DMT_CPU_GPU Tuple4f operator<(Tuple4f v0, Tuple4f v1);
    DMT_CORE_API DMT_CPU_GPU Tuple4f operator<(Tuple4f v0, Tuple4f v1);
    DMT_CORE_API DMT_CPU_GPU Tuple2f operator>=(Tuple2f v0, Tuple2f v1);
    DMT_CORE_API DMT_CPU_GPU Tuple3f operator>=(Tuple3f v0, Tuple3f v1);
    DMT_CORE_API DMT_CPU_GPU Tuple2f operator>=(Tuple2f v0, Tuple2f v1);
    DMT_CORE_API DMT_CPU_GPU Tuple3f operator<=(Tuple3f v0, Tuple3f v1);
    DMT_CORE_API DMT_CPU_GPU Tuple4f operator<=(Tuple4f v0, Tuple4f v1);
    DMT_CORE_API DMT_CPU_GPU Tuple4f operator<=(Tuple4f v0, Tuple4f v1);
    DMT_CORE_API DMT_CPU_GPU Tuple2f operator==(Tuple2f v0, Tuple2f v1);
    DMT_CORE_API DMT_CPU_GPU Tuple3f operator==(Tuple3f v0, Tuple3f v1);
    DMT_CORE_API DMT_CPU_GPU Tuple2f operator==(Tuple2f v0, Tuple2f v1);
    DMT_CORE_API DMT_CPU_GPU Tuple3f operator!=(Tuple3f v0, Tuple3f v1);
    DMT_CORE_API DMT_CPU_GPU Tuple4f operator!=(Tuple4f v0, Tuple4f v1);
    DMT_CORE_API DMT_CPU_GPU Tuple4f operator!=(Tuple4f v0, Tuple4f v1);

    DMT_CORE_API DMT_CPU_GPU bool all(Tuple2f v);
    DMT_CORE_API DMT_CPU_GPU bool all(Tuple3f v);
    DMT_CORE_API DMT_CPU_GPU bool all(Tuple4f v);

    DMT_CORE_API DMT_CPU_GPU bool any(Tuple2f v);
    DMT_CORE_API DMT_CPU_GPU bool any(Tuple3f v);
    DMT_CORE_API DMT_CPU_GPU bool any(Tuple4f v);

    DMT_CORE_API DMT_CPU_GPU Tuple2f::value_type distanceL2(Tuple2f a, Tuple2f b);
    DMT_CORE_API DMT_CPU_GPU Tuple3f::value_type distanceL2(Tuple3f a, Tuple3f b);
    DMT_CORE_API DMT_CPU_GPU Tuple4f::value_type distanceL2(Tuple4f a, Tuple4f b);

    DMT_CORE_API DMT_CPU_GPU Tuple2f::value_type dotSelf(Tuple2f v);
    DMT_CORE_API DMT_CPU_GPU Tuple3f::value_type dotSelf(Tuple3f v);
    DMT_CORE_API DMT_CPU_GPU Tuple4f::value_type dotSelf(Tuple4f v);

    // Vector Types: Geometric Functions ------------------------------------------------------------------------------
#if __cplusplus >= 202002L
    template <VectorNormalized N, Vector V>
        requires(std::is_same_v<typename N::value_type, typename V::value_type> && N::numComponents() == V::numComponents())
#else
    template <typename N, typename V>
#endif
    DMT_CPU_GPU inline N faceForward(N direction, V vector)
    {
        return (dot(direction, vector) < 0.f) ? -direction : direction;
    }

    DMT_CORE_API DMT_CPU_GPU float angleBetween(Normal3f a, Normal3f b);
    DMT_CORE_API DMT_CPU_GPU float angleBetween(Quaternion a, Quaternion b);

#if __cplusplus >= 202002L
    template <Vector V>
#else
    template <typename V>
#endif
    DMT_CPU_GPU inline V gramSchmidt(V v, V w)
    {
        V dt = {dot(v, w)};
        return v * dt * w;
    }

    DMT_CORE_API DMT_CPU_GPU Frame coordinateSystem(Normal3f xAxis);
    DMT_CORE_API DMT_CPU_GPU void  gramSchmidt(Vector3f n, Vector3f* a, Vector3f* b);

    DMT_CORE_API DMT_CPU_GPU Quaternion slerp(float t, Quaternion zero, Quaternion one);
    inline float Dot(Vector3f const& a, Vector3f const& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

    // Vector Types: Spherical Geometry Functions ---------------------------------------------------------------------
    DMT_CORE_API DMT_CPU_GPU float    sphericalTriangleArea(Vector3f edge0, Vector3f edge1, Vector3f edge2);
    DMT_CORE_API DMT_CPU_GPU float    sphericalQuadArea(Vector3f edge0, Vector3f edge1, Vector3f edge2, Vector3f edge3);
    DMT_CORE_API DMT_CPU_GPU Vector3f sphericalDirection(float sinTheta, float cosTheta, float phi);
    DMT_CORE_API DMT_CPU_GPU float    sphericalTheta(Vector3f v);
    DMT_CORE_API DMT_CPU_GPU float    sphericalPhi(Vector3f v);
    DMT_CORE_API DMT_CPU_GPU float    cosTheta(Vector3f v);
    DMT_CORE_API DMT_CPU_GPU float    cos2Theta(Vector3f v);
    DMT_CORE_API DMT_CPU_GPU float    absCosTheta(Vector3f v);
    DMT_CORE_API DMT_CPU_GPU float    sinTheta(Vector3f v);
    DMT_CORE_API DMT_CPU_GPU float    sin2Theta(Vector3f v);
    DMT_CORE_API DMT_CPU_GPU float    tanTheta(Vector3f v);
    DMT_CORE_API DMT_CPU_GPU float    tan2Theta(Vector3f v);
    DMT_CORE_API DMT_CPU_GPU float    cosPhi(Vector3f v);
    DMT_CORE_API DMT_CPU_GPU float    sinPhi(Vector3f v);
    DMT_CORE_API DMT_CPU_GPU float    cosDPhi(Vector3f wa, Vector3f wb);
    DMT_CORE_API DMT_CPU_GPU bool     sameHemisphere(Vector3f w, Normal3f ap);

    // Vector Types: Axis Aligned Bounding Boxes ----------------------------------------------------------------------
    enum class DMT_CORE_API EBoundsCorner : int32_t
    {
        eBottom  = 0,
        eLeft    = eBottom,
        eBack    = eLeft,
        eTop     = 0b100,
        eRight   = 0b001,
        eForward = 0b010,
        // 8 corners
        eRightTopForward    = eRight | eTop | eForward,
        eRightTopBack       = eRight | eTop | eBack,
        eRightBottomForward = eRight | eBottom | eForward,
        eRightBottomBack    = eRight | eBottom | eBack,
        eLeftTopForward     = eLeft | eTop | eForward,
        eLeftTopBack        = eLeft | eTop | eBack,
        eLeftBottomForward  = eLeft | eBottom | eForward,
        eLeftBottomBack     = eLeft | eBottom | eBack,
    };

    enum class DMT_CORE_API EBoundsCorner2 : int32_t
    {
        eLeft   = 0b00,
        eBottom = eLeft,
        eTop    = 0b10,
        eRight  = 0b01,
        // 4 corners
        eRightTop    = eRight | eTop,
        eRightBottom = eRight | eBottom,
        eLeftTop     = eLeft | eTop,
        eLeftBottom  = eLeft | eBottom,
    };

    struct DMT_CORE_API Bounds3f
    {
        DMT_CPU_GPU Point3f&       operator[](int32_t i);
        DMT_CPU_GPU Point3f const& operator[](int32_t i) const;
        DMT_CPU_GPU Point3f        corner(EBoundsCorner corner) const;
        DMT_CPU_GPU Vector3f       diagonal() const;
        DMT_CPU_GPU float          surfaceArea() const;
        DMT_CPU_GPU float          volume() const;
        DMT_CPU_GPU int32_t        maxDimention() const;
        DMT_CPU_GPU Point3f        lerp(Point3f t) const;
        DMT_CPU_GPU Point3f        centroid() const { return .5f * pMin + .5f * pMax; }
        DMT_CPU_GPU Vector3f       offset(Point3f p) const;
        DMT_CPU_GPU void           boundingSphere(Point3f& outCenter, float& outRadius) const;
        DMT_CPU_GPU bool           isEmpty() const;
        DMT_CPU_GPU bool           isDegenerate() const;
        DMT_CPU_GPU bool           operator==(Bounds3f const& that) const;
        DMT_CPU_GPU bool           intersectP(Point3f             o,
                                              Vector3f            d,
                                              float               tMax = std::numeric_limits<float>::infinity(),
                                              float* DMT_RESTRICT hit0 = nullptr,
                                              float* DMT_RESTRICT hit1 = nullptr) const;
        DMT_CPU_GPU bool intersectP(Point3f o, Vector3f d, float tMax, Vector3f invDir, int32_t dirIsNeg[3]) const;

        Point3f pMin;
        Point3f pMax;
    };

    DMT_CORE_API DMT_CPU_GPU Bounds3f makeBounds(Point3f p0, Point3f p1);
    DMT_CORE_API DMT_CPU_GPU Bounds3f bbEmpty();
    DMT_CORE_API DMT_CPU_GPU bool     inside(Point3f p, Bounds3f const& b);
    DMT_CORE_API DMT_CPU_GPU Bounds3f bbUnion(Bounds3f const& a, Bounds3f const& b);
    DMT_CORE_API DMT_CPU_GPU Bounds3f bbUnion(Bounds3f const& b, Point3f p);

    struct DMT_CORE_API Bounds2f
    {
        DMT_CPU_GPU Point2f&       operator[](int32_t i);
        DMT_CPU_GPU Point2f const& operator[](int32_t i) const;
        DMT_CPU_GPU Point2f        corner(EBoundsCorner2 corner) const;
        DMT_CPU_GPU Vector2f       diagonal() const;
        DMT_CPU_GPU float          surfaceArea() const;
        DMT_CPU_GPU float          volume() const;
        DMT_CPU_GPU int32_t        maxDimention() const;
        DMT_CPU_GPU Point2f        lerp(Point2f t) const;
        DMT_CPU_GPU Point2f        centroid() const { return .5f * pMin + .5f * pMax; }
        DMT_CPU_GPU Vector2f       offset(Point2f p) const;
        DMT_CPU_GPU void           boundingCircle(Point2f& outCenter, float& outRadius) const;
        DMT_CPU_GPU bool           isEmpty() const;
        DMT_CPU_GPU bool           isDegenerate() const;
        DMT_CPU_GPU bool           operator==(Bounds2f const& that) const;

        Point2f pMin;
        Point2f pMax;
    };

    DMT_CORE_API DMT_CPU_GPU Bounds2f makeBounds(Point2f p0, Point2f p1);
    DMT_CORE_API DMT_CPU_GPU Bounds2f bbEmpty2();
    DMT_CORE_API DMT_CPU_GPU bool     inside(Point2f p, Bounds2f const& b);
    DMT_CORE_API DMT_CPU_GPU Bounds2f bbUnion(Bounds2f const& a, Bounds2f const& b);
    DMT_CORE_API DMT_CPU_GPU Bounds2f bbUnion(Bounds2f const& b, Point2f p);


    struct DMT_CORE_API Bounds2i
    {
        DMT_CPU_GPU Bounds2i()
        {
            int32_t minNum = std::numeric_limits<int32_t>::lowest();
            int32_t maxNum = std::numeric_limits<int32_t>::max();
            pMin           = Point2i{{maxNum, maxNum}};
            pMax           = Point2i{{minNum, minNum}};
        }
        DMT_CPU_GPU explicit Bounds2i(Point2i p) : pMin(p), pMax(p) {}
        DMT_CPU_GPU Bounds2i(Point2i p1, Point2i p2) : pMin(min(p1, p2)), pMax(max(p1, p2)) {}

        DMT_CPU_GPU
        Vector2i diagonal() const { return pMax - pMin; }

        DMT_CPU_GPU int32_t area() const
        {
            Vector2i d = pMax - pMin;
            return d.x * d.y;
        }

        DMT_CPU_GPU bool isEmpty() const { return pMin.x >= pMax.x || pMin.y >= pMax.y; }

        DMT_CPU_GPU bool isDegenerate() const { return pMin.x > pMax.x || pMin.y > pMax.y; }

        DMT_CPU_GPU int maxDimension() const
        {
            Vector2i diag = diagonal();
            if (diag.x > diag.y)
                return 0;
            else
                return 1;
        }
        DMT_CPU_GPU Point2i operator[](int i) const
        {
            assert(i == 0 || i == 1);
            return (i == 0) ? pMin : pMax;
        }

        DMT_CPU_GPU Point2i& operator[](int i)
        {
            assert(i == 0 || i == 1);
            return (i == 0) ? pMin : pMax;
        }

        DMT_CPU_GPU bool operator==(Bounds2i const& b) const { return near(b.pMin, pMin) && near(b.pMax, pMax); }

        DMT_CPU_GPU Point2i corner(int corner) const
        {
            assert(corner >= 0 && corner < 4);
            return Point2i{{(*this)[(corner & 1)].x, (*this)[(corner & 2) ? 1 : 0].y}};
        }

        DMT_CPU_GPU Point2i lerp(Point2f t) const
        {
            return Point2i{{static_cast<int32_t>(dmt::lerp(t.x, pMin.x, pMax.x)),
                            static_cast<int32_t>(dmt::lerp(t.y, pMin.y, pMax.y))}};
        }

        DMT_CPU_GPU Vector2i offset(Point2i p) const
        {
            Vector2i o = p - pMin;
            if (pMax.x > pMin.x)
                o.x /= pMax.x - pMin.x;
            if (pMax.y > pMin.y)
                o.y /= pMax.y - pMin.y;
            return o;
        }

        DMT_CPU_GPU void boundingSphere(Point2i* c, float* rad) const;

        Point2i pMin, pMax;
    };

    DMT_CORE_API DMT_CPU_GPU bool inside(Point2i p, Bounds2i b);

    // Vector Types: Matrix 4x4 ---------------------------------------------------------------------------------------
    struct DMT_CORE_API Index2
    {
        int32_t row, col;
    };

    DMT_CORE_API DMT_CPU_GPU inline constexpr Index2 sym(Index2 i)
    {
        Index2 j{};
        j.row = i.col;
        j.col = i.row;
        return j;
    }

    // Column Major Order
    struct DMT_CORE_API alignas(16) Matrix4f
    {
        // clang-format off
        static DMT_CPU_GPU constexpr Matrix4f zero()
        {
            return {{
                0.f, 0.f, 0.f, 0.f, // column zero
                0.f, 0.f, 0.f, 0.f, // column one
                0.f, 0.f, 0.f, 0.f, // column two
                0.f, 0.f, 0.f, 0.f  // column three
            }};
        }
        static DMT_CPU_GPU constexpr Matrix4f identity()
        {
            return {{
                1.f, 0.f, 0.f, 0.f, // column zero
                0.f, 1.f, 0.f, 0.f, // column one
                0.f, 0.f, 1.f, 0.f, // column two
                0.f, 0.f, 0.f, 1.f  // column three
            }};
        }
        static DMT_CPU_GPU constexpr Matrix4f rowWise(float const arr[16]) {
            return {{
                arr[0], arr[4], arr[8],  arr[12],
                arr[1], arr[5], arr[9],  arr[13],
                arr[2], arr[6], arr[10], arr[14],
                arr[3], arr[7], arr[11], arr[15],
            }};
        }

        // clang-format on
        DMT_CPU_GPU inline float&       operator[](Index2 i) { return m[i.col * 4 + i.row]; }
        DMT_CPU_GPU inline float const& operator[](Index2 i) const { return m[i.col * 4 + i.row]; }
        DMT_CPU_GPU inline Vector4f&    operator[](int32_t i)
        {
#if defined(__CUDA_ARCH__)
            return *reinterpret_cast<Vector4f*>(&m[i * 4]);
#else
            return *std::bit_cast<Vector4f*>(&m[i * 4]);
#endif
        }
        DMT_CPU_GPU inline Vector4f const& operator[](int32_t i) const
        {
#if defined(__CUDA_ARCH__)
            return *reinterpret_cast<Vector4f const*>(&m[i * 4]);
#else
            return *std::bit_cast<Vector4f const*>(&m[i * 4]);
#endif
        }

        alignas(16) float m[16];
    };
#if !defined(__CUDA_ARCH__)
    static_assert(std::is_trivial_v<Matrix4f> && std::is_standard_layout_v<Matrix4f>);
#endif

    struct DMT_CORE_API SVD
    {
        Matrix4f unitary;
        Vector4f singularValues;
        Matrix4f vunitary;
    };

    struct DMT_CORE_API QR
    {
        Matrix4f qOrthogonal;
        Matrix4f rUpper;
    };

    // define a rotation in the plane defined by two axes
    DMT_CORE_API DMT_CPU_GPU Matrix4f givensRotation(int32_t axis0, int32_t axis1, float theta);
    DMT_CORE_API DMT_CPU_GPU QR       qr(Matrix4f const& m, int32_t numIter = 10);
#if !defined(__NVCC__) && !defined(__CUDA_ARCH__)
    DMT_CORE_API DMT_CPU SVD  svd(Matrix4f const& m);
    DMT_CORE_API DMT_CPU bool isSingular(Matrix4f const& m, float tolerance = 1e-6f);
#endif

    DMT_CORE_API DMT_CPU_GPU Matrix4f operator+(Matrix4f const& a, Matrix4f const& b);
    DMT_CORE_API DMT_CPU_GPU Matrix4f operator-(Matrix4f const& a, Matrix4f const& b);
    DMT_CORE_API DMT_CPU_GPU Matrix4f operator*(Matrix4f const& a, Matrix4f const& b);
    DMT_CORE_API DMT_CPU_GPU Matrix4f operator*(float v, Matrix4f const& m);
    DMT_CORE_API DMT_CPU_GPU Matrix4f operator*(Matrix4f const& m, float v);
    DMT_CORE_API DMT_CPU_GPU Matrix4f operator/(Matrix4f const& m, float v);

    DMT_CORE_API DMT_CPU_GPU Matrix4f& operator+=(Matrix4f& a, Matrix4f const& b);
    DMT_CORE_API DMT_CPU_GPU Matrix4f& operator-=(Matrix4f& a, Matrix4f const& b);
    DMT_CORE_API DMT_CPU_GPU Matrix4f& operator*=(Matrix4f& a, Matrix4f const& b);

    DMT_CORE_API DMT_CPU_GPU Matrix4f fromDiag(Tuple4f v);
    DMT_CORE_API DMT_CPU_GPU Matrix4f fromQuat(Quaternion q);

    DMT_CORE_API DMT_CPU_GPU bool     near(Matrix4f const& a, Matrix4f const& b);
    DMT_CORE_API DMT_CPU_GPU float    determinant(Matrix4f const& m);
    DMT_CORE_API DMT_CPU_GPU Matrix4f inverse(Matrix4f const& m);
    DMT_CORE_API DMT_CPU_GPU Matrix4f transpose(Matrix4f const& m);
    DMT_CORE_API DMT_CPU_GPU Vector4f mul(Matrix4f const& m, Vector4f v);
    DMT_CORE_API DMT_CPU_GPU Vector3f mul(Matrix4f const& m, Vector3f const& v);
    DMT_CORE_API DMT_CPU_GPU Normal3f mul(Matrix4f const& m, Normal3f const& v);
    DMT_CORE_API DMT_CPU_GPU Normal3f mulTranspose(Matrix4f const& m, Normal3f const& v);
    DMT_CORE_API DMT_CPU_GPU Point3f  mul(Matrix4f const& m, Point3f const& p);

    // quaternion rotations
    DMT_CORE_API DMT_CPU_GPU Quaternion            fromRadians(float theta, Normal3f axis);
    DMT_CORE_API DMT_CPU_GPU Quaternion            conj(Quaternion quat);
    DMT_CORE_API DMT_CPU_GPU DMT_FASTCALL Vector3f rotate(Quaternion quat, Vector3f v);
    DMT_CORE_API DMT_CPU_GPU DMT_FASTCALL Normal3f rotate(Quaternion quat, Normal3f v);

    // Vector Types: Interval -----------------------------------------------------------------------------------------
    struct DMT_CORE_API Point3fi
    {
        Point3fi() = default;
        DMT_CPU_GPU Point3fi(Point3f);
        DMT_CPU_GPU Point3fi(Point3f value, Vector3f error);

        DMT_CPU_GPU Point3f         midpoint() const;
        DMT_CPU_GPU Vector3f        width() const;
        inline DMT_CPU_GPU Vector3f error() const { return width() / 2; }

        float xLow = 0.f, yLow = 0.f, zLow = 0.f;
        float xHigh = 0.f, yHigh = 0.f, zHigh = 0.f;
    };
    DMT_CORE_API DMT_CPU_GPU Point3fi operator+(Point3fi const& a, Point3fi const& b);
    DMT_CORE_API DMT_CPU_GPU Point3fi operator-(Point3fi const& a, Point3fi const& b);
    DMT_CORE_API DMT_CPU_GPU Point3fi operator*(Point3fi const& a, Point3fi const& b);
    DMT_CORE_API DMT_CPU_GPU Point3fi operator/(Point3fi const& a, Point3fi const& b);
    DMT_CORE_API DMT_CPU_GPU Point3fi mul(Matrix4f const& m, Point3fi const& p);

    // Ray and RayDifferentials ---------------------------------------------------------------------------------------
    // TODO compress direction when you write compressed normal
    struct DMT_CORE_API Ray
    {
        Ray() = default;
        DMT_CPU_GPU Ray(Point3f o, Vector3f d, float time = 0.f, uintptr_t medium = 0);

        DMT_CPU_GPU bool      hasNaN() const;
        DMT_CPU_GPU uintptr_t getMedium() const;
        DMT_CPU_GPU Point3f   operator()(float t) const { return o * d * t; }
        uintptr_t             medium = 0; // information about hasDifferentials embedded in the low bit
        Point3f               o{};
        Vector3f              d{{0, 0, 1}};
        Vector3f d_inv{{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), 1}};
        float    time = 0;
    };


    struct DMT_CORE_API RayDifferential : public Ray
    {
        RayDifferential() = default;
        DMT_CPU_GPU RayDifferential(Point3f o, Vector3f d, float time = 0.f, uintptr_t medium = 0);
        DMT_CPU_GPU explicit RayDifferential(Ray const& ray);
        DMT_CPU_GPU void setDifferentials(Point3f _rxOrigin, Vector3f _rxDirection, Point3f _ryOrigin, Vector3f _ryDirection);
        DMT_CPU_GPU void scaleDifferentials(float s);

        bool     hasDifferentials = false;
        Point3f  rxOrigin, ryOrigin;
        Vector3f rxDirection, ryDirection;
    };
} // namespace dmt

namespace dstd {
    // TODO Move somewhere else
    template <typename T, int N>
    class array
    {
    public:
        using value_type     = T;
        using iterator       = value_type*;
        using const_iterator = value_type const*;
        using size_t         = std::size_t;

        array() = default;

        DMT_CPU_GPU array(std::initializer_list<T> v)
        {
            size_t i = 0;
            for (T const& val : v)
                values[i++] = val;
        }


        DMT_CPU_GPU void fill(T const& v)
        {
            for (int i = 0; i < N; ++i)
                values[i] = v;
        }


        DMT_CPU_GPU bool operator==(array<T, N> const& a) const
        {
            for (int i = 0; i < N; ++i)
                if (values[i] != a.values[i])
                    return false;
            return true;
        }

        DMT_CPU_GPU bool operator!=(array<T, N> const& a) const { return !(*this == a); }


        DMT_CPU_GPU iterator begin() { return values; }

        DMT_CPU_GPU iterator end() { return values + N; }

        DMT_CPU_GPU const_iterator begin() const { return values; }

        DMT_CPU_GPU const_iterator end() const { return values + N; }


        DMT_CPU_GPU size_t size() const { return N; }


        DMT_CPU_GPU T& operator[](size_t i) { return values[i]; }

        DMT_CPU_GPU T const& operator[](size_t i) const { return values[i]; }


        DMT_CPU_GPU T* data() { return values; }

        DMT_CPU_GPU T const* data() const { return values; }

    private:
        T values[N] = {};
    };
    //optional ----------------------------------------------------------------------
    template <typename T>

    class optional
    {
    public:
        using value_type = T;

        optional() = default;
        DMT_CPU_GPU
        optional(T const& v) : set(true) { new (ptr()) T(v); }
        DMT_CPU_GPU
        optional(T&& v) : set(true) { new (ptr()) T(std::move(v)); }
        DMT_CPU_GPU
        optional(optional const& v) : set(v.has_value())
        {
            if (v.has_value())
                new (ptr()) T(v.value());
        }
        DMT_CPU_GPU
        optional(optional&& v) : set(v.has_value())
        {
            if (v.has_value())
            {
                new (ptr()) T(std::move(v.value()));
                v.reset();
            }
        }

        DMT_CPU_GPU
        optional& operator=(T const& v)
        {
            reset();
            new (ptr()) T(v);
            set = true;
            return *this;
        }
        DMT_CPU_GPU
        optional& operator=(T&& v)
        {
            reset();
            new (ptr()) T(std::move(v));
            set = true;
            return *this;
        }
        DMT_CPU_GPU
        optional& operator=(optional const& v)
        {
            reset();
            if (v.has_value())
            {
                new (ptr()) T(v.value());
                set = true;
            }
            return *this;
        }
        DMT_CPU_GPU
        optional& operator=(optional&& v)
        {
            reset();
            if (v.has_value())
            {
                new (ptr()) T(std::move(v.value()));
                set = true;
                v.reset();
            }
            return *this;
        }

        DMT_CPU_GPU
        ~optional() { reset(); }

        DMT_CPU_GPU
        explicit operator bool() const { return set; }

        DMT_CPU_GPU
        T value_or(T const& alt) const { return set ? value() : alt; }

        DMT_CPU_GPU
        T* operator->() { return &value(); }
        DMT_CPU_GPU
        T const* operator->() const { return &value(); }
        DMT_CPU_GPU
        T& operator*() { return value(); }
        DMT_CPU_GPU
        T const& operator*() const { return value(); }
        DMT_CPU_GPU
        T& value()
        {
            assert(set);
            return *ptr();
        }
        DMT_CPU_GPU
        T const& value() const
        {
            assert(set);
            return *ptr();
        }

        DMT_CPU_GPU
        void reset()
        {
            if (set)
            {
                value().~T();
                set = false;
            }
        }

        DMT_CPU_GPU
        bool has_value() const { return set; }

    private:
#ifdef __CUDA_ARCH__
        // Work-around NVCC bug
        DMT_CPU_GPU
        T* ptr() { return reinterpret_cast<T*>(&optionalValue); }
        DMT_CPU_GPU
        T const* ptr() const { return reinterpret_cast<T const*>(&optionalValue); }
#else
        DMT_CPU_GPU T*       ptr() { return std::launder(reinterpret_cast<T*>(&optionalValue)); }
        DMT_CPU_GPU T const* ptr() const { return std::launder(reinterpret_cast<T const*>(&optionalValue)); }
#endif

#if !defined(__CUDA_ARCH__)
        std::aligned_storage_t<sizeof(T), alignof(T)> optionalValue;
#else
        alignas(T) unsigned char optionalValue[sizeof(T)];
#endif
        bool set = false;
    };

    template <typename T, size_t N>
    class InlinedVector
    {
        static_assert(N > 0, "InlinedVector must have non-zero capacity");

    public:
        using value_type = T;
        using size_type  = size_t;

        InlinedVector() : _size(0) {}

        /// @note by copy only
        InlinedVector(size_t sz, T const& v) : _size(sz)
        {
#if !defined(__CUDA_ARCH__)
            assert(sz <= N && "Exceeding capacity");
            for (size_t i = 0; i < sz; ++i)
                std::construct_at(&reinterpret_cast<T*>(&_storage)[i], v);
#else
            for (size_t i = 0; i < sz; ++i)
                new (&reinterpret_cast<T*>(&_storage)[i]) T{v};
#endif
        }

        ~InlinedVector() { clear(); }

        InlinedVector(InlinedVector const& other) : _size(0)
        {
            for (size_type i = 0; i < other._size; ++i)
                emplace_back(other[i]);
        }

        InlinedVector& operator=(InlinedVector const& other)
        {
            if (this != &other)
            {
                clear();
                for (size_type i = 0; i < other._size; ++i)
                    emplace_back(other[i]);
            }
            return *this;
        }

        InlinedVector(InlinedVector&& other) noexcept : _size(0)
        {
            for (size_type i = 0; i < other._size; ++i)
                emplace_back(std::move(other[i]));
            other.clear();
        }

        InlinedVector& operator=(InlinedVector&& other) noexcept
        {
            if (this != &other)
            {
                clear();
                for (size_type i = 0; i < other._size; ++i)
                    emplace_back(std::move(other[i]));
                other.clear();
            }
            return *this;
        }

        void push_back(T const& value)
        {
#if !defined(__CUDA_ARCH__)
            if (_size >= N)
                throw std::overflow_error("InlinedVector capacity exceeded");
#endif
            new (data() + _size) T(value);
            ++_size;
        }

        void push_back(T&& value)
        {
#if !defined(__CUDA_ARCH__)
            if (_size >= N)
                throw std::overflow_error("InlinedVector capacity exceeded");
#endif
            new (data() + _size) T(std::move(value));
            ++_size;
        }

        template <typename... Args>
        void emplace_back(Args&&... args)
        {
#if !defined(__CUDA_ARCH__)
            if (_size >= N)
                throw std::overflow_error("InlinedVector capacity exceeded");
#endif
            new (data() + _size) T(std::forward<Args>(args)...);
            ++_size;
        }

        void pop_back()
        {
#if !defined(__CUDA_ARCH__)
            if (_size == 0)
                throw std::underflow_error("InlinedVector is empty");
#endif
            --_size;
            data()[_size].~T();
        }

        T& operator[](size_type index)
        {
#if !defined(__CUDA_ARCH__)
            if (index >= _size)
                throw std::out_of_range("InlinedVector index out of range");
#endif
            return data()[index];
        }

        T const& operator[](size_type index) const
        {
#if !defined(__CUDA_ARCH__)
            if (index >= _size)
                throw std::out_of_range("InlinedVector index out of range");
#endif
            return data()[index];
        }

        void resize(size_type new_size)
        {
#if !defined(__CUDA_ARCH__)
            if (new_size > N)
                throw std::overflow_error("InlinedVector resize beyond capacity");
#endif

            if (new_size > _size)
            {
                while (_size < new_size)
                    emplace_back(); // default constructor
            }
            else
            {
                while (_size > new_size)
                    pop_back();
            }
        }

        void resize(size_type new_size, T const& value)
        {
#if !defined(__CUDA_ARCH__)
            if (new_size > N)
                throw std::overflow_error("InlinedVector resize beyond capacity");
#endif

            if (new_size > _size)
            {
                while (_size < new_size)
                    push_back(value);
            }
            else
            {
                while (_size > new_size)
                    pop_back();
            }
        }

        size_type           size() const { return _size; }
        constexpr size_type capacity() const { return N; }
        bool                empty() const { return _size == 0; }

        void clear()
        {
            for (size_type i = 0; i < _size; ++i)
                data()[i].~T();
            _size = 0;
        }

        T*       begin() { return data(); }
        T*       end() { return data() + _size; }
        T const* begin() const { return data(); }
        T const* end() const { return data() + _size; }

    private:
#if !defined(__CUDA_ARCH__)
        using Storage = typename std::aligned_storage<sizeof(T), alignof(T)>::type;
        Storage _storage[N];
#else
        alignas(T) unsigned char _storage[sizeof(T) * N];
#endif
        size_type _size;

        T*       data() { return reinterpret_cast<T*>(&_storage[0]); }
        T const* data() const { return reinterpret_cast<T const*>(&_storage[0]); }
    };

} // namespace dstd

namespace dmt {
    inline __host__ __device__ glm::vec4 glmZero() { return glm::vec4{0.f}; }
    inline __host__ __device__ glm::vec4 glmOne() { return glm::vec4{1.f}; }
    inline __host__ __device__ glm::vec4 glmLambdaMin() { return glm::vec4{360.f}; }
    inline __host__ __device__ glm::vec4 glmLambdaMax() { return glm::vec4{830.f}; }

    static_assert(alignof(Tuple4f) == alignof(glm::vec4));

    // Vector Types: Conversion to and from GLM ---------------------------------------------------------------------
    template <typename T, typename Enable = void>
    struct to_glm;

#if defined(__CUDA_ARCH__)
    template <typename T>
    struct to_glm<T,
                  ::cuda::std::enable_if_t<::cuda::std::is_integral_v<typename T::value_type> &&
                                           sizeof(typename T::value_type) == sizeof(int32_t) &&
                                           ::cuda::std::is_signed_v<typename T::value_type> && T::numComponents() == 2>>
    {
        using type = typename glm::ivec2;
    };

    template <typename T>
    struct to_glm<T,
                  ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<typename T::value_type> &&
                                           sizeof(typename T::value_type) == sizeof(float) && T::numComponents() == 2>>
    {
        using type = typename glm::vec2;
    };

    template <typename T>
    struct to_glm<T,
                  ::cuda::std::enable_if_t<::cuda::std::is_integral_v<typename T::value_type> &&
                                           sizeof(typename T::value_type) == sizeof(int32_t) &&
                                           ::cuda::std::is_signed_v<typename T::value_type> && T::numComponents() == 3>>
    {
        using type = typename glm::ivec3;
    };

    template <typename T>
    struct to_glm<T,
                  ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<typename T::value_type> &&
                                           sizeof(typename T::value_type) == sizeof(float) && T::numComponents() == 3>>
    {
        using type = typename glm::vec3;
    };

    template <typename T>
    struct to_glm<T,
                  ::cuda::std::enable_if_t<::cuda::std::is_integral_v<typename T::value_type> &&
                                           sizeof(typename T::value_type) == sizeof(int32_t) &&
                                           ::cuda::std::is_signed_v<typename T::value_type> && T::numComponents() == 4>>
    {
        using type = glm::ivec4;
    };

    template <typename T>
    struct to_glm<T,
                  ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<typename T::value_type> &&
                                           sizeof(typename T::value_type) == sizeof(float) && T::numComponents() == 4 &&
                                           !::cuda::std::is_same_v<T, Quaternion>>>
    {
        using type = glm::vec4;
    };
    template <typename T>
    struct to_glm<T,
                  ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<typename T::value_type> &&
                                           sizeof(typename T::value_type) == sizeof(float) && T::numComponents() == 4 &&
                                           ::cuda::std::is_same_v<T, Quaternion>>>
    {
        using type = glm::quat;
    };
#else
    template <typename T>
    struct to_glm<T,
                  std::enable_if_t<std::is_integral_v<typename T::value_type> && sizeof(typename T::value_type) == sizeof(int32_t) &&
                                   std::is_signed_v<typename T::value_type> && T::numComponents() == 2>>
    {
        using type = glm::ivec2;
    };

    template <typename T>
    struct to_glm<T,
                  std::enable_if_t<std::is_floating_point_v<typename T::value_type> &&
                                   sizeof(typename T::value_type) == sizeof(float) && T::numComponents() == 2>>
    {
        using type = glm::vec2;
    };

    template <typename T>
    struct to_glm<T,
                  std::enable_if_t<std::is_integral_v<typename T::value_type> && sizeof(typename T::value_type) == sizeof(int32_t) &&
                                   std::is_signed_v<typename T::value_type> && T::numComponents() == 3>>
    {
        using type = glm::ivec3;
    };

    template <typename T>
    struct to_glm<T,
                  std::enable_if_t<std::is_floating_point_v<typename T::value_type> &&
                                   sizeof(typename T::value_type) == sizeof(float) && T::numComponents() == 3>>
    {
        using type = glm::vec3;
    };

    template <typename T>
    struct to_glm<T,
                  std::enable_if_t<std::is_integral_v<typename T::value_type> && sizeof(typename T::value_type) == sizeof(int32_t) &&
                                   std::is_signed_v<typename T::value_type> && T::numComponents() == 4>>
    {
        using type = glm::ivec4;
    };

    template <typename T>
    struct to_glm<T,
                  std::enable_if_t<std::is_floating_point_v<typename T::value_type> && sizeof(typename T::value_type) == sizeof(float) &&
                                   T::numComponents() == 4 && !std::is_same_v<T, Quaternion>>>
    {
        using type = glm::vec4;
    };
    template <typename T>
    struct to_glm<T,
                  std::enable_if_t<std::is_floating_point_v<typename T::value_type> && sizeof(typename T::value_type) == sizeof(float) &&
                                   T::numComponents() == 4 && std::is_same_v<T, Quaternion>>>
    {
        using type = glm::quat;
    };
#endif
#if __cplusplus >= 202002L
    template <Vector T>
#else
    template <typename T>
#endif
    using to_glm_t = typename to_glm<T>::type;

#if __cplusplus >= 202002L
    template <Vector T>
#else
    template <typename T>
#endif
    inline __host__ __device__ to_glm_t<T>* toGLM(T* v)
    {
        static_assert(sizeof(T) == sizeof(to_glm_t<T>), "Size mismatch");
#if defined(__CUDA_ARCH__)
        return reinterpret_cast<to_glm_t<T>*>(v);
#else
        return std::bit_cast<to_glm_t<T>*>(v);
#endif
    }
#if __cplusplus >= 202002L
    template <Vector T>
#else
    template <typename T>
#endif
    inline __host__ __device__ to_glm_t<T> const* toGLM(T const* v)
    {
#if defined(__CUDA_ARCH__)
        return reinterpret_cast<to_glm_t<T> const*>(v);
#else
        return std::bit_cast<to_glm_t<T> const*>(v);
#endif
    }

#if __cplusplus >= 202002L
    template <int32_t n, Scalar T, glm::qualifier Q>
#else
    template <int32_t n, typename T, glm::qualifier Q>
#endif
    struct from_glm;
#if __cplusplus >= 202002L
    template <Scalar T, glm::qualifier Q>
#else
    template <typename T, glm::qualifier Q>
#endif
    struct from_glm<2, T, Q>
    {
        using type = Tuple2<T>;
    };
#if _cplusplus >= 202002L
    template <Scalar T, glm::qualifier Q>
#else
    template <typename T, glm::qualifier Q>
#endif
    struct from_glm<3, T, Q>
    {
        using type = Tuple3<T>;
    };
#if _cplusplus >= 202002L
    template <Scalar T, glm::qualifier Q>
#else
    template <typename T, glm::qualifier Q>
#endif
    struct from_glm<4, T, Q>
    {
        using type = Tuple4<T>;
    };
#if _cplusplus >= 202002L
    template <int32_t n, Scalar T, glm::qualifier Q>
#else
    template <int32_t n, typename T, glm::qualifier Q>
#endif
    using from_glm_t = typename from_glm<n, T, Q>::type;

#if _cplusplus >= 202002L
    template <int32_t n, Scalar T, glm::qualifier Q>
#else
    template <int32_t n, typename T, glm::qualifier Q>
#endif
    inline __host__ __device__ from_glm_t<n, T, Q>* fromGLM(glm::vec<n, T, Q>* v)
    {
#if defined(__CUDA_ARCH__)
        return reinterpret_cast<from_glm_t<n, T, Q>*>(v);
#else
        return std::bit_cast<from_glm_t<n, T, Q>*>(v);
#endif
    }
#if _cplusplus >= 202002L
    template <int32_t n, Scalar T, glm::qualifier Q>
#else
    template <int32_t n, typename T, glm::qualifier Q>
#endif
    inline __host__ __device__ from_glm_t<n, T, Q> const* fromGLM(glm::vec<n, T, Q> const* v)
    {
#if defined(__CUDA_ARCH__)
        return reinterpret_cast<from_glm_t<n, T, Q> const*>(v);
#else
        return std::bit_cast<from_glm_t<n, T, Q> const*>(v);
#endif
    }

    inline __host__ __device__ Quaternion* fromGLMquat(glm::quat* q)
    {
#if defined(__CUDA_ARCH__)
        return reinterpret_cast<Quaternion*>(q);
#else
        return std::bit_cast<Quaternion*>(q);
#endif
    }

    inline __host__ __device__ Quaternion const* fromGLMquat(glm::quat const* q)
    {
#if defined(__CUDA_ARCH__)
        return reinterpret_cast<Quaternion const*>(q);
#else
        return std::bit_cast<Quaternion const*>(q);
#endif
    }

    inline __host__ __device__ glm::mat4* toGLMmat(Matrix4f* m)
    {
#if defined(__CUDA_ARCH__)
        return reinterpret_cast<glm::mat4*>(m);
#else
        return std::bit_cast<glm::mat4*>(m);
#endif
    }
    inline __host__ __device__ glm::mat4 const* toGLMmat(Matrix4f const* m)
    {
#if defined(__CUDA_ARCH__)
        return reinterpret_cast<glm::mat4 const*>(m);
#else
        return std::bit_cast<glm::mat4 const*>(m);
#endif
    }
    inline __host__ __device__ Matrix4f* fromGLMmat(glm::mat4* m)
    {
#if defined(__CUDA_ARCH__)
        return reinterpret_cast<Matrix4f*>(m);
#else
        return std::bit_cast<Matrix4f*>(m);
#endif
    }
    inline __host__ __device__ Matrix4f const* fromGLMmat(glm::mat4 const* m)
    {
#if defined(__CUDA_ARCH__)
        return reinterpret_cast<Matrix4f const*>(m);
#else
        return std::bit_cast<Matrix4f const*>(m);
#endif
    }


} // namespace dmt

#if defined(DMT_OS_WINDOWS)
    #pragma pop_macro("near")
#endif

#if defined(DMT_CUDAUTILS_IMPLEMENTATION)

// TODO generated optimized assembly for x64 uses vectorized instructions for floating point types, like "[v]addps", but integer types
// don't seem to use SSE2

namespace dmt {
    // Vector Types: Static Assertions --------------------------------------------------------------------------------
    #if __cplusplus >= 202002L
    static_assert(VectorNormalized<Normal2f>);
    static_assert(VectorNormalized<Normal3f>);
    static_assert(VectorScalable<Vector2i>);
    static_assert(VectorScalable<Vector2f>);
    static_assert(VectorScalable<Vector3i>);
    static_assert(VectorScalable<Vector3f>);
    static_assert(VectorScalable<Vector4i>);
    static_assert(VectorScalable<Vector4f>);
    static_assert(VectorScalable<Point2i>);
    static_assert(VectorScalable<Point2f>);
    static_assert(VectorScalable<Point3i>);
    static_assert(VectorScalable<Point3f>);
    static_assert(VectorScalable<Point4i>);
    static_assert(VectorScalable<Point4f>);
    #endif

    // Vector Types: Basic Operations ---------------------------------------------------------------------------------
    __host__ __device__ bool operator==(Point2i a, Point2i b) { return a.x == b.x && a.y == b.y; }

    __host__ __device__ bool operator==(Point3i a, Point3i b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

    __host__ __device__ bool operator==(Point4i a, Point4i b)
    {
        return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
    }

    __host__ __device__ Point2i::operator Vector2i()
    {
    #if defined(__CUDA_ARCH__)
        return *reinterpret_cast<Vector2i const*>(this);
    #else
        return *std::bit_cast<Vector2i const*>(this);
    #endif
    }
    __host__ __device__ Point2f::operator Vector2f()
    {
    #if defined(__CUDA_ARCH__)
        return *reinterpret_cast<Vector2f const*>(this);
    #else
        return *std::bit_cast<Vector2f const*>(this);
    #endif
    }
    __host__ __device__ Point3i::operator Vector3i()
    {
    #if defined(__CUDA_ARCH__)
        return *reinterpret_cast<Vector3i const*>(this);
    #else
        return *std::bit_cast<Vector3i const*>(this);
    #endif
    }
    __host__ __device__ Point3f::operator Vector3f()
    {
    #if defined(__CUDA_ARCH__)
        return *reinterpret_cast<Vector3f const*>(this);
    #else
        return *std::bit_cast<Vector3f const*>(this);
    #endif
    }
    __host__ __device__ Point4i::operator Vector4i()
    {
    #if defined(__CUDA_ARCH__)
        return *reinterpret_cast<Vector4i const*>(this);
    #else
        return *std::bit_cast<Vector4i const*>(this);
    #endif
    }
    __host__ __device__ Point4f::operator Vector4f()
    {
    #if defined(__CUDA_ARCH__)
        return *reinterpret_cast<Vector4f const*>(this);
    #else
        return *std::bit_cast<Vector4f const*>(this);
    #endif
    }

    __host__ __device__ Point2i operator+(Point2i a, Vector2i b)
    {
        glm::ivec2 sum = *toGLM(&a) + *toGLM(&b);
        return *fromGLM(&sum);
    }
    __host__ __device__ Point2f operator+(Point2f a, Vector2f b)
    {
        glm::vec2 sum = *toGLM(&a) + *toGLM(&b);
        return *fromGLM(&sum);
    }
    __host__ __device__ Point3i operator+(Point3i a, Vector3i b)
    {
        glm::ivec3 sum = *toGLM(&a) + *toGLM(&b);
        return *fromGLM(&sum);
    }
    __host__ __device__ Point3f operator+(Point3f a, Vector3f b)
    {
        glm::vec3 sum = *toGLM(&a) + *toGLM(&b);
        return *fromGLM(&sum);
    }
    __host__ __device__ Point4i operator+(Point4i a, Vector4i b)
    {
        glm::ivec4 sum = *toGLM(&a) + *toGLM(&b);
        return *fromGLM(&sum);
    }
    __host__ __device__ Point4f operator+(Point4f a, Vector4f b)
    {
        glm::vec4 sum = *toGLM(&a) + *toGLM(&b);
        return *fromGLM(&sum);
    }
    __host__ __device__ Vector2i operator+(Vector2i a, Vector2i b)
    {
        glm::ivec2 sum = *toGLM(&a) + *toGLM(&b);
        return *fromGLM(&sum);
    }
    __host__ __device__ Vector2f operator+(Vector2f a, Vector2f b)
    {
        glm::vec2 sum = *toGLM(&a) + *toGLM(&b);
        return *fromGLM(&sum);
    }
    __host__ __device__ Vector3i operator+(Vector3i a, Vector3i b)
    {
        glm::ivec3 sum = *toGLM(&a) + *toGLM(&b);
        return *fromGLM(&sum);
    }
    __host__ __device__ Vector3f operator+(Vector3f a, Vector3f b)
    {
        glm::vec3 sum = *toGLM(&a) + *toGLM(&b);
        return *fromGLM(&sum);
    }
    __host__ __device__ Vector4i operator+(Vector4i a, Vector4i b)
    {
        glm::ivec4 sum = *toGLM(&a) + *toGLM(&b);
        return *fromGLM(&sum);
    }
    __host__ __device__ Vector4f operator+(Vector4f a, Vector4f b)
    {
        glm::vec4 sum = *toGLM(&a) + *toGLM(&b);
        return *fromGLM(&sum);
    }
    __host__ __device__ Normal2f operator+(Normal2f a, Normal2f b)
    {
        glm::vec2 sum = glm::normalize(*toGLM(&a) + *toGLM(&b));
        return *fromGLM(&sum);
    }
    __host__ __device__ Normal3f operator+(Normal3f a, Normal3f b)
    {
        glm::vec3 sum = glm::normalize(*toGLM(&a) + *toGLM(&b));
        return *fromGLM(&sum);
    }

    __host__ __device__ Quaternion operator+(Quaternion a, Quaternion b)
    {
        glm::quat sum = *toGLM(&a) + *toGLM(&b);
        return *fromGLMquat(&sum);
    }

    __host__ __device__ Vector2i operator-(Point2i a, Point2i b)
    {
        glm::ivec2 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector2f operator-(Point2f a, Point2f b)
    {
        glm::vec2 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector3i operator-(Point3i a, Point3i b)
    {
        glm::ivec3 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector3f operator-(Point3f a, Point3f b)
    {
        glm::vec3 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector4i operator-(Point4i a, Point4i b)
    {
        glm::ivec4 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector4f operator-(Point4f a, Point4f b)
    {
        glm::vec4 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector2i operator-(Point2i a, Vector2i b)
    {
        glm::ivec2 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector2f operator-(Point2f a, Vector2f b)
    {
        glm::vec2 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector3i operator-(Point3i a, Vector3i b)
    {
        glm::ivec3 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector3f operator-(Point3f a, Vector3f b)
    {
        glm::vec3 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector4i operator-(Point4i a, Vector4i b)
    {
        glm::ivec4 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector4f operator-(Point4f a, Vector4f b)
    {
        glm::vec4 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector2i operator-(Vector2i a, Vector2i b)
    {
        glm::ivec2 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector2f operator-(Vector2f a, Vector2f b)
    {
        glm::vec2 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector3i operator-(Vector3i a, Vector3i b)
    {
        glm::ivec3 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector3f operator-(Vector3f a, Vector3f b)
    {
        glm::vec3 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector4i operator-(Vector4i a, Vector4i b)
    {
        glm::ivec4 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector4f operator-(Vector4f a, Vector4f b)
    {
        glm::vec4 dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLM(&dif);
    }
    __host__ __device__ Normal2f operator-(Normal2f a, Normal2f b)
    {
        glm::vec2 dif = glm::normalize(*toGLM(&a) - *toGLM(&b));
        return *fromGLM(&dif);
    }
    __host__ __device__ Normal3f operator-(Normal3f a, Normal3f b)
    {
        glm::vec3 dif = glm::normalize(*toGLM(&a) - *toGLM(&b));
        return *fromGLM(&dif);
    }

    __host__ __device__ Quaternion operator-(Quaternion a, Quaternion b)
    {
        glm::quat dif = *toGLM(&a) - *toGLM(&b);
        return *fromGLMquat(&dif);
    }

    __host__ __device__ Vector2i operator-(Point2i v)
    {
        glm::ivec2 dif = -*toGLM(&v);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector2f operator-(Point2f v)
    {
        glm::vec2 dif = -*toGLM(&v);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector3i operator-(Point3i v)
    {
        glm::ivec3 dif = -*toGLM(&v);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector3f operator-(Point3f v)
    {
        glm::vec3 dif = -*toGLM(&v);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector4i operator-(Point4i v)
    {
        glm::ivec4 dif = -*toGLM(&v);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector4f operator-(Point4f v)
    {
        glm::vec4 dif = -*toGLM(&v);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector2i operator-(Vector2i v)
    {
        glm::ivec2 dif = -*toGLM(&v);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector2f operator-(Vector2f v)
    {
        glm::vec2 dif = -*toGLM(&v);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector3i operator-(Vector3i v)
    {
        glm::ivec3 dif = -*toGLM(&v);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector3f operator-(Vector3f v)
    {
        glm::vec3 dif = -*toGLM(&v);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector4i operator-(Vector4i v)
    {
        glm::ivec4 dif = -*toGLM(&v);
        return *fromGLM(&dif);
    }
    __host__ __device__ Vector4f operator-(Vector4f v)
    {
        glm::vec4 dif = -*toGLM(&v);
        return *fromGLM(&dif);
    }
    __host__ __device__ Normal2f operator-(Normal2f v)
    {
        glm::vec2 dif = -*toGLM(&v);
        return *fromGLM(&dif);
    }
    __host__ __device__ Normal3f operator-(Normal3f v)
    {
        glm::vec3 dif = -*toGLM(&v);
        return *fromGLM(&dif);
    }

    __host__ __device__ Quaternion operator-(Quaternion q)
    {
        glm::quat dif = -*toGLM(&q);
        return *fromGLMquat(&dif);
    }

    __host__ __device__ Vector2i operator*(Vector2i a, Vector2i b)
    {
        glm::ivec2 mul = *toGLM(&a) * *toGLM(&b);
        return *fromGLM(&mul);
    }
    __host__ __device__ Vector2f operator*(Vector2f a, Vector2f b)
    {
        glm::vec2 mul = *toGLM(&a) * *toGLM(&b);
        return *fromGLM(&mul);
    }
    __host__ __device__ Vector3i operator*(Vector3i a, Vector3i b)
    {
        glm::ivec3 mul = *toGLM(&a) * *toGLM(&b);
        return *fromGLM(&mul);
    }
    __host__ __device__ Vector3f operator*(Vector3f a, Vector3f b)
    {
        glm::vec3 mul = *toGLM(&a) * *toGLM(&b);
        return *fromGLM(&mul);
    }
    __host__ __device__ Vector4i operator*(Vector4i a, Vector4i b)
    {
        glm::ivec4 mul = *toGLM(&a) * *toGLM(&b);
        return *fromGLM(&mul);
    }
    __host__ __device__ Vector4f operator*(Vector4f a, Vector4f b)
    {
        glm::vec4 mul = *toGLM(&a) * *toGLM(&b);
        return *fromGLM(&mul);
    }

    __host__ __device__ Quaternion operator*(Quaternion a, Quaternion b)
    {
        glm::quat mul = *toGLM(&a) * *toGLM(&b);
        return *fromGLMquat(&mul);
    }

    __host__ __device__ Vector2i operator/(Vector2i a, Vector2i b)
    {
        glm::ivec2 div = *toGLM(&a) / *toGLM(&b);
        return *fromGLM(&div);
    }
    __host__ __device__ Vector2f operator/(Vector2f a, Vector2f b)
    {
        glm::vec2 div = *toGLM(&a) / *toGLM(&b);
        return *fromGLM(&div);
    }
    __host__ __device__ Vector3i operator/(Vector3i a, Vector3i b)
    {
        glm::ivec3 div = *toGLM(&a) / *toGLM(&b);
        return *fromGLM(&div);
    }
    __host__ __device__ Vector3f operator/(Vector3f a, Vector3f b)
    {
        glm::vec3 div = *toGLM(&a) / *toGLM(&b);
        return *fromGLM(&div);
    }
    __host__ __device__ Vector4i operator/(Vector4i a, Vector4i b)
    {
        glm::ivec4 div = *toGLM(&a) / *toGLM(&b);
        return *fromGLM(&div);
    }
    __host__ __device__ Vector4f operator/(Vector4f a, Vector4f b)
    {
        glm::vec4 div = *toGLM(&a) / *toGLM(&b);
        return *fromGLM(&div);
    }

    __host__ __device__ Point2i& operator+=(Point2i& a, Vector2i b)
    {
        *toGLM(&a) += *toGLM(&b);
        return a;
    }
    __host__ __device__ Point2f& operator+=(Point2f& a, Vector2f b)
    {
        *toGLM(&a) += *toGLM(&b);
        return a;
    }
    __host__ __device__ Point3i& operator+=(Point3i& a, Vector3i b)
    {
        *toGLM(&a) += *toGLM(&b);
        return a;
    }
    __host__ __device__ Point3f& operator+=(Point3f& a, Vector3f b)
    {
        *toGLM(&a) += *toGLM(&b);
        return a;
    }
    __host__ __device__ Point4i& operator+=(Point4i& a, Vector4i b)
    {
        *toGLM(&a) += *toGLM(&b);
        return a;
    }
    __host__ __device__ Point4f& operator+=(Point4f& a, Vector4f b)
    {
        *toGLM(&a) += *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector2i& operator+=(Vector2i& a, Vector2i b)
    {
        *toGLM(&a) += *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector2f& operator+=(Vector2f& a, Vector2f b)
    {
        *toGLM(&a) += *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector3i& operator+=(Vector3i& a, Vector3i b)
    {
        *toGLM(&a) += *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector3f& operator+=(Vector3f& a, Vector3f b)
    {
        *toGLM(&a) += *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector4i& operator+=(Vector4i& a, Vector4i b)
    {
        *toGLM(&a) += *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector4f& operator+=(Vector4f& a, Vector4f b)
    {
        *toGLM(&a) += *toGLM(&b);
        return a;
    }
    __host__ __device__ Normal2f& operator+=(Normal2f& a, Normal2f b)
    {
        *toGLM(&a) += *toGLM(&b);
        return a;
    }
    __host__ __device__ Normal3f& operator+=(Normal3f& a, Normal3f b)
    {
        *toGLM(&a) += *toGLM(&b);
        return a;
    }

    __host__ __device__ Quaternion& operator+=(Quaternion& a, Quaternion b)
    {
        *toGLM(&a) += *toGLM(&b);
        return a;
    }

    __host__ __device__ Vector2i& operator-=(Vector2i& a, Vector2i b)
    {
        *toGLM(&a) -= *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector2f& operator-=(Vector2f& a, Vector2f b)
    {
        *toGLM(&a) -= *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector3i& operator-=(Vector3i& a, Vector3i b)
    {
        *toGLM(&a) -= *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector3f& operator-=(Vector3f& a, Vector3f b)
    {
        *toGLM(&a) -= *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector4i& operator-=(Vector4i& a, Vector4i b)
    {
        *toGLM(&a) -= *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector4f& operator-=(Vector4f& a, Vector4f b)
    {
        *toGLM(&a) -= *toGLM(&b);
        return a;
    }
    __host__ __device__ Normal2f& operator-=(Normal2f& a, Normal2f b)
    {
        *toGLM(&a) -= *toGLM(&b);
        return a;
    }
    __host__ __device__ Normal3f& operator-=(Normal3f& a, Normal3f b)
    {
        *toGLM(&a) -= *toGLM(&b);
        return a;
    }

    __host__ __device__ Quaternion& operator-=(Quaternion& a, Quaternion b)
    {
        *toGLM(&a) -= *toGLM(&b);
        return a;
    }

    __host__ __device__ Vector2i& operator*=(Vector2i& a, Vector2i b)
    {
        *toGLM(&a) *= *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector2f& operator*=(Vector2f& a, Vector2f b)
    {
        *toGLM(&a) *= *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector3i& operator*=(Vector3i& a, Vector3i b)
    {
        *toGLM(&a) *= *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector3f& operator*=(Vector3f& a, Vector3f b)
    {
        *toGLM(&a) *= *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector4i& operator*=(Vector4i& a, Vector4i b)
    {
        *toGLM(&a) *= *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector4f& operator*=(Vector4f& a, Vector4f b)
    {
        *toGLM(&a) *= *toGLM(&b);
        return a;
    }

    __host__ __device__ Quaternion& operator*=(Quaternion& a, Quaternion b)
    {
        *toGLM(&a) *= *toGLM(&b);
        return a;
    }

    __host__ __device__ Vector2i& operator/=(Vector2i& a, Vector2i b)
    {
        *toGLM(&a) /= *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector2f& operator/=(Vector2f& a, Vector2f b)
    {
        *toGLM(&a) /= *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector3i& operator/=(Vector3i& a, Vector3i b)
    {
        *toGLM(&a) /= *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector3f& operator/=(Vector3f& a, Vector3f b)
    {
        *toGLM(&a) /= *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector4i& operator/=(Vector4i& a, Vector4i b)
    {
        *toGLM(&a) /= *toGLM(&b);
        return a;
    }
    __host__ __device__ Vector4f& operator/=(Vector4f& a, Vector4f b)
    {
        *toGLM(&a) /= *toGLM(&b);
        return a;
    }

    __host__ __device__ Normal2f normalFrom(Vector2f v)
    {
        glm::vec2 nrm = glm::normalize(*toGLM(&v));
        return *fromGLM(&nrm);
    }
    __host__ __device__ Normal3f normalFrom(Vector3f v)
    {
        glm::vec3 nrm = glm::normalize(*toGLM(&v));
        return *fromGLM(&nrm);
    }

    // Vector Types: Generic Tuple Operations -------------------------------------------------------------------------
    __host__ __device__ Tuple2f abs(Tuple2f v)
    {
        glm::vec2 res = glm::abs(*toGLM(&v));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple2i abs(Tuple2i v)
    {
        glm::ivec2 res = glm::abs(*toGLM(&v));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple3f abs(Tuple3f v)
    {
        glm::vec3 res = glm::abs(*toGLM(&v));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple3i abs(Tuple3i v)
    {
        glm::ivec3 res = glm::abs(*toGLM(&v));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple4f abs(Tuple4f v)
    {
        glm::vec4 res = glm::abs(*toGLM(&v));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple4i abs(Tuple4i v)
    {
        glm::ivec4 res = glm::abs(*toGLM(&v));
        return *fromGLM(&res);
    }

    __host__ __device__ Tuple2f ceil(Tuple2f v)
    {
        glm::vec2 res = glm::ceil(*toGLM(&v));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple3f ceil(Tuple3f v)
    {
        glm::vec3 res = glm::ceil(*toGLM(&v));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple4f ceil(Tuple4f v)
    {
        glm::vec4 res = glm::ceil(*toGLM(&v));
        return *fromGLM(&res);
    }

    __host__ __device__ Tuple2f floor(Tuple2f v)
    {
        glm::vec2 res = glm::floor(*toGLM(&v));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple3f floor(Tuple3f v)
    {
        glm::vec3 res = glm::floor(*toGLM(&v));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple4f floor(Tuple4f v)
    {
        glm::vec4 res = glm::floor(*toGLM(&v));
        return *fromGLM(&res);
    }

    __host__ __device__ Tuple2f sqrt(Tuple2f v)
    {
        glm::vec2 res = glm::sqrt(*toGLM(&v));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple3f sqrt(Tuple3f v)
    {
        glm::vec3 res = glm::sqrt(*toGLM(&v));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple4f sqrt(Tuple4f v)
    {
        glm::vec4 res = glm::sqrt(*toGLM(&v));
        return *fromGLM(&res);
    }

    __host__ __device__ Tuple2f lerp(float t, Tuple2f zero, Tuple2f one)
    {
        glm::vec2 mx = glm::mix(*toGLM(&one), *toGLM(&zero), t);
        return *fromGLM(&mx);
    }
    __host__ __device__ Tuple3f lerp(float t, Tuple3f zero, Tuple3f one)
    {
        glm::vec3 mx = glm::mix(*toGLM(&one), *toGLM(&zero), t);
        return *fromGLM(&mx);
    }
    __host__ __device__ Tuple4f lerp(float t, Tuple4f zero, Tuple4f one)
    {
        glm::vec4 mx = glm::mix(*toGLM(&one), *toGLM(&zero), t);
        return *fromGLM(&mx);
    }

    __host__ __device__ Tuple2f fma(Tuple2f mult0, Tuple2f mult1, Tuple2f add)
    {
        glm::vec2 fmares = *toGLM(&mult0) * *toGLM(&mult1) + *toGLM(&add);
        return *fromGLM(&fmares);
    }
    __host__ __device__ Tuple2i fma(Tuple2i mult0, Tuple2i mult1, Tuple2i add)
    {
        glm::ivec2 fmares = *toGLM(&mult0) * *toGLM(&mult1) + *toGLM(&add);
        return *fromGLM(&fmares);
    }
    __host__ __device__ Tuple3f fma(Tuple3f mult0, Tuple3f mult1, Tuple3f add)
    {
        glm::vec3 fmares = *toGLM(&mult0) * *toGLM(&mult1) + *toGLM(&add);
        return *fromGLM(&fmares);
    }
    __host__ __device__ Tuple3i fma(Tuple3i mult0, Tuple3i mult1, Tuple3i add)
    {
        glm::ivec3 fmares = *toGLM(&mult0) * *toGLM(&mult1) + *toGLM(&add);
        return *fromGLM(&fmares);
    }
    __host__ __device__ Tuple4f fma(Tuple4f mult0, Tuple4f mult1, Tuple4f add)
    {
        glm::vec4 fmares = *toGLM(&mult0) * *toGLM(&mult1) + *toGLM(&add);
        return *fromGLM(&fmares);
    }
    __host__ __device__ Tuple4i fma(Tuple4i mult0, Tuple4i mult1, Tuple4i add)
    {
        glm::ivec4 fmares = *toGLM(&mult0) * *toGLM(&mult1) + *toGLM(&add);
        return *fromGLM(&fmares);
    }

    __host__ __device__ Tuple2f min(Tuple2f a, Tuple2f b)
    {
        glm::vec2 res = glm::min(*toGLM(&a), *toGLM(&b));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple2i min(Tuple2i a, Tuple2i b)
    {
        glm::ivec2 res = glm::min(*toGLM(&a), *toGLM(&b));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple3f min(Tuple3f a, Tuple3f b)
    {
        glm::vec3 res = glm::min(*toGLM(&a), *toGLM(&b));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple3i min(Tuple3i a, Tuple3i b)
    {
        glm::ivec3 res = glm::min(*toGLM(&a), *toGLM(&b));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple4f min(Tuple4f a, Tuple4f b)
    {
        glm::vec4 res = glm::min(*toGLM(&a), *toGLM(&b));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple4i min(Tuple4i a, Tuple4i b)
    {
        glm::ivec4 res = glm::min(*toGLM(&a), *toGLM(&b));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple2f max(Tuple2f a, Tuple2f b)
    {
        glm::vec2 res = glm::max(*toGLM(&a), *toGLM(&b));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple2i max(Tuple2i a, Tuple2i b)
    {
        glm::ivec2 res = glm::max(*toGLM(&a), *toGLM(&b));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple3f max(Tuple3f a, Tuple3f b)
    {
        glm::vec3 res = glm::max(*toGLM(&a), *toGLM(&b));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple3i max(Tuple3i a, Tuple3i b)
    {
        glm::ivec3 res = glm::max(*toGLM(&a), *toGLM(&b));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple4f max(Tuple4f a, Tuple4f b)
    {
        glm::vec4 res = glm::max(*toGLM(&a), *toGLM(&b));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple4i max(Tuple4i a, Tuple4i b)
    {
        glm::ivec4 res = glm::max(*toGLM(&a), *toGLM(&b));
        return *fromGLM(&res);
    }

    __host__ __device__ bool near(Tuple2f a, Tuple2f b, float tolerance)
    {
        auto bvec = glm::epsilonEqual(*toGLM(&a), *toGLM(&b), tolerance);
        return bvec.x && bvec.y;
    }
    __host__ __device__ bool near(Tuple2i a, Tuple2i b)
    {
        auto bvec = *toGLM(&a) = *toGLM(&b);
        return bvec.x && bvec.y;
    }
    __host__ __device__ bool near(Tuple3f a, Tuple3f b, float tolerance)
    {
        auto bvec = glm::epsilonEqual(*toGLM(&a), *toGLM(&b), tolerance);
        return bvec.x && bvec.y && bvec.z;
    }
    __host__ __device__ bool near(Tuple3i a, Tuple3i b)
    {
        auto bvec = *toGLM(&a) = *toGLM(&b);
        return bvec.x && bvec.y && bvec.z;
    }
    __host__ __device__ bool near(Tuple4f a, Tuple4f b, float tolerance)
    {
        auto bvec = glm::epsilonEqual(*toGLM(&a), *toGLM(&b), tolerance);
        return bvec.x && bvec.y && bvec.z && bvec.w;
    }
    __host__ __device__ bool near(Tuple4i a, Tuple4i b)
    {
        auto bvec = *toGLM(&a) = *toGLM(&b);
        return bvec.x && bvec.y && bvec.z && bvec.w;
    }

    __host__ __device__ Tuple2f::value_type dot(Tuple2f a, Tuple2f b) { return glm::dot(*toGLM(&a), *toGLM(&b)); }
    __host__ __device__ Tuple3f::value_type dot(Tuple3f a, Tuple3f b) { return glm::dot(*toGLM(&a), *toGLM(&b)); }
    __host__ __device__ Tuple4f::value_type dot(Tuple4f a, Tuple4f b) { return glm::dot(*toGLM(&a), *toGLM(&b)); }

    __host__ __device__ Tuple2f::value_type absDot(Tuple2f a, Tuple2f b)
    {
        return glm::abs(glm::dot(*toGLM(&a), *toGLM(&b)));
    }
    __host__ __device__ Tuple3f::value_type absDot(Tuple3f a, Tuple3f b)
    {
        return glm::abs(glm::dot(*toGLM(&a), *toGLM(&b)));
    }
    __host__ __device__ Tuple4f::value_type absDot(Tuple4f a, Tuple4f b)
    {
        return glm::abs(glm::dot(*toGLM(&a), *toGLM(&b)));
    }

    __host__ __device__ Tuple3f cross(Tuple3f a, Tuple3f b)
    {
        glm::vec3 x = glm::cross(*toGLM(&a), *toGLM(&b));
        return *fromGLM(&x);
    }

    __host__ __device__ Tuple2f normalize(Tuple2f v)
    {
        glm::vec2 res = glm::normalize(*toGLM(&v));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple3f normalize(Tuple3f v)
    {
        glm::vec3 res = glm::normalize(*toGLM(&v));
        return *fromGLM(&res);
    }
    __host__ __device__ Tuple4f normalize(Tuple4f v)
    {
        glm::vec4 res = glm::normalize(*toGLM(&v));
        return *fromGLM(&res);
    }

    __host__ __device__ Tuple2f::value_type normL2(Tuple2f v) { return glm::length(*toGLM(&v)); }
    __host__ __device__ Tuple3f::value_type normL2(Tuple3f v) { return glm::length(*toGLM(&v)); }
    __host__ __device__ Tuple4f::value_type normL2(Tuple4f v) { return glm::length(*toGLM(&v)); }

    __host__ __device__ bool all(Tuple2f v) { return v.x != 0 && v.y != 0; }
    __host__ __device__ bool all(Tuple3f v) { return v.x != 0 && v.y != 0 && v.z != 0; }
    __host__ __device__ bool all(Tuple4f v) { return v.x != 0 && v.y != 0 && v.z != 0 && v.w != 0; }

    __host__ __device__ bool any(Tuple2f v) { return v.x != 0 || v.y != 0; }
    __host__ __device__ bool any(Tuple3f v) { return v.x != 0 || v.y != 0 || v.z != 0; }
    __host__ __device__ bool any(Tuple4f v) { return v.x != 0 || v.y != 0 || v.z != 0 || v.w != 0; }

    __host__ __device__ Tuple2f operator>(Tuple2f v0, Tuple2f v1)
    {
        float const on   = fl::bitsToFloat(0xffffffff);
        float const zero = 0.0f;
        Tuple2f     result;
        result.x = (v0.x > v1.x) ? on : zero;
        result.y = (v0.y > v1.y) ? on : zero;
        return result;
    }

    __host__ __device__ Tuple3f operator>(Tuple3f v0, Tuple3f v1)
    {
        float const on   = fl::bitsToFloat(0xffffffff);
        float const zero = 0.0f;
        Tuple3f     result;
        result.x = (v0.x > v1.x) ? on : zero;
        result.y = (v0.y > v1.y) ? on : zero;
        result.z = (v0.z > v1.z) ? on : zero;
        return result;
    }

    __host__ __device__ Tuple3f operator<(Tuple3f v0, Tuple3f v1)
    {
        float const on   = fl::bitsToFloat(0xffffffff);
        float const zero = 0.0f;
        Tuple3f     result;
        result.x = (v0.x < v1.x) ? on : zero;
        result.y = (v0.y < v1.y) ? on : zero;
        result.z = (v0.z < v1.z) ? on : zero;
        return result;
    }

    __host__ __device__ Tuple4f operator<(Tuple4f v0, Tuple4f v1)
    {
        float const on   = fl::bitsToFloat(0xffffffff);
        float const zero = 0.0f;
        Tuple4f     result;
        result.x = (v0.x < v1.x) ? on : zero;
        result.y = (v0.y < v1.y) ? on : zero;
        result.z = (v0.z < v1.z) ? on : zero;
        result.w = (v0.w < v1.w) ? on : zero;
        return result;
    }

    __host__ __device__ Tuple2f operator>=(Tuple2f v0, Tuple2f v1)
    {
        float const on   = fl::bitsToFloat(0xffffffff);
        float const zero = 0.0f;
        Tuple2f     result;
        result.x = (v0.x >= v1.x) ? on : zero;
        result.y = (v0.y >= v1.y) ? on : zero;
        return result;
    }

    __host__ __device__ Tuple3f operator>=(Tuple3f v0, Tuple3f v1)
    {
        float const on   = fl::bitsToFloat(0xffffffff);
        float const zero = 0.0f;
        Tuple3f     result;
        result.x = (v0.x >= v1.x) ? on : zero;
        result.y = (v0.y >= v1.y) ? on : zero;
        result.z = (v0.z >= v1.z) ? on : zero;
        return result;
    }

    __host__ __device__ Tuple3f operator<=(Tuple3f v0, Tuple3f v1)
    {
        float const on   = fl::bitsToFloat(0xffffffff);
        float const zero = 0.0f;
        Tuple3f     result;
        result.x = (v0.x <= v1.x) ? on : zero;
        result.y = (v0.y <= v1.y) ? on : zero;
        result.z = (v0.z <= v1.z) ? on : zero;
        return result;
    }

    __host__ __device__ Tuple4f operator<=(Tuple4f v0, Tuple4f v1)
    {
        float const on   = fl::bitsToFloat(0xffffffff);
        float const zero = 0.0f;
        Tuple4f     result;
        result.x = (v0.x <= v1.x) ? on : zero;
        result.y = (v0.y <= v1.y) ? on : zero;
        result.z = (v0.z <= v1.z) ? on : zero;
        result.w = (v0.w <= v1.w) ? on : zero;
        return result;
    }

    __host__ __device__ Tuple2f operator==(Tuple2f v0, Tuple2f v1)
    {
        float const on   = fl::bitsToFloat(0xffffffff);
        float const zero = 0.0f;
        Tuple2f     result;
        result.x = (v0.x == v1.x) ? on : zero;
        result.y = (v0.y == v1.y) ? on : zero;
        return result;
    }

    __host__ __device__ Tuple3f operator==(Tuple3f v0, Tuple3f v1)
    {
        float const on   = fl::bitsToFloat(0xffffffff);
        float const zero = 0.0f;
        Tuple3f     result;
        result.x = (v0.x == v1.x) ? on : zero;
        result.y = (v0.y == v1.y) ? on : zero;
        result.z = (v0.z == v1.z) ? on : zero;
        return result;
    }

    __host__ __device__ Tuple3f operator!=(Tuple3f v0, Tuple3f v1)
    {
        float const on   = fl::bitsToFloat(0xffffffff);
        float const zero = 0.0f;
        Tuple3f     result;
        result.x = (v0.x != v1.x) ? on : zero;
        result.y = (v0.y != v1.y) ? on : zero;
        result.z = (v0.z != v1.z) ? on : zero;
        return result;
    }

    __host__ __device__ Tuple4f operator!=(Tuple4f v0, Tuple4f v1)
    {
        float const on   = fl::bitsToFloat(0xffffffff);
        float const zero = 0.0f;
        Tuple4f     result;
        result.x = (v0.x != v1.x) ? on : zero;
        result.y = (v0.y != v1.y) ? on : zero;
        result.z = (v0.z != v1.z) ? on : zero;
        result.w = (v0.w != v1.w) ? on : zero;
        return result;
    }

    __host__ __device__ Tuple2f::value_type distanceL2(Tuple2f a, Tuple2f b)
    {
        return glm::distance(*toGLM(&a), *toGLM(&b));
    }
    __host__ __device__ Tuple3f::value_type distanceL2(Tuple3f a, Tuple3f b)
    {
        return glm::distance(*toGLM(&a), *toGLM(&b));
    }
    __host__ __device__ Tuple4f::value_type distanceL2(Tuple4f a, Tuple4f b)
    {
        return glm::distance(*toGLM(&a), *toGLM(&b));
    }

    __host__ __device__ Tuple2f::value_type dotSelf(Tuple2f v) { return glm::length2(*toGLM(&v)); }
    __host__ __device__ Tuple3f::value_type dotSelf(Tuple3f v) { return glm::length2(*toGLM(&v)); }
    __host__ __device__ Tuple4f::value_type dotSelf(Tuple4f v) { return glm::length2(*toGLM(&v)); }

    // Vector Types: Geometric Functions ------------------------------------------------------------------------------
    __host__ __device__ float angleBetween(Normal3f a, Normal3f b) { return glm::dot(*toGLM(&a), *toGLM(&b)); }
    __host__ __device__ float angleBetween(Quaternion a, Quaternion b)
    {
        if (dot(a, b) < 0.f)
            return fl::twoPi() * fl::asinClamp(normL2(a + b) / 2);
        else
            return 2 * fl::asinClamp(normL2(b - a) / 2);
    }

    // quaternion rotation
    __host__ __device__ Quaternion fromRadians(float theta, Normal3f axis)
    {
        Quaternion  quat{};
        float const sinHalfAngle = sinf(theta * 0.5f);

        quat.w = cosf(theta * 0.5f);
        quat.x = axis.x * sinHalfAngle;
        quat.y = axis.y * sinHalfAngle;
        quat.z = axis.z * sinHalfAngle;
        assert(fl::abs(normL2(quat) - 1.f) < 1e-5f);

        return quat;
    }

    __host__ __device__ Quaternion conj(Quaternion quat)
    {
        Quaternion quatConj = -quat;
        quatConj.w          = quat.w;
        return quatConj;
    }

    __host__ __device__ DMT_FASTCALL Vector3f rotate(Quaternion quat, Vector3f v)
    {
        Quaternion pure{};
        pure.x                    = v.x;
        pure.y                    = v.y;
        pure.z                    = v.z;
        Quaternion const quatConj = conj(quat);

        pure = quat * pure * quatConj;

        return {pure.x, pure.y, pure.z};
    }

    __host__ __device__ DMT_FASTCALL Normal3f rotate(Quaternion quat, Normal3f v)
    {
        return normalize(rotate(quat, v.asVec()));
    }

    // frame and other utilities

    __host__ __device__ Frame coordinateSystem(Normal3f xAxis)
    {
        Frame frame;
        frame.xAxis    = xAxis;
        float     sign = fl::copysign(1.f, xAxis.z);
        float     a    = -1.f / (sign + xAxis.z);
        float     b    = xAxis.z * xAxis.y * a;
        glm::vec3 y    = {(1 + sign + xAxis.x * xAxis.x * a), (sign * b), (-sign * xAxis.x)};
        glm::vec3 z    = {(b), (sign + xAxis.y * xAxis.y * a), (-xAxis.y)};
        frame.yAxis    = normalFrom(*fromGLM(&y));
        frame.zAxis    = normalFrom(*fromGLM(&z));
        return frame;
    }

    __host__ __device__ void gramSchmidt(Vector3f n, Vector3f* a, Vector3f* b)
    {
        assert(a && b && fl::abs(normL2(n) - 1.f) < 1e-5f);
        if (n.x != n.y || n.x != n.z)
            *a = {n.z - n.y, n.x - n.z, n.y - n.x}; //(1,1,1)x N
        else
            *a = {n.z - n.y, n.x + n.z, -n.y - n.x}; //(-1,1,1)x N

        *a = normalize(*a);
        *b = cross(n, *a);
    }

    __host__ __device__ Quaternion slerp(float t, Quaternion zero, Quaternion one)
    {
        glm::quat q = glm::slerp(*toGLM(&one), *toGLM(&zero), t);
        return *fromGLMquat(&q);
    }

    // Vector Types: Spherical Geometry Functions ---------------------------------------------------------------------
    // https://brsr.github.io/2021/05/01/vector-spherical-geometry.html (Section Area of a Triangle)
    __host__ __device__ float sphericalTriangleArea(Vector3f edge0, Vector3f edge1, Vector3f edge2)
    {
        float scalarTripleProduct = dot(edge0, cross(edge1, edge2));
        float cosExcess           = 1 + dot(edge0, edge1) + dot(edge0, edge2) + dot(edge1, edge2);
        return glm::abs(2.f * fl::atan2(scalarTripleProduct, cosExcess));
    }
    __host__ __device__ float sphericalQuadArea(Vector3f edge0, Vector3f edge1, Vector3f edge2, Vector3f edge3)
    {
        Vector3f axb = cross(edge0, edge1), bxc = cross(edge1, edge2);
        Vector3f cxd = cross(edge2, edge3), dxa = cross(edge3, edge0);
        if (fl::nearZero(dotSelf(axb)) || fl::nearZero(dotSelf(bxc)) || fl::nearZero(dotSelf(cxd)) ||
            fl::nearZero(dotSelf(dxa)))
            return 0.f;

        axb = normalize(axb);
        bxc = normalize(bxc);
        cxd = normalize(cxd);
        dxa = normalize(dxa);

        float alpha = angleBetween(dxa, -axb);
        float beta  = angleBetween(axb, -bxc);
        float gamma = angleBetween(bxc, -cxd);
        float delta = angleBetween(cxd, -dxa);

        return glm::abs(alpha + beta + gamma + delta - fl::twoPi());
    }
    __host__ __device__ Vector3f sphericalDirection(float sinTheta, float cosTheta, float phi)
    {
        float clampedSinTheta = glm::clamp(sinTheta, -1.f, 1.f);
        return {{
            (clampedSinTheta * glm::cos(phi)),
            (clampedSinTheta * glm::sin(phi)),
            (glm::clamp(cosTheta, -1.f, 1.f)), // -1 ?
        }};
    }
    __host__ __device__ float sphericalTheta(Vector3f v) { return fl::acosClamp(v.z); }
    __host__ __device__ float sphericalPhi(Vector3f v)
    {
        float p = fl::atan2(v.y, v.z);
        return (p < 0.f) ? (p + fl::twoPi()) : p;
    }
    __host__ __device__ float cosTheta(Vector3f v) { return v.z; }
    __host__ __device__ float cos2Theta(Vector3f v) { return v.z * v.z; }
    __host__ __device__ float absCosTheta(Vector3f v) { return glm::abs(v.z); }
    __host__ __device__ float sin2Theta(Vector3f v) { return glm::max(0.f, 1.f - cos2Theta(v)); }
    __host__ __device__ float sinTheta(Vector3f v) { return glm::sqrt(sin2Theta(v)); }
    __host__ __device__ float tanTheta(Vector3f v) { return sinTheta(v) / cosTheta(v); }
    __host__ __device__ float tan2Theta(Vector3f v) { return sin2Theta(v) / cos2Theta(v); }
    __host__ __device__ float cosPhi(Vector3f v)
    {
        float sinTheta_ = sinTheta(v);
        return fl::nearZero(sinTheta_) ? 1.f : glm::clamp(v.x / sinTheta_, -1.f, 1.f);
    }
    __host__ __device__ float sinPhi(Vector3f v)
    {
        float sinTheta_ = sinTheta(v);
        return fl::nearZero(sinTheta_) ? 0.f : glm::clamp(v.y / sinTheta_, -1.f, 1.f);
    }
    __host__ __device__ float cosDPhi(Vector3f wa, Vector3f wb)
    {
        float waxy = wa.x * wa.x + wa.y * wa.y;
        float wbxy = wb.x * wb.x + wb.y * wb.y;
        if (fl::nearZero(waxy) || fl::nearZero(wbxy))
            return 1.f;

        return glm::clamp((wa.x * wb.x + wa.y * wb.y) / glm::sqrt(waxy * wbxy), -1.f, 1.f);
    }
    __host__ __device__ bool sameHemisphere(Vector3f w, Normal3f ap) { return w.z * ap.z > 0; }

    // Vector Types: Frame --------------------------------------------------------------------------------------------
    __host__ __device__ Frame::Frame(Normal3f x, Normal3f y, Normal3f z) : xAxis(x), yAxis(y), zAxis(z) {}

    __host__ __device__ Frame Frame::fromXZ(Normal3f x, Normal3f z) { return {x, cross(z, x), z}; }
    __host__ __device__ Frame Frame::fromXY(Normal3f x, Normal3f y) { return {x, y, cross(x, y)}; }
    __host__ __device__ Frame Frame::fromZ(Normal3f z)
    {
        Frame frame{};
        frame.zAxis = z;
        gramSchmidt(frame.zAxis.asVec(), &frame.xAxis.asVec(), &frame.yAxis.asVec());
        return frame;
    }
    __host__ __device__ Vector3f Frame::toLocal(Vector3f v) const
    {
        return {{dot(v, xAxis), dot(v, yAxis), dot(v, zAxis)}};
    }
    __host__ __device__ Normal3f Frame::toLocal(Normal3f n) const
    {
        assert(glm::epsilonEqual(dotSelf(n), 1.f, 1e-6f));
        // normalFrom = normalization. Mathematically, it's not needed
        // but floating point errors can prove otherwise. Stick an assertion and see if it explodes
        Normal3f ret{{dot(n, xAxis), dot(n, yAxis), dot(n, zAxis)}};
        assert(glm::epsilonEqual(dotSelf(ret), 1.f, 1e-6f));
        return ret;
    }
    __host__ __device__ Vector3f Frame::fromLocal(Vector3f v) const
    {
        return v.x * xAxis.asVec() + v.y * yAxis.asVec() + v.z * zAxis.asVec();
    }
    __host__ __device__ Normal3f Frame::fromLocal(Normal3f n) const
    {
        assert(glm::epsilonEqual(dotSelf(n), 1.f, 1e-6f));
        // normalFrom = normalization. Mathematically, it's not needed
        // but floating point errors can prove otherwise. Stick an assertion and see if it explodes
        Normal3f ret = n.x * xAxis.asVec() + n.y * yAxis.asVec() + n.z * zAxis.asVec();
        assert(glm::epsilonEqual(dotSelf(ret), 1.f, 1e-6f));
        return ret;
    }

    // Vector Types: Axis Aligned Bounding Boxes ----------------------------------------------------------------------
    // TODO: If more types of bounds are needed, refactor these into translation unit private templated functions called by the front facing ones
    __host__ __device__ bool inside(Point3f p, Bounds3f const& b)
    {
        return (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y && p.y <= b.pMax.y && p.z >= b.pMin.z &&
                p.z <= b.pMax.z);
    }

    __host__ __device__ Bounds3f makeBounds(Point3f p0, Point3f p1)
    {
        return {.pMin = min(p0, p1), .pMax = max(p0, p1)};
    }

    __host__ __device__ Bounds3f bbEmpty()
    {
        return {.pMin = {{fl::infinity(), fl::infinity(), fl::infinity()}},
                .pMax = {{-fl::infinity(), -fl::infinity(), -fl::infinity()}}};
    }

    __host__ __device__ Bounds3f bbUnion(Bounds3f const& a, Bounds3f const& b)
    {
        Bounds3f bRet;
        bRet.pMin = min(a.pMin, b.pMin);
        bRet.pMax = max(a.pMax, b.pMax);
        return bRet;
    }

    __host__ __device__ Bounds3f bbUnion(Bounds3f const& b, Point3f p)
    {
        Bounds3f bRet;
        bRet.pMin = min(b.pMin, p);
        bRet.pMax = max(b.pMax, p);
        return bRet;
    }

    __host__ __device__ Point3f& Bounds3f::operator[](int32_t i)
    {
        assert(i == 0 || i == 1);
        return i == 0 ? pMin : pMax;
    }

    __host__ __device__ Point3f const& Bounds3f::operator[](int32_t i) const
    {
        assert(i == 0 || i == 1);
        return i == 0 ? pMin : pMax;
    }

    __host__ __device__ Point3f Bounds3f::corner(EBoundsCorner corner) const
    {
        using enum EBoundsCorner;
        Point3f const ret{{
            operator[](toUnderlying(corner) & toUnderlying(eRight)).x,
            operator[]((toUnderlying(corner) & toUnderlying(eForward)) >> 1).y,
            operator[]((toUnderlying(corner) & toUnderlying(eTop)) >> 2).z,
        }};
        return ret;
    }

    __host__ __device__ Vector3f Bounds3f::diagonal() const { return pMax - pMin; }

    __host__ __device__ float Bounds3f::surfaceArea() const
    {
        Vector3f const d = diagonal();
        return 2 * (d.x * (d.y + d.z) + d.y * d.z);
    }

    __host__ __device__ float Bounds3f::volume() const
    {
        Vector3f const d = diagonal();
        return d.x * d.y * d.z;
    }

    __host__ __device__ int32_t Bounds3f::maxDimention() const
    {
        Vector3f const d = diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }

    __host__ __device__ Point3f Bounds3f::lerp(Point3f t) const
    {
        Point3f const ret{{glm::lerp(pMax.x, pMin.x, t.x), glm::lerp(pMax.y, pMin.y, t.y), glm::lerp(pMax.z, pMin.z, t.z)}};
        return ret;
    }

    __host__ __device__ Vector3f Bounds3f::offset(Point3f p) const
    {
        Vector3f o = p - pMin;
        if (pMax.x > pMin.x)
            o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y)
            o.y /= pMax.y - pMin.y;
        if (pMax.z > pMin.z)
            o.z /= pMax.z - pMin.z;
        return o;
    }

    __host__ __device__ void Bounds3f::boundingSphere(Point3f& outCenter, float& outRadius) const
    {
        outCenter = (pMin + pMax) / 2.f;
        outRadius = inside(outCenter, *this) ? glm::distance(*toGLM(&outCenter), *toGLM(&pMax)) : 0.f;
    }

    __host__ __device__ bool Bounds3f::isEmpty() const
    {
        return pMin.x >= pMax.x || pMin.y >= pMax.y || pMin.z >= pMax.z;
    }

    __host__ __device__ bool Bounds3f::isDegenerate() const
    {
        return pMin.x > pMax.x || pMin.y > pMax.y || pMin.z > pMax.z;
    }

    __host__ __device__ bool Bounds3f::operator==(Bounds3f const& that) const
    {
        bool b = near(pMin, that.pMin) && near(pMax, that.pMax);
        return b;
    }

    __host__ __device__ bool Bounds3f::intersectP(Point3f o, Vector3f d, float tMax, float* DMT_RESTRICT hit0, float* DMT_RESTRICT hit1) const
    {
        float t0 = 0, t1 = tMax;
        for (int32_t i = 0; i < 3; ++i)
        {
            float invRayDir = 1 / d[i];
            float tNear     = (pMin[i] - o[i]) * invRayDir;
            float tFar      = (pMax[i] - o[i]) * invRayDir;
            if (tNear > tFar)
            {
                float tmp = tNear;
                tNear     = tFar;
                tFar      = tmp;
            }
            tFar *= 1 + 2 * fl::gamma(3);

            t0 = tNear > t0 ? tNear : t0;
            t1 = tFar < t1 ? tFar : t1;
            if (t0 > t1)
                return false;
        }
        if (hit0)
            *hit0 = t0;
        if (hit1)
            *hit1 = t1;
        return true;
    }

    __host__ __device__ bool Bounds3f::intersectP(Point3f o, Vector3f d, float rayMax, Vector3f invDir, int32_t dirIsNeg[3]) const
    {
        Bounds3f const& self = *this;
        // check ray intersection agains x and y slabs
        float tMin  = (self[dirIsNeg[0]].x - o.x) * invDir.x;
        float tMax  = (self[1 - dirIsNeg[0]].x - o.x) * invDir.x;
        float tyMin = (self[dirIsNeg[1]].y - o.y) * invDir.y;
        float tyMax = (self[1 - dirIsNeg[1]].y - o.y) * invDir.y;

        // update tMax and tyMax to ensure robust intersection
        tMax *= 1 + 2 * fl::gamma(3);
        tyMax *= 1 + 2 * fl::gamma(3);

        if (tMin > tyMax || tyMin > tMax)
            return false;
        if (tyMin > tMin)
            tMin = tyMin;
        if (tyMax < tMax)
            tMax = tyMax;

        // check ray intersection against z slab
        float tzMin = (self[dirIsNeg[2]].z - o.z) * invDir.z;
        float tzMax = (self[1 - dirIsNeg[2]].z - o.z) * invDir.z;
        tzMax *= 1 + 2 * fl::gamma(3);

        if (tMin > tzMax || tzMin > tMax)
            return false;
        if (tzMin > tMin)
            tMin = tzMin;
        if (tzMax < tMax)
            tMax = tzMax;

        return (tMin < rayMax) && (tMax > 0);
    }

    // Bounds2f
    __host__ __device__ Bounds2f makeBounds(Point2f p0, Point2f p1)
    {
        return {.pMin = min(p0, p1), .pMax = max(p0, p1)};
    }

    __host__ __device__ Bounds2f bbEmpty2()
    {
        return {.pMin = {{fl::infinity(), fl::infinity()}}, .pMax = {{-fl::infinity(), -fl::infinity()}}};
    }

    __host__ __device__ bool inside(Point2f p, Bounds2f const& b)
    {
        return (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y && p.y <= b.pMax.y);
    }

    __host__ __device__ Bounds2f bbUnion(Bounds2f const& a, Bounds2f const& b)
    {
        Bounds2f bRet;
        bRet.pMin = min(a.pMin, b.pMin);
        bRet.pMax = max(a.pMax, b.pMax);
        return bRet;
    }

    __host__ __device__ Bounds2f bbUnion(Bounds2f const& b, Point2f p)
    {
        Bounds2f bRet;
        bRet.pMin = min(b.pMin, p);
        bRet.pMax = max(b.pMax, p);
        return bRet;
    }

    __host__ __device__ Point2f& Bounds2f::operator[](int32_t i)
    {
        assert(i == 0 || i == 1);
        return i == 0 ? pMin : pMax;
    }

    __host__ __device__ Point2f const& Bounds2f::operator[](int32_t i) const
    {
        assert(i == 0 || i == 1);
        return i == 0 ? pMin : pMax;
    }

    __host__ __device__ Point2f Bounds2f::corner(EBoundsCorner2 corner) const
    {
        using enum EBoundsCorner2;
        Point2f const ret{{
            operator[](toUnderlying(corner) & toUnderlying(eRight)).x,
            operator[]((toUnderlying(corner) & toUnderlying(eTop)) >> 1).y,
        }};
        return ret;
    }

    __host__ __device__ Vector2f Bounds2f::diagonal() const { return pMax - pMin; }

    __host__ __device__ float Bounds2f::surfaceArea() const
    {
        Vector2f const d = diagonal();
        return d.x * d.y;
    }

    __host__ __device__ float Bounds2f::volume() const { return 0; }

    __host__ __device__ int32_t Bounds2f::maxDimention() const
    {
        Vector2f const d = diagonal();
        if (d.x > d.y)
            return 0;
        else
            return 1;
    }

    __host__ __device__ Point2f Bounds2f::lerp(Point2f t) const
    {
        Point2f const ret{{glm::lerp(pMax.x, pMin.x, t.x), glm::lerp(pMax.y, pMin.y, t.y)}};
        return ret;
    }

    __host__ __device__ Vector2f Bounds2f::offset(Point2f p) const
    {
        Vector2f o = p - pMin;
        if (pMax.x > pMin.x)
            o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y)
            o.y /= pMax.y - pMin.y;
        return o;
    }

    __host__ __device__ void Bounds2f::boundingCircle(Point2f& outCenter, float& outRadius) const
    {
        outCenter = (pMin + pMax) / 2.f;
        outRadius = inside(outCenter, *this) ? glm::distance(*toGLM(&outCenter), *toGLM(&pMax)) : 0.f;
    }

    __host__ __device__ bool Bounds2f::isEmpty() const { return pMin.x >= pMax.x || pMin.y >= pMax.y; }

    __host__ __device__ bool Bounds2f::isDegenerate() const { return pMin.x > pMax.x || pMin.y > pMax.y; }

    __host__ __device__ bool Bounds2f::operator==(Bounds2f const& that) const
    {
        bool b = near(pMin, that.pMin) && near(pMax, that.pMax);
        return b;
    }

    // Bounds2i
    static glm::bvec2 operator>=(Point2i a, Point2i b) { return {a.x >= b.x, a.y >= b.y}; }
    static glm::bvec2 operator<=(Point2i a, Point2i b) { return {a.x <= b.x, a.y <= b.y}; }

    __host__ __device__ bool inside(Point2i p, Bounds2i b) { return glm::all(p >= b.pMin && p <= b.pMax); }

    __host__ __device__ void Bounds2i::boundingSphere(Point2i* c, float* rad) const
    {
        *c   = (pMin + pMax) / 2;
        *rad = inside(*c, *this) ? distanceL2(Point2f{{static_cast<float>(c->x), static_cast<float>(c->y)}},
                                              Point2f{{static_cast<float>(pMax.x), static_cast<float>(pMax.y)}})
                                 : 0;
    }

    // Vector Types: Matrix 4x4 ---------------------------------------------------------------------------------------
    __host__ __device__ Matrix4f givensRotation(int32_t axis0, int32_t axis1, float theta)
    {
        assert(axis0 > axis1 && axis0 >= 0 && axis1 >= 0 && axis0 < 4 && axis1 < 4);
        Matrix4f ret{Matrix4f::identity()};
        float    cosTheta        = glm::cos(theta);
        float    sinTheta        = glm::sin(theta);
        ret.m[axis0 + 4 * axis1] = sinTheta;
        ret.m[axis1 + 4 * axis0] = -sinTheta;
        ret.m[axis0 + 4 * axis0] = ret.m[axis1 + 4 * axis1] = cosTheta;
        return ret;
    }

    __host__ __device__ QR qr(Matrix4f const& m, int32_t numIter)
    {
        QR                    ret;
        alignas(16) glm::mat4 QT{1.f};
        alignas(16) glm::mat4 R = *toGLMmat(&m);
        for (int32_t iter = 0; iter < numIter; ++iter)
        {
            for (int32_t idx = 0; idx < 3; ++idx)
            {
                bool converge = true;
                for (int32_t numel = 0; numel <= idx; ++numel)
                {
                    Index2 i{.row = 3 - idx + numel, .col = numel};
                    assert(i.row > i.col);
                    float elem1 = reinterpret_cast<float*>(&R)[i.row + i.col * 4];
                    if (glm::epsilonEqual(elem1, 0.f, 1e-6f))
                        continue;

                    converge = false;

                    int32_t const cIdx  = glm::min(i.row, i.col);
                    float         elem0 = reinterpret_cast<float*>(&R)[cIdx + cIdx * 4];
                    //float theta = glm::acos(elem0 / glm::sqrt(elem0 * elem0 + elem1 * elem1));
                    float    theta = fl::atan2(-elem1, elem0);
                    Matrix4f G     = givensRotation(i.row, i.col, theta);
                    QT             = *toGLMmat(&G) * QT;
                    R              = *toGLMmat(&G) * R;
                }

                if (converge)
                    break;
            }
        }
        QT = glm::transpose(QT);

        ret.qOrthogonal = *fromGLMmat(&QT);
        ret.rUpper      = *fromGLMmat(&R);
        return ret;
    }

    #if !defined(__NVCC__) && !defined(__CUDA_ARCH__)
    __host__ SVD svd(Matrix4f const& m, uint32_t maxIterations = 100)
    {
        static constexpr int32_t numCols = 4;
        static constexpr int32_t numRows = 4;

        SVD             ret{.unitary = m, .singularValues = {{}}, .vunitary = Matrix4f::identity()};
        Eigen::Matrix4f matrix;
        matrix << m.m[0], m.m[1], m.m[2], m.m[3], m.m[4], m.m[5], m.m[6], m.m[7], m.m[8], m.m[9], m.m[10], m.m[11],
            m.m[12], m.m[13], m.m[14], m.m[15];
        Eigen::JacobiSVD<Eigen::Matrix4f> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::Vector4f singularValues = svd.singularValues();
        Eigen::Matrix4f U              = svd.matrixU();
        Eigen::Matrix4f V              = svd.matrixV();

        for (int32_t row = 0; row < numRows; ++row)
        {
            for (int32_t col = 0; col < numCols; ++col)
            {
                ret.unitary[{row, col}]  = U(row, col);
                ret.vunitary[{row, col}] = V(row, col);
            }
        }
        ret.singularValues.x = singularValues[0];
        ret.singularValues.y = singularValues[1];
        ret.singularValues.z = singularValues[2];
        ret.singularValues.w = singularValues[3];

        return ret;
    }

    __host__ bool isSingular(Matrix4f const& m, float tolerance)
    {
        assert(tolerance > 0.f);
        // Convert the input Matrix4f to Eigen::Matrix4f
        Eigen::Matrix4f matrix;
        matrix << m.m[0], m.m[1], m.m[2], m.m[3], m.m[4], m.m[5], m.m[6], m.m[7], m.m[8], m.m[9], m.m[10], m.m[11],
            m.m[12], m.m[13], m.m[14], m.m[15];

        // Perform SVD using JacobiSVD with only singular values computation
        Eigen::JacobiSVD<Eigen::Matrix4f> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Vector4f                   singularValues = svd.singularValues();

        // Check if any singular value is below the tolerance
        for (int i = 0; i < 4; ++i)
        {
            if (singularValues[i] < tolerance)
            {
                return true; // Matrix is singular
            }
        }

        return false; // Matrix is non-singular
    }
    #endif

    __host__ __device__ Matrix4f operator+(Matrix4f const& a, Matrix4f const& b)
    {
        alignas(16) glm::mat4 sum = *toGLMmat(&a) + *toGLMmat(&b);
        return *fromGLMmat(&sum);
    }
    __host__ __device__ Matrix4f operator-(Matrix4f const& a, Matrix4f const& b)
    {
        alignas(16) glm::mat4 sub = *toGLMmat(&a) - *toGLMmat(&b);
        return *fromGLMmat(&sub);
    }
    __host__ __device__ Matrix4f operator*(Matrix4f const& a, Matrix4f const& b)
    {
        alignas(16) glm::mat4 prod = *toGLMmat(&a) * *toGLMmat(&b);
        return *fromGLMmat(&prod);
    }

    __host__ __device__ Matrix4f operator*(float v, Matrix4f const& m)
    {
        Matrix4f ret = m;
        for (int32_t i = 0; i < 16; ++i)
            ret.m[i] *= v;
        return ret;
    }

    __host__ __device__ Matrix4f operator*(Matrix4f const& m, float v)
    {
        Matrix4f ret = m;
        for (int32_t i = 0; i < 16; ++i)
            ret.m[i] *= v;
        return ret;
    }

    __host__ __device__ Matrix4f operator/(Matrix4f const& m, float v)
    {
        Matrix4f ret = m;
        for (int32_t i = 0; i < 16; ++i)
            ret.m[i] /= v;
        return ret;
    }

    __host__ __device__ Matrix4f& operator+=(Matrix4f& a, Matrix4f const& b)
    {
        *toGLMmat(&a) += *toGLMmat(&b);
        return a;
    }
    __host__ __device__ Matrix4f& operator-=(Matrix4f& a, Matrix4f const& b)
    {
        *toGLMmat(&a) -= *toGLMmat(&b);
        return a;
    }
    __host__ __device__ Matrix4f& operator*=(Matrix4f& a, Matrix4f const& b)
    {
        *toGLMmat(&a) *= *toGLMmat(&b);
        return a;
    }

    __host__ __device__ Matrix4f fromDiag(Tuple4f v)
    {
        Matrix4f m{};
        for (int32_t i = 0; i < 16; i += 5)
        {
            int32_t j = i >> 2;
            m.m[i]    = v[j];
        }
        return m;
    }

    __host__ __device__ Matrix4f fromQuat(Quaternion q)
    {
        alignas(16) glm::mat4 mat = glm::mat4_cast(*toGLM(&q));
        return *fromGLMmat(&mat);
    }

    __host__ __device__ bool near(Matrix4f const& a, Matrix4f const& b)
    {
        bool result = true;
        for (uint32_t i = 0; i < 4; ++i)
        {
            result = near(a[i], b[i]);
            if (!result)
                return result;
        }
        return result;
    }

    __host__ __device__ float determinant(Matrix4f const& m) { return glm::determinant(*toGLMmat(&m)); }

    __host__ __device__ Matrix4f inverse(Matrix4f const& m)
    {
        alignas(16) glm::mat4 minv = glm::inverse(*toGLMmat(&m));
        return *fromGLMmat(&minv);
    }

    __host__ __device__ Matrix4f transpose(Matrix4f const& m)
    {
        alignas(16) glm::mat4 t = glm::transpose(*toGLMmat(&m));
        return *fromGLMmat(&t);
    }

    __host__ __device__ Vector4f mul(Matrix4f const& m, Vector4f v)
    {
        glm::vec4 res = *toGLMmat(&m) * *toGLM(&v);
        return *fromGLM(&res);
    }

    __host__ __device__ Vector3f mul(Matrix4f const& m, Vector3f const& v)
    {
        glm::vec4 w{v.x, v.y, v.z, 0.f};
        w = *toGLMmat(&m) * w;
        assert(fl::nearZero(w.w));
        return {{w.x, w.y, w.z}};
    }

    __host__ __device__ Normal3f mul(Matrix4f const& m, Normal3f const& v)
    {
        glm::vec4 w{v.x, v.y, v.z, 0.f};
        w = *toGLMmat(&m) * w;
        assert(fl::nearZero(w.w));
        w = glm::normalize(w);
        return {{w.x, w.y, w.z}};
    }

    __host__ __device__ Normal3f mulTranspose(Matrix4f const& m, Normal3f const& v)
    {
        Normal3f         ret;
        glm::vec4        w{v.x, v.y, v.z, 0.f};
        glm::mat4 const& glmMat = *toGLMmat(&m);

    #if 0 // 1Mil dollar question: who is the transpose?
        ret.x = glmMat[0][0] * w.x + glmMat[1][0] * w.y + glmMat[2][0] * w.z + glmMat[3][0] * w.w;
        ret.y = glmMat[0][1] * w.x + glmMat[1][1] * w.y + glmMat[2][1] * w.z + glmMat[3][1] * w.w;
        ret.z = glmMat[0][2] * w.x + glmMat[1][2] * w.y + glmMat[2][2] * w.z + glmMat[3][2] * w.w;
    #else
        ret.x = glmMat[0][0] * w.x + glmMat[0][1] * w.y + glmMat[0][2] * w.z + glmMat[0][3] * w.w;
        ret.y = glmMat[1][0] * w.x + glmMat[1][1] * w.y + glmMat[1][2] * w.z + glmMat[1][3] * w.w;
        ret.z = glmMat[2][0] * w.x + glmMat[2][1] * w.y + glmMat[2][2] * w.z + glmMat[2][3] * w.w;
    #endif
        ret = normalize(ret);

        return ret;
    }

    __host__ __device__ Point3f mul(Matrix4f const& m, Point3f const& p)
    {
        glm::vec4 w{p.x, p.y, p.z, 1.f};
        w = *toGLMmat(&m) * w;
        w /= w.w;
        return {{w.x, w.y, w.z}};
    }

    __host__ __device__ Point3fi mul(Matrix4f const& mat, Point3fi const& point)
    {
        // Multiply the matrix with the point (considering point as a 4D vector with w=1)
        Point3fi result;

        // Compute xLow and xHigh
        result.xLow  = mat[0][0] * point.xLow + mat[1][0] * point.yLow + mat[2][0] * point.zLow + mat[3][0];
        result.xHigh = mat[0][0] * point.xHigh + mat[1][0] * point.yHigh + mat[2][0] * point.zHigh + mat[3][0];

        // Compute yLow and yHigh
        result.yLow  = mat[0][1] * point.xLow + mat[1][1] * point.yLow + mat[2][1] * point.zLow + mat[3][1];
        result.yHigh = mat[0][1] * point.xHigh + mat[1][1] * point.yHigh + mat[2][1] * point.zHigh + mat[3][1];

        // Compute zLow and zHigh
        result.zLow  = mat[0][2] * point.xLow + mat[1][2] * point.yLow + mat[2][2] * point.zLow + mat[3][2];
        result.zHigh = mat[0][2] * point.xHigh + mat[1][2] * point.yHigh + mat[2][2] * point.zHigh + mat[3][2];

        return result;
    }

    // Vector Types: Intervals ----------------------------------------------------------------------------------------
    __host__ __device__ Point3fi::Point3fi(Point3f p) :
    xLow(p.x),
    yLow(p.y),
    zLow(p.z),
    xHigh(p.x),
    yHigh(p.y),
    zHigh(p.z)
    {
    }
    __host__ __device__ Point3fi::Point3fi(Point3f v, Vector3f error)
    {
        if (error.x == 0.f)
            xLow = xHigh = v.x;
        else
        {
            xLow  = fl::subRoundDown(v.x, error.x);
            xHigh = fl::addRoundUp(v.x, error.x);
        }
        if (error.y == 0.f)
            yLow = yHigh = v.y;
        else
        {
            yLow  = fl::subRoundDown(v.y, error.y);
            yHigh = fl::addRoundUp(v.y, error.y);
        }
        if (error.z == 0.f)
            zLow = zHigh = v.z;
        else
        {
            zLow  = fl::subRoundDown(v.z, error.z);
            zHigh = fl::addRoundUp(v.z, error.z);
        }
    }
    __host__ __device__ Point3f Point3fi::midpoint() const
    {
        return {static_cast<Tuple3<float>>(
            (*std::bit_cast<Vector3f const*>(&xLow) + *std::bit_cast<Vector3f const*>(&xHigh)) * 0.5f)};
    }
    __host__ __device__ Vector3f Point3fi::width() const
    {
        return *std::bit_cast<Point3f const*>(&xHigh) - *std::bit_cast<Point3f const*>(&xLow);
    }
    __host__ __device__ Point3fi operator+(Point3fi const& a, Point3fi const& b)
    {
        Point3fi ret;
    #if defined(__CUDA_ARCH__)
        ret.xLow  = __fadd_rd(a.xLow, b.xLow);
        ret.yLow  = __fadd_rd(a.yLow, b.yLow);
        ret.zLow  = __fadd_rd(a.zLow, b.zLow);
        ret.xHigh = __fadd_ru(a.xHigh, b.xHigh);
        ret.yHigh = __fadd_ru(a.yHigh, b.yHigh);
        ret.zHigh = __fadd_ru(a.zHigh, b.zHigh);
    #else
        uint32_t oldRounding = _MM_GET_ROUNDING_MODE();
        __m128   aLow        = _mm_loadu_ps(&a.xLow);
        __m128   bLow        = _mm_loadu_ps(&b.xLow);
        __m128   aHigh       = _mm_loadu_ps(&a.zLow);
        __m128   bHigh       = _mm_loadu_ps(&b.zLow);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        __m128 lowSum = _mm_add_ps(aLow, bLow);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        __m128 highSum = _mm_add_ps(aHigh, bHigh);

        _MM_SET_ROUNDING_MODE(oldRounding);
        _mm_storeu_ps(&ret.zLow, highSum);
        alignas(16) float low[4];
        _mm_storeu_ps(low, lowSum);
        ret.xLow = low[0];
        ret.yLow = low[1];
        ret.zLow = low[2];
    #endif
        return ret;
    }
    __host__ __device__ Point3fi operator-(Point3fi const& a, Point3fi const& b)
    {
        Point3fi ret;
    #if defined(__CUDA_ARCH__)
        ret.xLow  = __fsub_rd(a.xLow, b.xLow);
        ret.yLow  = __fsub_rd(a.yLow, b.yLow);
        ret.zLow  = __fsub_rd(a.zLow, b.zLow);
        ret.xHigh = __fsub_ru(a.xHigh, b.xHigh);
        ret.yHigh = __fsub_ru(a.yHigh, b.yHigh);
        ret.zHigh = __fsub_ru(a.zHigh, b.zHigh);
    #else
        uint32_t oldRounding = _MM_GET_ROUNDING_MODE();
        __m128   aLow        = _mm_loadu_ps(&a.xLow);
        __m128   bLow        = _mm_loadu_ps(&b.xLow);
        __m128   aHigh       = _mm_loadu_ps(&a.zLow);
        __m128   bHigh       = _mm_loadu_ps(&b.zLow);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        __m128 lowSum = _mm_sub_ps(aLow, bLow);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        __m128 highSum = _mm_sub_ps(aHigh, bHigh);

        _MM_SET_ROUNDING_MODE(oldRounding);
        _mm_storeu_ps(&ret.zLow, highSum);
        alignas(16) float low[4];
        _mm_storeu_ps(low, lowSum);
        ret.xLow = low[0];
        ret.yLow = low[1];
        ret.zLow = low[2];
    #endif
        return ret;
    }
    __host__ __device__ Point3fi operator*(Point3fi const& a, Point3fi const& b)
    {
        Point3fi ret;
    #if defined(__CUDA_ARCH__)
        ret.xLow  = __fmul_rd(a.xLow, b.xLow);
        ret.yLow  = __fmul_rd(a.yLow, b.yLow);
        ret.zLow  = __fmul_rd(a.zLow, b.zLow);
        ret.xHigh = __fmul_ru(a.xHigh, b.xHigh);
        ret.yHigh = __fmul_ru(a.yHigh, b.yHigh);
        ret.zHigh = __fmul_ru(a.zHigh, b.zHigh);
    #else
        uint32_t oldRounding = _MM_GET_ROUNDING_MODE();
        __m128   aLow        = _mm_loadu_ps(&a.xLow);
        __m128   bLow        = _mm_loadu_ps(&b.xLow);
        __m128   aHigh       = _mm_loadu_ps(&a.zLow);
        __m128   bHigh       = _mm_loadu_ps(&b.zLow);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        __m128 lowSum = _mm_mul_ps(aLow, bLow);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        __m128 highSum = _mm_mul_ps(aHigh, bHigh);

        _MM_SET_ROUNDING_MODE(oldRounding);
        _mm_storeu_ps(&ret.zLow, highSum);
        alignas(16) float low[4];
        _mm_storeu_ps(low, lowSum);
        ret.xLow = low[0];
        ret.yLow = low[1];
        ret.zLow = low[2];
    #endif
        return ret;
    }
    __host__ __device__ Point3fi operator/(Point3fi const& a, Point3fi const& b)
    {
        Point3fi ret;
    #if defined(__CUDA_ARCH__)
        ret.xLow  = __fdiv_rd(a.xLow, b.xLow);
        ret.yLow  = __fdiv_rd(a.yLow, b.yLow);
        ret.zLow  = __fdiv_rd(a.zLow, b.zLow);
        ret.xHigh = __fdiv_ru(a.xHigh, b.xHigh);
        ret.yHigh = __fdiv_ru(a.yHigh, b.yHigh);
        ret.zHigh = __fdiv_ru(a.zHigh, b.zHigh);
    #else
        uint32_t oldRounding = _MM_GET_ROUNDING_MODE();
        __m128   aLow        = _mm_loadu_ps(&a.xLow);
        __m128   bLow        = _mm_loadu_ps(&b.xLow);
        __m128   aHigh       = _mm_loadu_ps(&a.zLow);
        __m128   bHigh       = _mm_loadu_ps(&b.zLow);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
        __m128 lowSum = _mm_div_ps(aLow, bLow);

        _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
        __m128 highSum = _mm_div_ps(aHigh, bHigh);

        _MM_SET_ROUNDING_MODE(oldRounding);
        _mm_storeu_ps(&ret.zLow, highSum);
        alignas(16) float low[4];
        _mm_storeu_ps(low, lowSum);
        ret.xLow = low[0];
        ret.yLow = low[1];
        ret.zLow = low[2];
    #endif
        return ret;
    }

    // Ray and RayDifferentials ---------------------------------------------------------------------------------------
    __host__ __device__ Ray::Ray(Point3f o, Vector3f d, float time, uintptr_t medium) :
    medium(medium),
    o(o),
    d(d),
    d_inv(1 / d.x, 1 / d.y, 1 / d.z),
    time(time)
    {
    }

    __host__ __device__ bool Ray::hasNaN() const { return ::dmt::hasNaN(o) || ::dmt::hasNaN(d); }

    __host__ __device__ uintptr_t Ray::getMedium() const { return medium & ~1; }

    __host__ __device__ RayDifferential::RayDifferential(Point3f o, Vector3f d, float time, uintptr_t medium) :
    Ray(o, d, time, medium)
    {
    }

    __host__ __device__ RayDifferential::RayDifferential(Ray const& ray) : Ray(ray) {}

    __host__ __device__ void RayDifferential::setDifferentials(
        Point3f  _rxOrigin,
        Vector3f _rxDirection,
        Point3f  _ryOrigin,
        Vector3f _ryDirection)
    {
        rxOrigin    = _rxOrigin;
        ryOrigin    = _ryOrigin;
        rxDirection = _rxDirection;
        ryDirection = _ryDirection;
        medium |= 1;
    }

    __host__ __device__ void RayDifferential::scaleDifferentials(float s)
    {
        rxOrigin    = o + (rxOrigin - o) * s;
        ryOrigin    = o + (ryOrigin - o) * s;
        rxDirection = d + (rxDirection - d) * s;
        ryDirection = d + (ryDirection - d) * s;
    }

    // math utilities: vector -----------------------------------------------------------------------------------------
} // namespace dmt
#endif