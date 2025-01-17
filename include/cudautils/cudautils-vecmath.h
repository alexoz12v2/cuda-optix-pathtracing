#pragma once

#include "dmtmacros.h"

#include <platform/platform-utils.h>
#include "cudautils/cudautils-float.h"

#include <array>

#include <cassert>
#include <cstdint>
#include <cmath>


namespace dmt {
    // Vector Types ---------------------------------------------------------------------------------------------------
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

    template <Scalar S>
    struct Tuple2
    {
        using value_type = S;
        static consteval int32_t numComponents() { return 2; }
        DMT_CPU_GPU S&           operator[](int32_t i)
        {
            assert(i >= 0 && i < 2);
            return *(std::bit_cast<S*>(this) + i);
        }
        DMT_CPU_GPU S const& operator[](int32_t i) const
        {
            assert(i >= 0 && i < 2);
            return *(std::bit_cast<S const*>(this) + i);
        }
        S x, y;
    };

    template <Scalar S>
    struct Tuple3
    {
        using value_type = S;
        static consteval int32_t numComponents() { return 3; }
        DMT_CPU_GPU S&           operator[](int32_t i)
        {
            assert(i >= 0 && i < 3);
            return *(std::bit_cast<S*>(this) + i);
        }
        DMT_CPU_GPU S const& operator[](int32_t i) const
        {
            assert(i >= 0 && i < 3);
            return *(std::bit_cast<S const*>(this) + i);
        }
        S x, y, z;
    };

    template <Scalar S>
    struct Tuple4
    {
        using value_type = S;
        static consteval int32_t numComponents() { return 4; }
        DMT_CPU_GPU S&           operator[](int32_t i)
        {
            assert(i >= 0 && i < 4);
            return *(std::bit_cast<S*>(this) + i);
        }
        DMT_CPU_GPU S const& operator[](int32_t i) const
        {
            assert(i >= 0 && i < 4);
            return *(std::bit_cast<S const*>(this) + i);
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

    struct Vector2i : public Tuple2i { Vector2i() = default; DMT_CPU_GPU Vector2i(Tuple2i t) : Tuple2i(t) {} };
    struct Vector2f : public Tuple2f { Vector2f() = default; DMT_CPU_GPU Vector2f(Tuple2f t) : Tuple2f(t) {} };
    struct Vector3i : public Tuple3i { Vector3i() = default; DMT_CPU_GPU Vector3i(Tuple3i t) : Tuple3i(t) {} };
    struct Vector3f : public Tuple3f { Vector3f() = default; DMT_CPU_GPU Vector3f(Tuple3f t) : Tuple3f(t) {} };
    struct Vector4i : public Tuple4i { Vector4i() = default; DMT_CPU_GPU Vector4i(Tuple4i t) : Tuple4i(t) {} };
    struct Vector4f : public Tuple4f { Vector4f() = default; DMT_CPU_GPU Vector4f(Tuple4f t) : Tuple4f(t) {} };

    struct Point2i : public Tuple2i { Point2i() = default; DMT_CPU_GPU Point2i(Tuple2i t) : Tuple2i(t) {} explicit DMT_CPU_GPU operator Vector2i(); };
    struct Point2f : public Tuple2f { Point2f() = default; DMT_CPU_GPU Point2f(Tuple2f t) : Tuple2f(t) {} explicit DMT_CPU_GPU operator Vector2f(); };
    struct Point3i : public Tuple3i { Point3i() = default; DMT_CPU_GPU Point3i(Tuple3i t) : Tuple3i(t) {} explicit DMT_CPU_GPU operator Vector3i(); };
    struct Point3f : public Tuple3f { Point3f() = default; DMT_CPU_GPU Point3f(Tuple3f t) : Tuple3f(t) {} explicit DMT_CPU_GPU operator Vector3f(); };
    struct Point4i : public Tuple4i { Point4i() = default; DMT_CPU_GPU Point4i(Tuple4i t) : Tuple4i(t) {} explicit DMT_CPU_GPU operator Vector4i(); };
    struct Point4f : public Tuple4f { Point4f() = default; DMT_CPU_GPU Point4f(Tuple4f t) : Tuple4f(t) {} explicit DMT_CPU_GPU operator Vector4f(); };

    struct Normal2f : public Tuple2f, public Normalized { Normal2f() = default; DMT_CPU_GPU Normal2f(Tuple2f t) : Tuple2f(t) {} };
    struct Normal3f : public Tuple3f, public Normalized { Normal3f() = default; DMT_CPU_GPU Normal3f(Tuple3f t) : Tuple3f(t) {} };

    struct Quaternion : public Tuple4f { Quaternion() = default; DMT_CPU_GPU Quaternion(Tuple4f t) : Tuple4f(t) {} };

    struct Frame { Normal3f xAxis, yAxis, zAxis; };
    // clang-format on
    // TODO compressed normal, half floats


    // Vector Types: Fundamental Operations ---------------------------------------------------------------------------
    DMT_CPU_GPU Point2i    operator+(Point2i a, Vector2i b);
    DMT_CPU_GPU Point2f    operator+(Point2f a, Vector2f b);
    DMT_CPU_GPU Point3i    operator+(Point3i a, Vector3i b);
    DMT_CPU_GPU Point3f    operator+(Point3f a, Vector3f b);
    DMT_CPU_GPU Point4i    operator+(Point4i a, Vector4i b);
    DMT_CPU_GPU Point4f    operator+(Point4f a, Vector4f b);
    DMT_CPU_GPU Vector2i   operator+(Vector2i a, Vector2i b);
    DMT_CPU_GPU Vector2f   operator+(Vector2f a, Vector2f b);
    DMT_CPU_GPU Vector3i   operator+(Vector3i a, Vector3i b);
    DMT_CPU_GPU Vector3f   operator+(Vector3f a, Vector3f b);
    DMT_CPU_GPU Vector4i   operator+(Vector4i a, Vector4i b);
    DMT_CPU_GPU Vector4f   operator+(Vector4f a, Vector4f b);
    DMT_CPU_GPU Normal2f   operator+(Normal2f a, Normal2f b);
    DMT_CPU_GPU Normal3f   operator+(Normal3f a, Normal3f b);
    DMT_CPU_GPU Quaternion operator+(Quaternion a, Quaternion b);

    DMT_CPU_GPU Vector2i   operator-(Point2i a, Point2i b);
    DMT_CPU_GPU Vector2f   operator-(Point2f a, Point2f b);
    DMT_CPU_GPU Vector3i   operator-(Point3i a, Point3i b);
    DMT_CPU_GPU Vector3f   operator-(Point3f a, Point3f b);
    DMT_CPU_GPU Vector4i   operator-(Point4i a, Point4i b);
    DMT_CPU_GPU Vector4f   operator-(Point4f a, Point4f b);
    DMT_CPU_GPU Vector2i   operator-(Vector2i a, Vector2i b);
    DMT_CPU_GPU Vector2f   operator-(Vector2f a, Vector2f b);
    DMT_CPU_GPU Vector3i   operator-(Vector3i a, Vector3i b);
    DMT_CPU_GPU Vector3f   operator-(Vector3f a, Vector3f b);
    DMT_CPU_GPU Vector4i   operator-(Vector4i a, Vector4i b);
    DMT_CPU_GPU Vector4f   operator-(Vector4f a, Vector4f b);
    DMT_CPU_GPU Normal2f   operator-(Normal2f a, Normal2f b);
    DMT_CPU_GPU Normal3f   operator-(Normal3f a, Normal3f b);
    DMT_CPU_GPU Quaternion operator-(Quaternion a, Quaternion b);

    DMT_CPU_GPU Vector2i   operator-(Point2i v);
    DMT_CPU_GPU Vector2f   operator-(Point2f v);
    DMT_CPU_GPU Vector3i   operator-(Point3i v);
    DMT_CPU_GPU Vector3f   operator-(Point3f v);
    DMT_CPU_GPU Vector4i   operator-(Point4i v);
    DMT_CPU_GPU Vector4f   operator-(Point4f v);
    DMT_CPU_GPU Vector2i   operator-(Vector2i v);
    DMT_CPU_GPU Vector2f   operator-(Vector2f v);
    DMT_CPU_GPU Vector3i   operator-(Vector3i v);
    DMT_CPU_GPU Vector3f   operator-(Vector3f v);
    DMT_CPU_GPU Vector4i   operator-(Vector4i v);
    DMT_CPU_GPU Vector4f   operator-(Vector4f v);
    DMT_CPU_GPU Normal2f   operator-(Normal2f v);
    DMT_CPU_GPU Normal3f   operator-(Normal3f v);
    DMT_CPU_GPU Quaternion operator-(Quaternion q);

    DMT_CPU_GPU Vector2i   operator*(Vector2i a, Vector2i b);
    DMT_CPU_GPU Vector2f   operator*(Vector2f a, Vector2f b);
    DMT_CPU_GPU Vector3i   operator*(Vector3i a, Vector3i b);
    DMT_CPU_GPU Vector3f   operator*(Vector3f a, Vector3f b);
    DMT_CPU_GPU Vector4i   operator*(Vector4i a, Vector4i b);
    DMT_CPU_GPU Vector4f   operator*(Vector4f a, Vector4f b);
    DMT_CPU_GPU Quaternion operator*(Quaternion a, Quaternion b);

    DMT_CPU_GPU Vector2i operator/(Vector2i a, Vector2i b);
    DMT_CPU_GPU Vector2f operator/(Vector2f a, Vector2f b);
    DMT_CPU_GPU Vector3i operator/(Vector3i a, Vector3i b);
    DMT_CPU_GPU Vector3f operator/(Vector3f a, Vector3f b);
    DMT_CPU_GPU Vector4i operator/(Vector4i a, Vector4i b);
    DMT_CPU_GPU Vector4f operator/(Vector4f a, Vector4f b);

    DMT_CPU_GPU Point2i&    operator+=(Point2i& a, Vector2i b);
    DMT_CPU_GPU Point2f&    operator+=(Point2f& a, Vector2f b);
    DMT_CPU_GPU Point3i&    operator+=(Point3i& a, Vector3i b);
    DMT_CPU_GPU Point3f&    operator+=(Point3f& a, Vector3f b);
    DMT_CPU_GPU Point4i&    operator+=(Point4i& a, Vector4i b);
    DMT_CPU_GPU Point4f&    operator+=(Point4f& a, Vector4f b);
    DMT_CPU_GPU Vector2i&   operator+=(Vector2i& a, Vector2i b);
    DMT_CPU_GPU Vector2f&   operator+=(Vector2f& a, Vector2f b);
    DMT_CPU_GPU Vector3i&   operator+=(Vector3i& a, Vector3i b);
    DMT_CPU_GPU Vector3f&   operator+=(Vector3f& a, Vector3f b);
    DMT_CPU_GPU Vector4i&   operator+=(Vector4i& a, Vector4i b);
    DMT_CPU_GPU Vector4f&   operator+=(Vector4f& a, Vector4f b);
    DMT_CPU_GPU Normal2f&   operator+=(Normal2f& a, Normal2f b);
    DMT_CPU_GPU Normal3f&   operator+=(Normal3f& a, Normal3f b);
    DMT_CPU_GPU Quaternion& operator+=(Quaternion& a, Quaternion b);

    DMT_CPU_GPU Vector2i&   operator-=(Vector2i& a, Vector2i b);
    DMT_CPU_GPU Vector2f&   operator-=(Vector2f& a, Vector2f b);
    DMT_CPU_GPU Vector3i&   operator-=(Vector3i& a, Vector3i b);
    DMT_CPU_GPU Vector3f&   operator-=(Vector3f& a, Vector3f b);
    DMT_CPU_GPU Vector4i&   operator-=(Vector4i& a, Vector4i b);
    DMT_CPU_GPU Vector4f&   operator-=(Vector4f& a, Vector4f b);
    DMT_CPU_GPU Normal2f&   operator-=(Normal2f& a, Normal2f b);
    DMT_CPU_GPU Normal3f&   operator-=(Normal3f& a, Normal3f b);
    DMT_CPU_GPU Quaternion& operator-=(Quaternion& a, Quaternion b);

    DMT_CPU_GPU Vector2i&   operator*=(Vector2i& a, Vector2i b);
    DMT_CPU_GPU Vector2f&   operator*=(Vector2f& a, Vector2f b);
    DMT_CPU_GPU Vector3i&   operator*=(Vector3i& a, Vector3i b);
    DMT_CPU_GPU Vector3f&   operator*=(Vector3f& a, Vector3f b);
    DMT_CPU_GPU Vector4i&   operator*=(Vector4i& a, Vector4i b);
    DMT_CPU_GPU Vector4f&   operator*=(Vector4f& a, Vector4f b);
    DMT_CPU_GPU Quaternion& operator*=(Quaternion& a, Quaternion b);

    DMT_CPU_GPU Vector2i& operator/=(Vector2i& a, Vector2i b);
    DMT_CPU_GPU Vector2f& operator/=(Vector2f& a, Vector2f b);
    DMT_CPU_GPU Vector3i& operator/=(Vector3i& a, Vector3i b);
    DMT_CPU_GPU Vector3f& operator/=(Vector3f& a, Vector3f b);
    DMT_CPU_GPU Vector4i& operator/=(Vector4i& a, Vector4i b);
    DMT_CPU_GPU Vector4f& operator/=(Vector4f& a, Vector4f b);


    // Vector Types: Common Inline Methods ----------------------------------------------------------------------------
    template <Vector T>
    DMT_CPU_GPU inline T bcast(typename T::value_type t)
    {
        T ret;
        for (int32_t i = 0; i < T::numComponents(); ++i)
            ret[i] = t;
        return ret;
    }

    template <VectorScalable T>
    DMT_CPU_GPU inline T operator*(T v, typename T::value_type k)
    {
        T ret;
        for (int32_t i = 0; i < T::numComponents(); ++i)
            ret[i] = k * v[i];
        return ret;
    }
    template <VectorScalable T>
    DMT_CPU_GPU inline T operator*(typename T::value_type k, T v)
    {
        T ret;
        for (int32_t i = 0; i < T::numComponents(); ++i)
            ret[i] = k * v[i];
        return ret;
    }
    template <VectorScalable T>
    DMT_CPU_GPU inline T operator/(T v, typename T::value_type k)
    {
        T ret;
        for (int32_t i = 0; i < T::numComponents(); ++i)
            ret[i] = v[i] / k;
        return ret;
    }
    template <VectorScalable T>
    DMT_CPU_GPU inline T operator/(typename T::value_type k, T v)
    {
        T ret;
        for (int32_t i = 0; i < T::numComponents(); ++i)
            ret[i] = k / v[i];
        return ret;
    }

    template <VectorScalable T>
    DMT_CPU_GPU inline T& operator*=(T& v, typename T::value_type k)
    {
        for (int32_t i = 0; i < T::numComponents(); ++i)
            v[i] *= k;
        return v;
    }
    template <VectorScalable T>
    DMT_CPU_GPU inline T& operator/=(T& v, typename T::value_type k)
    {
        T ret;
        for (int32_t i = 0; i < T::numComponents(); ++i)
            ret[i] /= k;
        return ret;
    }

    // Vector Types: Component Operations -----------------------------------------------------------------------------
    template <Vector T>
    DMT_CPU_GPU inline int32_t maxComponentIndex(T v)
    {
        int32_t max = 0;
        for (int32_t i = 1; i < T::numComponents(); ++i)
            if (v[max] < v[i])
                max = i;

        return max;
    }

    template <Vector T>
    DMT_CPU_GPU inline int32_t minComponentIndex(T v)
    {
        int32_t min = 0;
        for (int32_t i = 1; i < T::numComponents(); ++i)
            if (v[min] > v[i])
                min = i;

        return min;
    }

    template <Vector T>
        requires(T::numComponents() == 2)
    DMT_CPU_GPU inline T permute(T v, int32_t i0, int32_t i1)
    {
        T ret;
        ret[0] = v[i0];
        ret[1] = v[i1];
        return ret;
    }
    template <Vector T>
        requires(T::numComponents() == 3)
    DMT_CPU_GPU inline T permute(T v, int32_t i0, int32_t i1, int32_t i2)
    {
        T ret;
        ret[0] = v[i0];
        ret[1] = v[i1];
        ret[2] = v[i2];
        return ret;
    }
    template <Vector T>
        requires(T::numComponents() == 4)
    DMT_CPU_GPU inline T permute(T v, int32_t i0, int32_t i1, int32_t i2, int32_t i3)
    {
        T ret;
        ret[0] = v[i0];
        ret[1] = v[i1];
        ret[2] = v[i2];
        ret[3] = v[i3];
        return ret;
    }

    template <Vector T>
    DMT_CPU_GPU inline T::value_type hprod(T v)
    {
        typename T::value_type ret = v[0];
        for (int32_t i = 1; i < T::numComponents(); ++i)
            ret *= v[i];
        return ret;
    }

    template <Vector T>
        requires std::is_floating_point_v<typename T::value_type>
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

    DMT_CPU_GPU Normal2f normalFrom(Vector2f v);
    DMT_CPU_GPU Normal3f normalFrom(Vector3f v);

    // Vector Types: Generic Tuple Operations -------------------------------------------------------------------------
    // If any of these are used to initialize a Normal, it has to be manually normalized
    DMT_CPU_GPU Tuple2f abs(Tuple2f v);
    DMT_CPU_GPU Tuple2i abs(Tuple2i v);
    DMT_CPU_GPU Tuple3f abs(Tuple3f v);
    DMT_CPU_GPU Tuple3i abs(Tuple3i v);
    DMT_CPU_GPU Tuple4f abs(Tuple4f v);
    DMT_CPU_GPU Tuple4i abs(Tuple4i v);

    DMT_CPU_GPU Tuple2f ceil(Tuple2f v);
    DMT_CPU_GPU Tuple3f ceil(Tuple3f v);
    DMT_CPU_GPU Tuple4f ceil(Tuple4f v);

    DMT_CPU_GPU Tuple2f floor(Tuple2f v);
    DMT_CPU_GPU Tuple3f floor(Tuple3f v);
    DMT_CPU_GPU Tuple4f floor(Tuple4f v);

    DMT_CPU_GPU Tuple2f lerp(float t, Tuple2f zero, Tuple2f one);
    DMT_CPU_GPU Tuple3f lerp(float t, Tuple3f zero, Tuple3f one);
    DMT_CPU_GPU Tuple4f lerp(float t, Tuple4f zero, Tuple4f one);

    DMT_CPU_GPU Tuple2f fma(Tuple2f mult0, Tuple2f mult1, Tuple2f add);
    DMT_CPU_GPU Tuple2i fma(Tuple2i mult0, Tuple2i mult1, Tuple2i add);
    DMT_CPU_GPU Tuple3f fma(Tuple3f mult0, Tuple3f mult1, Tuple3f add);
    DMT_CPU_GPU Tuple3i fma(Tuple3i mult0, Tuple3i mult1, Tuple3i add);
    DMT_CPU_GPU Tuple4f fma(Tuple4f mult0, Tuple4f mult1, Tuple4f add);
    DMT_CPU_GPU Tuple4i fma(Tuple4i mult0, Tuple4i mult1, Tuple4i add);

    DMT_CPU_GPU Tuple2f min(Tuple2f a, Tuple2f b);
    DMT_CPU_GPU Tuple2i min(Tuple2i a, Tuple2i b);
    DMT_CPU_GPU Tuple3f min(Tuple3f a, Tuple3f b);
    DMT_CPU_GPU Tuple3i min(Tuple3i a, Tuple3i b);
    DMT_CPU_GPU Tuple4f min(Tuple4f a, Tuple4f b);
    DMT_CPU_GPU Tuple4i min(Tuple4i a, Tuple4i b);

    DMT_CPU_GPU Tuple2f max(Tuple2f a, Tuple2f b);
    DMT_CPU_GPU Tuple2i max(Tuple2i a, Tuple2i b);
    DMT_CPU_GPU Tuple3f max(Tuple3f a, Tuple3f b);
    DMT_CPU_GPU Tuple3i max(Tuple3i a, Tuple3i b);
    DMT_CPU_GPU Tuple4f max(Tuple4f a, Tuple4f b);
    DMT_CPU_GPU Tuple4i max(Tuple4i a, Tuple4i b);

    DMT_CPU_GPU bool near(Tuple2f a, Tuple2f b, float tolerance = fl::eqTol());
    DMT_CPU_GPU bool near(Tuple2i a, Tuple2i b);
    DMT_CPU_GPU bool near(Tuple3f a, Tuple3f b, float tolerance = fl::eqTol());
    DMT_CPU_GPU bool near(Tuple3i a, Tuple3i b);
    DMT_CPU_GPU bool near(Tuple4f a, Tuple4f b, float tolerance = fl::eqTol());
    DMT_CPU_GPU bool near(Tuple4i a, Tuple4i b);

    DMT_CPU_GPU Tuple2f::value_type dot(Tuple2f a, Tuple2f b);
    DMT_CPU_GPU Tuple3f::value_type dot(Tuple3f a, Tuple3f b);
    DMT_CPU_GPU Tuple4f::value_type dot(Tuple4f a, Tuple4f b);

    DMT_CPU_GPU Tuple2f::value_type absDot(Tuple2f a, Tuple2f b);
    DMT_CPU_GPU Tuple3f::value_type absDot(Tuple3f a, Tuple3f b);
    DMT_CPU_GPU Tuple4f::value_type absDot(Tuple4f a, Tuple4f b);

    DMT_CPU_GPU Tuple3f cross(Tuple3f a, Tuple3f b);

    DMT_CPU_GPU Tuple2f normalize(Tuple2f v);
    DMT_CPU_GPU Tuple3f normalize(Tuple3f v);
    DMT_CPU_GPU Tuple4f normalize(Tuple4f v);

    DMT_CPU_GPU Tuple2f::value_type normL2(Tuple2f v);
    DMT_CPU_GPU Tuple3f::value_type normL2(Tuple3f v);
    DMT_CPU_GPU Tuple4f::value_type normL2(Tuple4f v);

    DMT_CPU_GPU Tuple2f::value_type distanceL2(Tuple2f a, Tuple2f b);
    DMT_CPU_GPU Tuple3f::value_type distanceL2(Tuple3f a, Tuple3f b);
    DMT_CPU_GPU Tuple4f::value_type distanceL2(Tuple4f a, Tuple4f b);

    DMT_CPU_GPU Tuple2f::value_type dotSelf(Tuple2f v);
    DMT_CPU_GPU Tuple3f::value_type dotSelf(Tuple3f v);
    DMT_CPU_GPU Tuple4f::value_type dotSelf(Tuple4f v);

    // Vector Types: Geometric Functions ------------------------------------------------------------------------------
    template <VectorNormalized N, Vector V>
        requires(std::is_same_v<typename N::value_type, typename V::value_type> && N::numComponents() == V::numComponents())
    DMT_CPU_GPU inline N faceForward(N direction, V vector)
    {
        return (dot(direction, vector) < 0.f) ? -direction : direction;
    }

    DMT_CPU_GPU float angleBetween(Normal3f a, Normal3f b);
    DMT_CPU_GPU float angleBetween(Quaternion a, Quaternion b);

    template <Vector V>
    DMT_CPU_GPU inline V gramSchmidt(V v, V w)
    {
        V dt = {dot(v, w)};
        return v * dt * w;
    }

    DMT_CPU_GPU Frame coordinateSystem(Normal3f xAxis);

    DMT_CPU_GPU Quaternion slerp(float t, Quaternion zero, Quaternion one);

    // Vector Types: Spherical Geometry Functions ---------------------------------------------------------------------
    DMT_CPU_GPU float    sphericalTriangleArea(Vector3f edge0, Vector3f edge1, Vector3f edge2);
    DMT_CPU_GPU float    sphericalQuadArea(Vector3f edge0, Vector3f edge1, Vector3f edge2, Vector3f edge3);
    DMT_CPU_GPU Vector3f sphericalDirection(float sinTheta, float cosTheta, float phi);
    DMT_CPU_GPU float    sphericalTheta(Vector3f v);
    DMT_CPU_GPU float    sphericalPhi(Vector3f v);
    DMT_CPU_GPU float    cosTheta(Vector3f v);
    DMT_CPU_GPU float    cos2Theta(Vector3f v);
    DMT_CPU_GPU float    absCosTheta(Vector3f v);
    DMT_CPU_GPU float    sinTheta(Vector3f v);
    DMT_CPU_GPU float    sin2Theta(Vector3f v);
    DMT_CPU_GPU float    tanTheta(Vector3f v);
    DMT_CPU_GPU float    tan2Theta(Vector3f v);
    DMT_CPU_GPU float    cosPhi(Vector3f v);
    DMT_CPU_GPU float    sinPhi(Vector3f v);
    DMT_CPU_GPU float    cosDPhi(Vector3f wa, Vector3f wb);
    DMT_CPU_GPU bool     sameHemisphere(Vector3f w, Normal3f ap);

    // Vector Types: Axis Aligned Bounding Boxes ----------------------------------------------------------------------
    enum EBoundsCorner : int32_t
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
    struct Bounds3f
    {
        DMT_CPU_GPU Point3f&       operator[](int32_t i);
        DMT_CPU_GPU Point3f const& operator[](int32_t i) const;
        DMT_CPU_GPU Point3f        corner(EBoundsCorner corner) const;
        DMT_CPU_GPU Vector3f       diagonal() const;
        DMT_CPU_GPU float          surfaceAraa() const;
        DMT_CPU_GPU float          volume() const;
        DMT_CPU_GPU int32_t        maxDimention() const;
        DMT_CPU_GPU Point3f        lerp(Point3f t) const;
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

    DMT_CPU_GPU bool     inside(Point3f p, Bounds3f const& b);
    DMT_CPU_GPU Bounds3f bbUnion(Bounds3f const& a, Bounds3f const& b);
    DMT_CPU_GPU Bounds3f bbUnion(Bounds3f const& b, Point3f p);

    // Vector Types: Matrix 4x4 ---------------------------------------------------------------------------------------
    struct Index2
    {
        int32_t row, col;
    };

    DMT_CPU_GPU inline constexpr Index2 sym(Index2 i)
    {
        Index2 const j{
            .row = i.col,
            .col = i.row,
        };
        return j;
    }

    // Column Major Order
    struct Matrix4f
    {
        // clang-format off
        static DMT_CPU_GPU inline constexpr Matrix4f zero()
        {
            return {{
                0.f, 0.f, 0.f, 0.f, // column zero
                0.f, 0.f, 0.f, 0.f, // column one
                0.f, 0.f, 0.f, 0.f, // column two
                0.f, 0.f, 0.f, 0.f  // column three
            }};
        }
        static DMT_CPU_GPU inline constexpr Matrix4f identity()
        {
            return {{
                1.f, 0.f, 0.f, 0.f, // column zero
                0.f, 1.f, 0.f, 0.f, // column one
                0.f, 0.f, 1.f, 0.f, // column two
                0.f, 0.f, 0.f, 1.f  // column three
            }};
        }
        // clang-format on
        DMT_CPU_GPU inline float&          operator[](Index2 i) { return m[i.col * 4 + i.row]; }
        DMT_CPU_GPU inline float const&    operator[](Index2 i) const { return m[i.col * 4 + i.row]; }
        DMT_CPU_GPU inline Vector4f&       operator[](int32_t i) { return *std::bit_cast<Vector4f*>(&m[i * 4]); }
        DMT_CPU_GPU inline Vector4f const& operator[](int32_t i) const
        {
            return *std::bit_cast<Vector4f const*>(&m[i * 4]);
        }

        float m[16];
    };
    static_assert(std::is_trivial_v<Matrix4f> && std::is_standard_layout_v<Matrix4f>);

    struct SVD
    {
        Matrix4f unitary;
        Vector4f singularValues;
        Matrix4f vunitary;
    };

    struct QR
    {
        Matrix4f qOrthogonal;
        Matrix4f rUpper;
    };

    // define a rotation in the plane defined by two axes
    DMT_CPU_GPU Matrix4f givensRotation(int32_t axis0, int32_t axis1, float theta);
    DMT_CPU_GPU QR       qr(Matrix4f const& m, int32_t numIter = 10);
    DMT_CPU SVD          svd(Matrix4f const& m);
    DMT_CPU bool         isSingular(Matrix4f const& m, float tolerance = 1e-6f);

    DMT_CPU_GPU Matrix4f operator+(Matrix4f const& a, Matrix4f const& b);
    DMT_CPU_GPU Matrix4f operator-(Matrix4f const& a, Matrix4f const& b);
    DMT_CPU_GPU Matrix4f operator*(Matrix4f const& a, Matrix4f const& b);
    DMT_CPU_GPU Matrix4f operator*(float v, Matrix4f const& m);
    DMT_CPU_GPU Matrix4f operator*(Matrix4f const& m, float v);
    DMT_CPU_GPU Matrix4f operator/(Matrix4f const& m, float v);

    DMT_CPU_GPU Matrix4f& operator+=(Matrix4f& a, Matrix4f const& b);
    DMT_CPU_GPU Matrix4f& operator-=(Matrix4f& a, Matrix4f const& b);
    DMT_CPU_GPU Matrix4f& operator*=(Matrix4f& a, Matrix4f const& b);

    DMT_CPU_GPU Matrix4f fromDiag(Tuple4f v);
    DMT_CPU_GPU Matrix4f fromQuat(Quaternion q);

    DMT_CPU_GPU bool     near(Matrix4f const& a, Matrix4f const& b);
    DMT_CPU_GPU float    determinant(Matrix4f const& m);
    DMT_CPU_GPU Matrix4f inverse(Matrix4f const& m);
    DMT_CPU_GPU Matrix4f transpose(Matrix4f const& m);
    DMT_CPU_GPU Vector4f mul(Matrix4f const& m, Vector4f v);
    DMT_CPU_GPU Vector3f mul(Matrix4f const& m, Vector3f const& v);
    DMT_CPU_GPU Normal3f mul(Matrix4f const& m, Normal3f const& v);
    DMT_CPU_GPU Point3f  mul(Matrix4f const& m, Point3f const& p);

    // Ray and RayDifferentials ---------------------------------------------------------------------------------------
    // TODO compress direction when you write compressed normal
    struct Ray
    {
        Ray() = default;
        DMT_CPU_GPU Ray(Point3f o, Vector3f d, float time = 0.f, uintptr_t medium = 0);

        DMT_CPU_GPU bool      hasNaN() const;
        DMT_CPU_GPU uintptr_t getMedium() const;

        uintptr_t medium = 0; // information about hasDifferentials embedded in the low bit
        Point3f   o{};
        Vector3f  d{{0, 0, 1}};
        float     time = 0;
    };

    struct RayDifferential : public Ray
    {
        RayDifferential() = default;
        DMT_CPU_GPU RayDifferential(Point3f o, Vector3f d, float time = 0.f, uintptr_t medium = 0);
        DMT_CPU_GPU explicit RayDifferential(Ray const& ray);
        DMT_CPU_GPU void setDifferentials(Point3f _rxOrigin, Vector3f _rxDirection, Point3f _ryOrigin, Vector3f _ryDirection);
        DMT_CPU_GPU void scaleDifferentials(float s);
        DMT_CPU_GPU bool hasDifferentials() const;

        Point3f  rxOrigin, ryOrigin;
        Vector3f rxDirection, ryDirection;
    };

    // Transform, AnimatedTransform, CameraTransform ------------------------------------------------------------------
    class Transform
    {
    public:
        Matrix4f m;    // Transformation matrix
        Matrix4f mInv; // Inverse transformation matrix

        // Default constructor
        DMT_CPU_GPU Transform();

        // Constructor with an initial matrix
        DMT_CPU_GPU explicit Transform(Matrix4f const& matrix);

        // Apply translation
        DMT_CPU_GPU void translate_(Vector3f const& translation);

        // Apply scaling
        DMT_CPU_GPU void scale_(Vector3f const& scaling);

        // Apply rotation (angle in degrees)
        DMT_CPU_GPU void rotate_(float angle, Vector3f const& axis);

        // Combine with another transform
        DMT_CPU_GPU Transform combine(Transform const& other) const;

        // Combine with another transform
        DMT_CPU_GPU void combine_(Transform const& other);

        DMT_CPU_GPU void lookAt_(Vector3f pos, Vector3f look, Vector3f up);

        DMT_CPU_GPU void concatTrasform_(std::array<float, 16> const& transform);

        // Reset to identity matrix
        DMT_CPU_GPU void reset_();

        // Swap m and mInv
        DMT_CPU_GPU void inverse_();

        DMT_CPU_GPU void decompose(Vector3f& outT, Quaternion& outR, Matrix4f& outS) const;

        // Equality comparison
        DMT_CPU_GPU bool operator==(Transform const& other) const;

        // Inequality comparison
        DMT_CPU_GPU bool operator!=(Transform const& other) const;

        DMT_CPU_GPU bool     hasScale(float tolerance = 1e-3f) const;
        DMT_CPU_GPU Vector3f applyInverse(Vector3f v) const;
        DMT_CPU_GPU Normal3f applyInverse(Point3f v) const;
        DMT_CPU_GPU Point3f  applyInverse(Normal3f v) const;
        DMT_CPU_GPU Vector3f operator()(Vector3f v) const;
        DMT_CPU_GPU Point3f  operator()(Point3f v) const;
        DMT_CPU_GPU Normal3f operator()(Normal3f v) const;
        DMT_CPU_GPU Bounds3f operator()(Bounds3f const& b) const;
    };

    class AnimatedTransform
    {
    public:
        AnimatedTransform() = default;
        DMT_CPU_GPU AnimatedTransform(Transform const& startTransform, float startTime, Transform const& endTransform, float endTime);

        DMT_CPU_GPU bool     isAnimated() const;
        DMT_CPU_GPU Vector3f applyInverse(Vector3f vpn, float time) const;
        DMT_CPU_GPU Point3f  applyInverse(Point3f vpn, float time) const;
        DMT_CPU_GPU Normal3f applyInverse(Normal3f vpn, float time) const;
        DMT_CPU_GPU Vector3f operator()(Vector3f vpn, float time) const;
        DMT_CPU_GPU Point3f  operator()(Point3f vpn, float time) const;
        DMT_CPU_GPU Normal3f operator()(Normal3f vpn, float time) const;
        // TODO: Interaction methods

        DMT_CPU_GPU bool      hasScale() const;
        DMT_CPU_GPU bool      hasRotation() const;
        DMT_CPU_GPU Transform interpolate(float time) const;
        // TODO: Ray and RayDifferential methods
        DMT_CPU_GPU Bounds3f motionBounds(Bounds3f const& b) const;
        DMT_CPU_GPU Bounds3f boundsPointMotion(Point3f p) const;

    private:
        DMT_CPU_GPU static void findZeros(
            float            c1,
            float            c2,
            float            c3,
            float            c4,
            float            c5,
            float            theta,
            Intervalf        tInterval,
            ArrayView<float> zeros,
            int32_t&         outNumZeros,
            int32_t          depth = 8);

    public:
        Transform startTransform;
        Transform endTransform;
        float     startTime = 0;
        float     endTime   = 1;

    private:
        struct DerivativeTerm
        {
            DMT_CPU_GPU       DerivativeTerm();
            DMT_CPU_GPU       DerivativeTerm(float c, float x, float y, float z);
            DMT_CPU_GPU float eval(Point3f p) const;

            float kc, kx, ky, kz;
        };

        enum EState : int32_t
        {
            eNone        = 0,
            eAnimated    = 1,
            eHasRotation = 2,
            eAll         = eAnimated | eHasRotation,
        };

        // rigid transformations
        DerivativeTerm m_c1[3], m_c2[3], m_c3[3], m_c4[3], m_c5[3];
        Matrix4f       m_s[2];
        Quaternion     m_r[2];
        Vector3f       m_t[2];
        EState         m_state;
    };

    struct CameraTransform
    {
        // requires initialized context
        DMT_CPU_GPU           CameraTransform(AnimatedTransform const& worldFromCamera, ERenderCoordSys renderCoordSys);
        DMT_CPU_GPU Transform renderFromWorld() const;

        AnimatedTransform renderFromCamera;
        Transform         worldFromRender;
    };

} // namespace dmt

/**
 * List of PBRT's SOA classes:
 * - pbrt core classes
 *   - Interval
 *   - Point2f
 *   - Point2i
 *   - Point3f
 *   - Vector3f
 *   - Normal3f
 *   - Point3fi
 *   - Ray
 *   - SubsurfaceInteraction
 *   - Frame
 *   - VisibleSurface
 *   - MediumInterface
 *   - TabulatedBSSRDF
 *   - LightSampleContext
 * - wavefront workitems
 *   - PixelSampleState
 *   - RayWorkItem
 *   - EscapedRayWorkItem
 *   - ShadowRayWorkItem
 *   - GetBSSRDFAndProbaRayWorkItem
 *   - SubsurfaceScatterWorkItem
 *   - MediumSampleWorkItem
 *   - MediumScatterWrokItem<ConcretePhaseFunction>
 *   - MaterialEvalWorkItem<ConcreteMaterial>
 */
namespace dmt {
}
