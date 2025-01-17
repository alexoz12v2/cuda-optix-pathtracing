#pragma once

#include "dmtmacros.h"

#include <platform/platform-utils.h>
#include "cudautils/cudautils-float.h"

#include <array>

#include <cassert>
#include <cstdint>
#include <cmath>

namespace dmt {
    // Enums ----------------------------------------------------------------------------------------------------------
    //I don't understand
    enum class ERenderCoordSys : uint8_t
    {
        eCameraWorld = 0,
        eCamera,
        eWorld,
        eCount
    };

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
    using Tuple4i = Tuple4<float>;

    struct Normalized { struct normalized { }; };

    struct Point2i : public Tuple2i { };
    struct Point2f : public Tuple2f { };
    struct Point3i : public Tuple3i { };
    struct Point3f : public Tuple3f { };
    struct Point4i : public Tuple4i { };
    struct Point4f : public Tuple4f { };

    struct Vector2i : public Tuple2i { };
    struct Vector2f : public Tuple2f { };
    struct Vector3i : public Tuple3i { };
    struct Vector3f : public Tuple3f { };
    struct Vector4i : public Tuple4i { };
    struct Vector4f : public Tuple4f { };

    struct Normal2f : public Tuple2f, public Normalized { };
    struct Normal3f : public Tuple3f, public Normalized { };

    struct Quaternion : public Tuple4f { };

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

    DMT_CPU_GPU Vector2i   operator/(Vector2i a, Vector2i b);
    DMT_CPU_GPU Vector2f   operator/(Vector2f a, Vector2f b);
    DMT_CPU_GPU Vector3i   operator/(Vector3i a, Vector3i b);
    DMT_CPU_GPU Vector3f   operator/(Vector3f a, Vector3f b);
    DMT_CPU_GPU Vector4i   operator/(Vector4i a, Vector4i b);
    DMT_CPU_GPU Vector4f   operator/(Vector4f a, Vector4f b);
    DMT_CPU_GPU Quaternion operator/(Quaternion a, Quaternion b);

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

    DMT_CPU_GPU Vector2i&   operator/=(Vector2i& a, Vector2i b);
    DMT_CPU_GPU Vector2f&   operator/=(Vector2f& a, Vector2f b);
    DMT_CPU_GPU Vector3i&   operator/=(Vector3i& a, Vector3i b);
    DMT_CPU_GPU Vector3f&   operator/=(Vector3f& a, Vector3f b);
    DMT_CPU_GPU Vector4i&   operator/=(Vector4i& a, Vector4i b);
    DMT_CPU_GPU Vector4f&   operator/=(Vector4f& a, Vector4f b);
    DMT_CPU_GPU Quaternion& operator/=(Quaternion& a, Quaternion b);


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
    DMT_CPU_GPU inline T& operator*(T& v, typename T::value_type k)
    {
        for (int32_t i = 0; i < T::numComponents(); ++i)
            v[i] *= k;
        return ret;
    }
    template <VectorScalable T>
    DMT_CPU_GPU inline T& operator/(T& v, typename T::value_type k)
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
    DMT_CPU_GPU Tuple2i::value_type dot(Tuple2i a, Tuple2i b);
    DMT_CPU_GPU Tuple3f::value_type dot(Tuple3f a, Tuple3f b);
    DMT_CPU_GPU Tuple3i::value_type dot(Tuple3i a, Tuple3i b);
    DMT_CPU_GPU Tuple4f::value_type dot(Tuple4f a, Tuple4f b);
    DMT_CPU_GPU Tuple4i::value_type dot(Tuple4i a, Tuple4i b);

    DMT_CPU_GPU Tuple2f::value_type absDot(Tuple2f a, Tuple2f b);
    DMT_CPU_GPU Tuple2i::value_type absDot(Tuple2i a, Tuple2i b);
    DMT_CPU_GPU Tuple3f::value_type absDot(Tuple3f a, Tuple3f b);
    DMT_CPU_GPU Tuple3i::value_type absDot(Tuple3i a, Tuple3i b);
    DMT_CPU_GPU Tuple4f::value_type absDot(Tuple4f a, Tuple4f b);
    DMT_CPU_GPU Tuple4i::value_type absDot(Tuple4i a, Tuple4i b);

    DMT_CPU_GPU Tuple3f cross(Tuple3f a, Tuple3f b);
    DMT_CPU_GPU Tuple3i cross(Tuple3i a, Tuple3i b);

    DMT_CPU_GPU Tuple2f normalize(Tuple2f v);
    DMT_CPU_GPU Tuple3f normalize(Tuple3f v);
    DMT_CPU_GPU Tuple4f normalize(Tuple4f v);

    DMT_CPU_GPU Tuple2f::value_type normL2(Tuple2f v);
    DMT_CPU_GPU Tuple2i::value_type normL2(Tuple2i v);
    DMT_CPU_GPU Tuple3f::value_type normL2(Tuple3f v);
    DMT_CPU_GPU Tuple3i::value_type normL2(Tuple3i v);
    DMT_CPU_GPU Tuple4f::value_type normL2(Tuple4f v);
    DMT_CPU_GPU Tuple4i::value_type normL2(Tuple4i v);

    DMT_CPU_GPU Tuple2f::value_type distanceL2(Tuple2f a, Tuple2f b);
    DMT_CPU_GPU Tuple2i::value_type distanceL2(Tuple2i a, Tuple2i b);
    DMT_CPU_GPU Tuple3f::value_type distanceL2(Tuple3f a, Tuple3f b);
    DMT_CPU_GPU Tuple3i::value_type distanceL2(Tuple3i a, Tuple3i b);
    DMT_CPU_GPU Tuple4f::value_type distanceL2(Tuple4f a, Tuple4f b);
    DMT_CPU_GPU Tuple4i::value_type distanceL2(Tuple4i a, Tuple4i b);

    DMT_CPU_GPU Tuple2f::value_type dotSelf(Tuple2f v);
    DMT_CPU_GPU Tuple2i::value_type dotSelf(Tuple2i v);
    DMT_CPU_GPU Tuple3f::value_type dotSelf(Tuple3f v);
    DMT_CPU_GPU Tuple3i::value_type dotSelf(Tuple3i v);
    DMT_CPU_GPU Tuple4f::value_type dotSelf(Tuple4f v);
    DMT_CPU_GPU Tuple4i::value_type dotSelf(Tuple4i v);

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
        return v * {dot(v, w)} * w;
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

    enum class EVecType32 : uint32_t
    {
        eVector = 0,
        ePoint  = 0x3f80'0000, // 1.f
    };
    namespace fl {
        DMT_CPU_GPU inline float bitsToFloat(EVecType32 e)
        {
            DMT_CPU_GPU inline float bitsToFloat(EVecType32 e)
            {
                return bitsToFloat(static_cast<std::underlying_type_t<EVecType32>>(e));
            }
        } // namespace fl

        // TODO SOA
        struct Intervalf
        {
        public:
            Intervalf() = default;
            DMT_CPU_GPU explicit Intervalf(float v);
            DMT_CPU_GPU                  Intervalf(float low, float high);
            DMT_CPU_GPU static Intervalf fromValueAndError(float v, float err);

        public:
            DMT_CPU_GPU float midpoint() const;
            DMT_CPU_GPU float width() const;

        public:
            float low  = 0.f;
            float high = 0.f;
        };
        DMT_CPU_GPU Intervalf operator+(Intervalf a, Intervalf b);
        DMT_CPU_GPU Intervalf operator-(Intervalf a, Intervalf b);
        DMT_CPU_GPU Intervalf operator/(Intervalf a, Intervalf b);
        DMT_CPU_GPU Intervalf operator*(Intervalf a, Intervalf b);

        // vectors

        using Vec3f  = glm::vec3;
        using Pt3f   = Vec3f;
        using Norm3f = Vec3f; // how to make sure that it always have unit length? Store compressed?
        using Vec4f  = glm::vec4;
        using Mat4f  = glm::mat4;
        using Quat   = glm::quat;
        using Pt2f   = glm::vec2;
    } // namespace fl


    class Transform
    {
    public:
        Mat4f m;    // Transformation matrix
        Mat4f mInv; // Inverse transformation matrix

        // Default constructor
        DMT_CPU_GPU Transform();

        // Constructor with an initial matrix
        DMT_CPU_GPU explicit Transform(Mat4f const& matrix);

        // Apply translation
        DMT_CPU_GPU void translate_(Vec3f const& translation);

        // Apply scaling
        DMT_CPU_GPU void scale_(Vec3f const& scaling);

        // Apply rotation (angle in degrees)
        DMT_CPU_GPU void rotate_(float angle, Vec3f const& axis);

        // Combine with another transform
        DMT_CPU_GPU Transform combine(Transform const& other) const;

        // Combine with another transform
        DMT_CPU_GPU void combine_(Transform const& other);

        DMT_CPU_GPU void lookAt_(Vec3f pos, Vec3f look, Vec3f up);

        DMT_CPU_GPU void concatTrasform_(std::array<float, 16> const& transform);

        // Reset to identity matrix
        DMT_CPU_GPU void reset_();

        // Swap m and mInv
        DMT_CPU_GPU void inverse_();
        // Apply the transform to a point
        DMT_CPU_GPU Vec3f applyToPoint(Vec3f const& point) const;

        // Apply the inverse transform to a point
        DMT_CPU_GPU Vec3f applyInverseToPoint(Vec3f const& point) const;

        DMT_CPU_GPU void decompose(Vec3f& outT, Quat& outR, Mat4f& outS) const;

        // Equality comparison
        DMT_CPU_GPU bool operator==(Transform const& other) const;

        // Inequality comparison
        DMT_CPU_GPU bool operator!=(Transform const& other) const;

        DMT_CPU_GPU bool            hasScale(float tolerance = 1e-3f) const;
        DMT_CPU_GPU Vec3f           applyInverse(Vec3f vpn, EVecType32 type, bool normalize = false) const;
        DMT_CPU_GPU Vec3f           operator()(Vec3f vpn, EVecType32 type, bool normalize = false) const;
        DMT_CPU_GPU bool            hasScale(float tolerance = 1e-3f) const;
        DMT_CPU_GPU Vec3f           applyInverse(Vec3f vpn, EVecType32 type, bool normalize = false) const;
        DMT_CPU_GPU Vec3f           operator()(Vec3f vpn, EVecType32 type, bool normalize = false) const;
        DMT_CPU_GPU struct Bounds3f operator()(struct Bounds3f const& b) const;
    };

    // retrun a Tranfrom with the Inverse mat of t passed
    DMT_CPU_GPU inline Transform Inverse(Transform const& t) { return Transform(t.mInv); }

    struct Bounds3f
    {
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

        DMT_CPU_GPU Vec3f&       operator[](int32_t i);
        DMT_CPU_GPU Vec3f const& operator[](int32_t i) const;
        DMT_CPU_GPU Pt3f         corner(EBoundsCorner corner) const;
        DMT_CPU_GPU Vec3f        diagonal() const;
        DMT_CPU_GPU float        surfaceAraa() const;
        DMT_CPU_GPU float        volume() const;
        DMT_CPU_GPU int32_t      maxDimention() const;
        DMT_CPU_GPU Pt3f         lerp(Pt3f t) const;
        DMT_CPU_GPU Vec3f        offset(Pt3f p) const;
        DMT_CPU_GPU void         boundingSphere(Pt3f& outCenter, float& outRadius) const;
        DMT_CPU_GPU bool         isEmpty() const;
        DMT_CPU_GPU bool         isDegenerate() const;
        DMT_CPU_GPU bool         operator==(Bounds3f const& that) const;
        DMT_CPU_GPU bool         intersectP(Pt3f                o,
                                            Vec3f               d,
                                            float               tMax = std::numeric_limits<float>::infinity(),
                                            float* DMT_RESTRICT hit0 = nullptr,
                                            float* DMT_RESTRICT hit1 = nullptr) const;
        DMT_CPU_GPU bool intersectP(Pt3f o, Vec3f d, float tMax, Vec3f invDir, std::array<int32_t, 3> dirIsNeg) const;

        Vec3f pMin;
        Vec3f pMax;
    };

    DMT_CPU_GPU bool     inside(Pt3f p, Bounds3f const& b);
    DMT_CPU_GPU bool     almostEqual(Mat4f const& a, Mat4f const& b);
    DMT_CPU_GPU Bounds3f bbUnion(Bounds3f const& a, Bounds3f const& b);
    DMT_CPU_GPU Bounds3f bbUnion(Bounds3f const& b, Pt3f p);

    class AnimatedTransform
    {
    public:
        AnimatedTransform() = default;
        DMT_CPU_GPU AnimatedTransform(Transform const& startTransform, float startTime, Transform const& endTransform, float endTime);

        DMT_CPU_GPU bool  isAnimated() const;
        DMT_CPU_GPU Vec3f applyInverse(Vec3f vpn, float time, EVecType32 type, bool normalize = false) const;
        DMT_CPU_GPU Vec3f operator()(Vec3f vpn, float time, EVecType32 type, bool normalize = false) const;
        // TODO: Interaction methods

        DMT_CPU_GPU bool      hasScale() const;
        DMT_CPU_GPU bool      hasRotation() const;
        DMT_CPU_GPU Transform interpolate(float time) const;
        // TODO: Ray and RayDifferential methods
        DMT_CPU_GPU Bounds3f motionBounds(Bounds3f const& b) const;
        DMT_CPU_GPU Bounds3f boundsPointMotion(Pt3f p) const;

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
            DMT_CPU_GPU float eval(Pt3f p) const;

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
        Mat4f          m_s[2];
        Quat           m_r[2];
        Vec3f          m_t[2];
        EState         m_state;
    };

    struct CameraTransform
    {
        // requires initialized context
        DMT_CPU_GPU       CameraTransform(AnimatedTransform const& worldFromCamera, ERenderCoordSys renderCoordSys);
        AnimatedTransform renderFromCamera;
        Transform         worldFromRender;

        DMT_CPU_GPU Transform RenderFromWorld() const { return worldFromRender; }
    };

} // namespace dmt

namespace dmt::soa {
    using namespace dmt;
}