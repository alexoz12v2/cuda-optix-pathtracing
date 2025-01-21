#include "cudautils.h"

#include "cudautils-vecconv.cuh"

#if defined(DMT_ARCH_X86_64)
#include <immintrin.h>
#endif

// TODO generated optimized assembly for x64 uses vectorized instructions for floating point types, like "[v]addps", but integer types
// don't seem to use SSE2

namespace dmt {
    // Vector Types: Static Assertions --------------------------------------------------------------------------------
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

    // Vector Types: Basic Operations ---------------------------------------------------------------------------------
    __host__ __device__ Point2i::operator Vector2i() { return *std::bit_cast<Vector2i const*>(this); }
    __host__ __device__ Point2f::operator Vector2f() { return *std::bit_cast<Vector2f const*>(this); }
    __host__ __device__ Point3i::operator Vector3i() { return *std::bit_cast<Vector3i const*>(this); }
    __host__ __device__ Point3f::operator Vector3f() { return *std::bit_cast<Vector3f const*>(this); }
    __host__ __device__ Point4i::operator Vector4i() { return *std::bit_cast<Vector4i const*>(this); }
    __host__ __device__ Point4f::operator Vector4f() { return *std::bit_cast<Vector4f const*>(this); }

    __host__ __device__ Point2i  operator+(Point2i a, Vector2i b) { return {fromGLM(toGLM(a) + toGLM(b))}; }
    __host__ __device__ Point2f  operator+(Point2f a, Vector2f b) { return {fromGLM(toGLM(a) + toGLM(b))}; }
    __host__ __device__ Point3i  operator+(Point3i a, Vector3i b) { return {fromGLM(toGLM(a) + toGLM(b))}; }
    __host__ __device__ Point3f  operator+(Point3f a, Vector3f b) { return {fromGLM(toGLM(a) + toGLM(b))}; }
    __host__ __device__ Point4i  operator+(Point4i a, Vector4i b) { return {fromGLM(toGLM(a) + toGLM(b))}; }
    __host__ __device__ Point4f  operator+(Point4f a, Vector4f b) { return {fromGLM(toGLM(a) + toGLM(b))}; }
    __host__ __device__ Vector2i operator+(Vector2i a, Vector2i b) { return {fromGLM(toGLM(a) + toGLM(b))}; }
    __host__ __device__ Vector2f operator+(Vector2f a, Vector2f b) { return {fromGLM(toGLM(a) + toGLM(b))}; }
    __host__ __device__ Vector3i operator+(Vector3i a, Vector3i b) { return {fromGLM(toGLM(a) + toGLM(b))}; }
    __host__ __device__ Vector3f operator+(Vector3f a, Vector3f b) { return {fromGLM(toGLM(a) + toGLM(b))}; }
    __host__ __device__ Vector4i operator+(Vector4i a, Vector4i b) { return {fromGLM(toGLM(a) + toGLM(b))}; }
    __host__ __device__ Vector4f operator+(Vector4f a, Vector4f b) { return {fromGLM(toGLM(a) + toGLM(b))}; }
    __host__ __device__ Normal2f operator+(Normal2f a, Normal2f b)
    {
        return {fromGLM(glm::normalize(toGLM(a) + toGLM(b)))};
    }
    __host__ __device__ Normal3f operator+(Normal3f a, Normal3f b)
    {
        return {fromGLM(glm::normalize(toGLM(a) + toGLM(b)))};
    }

    __host__ __device__ Quaternion operator+(Quaternion a, Quaternion b) { return fromGLMquat(toGLM(a) + toGLM(b)); }

    __host__ __device__ Vector2i operator-(Point2i a, Point2i b) { return {fromGLM(toGLM(a) - toGLM(b))}; }
    __host__ __device__ Vector2f operator-(Point2f a, Point2f b) { return {fromGLM(toGLM(a) - toGLM(b))}; }
    __host__ __device__ Vector3i operator-(Point3i a, Point3i b) { return {fromGLM(toGLM(a) - toGLM(b))}; }
    __host__ __device__ Vector3f operator-(Point3f a, Point3f b) { return {fromGLM(toGLM(a) - toGLM(b))}; }
    __host__ __device__ Vector4i operator-(Point4i a, Point4i b) { return {fromGLM(toGLM(a) - toGLM(b))}; }
    __host__ __device__ Vector4f operator-(Point4f a, Point4f b) { return {fromGLM(toGLM(a) - toGLM(b))}; }
    __host__ __device__ Vector2i operator-(Vector2i a, Vector2i b) { return {fromGLM(toGLM(a) - toGLM(b))}; }
    __host__ __device__ Vector2f operator-(Vector2f a, Vector2f b) { return {fromGLM(toGLM(a) - toGLM(b))}; }
    __host__ __device__ Vector3i operator-(Vector3i a, Vector3i b) { return {fromGLM(toGLM(a) - toGLM(b))}; }
    __host__ __device__ Vector3f operator-(Vector3f a, Vector3f b) { return {fromGLM(toGLM(a) - toGLM(b))}; }
    __host__ __device__ Vector4i operator-(Vector4i a, Vector4i b) { return {fromGLM(toGLM(a) - toGLM(b))}; }
    __host__ __device__ Vector4f operator-(Vector4f a, Vector4f b) { return {fromGLM(toGLM(a) - toGLM(b))}; }
    __host__ __device__ Normal2f operator-(Normal2f a, Normal2f b)
    {
        return {fromGLM(glm::normalize(toGLM(a) - toGLM(b)))};
    }
    __host__ __device__ Normal3f operator-(Normal3f a, Normal3f b)
    {
        return {fromGLM(glm::normalize(toGLM(a) - toGLM(b)))};
    }

    __host__ __device__ Quaternion operator-(Quaternion a, Quaternion b) { return fromGLMquat(toGLM(a) - toGLM(b)); }

    __host__ __device__ Vector2i operator-(Point2i v) { return {fromGLM(-toGLM(v))}; }
    __host__ __device__ Vector2f operator-(Point2f v) { return {fromGLM(-toGLM(v))}; }
    __host__ __device__ Vector3i operator-(Point3i v) { return {fromGLM(-toGLM(v))}; }
    __host__ __device__ Vector3f operator-(Point3f v) { return {fromGLM(-toGLM(v))}; }
    __host__ __device__ Vector4i operator-(Point4i v) { return {fromGLM(-toGLM(v))}; }
    __host__ __device__ Vector4f operator-(Point4f v) { return {fromGLM(-toGLM(v))}; }
    __host__ __device__ Vector2i operator-(Vector2i v) { return {fromGLM(-toGLM(v))}; }
    __host__ __device__ Vector2f operator-(Vector2f v) { return {fromGLM(-toGLM(v))}; }
    __host__ __device__ Vector3i operator-(Vector3i v) { return {fromGLM(-toGLM(v))}; }
    __host__ __device__ Vector3f operator-(Vector3f v) { return {fromGLM(-toGLM(v))}; }
    __host__ __device__ Vector4i operator-(Vector4i v) { return {fromGLM(-toGLM(v))}; }
    __host__ __device__ Vector4f operator-(Vector4f v) { return {fromGLM(-toGLM(v))}; }
    __host__ __device__ Normal2f operator-(Normal2f v) { return {fromGLM(-toGLM(v))}; }
    __host__ __device__ Normal3f operator-(Normal3f v) { return {fromGLM(-toGLM(v))}; }

    __host__ __device__ Quaternion operator-(Quaternion q) { return fromGLMquat(-toGLM(q)); }

    __host__ __device__ Vector2i operator*(Vector2i a, Vector2i b) { return {fromGLM(toGLM(a) * toGLM(b))}; }
    __host__ __device__ Vector2f operator*(Vector2f a, Vector2f b) { return {fromGLM(toGLM(a) * toGLM(b))}; }
    __host__ __device__ Vector3i operator*(Vector3i a, Vector3i b) { return {fromGLM(toGLM(a) * toGLM(b))}; }
    __host__ __device__ Vector3f operator*(Vector3f a, Vector3f b) { return {fromGLM(toGLM(a) * toGLM(b))}; }
    __host__ __device__ Vector4i operator*(Vector4i a, Vector4i b) { return {fromGLM(toGLM(a) * toGLM(b))}; }
    __host__ __device__ Vector4f operator*(Vector4f a, Vector4f b) { return {fromGLM(toGLM(a) * toGLM(b))}; }

    __host__ __device__ Quaternion operator*(Quaternion a, Quaternion b) { return fromGLMquat(toGLM(a) * toGLM(b)); }

    __host__ __device__ Vector2i operator/(Vector2i a, Vector2i b) { return {fromGLM(toGLM(a) / toGLM(b))}; }
    __host__ __device__ Vector2f operator/(Vector2f a, Vector2f b) { return {fromGLM(toGLM(a) / toGLM(b))}; }
    __host__ __device__ Vector3i operator/(Vector3i a, Vector3i b) { return {fromGLM(toGLM(a) / toGLM(b))}; }
    __host__ __device__ Vector3f operator/(Vector3f a, Vector3f b) { return {fromGLM(toGLM(a) / toGLM(b))}; }
    __host__ __device__ Vector4i operator/(Vector4i a, Vector4i b) { return {fromGLM(toGLM(a) / toGLM(b))}; }
    __host__ __device__ Vector4f operator/(Vector4f a, Vector4f b) { return {fromGLM(toGLM(a) / toGLM(b))}; }

    __host__ __device__ Point2i& operator+=(Point2i& a, Vector2i b)
    {
        toGLM(a) += toGLM(b);
        return a;
    }
    __host__ __device__ Point2f& operator+=(Point2f& a, Vector2f b)
    {
        toGLM(a) += toGLM(b);
        return a;
    }
    __host__ __device__ Point3i& operator+=(Point3i& a, Vector3i b)
    {
        toGLM(a) += toGLM(b);
        return a;
    }
    __host__ __device__ Point3f& operator+=(Point3f& a, Vector3f b)
    {
        toGLM(a) += toGLM(b);
        return a;
    }
    __host__ __device__ Point4i& operator+=(Point4i& a, Vector4i b)
    {
        toGLM(a) += toGLM(b);
        return a;
    }
    __host__ __device__ Point4f& operator+=(Point4f& a, Vector4f b)
    {
        toGLM(a) += toGLM(b);
        return a;
    }
    __host__ __device__ Vector2i& operator+=(Vector2i& a, Vector2i b)
    {
        toGLM(a) += toGLM(b);
        return a;
    }
    __host__ __device__ Vector2f& operator+=(Vector2f& a, Vector2f b)
    {
        toGLM(a) += toGLM(b);
        return a;
    }
    __host__ __device__ Vector3i& operator+=(Vector3i& a, Vector3i b)
    {
        toGLM(a) += toGLM(b);
        return a;
    }
    __host__ __device__ Vector3f& operator+=(Vector3f& a, Vector3f b)
    {
        toGLM(a) += toGLM(b);
        return a;
    }
    __host__ __device__ Vector4i& operator+=(Vector4i& a, Vector4i b)
    {
        toGLM(a) += toGLM(b);
        return a;
    }
    __host__ __device__ Vector4f& operator+=(Vector4f& a, Vector4f b)
    {
        toGLM(a) += toGLM(b);
        return a;
    }
    __host__ __device__ Normal2f& operator+=(Normal2f& a, Normal2f b)
    {
        toGLM(a) += toGLM(b);
        return a;
    }
    __host__ __device__ Normal3f& operator+=(Normal3f& a, Normal3f b)
    {
        toGLM(a) += toGLM(b);
        return a;
    }

    __host__ __device__ Quaternion& operator+=(Quaternion& a, Quaternion b)
    {
        toGLM(a) += toGLM(b);
        return a;
    }

    __host__ __device__ Vector2i& operator-=(Vector2i& a, Vector2i b)
    {
        toGLM(a) -= toGLM(b);
        return a;
    }
    __host__ __device__ Vector2f& operator-=(Vector2f& a, Vector2f b)
    {
        toGLM(a) -= toGLM(b);
        return a;
    }
    __host__ __device__ Vector3i& operator-=(Vector3i& a, Vector3i b)
    {
        toGLM(a) -= toGLM(b);
        return a;
    }
    __host__ __device__ Vector3f& operator-=(Vector3f& a, Vector3f b)
    {
        toGLM(a) -= toGLM(b);
        return a;
    }
    __host__ __device__ Vector4i& operator-=(Vector4i& a, Vector4i b)
    {
        toGLM(a) -= toGLM(b);
        return a;
    }
    __host__ __device__ Vector4f& operator-=(Vector4f& a, Vector4f b)
    {
        toGLM(a) -= toGLM(b);
        return a;
    }
    __host__ __device__ Normal2f& operator-=(Normal2f& a, Normal2f b)
    {
        toGLM(a) -= toGLM(b);
        return a;
    }
    __host__ __device__ Normal3f& operator-=(Normal3f& a, Normal3f b)
    {
        toGLM(a) -= toGLM(b);
        return a;
    }

    __host__ __device__ Quaternion& operator-=(Quaternion& a, Quaternion b)
    {
        toGLM(a) -= toGLM(b);
        return a;
    }

    __host__ __device__ Vector2i& operator*=(Vector2i& a, Vector2i b)
    {
        toGLM(a) *= toGLM(b);
        return a;
    }
    __host__ __device__ Vector2f& operator*=(Vector2f& a, Vector2f b)
    {
        toGLM(a) *= toGLM(b);
        return a;
    }
    __host__ __device__ Vector3i& operator*=(Vector3i& a, Vector3i b)
    {
        toGLM(a) *= toGLM(b);
        return a;
    }
    __host__ __device__ Vector3f& operator*=(Vector3f& a, Vector3f b)
    {
        toGLM(a) *= toGLM(b);
        return a;
    }
    __host__ __device__ Vector4i& operator*=(Vector4i& a, Vector4i b)
    {
        toGLM(a) *= toGLM(b);
        return a;
    }
    __host__ __device__ Vector4f& operator*=(Vector4f& a, Vector4f b)
    {
        toGLM(a) *= toGLM(b);
        return a;
    }

    __host__ __device__ Quaternion& operator*=(Quaternion& a, Quaternion b)
    {
        toGLM(a) *= toGLM(b);
        return a;
    }

    __host__ __device__ Vector2i& operator/=(Vector2i& a, Vector2i b)
    {
        toGLM(a) /= toGLM(b);
        return a;
    }
    __host__ __device__ Vector2f& operator/=(Vector2f& a, Vector2f b)
    {
        toGLM(a) /= toGLM(b);
        return a;
    }
    __host__ __device__ Vector3i& operator/=(Vector3i& a, Vector3i b)
    {
        toGLM(a) /= toGLM(b);
        return a;
    }
    __host__ __device__ Vector3f& operator/=(Vector3f& a, Vector3f b)
    {
        toGLM(a) /= toGLM(b);
        return a;
    }
    __host__ __device__ Vector4i& operator/=(Vector4i& a, Vector4i b)
    {
        toGLM(a) /= toGLM(b);
        return a;
    }
    __host__ __device__ Vector4f& operator/=(Vector4f& a, Vector4f b)
    {
        toGLM(a) /= toGLM(b);
        return a;
    }

    __host__ __device__ Normal2f normalFrom(Vector2f v) { return {fromGLM(glm::normalize(toGLM(v)))}; }
    __host__ __device__ Normal3f normalFrom(Vector3f v) { return {fromGLM(glm::normalize(toGLM(v)))}; }

    // Vector Types: Generic Tuple Operations -------------------------------------------------------------------------
    __host__ __device__ Tuple2f abs(Tuple2f v) { return {fromGLM(glm::abs(toGLM(v)))}; }
    __host__ __device__ Tuple2i abs(Tuple2i v) { return {fromGLM(glm::abs(toGLM(v)))}; }
    __host__ __device__ Tuple3f abs(Tuple3f v) { return {fromGLM(glm::abs(toGLM(v)))}; }
    __host__ __device__ Tuple3i abs(Tuple3i v) { return {fromGLM(glm::abs(toGLM(v)))}; }
    __host__ __device__ Tuple4f abs(Tuple4f v) { return {fromGLM(glm::abs(toGLM(v)))}; }
    __host__ __device__ Tuple4i abs(Tuple4i v) { return {fromGLM(glm::abs(toGLM(v)))}; }

    __host__ __device__ Tuple2f ceil(Tuple2f v) { return {fromGLM(glm::ceil(toGLM(v)))}; }
    __host__ __device__ Tuple3f ceil(Tuple3f v) { return {fromGLM(glm::ceil(toGLM(v)))}; }
    __host__ __device__ Tuple4f ceil(Tuple4f v) { return {fromGLM(glm::ceil(toGLM(v)))}; }

    __host__ __device__ Tuple2f floor(Tuple2f v) { return {fromGLM(glm::floor(toGLM(v)))}; }
    __host__ __device__ Tuple3f floor(Tuple3f v) { return {fromGLM(glm::floor(toGLM(v)))}; }
    __host__ __device__ Tuple4f floor(Tuple4f v) { return {fromGLM(glm::floor(toGLM(v)))}; }

    __host__ __device__ Tuple2f lerp(float t, Tuple2f zero, Tuple2f one)
    {
        return {fromGLM(glm::mix(toGLM(one), toGLM(zero), t))};
    }
    __host__ __device__ Tuple3f lerp(float t, Tuple3f zero, Tuple3f one)
    {
        return {fromGLM(glm::mix(toGLM(one), toGLM(zero), t))};
    }
    __host__ __device__ Tuple4f lerp(float t, Tuple4f zero, Tuple4f one)
    {
        return {fromGLM(glm::mix(toGLM(one), toGLM(zero), t))};
    }

    __host__ __device__ Tuple2f fma(Tuple2f mult0, Tuple2f mult1, Tuple2f add)
    {
        return {fromGLM(glm::fma(toGLM(mult0), toGLM(mult1), toGLM(add)))};
    }
    __host__ __device__ Tuple2i fma(Tuple2i mult0, Tuple2i mult1, Tuple2i add)
    {
        return {fromGLM(glm::fma(toGLM(mult0), toGLM(mult1), toGLM(add)))};
    }
    __host__ __device__ Tuple3f fma(Tuple3f mult0, Tuple3f mult1, Tuple3f add)
    {
        return {fromGLM(glm::fma(toGLM(mult0), toGLM(mult1), toGLM(add)))};
    }
    __host__ __device__ Tuple3i fma(Tuple3i mult0, Tuple3i mult1, Tuple3i add)
    {
        return {fromGLM(glm::fma(toGLM(mult0), toGLM(mult1), toGLM(add)))};
    }
    __host__ __device__ Tuple4f fma(Tuple4f mult0, Tuple4f mult1, Tuple4f add)
    {
        return {fromGLM(glm::fma(toGLM(mult0), toGLM(mult1), toGLM(add)))};
    }
    __host__ __device__ Tuple4i fma(Tuple4i mult0, Tuple4i mult1, Tuple4i add)
    {
        return {fromGLM(glm::fma(toGLM(mult0), toGLM(mult1), toGLM(add)))};
    }

    __host__ __device__ Tuple2f min(Tuple2f a, Tuple2f b) { return {fromGLM(glm::min(toGLM(a), toGLM(b)))}; }
    __host__ __device__ Tuple2i min(Tuple2i a, Tuple2i b) { return {fromGLM(glm::min(toGLM(a), toGLM(b)))}; }
    __host__ __device__ Tuple3f min(Tuple3f a, Tuple3f b) { return {fromGLM(glm::min(toGLM(a), toGLM(b)))}; }
    __host__ __device__ Tuple3i min(Tuple3i a, Tuple3i b) { return {fromGLM(glm::min(toGLM(a), toGLM(b)))}; }
    __host__ __device__ Tuple4f min(Tuple4f a, Tuple4f b) { return {fromGLM(glm::min(toGLM(a), toGLM(b)))}; }
    __host__ __device__ Tuple4i min(Tuple4i a, Tuple4i b) { return {fromGLM(glm::min(toGLM(a), toGLM(b)))}; }

    __host__ __device__ Tuple2f max(Tuple2f a, Tuple2f b) { return {fromGLM(glm::max(toGLM(a), toGLM(b)))}; }
    __host__ __device__ Tuple2i max(Tuple2i a, Tuple2i b) { return {fromGLM(glm::max(toGLM(a), toGLM(b)))}; }
    __host__ __device__ Tuple3f max(Tuple3f a, Tuple3f b) { return {fromGLM(glm::max(toGLM(a), toGLM(b)))}; }
    __host__ __device__ Tuple3i max(Tuple3i a, Tuple3i b) { return {fromGLM(glm::max(toGLM(a), toGLM(b)))}; }
    __host__ __device__ Tuple4f max(Tuple4f a, Tuple4f b) { return {fromGLM(glm::max(toGLM(a), toGLM(b)))}; }
    __host__ __device__ Tuple4i max(Tuple4i a, Tuple4i b) { return {fromGLM(glm::max(toGLM(a), toGLM(b)))}; }

    __host__ __device__ bool near(Tuple2f a, Tuple2f b, float tolerance)
    {
        auto bvec = glm::epsilonEqual(toGLM(a), toGLM(b), tolerance);
        return bvec.x && bvec.y;
    }
    __host__ __device__ bool near(Tuple2i a, Tuple2i b)
    {
        auto bvec = toGLM(a) = toGLM(b);
        return bvec.x && bvec.y;
    }
    __host__ __device__ bool near(Tuple3f a, Tuple3f b, float tolerance)
    {
        auto bvec = glm::epsilonEqual(toGLM(a), toGLM(b), tolerance);
        return bvec.x && bvec.y && bvec.z;
    }
    __host__ __device__ bool near(Tuple3i a, Tuple3i b)
    {
        auto bvec = toGLM(a) = toGLM(b);
        return bvec.x && bvec.y && bvec.z;
    }
    __host__ __device__ bool near(Tuple4f a, Tuple4f b, float tolerance)
    {
        auto bvec = glm::epsilonEqual(toGLM(a), toGLM(b), tolerance);
        return bvec.x && bvec.y && bvec.z && bvec.w;
    }
    __host__ __device__ bool near(Tuple4i a, Tuple4i b)
    {
        auto bvec = toGLM(a) = toGLM(b);
        return bvec.x && bvec.y && bvec.z && bvec.w;
    }

    __host__ __device__ Tuple2f::value_type dot(Tuple2f a, Tuple2f b) { return glm::dot(toGLM(a), toGLM(b)); }
    __host__ __device__ Tuple3f::value_type dot(Tuple3f a, Tuple3f b) { return glm::dot(toGLM(a), toGLM(b)); }
    __host__ __device__ Tuple4f::value_type dot(Tuple4f a, Tuple4f b) { return glm::dot(toGLM(a), toGLM(b)); }

    __host__ __device__ Tuple2f::value_type absDot(Tuple2f a, Tuple2f b)
    {
        return glm::abs(glm::dot(toGLM(a), toGLM(b)));
    }
    __host__ __device__ Tuple3f::value_type absDot(Tuple3f a, Tuple3f b)
    {
        return glm::abs(glm::dot(toGLM(a), toGLM(b)));
    }
    __host__ __device__ Tuple4f::value_type absDot(Tuple4f a, Tuple4f b)
    {
        return glm::abs(glm::dot(toGLM(a), toGLM(b)));
    }

    __host__ __device__ Tuple3f cross(Tuple3f a, Tuple3f b) { return {fromGLM(glm::cross(toGLM(a), toGLM(b)))}; }

    __host__ __device__ Tuple2f normalize(Tuple2f v) { return {fromGLM(glm::normalize(toGLM(v)))}; }
    __host__ __device__ Tuple3f normalize(Tuple3f v) { return {fromGLM(glm::normalize(toGLM(v)))}; }
    __host__ __device__ Tuple4f normalize(Tuple4f v) { return {fromGLM(glm::normalize(toGLM(v)))}; }

    __host__ __device__ Tuple2f::value_type normL2(Tuple2f v) { return glm::length(toGLM(v)); }
    __host__ __device__ Tuple3f::value_type normL2(Tuple3f v) { return glm::length(toGLM(v)); }
    __host__ __device__ Tuple4f::value_type normL2(Tuple4f v) { return glm::length(toGLM(v)); }

    __host__ __device__ Tuple2f::value_type distanceL2(Tuple2f a, Tuple2f b)
    {
        return glm::distance(toGLM(a), toGLM(b));
    }
    __host__ __device__ Tuple3f::value_type distanceL2(Tuple3f a, Tuple3f b)
    {
        return glm::distance(toGLM(a), toGLM(b));
    }
    __host__ __device__ Tuple4f::value_type distanceL2(Tuple4f a, Tuple4f b)
    {
        return glm::distance(toGLM(a), toGLM(b));
    }

    __host__ __device__ Tuple2f::value_type dotSelf(Tuple2f v) { return glm::length2(toGLM(v)); }
    __host__ __device__ Tuple3f::value_type dotSelf(Tuple3f v) { return glm::length2(toGLM(v)); }
    __host__ __device__ Tuple4f::value_type dotSelf(Tuple4f v) { return glm::length2(toGLM(v)); }

    // Vector Types: Geometric Functions ------------------------------------------------------------------------------
    __host__ __device__ float angleBetween(Normal3f a, Normal3f b) { return glm::dot(toGLM(a), toGLM(b)); }
    __host__ __device__ float angleBetween(Quaternion a, Quaternion b)
    {
        if (dot(a, b) < 0.f)
            return fl::twoPi() * fl::asinClamp(normL2(a + b) / 2);
        else
            return 2 * fl::asinClamp(normL2(b - a) / 2);
    }

    __host__ __device__ Frame coordinateSystem(Normal3f xAxis)
    {
        Frame frame;
        frame.xAxis    = xAxis;
        float     sign = fl::copysign(1.f, xAxis.z);
        float     a    = -1.f / (sign + xAxis.z);
        float     b    = xAxis.z * xAxis.y * a;
        glm::vec3 y    = {(1 + sign + xAxis.x * xAxis.x * a), (sign * b), (-sign * xAxis.x)};
        glm::vec3 z    = {(b), (sign + xAxis.y * xAxis.y * a), (-xAxis.y)};
        frame.yAxis    = {fromGLM(glm::normalize(y))};
        frame.zAxis    = {fromGLM(glm::normalize(z))};
        return frame;
    }

    __host__ __device__ Quaternion slerp(float t, Quaternion zero, Quaternion one)
    {
        return fromGLMquat(glm::slerp(toGLM(one), toGLM(zero), t));
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
            .x = (clampedSinTheta * glm::cos(phi)),
            .y = (clampedSinTheta * glm::sin(phi)),
            .z = (glm::clamp(cosTheta, -1.f, 1.f)), // -1 ?
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

    __host__ __device__ Frame    Frame::fromXZ(Normal3f x, Normal3f z) { return {x, cross(z, x), z}; }
    __host__ __device__ Frame    Frame::fromXY(Normal3f x, Normal3f y) { return {x, y, cross(x, y)}; }
    __host__ __device__ Frame    Frame::fromZ(Normal3f z) { return coordinateSystem(z); }
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
        Point3f const ret{{
            operator[](corner& eRight).x,
            operator[]((corner & eForward) >> 1).y,
            operator[]((corner & eTop) >> 2).z,
        }};
        return ret;
    }

    __host__ __device__ Vector3f Bounds3f::diagonal() const { return pMax - pMin; }

    __host__ __device__ float Bounds3f::surfaceAraa() const
    {
        Vector3f const d = diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
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
        outRadius = inside(outCenter, *this) ? glm::distance(toGLM(outCenter), toGLM(pMax)) : 0.f;
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
        QR        ret;
        glm::mat4 QT{1.f};
        glm::mat4 R = toGLMmat(m);
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
                    QT             = toGLMmat(G) * QT;
                    R              = toGLMmat(G) * R;
                }

                if (converge)
                    break;
            }
        }
        ret.qOrthogonal = fromGLMmat(glm::transpose(QT));
        ret.rUpper      = fromGLMmat(R);
        return ret;
    }

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

    __host__ __device__ Matrix4f operator+(Matrix4f const& a, Matrix4f const& b)
    {
        return fromGLMmat(toGLMmat(a) + toGLMmat(b));
    }
    __host__ __device__ Matrix4f operator-(Matrix4f const& a, Matrix4f const& b)
    {
        return fromGLMmat(toGLMmat(a) - toGLMmat(b));
    }
    __host__ __device__ Matrix4f operator*(Matrix4f const& a, Matrix4f const& b)
    {
        return fromGLMmat(toGLMmat(a) * toGLMmat(b));
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
        toGLMmat(a) += toGLMmat(b);
        return a;
    }
    __host__ __device__ Matrix4f& operator-=(Matrix4f& a, Matrix4f const& b)
    {
        toGLMmat(a) -= toGLMmat(b);
        return a;
    }
    __host__ __device__ Matrix4f& operator*=(Matrix4f& a, Matrix4f const& b)
    {
        toGLMmat(a) *= toGLMmat(b);
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

    __host__ __device__ Matrix4f fromQuat(Quaternion q) { return fromGLMmat(glm::mat4_cast(toGLM(q))); }

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

    __host__ __device__ float determinant(Matrix4f const& m) { return glm::determinant(toGLMmat(m)); }

    __host__ __device__ Matrix4f inverse(Matrix4f const& m) { return fromGLMmat(glm::inverse(toGLMmat(m))); }

    __host__ __device__ Matrix4f transpose(Matrix4f const& m) { return fromGLMmat(glm::transpose(toGLMmat(m))); }

    __host__ __device__ Vector4f mul(Matrix4f const& m, Vector4f v) { return {fromGLM(toGLMmat(m) * toGLM(v))}; }

    __host__ __device__ Vector3f mul(Matrix4f const& m, Vector3f const& v)
    {
        glm::vec4 w{v.x, v.y, v.z, 0.f};
        w = toGLMmat(m) * w;
        assert(fl::nearZero(w.w));
        return {{.x = w.x, .y = w.y, .z = w.y}};
    }

    __host__ __device__ Normal3f mul(Matrix4f const& m, Normal3f const& v)
    {
        glm::vec4 w{v.x, v.y, v.z, 0.f};
        w = toGLMmat(m) * w;
        assert(fl::nearZero(w.w));
        w = glm::normalize(w);
        return {{.x = w.x, .y = w.y, .z = w.y}};
    }

    __host__ __device__ Normal3f mulTranspose(Matrix4f const& m, Normal3f const& v)
    {
        Normal3f         ret;
        glm::vec4        w{v.x, v.y, v.z, 0.f};
        glm::mat4 const& glmMat = toGLMmat(m);
        ret.x                   = glmMat[0][0] * w.x + glmMat[1][0] * w.y + glmMat[2][0] * w.z + glmMat[3][0] * w.w;
        ret.y                   = glmMat[0][1] * w.x + glmMat[1][1] * w.y + glmMat[2][1] * w.z + glmMat[3][1] * w.w;
        ret.z                   = glmMat[0][2] * w.x + glmMat[1][2] * w.y + glmMat[2][2] * w.z + glmMat[3][2] * w.w;
        return ret;
    }

    __host__ __device__ Point3f mul(Matrix4f const& m, Point3f const& p)
    {
        glm::vec4 w{p.x, p.y, p.z, 1.f};
        w = toGLMmat(m) * w;
        w /= w.w;
        return {{.x = w.x, .y = w.y, .z = w.y}};
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
        ret.xLow             = low[0];
        ret.yLow             = low[1];
        ret.zLow             = low[2];
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
        ret.xLow             = low[0];
        ret.yLow             = low[1];
        ret.zLow             = low[2];
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
        ret.xLow             = low[0];
        ret.yLow             = low[1];
        ret.zLow             = low[2];
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

    __host__ __device__ bool RayDifferential::hasDifferentials() const { return medium & 1; }


    // math utilities: vector -----------------------------------------------------------------------------------------
} // namespace dmt

namespace dmt {
}
