#include "cudautils/cudautils-vecmath.cuh"

// GLM and Eigen
#include "cudautils-include-glm.cuh"

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
    __host__ __device__ Frame Frame::fromXZ(Vector3f x, Vector3f z)
    {
        // Normalize x first
        Vector3f X = normalize(x);

        // Remove the component of z along x
        Vector3f Z = z - dot(z, X) * X;

        // Handle near-parallel case
        if (dotSelf(Z) < 1e-8f)
        {
            // Pick an arbitrary vector not parallel to X
            Vector3f tmp = (fabs(X.x) < 0.9f) ? Vector3f(1, 0, 0) : Vector3f(0, 1, 0);
            Z            = cross(X, tmp);
        }

        Z = normalize(Z);

        // Y completes the right-handed basis
        Vector3f Y = cross(Z, X);

        return {X, Y, Z};
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

    __host__ __device__ Bounds3f makeBounds(Point3f p0, Point3f p1) { return {min(p0, p1), max(p0, p1)}; }

    __host__ __device__ Bounds3f bbEmpty()
    {
        return {{{fl::infinity(), fl::infinity(), fl::infinity()}}, {{-fl::infinity(), -fl::infinity(), -fl::infinity()}}};
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
            operator[](static_cast<int32_t>(corner) & static_cast<int32_t>(EBoundsCorner::eRight)).x,
            operator[]((static_cast<int32_t>(corner) & static_cast<int32_t>(EBoundsCorner::eForward)) >> 1).y,
            operator[]((static_cast<int32_t>(corner) & static_cast<int32_t>(EBoundsCorner::eTop)) >> 2).z,
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
    __host__ __device__ Bounds2f makeBounds(Point2f p0, Point2f p1) { return {min(p0, p1), max(p0, p1)}; }

    __host__ __device__ Bounds2f bbEmpty2()
    {
        return {{{fl::infinity(), fl::infinity()}}, {{-fl::infinity(), -fl::infinity()}}};
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
        Point2f const ret{{
            operator[](static_cast<int32_t>(corner) & static_cast<int32_t>(EBoundsCorner2::eRight)).x,
            operator[]((static_cast<int32_t>(corner) & static_cast<int32_t>(EBoundsCorner2::eTop)) >> 1).y,
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

    __host__ __device__ bool inside(Point2i p, Bounds2i b)
    {
        return cuTypes_all(to_int2(p) >= to_int2(b.pMin) && to_int2(p) <= to_int2(b.pMax));
    }

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
                    Index2 i{3 - idx + numel, numel};
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

#if !defined(_CUDACC__) && !defined(__CUDA_ARCH__)
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
        w.x /= w.w;
        w.y /= w.w;
        w.z /= w.w;
        w.w = 1.f;
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
            (*reinterpret_cast<Vector3f const*>(&xLow) + *reinterpret_cast<Vector3f const*>(&xHigh)) * 0.5f)};
    }
    __host__ __device__ Vector3f Point3fi::width() const
    {
        return *reinterpret_cast<Point3f const*>(&xHigh) - *reinterpret_cast<Point3f const*>(&xLow);
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