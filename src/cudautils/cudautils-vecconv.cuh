#pragma once

#if defined(__NVCC__)
#pragma nv_diag_suppress 20012
#endif
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/vec3.hpp>   // Vec3f
#include <glm/vec4.hpp>   // Vec4f
#include <glm/ext/quaternion_float.hpp>
#if defined(__NVCC__)
#pragma nv_diag_default 20012
#endif


namespace dmt {
    // Vector Types: Conversion to and from GLM ---------------------------------------------------------------------
    template <typename T, typename Enable = void>
    struct to_glm;

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
    template <Vector T>
    using to_glm_t = to_glm<T>::type;

    template <Vector T>
    inline constexpr __host__ __device__ to_glm_t<T>& toGLM(T& v)
    {
        return *std::bit_cast<to_glm_t<T>*>(&v);
    }
    template <Vector T>
    inline constexpr __host__ __device__ to_glm_t<T> const& toGLM(T const& v)
    {
        return *std::bit_cast<to_glm_t<T> const*>(&v);
    }

    template <int32_t n, Scalar T, glm::qualifier Q>
    struct from_glm;
    template <Scalar T, glm::qualifier Q>
    struct from_glm<2, T, Q>
    {
        using type = Tuple2<T>;
    };
    template <Scalar T, glm::qualifier Q>
    struct from_glm<3, T, Q>
    {
        using type = Tuple3<T>;
    };
    template <Scalar T, glm::qualifier Q>
    struct from_glm<4, T, Q>
    {
        using type = Tuple4<T>;
    };
    template <int32_t n, Scalar T, glm::qualifier Q>
    using from_glm_t = from_glm<n, T, Q>::type;

    template <int32_t n, Scalar T, glm::qualifier Q>
    inline constexpr __host__ __device__ from_glm_t<n, T, Q>& fromGLM(glm::vec<n, T, Q>& v)
    {
        return *std::bit_cast<from_glm_t<n, T, Q>*>(&v);
    }
    template <int32_t n, Scalar T, glm::qualifier Q>
    inline constexpr __host__ __device__ from_glm_t<n, T, Q> const& fromGLM(glm::vec<n, T, Q> const& v)
    {
        return *std::bit_cast<from_glm_t<n, T, Q> const*>(&v);
    }

    inline constexpr __host__ __device__ Quaternion& fromGLMquat(glm::quat& q)
    {
        return *std::bit_cast<Quaternion*>(&q);
    }

    inline constexpr __host__ __device__ Quaternion const& fromGLMquat(glm::quat const& q)
    {
        return *std::bit_cast<Quaternion const*>(&q);
    }

    inline constexpr __host__ __device__ glm::mat4& toGLMmat(Matrix4f& m) { return *std::bit_cast<glm::mat4*>(&m); }
    inline constexpr __host__ __device__ glm::mat4 const& toGLMmat(Matrix4f const& m)
    {
        return *std::bit_cast<glm::mat4 const*>(&m);
    }
    inline constexpr __host__ __device__ Matrix4f& fromGLMmat(glm::mat4& m) { return *std::bit_cast<Matrix4f*>(&m); }
    inline constexpr __host__ __device__ Matrix4f const& fromGLMmat(glm::mat4 const& m)
    {
        return *std::bit_cast<Matrix4f const*>(&m);
    }
} // namespace dmt