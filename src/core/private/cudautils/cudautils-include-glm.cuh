#include "cudautils/cudautils-macro.cuh"

#if defined(__CUDACC__)
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

#if !defined(__CUDACC__) && !defined(__CUDA_ARCH__)
    #include <Eigen/Dense>
#endif
#if defined(__CUDACC__)
    #pragma nv_diag_default 20012
    #pragma nv_diag_default 3012
#endif
#undef diag_suppress

#if !defined(__CUDACC__) && !defined(__CUDA_ARCH__)
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

#if defined(DMT_OS_WINDOWS)
    #pragma pop_macro("near")
#endif


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
