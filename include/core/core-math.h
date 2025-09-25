#pragma once

#include "core/core-macros.h"
#include "cudautils/cudautils.h"
#include "core/core-dstd.h"

#include "platform/platform-memory.h"

#if !defined(DMT_ARCH_X86_64)
    #error "Support only for AVX2 capable x86_64 CPU"
#endif


#if !defined(DMT_ARCH_X86_64)
    #error "Support only for AVX2 capable x86_64 CPU"
#endif

#include <immintrin.h>

namespace dmt::arch {
    DMT_CORE_API float hmin_ps(__m128 v);
    DMT_CORE_API float hmax_ps(__m128 v);
    DMT_CORE_API float hmin_ps(__m256 v);
    DMT_CORE_API float hmax_ps(__m256 v);

    DMT_CORE_API void transpose3x2(float const* src, float* x, float* y, float* z);
    DMT_CORE_API void transpose3x4(float const* src, float* x, float* y, float* z);
    DMT_CORE_API void transpose3x8(float const* src, float* x, float* y, float* z);
    DMT_CORE_API void transpose3xN(float const* src, float* x, float* y, float* z, size_t N);
} // namespace dmt::arch

namespace dmt::color {
    DMT_CORE_API RGB     rgbFromHsv(Point3f hsv);
    DMT_CORE_API Point3f hsvFromRgb(RGB rgb);
} // namespace dmt::color

namespace dmt::numbers {
    template <std::unsigned_integral... Args>
    constexpr uint64_t DMT_FASTCALL hashIntegers(Args... args)
        requires(sizeof...(Args) > 0)
    {
        uint64_t hash    = 0;
        auto     combine = [](uint64_t seed, uint64_t value) -> uint64_t {
            return seed ^ (value + 0x9e3779b97f4a7c15 + (seed << 6) + (seed >> 2));
        };

        ((hash = combine(hash, static_cast<uint64_t>(args))), ...);
        return hash;
    }

    DMT_FORCEINLINE constexpr uint64_t mixBits(uint64_t v)
    {
        v ^= (v >> 31);
        v *= 0x7fb5d329728ea185;
        v ^= (v >> 27);
        v *= 0x81dadef4bc2dd44d;
        v ^= (v >> 33);
        return v;
    }

    DMT_CORE_API uint16_t permutationElement(int32_t i, uint32_t l, uint64_t p);
} // namespace dmt::numbers

namespace dmt {
    struct DMT_CORE_API TriangleData
    {
        Point3f v0, v1, v2;
        // TODO remove
        RGB color;
    };

    // https: //www.readkong.com/page/octahedron-environment-maps-6054207
    struct DMT_CORE_API OctahedralNorm
    {
        uint16_t x, y;
    };

    DMT_CORE_API OctahedralNorm octaFromNorm(Normal3f n);
    DMT_CORE_API Normal3f       normFromOcta(OctahedralNorm o);

    DMT_CORE_API void      extractAffineTransform(Matrix4f const& m, float affineTransform[12]);
    DMT_CORE_API Matrix4f  matrixFromAffine(float const affineTransform[12]);
    DMT_CORE_API Transform transformFromAffine(float const affineTransform[12]);

    DMT_FORCEINLINE DMT_CORE_API inline void coordinateSystemFallback(Vector3f const& n, Vector3f* t, Vector3f* b)
    {
        if (fabsf(n.x) > fabsf(n.z))
            *t = Vector3f(-n.y, n.x, 0.f);
        else
            *t = Vector3f(0.f, -n.z, n.y);

        *t = normalize(*t);
        *b = cross(n, *t);
    }

    DMT_CORE_API float lookupTableRead(float const* table, float x, int32_t size);
    DMT_CORE_API float lookupTableRead2D(float const* table, float x, float y, int32_t sizex, int32_t sizey);

    /// Maybe move to `cudautils`. This is used to sample from a PMF whose waveform is known, starting from a uniformly distributed random
    /// number in a continuous domain. Furthermore, such initial sample is remapped such that it can be reused
    /// How is it remapped: The offset between the 2 bounding CDF values is in itself still a random number, which can be remapped in the 0,1 range
    DMT_CORE_API int32_t sampleDiscrete(float const* weights, uint32_t weightCount, float u, float* pmf, float* uRemapped);

    /// @{
    DMT_CORE_API uint32_t                        decodeMorton2D(uint32_t morton);
    DMT_CORE_API uint32_t                        encodeMorton2D(uint32_t x, uint32_t y);
    DMT_CORE_API DMT_FORCEINLINE inline uint32_t decodeMortonX(uint32_t morton) { return decodeMorton2D(morton); }
    DMT_CORE_API DMT_FORCEINLINE inline uint32_t decodeMortonY(uint32_t morton) { return decodeMorton2D(morton >> 1); }
    /// @}

    // TODO move elsewhere?
    DMT_FORCEINLINE inline void resetMonotonicBufferPointer(std::pmr::monotonic_buffer_resource& resource,
                                                            unsigned char*                       ptr,
                                                            uint32_t                             bytes)
    {
        // https://developercommunity.visualstudio.com/t/monotonic_buffer_resourcerelease-does/10624172
        auto* upstream = resource.upstream_resource();
        std::destroy_at(&resource);
        std::construct_at(&resource, ptr, bytes, upstream);
    }


    // distributions
    /**
     * @brief class to represent a 1D PDF and its CDF functions as
     * <ul>
     *  <li>PDF -> constant interpolated piecewise function</li>
     *  <li>CDF -> linearly interpolated piecewise function</li>
     * </ul>
     * We store the absolute value of the sampled function and its CDF computed with a normalized running sum
     * TODO: If necessary, define copy-control, otherwise, just avoid it and pass them with const&
     */
    class PiecewiseConstant1D
    {
    public:
        DMT_CORE_API PiecewiseConstant1D(std::span<float const>     func,
                                         float                      min,
                                         float                      max,
                                         std::pmr::memory_resource* memory = std::pmr::get_default_resource());

        float DMT_CORE_API    integral() const;
        uint32_t DMT_CORE_API size() const;

        /// @warning returns nan if x outside range min, max
        float DMT_CORE_API invert(float x) const;

        float DMT_CORE_API sample(float u, float* pdf = nullptr, int32_t* offset = nullptr) const;

        DMT_FORCEINLINE inline std::span<float const> absFunc() const { return {m_buffer.get(), m_funcCount}; }
        DMT_FORCEINLINE inline std::span<float const> CDF() const
        {
            return {m_buffer.get() + m_funcCount, m_funcCount};
        }
        DMT_FORCEINLINE inline std::span<float const> absFunc() { return {m_buffer.get(), m_funcCount}; }
        DMT_FORCEINLINE inline std::span<float const> CDF() { return {m_buffer.get() + m_funcCount, m_funcCount}; }

    private:
        DMT_FORCEINLINE inline std::span<float> absFunc_() { return {m_buffer.get(), m_funcCount}; }
        DMT_FORCEINLINE inline std::span<float> CDF_() { return {m_buffer.get() + m_funcCount, m_funcCount}; }

    private:
        UniqueRef<float[]> m_buffer;    // first half abs(func), second half CDF
        uint32_t           m_funcCount; // half the count of floats in buffer
        float              m_min, m_max;
        float              m_integral;
    };

    /**
     * @brief class to represent a 2D PDF and its CDF functions by decomposing a 2D distribution
     * proportional to the input function into
     * - the marginal PDF (1D piecewise constant) over the first axis
     * - a list, for each value in the first axis, of conditional PDFs, because p(y | x) = p(x, y) / p(x)
     * @see PiecewiseConstant1D
     */
    class PiecewiseConstant2D
    {
    public:
        DMT_CORE_API PiecewiseConstant2D(dstd::Array2D<float> const& data,
                                         Bounds2f                    domain,
                                         std::pmr::memory_resource*  memory = std::pmr::get_default_resource(),
                                         std::pmr::memory_resource*  temp   = std::pmr::get_default_resource());

        Bounds2f DMT_CORE_API domain() const;
        Point2i DMT_CORE_API  resolution() const;
        float DMT_CORE_API    integral() const;

        /// @warning returns nan if p outside bounds
        Point2f DMT_CORE_API invert(Point2f p) const;

        Point2f DMT_CORE_API sample(Point2f u, float* pdf = nullptr, Point2i* offset = nullptr) const;

        float DMT_CORE_API     pdf(Point2f pr) const;
        std::span<float const> conditionalCdfRow(int y) const { return m_pConditionalV[y].CDF(); }

        std::span<float const> marginalCdf() const { return m_pMarginalV.CDF(); }

    private: // assume variables are u and v, v -> marginal, u -> computed through conditional probability
        Bounds2f                              m_domain;
        std::pmr::vector<PiecewiseConstant1D> m_pConditionalV;
        PiecewiseConstant1D                   m_pMarginalV;
    };
} // namespace dmt

namespace dmt::transforms {
    /**
     * `F/a  0  0       0`
     * `0   -F  0       0`
     * `0    0  n/(f-n) fn/(f-n)`
     * `0    0 -1       0`
     * where
     * - F = 1/(tan(fov/2))
     * - f = far clip plane
     * - n = near clip plane
     * - a = aspect ratio
     * This is used to initialize the screenFromCamera Matrix
     */
    DMT_CORE_API Transform DMT_FASTCALL persp(float fovRadians, float aspectRatio, float near, float far);

    DMT_CORE_API Transform DMT_FASTCALL scale(Vector3f s);

    DMT_CORE_API Transform DMT_FASTCALL translate(Vector3f v);

    /**
     * Build matrix column-wise (camera space → camera-world space)
     * Camera space (left handed):
     * - X: right | Y: up      | Z: forward
     * Camera-World space (right handed):
     * - X: right | Y: forward | Z: up
     * So camera X → world right, Y → world up, Z → world forward
     */
    DMT_CORE_API Transform DMT_FASTCALL cameraWorldFromCamera(Normal3f cameraDirection);

    /**
     * Build matrix column-wise (camera space → world space)
     * Camera space (left handed):
     * - X: right | Y: up      | Z: forward
     * - origin: camera position
     * World space (right handed):
     * - X: right | Y: forward | Z: up
     * - origin: 0,0,0
     * So camera X → world right, Y → world up, Z → world forward
     */
    DMT_CORE_API Transform DMT_FASTCALL worldFromCamera(Normal3f cameraDirection, Point3f cameraPosition);

    DMT_CORE_API Transform DMT_FASTCALL
        cameraFromRaster_Perspective(float focalLength, float sensorHeight, uint32_t xRes, uint32_t yRes);
} // namespace dmt::transforms