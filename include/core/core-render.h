#pragma once

#include "core/core-macros.h"
#include "core/core-bsdf.h"
#include "core/core-texture.h"
#include "core/core-texture-cache.h"
#include "core/core-trianglemesh.h"
#include "core/core-light.h"

#include "platform/platform-memory.h"
#include "platform/platform-threadPool.h"

namespace dmt {
    struct Parameters
    {
        float               focalLength     = 20.f;
        float               sensorSize      = 36.f;
        Vector3f            cameraDirection = {0, 1, 0};
        Point2i             filmResolution  = {128, 128};
        int32_t             samplesPerPixel = 1;
        int32_t             maxDepth        = 5;
        UniqueRef<EnvLight> envLight        = nullptr;
    };

    class Renderer
    {
    public:
        DMT_CORE_API Renderer(size_t tmpSize = 4096);
        Renderer(Renderer const&)                = delete;
        Renderer(Renderer&&) noexcept            = delete;
        Renderer& operator=(Renderer const&)     = delete;
        Renderer& operator=(Renderer&&) noexcept = delete;
        DMT_CORE_API ~Renderer() noexcept;

    public:
        /// once started, public members shouldn't be changed anymore
        DMT_CORE_API void startRenderThread();

    public:
        // read only stuff for workers and render thread
        Scene        scene;
        Parameters   params;
        TextureCache texCache;

    private:
        std::pmr::monotonic_buffer_resource tmpMem();

        void resetBigTmp();

    private:
        // small local temp memory
        unsigned char m_smallBuffer[256];

        // bigger local temp memory
        UniqueRef<unsigned char[]> m_bigBuffer;
        size_t                     m_bigBufferSize;

        // upstream arena instance
        std::pmr::monotonic_buffer_resource m_bigTmpMem;

        std::pmr::synchronized_pool_resource m_poolMem;

        // threading resources
        os::Thread   m_renderThread;
        ThreadPoolV2 m_workers;

        UniqueRef<unsigned char[]> m_renderThreadData;
    };
} // namespace dmt

namespace dmt::sampling {
    inline constexpr uint32_t                        NumPrimes = 16;
    inline constexpr std::array<uint32_t, NumPrimes> Primes{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};

    struct DigitPermutation
    {
        uint16_t const* perms;
        uint32_t        nDigits;
        uint32_t        base;

        inline uint32_t permsCount() const { return nDigits * base; }
        inline int32_t  permute(int32_t digitIndex, int32_t digitValue) const
        {
            return perms[digitIndex * base + digitValue];
        }
    };
    static_assert(std::is_standard_layout_v<DigitPermutation> && std::is_trivial_v<DigitPermutation>);

    class DigitPermutations
    {
    public:
        DMT_CORE_API DigitPermutations(uint32_t maxPrimeIndex, uint32_t seed, std::pmr::memory_resource* _memory);
        DigitPermutations(DigitPermutations const&)                         = delete;
        DigitPermutations&              operator=(DigitPermutations const&) = delete;
        DMT_CORE_API                    DigitPermutations(DigitPermutations&& other) noexcept;
        DMT_CORE_API DigitPermutations& operator=(DigitPermutations&& other) noexcept;
        DMT_FORCEINLINE ~DigitPermutations() noexcept { destroy(); }

        DMT_FORCEINLINE bool             isValid() const { return m_buffer; }
        DMT_FORCEINLINE uint32_t         numPrimes() const { return m_maxPrimeIndex; }
        DMT_FORCEINLINE DigitPermutation permutation(uint32_t primeIndex) const
        {
            return {.perms   = permBuffer()[primeIndex].get(),
                    .nDigits = nDigitsBuffer()[primeIndex],
                    .base    = Primes[primeIndex]};
        }

    private:
        DMT_CORE_API void destroy() noexcept;

        // clang-format off
        DMT_FORCEINLINE UniqueRef<uint16_t[]>* permBuffer() { return reinterpret_cast<UniqueRef<uint16_t[]>*>(m_buffer); }
        DMT_FORCEINLINE uint8_t* nDigitsBuffer() { return reinterpret_cast<uint8_t*>(permBuffer() + m_maxPrimeIndex); }
        DMT_FORCEINLINE UniqueRef<uint16_t[]> const* permBuffer() const { return reinterpret_cast<UniqueRef<uint16_t[]> const*>(m_buffer); }
        DMT_FORCEINLINE uint8_t const* nDigitsBuffer() const { return reinterpret_cast<uint8_t const*>(permBuffer() + m_maxPrimeIndex); }
        // clang-format on

    private:
        std::pmr::memory_resource* m_memory        = nullptr; // memory_resource must outlive object
        uint32_t                   m_maxPrimeIndex = 0;
        unsigned char*             m_buffer        = nullptr;
    };

    template <typename T>
    concept Sampler = requires(std::remove_cvref_t<T>& t) {
        {
            t.startPixelSample(std::declval<Point2i>(), std::declval<int32_t>(), std::declval<int32_t>())
        };
        {
            t.get1D()
        } -> std::floating_point;
        {
            t.get2D()
        } -> std::same_as<Point2f>;
        {
            t.getPixel2D()
        } -> std::same_as<Point2f>;
    } && !std::is_pointer_v<std::remove_cvref_t<T>>;

    class HaltonOwen
    {
    public:
        DMT_CORE_API HaltonOwen(int32_t samplesPerPixel, Point2i resolution, int32_t seed);

        DMT_CORE_API void    startPixelSample(Point2i p, int32_t sampleIndex, int32_t dim = 0);
        DMT_CORE_API float   get1D();
        DMT_CORE_API Point2f get2D();
        DMT_CORE_API Point2f getPixel2D();

    private:
        static constexpr int32_t                            MaxHaltonResolution = 128;
        static constexpr int32_t                            NumDimensions       = 2; // width, height
        static constexpr std::array<int32_t, NumDimensions> DimPrimes{2, 3};
        static constexpr std::array<int32_t, NumDimensions> DimPrimeIndices{0, 1};

        int64_t m_haltonIndex = 0;
        int32_t m_dimension   = 0;
        int32_t m_multInvs[NumDimensions]{};
        int32_t m_baseScales[NumDimensions]{};
        int32_t m_baseExponents[NumDimensions]{};
    };
    static_assert(Sampler<HaltonOwen>);

} // namespace dmt::sampling

namespace dmt::filtering {
    // imagine placing a 2D filter function over each pixel center coordinate (index + 0.5). Instead of using
    // directly samples computed from a given pixel, we use inverse transform method to repurpose the sample
    // according to the PDF associated to the filter function. Such strategy is equivalent to applying the low pass filter
    // to the reconstructed signal, which is then computed with a weighted sum of all contributions on the pixel. The weight
    // is directly proportional to the PDF value of the repurposed sample
    struct FilterSample
    {
        Point2f p;
        float   weight;
    };

    // to be able to sample from a filter function, you need to store its function2D and construct a distribution from it
    // then, you also need to keep storing its original function to compute a weight associated to the sample with f(sampleIdx) / pdf
    // note that we need to store f too as it is the signed, unnormalized version of the absFunc from the distirbution
    /// @note doesn't have copy control, pass around as reference
    template <typename T>
    concept Filter = requires(std::remove_cvref_t<T> const& t) {
        {
            t.evaluate(std::declval<Point2f>())
        } -> std::floating_point;
        {
            t.radius()
        } -> std::same_as<Vector2f>;
    } && !std::is_pointer_v<std::remove_cvref_t<T>>;

    class FilterSampler
    {
    public:
        static constexpr int32_t NumSamplesPerAxisPerDomainUnit = 32;

    private:
        template <Filter T>
        static dstd::Array2D<float> evaluateFunction(T const& filter, Bounds2f domain, std::pmr::memory_resource* memory)
        {
            uint32_t xSize = static_cast<uint32_t>(NumSamplesPerAxisPerDomainUnit * filter.radius().x);
            if (!isPOT(xSize))
                xSize = nextPOT(xSize);
            uint32_t ySize = static_cast<uint32_t>(NumSamplesPerAxisPerDomainUnit * filter.radius().y);
            if (!isPOT(ySize))
                ySize = nextPOT(ySize);

            dstd::Array2D<float> nrvo{xSize, ySize, memory};
            // tabularize filter function
            for (uint32_t y = 0; y < ySize; ++y)
            {
                for (uint32_t x = 0; x < xSize; ++x)
                {
                    Point2f p  = domain.lerp({{(x + 0.5f) / xSize, (y + 0.5f) / ySize}});
                    nrvo(x, y) = filter.evaluate(p); // TODO vectorize
                }
            }

            return nrvo;
        }

        template <Filter T>
        static Bounds2f domainFromFilter(T const& filter)
        {
            return makeBounds(-filter.radius(), filter.radius());
        }

    public:
        template <Filter T>
        FilterSampler(T const&                   filter,
                      std::pmr::memory_resource* memory = std::pmr::get_default_resource(),
                      std::pmr::memory_resource* temp   = std::pmr::get_default_resource()) :
        m_f(evaluateFunction(filter, domainFromFilter(filter), memory)),
        m_distrib(m_f, domainFromFilter(filter), memory, temp)
        {
        }

        DMT_FORCEINLINE Bounds2f domain() const { return m_distrib.domain(); }

        DMT_FORCEINLINE FilterSample sample(Point2f u) const
        {
            FilterSample result;
            float        pdf;
            Point2i      pi;

            result.p      = m_distrib.sample(u, &pdf, &pi);
            result.weight = m_f(pi.x, pi.y) / pdf;

            return result;
        }

    private:
        dstd::Array2D<float> m_f;
        PiecewiseConstant2D  m_distrib;
    };

    class Mitchell
    {
    public:
        inline Mitchell(Vector2f                   radius = {{2.f, 2.f}},
                        float                      b      = 1.f / 3.f,
                        float                      c      = 1.f / 3.f,
                        std::pmr::memory_resource* memory = std::pmr::get_default_resource(),
                        std::pmr::memory_resource* temp   = std::pmr::get_default_resource()) :
        m_radius(radius),
        m_b(b),
        m_c(c),
        m_sampler(*this, memory, temp) // needs radius, b and c, cause evaluate and radius needs to work at this point
        {
        }

        DMT_FORCEINLINE FilterSample sample(Point2f u) const { return m_sampler.sample(u); }

        DMT_FORCEINLINE float evaluate(Point2f p) const
        {
            return mitchell1D(2 * p.x / m_radius.x, m_b, m_c) * mitchell1D(2 * p.y / m_radius.y, m_b, m_c);
        }

        // domain, for the x axis and y axis, in which the filter is != 0 starting from 0 (low pass filter)
        DMT_FORCEINLINE Vector2f radius() const { return m_radius; }

        // 2D integral of the filter
        DMT_FORCEINLINE float integral() const { return m_radius.x * m_radius.y / 4; }

    private:
        static inline float mitchell1D(float x, float b, float c)
        {
            x = std::abs(x);
            if (x <= 1)
                return ((12 - 9 * b - 6 * c) * x * x * x + (-18 + 12 * b + 6 * c) * x * x + (6 - 2 * b)) * (1.f / 6.f);
            else if (x <= 2)
                return ((-b - 6 * c) * x * x * x + (6 * b + 30 * c) * x * x + (-12 * b - 48 * c) * x + (8 * b + 24 * c)) *
                       (1.f / 6.f);
            else
                return 0;
        }

    private:
        Vector2f      m_radius;
        float         m_b, m_c;
        FilterSampler m_sampler;
    };
    static_assert(Filter<Mitchell>);
} // namespace dmt::filtering

namespace dmt::film {
    template <typename T>
    concept Film = requires(std::remove_cvref_t<T>& t) {
        {
            t.addSample(std::declval<Point2i>(), std::declval<RGB>(), 0.f)
        };
    } && !std::is_pointer_v<std::remove_cvref_t<T>>;

    class RGBFilm
    {
    private:
        struct Pixel
        {
            double rgbSum[3];
            double weightSum;
        };

    public:
        //RGBFilm(Point2i resolution, float maxComponentValue = fl::infinity(), std::pmr::memory_resource* memory = std::pmr::get_default_resource()):
        //m_pixels(makeUniqueRef<Pixel[]>(memory, resolution.x * resolution.y), m_resolution(resolution), m_maxComponentValue(maxComponentValue) {}
        inline RGBFilm(Point2i                    res,
                       float                      maxComp = fl::infinity(),
                       std::pmr::memory_resource* memory  = std::pmr::get_default_resource()) :
        m_pixels(makeUniqueRef<Pixel[]>(memory, res.x * res.y)),
        m_resolution(res),
        m_maxComponentValue(maxComp)
        {
            std::memset(m_pixels.get(), 0, sizeof(Pixel) * res.x * res.y);
        }

        // TODO switch to EXR
        DMT_CORE_API void writeImage(os::Path const&            imagePath,
                                     std::pmr::memory_resource* temp = std::pmr::get_default_resource());
        DMT_CORE_API void addSample(Point2i pixel, RGB sample, float weight);

        DMT_FORCEINLINE Point2i resolution() const { return m_resolution; }

    private:
        UniqueRef<Pixel[]> m_pixels;
        Point2i            m_resolution;
        float              m_maxComponentValue;
    };
    static_assert(Film<RGBFilm>);
} // namespace dmt::film

namespace dmt::camera {
    struct CameraSample
    {
        /// point on the film to which the generated ray should carry radiance (meaning pixelPosition + random offset)
        Point2f pFilm;

        /// point on the lens the ray passes through
        Point2f pLens;

        /// time at which the ray should sample the scene. If the camera itself is in motion,
        /// the time value determines what camera position to use when generating the ray
        float time;

        /// scale factor that is applied when the ray’s radiance is added to the image stored by the film;
        /// it accounts for the reconstruction filter used to filter image samples at each pixel
        float filterWeight;
    };

    template <filtering::Filter F, sampling::Sampler S>
    CameraSample getCameraSample(S& sampler, Point2i pPixel, F const& filter)
    {
        filtering::FilterSample const fs = filter.sample(sampler.getPixel2D());
        Point2f const                 pPixelf{{static_cast<float>(pPixel.x), static_cast<float>(pPixel.y)}};
        Vector2f const                pixelShift{{0.5f, 0.5f}};
        CameraSample const            res{.pFilm        = (pPixelf + pixelShift) + fs.p,
                                          .pLens        = sampler.get2D(),
                                          .time         = sampler.get1D(),
                                          .filterWeight = fs.weight};
        assert(res.pFilm.x >= pPixel.x + 0.5f - filter.radius().x);
        assert(res.pFilm.x <= pPixel.x + 0.5f + filter.radius().x);
        assert(res.pFilm.y >= pPixel.y + 0.5f - filter.radius().y);
        assert(res.pFilm.y <= pPixel.y + 0.5f + filter.radius().y);
        return res;
    }

    DMT_CORE_API Ray generateRay(CameraSample const& cs, Transform const& cameraFromRaster, Transform const& renderFromCamera);

    DMT_CORE_API ApproxDifferentialsContext minDifferentialsFromCamera(
        Transform const&     cameraFromRaster,
        Transform const&     renderFromCamera,
        film::RGBFilm const& film,
        uint32_t             samplesPerPixel);

} // namespace dmt::camera
