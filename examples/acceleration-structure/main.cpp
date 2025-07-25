#include "utilities.h"

#define DMT_ENTRY_POINT
#include "platform/platform.h"
#include "core/core-cudautils-cpubuild.h"
#include "core/core-bvh-builder.h"
#include "core/core-primitive.h"

#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <iomanip>

namespace dmt::ddbg {
    std::string printBVHToString(BVHBuildNode* node, int depth = 0, std::string const& prefix = "")
    {
        std::string result;
        if (depth == 0)
            result += "\n";

        constexpr auto boundsToString = [](Bounds3f const& b) -> std::string {
            std::ostringstream oss;
            oss << "Bounds[( " << b.pMin.x << ", " << b.pMin.y << ", " << b.pMin.z << " ) - ( " << b.pMax.x << ", "
                << b.pMax.y << ", " << b.pMax.z << " )]";
            return oss.str();
        };

        if (!node)
            return result;

        std::string indent(depth * 2, ' ');

        if (node->childCount == 0 && node->primitiveCount > 0)
        {
            result += indent + prefix + "Leaf [count: " + std::to_string(node->primitiveCount) + ", ";
            result += boundsToString(node->bounds) + "]\n";
        }
        else
        {
            result += indent + prefix + "Internal [children: " + std::to_string(node->childCount) + ", ";
            result += boundsToString(node->bounds) + "]\n";

            for (uint32_t i = 0; i < node->childCount; ++i)
            {
                bool        isLast      = (i == node->childCount - 1);
                std::string childPrefix = isLast ? "└─ " : "├─ ";
                result += printBVHToString(node->children[i], depth + 1, childPrefix);
            }
        }

        return result;
    }

    std::vector<TriangleData> makeCubeTriangles()
    {
        std::vector<TriangleData> tris;

        float const                  s        = 1.0f;
        std::array<Point3f, 8> const vertices = {
            {{{-s, -s, -s}}, {{s, -s, -s}}, {{s, s, -s}}, {{-s, s, -s}}, {{-s, -s, s}}, {{s, -s, s}}, {{s, s, s}}, {{-s, s, s}}}};

        std::array<std::array<int, 3>, 12> const indices = {{
            // Front face
            {0, 1, 2},
            {2, 3, 0},
            // Back face
            {4, 7, 6},
            {6, 5, 4},
            // Left face
            {0, 3, 7},
            {7, 4, 0},
            // Right face
            {1, 5, 6},
            {6, 2, 1},
            // Top face
            {3, 2, 6},
            {6, 7, 3},
            // Bottom face
            {0, 4, 5},
            {5, 1, 0},
        }};

        for (auto const& idx : indices)
        {
            tris.push_back({{{vertices[idx[0]].x, vertices[idx[0]].y, vertices[idx[0]].z}},
                            {{vertices[idx[1]].x, vertices[idx[1]].y, vertices[idx[1]].z}},
                            {{vertices[idx[2]].x, vertices[idx[2]].y, vertices[idx[2]].z}}});
        }

        return tris;
    }

    std::vector<TriangleData> makePlaneTriangles(float size = 1.0f)
    {
        std::vector<TriangleData> tris;

        float const s = size * 0.5f;

        Point3f const v0{{-s, 0.0f, -s}};
        Point3f const v1{{s, 0.0f, -s}};
        Point3f const v2{{s, 0.0f, s}};
        Point3f const v3{{-s, 0.0f, s}};

        // Triangle 1: v0, v1, v2
        tris.emplace_back(v0, v1, v2);

        // Triangle 2: v2, v3, v0
        tris.emplace_back(v2, v3, v0);

        return tris;
    }

    std::pmr::vector<TriangleData> debugScene(std::pmr::memory_resource* memory = std::pmr::get_default_resource())
    {
        std::pmr::vector<TriangleData> scene{memory};

        auto cube  = makeCubeTriangles();
        auto plane = makePlaneTriangles(4.0f); // Large ground plane

        scene.insert(scene.end(), cube.begin(), cube.end());
        scene.insert(scene.end(), plane.begin(), plane.end());
        return scene;
    }

    std::pmr::vector<Primitive const*> rawPtrsCopy(std::pmr::vector<dmt::UniqueRef<Primitive>> const& ownedPrimitives,
                                                   std::pmr::memory_resource* memory = std::pmr::get_default_resource())
    {
        std::pmr::vector<Primitive const*> rawPtrs{memory};
        rawPtrs.reserve(ownedPrimitives.size());

        for (auto const& ref : ownedPrimitives)
            rawPtrs.push_back(ref.get());

        return rawPtrs;
    }
} // namespace dmt::ddbg

namespace dmt::test {
    void testBoundsEquality(std::span<TriangleData> scene, std::span<dmt::Primitive const*> spanPrims)
    {
        Context ctx;
        assert(ctx.isValid() && "Need valid context");

        Bounds3f const
            sceneBounds = std::transform_reduce(scene.begin(), scene.end(), bbEmpty(), [](dmt::Bounds3f a, dmt::Bounds3f b) {
            return bbUnion(a, b);
        }, [](TriangleData const& t) { return Bounds3f{min(min(t.v0, t.v1), t.v2), max(max(t.v0, t.v1), t.v2)}; });

        Bounds3f const primsBounds = std::transform_reduce( //
            spanPrims.begin(),
            spanPrims.end(),
            bbEmpty(),
            [](Bounds3f a, Bounds3f b) { return bbUnion(a, b); },
            [](Primitive const* p) { return p->bounds(); });

        if (sceneBounds != primsBounds)
        {
            ctx.error("{{ {{ {} {} {} }} {{ {} {} {} }}}}",
                      std::make_tuple(sceneBounds.pMin.x,
                                      sceneBounds.pMin.y,
                                      sceneBounds.pMin.z,
                                      sceneBounds.pMax.x,
                                      sceneBounds.pMax.y,
                                      sceneBounds.pMax.z));
            ctx.error("vs", {});
            ctx.error("{{ {{ {} {} {} }} {{ {} {} {} }}}}",
                      std::make_tuple(primsBounds.pMin.x,
                                      primsBounds.pMin.y,
                                      primsBounds.pMin.z,
                                      primsBounds.pMax.x,
                                      primsBounds.pMax.y,
                                      primsBounds.pMax.z));
            assert(false && "why");
        }
        else
        {
            ctx.log("{{ {{ {} {} {} }} {{ {} {} {} }}}}",
                    std::make_tuple(primsBounds.pMin.x,
                                    primsBounds.pMin.y,
                                    primsBounds.pMin.z,
                                    primsBounds.pMax.x,
                                    primsBounds.pMax.y,
                                    primsBounds.pMax.z));
        }
    }
} // namespace dmt::test

namespace dmt::sampling {
    inline constexpr uint32_t                        NumPrimes = 16;
    inline constexpr std::array<uint32_t, NumPrimes> Primes{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};

    struct DigitPermutation
    {
        uint16_t const* perms;
        uint32_t        nDigits;
        uint32_t        base;

        uint32_t permsCount() const { return nDigits * base; }
        int32_t  permute(int32_t digitIndex, int32_t digitValue) const { return perms[digitIndex * base + digitValue]; }
    };
    static_assert(std::is_standard_layout_v<DigitPermutation> && std::is_trivial_v<DigitPermutation>);

    class DigitPermutations
    {
    public:
        DigitPermutations(uint32_t maxPrimeIndex, uint32_t seed, std::pmr::memory_resource* _memory) :
        m_memory{_memory},
        m_maxPrimeIndex{maxPrimeIndex}
        {
            assert(maxPrimeIndex <= NumPrimes);

            m_buffer = reinterpret_cast<unsigned char*>(
                m_memory->allocate(m_maxPrimeIndex * (sizeof(UniqueRef<uint16_t[]>) + sizeof(uint8_t))));
            if (!m_buffer)
                return;

            for (uint32_t i = 0; i < maxPrimeIndex; ++i)
            {
                // compute number of digits for the base
                uint32_t const base     = Primes[i];
                float const    invBase  = dmt::fl::rcp(base);
                uint8_t        nDigits  = 0;
                float          invBaseM = 1.f;
                // until floating point subtracton of the current power has an effect
                while (1 - (base - 1) * invBaseM < 1)
                {
                    ++nDigits;
                    invBaseM *= invBase;
                }

                nDigitsBuffer()[i] = nDigits;
                std::construct_at(&permBuffer()[i], makeUniqueRef<uint16_t[]>(m_memory, nDigits * base));

                // compute permutations for all digits
                uint16_t* permutations = permBuffer()[i].get();
                for (int32_t digitIndex = 0; digitIndex < nDigits; ++digitIndex)
                {
                    uint64_t const dseed = dmt::numbers::hashIntegers(base, static_cast<uint32_t>(digitIndex), seed);
                    for (int32_t digitValue = 0; digitValue < base; ++digitValue)
                    {
                        int32_t const index = digitIndex * base + digitValue;
                        // only one?
                        permutations[index] = dmt::numbers::permutationElement(digitValue, digitValue, base, dseed);
                    }
                }
            }
        }

        // Delete copy constructor and copy assignment
        DigitPermutations(DigitPermutations const&)            = delete;
        DigitPermutations& operator=(DigitPermutations const&) = delete;

        // Move constructor
        DigitPermutations(DigitPermutations&& other) noexcept :
        m_memory(other.m_memory),
        m_maxPrimeIndex(other.m_maxPrimeIndex),
        m_buffer(other.m_buffer)
        {
            other.m_memory        = nullptr;
            other.m_maxPrimeIndex = 0;
            other.m_buffer        = nullptr;
        }

        // Move assignment
        DigitPermutations& operator=(DigitPermutations&& other) noexcept
        {
            if (this != &other)
            {
                // Free existing buffer
                destroy();

                // Steal other's resources
                m_memory        = other.m_memory;
                m_maxPrimeIndex = other.m_maxPrimeIndex;
                m_buffer        = other.m_buffer;

                other.m_memory        = nullptr;
                other.m_maxPrimeIndex = 0;
                other.m_buffer        = nullptr;
            }
            return *this;
        }

        // Destructor
        ~DigitPermutations() noexcept { destroy(); }

        bool             isValid() const { return m_buffer; }
        uint32_t         numPrimes() const { return m_maxPrimeIndex; }
        DigitPermutation permutation(uint32_t primeIndex) const
        {
            return {.perms   = permBuffer()[primeIndex].get(),
                    .nDigits = nDigitsBuffer()[primeIndex],
                    .base    = Primes[primeIndex]};
        }

    private:
        void destroy() noexcept
        {
            if (m_buffer && m_memory)
            {
                // Destroy all UniqueRefs explicitly
                for (uint32_t i = 0; i < m_maxPrimeIndex; ++i)
                {
                    std::destroy_at(&permBuffer()[i]);
                }

                m_memory->deallocate(m_buffer, m_maxPrimeIndex * (sizeof(UniqueRef<uint16_t[]>) + sizeof(uint8_t)));
            }

            m_buffer = nullptr;
        }


        DMT_FORCEINLINE UniqueRef<uint16_t[]>* permBuffer()
        {
            return reinterpret_cast<UniqueRef<uint16_t[]>*>(m_buffer);
        }

        DMT_FORCEINLINE uint8_t* nDigitsBuffer()
        { //
            return reinterpret_cast<uint8_t*>(permBuffer() + m_maxPrimeIndex);
        }

        DMT_FORCEINLINE UniqueRef<uint16_t[]> const* permBuffer() const
        {
            return reinterpret_cast<UniqueRef<uint16_t[]> const*>(m_buffer);
        }

        DMT_FORCEINLINE uint8_t const* nDigitsBuffer() const
        {
            return reinterpret_cast<uint8_t const*>(permBuffer() + m_maxPrimeIndex);
        }

    private:
        std::pmr::memory_resource* m_memory        = nullptr; // memory_resource must outlive object
        uint32_t                   m_maxPrimeIndex = 0;
        unsigned char*             m_buffer        = nullptr;
    };

    float radicalInverse(uint32_t primeIndex, uint64_t num)
    {
        assert(primeIndex < NumPrimes && "Exceeding number of primes for radical inverse computation");

        uint32_t const base    = Primes[primeIndex];
        float const    invBase = dmt::fl::rcp(base);
        float          result  = 0.f;
        float          factor  = invBase;

        while (num > 0)
        {
            uint64_t const digit = num % base;

            result += digit * factor;
            num /= base;
            factor *= invBase;
        }

        return result;
    }

    uint64_t inverseRadicalInverse(uint64_t inverse, int32_t base, int32_t nDigits)
    {
        uint64_t index = 0;
        for (int32_t i = 0; i < nDigits; ++i)
        {
            uint64_t const digit = inverse % base;
            inverse /= base;
            index *= base;
            index += digit;
        }
        return index;
    }

    float scrambledRadicalInverse(DigitPermutation const perm, uint64_t num)
    {
        float const invBase       = dmt::fl::rcp(perm.base);
        int32_t     digitIndex    = 0;
        float       invBaseM      = 1.f;
        uint64_t    reverseDigits = 0;
        while (1 - (perm.base - 1) * invBaseM < 1)
        {
            uint64_t const next       = num / perm.base;
            int32_t const  digitValue = num - next * perm.base; // basically num % perm.base
            reverseDigits *= perm.base + perm.permute(digitIndex, digitValue);
            invBaseM *= invBase;
            ++digitIndex;
            num = next;
        }

        return std::min(invBaseM * reverseDigits, dmt::fl::oneMinusEps());
    }

    float owenScrambledRadicalInverse(int32_t primeIndex, uint64_t num, uint32_t hash)
    {
        assert(primeIndex < NumPrimes && "primeIndex too high");
        namespace dn                 = dmt::numbers;
        uint32_t const base          = Primes[primeIndex];
        float const    invBase       = dmt::fl::rcp(base);
        int32_t        digitIndex    = 0;
        float          invBaseM      = 1.f;
        uint64_t       reverseDigits = 0;
        while (1 - (base - 1) * invBaseM < 1)
        {
            uint64_t const next = num / base;
            // second param ignored. TODO: SIMD
            int32_t const digitValue = dn::permutationElement(num - next * base, base, base, dn::mixBits(hash ^ reverseDigits));
            reverseDigits = reverseDigits * base + digitValue;
            invBaseM *= invBase;
            ++digitIndex;
            num = next;
        }

        return std::min(invBaseM * reverseDigits, dmt::fl::oneMinusEps());
    }

    class HaltonOwen
    {
    public:
        HaltonOwen(int32_t samplesPerPixel, Point2i resolution, int32_t seed)
        {
            for (int32_t i = 0; i < NumDimensions; ++i)
            {
                // using smallest primes for the 2 dimensions
                int32_t const base  = DimPrimes[i];
                int32_t       scale = 1, exp = 0;
                while (scale < std::min(resolution[i], MaxHaltonResolution))
                {
                    scale *= base;
                    ++exp;
                }
                m_baseScales[i]    = scale;
                m_baseExponents[i] = exp;
            }

            m_multInvs[0] = multiplicativeInverse(m_baseScales[1], m_baseScales[0]);
            m_multInvs[1] = multiplicativeInverse(m_baseScales[0], m_baseScales[1]);
        }

        void startPixelSample(Point2i p, int32_t sampleIndex, int32_t dim = 0)
        {
            int32_t const sampleStride = m_baseScales[0] * m_baseScales[1];
            m_haltonIndex              = 0;

            if (sampleStride > 1)
            {
                // reproject pixel coordinates onto halton grid
                Point2i pm{{p[0] % MaxHaltonResolution, p[1] % MaxHaltonResolution}};
                for (int32_t i = 0; i < NumDimensions; ++i)
                {
                    // scaling by base^exponent makes it so that the first exponent digits to the right of the . in the radical inverse
                    // are shifted to the left. Hence, if you scale by base^exponent, the integer part of
                    // the result will be in [0, base^exponent - 1] -> call it x (or y, depends on dimension)
                    // let x_r be radical inverse of the last j digits of x (pixel coord), j exponent in dimension 0
                    // let y_r be radical inverse of the last k digits of y (pixel coord), k exponent in dimension 1
                    // x_r = (haltonIndex mod 2^j), y_r = (haltonIndex mod 3^k), solve for haltonIndex
                    uint64_t const dimOffset = inverseRadicalInverse(pm[i], DimPrimes[i], m_baseExponents[i]);
                    m_haltonIndex += dimOffset * (sampleStride / m_baseScales[i]) * m_multInvs[i];
                }
                m_haltonIndex %= sampleStride;
            }

            m_haltonIndex += sampleIndex * sampleStride;
            m_dimension = std::max(2, dim);
        }

        float get1D()
        {
            if (m_dimension >= NumPrimes)
                m_dimension = 2;
            return sampleDim(m_dimension++, m_haltonIndex);
        }

        Point2f get2D()
        {
            if (m_dimension + 1 >= NumPrimes)
                m_dimension = 2;
            int const dim = m_dimension;
            m_dimension += 2;
            return {{sampleDim(dim, m_haltonIndex), sampleDim(dim + 1, m_haltonIndex)}};
        }

        Point2f getPixel2D()
        {
            return {{radicalInverse(DimPrimeIndices[0], m_haltonIndex >> m_baseExponents[0]),
                     radicalInverse(DimPrimeIndices[1], m_haltonIndex / m_baseScales[1])}};
        }

    private:
        // GCD(a, b) % b
        static inline constexpr uint64_t multiplicativeInverse(int64_t a, int64_t n)
        {
            constexpr auto extendedGCD = [](auto&& f, uint64_t _a, uint64_t _b, int64_t* _x, int64_t* _y) -> void {
                if (_b == 0)
                {
                    *_x = 1;
                    *_y = 0;
                }
                else
                {
                    int64_t d = _a / _b, xp, yp;
                    f(f, _b, _a % _b, &xp, &yp);
                    *_x = yp;
                    *_y = xp - (d * yp);
                }
            };
            int64_t x, y;
            extendedGCD(extendedGCD, a, n, &x, &y);
            return x % n;
        }

        static float sampleDim(int dim, int64_t haltonIndex)
        {
            namespace dn = dmt::numbers;
            return owenScrambledRadicalInverse(dim, haltonIndex, dn::mixBits(1 + dim << 4));
        }

    private:
        static constexpr int32_t                            MaxHaltonResolution = 128;
        static constexpr int32_t                            NumDimensions       = 2; // width, height
        static constexpr std::array<int32_t, NumDimensions> DimPrimes{2, 3};
        static constexpr std::array<int32_t, NumDimensions> DimPrimeIndices{0, 1};

        int64_t m_haltonIndex = 0;
        int32_t m_samplesPerPixel;
        int32_t m_dimension = 0;
        int32_t m_multInvs[NumDimensions];
        int32_t m_baseScales[NumDimensions];
        int32_t m_baseExponents[NumDimensions];
    };

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
        { t.evaluate(std::declval<Point2f>()) } -> std::floating_point;
        { t.radius() } -> std::same_as<Vector2f>;
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
            return {-filter.radius(), filter.radius()};
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

        Bounds2f domain() const { return m_distrib.domain(); }

        FilterSample sample(Point2f u) const
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
        Mitchell(Vector2f                   radius = {{2.f, 2.f}},
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

        FilterSample sample(Point2f u) const { return m_sampler.sample(u); }

        float evaluate(Point2f p) const
        {
            return mitchell1D(2 * p.x / m_radius.x, m_b, m_c) * mitchell1D(2 * p.y / m_radius.y, m_b, m_c);
        }

        // domain, for the x axis and y axis, in which the filter is != 0 starting from 0 (low pass filter)
        Vector2f radius() const { return m_radius; }

        // 2D integral of the filter
        float integral() const { return m_radius.x * m_radius.y / 4; }

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
} // namespace dmt::filtering

namespace dmt {
    void runMainProgram(std::span<TriangleData>     scene,
                        std::span<Primitive const*> primsView,
                        BVHBuildNode*               bvh,
                        unsigned char*              scratchBuffer)
    {
        // primsView primitive coordinates defined in world space
        static constexpr uint32_t Width = 1280, Height = 720, NumChannels = 3;
        static constexpr int32_t  samplesPerPixel = 4;

        std::pmr::synchronized_pool_resource pool;
        ThreadPoolV2                         threadpool{std::thread::hardware_concurrency(), &pool};
        auto                                 image = makeUniqueRef<uint8_t[]>(&pool, Width * Height * NumChannels);
        dmt::sampling::HaltonOwen            sampler{samplesPerPixel, {{Width, Height}}, 123432};

        // define camera (image plane physical dims, resolution given by image)
        Vector3f const cameraPosition{{0.f, -4.f, 2.f}};
        Normal3f const cameraDirection = normalFrom({{0.f, 1.f, 0.3f}});
        float const    focalLength     = 35e-3f; // 35 mm
        // instead of pixelLength, use fovRadians
        float const pixelLength = 13e-6f; // 169 um^2 square pixel

        // for each pixel, for each sample within the pixel (halton + owen scrambling)
        for (Point2i pixel : dmt::ScanlineRange2D({{Width, Height}}))
        {
            for (int32_t sampleIndex = 0; sampleIndex < samplesPerPixel; ++sampleIndex)
            {
                sampler.startPixelSample(pixel, sampleIndex);
                Point2f sample = sampler.getPixel2D();
                // shoot ray, register intersection
            }
        }
    }
} // namespace dmt

int32_t guardedMain()
{
    dmt::Ctx::init();
    class Janitor
    {
    public:
        ~Janitor() { dmt::Ctx::destroy(); }
    } j;

    {
        dmt::Context ctx;
        ctx.log("Hello Cruel World", {});

        // Sample scene
        std::unique_ptr<unsigned char[]> bufferPtr    = std::make_unique<unsigned char[]>(2048);
        auto                             bufferMemory = std::pmr::monotonic_buffer_resource(bufferPtr.get(), 2048);
        auto                             scene        = dmt::ddbg::debugScene();
        dmt::reorderByMorton(scene);
        auto prims     = dmt::makeSinglePrimitivesFromTriangles(scene);
        auto primsView = dmt::ddbg::rawPtrsCopy(prims);

        std::span<dmt::Primitive const*> spanPrims{primsView};
        dmt::test::testBoundsEquality(scene, spanPrims);

        // check that prims bounds equal scene bounds
        auto* rootNode = dmt::bvh::build(spanPrims, &bufferMemory);

        if (ctx.isLogEnabled())
        {
            ctx.log("-- Before Primitive Packing --", {});
            std::string tee = dmt::ddbg::printBVHToString(rootNode);
            ctx.log(std::string_view{tee}, {});
        }

        dmt::test::bvhTestRays(rootNode);

        dmt::bvh::groupTrianglesInBVHLeaves(rootNode, prims, &bufferMemory);

        bufferMemory.release();

        if (ctx.isLogEnabled())
        {
            ctx.log("-- After Primitive Packing --", {});
            std::string tee = dmt::ddbg::printBVHToString(rootNode);
            ctx.log(std::string_view{tee}, {});
        }

        dmt::test::bvhTestRays(rootNode);

        dmt::test::testDistribution1D();
        dmt::test::testDistribution2D();

        dmt::runMainProgram(scene, spanPrims, rootNode, bufferPtr.get());

        dmt::bvh::cleanup(rootNode);

        ctx.log("Goodbye!", {});
    }

    return 0;
}
