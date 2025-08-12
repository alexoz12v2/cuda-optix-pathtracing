#include "utilities.h"

#define DMT_ENTRY_POINT
#include "platform/platform.h"
#include "core/core-cudautils-cpubuild.h"
#include "core/core-bvh-builder.h"
#include "core/core-bsdf.h"
#include "core/core-primitive.h"
#include "core/core-light.h"
#include "core/core-texture.h"

#include <stb_image_write.h>

#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <iomanip>

#define DMT_DBG_PX_X       84
#define DMT_DBG_PX_Y       84
#define DMT_DBG_SAMPLE_IDX 0x70

namespace dmt::ddbg {
    void printTrianglePrimitives(std::span<Primitive const*> bvhPrimitives)
    {
        Context ctx;
        if (!ctx.isValid())
            return;

        auto const printTri = [&ctx](int32_t choice, size_t i, Point3f v0, Point3f v1, Point3f v2, RGB color) {
            std::string_view prefix = choice == 0   ? "Triangle"
                                      : choice == 1 ? "Triangles2"
                                      : choice == 2 ? "Triangles4"
                                                    : "Triangles8";
            // clang-format off
            ctx.log("{} {}: {{ v0: {{{} {} {}}}, v1: {{{} {} {}}}, v2: {{{} {} {}}}, c: {{{} {} {}}} }}",
                    std::make_tuple(prefix, i,
                                    v0.x, v0.y, v0.z,
                                    v1.x, v1.y, v1.z,
                                    v2.x, v2.y, v2.z,
                                    color.r, color.g, color.b));
            // clang-format on
        };

        size_t index = 0;
        for (auto const* prim : bvhPrimitives)
        {
            if (auto const* p = dynamic_cast<Triangle const*>(prim); p)
            {
                auto const [v0, v1, v2, c] = p->tri;
                printTri(0, index++, v0, v1, v2, c);
            }
            else if (auto const* p = dynamic_cast<Triangles2 const*>(prim); p)
            {
                for (int32_t i = 0; i < 2; ++i)
                {
                    Point3f const v0{p->xs[i * 3 + 0], p->ys[i * 3 + 0], p->zs[i * 3 + 0]};
                    Point3f const v1{p->xs[i * 3 + 1], p->ys[i * 3 + 1], p->zs[i * 3 + 1]};
                    Point3f const v2{p->xs[i * 3 + 2], p->ys[i * 3 + 2], p->zs[i * 3 + 2]};
                    RGB const     c{p->colors[i]};
                    printTri(1, index++, v0, v1, v2, c);
                }
            }
            else if (auto const p = dynamic_cast<Triangles4 const*>(prim); p)
            {
                for (int32_t i = 0; i < 4; ++i)
                {
                    Point3f const v0{p->xs[i * 3 + 0], p->ys[i * 3 + 0], p->zs[i * 3 + 0]};
                    Point3f const v1{p->xs[i * 3 + 1], p->ys[i * 3 + 1], p->zs[i * 3 + 1]};
                    Point3f const v2{p->xs[i * 3 + 2], p->ys[i * 3 + 2], p->zs[i * 3 + 2]};
                    RGB const     c{p->colors[i]};
                    printTri(2, index++, v0, v1, v2, c);
                }
            }
            else if (auto const p = dynamic_cast<Triangles8 const*>(prim); p)
            {
                for (int32_t i = 0; i < 8; ++i)
                {
                    Point3f const v0{p->xs[i * 3 + 0], p->ys[i * 3 + 0], p->zs[i * 3 + 0]};
                    Point3f const v1{p->xs[i * 3 + 1], p->ys[i * 3 + 1], p->zs[i * 3 + 1]};
                    Point3f const v2{p->xs[i * 3 + 2], p->ys[i * 3 + 2], p->zs[i * 3 + 2]};
                    RGB const     c{p->colors[i]};
                    printTri(3, index++, v0, v1, v2, c);
                }
            }
        }
    }

    void printBVHToString(BVHBuildNode* node, int depth = 0, std::string const& prefix = "")
    {
        Context ctx;
        if (!ctx.isValid())
            return;

        constexpr auto boundsToString = [](Bounds3f const& b) -> std::string {
            std::ostringstream oss;
            oss << "Bounds[( " << b.pMin.x << ", " << b.pMin.y << ", " << b.pMin.z << " ) - ( " << b.pMax.x << ", "
                << b.pMax.y << ", " << b.pMax.z << " )]";
            return oss.str();
        };

        if (!node)
            return;

        std::string indent(depth * 2, ' ');

        if (node->childCount == 0 && node->primitiveCount > 0)
        {
            ctx.log("{}{}Leaf [count: {} {}]",
                    std::make_tuple(indent, prefix, std::to_string(node->primitiveCount), boundsToString(node->bounds)));
        }
        else
        {
            ctx.log("{}{}Internal [children: {}, {}]",
                    std::make_tuple(indent, prefix, std::to_string(node->childCount), boundsToString(node->bounds)));

            for (uint32_t i = 0; i < node->childCount; ++i)
            {
                bool        isLast      = (i == node->childCount - 1);
                std::string childPrefix = isLast ? "└─ " : "├─ ";
                printBVHToString(node->children[i], depth + 1, childPrefix);
            }
        }
    }

    static RGB hsvToRgb(float h, float s, float v)
    {
        h = std::fmodf(h, 1.0f);
        if (h < 0.0f)
            h += 1.0f;

        float c = v * s;
        float x = c * (1 - std::fabs(std::fmodf(h * 6.0f, 2.0f) - 1));
        float m = v - c;

        float r1, g1, b1;
        // clang-format off
        if      (h < 1.0f / 6.0f) { r1 = c; g1 = x; b1 = 0; }
        else if (h < 2.0f / 6.0f) { r1 = x; g1 = c; b1 = 0; }
        else if (h < 3.0f / 6.0f) { r1 = 0; g1 = c; b1 = x; }
        else if (h < 4.0f / 6.0f) { r1 = 0; g1 = x; b1 = c; }
        else if (h < 5.0f / 6.0f) { r1 = x; g1 = 0; b1 = c; }
        else                      { r1 = c; g1 = 0; b1 = x; }
        // clang-format on

        return {r1 + m, g1 + m, b1 + m};
    }

    RGB makeRGBFromHSVinterp(float h, float s, float vMax, float vMin, uint32_t num, uint32_t index)
    {
        if (num == 0)
            return {0, 0, 0};
        float v = (static_cast<float>(index) / static_cast<float>(num - 1)) * vMax + vMin;
        return hsvToRgb(h, s, v);
    }

    std::vector<TriangleData> makeCubeTriangles(Point3f centerPosition, float size, float zRadians, float hue, float saturation)
    {
        std::vector<TriangleData> tris;

        float const h = size * 0.5f; // half size

        // Cube in local space
        std::array<Point3f, 8> const localVerts =
            {Point3f{{-h, -h, -h}},
             Point3f{{+h, -h, -h}},
             Point3f{{+h, +h, -h}},
             Point3f{{-h, +h, -h}},
             Point3f{{-h, -h, +h}},
             Point3f{{+h, -h, +h}},
             Point3f{{+h, +h, +h}},
             Point3f{{-h, +h, +h}}};

        // Precompute rotation matrix around Z axis
        float const cosZ = std::cos(zRadians);
        float const sinZ = std::sin(zRadians);

        auto const rotateZ = [cosZ, sinZ](Point3f const& p) -> Point3f {
            return Point3f{{cosZ * p.x - sinZ * p.y, sinZ * p.x + cosZ * p.y, p.z}};
        };

        // Local to world transform (rotation + translation)
        auto const toWorld = [rotateZ, centerPosition](Point3f const& p) -> Point3f {
            Point3f rotated = rotateZ(p);
            return Point3f{{rotated.x + centerPosition.x, rotated.y + centerPosition.y, rotated.z + centerPosition.z}};
        };

        // Face indices
        // clang-format off
        std::array<std::array<int, 3>, 12> const indices = {{
            {0, 1, 2}, {2, 3, 0}, // Front (+Y)
            {5, 4, 7}, {7, 6, 5}, // Back (-Y)
            {4, 0, 3}, {3, 7, 4}, // Left (-X)
            {1, 5, 6}, {6, 2, 1}, // Right (+X)
            {3, 2, 6}, {6, 7, 3}, // Top (+Z)
            {4, 5, 1}, {1, 0, 4}, // Bottom (-Z)
        }};
        // clang-format on

        uint32_t i = 0;
        for (auto const& idx : indices)
        {
            tris.emplace_back(toWorld(localVerts[idx[0]]),
                              toWorld(localVerts[idx[1]]),
                              toWorld(localVerts[idx[2]]),
                              makeRGBFromHSVinterp(hue, saturation, 0.85f, 0.03f, 12, i));
            ++i;
        }

        return tris;
    }

    std::vector<TriangleData> makePlaneTriangles(Point3f centerPosition, float size, float hue, float saturation)
    {
        std::vector<TriangleData> tris;
        float const               h = size * 0.5f;

        // On X-Y plane (Z = 0), +Y is forward
        Point3f const v0 = {{centerPosition.x - h, centerPosition.y - h, centerPosition.z}};
        Point3f const v1 = {{centerPosition.x + h, centerPosition.y - h, centerPosition.z}};
        Point3f const v2 = {{centerPosition.x + h, centerPosition.y + h, centerPosition.z}};
        Point3f const v3 = {{centerPosition.x - h, centerPosition.y + h, centerPosition.z}};

        // Triangle 1: v0, v1, v2
        tris.emplace_back(v0, v1, v2, makeRGBFromHSVinterp(hue, saturation, 0.2f, 0.1f, 2, 0));
        // Triangle 2: v2, v3, v0
        tris.emplace_back(v2, v3, v0, makeRGBFromHSVinterp(hue, saturation, 0.2f, 0.1f, 2, 1));

        return tris;
    }

    std::pmr::vector<TriangleData> debugScene(std::pmr::memory_resource* memory = std::pmr::get_default_resource())
    {
        std::pmr::vector<TriangleData> scene{memory};

        auto cube  = makeCubeTriangles({0.f, 1.5f, -0.75f}, 0.5f, fl::pi() * 0.25f, 0.02f, 0.7f);
        auto cube1 = makeCubeTriangles({0.5f, 1.2f, -0.9f}, 0.42f, fl::pi() * 0.7f, 0.5f, 0.7f);
        auto cube2 = makeCubeTriangles({0.5f, 1.7f, -0.9f}, 0.42f, fl::pi() * 0.7f, 0.f, 0.3f);
        auto cube3 = makeCubeTriangles({0.5f, 1.7f, -0.6f}, 0.42f, fl::pi() * 0.7f, 0.7f, 0.3f);
        auto plane = makePlaneTriangles({0.f, 1.f, -1.f}, 3.0f, 0.23f, 0.7f);

        scene.insert(scene.end(), cube.begin(), cube.end());
        scene.insert(scene.end(), plane.begin(), plane.end());
        scene.insert(scene.end(), cube1.begin(), cube1.end());
        scene.insert(scene.end(), cube2.begin(), cube2.end());
        scene.insert(scene.end(), cube3.begin(), cube3.end());

        // triangle in front of camera
        //Point3f const v0{-0.5f, 0.f, -0.5f};
        //Point3f const v1{0.5f, 0, -0.5f};
        //Point3f const v2{0, 0, 0.5f};

        //Point3f const v3{0.9f, 0, 0.5f};

        //scene.emplace_back(v0, v1, v2);
        //scene.emplace_back(v2, v1, v3);

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
                        permutations[index] = dmt::numbers::permutationElement(digitValue, base, dseed);
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
            int32_t const digitValue = dn::permutationElement(num - next * base, base, dn::mixBits(hash ^ reverseDigits));
            reverseDigits = reverseDigits * base + digitValue;
            invBaseM *= invBase;
            ++digitIndex;
            num = next;
        }

        return std::min(invBaseM * reverseDigits, dmt::fl::oneMinusEps());
    }

    template <typename T>
    concept Sampler = requires(std::remove_cvref_t<T>& t) {
        { t.startPixelSample(std::declval<Point2i>(), std::declval<int32_t>(), std::declval<int32_t>()) };
        { t.get1D() } -> std::floating_point;
        { t.get2D() } -> std::same_as<Point2f>;
        { t.getPixel2D() } -> std::same_as<Point2f>;
    } && !std::is_pointer_v<std::remove_cvref_t<T>>;

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
            Point2f p{};
            if (m_dimension + 1 >= NumPrimes)
                m_dimension = 2;
            int const dim = m_dimension;
            m_dimension += 2;

            p.x = sampleDim(dim, m_haltonIndex);
            p.y = sampleDim(dim + 1, m_haltonIndex);
            return p;
        }

        Point2f getPixel2D()
        {
            Point2f p{.5f, .5f};
            p.x = radicalInverse(DimPrimeIndices[0], m_haltonIndex >> m_baseExponents[0]);
            p.y = radicalInverse(DimPrimeIndices[1], m_haltonIndex / m_baseScales[1]);
            assert(p.x <= 1.f && p.x >= 0.f && p.y <= 1.f && p.y >= 0.f && "Out of bounds");
            return p;
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
            float const res = owenScrambledRadicalInverse(dim, haltonIndex, dn::mixBits(1 + static_cast<uint64_t>(dim) << 4));
            assert(res >= 0 && res <= 1.f && "Owen scrabled radical inverse broken");
            return res;
        }

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
    static_assert(Filter<Mitchell>);
} // namespace dmt::filtering

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

    // WARNING: DOESN'T WORK
    Ray generateRay(CameraSample const& cs, Transform const& cameraFromRaster, Transform const& renderFromCamera)
    {
        Point3f const pxImage{{cs.pFilm.x, cs.pFilm.y, 0}};
        Point3f const pCamera{{cameraFromRaster(pxImage)}};
        // TODO add lens?
        // time should use lerp(cs.time, shutterOpen, shutterClose)
        Ray const ray{Point3f{{0, 0, 0}}, normalize(pCamera), cs.time};
        // TODO handle tMax better
        float tMax = 1e5f;
        return renderFromCamera(ray, &tMax);
    }
} // namespace dmt::camera

namespace dmt::film {
    template <typename T>
    concept Film = requires(std::remove_cvref_t<T>& t) {
        { t.addSample(std::declval<Point2i>(), std::declval<RGB>(), 0.f) };
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
        RGBFilm(Point2i                    res,
                float                      maxComp = fl::infinity(),
                std::pmr::memory_resource* memory  = std::pmr::get_default_resource()) :
        m_pixels(makeUniqueRef<Pixel[]>(memory, res.x * res.y)),
        m_resolution(res),
        m_maxComponentValue(maxComp)
        {
            std::memset(m_pixels.get(), 0, sizeof(Pixel) * res.x * res.y);
        }

        void writeImage(os::Path const& imagePath, std::pmr::memory_resource* temp = std::pmr::get_default_resource())
        {
            using std::uint8_t;

            int const   width     = m_resolution.x;
            int const   height    = m_resolution.y;
            int const   numPixels = width * height;
            int const   channels  = 3; // RGB, 24-bit
            float const gamma     = 1.0f / 2.2f;

            // Allocate 8-bit per channel RGB image buffer
            UniqueRef<uint8_t[]> image = makeUniqueRef<uint8_t[]>(temp, numPixels * channels);

            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    int const    idx = x + width * y;
                    Pixel const& px  = m_pixels[idx];

                    float const scale = px.weightSum > 0 ? 1.0f / px.weightSum : 0.0f;

                    // basic gamma correction
                    float const r = std::pow(float(px.rgbSum[0] * scale), gamma);
                    float const g = std::pow(float(px.rgbSum[1] * scale), gamma);
                    float const b = std::pow(float(px.rgbSum[2] * scale), gamma);

                    // Clamp and convert to 8-bit
                    constexpr auto toByte = [](float v) -> uint8_t {
                        return static_cast<uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
                    };

                    image[idx * 3 + 0] = toByte(r);
                    image[idx * 3 + 1] = toByte(g);
                    image[idx * 3 + 2] = toByte(b);
                }
            }

            // Write PNG using stb_image_write
            stbi_write_png(imagePath.toUnderlying(temp).c_str(), // path
                           width,
                           height,          // resolution
                           channels,        // number of channels (RGB)
                           image.get(),     // image buffer
                           width * channels // stride in bytes
            );
        }

        void addSample(Point2i pixel, RGB sample, float weight)
        {
            float const m = [](RGB rgb) {
                if (rgb.r > rgb.g && rgb.r > rgb.b)
                    return rgb.r;
                else if (rgb.g > rgb.r && rgb.g > rgb.b)
                    return rgb.g;
                else
                    return rgb.b;
            }(sample);
            if (m > m_maxComponentValue)
            {
                sample.r *= m_maxComponentValue / m;
                sample.g *= m_maxComponentValue / m;
                sample.b *= m_maxComponentValue / m;
            }
            assert(pixel.x < m_resolution.x && pixel.y < m_resolution.y && "Outside pixel boundaries");
            Pixel& px = m_pixels[pixel.x + m_resolution.x * pixel.y];
            for (int32_t c = 0; c < 3; ++c)
                px.rgbSum[c] += reinterpret_cast<float*>(&sample)[c];

            px.weightSum += weight;
        }

        Point2i resolution() const { return m_resolution; }

    private:
        UniqueRef<Pixel[]> m_pixels;
        Point2i            m_resolution;
        float              m_maxComponentValue;
    };
    static_assert(Film<RGBFilm>);
} // namespace dmt::film

namespace dmt {
    // TODO move elsewhere
    // returns a partially constructed `ApproxDifferentialsContext`, where only `p` and `n` are to change every time
    // IF camera is lensless, ray origin is the same for all differentials,
    ApproxDifferentialsContext minDifferentialsFromCamera(Transform const&     cameraFromRaster,
                                                          Transform const&     renderFromCamera,
                                                          film::RGBFilm const& film,
                                                          uint32_t             samplesPerPixel)
    {
        static constexpr uint32_t CellNum = 512;

        ApproxDifferentialsContext ret{};
        ret.samplesPerPixel     = samplesPerPixel;
        ret.minPosDifferentialX = Vector3f::s(fl::infinity());
        ret.minPosDifferentialY = Vector3f::s(fl::infinity());
        ret.minDirDifferentialX = Vector3f::s(fl::infinity());
        ret.minDirDifferentialY = Vector3f::s(fl::infinity());

        Vector3f const       dxCamera = cameraFromRaster(Point3f{1, 0, 0}) - cameraFromRaster(Point3f{0, 0, 0});
        Vector3f const       dyCamera = cameraFromRaster(Point3f{0, 1, 0}) - cameraFromRaster(Point3f{0, 0, 0});
        camera::CameraSample sample{};
        sample.pLens        = {0.5f, 0.5f};
        sample.time         = 0.5f;
        sample.filterWeight = 1.f;

        for (uint32_t i = 0; i < CellNum; ++i)
        {
            sample.pFilm.x = static_cast<float>(i) / (CellNum - 1) * film.resolution().x;
            sample.pFilm.y = static_cast<float>(i) / (CellNum - 1) * film.resolution().y;

            Ray const      ray      = camera::generateRay(sample, cameraFromRaster, renderFromCamera);
            Point3f const  rxOrigin = ray.o, ryOrigin = ray.o; // TOOD holds only for lensless perspective cameras
            Vector3f const rxDirection = normalize(renderFromCamera(renderFromCamera.applyInverse(ray.d) + dxCamera));
            Vector3f const ryDirection = normalize(renderFromCamera(renderFromCamera.applyInverse(ray.d) + dyCamera));

            Vector3f dox = renderFromCamera.applyInverse(rxOrigin - ray.o);
            Vector3f doy = renderFromCamera.applyInverse(ryOrigin - ray.o);
            if (dotSelf(dox) < dotSelf(ret.minPosDifferentialX))
                ret.minPosDifferentialX = dox;
            if (dotSelf(doy) < dotSelf(ret.minPosDifferentialY))
                ret.minPosDifferentialY = doy;

            Frame const f = Frame::fromZ(ray.d);

            Vector3f const df = normalize(f.toLocal(ray.d)); // should be 0 0 1
            assert(fl::abs(df.z - 1.f) < 1e-5f && fl::nearZero(df.x) && fl::nearZero(df.y));
            Vector3f const dxf = normalize(f.toLocal(rxDirection));
            Vector3f const dyf = normalize(f.toLocal(ryDirection));

            if (dotSelf(dxf - df) < dotSelf(ret.minDirDifferentialX))
                ret.minDirDifferentialX = dxf - df;
            if (dotSelf(dyf - df) < dotSelf(ret.minDirDifferentialY))
                ret.minDirDifferentialY = dyf - df;
        }

        return ret;
    }

    void resetMonotonicBufferPointer(std::pmr::monotonic_buffer_resource& resource, unsigned char* ptr, uint32_t bytes)
    {
        // https://developercommunity.visualstudio.com/t/monotonic_buffer_resourcerelease-does/10624172
        auto* upstream = resource.upstream_resource();
        std::destroy_at(&resource);
        std::construct_at(&resource, ptr, bytes, upstream);
    }

    // TODO proper path tracing
    RGB incidentRadiance(Ray const&                        ray,
                         BVHBuildNode*                     bvh,
                         sampling::HaltonOwen&             sampler,
                         EnvLight const&                   envLight,
                         ApproxDifferentialsContext const& diffCtx,
                         ImageTexturev2 const&             tex,
                         std::pmr::memory_resource*        temp)
    {
        // TODO: Doesn't work
        Context          ctx;
        Intersection     isect;
        Primitive const* prim = bvh::intersectWideBVHBuild(ray, bvh, &isect);
        // TODO: If you are inside an object, you need to accept also negative dot products
        if (prim && dot(-ray.d, isect.ng) > 0)
        {
            assert(fl::abs(normL2(isect.ng) - 1.f) < 1e-5f && "Geometric normal not of unit length");
            // return isect.color;
//#define DMT_TEST_OREN_NAYAR
#define DMT_TEST_TEXTURES
#if defined(DMT_TEST_OREN_NAYAR)
            float            pdf  = 1.f;
            oren_nayar::BRDF bsdf = oren_nayar::makeParams(0.7f, RGB{0.4f, 0.25f, 0.2f}, isect.ng, -ray.d, 100.f);
            Vector3f         wi   = oren_nayar::sample(isect.ng, isect.ng, sampler.get2D(), &pdf);

            float const cosThetaWi = dot(wi, isect.ng);
            if (pdf != 0)
                return oren_nayar::intensity(bsdf, -ray.d, wi) * cosThetaWi / pdf;
#else
            // BRUSHED GOLD
            Vector3f const wo = -ray.d;
            Point2f const  u  = sampler.get2D();
            float const    uc = sampler.get1D();

            RGB const eta = {0.155f, 0.424f, 1.345};
            RGB const k   = {3.911f, 2.345f, 1.770f};

            ggx::BSDF       bsdf   = ggx::makeConductor(wo, isect.ng, isect.ng, 0.2, 0.05, Vector3f::xAxis(), eta, k);
            ggx::BSDFSample sample = ggx::sample(bsdf, wo, isect.ng, u, uc);
            if (sample.pdf != 0)
            {
                float cosThetaWi = dot(sample.wi, isect.ng);
                assert(cosThetaWi > 0.f);
    #if !defined(DMT_TEST_TEXTURES)
                return (sample.f * bsdf.closure.sampleWeight * cosThetaWi / sample.pdf).saturate0();
    #else
                RGB                mult    = (sample.f * bsdf.closure.sampleWeight * cosThetaWi / sample.pdf);
                float              multLum = 0.2627f * mult.r + 0.6870f * mult.g + 0.0593 * mult.b;
                TextureEvalContext texCtx{};
                texCtx.p  = isect.p;
                texCtx.n  = isect.ng; // TODO should be shading normal
                texCtx.uv = isect.uv;

                // compute all differentials we need
                UVDifferentialsContext uvDiffCtx{};
                uvDiffCtx.dpdu = isect.dpdu;
                uvDiffCtx.dpdv = isect.dpdv;

                // for dpdx, dpdy, use approximagtion, which needs a bunch of data
                ApproxDifferentialsContext approxCtx = diffCtx;
                approxCtx.p                          = isect.p;
                approxCtx.n                          = isect.ng; // TODO should be shading normal
                approximate_dp_dxy(approxCtx, &uvDiffCtx.dpdx, &uvDiffCtx.dpdy);

                texCtx.dpdx = uvDiffCtx.dpdx;
                texCtx.dpdy = uvDiffCtx.dpdy;
                texCtx.dUV  = duv_From_dp_dxy(uvDiffCtx);

                if (tex.width == 0)
                {
                    CheckerTexture chtex{};
                    chtex.scaleU = 16.f;
                    chtex.scaleV = 16.f;
                    chtex.color1 = {0.6f, 0.1f, 0.05f};
                    chtex.color2 = {0.8f, 0.8f, 0.75f};

                    return chtex.evalRGB(texCtx) * multLum;
                }
                else
                {
                    return tex.evalRGB(texCtx) * multLum;
                }
    #endif
            }
#endif

            //ctx.error("Miss Me", {});
            return {};
        }

        float pdf = 0.f;
        return envLightEval(envLight, ray.d, &pdf);
    }

    // note: we are using float equality because the numbers should come from copying, no computation needed
    bool checkGrouping(std::span<Primitive const*> primsView, std::span<Primitive const*> primsGrouped)
    {
        if (primsView.size() != primsGrouped.size() * 8)
            return false;

        for (size_t i = 0; i < primsGrouped.size(); ++i)
        {
            auto const* group = dynamic_cast<Triangles8 const*>(primsGrouped[i]);
            if (!group)
                return false;

            for (int j = 0; j < 8; ++j)
            {
                size_t      primIndex = (i << 3) + j;
                auto const* tri       = dynamic_cast<Triangle const*>(primsView[primIndex]);
                if (!tri)
                    return false;

                Point3f v0 = tri->tri.v0;
                Point3f v1 = tri->tri.v1;
                Point3f v2 = tri->tri.v2;

                auto idx = 3 * j;

                // Compare vertices
                if (group->xs[idx + 0] != v0.x || group->ys[idx + 0] != v0.y || group->zs[idx + 0] != v0.z)
                    return false;
                if (group->xs[idx + 1] != v1.x || group->ys[idx + 1] != v1.y || group->zs[idx + 1] != v1.z)
                    return false;
                if (group->xs[idx + 2] != v2.x || group->ys[idx + 2] != v2.y || group->zs[idx + 2] != v2.z)
                    return false;

                // Optional: Compare color (if needed)
                RGB c = tri->tri.color;
                RGB g = group->colors[j];
                if (c.r != g.r || c.g != g.g || c.b != g.b)
                    return false;
            }
        }

        return true;
    }

    bool checkBvhPrimitivesEquality(std::span<Primitive const*> primsView, std::span<Primitive const*> primsGrouped)
    {
        if (primsGrouped.size() > primsView.size())
            return false;

        // assume all Primitives in primsView are Triangle
        UniqueRef<bool[]> checked = makeUniqueRef<bool[]>(std::pmr::get_default_resource(), primsView.size());
        std::fill(checked.get(), checked.get() + primsView.size(), false);

        auto const matchTriangle = [&](Point3f const& v0, Point3f const& v1, Point3f const& v2, RGB const& color) -> bool {
            for (size_t i = 0; i < primsView.size(); ++i)
            {
                if (checked[i])
                    continue;

                auto const* tri = dynamic_cast<Triangle const*>(primsView[i]);
                if (!tri)
                    return false;

                if ((tri->tri.v0.x == v0.x && tri->tri.v0.y == v0.y && tri->tri.v0.z == v0.z) &&
                    (tri->tri.v1.x == v1.x && tri->tri.v1.y == v1.y && tri->tri.v1.z == v1.z) &&
                    (tri->tri.v2.x == v2.x && tri->tri.v2.y == v2.y && tri->tri.v2.z == v2.z) &&
                    (tri->tri.color.r == color.r && tri->tri.color.g == color.g && tri->tri.color.b == color.b))
                {
                    checked[i] = true;
                    return true;
                }
            }
            return false;
        };

        for (Primitive const* group : primsGrouped)
        {
            if (auto const* p = dynamic_cast<Triangle const*>(group); p)
            {
                if (!matchTriangle(p->tri.v0, p->tri.v1, p->tri.v2, p->tri.color))
                    return false;
            }
            else if (auto const* p = dynamic_cast<Triangles2 const*>(group); p)
            {
                for (int i = 0; i < 2; ++i)
                {
                    Point3f const v0{p->xs[i * 3 + 0], p->ys[i * 3 + 0], p->zs[i * 3 + 0]};
                    Point3f const v1{p->xs[i * 3 + 1], p->ys[i * 3 + 1], p->zs[i * 3 + 1]};
                    Point3f const v2{p->xs[i * 3 + 2], p->ys[i * 3 + 2], p->zs[i * 3 + 2]};
                    if (!matchTriangle(v0, v1, v2, p->colors[i]))
                        return false;
                }
            }
            else if (auto const* p = dynamic_cast<Triangles4 const*>(group); p)
            {
                for (int i = 0; i < 4; ++i)
                {
                    Point3f const v0{p->xs[i * 3 + 0], p->ys[i * 3 + 0], p->zs[i * 3 + 0]};
                    Point3f const v1{p->xs[i * 3 + 1], p->ys[i * 3 + 1], p->zs[i * 3 + 1]};
                    Point3f const v2{p->xs[i * 3 + 2], p->ys[i * 3 + 2], p->zs[i * 3 + 2]};
                    if (!matchTriangle(v0, v1, v2, p->colors[i]))
                        return false;
                }
            }
            else if (auto const* p = dynamic_cast<Triangles8 const*>(group); p)
            {
                for (int i = 0; i < 8; ++i)
                {
                    Point3f const v0{p->xs[i * 3 + 0], p->ys[i * 3 + 0], p->zs[i * 3 + 0]};
                    Point3f const v1{p->xs[i * 3 + 1], p->ys[i * 3 + 1], p->zs[i * 3 + 1]};
                    Point3f const v2{p->xs[i * 3 + 2], p->ys[i * 3 + 2], p->zs[i * 3 + 2]};
                    if (!matchTriangle(v0, v1, v2, p->colors[i]))
                        return false;
                }
            }
            else
            {
                return false; // Unknown type
            }
        }

        // All grouped triangles matched, ensure no extras
        for (size_t i = 0; i < primsView.size(); ++i)
        {
            if (!checked[i])
                return false;
        }

        return true;
    }

    void writeIntersectionTestImage(
        std::pmr::monotonic_buffer_resource&  scratch,
        unsigned char*                        scratchBuffer,
        size_t                                ScratchBufferBytes,
        std::pmr::synchronized_pool_resource& pool,
        std::span<UniqueRef<Primitive>>       primitives,
        uint32_t                              Width,
        uint32_t                              Height,
        sampling::HaltonOwen&                 sampler,
        filtering::Mitchell const&            filter,
        Transform const&                      cameraFromRaster,
        Transform const&                      renderFromCamera)
    {
        Context       ctx;
        Point2i const res{static_cast<int32_t>(Width), static_cast<int32_t>(Height)};
        film::RGBFilm film{res, 1e5f, &pool};
        ctx.log("Executing test run with intersection tests against all primitives", {});

        UniqueRef<unsigned char[]> maskImage = makeUniqueRef<unsigned char[]>(&pool, res.x * res.y);

        for (Point2i pixel : ScanlineRange2D(res))
        {
            if (pixel.x % 32 == 0 && pixel.y % 32 == 0)
                ctx.log("Starting Pixel {{ {} {} }}", std::make_tuple(pixel.x, pixel.y));

            if (pixel.x == DMT_DBG_PX_X && pixel.y == DMT_DBG_PX_Y)
                int i = 0;

            for (int32_t sampleIndex = 0; sampleIndex < 1; ++sampleIndex)
            {
                sampler.startPixelSample(pixel, sampleIndex);
                camera::CameraSample /*const*/ cs = camera::getCameraSample(sampler, pixel, filter);
                cs.pFilm.x                        = static_cast<float>(pixel.x) + 0.5f;
                cs.pFilm.y                        = static_cast<float>(pixel.y) + 0.5f;
                cs.filterWeight                   = 1.f;
                Ray const ray{camera::generateRay(cs, cameraFromRaster, renderFromCamera)};

                float        nearest = fl::infinity();
                Intersection isect{};

                for (size_t i = 0; i < primitives.size(); ++i)
                {
                    if (auto si = primitives[i]->intersect(ray, fl::infinity()); si.hit && si.t < nearest)
                    {
                        isect   = si;
                        nearest = si.t;
                    }
                }

                RGB radiance = {};
                if (isect.hit)
                {
                    radiance = dot(-ray.d, isect.ng) > 0.f ? RGB{0, 0, 1} : RGB{1, 0, 0};
                }
                else
                {
                    radiance = RGB{.r = 0.255, .g = 0.102, .b = 0.898};
                }

                maskImage[pixel.x + res.x * pixel.y] = isect.hit ? 255 : 0;

                film.addSample(pixel, radiance, cs.filterWeight);

                resetMonotonicBufferPointer(scratch, scratchBuffer, ScratchBufferBytes);
            }
        }

        os::Path imagePath = os::Path::executableDir(&pool);
        imagePath /= "test.png";
        ctx.log("Writing test image into path \"{}\"", std::make_tuple(imagePath.toUnderlying(&scratch)));
        film.writeImage(imagePath);

        imagePath = os::Path::executableDir(&pool);
        imagePath /= "mask.png";
        ctx.log("Writing test mask into path \"{}\"", std::make_tuple(imagePath.toUnderlying(&scratch)));
        stbi_write_png(imagePath.toUnderlying(&scratch).c_str(), res.x, res.y, 1, maskImage.get(), 0);
    }

    static UniqueRef<RGB[]> background(int32_t*                   xRes,
                                       int32_t*                   yRes,
                                       std::pmr::memory_resource* memory,
                                       std::pmr::memory_resource* temp)
    {
        // TODO better path
        os::Path imagePath = os::Path::executableDir(temp);
        imagePath /= "kloppenheim_02_2k.exr";
        if (imagePath.isValid() && imagePath.isFile())
        {
            openEXR(imagePath, nullptr, xRes, yRes, temp);
            auto image = makeUniqueRef<RGB[]>(memory, static_cast<size_t>(*xRes) * *yRes);
            std::memset(image.get(), 0, static_cast<size_t>(*xRes) * *yRes * sizeof(RGB));

            auto* raw = image.get();
            if (openEXR(imagePath, &raw, xRes, yRes, temp))
            {
                Context ctx;
                ctx.warn("Writing EXR image into PNG to check if there's a y flipping or something", {});
                writePNG(os::Path::executableDir() / "background.png", image.get(), *xRes, *yRes);
                return image;
            }
        }

        *xRes = 4;
        *yRes = 2;

        auto constImage = makeUniqueRef<RGB[]>(memory, 4ull * 2);
        for (int32_t i = 0; i < 4 * 2; ++i)
            constImage[i] = RGB{.r = 0.255, .g = 0.102, .b = 0.898};
        return constImage;
    }

    static ImageTexturev2 openTestTexture()
    {
        ImageTexturev2 tex{};
        Context        ctx;
        assert(ctx.isValid());
        os::Path imageDirectory = os::Path::executableDir();
        imageDirectory /= "tex";
        os::Path const diffuse = imageDirectory / "white_sandstone_bricks_03_diff_4k.exr";
        if (!diffuse.isValid() || !diffuse.isFile())
            return tex;

        auto*   mem  = std::pmr::get_default_resource();
        int32_t xRes = 0;
        int32_t yRes = 0;
        // First probe for resolution
        if (!openEXR(diffuse, nullptr, &xRes, &yRes) || xRes <= 0 || yRes <= 0)
        {
            ctx.error("Couldn't probe EXR file \"{}\"", std::make_tuple(diffuse.toUnderlying()));
            return tex;
        }

        // allocate image and actually load it
        size_t           basePixelCount = size_t(xRes) * size_t(yRes);
        UniqueRef<RGB[]> image          = makeUniqueRef<RGB[]>(mem, basePixelCount);
        if (!image)
        {
            ctx.error("Failed to allocate memory for image.", {});
            return tex;
        }

        RGB* ptr = image.get();
        if (!openEXR(diffuse, &ptr, &xRes, &yRes))
        {
            ctx.error("Couldn't open file at \"{}\"", std::make_tuple(diffuse.toUnderlying()));
            return tex;
        }
        tex = makeRGBMipmappedTexture(image.get(), xRes, yRes, TexWrapMode::eClamp, TexWrapMode::eClamp, TexFormat::FloatRGB, mem);
        return tex;
    }

    static void pixelEval(
        dmt::Point2i                           pixel,
        int32_t const                          SamplesPerPixel,
        dmt::sampling::HaltonOwen&             sampler,
        dmt::filtering::Mitchell const&        filter,
        dmt::Transform const&                  cameraFromRaster,
        dmt::Transform const&                  renderFromCamera,
        dmt::BVHBuildNode*                     bvhRoot,
        dmt::EnvLight const&                   backgroundLight,
        dmt::ApproxDifferentialsContext const& diffCtx,
        dmt::ImageTexturev2 const&             tex,
        dmt::film::RGBFilm&                    film)
    {
        Context ctx;
        UniqueRef<unsigned char[]> monotonicBufferMemory = makeUniqueRef<unsigned char[]>(std::pmr::get_default_resource(), 4096);
        // TODO reduce logging. remove later
        if (pixel.x % 32 == 0 && pixel.y % 32 == 0)
            ctx.log("Starting Pixel {{ {} {} }}", std::make_tuple(pixel.x, pixel.y)); // TODO more samples when filter introduced
        if (pixel.x == DMT_DBG_PX_X && pixel.y == DMT_DBG_PX_Y)
        {
            ctx.warn("Selected Pixel {} {}", std::make_tuple(pixel.x, pixel.y));
            int i = 0;
        }

        for (int32_t sampleIndex = 0; sampleIndex < SamplesPerPixel; ++sampleIndex)
        {
            std::pmr::monotonic_buffer_resource scratch{monotonicBufferMemory.get(), 4096, std::pmr::null_memory_resource()};

            if (sampleIndex == DMT_DBG_SAMPLE_IDX)
                int i = 0;
            sampler.startPixelSample(pixel, sampleIndex);
            camera::CameraSample cs = camera::getCameraSample(sampler, pixel, filter);
#if 1
            cs.pFilm.x      = static_cast<float>(pixel.x) + 0.5f;
            cs.pFilm.y      = static_cast<float>(pixel.y) + 0.5f;
            cs.filterWeight = 1.f;
#endif
            Ray const ray{camera::generateRay(cs, cameraFromRaster, renderFromCamera)};
            RGB const radiance = incidentRadiance(ray, bvhRoot, sampler, backgroundLight, diffCtx, tex, &scratch);
            film.addSample(pixel, radiance, cs.filterWeight);

            if (pixel.x == DMT_DBG_PX_X && pixel.y == DMT_DBG_PX_Y)
            {
                ctx.warn("  radiance sample {}: {} {} {} W m-2 sr-1",
                         std::make_tuple(sampleIndex, radiance.r, radiance.g, radiance.b));
            }
        }
    }

    namespace job {

        struct EvalTileData
        {
            dmt::Point2i                    StartPixel;
            dmt::Point2i                    tileResolution;
            int32_t                         SamplesPerPixel;
            dmt::sampling::HaltonOwen*      samplers;
            dmt::filtering::Mitchell const* filter;
            dmt::Transform const*           cameraFromRaster;
            dmt::Transform const*           renderFromCamera;
            dmt::BVHBuildNode*              bvhRoot;
            dmt::EnvLight const*            backgroundLight;
            dmt::ApproxDifferentialsContext diffCtx;
            dmt::ImageTexturev2 const*      tex;
            dmt::film::RGBFilm*             film;
        };

        static void evalTile(uintptr_t _data, uint32_t tid)
        {
            EvalTileData& data = *std::bit_cast<EvalTileData*>(_data);

            dmt::sampling::HaltonOwen& sampler = data.samplers[tid];

            int32_t xEnd = data.StartPixel.x + data.tileResolution.x;
            int32_t yEnd = data.StartPixel.y + data.tileResolution.y;

            for (int32_t y = data.StartPixel.y; y < yEnd; ++y)
            {
                for (int32_t x = data.StartPixel.x; x < xEnd; ++x)
                {
                    pixelEval({x, y},
                              data.SamplesPerPixel,
                              sampler,
                              *data.filter,
                              *data.cameraFromRaster,
                              *data.renderFromCamera,
                              data.bvhRoot,
                              *data.backgroundLight,
                              data.diffCtx,
                              *data.tex,
                              *data.film);
                }
            }
        }
    } // namespace job

    static void runMainProgram()
    {
        // primsView primitive coordinates defined in world space
        static constexpr uint32_t Width              = 128;
        static constexpr uint32_t Height             = 128;
        static constexpr uint32_t TileWidth          = 32;
        static constexpr uint32_t TileHeight         = 32;
        static constexpr uint32_t NumTileX           = ceilDiv(Width, TileWidth);
        static constexpr uint32_t NumTileY           = ceilDiv(Height, TileHeight);
        static constexpr uint32_t NumJobs            = NumTileX * NumTileY;
        static constexpr uint32_t NumChannels        = 3;
        static constexpr int32_t  SamplesPerPixel    = 1;
        static constexpr uint32_t ScratchBufferBytes = 4096;

        Context ctx;
        assert(ctx.isValid() && "Invalid Context");

        Scene scene;
        { // prepare geometry
            scene.geometry.reserve(16);
            scene.geometry.emplace_back(makeUniqueRef<TriangleMesh>(std::pmr::get_default_resource()));
            auto& cube = *scene.geometry.back();
            TriangleMesh::unitCube(cube);
        }
        { // instance geometry
            {
                scene.instances.emplace_back(makeUniqueRef<Instance>(std::pmr::get_default_resource()));
                auto& cubeInstance   = *scene.instances.back();
                cubeInstance.meshIdx = 0;

                Transform const t = Transform::translate({0.f, 1.5f, -0.75f}) *
                                    Transform::rotate(45.f, Vector3f::zAxis()) * Transform::scale(0.5f);

                extractAffineTransform(t.m, cubeInstance.affineTransform);
                cubeInstance.bounds = scene.geometry[cubeInstance.meshIdx]->transformedBounds(t);
                cubeInstance.color  = ddbg::hsvToRgb(0.3f, 0.7f, 0.6f);
            }
            {
                scene.instances.emplace_back(makeUniqueRef<Instance>(std::pmr::get_default_resource()));
                auto& cubeInstance   = *scene.instances.back();
                cubeInstance.meshIdx = 0;

                Transform const t = Transform::translate({0.5f, 1.2f, -0.9f}) *
                                    Transform::rotate(70.f, Vector3f::zAxis()) * Transform::scale(0.42f);

                extractAffineTransform(t.m, cubeInstance.affineTransform);
                cubeInstance.bounds = scene.geometry[cubeInstance.meshIdx]->transformedBounds(t);
                cubeInstance.color  = ddbg::hsvToRgb(0.5f, 0.7f, 0.6f);
            }
            {
                scene.instances.emplace_back(makeUniqueRef<Instance>(std::pmr::get_default_resource()));
                auto& cubeInstance   = *scene.instances.back();
                cubeInstance.meshIdx = 0;

                Transform const t = Transform::translate({0.5f, 1.7f, -0.6f}) *
                                    Transform::rotate(55.f, normalize(Vector3f{Vector3f::zAxis()} + Vector3f::xAxis())) *
                                    Transform::scale(0.42f);

                extractAffineTransform(t.m, cubeInstance.affineTransform);
                cubeInstance.bounds = scene.geometry[cubeInstance.meshIdx]->transformedBounds(t);
                cubeInstance.color  = ddbg::hsvToRgb(0.7f, 0.3f, 0.6f);
            }
        }

        // memory resources (TODO with our allocators)
        UniqueRef<unsigned char[]> scratchBuffer = makeUniqueRef<unsigned char[]>(std::pmr::get_default_resource(),
                                                                                  ScratchBufferBytes);
        std::pmr::monotonic_buffer_resource scratch{scratchBuffer.get(), ScratchBufferBytes, std::pmr::null_memory_resource()};
        std::pmr::synchronized_pool_resource pool;

        // compute per instance BVH and total BVH
        std::pmr::vector<BVHBuildNode*>        perInstanceBvhNodes{&pool};
        std::pmr::vector<UniqueRef<Primitive>> primitives{&pool};
        perInstanceBvhNodes.reserve(64);
        primitives.reserve(256);

        ctx.warn("Building Per Instance BVHs", {});
        for (size_t instanceIdx = 0; instanceIdx < scene.instances.size(); ++instanceIdx)
        {
            ctx.warn("BVH[{}]", std::make_tuple(instanceIdx));
            perInstanceBvhNodes.push_back(bvh::buildForInstance(scene, instanceIdx, primitives, &scratch, &pool));
            resetMonotonicBufferPointer(scratch, scratchBuffer.get(), ScratchBufferBytes);
            ddbg::printBVHToString(perInstanceBvhNodes.back());
        }

        auto* bvhRoot = reinterpret_cast<BVHBuildNode*>(pool.allocate(sizeof(BVHBuildNode)));
        std::memset(bvhRoot, 0, sizeof(BVHBuildNode));
        bvh::buildCombined(bvhRoot, perInstanceBvhNodes, &scratch, &pool);

        // rendering resources
        ThreadPoolV2        threadpool{std::thread::hardware_concurrency(), &pool};
        film::RGBFilm       film{{{Width, Height}}, 1e5f, &pool};
        filtering::Mitchell filter{{{2.f, 2.f}}, 1.f / 3.f, 1.f / 3.f, &pool, &scratch};
        resetMonotonicBufferPointer(scratch, scratchBuffer.get(), ScratchBufferBytes);

        // define camera (image plane physical dims, resolution given by image)
        Vector3f const cameraPosition{0.f, 0.f, 0.f};
        Normal3f const cameraDirection{0.f, 1.f, -0.5f};
        float const    focalLength  = 20e-3f; // 20 mm
        float const    sensorHeight = 36e-3f; // 36mm
        float const    aspectRatio  = static_cast<float>(Width) / Height;

        Transform const cameraFromRaster = transforms::cameraFromRaster_Perspective(focalLength, sensorHeight, Width, Height);
        Transform const renderFromCamera = transforms::worldFromCamera(cameraDirection, cameraPosition);
        ApproxDifferentialsContext diffCtx = minDifferentialsFromCamera(cameraFromRaster, renderFromCamera, film, SamplesPerPixel);

        ctx.warn(" -- Opening texture --", {});
        auto           start = std::chrono::high_resolution_clock::now();
        ImageTexturev2 tex   = openTestTexture();
        auto           end   = std::chrono::high_resolution_clock::now();

        uint32_t millis = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        ctx.warn(" -- Opened texture in {} ms --", std::make_tuple(millis));

        // TODO: remove, open background image (if possible)
        int32_t  xResBackground = 0, yResBackground = 0;
        auto     backImage = background(&xResBackground, &yResBackground, &pool, &scratch);
        EnvLight backgroundLight{backImage.get(), xResBackground, yResBackground, {0, 0, 0, 1}, 1.f, &pool};

        // TODO: translate the BVH to be in camera-world (render) space, then switch renderFromCamera to use camera-world
        ctx.warn("Printing Global BVH", {});
        ddbg::printBVHToString(bvhRoot);

#define DMT_SINGLE_THREADED
#if defined(DMT_SINGLE_THREADED)
        sampling::HaltonOwen sampler{SamplesPerPixel, {{Width, Height}}, 123432};
        // for each pixel, for each sample within the pixel (halton + owen scrambling)
        for (Point2i pixel : ScanlineRange2D({{Width, Height}}))
        {
            pixelEval(pixel,
                      SamplesPerPixel,
                      sampler,
                      filter,
                      cameraFromRaster,
                      renderFromCamera,
                      bvhRoot,
                      backgroundLight,
                      diffCtx,
                      tex,
                      film);
        }
#else
        UniqueRef<job::EvalTileData[]> jobData = makeUniqueRef<job::EvalTileData[]>(&pool, NumJobs);
        UniqueRef<sampling::HaltonOwen[]> tlsSamplers = makeUniqueRef<sampling::HaltonOwen[]>(&pool, threadpool.numThreads());
        for (uint32_t tidx = 0; tidx < threadpool.numThreads(); ++tidx)
        {
            int seed = 18123 * sinf(32424 * tidx);
            std::construct_at<sampling::HaltonOwen>(&tlsSamplers[tidx], SamplesPerPixel, Point2i{Width, Height}, seed);
        }

        Point2i startPix{0, 0};
        Point2i tileSize{TileWidth, TileHeight};
        for (uint32_t job = 0; job < NumJobs; ++job)
        {
            jobData[job].StartPixel       = startPix;
            jobData[job].tileResolution   = tileSize;
            jobData[job].SamplesPerPixel  = SamplesPerPixel;
            jobData[job].samplers         = tlsSamplers.get();
            jobData[job].filter           = &filter;
            jobData[job].cameraFromRaster = &cameraFromRaster;
            jobData[job].renderFromCamera = &renderFromCamera;
            jobData[job].bvhRoot          = bvhRoot;
            jobData[job].backgroundLight  = &backgroundLight;
            jobData[job].diffCtx          = diffCtx;
            jobData[job].tex              = &tex;
            jobData[job].film             = &film;
            if (job != 0 && job % NumTileX == 0)
            {
                startPix.x = 0;
                startPix.y += TileHeight;
                tileSize.x = Width;
                tileSize.y = fminf(TileHeight, Height - startPix.y);
            }
            else
            {
                startPix.x += TileWidth;
                tileSize.x = fminf(TileWidth, Width - startPix.x);
            }

            threadpool.addJob({job::evalTile, std::bit_cast<uintptr_t>(&jobData[job])}, EJobLayer::ePriority0);
        }
        threadpool.kickJobs();
        threadpool.waitForAll();
        ctx.warn("ALL SHOULD BE FINISHED", {});
#endif

        os::Path imagePath = os::Path::executableDir(&pool);
        imagePath /= "image.png";
        ctx.log("Writing image into path \"{}\"", std::make_tuple(imagePath.toUnderlying(&scratch)));
        film.writeImage(imagePath);

        // writeIntersectionTestImage(scratch, scratchBuffer.get(), ScratchBufferBytes, pool, primitives, Width, Height, sampler, filter, cameraFromRaster, renderFromCamera);
        bvh::cleanup(bvhRoot, &pool);
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

#if 0
        dmt::reorderByMorton(scene);
#else
        ctx.warn("No morton reordering of primitives, still to test", {});
#endif

#if defined(DMT_ENABLE_EXE_TESTS)
        auto primsGrouped     = dmt::makePrimitivesFromTriangles(scene);
        auto primsGroupedView = dmt::ddbg::rawPtrsCopy(primsGrouped);

        auto prims     = dmt::makeSinglePrimitivesFromTriangles(scene);
        auto primsView = dmt::ddbg::rawPtrsCopy(prims);

        std::span<dmt::Primitive const*> spanPrims{primsView};
        dmt::test::testBoundsEquality(scene, spanPrims);

        // check that prims bounds equal scene bounds
        auto* rootNode = dmt::bvh::build(spanPrims, &bufferMemory);

        if (ctx.isLogEnabled())
        {
            ctx.log("-- Before Primitive Packing --", {});
            dmt::ddbg::printBVHToString(rootNode);
        }

        dmt::test::bvhTestRays(rootNode);

        auto bvhPrimitivesBefore = dmt::bvh::extractPrimitivesFromBuild(rootNode);
        if (!dmt::checkBvhPrimitivesEquality(primsView, bvhPrimitivesBefore))
        {
            ctx.error("BVH Primitive equality failed before gruoping", {});
            assert(false && "BVH Primitive equality failed before gruoping");
        }

        dmt::ddbg::printTrianglePrimitives(bvhPrimitivesBefore);

        dmt::bvh::groupTrianglesInBVHLeaves(rootNode, prims, &bufferMemory);
        auto bvhPrimitives = dmt::bvh::extractPrimitivesFromBuild(rootNode);
        if (!dmt::checkBvhPrimitivesEquality(primsView, bvhPrimitives))
        {
            ctx.error("BVH Primitive equality failed after gruoping", {});
            assert(false && "BVH Primitive equality failed after gruoping");
        }

        if (ctx.isLogEnabled())
        {
            ctx.log("-- After Primitive Packing --", {});
            dmt::ddbg::printBVHToString(rootNode);
        }

        dmt::ddbg::printTrianglePrimitives(bvhPrimitives);
        dmt::test::testIndexedTriangleGrouping();

        dmt::resetMonotonicBufferPointer(bufferMemory, bufferPtr.get(), 2048);

        dmt::test::bvhTestRays(rootNode);
        dmt::bvh::cleanup(rootNode);

        dmt::test::testQuaternionRotation();
        dmt::test::testDistribution1D();
        dmt::test::testDistribution2D();
        dmt::test::testOctahedralProj();
        dmt::test::testGGXconductor(4096);
        dmt::test::testEnvironmentalLightConstantValue();
        dmt::test::testSphereLightPDFAnalyticCheck();
        dmt::test::testMipmappedTexturePrinting();
#endif

        dmt::runMainProgram();

        ctx.log("Goodbye!", {});
    }

    return 0;
}
