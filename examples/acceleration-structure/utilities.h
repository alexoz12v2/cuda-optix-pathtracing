#pragma once

#include "core/core-bvh-builder.h"
#include "core/core-cudautils-cpubuild.h"
#include "core/core-math.h"
#include "core/core-primitive.h"
#include "core/core-dstd.h"

#include <memory_resource>

namespace dmt {
    // intersection/math algorithms
    bool DMT_FASTCALL slabTest(Point3f         rayOrigin,
                               Vector3f        rayDirection,
                               Bounds3f const& box,
                               float*          outTmin = nullptr,
                               float*          outTmax = nullptr);

    std::pmr::vector<dmt::UniqueRef<Primitive>> makeSinglePrimitivesFromTriangles(
        std::span<TriangleData const> tris,
        std::pmr::memory_resource*    memory = std::pmr::get_default_resource());

    std::pmr::vector<dmt::UniqueRef<Primitive>> makePrimitivesFromTriangles(
        std::span<TriangleData const> tris,
        std::pmr::memory_resource*    memory = std::pmr::get_default_resource());

    uint32_t morton3D(float x, float y, float z);

    void reorderByMorton(std::span<TriangleData> tris);

    class ScanlineRange2D
    {
    public:
        struct End
        {
        };
        class Iterator
        {
        public:
            using difference_type   = std::ptrdiff_t;
            using value_type        = Point2i;
            using iterator_category = std::forward_iterator_tag;

            Iterator();
            Iterator(Point2i p, Point2i res);

            value_type operator*() const;

            Iterator& operator++();
            Iterator  operator++(int);

            bool operator==(End) const;
            bool operator==(Iterator const& other) const;

        private:
            Point2i m_p;
            Point2i m_res;
        };
        static_assert(std::forward_iterator<Iterator>, "Failed");

    public:
        explicit ScanlineRange2D(Point2i resolution);

        Iterator begin() const;
        End      end() const;

    private:
        Point2i m_resolution;
    };

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
        PiecewiseConstant1D(std::span<float const>     func,
                            float                      min,
                            float                      max,
                            std::pmr::memory_resource* memory = std::pmr::get_default_resource());

        float    integral() const;
        uint32_t size() const;

        /// @warning returns nan if x outside range min, max
        float invert(float x) const;

        float sample(float u, float* pdf = nullptr, int32_t* offset = nullptr) const;

        DMT_FORCEINLINE std::span<float const> absFunc() const { return {m_buffer.get(), m_funcCount}; }
        DMT_FORCEINLINE std::span<float const> CDF() const { return {m_buffer.get() + m_funcCount, m_funcCount}; }
        DMT_FORCEINLINE std::span<float const> absFunc() { return {m_buffer.get(), m_funcCount}; }
        DMT_FORCEINLINE std::span<float const> CDF() { return {m_buffer.get() + m_funcCount, m_funcCount}; }

    private:
        DMT_FORCEINLINE std::span<float> absFunc_() { return {m_buffer.get(), m_funcCount}; }
        DMT_FORCEINLINE std::span<float> CDF_() { return {m_buffer.get() + m_funcCount, m_funcCount}; }

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
        PiecewiseConstant2D(dstd::Array2D<float> const& data,
                            Bounds2f                    domain,
                            std::pmr::memory_resource*  memory = std::pmr::get_default_resource(),
                            std::pmr::memory_resource*  temp   = std::pmr::get_default_resource());

        Bounds2f domain() const;
        Point2i  resolution() const;
        float    integral() const;

        /// @warning returns nan if p outside bounds
        Point2f invert(Point2f p) const;

        Point2f sample(Point2f u, float* pdf = nullptr, Point2i* offset = nullptr) const;

        float pdf(Point2f pr) const;

    private: // assume variables are u and v, v -> marginal, u -> computed through conditional probability
        Bounds2f                              m_domain;
        std::pmr::vector<PiecewiseConstant1D> m_pConditionalV;
        PiecewiseConstant1D                   m_pMarginalV;
    };
} // namespace dmt

namespace dmt::test {
    void bvhTestRays(BVHBuildNode* rootNode);
    void testDistribution1D();
    void testDistribution2D();
} // namespace dmt::test

namespace dmt::bvh {
    BVHBuildNode* traverseBVHBuild(Ray                        ray,
                                   BVHBuildNode*              bvh,
                                   std::pmr::memory_resource* memory = std::pmr::get_default_resource());
}

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

    uint16_t permutationElement(int32_t i, int32_t j, uint32_t l, uint64_t p);
} // namespace dmt::numbers
