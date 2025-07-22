
#define DMT_ENTRY_POINT
#include "platform/platform.h"
#define DMT_CUDAUTILS_IMPL
#include "cudautils/cudautils-vecmath.h"

#include <immintrin.h>

#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <iomanip>
#include <deque>
#include <stack>
#include "platform-launch.h"

namespace dstd {
    template <std::copyable T, typename A, typename UnaryPredicate>
        requires(std::is_invocable_r_v<bool, UnaryPredicate, T &&>)
    std::optional<T> move_to_back_and_pop_if(std::vector<T, A>& vec, UnaryPredicate pred)
    {
        auto it = std::find_if(vec.begin(), vec.end(), pred);
        if (it != vec.end())
        {
            if (it != vec.end() - 1)
            {
                std::iter_swap(it, vec.end() - 1); // move found element to back
            }
            T res = vec.back();
            vec.pop_back(); // remove it
            return res;
        }
        return std::nullopt; // no match
    }

    template <typename In, typename Out, typename Func>
    class transform_span
    {
    public:
        using value_type = Out;
        using size_type  = std::size_t;

        transform_span(In* data, size_type size, Func func) : _data(data), _size(size), _func(std::move(func)) {}

        value_type operator[](size_type i) const { return _func(_data[i]); }

        size_type size() const noexcept { return _size; }

        bool empty() const noexcept { return _size == 0; }

        In* data() const noexcept { return _data; }

    private:
        In*       _data;
        size_type _size;
        Func      _func;
    };
} // namespace dstd

namespace dmt {
    inline constexpr int LogBranchingFactor    = 3;
    inline constexpr int BranchingFactor       = 1 << LogBranchingFactor;
    inline constexpr int LeavesBranchingFactor = 1 << LogBranchingFactor; // TODO better
    inline constexpr int TrianglesPerLeaf      = 4;
    inline constexpr int maxPrimsInNode        = 8;

    inline float hmin_ps(__m128 v)
    {
        __m128 shuf = _mm_movehdup_ps(v); // (v1,v1,v3,v3)
        __m128 mins = _mm_min_ps(v, shuf);
        shuf        = _mm_movehl_ps(shuf, mins); // (v2,v3)
        mins        = _mm_min_ss(mins, shuf);
        return _mm_cvtss_f32(mins);
    }

    inline float hmax_ps(__m128 v)
    {
        __m128 shuf = _mm_movehdup_ps(v);
        __m128 maxs = _mm_max_ps(v, shuf);
        shuf        = _mm_movehl_ps(shuf, maxs);
        maxs        = _mm_max_ss(maxs, shuf);
        return _mm_cvtss_f32(maxs);
    }

    inline float hmin_ps(__m256 v)
    {
        __m128 low  = _mm256_castps256_ps128(v);   // lower 128
        __m128 high = _mm256_extractf128_ps(v, 1); // upper 128
        __m128 min1 = _mm_min_ps(low, high);
        __m128 shuf = _mm_movehdup_ps(min1);
        __m128 min2 = _mm_min_ps(min1, shuf);
        shuf        = _mm_movehl_ps(shuf, min2);
        min2        = _mm_min_ss(min2, shuf);
        return _mm_cvtss_f32(min2);
    }

    inline float hmax_ps(__m256 v)
    {
        __m128 low  = _mm256_castps256_ps128(v);
        __m128 high = _mm256_extractf128_ps(v, 1);
        __m128 max1 = _mm_max_ps(low, high);
        __m128 shuf = _mm_movehdup_ps(max1);
        __m128 max2 = _mm_max_ps(max1, shuf);
        shuf        = _mm_movehl_ps(shuf, max2);
        max2        = _mm_max_ss(max2, shuf);
        return _mm_cvtss_f32(max2);
    }

    class DMT_INTERFACE Primitive
    {
    public:
        virtual ~Primitive() {};

        virtual Bounds3f bounds() const = 0;
    };

    struct TriangleData
    {
        Point3f v0, v1, v2;
    };

    // TODO: Indexed variants
    class Triangle : public Primitive
    {
    public:
        TriangleData tri;
        Bounds3f     bounds() const override
        {
            Point3f p0{{tri.v0.x, tri.v0.y, tri.v0.z}};
            Point3f p1{{tri.v1.x, tri.v1.y, tri.v1.z}};
            Point3f p2{{tri.v2.x, tri.v2.y, tri.v2.z}};
            return bbUnion(bbUnion(Bounds3f{p0, p0}, Bounds3f{p1, p1}), Bounds3f{p2, p2});
        }
    };

    class Triangles2 : public Primitive
    {
    public:
        static constexpr int32_t numTriangles = 2;
        float                    xs[3 * numTriangles], ys[3 * numTriangles], zs[3 * numTriangles];
        Bounds3f                 bounds() const override
        {
            // Load all 6 floats (3 vertices * 2 triangles) for x, y, z
            __m128 x0 = _mm_loadu_ps(xs);     // xs[0..3]
            __m128 x1 = _mm_loadu_ps(xs + 2); // xs[2..5] overlaps last part of x0

            __m128 y0 = _mm_loadu_ps(ys);
            __m128 y1 = _mm_loadu_ps(ys + 2);

            __m128 z0 = _mm_loadu_ps(zs);
            __m128 z1 = _mm_loadu_ps(zs + 2);

            __m128 xMin = _mm_min_ps(x0, x1);
            __m128 xMax = _mm_max_ps(x0, x1);

            __m128 yMin = _mm_min_ps(y0, y1);
            __m128 yMax = _mm_max_ps(y0, y1);

            __m128 zMin = _mm_min_ps(z0, z1);
            __m128 zMax = _mm_max_ps(z0, z1);

            float x_min = hmin_ps(xMin);
            float x_max = hmax_ps(xMax);
            float y_min = hmin_ps(yMin);
            float y_max = hmax_ps(yMax);
            float z_min = hmin_ps(zMin);
            float z_max = hmax_ps(zMax);

            return Bounds3f{Point3f{{x_min, y_min, z_min}}, Point3f{{x_max, y_max, z_max}}};
        }
    };

    class Triangles4 : public Primitive
    {
    public:
        static constexpr int32_t numTriangles = 4;
        float                    xs[3 * numTriangles], ys[3 * numTriangles], zs[3 * numTriangles];
        Bounds3f                 bounds() const override
        {
            // Load all 12 floats (3 vertices * 4 triangles) for x, y, z
            __m128 x0 = _mm_loadu_ps(xs);     // xs[0..3]
            __m128 x1 = _mm_loadu_ps(xs + 4); // xs[4..7]
            __m128 x2 = _mm_loadu_ps(xs + 8); // xs[8..11]

            __m128 y0 = _mm_loadu_ps(ys);
            __m128 y1 = _mm_loadu_ps(ys + 4);
            __m128 y2 = _mm_loadu_ps(ys + 8);

            __m128 z0 = _mm_loadu_ps(zs);
            __m128 z1 = _mm_loadu_ps(zs + 4);
            __m128 z2 = _mm_loadu_ps(zs + 8);

            __m128 xMin = _mm_min_ps(_mm_min_ps(x0, x1), x2);
            __m128 xMax = _mm_max_ps(_mm_max_ps(x0, x1), x2);

            __m128 yMin = _mm_min_ps(_mm_min_ps(y0, y1), y2);
            __m128 yMax = _mm_max_ps(_mm_max_ps(y0, y1), y2);

            __m128 zMin = _mm_min_ps(_mm_min_ps(z0, z1), z2);
            __m128 zMax = _mm_max_ps(_mm_max_ps(z0, z1), z2);

            float x_min = hmin_ps(xMin);
            float x_max = hmax_ps(xMax);
            float y_min = hmin_ps(yMin);
            float y_max = hmax_ps(yMax);
            float z_min = hmin_ps(zMin);
            float z_max = hmax_ps(zMax);

            return Bounds3f{Point3f{{x_min, y_min, z_min}}, Point3f{{x_max, y_max, z_max}}};
        }
    };

    class Triangles8 : public Primitive
    {
    public:
        static constexpr int32_t numTriangles = 8;

        Bounds3f bounds() const override
        {
            // Load all 24 floats (3 * 8) for x, y, z
            __m256 x0 = _mm256_loadu_ps(xs);      // xs[0]..xs[7]
            __m256 x1 = _mm256_loadu_ps(xs + 8);  // xs[8]..xs[15]
            __m256 x2 = _mm256_loadu_ps(xs + 16); // xs[16]..xs[23]

            __m256 y0 = _mm256_loadu_ps(ys);
            __m256 y1 = _mm256_loadu_ps(ys + 8);
            __m256 y2 = _mm256_loadu_ps(ys + 16);

            __m256 z0 = _mm256_loadu_ps(zs);
            __m256 z1 = _mm256_loadu_ps(zs + 8);
            __m256 z2 = _mm256_loadu_ps(zs + 16);

            // Reduce min/max across the 24 values using AVX2
            __m256 xMin = _mm256_min_ps(_mm256_min_ps(x0, x1), x2);
            __m256 xMax = _mm256_max_ps(_mm256_max_ps(x0, x1), x2);

            __m256 yMin = _mm256_min_ps(_mm256_min_ps(y0, y1), y2);
            __m256 yMax = _mm256_max_ps(_mm256_max_ps(y0, y1), y2);

            __m256 zMin = _mm256_min_ps(_mm256_min_ps(z0, z1), z2);
            __m256 zMax = _mm256_max_ps(_mm256_max_ps(z0, z1), z2);

            // Final horizontal reduction to scalars
            float x_min = hmin_ps(xMin);
            float x_max = hmax_ps(xMax);

            float y_min = hmin_ps(yMin);
            float y_max = hmax_ps(yMax);

            float z_min = hmin_ps(zMin);
            float z_max = hmax_ps(zMax);

            return Bounds3f{Point3f{{x_min, y_min, z_min}}, Point3f{{x_max, y_max, z_max}}};
        }

        // x_triangle0_vertex0 | x_triangle0_vertex1 | x_triangle0_vertex2 | x_triangle1_vertex0 | ...
        float xs[3 * numTriangles];
        float ys[3 * numTriangles];
        float zs[3 * numTriangles];
    };

    std::pmr::vector<dmt::UniqueRef<Primitive>> makeSinglePrimitivesFromTriangles(
        std::span<TriangleData const> tris,
        std::pmr::memory_resource*    memory = std::pmr::get_default_resource())
    {
        std::pmr::vector<dmt::UniqueRef<Primitive>> out(memory);
        for (uint64_t i = 0; i < tris.size(); ++i)
        {
            Triangle group{};
            group.tri = tris[i];
            out.push_back(dmt::makeUniqueRef<Triangle>(memory, std::move(group)));
        }

        return out;
    }

    std::pmr::vector<dmt::UniqueRef<Primitive>> makePrimitivesFromTriangles(
        std::span<TriangleData const> tris,
        std::pmr::memory_resource*    memory = std::pmr::get_default_resource())
    {
        std::pmr::vector<dmt::UniqueRef<Primitive>> out(memory);
        size_t                                      i = 0;

        // Pass 1: Triangles8
        for (; i + 8 <= tris.size(); i += 8)
        {
            Triangles8 group{};
            for (int j = 0; j < 8; ++j)
            {
                group.xs[3 * j + 0] = tris[i + j].v0.x;
                group.xs[3 * j + 1] = tris[i + j].v1.x;
                group.xs[3 * j + 2] = tris[i + j].v2.x;

                group.ys[3 * j + 0] = tris[i + j].v0.y;
                group.ys[3 * j + 1] = tris[i + j].v1.y;
                group.ys[3 * j + 2] = tris[i + j].v2.y;

                group.zs[3 * j + 0] = tris[i + j].v0.z;
                group.zs[3 * j + 1] = tris[i + j].v1.z;
                group.zs[3 * j + 2] = tris[i + j].v2.z;
            }
            out.push_back(dmt::makeUniqueRef<Triangles8>(memory, std::move(group)));
        }

        // Pass 2: Triangles4
        for (; i + 4 <= tris.size(); i += 4)
        {
            Triangles4 group{};
            for (int j = 0; j < 4; ++j)
            {
                group.xs[3 * j + 0] = tris[i + j].v0.x;
                group.xs[3 * j + 1] = tris[i + j].v1.x;
                group.xs[3 * j + 2] = tris[i + j].v2.x;

                group.ys[3 * j + 0] = tris[i + j].v0.y;
                group.ys[3 * j + 1] = tris[i + j].v1.y;
                group.ys[3 * j + 2] = tris[i + j].v2.y;

                group.zs[3 * j + 0] = tris[i + j].v0.z;
                group.zs[3 * j + 1] = tris[i + j].v1.z;
                group.zs[3 * j + 2] = tris[i + j].v2.z;
            }
            out.push_back(dmt::makeUniqueRef<Triangles4>(memory, std::move(group)));
        }

        // Pass 3: Triangles2
        for (; i + 2 <= tris.size(); i += 2)
        {
            Triangles2 group{};
            for (int j = 0; j < 2; ++j)
            {
                group.xs[3 * j + 0] = tris[i + j].v0.x;
                group.xs[3 * j + 1] = tris[i + j].v1.x;
                group.xs[3 * j + 2] = tris[i + j].v2.x;

                group.ys[3 * j + 0] = tris[i + j].v0.y;
                group.ys[3 * j + 1] = tris[i + j].v1.y;
                group.ys[3 * j + 2] = tris[i + j].v2.y;

                group.zs[3 * j + 0] = tris[i + j].v0.z;
                group.zs[3 * j + 1] = tris[i + j].v1.z;
                group.zs[3 * j + 2] = tris[i + j].v2.z;
            }
            out.push_back(dmt::makeUniqueRef<Triangles2>(memory, std::move(group)));
        }

        // Pass 4: Individual Triangle
        for (; i < tris.size(); ++i)
        {
            Triangle group{};
            group.tri = tris[i];
            out.push_back(dmt::makeUniqueRef<Triangle>(memory, std::move(group)));
        }

        return out;
    }

    struct BVHBuildNode
    {
        Bounds3f         bounds;
        Primitive const* primitives[LeavesBranchingFactor];
        BVHBuildNode*    children[BranchingFactor];
        uint32_t         childCount;
        uint32_t         primitiveCount;
    };
    static_assert(std::is_trivial_v<BVHBuildNode> && std::is_standard_layout_v<BVHBuildNode>,
                  "needed to use aggregate init and memset/memcpy");

    namespace bvh {
        DMT_FORCEINLINE float evaluateSAH(std::span<Primitive const*> nodePrims, int32_t axis, float splitPos)
        {
            Bounds3f leftBounds = bbEmpty(), rightBounds = bbEmpty();
            int32_t  leftCount = 0, rightCount = 0;

            for (auto const* prim : nodePrims)
            {
                Bounds3f const b = prim->bounds();
                if (b.centroid()[axis] < splitPos)
                {
                    ++leftCount;
                    leftBounds = bbUnion(leftBounds, b);
                }
                else
                {
                    ++rightCount;
                    rightBounds = bbUnion(rightBounds, b);
                }
            }

            float const cost = leftCount * leftBounds.surfaceArea() + rightCount * rightBounds.surfaceArea();
            return cost > 0.f ? cost : 1e30f;
        }

        BVHBuildNode* build(std::span<Primitive const*> const prims,
                            std::pmr::memory_resource* temp, // allocate only stuff which doesn't need destruction!
                            std::pmr::memory_resource* memory = std::pmr::get_default_resource())
        {
            std::pmr::vector<Primitive const*> shufflingPrims{temp};
            using Iter = decltype(shufflingPrims)::iterator;
            shufflingPrims.reserve(nextPOT(prims.size()));
            std::copy(prims.begin(), prims.end(), std::back_inserter(shufflingPrims));

            constexpr auto func =
                [](auto&&                     _f,
                   Iter                       _primsBeg,
                   Iter                       _primsEnd,
                   BVHBuildNode*              _parent,
                   std::pmr::memory_resource* _memory,
                   std::pmr::memory_resource* _temp) -> void {
                constexpr int32_t NumChildren    = 8;
                constexpr int32_t MinNumBin      = 16;
                constexpr int32_t MaxNumBin      = 128;
                constexpr float   BinScaleFactor = 2.f;

                assert(std::distance(_primsBeg, _primsEnd) >= 0 && "negative primitive range in BVH construction");
                if (auto dist = std::distance(_primsBeg, _primsEnd); dist < LeavesBranchingFactor && dist >= 0)
                {
                    assert(_parent->childCount == 0 && "Leaf BVH Node shouldn't have children");
                    _parent->primitiveCount = std::distance(_primsBeg, _primsEnd);
                    std::copy(_primsBeg, _primsEnd, _parent->primitives);
                }
                else
                {
                    struct WorkingStackItem
                    {
                        BVHBuildNode node;
                        Iter         beg, end;
                    };
                    std::pmr::vector<WorkingStackItem> childCandidates{_temp};
                    childCandidates.reserve(NumChildren);
                    childCandidates.emplace_back();
                    std::memset(&childCandidates.back().node, 0, sizeof(BVHBuildNode));
                    childCandidates.back()
                        .node.bounds = std::transform_reduce(_primsBeg, _primsEnd, bbEmpty(), [](Bounds3f a, Bounds3f b) {
                        return bbUnion(a, b);
                    }, [](Primitive const* prim) { return prim->bounds(); });
                    childCandidates.back().beg = _primsBeg;
                    childCandidates.back().end = _primsEnd;

                    bool shouldContinue = true;
                    while (childCandidates.size() < NumChildren && shouldContinue)
                    {
                        auto maybeNode = dstd::move_to_back_and_pop_if(childCandidates, [](WorkingStackItem const& wItem) {
                            return std::distance(wItem.beg, wItem.end) >= LeavesBranchingFactor;
                        });

                        if (!maybeNode)
                            shouldContinue = false;
                        else
                        {
                            auto          current              = maybeNode->node;
                            auto          primsBeg             = maybeNode->beg;
                            auto          primsEnd             = maybeNode->end;
                            int32_t const axis                 = current.bounds.maxDimention(); // common estimate
                            float const   extent               = current.bounds.pMax[axis] - current.bounds.pMin[axis];
                            int32_t const numSplits            = std::clamp(static_cast<int>(
                                                                     extent * BinScaleFactor *
                                                                     std::log2(std::distance(primsBeg, primsEnd) + 1)),
                                                                 MinNumBin,
                                                                 MaxNumBin);
                            float const   splitLength          = extent / numSplits;
                            float         minimumSplitCost     = std::numeric_limits<float>::infinity();
                            float         minimumSplitPosition = 0.f;
                            Iter          primsMid             = primsBeg;

                            // check for degenerate bounding box (flat in one dimension)
                            constexpr float DegenerateEpsilon = 1e-5f;

                            Bounds3f const& bounds = current.bounds;
                            Vector3f const  diag   = bounds.pMax - bounds.pMin;
                            bool const degenerate  = (diag[0] < DegenerateEpsilon) || (diag[1] < DegenerateEpsilon) ||
                                                    (diag[2] < DegenerateEpsilon);

                            if (!degenerate)
                            {
                                for (uint64_t i = 0; i < numSplits - 1; ++i)
                                {
                                    float const splitPosition = current.bounds.pMin[axis] + (i + 1) * splitLength;
                                    float const splitCost     = evaluateSAH(std::span{primsBeg, primsEnd},
                                                                        axis,
                                                                        splitPosition); // TODO refactor
                                    if (splitCost < minimumSplitCost)
                                    {
                                        minimumSplitCost     = splitCost;
                                        minimumSplitPosition = splitPosition;
                                    }
                                }

                                primsMid = std::partition(primsBeg,
                                                          primsEnd,
                                                          [axis, mid = minimumSplitPosition](Primitive const* p) {
                                    return p->bounds().centroid()[axis] < mid;
                                });
                            }

                            if (primsMid == primsBeg || primsMid == primsEnd)
                            {
                                // Binning failed — fallback to split by median centroid
                                // find best axis (maxDim may not be optimal one)
                                int  bestAxis           = axis;
                                bool validCentroidSplit = false;
                                for (int i = 0; i < 3; ++i)
                                {
                                    int  tryAxis       = (axis + i) % 3;
                                    auto firstCentroid = (*primsBeg)->bounds().centroid()[tryAxis];
                                    validCentroidSplit = std::any_of(primsBeg + 1, primsEnd, [&](auto* p) {
                                        return std::abs(p->bounds().centroid()[tryAxis] - firstCentroid) > 1e-5f;
                                    });

                                    if (validCentroidSplit)
                                    {
                                        bestAxis = tryAxis;
                                        break;
                                    }
                                }

                                // 1. Collect centroid values
                                std::pmr::vector<std::pair<float, Primitive const*>> centroids{_temp};
                                centroids.reserve(nextPOT(static_cast<uint64_t>(std::distance(primsBeg, primsEnd))));
                                for (auto it = primsBeg; it != primsEnd; ++it)
                                {
                                    centroids.emplace_back((*it)->bounds().centroid()[bestAxis], *it);
                                }

                                // 2. Sort and compute median
                                std::sort(centroids.begin(), centroids.end(), [](auto const& a, auto const& b) {
                                    return a.first < b.first;
                                });

                                size_t midIdx = centroids.size() / 2;

                                primsMid = primsBeg + midIdx;
                            }

                            // last resort tie breaker: Index median split
                            if (primsMid == primsBeg || primsMid == primsEnd)
                            {
                                auto count = std::distance(primsBeg, primsEnd);
                                primsMid   = primsBeg + count / 2;
                            }

                            childCandidates.emplace_back(); // after each iter, size + 1
                            std::memset(&childCandidates.back().node, 0, sizeof(BVHBuildNode));
                            childCandidates.emplace_back();
                            std::memset(&childCandidates.back().node, 0, sizeof(BVHBuildNode));
                            auto last  = childCandidates.rbegin();
                            auto first = childCandidates.rbegin() + 1;

                            // Sort so first always points to the larger half
                            if (std::distance(primsBeg, primsMid) > std::distance(primsMid, primsEnd))
                                std::swap(first, last);

                            first->beg = primsMid;
                            first->end = primsEnd;
                            first->node.bounds = std::transform_reduce(primsMid, primsEnd, bbEmpty(), [](Bounds3f a, Bounds3f b) {
                                return bbUnion(a, b);
                            }, [](Primitive const* prim) { return prim->bounds(); });

                            last->beg = primsBeg;
                            last->end = primsMid;
                            last->node.bounds = std::transform_reduce(primsBeg, primsMid, bbEmpty(), [](Bounds3f a, Bounds3f b) {
                                return bbUnion(a, b);
                            }, [](Primitive const* prim) { return prim->bounds(); });
                        } // end else (maybeNode)
                    } // end while (on workItem list)

                    for (auto const& workItem : childCandidates)
                    {
                        auto* node = reinterpret_cast<BVHBuildNode*>(_memory->allocate(sizeof(BVHBuildNode)));
                        std::memset(node, 0, sizeof(BVHBuildNode));
                        node->bounds = workItem.node.bounds;
                        assert(_parent->childCount < BranchingFactor - 1 && "Too many children for BVH Node");
                        _parent->children[_parent->childCount++] = node;
                        _f(_f, workItem.beg, workItem.end, node, _memory, _temp);
                    }
                }
            };

            auto* root = reinterpret_cast<BVHBuildNode*>(memory->allocate(sizeof(BVHBuildNode)));
            std::memset(root, 0, sizeof(BVHBuildNode));
            root->bounds = std::transform_reduce(shufflingPrims.begin(), shufflingPrims.end(), bbEmpty(), [](Bounds3f a, Bounds3f b) {
                return bbUnion(a, b);
            }, [](Primitive const* p) { return p->bounds(); });
            func(func, shufflingPrims.begin(), shufflingPrims.end(), root, memory, temp);
            return root;
        }

        void cleanup(BVHBuildNode* node, std::pmr::memory_resource* memory = std::pmr::get_default_resource())
        {
            if (!node)
                return;

            // Recursively clean up children
            for (uint32_t i = 0; i < node->childCount; ++i)
                cleanup(node->children[i], memory);

            // Deallocate this node
            memory->deallocate(node, sizeof(BVHBuildNode), alignof(BVHBuildNode));
        }

        std::size_t groupTrianglesInBVHLeaves(BVHBuildNode*                                node,
                                              std::pmr::vector<dmt::UniqueRef<Primitive>>& out,
                                              std::pmr::memory_resource*                   temp,
                                              std::pmr::memory_resource* memory = std::pmr::get_default_resource())
        {
            using dmt::makeUniqueRef;

            std::size_t outputStartIndex = out.size();

            std::function<void(BVHBuildNode*)> traverse;
            traverse = [&](BVHBuildNode* current) {
                if (current->childCount == 0)
                {
                    std::pmr::vector<Triangle> triangles{temp};

                    // Save the output starting index for this leaf
                    std::size_t leafStart = out.size();

                    for (uint64_t i = 0; i < current->primitiveCount; ++i)
                    {
                        Primitive const* p = current->primitives[i];
                        if (auto t = dynamic_cast<Triangle const*>(p))
                            triangles.push_back(*t);
                        else
                            out.push_back(dmt::UniqueRef<Primitive>(const_cast<Primitive*>(p)));
                    }

                    // Group triangles
                    std::size_t i = 0;
                    while (i + 8 <= triangles.size())
                    {
                        Triangles8 group{};
                        for (int j = 0; j < 8; ++j)
                        {
                            group.xs[3 * j + 0] = triangles[i + j].tri.v0.x;
                            group.xs[3 * j + 1] = triangles[i + j].tri.v1.x;
                            group.xs[3 * j + 2] = triangles[i + j].tri.v2.x;

                            group.ys[3 * j + 0] = triangles[i + j].tri.v0.y;
                            group.ys[3 * j + 1] = triangles[i + j].tri.v1.y;
                            group.ys[3 * j + 2] = triangles[i + j].tri.v2.y;

                            group.zs[3 * j + 0] = triangles[i + j].tri.v0.z;
                            group.zs[3 * j + 1] = triangles[i + j].tri.v1.z;
                            group.zs[3 * j + 2] = triangles[i + j].tri.v2.z;
                        }
                        out.push_back(makeUniqueRef<Triangles8>(memory, std::move(group)));
                        i += 8;
                    }

                    if (i + 4 <= triangles.size())
                    {
                        Triangles4 group{};
                        for (int j = 0; j < 4; ++j)
                        {
                            group.xs[3 * j + 0] = triangles[i + j].tri.v0.x;
                            group.xs[3 * j + 1] = triangles[i + j].tri.v1.x;
                            group.xs[3 * j + 2] = triangles[i + j].tri.v2.x;

                            group.ys[3 * j + 0] = triangles[i + j].tri.v0.y;
                            group.ys[3 * j + 1] = triangles[i + j].tri.v1.y;
                            group.ys[3 * j + 2] = triangles[i + j].tri.v2.y;

                            group.zs[3 * j + 0] = triangles[i + j].tri.v0.z;
                            group.zs[3 * j + 1] = triangles[i + j].tri.v1.z;
                            group.zs[3 * j + 2] = triangles[i + j].tri.v2.z;
                        }
                        out.push_back(makeUniqueRef<Triangles4>(memory, std::move(group)));
                        i += 4;
                    }

                    if (i + 2 <= triangles.size())
                    {
                        Triangles2 group{};
                        for (int j = 0; j < 2; ++j)
                        {
                            group.xs[3 * j + 0] = triangles[i + j].tri.v0.x;
                            group.xs[3 * j + 1] = triangles[i + j].tri.v1.x;
                            group.xs[3 * j + 2] = triangles[i + j].tri.v2.x;

                            group.ys[3 * j + 0] = triangles[i + j].tri.v0.y;
                            group.ys[3 * j + 1] = triangles[i + j].tri.v1.y;
                            group.ys[3 * j + 2] = triangles[i + j].tri.v2.y;

                            group.zs[3 * j + 0] = triangles[i + j].tri.v0.z;
                            group.zs[3 * j + 1] = triangles[i + j].tri.v1.z;
                            group.zs[3 * j + 2] = triangles[i + j].tri.v2.z;
                        }
                        out.push_back(makeUniqueRef<Triangles2>(memory, std::move(group)));
                        i += 2;
                    }

                    for (; i < triangles.size(); ++i)
                    {
                        Triangle t = triangles[i];
                        out.push_back(makeUniqueRef<Triangle>(memory, std::move(t)));
                    }

                    // Update the current leaf's primitive list to point to the new ones
                    std::memset(current->primitives, 0, current->primitiveCount * sizeof(uintptr_t));
                    current->primitiveCount = static_cast<uint32_t>(out.size() - leafStart);
                    for (uint64_t i = 0; i < current->primitiveCount; ++i)
                        current->primitives[i] = out[leafStart + i].get();
                }
                else
                {
                    for (uint64_t i = 0; i < current->childCount; ++i)
                    {
                        traverse(current->children[i]);
                    }
                }
            };

            traverse(node);
            return outputStartIndex;
        }
    } // namespace bvh

    uint32_t morton3D(float x, float y, float z)
    {
        constexpr auto expandBits = [](uint32_t v) -> uint32_t {
            // Expands 10 bits into 30 bits by inserting 2 zeros between each bit
            v = (v * 0x00010001u) & 0xFF0000FFu;
            v = (v * 0x00000101u) & 0x0F00F00Fu;
            v = (v * 0x00000011u) & 0xC30C30C3u;
            v = (v * 0x00000005u) & 0x49249249u;
            return v;
        };
        // Assumes x, y, z are ∈ [0, 1]
        x = std::clamp(x * 1024.0f, 0.0f, 1023.0f);
        y = std::clamp(y * 1024.0f, 0.0f, 1023.0f);
        z = std::clamp(z * 1024.0f, 0.0f, 1023.0f);

        uint32_t xx = expandBits(static_cast<uint32_t>(x));
        uint32_t yy = expandBits(static_cast<uint32_t>(y));
        uint32_t zz = expandBits(static_cast<uint32_t>(z));

        return (xx << 2) | (yy << 1) | zz;
    }

    void reorderByMorton(std::span<TriangleData> tris)
    {
        Bounds3f bounds = bbEmpty();
        for (auto& t : tris)
            bounds = bbUnion(bounds,
                             dmt::Bounds3f{dmt::min(dmt::min(t.v0, t.v1), t.v2), dmt::max(dmt::max(t.v0, t.v1), t.v2)});

        auto const getMortonIndex = [bounds](TriangleData const& t) -> uint32_t {
            Point3f  c = (t.v0 + t.v1 + t.v2) / 3;
            Point3f  n = bounds.offset(c); // Normalize to [0,1]
            uint32_t m = morton3D(n.x, n.y, n.z);
            return m;
        };

        std::sort(tris.begin(), tris.end(), [&getMortonIndex](auto const& a, auto const& b) {
            return getMortonIndex(a) < getMortonIndex(b);
        });
    }


    namespace ddbg {
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
    } // namespace ddbg

    bool slabTest(Point3f rayOrigin, Vector3f rayDirection, Bounds3f const& box, float* outTmin = nullptr, float* outTmax = nullptr)
    {
        float tmin = -std::numeric_limits<float>::infinity();
        float tmax = std::numeric_limits<float>::infinity();

        for (int i = 0; i < 3; ++i)
        {
            float invD = 1.0f / rayDirection[i];
            float t0   = (box.pMin[i] - rayOrigin[i]) * invD;
            float t1   = (box.pMax[i] - rayOrigin[i]) * invD;

            if (invD < 0.0f)
                std::swap(t0, t1);

            tmin = std::max(tmin, t0);
            tmax = std::min(tmax, t1);

            if (tmax < tmin)
                return false; // No intersection
        }

        if (outTmin)
            *outTmin = tmin;
        if (outTmax)
            *outTmax = tmax;

        return true; // Intersection occurred
    }

    BVHBuildNode* traverseBVHBuild(Ray                        ray,
                                   BVHBuildNode*              bvh,
                                   std::pmr::memory_resource* memory = std::pmr::get_default_resource())
    {
        std::pmr::vector<BVHBuildNode*> activeNodeStack;
        activeNodeStack.reserve(64);
        activeNodeStack.push_back(bvh);

        BVHBuildNode* intersection = nullptr;
        while (!activeNodeStack.empty())
        {
            BVHBuildNode* current = activeNodeStack.back();
            activeNodeStack.pop_back();

            if (current->childCount > 0)
            {
                // children order of traversal: 1) Distance Heuristic: from smallest to highest tmin - ray origin 2) Sign Heuristic
                // start with distance heuristic
                struct
                {
                    uint32_t i = static_cast<uint32_t>(-1);
                    float    d = fl::infinity();
                } tmins[BranchingFactor];
                uint32_t currentIndex = 0;

                for (uint32_t i = 0; i < current->childCount; ++i)
                {
                    float tmin = fl::infinity();
                    if (slabTest(ray.o, ray.d, current->children[i]->bounds, &tmin))
                    {
                        tmins[currentIndex].d = tmin;
                        tmins[currentIndex].i = i;
                        ++currentIndex;
                    }
                }

                std::sort(std::begin(tmins), std::begin(tmins) + currentIndex, [](auto const& a, auto const& b) {
                    return a.d > b.d;
                });

                for (uint32_t i = 0; i < currentIndex; ++i)
                    activeNodeStack.push_back(current->children[tmins[i].i]);
            }
            else
            {
                // TODO handle any-hit, closest-hit, ...
                // for now, stop at the first leaf intersection
                intersection = current;
                break;
            }
        }

        return intersection;
    }
} // namespace dmt

void bvhTestRays(dmt::BVHBuildNode* rootNode)
{
    dmt::Context ctx;
    assert(ctx.isValid() && "Invalid context");
    std::vector<dmt::Ray> testRays = {
        // Straight through center of scene
        dmt::Ray({{0.5f, 0.5f, -1.0f}}, {{0, 0, 1}}),
        dmt::Ray({{1.5f, 1.5f, -1.0f}}, {{0, 0, 1}}),
        dmt::Ray({{2.5f, 2.5f, -1.0f}}, {{0, 0, 1}}),
        dmt::Ray({{3.5f, 3.5f, -1.0f}}, {{0, 0, 1}}),

        // Grazing edges
        dmt::Ray({{1.0f, 1.0f, -1.0f}}, {{0, 0, 1}}),
        dmt::Ray({{4.0f, 4.0f, -1.0f}}, {{0, 0, 1}}),

        // Missing all
        dmt::Ray({{5.0f, 5.0f, -1.0f}}, {{0, 0, 1}}),
        dmt::Ray({{-1.0f, -1.0f, -1.0f}}, {{0, 0, 1}}),

        // Diagonal through stack
        dmt::Ray({{-1.0f, -1.0f, 1.0f}}, {{1, 1, 1}}),
        dmt::Ray({{0.5f, 0.5f, 0.5f}}, {{1, 1, 1}}),

        // Through nested box
        dmt::Ray({{1.0f, 1.0f, 1.0f}}, {{0, 1, 0}}),
    };

    for (size_t i = 0; i < testRays.size(); ++i)
    {
        dmt::BVHBuildNode* hit = dmt::traverseBVHBuild(testRays[i], rootNode);
        if (hit)
        {
            ctx.log("Ray {} hit leaf bounding box: min = ({}, {}, {}), max = ({}, {}, {})",
                    std::make_tuple(i,
                                    hit->bounds.pMin.x,
                                    hit->bounds.pMin.y,
                                    hit->bounds.pMin.z,
                                    hit->bounds.pMax.x,
                                    hit->bounds.pMax.y,
                                    hit->bounds.pMax.z));
        }
        else
        {
            ctx.log("Ray {} missed the scene.", std::make_tuple(i));
        }
    }
}

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

        // check that prims bounds equal scene bounds
        dmt::Bounds3f const
            sceneBounds = std::transform_reduce(scene.begin(), scene.end(), dmt::bbEmpty(), [](dmt::Bounds3f a, dmt::Bounds3f b) {
            return dmt::bbUnion(a, b);
        }, [](dmt::TriangleData const& t) {
            return dmt::Bounds3f{dmt::min(dmt::min(t.v0, t.v1), t.v2), dmt::max(dmt::max(t.v0, t.v1), t.v2)};
        });
        dmt::Bounds3f const primsBounds = std::transform_reduce( //
            spanPrims.begin(),
            spanPrims.end(),
            dmt::bbEmpty(),
            [](dmt::Bounds3f a, dmt::Bounds3f b) { return dmt::bbUnion(a, b); },
            [](dmt::Primitive const* p) { return p->bounds(); });

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

        auto* rootNode = dmt::bvh::build(spanPrims, &bufferMemory);

        if (ctx.isLogEnabled())
        {
            ctx.log("-- Before Primitive Packing --", {});
            std::string tee = dmt::ddbg::printBVHToString(rootNode);
            ctx.log(std::string_view{tee}, {});
        }

        bvhTestRays(rootNode);

        dmt::bvh::groupTrianglesInBVHLeaves(rootNode, prims, &bufferMemory);

        if (ctx.isLogEnabled())
        {
            ctx.log("-- After Primitive Packing --", {});
            std::string tee = dmt::ddbg::printBVHToString(rootNode);
            ctx.log(std::string_view{tee}, {});
        }

        bvhTestRays(rootNode);

        dmt::bvh::cleanup(rootNode);
    }

    return 0;
}
