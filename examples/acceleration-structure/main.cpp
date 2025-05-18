
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

namespace dmt {
    struct BVHNodeAVX2
    {
        __m256 xMin;
        __m256 xMax;
        __m256 yMin;
        __m256 yMax;
        __m256 zMin;
        __m256 zMax;

        // sign - node data. sign (3 bytes) contains 8 3-bit data indicating the sign permutations
        // node data (5 bytes) contains: | is inner | child cluster or primitive cluster offset in leaf | mask to identify valid node or number of primitive cluster in leaf |
        __m256 sn;
    };

    inline constexpr int LogBranchingFactor = 3;
    inline constexpr int BranchingFactor    = 1 << LogBranchingFactor;
    inline constexpr int TrianglesPerLeaf   = 4;

    struct BVHBuildNode
    {
        Bounds3f      bounds;
        void*         primitive;
        BVHBuildNode* children[BranchingFactor];
        uint32_t      childCount;
    };

    struct NodeState
    {
        Bounds3f bounds;
        uint32_t count;
    };

    namespace bvh_debug {
        uint32_t leafCount(BVHBuildNode const* node)
        {
            Context ctx;
            if (!node)
                return 0;

            // If the node has no children, it's a leaf
            if (node->childCount == 0)
            {
                ctx.log("bounds: {{ min: [{} {} {}], max: [{} {} {}] }}",
                        std::make_tuple(node->bounds.pMin.x,
                                        node->bounds.pMin.y,
                                        node->bounds.pMin.z,
                                        node->bounds.pMax.x,
                                        node->bounds.pMax.y,
                                        node->bounds.pMax.z));
                return 1;
            }

            uint32_t count = 0;
            for (uint32_t i = 0; i < node->childCount; ++i)
            {
                count += leafCount(node->children[i]);
            }
            return count;
        }
    } // namespace bvh_debug

    void cleanupBuildBVH(BVHBuildNode* node, std::pmr::memory_resource* alloc = std::pmr::get_default_resource())
    {
        if (node->childCount == 0)
        {
            alloc->deallocate(node, sizeof(BVHBuildNode));
        }
        else
        {
            for (uint32_t i = 0; i < node->childCount; ++i)
            {
                cleanupBuildBVH(node->children[i], alloc);
            }
        }
    }

    namespace bvh::detail {
        std::array<uint32_t, 3> axesFromSelected(uint32_t used)
        {
            std::array<uint32_t, 3> axes;
            axes[0] = used;

            uint32_t idx = 1;
            for (uint32_t i = 0; i < 3; ++i)
            {
                if (i != used)
                {
                    axes[idx++] = i;
                }
            }

            return axes;
        }
    } // namespace bvh::detail

    BVHBuildNode* buildBVH(std::span<Bounds3f>        primitives,
                           std::pmr::memory_resource* alloc = std::pmr::get_default_resource())
    {
        static constexpr int NumBins   = 12;
        static constexpr int NumSplits = NumBins - 1;

        using Iter = std::pmr::vector<Bounds3f>::iterator;

        struct StackFrame
        {
            BVHBuildNode* node;
            Iter          beg, end;
            Bounds3f      bounds;
        };

        std::pmr::vector<Bounds3f> prims{alloc};
        prims.reserve(primitives.size());
        for (auto const& prim : primitives)
            prims.push_back(prim);

        if (prims.empty())
            return nullptr;

        Bounds3f totalBounds = prims[0];
        for (size_t i = 1; i < prims.size(); ++i)
            totalBounds = bbUnion(totalBounds, prims[i]);

        BVHBuildNode*               root = new (alloc->allocate(sizeof(BVHBuildNode))) BVHBuildNode{};
        std::pmr::deque<StackFrame> stack{alloc};
        stack.push_back({root, prims.begin(), prims.end(), totalBounds});

        while (!stack.empty())
        {
            StackFrame frame = stack.back();
            stack.pop_back();

            BVHBuildNode* node = frame.node;
            Iter          beg = frame.beg, end = frame.end;
            node->bounds = frame.bounds;

            size_t count = end - beg;
            if (count == 1)
            {
                // Leaf node (already grouped beforehand)
                node->childCount = 0;
                node->primitive  = nullptr; // Replace with actual cluster ref if needed
            }
            else
            {
                // 8-way split using 3 SAH passes (2^3 = 8)
                std::pmr::deque<StackFrame> childrenStack{alloc};
                std::pmr::deque<StackFrame> singletonStack{alloc};
                childrenStack.push_back({node, beg, end, frame.bounds});

                int splitCount = 0;
                while (!childrenStack.empty() && childrenStack.size() < BranchingFactor && splitCount < BranchingFactor - 1)
                {
                    StackFrame current = childrenStack.front();
                    childrenStack.pop_front();

                    Iter cbeg = current.beg, cend = current.end;
                    if (cend - cbeg <= 1)
                    {
                        assert(!current.node);
                        singletonStack.push_back(current);
                    }
                    if (cend - cbeg > 1)
                    {
                        auto splitDims = bvh::detail::axesFromSelected(current.bounds.maxDimention());
                        bool splitDone = false;
                        for (uint32_t i = 0; i < 3 && !splitDone; ++i)
                        {
                            uint32_t splitDim = current.bounds.maxDimention();
                            std::sort(cbeg, cend, [splitDim](Bounds3f const& a, Bounds3f const& b) {
                                return a.centroid()[splitDim] < b.centroid()[splitDim];
                            });

                            std::array<NodeState, NumBins> bins{};
                            for (auto it = cbeg; it != cend; ++it)
                            {
                                int b = std::clamp(static_cast<int>(NumBins * current.bounds.offset(it->centroid())[splitDim]),
                                                   0,
                                                   NumBins - 1);
                                bins[b].count++;
                                bins[b].bounds = bbUnion(bins[b].bounds, *it);
                            }

                            std::array<float, NumSplits> costs{};
                            int                          leftCount  = 0;
                            Bounds3f                     leftBounds = bbEmpty();
                            for (int i = 0; i < NumSplits; ++i)
                            {
                                leftBounds = bbUnion(leftBounds, bins[i].bounds);
                                leftCount += bins[i].count;
                                costs[i] += leftCount * leftBounds.surfaceArea();
                            }

                            int      rightCount  = 0;
                            Bounds3f rightBounds = bbEmpty();
                            for (int i = NumSplits; i > 0; --i)
                            {
                                rightBounds = bbUnion(rightBounds, bins[i].bounds);
                                rightCount += bins[i].count;
                                costs[i - 1] += rightCount * rightBounds.surfaceArea();
                            }

                            int   bestSplit = std::min_element(costs.begin(), costs.end()) - costs.begin();
                            float splitVal  = static_cast<float>(bestSplit + 1) / NumBins;

                            Iter mid = std::partition(cbeg, cend, [&](Bounds3f const& b) {
                                return current.bounds.offset(b.centroid())[splitDim] < splitVal;
                            });

                            if (mid != cbeg && mid != cend)
                            {
                                // Push both halves
                                Bounds3f leftB = bbEmpty(), rightB = bbEmpty();
                                for (auto it = cbeg; it != mid; ++it)
                                    leftB = bbUnion(leftB, *it);
                                for (auto it = mid; it != cend; ++it)
                                    rightB = bbUnion(rightB, *it);

                                childrenStack.emplace_back(nullptr, cbeg, mid, leftB);
                                childrenStack.emplace_back(nullptr, mid, cend, rightB);
                                ++splitCount;
                                splitDone = true;
                            }
                        }

                        if (!splitDone)
                        {
                            // fallback to equal count splitting
                            uint32_t fallbackDim = current.bounds.maxDimention(); // could reuse `splitDim` from earlier

                            std::sort(cbeg, cend, [fallbackDim](Bounds3f const& a, Bounds3f const& b) {
                                return a.centroid()[fallbackDim] < b.centroid()[fallbackDim];
                            });

                            Iter mid = cbeg + (cend - cbeg) / 2;

                            if (mid != cbeg && mid != cend)
                            {
                                Bounds3f leftB = bbEmpty(), rightB = bbEmpty();
                                for (auto it = cbeg; it != mid; ++it)
                                    leftB = bbUnion(leftB, *it);
                                for (auto it = mid; it != cend; ++it)
                                    rightB = bbUnion(rightB, *it);

                                childrenStack.emplace_back(nullptr, cbeg, mid, leftB);
                                childrenStack.emplace_back(nullptr, mid, cend, rightB);
                                ++splitCount;
                            }
                            else
                            {
                                assert(false && "Equal Count splitting shouldn't fail with more than 1 primitive");
                            }
                        }
                    }
                }

                for (auto const& frame : singletonStack)
                    childrenStack.push_back(frame);

                // Finalize 8 children
                node->childCount = childrenStack.size();
                for (uint32_t i = 0; i < node->childCount; ++i)
                {
                    BVHBuildNode* child   = new (alloc->allocate(sizeof(BVHBuildNode))) BVHBuildNode{};
                    node->children[i]     = child;
                    StackFrame childFrame = childrenStack[i];
                    childFrame.node       = child;
                    stack.push_back(childFrame);
                }
            }
        }

        assert(bvh_debug::leafCount(root) == primitives.size());

        return root;
    }

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

    namespace bvh_debug {
        std::string boundsToJson(Bounds3f const& bounds)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3);
            oss << "{ \"pMin\": [" << bounds.pMin.x << ", " << bounds.pMin.y << ", " << bounds.pMin.z << "], ";
            oss << "\"pMax\": [" << bounds.pMax.x << ", " << bounds.pMax.y << ", " << bounds.pMax.z << "] }";
            return oss.str();
        }

        void serializeBVHNode(std::ostringstream& oss, BVHBuildNode const* node)
        {
            assert(node);
            oss << "{ \"bounds\": " << boundsToJson(node->bounds);

            if (node->childCount > 0)
            {
                oss << ", \"children\": [";
                for (uint32_t i = 0; i < node->childCount; ++i)
                {
                    if (i > 0)
                        oss << ", ";
                    assert(node->children[i]);
                    serializeBVHNode(oss, node->children[i]);
                }
                oss << "] }";
            }
            else
            {
                oss << ", \"primitive\": " << reinterpret_cast<uintptr_t>(node->primitive) << " }";
            }
        }

        std::string bvhToJson(BVHBuildNode const* root)
        {
            std::ostringstream oss;
            serializeBVHNode(oss, root);
            return oss.str();
        }
    } // namespace bvh_debug
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

        // Sample AABBs for the scene
        std::vector<dmt::Bounds3f> primitives = {
            // Ground layer
            dmt::Bounds3f({{0, 0, 0}}, {{1, 1, 1}}),
            dmt::Bounds3f({{1, 0.5, 0.5}}, {{2, 1.5, 1.5}}),     // overlaps with previous
            dmt::Bounds3f({{3, 0, 0}}, {{4, 1, 1}}),             // separated
            dmt::Bounds3f({{2.5, 0.2, 0.2}}, {{3.5, 1.2, 1.2}}), // overlaps with above

            // Mid-layer
            dmt::Bounds3f({{0, 1, 1}}, {{1, 2, 2}}),
            dmt::Bounds3f({{0.8, 1.5, 1.5}}, {{1.8, 2.5, 2.5}}), // partial overlap
            dmt::Bounds3f({{2, 1, 1}}, {{3, 2, 2}}),
            dmt::Bounds3f({{2.1, 1.1, 1.1}}, {{2.9, 1.9, 1.9}}), // completely inside previous

            // Upper layer
            dmt::Bounds3f({{0, 2, 2}}, {{1, 3, 3}}),
            dmt::Bounds3f({{1.5, 2.5, 2.5}}, {{2.5, 3.5, 3.5}}), // isolated
            dmt::Bounds3f({{3, 2, 2}}, {{4, 3, 3}}),
            dmt::Bounds3f({{3.5, 2.5, 2.5}}, {{4.5, 3.5, 3.5}}), // overlaps with previous

            // Random floating boxes
            dmt::Bounds3f({{1, 3, 4}}, {{2, 4, 5}}),
            dmt::Bounds3f({{2.2, 3.2, 4.2}}, {{3.2, 4.2, 5.2}}), // overlaps slightly
            dmt::Bounds3f({{3.5, 3, 4}}, {{4.5, 4, 5}}),         // isolated

            // Diagonal stack
            dmt::Bounds3f({{0, 0, 2}}, {{1, 1, 3}}),
            dmt::Bounds3f({{1, 1, 3}}, {{2, 2, 4}}),
            dmt::Bounds3f({{2, 2, 4}}, {{3, 3, 5}}),
            dmt::Bounds3f({{3, 3, 5}}, {{4, 4, 6}}), // all stacked touching corners

            // One inside another
            dmt::Bounds3f({{0.5, 0.5, 0.5}}, {{3.5, 3.5, 3.5}}) // wraps several smaller boxes
        };

        auto* rootNode = dmt::buildBVH(primitives);

        if (ctx.isLogEnabled())
        {
            std::string json = dmt::bvh_debug::bvhToJson(rootNode);
            ctx.log(std::string_view{json}, {});
        }

        // make tests with some rays
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

        dmt::cleanupBuildBVH(rootNode);
    }

    return 0;
}
