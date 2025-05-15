
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
                while (!childrenStack.empty() && childrenStack.size() < BranchingFactor && splitCount < BranchingFactor)
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
            // Row 1
            dmt::Bounds3f({{0, 0, 0}}, {{1, 1, 0}}), // Triangle 1 (quad made up of 4 triangles)
            dmt::Bounds3f({{1, 0, 0}}, {{2, 1, 0}}), // Triangle 2
            dmt::Bounds3f({{2, 0, 0}}, {{3, 1, 0}}), // Triangle 3
            dmt::Bounds3f({{3, 0, 0}}, {{4, 1, 0}}), // Triangle 4

            // Row 2
            dmt::Bounds3f({{0, 1, 0}}, {{1, 2, 0}}), // Triangle 5
            dmt::Bounds3f({{1, 1, 0}}, {{2, 2, 0}}), // Triangle 6
            dmt::Bounds3f({{2, 1, 0}}, {{3, 2, 0}}), // Triangle 7
            dmt::Bounds3f({{3, 1, 0}}, {{4, 2, 0}}), // Triangle 8

            // Row 3
            dmt::Bounds3f({{0, 2, 0}}, {{1, 3, 0}}), // Triangle 9
            dmt::Bounds3f({{1, 2, 0}}, {{2, 3, 0}}), // Triangle 10
            dmt::Bounds3f({{2, 2, 0}}, {{3, 3, 0}}), // Triangle 11
            dmt::Bounds3f({{3, 2, 0}}, {{4, 3, 0}}), // Triangle 12

            // Row 4
            dmt::Bounds3f({{0, 3, 0}}, {{1, 4, 0}}), // Triangle 13
            dmt::Bounds3f({{1, 3, 0}}, {{2, 4, 0}}), // Triangle 14
            dmt::Bounds3f({{2, 3, 0}}, {{3, 4, 0}}), // Triangle 15
            dmt::Bounds3f({{3, 3, 0}}, {{4, 4, 0}}), // Triangle 16

            // Row 5
            dmt::Bounds3f({{0, 4, 0}}, {{1, 5, 0}}), // Triangle 17
            dmt::Bounds3f({{1, 4, 0}}, {{2, 5, 0}}), // Triangle 18
            dmt::Bounds3f({{2, 4, 0}}, {{3, 5, 0}}), // Triangle 19
            dmt::Bounds3f({{3, 4, 0}}, {{4, 5, 0}})  // Triangle 20
        };

        auto* rootNode = dmt::buildBVH(primitives);

        if (ctx.isLogEnabled())
        {
            std::string json = dmt::bvh_debug::bvhToJson(rootNode);
            ctx.log(std::string_view{json}, {});
        }

        dmt::cleanupBuildBVH(rootNode);
    }

    return 0;
}
