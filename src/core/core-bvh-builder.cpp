#include "core-bvh-builder.h"

#include "core-dstd.h"

#include <numeric>
#include <cstdlib>

namespace dmt::bvh {
    static DMT_FORCEINLINE float evaluateSAH(std::span<Primitive const*> nodePrims, int32_t axis, float splitPos)
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
                        std::pmr::memory_resource*        temp, // allocate only stuff which doesn't need destruction!
                        std::pmr::memory_resource*        memory)
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

                        Bounds3f const& bounds     = current.bounds;
                        Vector3f const  diag       = bounds.pMax - bounds.pMin;
                        bool const      degenerate = (diag[0] < DegenerateEpsilon) || (diag[1] < DegenerateEpsilon) ||
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

    void cleanup(BVHBuildNode* node, std::pmr::memory_resource* memory)
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
                                          std::pmr::memory_resource*                   memory)
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
} // namespace dmt::bvh