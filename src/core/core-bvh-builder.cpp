#include "core-bvh-builder.h"

#include "core-dstd.h"

#include <numeric>
#include <cstdlib>

namespace dmt::bvh {
    template <typename T, typename F>
        requires std::is_invocable_r_v<Bounds3f, F, T>
    static float evaluateSAH(std::span<T> nodePrims, F&& boundsFunc, int32_t axis, float splitPos)
    {
        Bounds3f leftBounds = bbEmpty(), rightBounds = bbEmpty();
        int32_t  leftCount = 0, rightCount = 0;

        for (auto const* prim : nodePrims)
        {
            Bounds3f const b = boundsFunc(prim);
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

    static Bounds3f bbUnionPrimitives(std::span<Primitive const*> prims)
    {
        Bounds3f bounds = bbEmpty();
        for (Primitive const* prim : prims)
            bounds = bbUnion(prim->bounds(), bounds);

        return bounds;
    }

    static Bounds3f bbUnionNodes(std::span<BVHBuildNode*> nodes)
    {
        Bounds3f bounds = bbEmpty();
        for (BVHBuildNode* node : nodes)
            bounds = bbUnion(node->bounds, bounds);

        return bounds;
    }

    static int32_t estimateBinningSplitNumber(float extent, dmt::Primitive const** primsBeg, dmt::Primitive const** primsEnd)
    {
        return std::clamp(static_cast<int>(extent * BinScaleFactor * std::log2(std::distance(primsBeg, primsEnd) + 1)),
                          MinNumBin,
                          MaxNumBin);
    }

    static void buildRecursive(Primitive const**          _primsBeg,
                               Primitive const**          _primsEnd,
                               BVHBuildNode*              _parent,
                               std::pmr::memory_resource* _memory,
                               std::pmr::memory_resource* _temp)
    {
        assert(std::distance(_primsBeg, _primsEnd) >= 0 && "negative primitive range in BVH construction");
        if (auto dist = std::distance(_primsBeg, _primsEnd); dist <= LeavesBranchingFactor && dist >= 0)
        {
            assert(_parent->childCount == 0 && "Leaf BVH Node shouldn't have children");
            _parent->primitiveCount = std::distance(_primsBeg, _primsEnd);
            std::copy(_primsBeg, _primsEnd, _parent->primitives);
        }
        else
        {
            struct WorkingStackItem
            {
                BVHBuildNode      node;
                Primitive const **beg, **end;
            };
            std::pmr::vector<WorkingStackItem> childCandidates{_temp};
            childCandidates.reserve(BranchingFactor);
            childCandidates.emplace_back();
            std::memset(&childCandidates.back().node, 0, sizeof(BVHBuildNode));
            childCandidates.back().node.bounds = bbUnionPrimitives(std::span{_primsBeg, _primsEnd});
            childCandidates.back().beg         = _primsBeg;
            childCandidates.back().end         = _primsEnd;

            bool shouldContinue = true;
            while (childCandidates.size() < BranchingFactor && shouldContinue)
            {
                auto maybeNode = dstd::move_to_back_and_pop_if(childCandidates, [](WorkingStackItem const& wItem) {
                    return std::distance(wItem.beg, wItem.end) >= LeavesBranchingFactor;
                });

                if (!maybeNode)
                    shouldContinue = false;
                else
                {
                    BVHBuildNode const& current  = maybeNode->node;
                    Primitive const**   primsBeg = maybeNode->beg;
                    Primitive const**   primsEnd = maybeNode->end;
                    int32_t const       axis     = current.bounds.maxDimention(); // common estimate
                    Primitive const**   primsMid = primsBeg;

                    // check for degenerate bounding box (flat in one dimension)
                    Bounds3f const& bounds     = current.bounds;
                    Vector3f const  diag       = bounds.pMax - bounds.pMin;
                    bool const      degenerate = (diag[0] < DegenerateEpsilon) || (diag[1] < DegenerateEpsilon) ||
                                            (diag[2] < DegenerateEpsilon);

                    if (!degenerate)
                    {
                        float const   extent               = current.bounds.pMax[axis] - current.bounds.pMin[axis];
                        float         minimumSplitCost     = fl::infinity();
                        float         minimumSplitPosition = 0.f;
                        int32_t const numSplits            = estimateBinningSplitNumber(extent, primsBeg, primsEnd);
                        float const   splitLength          = extent / numSplits;
                        for (uint64_t i = 0; i < numSplits - 1; ++i)
                        {

                            float const splitPosition = current.bounds.pMin[axis] + (i + 1) * splitLength;
                            float const splitCost = evaluateSAH(std::span{primsBeg, primsEnd}, [](Primitive const* p) {
                                return p->bounds();
                            }, axis, splitPosition);
                            if (splitCost < minimumSplitCost)
                            {
                                minimumSplitCost     = splitCost;
                                minimumSplitPosition = splitPosition;
                            }
                        }

                        primsMid = std::partition(primsBeg, primsEnd, [axis, mid = minimumSplitPosition](Primitive const* p) {
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

                    first->beg         = primsMid;
                    first->end         = primsEnd;
                    first->node.bounds = bbUnionPrimitives(std::span{primsMid, primsEnd});

                    last->beg         = primsBeg;
                    last->end         = primsMid;
                    last->node.bounds = bbUnionPrimitives(std::span{primsBeg, primsMid});
                } // end else (maybeNode)
            } // end while (on workItem list)

            for (auto const& workItem : childCandidates)
            {
                auto* node = reinterpret_cast<BVHBuildNode*>(_memory->allocate(sizeof(BVHBuildNode)));
                std::memset(node, 0, sizeof(BVHBuildNode));
                node->bounds = workItem.node.bounds;
                assert(_parent->childCount < BranchingFactor && "Too many children for BVH Node");
                _parent->children[_parent->childCount++] = node;
                buildRecursive(workItem.beg, workItem.end, node, _memory, _temp);
            }
        }
    };

    BVHBuildNode* buildCombined(BVHBuildNode*              root,
                                std::span<BVHBuildNode*>   nodes,
                                std::pmr::memory_resource* temp,
                                std::pmr::memory_resource* memory)
    {
        // done by caller
        // auto* root = reinterpret_cast<BVHBuildNode*>(memory->allocate(sizeof(BVHBuildNode)));
        // std::memset(root, 0, sizeof(BVHBuildNode));

        // TODO maybe: order according to sign heuristic
        root->bounds = bbUnionNodes(nodes);
        if (nodes.size() <= BranchingFactor)
        {
            root->childCount = static_cast<uint32_t>(nodes.size());
            for (uint32_t i = 0; i < root->childCount; ++i)
                root->children[i] = nodes[i];
        }
        else
        {
            struct Pair
            {
                Bounds3f                 bounds;
                std::span<BVHBuildNode*> nodes;
            };
            std::pmr::vector<Pair> stack{temp};
            stack.emplace_back(root->bounds, nodes);

            size_t   lastCount          = 0;
            uint32_t proposedChildCount = 1;
            while (lastCount != stack.size() && proposedChildCount < BranchingFactor)
            {
                lastCount                      = stack.size();
                auto const [bounds, currNodes] = stack.back();
                stack.pop_back();

                // TODO copy and abstract the else branch in build recursive. for now keep the binning part
                int32_t const axis        = bounds.maxDimention();
                float const   extent      = bounds.pMax[axis] - bounds.pMin[axis];
                int32_t const numSplits   = MinNumBin; // TODO better
                float const   splitLength = extent / numSplits;


                float minimumSplitCost     = fl::infinity();
                float minimumSplitPosition = 0.f;

                for (int32_t i = 0; i < numSplits - 1; ++i)
                {
                    float const splitPosition = bounds.pMin[axis] + (i + 1) * splitLength;

                    float splitCost = evaluateSAH(currNodes, [](BVHBuildNode const* p) {
                        return p->bounds;
                    }, axis, splitPosition);

                    if (splitCost < minimumSplitCost)
                    {
                        minimumSplitCost     = splitCost;
                        minimumSplitPosition = splitPosition;
                    }
                }

                auto mid = std::partition(currNodes.begin(),
                                          currNodes.end(),
                                          [axis, middle = minimumSplitPosition](BVHBuildNode* p) {
                    return p->bounds.centroid()[axis] < middle;
                });

                if (mid != currNodes.begin() && mid != currNodes.end())
                {
                    std::span<BVHBuildNode*> left{currNodes.begin(), mid};
                    std::span<BVHBuildNode*> right{mid, currNodes.end()};

                    stack.emplace_back(bbUnionNodes(left), left);
                    stack.emplace_back(bbUnionNodes(right), right);
                    ++proposedChildCount;
                }
                else
                    stack.emplace_back(bounds, currNodes);
            } // end while

            // TODO move fallback elsewhere
            if (proposedChildCount == 1 || stack.size() == 1)
            {
                auto const [bounds, currNodes] = stack.back();
                stack.pop_back();
                assert(currNodes.size() > BranchingFactor &&
                       "if less than branching factor than it "
                       "should create a leaf");

                // Split into 8 equal-sized spans
                size_t total     = currNodes.size();
                size_t baseSize  = total / BranchingFactor;
                size_t remainder = total % BranchingFactor;

                BVHBuildNode** it = &currNodes[0];

                for (int i = 0; i < BranchingFactor; ++i)
                {
                    size_t                   count = baseSize + (i < remainder ? 1 : 0);
                    std::span<BVHBuildNode*> subspan{it, count};
                    stack.emplace_back(bbUnionNodes(subspan), subspan);
                    it += count;
                }

                proposedChildCount = BranchingFactor;
            }

            assert(proposedChildCount > 1 && proposedChildCount <= BranchingFactor && !stack.empty() &&
                   "Binning failed miserably");

            root->childCount = static_cast<uint32_t>(stack.size());
            for (uint32_t i = 0; i < root->childCount; ++i)
            {
                auto* node = reinterpret_cast<BVHBuildNode*>(memory->allocate(sizeof(BVHBuildNode)));
                std::memset(node, 0, sizeof(BVHBuildNode));
                node->childCount = static_cast<uint32_t>(stack[i].nodes.size());
                node->bounds     = stack[i].bounds;
                for (uint32_t j = 0; j < node->childCount; ++j)
                    node->children[j] = stack[i].nodes[j];

                root->children[i] = node;
            }
        }

        return root;
    }

    BVHBuildNode* buildForInstance(Scene const&                            scene,
                                   size_t                                  instanceIdx,
                                   std::pmr::vector<UniqueRef<Primitive>>& outPrims,
                                   std::pmr::memory_resource*              temp,
                                   std::pmr::memory_resource*              memory)
    {
        Instance const*     instance   = scene.instances[instanceIdx].get();
        Transform const     m          = transformFromAffine(instance->affineTransform);
        TriangleMesh const& mesh       = *scene.geometry[instance->meshIdx];
        size_t const        primsStart = outPrims.size();

        outPrims.reserve(primsStart + mesh.triCount());

        for (size_t triIdx = 0; triIdx < mesh.triCount(); ++triIdx)
        {
            TriangleIndexed tri;
            tri.scene       = &scene;
            tri.instanceIdx = instanceIdx;
            tri.triIdx      = triIdx;
            outPrims.push_back(makeUniqueRef<TriangleIndexed>(memory, tri));
        }

        auto* root = reinterpret_cast<BVHBuildNode*>(memory->allocate(sizeof(BVHBuildNode)));
        std::memset(root, 0, sizeof(BVHBuildNode));
        root->bounds = instance->bounds;

        std::pmr::vector<Primitive const*> tempCopy{temp};

        size_t const numel = outPrims.size() - primsStart;
        tempCopy.reserve(numel);
        for (size_t i = 0; i < numel; ++i)
            tempCopy.push_back(outPrims[primsStart + i].get());

        buildRecursive(tempCopy.data(), tempCopy.data() + tempCopy.size(), root, memory, temp);
        return root;
    }

    BVHBuildNode* build(std::span<Primitive const*> const prims,
                        std::pmr::memory_resource*        temp, // allocate only stuff which doesn't need destruction!
                        std::pmr::memory_resource*        memory)
    {
        std::pmr::vector<Primitive const*> shufflingPrims{temp};
        shufflingPrims.reserve(nextPOT(prims.size()));
        std::copy(prims.begin(), prims.end(), std::back_inserter(shufflingPrims));

        auto* root = reinterpret_cast<BVHBuildNode*>(memory->allocate(sizeof(BVHBuildNode)));
        std::memset(root, 0, sizeof(BVHBuildNode));
        root->bounds = std::transform_reduce(shufflingPrims.begin(), shufflingPrims.end(), bbEmpty(), [](Bounds3f a, Bounds3f b) {
            return bbUnion(a, b);
        }, [](Primitive const* p) { return p->bounds(); });
        buildRecursive(shufflingPrims.data(), shufflingPrims.data() + shufflingPrims.size(), root, memory, temp);
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

#define DMT_BVH_TRIANGLES_GROUP8
#define DMT_BVH_TRIANGLES_GROUP4
#define DMT_BVH_TRIANGLES_GROUP2

                // Group triangles TODO Remove color
                std::size_t i = 0;
#ifdef DMT_BVH_TRIANGLES_GROUP8
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
                        group.colors[j]     = triangles[i + j].tri.color;
                    }
                    out.push_back(makeUniqueRef<Triangles8>(memory, std::move(group)));
                    i += 8;
                }
#endif

#ifdef DMT_BVH_TRIANGLES_GROUP4
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
                        group.colors[j]     = triangles[i + j].tri.color;
                    }
                    out.push_back(makeUniqueRef<Triangles4>(memory, std::move(group)));
                    i += 4;
                }
#endif

#ifdef DMT_BVH_TRIANGLES_GROUP2
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
                        group.colors[j]     = triangles[i + j].tri.color;
                    }
                    out.push_back(makeUniqueRef<Triangles2>(memory, std::move(group)));
                    i += 2;
                }
#endif

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