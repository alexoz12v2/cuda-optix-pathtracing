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
            uint8_t                            idxNode = 0;
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
                    BVHBuildNode&     current  = maybeNode->node;
                    Primitive const** primsBeg = maybeNode->beg;
                    Primitive const** primsEnd = maybeNode->end;
                    int32_t const     axis     = current.bounds.maxDimention(); // common estimate
                    Primitive const** primsMid = primsBeg;


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
                        assert(false);
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

    static bool allSingleIndexedPrimitives(Primitive const* primitives[LeavesBranchingFactor], uint32_t count)
    {
        for (uint32_t i = 0; i < count; ++i)
        {
            if (!dynamic_cast<TriangleIndexed const*>(primitives[i]))
                return false;
        }
        return true;
    }

    static bool groupPrimitivesRecursive(
        Scene const&                              scene,
        size_t                                    instanceIdx,
        BVHBuildNode*                             node,
        std::pmr::vector<Primitive const*> const& middlePrims,
        std::pmr::vector<UniqueRef<Primitive>>&   outPrims,
        std::pmr::memory_resource*                memory)
    {
        assert(node && "nullptr");
        if (!node)
            return false;

        if (node->childCount == 0) // leaf
        {
            size_t triCount = node->primitiveCount;
            bool   valid    = allSingleIndexedPrimitives(node->primitives, node->primitiveCount);
            assert(valid && "Uncompatible primitive types");
            if (!valid) // if valid we can reinterpret_cast to TriangleIndexed array
                return false;
            TriangleIndexed const** tris = reinterpret_cast<TriangleIndexed const**>(node->primitives);

            size_t i          = 0;
            size_t groupCount = 0;
            while (i < triCount)
            {
                size_t const remaining = triCount - i;
#if defined(DMT_GROUP_PRIMITIVES)
                if (remaining >= 8)
                {
                    auto* grouped = std::construct_at(
                        reinterpret_cast<TrianglesIndexed8*>(memory->allocate(sizeof(TrianglesIndexed8))));
                    grouped->scene       = &scene;
                    grouped->instanceIdx = instanceIdx;
                    for (size_t j = 0; j < 8; ++j)
                        grouped->triIdxs[j] = tris[i + j]->triIdx;

                    outPrims.push_back(UniqueRef<TrianglesIndexed8>(grouped, PmrDeleter::create<TrianglesIndexed8>(memory)));
                    i += 8;
                }
                else if (remaining >= 4)
                {
                    auto* grouped = std::construct_at(
                        reinterpret_cast<TrianglesIndexed4*>(memory->allocate(sizeof(TrianglesIndexed4))));
                    grouped->scene       = &scene;
                    grouped->instanceIdx = instanceIdx;
                    for (size_t j = 0; j < 4; ++j)
                        grouped->triIdxs[j] = tris[i + j]->triIdx;

                    outPrims.push_back(UniqueRef<TrianglesIndexed4>(grouped, PmrDeleter::create<TrianglesIndexed4>(memory)));
                    i += 4;
                }
                else if (remaining >= 2)
                {
                    auto* grouped = std::construct_at(
                        reinterpret_cast<TrianglesIndexed2*>(memory->allocate(sizeof(TrianglesIndexed2))));
                    grouped->scene       = &scene;
                    grouped->instanceIdx = instanceIdx;
                    for (size_t j = 0; j < 2; ++j)
                        grouped->triIdxs[j] = tris[i + j]->triIdx;

                    outPrims.push_back(UniqueRef<TrianglesIndexed2>(grouped, PmrDeleter::create<TrianglesIndexed2>(memory)));
                    i += 2;
                }
                else
#endif
                {
                    auto* single = std::construct_at(
                        reinterpret_cast<TriangleIndexed*>(memory->allocate(sizeof(TriangleIndexed))));
                    single->scene       = &scene;
                    single->instanceIdx = instanceIdx;
                    *single             = *tris[i]; // shallow copy
                    outPrims.push_back(UniqueRef<TriangleIndexed>(single, PmrDeleter::create<TriangleIndexed>(memory)));
                    i += 1;
                }

                ++groupCount;
            }

            // store grouped primitive in node
            assert(groupCount <= node->primitiveCount && "Grouping should always diminish primitives for each leaf");
            node->primitiveCount = static_cast<uint32_t>(groupCount);
            std::memset(node->primitives, 0, sizeof(node->primitives));

            size_t const offset = outPrims.size() - groupCount;
            for (uint32_t idx = 0; idx < node->primitiveCount; ++idx)
            {
                node->primitives[idx] = outPrims[offset + idx].get();
            }
        }
        else
        {
            for (uint32_t i = 0; i < node->childCount; ++i)
                groupPrimitivesRecursive(scene, instanceIdx, node->children[i], middlePrims, outPrims, memory);
        }
        return true;
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

        std::pmr::vector<Primitive*> middlePrims{temp};

        middlePrims.reserve(mesh.triCount());
        outPrims.reserve(primsStart + mesh.triCount());

        for (size_t triIdx = 0; triIdx < mesh.triCount(); ++triIdx)
        {
            // WARNING: using non-temp memory. You need to free it later
            auto* tri = reinterpret_cast<TriangleIndexed*>(memory->allocate(sizeof(TriangleIndexed)));
            std::construct_at(tri);
            tri->scene       = &scene;
            tri->instanceIdx = instanceIdx;
            tri->triIdx      = triIdx;
            //outPrims.push_back(makeUniqueRef<TriangleIndexed>(memory, tri));
            middlePrims.push_back(tri);
        }

        auto* root = reinterpret_cast<BVHBuildNode*>(memory->allocate(sizeof(BVHBuildNode)));
        std::memset(root, 0, sizeof(BVHBuildNode));
        root->bounds          = instance->bounds;
        Primitive const** beg = const_cast<Primitive const**>(middlePrims.data());
        Primitive const** end = const_cast<Primitive const**>(middlePrims.data() + middlePrims.size());

        buildRecursive(beg, end, root, memory, temp);

        // Traverse the BVH, and, for each node having 2, 4, 8 TriangleIndexed primitives, group them into TrianglesIndexed2,
        // TrianglesIndexed4, TrianglesIndexed8
        // Once done, copy to outPrims, transfering pointer ownership
#if defined(DMT_GROUP_PRIMITIVES)
        groupPrimitivesRecursive(scene, instanceIdx, root, middlePrims, outPrims, memory);
#else
        for (uint32_t i = 0; i < middlePrims.size(); ++i)
            outPrims.push_back(UniqueRef<Primitive>{middlePrims[i], PmrDeleter::create<Primitive>(memory)});
#endif

        return root;
    }

    static uint32_t countBVHBuildNodesRecursive(BVHBuildNode* bvh)
    {
        if (!bvh || bvh->childCount == 0)
            return 0;

        uint32_t count = bvh->childCount;
        for (uint32_t i = 0; i < bvh->childCount; ++i)
            count += countBVHBuildNodesRecursive(bvh->children[i]);

        return count;
    }

    static void buildPermutationsBoxproj(Bounds3f const* children, uint32_t childCount, std::array<uint32_t, 8>& outPerm)
    {
        static_assert(BranchingFactor == SIMDWidth && SIMDWidth == 8);

        uint32_t const dummyIndex = childCount;
        for (int32_t mask = 0; mask < 8; ++mask)
        {
            Vector3f dir{};
            dir.x = (mask & 1) ? -1.f : 1.f;
            dir.y = (mask & 2) ? -1.f : 1.f;
            dir.z = (mask & 4) ? -1.f : 1.f;

            std::array<std::pair<double, int>, 8> kv{};
            for (uint32_t i = 0; i < 8; ++i)
            {
                if (children[i].isEmpty())
                    kv[i] = {fl::infinity(), i};
                else
                {
                    double const nearCornerx = (dir.x > 0.f) ? children[i].pMin.x : children[i].pMax.x;
                    double const nearCornery = (dir.y > 0.f) ? children[i].pMin.y : children[i].pMax.y;
                    double const nearCornerz = (dir.z > 0.f) ? children[i].pMin.z : children[i].pMax.z;

                    double key = nearCornerx * dir.x + nearCornery * dir.y + nearCornerz * dir.z;
                    if (!std::isfinite(key))
                        key = std::isnan(key) ? fl::infinity() : key;

                    kv[i] = {key, i};
                }
            }

            std::stable_sort(kv.begin(), kv.end(), [](auto const& a, auto const& b) {
                if (a.first < b.first)
                    return true;
                if (a.first > b.first)
                    return false;
                return a.second < b.second;
            });

            uint32_t packed = 0;
            for (int32_t slot = 0; slot < 8; ++slot)
            {
                int32_t const  idx   = kv[slot].second;
                uint32_t const field = (static_cast<uint32_t>(idx) & 0x7u);
                packed |= (field << (slot * 3));
            }
            outPerm[mask] = packed;
        }
    }

    // --- Two-pass builder ---
    struct IndexMapPair
    {
        uint32_t index;
        uint8_t  isLeaf;
    };

    // Helper: walk tree and assign node -> finalBufferIndex (DFS, preserves DFS order)
    static uint32_t assignNodeIndicesDFS(BVHBuildNode* root, std::unordered_map<BVHBuildNode*, IndexMapPair>& outIndexMap)
    {
        uint32_t                   counter = 0;
        std::vector<BVHBuildNode*> st;
        st.push_back(root);

        // we will do a pre-order DFS where we assign index when visiting the node
        while (!st.empty())
        {
            BVHBuildNode* node = st.back();
            st.pop_back();

            // assign index
            outIndexMap[node].index  = counter++;
            outIndexMap[node].isLeaf = node->childCount == 0 ? 0xff : 0;
            // push children in reverse so child 0 processed first (optional)
            for (int i = (int)node->childCount - 1; i >= 0; --i)
                st.push_back(node->children[i]);
        }
        return counter;
    }

    // Build the BVH WiVe clusters in two passes
    BVHWiVeCluster* buildBVHWive(BVHBuildNode*              root,
                                 uint32_t*                  pnodeCount,
                                 std::pmr::memory_resource* _temp,
                                 std::pmr::memory_resource* memory)
    {
        if (!root)
            return nullptr;

        // PASS 1: assign indices to nodes (DFS order)
        std::unordered_map<BVHBuildNode*, IndexMapPair> indexMap;
        indexMap.reserve(4096);
        uint32_t nodeCount = assignNodeIndicesDFS(root, indexMap);
        if (pnodeCount)
            *pnodeCount = nodeCount;

        // We will allocate one extra slot for the global dummy cluster at the very end
        uint32_t const dummyClusterIndex = nodeCount; // global index of dummy cluster
        uint32_t       totalClusters     = nodeCount + 1u;

        // allocate buffer
        auto* buffer = reinterpret_cast<BVHWiVeCluster*>(
            memory->allocate(totalClusters * sizeof(BVHWiVeCluster), alignof(BVHWiVeCluster)));
        if (!buffer)
            return nullptr;
        std::memset(buffer, 0, totalClusters * sizeof(BVHWiVeCluster));

        // Initialize the global dummy cluster at the last index: set lanes to empty bounds
        {
            BVHWiVeCluster& d = buffer[dummyClusterIndex];
            for (int i = 0; i < 8; ++i)
            {
                d.bxmin[i]       = fl::infinity();
                d.bxmax[i]       = -fl::infinity();
                d.bymin[i]       = fl::infinity();
                d.bymax[i]       = -fl::infinity();
                d.bzmin[i]       = fl::infinity();
                d.bzmax[i]       = -fl::infinity();
                d.slotEntries[i] = 0; // no children
            }
        }

        // PASS 2: build clusters using the known indices
        // We'll traverse the tree again in DFS order, using the same assignment as pass1 (pre-order).
        std::pmr::vector<BVHBuildNode*> st{_temp};
        st.push_back(root);

        while (!st.empty())
        {
            BVHBuildNode* node = st.back();
            st.pop_back();

            uint32_t        nodeIndex = indexMap[node].index;
            BVHWiVeCluster& out       = buffer[nodeIndex];

            // copy children bounds into lanes; pad with bbEmpty()
            uint32_t childCount = node->childCount;
            if (childCount > 0)
            {
                Bounds3f childrenBounds[8];
                for (uint32_t i = 0; i < childCount; ++i)
                {
                    childrenBounds[i] = node->children[i]->bounds;
                    // also push children onto the stack so we process them in the same DFS order
                    st.push_back(node->children[i]);
                }
                for (uint32_t i = childCount; i < 8; ++i)
                    childrenBounds[i] = bbEmpty();

                // copy lane-wise arrays (x min/max, etc.)
                for (uint32_t i = 0; i < 8; ++i)
                {
                    out.bxmin[i] = childrenBounds[i].pMin.x;
                    out.bxmax[i] = childrenBounds[i].pMax.x;
                    out.bymin[i] = childrenBounds[i].pMin.y;
                    out.bymax[i] = childrenBounds[i].pMax.y;
                    out.bzmin[i] = childrenBounds[i].pMin.z;
                    out.bzmax[i] = childrenBounds[i].pMax.z;
                }

                // build the 8 permutations for this node (one per mask), the low 24-bit word per mask
                std::array<uint32_t, 8> perms24{};
                buildPermutationsBoxproj(childrenBounds, childCount, perms24);

                // Now compute global child offsets for each lane index 0..7.
                // For a packed permutation (per mask) that contains lane indices (0..7),
                // we must translate laneIndex -> global cluster index.
                // For lanes >= childCount (padding), we point them to the dummy cluster global index.
                uint32_t laneGlobalIndex[8]{};
                uint8_t  laneGlobalIsLeaf[8]{};
                for (uint32_t lane = 0; lane < 8; ++lane)
                {
                    if (lane < childCount)
                    {
                        BVHBuildNode* childNode = node->children[lane];
                        laneGlobalIndex[lane]   = indexMap[childNode].index;
                        laneGlobalIsLeaf[lane]  = indexMap[childNode].isLeaf;
                    }
                    else
                    {
                        laneGlobalIndex[lane]  = dummyClusterIndex;
                        laneGlobalIsLeaf[lane] = 0xfe;
                    }
                }

                for (int mask = 0; mask < 8; ++mask)
                {
                    uint32_t packed24 = perms24[mask] & 0x00FF'FFFFu; // low 24 bits
                    assert((perms24[mask] & 0x00FF'FFFFu) == packed24);
                    for (int slot = 0; slot < 8; ++slot)
                    {
                        uint32_t slotIndex3bit = (packed24 >> (slot * 3)) & 0x7u; // 0..7
                        uint32_t childGlobal   = laneGlobalIndex[slotIndex3bit];  // translate lane -> global cluster
                        // isLeaf (8 bit) + offset (32 bit) + permLow24 (24 bit)
                        uint64_t entry = ((uint64_t)childGlobal << 24) | (uint64_t)packed24;
                        entry |= (((uint64_t)laneGlobalIsLeaf[slotIndex3bit]) << 56);

                        out.slotEntries[slot] = entry; // example: storing last mask's entries (adapt as needed)
                    }
                }
            }
            else // LEAF NODE
            {
                assert(node->primitiveCount > 0);
                assert(dynamic_cast<TriangleIndexed const*>(node->primitives[0]));
                BVHWiVeLeaf&   out         = reinterpret_cast<BVHWiVeLeaf*>(buffer)[nodeIndex];
                Scene const*   scene       = dynamic_cast<TriangleIndexed const*>(node->primitives[0])->scene;
                uint32_t const instanceIdx = static_cast<uint32_t>(
                    dynamic_cast<TriangleIndexed const*>(node->primitives[0])->instanceIdx);
                TriangleMesh const& mesh  = *scene->geometry[scene->instances[instanceIdx]->meshIdx];
                Transform           xform = transformFromAffine(scene->instances[instanceIdx]->affineTransform);

                out.instanceIdx = instanceIdx;
                out.triCount    = static_cast<uint8_t>(node->primitiveCount);

                for (uint32_t tri = 0; tri < out.triCount; ++tri)
                {
                    TriangleIndexed const* triangle = dynamic_cast<TriangleIndexed const*>(node->primitives[tri]);
                    assert(triangle && scene == triangle->scene && instanceIdx == triangle->instanceIdx);
                    out.triIdx[tri] = triangle->triIdx;
                    auto index      = mesh.getIndexedTri(triangle->triIdx);
                    out.v0s[tri]    = xform(mesh.getPosition(index.v[0].positionIdx));
                    out.v1s[tri]    = xform(mesh.getPosition(index.v[1].positionIdx));
                    out.v2s[tri]    = xform(mesh.getPosition(index.v[2].positionIdx));
                }
            }
        }

        return buffer;
    }

    static DMT_FORCEINLINE void swap_avx(__m256& a, __m256& b)
    {
        __m256 tmp = a;
        a          = b;
        b          = tmp;
    }

    static DMT_FORCEINLINE __m256i expand_mask_i32(int32_t packedMask)
    {
        uint8_t idx8[8];
        for (int i = 0; i < 8; i++)
        {
            idx8[i] = (packedMask >> (3 * i)) & 0x7; // isolate 3 bits
        }
        // Load 8 bytes into low 64 bits of XMM
        __m128i m8 = _mm_loadl_epi64((__m128i const*)idx8);
        // Zero-extend to 8x 32-bit ints
        return _mm256_cvtepu8_epi32(m8);
    }

    static std::array<__m256i, 256> initPermTable()
    {
        std::array<__m256i, 256> permTable{};
        for (int m = 0; m < 256; ++m)
        {
            int idxs[8];
            int p = 0;
            // push indices of set bits in low-to-high lane order (0..7)
            for (int i = 0; i < 8; ++i)
            {
                if (m & (1 << i))
                    idxs[p++] = i;
            }
            // fill remaining slots with a safe value (e.g. 0). They will be ignored
            // because we will only process 'p' lanes afterward.
            for (; p < 8; ++p)
                idxs[p] = 0;

            // _mm256_setr_epi32 would be ideal (sets from low->high) but many toolchains
            // expose only _mm256_set_epi32 (high->low). Use setr if available:
            permTable[m] = _mm256_setr_epi32(idxs[0], idxs[1], idxs[2], idxs[3], idxs[4], idxs[5], idxs[6], idxs[7]);
        }
        return permTable;
    }

    static std::array<__m256i, 256> s_permTable = initPermTable();

    struct StackEntry
    {
        BVHWiVeCluster const* node;
        bool                  isInner;
        float                 tmin; // nearest possible t for this entry
        // optionally store other metadata if needed
    };

    bool traverseRay(Ray const&                 ray,
                     BVHWiVeCluster const*      bvh,
                     uint32_t                   nodeCount,
                     uint32_t*                  instanceIdx,
                     size_t*                    triIdx,
                     triangle::Triisect*        outTri,
                     std::pmr::memory_resource* temp)
    {
        std::pmr::vector<StackEntry> stack{temp};
        stack.reserve(64);

        __m256 const  rayIDirX  = _mm256_set1_ps(ray.d_inv.x);
        __m256 const  rayIDirY  = _mm256_set1_ps(ray.d_inv.y);
        __m256 const  rayIDirZ  = _mm256_set1_ps(ray.d_inv.z);
        __m256 const  rayOrgX   = _mm256_set1_ps(ray.o.x);
        __m256 const  rayOrgY   = _mm256_set1_ps(ray.o.y);
        __m256 const  rayOrgZ   = _mm256_set1_ps(ray.o.z);
        uint8_t const raySignX  = ray.d.x < 0.f;
        uint8_t const raySignY  = ray.d.y < 0.f;
        uint8_t const raySignZ  = ray.d.z < 0.f;
        uint8_t const signIndex = (raySignZ << 2) | (raySignY << 1) | raySignX;

        float const initial_tNear = 1e-5f;
        float       scalar_tFar   = 1e5f; // scalar cutoff updated from leaf hits
        __m256      tNear         = _mm256_set1_ps(initial_tNear);
        __m256      tFarVec       = _mm256_set1_ps(scalar_tFar); // will be updated when isect changes

        int const     scale  = sizeof(uint64_t);
        __m256i const vindex = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

        // current best intersection
        auto isect = triangle::Triisect::nothing();
        isect.t    = scalar_tFar;

        // start stack with root
        stack.push_back({&bvh[0], true, 0.0f});

        while (!stack.empty())
        {
            // pop an entry (LIFO)
            StackEntry entry = stack.back();
            stack.pop_back();

            // prune early if entry's nearest possible t is >= current best hit
            if (entry.tmin >= isect.t)
                continue;

            if (entry.isInner)
            {
                BVHWiVeCluster const* current = entry.node;

                // load cluster AABB arrays (8 lanes)
                __m256 bxmin = _mm256_loadu_ps(current->bxmin);
                __m256 bxmax = _mm256_loadu_ps(current->bxmax);
                __m256 bymin = _mm256_loadu_ps(current->bymin);
                __m256 bymax = _mm256_loadu_ps(current->bymax);
                __m256 bzmin = _mm256_loadu_ps(current->bzmin);
                __m256 bzmax = _mm256_loadu_ps(current->bzmax);

                if (raySignX)
                    swap_avx(bxmin, bxmax);
                if (raySignY)
                    swap_avx(bymin, bymax);
                if (raySignZ)
                    swap_avx(bzmin, bzmax);

                // use current scalar_tFar as broadcasted vector (we update tFarVec below when isect changes)
                tFarVec = _mm256_set1_ps(isect.t);

                __m256 const txMin = _mm256_max_ps(_mm256_mul_ps(_mm256_sub_ps(bxmin, rayOrgX), rayIDirX), tNear);
                __m256 const txMax = _mm256_min_ps(_mm256_mul_ps(_mm256_sub_ps(bxmax, rayOrgX), rayIDirX), tFarVec);
                __m256 const tyMin = _mm256_max_ps(_mm256_mul_ps(_mm256_sub_ps(bymin, rayOrgY), rayIDirY), tNear);
                __m256 const tyMax = _mm256_min_ps(_mm256_mul_ps(_mm256_sub_ps(bymax, rayOrgY), rayIDirY), tFarVec);
                __m256 const tzMin = _mm256_max_ps(_mm256_mul_ps(_mm256_sub_ps(bzmin, rayOrgZ), rayIDirZ), tNear);
                __m256 const tzMax = _mm256_min_ps(_mm256_mul_ps(_mm256_sub_ps(bzmax, rayOrgZ), rayIDirZ), tFarVec);

                // sign-order permutation for this cluster
                uint32_t packedEntries = static_cast<uint32_t>(current->slotEntries[signIndex]) & 0x00ff'ffff;
                __m256i  perm          = expand_mask_i32(static_cast<int32_t>(packedEntries));

                // apply sign permutation to tMin/tMax
                __m256 tMin_p = _mm256_permutevar8x32_ps(_mm256_max_ps(_mm256_max_ps(txMin, tyMin), tzMin), perm);
                __m256 tMax_p = _mm256_permutevar8x32_ps(_mm256_min_ps(_mm256_min_ps(txMax, tyMax), tzMax), perm);

                // intersection mask and compress to low lanes
                __m256 cmpMaskVec = _mm256_cmp_ps(tMin_p, tMax_p, _CMP_LT_OQ);
                int    mask8      = _mm256_movemask_ps(cmpMaskVec);
                if (mask8 == 0)
                    continue;

                __m256i packPerm = s_permTable[mask8];

                __m256 packed_tMin = _mm256_permutevar8x32_ps(tMin_p, packPerm);
                // __m256 packed_tMax = _mm256_permutevar8x32_ps(tMax_p, packPerm); // unused here, but could be used

                // gather node offsets and flags (these gathers fetch 8 x int32 entries)
                int const* baseOff = reinterpret_cast<int const*>(
                    reinterpret_cast<unsigned char const*>(current->slotEntries) + 3);
                __m256i gathered_node = _mm256_i32gather_epi32(baseOff, vindex, scale);
                __m256i node_perm     = _mm256_permutevar8x32_epi32(gathered_node, perm); // sign-ordered nodes
                __m256i packed_node   = _mm256_permutevar8x32_epi32(node_perm, packPerm); // compress active lanes

                int const* flagOff = reinterpret_cast<int const*>(
                    reinterpret_cast<unsigned char const*>(current->slotEntries) + 4);
                __m256i gathered_flags = _mm256_i32gather_epi32(flagOff, vindex, scale);
                __m256i flags_perm     = _mm256_srli_epi32(_mm256_permutevar8x32_epi32(gathered_flags, perm), 24);
                __m256i packed_flags   = _mm256_permutevar8x32_epi32(flags_perm, packPerm);

                // store packed arrays to read lanes cheaply
                alignas(32) int   nodesArr[8];
                alignas(32) int   flagsArr[8];
                alignas(32) float tminArr[8];
                _mm256_store_si256((__m256i*)nodesArr, packed_node);
                _mm256_store_si256((__m256i*)flagsArr, packed_flags);
                _mm256_store_ps(tminArr, packed_tMin);

                int k = _mm_popcnt_u64(static_cast<int64_t>(mask8)); // number active

                // push in reverse order so that lane 0 (nearest) is on top of stack after all pushes:
                for (int lane = k - 1; lane >= 0; --lane)
                {
                    int   nodeVal      = nodesArr[lane];        // node index (or pointer offset)
                    int   flagVal      = flagsArr[lane] & 0xff; // flags stored in low byte
                    bool  childIsInner = (flagVal == 0);        // adapt to your flags scheme
                    float childTmin    = tminArr[lane];

                    // small sanity: if childTmin >= current best, skip pushing it
                    if (childTmin >= isect.t)
                        continue;

                    // push the child entry (node pointer resolved from gathered nodeVal)
                    // Here we assume nodeVal is an index into bvh[].
                    BVHWiVeCluster const* childPtr = &bvh[nodeVal];
                    stack.push_back({childPtr, childIsInner, childTmin});
                }
            }
            else
            {
                // leaf node processing
                BVHWiVeLeaf const* leaf = reinterpret_cast<BVHWiVeLeaf const*>(entry.node);

                uint8_t  remaining = leaf->triCount;
                uint32_t currIdx   = 0;

                // If you have vectorized triangle intersection helpers, use them.
                // Here I assume triangle::intersect4/intersect exist and return Triisect
                // with .hit boolean and .t and indices.
                while (remaining >= 4)
                {
                    auto sect = triangle::intersect4(ray, isect.t, leaf->v0s + currIdx, leaf->v1s + currIdx, leaf->v2s + currIdx, 0xF);
                    remaining -= 4;
                    currIdx += 4;
                    if (sect && sect.t < isect.t)
                    {
                        isect = sect;
                        // update scalar cutoff
                        scalar_tFar  = isect.t;
                        *instanceIdx = leaf->instanceIdx;
                        *triIdx      = leaf->triIdx[sect.index];
                    }
                }
                while (remaining >= 2)
                {
                    auto sect = triangle::intersect4(ray, isect.t, leaf->v0s + currIdx, leaf->v1s + currIdx, leaf->v2s + currIdx, 0x3);
                    remaining -= 2;
                    currIdx += 2;
                    if (sect && sect.t < isect.t)
                    {
                        isect        = sect;
                        scalar_tFar  = isect.t;
                        *instanceIdx = leaf->instanceIdx;
                        *triIdx      = leaf->triIdx[sect.index];
                    }
                }
                while (remaining > 0)
                {
                    auto sect = triangle::intersect(ray, isect.t, leaf->v0s[currIdx], leaf->v1s[currIdx], leaf->v2s[currIdx], 0);
                    --remaining;
                    ++currIdx;
                    if (sect && sect.t < isect.t)
                    {
                        isect        = sect;
                        scalar_tFar  = isect.t;
                        *instanceIdx = leaf->instanceIdx;
                        *triIdx      = leaf->triIdx[sect.index];
                    }
                }

                // If we found a hit in this leaf, update tFarVec used for AABB tests
                tFarVec = _mm256_set1_ps(scalar_tFar);

                // Optional pruning: pop any stacked entries that can't beat the new isect.t
                // (we'll actually check at top of loop, so explicit prune is optional; but we can drop any strictly worse)
                while (!stack.empty() && stack.back().tmin >= isect.t)
                    stack.pop_back();

                // continue traversal to possibly find a closer hit
            }
        } // while stack

        // After traversal, if we found a hit, write out hit info and return true
        if (isect && isect.t < 1e5f)
        {
            *outTri = isect;
            return true;
        }

        return false;
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