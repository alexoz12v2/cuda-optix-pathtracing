#include "core-light-tree-builder.h"

namespace dmt {

    static void DMT_FASTCALL
        directionConesUnion(Vector3f w0, float cosTheta_0, Vector3f w1, float cosTheta_1, Vector3f* w, float* cosTheta)
    {
        // assume both cones are valid (or not w = 0 means empty bounds (TEST))
        // assert(fl::abs(normL2(w0) - 1.f) < 1e-5f && cosTheta_0 >= -1.f && cosTheta_0 <= 1.f);
        // assert(fl::abs(normL2(w1) - 1.f) < 1e-5f && cosTheta_1 >= -1.f && cosTheta_1 <= 1.f);
        assert(w && cosTheta);

        // if one cone is inside another, then return the outer one
        float const theta_0 = fl::safeacos(cosTheta_0), theta_1 = fl::safeacos(cosTheta_1);
        float const theta_diff = angleBetween(w0, w1);
        if (fminf(theta_diff + theta_1, fl::pi()) <= theta_0) // 1 inside 0
        {
            *w = w0, *cosTheta = cosTheta_0;
            return;
        }
        if (fminf(theta_diff + theta_0, fl::pi()) <= theta_1) // 0 inside 1
        {
            *w = w1, *cosTheta = cosTheta_1;
            return;
        }

        // combined spread angle = angleBetweenVectors + Theta0 + Theta1
        float const    theta_combined = (theta_0 + theta_diff + theta_1) * 0.5f;
        Vector3f const wRotate        = normalize(cross(w0, w1));
        if (theta_combined >= fl::pi() || dotSelf(wRotate) == 0) // return whole sphere
        {
            *w = {0, 0, 1}, *cosTheta = -1;
            return;
        }

        // combined cone axis = rotate by spread angle - one of the two theta,
        // with respect to the cross axis, one of the two normal axes
        float const thetaRotate = theta_combined - theta_0;
        Quaternion  rotateQuat  = fromRadians(thetaRotate, wRotate);

        *w        = normalize(rotate(rotateQuat, w0));
        *cosTheta = cosf(theta_combined);
    }

    // -- `LightBounds` Methods --
    LightBounds lbUnion(LightBounds const& lb0, LightBounds const& lb1)
    {
        LightBounds lb{};
        // compute the union of the spatial bounds
        lb.bounds = bbUnion(lb0.bounds, lb1.bounds);

        // compute the union of the two directional cones
        directionConesUnion(lb0.w, lb0.cosTheta_o, lb1.w, lb1.cosTheta_o, &lb.w, &lb.cosTheta_o);

        // conservative approach: The falloff angle is the maximum one (minimum cosine)
        lb.cosTheta_e = fmaxf(lb0.cosTheta_e, lb1.cosTheta_e);

        lb.phi = lb0.phi + lb1.phi;

        return lb;
    }

    static LightBounds lbEmpty()
    {
        LightBounds lb{};
        lb.bounds = bbEmpty();
        return lb;
    }

    static float DMT_FASTCALL cosSubClamped(float sinTheta_a, float cosTheta_a, float sinTheta_b, float cosTheta_b)
    {
        if (cosTheta_a > cosTheta_b)
            return 1;
        return cosTheta_a * cosTheta_b + sinTheta_a * sinTheta_b;
    }

    static float DMT_FASTCALL sinSubClamped(float sinTheta_a, float cosTheta_a, float sinTheta_b, float cosTheta_b)
    {
        if (cosTheta_a > cosTheta_b)
            return 0;
        return sinTheta_a * cosTheta_b - cosTheta_a * sinTheta_b;
    }

    static std::pair<float, float> sinCosThetaBoundsSubtended(Bounds3f const& b, Point3f p)
    { // compute boundsing sphere
        float   radius = 0;
        Point3f center{};
        b.boundingSphere(center, radius);
        float const radius2 = radius * radius;
        float const dist2   = dot(center, p);
        if (dist2 < radius2) // if inside sphere then you see pi
            return {0.f, -1.f};

        float const sin2ThetaMax = radius2 / dist2;
        return std::make_pair(fl::sqrt(sin2ThetaMax), fl::safeSqrt(1.f - sin2ThetaMax));
    }

    DMT_FASTCALL float lbImportance(LightBounds const& lb0, Point3f p, Normal3f n)
    {
        assert(fl::abs(fl::normL2(n) - 1.f) < 1e-5f);

        Point3f const  pc         = lb0.bounds.centroid();
        Vector3f const wi         = normalize(p - pc);
        float const    sinTheta_o = fl::safeSqrt(1.f - lb0.cosTheta_o * lb0.cosTheta_o);

        // compute clamped dsquared distance from reference point = max between distSquared and halfDiag
        float const distSqr = fmaxf(dotSelf(p - pc), normL2(lb0.bounds.diagonal()) * 0.5f);

        // compute sine and cosine of angle between normal cone axis and vec(ref point -> bounds centroid)
        float cosTheta_w = dot(wi, lb0.w);
        if (lb0.twoSided)
            cosTheta_w = fl::abs(cosTheta_w);
        float const sinTheta_w = fl::safeSqrt(1.f - fl::sqr(cosTheta_w));

        // compute sine and cosine for angle of bounds
        auto const [sinTheta_b, cosTheta_b] = sinCosThetaBoundsSubtended(lb0.bounds, p);

        // compute cos(theta_w - theta_o - theta_b) (see paper illustrations)
        float const cosTheta_wo = cosSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, lb0.cosTheta_o);
        float const sinTheta_wo = sinSubClamped(sinTheta_w, cosTheta_w, sinTheta_o, lb0.cosTheta_o);
        float const cosTheta_p  = cosSubClamped(sinTheta_wo, cosTheta_wo, sinTheta_b, cosTheta_b);
        // if outside angular falloff, then no importance
        if (cosTheta_p <= lb0.cosTheta_e)
            return 0;

        // further decrease the importance as the angle between the reference point normal and light cone normal axis increase
        float const cosTheta_i  = absDot(wi, n);
        float const sinTheta_i  = fl::safeSqrt(1.f - cosTheta_i * cosTheta_i);
        float const cosTheta_ib = cosSubClamped(sinTheta_i, cosTheta_i, sinTheta_b, cosTheta_b);

        // lambert's cosine law + square attenuation to estimate importance
        float const importance = fmaxf(lb0.phi * cosTheta_ib * cosTheta_p / distSqr, 0);
        return importance;
    }

    // -- `LightBounds` Factory Methods, for each finite position light type --
    LightBounds makeLBFromLight(Light const& light)
    {
        LightBounds lb{};
        if (light.type == LightType::ePoint)
        {
            Vector3f const halfExtent = Vector3f::s(light.data.point.radius);
            float const    phi        = 4 * fl::pi() * light.strength.max() * light.data.point.evalFac;

            lb.bounds     = makeBounds(light.co - halfExtent, light.co + halfExtent);
            lb.phi        = phi;
            lb.twoSided   = false;
            lb.w          = {0, 0, 1};
            lb.cosTheta_o = -1; // cos(pi)
            lb.cosTheta_e = 0;  // cos(pi / 2)
        }
        else if (light.type == LightType::eSpot)
        {
            assert(fl::abs(normL2(light.data.spot.direction) - 1.f) < 1e-5f &&
                   fl::abs(light.data.spot.cosHalfLargerSpread) < 1.f && fl::abs(light.data.spot.cosHalfSpotAngle) < 1.f);
            Vector3f const halfExtent = Vector3f::s(light.data.spot.radius);
            float const    phi        = 4 * fl::pi() * light.data.spot.evalFac * light.strength.max();
            float const    cosTheta_e = cosf(
                fl::safeacos(light.data.spot.cosHalfLargerSpread) - fl::safeacos(light.data.spot.cosHalfSpotAngle));

            lb.bounds     = makeBounds(light.co - halfExtent, light.co + halfExtent);
            lb.phi        = phi;
            lb.twoSided   = false;
            lb.w          = light.data.spot.direction;
            lb.cosTheta_o = light.data.spot.cosHalfSpotAngle;
            lb.cosTheta_e = cosTheta_e;
        }
        else if (light.type == LightType::eMesh)
        {
            assert(false && "Not yet implemented!");
        }
        else
        {
            assert(false && "Invalid Light type for Bounds!");
        }

        return lb;
    }

    // -- LightTreeBuildNode utils --
    DMT_FASTCALL float adaptiveSplittingHeuristic(LightTreeBuildNode const& node, Point3f p)
    {
        if (node.leaf)
            return 1.f;

        // intersect line (point - centroid) with 2 planes of the AABB. the two intersections are a (tMin) and b (tMax)
        // alternative: approximate AABB with a sphere bounds, therefore a/b = sqrt(distance squared clamped) -/+ haldDiagonal of AABB
        Point3f const  pc       = node.lb.bounds.centroid();
        Vector3f const wi       = normalize(p - pc);
        float const    halfDiag = normL2(node.lb.bounds.diagonal()) * 0.5f;
        float const    dist     = fl::safeSqrt(fmaxf(dotSelf(p - pc), halfDiag));

        float const a = fmaxf(dist - halfDiag, 0.f);
        float const b = dist + halfDiag;

        float gExpected2 = 0.f;
        float gVariance  = 0.f;
        if (a > 0 && b > 0)
        {
            float const a3 = a * a * a;
            float const b3 = b * b * b;

            float const a_minus_b   = a - b;
            float const a3_minus_b3 = a_minus_b * (a * a + a * b + b * b);

            gExpected2 = fl::sqr(fl::rcp(a * b));
            gVariance  = a3_minus_b3 / (3 * a_minus_b * a3 * b3) - gExpected2;
        }

        float const eExpected2 = fl::sqr(node.lb.phi / node.numEmitters);
        float const eVariance  = node.varPhi;

        float const sigma2 = (eVariance * gVariance + eVariance * gExpected2 + eExpected2 * gVariance) *
                             fl::sqr(node.numEmitters);
        float const probSigma = std::pow(fmaxf(fl::rcp(1 + fl::sqrt(sigma2)), 0.f), 0.25f);
        assert(probSigma >= 0.f && probSigma <= 1.f);
        return probSigma;
    }

    /// regularization factor to favour less thin boxes
    DMT_FORCEINLINE static float lightTreeBounds_Kr(Bounds3f const& b, int32_t axis)
    {
        return maxComponent(b.diagonal()) / b.diagonal()[axis];
    }

    /// spatial cost function factor
    DMT_FORCEINLINE static float lightTreeBounds_Ma(Bounds3f const& b) { return b.surfaceArea(); }

    /// orientation cost function factor
    DMT_FORCEINLINE static float lightTreeBounds_Momega(float cosTheta_e, float cosTheta_o)
    {
        assert(cosTheta_o >= cosTheta_e);
        float const theta_e = fl::safeacos(cosTheta_e);
        float const theta_o = fl::safeacos(cosTheta_o);
        float const theta_w = fminf(theta_o + theta_e, fl::pi());

        float const sinTheta_o    = sinf(theta_o);
        float const cosTheta_diff = sinf(theta_o - 2 * theta_w);

        float const Momega = fl::twoPi() * (1 - cosTheta_o) +
                             fl::piOver2() *
                                 (2 * theta_w * sinTheta_o - cosTheta_diff - 2 * theta_o * sinTheta_o + cosTheta_o);
        return Momega;
    }

    DMT_FASTCALL float summedAreaOrientationHeuristic(
        LightBounds const& lbLeft,
        LightBounds const& lbRight,
        float              parent_Kr,
        float              parent_Ma,
        float              parent_Momega)
    {
        float const Ma_L = lightTreeBounds_Ma(lbLeft.bounds);
        float const Ma_R = lightTreeBounds_Ma(lbRight.bounds);

        float const Momega_L = lightTreeBounds_Momega(lbLeft.cosTheta_e, lbLeft.cosTheta_o);
        float const Momega_R = lightTreeBounds_Momega(lbRight.cosTheta_e, lbRight.cosTheta_o);

        float const cost = parent_Kr * (lbLeft.phi * Ma_L * Momega_L + lbRight.phi * Ma_R * Momega_R) /
                           (parent_Ma * parent_Momega);
        return cost;
    }

    static LightBounds lbUnionAll(std::span<Light const> lights)
    {
        LightBounds lb = makeLBFromLight(lights[0]);
        for (size_t i = 1; i < lights.size(); ++i)
            lb = lbUnion(makeLBFromLight(lights[i]), lb);
        return lb;
    }

    static LightTreeBuildNode makeLeaf(Light const* ptr, uint32_t idx)
    {
        assert(ptr);
        LightTreeBuildNode node{};
        node.leaf             = 1;
        node.data.emitter.idx = idx;
        node.data.emitter.ptr = ptr;
        node.lb               = makeLBFromLight(*ptr);

        return node;
    }

    static LightTreeBuildNode makeInterior()
    {
        LightTreeBuildNode node{};
        return node;
    }

    static size_t lightTreeBuildRecursive(
        std::span<Light>                      lights,
        uint32_t                              start,
        uint32_t                              end,
        uint32_t                              trail,
        uint32_t                              tzcount,
        std::pmr::vector<LightTreeBuildNode>* nodes,
        std::pmr::vector<LightTrailPair>*     bitTrails,
        std::pmr::memory_resource*            temp)
    {
        assert(lights.size() > 0);
        LightTreeBuildNode& current = nodes->back();
        if (lights.size() == 1)
        {
            assert(start == end + 1);
            current = makeLeaf(lights.data(), start);
            bitTrails->emplace_back(start, trail);

            return 1;
        }
        else // Binning
        {
            int32_t const axis     = current.lb.bounds.maxDimention();
            float const   splitLen = current.lb.bounds.diagonal()[axis] / LightTreeNumBins;

            float const Kr     = lightTreeBounds_Kr(current.lb.bounds, axis);
            float const Ma     = lightTreeBounds_Ma(current.lb.bounds);
            float const Momega = lightTreeBounds_Momega(current.lb.cosTheta_e, current.lb.cosTheta_o);

            LightBounds lbLeft = lbEmpty(), lbRight = lbEmpty();
            float       minSplitPos = 0.f;
            float       minCost     = fl::infinity();
            for (uint32_t i = 1; i < LightTreeNumBins - 1; ++i)
            {
                float const splitPos  = static_cast<float>(i) * splitLen;
                LightBounds lbLeftTmp = lbEmpty(), lbRightTmp = lbEmpty();
                for (Light const& light : lights)
                {
                    if (light.co[axis] < splitPos) // left
                        lbLeftTmp = lbUnion(lbLeftTmp, makeLBFromLight(light));
                    else
                        lbRightTmp = lbUnion(lbRightTmp, makeLBFromLight(light));
                }

                if (float cost = summedAreaOrientationHeuristic(lbLeftTmp, lbRightTmp, Kr, Ma, Momega); cost < minCost)
                {
                    minCost     = cost;
                    minSplitPos = splitPos;
                    lbLeft      = lbLeftTmp;
                    lbRight     = lbRightTmp;
                }
            }
            assert(minSplitPos != 0);

            Light*   midPtr = std::partition(lights.data() + start,
                                           lights.data() + end,
                                           [axis, s = minSplitPos](Light const& a) { return a.co[axis] < s; });
            uint32_t mid    = midPtr - (lights.data() + start);

            if (mid == start)
            {
                ++mid;
                assert(mid <= end);
            }

            // recursion
            nodes->emplace_back(makeInterior());
            LightTreeBuildNode& left      = nodes->back();
            uint32_t const      leftTrail = trail;
            assert(leftTrail > trail);
            current.data.children[0] = &left;
            current.numEmitters += lightTreeBuildRecursive(lights, start, mid, leftTrail, ++tzcount, nodes, bitTrails, temp);

            if (mid != end)
            {
                nodes->emplace_back(makeInterior());
                LightTreeBuildNode& right      = nodes->back();
                uint32_t const      rightTrail = trail | (1u << tzcount);
                current.data.children[1]       = &left;
                current.numEmitters += lightTreeBuildRecursive(lights, mid, end, rightTrail, ++tzcount, nodes, bitTrails, temp);
            }
            else
            {
                assert(!current.data.children[1]);
            }

            return current.numEmitters;
        }
    }

    static float sampleVariance(std::pmr::vector<float> const& phis)
    {
        assert(phis.size() > 1);
        float mean     = 0.f;
        float variance = 0.f;

        for (float f : phis)
            mean += f;
        mean /= phis.size();

        for (float f : phis)
            variance += fl::sqr(f - mean);
        variance /= (phis.size() - 1);

        return variance;
    }

    static void fillPhisRecursive(LightTreeBuildNode& node, std::pmr::vector<float>* phis)
    {
        assert(phis && !node.leaf && node.data.children[0] && node.data.children[1]);
        if (node.data.children[0]->leaf)
            phis->push_back(node.data.children[0]->lb.phi);
        else
            fillPhisRecursive(*node.data.children[0], phis);

        if (node.data.children[1]->leaf)
            phis->push_back(node.data.children[0]->lb.phi);
        else
            fillPhisRecursive(*node.data.children[1], phis);
    }

    static void lightTreeComputeVariances(std::pmr::vector<LightTreeBuildNode>* nodes, std::pmr::memory_resource* temp)
    {
        // TODO maybe: use a map LightTreeBuildNode -> some phis?
        std::pmr::vector<float> phis{temp};
        phis.reserve((*nodes)[0].numEmitters);
        for (LightTreeBuildNode& node : *nodes)
        {
            if (!node.leaf)
            {
                phis.clear();
                fillPhisRecursive(node, &phis);
                node.varPhi = sampleVariance(phis);
            }
        }
    }

    size_t lightTreeBuild(std::span<Light>                      lights,
                          std::pmr::vector<LightTreeBuildNode>* nodes,
                          std::pmr::vector<LightTrailPair>*     bitTrails,
                          std::pmr::memory_resource*            temp)
    {
        if (nodes->size() != 0 || bitTrails->size() != 0)
            return 0;

        nodes->reserve(64);
        bitTrails->reserve(64);

        // top down pass to construct the whole tree and bit trails
        nodes->emplace_back(makeInterior());
        auto& node         = nodes->back();
        node.lb            = lbUnionAll(lights);
        size_t numEmitters = lightTreeBuildRecursive(lights, 0, lights.size(), 0, 0, nodes, bitTrails, temp);
        assert(numEmitters == lights.size());

        std::sort(bitTrails->begin(), bitTrails->end(), [](LightTrailPair p0, LightTrailPair p1) {
            return p0.lightIdx < p1.lightIdx;
        });

        // bottom up pass to precompute variances (leaf nodes have leaf.lb.phi)
        lightTreeComputeVariances(nodes, temp);
        return numEmitters;
    }

    LightSplit lightTreeAdaptiveSplit(LightTreeBuildNode const& ltRoot, Point3f p, float precision, std::pmr::memory_resource* memory)
    {
        LightSplit split{};
        if (ltRoot.leaf)
        {
            split.count    = 1;
            split.nodes[0] = &ltRoot;
        }
        else
        {
            // breadth-first visit
            std::pmr::vector<LightTreeBuildNode const*> parentStack{memory}, siblingStack{memory};
            parentStack.reserve(64), siblingStack.reserve(64);
            siblingStack.push_back(&ltRoot);
            while ((!siblingStack.empty() && !parentStack.empty()) || split.count < LightTreeMaxSplitSize)
            {
                if (!siblingStack.empty())
                {
                    LightTreeBuildNode const* sibling = siblingStack.back();
                    siblingStack.pop_back();
                    bool enough = adaptiveSplittingHeuristic(*sibling, p) >= precision;
                    if (split.count + 1 == LightTreeMaxSplitSize || enough)
                    {
                        split.nodes[split.count] = sibling;
                        ++split.count;
                        assert(split.count <= LightTreeMaxSplitSize);
                    }
                    else
                    {
                        assert(!enough);
                        if (!sibling->leaf)
                            parentStack.push_back(sibling);
                    }
                }
                else if (!parentStack.empty())
                {
                    LightTreeBuildNode const* parent = parentStack.back();
                    parentStack.pop_back();

                    assert(!parent->leaf && parent->data.children[0] && parent->data.children[1]);
                    siblingStack.push_back(parent->data.children[0]);
                    siblingStack.push_back(parent->data.children[1]);
                }
            }
        }

        return split;
    }

    SelectedLights DMT_FASTCALL selectLightsFromSplit(LightSplit const& lightSplit, Point3f p, Vector3f n, float u, float startPMF)
    {
        assert(fl::abs(normL2(n) - 1.f) < 1e-5f);
        SelectedLights selectedLights{};
        uint32_t       moreLights = lightSplit.count;
        uint32_t       splitIndex = 0;
        while (selectedLights.count < LightTreeMaxSplitSize && moreLights) // I think that moreLights is sufficient
        {
            LightTreeBuildNode const* node        = lightSplit.nodes[splitIndex];
            float                     pmf         = startPMF;
            bool                      pathSampled = true;
            ++splitIndex;
            while (!pathSampled)
            {
                if (!node->leaf)
                {
                    float const importances[2]{lbImportance(node->data.children[0]->lb, p, n),
                                               lbImportance(node->data.children[1]->lb, p, n)};
                    if (importances[0] == 0.f && importances[1] == 0.f)
                        pathSampled = true;
                    else
                    {
                        float   nodePMF     = 0;
                        int32_t chosenChild = sampleDiscrete(importances, 2, u, &nodePMF, &u);
                        assert(chosenChild >= 0 && chosenChild <= 1);
                        pmf *= nodePMF;
                        node = node->data.children[chosenChild];
                    }
                }
                else
                {
                    pathSampled = true;
                    --moreLights;
                    if (lbImportance(node->lb, p, n) > 0.f)
                    {
                        selectedLights.indices[selectedLights.count] = node->data.emitter.idx;
                        selectedLights.pmfs[selectedLights.count]    = pmf;
                        ++selectedLights.count;
                    }
                }
            }
        }

        return selectedLights;
    }

    float lightSelectionPMF(LightTreeBuildNode const& ltRoot, Point3f p, Vector3f n, uint32_t trail, float startPMF)
    {
        assert(fl::abs(normL2(n) - 1.f) < 1e-5f);
        LightTreeBuildNode const* node = &ltRoot;
        while (!node->leaf)
        {
            float const importances[2]{lbImportance(node->data.children[0]->lb, p, n),
                                       lbImportance(node->data.children[1]->lb, p, n)};
            startPMF *= importances[trail & 1] / (importances[0] + importances[1]);
            node = node->data.children[trail & 1];
            trail >>= 1;
        }

        return startPMF;
    }
} // namespace dmt