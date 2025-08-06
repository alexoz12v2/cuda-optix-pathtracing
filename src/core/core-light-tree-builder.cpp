#include "core-light-tree-builder.h"

namespace dmt {

    static void DMT_FASTCALL
        directionConesUnion(Vector3f w0, float cosTheta_0, Vector3f w1, float cosTheta_1, Vector3f* w, float* cosTheta)
    {
        // assume both cones are valid
        assert(fl::abs(normL2(w0) - 1.f) < 1e-5f && cosTheta_0 >= -1.f && cosTheta_0 <= 1.f);
        assert(fl::abs(normL2(w1) - 1.f) < 1e-5f && cosTheta_1 >= -1.f && cosTheta_1 <= 1.f);
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

    float energyVarianceFromLight(Light const& light)
    {
        float var = fl::infinity();
        if (light.type == LightType::ePoint)
        {
            // we sample from uniform sphere if we had a transmission (f(w) = 1/4pi) and cosWeighted hemisphere (f(w) = z/pi for z>=0) otherwise
            // the former has variance of 1, while the latter has variance of 0.611, hence take the higest to be conservative
        }
        else if (light.type == LightType::eSpot)
        {
        }
        else if (light.type == LightType::eMesh)
        {
            assert(false && "not yet implemented!");
        }
        else
        {
            assert(false && "invalid light type!");
        }
        return var;
    }
} // namespace dmt