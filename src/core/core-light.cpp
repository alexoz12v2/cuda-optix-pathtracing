#include "core-light.h"

namespace dmt {

    // -- Factory Methods --
    // ---- Point Light ----
    Light makePointLight(Transform const& t, RGB emission, float radius, float factor)
    {
        Light light{};
        light.lightFromRender = t;

        light.co.x = light.lightFromRender.m.m[12];
        light.co.y = light.lightFromRender.m.m[13];
        light.co.z = light.lightFromRender.m.m[14];

        light.type = LightType::ePoint;

        light.strength = emission;

        light.data.point.evalFac = factor;
        light.data.point.radius  = radius;
        return light;
    }
    // ---- Spot Light ----
    Light makeSpotLight(Transform const& t, RGB emission, float cosTheta0, float cosThetae, float radius, float factor)
    {
        cosTheta0 = fmaxf(cosTheta0, cosThetae);
        cosThetae = fminf(cosTheta0, cosThetae);

        Light light{};
        light.type     = LightType::eSpot;
        light.strength = emission;

        // --- Extract apex position from translation ---
        light.co.x = t.m.m[12];
        light.co.y = t.m.m[13];
        light.co.z = t.m.m[14];

        // --- Light cone direction in world space ---
        Vector3f coneDirWorld = normalize(t(Vector3f{0, 1, 0})); // y_world = default world space direction
        Vector3f zLight       = coneDirWorld;                    // z_local = forward direction in light space

        // --- Construct orthonormal frame for light ---
        Vector3f xLight, yLight;
        if (fabsf(zLight.x) > 0.1f)
            xLight = normalize(cross(Vector3f(0, 1, 0), zLight));
        else
            xLight = normalize(cross(Vector3f(1, 0, 0), zLight));
        yLight = cross(zLight, xLight);

        // --- Build 4x4 matrix mapping world -> light space ---
        // Columns = x, y, z axes; origin = apex
        // Note: This is the inverse of the usual "frame to world" transform
        // clang-format off
        float const arr[16] {
            xLight.x, xLight.y, xLight.z, -dot(xLight, light.co),
            yLight.x, yLight.y, yLight.z, -dot(yLight, light.co),
            zLight.x, zLight.y, zLight.z, -dot(zLight, light.co),
            0.f,      0.f,      0.f,      1.f
        };
        // clang-format on
        light.lightFromRender = Transform{Matrix4f::rowWise(arr)};

        // --- Spot parameters ---
        light.data.spot.direction           = coneDirWorld; // world-space direction
        light.data.spot.radius              = radius;
        light.data.spot.evalFac             = factor;
        light.data.spot.cosHalfSpotAngle    = fl::clamp(cosTheta0, -1.f, 1.f);
        light.data.spot.cosHalfLargerSpread = fl::clamp(cosThetae, -1.f, 1.f);

        return light;
    }

    // ---- Env Light ----
    static PiecewiseConstant2D distributionFromImage(
        RGB const*                 image,
        int32_t                    xRes,
        int32_t                    yRes,
        std::pmr::memory_resource* memory,
        std::pmr::memory_resource* temp)
    {
        Bounds2f const domain = makeBounds({0, 0}, {1, 1});
// TODO AVX
#if 1
        dstd::Array2D<float> func{static_cast<uint32_t>(xRes), static_cast<uint32_t>(yRes), temp};
        for (int32_t y = 0; y < yRes; ++y)
        {
            for (int32_t x = 0; x < xRes; ++x)
            {
                float const value = image[x + y * xRes].avg(); // probably not the best estimate for radiance?
                // assuming jacobian is constant over texel region
                // and equal to 1 no matter where you are in the domain
                func(x, y) = value;
            }
        }
#endif
        return PiecewiseConstant2D{func, domain, memory, temp};
    }

    EnvLight::EnvLight(RGB const*                 image,
                       int32_t                    xRes,
                       int32_t                    yRes,
                       Quaternion                 quat,
                       float                      scale,
                       std::pmr::memory_resource* memory,
                       std::pmr::memory_resource* temp) :
    distrib(distributionFromImage(image, xRes, yRes, memory, temp)),
    lightFromRender(normalize(quat)),
    imageBuffer(image),
    xResolution(xRes),
    yResolution(yRes),
    evalFac(scale)
    {
        assert(imageBuffer && isPOT(xRes) && isPOT(yRes) && xRes == 2 * yRes);
    }

    // -- Methods Light Specific --
    // ---- Point Light ----
    bool pointLightSampleFromContext(Light const& light, Point2f u, Point3f p, Vector3f n, bool hadTransmission, LightSample* sample)
    {
        assert(light.type == LightType::ePoint);
        float const radiusSqr = light.data.point.radius * light.data.point.radius;
        Vector3f    lightN    = p - light.co;
        float const distSqr   = dotSelf(lightN);
        float const dist      = fl::sqrt(distSqr);
        assert(!fl::isInfOrNaN(dist));
        lightN /= dist;

        bool const effectivelyDelta = (light.data.point.radius / dist) < 1e-3f;

        sample->evalFac = light.data.point.evalFac;

        float cosTheta = 0.f;
        if (distSqr > radiusSqr) // outside sphere
        {
            float const oneMinusCos = sin_sqr_to_one_minus_cos(radiusSqr / distSqr);

            sample->d = sampleUniformCone(-lightN, oneMinusCos, u, &cosTheta, &sample->pdf);
            if (effectivelyDelta)
            {
                sample->pdf   = 1.f;
                sample->delta = true;
            }
        }
        else // inside sphere
        {
            if (hadTransmission)
            {
                sample->d   = sampleUniformSphere(u);
                sample->pdf = 1.f / (4.f * fl::pi());
            }
            else
            {
                sample->d = sampleCosHemisphere(n, u, &sample->pdf);
            }
            cosTheta = -dot(sample->d, lightN);
        }

        // law of cosines ( a = b cosGamma +- sqrt(c2 - b2 sin2Gamma) )
        // (you know center - pSurf distance and angle between center - pSurf and pSurf - pLight, and want to know distance)
        sample->t = dist * cosTheta -
                    copysignf(fl::safeSqrt(radiusSqr - distSqr + distSqr * fl::sqr(cosTheta)), distSqr - radiusSqr);

        // remap sampled point onto the sphere to prevent precision issues with small radius
        sample->p  = p + sample->d * sample->t;
        sample->ng = normalize(sample->p - light.co);
        sample->p  = sample->ng * light.data.point.radius + light.co;

        // texture coordinates
        Point2f const uv = mapToSphere(light.lightFromRender(normalFrom(sample->ng)));

        // remap to barycentric coords
        sample->uv[0] = uv.y;
        sample->uv[1] = 1.f - uv.x - uv.y;
        return true;
    }

    bool pointLightIntersect(Light const& light, Ray const& ray, float* t)
    {
        assert(light.type == LightType::ePoint || light.type == LightType::eSpot);
        float const radius = light.type == LightType::ePoint ? light.data.point.radius : light.data.spot.radius;
        if (radius > 0.f)
        {
            Point3f p{};
            return raySphereIntersect(ray.o, ray.d, 0, 1e5f, light.co, radius, &p, t);
        }
        else
        {
            return false;
        }
    }

    // ---- Spot Light ----
    /// transform vector into light#s local coordinate system and invert its z (to go towards cone apex)
    static Vector3f spotLightToLocal(Transform const& lightFromRender, Vector3f _dir)
    {
        Vector3f dir = _dir;
        dir          = normalize(lightFromRender(dir));
        return dir;
    }

    static float spotLightAttenuation(SpotLight const& light, Vector3f rayD)
    {
        // assert(rayD.z > 0); // TODO if negative should return 0 hopefully?
        float cosTheta = rayD.z; // assuming normalized direction in light space
        return smoothstep(light.cosHalfLargerSpread, light.cosHalfSpotAngle, cosTheta);
    }

    static float halfCotHalfSpotAngle(float cosHalfSpotAngle)
    {
        float const sinHalfSpotAngle = fl::safeSqrt(1.f - cosHalfSpotAngle * cosHalfSpotAngle);
        if (sinHalfSpotAngle != 0)
            return 0.5f * cosHalfSpotAngle / sinHalfSpotAngle;
        else
            return fl::infinity();
    }

    static Point2f spotLightUV(Vector3f dir, float halfCotHalfSpotAngle)
    {
        // Ensures that the spot light projects the full image regardless of the spot angle.
        float const factor = halfCotHalfSpotAngle / dir.z;

        // NOTE: Return barycentric coordinates in the same notation as Embree and OptiX.
        Point2f const uv{dir.y * factor + 0.5f, -(dir.x + dir.y) * factor};
        return uv;
    }

    // Samples a direction uniformly on a spherical cap
    // axis: the central direction of the cap (normalized)
    // cosThetaMax: cosine of the maximum polar angle of the cap
    // u: 2D uniform random sample in [0,1)^2
    static Vector3f sampleUniformSphereCap(Vector3f const& axis, float cosThetaMax, Point2f const& u)
    {
        float cosTheta = (1.f - u.x) + u.x * cosThetaMax; // linear interpolation: cosTheta in [cosThetaMax,1]
        float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
        float phi      = 2.f * fl::pi() * u.y;

        // Local spherical coordinates
        float x = sinTheta * cosf(phi);
        float y = sinTheta * sinf(phi);
        float z = cosTheta;

        // Build an orthonormal basis around axis
        Vector3f w = normalize(axis);
        Vector3f uVec, vVec;
        if (fabsf(w.x) > 0.1f)
            uVec = normalize(cross(Vector3f(0, 1, 0), w));
        else
            uVec = normalize(cross(Vector3f(1, 0, 0), w));
        vVec = cross(w, uVec);

        // Transform local coords to world
        return normalize(x * uVec + y * vVec + z * w);
    }


    bool spotLightSampleFromContext(Light const& light, Point2f u, Point3f p, Vector3f n, bool ht, LightSample* sample)
    {
        float const radiusSqr = light.data.spot.radius * light.data.spot.radius;

        Vector3f    lightN  = p - light.co;
        float const distSqr = dotSelf(lightN);
        float const dist    = sqrtf(distSqr);
        lightN /= dist;

        bool const effectivelyDelta = (light.data.spot.radius / dist) < 1e-3f;

        sample->evalFac = light.data.spot.evalFac;

        float cosTheta = 0.f;
        if (distSqr > radiusSqr)
        {
            // if outside sphere
            float const one_minus_cos_half_spot_spread = 1.f - light.data.spot.cosHalfLargerSpread;
            float const one_minus_cos_half_angle       = sin_sqr_to_one_minus_cos(radiusSqr / distSqr);
            if (one_minus_cos_half_angle < one_minus_cos_half_spot_spread)
            {
                // if direction towards apex, sample visible part of the sphere
                sample->d = sampleUniformCone(-lightN, one_minus_cos_half_angle, u, &cosTheta, &sample->pdf);
            }
            else
            {
                // sample spread cone (note: Here we compute also sample->t, hence after the attenuation step we won't compute it again)
                sample->d = sampleUniformCone(-light.data.spot.direction,
                                              one_minus_cos_half_spot_spread,
                                              u,
                                              &cosTheta,
                                              &sample->pdf);
            }
            if (!raySphereIntersect(p,
                                    sample->d,
                                    0.f,
                                    fl::infinity(),
                                    light.co,
                                    light.data.spot.radius,
                                    &sample->p,
                                    &sample->t))
            {
                // direction doesn't intersect light
                return false;
            }
            if (effectivelyDelta)
            {
                sample->pdf   = 1.f;
                sample->delta = true;
            }
        }
        else
        {
            // inside sphere
            if (ht)
            { // if current path had transmission, sample uniform sphere
                sample->d   = sampleUniformSphere(u);
                sample->pdf = 1.f / (4.f * fl::pi());
            }
            else
            { // else sample cosine weighted hemisphere
                sample->d = sampleCosHemisphere(n, u, &sample->pdf);
            }
            cosTheta = -dot(sample->d, lightN);

            // law of cosines (see point light for more detailed comment)
            sample->t = dist * cosTheta *
                        copysignf(fl::safeSqrt(radiusSqr - distSqr + distSqr * fl::sqr(cosTheta)), distSqr - radiusSqr);
            sample->p = p + sample->d * sample->t;
        }

        // attenuation step
        Vector3f const localRayDir = spotLightToLocal(light.lightFromRender, -sample->d);
        if (distSqr > radiusSqr)
            sample->evalFac *= spotLightAttenuation(light.data.spot, localRayDir);
        if (sample->evalFac == 0.f)
            return false;

        // remap sampled point onto sphere to prevent issues with small radii
        sample->ng = normalize(sample->p - light.co);
        sample->p  = sample->ng * light.data.spot.radius + light.co;

        // texture coordinates
        sample->uv = spotLightUV(localRayDir, halfCotHalfSpotAngle(light.data.spot.cosHalfSpotAngle));

        return true;
    }

    bool spotLightIntersect(Light const& light, Ray const& ray, float* t)
    {
        // spot light is basically a one sided point light (excessive angle with respoect to cone is accounted for in sampling)
        if (dot(ray.d, ray.o - light.co) >= 0.f)
            return false;

        return pointLightIntersect(light, ray, t);
    }

    // -- Methods --
    bool lightSampleFromContext(Light const& light, LightSampleContext const& lsCtx, Point2f u, LightSample* sample)
    {
        sample->type = light.type;
        sample->p    = {};
        sample->d    = {};
        if (light.type == LightType::ePoint)
        {
            return pointLightSampleFromContext(light, u, lsCtx.p, lsCtx.n, lsCtx.hadTransmission, sample);
        }
        else if (light.type == LightType::eSpot)
        {
            return spotLightSampleFromContext(light, u, lsCtx.p, lsCtx.n, lsCtx.hadTransmission, sample);
        }
        else
            return false;
    }

    bool lightIntersect(Light const& light, Ray const& ray, float* t)
    {

        if (light.type == LightType::ePoint)
            return pointLightIntersect(light, ray, t);
        else if (light.type == LightType::eSpot)
            return spotLightIntersect(light, ray, t);
        else
            return false;
    }

    RGB lightEval(Light const& light, LightSample const* sample)
    {
        RGB eval = RGB::fromScalar(1.f);

        //eval *= sample->evalFac;
        eval *= light.strength;

        // Distance-based attenuation (optional)
        // Avoid division by zero
        float attenuation = 1.f;
        if (sample->t > 1e-6f)
        {
            // Example: inverse-square law
            attenuation = 1.f / (sample->t * sample->t);

            // Alternatively, smooth falloff: linear
            // attenuation = std::max(0.f, 1.f - sample->t / maxDistance);

            // Or exponential falloff
            // attenuation = expf(-decayRate * sample->t);
        }

        eval *= attenuation;

        return eval;
    }

    // --- Env Light ----
    bool envLightSampleFromContext(EnvLight const& light, LightSampleContext const& lsCtx, Point2f u, LightSample* sample)
    {
        // sample distribution
        float   mapPDF = 0.f;
        Point2f uv     = light.distrib.sample(u, &mapPDF);
        if (mapPDF == 0.f)
            return false;

        // update PDF and compute incident direction
        // 2 pi u = 1 - phi <-> phi = 1 - 2 pi u
        // pi v = pi - theta <-> theta = pi (1 - v)
        float const phi = fl::clamp(1.f - fl::twoPi() * uv[0], -fl::pi(), fl::pi());
#if defined(DMT_EQUIRECTANGULAR_FLIP_Y)
        float const theta = fl::clamp(fl::pi() * (1.f - uv[1]), 0, fl::pi());
#else
        float const theta = fl::clamp(fl::pi() * uv[1], 0, fl::pi());
#endif
        Vector3f const wLight = cartesianFromSpherical(1.f, phi, theta);

        // inverse quaternion to get wrold space orientation. notes on quaternions
        //  - pure quaternion -> w = 0
        //  - rotation unit quaterion -> cosTheta/2 + unitVector sinTheta/2 where cosTheta/2 is w and unitVector is the rotation axis
        //  - rotation is applied by conjugation of a point represented as a pure quaterion -> Quat * Pure * Quat^-1, then extract x,y,z from result
        //    where the inverse quaterion is equal to the conjugate as it is unit, which is cosTheta/2 - unitVector sinTheta/2 where cosTheta/2
        Quaternion const pure_wLight{wLight.x, wLight.y, wLight.z, 0.f};
        Quaternion       lightFromRenderConj = -light.lightFromRender;
        lightFromRenderConj.w                = light.lightFromRender.w;
        Quaternion const pure_wi             = lightFromRenderConj * pure_wLight * light.lightFromRender;
        assert(fl::abs(pure_wi.w) < 1e-5f);
        Vector3f const wi{pure_wi.x, pure_wi.y, pure_wi.z};

        // compute PDF
        sample->pdf     = mapPDF / (4 * fl::pi());
        sample->p       = lsCtx.p; // for lack of a better value, use the last intersection position
        sample->d       = wi;
        sample->ng      = wi;
        sample->uv      = uv; // used in eval to sample the texture
        sample->evalFac = light.evalFac;
        sample->t       = fl::infinity();
        sample->type    = LightType::eEnv;
        return true;
    }

    RGB envLightEval(EnvLight const& light, LightSample const* sample)
    {
        // for now nearest filter (ZOH)
        int32_t const xIdx = static_cast<int32_t>(roundf(fl::clamp01(sample->uv[0]) * (light.xResolution - 1)));
        int32_t const yIdx = static_cast<int32_t>(roundf(fl::clamp01(sample->uv[1]) * (light.yResolution - 1)));
        RGB const     s    = light.imageBuffer[xIdx + yIdx * light.xResolution];
        return s;
    }

    RGB envLightEval(EnvLight const& light, Vector3f wi, float* pdf)
    {
        assert(fl::abs(normL2(wi) - 1.f) < 1e-5f);

        // rotate to light space
        Quaternion const pure_wi{wi.x, wi.y, wi.z, 0.f};
        Quaternion       lightFromRenderConj = -light.lightFromRender;
        lightFromRenderConj.w                = light.lightFromRender.w;
        Quaternion const pure_wLight         = light.lightFromRender * pure_wi * lightFromRenderConj;
        Vector3f const   wLight{pure_wLight.x, pure_wLight.y, pure_wLight.z};

        // spherical coordinates
        float const theta = std::acos(fl::clamp(wLight.z, -1.f, 1.f));
        float const phi   = std::atan2(wLight.y, wLight.x);

        // map to [0,1]^2
        Point2f const uv
        {
            0.5f * (1.f + phi / fl::pi()), // phi in [-pi, pi]
#define DMT_EQUIRECTANGULAR_FLIP_Y
#if defined(DMT_EQUIRECTANGULAR_FLIP_Y)
                1.f - theta / fl::pi()
#else
                theta / fl::pi()
#endif
        };

        // compute PDF
        *pdf = light.distrib.pdf(uv) / (4.f * fl::pi());

        // nearest-neighbor sampling
        int32_t const xIdx = std::clamp<int32_t>(static_cast<int32_t>(uv[0] * light.xResolution), 0, light.xResolution - 1);
        int32_t const yIdx = std::clamp<int32_t>(static_cast<int32_t>(uv[1] * light.yResolution), 0, light.yResolution - 1);
        RGB const s = light.imageBuffer[xIdx + yIdx * light.xResolution];
        return s;
    }
} // namespace dmt