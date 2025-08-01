#include "core-bsdf.h"

namespace dmt {
    BsdfClosure makeClosure(RGB weight)
    {
        return {.weight = weight, .N{0, 0, 1}, .sampleWeight = maxComponent(max(weight.asVec(), Vector3f::zero()))};
    }

    Vector3f ensureValidSpecularReflection(Vector3f const Ng, Vector3f const I, Vector3f N)
    {
        Vector3f const R = 2 * dot(N, I) * N - I;

        float const Iz = dot(I, Ng);
        assert(Iz >= 0);

        // Reflection rays may always be at least as shallow as the incoming ray.
        float const threshold = fminf(0.9f * Iz, 0.01f);
        if (dot(Ng, R) >= threshold)
        {
            return N;
        }

        // Form coordinate system with Ng as the Z axis and N inside the X-Z-plane.
        // The X axis is found by normalizing the component of N that's orthogonal to Ng.
        // The Y axis isn't actually needed.
        Vector3f const X = safeNormalizeFallback(N - dot(N, Ng) * Ng, N);

        // Calculate N.z and N.x in the local coordinate system.
        //
        // The goal of this computation is to find a N' that is rotated towards Ng just enough
        // to lift R' above the threshold (here called t), therefore dot(R', Ng) = t.
        //
        // According to the standard reflection equation,
        // this means that we want dot(2*dot(N', I)*N' - I, Ng) = t.
        //
        // Since the Z axis of our local coordinate system is Ng, dot(x, Ng) is just x.z, so we get
        // 2*dot(N', I)*N'.z - I.z = t.
        //
        // The rotation is simple to express in the coordinate system we formed -
        // since N lies in the X-Z-plane, we know that N' will also lie in the X-Z-plane,
        // so N'.y = 0 and therefore dot(N', I) = N'.x*I.x + N'.z*I.z .
        //
        // Furthermore, we want N' to be normalized, so N'.x = sqrt(1 - N'.z^2).
        //
        // With these simplifications, we get the equation
        // 2*(sqrt(1 - N'.z^2)*I.x + N'.z*I.z)*N'.z - I.z = t,
        // or
        // 2*sqrt(1 - N'.z^2)*I.x*N'.z = t + I.z * (1 - 2*N'.z^2),
        // after rearranging terms.
        // Raise both sides to the power of two and substitute terms with
        // a = I.x^2 + I.z^2,
        // b = 2*(a + Iz*t),
        // c = (Iz + t)^2,
        // we obtain
        // 4*a*N'.z^4 - 2*b*N'.z^2 + c = 0.
        //
        // The only unknown here is N'.z, so we can solve for that.
        //
        // The equation has four solutions in general, two can immediately be discarded because they're
        // negative so N' would lie in the lower hemisphere; one solves
        // 2*sqrt(1 - N'.z^2)*I.x*N'.z = -(t + I.z * (1 - 2*N'.z^2))
        // instead of the original equation (before squaring both sides).
        // Therefore only one root is valid.
        float const Ix = dot(I, X);

        float const a = Ix * Ix + Iz * Iz;
        float const b = 2.0f * (a + Iz * threshold);
        float       c = threshold + Iz;
        c *= c;

        // In order that the root formula solves 2*sqrt(1 - N'.z^2)*I.x*N'.z = t + I.z - 2*I.z*N'.z^2,
        // Ix and (t + I.z * (1 - 2*N'.z^2)) must have the same sign (the rest terms are non-negative by
        // definition).
        float const Nz2 = (Ix < 0) ? 0.25f * (b + fl::safeSqrt(b * b - 4.0f * a * c)) / a
                                   : 0.25f * (b - fl::safeSqrt(b * b - 4.0f * a * c)) / a;

        float const Nx = fl::safeSqrt(1.0f - Nz2);
        float const Nz = fl::safeSqrt(Nz2);

        return Nx * X + Nz * Ng;
    }
} // namespace dmt

namespace dmt::fresnel {
    // TODO compare performance/quality with schlick's approx
    float reflectanceDielectric(float cosThetai, float relIOR)
    {
        cosThetai = fl::clamp(cosThetai, -1, 1);
        if (cosThetai < 0)
        {
            relIOR    = fl::rcp(relIOR);
            cosThetai = -cosThetai;
        }

        // snell's law. Return 1.f in case of total reflection
        float const sin2Thetai = 1 - cosThetai * cosThetai;
        float const sin2Thetat = sin2Thetai / relIOR * relIOR;
        if (sin2Thetat >= 1.f)
            return 1.f;
        float const cosThetat = fl::safeSqrt(1.f - sin2Thetat);

        float const rParl = (relIOR * cosThetai - cosThetat) / (relIOR * cosThetai + cosThetat);
        float const rPerp = (cosThetai - relIOR * cosThetat) / (cosThetai + relIOR * cosThetat);
        return 0.5f * (rParl * rParl + rPerp * rPerp);
    }

    RGB DMT_FASTCALL reflectanceConductor(float cosThetai, RGB eta, RGB etak)
    {
        RGB reflectance{};
        cosThetai = fl::clamp(cosThetai, 0, 1);

        float    cosTheta2i = cosThetai * cosThetai;
        float    sinTheta2i = 1 - cosTheta2i;
        Vector3f eta2       = eta.mul(eta);
        Vector3f etak2      = etak.mul(etak);

        Vector3f t0       = eta2 - etak2 - Vector3f::s(sinTheta2i);
        Vector3f a2plusb2 = sqrt(t0 * t0 + 4 * eta2 * etak2);
        Vector3f t1       = a2plusb2 + Vector3f::s(cosTheta2i);
        Vector3f a        = sqrt(0.5f * (a2plusb2 + t0));
        Vector3f t2       = 2 * a * cosThetai;
        Vector3f Rs       = (t1 - t2) / (t1 + t2);

        Vector3f t3 = cosTheta2i * a2plusb2 + Vector3f::s(sinTheta2i * sinTheta2i);
        Vector3f t4 = t2 * sinTheta2i;
        Vector3f Rp = Rs * (t3 - t4) / (t3 + t4);

        return RGB::fromVec(0.5f * (Rp + Rs));
    }
} // namespace dmt::fresnel

namespace dmt::oren_nayar {
    BRDF makeParams(float roughness, RGB color, Vector3f ns, Vector3f wi, RGB weight)
    {
        static constexpr float piOver2Minus2Over3 = fl::piOver2() - 2.f / 3.f;
        static constexpr float _2piM5_6Over3      = (2.f * fl::pi() - 5.6f) / 3.f;
        static constexpr float _1OverPi           = 1.f / fl::pi();

        BRDF params{};
        params.closure   = makeClosure(weight);
        params.closure.N = ns;

        float const sigma  = fl::clamp01(roughness);
        RGB const   albedo = color.saturate();

        params.a = fl::rcp(fl::pi() + piOver2Minus2Over3 * sigma);
        params.b = params.a * sigma;

        // energy compensation for multiscatter term
        float const    Eavg = params.a * params.a + _2piM5_6Over3 * params.b;
        Vector3f const Ems  = _1OverPi * albedo.mul(albedo) * (Eavg / (1.f - Eavg)) /
                             (Vector3f::one() - albedo.asVec() * (1.f - Eavg));
        float const nv          = fmaxf(dot(ns, wi), 0.f);
        float const Ev          = params.a * fl::pi() + params.b * G(nv);
        params.multiscatterTerm = RGB::fromVec(Ems * (1.f - Ev));
        return params;
    }

    Vector3f DMT_FASTCALL sample(Vector3f ns, Vector3f ng, Point2f u, float* pdf)
    {
        Vector3f const wi = sampleCosHemisphere(ns, u, pdf);
        if (*pdf && dot(wi, ng) <= 0.f)
            *pdf = 0.f;

        return wi;
    }

    float DMT_FASTCALL G(float cosTheta)
    {
        if (cosTheta < 1e-6f)
            return (fl::piOver2() - 2.f / 3.f) - cosTheta;

        float const sinTheta = fl::safeSqrt(1.f - cosTheta * cosTheta);
        float const theta    = fl::acosClamp(cosTheta);
        return sinTheta * (theta - 2.0f / 3.0f - sinTheta * cosTheta) +
               2.0f / 3.0f * (sinTheta / cosTheta) * (1.0f - sinTheta * sinTheta * sinTheta);
    }

    RGB intensity(BRDF const& bsdf, Vector3f wo, Vector3f wi)
    {
        static constexpr float _1OverPi = 1.f / fl::pi();

        // let alpha be the angle between normal and light vector
        float const cosAlpha = absDot(bsdf.closure.N, wi);
        if (bsdf.b <= 0.f) // lambert fallback
            return RGB::fromScalar(cosAlpha * _1OverPi);

        // let gamma be the angle between view vector and normal
        float const cosGamma = absDot(bsdf.closure.N, wo);

        // formula
        float t = dot(wo, wi) - cosAlpha * cosGamma;
        if (t >= 0.f)
            t /= fmaxf(cosAlpha, cosGamma) + fl::minNormalized();

        float const singleScatter = bsdf.a + bsdf.b * t;
        float const El            = bsdf.a * fl::pi() + bsdf.b * G(cosAlpha);
        RGB const   multiScatter  = bsdf.multiscatterTerm * (1.f - El);
        return (RGB::fromScalar(singleScatter) + multiScatter) * cosAlpha;
    }
} // namespace dmt::oren_nayar

namespace dmt::ggx {
    /// Cycles' implementation
    static void energyPreservation(BSDF& outBsdf, Vector3f wo, RGB Fss)
    {
        float const mu    = dot(wo, outBsdf.closure.N);
        float const rough = sqrtf(sqrtf(outBsdf.alphax * outBsdf.alphay));

        // TODO
        float const E    = /*table lookup*/ 1.f;
        float const Eavg = /*table lookup*/ 1.f;

        float const missingFactor = (1.f - E) / E;
        outBsdf.energyScale       = 1.f + missingFactor;

        // Check if we need to account for extra darkening/saturation due to multi-bounce Fresnel.
        // if (!isequal(Fss, one_spectrum())) // Basically always, so skip this
        {
            // Fms here is based on the appendix of "Revisiting Physically Based Shading at Imageworks"
            // by Christopher Kulla and Alejandro Conty,
            // with one Fss cancelled out since this is just a multiplier on top of
            // the single-scattering BSDF, which already contains one bounce of Fresnel.
            RGB const Fms = Fss * Eavg / (RGB::one() - Fss * (1.0f - Eavg));
            // Since we already include the energy compensation in bsdf->energy_scale,
            // this term is what's needed to make the full BSDF * weight * energy_scale
            // computation work out to the correct value. */
            RGB const darkening = (RGB::one() + Fms * missingFactor) / outBsdf.energyScale;
            outBsdf.closure.weight *= darkening;
            outBsdf.closure.sampleWeight *= darkening.avg();
        }
    }

    static RGB estimateAlbedo(BSDF const& bsdf, Vector3f w)
    {
        float const cosNI = dot(w, bsdf.closure.N);
        RGB         reflectance{}, transmittance{};
        if (bsdf.isConductor)
            reflectance = fresnel::reflectanceConductor(cosNI, bsdf.fresnel.c.eta, bsdf.fresnel.c.etak);
        else
        {
            float const f = fresnel::reflectanceDielectric(cosNI, bsdf.eta);
            reflectance   = f * bsdf.fresnel.d.reflectanceTint;
            transmittance = (1.f - f) * bsdf.fresnel.d.transmittanceTimt;
        }

        //if (bsdf.isConductor) {
        //    // lookup table if you use schlick's generalized approx or 82 tint. We are using the full formula
        //    // is it needed? TODO
        //}
        return reflectance + transmittance;
    }

    BSDF DMT_FASTCALL
        makeDielectric(Vector3f wo, Vector3f ns, Vector3f ng, float ior, float alphax, float alphay, RGB r, RGB t, RGB weight)
    {
        BSDF bsdf{};
        bsdf.closure                     = makeClosure(weight);
        bsdf.closure.N                   = ensureValidSpecularReflection(ns, wo, ng);
        bsdf.T                           = {};
        bsdf.eta                         = fmaxf(ior, 1e-5f);
        bsdf.alphax                      = fl::clamp01(alphax);
        bsdf.alphay                      = fl::clamp01(alphay);
        bsdf.energyScale                 = 1.f;
        bsdf.isConductor                 = 0;
        bsdf.fresnel.d.reflectanceTint   = r.saturate();
        bsdf.fresnel.d.transmittanceTimt = t.saturate();
        bsdf.closure.sampleWeight *= estimateAlbedo(bsdf, wo).avg();

        // energy preservation, assuming transmission tint is what makes up most of the color
        energyPreservation(bsdf, wo, bsdf.fresnel.d.transmittanceTimt);
        return bsdf;
    }

    BSDF DMT_FASTCALL
        makeConductor(Vector3f wo, Vector3f ns, Vector3f ng, float alphax, float alphay, Vector3f tangent, RGB eta, RGB etak, RGB weight)
    {
        BSDF bsdf{};
        bsdf.closure        = makeClosure(weight);
        bsdf.closure.N      = ensureValidSpecularReflection(ns, wo, ng);
        bsdf.eta            = 1.f;
        bsdf.alphax         = fl::clamp01(alphax);
        bsdf.alphay         = fl::clamp01(alphay);
        bsdf.isConductor    = 1;
        bsdf.fresnel.c.eta  = eta.saturate();
        bsdf.fresnel.c.etak = eta.saturate();
        bsdf.closure.sampleWeight *= estimateAlbedo(bsdf, wo).avg();

        // energy preservation, assuming transmission tint is what makes up most of the color
        // fit the F82-tint model (TODO: we lack the lookup table), based on 0deg and 82 deg
        // and then compute Fss
        RGB const F0  = fresnel::reflectanceConductor(1.f, bsdf.fresnel.c.eta, bsdf.fresnel.c.etak);
        RGB const F82 = fresnel::reflectanceConductor(1.f / 7.f, bsdf.fresnel.c.eta, bsdf.fresnel.c.etak);
        const RGB B   = (RGB::fromVec(lerp(0.46266436f, F0.asVec(), Vector3f::one())) - F82) * 17.651384f;
        const RGB Fss = (RGB::fromVec(lerp(1.0f / 21.0f, F0.asVec(), Vector3f::one())) - B * (1.0f / 126.0f)).saturate();
        energyPreservation(bsdf, wo, Fss);
        return bsdf;
    }

    float DMT_FASTCALL auxiliaryLambda(Vector3f w, float alphax, float alphay)
    {
        assert(fl::abs(1.f - normL2(w)) < 1e-5f && "Direction should be unit vector");
        float const cosTheta  = w.z;
        float const sinTheta2 = 1.f - cosTheta * cosTheta;
        float const cosPhi2   = w.x * w.x / sinTheta2;
        float const sinPhi2   = w.y * w.y / sinTheta2;
        return fl::sqrt(alphax * alphax * cosPhi2 + alphay * alphay * sinPhi2);
    }

    float DMT_FASTCALL smithG1(Vector3f w, float alphax, float alphay)
    {
        return fl::rcp(1.f + auxiliaryLambda(w, alphax, alphay));
    }

    float DMT_FASTCALL heightCorrG(Vector3f wo, Vector3f wi, float alphax, float alphay)
    {
        return fl::rcp(1.f + auxiliaryLambda(wo, alphax, alphay) + auxiliaryLambda(wi, alphax, alphay));
    }

    float DMT_FASTCALL NDF(Vector3f wm, float alphax, float alphay)
    {
        assert(fl::abs(1.f - normL2(wm)) < 1e-5f && "Direction should be unit vector");
        float const cosTheta2 = wm.z * wm.z;
        float const sinTheta2 = 1.f - cosTheta2;
        float const tanTheta2 = sinTheta2 / cosTheta2;
        float const cosPhi2   = wm.x * wm.x / sinTheta2;
        float const sinPhi2   = wm.y * wm.y / sinTheta2;

        return fl::rcp(fl::pi() * alphax * alphay * cosTheta2 * cosTheta2 *
                       (1.f + tanTheta2 * (cosPhi2 / (alphax * alphax) + sinPhi2 / (alphay * alphay))));
    }

    float DMT_FASTCALL PDF(Vector3f w, Vector3f wm, float alphax, float alphay)
    {
        // absCosTheta = abs(z)
        return smithG1(w, alphax, alphay) / absCosTheta(w) * NDF(wm, alphax, alphay) * absDot(w, wm);
    }

    Vector3f DMT_FASTCALL sampleMicroNormal(Vector3f wi, Point2f u, float alphax, float alphay)
    {
        // Section 3.2: transform w to hemispherical configuration
        Vector3f const wi_ = normalize(Vector3f{alphax * wi.x, alphay * wi.y, wi.z});

        // Section 4.1: find orthonormal basis for visible normal sampling
        float const lensq = wi_.x * wi_.x + wi_.y * wi_.y;
        Vector3f    T1, T2;
        if (lensq < 1e-7f)
        {
            T1 = Vector3f{-wi_.y, wi_.x, 0.f} * fl::rsqrt(lensq);
            T2 = cross(wi_, T1);
        }
        else
        {
            // normal incidence, any base is fine
            T1 = Vector3f::xAxis();
            T2 = Vector3f::yAxis();
        }

        // Section 4.2: parametrization of the projected area
        Point2f t = sampleUniformDisk(u);
        t.y       = fl::lerp(0.5f * (1.f + wi_.z), fl::safeSqrt(1.f - t.x * t.x), t.y);

        // Section 4.3: Reprojection onto hemisphere
        Vector3f const H_Local = hemisphereFromDisk(t);
        Vector3f const H_      = H_Local.x * T1 + H_Local.y * T2 + H_Local.z * wi_;

        // Section 3.4: Transforming hte normal back to the ellipsoid configuration
        return normalize(Vector3f{alphax * H_.x, alphay * H_.y, fmaxf(0.f, H_.z)});
    }
} // namespace dmt::ggx