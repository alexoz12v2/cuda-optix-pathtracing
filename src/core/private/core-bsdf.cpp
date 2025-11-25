#include "core-bsdf.h"

#include "core-math.h"
#include "cudautils/cudautils-float.cuh"
#include "cudautils/cudautils-numbers.cuh"

// TODO remove
#if 1
    #include "platform-context.h"
#endif

namespace dmt {
    bool refract(Vector3f wi, Normal3f n, float eta, float* etap, Vector3f* wt)
    {
        float cosThetai = dot(wi, n);
        if (cosThetai < 0) // inside -> outside
        {
            eta       = fl::rcp(eta);
            cosThetai = -cosThetai;
            n         = -n;
        }

        // snell: cosThetat = sqrt(1-sin2Thetai / eta2). if radicand is negative, total internal reflection
        float const sin2Thetai = fmaxf(0.f, cosThetai * cosThetai);
        float       sin2Thetat = sin2Thetai / (eta * eta);
        if (sin2Thetat > 1.f)
            return false;

        float const cosThetat = fl::safeSqrt(1.f - sin2Thetat);

        *wt = -wi / eta + (cosThetai / eta - cosThetat) * n.asVec();

        if (etap)
            *etap = eta;

        return true;
    }

    BsdfClosure makeClosure(RGB weight)
    {
        return {.weight = weight, .N{0, 0, 1}, .sampleWeight = maxComponent(max(weight.asVec(), Vector3f::zero()))};
    }

    Vector3f ensureValidSpecularReflection(Vector3f Ng, Vector3f const I, Vector3f N)
    {
        Vector3f const R = 2 * dot(N, I) * N - I;

        float Iz = dot(I, Ng);
#if defined(DMT_CRASH_BSDF)
        assert(Iz >= 0); // TODO remove, caller should flip Normals
#else
        if (Iz < 0.0f)
        {
            // Either this was a backface or conventions differ. Flip both Ng and shading normal N
            // so that Ng faces the incoming direction. This is safe for two-sided materials;
            // for one-sided you should have culled this intersection earlier.
            Ng = -Ng;
            N  = -N;
            Iz = -Iz;
        }

        // Now Iz >= 0 (or zero)
        // If shading normal is in wrong hemisphere relative to Ng, flip it
        if (dot(N, Ng) < 0.0f)
        {
            N = -N;
        }
#endif

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
    BRDF makeParams(float roughness, RGB color, Vector3f ns, Vector3f wi, float multiscatterMultiplier, RGB weight)
    {
        static constexpr float piOver2Minus2Over3 = fl::piOver2() - 2.f / 3.f;
        static constexpr float _2piM5_6Over3      = (2.f * fl::pi() - 5.6f) / 3.f;
        static constexpr float _1OverPi           = 1.f / fl::pi();

        BRDF params{};
        params.closure                = makeClosure(weight);
        params.closure.N              = ns;
        params.multiscatterMultiplier = multiscatterMultiplier;

        float const sigma = fl::clamp01(roughness);
        params.albedo     = color.saturate();

        params.a = fl::rcp(fl::pi() + piOver2Minus2Over3 * sigma);
        params.b = params.a * sigma;

        // energy compensation for multiscatter term
        float const    Eavg = params.a * params.a + _2piM5_6Over3 * params.b;
        Vector3f const Ems  = _1OverPi * params.albedo.mul(params.albedo) * (Eavg / (1.f - Eavg)) /
                             (Vector3f{1, 1, 1} - params.albedo.asVec() * (1.f - Eavg));
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
        float const nl = absDot(bsdf.closure.N, wi);
        if (bsdf.b <= 0.f) // lambert fallback
            return RGB::fromScalar(nl * _1OverPi);

        // let gamma be the angle between view vector and normal
        float const nv = absDot(bsdf.closure.N, wo);

        // formula
        float t = dot(wo, wi) - nl * nv;
        if (t >= 0.f)
            t /= fmaxf(nl, nv) + fl::minNormalized();

        float const singleScatter = bsdf.a + bsdf.b * t;
        float const El            = bsdf.a * fl::pi() + bsdf.b * G(nl);
        RGB const   multiScatter  = bsdf.multiscatterMultiplier * bsdf.multiscatterTerm * (1.f - El);
        return bsdf.albedo * (RGB::fromScalar(singleScatter) + multiScatter) * nl;
    }
} // namespace dmt::oren_nayar

namespace dmt::ggx {

    // TODO: move to a Windows Heap memory dedicated to tables
    static float const table_ggx_E[1024] =
        {1.000000f, 0.980405f, 0.994967f, 0.997749f, 0.998725f, 0.999173f, 0.999411f, 0.999550f, 0.999634f, 0.999686f,
         0.999716f, 0.999732f, 0.999738f, 0.999735f, 0.999726f, 0.999712f, 0.999693f, 0.999671f, 0.999645f, 0.999615f,
         0.999583f, 0.999548f, 0.999511f, 0.999471f, 0.999429f, 0.999385f, 0.999338f, 0.999290f, 0.999240f, 0.999188f,
         0.999134f, 0.999079f, 1.000000f, 0.999451f, 0.990086f, 0.954714f, 0.911203f, 0.891678f, 0.893893f, 0.905010f,
         0.917411f, 0.928221f, 0.936755f, 0.943104f, 0.947567f, 0.950455f, 0.952036f, 0.952526f, 0.952096f, 0.950879f,
         0.948983f, 0.946495f, 0.943484f, 0.940010f, 0.936122f, 0.931864f, 0.927272f, 0.922380f, 0.917217f, 0.911810f,
         0.906184f, 0.900361f, 0.894361f, 0.888202f, 1.000000f, 0.999866f, 0.997676f, 0.987331f, 0.962386f, 0.929174f,
         0.902886f, 0.890270f, 0.888687f, 0.893114f, 0.899716f, 0.906297f, 0.911810f, 0.915853f, 0.918345f, 0.919348f,
         0.918982f, 0.917380f, 0.914671f, 0.910973f, 0.906393f, 0.901026f, 0.894957f, 0.888261f, 0.881006f, 0.873254f,
         0.865061f, 0.856478f, 0.847553f, 0.838329f, 0.828845f, 0.819138f, 1.000000f, 0.999941f, 0.998997f, 0.994519f,
         0.982075f, 0.959460f, 0.931758f, 0.907714f, 0.892271f, 0.885248f, 0.884058f, 0.885962f, 0.888923f, 0.891666f,
         0.893488f, 0.894054f, 0.893246f, 0.891062f, 0.887565f, 0.882845f, 0.877003f, 0.870143f, 0.862365f, 0.853766f,
         0.844436f, 0.834459f, 0.823915f, 0.812876f, 0.801410f, 0.789581f, 0.777445f, 0.765057f, 1.000000f, 0.999967f,
         0.999442f, 0.996987f, 0.989925f, 0.975437f, 0.953654f, 0.929078f, 0.907414f, 0.891877f, 0.882635f, 0.878166f,
         0.876572f, 0.876236f, 0.876004f, 0.875144f, 0.873238f, 0.870080f, 0.865601f, 0.859812f, 0.852774f, 0.844571f,
         0.835302f, 0.825069f, 0.813976f, 0.802123f, 0.789609f, 0.776523f, 0.762954f, 0.748982f, 0.734682f, 0.720122f,
         1.000000f, 0.999979f, 0.999644f, 0.998096f, 0.993618f, 0.983950f, 0.967765f, 0.946488f, 0.923989f, 0.904160f,
         0.889039f, 0.878704f, 0.872099f, 0.867839f, 0.864665f, 0.861611f, 0.858015f, 0.853468f, 0.847745f, 0.840752f,
         0.832480f, 0.822973f, 0.812312f, 0.800592f, 0.787922f, 0.774411f, 0.760171f, 0.745308f, 0.729926f, 0.714121f,
         0.697985f, 0.681600f, 1.000000f, 0.999985f, 0.999752f, 0.998684f, 0.995603f, 0.988803f, 0.976727f, 0.959272f,
         0.938451f, 0.917446f, 0.898932f, 0.884154f, 0.873043f, 0.864773f, 0.858278f, 0.852565f, 0.846845f, 0.840555f,
         0.833331f, 0.824969f, 0.815378f, 0.804550f, 0.792533f, 0.779405f, 0.765271f, 0.750246f, 0.734447f, 0.717996f,
         0.701008f, 0.683595f, 0.665862f, 0.647905f, 1.000000f, 0.999989f, 0.999816f, 0.999032f, 0.996781f, 0.991766f,
         0.982561f, 0.968421f, 0.950089f, 0.929705f, 0.909775f, 0.892112f, 0.877428f, 0.865535f, 0.855737f, 0.847185f,
         0.839084f, 0.830796f, 0.821858f, 0.811969f, 0.800961f, 0.788766f, 0.775391f, 0.760894f, 0.745367f, 0.728923f,
         0.711685f, 0.693783f, 0.675344f, 0.656491f, 0.637343f, 0.618009f, 1.000000f, 0.999991f, 0.999858f, 0.999255f,
         0.997533f, 0.993686f, 0.986490f, 0.974988f, 0.959173f, 0.940264f, 0.920251f, 0.901030f, 0.883794f, 0.868900f,
         0.856078f, 0.844728f, 0.834154f, 0.823720f, 0.812914f, 0.801368f, 0.788846f, 0.775227f, 0.760476f, 0.744623f,
         0.727744f, 0.709946f, 0.691355f, 0.672105f, 0.652332f, 0.632172f, 0.611753f, 0.591195f, 1.000000f, 0.999993f,
         0.999886f, 0.999406f, 0.998040f, 0.994992f, 0.989227f, 0.979769f, 0.966203f, 0.949069f, 0.929766f, 0.909990f,
         0.891126f, 0.873921f, 0.858498f, 0.844545f, 0.831539f, 0.818909f, 0.806146f, 0.792847f, 0.778733f, 0.763633f,
         0.747476f, 0.730264f, 0.712056f, 0.692949f, 0.673066f, 0.652547f, 0.631533f, 0.610170f, 0.588596f, 0.566939f,
         1.000000f, 0.999995f, 0.999906f, 0.999513f, 0.998399f, 0.995917f, 0.991195f, 0.983312f, 0.971656f, 0.956303f,
         0.938125f, 0.918488f, 0.898757f, 0.879901f, 0.862353f, 0.846089f, 0.830787f, 0.815992f, 0.801242f, 0.786135f,
         0.770368f, 0.753741f, 0.736147f, 0.717566f, 0.698037f, 0.677649f, 0.656520f, 0.634791f, 0.612611f, 0.590130f,
         0.567495f, 0.544843f, 1.000000f, 0.999996f, 0.999921f, 0.999591f, 0.998661f, 0.996594f, 0.992650f, 0.985988f,
         0.975917f, 0.962217f, 0.945336f, 0.926280f, 0.906266f, 0.886338f, 0.867144f, 0.848905f, 0.831508f, 0.814643f,
         0.797929f, 0.780995f, 0.763540f, 0.745347f, 0.726289f, 0.706324f, 0.685477f, 0.663824f, 0.641483f, 0.618591f,
         0.595303f, 0.571774f, 0.548156f, 0.524596f, 1.000000f, 0.999996f, 0.999933f, 0.999651f, 0.998859f, 0.997104f,
         0.993752f, 0.988046f, 0.979280f, 0.967055f, 0.951498f, 0.933282f, 0.913407f, 0.892888f, 0.872490f, 0.852624f,
         0.833371f, 0.814578f, 0.795964f, 0.777219f, 0.758062f, 0.738279f, 0.717734f, 0.696371f, 0.674202f, 0.651296f,
         0.627765f, 0.603747f, 0.579398f, 0.554878f, 0.530346f, 0.505951f, 1.000000f, 0.999997f, 0.999941f, 0.999696f,
         0.999012f, 0.997497f, 0.994604f, 0.989656f, 0.981964f, 0.971026f, 0.956742f, 0.939495f, 0.920052f, 0.899322f,
         0.878109f, 0.856955f, 0.836102f, 0.815549f, 0.795133f, 0.774620f, 0.753771f, 0.732389f, 0.710341f, 0.687567f,
         0.664071f, 0.639918f, 0.615213f, 0.590097f, 0.564725f, 0.539263f, 0.513871f, 0.488706f, 1.000000f, 0.999997f,
         0.999949f, 0.999733f, 0.999132f, 0.997806f, 0.995276f, 0.990935f, 0.984129f, 0.974304f, 0.961200f, 0.944967f,
         0.926144f, 0.905496f, 0.883799f, 0.861667f, 0.839472f, 0.817348f, 0.795251f, 0.773035f, 0.750522f, 0.727545f,
         0.703988f, 0.679793f, 0.654965f, 0.629565f, 0.603699f, 0.577506f, 0.551143f, 0.524778f, 0.498576f, 0.472695f,
         1.000000f, 0.999997f, 0.999954f, 0.999762f, 0.999228f, 0.998053f, 0.995814f, 0.991966f, 0.985893f, 0.977026f,
         0.964995f, 0.949768f, 0.931677f, 0.911324f, 0.889415f, 0.866586f, 0.843295f, 0.819794f, 0.796153f, 0.772321f,
         0.748187f, 0.723632f, 0.698566f, 0.672945f, 0.646780f, 0.620134f, 0.593114f, 0.565860f, 0.538532f, 0.511298f,
         0.484328f, 0.457779f, 1.000000f, 0.999998f, 0.999959f, 0.999786f, 0.999307f, 0.998254f, 0.996252f, 0.992808f,
         0.987348f, 0.979302f, 0.968235f, 0.953973f, 0.936669f, 0.916763f, 0.894859f, 0.871576f, 0.847421f, 0.822737f,
         0.797698f, 0.772347f, 0.746651f, 0.720547f, 0.693982f, 0.666934f, 0.639428f, 0.611535f, 0.583366f, 0.555065f,
         0.526792f, 0.498719f, 0.471015f, 0.443841f, 1.000000f, 0.999998f, 0.999962f, 0.999805f, 0.999371f, 0.998420f,
         0.996612f, 0.993502f, 0.988557f, 0.981218f, 0.971011f, 0.957656f, 0.941156f, 0.921798f, 0.900070f, 0.876539f,
         0.851730f, 0.826051f, 0.799763f, 0.773002f, 0.745814f, 0.718199f, 0.690150f, 0.661679f, 0.632832f, 0.603691f,
         0.574377f, 0.545037f, 0.515837f, 0.486949f, 0.458544f, 0.430781f, 1.000000f, 0.999998f, 0.999966f, 0.999822f,
         0.999425f, 0.998558f, 0.996912f, 0.994082f, 0.989572f, 0.982843f, 0.973398f, 0.960884f, 0.945180f, 0.926433f,
         0.905008f, 0.881402f, 0.856128f, 0.829631f, 0.802244f, 0.774186f, 0.745583f, 0.716504f, 0.686997f, 0.657112f,
         0.626924f, 0.596535f, 0.566077f, 0.535706f, 0.505592f, 0.475910f, 0.446831f, 0.418514f, 1.000000f, 0.999998f,
         0.999968f, 0.999836f, 0.999471f, 0.998674f, 0.997165f, 0.994570f, 0.990430f, 0.984229f, 0.975460f, 0.963718f,
         0.948785f, 0.930682f, 0.909656f, 0.886116f, 0.860541f, 0.833390f, 0.805050f, 0.775811f, 0.745878f, 0.715390f,
         0.684454f, 0.653169f, 0.621644f, 0.590007f, 0.558407f, 0.527011f, 0.495995f, 0.465536f, 0.435808f, 0.406965f,
         1.000000f, 0.999999f, 0.999970f, 0.999848f, 0.999509f, 0.998773f, 0.997379f, 0.994985f, 0.991163f, 0.985419f,
         0.977249f, 0.966213f, 0.952014f, 0.934568f, 0.914006f, 0.890646f, 0.864912f, 0.837258f, 0.808105f, 0.777803f,
         0.746627f, 0.714789f, 0.682460f, 0.649794f, 0.616939f, 0.584055f, 0.551314f, 0.518896f, 0.486987f, 0.455768f,
         0.425410f, 0.396069f, 1.000000f, 0.999999f, 0.999973f, 0.999858f, 0.999542f, 0.998857f, 0.997562f, 0.995341f,
         0.991792f, 0.986447f, 0.978809f, 0.968414f, 0.954908f, 0.938115f, 0.918063f, 0.894970f, 0.869199f, 0.841178f,
         0.811344f, 0.780093f, 0.747765f, 0.714642f, 0.680962f, 0.646935f, 0.612760f, 0.578633f, 0.544751f, 0.511315f,
         0.478520f, 0.446553f, 0.415586f, 0.385770f, 1.000000f, 0.999999f, 0.999974f, 0.999867f, 0.999571f, 0.998930f,
         0.997720f, 0.995648f, 0.992336f, 0.987340f, 0.980174f, 0.970361f, 0.957504f, 0.941351f, 0.921833f, 0.899078f,
         0.873371f, 0.845104f, 0.814712f, 0.782625f, 0.749237f, 0.714896f, 0.679909f, 0.644547f, 0.609064f, 0.573697f,
         0.538677f, 0.504225f, 0.470550f, 0.437846f, 0.406286f, 0.376017f, 1.000000f, 0.999999f, 0.999976f, 0.999874f,
         0.999596f, 0.998993f, 0.997858f, 0.995914f, 0.992810f, 0.988121f, 0.981374f, 0.972090f, 0.959837f, 0.944301f,
         0.925331f, 0.902963f, 0.877406f, 0.848999f, 0.818163f, 0.785347f, 0.750991f, 0.715502f, 0.679256f, 0.642589f,
         0.605812f, 0.569211f, 0.533054f, 0.497587f, 0.463038f, 0.429607f, 0.397469f, 0.366766f, 1.000000f, 0.999999f,
         0.999977f, 0.999881f, 0.999618f, 0.999049f, 0.997978f, 0.996147f, 0.993224f, 0.988806f, 0.982434f, 0.973628f,
         0.961935f, 0.946991f, 0.928572f, 0.906628f, 0.881288f, 0.852834f, 0.821659f, 0.788217f, 0.752981f, 0.716417f,
         0.678962f, 0.641022f, 0.602968f, 0.565141f, 0.527849f, 0.491370f, 0.455949f, 0.421800f, 0.389096f, 0.357977f,
         1.000000f, 0.999999f, 0.999978f, 0.999887f, 0.999637f, 0.999097f, 0.998083f, 0.996352f, 0.993588f, 0.989410f,
         0.983373f, 0.975002f, 0.963827f, 0.949446f, 0.931571f, 0.910076f, 0.885009f, 0.856589f, 0.825169f, 0.791196f,
         0.755170f, 0.717601f, 0.678991f, 0.639812f, 0.600501f, 0.561455f, 0.523031f, 0.485541f, 0.449253f, 0.414392f,
         0.381135f, 0.349616f, 1.000000f, 0.999999f, 0.999979f, 0.999892f, 0.999655f, 0.999141f, 0.998177f, 0.996533f,
         0.993910f, 0.989945f, 0.984209f, 0.976232f, 0.965536f, 0.951688f, 0.934346f, 0.913314f, 0.888564f, 0.860245f,
         0.828665f, 0.794253f, 0.757521f, 0.719020f, 0.679309f, 0.638927f, 0.598380f, 0.558126f, 0.518574f, 0.480074f,
         0.442922f, 0.407354f, 0.373554f, 0.341651f, 1.000000f, 0.999999f, 0.999980f, 0.999897f, 0.999669f, 0.999179f,
         0.998260f, 0.996693f, 0.994197f, 0.990422f, 0.984956f, 0.977337f, 0.967083f, 0.953737f, 0.936913f, 0.916351f,
         0.891951f, 0.863792f, 0.832128f, 0.797360f, 0.760004f, 0.720642f, 0.679885f, 0.638338f, 0.596578f, 0.555128f,
         0.514452f, 0.474944f, 0.436929f, 0.400661f, 0.366328f, 0.334053f, 1.000000f, 0.999999f, 0.999981f, 0.999901f,
         0.999683f, 0.999213f, 0.998334f, 0.996836f, 0.994452f, 0.990848f, 0.985625f, 0.978332f, 0.968486f, 0.955612f,
         0.939288f, 0.919197f, 0.895172f, 0.867221f, 0.835539f, 0.800494f, 0.762592f, 0.722438f, 0.680691f, 0.638020f,
         0.595071f, 0.552438f, 0.510643f, 0.470129f, 0.431253f, 0.394289f, 0.359430f, 0.326796f, 1.000000f, 0.999999f,
         0.999982f, 0.999905f, 0.999695f, 0.999244f, 0.998400f, 0.996965f, 0.994681f, 0.991230f, 0.986227f, 0.979231f,
         0.969761f, 0.957330f, 0.941486f, 0.921863f, 0.898229f, 0.870526f, 0.838887f, 0.803634f, 0.765260f, 0.724383f,
         0.681702f, 0.637948f, 0.593837f, 0.550034f, 0.507126f, 0.465607f, 0.425873f, 0.388216f, 0.352839f, 0.319858f,
         1.000000f, 0.999999f, 0.999982f, 0.999908f, 0.999706f, 0.999271f, 0.998459f, 0.997080f, 0.994886f, 0.991574f,
         0.986770f, 0.980045f, 0.970922f, 0.958906f, 0.943521f, 0.924358f, 0.901128f, 0.873705f, 0.842159f, 0.806765f,
         0.767988f, 0.726453f, 0.682895f, 0.638100f, 0.592854f, 0.547896f, 0.503881f, 0.461361f, 0.420769f, 0.382424f,
         0.346535f, 0.313216f, 1.000000f, 0.999999f, 0.999983f, 0.999911f, 0.999716f, 0.999296f, 0.998513f, 0.997184f,
         0.995072f, 0.991884f, 0.987261f, 0.980785f, 0.971982f, 0.960355f, 0.945407f, 0.926694f, 0.903873f, 0.876756f,
         0.845349f, 0.809871f, 0.770757f, 0.728630f, 0.684249f, 0.638454f, 0.592102f, 0.546006f, 0.500893f, 0.457373f,
         0.415925f, 0.376893f, 0.340499f, 0.306853f};

    static float const table_ggx_Eavg[32] =
        {1.000000f, 0.999992f, 0.999897f, 0.999548f, 0.998729f, 0.997199f, 0.994703f, 0.990986f,
         0.985805f, 0.978930f, 0.970160f, 0.959321f, 0.946279f, 0.930937f, 0.913247f, 0.893209f,
         0.870874f, 0.846345f, 0.819774f, 0.791360f, 0.761345f, 0.730001f, 0.697631f, 0.664547f,
         0.631068f, 0.597509f, 0.564165f, 0.531311f, 0.499191f, 0.468013f, 0.437950f, 0.409137f};

    /// Cycles' implementation
    static void energyPreservation(BSDF& outBsdf, Vector3f wo, RGB Fss)
    {
        float const mu    = dot(wo, outBsdf.closure.N);
        float const rough = sqrtf(sqrtf(outBsdf.alphax * outBsdf.alphay));

        float const E    = lookupTableRead2D(table_ggx_E, rough, mu, 32, 32);
        float const Eavg = lookupTableRead(table_ggx_Eavg, rough, 32);

        float const missingFactor = (1.f - E) / E;
        outBsdf.energyScale       = 1.f + missingFactor;

        // Check if we need to account for extra darkening/saturation due to multi-bounce Fresnel.
        if (Fss != RGB::one())
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
            transmittance = (1.f - f) * bsdf.fresnel.d.transmittanceTint;
        }

        //if (bsdf.isConductor) {
        //    // lookup table if you use schlick's generalized approx or 82 tint. We are using the full formula
        //    // is it needed? TODO
        //}
        return reflectance + transmittance;
    }

    static Vector3f orthogonalizeTangent(Vector3f normal, Vector3f proposedTangent)
    {
        assert(fl::abs(normL2(normal) - 1.f) < 1e-5f && "Unit vector expected");

        Vector3f projected = proposedTangent - dot(proposedTangent, normal) * normal;
        return normalize(projected);
    }

    BSDF DMT_FASTCALL
        makeDielectric(Vector3f wo, Vector3f ns, Vector3f ng, float ior, float alphax, float alphay, RGB r, RGB t, RGB weight)
    {
        BSDF bsdf{};
        bsdf.closure                     = makeClosure(weight);
        bsdf.closure.N                   = ensureValidSpecularReflection(ns, wo, ng);
        bsdf.T                           = Vector3f::xAxis(); // TODO expose as param
        bsdf.eta                         = fmaxf(ior, 1e-5f);
        bsdf.alphax                      = fl::clamp(alphax, 1e-4f, 1.f);
        bsdf.alphay                      = fl::clamp(alphay, 1e-4f, 1.f);
        bsdf.energyScale                 = 1.f;
        bsdf.isConductor                 = 0;
        bsdf.fresnel.d.reflectanceTint   = r.saturate();
        bsdf.fresnel.d.transmittanceTint = t.saturate();
        bsdf.closure.sampleWeight *= estimateAlbedo(bsdf, wo).avg();

        // energy preservation, assuming transmission tint is what makes up most of the color
        energyPreservation(bsdf, wo, bsdf.fresnel.d.transmittanceTint);
        return bsdf;
    }

    BSDF DMT_FASTCALL
        makeConductor(Vector3f wo, Vector3f ns, Vector3f ng, float alphax, float alphay, Vector3f tangent, RGB eta, RGB etak, RGB weight)
    {
        BSDF model{};
        model.closure   = makeClosure(weight);
        model.closure.N = ensureValidSpecularReflection(ns, wo, ng);
        model.T         = dotSelf(tangent) < 1e-6f ? Vector3f::xAxis() : orthogonalizeTangent(model.closure.N, tangent);
        model.eta       = 1.f;
        model.alphax    = fl::clamp(alphax, 1e-4f, 1.f);
        model.alphay    = fl::clamp(alphay, 1e-4f, 1.f);
        model.isConductor    = 1;
        model.fresnel.c.eta  = eta;
        model.fresnel.c.etak = etak;
        model.closure.sampleWeight *= estimateAlbedo(model, wo).avg();

        // energy preservation, assuming transmission tint is what makes up most of the color
        // fit the F82-tint model (TODO: we lack the lookup table), based on 0deg and 82 deg
        // and then compute Fss
        RGB const F0  = fresnel::reflectanceConductor(1.f, model.fresnel.c.eta, model.fresnel.c.etak);
        RGB const F82 = fresnel::reflectanceConductor(1.f / 7.f, model.fresnel.c.eta, model.fresnel.c.etak);
        const RGB B   = (RGB::fromVec(lerp(0.46266436f, F0.asVec(), Vector3f::one())) - F82) * 17.651384f;
        const RGB Fss = (RGB::fromVec(lerp(1.0f / 21.0f, F0.asVec(), Vector3f::one())) - B * (1.0f / 126.0f)).saturate();
        energyPreservation(model, wo, Fss);
        return model;
    }

    float DMT_FASTCALL auxiliaryLambda(Vector3f w, float alphax, float alphay)
    {
        assert(fl::abs(1.f - normL2(w)) < 1e-5f && "Direction should be unit vector");
#if 0
        float const cosTheta  = w.z;
        float const sinTheta2 = 1.f - cosTheta * cosTheta;
        float const cosPhi2   = w.x * w.x / sinTheta2;
        float const sinPhi2   = w.y * w.y / sinTheta2;
        return fl::sqrt(alphax * alphax * cosPhi2 + alphay * alphay * sinPhi2);
#else
        float xWeight = alphax * w.x;
        float yWeight = alphay * w.y;

        xWeight *= xWeight;
        yWeight *= yWeight;

        float const sqr_alpha_tan_n = (xWeight + yWeight) / (w.z * w.z);
        return 0.5f * (sqrtf(1.0f + sqr_alpha_tan_n) - 1.0f);
#endif
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

        float D = 0;
#if 1
        float const cos2Theta = wm.z * wm.z;
        float const sin2Theta = fmaxf(0.f, 1.f - cos2Theta);
        float const sinTheta  = fl::sqrt(sin2Theta);
        float const tan2Theta = sin2Theta / cos2Theta;
        if (tan2Theta > 1e5f)
            return 0;
        float const cos4Theta = cos2Theta * cos2Theta;
        if (cos4Theta < 1e-8f)
            return 0;

        float cosPhi = sin2Theta < 1e-7f ? 1.f : fl::clamp(wm.x / sinTheta, -1.f, 1.f);
        float sinPhi = cos2Theta < 1e-7f ? 1.f : fl::clamp(wm.y / sinTheta, -1.f, 1.f);

        cosPhi /= alphax;
        sinPhi /= alphay;
        cosPhi *= cosPhi;
        sinPhi *= sinPhi;

        float den = 1.f + tan2Theta * (cosPhi + sinPhi);
        den *= den;

        D = fl::rcp(fl::pi() * alphax * alphay * cos4Theta * den);
#else
        wm /= {alphax, alphay, 1.0f};

        float const cos_NH2 = wm.z * wm.z;
        float const alpha2  = alphax * alphay;
        float const HdotH   = dotSelf(wm);

        D = fl::rcpPi() / (alpha2 * HdotH * HdotH);
#endif
        assert(!fl::isInfOrNaN(D) && D >= 0.f);

        // Optional: clamp extreme D values (TODO make it configurable?)
        return fminf(D, 100.f);
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
        if (lensq > 1e-7f)
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

    BSDFSample DMT_FASTCALL sample(BSDF const& bsdf, Vector3f w, Vector3f ng, Point2f u, float uc)
    {
        BSDFSample bs{};
        bs.eta = 1.f;
        bs.f   = RGB::fromScalar(0.f);
        bs.pdf = 0.f;

        // --- tangent frame (shading normal) ---
        Frame const    tangentSpace = Frame::fromXZ(bsdf.T, bsdf.closure.N);
        Vector3f const wLocal       = tangentSpace.toLocal(w); // incoming direction in tangent space
        float const    cosI         = wLocal.z;

        // sanity: incoming should be in the hemisphere of N
        if (cosI <= 0.f)
            return bs;

        // --- roughness / singular check ---
        bool const singular = effectivelySmooth(bsdf.alphax, bsdf.alphay);

        // --- Fresnel-based mixture (macro-level) ---
        // Use macroscopic cosine (cosI) to decide split between reflection/transmission.
        RGB reflectTint{};
        RGB transTint{};
        if (bsdf.isConductor)
        {
            // conductor: all reflection (no transmission), reflectance depends on micro-normal at sampling time,
            // but for the mixture we treat it as pure reflection.
            reflectTint = RGB::one(); // tint handled per-channel in conductor fresnel below
            transTint   = RGB::fromScalar(0.f);
        }
        else
        {
            // dielectric: use Fresnel at macro angle as heuristic to mix
            float Fmac  = fresnel::reflectanceDielectric(cosI, bsdf.eta);
            reflectTint = RGB::fromScalar(Fmac) * bsdf.fresnel.d.reflectanceTint;
            transTint   = RGB::fromScalar(1.f - Fmac) * bsdf.fresnel.d.transmittanceTint;
        }

        float denomMix   = (reflectTint + transTint).avg();
        float pdfReflect = (denomMix > 1e-12f) ? reflectTint.avg() / denomMix
                                               : (reflectTint.avg() > transTint.avg() ? 1.0f : 0.0f);

        bool doRefract = (!bsdf.isConductor) && (uc >= pdfReflect);
#if 1
        if (doRefract)
        {
            Context ctx;
            ctx.trace("Transmitted", {});
        }
#endif

        // --- sample (visible) micro-normal in local frame ---
        Vector3f wmLocal;
        if (singular)
        {
            // delta case: micro-normal aligns with shading normal (tangent space z)
            wmLocal = Vector3f(0.f, 0.f, 1.f);
            bs.wm   = bsdf.closure.N;
        }
        else
        {
            wmLocal = sampleMicroNormal(wLocal, u, bsdf.alphax, bsdf.alphay); // returns normalized micro-normal
            bs.wm   = normalize(tangentSpace.fromLocal(wmLocal));
        }

        // cos between incoming and micro-normal
        float const cosWH = dot(wLocal, wmLocal);
        if (cosWH <= 0.f)
            return bs;

        // --- compute outgoing local direction (woLocal) depending on branch ---
        Vector3f woLocal; // outgoing direction in local frame
        if (singular)
        {
            // deterministic specular: use geometric normal (closure.N) for reflection/refraction
            if (doRefract)
            {
                // refract using geometric normal
                Vector3f wt;
                float    etaRel = bsdf.eta;
                // refract expects wi (incoming), Normal3f n, eta, etap, wt
                // create Normal3f from closure.N (Normal3f normalizes internally)
                if (!refract(w, Normal3f(bsdf.closure.N), etaRel, &bs.eta, &wt))
                    return bs; // TIR or failure
                bs.wi = wt;
                // delta transmission: assign pdf equal to mixture probability for transmission branch
                bs.pdf = (1.0f - pdfReflect);
                // delta BTDF: approximate via transTint / (eta^2) (radiance symmetry)
                bs.f = transTint / (bs.eta * bs.eta);
                return bs;
            }
            else
            {
                // perfect mirror reflection
                Vector3f wr = reflect(w, Normal3f(bsdf.closure.N).asVec()); // reflect expects Vector3f,Normal3f
                bs.wi       = wr;
                bs.pdf      = pdfReflect;
                // delta BRDF: reflectance uses conductor or dielectric microfacet F at cos between wo and normal
                if (bsdf.isConductor)
                {
                    // conductor uses complex IOR Fresnel (per-channel)
                    bs.f = fresnel::reflectanceConductor(fabsf(dot(wr, bsdf.closure.N)),
                                                         bsdf.fresnel.c.eta,
                                                         bsdf.fresnel.c.etak);
                }
                else
                {
                    float F = fresnel::reflectanceDielectric(fabsf(dot(wr, bsdf.closure.N)), bsdf.eta);
                    bs.f    = bsdf.fresnel.d.reflectanceTint * F;
                }
                return bs;
            }
        }

        // Non-singular: compute outgoing in local space using micro-normal wmLocal
        if (doRefract)
        {
            // Transmission: need eta_i / eta_o consistent with refract helper semantics.
            float etaI = 1.f;
            float etaO = bsdf.eta;
            // if incoming is below geometric normal we would have flipped earlier; here cosI > 0 so assume outside->inside
            float relEta = etaI / etaO;

            // compute sin^2 theta_t to check TIR
            float sin2ThetaT = relEta * relEta * (fmaxf(0.f, 1.f - cosWH * cosWH));
            if (sin2ThetaT >= 1.f)
                return bs; // TIR for this micro-normal

            float cosThetaT = fl::safeSqrt(1.f - sin2ThetaT);

            // compute outgoing local vector (Walter eqn mapping)
            // Using relation: woLocal = normalize(relEta * -wLocal + (relEta * cosWH - cosThetaT) * wmLocal)
            woLocal = normalize(relEta * -wLocal + (relEta * cosWH - cosThetaT) * wmLocal);
            bs.eta  = relEta;
        }
        else
        {
            // Reflection: standard microfacet reflection around micro-normal
            woLocal = normalize(2.f * cosWH * wmLocal - wLocal);
            bs.eta  = 1.f;
        }

        // ensure outgoing is in the upper hemisphere
        if (!doRefract && woLocal.z <= 0.f)
        {
            if (doRefract)
            {
                Context ctx;
                ctx.error("Trasmission died", {});
            }
            return bs;
        }

        // convert to world-space outgoing direction
        bs.wi = normalize(tangentSpace.fromLocal(woLocal));

        // --- evaluate microfacet terms ---
        float const D = NDF(wmLocal, bsdf.alphax, bsdf.alphay);
        float const G = heightCorrG(woLocal, wLocal, bsdf.alphax, bsdf.alphay);

        // visible-normal PDF for the sampled micro-normal (in the local frame)
        float const pdf_m = fmaxf(1e-12f, PDF(wLocal, wmLocal, bsdf.alphax, bsdf.alphay));

        // compute Fresnel at micro-normal (cos between incident and micro-normal)
        float const F_micro = bsdf.isConductor
                                  ? fresnel::reflectanceConductor(fabsf(cosWH), bsdf.fresnel.c.eta, bsdf.fresnel.c.etak).avg()
                                  : fresnel::reflectanceDielectric(fabsf(cosWH), bsdf.eta);

        // compute common helper values
        float const cosO = woLocal.z;
        float const cosN = wLocal.z; // same as cosI earlier

        // --- assemble PDF and BSDF depending on branch ---
        if (!doRefract)
        {
            // Reflection (Walter eq. for reflection):
            // f = (F * D * G) / (4 * cosI * cosO)
            float denom = 4.f * cosN * cosO;
            if (denom <= 1e-12f)
                return bs;

            // per-channel Fresnel: conductor returns RGB; dielectric returns scalar F_micro
            RGB Fcol = bsdf.isConductor
                           ? fresnel::reflectanceConductor(fabsf(cosWH), bsdf.fresnel.c.eta, bsdf.fresnel.c.etak)
                           : bsdf.fresnel.d.reflectanceTint * RGB::fromScalar(F_micro);

            bs.f = (Fcol * (D * G)) / denom;

            // PDF change of variable: p(wo) = pdf_m / (4 * |dot(wo, wm)|)
            float denomPDF = 4.f * fabsf(cosWH);
            if (denomPDF <= 1e-12f)
                return bs;

            bs.pdf = pdfReflect * (pdf_m / denomPDF);
            // mixture factor applied (pdfReflect)
        }
        else
        {
            // Transmission (Walter's BTDF eq.21)
            // denom_term = (eta_i * (i·h) + eta_o * (o·h))^2
            float const i_dot_h = dot(wLocal, wmLocal);
            float const o_dot_h = dot(woLocal, wmLocal);
            float const i_dot_n = fabsf(cosN);
            float const o_dot_n = fabsf(cosO);

            // eta_i = 1, eta_o = bsdf.eta under our convention (outside->inside)
            float const eta_i     = 1.f;
            float const eta_o     = bsdf.eta;
            float       denomTerm = eta_i * i_dot_h + eta_o * o_dot_h;
            denomTerm             = denomTerm * denomTerm;
            if (denomTerm <= 1e-12f)
                return bs;

            // scalar 1 - F (use scalar Fresnel for dielectric)
            float const oneMinusF = 1.f - F_micro;

            // BTDF scalar factor per Walter eq.21:
            float bt_scalar = (fabsf(i_dot_h) * fabsf(o_dot_h)) / (i_dot_n * o_dot_n);
            bt_scalar *= (eta_o * eta_o) * oneMinusF * G * D / denomTerm;

            // transmittance tint (RGB) applied
            bs.f = bsdf.fresnel.d.transmittanceTint * bt_scalar;

            // change-of-variable Jacobian (eq.17 swapped): (eta_i^2 * |i·h|) / denomTerm
            float const jacobian = (eta_i * eta_i * fabsf(i_dot_h)) / denomTerm;

            bs.pdf = (1.0f - pdfReflect) * pdf_m * jacobian;
#if 1
            {
                Context ctx;
                ctx.trace("Transmission still alive with PDF {}", std::make_tuple(bs.pdf));
            }
#endif
        }

        // final safety clamps
        if (fl::isInfOrNaN(bs.pdf) || bs.pdf <= 0.f) // NaN or <=0
        {
            // nothing valid
            bs.pdf = 0.f;
            bs.f   = RGB::fromScalar(0.f);
            return bs;
        }

        // ensure non-negative energy (tiny negative due to FP can happen)
        bs.f = bs.f.saturate0();

        return bs;
    }

    RGB DMT_FASTCALL eval(BSDF const& bsdf, Vector3f wo, Vector3f wi, Vector3f ng, float* pdf)
    {
        BSDFSample dummy{};
        *pdf = 0.f;

        // --- tangent frame ---
        Frame const    tangentSpace = Frame::fromXZ(bsdf.T, bsdf.closure.N);
        Vector3f const woLocal      = normalize(tangentSpace.toLocal(wo));
        Vector3f const wiLocal      = normalize(tangentSpace.toLocal(wi));

        float const cosNO = woLocal.z;
        float const cosNI = wiLocal.z;
        if (cosNO <= 0.f || cosNI <= 0.f)
            return RGB::fromScalar(0.f); // below hemisphere, invalid

        // --- determine reflection vs transmission ---
        bool const isTransmission = dot(bsdf.closure.N, wo) * dot(bsdf.closure.N, wi) < 0.f;
        bool const canRefract     = !bsdf.isConductor && (bsdf.eta - 1.f) > 1e-3f &&
                                bsdf.fresnel.d.transmittanceTint.max() > 1e-4f;

        if (isTransmission && !canRefract)
            return RGB::fromScalar(0.f);

        // --- half-vector ---
        Vector3f H      = isTransmission ? normalize(bsdf.eta * wo + wi) : normalize(wo + wi);
        Vector3f HLocal = normalize(tangentSpace.toLocal(H));

        // --- Fresnel ---
        float const cosHI = dot(wiLocal, HLocal);
        RGB         reflectance{}, transmittance{};
        if (bsdf.isConductor)
            reflectance = fresnel::reflectanceConductor(cosHI, bsdf.fresnel.c.eta, bsdf.fresnel.c.etak);
        else
        {
            float F       = fresnel::reflectanceDielectric(fabsf(cosHI), bsdf.eta);
            reflectance   = F * bsdf.fresnel.d.reflectanceTint;
            transmittance = (1.f - F) * bsdf.fresnel.d.transmittanceTint;
        }

        if (reflectance.max() < 1e-7f && transmittance.max() < 1e-7f)
            return RGB::fromScalar(0.f);

        // --- microfacet terms ---
        float const D       = NDF(HLocal, bsdf.alphax, bsdf.alphay);
        float const lambdaI = auxiliaryLambda(wiLocal, bsdf.alphax, bsdf.alphay);
        float const lambdaO = auxiliaryLambda(woLocal, bsdf.alphax, bsdf.alphay);
        float const G       = heightCorrG(woLocal, wiLocal, bsdf.alphax, bsdf.alphay);

        // --- visible-normal PDF ---
        float const pdf_m = fmaxf(1e-12f, PDF(wiLocal, HLocal, bsdf.alphax, bsdf.alphay));

        float pdfReflect = 0.f;
        float lobePDF    = 1.f;

        RGB f = RGB::fromScalar(0.f);

        if (!isTransmission)
        {
            // --- reflection ---
            float denom = 4.f * cosNI * cosNO;
            if (denom <= 1e-12f)
                return RGB::fromScalar(0.f);

            RGB Fcol = bsdf.isConductor
                           ? fresnel::reflectanceConductor(fabsf(cosHI), bsdf.fresnel.c.eta, bsdf.fresnel.c.etak)
                           : bsdf.fresnel.d.reflectanceTint *
                                 RGB::fromScalar(fresnel::reflectanceDielectric(fabsf(cosHI), bsdf.eta));

            f = Fcol * D * G / denom;

            pdfReflect = reflectance.avg() / (reflectance + transmittance).avg();
            lobePDF    = pdfReflect;

            *pdf = pdf_m / (4.f * fabsf(cosHI)) * lobePDF;
        }
        else
        {
            // --- transmission (BTDF) ---
            float const i_dot_h = dot(wiLocal, HLocal);
            float const o_dot_h = dot(woLocal, HLocal);
            float const i_dot_n = cosNI;
            float const o_dot_n = cosNO;

            float const eta_i = 1.f;
            float const eta_o = bsdf.eta;

            float denomTerm = eta_i * i_dot_h + eta_o * o_dot_h;
            denomTerm       = denomTerm * denomTerm;
            if (denomTerm <= 1e-12f)
                return RGB::fromScalar(0.f);

            float const oneMinusF = 1.f - fresnel::reflectanceDielectric(fabsf(cosHI), bsdf.eta);
            float       bt_scalar = (fabsf(i_dot_h) * fabsf(o_dot_h)) / (i_dot_n * o_dot_n) * G * D * oneMinusF *
                              (eta_o * eta_o) / denomTerm;

            f = bt_scalar * bsdf.fresnel.d.transmittanceTint;

            pdfReflect = reflectance.avg() / (reflectance + transmittance).avg();
            lobePDF    = 1.f - pdfReflect;

            float jacobian = (eta_i * eta_i * fabsf(i_dot_h)) / denomTerm;
            *pdf           = pdf_m * jacobian * lobePDF;
        }

        f = f.saturate0();
        if (*pdf <= 0.f || fl::isInfOrNaN(*pdf))
        {
            *pdf = 0.f;
            return RGB::fromScalar(0.f);
        }

        return f;
    }

} // namespace dmt::ggx