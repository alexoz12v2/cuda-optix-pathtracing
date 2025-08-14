#include "core-material.h"

namespace dmt {
    static int32_t mipLevelsFromResolution(int32_t xRes, int32_t yRes)
    {
        int levels = 0;
        int w = xRes, h = yRes;
        while (w > 0 || h > 0) // better for EWA (more memory usage though)
        {
            ++levels;
            w >>= 1;
            h >>= 1;
        }
        return levels;
    }
    static RGB rgbFromByte3(uint32_t encoded)
    {
        RGB res{};
        res.r = static_cast<float>((encoded & 0x0000'ffff'0000'0000) >> 32) / 255.f;
        res.g = static_cast<float>((encoded & 0x0000'0000'ffff'0000) >> 16) / 255.f;
        res.b = static_cast<float>(encoded & 0x0000'0000'0000'ffff) / 255.f;
        return res;
    }

    static RGB sampleMippedTexture(TextureEvalContext const& ctx,
                                   TextureCache&             texCache,
                                   uint64_t                  key,
                                   int32_t                   width,
                                   int32_t                   height,
                                   bool                      normal)
    {
        // TODO; here we are assuming that uv are equal to texture coordinates. In a more general scenario, we would
        // compute a mapping function to generate texture coordinates form the `TextureEvalContext`. Assume s = u, t = v
        Vector2f const dstdx = Vector2f{ctx.dUV.dudx, ctx.dUV.dvdx};
        Vector2f const dstdy = Vector2f{ctx.dUV.dudy, ctx.dUV.dvdy};
        Point2f const  st    = ctx.uv;

        // choose ellipse axes and clamp them if necessary
        bool const     dxLonger   = dotSelf(dstdx) > dotSelf(dstdy);
        Vector2f const dst0       = dxLonger ? dstdx : dstdy;
        Vector2f       dst1       = dxLonger ? dstdy : dstdx;
        float          shorterLen = normL2(dst1);
        float const    longerLen  = normL2(dst0);
        assert(shorterLen != 0 && "you should implement a trilinear filtering fallback");

        if (float den = shorterLen * MaxAnisotropy; den < longerLen)
        {
            float const scale = longerLen / den;
            dst1 *= scale;
            shorterLen *= scale;
        }
        assert(shorterLen != 0 && "you should implement a trilinear filtering fallback");

        // choose 2 level of details and perform 2 EWA filtering lookup
        auto const lodRes = computeTextureLOD_from_dudv(ctx.dUV.dudx, ctx.dUV.dudy, ctx.dUV.dvdx, ctx.dUV.dvdy, width, height);
        int32_t const mipLevels = mipLevelsFromResolution(width, height);

        // for EWA
        float lambda = lodRes.lod_minor; // tends to preserve more detail; EWA addresses aliasing anisotropically
        int   ilod   = std::clamp(int(std::floor(lambda)), 0, mipLevels - 1);
        float t      = lambda - ilod;

        Point2i levelResLod0 = {std::max(1, width >> ilod), std::max(1, height >> ilod)};
        Point2i levelResLod1 = {std::max(1, width >> (ilod + 1)), std::max(1, height >> (ilod + 1))};

        uint32_t    bytesLod0;
        uint32_t    bytesLod1;
        TexFormat   texFormat;
        auto const* mortonLevelBufferLod0 = reinterpret_cast<unsigned char const*>(
            texCache.getOrInsert(key, ilod, bytesLod0, texFormat));
        auto const* mortonLevelBufferLod1 = reinterpret_cast<unsigned char const*>(
            texCache.getOrInsert(key, ilod + 1, bytesLod1, texFormat));

        // TODO handle wrap mode
        EWAParams paramsLod0{mortonLevelBufferLod0, levelResLod0, TexWrapMode::eMirror, TexWrapMode::eMirror, texFormat, normal};
        EWAParams paramsLod1{mortonLevelBufferLod1, levelResLod1, TexWrapMode::eMirror, TexWrapMode::eMirror, texFormat, normal};

        return lerp(EWAFormula(paramsLod0, st, dst0, dst1), EWAFormula(paramsLod1, st, dst0, dst1), t);
    }

    BSDFSample materialSample(SurfaceMaterial const&    material,
                              TextureCache&             cache,
                              TextureEvalContext const& texCtx,
                              Vector3f                  w,
                              Vector3f                  ng,
                              Point2f                   u,
                              float                     uc)
    {
        // sample normal
        Normal3f ns = normFromOcta(material.normalvalue);
        if (material.texMatMap & SurfaceMaterial::NormalMask)
        {
            int w = static_cast<int>(material.normalWidth), h = static_cast<int>(material.normalHeight);
            ns = sampleMippedTexture(texCtx, cache, material.normalkey, w, h, true).asVec();
            assert(fl::abs(dotSelf(ns) - 1.f) < 1e-5f);
        }

        // sample metallic texture with fallback. If 0 then pure dielectric, if 1 then pure conductor, otherwise compute both and lerp
        float metallic = material.metallicvalue;
        if (material.texMatMap & SurfaceMaterial::MetallicMask)
        {
            int w = static_cast<int>(material.metallicWidth), h = static_cast<int>(material.metallicHeight);
            metallic = sampleMippedTexture(texCtx, cache, material.metallickey, w, h, false).r;
        }

        // sample roughness
        float roughness = material.roughnessvalue;
        if (material.texMatMap & SurfaceMaterial::RoughnessMask)
        {
            int w = static_cast<int>(material.roughnessWidth), h = static_cast<int>(material.roughnessHeight);
            roughness = sampleMippedTexture(texCtx, cache, material.roughnesskey, w, h, false).r;
        }

        // sample diffuse (TODO colorspace?)
        RGB diffuse = rgbFromByte3(material.diffusevalue);
        if (material.texMatMap & SurfaceMaterial::DiffuseMask)
        {
            int w = static_cast<int>(material.diffuseWidth), h = static_cast<int>(material.diffuseHeight);
            diffuse = sampleMippedTexture(texCtx, cache, material.diffusekey, w, h, false);
        }

        // --- Step 2: Early exit for pure diffuse ---
        if (metallic <= 0.f && material.isDiffuseOpaque)
        {
            oren_nayar::BRDF brdf = oren_nayar::makeParams(roughness, diffuse, ns, w, material.multiscatterMultiplier, {1, 1, 1});
            float      pdf = 0.f;
            Vector3f   wr  = oren_nayar::sample(ns, ng, u, &pdf);
            BSDFSample result{};
            result.f   = oren_nayar::intensity(brdf, w, wr);
            result.pdf = pdf;
            result.eta = 1.f;
            result.wi  = wr;
            result.wm  = ns;
            return result;
        }

        // --- Step 3: Common GGX preparation ---
        float alphay = roughness;
        float alphax = material.anisotropy * roughness;

        ggx::BSDF dielectric = //
            ggx::makeDielectric(w, ns, ng, material.ior, alphax, alphay, material.reflectanceTint, material.transmittanceTint, diffuse);

        ggx::BSDF conductor = ggx::makeConductor(w, ns, ng, alphax, alphay, {1, 0, 0}, material.eta, material.etak, diffuse);

        // --- Step 4: Handle pure cases quickly ---
        if (metallic <= 0.f)
            return ggx::sample(dielectric, w, ng, u, uc);
        if (metallic >= 1.f)
            return ggx::sample(conductor, w, ng, u, uc);

        // --- Step 5: Mixed metallic: sample both and lerp ---
        // NOTE: Not exactly physically/numerically accurate for an unbiased monte carlo
        BSDFSample sDielectric = ggx::sample(dielectric, w, ng, u, uc);
        BSDFSample sConductor  = ggx::sample(conductor, w, ng, u, uc);

        BSDFSample result{};
        result.f   = lerp(sDielectric.f, sConductor.f, metallic);
        result.pdf = lerp(sDielectric.pdf, sConductor.pdf, metallic);
        result.eta = 1.f;
        result.wi  = sConductor.wi;
        result.wm  = sConductor.wm;
        return result;
    }
} // namespace dmt