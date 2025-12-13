#include "core-material.h"
#include "core-texture.h"
#include "cudautils/cudautils-vecmath.cuh"
#include "platform-utils.h"

namespace /*static*/ {
    using namespace dmt;
    int32_t mipLevelsFromResolution(int32_t xRes, int32_t yRes)
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

    static RGB sampleBilinearTexel(
        Point2f              st,
        Point2i              res,
        TexWrapMode          wrapX,
        TexWrapMode          wrapY,
        TexFormat            texFormat,
        bool                 isNormal,
        unsigned char const* mortonLevelBuffer)
    {
        // Map to texel space (same convention as EWA)
        float x = st.x * res.x - 0.5f;
        float y = st.y * res.y - 0.5f;

        int x0 = int(floor(x));
        int y0 = int(floor(y));
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float tx = x - x0;
        float ty = y - y0;

        RGB c00 = sampleMortonTexel(x0, y0, res, wrapX, wrapY, texFormat, false, mortonLevelBuffer);
        RGB c10 = sampleMortonTexel(x1, y0, res, wrapX, wrapY, texFormat, false, mortonLevelBuffer);
        RGB c01 = sampleMortonTexel(x0, y1, res, wrapX, wrapY, texFormat, false, mortonLevelBuffer);
        RGB c11 = sampleMortonTexel(x1, y1, res, wrapX, wrapY, texFormat, false, mortonLevelBuffer);

        RGB cx0 = lerp(c00, c10, tx);
        RGB cx1 = lerp(c01, c11, tx);
        RGB c   = lerp(cx0, cx1, ty);

        if (isNormal)
        {
            c.r = c.r * 2.0f - 1.0f;
            c.g = c.g * 2.0f - 1.0f;
        }

        return c;
    }

    static RGB sampleTrilinear(
        Point2f       st,
        int           width,
        int           height,
        int           mipLevels,
        float         lod,
        TexWrapMode   wrapX,
        TexWrapMode   wrapY,
        TexFormat&    texFormat,
        bool          isNormal,
        TextureCache& texCache,
        uint64_t      key)
    {
        int   ilod = clamp(int(floor(lod)), 0, mipLevels - 1);
        float t    = lod - ilod;

        Point2i res0 = {std::max(1, width >> ilod), std::max(1, height >> ilod)};
        Point2i res1 = {std::max(1, width >> (ilod + 1)), std::max(1, height >> (ilod + 1))};

        uint32_t bytes0 = 0, bytes1 = 0;

        auto* buf0 = static_cast<unsigned char const*>(texCache.getOrInsert(key, ilod, bytes0, texFormat));
        auto* buf1 = static_cast<unsigned char const*>(texCache.getOrInsert(key, ilod + 1, bytes1, texFormat));

        RGB c0 = sampleBilinearTexel(st, res0, wrapX, wrapY, texFormat, isNormal, buf0);
        RGB c1 = sampleBilinearTexel(st, res1, wrapX, wrapY, texFormat, isNormal, buf1);

        return lerp(c0, c1, t);
    }

    RGB sampleMippedTexture(TextureEvalContext const& ctx,
                            TextureCache&             texCache,
                            uint64_t                  key,
                            int32_t                   width,
                            int32_t                   height,
                            bool                      normal,
                            TexFormat&                outTexFormat)
    {
        // TODO; here we are assuming that uv are equal to texture coordinates. In a more general scenario, we would
        // compute a mapping function to generate texture coordinates form the `TextureEvalContext`. Assume s = u, t = v
        Vector2f const dstdx     = Vector2f{ctx.dUV.dudx, ctx.dUV.dvdx};
        Vector2f const dstdy     = Vector2f{ctx.dUV.dudy, ctx.dUV.dvdy};
        Point2f const  st        = ctx.uv;
        int32_t const  mipLevels = mipLevelsFromResolution(width, height);

        // choose ellipse axes and clamp them if necessary
        bool const     dxLonger   = dotSelf(dstdx) > dotSelf(dstdy);
        Vector2f const dst0       = dxLonger ? dstdx : dstdy;
        Vector2f       dst1       = dxLonger ? dstdy : dstdx;
        float          shorterLen = normL2(dst1);
        float const    longerLen  = normL2(dst0);

        // TODO Better temporary fix
        bool const validDiffs = fl::nearZero(ctx.dUV.dudx) || fl::nearZero(ctx.dUV.dudy) ||
                                fl::nearZero(ctx.dUV.dvdx) || fl::nearZero(ctx.dUV.dvdy);
        if (!validDiffs)
        {
            // compute 1 scalar LOD with *Isotropic Footprint Estimation*
            float const dud = fmaxf(fl::abs(ctx.dUV.dudx), fl::abs(ctx.dUV.dudy));
            float const dvd = fmaxf(fl::abs(ctx.dUV.dvdx), fl::abs(ctx.dUV.dvdy));

            float const rho = fmaxf(dud * width, dvd * height);
            float const lod = fmaxf(log2(fmaxf(rho, 1e-8f)), 0);
            return sampleTrilinear(ctx.uv,
                                   width,
                                   height,
                                   mipLevels,
                                   lod,
                                   TexWrapMode::eMirror,
                                   TexWrapMode::eMirror,
                                   outTexFormat,
                                   normal,
                                   texCache,
                                   key);
        }

        if (float den = shorterLen * maxAnisotropy; den < longerLen)
        {
            float const scale = longerLen / den;
            dst1 *= scale;
            shorterLen *= scale;
        }
        if (shorterLen == 0)
        {
            // compute 1 scalar LOD with *Isotropic Footprint Estimation*
            float const dud = fmaxf(fl::abs(ctx.dUV.dudx), fl::abs(ctx.dUV.dudy));
            float const dvd = fmaxf(fl::abs(ctx.dUV.dvdx), fl::abs(ctx.dUV.dvdy));

            float const rho = fmaxf(dud * width, dvd * height);
            float const lod = fmaxf(log2(fmaxf(rho, 1e-8f)), 0);
            return sampleTrilinear(ctx.uv,
                                   width,
                                   height,
                                   mipLevels,
                                   lod,
                                   TexWrapMode::eMirror,
                                   TexWrapMode::eMirror,
                                   outTexFormat,
                                   normal,
                                   texCache,
                                   key);
        }

        // choose 2 level of details and perform 2 EWA filtering lookup
        auto const lodRes = computeTextureLOD_from_dudv(ctx.dUV.dudx, ctx.dUV.dudy, ctx.dUV.dvdx, ctx.dUV.dvdy, width, height);

        // for EWA
        float const lambda = lodRes.lod_minor; // tends to preserve more detail; EWA addresses aliasing anisotropically
        int const   ilod   = std::clamp(static_cast<int>(std::floor(lambda)), 0, mipLevels - 1);
        float const t      = lambda - ilod;

        Point2i const levelResLod0 = {std::max(1, width >> ilod), std::max(1, height >> ilod)};
        Point2i const levelResLod1 = {std::max(1, width >> (ilod + 1)), std::max(1, height >> (ilod + 1))};

        uint32_t    bytesLod0             = 0;
        uint32_t    bytesLod1             = 0;
        auto const* mortonLevelBufferLod0 = static_cast<unsigned char const*>(
            texCache.getOrInsert(key, ilod, bytesLod0, outTexFormat));
        auto const* mortonLevelBufferLod1 = static_cast<unsigned char const*>(
            texCache.getOrInsert(key, ilod + 1, bytesLod1, outTexFormat));
        assert(mortonLevelBufferLod0 && mortonLevelBufferLod1);
        assert(isAligned(mortonLevelBufferLod0, alignPerPixel(outTexFormat)));
        assert(isAligned(mortonLevelBufferLod1, alignPerPixel(outTexFormat)));

        // TODO handle wrap mode
        EWAParams const paramsLod0{.mortonLevelBuffer = mortonLevelBufferLod0,
                                   .levelRes          = levelResLod0,
                                   .wrapX             = TexWrapMode::eMirror,
                                   .wrapY             = TexWrapMode::eMirror,
                                   .texFormat         = outTexFormat,
                                   .isNormal          = normal};
        EWAParams const paramsLod1{.mortonLevelBuffer = mortonLevelBufferLod1,
                                   .levelRes          = levelResLod1,
                                   .wrapX             = TexWrapMode::eMirror,
                                   .wrapY             = TexWrapMode::eMirror,
                                   .texFormat         = outTexFormat,
                                   .isNormal          = normal};
        return lerp(EWAFormula(paramsLod0, st, dst0, dst1), EWAFormula(paramsLod1, st, dst0, dst1), t);
    }
} // namespace

namespace dmt {

    BSDFSample materialSample(SurfaceMaterial const&    material,
                              TextureCache&             cache,
                              TextureEvalContext const& texCtx,
                              Vector3f                  w,
                              Vector3f                  ng,
                              Point2f                   u,
                              float                     uc)
    {
        // sample normal
        Normal3f  ns;
        Vector3f  nMap;
        RGB       nMapSampled;
        TexFormat texFormat{};
        if (material.useShadingNormals)
        {
            ns = normFromOcta(material.normalvalue);
            if (material.texMatMap & SurfaceMaterial::NormalMask)
            {
                int width   = static_cast<int>(material.normalWidth);
                int height  = static_cast<int>(material.normalHeight);
                nMapSampled = sampleMippedTexture(texCtx, cache, material.normalkey, width, height, true, texFormat);
                nMap        = nMapSampled.asVec();
                if (texFormat != TexFormat::FloatRGB)
                    nMap = normalize(map(nMap, [](float const x) { return fl::quantize(x); }));
                Frame const frame = Frame::fromZ(ng);
                ns                = safeNormalizeFallback(frame.fromLocal(nMap), ng);
#if defined(DMT_NORMAL_MAP_D3D_STYLE)
                nMap.y = -nMap.y;
#endif
                // if normal and image not 32 bit, then quantize and normalize the sampled result
                assert(fl::abs(dotSelf(ns) - 1.f) < 1e-5f);
            }
        }
        else
            ns = ng;

        // sample metallic texture with fallback. If 0 then pure dielectric, if 1 then pure conductor, otherwise compute both and lerp
        float metallic = material.metallicvalue;
        if (material.texMatMap & SurfaceMaterial::MetallicMask)
        {
            int width = static_cast<int>(material.metallicWidth), h = static_cast<int>(material.metallicHeight);
            metallic = sampleMippedTexture(texCtx, cache, material.metallickey, width, h, false, texFormat).r;
        }

        // sample roughness
        float roughness = material.roughnessvalue;
        if (material.texMatMap & SurfaceMaterial::RoughnessMask)
        {
            int w = static_cast<int>(material.roughnessWidth), h = static_cast<int>(material.roughnessHeight);
            roughness = sampleMippedTexture(texCtx, cache, material.roughnesskey, w, h, false, texFormat).r;
        }

        // sample diffuse (TODO colorspace?)
        RGB diffuse = rgbFromByte3(material.diffusevalue);
        if (material.texMatMap & SurfaceMaterial::DiffuseMask)
        {
            int w = static_cast<int>(material.diffuseWidth), h = static_cast<int>(material.diffuseHeight);
            diffuse = sampleMippedTexture(texCtx, cache, material.diffusekey, w, h, false, texFormat);
        }

        if (dot(ng, w) < 0) // TODO maybe flip sign bit
            ng = ng * Vector3f::s(-1);

        // --- Step 2: Early exit for pure diffuse ---
        if (metallic <= 0.f && material.isDiffuseOpaque)
        {
            if (dot(ng, ns) < 0)
                ns = ns * Vector3f::s(-1);
            oren_nayar::BRDF brdf = oren_nayar::makeParams(roughness, diffuse, ns, w, material.multiscatterMultiplier, {1, 1, 1});
            float      pdf = 0.f;
            Vector3f   wr  = oren_nayar::sample(ns, ng, u, &pdf);
            BSDFSample result{};
            result.f   = oren_nayar::intensity(brdf, w, wr);
            result.pdf = pdf;
            result.eta = 1.f;
            result.wi  = wr;
            result.wm  = ns;
            assert(dot(result.wi, ns) > 0);
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

    BSDFEval materialEval(SurfaceMaterial const&    material,
                          TextureCache&             cache,
                          TextureEvalContext const& texCtx,
                          Vector3f                  wo,
                          Vector3f                  wi,
                          Vector3f                  ng)
    {
        // --- Step 1: Evaluate shading normal ---
        Normal3f  ns;
        TexFormat texFormat{};

        if (material.useShadingNormals)
        {
            ns = normFromOcta(material.normalvalue);
            if (material.texMatMap & SurfaceMaterial::NormalMask)
            {
                int      w = static_cast<int>(material.normalWidth), h = static_cast<int>(material.normalHeight);
                Vector3f nMap = sampleMippedTexture(texCtx, cache, material.normalkey, w, h, true, texFormat).asVec();
                if (texFormat != TexFormat::FloatRGB)
                    nMap = normalize(map(nMap, [](float const x) { return fl::quantize(x); }));
                Frame const frame = Frame::fromZ(ng);
                ns                = safeNormalizeFallback(frame.fromLocal(nMap), ng);
            }
        }
        else
            ns = ng;

        // --- Step 2: Material parameters ---
        float metallic = material.metallicvalue;
        if (material.texMatMap & SurfaceMaterial::MetallicMask)
        {
            int w = static_cast<int>(material.metallicWidth), h = static_cast<int>(material.metallicHeight);
            metallic = sampleMippedTexture(texCtx, cache, material.metallickey, w, h, false, texFormat).r;
        }

        float roughness = material.roughnessvalue;
        if (material.texMatMap & SurfaceMaterial::RoughnessMask)
        {
            int w = static_cast<int>(material.roughnessWidth), h = static_cast<int>(material.roughnessHeight);
            roughness = sampleMippedTexture(texCtx, cache, material.roughnesskey, w, h, false, texFormat).r;
        }

        RGB diffuse = rgbFromByte3(material.diffusevalue);
        if (material.texMatMap & SurfaceMaterial::DiffuseMask)
        {
            int w = static_cast<int>(material.diffuseWidth), h = static_cast<int>(material.diffuseHeight);
            diffuse = sampleMippedTexture(texCtx, cache, material.diffusekey, w, h, false, texFormat);
        }

        if (dot(ng, wo) < 0) // TODO maybe flip sign bit
            ng = ng * Vector3f::s(-1);

        // --- Step 3: Early exit for pure diffuse ---
        if (metallic <= 0.f && material.isDiffuseOpaque)
        {
            if (dot(ng, ns) < 0)
                ns = ns * Vector3f::s(-1);
            oren_nayar::BRDF
                  brdf = oren_nayar::makeParams(roughness, diffuse, ns, wi, material.multiscatterMultiplier, {1, 1, 1});
            float pdf  = std::max(0.f, dot(ns, wi)) * fl::rcpPi(); // cosine-weighted hemisphere pdf
            RGB   f    = oren_nayar::intensity(brdf, wo, wi);

            BSDFEval result{};
            result.f   = f;
            result.pdf = pdf;
            result.eta = 1.f;
            result.wi  = wi;
            result.wm  = ns;
            return result;
        }

        // --- Step 4: Setup GGX ---
        float alphay = roughness;
        float alphax = material.anisotropy * roughness;

        ggx::BSDF dielectric = ggx::makeDielectric(wo,
                                                   ns,
                                                   ng,
                                                   material.ior,
                                                   alphax,
                                                   alphay,
                                                   material.reflectanceTint,
                                                   material.transmittanceTint,
                                                   diffuse);

        ggx::BSDF conductor = ggx::makeConductor(wo, ns, ng, alphax, alphay, {1, 0, 0}, material.eta, material.etak, diffuse);

        // --- Step 5: Evaluate ---
        float pdfDielectric = 0.f, pdfConductor = 0.f;
        RGB   fDielectric = ggx::eval(dielectric, wo, wi, ng, &pdfDielectric);
        RGB   fConductor  = ggx::eval(conductor, wo, wi, ng, &pdfConductor);

        BSDFEval result{};
        if (metallic <= 0.f)
        {
            result.f   = fDielectric;
            result.pdf = pdfDielectric;
            result.eta = dielectric.eta;
        }
        else if (metallic >= 1.f)
        {
            result.f   = fConductor;
            result.pdf = pdfConductor;
            result.eta = 1.f;
        }
        else
        {
            // blend (not physically rigorous, matches your sampler)
            result.f   = lerp(fDielectric, fConductor, metallic);
            result.pdf = lerp(pdfDielectric, pdfConductor, metallic);
            result.eta = 1.f;
        }

        result.wi = wi;
        result.wm = ns;
        return result;
    }

    Normal3f materialShadingNormal(SurfaceMaterial const& material, TextureCache& cache, TextureEvalContext const& texCtx, Vector3f ng)
    {
        if (!material.useShadingNormals)
            return ng;
        // Default: object-space/oct-encoded normal
        Normal3f ns = normFromOcta(material.normalvalue);

        // If there�s a normal map, sample it
        if (material.texMatMap & SurfaceMaterial::NormalMask)
        {
            int const w = static_cast<int>(material.normalWidth);
            int const h = static_cast<int>(material.normalHeight);

            TexFormat format{};
            Vector3f  nMap = sampleMippedTexture(texCtx, cache, material.normalkey, w, h, true, format).asVec();
            if (format != TexFormat::FloatRGB)
                nMap = normalize(map(nMap, [](float const x) { return fl::quantize(x, 4); }));
            Frame const tangentFrame = Frame::fromZ(ng);

            // Ensure normalized, fallback to ng if something degenerate
            ns = safeNormalizeFallback(tangentFrame.fromLocal(nMap), ng);
        }

        return ns;
    }
} // namespace dmt