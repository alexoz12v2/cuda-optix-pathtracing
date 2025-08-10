#include "core-texture.h"

namespace dmt {
    void approximate_dp_dxy(ApproxDifferentialsContext const& apctx, Vector3f* dpdx, Vector3f* dpdy)
    {
        assert(fl::abs(normL2(apctx.n) - 1.f) < 1e-5f && dpdx && dpdy);

        // compute tangent plane equation for ray differential intersection
        Point3f const   pCamera         = apctx.cameraFromRender(apctx.p);
        Transform const downZFromCamera = Transform::rotateFromTo(normalize(pCamera), {0, 0, 1});
        Point3f const   pDownZ          = downZFromCamera(pCamera);
        Normal3f const  nDownZ = downZFromCamera(apctx.cameraFromRender(normalFrom(apctx.n))); // apply -T automatically
        float const     d      = nDownZ.z * pDownZ.z;

        // find intersection points for approximated camera differential rays
        Ray const   xRay(Point3f{0, 0, 0} + apctx.minPosDifferentialX,
                       normalize(Vector3f{0, 0, 1} + apctx.minDirDifferentialX));
        Ray const   yRay(Point3f{0, 0, 0} + apctx.minPosDifferentialY,
                       normalize(Vector3f{0, 0, 1} + apctx.minDirDifferentialY));
        float const tx = -(dot(nDownZ, xRay.o) - d) / dot(nDownZ, xRay.d); // plane eq = ray eq
        float const ty = -(dot(nDownZ, yRay.o) - d) / dot(nDownZ, yRay.d);

        Point3f const px = xRay(tx), py = yRay(ty);

        // estimate dpdx and dpdy in  tangent plane at intersection point
        float const sppScale = fmaxf(.125f, fl::rsqrt(static_cast<float>(apctx.samplesPerPixel)));

        *dpdx = sppScale * apctx.cameraFromRender.applyInverse(downZFromCamera.applyInverse(px - pDownZ));
        *dpdy = sppScale * apctx.cameraFromRender.applyInverse(downZFromCamera.applyInverse(py - pDownZ));
    }

    UVDifferentials duv_From_dp_dxy(UVDifferentialsContext const& uvctx)
    {
        UVDifferentials ret{};

        // Least Square method to solve 2 linear systems with chain rules
        float const ata00 = dot(uvctx.dpdu, uvctx.dpdu);
        float const ata01 = dot(uvctx.dpdu, uvctx.dpdv);
        float const ata11 = dot(uvctx.dpdv, uvctx.dpdv);

        float invDet = fl::rcp(fl::FMA(ata00, ata11, -ata01 * ata01));
        invDet       = !fl::isinf(invDet) ? invDet : 0.f;

        // Compute $\transpose{\XFORM{A}} \VEC{b}$ for $x$ and $y$
        float const atb0x = dot(uvctx.dpdu, uvctx.dpdx);
        float const atb1x = dot(uvctx.dpdv, uvctx.dpdx);
        float const atb0y = dot(uvctx.dpdu, uvctx.dpdy);
        float const atb1y = dot(uvctx.dpdv, uvctx.dpdy);

        // Compute $u$ and $v$ derivatives with respect to $x$ and $y$
        ret.dudx = fl::FMA(ata11, atb0x, -ata01 * atb1x) * invDet;
        ret.dvdx = fl::FMA(ata00, atb1x, -ata01 * atb0x) * invDet;
        ret.dudy = fl::FMA(ata11, atb0y, -ata01 * atb1y) * invDet;
        ret.dvdy = fl::FMA(ata00, atb1y, -ata01 * atb0y) * invDet;

        // Clamp derivatives of $u$ and $v$ to reasonable values
        ret.dudx = !fl::isinf(ret.dudx) ? fl::clamp(ret.dudx, -1e8f, 1e8f) : 0.f;
        ret.dvdx = !fl::isinf(ret.dvdx) ? fl::clamp(ret.dvdx, -1e8f, 1e8f) : 0.f;
        ret.dudy = !fl::isinf(ret.dudy) ? fl::clamp(ret.dudy, -1e8f, 1e8f) : 0.f;
        ret.dvdy = !fl::isinf(ret.dvdy) ? fl::clamp(ret.dvdy, -1e8f, 1e8f) : 0.f;

        return ret;
    }

    // -- Checker Texture --
    float CheckerTexture::evalFloat(TextureEvalContext const& ctx) const
    {
        // Compute approximate filter footprint in texture space
        float dudx = ctx.dUV.dudx * scaleU;
        float dudy = ctx.dUV.dudy * scaleU;
        float dvdx = ctx.dUV.dvdx * scaleV;
        float dvdy = ctx.dUV.dvdy * scaleV;

        float footprintU = fmaxf(fabsf(dudx), fabsf(dudy));
        float footprintV = fmaxf(fabsf(dvdx), fabsf(dvdy));

        float u = ctx.uv.x * scaleU;
        float v = ctx.uv.y * scaleV;

        // If footprint is large (pixel spans multiple checkers), blend
        if (footprintU > 2.0f || footprintV > 2.0f)
        {
            return 0.5f * (1.0f + 0.0f); // average of the two
        }

        int  iu      = static_cast<int>(floorf(u));
        int  iv      = static_cast<int>(floorf(v));
        bool checker = (iu + iv) % 2 == 0;
        return checker ? 1.0f : 0.0f;
    }

    RGB CheckerTexture::evalRGB(TextureEvalContext const& ctx) const
    {
        float dudx = ctx.dUV.dudx * scaleU;
        float dudy = ctx.dUV.dudy * scaleU;
        float dvdx = ctx.dUV.dvdx * scaleV;
        float dvdy = ctx.dUV.dvdy * scaleV;

        float footprintU = fmaxf(fabsf(dudx), fabsf(dudy));
        float footprintV = fmaxf(fabsf(dvdx), fabsf(dvdy));

        float u = ctx.uv.x * scaleU;
        float v = ctx.uv.y * scaleV;

        if (footprintU > 2.0f || footprintV > 2.0f)
        {
            return 0.5f * (color1 + color2);
        }

        int  iu      = static_cast<int>(floorf(u));
        int  iv      = static_cast<int>(floorf(v));
        bool checker = (iu + iv) % 2 == 0;
        return checker ? color1 : color2;
    }

    // -- Image Texture --
    static int32_t log2Int(float v)
    {
        if (v < 1.f)
            return -log2Int(fl::rcp(v));
        uint32_t const midsignif = 0b00000000001101010000010011110011; // significand of floating point pow(2, 1.5)
        return fl::exponent(v) + (fl::significand(v) >= midsignif ? 1 : 0);
    }

    size_t mortonLevelOffset(int baseW, int baseH, int level)
    {
        size_t offset = 0;
        for (int l = 0; l < level; ++l)
        {
            int w = baseW >> l;
            int h = baseH >> l;
            offset += size_t(w) * size_t(h);
        }
        return offset;
    }

    size_t mipChainPixelCount(int w, int h)
    {
        size_t total = 0;
        while (w > 0 || h > 0)
        {
            total += size_t(w) * size_t(h);
            w >>= 1;
            h >>= 1;
        }
        return total;
    }

    ImageTexturev2 makeRGBMipmappedTexture(
        RGB const*                 image,
        int32_t                    xRes,
        int32_t                    yRes,
        TexWrapMode                wrapModeX,
        TexWrapMode                wrapModeY,
        std::pmr::memory_resource* memory)
    {
        assert(memory && image && xRes > 0 && yRes > 0);
        ImageTexturev2 tex{};

        if (!isPOT(xRes) || !isPOT(yRes))
            return tex;

        tex.width     = xRes;
        tex.height    = yRes;
        tex.isRGB     = 1;
        tex.deviceIdx = -1; // TODO: upload to GPU later
        tex.wrapModeX = wrapModeX;
        tex.wrapModeY = wrapModeY;

        // Mip levels: stop when both dims hit 1
        int levels = 0;
        int w = xRes, h = yRes;
#if defined(DMT_IGNORE_ANISOTROPY)
    #error "you should modify EWA implementation!"
        while (w > 0 && h > 0) // PBRT uses (w > 0 || h > 0) ie create more levels for anisotropy textures
#else
        while (w > 0 || h > 0) // better for EWA (more memory usage though)
#endif
        {
            ++levels;
            w >>= 1;
            h >>= 1;
        }
        tex.mipLevels = levels;

        // Allocate exact space for the mip chain
        size_t totalPixels = mipChainPixelCount(xRes, yRes);
        size_t totalBytes  = totalPixels * sizeof(RGB);
        void*  buffer      = memory->allocate(totalBytes, 32);
        if (!buffer)
            return tex;

        tex.data.rgb = reinterpret_cast<RGB*>(buffer);

        // --- Copy level 0 (base image) in Morton order ---
        for (int v = 0; v < yRes; ++v)
        {
            for (int u = 0; u < xRes; ++u)
            {
                uint32_t mortonIdx      = encodeMorton2D(u, v);
                tex.data.rgb[mortonIdx] = image[u + v * xRes];
            }
        }

        // --- Generate lower mips ---
        for (int level = 1; level < levels; ++level)
        {
            int prevW = xRes >> (level - 1);
            int prevH = yRes >> (level - 1);
            int currW = xRes >> level;
            int currH = yRes >> level;

            RGB* prevLevel = tex.data.rgb + mortonLevelOffset(xRes, yRes, level - 1);
            RGB* currLevel = tex.data.rgb + mortonLevelOffset(xRes, yRes, level);

            for (int v = 0; v < currH; ++v)
            {
                for (int u = 0; u < currW; ++u)
                { // clang-format off
                    RGB c00 = prevLevel[encodeMorton2D(u * 2,     v * 2)];
                    RGB c10 = prevLevel[encodeMorton2D(u * 2 + 1, v * 2)];
                    RGB c01 = prevLevel[encodeMorton2D(u * 2,     v * 2 + 1)];
                    RGB c11 = prevLevel[encodeMorton2D(u * 2 + 1, v * 2 + 1)];
                    // clang-format on
                    currLevel[encodeMorton2D(u, v)] = 0.25f * (c00 + c10 + c01 + c11);
                }
            }
        }

        return tex;
    }

    void freeImageTexture(ImageTexturev2& tex, std::pmr::memory_resource* memory)
    {
        size_t totalPixels = mipChainPixelCount(tex.width, tex.height);
        size_t totalBytes  = totalPixels;
        if (tex.isRGB)
            totalBytes *= sizeof(RGB);
        else if (tex.isNormal)
            totalBytes *= sizeof(OctahedralNorm);
        else
            totalBytes *= sizeof(float);
        memory->deallocate(tex.data.L, totalBytes, 32);
        std::memset(&tex, 0, sizeof(ImageTexturev2));
    }

    float ImageTexturev2::evalFloat(TextureEvalContext const& ctx) const
    {
        if (isRGB || isNormal)
            return 0.f;
        // TODO
        return 0.f;
    }

    static RGB sampleMortonTexel(int s, int t, Point2i levelRes, TexWrapMode wrapX, TexWrapMode wrapY, RGB const* mortonLevelBuffer)
    {
        constexpr auto wrapCoord = [](int coord, int size, TexWrapMode mode) -> int {
            switch (mode)
            {
                case TexWrapMode::eRepeat:
                    coord = coord % size;
                    if (coord < 0)
                        coord += size;
                    return coord;
                case TexWrapMode::eMirror:
                {
                    int period = size * 2;
                    coord      = coord % period;
                    if (coord < 0)
                        coord += period;
                    return coord < size ? coord : (period - coord - 1);
                }
                case TexWrapMode::eClamp: [[fallthrough]];
                default: return std::clamp(coord, 0, size - 1);
            }
        };

        int u = wrapCoord(s, levelRes.x, wrapX);
        int v = wrapCoord(t, levelRes.y, wrapY);

        uint32_t mortonIdx = encodeMorton2D(u, v);
        return mortonLevelBuffer[mortonIdx];
    }

    // TODO __constant__ memory for GPU
    static auto makeEwaWeightLUT(float alpha = 2.0f)
    {
        std::array<float, EWA_LUT_SIZE> lut{};
        for (uint32_t i = 0; i < EWA_LUT_SIZE; ++i)
        {
            float t = float(i) / float(EWA_LUT_SIZE - 1); // normalized [0,1]
            lut[i]  = std::exp(-alpha * t);
        }
        return lut;
    }

    // computed when DLL is brought up to memory
    std::array<float, EWA_LUT_SIZE> EwaWeightLUT = makeEwaWeightLUT(2.f);

    struct LODResult
    {
        float sigma_max; // major axis length (texels)
        float sigma_min; // minor axis length (texels)
        float lod_major; // = max(0, log2(sigma_max))
        float lod_minor; // = max(0, log2(sigma_min))  <-- PBRT/EWA uses this typically
    };

    // ctx.dUV contains dudx, dudy, dvdx, dvdy
    // texWidth/texHeight are base-level resolution (not mip-resolved)
    inline LODResult computeTextureLOD_from_dudv(float dudx, float dudy, float dvdx, float dvdy, int texWidth, int texHeight)
    {
        // Convert derivatives to texel units (important!)
        // a = dstdx_texel = (dudx * texWidth, dvdx * texHeight)
        // b = dstdy_texel = (dudy * texWidth, dvdy * texHeight)
        float a0 = dudx * float(texWidth);
        float a1 = dvdx * float(texHeight);
        float b0 = dudy * float(texWidth);
        float b1 = dvdy * float(texHeight);

        // Build symmetric matrix M = a a^T + b b^T
        // M = [ E  G ]
        //     [ G  F ]
        float E = a0 * a0 + a1 * a1; // dot(a,a)
        float F = b0 * b0 + b1 * b1; // dot(b,b)
        float G = a0 * b0 + a1 * b1; // dot(a,b)

        // Numerical guard
        float const eps = 1e-12f;

        // Compute eigenvalues of 2x2 symmetric matrix
        // trace = E + F
        // det = E*F - G*G
        float trace = E + F;
        float det   = E * F - G * G;

        // Clamp tiny negative det due to numerical noise
        if (det < 0.f && det > -eps)
            det = 0.f;

        // If both a and b are (almost) zero, return tiny sigma
        if (trace <= eps)
        {
            LODResult r{};
            r.sigma_max = 0.f;
            r.sigma_min = 0.f;
            r.lod_major = 0.f;
            r.lod_minor = 0.f;
            return r;
        }

        // eigenvalues = 0.5 * (trace ± sqrt(trace^2 - 4*det))
        float discr = trace * trace - 4.f * det;
        if (discr < 0.f)
            discr = 0.f; // numeric safety
        float sqrtD = std::sqrt(discr);

        float lambda1 = 0.5f * (trace + sqrtD); // >=
        float lambda2 = 0.5f * (trace - sqrtD); // <=

        // Clamp tiny negatives
        if (lambda1 < 0.f && lambda1 > -eps)
            lambda1 = 0.f;
        if (lambda2 < 0.f && lambda2 > -eps)
            lambda2 = 0.f;

        // singular values = sqrt(eigenvalues)
        float sigma1 = lambda1 > 0.f ? std::sqrt(lambda1) : 0.f;
        float sigma2 = lambda2 > 0.f ? std::sqrt(lambda2) : 0.f;

        // LODs = log2(sigma) but clamp to >= 0 (base level)
        float lod1 = (sigma1 > 0.f) ? std::log2(sigma1) : -INFINITY;
        float lod2 = (sigma2 > 0.f) ? std::log2(sigma2) : -INFINITY;

        LODResult res;
        res.sigma_max = std::max(sigma1, sigma2);
        res.sigma_min = std::min(sigma1, sigma2);
        res.lod_major = std::max(0.f, lod1);
        res.lod_minor = std::max(0.f, lod2);
        return res;
    }

    // TODO with all types of data (normals are an exception)
    static RGB EWA(ImageTexturev2 const* tex, int32_t ilod, Point2f st_in, Vector2f dst0_in, Vector2f dst1_in)
    {
        // If beyond last level: return a single texel
        if (ilod >= tex->mipLevels)
        {
            size_t off = mortonLevelOffset(tex->width, tex->height, tex->mipLevels - 1);
            return tex->data.rgb[off];
        }

        // Pointer to this mip level
        RGB const* mortonLevelBuffer = tex->data.rgb + mortonLevelOffset(tex->width, tex->height, ilod);

        // Mip level resolution
        Point2i levelRes = {std::max(1, tex->width >> ilod), std::max(1, tex->height >> ilod)};

        // Scale derivatives to this mip level in texel space
        Vector2f dst0 = {dst0_in.x * levelRes.x, dst0_in.y * levelRes.y};
        Vector2f dst1 = {dst1_in.x * levelRes.x, dst1_in.y * levelRes.y};

        // Map st to texel space (centered at -0.5)
        float sx = st_in.x * float(levelRes.x) - 0.5f;
        float sy = st_in.y * float(levelRes.y) - 0.5f;

        // Ellipse coefficients (PBRT style)
        float A = dst0.y * dst0.y + dst1.y * dst1.y;
        float B = -2.0f * (dst0.x * dst0.y + dst1.x * dst1.y);
        float C = dst0.x * dst0.x + dst1.x * dst1.x;

        // Normalize
        float invF = 1.0f / (A * C - 0.25f * B * B);
        if (!(invF > 0.0f))
        {
            // Degenerate: nearest
            return sampleMortonTexel(int(std::floor(sx + 0.5f)),
                                     int(std::floor(sy + 0.5f)),
                                     levelRes,
                                     tex->wrapModeX,
                                     tex->wrapModeY,
                                     mortonLevelBuffer);
        }
        A *= invF;
        B *= invF;
        C *= invF;

        // Add 1 to major diag terms for pixel reconstruction filter
        A += 1.0f;
        C += 1.0f;

        // Bounding box extents
        float invDet  = 1.0f / (A * C - 0.25f * B * B);
        float uRadius = std::sqrt(C * invDet);
        float vRadius = std::sqrt(A * invDet);

#if 1 // Debug radii
        if (uRadius > 20.0f || vRadius > 20.0f)
        {
            Context ctx;
            ctx.warn("    LOD {} | uRad={} vRad={} | A={} B={} C={}", std::make_tuple(ilod, uRadius, vRadius, A, B, C));
        }
#endif

        int s0 = int(std::ceil(sx - uRadius));
        int s1 = int(std::floor(sx + uRadius));
        int t0 = int(std::ceil(sy - vRadius));
        int t1 = int(std::floor(sy + vRadius));

        RGB   sum{};
        float sumW = 0.0f;

        for (int t = t0; t <= t1; ++t)
        {
            float tt = float(t) - sy;
            for (int s = s0; s <= s1; ++s)
            {
                float ss = float(s) - sx;
                float r2 = A * ss * ss + B * ss * tt + C * tt * tt;

                if (r2 < 1.0f)
                {
                    float r2c = std::clamp(r2, 0.0f, 0.999999f);
                    int   idx = int(r2c * float(EWA_LUT_SIZE - 1));
                    idx       = std::min(std::max(idx, 0), int(EWA_LUT_SIZE - 1));

                    float w     = EwaWeightLUT[idx];
                    RGB   texel = sampleMortonTexel(s, t, levelRes, tex->wrapModeX, tex->wrapModeY, mortonLevelBuffer);
                    sum += texel * w;
                    sumW += w;
                }
            }
        }

        if (!(sumW > 0.0f))
        {
            return sampleMortonTexel(int(std::floor(sx + 0.5f)),
                                     int(std::floor(sy + 0.5f)),
                                     levelRes,
                                     tex->wrapModeX,
                                     tex->wrapModeY,
                                     mortonLevelBuffer);
        }

        return sum / sumW;
    }

    RGB ImageTexturev2::evalRGB(TextureEvalContext const& ctx) const
    {
        if (!isRGB)
            return RGB::fromScalar(0.f);

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
        auto lodRes = computeTextureLOD_from_dudv(ctx.dUV.dudx, ctx.dUV.dudy, ctx.dUV.dvdx, ctx.dUV.dvdy, width, height);

#if 1
        Context actx;
        // clang-format off
        actx.log(" [TEXTURE SAMPLING]: st {} {} | p {} {} {} |", std::make_tuple(st.x, st.y, ctx.p.x, ctx.p.y, ctx.p.z));
        actx.log("         \xCF\x83_min {} | \xCF\x83_max {} (If too higher than texel resolution, then differential computation is wrong!)", std::make_tuple(lodRes.sigma_min, lodRes.sigma_max));
        // clang-format on
#endif

        // for EWA
        float lambda = lodRes.lod_minor; // tends to preserve more detail; EWA addresses aliasing anisotropically
        int   ilod   = std::clamp(int(std::floor(lambda)), 0, mipLevels - 1);
        float t      = lambda - ilod;

        // for trilinear filtering use lodRes.lod_major

        return lerp(EWA(this, ilod, st, dst0, dst1), EWA(this, ilod + 1, st, dst0, dst1), t);
    }

    // -- Texture Dispatch Type --
    float TextureVariant::evalFloat(TextureEvalContext const& ctx) const
    {
        switch (type)
        {
            case TextureType::eImage: return payload.image.evalFloat(ctx);
            case TextureType::eChecker: return payload.checker.evalFloat(ctx);
            default: assert(false); return 0.0f;
        }
    }

    RGB TextureVariant::evalRGB(TextureEvalContext const& ctx) const
    {
        switch (type)
        {
            case TextureType::eImage: return payload.image.evalRGB(ctx);
            case TextureType::eChecker: return payload.checker.evalRGB(ctx);
            default: assert(false); return RGB::fromScalar(0.0f);
        }
    }
} // namespace dmt