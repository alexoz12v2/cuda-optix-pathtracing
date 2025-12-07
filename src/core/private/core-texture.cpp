#include "core-texture.h"

#include "core-texture-cache.h"

#include <algorithm>
#include <cmath>

namespace dmt {
    struct Byte3
    {
        Byte3() = default;
        Byte3(RGB rgb) :
        r{static_cast<uint8_t>(fminf(rgb.r * 255.f, 255.f))},
        g{static_cast<uint8_t>(fminf(rgb.g * 255.f, 255.f))},
        b{static_cast<uint8_t>(fminf(rgb.b * 255.f, 255.f))}
        {
        }

        explicit operator RGB() const
        {
            RGB rgb{};
            rgb.r = static_cast<float>(r) / 255.f;
            rgb.g = static_cast<float>(g) / 255.f;
            rgb.b = static_cast<float>(b) / 255.f;
            return rgb;
        }

        uint8_t r, g, b;
    };

    static uint8_t toByte(float t) { return static_cast<uint8_t>(fl::clamp(t * 255.f, 0, 255.f)); }

    struct Half3
    {
        Half3() = default;
        Half3(RGB rgb) : r{rgb.r}, g{rgb.g}, b{rgb.b} {}

        explicit operator RGB() const
        {
            RGB rgb{};
            rgb.r = static_cast<float>(r);
            rgb.g = static_cast<float>(g);
            rgb.b = static_cast<float>(b);
            return rgb;
        }

        Half r, g, b;
    };

    uint32_t bytesPerPixel(TexFormat texFormat)
    {
        using enum TexFormat;
        switch (texFormat)
        {
            case FloatRGB: return 12;
            case HalfRGB: return 6;
            case ByteRGB: return 3;
            case FloatGray: return 4;
            case HalfGray: return 2;
            case ByteGray: [[fallthrough]];
            default: return 1;
        }
    }

    size_t alignPerPixel(TexFormat tex)
    {
        using enum TexFormat;
        switch (tex)
        {
            case FloatRGB: [[fallthrough]];
            case FloatGray: return 4;
            case HalfRGB: [[fallthrough]];
            case HalfGray: return 2;
            case ByteRGB: [[fallthrough]];
            case ByteGray: [[fallthrough]];
            case Count: return 1;
        }
        return 1;
    }

    template <typename T>
    concept PixelType = std::is_same_v<T, RGB> || std::is_same_v<T, Half3> || std::is_same_v<T, Byte3> ||
                        std::is_same_v<T, float> || std::is_same_v<T, Half> || std::is_same_v<T, uint8_t>;

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

#if 0
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
#else
    UVDifferentials duv_From_dp_dxy(UVDifferentialsContext const& uvctx)
    {
        UVDifferentials ret{};

        // Local aliases for readability
        auto const& dpdx = uvctx.dpdx;
        auto const& dpdy = uvctx.dpdy;
        auto        dpdu = uvctx.dpdu; // make mutable copy for possible fallback
        auto        dpdv = uvctx.dpdv;

        // Small constants
        constexpr float EPS_DET     = 1e-12f; // threshold for determinant singularity
        constexpr float REG_SCALE   = 1e-6f;  // regularization scale relative to ATA diag
        constexpr float PERTURB_FAC = 1e-6f;  // factor for perturbing dpdy if dpdx==dpdy
        constexpr float MAX_DERIV   = 1e8f;

        // If dpdx and dpdy are identical (or very close), perturb dpdy slightly so we keep a usable differential
        // (this prevents exact collapse of x/y derivatives)
        Vector3f rdpdy = dpdy;
        if (normL2(dpdx - dpdy) < 1e-12f * fmaxf(1.0f, normL2(dpdx)))
        {
            // try to compute a reasonable orthogonal direction to perturb along:
            Vector3f n    = cross(dpdu, dpdv);
            float    nlen = normL2(n);
            if (nlen < 1e-12f)
            {
                // UV basis degenerate; try alternative normals
                n    = cross(dpdu, dpdx);
                nlen = normL2(n);
                if (nlen < 1e-12f)
                {
                    n    = cross(dpdv, dpdx);
                    nlen = normL2(n);
                }
            }
            if (nlen < 1e-12f)
            {
                // Last resort: pick a world-space axis
                n    = {0.0f, 0.0f, 1.0f};
                nlen = 1.0f;
            }
            Vector3f perp = normalize(n);
            // scale perturbation by magnitude of dpdx so small differentials still get something tiny
            float scale = PERTURB_FAC * fmaxf(1.0f, normL2(dpdx));
            rdpdy       = dpdy + perp * scale;
        }

        // ATA components (2x2 symmetric)
        float const ata00 = dot(dpdu, dpdu);
        float const ata01 = dot(dpdu, dpdv);
        float const ata11 = dot(dpdv, dpdv);

        // Right-hand sides: A^T * b for dpdx and rdpdy
        float const atb0x = dot(dpdu, dpdx);
        float const atb1x = dot(dpdv, dpdx);
        float const atb0y = dot(dpdu, rdpdy);
        float const atb1y = dot(dpdv, rdpdy);

        // Compute (regularized) determinant and inverse
        float det = ata00 * ata11 - ata01 * ata01;

        float dudx = 0.0f;
        float dvdx = 0.0f;
        float dudy = 0.0f;
        float dvdy = 0.0f;

        if (!fl::isinf(det) && fabs(det) > EPS_DET)
        {
            // Well-conditioned: direct inverse
            float invDet = 1.0f / det;

            dudx = fl::FMA(ata11, atb0x, -ata01 * atb1x) * invDet;
            dvdx = fl::FMA(ata00, atb1x, -ata01 * atb0x) * invDet;
            dudy = fl::FMA(ata11, atb0y, -ata01 * atb1y) * invDet;
            dvdy = fl::FMA(ata00, atb1y, -ata01 * atb0y) * invDet;
        }
        else
        {
            // Ill-conditioned or singular: use Tikhonov regularization (ridge) to form a stable pseudo-inverse
            // (A^T A + lambda I)^{-1} A^T b
            float lambda = REG_SCALE * fmaxf(1.0f, fmaxf(ata00, ata11));
            float r00    = ata00 + lambda;
            float r11    = ata11 + lambda;
            float r01    = ata01;

            float rdet = r00 * r11 - r01 * r01;

            if (!fl::isinf(rdet) && fabs(rdet) > 0.0f)
            {
                float invRdet = 1.0f / rdet;

                // compute inverse of (A^T A + lambda I) times A^T b
                dudx = fl::FMA(r11, atb0x, -r01 * atb1x) * invRdet;
                dvdx = fl::FMA(r00, atb1x, -r01 * atb0x) * invRdet;
                dudy = fl::FMA(r11, atb0y, -r01 * atb1y) * invRdet;
                dvdy = fl::FMA(r00, atb1y, -r01 * atb0y) * invRdet;
            }
            else
            {
                // As a last resort, fall back to geometric gradient approach:
                // estimate gradient directions for u and v via cross-normal relationships.
                Vector3f normal = cross(dpdu, dpdv);
                float    nlen   = normL2(normal);
                if (nlen < 1e-12f)
                {
                    // still degenerate: try alternatives
                    normal = cross(dpdu, dpdx);
                    nlen   = normL2(normal);
                    if (nlen < 1e-12f)
                    {
                        normal = cross(dpdv, dpdx);
                        nlen   = normL2(normal);
                    }
                }
                if (nlen < 1e-12f)
                {
                    normal = {0.0f, 0.0f, 1.0f}; // final fallback
                }
                else
                {
                    normal = normalize(normal);
                }

                // Build approximate object-space gradients for u and v.
                // These are not exact inverses but are stable and useful for MIP selection.
                Vector3f grad_u = normalize(cross(normal, dpdv));
                Vector3f grad_v = normalize(cross(dpdu, normal));

                // project dpdx/rdpdy onto these approximate gradients
                dudx = dot(grad_u, dpdx);
                dvdx = dot(grad_v, dpdx);
                dudy = dot(grad_u, rdpdy);
                dvdy = dot(grad_v, rdpdy);
            }
        }

        // Clamp and sanitize
        ret.dudx = !fl::isinf(dudx) ? fl::clamp(dudx, -MAX_DERIV, MAX_DERIV) : 0.f;
        ret.dvdx = !fl::isinf(dvdx) ? fl::clamp(dvdx, -MAX_DERIV, MAX_DERIV) : 0.f;
        ret.dudy = !fl::isinf(dudy) ? fl::clamp(dudy, -MAX_DERIV, MAX_DERIV) : 0.f;
        ret.dvdy = !fl::isinf(dvdy) ? fl::clamp(dvdy, -MAX_DERIV, MAX_DERIV) : 0.f;

        return ret;
    }

#endif

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

    template <typename T>
    struct Tag
    {
    };

    ImageTexturev2 makeRGBMipmappedTexture(
        void const*                image,
        int32_t                    xRes,
        int32_t                    yRes,
        TexWrapMode                wrapModeX,
        TexWrapMode                wrapModeY,
        TexFormat                  texFormat,
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
        tex.texFormat = texFormat;

        // Mip levels: stop when both dims hit 1
        int levels = 0;
        int w = xRes, h = yRes;
        while (w > 0 || h > 0) // better for EWA (more memory usage though)
        {
            ++levels;
            w >>= 1;
            h >>= 1;
        }
        tex.mipLevels = levels;

        // Allocate exact space for the mip chain
        size_t totalPixels = mipChainPixelCount(xRes, yRes);
        size_t pixelSize   = bytesPerPixel(tex.texFormat);
        size_t totalBytes  = totalPixels * pixelSize;
        void*  buffer      = memory->allocate(totalBytes, 32);
        if (!buffer)
            return tex;

        tex.data = buffer;

        // --- Copy level 0 (base image) in Morton order ---
        auto const copyPixel = [pixelSize, xRes](void* dst, void const* src, uint32_t u, uint32_t v) {
            uint32_t mortonIdx = encodeMorton2D(u, v);
            std::memcpy(reinterpret_cast<unsigned char*>(dst) + pixelSize * mortonIdx,
                        reinterpret_cast<unsigned char const*>(src) + (u + static_cast<size_t>(v) * xRes) * pixelSize,
                        pixelSize);
        };

        auto const extractPixel = [pixelSize]<PixelType T>(void const* src, uint32_t u, uint32_t v, Tag<T>) -> T {
            uint32_t mortonIdx = encodeMorton2D(u, v);
            auto const* ref = reinterpret_cast<T const*>(reinterpret_cast<unsigned char const*>(src) + pixelSize * mortonIdx);
            return *ref;
        };

        for (int v = 0; v < yRes; ++v)
        {
            for (int u = 0; u < xRes; ++u)
            {
                copyPixel(tex.data, image, u, v);
            }
        }

        // --- Generate lower mips ---
        for (int level = 1; level < levels; ++level)
        {
            int prevW = xRes >> (level - 1);
            int prevH = yRes >> (level - 1);
            int currW = xRes >> level;
            int currH = yRes >> level;

            auto* prevLevel = reinterpret_cast<unsigned char*>(tex.data) +
                              mortonLevelOffset(xRes, yRes, level - 1) * pixelSize;
            auto* currLevel = reinterpret_cast<unsigned char*>(tex.data) + mortonLevelOffset(xRes, yRes, level) * pixelSize;
            assert(isAligned(prevLevel, alignPerPixel(tex.texFormat)));
            assert(isAligned(currLevel, alignPerPixel(tex.texFormat)));

            for (int v = 0; v < currH; ++v)
            {
                for (int u = 0; u < currW; ++u)
                {
                    using enum TexFormat;
                    RGB   c00{}, c10{}, c01{}, c11{};
                    float f00{}, f10{}, f01{}, f11{};
                    switch (tex.texFormat)
                    {
                        case FloatRGB:
                            c00 = extractPixel(prevLevel, u * 2, v * 2, Tag<RGB>{});
                            c10 = extractPixel(prevLevel, u * 2 + 1, v * 2, Tag<RGB>{});
                            c01 = extractPixel(prevLevel, u * 2, v * 2 + 1, Tag<RGB>{});
                            c11 = extractPixel(prevLevel, u * 2 + 1, v * 2 + 1, Tag<RGB>{});
                            break;
                        case HalfRGB:
                            c00 = static_cast<RGB>(extractPixel(prevLevel, u * 2, v * 2, Tag<Half3>{}));
                            c10 = static_cast<RGB>(extractPixel(prevLevel, u * 2 + 1, v * 2, Tag<Half3>{}));
                            c01 = static_cast<RGB>(extractPixel(prevLevel, u * 2, v * 2 + 1, Tag<Half3>{}));
                            c11 = static_cast<RGB>(extractPixel(prevLevel, u * 2 + 1, v * 2 + 1, Tag<Half3>{}));
                            break;
                        case ByteRGB:
                            c00 = static_cast<RGB>(extractPixel(prevLevel, u * 2, v * 2, Tag<Byte3>{}));
                            c10 = static_cast<RGB>(extractPixel(prevLevel, u * 2 + 1, v * 2, Tag<Byte3>{}));
                            c01 = static_cast<RGB>(extractPixel(prevLevel, u * 2, v * 2 + 1, Tag<Byte3>{}));
                            c11 = static_cast<RGB>(extractPixel(prevLevel, u * 2 + 1, v * 2 + 1, Tag<Byte3>{}));
                            break;
                        case FloatGray:
                            f00 = extractPixel(prevLevel, u * 2, v * 2, Tag<float>{});
                            f10 = extractPixel(prevLevel, u * 2 + 1, v * 2, Tag<float>{});
                            f01 = extractPixel(prevLevel, u * 2, v * 2 + 1, Tag<float>{});
                            f11 = extractPixel(prevLevel, u * 2 + 1, v * 2 + 1, Tag<float>{});
                            break;
                        case HalfGray:
                            f00 = static_cast<float>(extractPixel(prevLevel, u * 2, v * 2, Tag<Half>{}));
                            f10 = static_cast<float>(extractPixel(prevLevel, u * 2 + 1, v * 2, Tag<Half>{}));
                            f01 = static_cast<float>(extractPixel(prevLevel, u * 2, v * 2 + 1, Tag<Half>{}));
                            f11 = static_cast<float>(extractPixel(prevLevel, u * 2 + 1, v * 2 + 1, Tag<Half>{}));
                            break;
                        case ByteGray:
                            f00 = static_cast<float>(extractPixel(prevLevel, u * 2, v * 2, Tag<uint8_t>{})) / 255.f;
                            f10 = static_cast<float>(extractPixel(prevLevel, u * 2 + 1, v * 2, Tag<uint8_t>{})) / 255.f;
                            f01 = static_cast<float>(extractPixel(prevLevel, u * 2, v * 2 + 1, Tag<uint8_t>{})) / 255.f;
                            f11 = static_cast<float>(extractPixel(prevLevel, u * 2 + 1, v * 2 + 1, Tag<uint8_t>{})) / 255.f;
                            break;
                    }

                    RGB toEncode = RGB::fromScalar(0.25f);
                    switch (tex.texFormat)
                    {
                        case FloatRGB: [[fallthrough]];
                        case HalfRGB: [[fallthrough]];
                        case ByteRGB: toEncode *= c00 + c10 + c01 + c11; break;
                        case FloatGray: [[fallthrough]];
                        case HalfGray: [[fallthrough]];
                        case ByteGray: toEncode *= f00 + f10 + f01 + f11; break;
                    }

                    void* dst = currLevel + encodeMorton2D(u, v) * pixelSize;
                    switch (tex.texFormat)
                    {
                        case FloatRGB:
                        {
                            assert(pixelSize == sizeof(RGB));
                            std::memcpy(dst, &toEncode, pixelSize);
                            break;
                        }
                        case HalfRGB:
                        {
                            Half const half[3]{Half{toEncode.r}, Half{toEncode.g}, Half{toEncode.b}};
                            assert(pixelSize == 3 * sizeof(Half));
                            std::memcpy(dst, &half, pixelSize);
                            break;
                        }
                        case ByteRGB:
                        {
                            uint8_t const byte[3]{toByte(toEncode.r), toByte(toEncode.g), toByte(toEncode.b)};
                            assert(pixelSize == 3 * sizeof(uint8_t));
                            std::memcpy(dst, &byte, pixelSize);
                            break;
                        }
                        case FloatGray:
                        {
                            assert(toEncode.r == toEncode.g && toEncode.g == toEncode.b);
                            std::memcpy(dst, &toEncode.r, pixelSize);
                            break;
                        }
                        case HalfGray:
                        {
                            assert(toEncode.r == toEncode.g && toEncode.g == toEncode.b);
                            assert(pixelSize == sizeof(Half));
                            Half const half{toEncode.r};
                            std::memcpy(dst, &half, pixelSize);
                            break;
                        }
                        case ByteGray:
                        {
                            assert(toEncode.r == toEncode.g && toEncode.g == toEncode.b);
                            assert(pixelSize == sizeof(uint8_t));
                            uint8_t const byte = toEncode.r * 255.f;
                            std::memcpy(dst, &byte, pixelSize);
                            break;
                        }
                    }
                }
            }
        }

        return tex;
    }

    void freeImageTexture(ImageTexturev2& tex, std::pmr::memory_resource* memory)
    {
        size_t totalPixels = mipChainPixelCount(tex.width, tex.height);
        size_t totalBytes  = totalPixels * bytesPerPixel(tex.texFormat);

        memory->deallocate(tex.data, totalBytes, 32);
        std::memset(&tex, 0, sizeof(ImageTexturev2));
    }

    float ImageTexturev2::evalFloat(TextureEvalContext const& ctx) const
    {
        if (isRGB || isNormal)
            return 0.f;

        return evalRGB(ctx).r;
    }

    static RGB sampleMortonTexel(
        int         s,
        int         t,
        Point2i     levelRes,
        TexWrapMode wrapX,
        TexWrapMode wrapY,
        TexFormat   format,           // NEW: source format
        bool        isNormalMap,      // for special handling if needed
        void const* mortonLevelBuffer // NEW: generic pointer
    )
    {
        assert(isAligned(mortonLevelBuffer, alignPerPixel(format)));
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

        RGB result{};
        switch (format)
        {
            case TexFormat::FloatRGB:
            {
                auto data = static_cast<RGB const*>(mortonLevelBuffer);
                result    = data[mortonIdx];
                break;
            }
            case TexFormat::HalfRGB:
            {
                auto         data = static_cast<Half3 const*>(mortonLevelBuffer);
                Half3 const& h    = data[mortonIdx];
                result.r          = static_cast<float>(h.r);
                result.g          = static_cast<float>(h.g);
                result.b          = static_cast<float>(h.b);
                break;
            }
            case TexFormat::ByteRGB:
            {
                auto         data = static_cast<Byte3 const*>(mortonLevelBuffer);
                Byte3 const& b8   = data[mortonIdx];
                result.r          = b8.r / 255.0f;
                result.g          = b8.g / 255.0f;
                result.b          = b8.b / 255.0f;
                break;
            }
            case TexFormat::FloatGray:
            {
                auto  data = static_cast<float const*>(mortonLevelBuffer);
                float g    = data[mortonIdx];
                result     = {g, g, g};
                break;
            }
            case TexFormat::HalfGray:
            {
                auto  data = static_cast<Half const*>(mortonLevelBuffer);
                float g    = static_cast<float>(data[mortonIdx]);
                result     = {g, g, g};
                break;
            }
            case TexFormat::ByteGray:
            {
                auto  data = static_cast<uint8_t const*>(mortonLevelBuffer);
                float g    = data[mortonIdx] / 255.0f;
                result     = {g, g, g};
                break;
            }
            default:
                // Return magenta to signal error
                result = {1.0f, 0.0f, 1.0f};
                break;
        }

        // Optional: if normal map in [0..1], remap to [-1..1]
        if (isNormalMap)
        {
            result.r = result.r * 2.0f - 1.0f;
            result.g = result.g * 2.0f - 1.0f;
            result.b = result.b * 2.0f - 1.0f;
        }

        return result;
    }

    // clang-format off
    static std::array<float, EWA_LUT_SIZE> const EwaWeightLUT = {
    // MIPMap EWA Lookup Table Values
    0.864664733f, 0.849040031f, 0.83365953f, 0.818519294f, 0.80361563f,
    0.788944781f, 0.774503231f, 0.760287285f, 0.746293485f, 0.732518315f,
    0.718958378f, 0.705610275f, 0.692470789f, 0.679536581f, 0.666804492f,
    0.654271305f, 0.641933978f, 0.629789352f, 0.617834508f, 0.606066525f,
    0.594482362f, 0.583079159f, 0.571854174f, 0.560804546f, 0.549927592f,
    0.539220572f, 0.528680861f, 0.518305838f, 0.50809288f, 0.498039544f,
    0.488143265f, 0.478401601f, 0.468812168f, 0.45937258f, 0.450080454f,
    0.440933526f, 0.431929469f, 0.423066139f, 0.414341331f, 0.405752778f,
    0.397298455f, 0.388976216f, 0.380784035f, 0.372719884f, 0.364781618f,
    0.356967449f, 0.34927541f, 0.341703475f, 0.334249914f, 0.32691282f,
    0.319690347f, 0.312580705f, 0.305582166f, 0.298692942f, 0.291911423f,
    0.285235822f, 0.278664529f, 0.272195935f, 0.265828371f, 0.259560347f,
    0.253390193f, 0.247316495f, 0.241337672f, 0.235452279f, 0.229658857f,
    0.223955944f, 0.21834214f, 0.212816045f, 0.207376286f, 0.202021524f,
    0.196750447f, 0.191561714f, 0.186454013f, 0.181426153f, 0.176476851f,
    0.171604887f, 0.166809067f, 0.162088141f, 0.157441005f, 0.152866468f,
    0.148363426f, 0.143930718f, 0.139567271f, 0.135272011f, 0.131043866f,
    0.126881793f, 0.122784719f, 0.11875169f, 0.114781633f, 0.11087364f,
    0.107026696f, 0.103239879f, 0.0995122194f, 0.0958427936f, 0.0922307223f,
    0.0886750817f, 0.0851749927f, 0.0817295909f, 0.0783380121f, 0.0749994367f,
    0.0717130303f, 0.0684779733f, 0.0652934611f, 0.0621587038f, 0.0590728968f,
    0.0560353249f, 0.0530452281f, 0.0501018465f, 0.0472044498f, 0.0443523228f,
    0.0415447652f, 0.0387810767f, 0.0360605568f, 0.0333825648f, 0.0307464004f,
    0.0281514227f, 0.0255970061f, 0.0230824798f, 0.0206072628f, 0.0181707144f,
    0.0157722086f, 0.013411209f, 0.0110870898f, 0.0087992847f, 0.0065472275f,
    0.00433036685f, 0.0021481365f, 0.f };
    // clang-format on

    // ctx.dUV contains dudx, dudy, dvdx, dvdy
    // texWidth/texHeight are base-level resolution (not mip-resolved)
    LODResult computeTextureLOD_from_dudv(float dudx, float dudy, float dvdx, float dvdy, int texWidth, int texHeight)
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

        // eigenvalues = 0.5 * (trace � sqrt(trace^2 - 4*det))
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

        LODResult res{};
        res.sigma_max = std::max(sigma1, sigma2);
        res.sigma_min = std::min(sigma1, sigma2);
        res.lod_major = std::max(0.f, lod1);
        res.lod_minor = std::max(0.f, lod2);
        return res;
    }

#define DMT_SAFETY_FIRST_EWA

    RGB EWAFormula(EWAParams const& p, Point2f st_in, Vector2f dst0_in, Vector2f dst1_in)
    {
        using enum TexFormat;

        // Scale derivatives to mip level texel space
        Vector2f               dst0     = {dst0_in.x * p.levelRes.x, dst0_in.y * p.levelRes.y};
        Vector2f               dst1     = {dst1_in.x * p.levelRes.x, dst1_in.y * p.levelRes.y};
        static constexpr float MinDeriv = 1e-4f;
#if defined(DMT_OS_WINDOWS) && defined(DMT_DEBUG)
        if (std::abs(dst0.x) < MinDeriv && std::abs(dst0.y) < MinDeriv && std::abs(dst1.x) < MinDeriv &&
            std::abs(dst1.y) < MinDeriv)
            __debugbreak();
#endif

        // Map st to texel space (centered at -0.5)
        int const sx = st_in.x * static_cast<float>(p.levelRes.x) - 0.5f;
        int const sy = st_in.y * static_cast<float>(p.levelRes.y) - 0.5f;

        // Ellipse coefficients (PBRT style)
        float A = dst0.y * dst0.y + dst1.y * dst1.y + 1;
        float B = -2.0f * (dst0.x * dst0.y + dst1.x * dst1.y);
        float C = dst0.x * dst0.x + dst1.x * dst1.x + 1;

        // Normalize
        float const det = (A * C - 0.25f * B * B);
#if defined(DMT_OS_WINDOWS) || defined(DMT_DEBUG)
        if (det <= 1e-12f)
            __debugbreak();
#endif
        float const invF = 1.0f / det;
        A *= invF;
        B *= invF;
        C *= invF;

        // Bounding box
        float const detBB = A * C - 0.25f * B * B;
#if defined(DMT_OS_WINDOWS) || defined(DMT_DEBUG)
        if (detBB <= 1e-12f)
            __debugbreak();
#endif
        float const invDet  = 1.0f / detBB;
        float const uRadius = fl::safeSqrt(C * invDet);
        float const vRadius = fl::safeSqrt(A * invDet);

        int const s0 = static_cast<int>(ceilf(sx - uRadius));
        int const s1 = static_cast<int>(floorf(sx + uRadius));
        int const t0 = static_cast<int>(ceilf(sy - vRadius));
        int const t1 = static_cast<int>(floorf(sy + vRadius));
        // TODO: Remove. Assumes max texture coord is 2
#if defined(DMT_OS_WINDOWS) || defined(DMT_DEBUG)
        if (s0 > p.levelRes.x * 2 || s1 > p.levelRes.x * 2)
            __debugbreak();
        if (t0 > p.levelRes.y * 2 || t1 > p.levelRes.y * 2)
            __debugbreak();
#endif

        RGB   sum{};
        float sumW = 0.0f;

        for (int t = t0; t <= t1; ++t)
        {
            float const tt = static_cast<float>(t) - sy;
            for (int s = s0; s <= s1; ++s)
            {
                float const ss = static_cast<float>(s) - sx;
                float const r2 = A * ss * ss + B * ss * tt + C * tt * tt;
#if defined(DMT_OS_WINDOWS) || defined(DMT_DEBUG)
                if (r2 < 0)
                    __debugbreak();
#endif

                if (r2 < 1.0f)
                {
                    int const idx = fminf(r2 * EWA_LUT_SIZE, EWA_LUT_SIZE - 1);

                    float w = EwaWeightLUT[idx];
                    RGB texel = sampleMortonTexel(s, t, p.levelRes, p.wrapX, p.wrapY, p.texFormat, p.isNormal, p.mortonLevelBuffer);
                    sum += texel * w;
                    sumW += w;
                }
            }
        }

        if (!(sumW > 0.0f))
        {
            return sampleMortonTexel(int(std::floor(sx + 0.5f)),
                                     int(std::floor(sy + 0.5f)),
                                     p.levelRes,
                                     p.wrapX,
                                     p.wrapY,
                                     p.texFormat,
                                     p.isNormal,
                                     p.mortonLevelBuffer);
        }

        return sum / sumW;
    }

    // TODO with all types of data (normals are an exception)
    static RGB EWA(ImageTexturev2 const* tex, int32_t ilod, Point2f st_in, Vector2f dst0_in, Vector2f dst1_in)
    {
        using enum TexFormat;

        size_t const pixelSize    = bytesPerPixel(tex->texFormat);
        auto const   extractPixel = [pixelSize]<PixelType T>(void const* src, uint32_t u, uint32_t v, Tag<T>) -> T {
            uint32_t mortonIdx = encodeMorton2D(u, v);
            auto const* ref = reinterpret_cast<T const*>(reinterpret_cast<unsigned char const*>(src) + pixelSize * mortonIdx);
            return *ref;
        };
        // If beyond last level: return a single texel
        if (ilod >= tex->mipLevels)
        {
            size_t      off = mortonLevelOffset(tex->width, tex->height, tex->mipLevels - 1);
            void const* src = reinterpret_cast<unsigned char const*>(tex->data) + off * pixelSize;
            switch (tex->texFormat)
            {
                case FloatRGB: return extractPixel(src, 0, 0, Tag<RGB>{});
                case HalfRGB: return static_cast<RGB>(extractPixel(src, 0, 0, Tag<Half3>{}));
                case ByteRGB: return static_cast<RGB>(extractPixel(src, 0, 0, Tag<Byte3>{}));
                case FloatGray:
                {
                    float f = extractPixel(src, 0, 0, Tag<float>{});
                    return RGB::fromScalar(f);
                }
                case HalfGray:
                {
                    float f = static_cast<float>(extractPixel(src, 0, 0, Tag<Half>{}));
                    return RGB::fromScalar(f);
                }
                case ByteGray:
                {
                    float f = fl::clamp01(extractPixel(src, 0, 0, Tag<uint8_t>{}) / 255.f);
                    return RGB::fromScalar(f);
                }
            }
        }

        // Pointer to this mip level
        auto const* mortonLevelBuffer = reinterpret_cast<unsigned char const*>(tex->data) +
                                        mortonLevelOffset(tex->width, tex->height, ilod) * pixelSize;

        // Mip level resolution
        Point2i levelRes = {std::max(1, tex->width >> ilod), std::max(1, tex->height >> ilod)};

        EWAParams params{mortonLevelBuffer,
                         levelRes,
                         tex->wrapModeX,
                         tex->wrapModeY,
                         tex->texFormat,
                         static_cast<bool>(tex->isNormal)};

        return EWAFormula(params, st_in, dst0_in, dst1_in);
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

        // for EWA
        float lambda = lodRes.lod_minor; // tends to preserve more detail; EWA addresses aliasing anisotropically
        int   ilod   = std::clamp<int>(int(std::floor(lambda)), 0, mipLevels - 1);
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