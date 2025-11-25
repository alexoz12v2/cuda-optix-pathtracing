#include "core-parser.h"

#include "ImfRgba.h"
#include "core-light.h"
#include "core-math.h"
#include "core-render.h"
#include "core-texture.h"
#include "core-trianglemesh.h"
#include "cudautils/cudautils-transform.cuh"
#include "cudautils/cudautils-vecmath.cuh"
#include "platform-memory.h"
#include "platform/platform-context.h"

// json parsing
#include <algorithm>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>

// image parsing
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <ImfRgbaFile.h>
#include <ImfArray.h>

// C++ standard
#include <fstream>
#include <unordered_map>
#include <unordered_set>

namespace dmt {

    struct TexInfo
    {
        std::string texType;
        std::string texPath;
        uint32_t    xRes;
        uint32_t    yRes;
    };

    struct ParserState
    {
        std::unordered_map<std::string, Transform> transforms;

        std::unordered_map<std::string, Light>    lights;
        std::unordered_map<std::string, TexInfo>  texturesPath;
        std::unordered_map<std::string, uint32_t> materials;
        std::unordered_map<std::string, uint32_t> objects;
        std::vector<Transform>                    stackTrasforms;
    };
} // namespace dmt

namespace dmt::dict {
    using namespace std::string_view_literals;
    // Objects
    SText camera = "camera"sv;
    SText film   = "film"sv;
    SText world  = "world"sv;
    // Array
    SText textures   = "textures"sv;
    SText materials  = "materials"sv;
    SText objects    = "objects"sv;
    SText transforms = "transforms"sv;
    // String
    SText envlight = "envlight"sv;
} // namespace dmt::dict

namespace dmt::parse_helpers {
    using json = nlohmann::json;

    enum class ExtractVec3fResult
    {
        eOk = 0,
        eNotArray,
        eIncorrectSize,
        eInvalidType0,
        eInvalidType1,
        eInvalidType2,

        eCount
    };

    enum class TempTexObjResult
    {
        eOk = 0,
        eNotExists,
        eFormatNotSupported,
        eFailLoad,
        eNumChannelsIncorrect,

        eCount
    };

    static std::string_view tempTexObjResultToString(TempTexObjResult result)
    {
        switch (result)
        {
            case TempTexObjResult::eOk: return "Operation was successful.";
            case TempTexObjResult::eNotExists: return "The texture object does not exist.";
            case TempTexObjResult::eFormatNotSupported: return "The texture format is not supported.";
            case TempTexObjResult::eFailLoad: return "Failed to load the texture.";
            case TempTexObjResult::eNumChannelsIncorrect: return "The number of texture channels is incorrect.";
            default: return "Unknown error.";
        }
    }


    static bool hasOnlyAllowedkeys(json const& object, std::unordered_set<std::string> const& allowedKeys)
    {
        if (!object.is_object())
            return false;
        for (auto const& [key, val] : object.items())
        {
            if (!allowedKeys.contains(key))
                return false;
        }
        return true;
    }

    static bool hasAllAllowedkeys(json const& object, std::unordered_set<std::string> const& allowedKeys)
    {
        if (!hasOnlyAllowedkeys(object, allowedKeys))
            return false;
        for (auto const& key : allowedKeys)
        {
            if (!object.contains(key))
                return false;
        }
        return true;
    }

    inline RGB* allocAlignedRGB(size_t count)
    {
        void* ptr = nullptr;
        // allocate with 16-byte alignment
#if defined(DMT_OS_LINUX)
        if (posix_memalign(&ptr, std::max<size_t>(alignof(RGB), alignof(EnvLight)), sizeof(RGB) * count) != 0)
#elif defined(DMT_OS_WINDOWS)
        if (ptr = _aligned_malloc(sizeof(RGB) * count, std::max<size_t>(alignof(RGB), alignof(EnvLight))))
#else
#endif
            return nullptr;
        return reinterpret_cast<RGB*>(ptr);
    }

    static RGB* loadImageAsRGB(os::Path const& path, int& outWidth, int& outHeight)
    {
        if (!path.isValid() || !path.isFile())
            return nullptr;

        std::pmr::string pathStr = path.toUnderlying();
        std::string      pathExt;
        if (pathStr.find_last_of('.') == std::pmr::string::npos)
            return nullptr;
        std::copy(pathStr.begin() + static_cast<ptrdiff_t>(pathStr.find_last_of('.')),
                  pathStr.end(),
                  std::back_inserter(pathExt));

        std::ranges::transform(pathExt, pathExt.begin(), [](char c) { return std::tolower(c); });

        int  xRes     = 0;
        int  yRes     = 0;
        int  channels = 0;
        RGB* result   = nullptr;

        if (pathExt.ends_with("png") || pathExt.ends_with("jpg") || pathExt.ends_with("jpeg"))
        {
            // Decode 8-bit LDR with stb_image
            unsigned char* data = stbi_load(pathStr.c_str(), &xRes, &yRes, &channels, 0);
            if (!data)
                return nullptr;

            result = allocAlignedRGB(sizeof(RGB) * xRes * yRes);
            if (!result)
            {
                stbi_image_free(data);
                return nullptr;
            }

            for (int i = 0; i < xRes * yRes; ++i)
            {
                if (channels >= 3)
                {
                    result[i].r = data[i * channels + 0] / 255.0f;
                    result[i].g = data[i * channels + 1] / 255.0f;
                    result[i].b = data[i * channels + 2] / 255.0f;
                }
                else
                {
                    float g     = data[i * channels + 0] / 255.0f;
                    result[i].r = result[i].g = result[i].b = g;
                }
            }
            stbi_image_free(data);
        }
        else if (pathExt.ends_with("exr"))
        {
            try
            {
                Imf::RgbaInputFile file(pathStr.c_str());
                Imath::Box2i       dw = file.dataWindow();
                xRes                  = dw.max.x - dw.min.x + 1;
                yRes                  = dw.max.y - dw.min.y + 1;

                Imf::Array2D<Imf::Rgba> pixels;
                pixels.resizeErase(yRes, xRes);
                file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * xRes, 1, xRes);
                file.readPixels(dw.min.y, dw.max.y);

                result = allocAlignedRGB(sizeof(RGB) * xRes * yRes);
                if (!result)
                    return nullptr;

                for (int y = 0; y < yRes; ++y)
                {
                    for (int x = 0; x < xRes; ++x)
                    {
                        Imf::Rgba const& px  = pixels[y][x];
                        int              idx = y * xRes + x;
                        result[idx].r        = (float)px.r;
                        result[idx].g        = (float)px.g;
                        result[idx].b        = (float)px.b;
                    }
                }
            } catch (...)
            {
                return nullptr;
            }
        }
        else if (pathExt.ends_with("hdr"))
        {
            float* data = stbi_loadf(pathStr.c_str(), &xRes, &yRes, &channels, 0);
            if (!data)
                return nullptr;

            result = allocAlignedRGB(sizeof(RGB) * xRes * yRes);
            if (!result)
            {
                stbi_image_free(data);
                return nullptr;
            }

            for (int i = 0; i < xRes * yRes; ++i)
            {
                if (channels >= 3)
                {
                    result[i].r = data[i * channels + 0];
                    result[i].g = data[i * channels + 1];
                    result[i].b = data[i * channels + 2];
                }
                else
                {
                    float g     = data[i * channels + 0];
                    result[i].r = result[i].g = result[i].b = g;
                }
            }
            stbi_image_free(data);
        }
        else
        {
            return nullptr; // unsupported
        }

        outWidth  = xRes;
        outHeight = yRes;
        return result;
    }

    static std::string_view writeOpenEXRChannel(Imf::RgbaChannels channels)
    {
        using enum Imf::RgbaChannels;
        switch (channels)
        {
            case WRITE_R: return "(OpenEXR) Red";
            case WRITE_G: return "(OpenEXR) Green";
            case WRITE_B: return "(OpenEXR) Blue";
            case WRITE_A: return "(OpenEXR) Alpha";
            case WRITE_Y: return "(OpenEXR) Luminance, for black-and-white images, or in combination with chroma";
            case WRITE_C:
                return "(OpenEXR) Chroma (two subsampled channels, RY and BY, supported only for scanline-based files)";
            case WRITE_RGB: return "(OpenEXR) Red, green, blue";
            case WRITE_RGBA: return "(OpenEXR) Red, green, blue, alpha";
            case WRITE_YC: return "(OpenEXR) Luminance, chroma";
            case WRITE_YA: return "(OpenEXR) Luminance, alpha";
            case WRITE_YCA: return "(OpenEXR) Luminance, chroma, alpha";
        }
    }

    static TempTexObjResult tempTexObj(os::Path const& path, bool isRGB, ImageTexturev2* out)
    {
        using enum TempTexObjResult;
        if (!path.isValid() || !path.isFile())
            return eNotExists;

        std::pmr::string           pathStr = path.toUnderlying();
        std::pmr::memory_resource* mem     = std::pmr::get_default_resource();

        int         xRes = 0, yRes = 0, channels = isRGB ? 3 : 1;
        void const* data = nullptr;
        TexFormat   format;

        std::string pathExt;
        if (pathStr.find_last_of('.') == std::pmr::string::npos)
            return eNotExists;

        std::copy(pathStr.begin() + static_cast<ptrdiff_t>(pathStr.find_last_of('.')),
                  pathStr.end(),
                  std::back_inserter(pathExt));

        std::ranges::transform(pathExt, pathExt.begin(), [](char c) { return std::tolower(c); });

        if (pathExt.ends_with("png") || pathExt.ends_with("jpg") || pathExt.ends_with("jpeg"))
        {
            // PNG file, no header parsing, 8 bppc only
            format = isRGB ? TexFormat::ByteRGB : TexFormat::ByteGray;
            data   = stbi_load(pathStr.c_str(), &xRes, &yRes, &channels, 0);
            if (!data)
                return eFailLoad;
            if (channels != (isRGB ? 3 : 1))
            {
                Context ctx;
                ctx.error("  texture channels: {}", std::make_tuple(channels));
                ctx.error("  texture expected: {}", std::make_tuple(isRGB ? 3 : 1));
                return eNumChannelsIncorrect;
            }
        }
        else if (pathExt.ends_with("exr"))
        {
            // OpenEXR format, half16 per channel
            try
            {
                Imf::RgbaInputFile file(pathStr.c_str());
                Imath::Box2i       dw = file.dataWindow();

                if (file.channels() != (isRGB ? Imf::RgbaChannels::WRITE_RGB : Imf::RgbaChannels::WRITE_Y))
                {
                    Context ctx;
                    ctx.error("  texture channels: {}", std::make_tuple(writeOpenEXRChannel(file.channels())));
                    ctx.error("  texture expected: {}",
                              std::make_tuple(writeOpenEXRChannel(
                                  isRGB ? Imf::RgbaChannels::WRITE_RGB : Imf::RgbaChannels::WRITE_Y)));
                    return eNumChannelsIncorrect;
                }

                if (file.pixelAspectRatio() != 1.f)
                {
                    Context ctx;
                    ctx.warn("  texture EXR pixel aspect ratio different than 1", {});
                }

                if (file.lineOrder() != Imf::LineOrder::INCREASING_Y)
                {
                    Context ctx;
                    ctx.warn("  texture EXR line order differnet than increasing Y", {});
                }

                if (file.parts() > 1)
                {
                    Context ctx;
                    ctx.error("  OpenEXR: No support for multi-view/multi-part EXR files", {});
                    return eFormatNotSupported;
                }

                if (file.header().hasType() && file.header().type() != "scanlineimage")
                {
                    Context ctx;
                    ctx.error("  OpenEXR: Expected type '{}', but got '{}'",
                              std::make_tuple("scanlineimage", file.header().type()));
                    return eFormatNotSupported;
                }

                xRes = dw.max.x - dw.min.x + 1;
                yRes = dw.max.y - dw.min.y + 1;

                // Allocate OpenEXR storage
                Imf::Array2D<Imf::Rgba> pixels;
                pixels.resizeErase(yRes, xRes);

                file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * xRes, 1, xRes);
                file.readPixels(dw.min.y, dw.max.y);

                if (isRGB)
                {
                    format    = TexFormat::HalfRGB;
                    half* buf = (half*)malloc(sizeof(half) * xRes * yRes * 3);
                    if (!buf)
                        throw std::runtime_error("what");
                    for (int y = 0; y < yRes; ++y)
                    {
                        for (int x = 0; x < xRes; ++x)
                        {
                            Imf::Rgba const& px         = pixels[y][x];
                            buf[(y * xRes + x) * 3 + 0] = px.r;
                            buf[(y * xRes + x) * 3 + 1] = px.g;
                            buf[(y * xRes + x) * 3 + 2] = px.b;
                        }
                    }
                    data = buf;
                }
                else
                {
                    format    = TexFormat::HalfGray;
                    half* buf = (half*)malloc(sizeof(half) * xRes * yRes);
                    if (!buf)
                        throw std::runtime_error("what");
                    for (int y = 0; y < yRes; ++y)
                    {
                        for (int x = 0; x < xRes; ++x)
                        {
                            Imf::Rgba const& px = pixels[y][x];
                            assert((px.b == 0 && px.g == 0 && (px.a == 1 || px.a == 0)) || (px.r == px.g && px.g == px.b));
                            buf[y * xRes + x] = px.r;
                        }
                    }
                    data = buf;
                }
            } catch (std::exception const& e)
            {
                return eFailLoad;
            }
        }
        else if (pathExt.ends_with("hdr"))
        {
            // Radiance RGBE pixel format through stb_image, which expands it into float32
            format = isRGB ? TexFormat::FloatRGB : TexFormat::FloatGray;
            data   = stbi_loadf(pathStr.c_str(), &xRes, &yRes, &channels, 0);
            if (!data)
                return eFailLoad;
            if (channels != (isRGB ? 3 : 1))
            {
                Context ctx;
                ctx.error("  texture channels: {}", std::make_tuple(channels));
                return eNumChannelsIncorrect;
            }
        }
        else
        {
            return eFormatNotSupported;
        }

        *out = makeRGBMipmappedTexture(data, xRes, yRes, TexWrapMode::eClamp, TexWrapMode::eClamp, format, mem);
        return eOk;
    }

    static void freeTempTexObj(os::Path const& path, ImageTexturev2& in)
    {
        std::pmr::string pathStr = path.toUnderlying();
        std::string      pathExt;
        if (pathStr.find_last_of('.') == std::pmr::string::npos)
            return;

        std::copy(pathStr.begin() + static_cast<ptrdiff_t>(pathStr.find_last_of('.')),
                  pathStr.end(),
                  std::back_inserter(pathExt));

        std::ranges::transform(pathExt, pathExt.begin(), [](char c) { return std::tolower(c); });
        if (pathExt.ends_with("png") || pathExt.ends_with("hdr") || pathExt.ends_with("jpg") || pathExt.ends_with("jpeg"))
        {
            stbi_image_free(in.data);
        }
        else if (pathExt.ends_with("exr"))
        {
            free(in.data);
        }
    }

    static ExtractVec3fResult extractVec3f(json const& vec, Vector3f* out)
    {
        using enum ExtractVec3fResult;
        assert(out);
        if (!vec.is_array())
            return eNotArray;
        if (vec.size() != 3)
            return eIncorrectSize;
        for (uint32_t i = 0; i < 3; ++i)
        {
            try
            {
                if (!vec[i].is_number())
                    return static_cast<ExtractVec3fResult>(dmt::toUnderlying(eInvalidType0) + i);
                float elem = static_cast<float>(vec[i]);
                (*out)[i]  = elem;
            } catch (...)
            {
                return static_cast<ExtractVec3fResult>(dmt::toUnderlying(eInvalidType0) + i);
            }
        }

        return eOk;
    }

    static bool parseFilm(Parser& parser, json const& film, Parameters* param)
    {
        Context ctx;

        if (film.size() > 3)
        {
            //insert error message
            return false;
        }

        try
        {
            //keys check
            if (!film.contains("resolutionX"))
            {
                //insert error message
                return false;
            }

            if (!film.contains("resolutionY"))
            {
                //insert error message
                return false;
            }

            //Values check
            if (!film["resolutionX"].is_number())
            {
                //insert error message
                return false;
            }

            if (!film["resolutionY"].is_number())
            {
                //insert error message
                return false;
            }

            if (film.contains("samples"))
            {
                if (!film["samples"].is_number_integer() || static_cast<int32_t>(film["samples"]) <= 0)
                {
                    ctx.error("'samples' Should be positive integer", {});
                    return false;
                }
                param->samplesPerPixel = film["samples"];
            }

            param->filmResolution = Point2i{film["resolutionX"], film["resolutionY"]};

        } catch (...)
        {
            //insert error message
            return false;
        }


        return true;
    }

    static bool parseTexture(Parser& parser, ParserState& state, json const& texture, Renderer* rend)
    {
        Context ctx;

        thread_local std::unordered_set<std::string> const allowed{"name", "type", "path"};

        if (!hasOnlyAllowedkeys(texture, allowed))
        {
            ctx.error("Texture object should contain only attributes 'path', 'name', 'type'", {});
            return false;
        }

        if (!texture.contains("name") || !texture["name"].is_string())
        {
            ctx.error("Texture object should contain attribute 'name' and it should be a string", {});
            return false;
        }

        std::string texName = texture["name"];
        ctx.log("Texture: Loading texture '{}'", std::make_tuple(texName));

        if (!texture.contains("type") || !texture["type"].is_string())
        {
            ctx.error("Texture object should contain attribute 'type' and it should be a string", {});
            return false;
        }

        if (!texture.contains("path") || !texture["path"].is_string())
        {
            ctx.error("Texture object should contain attribute 'path' and it should be a string", {});
            return false;
        }

        if (state.texturesPath.contains(static_cast<std::string>(texture["name"])))
        {
            ctx.error("texture {} already exists", std::make_tuple(static_cast<std::string>(texture["name"])));
            return false;
        }

        state.texturesPath[texName].texType = texture["type"];

        if (state.texturesPath[texName].texType != "diffuse" && state.texturesPath[texName].texType != "normal" &&
            state.texturesPath[texName].texType != "metallic" && state.texturesPath[texName].texType != "roughness")
        {
            ctx.error("texture: unrecognized type '{}'", std::make_tuple(state.texturesPath[texName].texType));
            return false;
        }

        os::Path texPath = parser.fileDirectory() / static_cast<std::string>(texture["path"]).c_str();

        if (!texPath.isValid() || !texPath.isFile())
        {
            //insert error message
            return false;
        }

        state.texturesPath[texName].texPath = std::string(texPath.toUnderlying());
        ImageTexturev2   imgTex{};
        TempTexObjResult res = tempTexObj(texPath,
                                          state.texturesPath[texName].texType == "diffuse" ||
                                              state.texturesPath[texName].texType == "normal",
                                          &imgTex);
        if (res != TempTexObjResult::eOk)
        {
            ctx.error("Error loading texture '{}'", std::make_tuple(texName));
            ctx.error("{}", std::make_tuple(tempTexObjResultToString(res)));
            if (res == TempTexObjResult::eNumChannelsIncorrect)
            {
                ctx.error("metallic, roughness expect 1 channel", {});
                ctx.error("diffuse, normal, expect 3 channels", {});
            }
            return false;
        }
        state.texturesPath[texName].xRes = static_cast<uint32_t>(imgTex.width);
        state.texturesPath[texName].yRes = static_cast<uint32_t>(imgTex.height);

        rend->texCache.MipcFiles.createCacheFile(baseKeyFromPath(texPath), imgTex);
        freeTempTexObj(texPath, imgTex);
        return true;
    }

    static bool parseMaterial(Parser& parser, ParserState& state, json const& material, Renderer* rend)
    {
        Context         ctx;
        SurfaceMaterial mat{};
        if (!material.is_object())
        {
            ctx.error("The 'materials' array should contain only objects", {});
            return false;
        }

        if (!material.contains("name") || !material["name"].is_string())
        {
            ctx.error(
                "The material should have a unique 'name' *string*"
                "field in the materials namespace",
                {});
            return false;
        }

        std::string const name = material["name"];

        if (state.materials.contains(name))
        {
            ctx.error("Duplicate material name '{}'", std::make_tuple(name));
            return false;
        }

        ctx.log("Material: Loading material '{}'", std::make_tuple(name));

        thread_local std::unordered_set<std::string> const
            allowedKeys{"name",
                        "diffuse",
                        "metallic",
                        "normal",
                        "roughness",
                        "ior",
                        "eta",
                        "etak",
                        "ggx-anisotropy",
                        "ggx-dielectric",
                        "oren-nayar-dielectric"};
        if (!hasOnlyAllowedkeys(material, allowedKeys))
        {
            std::string extraneous;
            for (auto const& [key, val] : material.items())
            {
                if (!allowedKeys.contains(key))
                {
                    extraneous = key;
                    break;
                }
            }
            ctx.error("material '{}' has extraneous key '{}'", std::make_tuple(name, extraneous));
            return false;
        }

        try
        {
            // - mandatory fields parsing
            if (!material.contains("diffuse"))
            {
                ctx.error("material should specify a 'diffuse' either as RGB or texture name", {});
                return false;
            }

            if (!material.contains("metallic"))
            {
                ctx.error("material should specify a 'metallic' either as float or texture name", {});
                return false;
            }

            if (!material.contains("roughness"))
            {
                ctx.error("material should specify a 'roughness' either as float or texture name", {});
                return false;
            }

            // -- diffuse
            if (material["diffuse"].is_array())
            {
                Vector3f tmp{};
                if (extractVec3f(material["diffuse"], &tmp) != ExtractVec3fResult::eOk)
                {
                    ctx.error("'diffuse' constant expected to be RGB value", {});
                    return false;
                }
                mat.diffusevalue = byte3FromRGB(RGB::fromVec(min(max(tmp, {0, 0, 0}), {1, 1, 1})));
            }
            else if (material["diffuse"].is_string())
            {
                std::string diffuseName = material["diffuse"];
                if (!state.texturesPath.contains(diffuseName))
                {
                    ctx.error("'diffuse' texture name should be an existing named texture", {});
                    return false;
                }

                TexInfo const& texInfo = state.texturesPath.at(diffuseName);
                if (texInfo.texType != "diffuse")
                {
                    ctx.error("'diffuse' material texture should point to a 'diffuse' texture", {});
                    return false;
                }
                mat.diffusekey    = baseKeyFromPath(texInfo.texPath);
                mat.diffuseWidth  = texInfo.xRes;
                mat.diffuseHeight = texInfo.yRes;
                mat.texMatMap |= SurfaceMaterial::DiffuseMask;
            }
            else
            {
                ctx.error("material 'diffuse' should be either texture name or RGB", {});
                return false;
            }

            // -- normal
            if (!material.contains("normal"))
            {
                mat.useShadingNormals = false;
            }
            else
            {
                mat.useShadingNormals = true;
                if (material["normal"].is_string())
                {
                    std::string normalName = material["normal"];
                    if (!state.texturesPath.contains(normalName))
                    {
                        ctx.error("'normal' texture name should be an existing named texture", {});
                        return false;
                    }

                    TexInfo const& texInfo = state.texturesPath.at(normalName);
                    if (texInfo.texType != "normal")
                    {
                        ctx.error("'normal' material texture should point to a 'normal' texture", {});
                        return false;
                    }
                    mat.normalkey    = baseKeyFromPath(texInfo.texPath);
                    mat.normalWidth  = texInfo.xRes;
                    mat.normalHeight = texInfo.yRes;
                    mat.texMatMap |= SurfaceMaterial::NormalMask;
                }
                else
                {
                    ctx.error("material 'normal' should be an RGB texture name", {});
                    return false;
                }
            }

            // -- roughness
            if (material["roughness"].is_number())
            {
                mat.roughnessvalue = fl::clamp01(material["roughness"]);
            }
            else if (material["roughness"].is_string())
            {
                std::string roughnessName = material["roughness"];
                if (!state.texturesPath.contains(roughnessName))
                {
                    ctx.error("'roughness' texture name should be an existing named texture", {});
                    return false;
                }
                TexInfo const& texInfo = state.texturesPath.at(roughnessName);
                if (texInfo.texType != "roughness")
                {
                    ctx.error("'roughness' material texture should point to a 'roughness' texture", {});
                    return false;
                }
                mat.roughnesskey    = baseKeyFromPath(texInfo.texPath);
                mat.roughnessWidth  = texInfo.xRes;
                mat.roughnessHeight = texInfo.yRes;
                mat.texMatMap |= SurfaceMaterial::RoughnessMask;
            }
            else
            {
                ctx.error("material 'roughness' should be either texture name or float", {});
                return false;
            }

            // -- metallic
            if (material["metallic"].is_number())
            {
                mat.metallicvalue = fl::clamp01(material["metallic"]);
            }
            else if (material["metallic"].is_string())
            {
                std::string metallicName = material["metallic"];
                if (!state.texturesPath.contains(metallicName))
                {
                    ctx.error("'metallic' texture name should be an existing named texture", {});
                    return false;
                }
                TexInfo const& texInfo = state.texturesPath.at(metallicName);
                if (texInfo.texType != "metallic")
                {
                    ctx.error("'metallic' material texture should point to a 'metallic' texture", {});
                    return false;
                }
                mat.metallickey    = baseKeyFromPath(texInfo.texPath);
                mat.metallicWidth  = texInfo.xRes;
                mat.metallicHeight = texInfo.yRes;
                mat.texMatMap |= SurfaceMaterial::MetallicMask;
            }
            else
            {
                ctx.error("material 'metallic' should be either texture name or float", {});
                return false;
            }

            // - optional fields for diffuse
            // -- ior
            // dielectric ior list: <https://balyberdin.com/tools/ior-list/>
            mat.ior = 1.4f; // default
            if (material.contains("ior"))
            {
                if (!material["ior"].is_number())
                {
                    ctx.error("'ior' from material, if present, should be a number between 1 and 10", {});
                    return false;
                }
                mat.ior = fmaxf(material["ior"], 1.f);
            }

            // -- eta
            static constexpr uint8_t etaPresent  = 0x1;
            static constexpr uint8_t etakPresent = 0x2;
            static constexpr uint8_t etaFull     = etaPresent | etakPresent;
            uint8_t                  etaPresence = 0;
            // conductor ior list: <https://chris.hindefjord.se/resources/rgb-ior-metals/>
            mat.eta  = {.r = 0.18299f, .g = 0.42108f, .b = 1.37340f};
            mat.etak = {.r = 3.42420f, .g = 2.34590f, .b = 1.77040f};
            if (material.contains("eta"))
            {
                etaPresence |= etaPresent;
                Vector3f tmp{};
                if (extractVec3f(material["eta"], &tmp) != ExtractVec3fResult::eOk)
                {
                    ctx.error("material error extracting RGB 'eta' field", {});
                    return false;
                }
                mat.eta = RGB::fromVec(max(tmp, {0, 0, 0}));
            }

            // -- etak
            if (material.contains("etak"))
            {
                etaPresence |= etakPresent;
                Vector3f tmp{};
                if (extractVec3f(material["etak"], &tmp) != ExtractVec3fResult::eOk)
                {
                    ctx.error("material error extracting RGB 'etak' field", {});
                    return false;
                }
                mat.etak = RGB::fromVec(max(tmp, {0, 0, 0}));
            }

            if (etaPresence != 0 && etaPresence != etaFull)
            {
                ctx.error("material: either specify both 'eta' and 'etak' or none of them (Gold)", {});
                return false;
            }

            // -- ggx-anisotropy
            if (material.contains("ggx-anisotropy"))
            {
                if (!material["ggx-anisotropy"].is_number() || static_cast<float>(material["ggx-anisotropy"]) < 0.f ||
                    static_cast<float>(material["ggx-anisotropy"]) > 1.f)
                {
                    ctx.error("material field 'ggx-anisotropy' should be a number from 0.0 to 1.0", {});
                    return false;
                }

                float const anisInterp = fl::clamp01(material["ggx-anisotropy"]);
                mat.anisotropy         = fl::lerp(anisInterp, 1.f, 8.f);
            }

            // -- ggx-dielectric
            bool dielectricFound = false;
            if (material.contains("ggx-dielectric"))
            {
                dielectricFound = true;

                json const& ggxDielectric = material["ggx-dielectric"];
                if (!ggxDielectric.is_object() && !ggxDielectric.is_null())
                {
                    ctx.error("material: 'ggx-dielectric' should be an object (or null to use defaults)", {});
                    return false;
                }
                mat.reflectanceTint   = {1, 1, 1};
                mat.transmittanceTint = {1, 1, 1};
                if (ggxDielectric.is_object())
                {
                    static std::unordered_set<std::string> const allowedKeys{"reflectance-tint", "transmittance-tint"};
                    if (!hasOnlyAllowedkeys(ggxDielectric, allowedKeys))
                    {
                        ctx.error("material: Unrecognized key inside 'ggx-dielectric'", {});
                        return false;
                    }
                    if (ggxDielectric.contains("reflectance-tint"))
                    {
                        Vector3f tmp{};
                        if (extractVec3f(ggxDielectric["reflectance-tint"], &tmp) != ExtractVec3fResult::eOk)
                        {
                            ctx.error("material: error extracting 'reflectance-tint'", {});
                            return false;
                        }
                        mat.reflectanceTint = RGB::fromVec(max(min(tmp, {1, 1, 1}), {0, 0, 0}));
                    }
                    if (ggxDielectric.contains("transmittance-tint"))
                    {
                        Vector3f tmp{};
                        if (extractVec3f(ggxDielectric["transmittance-tint"], &tmp) != ExtractVec3fResult::eOk)
                        {
                            ctx.error("material: error extracting 'transmittance-tint'", {});
                            return false;
                        }
                        mat.transmittanceTint = RGB::fromVec(min(max(tmp, {0, 0, 0}), {1, 1, 1}));
                    }
                }
            }

            // -- oren-nayar-dielectric
            if (material.contains("oren-nayar-dielectric"))
            {
                if (dielectricFound)
                {
                    ctx.error("material: Either specify 'ggx-dielectric' or 'oren-nayar-dielectric', not both", {});
                    return false;
                }
                dielectricFound                 = true;
                json const& orenNayarDielectric = material["oren-nayar-dielectric"];
                if (!orenNayarDielectric.is_object() && !orenNayarDielectric.is_null())
                {
                    ctx.error("material: 'oren-nayar-dielectric' should be an object (or null to use defaults)", {});
                    return false;
                }
                mat.multiscatterMultiplier = 1.f;
                if (orenNayarDielectric.contains("multiscatter-multiplier"))
                {
                    if (json const& j = orenNayarDielectric["multiscatter-multiplier"];
                        !j.is_number() || static_cast<float>(j) <= 0.f)
                    {
                        ctx.error("material: 'multiscatter-multiplier' should be a positive number", {});
                        return false;
                    }
                    mat.multiscatterMultiplier = orenNayarDielectric["multiscatter-multiplier"];
                }
            }
        } catch (...)
        {
            ctx.error("Unknown Error during material parsing", {});
            return false;
        }

        rend->scene.materials.push_back(mat);
        auto const& [it, wasInserted] = state.materials.try_emplace(name,
                                                                    static_cast<uint32_t>(rend->scene.materials.size() - 1));

        return wasInserted;
    }

    static bool parseCamera(Parser& parser, json const& camera, Parameters* param)
    {

        Context ctx;
        if (camera.size() > 4)
        {
            return false;
        }
        try
        {
            //check on key
            if (!camera.contains("focalLength"))
            {
                //insert error message
                return false;
            }

            if (!camera.contains("sensorSize"))
            {
                //insert error message
                return false;
            }

            if (!camera.contains("direction"))
            {
                //insert error message
                return false;
            }

            //check on type
            if (!camera["sensorSize"].is_number() || static_cast<float>(camera["sensorSize"]) <= 0.f)
            {
                ctx.error("'sensorSize' should be a positive number", {});
                return false;
            }

            if (!camera["focalLength"].is_number() || static_cast<float>(camera["focalLength"]) <= 0.f)
            {
                ctx.error("'focalLength' should be a positive number", {});
                return false;
            }

            json const&        jDir = camera["direction"];
            Vector3f           dir{};
            ExtractVec3fResult res = extractVec3f(jDir, &dir);

            if (res != ExtractVec3fResult::eOk)
            {
                //insert error message
                return false;
            }


            if (!camera["focalLength"].is_number())
            {
                //insert error message
                return false;
            }

            param->focalLength     = static_cast<float>(camera["focalLength"]) / 1000.f;
            param->sensorSize      = static_cast<float>(camera["sensorSize"]) / 1000.f;
            param->cameraDirection = dir;

            // optioanl position
            if (camera.contains("position"))
            {
                Vector3f tmp{};
                if (extractVec3f(camera["position"], &tmp) != ExtractVec3fResult::eOk)
                {
                    ctx.error("Error extracting camera position vector", {});
                    return false;
                }
                param->cameraPosition = tmp;
            }

            if (camera.contains("max-depth"))
            {
                if (!camera["max-depth"].is_number_integer() || static_cast<int>(camera["max-depth"]) <= 0)
                {
                    ctx.error("'max-depth' should be a positive integer", {});
                    return false;
                }
                param->maxDepth = camera["max-depth"];
            }
        } catch (...)
        {
            //insert error message
            return false;
        }

        return true;
    }

    static bool transform(Parser& parser, ParserState& state, json const& transform)
    {
        Context                                            ctx;
        thread_local std::unordered_set<std::string> const allowedKeys{"name", "srt"};
        if (!transform.is_object())
        {
            ctx.error("Error: Transform Expected a JSON Object", {});
            return false;
        }

        if (!hasOnlyAllowedkeys(transform, allowedKeys))
        {
            std::string extraneous;
            for (auto const& [key, value] : transform.items())
            {
                if (!allowedKeys.contains(key))
                {
                    extraneous = key;
                    break;
                }
            }
            ctx.error("Transform has extraneous key '{}'", std::make_tuple(extraneous));
            return false;
        }

        std::string name;
        json        srt;
        try
        {
            json const& jname = transform["name"];
            if (!jname.is_string() || static_cast<std::string>(jname).empty())
            {
                ctx.error("Expected 'name' to be a non-empty string", {});
                return false;
            }
            name = static_cast<std::string>(jname);
        } catch (...)
        {
            ctx.error("Expected 'name' and 'srt' as params for the transform object", {});
            return false;
        }

        ctx.log("Transform: Loading Transform '{}'", std::make_tuple(name));

        // check existance
        if (state.transforms.contains(name))
        {
            ctx.error("Transform with name {} already exists", std::make_tuple(name));
            return false;
        }

        // extract from srt values: "rotate-axis", "rotate-degrees", "translation-vector", "scale". If some of them
        // are absent, then we assume some defaults
        Vector3f rotateAxis    = {0, 0, 1};
        float    rotateDegrees = 0;
        Vector3f translationVector{};
        Vector3f scale{1, 1, 1}; // TODO should we allow small scales
        if (transform.contains("srt"))
        {
            srt = transform["srt"];
            try
            {
                if (srt.contains("rotate-axis"))
                {
                    Vector3f tmp{};
                    if (extractVec3f(srt["rotate-axis"], &tmp) != ExtractVec3fResult::eOk)
                    {
                        ctx.error("Error extracting field 'rotate-axis'", {});
                        return false;
                    }
                    rotateAxis = normalize(tmp);
                }
                if (srt.contains("rotate-degrees"))
                {
                    if (!srt["rotate-degrees"].is_number())
                    {
                        ctx.error("Error extracting field 'rotate-degrees'", {});
                        return false;
                    }
                    rotateDegrees = srt["rotate-degrees"];
                }
                if (srt.contains("translation-vector"))
                {
                    Vector3f tmp{};
                    if (extractVec3f(srt["translation-vector"], &tmp) != ExtractVec3fResult::eOk)
                    {
                        ctx.error("Error extracting field 'translation-vector'", {});
                        return false;
                    }
                    translationVector = tmp;
                }
                if (srt.contains("scale"))
                {
                    if (srt["scale"].is_number())
                        scale = Vector3f::s(static_cast<float>(srt["scale"]));
                    else if (Vector3f tmp{}; extractVec3f(srt["scale"], &tmp) == ExtractVec3fResult::eOk)
                        scale = tmp;
                    else
                    {
                        ctx.error("Error parsing field 'scale'", {});
                        return false;
                    }
                }
            } catch (...)
            {
                ctx.error("Unknown error during transform parsing", {});
                return false;
            }
        }

        Transform t = Transform::translate(translationVector) * Transform::rotate(rotateDegrees, rotateAxis) *
                      Transform::scale(scale);
        auto const& [it, wasInserted] = state.transforms.try_emplace(name, t);

        return wasInserted;
    }

    static bool light(Parser& parser, ParserState& state, json const& light)
    {
        Context                                            ctx;
        thread_local std::unordered_set<std::string> const allowedKeysPoint{"name", "type", "radiant-intensity"};
        thread_local std::unordered_set<std::string> const
            allowedKeysSpot{"name",
                            "type",
                            "cone-angle",
                            "falloff-percentage",
                            "radiant-intensity",
                            "cone-angle",
                            "falloff-percentage"};

        if (!light.is_object())
        {
            ctx.error("light is not an object", {});
            return false;
        }

        std::string name;
        std::string type;

        try
        {
            // compulsory name and type
            if (!light.contains("name") || !light["name"].is_string())
            {
                ctx.error("Absent or invalid type for light 'name'", {});
                return false;
            }

            name = light["name"];
            ctx.log("Light: Loading light '{}'", std::make_tuple(name));

            if (state.lights.contains(name))
            {
                ctx.error("Duplicated 'name' light found", {});
                return false;
            }

            if (!light.contains("type") || !light["type"].is_string())
            {
                ctx.error("Absent or invalid type for light 'type'", {});
                return false;
            }
            name = light["name"];
            type = light["type"];

            if (type == "point") // if point, then only radiant-intensity is the other allowed field
            {
                if (!hasOnlyAllowedkeys(light, allowedKeysPoint))
                {
                    std::string extraneous;
                    for (auto const& [key, value] : light.items())
                    {
                        if (!allowedKeysPoint.contains(key))
                        {
                            extraneous = key;
                            break;
                        }
                    }

                    ctx.error("Light: unexpected parameter '{}' for point light '{}'", std::make_tuple(extraneous, name));
                    return false;
                }

                RGB radiantIntensity{1, 1, 1};
                if (light.contains("radiant-intensity"))
                {
                    Vector3f tmp{};
                    if (extractVec3f(light["radiant-intensity"], &tmp) != ExtractVec3fResult::eOk)
                    {
                        ctx.error("Incorrect format for 'radiant-intensity' of point light", {});
                        return false;
                    }
                    radiantIntensity = RGB::fromVec(tmp);
                }

                auto const& [it, wasInserted] = //
                    state.lights.try_emplace(name, makePointLight(Transform{}, radiantIntensity));
                if (!wasInserted)
                    return false;
            }
            else if (type == "spot") // if spot, then radiant-intensity, cone-angle, falloff-percentage are the only allowed field
            {
                if (!hasOnlyAllowedkeys(light, allowedKeysSpot))
                {
                    std::string extraneous;
                    for (auto const& [key, value] : light.items())
                    {
                        if (!allowedKeysSpot.contains(key))
                        {
                            extraneous = key;
                            break;
                        }
                    }
                    ctx.error("Light: unexpected parameter '{}' for spot light '{}'", std::make_tuple(extraneous, name));
                    return false;
                }

                RGB   radiantIntensity{1, 1, 1};
                float coneAngle         = 60.f;
                float falloffPercentage = 10.f;

                if (light.contains("radiant-intensity"))
                {
                    Vector3f tmp{};
                    if (extractVec3f(light["radiant-intensity"], &tmp) != ExtractVec3fResult::eOk)
                    {
                        ctx.error("Incorrect format for 'radiant-intensity' of point light", {});
                        return false;
                    }
                    radiantIntensity = RGB::fromVec(max(tmp, {0, 0, 0}));
                }

                if (light.contains("cone-angle"))
                {
                    if (!light["cone-angle"].is_number())
                    {
                        ctx.error("'cone-angle' should be a number", {});
                        return false;
                    }
                    coneAngle = fl::clamp(light["cone-angle"], 10.f, 120.f);
                }

                if (light.contains("falloff-percentage"))
                {
                    if (!light["falloff-percentage"].is_number())
                    {
                        ctx.error("'falloff-percentage', should be a number", {});
                        return false;
                    }
                    falloffPercentage = fl::clamp(light["falloff-percentage"], 1.f, 80.f);
                }

                float const cosTheta0         = cosf(coneAngle * (1.f - falloffPercentage / 100.f) * fl::pi() / 180.f);
                float const cosThetae         = cosf(coneAngle * fl::pi() / 180.f);
                auto const& [it, wasInserted] = //
                    state.lights.try_emplace(name, makeSpotLight(Transform{}, radiantIntensity, cosTheta0, cosThetae));
                if (!wasInserted)
                    return false;
            }
            else
            {
                ctx.error("Undefined 'type' for light", {});
                return false;
            }
        } catch (...)
        {
            ctx.error("Unknown error parsing light", {});
            return false;
        }

        return true;
    }

    static bool envlight(Parser& parser, ParserState& state, json const& envLight, Renderer* renderer)
    {
        Context ctx;
        if (!envLight.is_string())
        {
            ctx.error("expected 'envLight' to be string", {});
            return false;
        }

        os::Path envPath = parser.fileDirectory() / static_cast<std::string>(envLight).c_str();
        if (!envPath.isValid() || !envPath.isFile())
        {
            ctx.error(
                "invalid path specified on 'envlight'."
                " Should be a supported image file relative to JSON file directory",
                {});
        }
        int32_t    xRes  = 0;
        int32_t    yRes  = 0;
        Quaternion quat  = Quaternion::quatIdentity();
        float      scale = 1.f; // TODO

        // NOTE: This image is leaked!!! (on purpose?, TODO: renderer's destructor should call free on the buffer)
        RGB* image = loadImageAsRGB(envPath, xRes, yRes);
        if (!image)
        {
            ctx.error("failed to load image at path '{}'", std::make_tuple(envPath.toUnderlying()));
            return false;
        }
        renderer->params.envLight = makeUniqueRef<EnvLight>(renderer->scene.memory(), image, xRes, yRes, quat, scale);
        if (!renderer->params.envLight)
        {
            ctx.error("failed to load image at path '{}'", std::make_tuple(envPath.toUnderlying()));
            return false;
        }

        return true;
    }

    static bool object(Parser& parser, ParserState& state, json const& object, Renderer* rend, MeshFbxParser& meshParser)
    {
        Context ctx;

        if (!object.is_object())
        {
            ctx.error("element of 'objects' should be a JSON object", {});
            return false;
        }

        std::string name;
        // grows depending on type. once type fully specified, check
        std::unordered_set<std::string> allowedKeys{"name", "type", "material"};
        allowedKeys.reserve(128);
        try
        {
            if (!object.contains("name") || !object["name"].is_string())
            {
                ctx.error("object specification mandates a unique 'name' as string", {});
                return false;
            }

            name = object["name"];
            ctx.log("Object: Loading object '{}'", std::make_tuple(name));

            if (state.objects.contains(name))
            {
                ctx.error("object 'name's should be unique", {});
                return false;
            }

            // get material id
            if (!object.contains("material") || !object["material"].is_string())
            {
                ctx.error("object should specify a string 'material' name", {});
                return false;
            }

            std::string const materialStr = object["material"];
            if (!state.materials.contains(materialStr))
            {
                ctx.error("couldn't find material of name `{}`", std::make_tuple(materialStr));
                return false;
            }
            uint32_t const materialIdx = state.materials.at(materialStr);

            // switch on type
            if (!object.contains("type") || !object["type"].is_string())
            {
                ctx.error("object should contain a string 'type'", {});
                return false;
            }
            std::string const type = object["type"];

            rend->scene.geometry.emplace_back(makeUniqueRef<TriangleMesh>(rend->scene.memory()));
            TriangleMesh& mesh = *rend->scene.geometry.back();
            if (type == "fbx" || type == "FBX")
            {
                allowedKeys.insert("path");
                if (!hasOnlyAllowedkeys(object, allowedKeys))
                {
                    std::string extraneous;
                    for (auto const& [key, value] : object.items())
                    {
                        if (!allowedKeys.contains(key))
                        {
                            extraneous = key;
                            break;
                        }
                    }

                    ctx.error("Object: unexpected parameter '{}' while loading object '{}'",
                              std::make_tuple(extraneous, name));
                    return false;
                }

                // check path is valid and file, then go to fbx parsing function
                if (!object.contains("path") || !object["path"].is_string())
                {
                    ctx.error("FBX object should contain a string 'path' field", {});
                    return false;
                }
                os::Path const fbxPath      = parser.fileDirectory() / static_cast<std::string>(object["path"]).c_str();
                std::pmr::string const path = fbxPath.toUnderlying();
                if (!fbxPath.isValid() || !fbxPath.isFile())
                {
                    ctx.error("FBX object 'path' field should point to a existing FBX file. Got `{}`",
                              std::make_tuple(path));
                    return false;
                }

                ctx.trace("Object: FBX Object detected. single-mesh FBX File Path: '{}'. Loading...",
                          std::make_tuple(path));
                if (!meshParser.ImportFBX(path.c_str(), &mesh))
                {
                    ctx.error("Object: ('{}') Error during parsing of FBX file '{}'", std::make_tuple(name, path));
                    return false;
                }

                for (size_t tri = 0; tri < mesh.triCount(); ++tri)
                {
                    mesh.getIndexedTriRef(tri).matIdx = static_cast<int32_t>(materialIdx);
                }
            }
            else if (type == "primitive")
            {
                allowedKeys.insert("shape");
                if (!hasOnlyAllowedkeys(object, allowedKeys))
                {
                    std::string extraneous;
                    for (auto const& [key, value] : object.items())
                    {
                        if (!allowedKeys.contains(key))
                        {
                            extraneous = key;
                            break;
                        }
                    }

                    ctx.error("Object: unexpected parameter '{}' while loading object '{}'",
                              std::make_tuple(extraneous, name));
                    return false;
                }

                if (!object.contains("shape") || !object["shape"].is_string())
                {
                    ctx.error("'shape' for primitive absent or ill-defined", {});
                    return false;
                }

                std::string const shape = object["shape"];
                if (shape == "cube")
                    TriangleMesh::unitCube(mesh, static_cast<int32_t>(materialIdx));
                else if (shape == "plane")
                    TriangleMesh::unitPlane(mesh, static_cast<int32_t>(materialIdx));
                else
                {
                    ctx.error("Unrecognized 'shape' `{}`", std::make_tuple(shape));
                    return false;
                }
            }
            else
            {
                ctx.error("Unrecognized type `{}` in object parsing", std::make_tuple(type));
                return false;
            }
        } catch (...)
        {
            ctx.error("Unknown error while parsing object", {});
            return false;
        }

        auto const& [it, wasInserted] = //
            state.objects.try_emplace(name, static_cast<uint32_t>(rend->scene.geometry.size() - 1));

        return wasInserted;
    }

    static bool parseWorldTranform(Parser& parse, ParserState& state, json const& worldObj, Renderer* rend)
    {

        //existing key,
        for (auto const& [key, value] : worldObj.items())
        {

            if (state.transforms.contains(key))
            {
                state.stackTrasforms.push_back(state.transforms[key]);
                if (!parse_helpers::parseWorldTranform(parse, state, value, rend))
                    return false;
                state.stackTrasforms.pop_back();
            }

            Transform currTransform = state.stackTrasforms.back();
            for (auto transform = state.stackTrasforms.rbegin() + 1; transform != state.stackTrasforms.rend(); transform++)
            {
                currTransform = (*transform) * currTransform;
            }

            //evaluate instances
            if (key == "instances")
            {
                if (!value.is_array())
                {
                    //insert error message
                    return false;
                }

                for (auto const& obj : value)
                {
                    if (!obj.is_string())
                        return false;
                    if (!state.objects.contains(static_cast<std::string>(obj)))
                        return false;

                    rend->scene.instances.emplace_back(makeUniqueRef<Instance>(rend->scene.memory()));
                    auto& instance   = *rend->scene.instances.back();
                    instance.meshIdx = state.objects[static_cast<std::string>(obj)];
                    extractAffineTransform(currTransform.m, instance.affineTransform);
                    instance.bounds = rend->scene.geometry[instance.meshIdx]->transformedBounds(currTransform);
                }
            }

            //evaluate the ligths
            else if (key == "lights")
            {

                if (!value.is_array())
                {
                    //insert error message
                    return false;
                }

                for (auto const& light : value)
                {
                    if (!light.is_string())
                        return false;
                    if (!state.lights.contains(light))
                    {
                        //insert error message
                        return false;
                    }

                    Light tmpLight = state.lights[light];
                    if (tmpLight.type == dmt::LightType::ePoint)
                    {

                        rend->scene.lights.push_back(dmt::makePointLight(currTransform,
                                                                         tmpLight.strength,
                                                                         tmpLight.data.point.radius,
                                                                         tmpLight.data.point.evalFac));
                    }
                    else if (tmpLight.type == dmt::LightType::eSpot)
                    {
                        rend->scene.lights.push_back(
                            dmt::makeSpotLight(currTransform,
                                               tmpLight.strength,
                                               tmpLight.data.spot.cosHalfSpotAngle,
                                               tmpLight.data.spot.cosHalfLargerSpread,
                                               tmpLight.data.spot.radius,
                                               tmpLight.data.spot.evalFac));
                    }
                    else
                        return false;
                }
            }
        }

        return true;
    }
} // namespace dmt::parse_helpers

namespace dmt {
    Parser::Parser(os::Path const& path, Renderer* renderer, std::pmr::memory_resource* mem) :
    m_renderer{renderer},
    m_path{path},
    m_tmp(mem)
    {
    }

    bool Parser::parse()
    {
        using json = nlohmann::json;
        Context ctx;
        if (!ctx.isValid())
            return false;

        ctx.log("Starting scene Parsing...", {});
#if defined(DMT_OS_LINUX)
        ctx.warn("  (remember to delete once in a while the content of ~/.cache/dmt-mipc/)", {});
#endif

        std::ifstream f(m_path.toUnderlying(m_tmp).c_str());
        if (!f)
            return false;

        static std::unordered_set<std::string> const
            allowedKeys{"camera",
                        "film",
                        "textures",
                        "materials",
                        "objects",
                        "lights",
                        "envlight",
                        "transforms",
                        "world"};

        try
        {
            json data = json::parse(f);
            if (!parse_helpers::hasAllAllowedkeys(data, allowedKeys))
                throw std::runtime_error("JSON value is lacking some keys");

            ParserState state;

            // parse camera object
            if (!parse_helpers::parseCamera(*this, data["camera"], &m_renderer->params))
                throw nullptr;

            // parse film object
            if (!parse_helpers::parseFilm(*this, data["film"], &m_renderer->params))
                throw nullptr;

            // parse texture array
            if (!data["textures"].is_array())
                throw std::runtime_error("'textures' should be a JSON array");

            for (json const& texture : data["textures"])
            {
                if (!parse_helpers::parseTexture(*this, state, texture, m_renderer))
                    throw nullptr;
            }

            // parse materials array
            if (!data["materials"].is_array())
                throw std::runtime_error("'materials' should be a JSON array");

            for (json const& material : data["materials"])
            {
                if (!parse_helpers::parseMaterial(*this, state, material, m_renderer))
                    throw nullptr;
            }

            // parse objects array
            if (!data["objects"].is_array())
                throw std::runtime_error("'objects' should be a JSON array");

            for (json const& object : data["objects"])
            {
                if (!parse_helpers::object(*this, state, object, m_renderer, m_fbxParser))
                    throw nullptr;
            }
            //parse ligths
            // parse objects array
            if (!data["lights"].is_array())
                throw std::runtime_error("'lights' should be a JSON array");

            for (json const& light : data["lights"])
            {
                if (!parse_helpers::light(*this, state, light))
                    throw nullptr;
            }
            // parse envlight
            if (!parse_helpers::envlight(*this, state, data["envlight"], m_renderer))
                throw nullptr;

            // parse transforms array
            if (!data["transforms"].is_array())
                throw std::runtime_error("'transforms' should be a JSON array");

            for (json const& transform : data["transforms"])
            {
                if (!parse_helpers::transform(*this, state, transform))
                    throw nullptr;
            }

            if (!data["world"].is_object())
                throw std::runtime_error("'world' should be an object");
            //existing key,
            for (auto const& [key, value] : data["world"].items())
            {
                if (state.transforms.contains(key))
                {
                    state.stackTrasforms.push_back(state.transforms[key]);
                    if (!parse_helpers::parseWorldTranform(*this, state, value, m_renderer))
                    {
                        throw std::runtime_error("Error to parse world");
                    }
                }
                state.stackTrasforms.pop_back();
            }
        } catch (...)
        {
            try
            {
                auto excPtr = std::current_exception();
                if (excPtr)
                    std::rethrow_exception(excPtr);
            } catch (std::exception const& e)
            {
                ctx.error("[Parser JSON Exception] {}", std::make_tuple(e.what()));
            }

            return false;
        }
        // true if successful
        return true;
    }
} // namespace dmt