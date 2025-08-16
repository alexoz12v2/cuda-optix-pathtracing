#define DMT_ENTRY_POINT
#define DMT_WINDOWS_CLI
#include "platform/platform.h"
#include "core/core-bvh-builder.h"
#include "core/core-cudautils-cpubuild.h"
#include "core/core-texture-cache.h"
#include "core/core-render.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8

#include <stb_image_write.h>
#include <stb_image.h>

namespace dmt {
    // TODO somehow integrate this with texture
    struct PNGInfoWithGamma
    {
        float gamma{0.0f}; // if present in gAMA chunk
        int   width{};
        int   height{};
        int   channels : 31 = 0;
        int   is_srgb  : 1  = false;
    };

    // Minimal helper to read a big-endian 32-bit integer from PNG file
    static uint32_t read_be32(FILE* f)
    {
        uint8_t buf[4];
        fread(buf, 1, 4, f);
        return (uint32_t(buf[0]) << 24) | (uint32_t(buf[1]) << 16) | (uint32_t(buf[2]) << 8) | uint32_t(buf[3]);
    }

    // Reads PNG chunks until IHDR is done and optional gAMA/sRGB/iCCP are parsed
    // https://www.w3.org/TR/2003/REC-PNG-20031110/#11addnlcolinfo
    static void PNGdetectColorspace(char const* filename, PNGInfoWithGamma& out)
    {
        FILE* f = fopen(filename, "rb");
        if (!f)
            return;

        // Verify PNG signature
        uint8_t sig[8];
        if (fread(sig, 1, 8, f) != 8 || memcmp(sig, "\x89PNG\r\n\x1a\n", 8) != 0)
        {
            fclose(f);
            return; // Not a PNG
        }

        bool done = false;
        while (!done)
        {
            uint32_t length  = read_be32(f);
            char     type[5] = {};
            fread(type, 1, 4, f);
            type[4] = 0;

            if (strcmp(type, "IHDR") == 0)
            {
                out.width  = int(read_be32(f));
                out.height = int(read_be32(f));
                fseek(f, length - 8, SEEK_CUR); // skip rest of IHDR
            }
            else if (strcmp(type, "gAMA") == 0)
            {
                uint32_t gamma_int = read_be32(f);
                out.gamma          = float(gamma_int) / 100000.0f;
                fseek(f, length - 4, SEEK_CUR);
            }
            else if (strcmp(type, "sRGB") == 0)
            {
                out.is_srgb = true;
                fseek(f, length, SEEK_CUR);
            }
            else if (strcmp(type, "iCCP") == 0)
            {
                out.is_srgb = true; // assume sRGB-like if ICC profile present
                fseek(f, length, SEEK_CUR);
            }
            else
            {
                fseek(f, length, SEEK_CUR); // skip other chunk
            }

            fseek(f, 4, SEEK_CUR); // skip CRC

            if (strcmp(type, "IDAT") == 0)
            {
                done = true; // Start of image data, we can stop scanning
            }
        }

        fclose(f);
    }

    unsigned char* stbi_load_with_gamma_info(char const* filename, int* x, int* y, int* comp, int req_comp, PNGInfoWithGamma* out_info)
    {
        if (out_info)
        {
            *out_info = {};
            PNGdetectColorspace(filename, *out_info);
        }

        unsigned char* data = stbi_load(filename, x, y, comp, req_comp);
        if (out_info)
        {
            out_info->channels = (req_comp != 0 ? req_comp : *comp);
        }
        return data;
    }
} // namespace dmt

namespace dmt {
    static BVHBuildNode* buildBVHBuildLayout(Scene& scene, std::pmr::synchronized_pool_resource& pool)
    {
        UniqueRef<unsigned char[]> bufTmp = makeUniqueRef<unsigned char[]>(std::pmr::get_default_resource(), 4096);
        std::pmr::monotonic_buffer_resource scratch{bufTmp.get(), 4096, std::pmr::null_memory_resource()};

        { // prepare geometry
            scene.geometry.reserve(16);
            scene.geometry.emplace_back(makeUniqueRef<TriangleMesh>(std::pmr::get_default_resource()));
            auto& cube = *scene.geometry.back();
            TriangleMesh::unitCube(cube);
        }
        { // instance geometry
            {
                scene.instances.emplace_back(makeUniqueRef<Instance>(std::pmr::get_default_resource()));
                auto& cubeInstance   = *scene.instances.back();
                cubeInstance.meshIdx = 0;

                Transform const t = Transform::translate({0.f, 1.5f, -0.75f}) *
                                    Transform::rotate(45.f, Vector3f::zAxis()) * Transform::scale(0.5f);

                extractAffineTransform(t.m, cubeInstance.affineTransform);
                cubeInstance.bounds = scene.geometry[cubeInstance.meshIdx]->transformedBounds(t);
                cubeInstance.color  = {0.3f, 0.7f, 0.6f};
            }
            {
                scene.instances.emplace_back(makeUniqueRef<Instance>(std::pmr::get_default_resource()));
                auto& cubeInstance   = *scene.instances.back();
                cubeInstance.meshIdx = 0;

                Transform const t = Transform::translate({0.5f, 1.2f, -0.9f}) *
                                    Transform::rotate(70.f, Vector3f::zAxis()) * Transform::scale(0.42f);

                extractAffineTransform(t.m, cubeInstance.affineTransform);
                cubeInstance.bounds = scene.geometry[cubeInstance.meshIdx]->transformedBounds(t);
                cubeInstance.color  = {0.5f, 0.7f, 0.6f};
            }
            {
                scene.instances.emplace_back(makeUniqueRef<Instance>(std::pmr::get_default_resource()));
                auto& cubeInstance   = *scene.instances.back();
                cubeInstance.meshIdx = 0;

                Transform const t = Transform::translate({0.5f, 1.7f, -0.6f}) *
                                    Transform::rotate(55.f, normalize(Vector3f{Vector3f::zAxis()} + Vector3f::xAxis())) *
                                    Transform::scale(0.42f);

                extractAffineTransform(t.m, cubeInstance.affineTransform);
                cubeInstance.bounds = scene.geometry[cubeInstance.meshIdx]->transformedBounds(t);
                cubeInstance.color  = {0.7f, 0.3f, 0.6f};
            }
            // compute per instance BVH and total BVH
            std::pmr::vector<BVHBuildNode*>        perInstanceBvhNodes{&pool};
            std::pmr::vector<UniqueRef<Primitive>> primitives{&pool};
            perInstanceBvhNodes.reserve(64);
            primitives.reserve(256);

            for (size_t instanceIdx = 0; instanceIdx < scene.instances.size(); ++instanceIdx)
            {
                perInstanceBvhNodes.push_back(bvh::buildForInstance(scene, instanceIdx, primitives, &scratch, &pool));
                resetMonotonicBufferPointer(scratch, bufTmp.get(), 4096);
            }

            auto* bvhRoot = reinterpret_cast<BVHBuildNode*>(pool.allocate(sizeof(BVHBuildNode)));
            std::memset(bvhRoot, 0, sizeof(BVHBuildNode));
            bvh::buildCombined(bvhRoot, perInstanceBvhNodes, &scratch, &pool);

            return bvhRoot;
        }
    }

    static void testTextureCache()
    {
        std::pmr::unsynchronized_pool_resource poolList;
        std::pmr::unsynchronized_pool_resource poolMap;

        auto* texTmpMem = std::pmr::get_default_resource();

        // 256 MB cache
        TextureCache texCache{1ull << 28, &poolMap, &poolList};
        int          width = 0, height = 0, comp = 0;
        os::Path     image = os::Path::executableDir();
        image /= "white.png";
        PNGInfoWithGamma pngInfo;
        uint8_t* data = stbi_load_with_gamma_info(image.toUnderlying().c_str(), &width, &height, &comp, 3, &pngInfo);
        if (!data)
        {
            // Todo context logging for CLI Windows
            std::cerr << "Couldn't load image data" << std::endl;
            return;
        }
        if (!isPOT(width) || !isPOT(height))
        {
            std::cerr << "We currently support only power of two resolution textures" << std::endl;
            stbi_image_free(data);
            return;
        }
        std::cout << "Does PNG file have gAMA header? " << (pngInfo.gamma != 0.f ? "yes" : "no") << std::endl;

        auto tex = makeRGBMipmappedTexture(data, width, height, TexWrapMode::eClamp, TexWrapMode::eClamp, TexFormat::ByteRGB, texTmpMem);
        stbi_image_free(data);
        if (!tex.data)
        {
            std::cerr << "Couldn't construct mipmapped image data" << std::endl;
            return;
        }

        uint64_t const fileKey = baseKeyFromPath(image);
        if (!texCache.MipcFiles.createCacheFile(fileKey, tex))
        {
            freeImageTexture(tex, texTmpMem);
            std::cerr << "Couldn't create mipc file data" << std::endl;
            return;
        }

        std::cout << "created mipc file data, see it on Temp and continue..." << std::endl;
        std::cin.get();

        uint32_t  bytes      = 0;
        size_t    alignedOff = 0;
        TexFormat format;

        int32_t startOffset = texCache.MipcFiles.copyMipOfKey(fileKey, 1, nullptr, &bytes, &alignedOff, &format);
        if (!startOffset)
        {
            freeImageTexture(tex, texTmpMem);
            std::cerr << "Couldn't extract mipc lod 1 metadata" << std::endl;
            return;
        }

        int width1   = width >> 1;  // set appropriately
        int height1  = height >> 1; // set appropriately
        int channels = 3;           // RGB

        if (bytes < static_cast<uint32_t>(width1) * height1 * channels)
        {
            freeImageTexture(tex, texTmpMem);
            std::cerr << "Insufficient bytes " << bytes << " for a " << width1 << "x" << height1 << std::endl;
            return;
        }
        auto lod1morton = makeUniqueRef<unsigned char[]>(texTmpMem, bytes);
        if (!texCache.MipcFiles.copyMipOfKey(fileKey, 1, lod1morton.get(), &bytes, &alignedOff, &format))
        {
            freeImageTexture(tex, texTmpMem);
            std::cerr << "Couldn't read mipc lod 1 metadata" << std::endl;
            return;
        }

        unsigned char* mipData = lod1morton.get() + startOffset; // skip filler

        // allocate row-major buffer
        std::vector<unsigned char> rowMajor(width1 * height1 * channels);

        // assume data is one byte per channel
        for (size_t mortonIndex = 0; mortonIndex < static_cast<size_t>(width1) * height1; ++mortonIndex)
        {
            unsigned int x = decodeMortonX(static_cast<unsigned int>(mortonIndex));
            unsigned int y = decodeMortonY(static_cast<unsigned int>(mortonIndex));

            if (x >= width1 || y >= height1)
                continue; // safety

            size_t dstIndex = (y * width1 + x) * channels;
            size_t srcIndex = mortonIndex * channels;
            assert(dstIndex + 2 < rowMajor.size());
            assert(srcIndex + 2 < bytes);

            rowMajor[dstIndex + 0] = mipData[srcIndex + 0];
            rowMajor[dstIndex + 1] = mipData[srcIndex + 1];
            rowMajor[dstIndex + 2] = mipData[srcIndex + 2];
        }

        // write PNG
        os::Path p = os::Path::executableDir() / "lod1.png";
        if (!stbi_write_png(p.toUnderlying().c_str(), width1, height1, channels, rowMajor.data(), width1 * channels))
        {
            freeImageTexture(tex, texTmpMem);
            std::cerr << "Failed to write PNG" << std::endl;
        }

        unsigned char* original = reinterpret_cast<unsigned char*>(tex.data) +
                                  mortonLevelOffset(width, height, 1) * bytesPerPixel(tex.texFormat);
        for (size_t mortonIndex = 0; mortonIndex < static_cast<size_t>(width) * height; ++mortonIndex)
        {
            unsigned int x = decodeMortonX(static_cast<unsigned int>(mortonIndex));
            unsigned int y = decodeMortonY(static_cast<unsigned int>(mortonIndex));

            if (x >= width1 || y >= height1)
                continue; // safety

            size_t dstIndex = (y * width1 + x) * channels;
            size_t srcIndex = mortonIndex * channels;
            assert(dstIndex + 2 < rowMajor.size());
            assert(srcIndex + 2 < bytes);

            rowMajor[dstIndex + 0] = original[srcIndex + 0];
            rowMajor[dstIndex + 1] = original[srcIndex + 1];
            rowMajor[dstIndex + 2] = original[srcIndex + 2];
        }

        os::Path p1 = os::Path::executableDir() / "lod1_tex.png";
        if (!stbi_write_png(p1.toUnderlying().c_str(), width1, height1, channels, rowMajor.data(), width1 * channels))
        {
            freeImageTexture(tex, texTmpMem);
            std::cerr << "Failed to write PNG" << std::endl;
        }

        freeImageTexture(tex, texTmpMem);
    }

    static void testTextureCacheLRU()
    {
        std::pmr::unsynchronized_pool_resource poolList;
        std::pmr::unsynchronized_pool_resource poolMap;

        auto* texTmpMem = std::pmr::get_default_resource();

        // Small cache so we can trigger eviction
        TextureCache     texCache{1ull << 26, &poolMap, &poolList}; // 64 MB
        int              width = 0, height = 0, comp = 0;
        os::Path         image = os::Path::executableDir() / "white.png";
        PNGInfoWithGamma pngInfo;
        uint8_t* data = stbi_load_with_gamma_info(image.toUnderlying().c_str(), &width, &height, &comp, 3, &pngInfo);
        if (!data)
        {
            std::cerr << "FAIL: Couldn't load white.png" << std::endl;
            return;
        }
        if (!isPOT(width) || !isPOT(height))
        {
            std::cerr << "FAIL: white.png is not POT" << std::endl;
            stbi_image_free(data);
            return;
        }

        auto tex = makeRGBMipmappedTexture(data, width, height, TexWrapMode::eClamp, TexWrapMode::eClamp, TexFormat::ByteRGB, texTmpMem);
        stbi_image_free(data);
        if (!tex.data)
        {
            std::cerr << "FAIL: Couldn't create mipmapped texture" << std::endl;
            return;
        }

        uint64_t const keyA = baseKeyFromPath(image);
        if (!texCache.MipcFiles.createCacheFile(keyA, tex))
        {
            std::cerr << "FAIL: Couldn't create mipc for keyA" << std::endl;
            freeImageTexture(tex, texTmpMem);
            return;
        }

        // Duplicate file to create a second texture key
        os::Path image2 = os::Path::executableDir() / "white2.png";
        {
            auto const* copied   = reinterpret_cast<wchar_t const*>(image2.internalData());
            auto const* existing = reinterpret_cast<wchar_t const*>(image.internalData());
            // TODO remove
            if (!CopyFileW(existing, copied, false))
            {
                std::cerr << "FAIL: couldn't copy file" << std::endl;
                freeImageTexture(tex, texTmpMem);
                return;
            }
        }

        uint64_t const keyB = baseKeyFromPath(image2);
        if (!texCache.MipcFiles.createCacheFile(keyB, tex))
        {
            std::cerr << "FAIL: Couldn't create mipc for keyB" << std::endl;
            freeImageTexture(tex, texTmpMem);
            return;
        }

        // --- Cache hit/miss test ---
        uint32_t  bytes = 0;
        TexFormat format;
        auto*     dataA1 = texCache.getOrInsert(keyA, 0, bytes, format); // miss -> load
        auto*     dataA2 = texCache.getOrInsert(keyA, 0, bytes, format); // hit -> same ptr
        if (dataA1 != dataA2)
        {
            std::cerr << "FAIL: Cache hit returned different pointer for keyA\n";
            return;
        }

        // --- Force eviction ---
        texCache.getOrInsert(keyB, 0, bytes, format);                // likely evicts A in small cache
        auto* dataA3 = texCache.getOrInsert(keyA, 0, bytes, format); // should reload
        if (dataA3 == dataA1)
        {
            std::cerr << "FAIL: Expected eviction of keyA but pointer is same\n";
            return;
        }

        // --- Validate that data loads correctly ---
        if (!dataA3)
        {
            std::cerr << "FAIL: dataA3 is null after reload\n";
            return;
        }

        std::cerr << "TextureCache LRU test passed!\n";
        freeImageTexture(tex, texTmpMem);
    }

} // namespace dmt

#define DMT_DBG_PX_X 50
#define DMT_DBG_PX_Y 58

int32_t guardedMain()
{
    dmt::Ctx::init();
    class Janitor
    {
    public:
        ~Janitor() { dmt::Ctx::destroy(); }
    } j;

    {
        dmt::Context ctx;
        ctx.log("Hello Cruel World", {});

        auto monotonicBuf = dmt::makeUniqueRef<unsigned char[]>(std::pmr::get_default_resource(), 4096);

        std::pmr::synchronized_pool_resource pool{};
        std::pmr::monotonic_buffer_resource  scratch{monotonicBuf.get(), 4096, std::pmr::null_memory_resource()};

        dmt::Scene           scene;
        dmt::BVHBuildNode*   bvhRoot   = dmt::buildBVHBuildLayout(scene, pool);
        uint32_t             nodeCount = 0;
        dmt::BVHWiVeCluster* bvh       = dmt::bvh::buildBVHWive(bvhRoot, &nodeCount, &scratch, &pool);

        static constexpr uint32_t       Width = 128, Height = 128, Channels = 1;
        dmt::os::Path                   imagePath = dmt::os::Path::executableDir() / "wive.png";
        dmt::UniqueRef<unsigned char[]> buffer = dmt::makeUniqueRef<unsigned char[]>(std::pmr::get_default_resource(),
                                                                                     static_cast<size_t>(Width) *
                                                                                         Height * Channels);
        // define camera (image plane physical dims, resolution given by image)
        dmt::Vector3f const cameraPosition{0.f, 0.f, 0.f};
        dmt::Normal3f const cameraDirection{0.f, 1.f, -0.5f};
        float const         focalLength  = 20e-3f; // 20 mm
        float const         sensorHeight = 36e-3f; // 36mm
        float const         aspectRatio  = static_cast<float>(Width) / Height;

        dmt::Transform const
            cameraFromRaster = dmt::transforms::cameraFromRaster_Perspective(focalLength, sensorHeight, Width, Height);
        dmt::Transform const renderFromCamera = dmt::transforms::worldFromCamera(cameraDirection, cameraPosition);

        std::memset(buffer.get(), 0, static_cast<size_t>(Width) * Height * Channels);
        for (int32_t y = 0; y < Height; ++y)
        {
            for (int32_t x = 0; x < Width; ++x)
            {
                if (x == DMT_DBG_PX_X && y == DMT_DBG_PX_Y)
                    int i = 0;
                using namespace dmt;
                Point3f const pxImage{x + 0.5f, y + 0.5f, 0};
                Point3f const pCamera{cameraFromRaster(pxImage)};
                Ray           ray{Point3f{0, 0, 0}, normalize(pCamera), 0};
                float         tMax = 1e5f;
                ray                = renderFromCamera(ray, &tMax);

                // Test
                uint32_t instanceIdx = 0;
                size_t   triIdx      = 0;
                auto     trisect     = dmt::triangle::Triisect::nothing();
                if (dmt::bvh::traverseRay(ray, bvh, nodeCount, &instanceIdx, &triIdx, &trisect))
                    buffer[x + static_cast<int64_t>(y) * Width] = 255;
            }
        }

        stbi_write_png(imagePath.toUnderlying().c_str(), Width, Height, 1, buffer.get(), Width);

        dmt::testTextureCache();
        dmt::testTextureCacheLRU();

        std::cout << "[Main Thread] Starting Render Thread" << std::endl;
        std::cin.get();
        {
            dmt::Renderer renderer;
            renderer.startRenderThread();
        }
        std::cout << "[Main Thread] Render Thread joined" << std::endl;
    }
    return 0;
}
