#define DMT_ENTRY_POINT
#define DMT_WINDOWS_CLI
#include "platform/platform.h"
#include "core/core-bvh-builder.h"
#include "core/core-cudautils-cpubuild.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace dmt {
    void resetMonotonicBufferPointer(std::pmr::monotonic_buffer_resource& resource, unsigned char* ptr, uint32_t bytes)
    {
        // https://developercommunity.visualstudio.com/t/monotonic_buffer_resourcerelease-does/10624172
        auto* upstream = resource.upstream_resource();
        std::destroy_at(&resource);
        std::construct_at(&resource, ptr, bytes, upstream);
    }

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
    }
    return 0;
}
