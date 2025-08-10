#define DMT_ENTRY_POINT
#define DMT_WINDOWS_CLI
#include "platform/platform.h"
#include "core/core-bvh-builder.h"

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
        dmt::BVHBuildNode*   bvhRoot = dmt::buildBVHBuildLayout(scene, pool);
        dmt::BVHWiVeCluster* bvh     = dmt::bvh::buildBVHWive(bvhRoot, &scratch, &pool);

        int i = 0;
    }
    return 0;
}
