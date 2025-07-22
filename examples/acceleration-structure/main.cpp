
#define DMT_ENTRY_POINT
#include "platform/platform.h"
#include "core/core-cudautils-cpubuild.h"
#include "core/core-bvh-builder.h"
#include "core/core-primitive.h"

#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <iomanip>

namespace dmt {
    std::pmr::vector<dmt::UniqueRef<Primitive>> makeSinglePrimitivesFromTriangles(
        std::span<TriangleData const> tris,
        std::pmr::memory_resource*    memory = std::pmr::get_default_resource())
    {
        std::pmr::vector<dmt::UniqueRef<Primitive>> out(memory);
        for (uint64_t i = 0; i < tris.size(); ++i)
        {
            Triangle group{};
            group.tri = tris[i];
            out.push_back(dmt::makeUniqueRef<Triangle>(memory, std::move(group)));
        }

        return out;
    }

    std::pmr::vector<dmt::UniqueRef<Primitive>> makePrimitivesFromTriangles(
        std::span<TriangleData const> tris,
        std::pmr::memory_resource*    memory = std::pmr::get_default_resource())
    {
        std::pmr::vector<dmt::UniqueRef<Primitive>> out(memory);
        size_t                                      i = 0;

        // Pass 1: Triangles8
        for (; i + 8 <= tris.size(); i += 8)
        {
            Triangles8 group{};
            for (int j = 0; j < 8; ++j)
            {
                group.xs[3 * j + 0] = tris[i + j].v0.x;
                group.xs[3 * j + 1] = tris[i + j].v1.x;
                group.xs[3 * j + 2] = tris[i + j].v2.x;

                group.ys[3 * j + 0] = tris[i + j].v0.y;
                group.ys[3 * j + 1] = tris[i + j].v1.y;
                group.ys[3 * j + 2] = tris[i + j].v2.y;

                group.zs[3 * j + 0] = tris[i + j].v0.z;
                group.zs[3 * j + 1] = tris[i + j].v1.z;
                group.zs[3 * j + 2] = tris[i + j].v2.z;
            }
            out.push_back(dmt::makeUniqueRef<Triangles8>(memory, std::move(group)));
        }

        // Pass 2: Triangles4
        for (; i + 4 <= tris.size(); i += 4)
        {
            Triangles4 group{};
            for (int j = 0; j < 4; ++j)
            {
                group.xs[3 * j + 0] = tris[i + j].v0.x;
                group.xs[3 * j + 1] = tris[i + j].v1.x;
                group.xs[3 * j + 2] = tris[i + j].v2.x;

                group.ys[3 * j + 0] = tris[i + j].v0.y;
                group.ys[3 * j + 1] = tris[i + j].v1.y;
                group.ys[3 * j + 2] = tris[i + j].v2.y;

                group.zs[3 * j + 0] = tris[i + j].v0.z;
                group.zs[3 * j + 1] = tris[i + j].v1.z;
                group.zs[3 * j + 2] = tris[i + j].v2.z;
            }
            out.push_back(dmt::makeUniqueRef<Triangles4>(memory, std::move(group)));
        }

        // Pass 3: Triangles2
        for (; i + 2 <= tris.size(); i += 2)
        {
            Triangles2 group{};
            for (int j = 0; j < 2; ++j)
            {
                group.xs[3 * j + 0] = tris[i + j].v0.x;
                group.xs[3 * j + 1] = tris[i + j].v1.x;
                group.xs[3 * j + 2] = tris[i + j].v2.x;

                group.ys[3 * j + 0] = tris[i + j].v0.y;
                group.ys[3 * j + 1] = tris[i + j].v1.y;
                group.ys[3 * j + 2] = tris[i + j].v2.y;

                group.zs[3 * j + 0] = tris[i + j].v0.z;
                group.zs[3 * j + 1] = tris[i + j].v1.z;
                group.zs[3 * j + 2] = tris[i + j].v2.z;
            }
            out.push_back(dmt::makeUniqueRef<Triangles2>(memory, std::move(group)));
        }

        // Pass 4: Individual Triangle
        for (; i < tris.size(); ++i)
        {
            Triangle group{};
            group.tri = tris[i];
            out.push_back(dmt::makeUniqueRef<Triangle>(memory, std::move(group)));
        }

        return out;
    }

    uint32_t morton3D(float x, float y, float z)
    {
        constexpr auto expandBits = [](uint32_t v) -> uint32_t {
            // Expands 10 bits into 30 bits by inserting 2 zeros between each bit
            v = (v * 0x00010001u) & 0xFF0000FFu;
            v = (v * 0x00000101u) & 0x0F00F00Fu;
            v = (v * 0x00000011u) & 0xC30C30C3u;
            v = (v * 0x00000005u) & 0x49249249u;
            return v;
        };
        // Assumes x, y, z are ∈ [0, 1]
        x = std::clamp(x * 1024.0f, 0.0f, 1023.0f);
        y = std::clamp(y * 1024.0f, 0.0f, 1023.0f);
        z = std::clamp(z * 1024.0f, 0.0f, 1023.0f);

        uint32_t xx = expandBits(static_cast<uint32_t>(x));
        uint32_t yy = expandBits(static_cast<uint32_t>(y));
        uint32_t zz = expandBits(static_cast<uint32_t>(z));

        return (xx << 2) | (yy << 1) | zz;
    }

    void reorderByMorton(std::span<TriangleData> tris)
    {
        Bounds3f bounds = bbEmpty();
        for (auto& t : tris)
            bounds = bbUnion(bounds,
                             dmt::Bounds3f{dmt::min(dmt::min(t.v0, t.v1), t.v2), dmt::max(dmt::max(t.v0, t.v1), t.v2)});

        auto const getMortonIndex = [bounds](TriangleData const& t) -> uint32_t {
            Point3f  c = (t.v0 + t.v1 + t.v2) / 3;
            Point3f  n = bounds.offset(c); // Normalize to [0,1]
            uint32_t m = morton3D(n.x, n.y, n.z);
            return m;
        };

        std::sort(tris.begin(), tris.end(), [&getMortonIndex](auto const& a, auto const& b) {
            return getMortonIndex(a) < getMortonIndex(b);
        });
    }


    namespace ddbg {
        std::string printBVHToString(BVHBuildNode* node, int depth = 0, std::string const& prefix = "")
        {
            std::string result;
            if (depth == 0)
                result += "\n";

            constexpr auto boundsToString = [](Bounds3f const& b) -> std::string {
                std::ostringstream oss;
                oss << "Bounds[( " << b.pMin.x << ", " << b.pMin.y << ", " << b.pMin.z << " ) - ( " << b.pMax.x << ", "
                    << b.pMax.y << ", " << b.pMax.z << " )]";
                return oss.str();
            };

            if (!node)
                return result;

            std::string indent(depth * 2, ' ');

            if (node->childCount == 0 && node->primitiveCount > 0)
            {
                result += indent + prefix + "Leaf [count: " + std::to_string(node->primitiveCount) + ", ";
                result += boundsToString(node->bounds) + "]\n";
            }
            else
            {
                result += indent + prefix + "Internal [children: " + std::to_string(node->childCount) + ", ";
                result += boundsToString(node->bounds) + "]\n";

                for (uint32_t i = 0; i < node->childCount; ++i)
                {
                    bool        isLast      = (i == node->childCount - 1);
                    std::string childPrefix = isLast ? "└─ " : "├─ ";
                    result += printBVHToString(node->children[i], depth + 1, childPrefix);
                }
            }

            return result;
        }

        std::vector<TriangleData> makeCubeTriangles()
        {
            std::vector<TriangleData> tris;

            float const                  s        = 1.0f;
            std::array<Point3f, 8> const vertices = {
                {{{-s, -s, -s}}, {{s, -s, -s}}, {{s, s, -s}}, {{-s, s, -s}}, {{-s, -s, s}}, {{s, -s, s}}, {{s, s, s}}, {{-s, s, s}}}};

            std::array<std::array<int, 3>, 12> const indices = {{
                // Front face
                {0, 1, 2},
                {2, 3, 0},
                // Back face
                {4, 7, 6},
                {6, 5, 4},
                // Left face
                {0, 3, 7},
                {7, 4, 0},
                // Right face
                {1, 5, 6},
                {6, 2, 1},
                // Top face
                {3, 2, 6},
                {6, 7, 3},
                // Bottom face
                {0, 4, 5},
                {5, 1, 0},
            }};

            for (auto const& idx : indices)
            {
                tris.push_back({{{vertices[idx[0]].x, vertices[idx[0]].y, vertices[idx[0]].z}},
                                {{vertices[idx[1]].x, vertices[idx[1]].y, vertices[idx[1]].z}},
                                {{vertices[idx[2]].x, vertices[idx[2]].y, vertices[idx[2]].z}}});
            }

            return tris;
        }

        std::vector<TriangleData> makePlaneTriangles(float size = 1.0f)
        {
            std::vector<TriangleData> tris;

            float const s = size * 0.5f;

            Point3f const v0{{-s, 0.0f, -s}};
            Point3f const v1{{s, 0.0f, -s}};
            Point3f const v2{{s, 0.0f, s}};
            Point3f const v3{{-s, 0.0f, s}};

            // Triangle 1: v0, v1, v2
            tris.emplace_back(v0, v1, v2);

            // Triangle 2: v2, v3, v0
            tris.emplace_back(v2, v3, v0);

            return tris;
        }

        std::pmr::vector<TriangleData> debugScene(std::pmr::memory_resource* memory = std::pmr::get_default_resource())
        {
            std::pmr::vector<TriangleData> scene{memory};

            auto cube  = makeCubeTriangles();
            auto plane = makePlaneTriangles(4.0f); // Large ground plane

            scene.insert(scene.end(), cube.begin(), cube.end());
            scene.insert(scene.end(), plane.begin(), plane.end());
            return scene;
        }

        std::pmr::vector<Primitive const*> rawPtrsCopy(std::pmr::vector<dmt::UniqueRef<Primitive>> const& ownedPrimitives,
                                                       std::pmr::memory_resource* memory = std::pmr::get_default_resource())
        {
            std::pmr::vector<Primitive const*> rawPtrs{memory};
            rawPtrs.reserve(ownedPrimitives.size());

            for (auto const& ref : ownedPrimitives)
                rawPtrs.push_back(ref.get());

            return rawPtrs;
        }
    } // namespace ddbg

    bool slabTest(Point3f rayOrigin, Vector3f rayDirection, Bounds3f const& box, float* outTmin = nullptr, float* outTmax = nullptr)
    {
        float tmin = -std::numeric_limits<float>::infinity();
        float tmax = std::numeric_limits<float>::infinity();

        for (int i = 0; i < 3; ++i)
        {
            float invD = 1.0f / rayDirection[i];
            float t0   = (box.pMin[i] - rayOrigin[i]) * invD;
            float t1   = (box.pMax[i] - rayOrigin[i]) * invD;

            if (invD < 0.0f)
                std::swap(t0, t1);

            tmin = std::max(tmin, t0);
            tmax = std::min(tmax, t1);

            if (tmax < tmin)
                return false; // No intersection
        }

        if (outTmin)
            *outTmin = tmin;
        if (outTmax)
            *outTmax = tmax;

        return true; // Intersection occurred
    }

    BVHBuildNode* traverseBVHBuild(Ray                        ray,
                                   BVHBuildNode*              bvh,
                                   std::pmr::memory_resource* memory = std::pmr::get_default_resource())
    {
        std::pmr::vector<BVHBuildNode*> activeNodeStack;
        activeNodeStack.reserve(64);
        activeNodeStack.push_back(bvh);

        BVHBuildNode* intersection = nullptr;
        while (!activeNodeStack.empty())
        {
            BVHBuildNode* current = activeNodeStack.back();
            activeNodeStack.pop_back();

            if (current->childCount > 0)
            {
                // children order of traversal: 1) Distance Heuristic: from smallest to highest tmin - ray origin 2) Sign Heuristic
                // start with distance heuristic
                struct
                {
                    uint32_t i = static_cast<uint32_t>(-1);
                    float    d = fl::infinity();
                } tmins[BranchingFactor];
                uint32_t currentIndex = 0;

                for (uint32_t i = 0; i < current->childCount; ++i)
                {
                    float tmin = fl::infinity();
                    if (slabTest(ray.o, ray.d, current->children[i]->bounds, &tmin))
                    {
                        tmins[currentIndex].d = tmin;
                        tmins[currentIndex].i = i;
                        ++currentIndex;
                    }
                }

                std::sort(std::begin(tmins), std::begin(tmins) + currentIndex, [](auto const& a, auto const& b) {
                    return a.d > b.d;
                });

                for (uint32_t i = 0; i < currentIndex; ++i)
                    activeNodeStack.push_back(current->children[tmins[i].i]);
            }
            else
            {
                // TODO handle any-hit, closest-hit, ...
                // for now, stop at the first leaf intersection
                intersection = current;
                break;
            }
        }

        return intersection;
    }
} // namespace dmt

void bvhTestRays(dmt::BVHBuildNode* rootNode)
{
    dmt::Context ctx;
    assert(ctx.isValid() && "Invalid context");
    std::vector<dmt::Ray> testRays = {
        // Straight through center of scene
        dmt::Ray({{0.5f, 0.5f, -1.0f}}, {{0, 0, 1}}),
        dmt::Ray({{1.5f, 1.5f, -1.0f}}, {{0, 0, 1}}),
        dmt::Ray({{2.5f, 2.5f, -1.0f}}, {{0, 0, 1}}),
        dmt::Ray({{3.5f, 3.5f, -1.0f}}, {{0, 0, 1}}),

        // Grazing edges
        dmt::Ray({{1.0f, 1.0f, -1.0f}}, {{0, 0, 1}}),
        dmt::Ray({{4.0f, 4.0f, -1.0f}}, {{0, 0, 1}}),

        // Missing all
        dmt::Ray({{5.0f, 5.0f, -1.0f}}, {{0, 0, 1}}),
        dmt::Ray({{-1.0f, -1.0f, -1.0f}}, {{0, 0, 1}}),

        // Diagonal through stack
        dmt::Ray({{-1.0f, -1.0f, 1.0f}}, {{1, 1, 1}}),
        dmt::Ray({{0.5f, 0.5f, 0.5f}}, {{1, 1, 1}}),

        // Through nested box
        dmt::Ray({{1.0f, 1.0f, 1.0f}}, {{0, 1, 0}}),
    };

    for (size_t i = 0; i < testRays.size(); ++i)
    {
        dmt::BVHBuildNode* hit = dmt::traverseBVHBuild(testRays[i], rootNode);
        if (hit)
        {
            ctx.log("Ray {} hit leaf bounding box: min = ({}, {}, {}), max = ({}, {}, {})",
                    std::make_tuple(i,
                                    hit->bounds.pMin.x,
                                    hit->bounds.pMin.y,
                                    hit->bounds.pMin.z,
                                    hit->bounds.pMax.x,
                                    hit->bounds.pMax.y,
                                    hit->bounds.pMax.z));
        }
        else
        {
            ctx.log("Ray {} missed the scene.", std::make_tuple(i));
        }
    }
}

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

        // Sample scene
        std::unique_ptr<unsigned char[]> bufferPtr    = std::make_unique<unsigned char[]>(2048);
        auto                             bufferMemory = std::pmr::monotonic_buffer_resource(bufferPtr.get(), 2048);
        auto                             scene        = dmt::ddbg::debugScene();
        dmt::reorderByMorton(scene);
        auto prims     = dmt::makeSinglePrimitivesFromTriangles(scene);
        auto primsView = dmt::ddbg::rawPtrsCopy(prims);

        std::span<dmt::Primitive const*> spanPrims{primsView};

        // check that prims bounds equal scene bounds
        dmt::Bounds3f const
            sceneBounds = std::transform_reduce(scene.begin(), scene.end(), dmt::bbEmpty(), [](dmt::Bounds3f a, dmt::Bounds3f b) {
            return dmt::bbUnion(a, b);
        }, [](dmt::TriangleData const& t) {
            return dmt::Bounds3f{dmt::min(dmt::min(t.v0, t.v1), t.v2), dmt::max(dmt::max(t.v0, t.v1), t.v2)};
        });
        dmt::Bounds3f const primsBounds = std::transform_reduce( //
            spanPrims.begin(),
            spanPrims.end(),
            dmt::bbEmpty(),
            [](dmt::Bounds3f a, dmt::Bounds3f b) { return dmt::bbUnion(a, b); },
            [](dmt::Primitive const* p) { return p->bounds(); });

        if (sceneBounds != primsBounds)
        {
            ctx.error("{{ {{ {} {} {} }} {{ {} {} {} }}}}",
                      std::make_tuple(sceneBounds.pMin.x,
                                      sceneBounds.pMin.y,
                                      sceneBounds.pMin.z,
                                      sceneBounds.pMax.x,
                                      sceneBounds.pMax.y,
                                      sceneBounds.pMax.z));
            ctx.error("vs", {});
            ctx.error("{{ {{ {} {} {} }} {{ {} {} {} }}}}",
                      std::make_tuple(primsBounds.pMin.x,
                                      primsBounds.pMin.y,
                                      primsBounds.pMin.z,
                                      primsBounds.pMax.x,
                                      primsBounds.pMax.y,
                                      primsBounds.pMax.z));
            assert(false && "why");
        }
        else
        {
            ctx.log("{{ {{ {} {} {} }} {{ {} {} {} }}}}",
                    std::make_tuple(primsBounds.pMin.x,
                                    primsBounds.pMin.y,
                                    primsBounds.pMin.z,
                                    primsBounds.pMax.x,
                                    primsBounds.pMax.y,
                                    primsBounds.pMax.z));
        }

        auto* rootNode = dmt::bvh::build(spanPrims, &bufferMemory);

        if (ctx.isLogEnabled())
        {
            ctx.log("-- Before Primitive Packing --", {});
            std::string tee = dmt::ddbg::printBVHToString(rootNode);
            ctx.log(std::string_view{tee}, {});
        }

        bvhTestRays(rootNode);

        dmt::bvh::groupTrianglesInBVHLeaves(rootNode, prims, &bufferMemory);

        if (ctx.isLogEnabled())
        {
            ctx.log("-- After Primitive Packing --", {});
            std::string tee = dmt::ddbg::printBVHToString(rootNode);
            ctx.log(std::string_view{tee}, {});
        }

        bvhTestRays(rootNode);

        dmt::bvh::cleanup(rootNode);
    }

    return 0;
}
