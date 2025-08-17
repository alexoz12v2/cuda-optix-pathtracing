#include "core-trianglemesh.h"

#include <numeric>

namespace dmt {
    TriangleMesh::TriangleMesh(size_t cap, std::pmr::memory_resource* memory) :
    m_positions{memory},
    m_normals{memory},
    m_uvs{memory},
    m_tris{memory}
    {
        m_positions.reserve(cap);
        m_normals.reserve(cap);
        m_uvs.reserve(cap);
        m_tris.reserve(cap);
    }

    TriangleMesh& TriangleMesh::addPosition(Point3f p)
    {
        m_positions.emplace_back(p);
        return *this;
    }

    TriangleMesh& TriangleMesh::addNormal(Normal3f n)
    {
        m_normals.emplace_back(n);
        return *this;
    }

    TriangleMesh& TriangleMesh::addUV(Point2f uv)
    {
        m_uvs.emplace_back(uv);
        return *this;
    }

    TriangleMesh& TriangleMesh::addIndexedTriangle(VertexIndex i0, VertexIndex i1, VertexIndex i2, int32_t matIdx)
    {
        m_tris.emplace_back(i0, i1, i2, matIdx);
        return *this;
    }

    Point3f TriangleMesh::getPosition(size_t idx) const { return m_positions[idx]; }

    Normal3f TriangleMesh::getNormal(size_t idx) const { return m_normals[idx]; }

    Point2f TriangleMesh::getUV(size_t idx) const { return m_uvs[idx]; }

    IndexedTri TriangleMesh::getIndexedTri(size_t idx) const { return m_tris[idx]; }

    Bounds3f TriangleMesh::transformedBounds(float const affine[12]) const
    {
        return transformedBounds(transformFromAffine(affine));
    }

    Bounds3f TriangleMesh::transformedBounds(Transform const& t) const
    {
        Bounds3f bounds = bbEmpty();
        for (auto const& triIndex : m_tris)
        {
            Point3f const v0 = t(m_positions[triIndex.v[0].positionIdx]);
            Point3f const v1 = t(m_positions[triIndex.v[1].positionIdx]);
            Point3f const v2 = t(m_positions[triIndex.v[2].positionIdx]);

            bounds = bbUnion(bbUnion(bbUnion(bounds, v0), v1), v2);
        }

        return bounds;
    }

    // static

    void TriangleMesh::unitCube(TriangleMesh& mesh, int32_t matIdx)
    {
        static constexpr float half = 0.5f;

        // divide uv space by 64 tiles
        static constexpr float uvtile        = 0.125f;
        static constexpr float uExtremeLeft  = uvtile;
        static constexpr float uLeft         = uvtile * 3;
        static constexpr float uRight        = uvtile * 5;
        static constexpr float uExtremeRight = uvtile * 7;

        static constexpr float vExtremeBottom = 0.f;
        static constexpr float vBottom        = uvtile * 2;
        static constexpr float vCenter        = uvtile * 4;
        static constexpr float vTop           = uvtile * 6;
        static constexpr float vExtremeTop    = 1.f;

        assert(mesh.positionCount() == 0 && "mesh is supposed to be empty to insert a primitive");

        mesh.addPosition({-half, -half, half})  // 0 left front top (8 vertices total)
            .addPosition({-half, -half, -half}) // 1 left front bottom
            .addPosition({half, -half, -half})  // 2 right front bottom
            .addPosition({half, -half, half})   // 3 right front top
            .addPosition({-half, half, half})   // 4 left back top
            .addPosition({-half, half, -half})  // 5 left back bottom
            .addPosition({half, half, -half})   // 6 right back bottom
            .addPosition({half, half, half});   // 7 right back top

        mesh.addNormal({1, 0, 0})   // 0 right   (+X) (6 normals total)
            .addNormal({-1, 0, 0})  // 1 left    (-X)
            .addNormal({0, 1, 0})   // 2 back    (+Y)
            .addNormal({0, -1, 0})  // 3 front   (-Y)
            .addNormal({0, 0, 1})   // 4 up      (+Z)
            .addNormal({0, 0, -1}); // 5 down    (-Z)

        // cross, starting from bottom left (open blender's cube on uv editor
        // and spreadsheet, face corner domain)
        mesh.addUV({uLeft, vExtremeBottom})  //  0 bottom-left of front
            .addUV({uRight, vExtremeBottom}) //  1 bottom-right of front
            .addUV({uRight, vBottom})        //  2 top-right of front
            .addUV({uRight, vCenter})        //  3 top-right of right
            .addUV({uExtremeRight, vCenter}) //  4 top-right of back
            .addUV({uExtremeRight, vTop})    //  5 bottom-right of back
            .addUV({uRight, vTop})           //  6 bottom-left of back
            .addUV({uRight, vExtremeTop})    //  7 top-left of back
            .addUV({uLeft, vExtremeTop})     //  8 top-left of left
            .addUV({uLeft, vTop})            //  9 bottom-left of left
            .addUV({uExtremeLeft, vTop})     // 10 bottom-right of left
            .addUV({uExtremeLeft, vCenter})  // 11 top-right of left
            .addUV({uLeft, vCenter})         // 12 top-left of front
            .addUV({uLeft, vBottom});        // 13 bottom-left of front

        // Faces: each face has 2 triangles
        // LEFT (-X)
        mesh.addIndexedTriangle({1, 1, 9}, {5, 1, 10}, {4, 1, 11}, matIdx)
            .addIndexedTriangle({1, 1, 9}, {4, 1, 11}, {0, 1, 12}, matIdx);
        // FRONT (+Y)
        mesh.addIndexedTriangle({2, 3, 6}, {1, 3, 9}, {0, 3, 12}, matIdx) //
            .addIndexedTriangle({2, 3, 6}, {0, 3, 12}, {3, 3, 3}, matIdx);
        // RIGHT (+X)
        mesh.addIndexedTriangle({6, 0, 5}, {2, 0, 6}, {3, 0, 3}, matIdx) //
            .addIndexedTriangle({6, 0, 5}, {3, 0, 3}, {7, 0, 4}, matIdx);
        // TOP (+Z)
        mesh.addIndexedTriangle({3, 4, 3}, {0, 4, 12}, {4, 4, 13}, matIdx) //
            .addIndexedTriangle({3, 4, 3}, {4, 4, 13}, {7, 4, 2}, matIdx);
        // BACK (-Y)
        mesh.addIndexedTriangle({7, 2, 2}, {4, 2, 13}, {5, 2, 0}, matIdx) //
            .addIndexedTriangle({7, 2, 2}, {5, 2, 0}, {6, 2, 1}, matIdx);
        // BOTTOM (-Z)
        mesh.addIndexedTriangle({6, 5, 7}, {5, 5, 8}, {1, 5, 9}, matIdx) //
            .addIndexedTriangle({6, 5, 7}, {1, 5, 9}, {2, 5, 6}, matIdx);
    }

    Scene::Scene(std::pmr::memory_resource* memory) : geometry{memory}, instances{memory}, m_memory{memory} {}

    BufferSpecification Buffers::specFromScene(Scene const& scene)
    {
        BufferSpecification buf{};
        for (auto const& mesh : scene.geometry)
        {
            buf.numVerts += mesh->positionCount();
            buf.numNormals += mesh->normalCount();
            buf.numUvs += mesh->uvCount();
            buf.numIndices += mesh->triCount();
        }

        buf.numMeshes    = scene.geometry.size();
        buf.numInstances = scene.instances.size();

        return buf;
    }

    Buffers::Buffers(Scene const& scene, std::pmr::memory_resource* memory) : Buffers(specFromScene(scene), memory) {}

    Buffers::Buffers(BufferSpecification const& cap, std::pmr::memory_resource* memory) :
    verts{makeUniqueRef<Point3f[]>(memory, cap.numVerts, alignment)},
    normals{makeUniqueRef<OctahedralNorm[]>(memory, cap.numNormals, alignment)},
    uvs{makeUniqueRef<Point2f[]>(memory, cap.numUvs, alignment)},
    indices{makeUniqueRef<IndexedTri[]>(memory, cap.numIndices, alignment)},
    instances{makeUniqueRef<Instance[]>(memory, cap.numInstances, alignment)},
    primOffset{makeUniqueRef<GeoOffsets[]>(memory, cap.numMeshes, alignment)},
    m_memory{memory},
    m_curr{},
    m_cap{cap}
    {
    }

    static constexpr bool operator==(BufferSpecification const& a, BufferSpecification const& b)
    {
        return a.numVerts == b.numVerts && a.numNormals == b.numNormals && a.numUvs == b.numUvs &&
               a.numIndices == b.numIndices && a.numMeshes == b.numMeshes && a.numInstances == b.numInstances;
    }

    bool Buffers::copySceneToBuffers(Scene const& scene)
    {
        assert(m_cap == specFromScene(scene) && "Inconsistent specification");

        // TODO if Scene in host memory and scene in device memory do something else
        // TODO parallel version?
        GeoOffsets runningOff{};
        for (size_t meshIdx = 0; meshIdx < m_cap.numMeshes; ++meshIdx)
        {
            TriangleMesh const& mesh = *scene.geometry[meshIdx];

            // save offsets
            primOffset[meshIdx] = runningOff;

            // save vertices (unchanged)
            for (size_t vertIdx = 0; vertIdx < mesh.positionCount(); ++vertIdx)
                verts[runningOff.positionIdx + vertIdx] = mesh.getPosition(vertIdx);

            // save normals
            for (size_t normIdx = 0; normIdx < mesh.normalCount(); ++normIdx)
                normals[runningOff.normalIdx + normIdx] = octaFromNorm(mesh.getNormal(normIdx));

            // save uvs
            for (size_t uvIdx = 0; uvIdx < mesh.uvCount(); ++uvIdx)
                uvs[runningOff.uvIdx + uvIdx] = mesh.getUV(uvIdx);

            // save _globalized_ indices
            for (size_t localIdx = 0; localIdx < mesh.triCount(); ++localIdx)
                indices[runningOff.indexOff + localIdx] = mesh.getIndexedTri(localIdx);

            runningOff.positionIdx += mesh.positionCount();
            runningOff.normalIdx += mesh.normalCount();
            runningOff.uvIdx += mesh.uvCount();
            runningOff.indexOff += mesh.triCount();
            ++m_curr.numMeshes;
        }

        for (size_t instanceIdx = 0; instanceIdx < m_cap.numInstances; ++instanceIdx)
        {
            instances[instanceIdx] = *scene.instances[instanceIdx];
            ++m_curr.numInstances;
        }

        assert(m_cap == m_curr && "Cap config and current config not equal");
        return true;
    }
} // namespace dmt