#pragma once

#include "core/core-macros.h"
#include "core/core-cudautils-cpubuild.h"
#include "core/core-math.h"
#include "core/core-material.h"
#include "core/core-light.h"

#include "platform/platform-memory.h"

namespace dmt {
    struct DMT_CORE_API VertexIndex
    {
        size_t positionIdx;
        size_t normalIdx;
        size_t uvIdx;
    };
    static_assert(std::is_trivial_v<VertexIndex> && std::is_standard_layout_v<VertexIndex>);

    struct DMT_CORE_API GeoOffsets
    {
        size_t positionIdx;
        size_t normalIdx;
        size_t uvIdx;
        size_t indexOff;
    };
    static_assert(std::is_trivial_v<GeoOffsets> && std::is_standard_layout_v<GeoOffsets>);

    struct DMT_CORE_API IndexedTri
    {
        IndexedTri(VertexIndex v0, VertexIndex v1, VertexIndex v2, int32_t matIdx) : v{v0, v1, v2}, matIdx{matIdx} {}

        VertexIndex v[3];
        int32_t     matIdx;

        VertexIndex&       operator[](int i) { return v[i]; }
        VertexIndex const& operator[](int i) const { return v[i]; }
    };

    class TriangleMesh
    {
    public: // primitive creation
        DMT_CORE_API static void unitCube(TriangleMesh& mesh, int32_t matIdx = -1);

    public:
        DMT_CORE_API TriangleMesh(size_t cap = 256, std::pmr::memory_resource* memory = std::pmr::get_default_resource());
        TriangleMesh(TriangleMesh const&)            = delete;
        TriangleMesh& operator=(TriangleMesh const&) = delete;

        DMT_CORE_API TriangleMesh& addPosition(Point3f p);
        DMT_CORE_API TriangleMesh& addNormal(Normal3f n);
        DMT_CORE_API TriangleMesh& addUV(Point2f uv);
        DMT_CORE_API TriangleMesh& addIndexedTriangle(VertexIndex i0, VertexIndex i1, VertexIndex i2, int32_t matIdx);

        DMT_CORE_API Point3f    getPosition(size_t idx) const;
        DMT_CORE_API uint32_t   getPositionSize();
        DMT_CORE_API Normal3f   getNormal(size_t idx) const;
        uint32_t                getNormalSize();
        DMT_CORE_API Point2f    getUV(size_t idx) const;
        DMT_CORE_API IndexedTri getIndexedTri(size_t idx) const;

        DMT_CORE_API size_t positionCount() const { return m_positions.size(); }
        DMT_CORE_API size_t normalCount() const { return m_normals.size(); }
        DMT_CORE_API size_t uvCount() const { return m_uvs.size(); }
        DMT_CORE_API size_t triCount() const { return m_tris.size(); }

        DMT_CORE_API Bounds3f transformedBounds(float const affine[12]) const;
        DMT_CORE_API Bounds3f transformedBounds(Transform const& t) const;

        DMT_CORE_API bool checkPosition(Point3f p, uint32_t& idx);
        DMT_CORE_API bool checkNormal(Point3f p, uint32_t& idx);

    private:
        std::pmr::vector<Point3f>    m_positions;
        std::pmr::vector<Normal3f>   m_normals;
        std::pmr::vector<Point2f>    m_uvs;
        std::pmr::vector<IndexedTri> m_tris;
        // TODO per face shader assignment
    };

    // TODO ideas for SOA Layout conversion: Compact normals in octahedral encoding, compact into half uvs
    // TODO move elsewhere
    struct DMT_CORE_API Instance
    {
        size_t   meshIdx;
        float    affineTransform[12]; // column major
        Bounds3f bounds;

        // TODO remove
        RGB color;
    };

    struct DMT_CORE_API TriBufferSpan
    {
        size_t triFirst;
        size_t triCount;
    };

    /// Takes after `Scene` from cycles
    class Scene
    {
    public:
        DMT_CORE_API explicit Scene(std::pmr::memory_resource* memory = std::pmr::get_default_resource());

        Scene(Scene const&)                = delete;
        Scene(Scene&&) noexcept            = delete;
        Scene& operator=(Scene const&)     = delete;
        Scene& operator=(Scene&&) noexcept = delete;

        std::pmr::vector<UniqueRef<TriangleMesh>> geometry;
        std::pmr::vector<UniqueRef<Instance>>     instances;
        std::pmr::vector<SurfaceMaterial>         materials;
        std::pmr::vector<Light>                   lights;

    private:
        std::pmr::memory_resource* m_memory;
    };

    struct BufferSpecification
    {
        size_t numVerts;
        size_t numNormals;
        size_t numUvs;
        size_t numIndices;
        size_t numMeshes;
        size_t numInstances;
    };
    static_assert(std::is_trivial_v<BufferSpecification> && std::is_standard_layout_v<BufferSpecification>);

    /// NOTE: All buffers should be 16 bytes aligned
    /// Takes after `DeviceScene` from cycles -> if `std::pmr::memory_resource` allocates GPU memory, you cannot access it from host code!
    class Buffers
    {
    public:
        static constexpr size_t alignment = 16;

        DMT_CORE_API Buffers(Scene const& scene, std::pmr::memory_resource* memory = std::pmr::get_default_resource());
        Buffers(Buffers const&)                = delete;
        Buffers(Buffers&&) noexcept            = delete;
        Buffers& operator=(Buffers const&)     = delete;
        Buffers& operator=(Buffers&&) noexcept = delete;

        DMT_CORE_API bool copySceneToBuffers(Scene const& scene);

        // Per vert: size = number of vertices
        UniqueRef<Point3f[]>        verts;
        UniqueRef<OctahedralNorm[]> normals;
        UniqueRef<Point2f[]>        uvs;

        // Per face: size = number of triangles
        UniqueRef<IndexedTri[]> indices;
        // TODO per face shader assignment

        // per mesh instance
        UniqueRef<Instance[]> instances;

        // per mesh
        UniqueRef<GeoOffsets[]> primOffset;

    private:
        Buffers(BufferSpecification const& cap, std::pmr::memory_resource* memory);
        static BufferSpecification specFromScene(Scene const& scene);

        std::pmr::memory_resource* m_memory;
        BufferSpecification        m_curr;
        BufferSpecification        m_cap;
    };
} // namespace dmt