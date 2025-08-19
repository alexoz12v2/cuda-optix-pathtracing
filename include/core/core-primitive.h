#pragma once

#include "core/core-macros.h"
#include "core/core-math.h"

#include "cudautils/cudautils-vecmath.h"
#include "cudautils/cudautils-color.h"

#include "core/core-trianglemesh.h"


// TODO remove
#include <random>

namespace dmt {
    struct Intersection
    {
        Point3f  p;
        Point2f  uv;
        Vector3f ng;
        Vector3f dpdu;
        Vector3f dpdv;
        float    t;
        bool     hit;
        RGB      color;
    };
    static_assert(std::is_standard_layout_v<Intersection> && std::is_trivial_v<Intersection>);

    class DMT_CORE_API DMT_INTERFACE Primitive
    {
    public:
        virtual ~Primitive() {};

        virtual Bounds3f     bounds() const                              = 0;
        virtual Intersection intersect(Ray const& ray, float tMax) const = 0;
    };

    // TODO: Indexed variants
    class DMT_CORE_API Triangle : public Primitive
    {
    public:
        // TODO remove
        using Primitive::Primitive;
        Bounds3f     bounds() const override;
        Intersection intersect(Ray const& ray, float tMax) const override;

        TriangleData tri;
    };

    class DMT_CORE_API Triangles2 : public Primitive
    {
    public:
        Bounds3f     bounds() const override;
        Intersection intersect(Ray const& ray, float tMax) const override;

        static constexpr int32_t numTriangles = 2;

        float xs[3 * numTriangles], ys[3 * numTriangles], zs[3 * numTriangles];
        // TODO remove
        RGB colors[numTriangles];
    };

    class DMT_CORE_API Triangles4 : public Primitive
    {
    public:
        static constexpr int32_t numTriangles = 4;

        Bounds3f     bounds() const override;
        Intersection intersect(Ray const& ray, float tMax) const override;

        float xs[3 * numTriangles], ys[3 * numTriangles], zs[3 * numTriangles];
        // TODO remove
        RGB colors[numTriangles];
    };

    class DMT_CORE_API Triangles8 : public Primitive
    {
    public:
        static constexpr int32_t numTriangles = 8;

        Bounds3f     bounds() const override;
        Intersection intersect(Ray const& ray, float tMax) const override;

        // x_triangle0_vertex0 | x_triangle0_vertex1 | x_triangle0_vertex2 | x_triangle1_vertex0 | ...
        float xs[3 * numTriangles];
        float ys[3 * numTriangles];
        float zs[3 * numTriangles];
        // TODO remove
        RGB colors[numTriangles];
    };

    class DMT_CORE_API TriangleIndexedBase : public Primitive
    {
    public:
        Scene const* scene;
        size_t       instanceIdx;

        Vector3f normalFromIndex(size_t tri) const;
        Point2f  uvFromIndex(size_t tri, float u, float v) const;
        void     compute_dpdu_dpdv(size_t triIdx, Vector3f* dpdu, Vector3f* dpdv) const;

        std::tuple<Point3f, Point3f, Point3f> worldSpacePts(size_t _triIdx) const;
    };

    class DMT_CORE_API TriangleIndexed : public TriangleIndexedBase
    {
    public:
        Bounds3f     bounds() const override;
        Intersection intersect(Ray const& ray, float tMax) const override;

        size_t triIdx;
    };

    class DMT_CORE_API TrianglesIndexed2 : public TriangleIndexedBase
    {
    public:
        Bounds3f     bounds() const override;
        Intersection intersect(Ray const& ray, float tMax) const override;

        size_t triIdxs[2];
    };

    class DMT_CORE_API TrianglesIndexed4 : public TriangleIndexedBase
    {
    public:
        Bounds3f     bounds() const override;
        Intersection intersect(Ray const& ray, float tMax) const override;

        size_t triIdxs[4];
    };

    class DMT_CORE_API TrianglesIndexed8 : public TriangleIndexedBase
    {
    public:
        Bounds3f     bounds() const override;
        Intersection intersect(Ray const& ray, float tMax) const override;

        size_t triIdxs[8];
    };
} // namespace dmt

namespace dmt::triangle {
    struct DMT_CORE_API Triisect
    {
        static constexpr float tol = 1e-7f; // or 6

        float    u;
        float    v;
        float    w;
        float    t;
        uint32_t index;

        operator bool() const { return !fl::isInfOrNaN(u); }

        static constexpr Triisect nothing()
        {
            return {.u = fl::infinity(), .v = fl::infinity(), .w = fl::infinity(), .t = fl::infinity(), .index = 0};
        }
    };


    Intersection DMT_FASTCALL DMT_CORE_API fromTrisect(Triisect trisect, Ray const& ray, RGB color, Point2f uv = {0, 0});

    Triisect DMT_FASTCALL DMT_CORE_API intersect(Ray const& ray, float tMax, Point3f v0, Point3f v1, Point3f v2, uint32_t index);

    // 0x3 -> intersect2, 0xf -> intersect4
    Triisect DMT_FASTCALL DMT_CORE_API
        intersect4(Ray const& ray, float tMax, Point3f const* v0s, Point3f const* v1s, Point3f const* v2s, int32_t mask);

    Triisect DMT_FASTCALL DMT_CORE_API
        intersect8(Ray const& ray, float tMax, Point3f const* v0s, Point3f const* v1s, Point3f const* v2s);
} // namespace dmt::triangle