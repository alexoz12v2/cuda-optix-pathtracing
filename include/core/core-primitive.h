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
        Point3f p;
        float   t;
        bool    hit;
        // TODO remove
        RGB color;
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

    class DMT_CORE_API TriangleIndexed : public Primitive
    {
    public:
        Bounds3f     bounds() const override;
        Intersection intersect(Ray const& ray, float tMax) const override;

        Scene const* scene;
        size_t       instanceIdx;
        size_t       triIdx;

    private:
        std::tuple<Point3f, Point3f, Point3f> worldSpacePts() const;
    };
} // namespace dmt