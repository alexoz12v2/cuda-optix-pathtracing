#include "core-primitive.h"

#if !defined(DMT_ARCH_X86_64)
    #error "Not correct CPU Arch"
#endif

namespace dmt {
    // -- bounds --
    Bounds3f Triangle::bounds() const
    {
        Point3f p0{{tri.v0.x, tri.v0.y, tri.v0.z}};
        Point3f p1{{tri.v1.x, tri.v1.y, tri.v1.z}};
        Point3f p2{{tri.v2.x, tri.v2.y, tri.v2.z}};
        return bbUnion(bbUnion(Bounds3f{p0, p0}, Bounds3f{p1, p1}), Bounds3f{p2, p2});
    }

    Bounds3f Triangles2::bounds() const
    {
        // Load all 6 floats (3 vertices * 2 triangles) for x, y, z
        __m128 x0 = _mm_loadu_ps(xs);     // xs[0..3]
        __m128 x1 = _mm_loadu_ps(xs + 2); // xs[2..5] overlaps last part of x0

        __m128 y0 = _mm_loadu_ps(ys);
        __m128 y1 = _mm_loadu_ps(ys + 2);

        __m128 z0 = _mm_loadu_ps(zs);
        __m128 z1 = _mm_loadu_ps(zs + 2);

        __m128 xMin = _mm_min_ps(x0, x1);
        __m128 xMax = _mm_max_ps(x0, x1);

        __m128 yMin = _mm_min_ps(y0, y1);
        __m128 yMax = _mm_max_ps(y0, y1);

        __m128 zMin = _mm_min_ps(z0, z1);
        __m128 zMax = _mm_max_ps(z0, z1);

        float x_min = arch::hmin_ps(xMin);
        float x_max = arch::hmax_ps(xMax);
        float y_min = arch::hmin_ps(yMin);
        float y_max = arch::hmax_ps(yMax);
        float z_min = arch::hmin_ps(zMin);
        float z_max = arch::hmax_ps(zMax);

        return Bounds3f{Point3f{{x_min, y_min, z_min}}, Point3f{{x_max, y_max, z_max}}};
    }

    Bounds3f Triangles4::bounds() const
    {
        // Load all 12 floats (3 vertices * 4 triangles) for x, y, z
        __m128 x0 = _mm_loadu_ps(xs);     // xs[0..3]
        __m128 x1 = _mm_loadu_ps(xs + 4); // xs[4..7]
        __m128 x2 = _mm_loadu_ps(xs + 8); // xs[8..11]

        __m128 y0 = _mm_loadu_ps(ys);
        __m128 y1 = _mm_loadu_ps(ys + 4);
        __m128 y2 = _mm_loadu_ps(ys + 8);

        __m128 z0 = _mm_loadu_ps(zs);
        __m128 z1 = _mm_loadu_ps(zs + 4);
        __m128 z2 = _mm_loadu_ps(zs + 8);

        __m128 xMin = _mm_min_ps(_mm_min_ps(x0, x1), x2);
        __m128 xMax = _mm_max_ps(_mm_max_ps(x0, x1), x2);

        __m128 yMin = _mm_min_ps(_mm_min_ps(y0, y1), y2);
        __m128 yMax = _mm_max_ps(_mm_max_ps(y0, y1), y2);

        __m128 zMin = _mm_min_ps(_mm_min_ps(z0, z1), z2);
        __m128 zMax = _mm_max_ps(_mm_max_ps(z0, z1), z2);

        float x_min = arch::hmin_ps(xMin);
        float x_max = arch::hmax_ps(xMax);
        float y_min = arch::hmin_ps(yMin);
        float y_max = arch::hmax_ps(yMax);
        float z_min = arch::hmin_ps(zMin);
        float z_max = arch::hmax_ps(zMax);

        return Bounds3f{Point3f{{x_min, y_min, z_min}}, Point3f{{x_max, y_max, z_max}}};
    }

    Bounds3f Triangles8::bounds() const
    {
        // Load all 24 floats (3 * 8) for x, y, z
        __m256 x0 = _mm256_loadu_ps(xs);      // xs[0]..xs[7]
        __m256 x1 = _mm256_loadu_ps(xs + 8);  // xs[8]..xs[15]
        __m256 x2 = _mm256_loadu_ps(xs + 16); // xs[16]..xs[23]

        __m256 y0 = _mm256_loadu_ps(ys);
        __m256 y1 = _mm256_loadu_ps(ys + 8);
        __m256 y2 = _mm256_loadu_ps(ys + 16);

        __m256 z0 = _mm256_loadu_ps(zs);
        __m256 z1 = _mm256_loadu_ps(zs + 8);
        __m256 z2 = _mm256_loadu_ps(zs + 16);

        // Reduce min/max across the 24 values using AVX2
        __m256 xMin = _mm256_min_ps(_mm256_min_ps(x0, x1), x2);
        __m256 xMax = _mm256_max_ps(_mm256_max_ps(x0, x1), x2);

        __m256 yMin = _mm256_min_ps(_mm256_min_ps(y0, y1), y2);
        __m256 yMax = _mm256_max_ps(_mm256_max_ps(y0, y1), y2);

        __m256 zMin = _mm256_min_ps(_mm256_min_ps(z0, z1), z2);
        __m256 zMax = _mm256_max_ps(_mm256_max_ps(z0, z1), z2);

        // Final horizontal reduction to scalars
        float x_min = arch::hmin_ps(xMin);
        float x_max = arch::hmax_ps(xMax);
        float y_min = arch::hmin_ps(yMin);
        float y_max = arch::hmax_ps(yMax);
        float z_min = arch::hmin_ps(zMin);
        float z_max = arch::hmax_ps(zMax);

        return Bounds3f{Point3f{{x_min, y_min, z_min}}, Point3f{{x_max, y_max, z_max}}};
    }

    // -- intersect --
    // TODO add clamp(u, 0, 1) and clamp(v, 0, 1) if you add them to the intersection struct for texture sampling (moller trumbore)
    Intersection Triangle::intersect(Ray const& ray, float tMax) const
    {
#if defined DMT_MOLLER_TRUMBORE
        // moller trumbore algorithm https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
        Vector3f const         e1  = tri.v1 - tri.v0;
        Vector3f const         e2  = tri.v2 - tri.v0;
        static constexpr float tol = 1e-7f; // or 6
        // det[a, b, c] = a . (b x c) -> all cyclic permutations + cross is anticommutative
        Vector3f const h = cross(ray.d, e2);
        float const    a = dot(e1, h);

        if (a > -tol && a < tol) // if determinant is zero, then ray || triangle
            return {.p = {{}}, .t = 0.f, .hit = false};

        float const    f = fl::rcp(a); // inverse for cramer's rule
        Vector3f const s = ray.o - tri.v0;
        float const    u = f * dot(s, h); // barycentric coord 0
        if (u < -tol || u > 1.f + tol)
            return {.p = {{}}, .t = 0.f, .hit = false};

        Vector3f    q = cross(s, e1);
        float const v = f * dot(ray.d, q); // barycentric coord 1
        if (v < -tol || (u + v) > 1.f + tol)
            return {.p = {{}}, .t = 0.f, .hit = false};

        float const t = f * dot(e2, q);
        if (t < -tol || t > tMax)
            return {.p = {{}}, .t = t, .hit = false};

        return {.p = ray.o + t * ray.d, .t = t, .hit = true};
#else
        // Woop's watertight algorithm https://jcgt.org/published/0002/01/05/paper.pdf
        // calculate dimension where the ray direction is maximal
        int32_t kz = maxComponentIndex(ray.d);
        int32_t kx = kz + 1;
        if (kx == 3)
            kx = 0;
        int ky = kx + 1;
        if (ky == 3)
            ky = 0;
        // swap kx and ky dimension to preserve winding direction of triangles
        if (ray.d[kz] < 0.0f)
            std::swap(kx, ky);

        // calculate shear constants
        float const Sx = ray.d[kx] / ray.d[kz];
        float const Sy = ray.d[ky] / ray.d[kz];
        float const Sz = fl::rcp(ray.d[kz]);

        // calculate vertices relative to ray origin
        Vector3f const A = tri.v0 - ray.o;
        Vector3f const B = tri.v1 - ray.o;
        Vector3f const C = tri.v2 - ray.o;

        // perfor shear and scale of vertices
        float const Ax = A[kx] - Sx * A[kz];
        float const Ay = A[ky] - Sy * A[kz];
        float const Bx = B[kx] - Sx * B[kz];
        float const By = B[ky] - Sy * B[kz];
        float const Cx = C[kx] - Sx * C[kz];
        float const Cy = C[ky] - Sy * C[kz];

        // calculate scaled barycentric coordinates
        float U = Cx * By - Cy * Bx;
        float V = Ax * Cy - Ay * Cx;
        float W = Bx * Ay - By * Ax;

        // fallback to test against edges using double precision
        if (fl::nearZero(U) || fl::nearZero(V) || fl::nearZero(W))
        {
            double CxBy = (double)Cx * (double)By;
            double CyBx = (double)Cy * (double)Bx;
            U           = (float)(CxBy - CyBx);
            double AxCy = (double)Ax * (double)Cy;
            double AyCx = (double)Ay * (double)Cx;
            V           = (float)(AxCy - AyCx);
            double BxAy = (double)Bx * (double)Ay;
            double ByAx = (double)By * (double)Ax;
            W           = (float)(BxAy - ByAx);
        }

        // Perform edge tests. Moving this test before and at the end of the previous conditional gives higher performance
    #ifdef BACKFACE_CULLING
        if (U < 0.0f || V < 0.0f || W < 0.0f)
            return {.p = {}, .t = 0, .hit = false};
    #else
        if ((U < 0.0f || V < 0.0f || W < 0.0f) && (U > 0.0f || V > 0.0f || W > 0.0f))
            return {.p = {}, .t = 0, .hit = false};
    #endif

        // calculate determinant
        float const det = U + V + W;
        if (fl::nearZero(det))
            return {.p = {}, .t = 0, .hit = false};

        // Calculate scaled z-coordinates of vertices and use them to calculate the hit distance
        float const Az = Sz * A[kz];
        float const Bz = Sz * B[kz];
        float const Cz = Sz * C[kz];
        float const T  = U * Az + V * Bz + W * Cz;
    #ifdef BACKFACE_CULLING
        if (T < 0.0f || T > tMax * det)
            return {.p = {}, .t = 0, .hit = false};
    #else
        int det_sign = fl::signBit(det);
        if (fl::xorf(T, det_sign) < 0.0f || fl::xorf(T, det_sign) > tMax * fl::xorf(det, det_sign))
            return {.p = {}, .t = 0, .hit = false};
    #endif
        // normalize U, V, W, and T
        float const rcpDet = 1.0f / det;
        float const u      = U * rcpDet;
        float const v      = V * rcpDet;
        float const w      = W * rcpDet;
        float const t      = T * rcpDet;

        return {.p = ray.o + t * ray.d, .t = t, .hit = true};
#endif
    }

    Intersection Triangles2::intersect(Ray const& ray, float tMax) const
    {
        // unpack origin, direction and constants
        __m128 const ox = _mm_set1_ps(ray.o.x);
        __m128 const oy = _mm_set1_ps(ray.o.y);
        __m128 const oz = _mm_set1_ps(ray.o.z);

        __m128 const dx = _mm_set1_ps(ray.d.x);
        __m128 const dy = _mm_set1_ps(ray.d.y);
        __m128 const dz = _mm_set1_ps(ray.d.z);

        static constexpr float eps1 = 1e-7f;

        __m128 const eps   = _mm_set1_ps(eps1);
        __m128 const tMaxV = _mm_set1_ps(tMax);
        __m128 const inf   = _mm_set1_ps(fl::infinity());
        __m128 const zero  = _mm_set1_ps(-eps1);
        __m128 const one   = _mm_set1_ps(1.f + eps1);

        // vertices. x1 - x0 1st tri, x4 - x3 2nd tri, hence interlaced format. either loadu and transpose or set
        // I need only the first lane, the second is duplicated and ignored
        __m128 const v0x = _mm_set_ps(xs[3], xs[0], xs[3], xs[0]);
        __m128 const v1x = _mm_set_ps(xs[4], xs[1], xs[4], xs[1]);
        __m128 const v2x = _mm_set_ps(xs[5], xs[2], xs[5], xs[2]);

        __m128 const v0y = _mm_set_ps(ys[3], ys[0], ys[3], ys[0]);
        __m128 const v1y = _mm_set_ps(ys[4], ys[1], ys[4], ys[1]);
        __m128 const v2y = _mm_set_ps(ys[5], ys[2], ys[5], ys[2]);

        __m128 const v0z = _mm_set_ps(zs[3], zs[0], zs[3], zs[0]);
        __m128 const v1z = _mm_set_ps(zs[4], zs[1], zs[4], zs[1]);
        __m128 const v2z = _mm_set_ps(zs[5], zs[2], zs[5], zs[2]);

        // e1 = v1 - v0, e2 = v2 - v0
        __m128 const e1x = _mm_sub_ps(v1x, v0x);
        __m128 const e1y = _mm_sub_ps(v1y, v0y);
        __m128 const e1z = _mm_sub_ps(v1z, v0z);

        __m128 const e2x = _mm_sub_ps(v2x, v0x);
        __m128 const e2y = _mm_sub_ps(v2y, v0y);
        __m128 const e2z = _mm_sub_ps(v2z, v0z);

        // h = cross(d, e2)
        __m128 const hx = _mm_sub_ps(_mm_mul_ps(dy, e2z), _mm_mul_ps(dz, e2y));
        __m128 const hy = _mm_sub_ps(_mm_mul_ps(dz, e2x), _mm_mul_ps(dx, e2z));
        __m128 const hz = _mm_sub_ps(_mm_mul_ps(dx, e2y), _mm_mul_ps(dy, e2x));

        // determinant a = dot(e1, h)
        __m128 const a = _mm_add_ps(_mm_add_ps(_mm_mul_ps(e1x, hx), _mm_mul_ps(e1y, hy)), _mm_mul_ps(e1z, hz));

        // if (std::abs(a) < 1e-8f)
        __m128 const mask = _mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), a), eps);
        __m128 const invA = _mm_rcp_ps(a);

        // s = o - v0
        __m128 const sx = _mm_sub_ps(ox, v0x);
        __m128 const sy = _mm_sub_ps(oy, v0y);
        __m128 const sz = _mm_sub_ps(oz, v0z);

        // first barycentric coord with Cramer's: u = invA * dot(s, h); check if in [0, 1]
        __m128 const u     = _mm_mul_ps(invA,
                                    _mm_add_ps(_mm_add_ps(_mm_mul_ps(sx, hx), _mm_mul_ps(sy, hy)), _mm_mul_ps(sz, hz)));
        __m128 const uMask = _mm_and_ps(_mm_cmpge_ps(u, zero), _mm_cmple_ps(u, one));

        // q = cross(s, e1)
        __m128 const qx = _mm_sub_ps(_mm_mul_ps(sy, e1z), _mm_mul_ps(sz, e1y));
        __m128 const qy = _mm_sub_ps(_mm_mul_ps(sz, e1x), _mm_mul_ps(sx, e1z));
        __m128 const qz = _mm_sub_ps(_mm_mul_ps(sx, e1y), _mm_mul_ps(sy, e1x));

        // second barycentric coord with Cramer's: v = invA * dot(ray.d, q); check if in [0, 1]
        __m128 const v     = _mm_mul_ps(invA,
                                    _mm_add_ps(_mm_add_ps(_mm_mul_ps(dx, qx), _mm_mul_ps(dy, qy)), _mm_mul_ps(dz, qz)));
        __m128 const vMask = _mm_and_ps(_mm_cmpge_ps(v, zero), _mm_cmple_ps(_mm_add_ps(v, u), one));

        // invA * dot(e2, q) ; should be greater than some tolerance and less than tMax
        __m128 const t     = _mm_mul_ps(invA,
                                    _mm_add_ps(_mm_add_ps(_mm_mul_ps(e2x, qx), _mm_mul_ps(e2y, qy)), _mm_mul_ps(e2z, qz)));
        __m128 const tMask = _mm_and_ps(_mm_cmpgt_ps(t, eps), _mm_cmplt_ps(t, tMaxV));

        // and all masks and, where mask on put t, where mask off put infinity
        __m128 const finalMask = _mm_and_ps(_mm_and_ps(mask, uMask), _mm_and_ps(vMask, tMask));
        __m128 const tResult   = _mm_or_ps(_mm_and_ps(finalMask, t), _mm_andnot_ps(finalMask, inf));

        alignas(16) float tVals[4];
        _mm_store_ps(tVals, tResult);

        int32_t bestIndex = -1;
        float   bestT     = tMax;
        for (int32_t i = 0; i < 2; ++i)
        {
            if (tVals[i] < bestT)
            {
                bestIndex = i;
                bestT     = tVals[i];
            }
        }

        if (bestIndex >= 0)
            return {.p = ray.o + bestT * ray.d, .t = bestT, .hit = true};
        else
            return {.p = {{}}, .t = 0.f, .hit = false};
    }

    Intersection Triangles4::intersect(Ray const& ray, float tMax) const
    { //
        // unpack origin, direction and constants
        __m128 const ox = _mm_set1_ps(ray.o.x);
        __m128 const oy = _mm_set1_ps(ray.o.y);
        __m128 const oz = _mm_set1_ps(ray.o.z);

        __m128 const dx = _mm_set1_ps(ray.d.x);
        __m128 const dy = _mm_set1_ps(ray.d.y);
        __m128 const dz = _mm_set1_ps(ray.d.z);

        static constexpr float eps1 = 1e-7f;

        __m128 const eps   = _mm_set1_ps(eps1);
        __m128 const tMaxV = _mm_set1_ps(tMax);
        __m128 const inf   = _mm_set1_ps(fl::infinity());
        __m128 const zero  = _mm_set1_ps(-eps1);
        __m128 const one   = _mm_set1_ps(1.f + eps1);

        // vertices. x1 - x0 1st tri, x4 - x3 2nd tri, hence interlaced format. either loadu and transpose or set
        __m128 const v0x = _mm_set_ps(xs[3], xs[0], xs[9], xs[6]);
        __m128 const v1x = _mm_set_ps(xs[4], xs[1], xs[10], xs[7]);
        __m128 const v2x = _mm_set_ps(xs[5], xs[2], xs[11], xs[8]);

        __m128 const v0y = _mm_set_ps(ys[3], ys[0], ys[9], ys[6]);
        __m128 const v1y = _mm_set_ps(ys[4], ys[1], ys[10], ys[7]);
        __m128 const v2y = _mm_set_ps(ys[5], ys[2], ys[11], ys[8]);

        __m128 const v0z = _mm_set_ps(zs[3], zs[0], zs[9], zs[6]);
        __m128 const v1z = _mm_set_ps(zs[4], zs[1], zs[10], zs[7]);
        __m128 const v2z = _mm_set_ps(zs[5], zs[2], zs[11], zs[8]);

        // e1 = v1 - v0, e2 = v2 - v0
        __m128 const e1x = _mm_sub_ps(v1x, v0x);
        __m128 const e1y = _mm_sub_ps(v1y, v0y);
        __m128 const e1z = _mm_sub_ps(v1z, v0z);

        __m128 const e2x = _mm_sub_ps(v2x, v0x);
        __m128 const e2y = _mm_sub_ps(v2y, v0y);
        __m128 const e2z = _mm_sub_ps(v2z, v0z);

        // h = cross(d, e2)
        __m128 const hx = _mm_sub_ps(_mm_mul_ps(dy, e2z), _mm_mul_ps(dz, e2y));
        __m128 const hy = _mm_sub_ps(_mm_mul_ps(dz, e2x), _mm_mul_ps(dx, e2z));
        __m128 const hz = _mm_sub_ps(_mm_mul_ps(dx, e2y), _mm_mul_ps(dy, e2x));

        // determinant a = dot(e1, h)
        __m128 const a = _mm_add_ps(_mm_add_ps(_mm_mul_ps(e1x, hx), _mm_mul_ps(e1y, hy)), _mm_mul_ps(e1z, hz));

        // if (std::abs(a) < 1e-8f)
        __m128 const mask = _mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), a), eps);
        __m128 const invA = _mm_rcp_ps(a);

        // s = o - v0
        __m128 const sx = _mm_sub_ps(ox, v0x);
        __m128 const sy = _mm_sub_ps(oy, v0y);
        __m128 const sz = _mm_sub_ps(oz, v0z);

        // first barycentric coord with Cramer's: u = invA * dot(s, h); check if in [0, 1]
        __m128 const u     = _mm_mul_ps(invA,
                                    _mm_add_ps(_mm_add_ps(_mm_mul_ps(sx, hx), _mm_mul_ps(sy, hy)), _mm_mul_ps(sz, hz)));
        __m128 const uMask = _mm_and_ps(_mm_cmpge_ps(u, zero), _mm_cmple_ps(u, one));

        // q = cross(s, e1)
        __m128 const qx = _mm_sub_ps(_mm_mul_ps(sy, e1z), _mm_mul_ps(sz, e1y));
        __m128 const qy = _mm_sub_ps(_mm_mul_ps(sz, e1x), _mm_mul_ps(sx, e1z));
        __m128 const qz = _mm_sub_ps(_mm_mul_ps(sx, e1y), _mm_mul_ps(sy, e1x));

        // second barycentric coord with Cramer's: v = invA * dot(ray.d, q); check if in [0, 1]
        __m128 const v     = _mm_mul_ps(invA,
                                    _mm_add_ps(_mm_add_ps(_mm_mul_ps(dx, qx), _mm_mul_ps(dy, qy)), _mm_mul_ps(dz, qz)));
        __m128 const vMask = _mm_and_ps(_mm_cmpge_ps(v, zero), _mm_cmple_ps(_mm_add_ps(v, u), one));

        // t = invA * dot(e2, q) ; should be greater than some tolerance and less than tMax
        __m128 const t     = _mm_mul_ps(invA,
                                    _mm_add_ps(_mm_add_ps(_mm_mul_ps(e2x, qx), _mm_mul_ps(e2y, qy)), _mm_mul_ps(e2z, qz)));
        __m128 const tMask = _mm_and_ps(_mm_cmpgt_ps(t, eps), _mm_cmplt_ps(t, tMaxV));

        // and all masks and, where mask on put t, where mask off put infinity
        __m128 const finalMask = _mm_and_ps(_mm_and_ps(mask, uMask), _mm_and_ps(vMask, tMask));
        __m128 const tResult   = _mm_or_ps(_mm_and_ps(finalMask, t), _mm_andnot_ps(finalMask, inf));

        alignas(16) float tVals[4];
        _mm_store_ps(tVals, tResult);

        int32_t bestIndex = -1;
        float   bestT     = tMax;
        for (int32_t i = 0; i < 4; ++i)
        {
            if (tVals[i] < bestT)
            {
                bestIndex = i;
                bestT     = tVals[i];
            }
        }

        if (bestIndex >= 0)
            return {.p = ray.o + bestT * ray.d, .t = bestT, .hit = true};
        else
            return {.p = {{}}, .t = 0.f, .hit = false};
    }

    Intersection Triangles8::intersect(Ray const& ray, float tMax) const
    {
        static constexpr int32_t FloatStride = sizeof(float);
        static constexpr float   eps1        = 1e-7;

        // index vector: [0, 3, 6, 9, 12, 15, 18, 21] for vertex0
        int32_t const v0Offsets[8] = {0, 3, 6, 9, 12, 15, 18, 21};
        int32_t const v1Offsets[8] = {1, 4, 7, 10, 13, 16, 19, 22};
        int32_t const v2Offsets[8] = {2, 5, 8, 11, 14, 17, 20, 23};

        __m256i const v0Index = _mm256_loadu_epi32(v0Offsets);
        __m256i const v1Index = _mm256_loadu_epi32(v1Offsets);
        __m256i const v2Index = _mm256_loadu_epi32(v2Offsets);

        // gather vertex components
        __m256 const v0x = _mm256_i32gather_ps(xs, v0Index, FloatStride);
        __m256 const v0y = _mm256_i32gather_ps(ys, v0Index, FloatStride);
        __m256 const v0z = _mm256_i32gather_ps(zs, v0Index, FloatStride);

        __m256 const v1x = _mm256_i32gather_ps(xs, v1Index, FloatStride);
        __m256 const v1y = _mm256_i32gather_ps(ys, v1Index, FloatStride);
        __m256 const v1z = _mm256_i32gather_ps(zs, v1Index, FloatStride);

        __m256 const v2x = _mm256_i32gather_ps(xs, v2Index, FloatStride);
        __m256 const v2y = _mm256_i32gather_ps(ys, v2Index, FloatStride);
        __m256 const v2z = _mm256_i32gather_ps(zs, v2Index, FloatStride);

        // ray origin and direction broadcasted
        __m256 const ox = _mm256_set1_ps(ray.o.x);
        __m256 const oy = _mm256_set1_ps(ray.o.y);
        __m256 const oz = _mm256_set1_ps(ray.o.z);

        __m256 const dx = _mm256_set1_ps(ray.d.x);
        __m256 const dy = _mm256_set1_ps(ray.d.y);
        __m256 const dz = _mm256_set1_ps(ray.d.z);

        // constants
        __m256 const eps   = _mm256_set1_ps(eps1);
        __m256 const zero  = _mm256_set1_ps(-eps1);
        __m256 const one   = _mm256_set1_ps(1.f + eps1);
        __m256 const tMaxV = _mm256_set1_ps(tMax);
        __m256 const inf   = _mm256_set1_ps(fl::infinity());

        // e1 = v1 - v0, e2 = v2 - v0
        __m256 const e1x = _mm256_sub_ps(v1x, v0x);
        __m256 const e1y = _mm256_sub_ps(v1y, v0y);
        __m256 const e1z = _mm256_sub_ps(v1z, v0z);

        __m256 const e2x = _mm256_sub_ps(v2x, v0x);
        __m256 const e2y = _mm256_sub_ps(v2y, v0y);
        __m256 const e2z = _mm256_sub_ps(v2z, v0z);

        // h = cross(d, e2)
        __m256 const hx = _mm256_sub_ps(_mm256_mul_ps(dy, e2z), _mm256_mul_ps(dz, e2y));
        __m256 const hy = _mm256_sub_ps(_mm256_mul_ps(dz, e2x), _mm256_mul_ps(dx, e2z));
        __m256 const hz = _mm256_sub_ps(_mm256_mul_ps(dx, e2y), _mm256_mul_ps(dy, e2x));

        // determinant a = dot(e1, h)
        __m256 const a = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(e1x, hx), _mm256_mul_ps(e1y, hy)),
                                       _mm256_mul_ps(e1z, hz));

        // if (std::abs(a) < 1e-8f)
        __m256 const mask = _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), a), eps, _CMP_GT_OQ);
        __m256 const invA = _mm256_rcp_ps(a);

        // s = o - v0
        __m256 const sx = _mm256_sub_ps(ox, v0x);
        __m256 const sy = _mm256_sub_ps(oy, v0y);
        __m256 const sz = _mm256_sub_ps(oz, v0z);

        // first barycentric coord with Cramer's: u = invA * dot(s, h); check if in [0, 1]
        __m256 const u     = _mm256_mul_ps(invA,
                                       _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(sx, hx), _mm256_mul_ps(sy, hy)),
                                                     _mm256_mul_ps(sz, hx)));
        __m256 const uMask = _mm256_and_ps(_mm256_cmp_ps(u, zero, _CMP_GE_OQ), _mm256_cmp_ps(u, one, _CMP_LE_OQ));

        // q = cross(s, e1)
        __m256 const qx = _mm256_sub_ps(_mm256_mul_ps(sy, e1z), _mm256_mul_ps(sz, e1y));
        __m256 const qy = _mm256_sub_ps(_mm256_mul_ps(sz, e1x), _mm256_mul_ps(sx, e1z));
        __m256 const qz = _mm256_sub_ps(_mm256_mul_ps(sx, e1y), _mm256_mul_ps(sy, e1x));

        // second barycentric coord with Cramer's: v = invA * dot(ray.d, q); check if in [0, 1]
        __m256 const v     = _mm256_mul_ps(invA,
                                       _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, qx), _mm256_mul_ps(dy, qy)),
                                                     _mm256_mul_ps(dz, qz)));
        __m256 const vMask = _mm256_and_ps(_mm256_cmp_ps(v, zero, _CMP_GE_OQ),
                                           _mm256_cmp_ps(_mm256_add_ps(v, u), one, _CMP_LE_OQ));

        // t = invA * dot(e2, q) ; should be greater than some tolerance and less than tMax
        __m256 const t     = _mm256_mul_ps(invA,
                                       _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(e2x, qx), _mm256_mul_ps(e2y, qy)),
                                                     _mm256_mul_ps(e2z, qz)));
        __m256 const tMask = _mm256_and_ps(_mm256_cmp_ps(t, eps, _CMP_GT_OQ), _mm256_cmp_ps(t, tMaxV, _CMP_LT_OQ));

        // and all masks and, where mask on put t, where mask off put infinity
        __m256 const finalMask = _mm256_and_ps(_mm256_and_ps(mask, uMask), _mm256_and_ps(vMask, tMask));
        __m256 const tResult   = _mm256_or_ps(_mm256_and_ps(finalMask, t), _mm256_andnot_ps(finalMask, inf));

        alignas(alignof(__m256)) float tVals[8];
        _mm256_store_ps(tVals, tResult);

        int32_t bestIndex = -1;
        float   bestT     = tMax;
        for (int32_t i = 0; i < 8; ++i)
        {
            if (tVals[i] < bestT)
            {
                bestIndex = i;
                bestT     = tVals[i];
            }
        }

        if (bestIndex >= 0)
            return {.p = ray.o + bestT * ray.d, .t = bestT, .hit = true};
        else
            return {.p = {{}}, .t = 0.f, .hit = false};
    }
} // namespace dmt