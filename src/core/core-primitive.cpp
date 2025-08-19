#include "core-primitive.h"

#include "core-math.h"

#if !defined(DMT_ARCH_X86_64)
    #error "Not correct CPU Arch"
#endif

//#define DMT_MOLLER_TRUMBORE
#define DMT_BACKFACE_CULLING
//#define DMT_LAST_RESORT

namespace dmt {
    namespace triangle {
        /// TODO REMOVE AND PUT NORMAL BUFFER EXTRACTION
        //static Vector3f computeNormal(Point3f p0, Point3f p1, Point3f p2)
        //{
        //    Vector3f edge1 = p1 - p0;
        //    Vector3f edge2 = p2 - p0;
        //    return normalize(cross(edge1, edge2)); // Right-handed normal
        //}

        //static float extract_component_m128(__m128 vec, int index)
        //{
        //    float arr[4];
        //    _mm_storeu_ps(arr, vec); // unaligned store
        //    return arr[index];       // runtime index
        //}

        //static float extract_component_m256(__m256 vec, int index)
        //{
        //    float arr[8];
        //    _mm256_storeu_ps(arr, vec); // unaligned store
        //    return arr[index];          // runtime index
        //}

        //static Point3f getPoint(__m128 x, __m128 y, __m128 z, int index)
        //{
        //    return {extract_component_m128(x, index), extract_component_m128(y, index), extract_component_m128(z, index)};
        //}

        //static Point3f getPoint(__m256 x, __m256 y, __m256 z, int index)
        //{
        //    return {extract_component_m256(x, index), extract_component_m256(y, index), extract_component_m256(z, index)};
        //}

        static Bounds3f bounds(Point3f p0, Point3f p1, Point3f p2) { return bbUnion(makeBounds(p0, p1), p2); }

        static Bounds3f bounds2(float const xs[6], float const ys[6], float const zs[6])
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

            return makeBounds({x_min, y_min, z_min}, {x_max, y_max, z_max});
        }

        Bounds3f bounds4(float const xs[12], float const ys[12], float const zs[12])
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

            return makeBounds({x_min, y_min, z_min}, {x_max, y_max, z_max});
        }

        static Bounds3f bounds8(float const xs[24], float const ys[24], float const zs[24])
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

        Triisect DMT_FASTCALL
            intersect4(Ray const& ray, float tMax, float tMin, Point3f const* v0s, Point3f const* v1s, Point3f const* v2s, int32_t mask)
        {
#if defined DMT_MOLLER_TRUMBORE
            // constants
            __m128 const  tol     = _mm_set1_ps(Triisect::tol);
            __m128 const  mtol    = _mm_set1_ps(-Triisect::tol);
            __m128 const  one     = _mm_set1_ps(Triisect::tol + 1.f);
            __m128i const iOne    = _mm_set1_epi32(1);
            __m128 const  signBit = _mm_set1_ps(-0.f);
            __m128 const  tMaxv   = _mm_set1_ps(tMax);
            __m128 const  tMinv   = _mm_set1_ps(tMin);
            __m128 const  argMask = _mm_castsi128_ps(
                _mm_set_epi32((mask & 0x08) ? -1 : 0, (mask & 0x04) ? -1 : 0, (mask & 0x02) ? -1 : 0, (mask & 0x01) ? -1 : 0));

            // ray
            __m128 const dx = _mm_set1_ps(ray.d.x);
            __m128 const dy = _mm_set1_ps(ray.d.y);
            __m128 const dz = _mm_set1_ps(ray.d.z);

            __m128 const ox = _mm_set1_ps(ray.o.x);
            __m128 const oy = _mm_set1_ps(ray.o.y);
            __m128 const oz = _mm_set1_ps(ray.o.z);

            // v0, v1, v2
            __m128i vIndex = _mm_set_epi32(9, 6, 3, 0);

            __m128 const v0x = _mm_i32gather_ps(reinterpret_cast<float const*>(v0s), vIndex, sizeof(float));
            __m128 const v1x = _mm_i32gather_ps(reinterpret_cast<float const*>(v1s), vIndex, sizeof(float));
            __m128 const v2x = _mm_i32gather_ps(reinterpret_cast<float const*>(v2s), vIndex, sizeof(float));

            vIndex           = _mm_add_epi32(vIndex, iOne);
            __m128 const v0y = _mm_i32gather_ps(reinterpret_cast<float const*>(v0s), vIndex, sizeof(float));
            __m128 const v1y = _mm_i32gather_ps(reinterpret_cast<float const*>(v1s), vIndex, sizeof(float));
            __m128 const v2y = _mm_i32gather_ps(reinterpret_cast<float const*>(v2s), vIndex, sizeof(float));

            vIndex           = _mm_add_epi32(vIndex, iOne);
            __m128 const v0z = _mm_i32gather_ps(reinterpret_cast<float const*>(v0s), vIndex, sizeof(float));
            __m128 const v1z = _mm_i32gather_ps(reinterpret_cast<float const*>(v1s), vIndex, sizeof(float));
            __m128 const v2z = _mm_i32gather_ps(reinterpret_cast<float const*>(v2s), vIndex, sizeof(float));

            // known term: o - v0
            __m128 const oMv0x = _mm_sub_ps(ox, v0x);
            __m128 const oMv0y = _mm_sub_ps(oy, v0y);
            __m128 const oMv0z = _mm_sub_ps(oz, v0z);

            // e1 = v1 - v0, e2 = v2 - v0
            __m128 const e1x = _mm_sub_ps(v1x, v0x);
            __m128 const e1y = _mm_sub_ps(v1y, v0y);
            __m128 const e1z = _mm_sub_ps(v1z, v0z);

            __m128 const e2x = _mm_sub_ps(v2x, v0x);
            __m128 const e2y = _mm_sub_ps(v2y, v0y);
            __m128 const e2z = _mm_sub_ps(v2z, v0z);

            // det[-d, e1, e2] = dot(e1, cross(d, e2))
            __m128 const dXe2x = _mm_sub_ps(_mm_mul_ps(dy, e2z), _mm_mul_ps(dz, e2y));
            __m128 const dXe2y = _mm_sub_ps(_mm_mul_ps(dz, e2x), _mm_mul_ps(dx, e2z));
            __m128 const dXe2z = _mm_sub_ps(_mm_mul_ps(dx, e2y), _mm_mul_ps(dy, e2x));

            __m128 const det    = _mm_add_ps(_mm_add_ps(_mm_mul_ps(e1x, dXe2x), _mm_mul_ps(e1y, dXe2y)),
                                          _mm_mul_ps(e1z, dXe2z));
            __m128 const invDet = _mm_rcp_ps(det);

            // abs(det) > tol
            __m128 const detMask = _mm_cmp_ps(_mm_andnot_ps(signBit, det), tol, _CMP_GT_OQ);

            // uDetNum = det[-d, oMv0, e2] = dot(oMv0, cross(d, e2)) = dot(oMv0, dXe2) | u = uDetNum * invDet
            __m128 const uDetNum = _mm_add_ps(_mm_add_ps(_mm_mul_ps(oMv0x, dXe2x), _mm_mul_ps(oMv0y, dXe2y)),
                                              _mm_mul_ps(oMv0z, dXe2z));
            __m128 const u       = _mm_mul_ps(uDetNum, invDet);

            // u >= -tol && u <= 1.f + tol
            __m128 const uMask = _mm_and_ps(_mm_cmp_ps(u, mtol, _CMP_GE_OQ), _mm_cmp_ps(u, one, _CMP_LE_OQ));

            // vDetNum = det[-d, e1, oMv0] = dot(d, cross(oMv0, e1)) = dot(d, oMv0_Xe1)
            __m128 const oMv0_Xe1x = _mm_sub_ps(_mm_mul_ps(oMv0y, e1z), _mm_mul_ps(oMv0z, e1y));
            __m128 const oMv0_Xe1y = _mm_sub_ps(_mm_mul_ps(oMv0z, e1x), _mm_mul_ps(oMv0x, e1z));
            __m128 const oMv0_Xe1z = _mm_sub_ps(_mm_mul_ps(oMv0x, e1y), _mm_mul_ps(oMv0y, e1x));

            __m128 const vDetNum = _mm_add_ps(_mm_add_ps(_mm_mul_ps(dx, oMv0_Xe1x), _mm_mul_ps(dy, oMv0_Xe1y)),
                                              _mm_mul_ps(dz, oMv0_Xe1z));

            __m128 const v = _mm_mul_ps(vDetNum, invDet);

            // v >= -tol && (u + v) <= 1 + tol
            __m128 const uPlusv = _mm_add_ps(u, v);
            __m128 const vMask  = _mm_and_ps(_mm_cmp_ps(v, mtol, _CMP_GE_OQ), _mm_cmp_ps(uPlusv, one, _CMP_LE_OQ));

            // tDetNum = det[oMv0, e1, e2] = dot(e2, cross(oMv0, e1)) = dot(e2, oMv0_Xe1)
            __m128 const tDetNum = _mm_add_ps(_mm_add_ps(_mm_mul_ps(e2x, oMv0_Xe1x), _mm_mul_ps(e2y, oMv0_Xe1y)),
                                              _mm_mul_ps(e2z, oMv0_Xe1z));
            __m128 const t       = _mm_mul_ps(tDetNum, invDet);

            // t >= -tol && t <= tMax
            __m128 const tMask = _mm_and_ps(_mm_and_ps(_mm_cmp_ps(t, mtol, _CMP_GE_OQ), _mm_cmp_ps(t, tMaxv, _CMP_LE_OQ)),
                                            _mm_cmp_ps(t, tMinv, _CMP_GT_OQ));

            // extract best result
            __m128 const finalMask = _mm_and_ps(argMask, _mm_and_ps(_mm_and_ps(detMask, uMask), _mm_and_ps(vMask, tMask)));

            // if at least 1 respects the condition (check by extracting sign bit from each element)
            if (_mm_movemask_ps(finalMask) == 0)
            {
                return Triisect::nothing();
            }
            else
            {
                __m128 const      tMasked = _mm_blendv_ps(_mm_set1_ps(fl::infinity()), t, finalMask);
                alignas(16) float tArray[4];
                alignas(16) float uArray[4];
                alignas(16) float vArray[4];

                _mm_store_ps(tArray, tMasked);
                _mm_store_ps(uArray, u);
                _mm_store_ps(vArray, v);

                float minT      = fl::infinity();
                int   bestIndex = -1;

                for (int i = 0; i < 4; ++i)
                {
                    if (tArray[i] < minT)
                    {
                        minT      = tArray[i];
                        bestIndex = i;
                    }
                }

                assert(bestIndex >= 0 && "if a mask was active at least 1 intersection");
                return {.u     = fl::clamp01(uArray[bestIndex]),
                        .v     = fl::clamp01(vArray[bestIndex]),
                        .w     = fl::clamp01(1.f - uArray[bestIndex] - vArray[bestIndex]),
                        .t     = minT,
                        .index = static_cast<uint32_t>(bestIndex)};
            }
#else
            //#error "not implemented"
            return Triisect::nothing();
#endif
        }

        Triisect DMT_FASTCALL
            intersect8(Ray const& ray, float tMax, float tMin, Point3f const* v0s, Point3f const* v1s, Point3f const* v2s)
        {
#if defined DMT_MOLLER_TRUMBORE
            // constants
            __m256 const  tol     = _mm256_set1_ps(Triisect::tol);
            __m256 const  mtol    = _mm256_set1_ps(-Triisect::tol);
            __m256 const  one     = _mm256_set1_ps(Triisect::tol + 1.f);
            __m256i const iOne    = _mm256_set1_epi32(1);
            __m256 const  signBit = _mm256_set1_ps(-0.f);
            __m256 const  tMaxv   = _mm256_set1_ps(tMax);
            __m256 const  tMinv   = _mm256_set1_ps(tMin);

            // ray
            __m256 const dx = _mm256_set1_ps(ray.d.x);
            __m256 const dy = _mm256_set1_ps(ray.d.y);
            __m256 const dz = _mm256_set1_ps(ray.d.z);

            __m256 const ox = _mm256_set1_ps(ray.o.x);
            __m256 const oy = _mm256_set1_ps(ray.o.y);
            __m256 const oz = _mm256_set1_ps(ray.o.z);

            // v0, v1, v2
            __m256i vIndex = _mm256_set_epi32(21, 18, 15, 12, 9, 6, 3, 0);

            __m256 const v0x = _mm256_i32gather_ps(reinterpret_cast<float const*>(v0s), vIndex, sizeof(float));
            __m256 const v1x = _mm256_i32gather_ps(reinterpret_cast<float const*>(v1s), vIndex, sizeof(float));
            __m256 const v2x = _mm256_i32gather_ps(reinterpret_cast<float const*>(v2s), vIndex, sizeof(float));

            vIndex           = _mm256_add_epi32(vIndex, iOne);
            __m256 const v0y = _mm256_i32gather_ps(reinterpret_cast<float const*>(v0s), vIndex, sizeof(float));
            __m256 const v1y = _mm256_i32gather_ps(reinterpret_cast<float const*>(v1s), vIndex, sizeof(float));
            __m256 const v2y = _mm256_i32gather_ps(reinterpret_cast<float const*>(v2s), vIndex, sizeof(float));

            vIndex           = _mm256_add_epi32(vIndex, iOne);
            __m256 const v0z = _mm256_i32gather_ps(reinterpret_cast<float const*>(v0s), vIndex, sizeof(float));
            __m256 const v1z = _mm256_i32gather_ps(reinterpret_cast<float const*>(v1s), vIndex, sizeof(float));
            __m256 const v2z = _mm256_i32gather_ps(reinterpret_cast<float const*>(v2s), vIndex, sizeof(float));

            // known term: o - v0
            __m256 const oMv0x = _mm256_sub_ps(ox, v0x);
            __m256 const oMv0y = _mm256_sub_ps(oy, v0y);
            __m256 const oMv0z = _mm256_sub_ps(oz, v0z);

            // e1 = v1 - v0, e2 = v2 - v0
            __m256 const e1x = _mm256_sub_ps(v1x, v0x);
            __m256 const e1y = _mm256_sub_ps(v1y, v0y);
            __m256 const e1z = _mm256_sub_ps(v1z, v0z);

            __m256 const e2x = _mm256_sub_ps(v2x, v0x);
            __m256 const e2y = _mm256_sub_ps(v2y, v0y);
            __m256 const e2z = _mm256_sub_ps(v2z, v0z);

            // det[-d, e1, e2] = dot(e1, cross(d, e2))
            __m256 const dXe2x = _mm256_sub_ps(_mm256_mul_ps(dy, e2z), _mm256_mul_ps(dz, e2y));
            __m256 const dXe2y = _mm256_sub_ps(_mm256_mul_ps(dz, e2x), _mm256_mul_ps(dx, e2z));
            __m256 const dXe2z = _mm256_sub_ps(_mm256_mul_ps(dx, e2y), _mm256_mul_ps(dy, e2x));

            __m256 const det    = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(e1x, dXe2x), _mm256_mul_ps(e1y, dXe2y)),
                                             _mm256_mul_ps(e1z, dXe2z));
            __m256 const invDet = _mm256_rcp_ps(det);

            // abs(det) > tol
            __m256 const detMask = _mm256_cmp_ps(_mm256_andnot_ps(signBit, det), tol, _CMP_GT_OQ);

            // uDetNum = det[-d, oMv0, e2] = dot(oMv0, cross(d, e2)) = dot(oMv0, dXe2) | u = uDetNum * invDet
            __m256 const uDetNum = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(oMv0x, dXe2x), _mm256_mul_ps(oMv0y, dXe2y)),
                                                 _mm256_mul_ps(oMv0z, dXe2z));
            __m256 const u = _mm256_mul_ps(uDetNum, invDet);

            // u >= -tol && u <= 1.f + tol
            __m256 const uMask = _mm256_and_ps(_mm256_cmp_ps(u, mtol, _CMP_GE_OQ), _mm256_cmp_ps(u, one, _CMP_LE_OQ));

            // vDetNum = det[-d, e1, oMv0] = dot(d, cross(oMv0, e1)) = dot(d, oMv0_Xe1)
            __m256 const oMv0_Xe1x = _mm256_sub_ps(_mm256_mul_ps(oMv0y, e1z), _mm256_mul_ps(oMv0z, e1y));
            __m256 const oMv0_Xe1y = _mm256_sub_ps(_mm256_mul_ps(oMv0z, e1x), _mm256_mul_ps(oMv0x, e1z));
            __m256 const oMv0_Xe1z = _mm256_sub_ps(_mm256_mul_ps(oMv0x, e1y), _mm256_mul_ps(oMv0y, e1x));

            __m256 const vDetNum = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, oMv0_Xe1x), _mm256_mul_ps(dy, oMv0_Xe1y)),
                                                 _mm256_mul_ps(dz, oMv0_Xe1z));

            __m256 const v = _mm256_mul_ps(vDetNum, invDet);

            // v >= -tol && (u + v) <= 1 + tol
            __m256 const vMask = _mm256_and_ps(_mm256_cmp_ps(v, mtol, _CMP_GE_OQ),
                                               _mm256_cmp_ps(_mm256_add_ps(u, v), one, _CMP_LE_OQ));

            // tDetNum = det[oMv0, e1, e2] = dot(e2, cross(oMv0, e1)) = dot(e2, oMv0_Xe1)
            __m256 const tDetNum = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(e2x, oMv0_Xe1x), _mm256_mul_ps(e2y, oMv0_Xe1y)),
                                                 _mm256_mul_ps(e2z, oMv0_Xe1z));
            __m256 const t = _mm256_mul_ps(tDetNum, invDet);

            // t >= -tol && t <= tMax
            __m256 const tMask = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(t, mtol, _CMP_GE_OQ),
                                                             _mm256_cmp_ps(t, tMaxv, _CMP_LE_OQ)),
                                               _mm256_cmp_ps(t, tMinv, _CMP_GT_OQ));

            // extract best result
            __m256 const mask = _mm256_and_ps(_mm256_and_ps(detMask, uMask), _mm256_and_ps(vMask, tMask));

            // if at least 1 respects the condition (check by extracting sign bit from each element)
            if (_mm256_movemask_ps(mask) == 0)
            {
                return Triisect::nothing();
            }
            else
            {
                __m256 const      tMasked = _mm256_blendv_ps(_mm256_set1_ps(fl::infinity()), t, mask);
                alignas(32) float tArray[8];
                alignas(32) float uArray[8];
                alignas(32) float vArray[8];

                _mm256_store_ps(tArray, tMasked);
                _mm256_store_ps(uArray, u);
                _mm256_store_ps(vArray, v);

                float minT      = fl::infinity();
                int   bestIndex = -1;

                for (int i = 0; i < 8; ++i)
                {
                    if (tArray[i] < minT)
                    {
                        minT      = tArray[i];
                        bestIndex = i;
                    }
                }

                assert(bestIndex >= 0 && "if a mask was active at least 1 intersection");
                return {.u     = fl::clamp01(uArray[bestIndex]),
                        .v     = fl::clamp01(vArray[bestIndex]),
                        .w     = fl::clamp01(1.f - uArray[bestIndex] - vArray[bestIndex]),
                        .t     = minT,
                        .index = static_cast<uint32_t>(bestIndex)};
            }
#else
            //#error "not implemented"
            return Triisect::nothing();
#endif
        }

#if 0
        Triisect DMT_FASTCALL intersect(Ray const& ray, float tMax, float tMin, Point3f v0, Point3f v1, Point3f v2, uint32_t index)
        {
    #if defined DMT_MOLLER_TRUMBORE
            // moller trumbore algorithm https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
            Vector3f const         e1  = v1 - v0;
            Vector3f const         e2  = v2 - v0;
            static constexpr float tol = -Triisect::tol;

            // det[a, b, c] = a . (b x c) -> all cyclic permutations + cross is anticommutative
            Vector3f const h = cross(ray.d, e2);
            float const    a = dot(e1, h);

            if (a > -tol && a < tol) // if determinant is zero, then ray || triangle
                return Triisect::nothing();

            float const    f = fl::rcp(a); // inverse for cramer's rule
            Vector3f const s = ray.o - v0;
            float const    u = f * dot(s, h); // barycentric coord 0
            if (u < -tol || u > 1.f + tol)
                return Triisect::nothing();

            Vector3f    q = cross(s, e1);
            float const v = f * dot(ray.d, q); // barycentric coord 1
            if (v < -tol || (u + v) > 1.f + tol)
                return Triisect::nothing();

            float const t = f * dot(e2, q);
            if (t <= tMin || t > tMax)
                return Triisect::nothing();

            return {.u = fl::clamp01(u), .v = fl::clamp01(v), .w = fl::clamp01(1.f - u - v), .t = t, .index = index};
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
            Vector3f const A = v0 - ray.o;
            Vector3f const B = v1 - ray.o;
            Vector3f const C = v2 - ray.o;

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
        #ifdef DMT_BACKFACE_CULLING
            if (U < 0.0f || V < 0.0f || W < 0.0f)
                return Triisect::nothing();
        #else
            if ((U < 0.0f || V < 0.0f || W < 0.0f) && (U > 0.0f || V > 0.0f || W > 0.0f))
                return Triisect::nothing();
        #endif

            // calculate determinant
            float const det = U + V + W;
            if (fl::nearZero(det))
                return Triisect::nothing();

            // Calculate scaled z-coordinates of vertices and use them to calculate the hit distance
            float const Az = Sz * A[kz];
            float const Bz = Sz * B[kz];
            float const Cz = Sz * C[kz];
            float const T  = U * Az + V * Bz + W * Cz;
        #ifdef DMT_BACKFACE_CULLING
            if (T < tMin * det || T > tMax * det)
                return Triisect::nothing();
        #else
            int det_sign = fl::signBit(det);
            if (fl::xorf(T, det_sign) < tMin * fl::xorf(det, det_sign) ||
                fl::xorf(T, det_sign) > tMax * fl::xorf(det, det_sign))
                return Triisect::nothing();
        #endif
            // normalize U, V, W, and T
            float const rcpDet = 1.0f / det;
            float const u      = U * rcpDet;
            float const v      = V * rcpDet;
            float const w      = W * rcpDet;
            float const t      = T * rcpDet;

            return {.u = u, .v = v, .w = w, .t = t};
    #endif
        }
#endif

        Triisect DMT_FASTCALL intersect(Ray const& ray, float tMax, float tMin, Point3f v0, Point3f v1, Point3f v2, uint32_t index)
        {
            // 1) choose k by absolute max component (robust for grazing)
            auto maxComponentIndexAbs = [](Vector3f const& v) -> int {
                float ax = fabsf(v.x), ay = fabsf(v.y), az = fabsf(v.z);
                if (az > ax && az > ay)
                    return 2;
                if (ay > ax)
                    return 1;
                return 0;
            };
            int kz = maxComponentIndexAbs(ray.d);
            int kx = (kz + 1) % 3;
            int ky = (kx + 1) % 3;
            // keep the original Woop sign swap (preserve winding)
            if (ray.d[kz] < 0.0f)
                std::swap(kx, ky);

            // shear constants (kz guaranteed to be nonzero-ish because chosen by abs max)
            float const Sx = ray.d[kx] / ray.d[kz];
            float const Sy = ray.d[ky] / ray.d[kz];
            float const Sz = 1.0f / ray.d[kz];

            // vertices relative to ray origin
            Vector3f const A = v0 - ray.o;
            Vector3f const B = v1 - ray.o;
            Vector3f const C = v2 - ray.o;

            // shear & scale
            float const Ax = A[kx] - Sx * A[kz];
            float const Ay = A[ky] - Sy * A[kz];
            float const Bx = B[kx] - Sx * B[kz];
            float const By = B[ky] - Sy * B[kz];
            float const Cx = C[kx] - Sx * C[kz];
            float const Cy = C[ky] - Sy * C[kz];

            // scaled barycentrics
            float U = Cx * By - Cy * Bx;
            float V = Ax * Cy - Ay * Cx;
            float W = Bx * Ay - By * Ax;

            // double fallback (no clamp)
            if (fl::nearZero(U, 0x1.0p-9f) || fl::nearZero(V, 0x1.0p-9f) || fl::nearZero(W, 0x1.0p-9f))
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

            // scaled epsilon = relative to triangle size (avoid absolute tiny thresholds)
            float triMax = fmaxf(fmaxf(fabsf(Ax), fabsf(Ay)),
                                 fmaxf(fabsf(Bx), fmaxf(fabsf(By), fmaxf(fabsf(Cx), fabsf(Cy)))));
            float eps    = 1e-7f * fmaxf(1.0f, triMax); // tune factor if needed

#ifdef DMT_BACKFACE_CULLING
            // tie-break rule: make one barycentric exclusive (W <= eps) and others inclusive
            if (U < -eps || V < -eps || W <= eps) // allow tiny positive W but reject <= eps
                return Triisect::nothing();
#else
            // two-sided: reject if signs mixed beyond eps
            bool anyNeg = (U < -eps) || (V < -eps) || (W < -eps);
            bool anyPos = (U > eps) || (V > eps) || (W > eps);
            if (anyNeg && anyPos)
                return Triisect::nothing();
#endif

            float const det = U + V + W;
#ifdef DMT_BACKFACE_CULLING
            // require det sufficiently positive
            if (det <= eps)
                return Triisect::nothing();
#else
            if (fabsf(det) <= eps)
                return Triisect::nothing();
#endif

            // scaled z coords and T
            float const Az = Sz * A[kz];
            float const Bz = Sz * B[kz];
            float const Cz = Sz * C[kz];
            float const T  = U * Az + V * Bz + W * Cz;

            // t interval test (use scaled det)
            if (T < tMin * det || T > tMax * det)
                return Triisect::nothing();

            float const rcpDet = 1.0f / det;
            float const u      = U * rcpDet;
            float const v      = V * rcpDet;
            float const w      = W * rcpDet;
            float const t      = T * rcpDet;

            // LAST RESORT: ensure geometric normal faces the ray (avoid backface sneaking through)
            // If you *prefer* no hit on grazing cases, replace next block with `if (det <= eps) return nothing;`.
            {
                Vector3f ng     = normalize(cross(v1 - v0, v2 - v0));
                float    facing = dot(ng, -ray.d); // positive if ng faces the ray
                if (facing <= 0.0f)
                {
#ifdef DMT_BACKFACE_CULLING
                    // backface got through; treat as miss
                    return Triisect::nothing();
#else
                    // flip normal for two-sided shading if desired (we still report hit)
                    // ng = -ng;
#endif
                }
            }

            return {.u = u, .v = v, .w = w, .t = t};
        }

        Intersection DMT_FASTCALL fromTrisect(Triisect trisect, Ray const& ray, RGB color, Point2f uv)
        {
            Intersection isect{};
            if (trisect)
            {
                isect.p     = ray.o + trisect.t * ray.d;
                isect.ng    = {};
                isect.t     = trisect.t;
                isect.hit   = true;
                isect.color = color;
                isect.uv    = uv;
            }
            return isect;
        }
    } // namespace triangle

    // -- bounds --
    Bounds3f Triangle::bounds() const
    {
        Point3f const p0{tri.v0.x, tri.v0.y, tri.v0.z};
        Point3f const p1{tri.v1.x, tri.v1.y, tri.v1.z};
        Point3f const p2{tri.v2.x, tri.v2.y, tri.v2.z};
        return triangle::bounds(p0, p1, p2);
    }

    Bounds3f Triangles2::bounds() const { return triangle::bounds2(xs, ys, zs); }

    Bounds3f Triangles4::bounds() const { return triangle::bounds4(xs, ys, zs); }

    Bounds3f Triangles8::bounds() const { return triangle::bounds8(xs, ys, zs); }

    // -- intersect --
    Intersection Triangle::intersect(Ray const& ray, float tMax) const
    {
        triangle::Triisect const trisect = triangle::intersect(ray, tMax, 1e-4f, tri.v0, tri.v1, tri.v2, 0);
        return triangle::fromTrisect(trisect, ray, tri.color);
    }

    Intersection Triangles2::intersect(Ray const& ray, float tMax) const
    {
        Point3f v0s[4]{};
        Point3f v1s[4]{};
        Point3f v2s[4]{};
        for (int i = 0; i < 4; ++i)
        {
            int j  = i & 0x3;
            v0s[i] = {xs[3 * j + 0], ys[3 * j + 0], zs[3 * j + 0]};
            v1s[i] = {xs[3 * j + 1], ys[3 * j + 1], zs[3 * j + 1]};
            v2s[i] = {xs[3 * j + 2], ys[3 * j + 2], zs[3 * j + 2]};
        }
        triangle::Triisect const trisect = triangle::intersect4(ray, fl::infinity(), 1e-4f, v0s, v1s, v2s, 0x3);
        assert(trisect.index < 2 && "out of bounds triangle2 intersection index");
        return triangle::fromTrisect(trisect, ray, colors[trisect.index]);
    }

    Intersection Triangles4::intersect(Ray const& ray, float tMax) const
    {
        Point3f v0s[4]{};
        Point3f v1s[4]{};
        Point3f v2s[4]{};
        for (int j = 0; j < 4; ++j)
        {
            v0s[j] = {xs[3 * j + 0], ys[3 * j + 0], zs[3 * j + 0]};
            v1s[j] = {xs[3 * j + 1], ys[3 * j + 1], zs[3 * j + 1]};
            v2s[j] = {xs[3 * j + 2], ys[3 * j + 2], zs[3 * j + 2]};
        }
        triangle::Triisect const trisect = triangle::intersect4(ray, tMax, 1e-4f, v0s, v1s, v2s, 0xf);
        return triangle::fromTrisect(trisect, ray, colors[trisect.index]);
    }

    Intersection Triangles8::intersect(Ray const& ray, float tMax) const
    {
        Point3f v0s[8]{};
        Point3f v1s[8]{};
        Point3f v2s[8]{};
        for (int i = 0; i < Triangles8::numTriangles; ++i)
        {
            v0s[i] = {xs[3 * i + 0], ys[3 * i + 0], zs[3 * i + 0]};
            v1s[i] = {xs[3 * i + 1], ys[3 * i + 1], zs[3 * i + 1]};
            v2s[i] = {xs[3 * i + 2], ys[3 * i + 2], zs[3 * i + 2]};
        }

        triangle::Triisect const trisect = triangle::intersect8(ray, tMax, 1e-4f, v0s, v1s, v2s);
        return triangle::fromTrisect(trisect, ray, colors[trisect.index]);
    }

    // INDEXED

    std::tuple<Point3f, Point3f, Point3f> TriangleIndexedBase::worldSpacePts(size_t _triIdx) const
    {
        Transform m = transformFromAffine(scene->instances[instanceIdx]->affineTransform); // maybe inefficient
        TriangleMesh const& mesh = *scene->geometry[scene->instances[instanceIdx]->meshIdx];
        IndexedTri const    tri  = mesh.getIndexedTri(_triIdx);

        Point3f const p0 = m(mesh.getPosition(tri[0].positionIdx));
        Point3f const p1 = m(mesh.getPosition(tri[1].positionIdx));
        Point3f const p2 = m(mesh.getPosition(tri[2].positionIdx));

        return {p0, p1, p2};
    }

    Vector3f TriangleIndexedBase::normalFromIndex(size_t tri) const
    {
        auto const* instance = scene->instances[instanceIdx].get();
        auto const* geo      = scene->geometry[instance->meshIdx].get();
        auto const  index    = geo->getIndexedTri(tri);
        Transform   t        = transformFromAffine(instance->affineTransform);

        // Get raw normals and sum them as plain vectors
        Vector3f n0 = geo->getNormal(index.v[0].normalIdx);
        Vector3f n1 = geo->getNormal(index.v[1].normalIdx);
        Vector3f n2 = geo->getNormal(index.v[2].normalIdx);

        // normalize done by constructor
        Normal3f avg = n0 + n1 + n2;

        assert(fl::abs(normL2(avg) - 1.f) < 1e-5f && "sum of normals should still be unit length");
        Normal3f transformed = t(avg);
        assert(fl::abs(normL2(transformed) - 1.f) < 1e-5f && "transformed normal should still be unit length");
        return transformed;
    }

    Point2f TriangleIndexedBase::uvFromIndex(size_t tri, float u, float v) const
    {
        auto const*   instance = scene->instances[instanceIdx].get();
        auto const*   geo      = scene->geometry[instance->meshIdx].get();
        auto const    index    = geo->getIndexedTri(tri);
        Point2f const uv0      = geo->getUV(index.v[0].uvIdx);
        Point2f const uv1      = geo->getUV(index.v[1].uvIdx);
        Point2f const uv2      = geo->getUV(index.v[2].uvIdx);

        // Perform the barycentric interpolation.
        float const w0 = 1.0f - u - v;
        float const w1 = u;
        float const w2 = v;

        Point2f const interpolatedUV = w0 * uv0 + w1 * uv1 + w2 * uv2;

        return interpolatedUV;
    }

    void TriangleIndexedBase::compute_dpdu_dpdv(size_t triIdx, Vector3f* dpdu, Vector3f* dpdv) const
    {
        assert(dpdu && dpdv);
        Transform           m    = transformFromAffine(scene->instances[instanceIdx]->affineTransform);
        TriangleMesh const& mesh = *scene->geometry[scene->instances[instanceIdx]->meshIdx];
        IndexedTri const    tri  = mesh.getIndexedTri(triIdx);

        // Positions
        Point3f p0 = m(mesh.getPosition(tri[0].positionIdx));
        Point3f p1 = m(mesh.getPosition(tri[1].positionIdx));
        Point3f p2 = m(mesh.getPosition(tri[2].positionIdx));

        // UVs
        Point2f uv0 = mesh.getUV(tri[0].uvIdx);
        Point2f uv1 = mesh.getUV(tri[1].uvIdx);
        Point2f uv2 = mesh.getUV(tri[2].uvIdx);

        Vector2f duv1 = uv1 - uv0;
        Vector2f duv2 = uv2 - uv0;

        Vector3f dp1 = p1 - p0;
        Vector3f dp2 = p2 - p0;

        float det = duv1.x * duv2.y - duv1.y * duv2.x;
        if (fl::abs(det) < 1e-8f)
        {
            // UVs are degenerate — fallback to geometric basis
            Vector3f ng = normalize(cross(p2 - p0, p1 - p0));
            coordinateSystemFallback(ng, dpdu, dpdv); // builds arbitrary tangent frame
            return;
        }

        float invDet = 1.0f / det;
        *dpdu        = (duv2.y * dp1 - duv1.y * dp2) * invDet;
        *dpdv        = (-duv2.x * dp1 + duv1.x * dp2) * invDet;
    }

    Bounds3f TriangleIndexed::bounds() const
    {
        auto const [p0, p1, p2] = worldSpacePts(triIdx);

        Point3f const pMin = min(min(p0, p1), p2);
        Point3f const pMax = max(max(p0, p1), p2);

        return makeBounds(pMin, pMax);
    }

    Intersection TriangleIndexed::intersect(Ray const& ray, float tMax) const
    {
        auto const [p0, p1, p2] = worldSpacePts(triIdx);

        triangle::Triisect const trisect = triangle::intersect(ray, tMax, 1e-4f, p0, p1, p2, 0);

        auto ret = triangle::fromTrisect(trisect,
                                         ray,
                                         scene->instances[instanceIdx]->color,
                                         uvFromIndex(triIdx, trisect.u, trisect.v));
        if (trisect)
        {
            ret.ng = normalFromIndex(triIdx);
            compute_dpdu_dpdv(triIdx, &ret.dpdu, &ret.dpdv);
        }
        return ret;
    }

    // clang-format off
    template <size_t N>
    struct TypeFromSize;
    template <> struct TypeFromSize<2> { using type = TrianglesIndexed2; };
    template <> struct TypeFromSize<4> { using type = TrianglesIndexed4; };
    template <> struct TypeFromSize<8> { using type = TrianglesIndexed8; };
    template <size_t N> using TypeFromSize_t = TypeFromSize<N>::type;
    // clang-format on

    template <size_t N>
        requires(N == 2 || N == 4 || N == 8)
    Bounds3f computeIndexedTriangleBounds(TypeFromSize_t<N> const* obj,
                                          size_t const             triIdxs[N],
                                          Bounds3f (*boundsFunc)(float const*, float const*, float const*))
    {
        constexpr size_t NumPoints = 3 * N;
        float            xs[NumPoints]{}, ys[NumPoints]{}, zs[NumPoints]{};
        float            aos[3 * NumPoints]{}; // 3 floats per point

        for (size_t triIdx = 0; triIdx < N; ++triIdx)
        {
            auto tuple = obj->worldSpacePts(triIdxs[triIdx]);
            std::memcpy(aos + 9 * triIdx, &tuple, sizeof(tuple));
        }

        arch::transpose3xN(aos, xs, ys, zs, NumPoints);
        return boundsFunc(xs, ys, zs);
    }

    Bounds3f TrianglesIndexed2::bounds() const
    {
        return computeIndexedTriangleBounds<2>(this, triIdxs, triangle::bounds2);
    }

    Bounds3f TrianglesIndexed4::bounds() const
    {
        return computeIndexedTriangleBounds<4>(this, triIdxs, triangle::bounds4);
    }

    Bounds3f TrianglesIndexed8::bounds() const
    {
        return computeIndexedTriangleBounds<8>(this, triIdxs, triangle::bounds8);
    }

    template <size_t N>
        requires(N == 2 || N == 4 || N == 8)
    void fillIndexedVerts(TypeFromSize_t<N> const* obj, size_t const triIdxs[N], Point3f v0s[N], Point3f v1s[N], Point3f v2s[N])
    {
        for (uint32_t tri = 0; tri < N; ++tri)
        {
            auto const [v0, v1, v2] = obj->worldSpacePts(triIdxs[tri]);

            v0s[tri] = v0;
            v1s[tri] = v1;
            v2s[tri] = v2;
        }
    }

    Intersection TrianglesIndexed2::intersect(Ray const& ray, float tMax) const
    {
        Point3f v0s[2]{};
        Point3f v1s[2]{};
        Point3f v2s[2]{};
        fillIndexedVerts<2>(this, triIdxs, v0s, v1s, v2s);
        triangle::Triisect trisect = triangle::intersect4(ray, tMax, 1e-4f, v0s, v1s, v2s, 0x3);

        auto ret = triangle::fromTrisect(trisect,
                                         ray,
                                         scene->instances[instanceIdx]->color,
                                         uvFromIndex(triIdxs[trisect.index], trisect.u, trisect.v));
        if (trisect)
        {
            ret.ng = normalFromIndex(triIdxs[trisect.index]);
            compute_dpdu_dpdv(triIdxs[trisect.index], &ret.dpdu, &ret.dpdv);
        }
        return ret;
    }

    Intersection TrianglesIndexed4::intersect(Ray const& ray, float tMax) const
    {
        Point3f v0s[4]{};
        Point3f v1s[4]{};
        Point3f v2s[4]{};
        fillIndexedVerts<4>(this, triIdxs, v0s, v1s, v2s);
        triangle::Triisect trisect = triangle::intersect4(ray, tMax, 1e-4f, v0s, v1s, v2s, 0xf);

        auto ret = triangle::fromTrisect(trisect,
                                         ray,
                                         scene->instances[instanceIdx]->color,
                                         uvFromIndex(triIdxs[trisect.index], trisect.u, trisect.v));
        if (trisect)
        {
            ret.ng = normalFromIndex(triIdxs[trisect.index]);
            compute_dpdu_dpdv(triIdxs[trisect.index], &ret.dpdu, &ret.dpdv);
        }
        return ret;
    }

    Intersection TrianglesIndexed8::intersect(Ray const& ray, float tMax) const
    {
        Point3f v0s[8]{};
        Point3f v1s[8]{};
        Point3f v2s[8]{};
        fillIndexedVerts<8>(this, triIdxs, v0s, v1s, v2s);
        triangle::Triisect trisect = triangle::intersect8(ray, tMax, 1e-4f, v0s, v1s, v2s);

        auto ret = triangle::fromTrisect(trisect,
                                         ray,
                                         scene->instances[instanceIdx]->color,
                                         uvFromIndex(triIdxs[trisect.index], trisect.u, trisect.v));
        if (trisect)
        {
            ret.ng = normalFromIndex(triIdxs[trisect.index]);
            compute_dpdu_dpdv(triIdxs[trisect.index], &ret.dpdu, &ret.dpdv);
        }
        return ret;
    }
} // namespace dmt