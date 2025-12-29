#include "shapes.cuh"

#include "extra_math.cuh"

inline float constexpr MOLLER_TRUMBORE_TOLERANCE = 1e-7f;

__device__ HitResult triangleIntersect(float4 x, float4 y, float4 z, Ray ray) {
  HitResult result;

  float3 const e0{x.y - x.x, y.y - y.x, z.y - z.x};
  float3 const e1{x.z - x.x, y.z - y.x, z.z - z.x};

  float3 const d_cross_e1{ray.d.y * e1.z - ray.d.z * e1.y,
                          ray.d.z * e1.x - ray.d.x * e1.z,
                          ray.d.x * e1.y - ray.d.y * e1.x};

  float const det =
      d_cross_e1.x * e0.x + d_cross_e1.y * e0.y + d_cross_e1.z * e0.z;
  if (fabsf(det) < MOLLER_TRUMBORE_TOLERANCE) return result;  // parallel

  float const invDet = 1.0f / det;
  float3 const o_to_v0{ray.o.x - x.x, ray.o.y - y.x, ray.o.z - z.x};
  float3 const t_cross_e0{o_to_v0.y * e0.z - o_to_v0.z * e0.y,
                          o_to_v0.z * e0.x - o_to_v0.x * e0.z,
                          o_to_v0.x * e0.y - o_to_v0.y * e0.x};

  float const u =
      invDet * (d_cross_e1.x * o_to_v0.x + d_cross_e1.y * o_to_v0.y +
                d_cross_e1.z * o_to_v0.z);
  float const v = invDet * (t_cross_e0.x * ray.d.x + t_cross_e0.y * ray.d.y +
                            t_cross_e0.z * ray.d.z);
  float const t = invDet * (t_cross_e0.x * e1.x + t_cross_e0.y * e1.y +
                            t_cross_e0.z * e1.z);

  bool const valid =
      (u >= -MOLLER_TRUMBORE_TOLERANCE && v >= -MOLLER_TRUMBORE_TOLERANCE &&
       (u + v) <= 1 + MOLLER_TRUMBORE_TOLERANCE) &&
      (t > 1e-4f);  // <-- reject t <= 0

  if (valid) {
    float3 const p0 = make_float3(x.x, y.x, z.x);
    float3 const p1 = make_float3(x.y, y.y, z.y);
    float3 const p2 = make_float3(x.z, y.z, z.z);

    result.hit = 1;
    result.t = t;
    result.pos = p0 + u * e0 + v * e1;
    // TODO indexed computation of normal. now gamble on counter-clockwise
    result.normal = normalize(cross(e1, e0));
#if 0
    if (result.pos.x <= -2.f && result.pos.y > 0.f) {
      printf("      - Intersection at %f %f %f | normal %x %x %x\n",
             result.pos.x, result.pos.y, result.pos.z,
             *(uint32_t*)(&result.normal.x), *(uint32_t*)(&result.normal.y),
             *(uint32_t*)(&result.normal.z));
    }
#endif
    assert(fabsf(length2(result.normal) - 1.f) < 1e-3f &&
           "Expected unit vector");
    result.error = errorFromTriangleIntersection(u, v, p0, p1, p2);
  }

  return result;
}

HitResult hostIntersectMT(const float3& o, const float3& d, const float3& v0,
                          const float3& v1, const float3& v2) {
#if 0
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "Ray Direction" << std::endl;
  std::cout << "    " << d.x << ", " << d.y << ", " << d.z << std::endl;
  std::cout << "Ray Origin" << std::endl;
  std::cout << "    " << o.x << ", " << o.y << ", " << o.z << std::endl;
  std::cout << "Triangle" << std::endl;
  std::cout << "    " << v0.x << ", " << v0.y << ", " << v0.z << std::endl;
  std::cout << "    " << v1.x << ", " << v1.y << ", " << v1.z << std::endl;
  std::cout << "    " << v2.x << ", " << v2.y << ", " << v2.z << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
#endif
  constexpr float EPS = 1e-7f;
  HitResult hit;

  float3 e0{v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
  float3 e1{v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};

  float3 p{d.y * e1.z - d.z * e1.y, d.z * e1.x - d.x * e1.z,
           d.x * e1.y - d.y * e1.x};

  float det = p.x * e0.x + p.y * e0.y + p.z * e0.z;
  if (fabs(det) < EPS) return hit;

  float invDet = 1.0f / det;

  float3 t{o.x - v0.x, o.y - v0.y, o.z - v0.z};

  float u = invDet * (p.x * t.x + p.y * t.y + p.z * t.z);
  if (u < -EPS || u > 1 + EPS) return hit;

  float3 q{t.y * e0.z - t.z * e0.y, t.z * e0.x - t.x * e0.z,
           t.x * e0.y - t.y * e0.x};

  float v = invDet * (q.x * d.x + q.y * d.y + q.z * d.z);
  if (v < -EPS || u + v > 1 + EPS) return hit;

  float tHit = invDet * (q.x * e1.x + q.y * e1.y + q.z * e1.z);
  if (tHit > EPS) {
    hit.hit = 1;
    hit.t = tHit;
    hit.pos = v0 + u * e0 + v * e1;
    // TODO indexed computation of normal. now gamble on counter-clockwise
    hit.normal = cross(e1, e0);
    hit.error = errorFromTriangleIntersection(u, v, v0, v1, v2);
  }

  return hit;
}
