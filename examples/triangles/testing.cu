#include "testing.h"

// float3 operators
#include "common.cuh"

#include <cmath>
#include <numbers>

// TODO: check winding
std::vector<Triangle> generateSphereMesh(float3 center, float radius,
                                         int latSubdiv, int lonSubdiv) {
  static float constexpr PI = std::numbers::pi_v<float>;
  std::vector<Triangle> triangles;
  triangles.reserve(2 * lonSubdiv * (latSubdiv - 1));

  for (int i = 0; i < latSubdiv; ++i) {
    // theta: 0 -> PI
    float const theta0 = PI * static_cast<float>(i) / latSubdiv;
    float const theta1 = PI * static_cast<float>(i + 1) / latSubdiv;

    float const y0 = cosf(theta0);
    float const y1 = cosf(theta1);
    float const r0 = sinf(theta0);
    float const r1 = sinf(theta1);

    for (int j = 0; j < lonSubdiv; ++j) {
      // phi: 0 -> 2PI
      float const phi0 = 2.f * PI * static_cast<float>(j) / lonSubdiv;
      float const phi1 =
          2.f * PI * static_cast<float>((j + 1) % lonSubdiv) / lonSubdiv;

      float const x00 = r0 * cosf(phi0);
      float const z00 = r0 * sinf(phi0);
      float const x01 = r1 * cosf(phi1);
      float const z01 = r1 * sinf(phi1);

      float const x10 = r1 * cosf(phi0);
      float const z10 = r1 * sinf(phi0);
      float const x11 = r0 * cosf(phi1);
      float const z11 = r0 * sinf(phi1);

      float3 const p00 = center + radius * make_float3(x00, y0, z00);
      float3 const p01 = center + radius * make_float3(x01, y0, z01);
      float3 const p10 = center + radius * make_float3(x10, y1, z10);
      float3 const p11 = center + radius * make_float3(x11, y1, z11);

      // skip degenerate caps
      if (i != 0) {
        triangles.emplace_back(p00, p10, p01);
      }
      if (i != latSubdiv - 1) {
        triangles.emplace_back(p01, p10, p11);
      }
    }
  }

  return triangles;
}
