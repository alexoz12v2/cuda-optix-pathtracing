#include "testing.h"

// float3 operators
#include "common.cuh"

#include <cmath>
#include <numbers>

std::vector<Triangle> generateSphereMesh(float3 center, float radius,
                                         int latSubdiv, int lonSubdiv) {
  std::vector<Triangle> triangles;
  triangles.reserve(2 * lonSubdiv * (latSubdiv - 1));

  float const PI = std::numbers::pi_v<float>;
  float3 const topPole = center + make_float3(0, radius, 0);
  float3 const bottomPole = center + make_float3(0, -radius, 0);

  // Loop over latitude bands (excluding poles)
  for (int i = 0; i < latSubdiv; ++i) {
    float const theta0 = PI * float(i) / latSubdiv;
    float const theta1 = PI * float(i + 1) / latSubdiv;

    float const y0 = radius * cosf(theta0);
    float const y1 = radius * cosf(theta1);
    float const r0 = radius * sinf(theta0);
    float const r1 = radius * sinf(theta1);

    for (int j = 0; j < lonSubdiv; ++j) {
      float const phi0 = 2.f * PI * float(j) / lonSubdiv;
      float const phi1 = 2.f * PI * float((j + 1) % lonSubdiv) / lonSubdiv;

      float3 p00 = center + make_float3(r0 * cosf(phi0), y0, r0 * sinf(phi0));
      float3 p01 = center + make_float3(r0 * cosf(phi1), y0, r0 * sinf(phi1));
      float3 p10 = center + make_float3(r1 * cosf(phi0), y1, r1 * sinf(phi0));
      float3 p11 = center + make_float3(r1 * cosf(phi1), y1, r1 * sinf(phi1));

      if (i == 0) {
        // Top cap: connect top pole to first latitude ring
        triangles.emplace_back(topPole, p10, p11);
      } else if (i == latSubdiv - 1) {
        // Bottom cap: connect last latitude ring to bottom pole
        triangles.emplace_back(p00, bottomPole, p01);
      } else {
        // Middle quad split into two triangles
        triangles.emplace_back(p00, p10, p01);
        triangles.emplace_back(p01, p10, p11);
      }
    }
  }

  return triangles;
}
