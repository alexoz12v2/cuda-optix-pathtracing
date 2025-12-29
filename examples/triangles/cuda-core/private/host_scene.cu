#include "host_scene.cuh"

#include "cuda-core/common_math.cuh"

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

// TODO rotation
std::vector<Triangle> generateCube(float3 center, float3 scale) {
  std::vector<Triangle> tris;
  tris.reserve(12);  // 12 tris, 2 per face

  // 8 corners
  float3 corners[8];
  for (int i = 0; i < 8; ++i) {
    float3 offset = make_float3(((i & 1) ? 0.5f : -0.5f) * scale.x,
                                ((i & 2) ? 0.5f : -0.5f) * scale.y,
                                ((i & 4) ? 0.5f : -0.5f) * scale.z);
    // TODO: rotation
    corners[i] = center + offset;
  }
  int faces[6][4] = {
      {0, 1, 3, 2},  // bottom (-y)
      {4, 5, 7, 6},  // top (+y)
      {0, 1, 5, 4},  // front (-z)
      {2, 3, 7, 6},  // back (+z)
      {0, 2, 6, 4},  // left (-x)
      {1, 3, 7, 5}   // right (+x)
  };

  for (int f = 0; f < 6; ++f) {
    int* idx = faces[f];
    tris.emplace_back(corners[idx[0]], corners[idx[1]], corners[idx[2]]);
    tris.emplace_back(corners[idx[0]], corners[idx[2]], corners[idx[3]]);
  }
  return tris;
}

void orthonormalTangentStable(float3 normal, float3* tangent,
                              float3* bitangent) {
  float3 n = normalize(normal);

  // Pick the major axis least aligned with normal
  float3 majorAxis;
  if (fabs(n.x) <= fabs(n.y) && fabs(n.x) <= fabs(n.z))
    majorAxis = make_float3(1, 0, 0);  // X is least aligned
  else if (fabs(n.y) <= fabs(n.x) && fabs(n.y) <= fabs(n.z))
    majorAxis = make_float3(0, 1, 0);  // Y is least aligned
  else
    majorAxis = make_float3(0, 0, 1);  // Z is least aligned

  *tangent = normalize(cross(majorAxis, n));
  *bitangent = cross(n, *tangent);  // completes the orthonormal basis
}

std::vector<Triangle> generatePlane(float3 center, float3 normal, float width,
                                    float height) {
  std::vector<Triangle> tris;
  tris.reserve(2);
  float3 tangent;
  float3 bitangent;
  orthonormalTangentStable(normal, &tangent, &bitangent);
  tangent *= width * 0.5f;
  bitangent *= height * 0.5f;

  float3 const p0 = center - tangent - bitangent;
  float3 const p1 = center + tangent - bitangent;
  float3 const p2 = center + tangent + bitangent;
  float3 const p3 = center - tangent + bitangent;

  // CCW winding
  tris.emplace_back(p0, p2, p1);
  tris.emplace_back(p0, p3, p2);
  return tris;
}
