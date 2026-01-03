#ifndef DMT_CUDA_CORE_HOST_SCENE_H
#define DMT_CUDA_CORE_HOST_SCENE_H

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

struct Triangle {
  Triangle() = default;
  Triangle(float3 const _v0, float3 const _v1, float3 const _v2)
      : v0(_v0), v1(_v1), v2(_v2) {}
  float3 v0;
  float3 v1;
  float3 v2;
};

template <typename T, typename... Vectors>
std::vector<T> concat(const std::vector<T>& first, const Vectors&... rest) {
  std::vector<T> result = first;
  (result.insert(result.end(), rest.begin(), rest.end()),
   ...);  // Fold expression
  return result;
}

struct HostTriangleScene {
  std::vector<Triangle> triangles;
  std::vector<uint32_t> nextMeshIndices;

  uint32_t meshCount() const {
    return static_cast<uint32_t>(nextMeshIndices.size());
  }

  void addModel(std::vector<Triangle> const& mesh) {
    triangles.insert(triangles.end(), mesh.begin(), mesh.end());
    if (!nextMeshIndices.empty())
      nextMeshIndices.push_back(nextMeshIndices.back() + mesh.size());
    else
      nextMeshIndices.push_back(mesh.size());
  }
};

std::vector<Triangle> generateSphereMesh(float3 center, float radius,
                                         int latSubdiv, int lonSubdiv);
std::vector<Triangle> generateCube(float3 center, float3 scale);
std::vector<Triangle> generatePlane(float3 center, float3 normal, float width,
                                    float height);

#endif  // DMT_CUDA_CORE_HOST_SCENE_H