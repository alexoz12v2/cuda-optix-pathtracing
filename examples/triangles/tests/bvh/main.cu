
#include "cuda-core/types.cuh"
#include "cuda-core/common_math.cuh"
#include "cuda-core/host_scene.cuh"

#ifdef DMT_OS_WINDOWS
#  include <Windows.h>
#elif defined(DMT_OS_LINUX)
#  include <unistd.h>
#  include <limits.h>
#endif

#include <iostream>
#include <span>
#include <numeric>

// - spatial splits
//     During the construction, both spatial splits and (standard) object splits
//     are evaluated, choosing the one minimizing the cost approximation.
// - Mixing AABBs and OOBs
//    To partition primitives during construction, five different splitting
//    strategies are performed choosing one with the smallest SAH cost: (1)
//    object and (2) spatial splitting in world space, (3) object and (4)
//    spatial splitting in a coordinate frame aligned to the orientation of hair
//    segments, and (5) clustering hair segments of similar orientation.

// A BVH node contains information such as child node pointers, the
//   number of leaves, and bounding box(es).
// the coordinates of child nodes’ bounding boxes relative to the parent using
//   fewer bits to reduce memory overhead. The quantized box must conservatively
//   cover the original bounding box

// in any-hit: accelerates tests simply by visiting the child node with
//   the larger surface area first.

// false-misses could occur if the distance between the entry and exit points
// (t_far − t_near) is less than or equal to two ULPs (unit of least precision)
// by simple error analysis. He proposed an algorithm that only needs adding two
// ULPs to each component of the inverse ray direction for computing tf ar
// before traversal.

// To prevent cores from stalling when
// a ray has been completely traversed, new rays are loaded from a
// global work queue using so-called persistent threads. The control
// flow divergence can be mitigated by breaking the traversal loop
// into two independent loops, i.e., iterating over the hierarchy and iterating
// over primitives

// addition, the topmost entries of the traversal stack
// – including the result of the child intersection tests – are stored in
// shared memory to reduce memory transfers.

struct AABB {
  AABB()
      : min(CUDART_INF_F, CUDART_INF_F, CUDART_INF_F),
        max(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F) {}
  float3 min;
  float3 max;
};

inline AABB bbUnion(AABB const& a, AABB const& b) {
  AABB const ret{
      .min = minv(a.min, b.min),
      .max = maxv(a.max, b.max),
  };
  return ret;
}

inline float surfaceArea(AABB const& box) {
  assert(maxComponentValue(box.min) < minComponentValue(box.max));
  const auto [x, y, z] = box.max - box.min;
  return 4.f * x * y + 2.f * y * z;
}

inline float surfaceArea(Triangle const& tri) {
  float3 const e0 = tri.v1 - tri.v0;
  float3 const e1 = tri.v2 - tri.v0;
  return length(cross(e0, e1)) * 0.5f;
}

inline AABB bounds(Triangle const& tri) {
  AABB const ret{
      .min = minv(minv(tri.v0, tri.v1), tri.v2),
      .max = maxv(maxv(tri.v0, tri.v1), tri.v2),
  };
  return ret;
}

inline float splitCost(std::span<Triangle> triangles, size_t splitPos,
                       float triCost = 1) {
  float const A_S =
      surfaceArea(std::accumulate(triangles.begin(), triangles.end(), AABB(),
                                  [](AABB const& acc, Triangle const& tri) {
                                    return bbUnion(acc, ::bounds(tri));
                                  }));
  float const A_S1 = surfaceArea(
      std::accumulate(triangles.begin(), triangles.begin() + splitPos, AABB(),
                      [](AABB const& acc, Triangle const& tri) {
                        return bbUnion(acc, ::bounds(tri));
                      }));
  float const A_S2 = surfaceArea(
      std::accumulate(triangles.begin() + splitPos, triangles.end(), AABB(),
                      [](AABB const& acc, Triangle const& tri) {
                        return bbUnion(acc, ::bounds(tri));
                      }));
  ptrdiff_t const NS1 = splitPos;
  ptrdiff_t const NS2 = triangles.size() - splitPos;
  return A_S1 / A_S * NS1 * triCost + A_S2 / A_S * NS2 * triCost;
}

class IBVHBuildNode {
 public:
  bool isLeaf() const { return do_isLeaf(); }
  AABB bounds() const { return do_bounds(); }
  float surfaceArea() const { return ::surfaceArea(bounds()); }
  virtual ~IBVHBuildNode() = default;

 protected:
  virtual bool do_isLeaf() const = 0;
  virtual AABB do_bounds() const = 0;
};

class BVHBuildInteriorNode final : public IBVHBuildNode {
  explicit BVHBuildInteriorNode(std::unique_ptr<IBVHBuildNode> left,
                                std::unique_ptr<IBVHBuildNode> right)
      : m_left(std::move(left)), m_right(std::move(right)) {}

 protected:
  bool do_isLeaf() const override { return false; }
  AABB do_bounds() const override {
    return bbUnion(m_left->bounds(), m_right->bounds());
  }

 private:
  std::unique_ptr<IBVHBuildNode> m_left;
  std::unique_ptr<IBVHBuildNode> m_right;
};

class BVHBuildLeafTriangles final : public IBVHBuildNode {
  explicit BVHBuildLeafTriangles(std::span<Triangle> triangles)
      : m_triangles(triangles.begin(), triangles.end()) {}
  size_t primitiveCount() const { return m_triangles.size(); }

 protected:
  bool do_isLeaf() const override { return true; }
  AABB do_bounds() const override {
    return std::accumulate(m_triangles.begin(), m_triangles.end(), AABB(),
                           [](AABB const& acc, Triangle const& tri) {
                             return bbUnion(acc, ::bounds(tri));
                           });
  }

 private:
  std::vector<Triangle> m_triangles;
};

namespace {

void testBvh() {
  std::cout << "Hello BVH World" << std::endl;
  // Test BVH Building from a list of bounding boxes
}

}  // namespace

// UNICODE and _UNICODE always defined
#ifdef _WIN32
int wmain() {
#else
int main() {
#endif
#ifdef DMT_OS_WINDOWS
  SetConsoleOutputCP(CP_UTF8);
  for (DWORD conoutHandleId : {STD_OUTPUT_HANDLE, STD_ERROR_HANDLE}) {
    HANDLE const hConsole = GetStdHandle(conoutHandleId);
    DWORD mode = 0;
    if (GetConsoleMode(hConsole, &mode)) {
      mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
      SetConsoleMode(hConsole, mode);
    }
  }
#endif
  CUDA_CHECK(cudaInitDevice(0, 0, 0));
  CUDA_CHECK(cudaSetDevice(0));
  testBvh();
}
