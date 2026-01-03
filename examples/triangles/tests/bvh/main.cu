
#include "cuda-core/types.cuh"
#include "cuda-core/common_math.cuh"
#include "cuda-core/host_scene.cuh"

#ifdef DMT_OS_WINDOWS
#  include <Windows.h>
#elif defined(DMT_OS_LINUX)
#  include <unistd.h>
#  include <limits.h>
#endif

#include <algorithm>
#include <array>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <span>

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

inline float3 bbCentroid(AABB const& a) { return (min + max) * 0.5f; }

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

class IBVHBuildNode {
 public:
  [[nodiscard]] bool isLeaf() const { return do_isLeaf(); }
  [[nodiscard]] AABB bounds() const { return do_bounds(); }
  [[nodiscard]] float surfaceArea() const { return ::surfaceArea(bounds()); }
  virtual ~IBVHBuildNode() = default;

 protected:
  virtual bool do_isLeaf() const = 0;
  virtual AABB do_bounds() const = 0;
};

enum class ESplitAxis : int { X = 0, Y = 1, Z = 2 };
inline operator int(ESplitAxis a) { return static_cast<int>(a); }
inline operator ESplitAxis(int a) {
  assert(a >= 0 && a <= 2);
  return static_cast<ESplitAxis>(a);
}
inline bool operator<=(int a, ESplitAxis b) {
  int bi = static_cast<int>(b);
  return a <= bi;
}

class BVHBuildInteriorNode final : public IBVHBuildNode {
 public:
  explicit BVHBuildInteriorNode(std::unique_ptr<IBVHBuildNode> left,
                                std::unique_ptr<IBVHBuildNode> right, int axis)
      : m_left(std::move(left)),
        m_right(std::move(right)),
        m_axis(static_cast<ESplitAxis>(axis)) {}

  [[nodiscard]] ESplitAxis splitAxis() const { return m_axis; }

 protected:
  bool do_isLeaf() const override { return false; }
  AABB do_bounds() const override {
    return bbUnion(m_left->bounds(), m_right->bounds());
  }

 private:
  std::unique_ptr<IBVHBuildNode> m_left;
  std::unique_ptr<IBVHBuildNode> m_right;
  ESplitAxis m_axis;
};

class BVHBuildLeafTriangles final : public IBVHBuildNode {
 public:
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

inline float3 triCentroid(Triangle const& tri) {
  return (tri.v0 + tri.v1 + tri.v2) / 3;
}

inline std::array<float, 2> leafCostAndSurfaceArea(
    std::span<Triangle> triangles, float rootSurfaceArea, float triCost = 1) {
  float const A_b =
      surfaceArea(std::accumulate(triangles.begin(), triangles.end(), AABB(),
                                  [](AABB const& acc, Triangle const& tri) {
                                    return bbUnion(acc, ::bounds(tri));
                                  }));
  return {
      A_b / rootSurfaceArea * static_cast<float>(triangles.size()) * triCost,
      A_b};
}

inline float SAH(float const setSurfaceArea, int leftCount, float leftArea,
                 int rightCount, float rightArea, float triCost = 1,
                 float boxCost = 1) {
  return 2 * boxCost + leftArea / setSurfaceArea * leftCount * triCost +
         rightArea / setSurfaceArea * rightCount * triCost;
}

// SAH, Top Down, binary BVH construction by minimizing cost function at
// each step
// - The number of bins can also be reduced during
//   construction, i.e., based on the current tree depth. The full binning
//   resolution is only required for the top of the tree, where picking
//   the best split position matters the mos
std::unique_ptr<IBVHBuildNode> BVHBuildCentroidBasedSAH(
    float rootSurfaceArea, std::span<Triangle> triangles, int bins,
    int leafThreshold) {
  assert(rootSurfaceArea > 0 && !triangles.empty() && bins > 0 &&
         (bins & (bins - 1)) == 0 && leafThreshold > 0 &&
         (leafThreshold & (leafThreshold - 1)) == 0);
  if (triangles.size() <= leafThreshold) {
    return std::make_unique<BVHBuildLeafTriangles>(triangles);
  }
  auto [bestCost, setSurfaceArea] =
      leafCostAndSurfaceArea(triangles, rootSurfaceArea);
  int bestAxis = -1;
  int bestSplitPositions = -1;
  // for each axis
  for (int axis = static_cast<int>(ESplitAxis::X); axis <= ESplitAxis::Z;
       ++axis) {
    // sort based on centroid order in the current axis
    std::ranges::sort(triangles,
                      [axis](Triangle const& first, Triangle const& second) {
                        return float3_at(triCentroid(first), axis) <
                               float3_at(triCentroid(second), axis);
                      });
    float const minC = float3_at(triCentroid(triangles.front()), axis);
    float const maxC = float3_at(triCentroid(triangles.back()), axis);
    float const extent = maxC - minC;

    // for each bin
    for (int bin = 0; bin < bins; ++bin) {
      float const split = minC + extent * (bin + 1) / bins;

      int const firstRight =
          std::ranges::find_if(triangles,
                               [axis, split](Triangle const& tri) {
                                 return float3_at(bbCentroid(bounds(tri)),
                                                  axis) >= split;
                               }) -
          triangles.begin();
      if (firstRight >= triangles.size() || firstRight == 0) {
        continue;
      }
      // surface area for the two partitions and SAH
      int const leftCount = firstRight;
      int const rightCount = static_cast<int>(triangles.size()) - leftCount;
      float const leftArea = surfaceArea(
          std::accumulate(triangles.begin(), triangles.begin() + leftCount,
                          AABB(), [](AABB const& acc, Triangle const& tri) {
                            return bbUnion(acc, bounds(tri));
                          }));
      float const rightArea = surfaceArea(
          std::accumulate(triangles.begin() + leftCount, triangles.end(),
                          AABB(), [](AABB const& acc, Triangle const& tri) {
                            return bbUnion(acc, bounds(tri));
                          }));
      float const cost =
          SAH(setSurfaceArea, leftCount, leftArea, rightCount, rightArea);
      if (cost < bestCost) {
        bestCost = cost;
        bestAxis = axis;
        bestSplitPositions = firstRight;
      }
    }
  }

  // if best is leaf, do it
  if (bestAxis == -1) {
    return std::make_unique<BVHBuildLeafTriangles>(triangles);
  }
  // otherwise sort at best axis
  std::ranges::sort(triangles,
                    [bestAxis](Triangle const& first, Triangle const& second) {
                      return float3_at(triCentroid(first), bestAxis) <
                             float3_at(triCentroid(second), bestAxis);
                    });
  // make inner node with recursion
  int const nextBins = std::max(1, bins >> 1);
  return std::make_unique<BVHBuildInteriorNode>(
      BVHBuildCentroidBasedSAH(
          rootSurfaceArea,
          std::span(triangles.begin(), triangles.begin() + bestSplitPositions),
          nextBins, leafThreshold),
      BVHBuildCentroidBasedSAH(
          rootSurfaceArea,
          std::span(triangles.begin() + bestSplitPositions, triangles.end()),
          nextBins, leafThreshold),
      bestAxis);
}

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
