
#include "cuda-core/types.cuh"
#include "cuda-core/common_math.cuh"

#ifdef DMT_OS_WINDOWS
#  include <Windows.h>
#elif defined(DMT_OS_LINUX)
#  include <unistd.h>
#  include <limits.h>
#endif

#include <iostream>

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
    }
  }
#endif
  CUDA_CHECK(cudaInitDevice(0, 0, 0));
  CUDA_CHECK(cudaSetDevice(0));
  testBvh();
}
