#include "cuda-core/types.cuh"
#include "cuda-core/bsdf.cuh"
#include "cuda-core/rng.cuh"
#include "cuda-core/common_math.cuh"
#include "cuda-core/extra_math.cuh"
#include "cuda-core/host_scene.cuh"
#include "cuda-core/host_utils.cuh"
#include "cuda-core/light.cuh"
#include "cuda-core/morton.cuh"
#include "cuda-core/shapes.cuh"
#include "cuda-core/kernels.cuh"

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>

#include <cassert>

#ifdef DMT_OS_WINDOWS
#  include <Windows.h>
#elif defined(DMT_OS_LINUX)
#  include <unistd.h>
#  include <limits.h>
#endif

namespace cg = cooperative_groups;

__device__ int atomicAggIncWrap(int* ptr, int const modulus) {
  cg::coalesced_group g = cg::coalesced_threads();
  int prev = 0;

  // elect the first active thread to perform atomic add
  if (g.thread_rank() == 0) {
    prev = atomicAdd(ptr, g.size());
  }

  // broadcast previous value within the warp
  // and add each active thread’s rank to it
  prev = g.thread_rank() + g.shfl(prev, 0);
  return prev % modulus;
}

__device__ int atomicAggDecWrap(int* ptr, int const modulus) {
  cg::coalesced_group g = cg::coalesced_threads();
  int prev = 0;

  // elect the first active thread to perform atomic add
  if (g.thread_rank() == 0) {
    prev = atomicSub(ptr, g.size());
  }

  // broadcast previous value within the warp
  // and add each active thread’s rank to it
  prev = g.thread_rank() + g.shfl(prev, 0);
  return prev % modulus;
}

// Synchronization → PTX instruction mapping
// __threadfence() ≈ membar.gl
// __threadfence_block() ≈ membar.cta
// __threadfence_system() ≈ membar.sys

// now back starts from 0 and front starts from 0
// Queue is empty → back == front
// Queue is full  → (back + 1) % queueCapacity == front

/// @param input buffer of input elements. can be any type of memory
/// @param inputSize  bytes of an element
/// @param inputCount count of input element
/// @param back GMEM pointer to atomically managed counter
/// @param front GMEM pointer to atomically managed counter
/// @param ready GMEM pointer to atomically managed boolean flag
/// @param queue GMEM work queue
/// @param queueCapacity maximum number of inputSize elements
/// @return whether each thread successfully inserted or not
__device__ bool pushAggGMEM(void const* input, int inputSize, int inputCount,
                            int* back, int* front, int* ready, void* queue,
                            int queueCapacity) {
  assert(!(reinterpret_cast<intptr_t>(input) & 0xf));  // 16 byte aligned
  assert(!(reinterpret_cast<intptr_t>(queue) & 0xf));  // 16 byte aligned
  assert(!(inputSize & 0xf));                          // 16 byte multiple
  assert((queueCapacity & (queueCapacity - 1)) == 0);
  int const modulusMask = queueCapacity - 1;
  cg::coalesced_group const g = cg::coalesced_threads();
  bool warpReady = false;
  // handle multiple producers
  do {
    if (g.thread_rank() == 0) {
#if __CUDA_ARCH__ >= 700
      warpReady = __nv_atomic_exchange_n(ready, 0, __NV_ATOMIC_ACQ_REL,
                                         __NV_THREAD_SCOPE_DEVICE) == 1;
#else
      // This is fugly 😡
      warpReady = atomicCAS(ready, 1, 0) == 1;
      // volatile → compiler cannot remove or move it
      // "memory" clobber → compiler assumes memory is affected
      asm volatile("membar.gl;" ::: "memory");
#endif
    }
    g.sync();
    warpReady = g.shfl(warpReady, 0);
  } while (warpReady);

  int const currentBack = atomicAdd(back, 0);
  int const currentFront = atomicAdd(front, 0);
  int const freePlaces =
      (currentFront - currentBack + queueCapacity) & (queueCapacity - 1);
  // number of elements to insert
  int const actualCount = min(freePlaces, inputCount);
  bool const valid = g.thread_rank() < actualCount;
  if (valid) {
    int const myPlace = (currentBack + g.thread_rank()) & modulusMask;
    memcpy(static_cast<uint8_t*>(queue) + myPlace * inputSize,
           static_cast<uint8_t const*>(input) + g.thread_rank() * inputSize,
           inputSize);
  }

  if (g.thread_rank() == 0) {
    atomicAdd(back, actualCount);
#if __CUDA_ARCH__ >= 700
    __nv_atomic_store_n(ready, 1, __NV_ATOMIC_RELEASE,
                        __NV_THREAD_SCOPE_DEVICE);
#else
    // equivalent of __threadfence
    asm volatile("membar.gl;" ::: "memory");
    atomicExch(ready, 1);
#endif
  }

  // last time, everyone gets together
  g.sync();
  return valid;
}

/// @param output buffer to receive elements. can be any type of memory
/// @param outputSize bytes of an element
/// @param outputCount number of elements to pop
/// @param back GMEM pointer to producer counter
/// @param front GMEM pointer to consumer counter
/// @param ready GMEM pointer to atomically managed boolean flag
/// @param queue GMEM work queue
/// @param queueCapacity maximum number of outputSize elements
/// @return whether each thread successfully popped an element
__device__ bool popAggGMEM(void* output, int outputSize, int outputCount,
                           int* back, int* front, int* ready, void* queue,
                           int queueCapacity) {
  assert(!(reinterpret_cast<intptr_t>(output) & 0xf));  // 16 byte aligned
  assert(!(reinterpret_cast<intptr_t>(queue) & 0xf));   // 16 byte aligned
  assert(!(outputSize & 0xf));                          // 16 byte multiple
  assert((queueCapacity & (queueCapacity - 1)) == 0);

  int const modulusMask = queueCapacity - 1;
  cg::coalesced_group const g = cg::coalesced_threads();
  bool warpReady = false;

  // acquire lock (same as push)
  do {
    if (g.thread_rank() == 0) {
#if __CUDA_ARCH__ >= 700
      warpReady = __nv_atomic_exchange_n(ready, 0, __NV_ATOMIC_ACQ_REL,
                                         __NV_THREAD_SCOPE_DEVICE) == 1;
#else
      warpReady = atomicCAS(ready, 1, 0) == 1;
      asm volatile("membar.gl;" ::: "memory");
#endif
    }
    g.sync();
    warpReady = g.shfl(warpReady, 0);
  } while (warpReady);

  // read current front/back
  int const currentFront = atomicAdd(front, 0);
  int const currentBack = atomicAdd(back, 0);

  // compute number of available elements
  int const available =
      (currentBack - currentFront + queueCapacity) & (queueCapacity - 1);
  int const actualCount = min(available, outputCount);
  bool const valid = g.thread_rank() < actualCount;

  // warp-coalesced copy from queue to output
  if (valid) {
    int const myPlace = (currentFront + g.thread_rank()) & modulusMask;
    memcpy(static_cast<uint8_t*>(output) + g.thread_rank() * outputSize,
           static_cast<uint8_t*>(queue) + myPlace * outputSize, outputSize);
  }

  // advance front by actualCount
  if (g.thread_rank() == 0) {
    atomicAdd(front, actualCount);
#if __CUDA_ARCH__ >= 700
    __nv_atomic_store_n(ready, 1, __NV_ATOMIC_RELEASE,
                        __NV_THREAD_SCOPE_DEVICE);
#else
    asm volatile("membar.gl;" ::: "memory");
    atomicExch(ready, 1);
#endif
  }

  g.sync();
  return valid;
}

// path state
// - pixel coords, sample index
// - samples per pixel
// - depth
// - throughput
// - transmission count
// - last was transmission
// - radiance
// - any specular bounces
// - last BSDF PDF
struct PathState {
  int pixelCoordX;
  int pixelCoordY;
  int sampleIndex;
  int spp;
  int depth;
  float3 throughput;
  float3 L;
  float lastBsdfPdf;
  int16_t transmissionCount;
  int8_t lastBouncetransmission;
  int8_t anySpecularBounces;
};
static_assert(sizeof(PathState) == 52);

// Kernels:
// - raygen (pixel coords, sample index -> ray) (camera, RNG)
// - anyhit (ray, path state, scene) -> HitResult
//   - intersect
// - closesthit (ray, path state, scene) -> HitResult
//   - intersect
// - miss (path state)
// - shade (path state, HitResult)

// ---------------------------------------------------------------------------
// Payload types
// ---------------------------------------------------------------------------

namespace {

void wavefrontMain() {}

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
  wavefrontMain();
}
