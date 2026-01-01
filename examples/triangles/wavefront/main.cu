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
// Queue is full  → (back + 1) % queueCapacity == front % queueCapacity

// TODO: Optimize for vectorized load/store instructions

__device__ unsigned pushAggGMEM(void const* input, int inputSize,
                                int* reserve_back, int* publish_back,
                                int* front, void* queue, int queueCapacity) {
  assert(!(reinterpret_cast<intptr_t>(queue) & 0xf));  // 16 byte aligned
  assert((queueCapacity & (queueCapacity - 1)) == 0);
  cg::coalesced_group const g = cg::coalesced_threads();
  int const inputCount = g.size();
  int actualCount = inputCount;
  // best effert to estimate current capacity
  {
    int const currentBack = atomicAdd(reserve_back, 0);
    int const currentFront = atomicAdd(front, 0);
    int const used =
        (currentBack - currentFront + queueCapacity) & (queueCapacity - 1);
    int const freePlaces = queueCapacity - 1 - used;
    actualCount = min(freePlaces, actualCount);
  }

  // reservation
  int base = 0;
  if (g.thread_rank() == 0) {
#if __CUDA_ARCH__ >= 700
    base = __nv_atomic_add(reserve_back, actualCount, __NV_ATOMIC_ACQ_REL,
                           __NV_THREAD_SCOPE_DEVICE);
#else
    base = atomicAdd(reserve_back, actualCount);
    // volatile → compiler cannot remove or move it
    // "memory" clobber → compiler assumes memory is affected
    // ensure queue is in its last up-to-date state
    asm volatile("membar.gl;" ::: "memory");
#endif
  }
  base = g.shfl(base, 0);

  int const idx = base + g.thread_rank();
  int const slot = idx & (queueCapacity - 1);

  // probably not needed since we locked the queue?
  bool const pushed = g.thread_rank() < actualCount;
  if (pushed) {
    // all threads are submitting their own element, which is in LMEM
    memcpy(static_cast<uint8_t*>(queue) + slot * inputSize, input, inputSize);
  }

  // need to be visible to consumers
  if (g.thread_rank() == 0) {
    // asm volatile("membar.gl;" ::: "memory");
    __threadfence();
    atomicAdd(publish_back, actualCount);
  }

  // last time, everyone gets together
  g.sync();

  // compute warp uniform return value (mask of who did the push successfully)
  unsigned const mask = g.ballot(pushed);
#ifndef NDEBUG
  if (g.thread_rank() == 0) {
    // A contiguous LSB mask looks like: 00011111
    // Adding 1 flips it to: 00100000
    // ANDing them gives 0
    // Any hole breaks this property
    // no holes: must be 0b000...011...1
    assert((mask & (mask + 1)) == 0);
  }
#endif
  return mask;
}

__device__ unsigned popAggGMEM(void* output, int outputSize, int* publish_back,
                               int* front, void* queue, int queueCapacity) {
  assert(!(reinterpret_cast<intptr_t>(queue) & 0xf));  // 16 byte aligned
  assert((queueCapacity & (queueCapacity - 1)) == 0);

  cg::coalesced_group const g = cg::coalesced_threads();
  int outputCount = g.size();
  // best effort to estimate current capacity
  {
    int const currentBack = atomicAdd(publish_back, 0);
    int const currentFront = atomicAdd(front, 0);
    int const used =
        (currentBack - currentFront + queueCapacity) & (queueCapacity - 1);
    outputCount = min(used, outputCount);
  }

  int base = 0;
  if (g.thread_rank() == 0) {
    base = atomicAdd(front, outputCount);
  }
  outputCount = g.shfl(outputCount, 0);
  base = g.shfl(base, 0);
  if (outputCount == 0) {
    return 0;
  }

  int const idx = base + g.thread_rank();
  int const slot = idx & (queueCapacity - 1);

  // warp-coalesced copy from queue to output
  bool const popped = g.thread_rank() < outputCount;
  if (popped) {
    // all threads are submitting their own element, which is in LMEM
    memcpy(output, static_cast<uint8_t*>(queue) + slot * outputSize,
           outputSize);
  }

  g.sync();

  // compute warp uniform return value
  unsigned const mask = g.ballot(popped);
#ifndef NDEBUG
  if (g.thread_rank() == 0) {
    // A contiguous LSB mask looks like: 00011111
    // Adding 1 flips it to: 00100000
    // ANDing them gives 0
    // Any hole breaks this property
    // no holes: must be 0b000...011...1
    assert((mask & (mask + 1)) == 0);
  }
#endif
  return mask;
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
struct TestPayload {
  float a;
  float b;
};

template <typename T>
  requires std::is_trivial_v<T> && std::is_standard_layout_v<T>
struct QueueGMEM {
  T* queue;
  int* front;
  int* reserve_back;
  int* publish_back;
  int queueCapacity;

  __device__ __forceinline__ unsigned push(T const* elem) {
    return pushAggGMEM(elem, sizeof(T), reserve_back, publish_back, front,
                       queue, queueCapacity);
  }
  __device__ __forceinline__ unsigned pop(T* elem) {
    return popAggGMEM(elem, sizeof(T), publish_back, front, queue,
                      queueCapacity);
  }
};

__global__ void kQueueTest(QueueGMEM<TestPayload> iqueue,
                           QueueGMEM<float> oqueue, int* producerDone) {
  static int constexpr PUSH_COUNT_PER_BLOCK = 1000;
  __shared__ extern int SMEM[];

  // three kind of kernels:
  // 0 -> push. 1 -> peek and replace. 2 -> pop and push result
  cg::grid_group const grid = cg::this_grid();
  // ensure first type has bound number of pushes per block
  if (blockIdx.x % 3 == 0) {
    int& pushCount = SMEM[0];
    if (threadIdx.x == 0) {
      pushCount = PUSH_COUNT_PER_BLOCK;
    }
    __syncthreads();
  }

  if (blockIdx.x % 3 == 0) {
    int& pushCount = SMEM[0];
    int rem = 0;
    while ((rem = atomicSub(&pushCount, 1)) > 0) {
      auto g = cg::coalesced_threads();
      TestPayload const tp{.a = 2.f * rem + blockIdx.x / 3 + g.thread_rank(),
                           .b = 3.f * rem + blockIdx.x / 3 + g.thread_rank()};
      if (g.thread_rank() < rem) {
        printf("[%u] Block Type 0: [%u] pushing %f %f\n", blockIdx.x,
               g.thread_rank(), tp.a, tp.b);
        int mask = 0;
        do {
          mask = iqueue.push(&tp);
        } while (0 == (mask & (1 << g.thread_rank())));
      }
    }
    if (threadIdx.x + blockDim.x * blockIdx.x == 0) {
      atomicExch(producerDone, 1);
    }
  } else if (blockIdx.x % 3 == 1) {
#if 0
    TestPayload tp{};
    while (true) {
      int actualCount = 0;
      cg::coalesced_group const g = cg::coalesced_threads();
      if (iqueue.pop(&tp, &actualCount)) {
        // only those who popped
        bool active = g.thread_rank() < actualCount;
        if (g.thread_rank() == 0 && g.meta_group_rank() == 0) {
          printf("Block Type 1 Extracted: %f %f\n", tp.a, tp.b);
        }
        while (true) {
          bool pushed = true;
          if (active) {
            pushed = iqueue.push(&tp, &actualCount);
          }
          pushed = g.all(pushed);
          if (pushed) {
            break;
          }
          if (cg::coalesced_threads().thread_rank() == 0) {
            printf("Block Type 1: re-pushing FAILED\n");
          }
        }
      } else {
        bool done = false;
        cg::coalesced_group const exiting = cg::coalesced_threads();
        if (exiting.thread_rank() == 0) {
          int const back = atomicAdd(iqueue.back, 0);
          int const front = atomicAdd(iqueue.front, 0);
          done = back == front;
        }
        exiting.sync();
        done = exiting.shfl(done, 0);
        if (done && atomicAdd(producerDone, 0) == 1) {
          break;
        }
      }
    }
#endif
  } else {
    TestPayload tp{};
    while (true) {
      cg::coalesced_group const g = cg::coalesced_threads();
      if (unsigned const mask = iqueue.pop(&tp); mask) {
        if (1 << g.thread_rank() & mask) {
          if (g.thread_rank() == 0) {
            printf("[%u] Block Type 2: [%u] Inputs: %f %f\n", blockIdx.x,
                   g.thread_rank(), tp.a, tp.b);
          }
          float const result = tp.a * tp.b;
          int omask = 0;
          do {
            omask = oqueue.push(&result);
          } while (0 == (omask & (1 << g.thread_rank())));
        }
      } else {
        bool done = false;
        cg::coalesced_group const exiting = cg::coalesced_threads();
        if (exiting.thread_rank() == 0) {
          int const back = atomicAdd(iqueue.reserve_back, 0);
          int const front = atomicAdd(iqueue.front, 0);
          done = back == front;
          // printf("Block Type 2: popping FAILED Maybe Done? %d\n", done);
        }
        exiting.sync();
        done = exiting.shfl(done, 0);
        if (done && atomicAdd(producerDone, 0) == 1) {
          break;
        }
      }
    }
  }

  // just to test it out. No need
  grid.sync();
}

namespace {
// Note: probably the queue doesn't need cooperative kernel launches. Just
// to experiment
void testQueueBehaviour() {
  // do we actually support cooperative kernel launches? (Pascal: yes.)
  int supportsCoopLaunch = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&supportsCoopLaunch,
                                    cudaDevAttrCooperativeLaunch, 0));
  if (!supportsCoopLaunch) {
    std::cerr << "CooperativeLaunch attribute not supported" << std::endl;
    exit(1);
  }
  // ensure that blocks can be all resident in the GPU at once
  // - either use 1 block per SM (*)
  // - or use the Occupancy API to figure out max blocks
  cudaDeviceProp deviceProp{};
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
  // allocate queues
  static constexpr int QUEUE_CAPACITY = 1 << 13;
  int init[3] = {0, 0, 1};

  QueueGMEM<TestPayload> iqueue{};
  iqueue.queueCapacity = QUEUE_CAPACITY;
  CUDA_CHECK(
      cudaMalloc(&iqueue.queue, sizeof(TestPayload) * iqueue.queueCapacity));
  CUDA_CHECK(cudaMalloc(&iqueue.front, sizeof(int) * 3));
  CUDA_CHECK(
      cudaMemcpy(iqueue.front, init, sizeof(int) * 3, cudaMemcpyHostToDevice));
  iqueue.reserve_back = iqueue.front + 1;
  iqueue.publish_back = iqueue.front + 1;

  QueueGMEM<float> oqueue{};
  oqueue.queueCapacity = QUEUE_CAPACITY;
  CUDA_CHECK(cudaMalloc(&oqueue.queue, sizeof(float) * oqueue.queueCapacity));
  CUDA_CHECK(cudaMalloc(&oqueue.front, sizeof(int) * 3));
  CUDA_CHECK(
      cudaMemcpy(oqueue.front, init, sizeof(int) * 3, cudaMemcpyHostToDevice));
  oqueue.reserve_back = oqueue.front + 1;
  oqueue.publish_back = oqueue.front + 1;

  int* d_producerDone = nullptr;
  CUDA_CHECK(cudaMalloc(&d_producerDone, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_producerDone, 0, sizeof(int)));

  // cooperative launch
  void* args[] = {&iqueue, &oqueue, &d_producerDone};
#if 0
  dim3 const gridDim{
      min(static_cast<unsigned int>(deviceProp.multiProcessorCount), 6), 1, 1};
  assert(gridDim.x >= 3);
  dim3 const blockDim{64, 1, 1};
#else
  dim3 const gridDim{6, 1, 1};
  dim3 const blockDim{1, 1, 1};
#endif
  std::cout << "Launching queue kernel with " << blockDim.x
            << " threads per block. blocks: " << gridDim.x << std::endl;
  CUDA_CHECK(cudaLaunchCooperativeKernel(kQueueTest, gridDim, blockDim, args,
                                         sizeof(int), nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> h_output;
  h_output.resize(oqueue.queueCapacity);
  CUDA_CHECK(cudaMemcpy(h_output.data(), oqueue.queue,
                        sizeof(float) * oqueue.queueCapacity,
                        cudaMemcpyDeviceToHost));
  // Print out some
  int h_front = 0, h_back = 0;
  CUDA_CHECK(
      cudaMemcpy(&h_front, oqueue.front, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_back, oqueue.publish_back, sizeof(int),
                        cudaMemcpyDeviceToHost));

  int produced =
      (h_back - h_front + oqueue.queueCapacity) & (oqueue.queueCapacity - 1);

  std::cout << "Produced output elements: " << produced << std::endl;

  int toPrint = std::min(produced, 10);
  for (int i = 0; i < toPrint; ++i) {
    int idx = (h_front + i) & (oqueue.queueCapacity - 1);
    std::cout << "oqueue[" << i << "] = " << h_output[idx] << std::endl;
  }
}

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
  CUDA_CHECK(cudaInitDevice(0, 0, 0));
  CUDA_CHECK(cudaSetDevice(0));
  testQueueBehaviour();
#if 0
  wavefrontMain();
#endif
}
