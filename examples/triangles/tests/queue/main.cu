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
#include "cuda-core/queue.cuh"

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

struct TestPayload {
  float a;
  float b;
};

namespace cg = cooperative_groups;

__global__ void kQueueTest(QueueGMEM<TestPayload> iqueue,
                           QueueGMEM<float> oqueue, int* producerDone) {
  static int constexpr PUSH_COUNT_PER_BLOCK = 1000;
  __shared__ extern int SMEM[];

  // three kind of kernels:
  // 0 -> push. 1 -> peek and replace. 2 -> pop and push result
  cg::grid_group const grid = cg::this_grid();
  int const producerBlocks = gridDim.x / 3;
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
    int ticket = 0;
    while ((ticket = atomicAggDecSaturate(&pushCount)) >= 0) {
      auto g = cg::coalesced_threads();
      TestPayload const tp{
          .a = 2.f * ticket + blockIdx.x / 3 + g.thread_rank(),
          .b = 3.f * ticket + blockIdx.x / 3 + g.thread_rank()};
      printf("[%u] Block Type 0: [%u] pushing %f %f\n", blockIdx.x,
             g.thread_rank(), tp.a, tp.b);
      int mask = 0;
      bool notDone = true;
      do {
        auto gg = cg::coalesced_threads();
        mask = iqueue.push(&tp);
        notDone = 0 == (mask & (1 << gg.thread_rank()));
        if (gg.thread_rank() == 0) {
          printf("[%u] Block Type 0: [%u] push status: %d\n", blockIdx.x,
                 gg.thread_rank(), notDone);
        }
        if (pushCount < 0) {
          break;
        }
      } while (notDone);
    }

    if (threadIdx.x == 0) {
      printf("[%u] Producer Dying\n", blockIdx.x);
    }

    if (threadIdx.x == 0) {
      while (atomicAdd(iqueue.reserve_back, 0) !=
             atomicAdd(iqueue.publish_back, 0)) {
        // busy wait
      }
      atomicAdd(producerDone, 1);
    }
  } else if (blockIdx.x % 3 == 1) {
  } else {
    TestPayload tp{};
    while (true) {
      cg::coalesced_group const g = cg::coalesced_threads();
      if (unsigned const mask = iqueue.pop(&tp); mask) {
        printf("[%u] Block Type 2: [%u] Mask: 0x%x \n", blockIdx.x,
               g.thread_rank(), mask);
        if (1 << g.thread_rank() & mask) {
          if (g.thread_rank() == 0) {
            printf("[%u] Block Type 2: [%u] Inputs: %f %f\n", blockIdx.x,
                   g.thread_rank(), tp.a, tp.b);
          }
          float const result = tp.a * tp.b;
          int omask = 0;
          bool notDone = true;
          do {
            cg::coalesced_group const gg = cg::coalesced_threads();
            omask = oqueue.push(&result);
            notDone = 0 == (omask & (1 << gg.thread_rank()));
          } while (notDone);
        }
      } else {
        bool done = false;
        cg::coalesced_group const exiting = cg::coalesced_threads();
        if (exiting.thread_rank() == 0) {
          int const back = atomicAdd(iqueue.publish_back, 0);
          int const front = atomicAdd(iqueue.front, 0);
          int const pd = atomicAdd(producerDone, 0);
          done = back == front && pd == producerBlocks;
          // stuck here, producer 3 absent
          printf(
              "[%u] Block Type 2: [%u] popping FAILED Maybe Done? %d. pd: %d "
              "of %d\n",
              blockIdx.x, g.thread_rank(), done, pd, producerBlocks);
        }
        exiting.sync();
        done = exiting.shfl(done, 0);
        if (done) {
          if (cg::coalesced_threads().thread_rank() == 0) {
            printf("[%u] consumer dying\n", blockIdx.x);
          }
          break;
        }
      }
    }
  }

  // just to test it out. No need
  grid.sync();
}

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
  int init[4] = {0, 0, 0, 0};

  QueueGMEM<TestPayload> iqueue{};
  iqueue.queueCapacity = QUEUE_CAPACITY;
  CUDA_CHECK(
      cudaMalloc(&iqueue.queue, sizeof(TestPayload) * iqueue.queueCapacity));
  CUDA_CHECK(cudaMalloc(&iqueue.front, sizeof(int) * 4));
  CUDA_CHECK(
      cudaMemcpy(iqueue.front, init, sizeof(int) * 4, cudaMemcpyHostToDevice));
  iqueue.reserve_back = iqueue.front + 1;
  iqueue.publish_back = iqueue.front + 2;
  iqueue.queue_open = iqueue.front + 3;

  QueueGMEM<float> oqueue{};
  oqueue.queueCapacity = QUEUE_CAPACITY;
  CUDA_CHECK(cudaMalloc(&oqueue.queue, sizeof(float) * oqueue.queueCapacity));
  CUDA_CHECK(cudaMalloc(&oqueue.front, sizeof(int) * 4));
  CUDA_CHECK(
      cudaMemcpy(oqueue.front, init, sizeof(int) * 4, cudaMemcpyHostToDevice));
  oqueue.reserve_back = oqueue.front + 1;
  oqueue.publish_back = oqueue.front + 2;
  oqueue.queue_open = oqueue.front + 3;

  int* d_producerDone = nullptr;
  CUDA_CHECK(cudaMalloc(&d_producerDone, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_producerDone, 0, sizeof(int)));

  // cooperative launch
  void* args[] = {&iqueue, &oqueue, &d_producerDone};
#if 1
  dim3 const gridDim{
      min(static_cast<unsigned int>(deviceProp.multiProcessorCount), 6), 1, 1};
  assert(gridDim.x >= 3);
  dim3 const blockDim{256, 1, 1};
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
  testQueueBehaviour();
}
