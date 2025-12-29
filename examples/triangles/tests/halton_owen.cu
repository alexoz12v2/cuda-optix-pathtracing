#include "halton_owen.cuh"

#include <cmath>
#include <numbers>

// ---------------------------------------------------------------------------
// Test Halton Owen: Kernel
// ---------------------------------------------------------------------------
__global__ void testForHaltonOwenKernel(DeviceHaltonOwen* d_haltonOwen) {
  uint32_t const mortonStart = blockIdx.x * blockDim.x + threadIdx.x;
  MortonLayout2D const layout = mortonLayout(gridDim.x * blockDim.x, 1);
  DeviceHaltonOwen& warpRng = d_haltonOwen[mortonStart / warpSize];
  DeviceHaltonOwenParams const params =
      warpRng.computeParams(layout.cols, layout.rows);
}

void testForHaltonOwenRunner() {
  int device = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  int nSM = prop.multiProcessorCount;
  int nThreadPerBlock = prop.maxThreadsPerBlock;
  int maxThreadPerSM = prop.maxThreadsPerMultiProcessor;
  std::wcout << "[DEBUG-INFO]: device: " << device << std::endl;
  std::wcout << "\tSM: " << nSM << std::endl;
  std::wcout << "\tMax threas per block: " << nThreadPerBlock << std::endl;
  std::wcout << "\tTotal constant memory: " << prop.totalConstMem << " bytes"
             << std::endl;
  std::wcout << "\tMax threas per SM: " << maxThreadPerSM << " bytes"
             << std::endl;
  std::wcout << "Total number of registers available per block: "
             << prop.regsPerBlock << std::endl;
  // A optimistic approssimation for the maximum occupancy: we assumed that are
  // used less regs per block, grid: x and block: x
  int nThreads = maxThreadPerSM / 2;
  int nBlocks = nSM * 2;
  int nWarps = nBlocks * nThreads / WARP_SIZE;
  std::vector<DeviceHaltonOwen> h_rng(nWarps);
  DeviceHaltonOwen* d_rng = nullptr;
  CUDA_CHECK(cudaMalloc(&d_rng, nWarps * sizeof(DeviceHaltonOwen)));
  CUDA_CHECK(cudaMemcpy(d_rng, h_rng.data(), nWarps * sizeof(DeviceHaltonOwen),
                        cudaMemcpyHostToDevice));
  std::vector<TestHoltonOwenVals> h_rngParams(nWarps);

  for (DeviceHaltonOwen& h_ho : h_rng) {
    // h_ho.computeParams();
  }
  CudaTimer timer;
  timer.begin();
  testForHaltonOwenKernel<<<nBlocks, nThreads>>>(d_rng);
  float ms = timer.end();
  cudaFree(d_rng);
}
