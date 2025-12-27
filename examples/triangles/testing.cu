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
// ---------------------------------------------------------------------------
// Test Halton Owen: Runner
// ---------------------------------------------------------------------------

struct TestHoltonOwenVals {
  float2 valGet2D;
  float2 valGetPixel2D;
  float valGet1D;
};

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
    //h_ho.computeParams();
  }
  CudaTimer timer;
  timer.begin();
  testForHaltonOwenKernel<<<nBlocks, nThreads>>>(d_rng);
  float ms = timer.end();
  cudaFree(d_rng);
}