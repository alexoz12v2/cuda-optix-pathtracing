#include <iostream>
#include <cuda_runtime.h>

__global__ void saxpy(int n, float a, float const* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

#ifdef DMT_OS_WINDOWS
int wmain()
#else
int main()
#endif
{
  int N = 1 << 20;  // 1M elements
  float a = 2.0f;

  // Allocate host memory
  float* h_x = new float[N];
  float* h_y = new float[N];

  for (int i = 0; i < N; i++) {
    h_x[i] = 1.0f;
    h_y[i] = 2.0f;
  }

  // Allocate device memory
  float *d_x, *d_y;
  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));

  // Copy inputs to device
  cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  saxpy<<<numBlocks, blockSize>>>(N, a, d_x, d_y);

  cudaDeviceSynchronize();

  // Copy result back
  cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Simple validation
  std::cout << "y[0] = " << h_y[0] << " (expected 4.0)" << std::endl;

  // Cleanup
  cudaFree(d_x);
  cudaFree(d_y);
  delete[] h_x;
  delete[] h_y;

  return 0;
}