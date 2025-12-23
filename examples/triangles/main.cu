#include <driver_types.h>
#include "common.cuh"

#ifdef DMT_OS_WINDOWS
#  include <Windows.h>
#endif

// cudaMallocPitch!

namespace {
}  // namespace

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------



// UNICODE and _UNICODE always defined
#ifdef _WIN32
int wmain() {
#else
int main() {
#endif
#ifdef DMT_OS_WINDOWS
  SetConsoleOutputCP(CP_UTF8);
#endif

#if 0
  DeviceCamera h_cam;
  DeviceCamera* d_cam = nullptr;
  cudaMalloc((void**) &d_cam, sizeof(DeviceCamera));
  cudaMemcpy(d_cam, &h_cam, sizeof(DeviceCamera), cudaMemcpyHostToDevice);
  
  dim3 grid(1,1,1);
  dim3 block(32,32,1);
  //
  //block2dim-> buffer
  //
  //launch 
  raygenKernel<<<grid, block>>>(d_cam);
#endif
}
