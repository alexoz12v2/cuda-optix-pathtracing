#include "common.cuh"

__device__ CameraSample raygen() { return {}; }

__global__ void raygenKernel(DeviceCamera const* d_cam,
                             DeviceHaltonOwen* d_haltonOwen,
                             CameraSample* d_samples) {
  // transforms
  // grid-style loop (2D)
  // - getCameraRay
  // - write to buffer
}

// 128-byte boundary aligned staring address
// 0x____'____'____'____'____'____'____'__00
// 0x____'____'____'____'____'____'____'__80

// 32-byte boundary aligned staring address
// 0x____'____'____'____'____'____'____'__00
// 0x____'____'____'____'____'____'____'__20
// 0x____'____'____'____'____'____'____'__40
// 0x____'____'____'____'____'____'____'__60
// 0x____'____'____'____'____'____'____'__80
// 0x____'____'____'____'____'____'____'__a0
// 0x____'____'____'____'____'____'____'__c0
// 0x____'____'____'____'____'____'____'__e0