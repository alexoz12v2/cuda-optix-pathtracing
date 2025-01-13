#include "dummy.h"
// #include <cudashared/cudashared.h>

#include <cassert>

__global__ void mulKernel(float* res) {//
    // this doesn't compile. Device Linking needs to happen where device functions are consumed, but 
    // this function has already been linked to a dependency shared library, so you cannot use it
    // res[threadIdx.x] = dmt::test::multiply(2.f, 3.f);
    res[threadIdx.x] = 2.f * 3.f;
}

namespace dmt::test {

void multiplyArr(float* ptr)
{
    cudaError_t cudaStatus;
    float*      d_ptr = nullptr;

    cudaStatus = cudaMalloc(&d_ptr, sizeof(float) * 32);
    assert(cudaStatus == ::cudaSuccess);

    mulKernel<<<1, 32>>>(d_ptr);
    assert(cudaStatus == ::cudaSuccess);
    cudaStatus = cudaDeviceSynchronize();
    assert(cudaStatus == ::cudaSuccess);

    cudaStatus = cudaMemcpy(ptr, d_ptr, 32 * sizeof(float), ::cudaMemcpyDeviceToHost);
    assert(cudaStatus == ::cudaSuccess);
}

}