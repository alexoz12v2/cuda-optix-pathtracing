#ifndef __NVCC__
    #define __NVCC__
#endif

#include "cuda-queue.h"

extern "C" __global__ void saxpy_grid_stride(int n, float a, float const* x, float* y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        y[i] = a * x[i] + y[i];
    }
}

extern "C" __global__ void kqueueDouble(dmt::ManagedQueue<int>* queue, dmt::ManagedQueue<int>* queue1)
{
    int num = 0;
    if (!queue->popDevice(&num))
        return;
    queue1->pushDevice(num * 2);
}

extern "C" __global__ void kmmqDouble(dmt::ManagedMultiQueue<double, int>* queue, dmt::ManagedMultiQueue<double, int>* queue1)
{
    int    num  = 0;
    double fnum = 0.0;
    if (!queue->popDevice(&fnum, &num))
        return;
    queue1->pushDevice(fnum * 2, num * 3);
}
