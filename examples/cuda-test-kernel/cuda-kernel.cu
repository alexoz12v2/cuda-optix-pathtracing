extern "C" __global__ void saxpy_grid_stride(int n, float a, float const* x, float* y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        y[i] = a * x[i] + y[i];
    }
}