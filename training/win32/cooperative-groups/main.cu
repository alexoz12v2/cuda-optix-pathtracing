// std stuff
#include <iostream>

// Windows Stuff
#include "Windows.h"
#include "ShlObj.h"
#include "Objbase.h"

// cuda stuff
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// our stuff
#include "the-macros.h"
#include "example.h"

using namespace dmt;

/// ## About CUDA Runtime and Context Management
/// A [Context](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#context)
/// is the CUDA Equivalent of a process. All resources and modules allocated for the process are associated to a
/// context.
/// An Implicit Context will be created for each device called Primary Context whenever you call `cudaInitDevice`



using Point3f = Vector3f;

struct Vector3f
{
    float x, y, z;
    __device__ __host__ Vector3f(float x, float y, float z):x(x), y(y), z(z){}
    __device__ __host__ float& operator[](int i)
    {
        return ((&x)[i]);
    }

    __device__ __host__ float& operator[](int i)
    {
        return ((&x)[i]);
    }

    __device__ __host__ const float& operator[](int i) const
    {
        return ((&x)[i]);
    }
};

struct Point2f
{
    float x, y;
    __device__ __host__ Point2f(float x, float y):x(x), y(y){}
    __device__ __host__ float& operator[](int i)
    {
        return ((&x)[i]);
    }

    __device__ __host__ float& operator[](int i)
    {
        return ((&x)[i]);
    }

    __device__ __host__ const float& operator[](int i) const
    {
        return ((&x)[i]);
    }
};

struct DeviceCamera 
{
    int resH = 256;
    int resW = 256;
    int samplesPP = 32; 
    Vector3f position{0.f, 0.f, 0.f};
};

struct CameraSample
{

    Point2f pointFilm{0.f, 0.f};
    Point2f pointLens{0.f, 0.f};
    float time = 0.f;
    float filterWeight = 0.f;

    __device__ __host__ CameraSample(){};
};

struct RaysPool
{
    Vector3f* pDirVec;
    Point3f* pOrgVec; 
    int dim = 0;
};

__device__ void generate_camera_ray_mega(cooperative_groups::thread_group g)
{
    int lane = g.thread_rank();
    if (lane == 0)
    {
        //gen ray
    }
    g.sync();
}

//__constant__ int d_lookup;

__global__ void megakernel(DeviceCamera* dp_dc)
{
    int nPixel = dp_dc->resH*dp_dc->resW;
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    cooperative_groups::thread_block_tile<32> warp = cooperative_groups::tiled_partition<32>(block);
    for(int i = threadIdx.x; i < nPixel; i += gridDim.x*blockDim.x)
    {
        //sampling 
        CameraSample cs;
        
    }
} 

int wmain()
{
    // - Setup console properly such that ANSI escape codes work
    for (HANDLE out : {GetStdHandle(STD_OUTPUT_HANDLE), GetStdHandle(STD_ERROR_HANDLE)})
    {
        DWORD mode = 0;
        GetConsoleMode(out, &mode);
        mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        mode |= DISABLE_NEWLINE_AUTO_RETURN;
        SetConsoleMode(out, mode);
    }
    std::ios::sync_with_stdio();

    //get device property 
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int nSM = prop.multiProcessorCount;
    int nThreadPerBlock = prop.maxThreadsPerBlock;

    DeviceCamera h_dc;
    DeviceCamera* dp_dc;
    cudaMallocHost(&dp_dc, sizeof(DeviceCamera));
    cudaMemcpy(dp_dc, &h_dc, sizeof(DeviceCamera), cudaMemcpyHostToDevice);
    //cudaMemcoyToSymnbol
    //Inital assumption: number of samples cannot exceed the number of threads per block
    //cycling for obtain more samples???
    megakernel<<<nSM, nThreadPerBlock>>>(dp_dc);
    // - Print some colored stuff
    std::cout << ANSI_RED "Hello Beautiful World" ANSI_RST << std::endl;

    // initialize COM Apartment for this process
    HRESULT const res = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (!SUCCEEDED(res))
        win32::printResultAndExitProcess(res);

    _1basics::printCudaCapableDevices();
}


