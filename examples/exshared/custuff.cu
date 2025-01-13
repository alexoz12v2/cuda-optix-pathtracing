#include "custuff.h"

#include <iostream>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>

class DMT_INTERFACE IStuff
{
public:
    __host__ __device__ IStuff() {}
    __host__            __device__ virtual ~IStuff() {}

    __host__ __device__ virtual int32_t giveMeNumber() = 0;
};

class Three : public IStuff
{
public:
    __host__ __device__ Three() : IStuff()
    { //
#if !defined(__CUDA_ARCH__)
        std::cout << "Constructor inside host" << std::endl;
#endif
    }

    __host__ __device__ ~Three() override
    {
#if !defined(__CUDA_ARCH__)
        std::cout << "Destructor called inside host" << std::endl;
        if (m_called)
            std::cout << "We called number" << std::endl;
#endif
    }
    __host__ __device__ int32_t giveMeNumber() override
    {
        m_called = true;
        return 3;
    }

private:
    bool m_called = false;
};

static __managed__ bool    s_deviceHasAccess = false;
static __managed__ int32_t s_number          = 0;

// A virtual function call on device code gives MMU Fault if the vtable doesn't reside in device memory
// the vtable is allocated on the device if the object is constructed on the device (for the first time),
// and on the host if it is constructed on the host, even if the underlying backing memory for the class is managed memory
// a Hack to force usage of virtual functions for a host constructed object seems to involve a call to
// the copy constructor on a __managed__ object inside the device code, to force vtable allocation upon
// the first usage of the class. This is horrible and I would prefer to allocate a vtable manually. Keeping
// this for reference
template <typename T>
__device__ void fixVirtualPointers(T* other)
{
    T temp = T(*other);              // object-copy moves the "guts" of the object w/o changing vtable
    memcpy(other, &temp, sizeof(T)); // pointer copy seems to move vtable
}

static __global__ void testVirtualFunctionsFromManagedObjectWithHostCopyControl(IStuff* stuff)
{
    int32_t gid = dmt::globalThreadIndex();
    if (gid == 0)
    { // you can't use dynamic_cast on device code. dam
        fixVirtualPointers(reinterpret_cast<Three*>(stuff));
        s_number = stuff->giveMeNumber();
    }
    __syncthreads();
}

static __global__ void testResource(dmt::DynaArray& arr)
{
    int32_t device;
    cudaGetDevice(&device);
    int32_t gid = dmt::globalThreadIndex();
    if (gid == 0)
    {
        bool res          = arr.resource()->deviceHasAccess(device);
        s_deviceHasAccess = res;
    }
    __syncthreads();
}

static __global__ void fillKernel(dmt::DynaArray& arr, float val, size_t maxSize)
{
    auto c = dmt::categoryOf(arr.resource());
    if (arr.lockForWrite()) // 1 insertion per warp
    {
        if (arr.size(false) < maxSize)
        {
            arr.push_back(&val, false, false);
        }
    }
    arr.unlockForWrite();
    __syncthreads();
}

static __global__ void changeValue(dmt::DynaArray& arr, float newVal)
{
    // when you want to change a value of the array, but you are not modifying the structure of the vector (ie not changing the size/capacity)
    // you can avoid using the warp exclusive write lock, and use a read lock, and making sure to compute a unique index into the array for
    // each thread
    int32_t gid = dmt::globalThreadIndex();
    arr.lockForRead();
    if (gid < arr.size(false))
    {
        auto& elem = *reinterpret_cast<float*>(arr.at(gid));
        elem       = newVal;
    }
    arr.unlockForRead();
}

static bool almostEqual(float a, float b, float epsilon = std::numeric_limits<float>::epsilon())
{
    return std::fabs(a - b) <= epsilon;
}

void fillVector(dmt::DynaArray& arr, float val, float after, float* cpu)
{
    dmt::AppContextJanitor j;
    cudaError_t            cudaStatus;
    size_t                 cap             = arr.capacity();
    uint32_t               threadsPerBlock = 32u; // note: equal to warp size
    uint32_t               blocks          = static_cast<uint32_t>(dmt::ceilDiv(cap, 32ull));

    Three* three = nullptr;
    cudaStatus   = cudaMallocManaged(&three, sizeof(Three));
    assert(cudaStatus == ::cudaSuccess);
    std::construct_at(three);

    j.actx.log("Calling managed object virtual function: {}", {reinterpret_cast<IStuff*>(three)->giveMeNumber()});
    testVirtualFunctionsFromManagedObjectWithHostCopyControl<<<1, threadsPerBlock>>>(three);
    cudaDeviceSynchronize();

    // std::destroy_at(three); once you move the vtable on the device, you don't have it on the host!!!
    cudaStatus = cudaFree(three);
    assert(cudaStatus == ::cudaSuccess);

    testResource<<<1, threadsPerBlock>>>(arr);
    cudaStatus = cudaGetLastError();
    assert(cudaStatus == ::cudaSuccess);
    cudaStatus = cudaDeviceSynchronize();
    assert(cudaStatus == ::cudaSuccess);
    if (s_deviceHasAccess)
        j.actx.log("Successfully tested memory resource for memory access from device");
    else
    {
        j.actx.error("something went wrong horribly");
        return;
    }

    fillKernel<<<blocks * cap, threadsPerBlock>>>(arr, val, cap); // 1 push_back per warp
    cudaStatus = cudaGetLastError();
    assert(cudaStatus == ::cudaSuccess);
    cudaDeviceSynchronize();

    // copy and check initial value
    arr.copyToHostSync(cpu);
    if (almostEqual(cpu[0], val))
        j.actx.log("Successfully put value {} inside DynaArray", {val});
    else
    {
        j.actx.error("Failed to put value {} inside DynaArray", {val});
        return;
    }

    changeValue<<<blocks, threadsPerBlock>>>(arr, after);
    cudaStatus = cudaGetLastError();
    assert(cudaStatus == ::cudaSuccess);
    cudaStatus = cudaDeviceSynchronize();
    assert(cudaStatus == ::cudaSuccess);

    arr.copyToHostSync(cpu);
    if (almostEqual(cpu[0], after))
        j.actx.log("Successfully updated value from {} to {}", {val, after});
    else
        j.actx.error("Failed to update value from {} to {}", {val, after});
}
