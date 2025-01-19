#include "dmtmacros.h"
#include <platform/platform.h>
#include <platform/platform-cuda-utils.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <limits>
#include <unordered_map>

#include <cstdio>
#include <cstdint>

namespace dmt::vtable {
    using namespace dmt;
    struct KeyValue 
    {
        static constexpr uint64_t empty = std::numeric_limits<uint64_t>::max();
        /** Obtained as concatenation of 8-bit device ID and 56 bits from the hashCRC64 of the name of the table */
        uint64_t key;
        BaseMemoryResource::VTableDevice value;
    };

    static constexpr uint64_t maxDevices = 16;
    // Unified, CudaMalloc, CudaMallocAsync, Buddy, MemPoolAsync
    static constexpr uint64_t numAllocators = 8;
    static constexpr size_t numEntries = maxDevices * numAllocators;
    static constexpr size_t numBytes = numEntries * sizeof(KeyValue);

    namespace detail 
    {
        DMT_FORCEINLINE __device__ uint64_t keyFrom(int32_t deviceID, sid_t name) 
        {
           assert(deviceID >= 0 && deviceID < dmt::vtable::maxDevices);
           uint64_t key = name | deviceID;
           assert(key != KeyValue::empty);
           return key;
        }
    }

    __managed__ BaseDeviceContainer* vtableLocker = nullptr;
    __managed__ KeyValue* vtable = nullptr;

    __host__ KeyValue* create() 
    {
        cudaError_t err = cudaMallocManaged(&vtableLocker, sizeof(BaseDeviceContainer));
        assert(err == ::cudaSuccess);
        err = cudaMallocManaged(&vtable, numBytes);
        assert(err == ::cudaSuccess);
        for (size_t i = 0; i < numEntries; ++i)
        {
            vtable[i].key = KeyValue::empty;
        }
        return vtable;
    }

    __host__ void destroy() 
    {
        vtableLocker->lockForWrite();
        cudaError_t err = cudaDeviceSynchronize();
        assert(err == ::cudaSuccess);
        err = cudaFree(vtable);
        assert(err == ::cudaSuccess);
        err = cudaFree(vtableLocker);
    }

    __device__ BaseMemoryResource::VTableDevice* lookup(int32_t deviceID, sid_t name)
    {
        BaseMemoryResource::VTableDevice* ret = nullptr;
        uint64_t const key = detail::keyFrom(deviceID, name);
        uint64_t const start = key % numEntries; // TODO POT masking
        uint64_t index = start;

        vtableLocker->lockForRead();

        do 
        {   
            KeyValue& current = vtable[index];
            if (current.key == key)
                ret = &current.value;
            else
                index = (index + 1) % numEntries;
        } while (index == start);

        vtableLocker->unlockForRead();

        return ret;
    }

    __device__ void insert(int32_t deviceID, sid_t name, void* userData, void(*putFunc)(void* _userData, BaseMemoryResource::VTableDevice& _toWrite))
    {
        uint64_t const key = detail::keyFrom(deviceID, name);
        uint64_t const start = key % numEntries; // TODO POT masking
        uint64_t index = start;
        if (vtableLocker->lockForWrite())
        {
            do 
            {
                KeyValue& current = vtable[index];
                if (current.key == KeyValue::empty)
                {
                    putFunc(userData, current.value);
                    vtableLocker->unlockForWrite();
                    return;
                }
                else
                    index = (index + 1) % numEntries;
            } while (index == start);

            // you shouldn't be here, die
            // TODO log macro?
            vtableLocker->unlockForWrite();
            asm("trap;");
        }
        // __syncthreads(); ?
    }
}

int32_t main()
{
    printf("Adding elements in the map from the host");

    getc(stdin);
}