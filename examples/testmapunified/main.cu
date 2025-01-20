#include "dmtmacros.h"
#include <platform/platform.h>
#include <platform/platform-cuda-utils.h>
#include <platform/platform-cuda-utils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <limits>
#include <unordered_map>
#include <memory_resource>

#include <cstdio>
#include <cstdint>

using dmt::sid_t;
using dmt::operator""_side;

int32_t main()
{
    printf("Adding elements in the map from the host\n");
    cudaFree(0); // force CUcontext creation

    getc(stdin);
}