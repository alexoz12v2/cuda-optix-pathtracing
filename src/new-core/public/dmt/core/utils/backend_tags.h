#pragma once

namespace dmt {

struct CpuTag {};
struct CudaTag {};

#ifdef __CUDA_ARCH__
#  define DMT_BACKEND_TAG ::dmt::CudaTag
#else
#  define DMT_BACKEND_TAG ::dmt::CpuTag
#endif

}  // namespace dmt