#pragma once

#include "cudautils/cudautils-macro.h"

#if defined(DMT_CUDAUTILS_IMPL) || defined(DMT_CUDAUTILS_FILTER_IMPL)
#include "cudautils-filter.cu"
#endif
