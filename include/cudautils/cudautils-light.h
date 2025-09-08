#pragma once

#include "cudautils/cudautils-macro.h"


struct DeviceLights
{
    bool hasEnv;
};

__device__ void env_eval(DeviceLights const& lights, float& Lr, float& Lg, float& Lb, float& pdf);