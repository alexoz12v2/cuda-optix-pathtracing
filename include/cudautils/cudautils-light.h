#pragma once

#include "cudautils/cudautils-macro.h"

namespace dmt {

    struct DeviceLights
    {
        bool hasEnv;
    };

    __device__ void env_eval(DeviceLights const& lights, float& Lr, float& Lg, float& Lb, float& pdf);
} // namespace dmt

#if defined(DMT_CUDAUTILS_IMPLEMENTATION)
namespace dmt {
    DMT_GPU void env_eval(DeviceLights const& lights, float& Lr, float& Lg, float& Lb, float& pdf)
    {
        if (!lights.hasEnv)
        {
            Lr = Lg = Lb = 0.f;
            pdf          = 0.f;
            return;
        }
        // TODO: sample HDRI; here: constant gray
        Lr = Lg = Lb = 0.1f;
        pdf          = 1.f;
    }
} // namespace dmt
#endif
