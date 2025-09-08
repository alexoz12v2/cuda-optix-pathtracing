#include "cudautils-light.h"

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
