#include "common.cuh"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Light specific sampling functions
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Light Sampling dispatcher
// ---------------------------------------------------------------------------
__host__ __device__ LightSample sampleLight(Light const& light,
                                            float3 const position,
                                            bool hadTransmission,
                                            float3 const normal, float* pdf) {
  LightSample sample{};
  // on device,we want to enter and exit with the same warp configuration
#ifdef __CUDA_ARCH__
  cg::coalesced_group theWarp = cooperative_groups::coalesced_threads();
#endif
  switch (light.type()) {
    case ELightType::ePoint:
      break;
    case ELightType::eSpot:
      break;
    case ELightType::eEnv:
      break;
    case ELightType::eDirectional:
      break;
  }

#ifdef __CUDA_ARCH__
  theWarp.sync();
#endif
  return sample;
}
