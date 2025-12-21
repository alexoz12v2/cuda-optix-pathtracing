#ifndef DUMBTRACER_CORE_CONSTANTS_H
#define DUMBTRACER_CORE_CONSTANTS_H

#include <cstdint>

namespace dmt {

    // Number of samples for each unit radius of Mitchell's filter tabularization
    inline constexpr int32_t NumSamplesPerAxisPerDomainUnit = 32;
    inline constexpr int32_t MaxResolutionGrid              = 64;
    inline constexpr int32_t EWA_LUT_SIZE                   = 128;
} // namespace dmt

#endif