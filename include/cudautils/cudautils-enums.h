#pragma once

#include "dmtmacros.h"

#include <cstdint>

namespace dmt {
    // Enums ----------------------------------------------------------------------------------------------------------
    //I don't understand
    enum class ERenderCoordSys : uint8_t
    {
        eCameraWorld = 0,
        eCamera,
        eWorld,
        eCount
    };
} // namespace dmt
