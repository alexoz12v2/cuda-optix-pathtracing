#pragma once

#include <platform/platform.h>
#include <platform/platform-cuda-utils.h>

void fillVector(dmt::DynaArray& arr, float val, float after, float* cpu);