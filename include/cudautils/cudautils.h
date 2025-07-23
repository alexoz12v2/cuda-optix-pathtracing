#pragma once

#include "dmtmacros.h"

#include "cudautils/cudautils-enums.h"
#include "cudautils/cudautils-float.h"
#include "cudautils/cudautils-vecmath.h"
#include "cudautils/cudautils-light.h"
#include "cudautils/cudautils-transform.h"
#include "cudautils/cudautils-camera.h"
#include "cudautils/cudautils-media.h"
#include "cudautils/cudautils-lightsampler.h"
#include "cudautils/cudautils-texture.h"
#include "cudautils/cudautils-material.h"
#include "cudautils/cudautils-sampler.h"
#include "cudautils/cudautils-film.h"
#include "cudautils/cudautils-filter.h"
#include "cudautils/cudautils-bxdf.h"
#include "cudautils/cudautils-spectrum.h"
#include "cudautils/cudautils-color.h"
#include "cudautils/cudautils-numbers.h"
#include "cudautils/cudautils-image.h"

#if defined(DMT_CUDAUTILS_IMPL)
#include "cudautils.cu"
#endif