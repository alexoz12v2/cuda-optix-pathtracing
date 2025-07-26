#pragma once

#include "cudautils/cudautils-macro.h"

class Film
{
};

#if defined(DMT_CUDAUTILS_IMPL) || defined(DMT_CUDAUTILS_FILM_IMPL)
    #include "cudautils-film.cu"
#endif