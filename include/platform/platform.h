#pragma once

// group 1: macros
#include "platform/platform-macros.h"

// group 2: launch
#include <platform/platform-launch.h>

// group 3: utils, logging, context (order matters)
#include <platform/platform-utils.h>
#include <platform/platform-logging.h>
#include <platform/platform-context.h>

// group 4: stuff. (order doesn't matter)
#include <platform/platform-memory.h>
#include <platform/platform-threadPool.h>
#include <platform/platform-file.h>

// group 5: generated
#include <platform/cuda-wrapper.h>

namespace dmt {
    /**
     * Checks the `CUresult` of a CUDA Driver API operation, and, if different than `::CUDA_SUCCESS`,
     * @note This is supposed to be used in CUDA calls whose failure is fatal
     */
    [[nodiscard]] DMT_PLATFORM_API bool cudaDriverCall(NvcudaLibraryFunctions const* cudaApi, CUresult result);

    /**
     * This function (and similiar for all loaded dlls) should be populated with more
     * manual fixes with respect to the generated version as soon as Access Violations are discovered
     */
    DMT_PLATFORM_API void fixCUDADriverSymbols(NvcudaLibraryFunctions* cudaApi);

    /**
     * Adds a ContextImpl object inside
     */
    class DMT_PLATFORM_API Ctx
    {
    public:
        /**
         * @warning The `resource` object should live beyond the `destroy` function call
         */
        static void init(bool                       destroyIfExising = false,
                         std::pmr::memory_resource* resource         = std::pmr::get_default_resource());
        static void destroy();

    private:
        Ctx()                                               = default;
        static inline std::pmr::memory_resource* m_resource = nullptr;
    };
} // namespace dmt
