#pragma once

#include "platform/platform-macros.h"
#include <platform/platform-launch.h> // must be included before all the rest

#include <platform/platform-utils.h>
#include <platform/platform-logging.h>
#include <platform/platform-memory.h>
#include <platform/platform-threadPool.h>
#include <platform/platform-display.h>
#include <platform/platform-file.h>
#include <platform/platform-context.h>
#include "platform/platform-logging-default-formatters.h" // must be after platform-logging
#include <platform/cuda-wrapper.h>

namespace dmt {
    /**
     * Checks the `CUresult` of a CUDA Driver API operation, and, if different than `::CUDA_SUCCESS`,
     * @note This is supposed to be used in CUDA calls whose failure is fatal
     */
    [[nodiscard]] DMT_PLATFORM_API bool cudaDriverCall(NvcudaLibraryFunctions* cudaApi, CUresult result);

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
