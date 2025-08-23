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
#include <platform/platform-memory-stackAllocator.h>

namespace dmt {
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
