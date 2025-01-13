#pragma once

#include "dmtmacros.h"
#include "middleware/middleware-macros.h"

#include <middleware/middleware-utils.h>
#include "middleware/middleware-parser.h"
#include "middleware/middleware-model.h"

#if !defined(DMT_NEEDS_MODULE)
#include <platform/platform.h>

#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/ext/matrix_transform.hpp>  // glm::translate, glm::rotate, glm::scale
#include <glm/ext/scalar_constants.hpp>  // glm::pi
#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/vec3.hpp>   // glm::vec3
#include <glm/vec4.hpp>   // glm::vec4

#include <array>
#include <atomic>
#include <forward_list>
#include <limits>
#include <map>
#include <memory_resource>
#include <set>
#include <stack>
#include <string_view>
#include <thread>
#include <vector>

#include <cassert>
#include <compare>
#include <cstdint>
#endif

// TODO switch all structures with stack allocator
// TODO remove default values from clases and leave them only in parsing functions

// stuff related to .pbrt file parsing + data structures
DMT_MODULE_EXPORT namespace dmt {} // namespace dmt

DMT_MODULE_EXPORT namespace dmt::job {
    using namespace dmt;
    struct ParseSceneHeaderData
    {
        std::string_view      filePath;
        AppContext*           actx;          // TODO not needed anymore, remove
        Options*              pInOutOptions; // when job kicked, you caller must wait on atomic
        std::atomic<uint32_t> done;          // should be zero when the job is kicked
        uint32_t              numChunkWorldBegin;
        uint32_t              offsetWorldBegin;
        uint32_t              numChunks;
    };

    void parseSceneHeader(uintptr_t address);
} // namespace dmt::job