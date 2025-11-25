#ifndef DMT_CORE_PUBLIC_CORE_PARSER_H
#define DMT_CORE_PUBLIC_CORE_PARSER_H

#include "core-macros.h"
#include "core-trianglemesh.h"
#include "core-dstd.h"
#include "core-material.h"
#include "core-mesh-parser.h"
#include "core-render.h"
#include "cudautils/cudautils-transform.cuh"

#include "platform-utils.h"

#include <memory_resource>

namespace dmt {
    class Parser
    {
    public:
        DMT_CORE_API Parser(os::Path const&            path,
                            Renderer*                  renderer,
                            std::pmr::memory_resource* mem = std::pmr::get_default_resource());
        Parser(Parser const&)                = delete;
        Parser(Parser&&) noexcept            = delete;
        Parser& operator=(Parser const&)     = delete;
        Parser& operator=(Parser&&) noexcept = delete;

    public:
        [[nodiscard]] DMT_FORCEINLINE bool isValid() const { return m_path.isValid() && m_path.isFile(); }
        [[nodiscard]] DMT_FORCEINLINE os::Path fileDirectory() const { return m_path.parent(); }

        DMT_CORE_API bool parse();

    private:
        /// component to handle FBX importing (1st triangular mesh only, ignoring textures and materials)
        MeshFbxParser m_fbxParser;

        /// Scene to be populated. Assumes no other accesses are being made to it
        Renderer* m_renderer;

        /// Path to json file
        os::Path m_path;

        /// Must outlive the `Parser` object
        std::pmr::memory_resource* m_tmp;
    };
} // namespace dmt
#endif // DMT_CORE_PUBLIC_CORE_PARSER_H
