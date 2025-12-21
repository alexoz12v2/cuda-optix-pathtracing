#ifndef DMT_CORE_PUBLIC_CORE_PARSER_H
#define DMT_CORE_PUBLIC_CORE_PARSER_H

#include "core-macros.h"
#include "core-trianglemesh.h"
#include "core-dstd.h"
#include "core-material.h"
#include "core-mesh-parser.h"
#include "core-types.h"

#include "cudautils/cudautils-transform.cuh"

#include "platform-utils.h"

#include <memory_resource>

namespace dmt {

class Parser {
 public:
  /// \warning moves the path object inside the class
  DMT_CORE_API explicit Parser(
      os::Path path,
      std::pmr::memory_resource* mem = std::pmr::get_default_resource());
  Parser(Parser const&) = delete;
  Parser(Parser&&) noexcept = delete;
  Parser& operator=(Parser const&) = delete;
  Parser& operator=(Parser&&) noexcept = delete;

  [[nodiscard]] DMT_FORCEINLINE bool isValid() const {
    return m_path.isValid() && m_path.isFile();
  }
  [[nodiscard]] DMT_FORCEINLINE os::Path fileDirectory() const {
    return m_path.parent();
  }

  DMT_CORE_API bool parse(ParsedObject& outObject);

 private:
  /// component to handle FBX importing (1st triangular mesh only, ignoring
  /// textures and materials)
  MeshFbxParser m_fbxParser;

  /// Path to json file
  os::Path m_path;

  /// Must outlive the `Parser` object
  std::pmr::memory_resource* m_tmp;
};
}  // namespace dmt

namespace dmt::parse_helpers {

enum class ExtractVec3fResult {
  eOk = 0,
  eNotArray,
  eIncorrectSize,
  eInvalidType0,
  eInvalidType1,
  eInvalidType2,

  eCount
};

enum class TempTexObjResult {
  eOk = 0,
  eNotExists,
  eFormatNotSupported,
  eFailLoad,
  eNumChannelsIncorrect,

  eCount
};

DMT_CORE_API RGB* loadImageAsRGB(os::Path const& path, int& outWidth,
                                 int& outHeight);
DMT_CORE_API TempTexObjResult tempTexObj(os::Path const& path, bool isRGB,
                                         ImageTexturev2* out);
DMT_CORE_API std::string_view tempTexObjResultToString(TempTexObjResult result);
DMT_CORE_API void freeTempTexObj(ImageTexturev2& in);
}  // namespace dmt::parse_helpers

#endif  // DMT_CORE_PUBLIC_CORE_PARSER_H
