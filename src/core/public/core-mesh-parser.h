#ifndef DMT_CORE_PUBLIC_CORE_MESH_PARSER_H
#define DMT_CORE_PUBLIC_CORE_MESH_PARSER_H

#include "core-macros.h"
#include "core-trianglemesh.h"
#include "cudautils/cudautils-transform.cuh"

// std library
#include <memory>
#include <string>
#include <unordered_map>

namespace dmt {
class MeshFbxParserImpl;
class DMT_CORE_API MeshFbxParser {
 public:
  MeshFbxParser();
  ~MeshFbxParser();

  bool ImportFBX(char const* fileName, TriangleMesh* outMesh) const;

 private:
  std::unique_ptr<MeshFbxParserImpl> m_pimpl = nullptr;
};
}  // namespace dmt
#endif  // DMT_CORE_PUBLIC_CORE_MESH_PARSER_H
