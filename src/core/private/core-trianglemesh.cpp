#include "core-trianglemesh.h"
#include "cudautils/cudautils-vecmath.cuh"

#include <numeric>

namespace dmt {
TriangleMesh::TriangleMesh(size_t cap, std::pmr::memory_resource* memory)
    : m_positions{memory}, m_normals{memory}, m_uvs{memory}, m_tris{memory} {
  m_positions.reserve(cap);
  m_normals.reserve(cap);
  m_uvs.reserve(cap);
  m_tris.reserve(cap);
}

TriangleMesh& TriangleMesh::addPosition(Point3f p) {
  m_positions.emplace_back(p);
  return *this;
}

bool TriangleMesh::checkPosition(Point3f p, uint32_t& idx) {
  for (size_t i = 0; i < m_positions.size(); ++i) {
    if (m_positions[i].x == p.x && m_positions[i].y == p.y &&
        m_positions[i].z == p.z) {
      idx = static_cast<uint32_t>(i);
      return true;
    }
  }
  return false;
}

bool TriangleMesh::checkNormal(Vector3f n, uint32_t& idx) {
  for (size_t i = 0; i < m_normals.size(); ++i) {
    // Compare un-normalized vectors by reversing normalization (or accept tiny
    // epsilon)
    Vector3f diff = Vector3f(m_normals[i].x, m_normals[i].y, m_normals[i].z) -
                    Vector3f(n.x, n.y, n.z);
    if (dotSelf(diff) < 1e-6f) {
      idx = static_cast<uint32_t>(i);
      return true;
    }
  }
  return false;
}

bool TriangleMesh::checkUV(Point2f uv, uint32_t& idx) {
  for (size_t i = 0; i < m_uvs.size(); ++i) {
    if (std::abs(m_uvs[i].x - uv.x) < 1e-6f &&
        std::abs(m_uvs[i].y - uv.y) < 1e-6f) {
      idx = static_cast<uint32_t>(i);
      return true;
    }
  }
  return false;
}

TriangleMesh& TriangleMesh::addNormal(Normal3f n) {
  m_normals.emplace_back(n);
  return *this;
}

TriangleMesh& TriangleMesh::addUV(Point2f uv) {
  m_uvs.emplace_back(uv);
  return *this;
}

TriangleMesh& TriangleMesh::addIndexedTriangle(VertexIndex i0, VertexIndex i1,
                                               VertexIndex i2, int32_t matIdx) {
  m_tris.emplace_back(i0, i1, i2, matIdx);
  return *this;
}

Point3f TriangleMesh::getPosition(size_t idx) const { return m_positions[idx]; }
uint32_t TriangleMesh::getPositionSize() const { return m_positions.size(); }
Normal3f TriangleMesh::getNormal(size_t idx) const { return m_normals[idx]; }
uint32_t TriangleMesh::getNormalSize() const { return m_normals.size(); }
Point2f TriangleMesh::getUV(size_t idx) const { return m_uvs[idx]; }

IndexedTri TriangleMesh::getIndexedTri(size_t idx) const { return m_tris[idx]; }

IndexedTri& TriangleMesh::getIndexedTriRef(size_t idx) { return m_tris[idx]; }

Bounds3f TriangleMesh::transformedBounds(float const affine[12]) const {
  return transformedBounds(transformFromAffine(affine));
}

Bounds3f TriangleMesh::transformedBounds(Transform const& t) const {
  Bounds3f bounds = bbEmpty();
  for (auto const& triIndex : m_tris) {
    Point3f const v0 = t(m_positions[triIndex.v[0].positionIdx]);
    Point3f const v1 = t(m_positions[triIndex.v[1].positionIdx]);
    Point3f const v2 = t(m_positions[triIndex.v[2].positionIdx]);

    bounds = bbUnion(bbUnion(bbUnion(bounds, v0), v1), v2);
  }

  return bounds;
}

// static

void TriangleMesh::unitCube(TriangleMesh& mesh, int32_t matIdx) {
  static constexpr float half = 0.5f;

  // divide uv space by 64 tiles
  static constexpr float uvtile = 0.125f;
  static constexpr float uExtremeLeft = uvtile;
  static constexpr float uLeft = uvtile * 3;
  static constexpr float uRight = uvtile * 5;
  static constexpr float uExtremeRight = uvtile * 7;

  static constexpr float vExtremeBottom = 0.f;
  static constexpr float vBottom = uvtile * 2;
  static constexpr float vCenter = uvtile * 4;
  static constexpr float vTop = uvtile * 6;
  static constexpr float vExtremeTop = 1.f;

  assert(mesh.positionCount() == 0 &&
         "mesh is supposed to be empty to insert a primitive");

  mesh.addPosition({-half, -half, half})  // 0 left front top (8 vertices total)
      .addPosition({-half, -half, -half})  // 1 left front bottom
      .addPosition({half, -half, -half})   // 2 right front bottom
      .addPosition({half, -half, half})    // 3 right front top
      .addPosition({-half, half, half})    // 4 left back top
      .addPosition({-half, half, -half})   // 5 left back bottom
      .addPosition({half, half, -half})    // 6 right back bottom
      .addPosition({half, half, half});    // 7 right back top

  mesh.addNormal({1, 0, 0})    // 0 right   (+X) (6 normals total)
      .addNormal({-1, 0, 0})   // 1 left    (-X)
      .addNormal({0, 1, 0})    // 2 back    (+Y)
      .addNormal({0, -1, 0})   // 3 front   (-Y)
      .addNormal({0, 0, 1})    // 4 up      (+Z)
      .addNormal({0, 0, -1});  // 5 down    (-Z)

  // cross, starting from bottom left (open blender's cube on uv editor
  // and spreadsheet, face corner domain)
  mesh.addUV({uLeft, vExtremeBottom})   //  0 bottom-left of front
      .addUV({uRight, vExtremeBottom})  //  1 bottom-right of front
      .addUV({uRight, vBottom})         //  2 top-right of front
      .addUV({uRight, vCenter})         //  3 top-right of right
      .addUV({uExtremeRight, vCenter})  //  4 top-right of back
      .addUV({uExtremeRight, vTop})     //  5 bottom-right of back
      .addUV({uRight, vTop})            //  6 bottom-left of back
      .addUV({uRight, vExtremeTop})     //  7 top-left of back
      .addUV({uLeft, vExtremeTop})      //  8 top-left of left
      .addUV({uLeft, vTop})             //  9 bottom-left of left
      .addUV({uExtremeLeft, vTop})      // 10 bottom-right of left
      .addUV({uExtremeLeft, vCenter})   // 11 top-right of left
      .addUV({uLeft, vCenter})          // 12 top-left of front
      .addUV({uLeft, vBottom});         // 13 bottom-left of front

  // Faces: each face has 2 triangles
  // LEFT (-X)
  mesh.addIndexedTriangle({1, 1, 6}, {4, 1, 4}, {5, 1, 5}, matIdx)
      .addIndexedTriangle({1, 1, 6}, {0, 1, 3}, {4, 1, 4}, matIdx);
  // FRONT (+Y)
  mesh.addIndexedTriangle({2, 3, 9}, {0, 3, 3}, {1, 3, 6}, matIdx)  //
      .addIndexedTriangle({2, 3, 9}, {3, 3, 12}, {0, 3, 3}, matIdx);
  // RIGHT (+X)
  mesh.addIndexedTriangle({6, 0, 9}, {3, 0, 12}, {2, 0, 10}, matIdx)  //
      .addIndexedTriangle({6, 0, 9}, {7, 0, 11}, {3, 0, 12}, matIdx);
  // TOP (+Z)
  mesh.addIndexedTriangle({0, 4, 3}, {3, 4, 12}, {4, 4, 2}, matIdx)  //
      .addIndexedTriangle({4, 4, 2}, {3, 4, 12}, {7, 4, 13}, matIdx);
  // BACK (-Y)
  mesh.addIndexedTriangle({7, 2, 13}, {5, 2, 1}, {4, 2, 2}, matIdx)  //
      .addIndexedTriangle({7, 2, 13}, {6, 2, 0}, {5, 2, 1}, matIdx);
  // BOTTOM (-Z)
  mesh.addIndexedTriangle({1, 5, 6}, {5, 5, 7}, {6, 5, 8}, matIdx)  //
      .addIndexedTriangle({2, 5, 9}, {1, 5, 6}, {6, 5, 8}, matIdx);
}

void TriangleMesh::unitPlane(TriangleMesh& mesh, int32_t matIdx) {
  static constexpr float center = 0.f;
  static constexpr float halfExtent = 0.5f;

  mesh.addPosition({center - halfExtent, center - halfExtent, 0})
      .addPosition({center + halfExtent, center - halfExtent, 0})
      .addPosition({center - halfExtent, center + halfExtent, 0})
      .addPosition({center + halfExtent, center + halfExtent, 0});

  mesh.addNormal({0, 0, 1});
  mesh.addUV({0, 0}).addUV({1, 0}).addUV({0, 1}).addUV({1, 1});

  mesh.addIndexedTriangle({0, 0, 0}, {3, 0, 3}, {2, 0, 2}, matIdx)  //
      .addIndexedTriangle({0, 0, 0}, {1, 0, 1}, {3, 0, 3}, matIdx);
}

void TriangleMesh::computeSmoothNormals(float smoothAngleDeg) {
  float const cosThreshold = std::cos(smoothAngleDeg * fl::pi() / 180.f);
  size_t const T = triCount();
  if (T == 0) return;

  // --- step 1: face normals (robust to degenerate triangles) ---
  std::vector<Vector3f> faceNormals(T);
  for (size_t t = 0; t < T; ++t) {
    auto const tri = getIndexedTri(t);
    Point3f const p0 = getPosition(tri.v[0].positionIdx);
    Point3f const p1 = getPosition(tri.v[1].positionIdx);
    Point3f const p2 = getPosition(tri.v[2].positionIdx);

    Vector3f n = cross(p1 - p0, p2 - p0);
    if (dotSelf(n) > 1e-12f)
      faceNormals[t] = normalize(n);
    else {
      // try alternate edge combination or fallback
      Vector3f alt =
          cross(p1 - p0, p2 - p0);  // same expression, kept for clarity
      if (dotSelf(alt) > 1e-12f)
        faceNormals[t] = normalize(alt);
      else
        faceNormals[t] = Vector3f{0.f, 1.f, 0.f};
    }
  }

  // --- step 2: vertex -> incident triangles adjacency ---
  size_t const V = positionCount();
  std::vector<std::vector<size_t>> vertexToTris(V);
  vertexToTris.assign(V, {});
  for (size_t t = 0; t < T; ++t) {
    auto const& tri = getIndexedTri(t);
    for (int i = 0; i < 3; ++i) vertexToTris[tri.v[i].positionIdx].push_back(t);
  }

  // --- step 3: clear normals (we will rebuild) ---
  m_normals.clear();

  // --- step 4: per-vertex clustering of incident faces by angle connectivity
  // ---
  for (size_t vi = 0; vi < V; ++vi) {
    auto& tris = vertexToTris[vi];
    if (tris.empty()) continue;

    // visited flags for the list of incident triangles
    std::vector<char> visited(tris.size(), 0);

    for (size_t ti = 0; ti < tris.size(); ++ti) {
      if (visited[ti]) continue;

      // BFS/DFS to collect connected component of faces (by angle test)
      std::vector<size_t> stack;
      std::vector<size_t> component;  // stores indices into tris[]
      stack.push_back(ti);
      visited[ti] = 1;

      while (!stack.empty()) {
        size_t cur_i = stack.back();
        stack.pop_back();
        component.push_back(cur_i);

        // neighbor test only among incident faces of this vertex
        for (size_t nj = 0; nj < tris.size(); ++nj) {
          if (visited[nj]) continue;
          size_t faceA = tris[cur_i];
          size_t faceB = tris[nj];
          if (dot(faceNormals[faceA], faceNormals[faceB]) >= cosThreshold) {
            visited[nj] = 1;
            stack.push_back(nj);
          }
        }
      }

      // compute averaged normal for this component
      Vector3f sum{0.f, 0.f, 0.f};
      for (size_t comp_i : component) sum += faceNormals[tris[comp_i]];

      if (dotSelf(sum) < 1e-12f) {
        // fallback: use first face normal in component
        sum = faceNormals[tris[component.front()]];
      }

      Vector3f finalN = normalize(sum);

      // deduplicate normals using existing checkNormal (preferred)
      uint32_t normalIndex = 0;
      Normal3f normalObj(finalN);
      if (!checkNormal(normalObj, normalIndex)) {
        addNormal(normalObj);
        normalIndex = static_cast<uint32_t>(m_normals.size() - 1);
      }

      // assign this normal index to all triangle-vertex slots in the component
      for (size_t comp_i : component) {
        size_t triIdx = tris[comp_i];
        auto& triRef = getIndexedTriRef(triIdx);
        for (int slot = 0; slot < 3; ++slot) {
          if (triRef.v[slot].positionIdx == vi)
            triRef.v[slot].normalIdx = normalIndex;
        }
      }
    }
  }

  // --- step 5: safety — make sure all triangle vertex normalIdx are valid ---
  if (m_normals.empty()) {
    // ensure at least one normal exists
    addNormal(Normal3f(Vector3f{0.f, 1.f, 0.f}));
  }

  for (size_t t = 0; t < T; ++t) {
    auto& triRef = getIndexedTriRef(t);
    for (int slot = 0; slot < 3; ++slot) {
      if (triRef.v[slot].normalIdx >= m_normals.size()) {
        // assign default normal index 0 if something slipped through
        triRef.v[slot].normalIdx = 0;
      }
    }
  }

  // sanity check
  for (Normal3f const& n : m_normals) {
    assert(fabsf(dotSelf(n) - 1) < 1e-5f);
  }
  for (auto const& tri : m_tris) {
    assert(tri.v[0].normalIdx < m_normals.size() &&
           fabsf(dotSelf(m_normals[tri.v[0].normalIdx]) - 1) < 1e-5f);
    assert(tri.v[1].normalIdx < m_normals.size() &&
           fabsf(dotSelf(m_normals[tri.v[1].normalIdx]) - 1) < 1e-5f);
    assert(tri.v[2].normalIdx < m_normals.size() &&
           fabsf(dotSelf(m_normals[tri.v[2].normalIdx]) - 1) < 1e-5f);
  }
}

Scene::Scene(std::pmr::memory_resource* memory)
    : geometry{memory}, instances{memory}, m_memory{memory} {}

BufferSpecification Buffers::specFromScene(Scene const& scene) {
  BufferSpecification buf{};
  for (auto const& mesh : scene.geometry) {
    buf.numVerts += mesh->positionCount();
    buf.numNormals += mesh->normalCount();
    buf.numUvs += mesh->uvCount();
    buf.numIndices += mesh->triCount();
  }

  buf.numMeshes = scene.geometry.size();
  buf.numInstances = scene.instances.size();

  return buf;
}

Buffers::Buffers(Scene const& scene, std::pmr::memory_resource* memory)
    : Buffers(specFromScene(scene), memory) {}

Buffers::Buffers(BufferSpecification const& cap,
                 std::pmr::memory_resource* memory)
    : verts{makeUniqueRef<Point3f[]>(memory, cap.numVerts, alignment)},
      normals{
          makeUniqueRef<OctahedralNorm[]>(memory, cap.numNormals, alignment)},
      uvs{makeUniqueRef<Point2f[]>(memory, cap.numUvs, alignment)},
      indices{makeUniqueRef<IndexedTri[]>(memory, cap.numIndices, alignment)},
      instances{makeUniqueRef<Instance[]>(memory, cap.numInstances, alignment)},
      primOffset{makeUniqueRef<GeoOffsets[]>(memory, cap.numMeshes, alignment)},
      m_memory{memory},
      m_curr{},
      m_cap{cap} {}

static constexpr bool operator==(BufferSpecification const& a,
                                 BufferSpecification const& b) {
  return a.numVerts == b.numVerts && a.numNormals == b.numNormals &&
         a.numUvs == b.numUvs && a.numIndices == b.numIndices &&
         a.numMeshes == b.numMeshes && a.numInstances == b.numInstances;
}

bool Buffers::copySceneToBuffers(Scene const& scene) {
  assert(m_cap == specFromScene(scene) && "Inconsistent specification");

  // TODO if Scene in host memory and scene in device memory do something else
  // TODO parallel version?
  GeoOffsets runningOff{};
  for (size_t meshIdx = 0; meshIdx < m_cap.numMeshes; ++meshIdx) {
    TriangleMesh const& mesh = *scene.geometry[meshIdx];

    // save offsets
    primOffset[meshIdx] = runningOff;

    // save vertices (unchanged)
    for (size_t vertIdx = 0; vertIdx < mesh.positionCount(); ++vertIdx)
      verts[runningOff.positionIdx + vertIdx] = mesh.getPosition(vertIdx);

    // save normals
    for (size_t normIdx = 0; normIdx < mesh.normalCount(); ++normIdx)
      normals[runningOff.normalIdx + normIdx] =
          octaFromNorm(mesh.getNormal(normIdx));

    // save uvs
    for (size_t uvIdx = 0; uvIdx < mesh.uvCount(); ++uvIdx)
      uvs[runningOff.uvIdx + uvIdx] = mesh.getUV(uvIdx);

    // save _globalized_ indices
    for (size_t localIdx = 0; localIdx < mesh.triCount(); ++localIdx)
      indices[runningOff.indexOff + localIdx] = mesh.getIndexedTri(localIdx);

    runningOff.positionIdx += mesh.positionCount();
    runningOff.normalIdx += mesh.normalCount();
    runningOff.uvIdx += mesh.uvCount();
    runningOff.indexOff += mesh.triCount();
    ++m_curr.numMeshes;
  }

  for (size_t instanceIdx = 0; instanceIdx < m_cap.numInstances;
       ++instanceIdx) {
    instances[instanceIdx] = *scene.instances[instanceIdx];
    ++m_curr.numInstances;
  }

  assert(m_cap == m_curr && "Cap config and current config not equal");
  return true;
}
}  // namespace dmt