# Cycles Scene Notes

We are referring to the `Scene` class used in cycles, which, among everything, stores a list of `Geometry` (look in the function `xml_read_scene`
to see how a scene description is parsed).

Observe how the `Geometry` Base class (whose subclasses are `Mesh`, `Hair`, `PointCloud`) have method `compute_bvh`, which returns a BVH Build Node.
This leads us to believe that cycles uses a Bottom-Up/Incremental approach to construct its BVH, which makes sense as it needs to efficiently
support moving objects, hence evolving Bounding Volume Hierarchies.

We are intereseted in methods `apply_transform`, `compute_bounds`, `is_instanced` of the `Geometry` class.

Let us start by analysing the `Mesh` subclass and how is constructed when parsing.

## `Geometry` Foundations

The `Geometry` base class implements the `Node` base class, which is an Hashable, reference counted, Table of values.

- `Geometry` owns its own BVH node
- Stores an index to the scene
- Is considered an instance if the `!transform_applied`
- Stores an Affine Transform

## `Mesh` Parsing

Let us break down what the method `xml_read_mesh` does, keeping in mind that cycles groups more Data Blocks (like the geometry itself) into Objects,
and that the parser contains a current state storing the current object and current affine transform

Scene Snippet Example:

```xml
<!-- Objects -->
<transform rotate="180 0 1 1">
 <state interpolation="smooth" shader="monkey">
  <include src="./objects/suzanne.xml" />
 </state>
</transform>

<transform rotate="90 1 0 0">
 <transform translate="0 0 1">
  <state shader="floor">
   <mesh P="-3 3 0  3 3 0  3 -3 0  -3 -3 0" nverts="4" verts="0 1 2 3" />
  </state>
 </transform>
</transform>
```

```cpp
static void xml_read_mesh(const XMLReadState &state, const xml_node node)
{
    Mesh* mesh = xml_add_mesh(state.scene, state.tfm, state.object);
    // ... details for shaders ...
    
    vector<float3> P;   // positions
    vector<float3> VN;  // normals
    vector<float> UV, T, TS;  // uv, uv tangents, uv tangent signs
    // index buffer is split in two, nverts[i] -> number of vertices on a face, verts[i:i+nverts[i]] -> indices of vertices in face. 
    // this split allows the renderer to support ngons
    vector<int> verts, nverts; 

    xml_read_float3_array(P, node, "P");
    xml_read_int_array(verts, node, "verts");
    xml_read_int_array(nverts, node, "nverts");

    // ... choose subdivision algorithm ...

    if (mesh->get_subdivision_type() == Mesh::SUBDIVISION_NONE) {
        // create vertices
        mesh->set_verts(P);

        // compute num triangles
        size_t num_triangles = 0;
        for (size_t i = 0; i < nverts.size(); ++i) { 
            num_triangles += nverts[i] - 2;
        }
        mesh->reserve_mesh(mesh->get_verts().size(), num_triangles);

        /* create triangles */
        int index_offset = 0;

        /* ... create triangles ... */

        if (xml_read_float3_array(VN, node, "N")) { /* ... handle vertex normal ... */ }
        if (xml_read_float_array(UV, node, "uv")) { /* ... handle uv ...*/}
        if (xml_read_float_array(T, node, "tangent")) { /* ... handle uv tangents ...*/}
        if (xml_read_float_array(TS, node, "tangent_sign")) {/* ... handle uv tangent sign ...*/}
    } else{/*... handle subdivision...*/}

    // ... handle generated coordinates ...
}
```

Let us ignore stuff like shaders and subdivision handling for now.

First, cycles will take the current `Mesh` instance from the scene Object if present, otherwise it creates a new Mesh object, add it to the geometry
vector of the scene and wire it to the current scene Object. Ignoring all the subdivision related stuff, the following are the data stored by a mesh:

```cpp
std::unique_ptr_with_size<int[]>    triangles;
std::unique_ptr_with_size<float3[]> verts;
std::unique_ptr_with_size<int[]>    shader;
std::unique_ptr_with_size<bool>     smooth;
// from Geometry base class, stores normals (complications arise because if a face is "Shade smooth", each vertex stores only 1 normal, while if
// "Shade Flat", each vertex stores 1 normal for each of its connected faces)
// Attribute also stores the uv
AttributeSet attributes; // list<Attribute>, where Attribute = vector<char>, type
```

Second, since cycles supports meshes with ngons, the first thing it does is triangularize each face and store them into the `Mesh` class

- Create Triangles:

```cpp
for (size_t i = 0; i < nverts.size(); i++) {
    for (int j = 0; j < nverts[i] - 2; j++) {
        const int v0 = verts[index_offset];
        const int v1 = verts[index_offset + j + 1];
        const int v2 = verts[index_offset + j + 2];

        assert(v0 < (int)P.size());
        assert(v1 < (int)P.size());
        assert(v2 < (int)P.size());

        mesh->add_triangle(v0, v1, v2, shader, state.smooth);
    }

    index_offset += nverts[i];
}
```

Once the meshes are constructed, pay attention to how they are managed. everytime `Scene::update` is called, hence everytime
`GeometryManager::device_update` is called, meshes stored in the `Scene` object as a `vector<Geometry>` are copied into a SOA format
inside chosen device memory, done by the function `GeometryManager::device_update_mesh`

We can achieve something similiar

1. Parse a mesh from an FBX file into a `TriangleMesh` class instance, which

    - Takes vertex positions and stores them in a vertex buffer
    - Takes the face indices and triangularizes the mesh
    - uv coordinates for each vertex (fix them to 0 if not present)

        - if uv coordinates are present, for each face in which there's flat shading, hence normals are specified per vertex,
          try to recover uv tangent vector (optional, because they can be generated on the fly knowing the triangle vertices and uv coords)

    - Generate a vertex normal buffer under the following conditions:

        - Normals specified in file (if not, generate them)
        - Normals can be for "smooth shading" or "flat shading", depending on the mapping mode and reference mode of the normal element in the fbx

            | MappingMode        | ReferenceMode    | Normal Resolution              | Typical Use               |
            | ------------------ | ---------------- | ------------------------------ | ------------------------- |
            | `eByControlPoint`  | `eDirect`        | One normal per shared vertex   | Smooth shading            |
            | `eByControlPoint`  | `eIndexToDirect` | Indexed normal per vertex      | Smooth shading, optimized |
            | `eByPolygonVertex` | `eDirect`        | One normal per triangle corner | Flat shading, sharp edges |
            | `eByPolygonVertex` | `eIndexToDirect` | Indexed normal per corner      | Flat/smooth mix           |
            | `eByPolygon`       | `eDirect`        | One normal per triangle        | Flat shading              |
            | `eByPolygon`       | `eIndexToDirect` | Indexed normal per triangle    | Flat shading              |
            | `eAllSame`         | `eDirect`        | Single normal for whole mesh   | Rare                      |

2. Create a `TriangleMesh` instance, which contains all buffers you need
3. Move All mesh instances in the scene in the SOA format and create instancing primitives
4. for each mesh instance primitive, compute a partial BVH referencing only the given mesh
5. combine all obtained bvh nodes into a sigle tree
