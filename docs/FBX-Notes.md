# FBX File Format Notes

## Intro

[FBX](https://aps.autodesk.com/developer/overview/fbx-sdk) is an Autodesk proprietary format
for Data exchange of 3D assets

They come both in binary and text (ASCII) format, though the latter is rarely used. That said, it can be useful to learn

Here's the general structure

```txt
FBXHeaderExtension: {
 ; some headers (we don't care)
 ; for example, here you'll find camera information
}
; Skipping useless header information like CreationTime,. ..
; These enumerate all the objects present in the scene
Definitions: {
 Count: 2
 ; then you group the Count number per ObjectType. We are interested in the Model Type
 ObjectType: "Model" {
   Count: 2
 }
}
; Now that we know there are 2 Models in the scene, w'll look at their properties
Objects: {
 ; type can be "Mesh", "Light", we care about meshes
  Model: "Model::Mesh", "Mesh" {
    Version: 232 ; don't care
    ; Transform Information. See later for more info (still name type value)
    ; Lcl stands for local since they should refer to parent?
    Properties60: {
      Property: "Lcl Translation", "Lcl Translation", "A+",0,0,0
      Property: "Lcl Rotation", "Lcl Rotation", "A+",0,0,0
      Property: "Lcl Scaling", "Lcl Scaling", "A+",1,1,1
    }
    NodeAttributeName: "Geometry::Mesh"
  }
  ; second object, light, omitted.
}
; Connection which define scene graph. In this case, root -> light -> mesh
; is Model::Scene a well known property/implicit name?
Connections: {
  Connect: "OO", "Model::Light", "Model::Scene"
  Connect: "OO", "Model::Mesh", "Model::Light"
}
; Takes = Animation Data. We don't care (emit error if found)
Takes: {
 ; ...
}
; Current FBX file format version is 7.7? hence here go legacy settings (ignored)
Version5: {
}
```

The **Scene Graph**

- Is a Node Hierarchy. Each node has
    - Reference to an **Attribute Object** which contain the content of the node
    - Array of **Takes** (Animation Data)
    - **Position**, **Rotation**, **Scaling** Relative to Parent

What if I want to visualize binary FBX as ASCII

- [FBX Format Converter](https://github.com/BobbyAnguelov/FbxFormatConverter)
    - Yes.
    - `FbxFormatConverter.exe -c <filepath|folderpath> [-o <filepath|folderpath>] {-ascii|-binary}`

## Code Notes

|                 |                                                                                          |
|-----------------|------------------------------------------------------------------------------------------| 
| `FbxManager`    | Singleton representing the main interface to the SDK to manage memory                    |
| `FbxIOSettings` | Configuration options for import/export operation. The import options Start with `IMP_*` |

- [Import Scene Tutorial](https://help.autodesk.com/cloudhelp/2018/ENU/FBX-Developer-Help/importing_and_exporting_a_scene/importing_a_scene.html)

## Concepts

### Fbx Stream

Basic abstract interface to read and write IO data. By default, the FBX SDK reads
and writes FBX Files into the filesystem. Advanced applications might need more control
such as

- Loading FBX Data from memory buffers
- Streaming FBX files from compressed/encrypted/network
- integrating with custom asset pipelines

### The FBX Importer

Once the importer has been initialized, **It has oly parsed the file header and selected a reader**. The scene
data hasn't been parsed yet

A Well Formed FBX File should contain **Only 1 Scene**, which we can import by creating a `FbxScene` object

When printing information about what the FBX file contains, we have different sets
of functions

- `FbxNode::GetSrcObjectCount<T>()` counts what exists of a given subtype
    - when applied to `FbxScene`, otherwise, counts outgoing connections to the object
    - for incoming connections, `GetDstObjectCount`, which, for scene, which is the root, is always 0

One of the direct children of `FbxScene` is `FbxGlobalSettings`, hence we have a direct
accessor to that without having to iterate through the hierarchy

```c++
FbxScene* scene = getScene();
FbxGlobalSettings const& theSettings = scene->GetGlobalSettings();
```

### (Global Settings) Coordinate System

FBX [**Stores the scene using Right-Handed, Y-Up system
**](https://help.autodesk.com/view/FBX/2020/ENU/?guid=FBX_Developer_Help_nodes_and_scene_graph_fbx_scenes_scene_axis_and_unit_conversion_html).

- Default Unit: Centimetres

This means that, if the application uses another unit or another coordinate frame, it needs to convert them accordingly
while loading a mesh.

- Note that this doesn't change the vertex data, only **node transforms**

```c++
FbxGlobalSettings const& theSettings = scene->GetGlobalSettings();
// getting Unit
theSettings.GetSystemUnit();
// Covner
```

An FBX Coordinate system is defined with 5 parameters

- Up Axis
    - The EUpVector specifies which axis has the up and down direction in the system.
- Up Axis Sign
    - The sign of the EUpVector is applied to represent the direction (1 is up and -1 is down relative to the observer).
- Front Axis Parity
    - `ParityEven` and `ParityOdd` denote the first one and the second one of the
      remain two axes in addition to the up axis.
    - With `ParityEven`, the front axis takes the numerically smaller available axis
- Front Axis Sign
    - Whether the chosen axis should be negated or not
- Handedness
    - Dictates, indirectly, the sign of the side axis to achieve the desired handedness
        - Right Handedness: `Right x Up = Front`
        - Left Handedness: `Right x Up = -Front` (equal to `Left x Up = Front`)

| Up Axis | Parity       | Front Axis | Side Axis |
|---------|--------------|------------|-----------|
| X       | `ParityEven` | Y          | Z         |
| X       | `ParityOdd`  | Z          | Y         |
| Y       | `ParityEven` | X          | Z         |
| Y       | `ParityOdd`  | Z          | X         |
| Z       | `ParityEven` | X          | Y         |
| Z       | `ParityOdd`  | Y          | X         |

Our desired system is

- Up Axis: +Z
- Front Axis: +Y
- Side Axis: +X

Hence, we can derive the 5 properties [easily](https://en.wikipedia.org/wiki/Right-hand_rule)

- Up Axis: Z
- Up Axis Sign: +1
- Parity: Odd
- Front Axis Sign: +1
- Handedness: **Left**

## Scene Graph as expressed in the Connection Section of the FBX ASCII file

```text
; Object connections
;------------------------------------------------------------------

Connections:  {
	
	;Model::Teapot_Node, Model::RootNode
	C: "OO",2601406783424,0
	
	;Geometry::Teapot, Model::Teapot_Node
	C: "OO",2601406785408,2601406783424
	
	;Material::, Model::Teapot_Node
	C: "OO",2601406176256,2601406783424
}
```

The format of a connection in the graph is the following:

- `C: "OO", childId, parentId`
    - Where `"OO"` stands for **Object to Object** connection

A new node is added to the graph only in **Model to Model** connections, eg

- `;Model::Teapot_Node, Model::RootNode`
    - The `Model` part is the **Object Type**, defined under the `Definitions` section
      of the file. Example: `ObjectType: "Model"`
    - The Name, ie `Model::Teapot_Node`, is the **Object Name**, defined under `Objects`.
      Having it with the Type Prefix makes it so that the object gets some default properties without having to define
      them in every object (TODO Example)

```text
; Under "Objects:"
	Model: 2601406783424, "Model::Teapot_Node", "Mesh" {
		Version: 232
		Properties70:  {
			P: "InheritType", "enum", "", "",1
			P: "ScalingMax", "Vector3D", "Vector", "",0,0,0
			P: "DefaultAttributeIndex", "int", "Integer", "",0
		}
		Shading: Y
		Culling: "CullingOff"
	}
```

Assigning to a node a connection to a geometry or a material, instead, creates an attribute belonging to that
node, and it's what's happening on the other two connections of our example. We'll End up with the following logical
structure

```text
FbxScene
└── FbxNode "RootNode"
    └── FbxNode "Teapot_Node"
        ├── NodeAttribute: FbxMesh "Teapot"
        ├── Material[0]: FbxSurfaceMaterial
```

### Geometry Object

An example for a geometry object follows. Read it carefully

```text
Definitions: {
    ; don't care about the rest
    ObjectType "Geometry": {
        Count: 1
        PropertyTemplate: "FbxMesh": {
            Properties70: {
                ; color, receives shadow, ... we don't care
            }
        }
    }
}
Objects: {
    ; don't care about the rest
    ; template: Geometry: <id>, "ObjectType::Name", "Mesh"
    Geometry: 2601406785408, "Geometry::Teapot", "Mesh" {
        ; probably stuff from exporter, don't care
        Properties70: {
            P: "smoothingAngle", "float", "","",45 
        }
        ; template: Vertices: *<count> { comma separated list of floating point }
        Vertices: *13974 { ... }         ; omitted. Note: These are positions
        PolygonVertexIndex: *27648 {...} ; omitted. Note: These are indices
        
        ; there may be more then 1 geometric normal? don't care, assume there's 1,
        ; otherwise reject the mesh
        LayerElementNormal: 0 {
            ; TODO more in depth
            MappingInformationType: "ByPolygonVertex"
            ReferenceInformationType: "Direct"
            Normals: *82944 { ... } ; omitted
            ; TODO: What's that? teapot has an array of 0s 
            ; probably weight per polygon index, the count is the same (depends on mapping type though)
            NormalsW: *27648 { ... } ; omitted
        }
        
        ; Geometric information on Color
        ; TODO we look it up if the material doesn't have a texture?
        ; as the index suggests, there may be more than one
        LayerElementColor: 0 {
            MappingInformation: "AllSame"
            ReferenceInformationType: "Direct"
            ; note: These *count are _per scalar_ 
            Colors: *4 { ... } ; omitted
        }
        
        ; UV Information. reject if absent ignore with warning if more than one
        LayerElementUV: 0 {
            MappingInformationType: "ByPolygonVertex"
            ReferenceInformationType: "IndexToDirect"
            UV: *55296 { ... } ; omitted
            ; Additional indirection buffer because we are using IndexToDirect
            UVIndex: *27648 { ... }
        }
        
        ; Material information. More than one may be applied. Reject if none ignore with warning if more than one
        LayerElementMaterial: 0 {
            ; We don't want a different material per polygon. Is that what it means?
            MappingInformationType: "ByPolygon"
            ReferenceInformationType: "IndexToDirect"
            Materials: *9216 { ... } ; material indices, but why 9216?
        }
        
        ; Geometry can have multiple layers! Reject those who have 0 layers and warning
        ; if more than 1
        Layer: 0 {
        	LayerElement:  {
				Type: "LayerElementNormal"
				TypedIndex: 0
			}
			LayerElement:  {
				Type: "LayerElementMaterial"
				TypedIndex: 0
			}
			LayerElement:  {
				Type: "LayerElementColor"
				TypedIndex: 0
			}
			LayerElement:  {
				Type: "LayerElementTexture"
				TypedIndex: 0
			}
			LayerElement:  {
				Type: "LayerElementUV"
				TypedIndex: 0
			}
        }
    }
}
```

Other than positions of vertices, which FBX Calls **Control Points**, and **indices**,
other information is stored in **Layers**, where each layer contains **Layer Elements**

- `FbxLayerElementNormal`
- `FbxLayerElementUV`
- `FbxLayerElementColor`
- `FbxLayerElementMaterial`
- `FbxLayerElementSmoothing`
    - Won't support this one, if found, crash

Each Layer elements has two pieces of information

- `MappingComponentType`: **Which mesh component does this value apply to**
    - **Blender** calls this _Domain_
- `ReferenceInformationType`: **How is the value stored and references**

More in depth

| Mapping Type       | Meaning                       |
|--------------------|-------------------------------|
| `eByControlPoint`  | Per Vertex                    |
| `eByPolygonVertex` | Per Vertex Corner             |
| `eByPolygon`       | Per Polygon                   |
| `eAllSame`         | One Value for the entire mesh |

Example mesh to visualize this

```text
Vertices (control points):
0 ---- 1
|    / |
|   /  |
3 ---- 2
    - Note: Winding doesn't matter. The stored order is what counts

Triangles:
T0: (0, 1, 2)
T1: (0, 2, 3)
- 4 Control Points
- 6 Polygon Vertices (3 per triangle)
- 2 polygons

Smooth normals: eByControlPoint (per Vertex) -> 4 Normals in total
UV seams or hard normals: eByPolygonVertex (per corner)
    Seam example:
        Triangle 0 Corners:
        (0,1,3) -> UV[0], UV[1], UV[2]
        
        Triangle 1 Corners:
        (1,2,3) -> UV[3], UV[4], UV[5]
        
        Even though vertex 1 and 3 appeared twice, they have different UVs in the two triangles
Material Assignment or Smoothing Groups: eByPolygon (per face)
    Polygon 0 -> material 0
    Polygon 1 -> material 1
```

Moving to `ReferenceInformationType`: How values are stored

| Reference Type   | Meaning                |
|------------------|------------------------|
| `eDirect`        | Values stored directly |
| `eIndexToDirect` | Values **indexed**     |

Example

```text
# eDirect
- if the normals are accessed with eByControlPoint
  - Normals.DirectArray(controlPointCount): [N0, N1, N2, N3]
  - access: normal = DirectArray[controlPointIndex]
  
# eIndexToDirect
We have a shared buffer accessed through an index buffer
- if the normals are accessed with eByPolygonVertex
  - Normals.DirectArray(arbitrarySize): [N0, N1, N2, N3] 
  - Normals.IndexArray(polygonVertexCount): [0, 1, 2, 0, 2, 3]
  - access: uvIndex = IndexArray[polygonVertexIndex]
            uv      = DirectArray[uvIndex]
```

## ANNEX

### Scene graph organization

traverse hierarchy `FbxScene`, `FbxNode`, `FbxNodeAttribute`

The Global scene settings node contains the scene's
axis system, system units, ambient lighting, and time setting.

This information is accessible through `FbxScene::GetGlobalSettings()`

The `FbxAnimEvaluator node evaluates the animated geometric transformations
of each node in the scene at a specific time.
Accessible by FbxScene::GetEvaluator()

### Fbx Nodes

Nodes are primarily used to specify the position, rotation and scale of scene
elements within a scene.

To get the root node of the hierarchy:

- `FbxScene::GetRootNode()`

To traverse the hierarchy use:

- `FbxNode::GetChild()`
- `FbxNode::GetParent()`
- `FbxNode::GetChildCount()`

Nodes are organized in a hierarchy such that the position, rotation and
scale of a node is described in relation to its parent's coordinate system.

The order in which the rotation and scaling transforms are applied to a parent and its
children is specified by the node's inherit type
The possible values are:

- `eInheritRrSs` : Scaling of parent is applied in the child world after the local child rotation.
- `eInheritRSrs` : Scaling of parent is applied in the parent world.
- `eInheritRrs`  : Scaling of parent does not affect the scaling of children.

Get transformation inherit type:

```cpp
void GetTransformationInheritType(FbxTransform::EInheritType& pInheritType) const;
```

FbxNodeAttribute: objects that are present in a scene. Eg: `FbxMesh`, `FbxLight` or `FbxCamera`

- More than one node attribute can be bound to a single node.
- Similarly, one node attribute can be bound to multiple nodes (instancing).

### Transformation data

This data is represented as a set of FbxPropertyT objects, accessible via
`FbxNode::LclTranslation`, `FbxNode::LclRotation`, `FbxNode::LclScaling.

Ex:
FbxDouble3 translation = lNode->LclTranslation.Get();
FbxDouble3 rotation = lNode->LclRotation.Get();
FbxDouble3 scaling = lNode->LclScaling.Get();

The transformations can be affect to FbxLimits and FbxConstraint, I think for animations.

A node's global and local transformation matrices can be respectively obtained
by calling FbxNode::EvaluateGlobalTransform() and FbxNode::EvaluateLocalTransform():

Meshes
FbxMesh -> abstracts the mesh information, like the vertices. A single instance
of FbxMesh can be bound to multiple instances of FbxNode. Scene geometry
uses the concept of layers and layers elements to define normal, material
texture maps.

Meshes-Vertex

The "Control points" are the per-face vertices.
The system used by FBX is right handed, Y-up axis system
FbxGeometry is the parent class of FbxMesh
the point are defined with FbxVector4, ex: FbxVector4 vertex0(-50, 0, 50);
Get the ControlPoints that are defined in a FbxMesh object:
FbxVector4* lControlPoints = lMesh->GetControlPoints(); lControlPoints is a vector.
Note that a scene axis conversion does not affect the vertex values of a mesh.

Polygon Management

Begins the process of adding a polygon to the mesh:

void BeginPolygon(int pMaterial=-1, int pTexture=-1, int pGroup=-1, bool pLegacy=true);

End writing a polygon, it should be called after adding one polygon:
void EndPolygon();

Add a polygon vertex to the current polygon.
void AddPolygon(int pIndex, int pTextureUVIndex = -1);
param pIndex Index in the table of the control points.
param pTextureUVIndex Index of texture UV coordinates to assign to this polygon if texture UV mapping type is \e
eByPolygonVertex. Otherwise it must be \c -1.
remark After adding all the polygons of the mesh, call function "BuildMeshEdgeArray" to generate edge data for the
mesh. */

To get the number of polygons in the mesh.
inline int GetPolygonCount() const { return mPolygons.GetCount(); }

Get the number of polygon vertices in a polygon.
inline int GetPolygonSize(int pPolygonIndex) const

Get the current group ID of the specified polygon.
int GetPolygonGroup(int pPolygonIndex) const;

Get a polygon vertex (i.e: an index to a control point).
inline int GetPolygonVertex(int pPolygonIndex, int pPositionInPolygon) const

Get the normal associated with the specified polygon vertex.
bool GetPolygonVertexNormal(int pPolyIndex, int pVertexIndex, FbxVector4& pNormal) const;

Get the normals associated with the mesh for every polygon vertex.
GetPolygonVertexNormals(FbxArray<FbxVector4>& pNormals) const;

Get the UV associated with the specified polygon vertex.
bool GetPolygonVertexUV(int pPolyIndex, int pVertexIndex, const char* pUVSetName, FbxVector2& pUV, bool& pUnmapped)
const;

Get the UVs associated with the mesh for every polygon vertex.
bool GetPolygonVertexUVs(const char* pUVSetName, FbxArray<FbxVector2>& pUVs, FbxArray<int>* pUnmappedUVId = NULL) const;

Get the array of polygon vertices (i.e: indices to the control points).
int* GetPolygonVertices() const;

Gets the number of polygon vertices in the mesh.
inline int GetPolygonVertexCount() const { return mPolygonVertices.Size();

Gets the start index into the array returned by GetPolygonVertices() for the given polygon.
int GetPolygonVertexIndex(int pPolygonIndex) const;

To access polygon information
int lStartIndex = mesh.GetPolygonVertexIndex(3);
if( lStartIndex == -1 ) return;
int* lVertices = mesh.GetPolygonVertices()[lStartIndex];
int lCount = mesh.GetPolygonSize(3);
for( int i = 0; i < lCount; ++i )
{
int vertexID = lVertices[i];
}

Mesh->UV info

Get the number of texture UV coordinates.
int GetTextureUVCount(FbxLayerElement::EType pTypeIdentifier=FbxLayerElement::eTextureDiffuse);
param pTypeIdentifier The texture channel the UV refers to.

Get the number of layer containing at least one channel UVMap.
int GetUVLayerCount() const;
return 0 if no UV maps have been defined.

Fills an array describing, for the given layer, which texture channel have UVs associated to it.
FbxArray<FbxLayerElement::EType> GetAllChannelUV(int pLayer);

Get a texture UV index associated with a polygon vertex (i.e: an index to a control point).
int GetTextureUVIndex(int pPolygonIndex, int pPositionInPolygon, FbxLayerElement::EType
pTypeIdentifier=FbxLayerElement::eTextureDiffuse);

Meshes-Normals

FbxLayerElementNormal contains the information about the normal vectors
of a mesh.
These can be mapped in a different ways:

control point (FbxLayerElement::eByControlPoint),
by polygon vertex (FbxLayerElement::eByPolygonVertex),
by polygon (FbxLayerElement::eByPolygon),
by edge (FbxLayerElement::eByEdge), or one mapping coordinate for the whole
surface (FbxLayerElement::eAllSame).

The array of normal vectors can be referenced by the array of control points in
different RecferceMode assigend in FbxLayerElement::SetReferenceMode(). These are:

FbxLayerElement::eDirect : This indicates that the mapping information for the
n'th element is found in the n'th place of FbxLayerElementTemplate::mDirectArray

FbxLayerElement::eIndex (FBX v5.0) : it is equal to eIndexToDirect

FbxLayerElement::eIndexToDirect : This indicates that each element of
FbxLayerElementTemplate::mIndexArray contains an index referring to an element
in FbxLayerElementTemplate::mDirectArray.

To obtain the reference mode used use the GetReferenceMode()

EReferenceMode GetReferenceMode() const { return mReferenceMode; }

Layer

For get a layer that can define the normal vectors information:

FbxLayer lLayer = lMesh->GetLayer(0);

To get the layer information about the normal use the:

const FbxLayerElementNormal* GetNormals() const;

So get the Mapping Mode.
EMappingMode GetMappingMode() const { return mMappingMode; }

So to access in the layer array use:
lLayerElementNormal->GetDirectArray()

Get the value in the array:

inline T GetAt(int pIndex) const
Returns the specified item's value.
param pIndex Index of the item
return The value of the specified item
remarks If the index is invalid, pItem is set to zero.

Instancing

We can have multiple instances of the same FbxNode node through a instancing.
this technique is realized creating a new Fbx object that point to a FbxNode node base:

Give a scene and the mesh base
create a FbxNode
FbxNode* lNode = FbxNode::Create(pScene,pName);
set the node attribute
lNode->SetNodeAttribute(pFirstCube);

So this can do for each scene element.

Materials

FbxSurfaceMaterial

Contains material settings

FbxSurfaceLambert

Ambient color property.
Ambient

Emissive color property.
Emissive

Diffuse color property.
Diffuse

NormalMap property. This property can be used to specify the distortion of the surface
normals and create the illusion of a bumpy surface.
NormalMap

Transparent color property.
TransparentColor

Reflection color property. This property is used to
implement reflection mapping.
Reflection

Reflection factor property. This property is used to
attenuate the reflection color.
ReflectionFactor

To create the material:

FbxString lMaterialName = "toto";
lMaterial = FbxSurfacePhong::Create(pScene, lMaterialName.Buffer());

To get the material ex:

lMaterial = lNode->GetSrcObject<FbxSurfacePhong>(0);

To assign a material to a polygon pass the material index to the BeginPolygon function:
lMesh->BeginPolygon(0); // Material index

Textures

FbxTexture

Is the base class for the textures and describes image mapping on top of geometry.

Importan attributes:
Type description

Alpha: EAlphaSource GetAlphaSource() const;
This property handles the default alpha value for textures.

CurrentMappingType: EMappingType GetMappingType() const;
eUV: Apply texture to the model according to UVs.
eBumpNormalMap: Bump map: Texture contains two direction vectors, that are used to convey relief in a texture.

WrapModeU: GetWrapModeU()
This property handles the texture wrap modes in U. Default value is eRepeat.

WrapModeV: GetWrapModeV()
This property handles the texture wrap modes in V. Default value is eRepeat.

enum EWrapMode: eRepeat, eClamp

UVSwap: bool GetSwapUV() const;
This property handles the swap UV flag.
If swap UV flag is enabled, the texture's width and height are swapped.
Default value is false.

PremultiplyAlpha: bool GetPremultiplyAlpha() const;
This property handles the PremultiplyAlpha flag. If PremultiplyAlpha flag is true,
the R, G, and B components you store have already been multiplied in with the alpha.
�Default value is true.

Texture positioning

Translation: GetTranslationU() GetTranslationV() GetDefaultT(FbxVector4& pR)
This property handles the default translation vector.
Rotation:
This property handles the default rotation vector.

Scaling: SetScale(double pU,double pV); GetScaleV(), GetScaleU() const; GetDefaultS(FbxVector4& pR)
This property handles the default scale vector.

RotationPivot: GetRotationU(), GetRotationV(), GetRotationW()  GetDefaultR(FbxVector4& pR)
This property handles the rotation pivot vector.

ScalingPivot;
This property handles the scaling pivot vector.
UV set to use.
This property handles the use of UV sets.
Default value is "default".

Referencing media

When a binary FBX file containing embedded media is imported the media will be extracted from
the file and copied into a subdirectory. By default, this subdirectory will be created at
the location of the FBX file with the same name as the file, with the extension .fbm.

FbxFileTexture

Represents any texture loaded from a file. To apply a texture to
geometry, first connect the geometry to a FbxSurfaceMaterial object
(e.g. FbxSurfaceLambert) and then connect one of its properties (e.g. Diffuse) to
the FbxFileTexture object.

The teture need to be connected to a property on the material. So you need to
find the material of the FbxNode associated to the mesh node.

To get the relative texture file path:

const char* GetRelativeFileName() const;
return The relative texture file path.
remarks An empty string is returned
if FbxFileTexture::SetRelativeFileName() has not been called before.

To get the materia use:

EMaterialUse GetMaterialUse() const;
Returns the material use.
return How the texture uses model material.

EMaterialUse values:

eModelMaterial, //! Texture uses model material.
eDefaultMaterial //! Texture does not use model material.

FbxLayerElementUV

To create the texture layers in the mesh:

FbxMesh::CreateElementUV()

There are created a number of texture layers in the mesh equal to the number
of used material channel like diffuse, ambient and emissive.

FxbGeometryElementUV -> define the mapping the texture's UV coordinates to each
of the polygon's vertices.

FbxgeometryBase

Is used to manage the control points, normals, binormals and tangents of
the geometries.

To get the geometry's UV element.
FbxGeometryElementUV* GetElementUV(int pIndex = 0, FbxLayerElement::EType pTypeIdentifier=FbxLayerElement::eUnknown);

param pIndex The UV geometry element index.
param pTypeIdentifier The texture channel the UVIndex refers to.
return A pointer to the geometry element or \c NULL if \e pIndex is out of range.
remarks If e pTypeIdentifier is not specified, the function will return the geometry
element regardless of its texture type.

To get the shading mode:

EShadingMode GetShadingMode() const
return The currently set shading mode.

It is usefull to find the type of shading of the node and if are the defined texture
for a node

enum EShadingMode:
eHardShading,
eWireFrame,
eFlatShading,
eLightShading,
eTextureShading,
eFullShading

World space
right x pos
Up z pos
forward y

camera space -> sys DirectX

To see:

FbxLayerElementBinormal

TriangleMesh
