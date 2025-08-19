#include "core-mesh-parser.h"

#include "fbxsdk.h"

namespace dmt {

    // Instantiate FBX SDK memory management object-> FbxManager
    //import a contents of an FBX file -> FbxIOSettings, FbxImporter, FbxScene
    //
    //Scene graph organization
    //
    //traverse hierarchy FbxScene, FbxNode, FbxNodeAttribute
    // elements information -> FbxNode, FbxNodeAttribute, FbxString
    //
    // FbxScene -> abstracts the scene graph, and it is organized as a hierarchy
    // of nodes
    // FbxNode -> abstracts the nodes of scene graph, anc contains the data of the
    // elements of the scene, the properties are described by FbxNodeAttribute
    //
    // The Global scene settings node contains the scene's
    // axis system, system units, ambient lighting, and time setting.
    // These information are accessable with FbxScene::GetGlobalSettings()
    //
    // The FbxAnimEvaluator node evaluates the animated geometric transformations
    // of each node in the scene at a specific time.
    // Accessible by FbxScene::GetEvaluator()
    //
    // Fbx Nodes
    //
    // Nodes are primarily used to specify the position, rotation and scale of scene
    // elements within a scene.
    //
    // To get the root node of the hirarchy:
    // FbxScene::GetRootNode()
    //
    // To traversed the hierarchy use:
    // FbxNode::GetChild()
    // FbxNode::GetParent()
    // FbxNode::GetChildCount()
    //
    // Nodes are organized in a hierarchy such that the position, rotation and
    // scale of a node is described in relation to its parent's coordinate system.
    //
    // The order in which the rotation and scaling transforms are applied to a parent and its
    // children is specified by the node's inherit type
    // The possible values are:
    //     - eInheritRrSs : Scaling of parent is applied in the child world after the local child rotation.
    //     - eInheritRSrs : Scaling of parent is applied in the parent world.
    //     - eInheritRrs  : Scaling of parent does not affect the scaling of children.
    //
    // Get transformation inherit type:
    // void GetTransformationInheritType(FbxTransform::EInheritType& pInheritType) const;
    //
    //
    // FbxNodeAttribute: objects thate are present in a scene. Like: FbxMesh, FbxLight or FbxCamera
    //
    // More than one node attribute can be bound to a single node.
    // Similarly, one node attribute can be bound to multiple nodes (instancing).
    //
    // Transformation data
    //
    // This data is represented as a set of FbxPropertyT objects, accessible via
    // FbxNode::LclTranslation, FbxNode::LclRotation, FbxNode::LclScaling.
    //
    // Ex:
    // FbxDouble3 translation = lNode->LclTranslation.Get();
    // FbxDouble3 rotation = lNode->LclRotation.Get();
    // FbxDouble3 scaling = lNode->LclScaling.Get();
    //
    // The transformations can be affect to FbxLimits and FbxConstraint, I think for animations.
    //
    // A node's global and local transformation matrices can be respectively obtained
    // by calling FbxNode::EvaluateGlobalTransform() and FbxNode::EvaluateLocalTransform():
    //
    // Meshes
    // FbxMesh -> abstracts the mesh information, like the vertices. A single instance
    // of FbxMesh can be bound to multiple instances of FbxNode. Scene geometry
    // uses the concept of layers and layers elements to define normal, material
    // texture maps.
    //
    // Meshes-Vertex
    //
    // The "Control points" are the per-face vertices.
    // The system used by FBX is right handed, Y-up axis system
    // FbxGeometry is the parent class of FbxMesh
    // the point are defined with FbxVector4, ex: FbxVector4 vertex0(-50, 0, 50);
    // Get the ControlPoints that are defined in a FbxMesh object:
    // FbxVector4* lControlPoints = lMesh->GetControlPoints(); lControlPoints is a vector.
    // Note that a scene axis conversion does not affect the vertex values of a mesh.
    //
    // Polygon Management
    //
    // Begins the process of adding a polygon to the mesh:
    //
    // void BeginPolygon(int pMaterial=-1, int pTexture=-1, int pGroup=-1, bool pLegacy=true);
    //
    // End writing a polygon, it should be called after adding one polygon:
    // void EndPolygon();
    //
    // Add a polygon vertex to the current polygon.
    // void AddPolygon(int pIndex, int pTextureUVIndex = -1);
    // param pIndex Index in the table of the control points.
    // param pTextureUVIndex Index of texture UV coordinates to assign to this polygon if texture UV mapping type is \e eByPolygonVertex. Otherwise it must be \c -1.
    // remark After adding all the polygons of the mesh, call function "BuildMeshEdgeArray" to generate edge data for the mesh. */
    //
    //
    // To get the number of polygons in the mesh.
    // inline int GetPolygonCount() const { return mPolygons.GetCount(); }
    //
    // Get the number of polygon vertices in a polygon.
    // inline int GetPolygonSize(int pPolygonIndex) const
    //
    // Get the current group ID of the specified polygon.
    // int GetPolygonGroup(int pPolygonIndex) const;
    //
    // Get a polygon vertex (i.e: an index to a control point).
    // inline int GetPolygonVertex(int pPolygonIndex, int pPositionInPolygon) const
    //
    // Get the normal associated with the specified polygon vertex.
    // bool GetPolygonVertexNormal(int pPolyIndex, int pVertexIndex, FbxVector4& pNormal) const;
    //
    // Get the normals associated with the mesh for every polygon vertex.
    // GetPolygonVertexNormals(FbxArray<FbxVector4>& pNormals) const;
    //
    // Get the UV associated with the specified polygon vertex.
    // bool GetPolygonVertexUV(int pPolyIndex, int pVertexIndex, const char* pUVSetName, FbxVector2& pUV, bool& pUnmapped) const;
    //
    // Get the UVs associated with the mesh for every polygon vertex.
    // bool GetPolygonVertexUVs(const char* pUVSetName, FbxArray<FbxVector2>& pUVs, FbxArray<int>* pUnmappedUVId = NULL) const;
    //
    // Get the array of polygon vertices (i.e: indices to the control points).
    // int* GetPolygonVertices() const;
    //
    //
    // Gets the number of polygon vertices in the mesh.
    // inline int GetPolygonVertexCount() const { return mPolygonVertices.Size();
    //
    // Gets the start index into the array returned by GetPolygonVertices() for the given polygon.
    // int GetPolygonVertexIndex(int pPolygonIndex) const;
    //
    // To access polygon information
    // int lStartIndex = mesh.GetPolygonVertexIndex(3);
    // if( lStartIndex == -1 ) return;
    // int* lVertices = mesh.GetPolygonVertices()[lStartIndex];
    // int lCount = mesh.GetPolygonSize(3);
    // for( int i = 0; i < lCount; ++i )
    // {
    //     int vertexID = lVertices[i];
    //	}
    //
    //
    // Mesh->UV info
    //
    // Get the number of texture UV coordinates.
    // int GetTextureUVCount(FbxLayerElement::EType pTypeIdentifier=FbxLayerElement::eTextureDiffuse);
    // param pTypeIdentifier The texture channel the UV refers to.
    //
    // Get the number of layer containing at least one channel UVMap.
    // int GetUVLayerCount() const;
    // return 0 if no UV maps have been defined.
    //
    //
    // Fills an array describing, for the given layer, which texture channel have UVs associated to it.
    // FbxArray<FbxLayerElement::EType> GetAllChannelUV(int pLayer);
    //
    // Get a texture UV index associated with a polygon vertex (i.e: an index to a control point).
    // int GetTextureUVIndex(int pPolygonIndex, int pPositionInPolygon, FbxLayerElement::EType pTypeIdentifier=FbxLayerElement::eTextureDiffuse);
    //
    // Meshes-Normals
    //
    // FbxLayerElementNormal contains the information about the normal vectors
    // of a mesh.
    // These can be mapped in a different ways:
    //
    // control point (FbxLayerElement::eByControlPoint),
    // by polygon vertex (FbxLayerElement::eByPolygonVertex),
    // by polygon (FbxLayerElement::eByPolygon),
    // by edge (FbxLayerElement::eByEdge), or one mapping coordinate for the whole
    // surface (FbxLayerElement::eAllSame).
    //
    // The array of normal vectors can be referenced by the array of control points in
    // different RecferceMode assigend in FbxLayerElement::SetReferenceMode(). These are:
    //
    // FbxLayerElement::eDirect : This indicates that the mapping information for the
    // n'th element is found in the n'th place of FbxLayerElementTemplate::mDirectArray
    //
    // FbxLayerElement::eIndex (FBX v5.0) : it is equal to eIndexToDirect
    //
    // FbxLayerElement::eIndexToDirect : This indicates that each element of
    // FbxLayerElementTemplate::mIndexArray contains an index referring to an element
    // in FbxLayerElementTemplate::mDirectArray.
    //
    // To obtain the reference mode used use the GetReferenceMode()
    //
    // EReferenceMode GetReferenceMode() const { return mReferenceMode; }
    //
    // Layer
    //
    // For get a layer that can define the normal vectors information:
    //
    // FbxLayer lLayer = lMesh->GetLayer(0);
    //
    // To get the layer information about the normal use the:
    //
    // const FbxLayerElementNormal* GetNormals() const;
    //
    // So get the Mapping Mode.
    // EMappingMode GetMappingMode() const { return mMappingMode; }
    //
    // So to access in the layer array use:
    // lLayerElementNormal->GetDirectArray()
    //
    // Get the value in the array:
    //
    // inline T  GetAt(int pIndex) const
    // Returns the specified item's value.
    // param pIndex               Index of the item
    // return                     The value of the specified item
    // remarks                    If the index is invalid, pItem is set to zero.
    //
    //
    // Instancing
    //
    // We can have multiple instances of the same FbxNode node through a instancing.
    // this technique is realized creating a new Fbx object that point to a FbxNode node base:
    //
    // Give a scene and the mesh base
    // create a FbxNode
    // FbxNode* lNode = FbxNode::Create(pScene,pName);
    // set the node attribute
    // lNode->SetNodeAttribute(pFirstCube);
    //
    //  So this can do for each scene element.
    //
    // Materials
    //
    // FbxSurfaceMaterial
    //
    // Contains material settings
    //
    // FbxSurfaceLambert
    //
    // Ambient color property.
    // Ambient
    //
    // Emissive color property.
    // Emissive
    //
    // Diffuse color property.
    // Diffuse
    //
    // NormalMap property. This property can be used to specify the distortion of the surface
    // normals and create the illusion of a bumpy surface.
    // NormalMap
    //
    // Transparent color property.
    // TransparentColor
    //
    // Reflection color property. This property is used to
    // implement reflection mapping.
    // Reflection
    //
    //
    // Reflection factor property. This property is used to
    // attenuate the reflection color.
    // ReflectionFactor
    //
    // To create the material:
    //
    // FbxString lMaterialName = "toto";
    // lMaterial = FbxSurfacePhong::Create(pScene, lMaterialName.Buffer());
    //
    // To get the material ex:
    //
    // lMaterial = lNode->GetSrcObject<FbxSurfacePhong>(0);
    //
    //  To assign a material to a polygon pass the material index to the BeginPolygon function:
    //  lMesh->BeginPolygon(0); // Material index
    //
    //
    // Textures
    //
    // FbxTexture
    //
    // Is the base class for the textures and describes image mapping on top of geometry.
    //
    // Importan attributes:
    // Type description
    //
    // Alpha: EAlphaSource GetAlphaSource() const;
    // This property handles the default alpha value for textures.
    //
    // CurrentMappingType: EMappingType GetMappingType() const;
    // eUV: Apply texture to the model according to UVs.
    // eBumpNormalMap: Bump map: Texture contains two direction vectors, that are used to convey relief in a texture.
    //
    // WrapModeU: GetWrapModeU()
    // This property handles the texture wrap modes in U. Default value is eRepeat.
    //
    //
    // WrapModeV: GetWrapModeV()
    // This property handles the texture wrap modes in V.  Default value is eRepeat.
    //
    // enum EWrapMode: eRepeat, eClamp
    //
    // UVSwap: bool GetSwapUV() const;
    // This property handles the swap UV flag.
    // If swap UV flag is enabled, the texture's width and height are swapped.
    // Default value is false.
    //
    // PremultiplyAlpha: bool GetPremultiplyAlpha() const;
    // This property handles the PremultiplyAlpha flag. If PremultiplyAlpha flag is true,
    // the R, G, and B components you store have already been multiplied in with the alpha.
    // ùDefault value is true.

    // Texture positioning
    //
    // Translation: GetTranslationU() GetTranslationV() GetDefaultT(FbxVector4& pR)
    // This property handles the default translation vector.
    // Rotation:
    // This property handles the default rotation vector.
    //
    // Scaling: SetScale(double pU,double pV); GetScaleV(), GetScaleU() const; GetDefaultS(FbxVector4& pR)
    // This property handles the default scale vector.
    //
    // RotationPivot: GetRotationU(), GetRotationV(), GetRotationW()  GetDefaultR(FbxVector4& pR)
    // This property handles the rotation pivot vector.
    //
    // ScalingPivot;
    // This property handles the scaling pivot vector.
    // UV set to use.
    // This property handles the use of UV sets.
    // Default value is "default".
    //
    //
    //
    // Referencing media
    //
    // When a binary FBX file containing embedded media is imported the media will be extracted from
    // the file and copied into a subdirectory. By default, this subdirectory will be created at
    // the location of the FBX file with the same name as the file, with the extension .fbm.
    //
    // FbxFileTexture
    //
    // Represents any texture loaded from a file. To apply a texture to
    // geometry, first connect the geometry to a FbxSurfaceMaterial object
    // (e.g. FbxSurfaceLambert) and then connect one of its properties (e.g. Diffuse) to
    // the FbxFileTexture object.
    //
    // The teture need to be connected to a property on the material. So you need to
    // find the material of the FbxNode associated to the mesh node.
    //
    // To get the relative texture file path:
    //
    // const char* GetRelativeFileName() const;
    // return             The relative texture file path.
    // remarks            An empty string is returned
    // if FbxFileTexture::SetRelativeFileName() has not been called before.
    //
    // To get the materia use:
    //
    // EMaterialUse GetMaterialUse() const;
    // Returns the material use.
    // return How the texture uses model material.
    //
    // EMaterialUse values:
    //
    //  eModelMaterial,		//! Texture uses model material.
    //  eDefaultMaterial	//! Texture does not use model material.
    //
    // FbxLayerElementUV
    //
    // To create the texture layers in the mesh:
    //
    // FbxMesh::CreateElementUV()
    //
    //
    // There are created a number of texture layers in the mesh equal to the number
    // of used material channel like diffuse, ambient and emissive.
    //
    // FxbGeometryElementUV -> define the mapping the texture's UV coordinates to each
    // of the polygon's vertices.
    //
    // FbxgeometryBase
    //
    // Is used to manage the control points, normals, binormals and tangents of
    // the geometries.
    //
    // To get the geometry's UV element.
    // FbxGeometryElementUV* GetElementUV(int pIndex = 0, FbxLayerElement::EType pTypeIdentifier=FbxLayerElement::eUnknown);
    //
    // param pIndex           The UV geometry element index.
    // param pTypeIdentifier  The texture channel the UVIndex refers to.
    // return                 A pointer to the geometry element or \c NULL if \e pIndex is out of range.
    // remarks                If e pTypeIdentifier is not specified, the function will return the geometry
    // element regardless of its texture type.
    //
    //
    // To get the shading mode:
    //
    // EShadingMode GetShadingMode() const
    // return The currently set shading mode.
    //
    // It is usefull to find the type of shading of the node and if are the defined texture
    // for a node
    //
    // enum EShadingMode:
    // eHardShading,
    // eWireFrame,
    // eFlatShading,
    // eLightShading,
    // eTextureShading,
    // eFullShading
    //
    // World space
    // right x pos
    // Up z pos
    // forward y
    //
    //camera space -> sys DirectX
    //
    //
    //
    // To see:
    //
    // FbxLayerElementBinormal
    //
    //
    //TriangleMesh

    static void TextureNames(FbxGeometry* pGeometry, std::pmr::unordered_map<char const *, char const *>& chTex);
    static void GetTextureName(FbxProperty property, std::pmr::unordered_map<char const *, char const *>& chTex, uint32_t texIdx, uint32_t matIdx);

    void        FbxDeleter::operator()(void* raw) const
    {
        auto* manager = reinterpret_cast<FbxManager*>(raw);
        if (manager)
        {
            manager->Destroy();
        }
    }

    void FbxSettingsDeleter::operator()(void* raw) const
    {
        auto* settings = reinterpret_cast<FbxIOSettings*>(raw);
        if (settings)
        {
            settings->Destroy();
        }
    }

    void FbxSceneDeleter::operator()(void* raw) const
    {
        auto* importer = reinterpret_cast<FbxScene*>(raw);
        if (importer)
        {
            importer->Destroy();
        }
    }

    void FbxDeleterResources::operator()(void* raw) const
    {

        auto* resources = reinterpret_cast<FbxResources*>(raw);

        if (resources->settings)
        {
            reinterpret_cast<FbxIOSettings*>(resources->settings)->Destroy();
        }
        if (resources->manager)
        {
            reinterpret_cast<FbxManager*>(resources->settings)->Destroy();
        }
    };

    static FbxManager*    getManager(dFbxManager& inst) { return reinterpret_cast<FbxManager*>(inst.get()); }
    static FbxIOSettings* getIoSettings(dFbxIOSettings& inst) { return reinterpret_cast<FbxIOSettings*>(inst.get()); }
    static FbxScene*      getScene(dFbxScene& inst) { return reinterpret_cast<FbxScene*>(inst.get()); }

    dFbxManager createFBXInstance() { return dFbxManager{FbxManager::Create(), FbxDeleter{}}; }

    MeshFbxPasser::MeshFbxPasser()
    {
        InitFbxManager();
        //m_settings = dFbxIOSettings{FbxIOSettings::Create(getManager(m_mng), IOSROOT), FbxSettingsDeleter{}};
        m_res.settings     = FbxIOSettings::Create(reinterpret_cast<FbxManager*>(m_res.manager), IOSROOT);
        FbxIOSettings* set = reinterpret_cast<FbxIOSettings*>(m_res.settings);
        //set flags for import settings
        set->SetBoolProp(IMP_FBX_MATERIAL, false);
        set->SetBoolProp(IMP_FBX_TEXTURE, true);
        set->SetBoolProp(IMP_FBX_LINK, false);
        set->SetBoolProp(IMP_FBX_SHAPE, false);
        set->SetBoolProp(IMP_FBX_GOBO, false);
        set->SetBoolProp(IMP_FBX_ANIMATION, false);
        set->SetBoolProp(IMP_FBX_GLOBAL_SETTINGS, true);
        set->SetBoolProp(IMP_FBX_NORMAL, true);
    }

    bool MeshFbxPasser::ImportFBX(char const*                fileName,
                                  TriangleMesh*              outMesh,
                                  std::pmr::memory_resource* memory = std::pmr::get_default_resource())
    {


        FbxManager* mng = reinterpret_cast<FbxManager*>(m_res.manager);


        //check FBX's filename
        os::Path const fbxDirectory = os::Path::executableDir() / fileName;
        if (!fbxDirectory.isValid() || !fbxDirectory.isFile())
            return false;

        m_fileName              = fbxDirectory.toUnderlying();
        std::string fileNameStr = std::string(fileName);

        if (fileNameStr.size() < 4)
            return false;

        //Define the scene name
        std::string sceneName = fileNameStr.substr(0, fileNameStr.size() - 4) + "_scene";
        //Initialize the scene
        FbxScene* pScene = FbxScene::Create(mng, sceneName.c_str());
        //Initialize the Importer
        FbxImporter* fbxImporter  = FbxImporter::Create(mng, sceneName.c_str());
        bool         importStatus = fbxImporter->Initialize(fileName, -1, mng->GetIOSettings());

        if (!importStatus)
            return false;
        //Import the scene
        importStatus = fbxImporter->Import(pScene);

        if (!importStatus)
            return false;

        fbxImporter->Destroy();

        //Channels<->TexturePath 
        m_ChannelsTexPath = std::pmr::unordered_map<char const*, char const*>{memory};
        m_ChannelsTexPath.reserve(4);
        //Get Info about the scene
        FbxAxisSystem as = pScene->GetGlobalSettings().GetAxisSystem();
        FbxSystemUnit su = pScene->GetGlobalSettings().GetSystemUnit();
        //unit sys conversion
        if (pScene->GetGlobalSettings().GetSystemUnit() == FbxSystemUnit::cm)
        {
            const FbxSystemUnit::ConversionOptions lConversionOptions = {
                true, /* mConvertRrsNodes */
                true, /* mConvertLimits */
                true, /* mConvertClusters */
                true, /* mConvertLightIntensity */
                true, /* mConvertPhotometricLProperties */
                true  /* mConvertCameraClipPlanes */
            };

            // Convert the scene to meters using the defined options.
            FbxSystemUnit::m.ConvertScene(pScene, lConversionOptions);
        }

        if (pScene->GetGlobalSettings().GetAxisSystem() != FbxAxisSystem::MayaZUp)
        {
            // Convert the scene to meters using the defined options.
            FbxAxisSystem::MayaZUp.ConvertScene(pScene);
        }

        assert(pScene->GetGlobalSettings().GetSystemUnit() == FbxSystemUnit::m);
        assert(pScene->GetGlobalSettings().GetAxisSystem() == FbxAxisSystem::MayaZUp);


        //attraverse the hirarchy
        FbxNode* lNode = pScene->GetRootNode();

        if (lNode)
        {
            for (uint32_t i = 0; i < lNode->GetChildCount(); i++)
            {
                FbxNode* lChild = lNode->GetChild(i);

                if (lChild->GetNodeAttribute() == nullptr)
                {
                    return false;
                }
                else
                {
                    FbxNodeAttribute::EType lAttributeType = (lChild->GetNodeAttribute()->GetAttributeType());


                    if (lAttributeType != FbxNodeAttribute::eMesh)
                        continue;

                    //Get Mesh Information
                    FbxMesh* lMesh                  = (FbxMesh*)lChild->GetNodeAttribute();
                    m_meshName                      = lMesh->GetName();
                    uint32_t    nPolygon            = lMesh->GetPolygonCount();
                    FbxVector4* lControlPointsArray = lMesh->GetControlPoints();
                    uint32_t    nElementNormal      = lMesh->GetElementNormalCount();
                    uint32_t    nElementBiNormal    = lMesh->GetElementBinormalCount();
                    FbxAMatrix  globalTransform     = lChild->EvaluateGlobalTransform();
                    //Polygon

                    if (nPolygon <= 0)
                        return false;

                    //TriangleMesh tMesh{nPolygon, memory};

                    //handle the mesh with Polygon groups
                    for (int l = 0; l < lMesh->GetElementPolygonGroupCount(); l++)
                    {
                        FbxGeometryElementPolygonGroup* lPolyGroup = lMesh->GetElementPolygonGroup(l);
                        switch (lPolyGroup->GetMappingMode())
                        {
                            case FbxGeometryElement::eByPolygon:
                                if (lPolyGroup->GetReferenceMode() == FbxGeometryElement::eIndexToDirect)
                                {
                                    int polyGroupId = lPolyGroup->GetIndexArray().GetAt(i);
                                    //todo read vertex points
                                }
                            default: break;
                        }
                    }
                    //FbxAxisSystem d;

                    //Handle single polygon
                    int vertexId = 0;
                    for (uint32_t k = 0; k < nPolygon; k++)
                    {
                        //polygon size
                        uint32_t polygonSize = lMesh->GetPolygonSize(k);
                        if (polygonSize > 4 && polygonSize > 2)
                            return false;

                        VertexIndex vIdx[4];

                        for (uint32_t j = 0; j < polygonSize; j++, vertexId++)
                        {
                            //normals handle
                            for (uint32_t k = 0; k < nElementNormal; k++)
                            {
                                FbxGeometryElementNormal* lENormal = lMesh->GetElementNormal(k);
                                Normal3f                  normal{};
                                if (lENormal->GetMappingMode() == FbxGeometryElement::eByControlPoint)
                                {
                                    switch (lENormal->GetReferenceMode())
                                    {
                                        case FbxGeometryElement::eDirect:
                                        {
                                            normal.x = static_cast<float>(lENormal->GetDirectArray().GetAt(vertexId)[0]);
                                            normal.y = static_cast<float>(lENormal->GetDirectArray().GetAt(vertexId)[1]);
                                            normal.z = static_cast<float>(lENormal->GetDirectArray().GetAt(vertexId)[2]);
                                        }
                                        break;
                                        case FbxGeometryElement::eIndexToDirect:
                                        {
                                            int id   = lENormal->GetIndexArray().GetAt(vertexId);
                                            normal.x = static_cast<float>(lENormal->GetDirectArray().GetAt(id)[0]);
                                            normal.y = static_cast<float>(lENormal->GetDirectArray().GetAt(id)[1]);
                                            normal.z = static_cast<float>(lENormal->GetDirectArray().GetAt(id)[2]);
                                        }
                                        break;
                                        default: break;
                                    }

                                    uint32_t nidx = 0;
                                    if (outMesh->checkNormal(normal, nidx))
                                    {
                                        vIdx[j].normalIdx = nidx;
                                    }
                                    else
                                    {
                                        outMesh->addNormal(normal);
                                        vIdx[j].normalIdx = outMesh->getNormalSize() - 1;
                                    }
                                }
                            }

                            int controlPointIndex = lMesh->GetPolygonVertex(k, j);


                            if (controlPointIndex < 0)
                            {
                                return false;
                            }
                            //vertex
                            uint32_t   idx         = 0;
                            FbxVector4 cp          = lControlPointsArray[controlPointIndex];
                            FbxVector4 transformed = globalTransform.MultT(cp);
                            Point3f    controlPoint{
                                   {static_cast<float>(cp[0]), static_cast<float>(cp[1]), static_cast<float>(cp[2])}};

                            if (outMesh->checkPosition(controlPoint, idx))
                            {
                                vIdx[j].positionIdx = idx;
                            }
                            else
                            {
                                outMesh->addPosition(controlPoint);
                                vIdx[j].positionIdx = outMesh->getPositionSize() - 1;
                            }
                            //add polygon
                            if (j > 1)
                            {
                                outMesh->addIndexedTriangle(vIdx[j - 2], vIdx[j], vIdx[j - 1], -1);
                            }
                        }
                    }

                    //handle textures
                    TextureNames(lMesh, this->m_ChannelsTexPath);
                }
            }
        }


        pScene->Destroy();


        return true;
    }

    char const* MeshFbxPasser::GetMeshName() { return m_meshName.c_str(); }

    MeshFbxPasser::~MeshFbxPasser() {}

    void MeshFbxPasser::InitFbxManager()
    {
        if (m_res.manager == nullptr)
        {
            m_res.manager = FbxManager::Create();
        }
    }

    void TextureNames(FbxGeometry* pGeometry, std::pmr::unordered_map<char const *, char const *>& chTex)
    {
         
        if(pGeometry->GetNode()==NULL)
        return;

        uint32_t nMat = pGeometry->GetNode()->GetSrcObjectCount<FbxSurfaceMaterial>();
        for (uint32_t matIdx = 0; matIdx < nMat; matIdx++)
        {
            FbxSurfaceMaterial* lMaterial = pGeometry->GetNode()->GetSrcObject<FbxSurfaceMaterial>(matIdx);

            //go through all the possible textures
            if(lMaterial)
            {

                uint32_t texIdx;
                FBXSDK_FOR_EACH_TEXTURE(texIdx)
                {
                    
                    FbxProperty property = lMaterial->FindProperty(FbxLayerElement::sTextureChannelNames[texIdx]);                   
                    GetTextureName(property, chTex, texIdx, matIdx); 
                }

            }

        }
    }

    void GetTextureName(FbxProperty property, std::pmr::unordered_map<char const *, char const *>& chTex, uint32_t texIdx, uint32_t matIdx) 
    { 
        if (property.IsValid())
        {
            uint32_t texCount = property.GetSrcObjectCount<FbxTexture>();

            for (uint32_t j = 0; j < texCount; ++j)
            {
                //Here we have to check if it's layeredtextures, or just textures:
                FbxLayeredTexture* lLayeredTexture = property.GetSrcObject<FbxLayeredTexture>(j);
                if (lLayeredTexture)
                {
                    uint32_t nTextures = lLayeredTexture->GetSrcObjectCount<FbxTexture>();
                    for (uint32_t k = 0; k < nTextures; ++k)
                    {
                        FbxTexture* lTexture = lLayeredTexture->GetSrcObject<FbxTexture>(k);
                        if (lTexture)
                        {
                            
                            FbxFileTexture* lFileTexture = FbxCast<FbxFileTexture>(lTexture);
                            chTex[FbxLayerElement::sTextureChannelNames[texIdx]] = lFileTexture->GetFileName();
                        }
                    }
                }
                else
                {
                    //no layered texture simply get on the property
                    FbxTexture* lTexture = property.GetSrcObject<FbxTexture>(j);
                    if (lTexture)
                    {
                        FbxFileTexture* lFileTexture = FbxCast<FbxFileTexture>(lTexture);
                        chTex[FbxLayerElement::sTextureChannelNames[texIdx]] = lFileTexture->GetFileName();
                    }
                }
            }
        }
    }


} // namespace dmt
