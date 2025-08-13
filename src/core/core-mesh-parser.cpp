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
    // To create the material:
    //
    // FbxString lMaterialName = "toto";
    // lMaterial = FbxSurfacePhong::Create(pScene, lMaterialName.Buffer());
    //
    // To get the material ex:
    //
    // lMaterial = lNode->GetSrcObject<FbxSurfacePhong>(0);
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
    // To see:
    //
    // FbxLayerElementBinormal
    //
    //
    //


    void FbxDeleter::operator()(void* raw) const
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

    static FbxManager*    getManager(dFbxManager& inst) { return reinterpret_cast<FbxManager*>(inst.get()); }
    static FbxIOSettings* getIoSettings(dFbxIOSettings& inst) { return reinterpret_cast<FbxIOSettings*>(inst.get()); }
    static FbxScene*      getScene(dFbxScene& inst) { return reinterpret_cast<FbxScene*>(inst.get()); }

    dFbxManager createFBXInstance() { return dFbxManager{FbxManager::Create(), FbxDeleter{}}; }

    MeshFbxPasser::MeshFbxPasser(dFbxManager& inst, char* fileName)
    {
        os::Path const fbxDirectory = os::Path::executableDir() / fileName;
        if (!fbxDirectory.isValid() || !fbxDirectory.isFile())
            return;

        m_fileName = fbxDirectory.toUnderlying();
        m_settings = dFbxIOSettings{FbxIOSettings::Create(getManager(inst), IOSROOT), FbxSettingsDeleter{}};

        //set flags for import settings
        getIoSettings(m_settings)->SetBoolProp(IMP_FBX_MATERIAL, false);
        getIoSettings(m_settings)->SetBoolProp(IMP_FBX_TEXTURE, true);
        getIoSettings(m_settings)->SetBoolProp(IMP_FBX_LINK, false);
        getIoSettings(m_settings)->SetBoolProp(IMP_FBX_SHAPE, false);
        getIoSettings(m_settings)->SetBoolProp(IMP_FBX_GOBO, false);
        getIoSettings(m_settings)->SetBoolProp(IMP_FBX_ANIMATION, false);
        getIoSettings(m_settings)->SetBoolProp(IMP_FBX_GLOBAL_SETTINGS, true);
        getIoSettings(m_settings)->SetBoolProp(IMP_FBX_NORMAL, true);
    }

    bool MeshFbxPasser::ImportFBX(dFbxManager& inst, char* fileName)
    {
        std::string fileNameStr = std::string(fileName);

        if (fileNameStr.size() < 4)
            return false;

        std::string sceneName = fileNameStr.substr(0, fileNameStr.size() - 4) + "_scene";

        FbxImporter* fbxImporter = FbxImporter::Create(getManager(inst), sceneName.c_str());

        //intialize the importer
        bool importStatus = fbxImporter->Initialize(fileName, -1, getManager(inst)->GetIOSettings());

        if (!importStatus)
            return false;


        //Load the scene
        m_scene = dFbxScene{FbxScene::Create(getManager(inst), sceneName.c_str())};

        fbxImporter->Destroy();

        //FbxNode*              lNode;
        //FbxMesh mesh;
        //FbxObject             obj;
        //FbxLayer layer;
        //FbxLayerElementNormal layerElementNormal;
        //FbxFileTexture*       texture;
        //FbxGeometry*          geometry;
        //FbxGeometryElementNormal* lGeometryElementNormal;
        //FbxGeometryBase*     geometry;

        return true;
    }

    MeshFbxPasser::~MeshFbxPasser() {}
} // namespace dmt
