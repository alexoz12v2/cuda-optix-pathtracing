#include "core-mesh-parser.h"

// our stuff
#include "platform-context.h"

// external
#include "fbxsdk.h"

// std
#include <string>
#include <memory>
#include <vector>
#include <memory_resource>
#include <unordered_set>
#include <unordered_map>

namespace dmt::detail {
    template <typename T>
        requires std::is_base_of_v<FbxObject, T>
    struct FbxObjectDestroyer
    {
        void operator()(T* settings) const
        {
            if (settings)
                settings->Destroy(true);
        }
    };
    template <typename T>
    using Handle = std::unique_ptr<T, detail::FbxObjectDestroyer<T>>;

    namespace {
        template <typename T, typename... Args>
        Handle<T> makeFbxHandle(Args&&... args)
        {
            return std::unique_ptr<T, FbxObjectDestroyer<T>>(std::forward<Args>(args)..., FbxObjectDestroyer<T>());
        }

        void printFileFormatAndVersion(Context& ctx, FbxIOPluginRegistry const* registry, FbxImporter* importer)
        {
            char const* description = registry->GetReaderFormatDescription(importer->GetFileFormat());
            char const* extension   = registry->GetReaderFormatExtension(importer->GetFileFormat());
            std::string msg         = "Mesh Importer Successfully Initialized. Detected File Format:\n\t";
            msg += description;
            msg += "(extension: ";
            msg += extension;
            msg += ")";
            if (_strcmpi("fbx", extension) == 0)
            {
                int fileMajor = 0, fileMinor = 0, fileRevision = 0;
                importer->GetFileVersion(fileMajor, fileMinor, fileRevision);
                msg += " FBX Version: " + std::to_string(fileMajor) + '.' + std::to_string(fileMinor) + '.' +
                       std::to_string(fileRevision);
            }
            ctx.trace("{}", std::make_tuple(msg));
        }

        std::string stringFromSystemUnit(FbxSystemUnit const& unit)
        {
            if (unit == FbxSystemUnit::cm)
                return "cm";
            if (unit == FbxSystemUnit::m)
                return "m";
            if (unit == FbxSystemUnit::Foot)
                return "foot";
            if (unit == FbxSystemUnit::dm)
                return "dm";
            if (unit == FbxSystemUnit::mm)
                return "mm";
            if (unit == FbxSystemUnit::Inch)
                return "inch";
            if (unit == FbxSystemUnit::km)
                return "km";
            if (unit == FbxSystemUnit::Yard)
                return "Yard";
            return "(Custom System Unit)";
        }

        std::string stringFromUpAxis(int upAxis)
        {
            switch (upAxis)
            {
                case 0: return "X";
                case 1: return "Y";
                case 2: return "Z";
                default: return "(Unknown Axis, uses default \"Yl\")";
            }
        }

        std::string stringFromAxisSystem(FbxAxisSystem const& axisSystem)
        {
            std::string result = "(AxisSystem) { \n\t(Axes) Up: ";

            int   upSign              = 0;
            int   frontSign           = 0;
            char  availableAxes[3][2] = {{1, 2}, {0, 2}, {0, 1}};
            char* axes                = nullptr;

            FbxAxisSystem::EUpVector const    up         = axisSystem.GetUpVector(upSign);
            FbxAxisSystem::EFrontVector const front      = axisSystem.GetFrontVector(frontSign);
            FbxAxisSystem::ECoordSystem const handedness = axisSystem.GetCoorSystem();

            result += (upSign < 0 ? '-' : '+');
            switch (up)
            { // clang-format off
                case FbxAxisSystem::eXAxis: axes = availableAxes[0]; result += 'X'; break;
                case FbxAxisSystem::eYAxis: axes = availableAxes[1]; result += 'Y'; break;
                case FbxAxisSystem::eZAxis: axes = availableAxes[2]; result += 'Z'; break;
            } // clang-format on
            assert(axes);
            bool const rightHanded = handedness == FbxAxisSystem::ECoordSystem::eRightHanded;
            bool const parityEven  = front == FbxAxisSystem::eParityEven;

            result += ", Front: ";
            result += (frontSign < 0 ? '-' : '+');
            char const frontAxis = axes[parityEven ? 0 : 1];
            char const sideAxis  = axes[parityEven ? 1 : 0];

            result += (frontAxis == 0 ? 'X' : (frontAxis == 1 ? 'Y' : 'Z'));
            result += ", Side: ";

            int const sideSign = upSign * frontSign * (rightHanded ? +1 : -1) * (parityEven ? +1 : -1);
            result += (sideSign < 0 ? '-' : '+');
            result += "XYZ"[sideAxis];

            result += "\n\t(Params) ...TODO }";

            return result;
        }

        void printSceneObjects(Context& ctx, FbxScene* scene)
        {
            std::string fbxStatistics = "Printing FBX Statistics for scene '";
            fbxStatistics += scene->GetName();

            fbxStatistics += "\n\tFBX Objects:          ";
            fbxStatistics += scene->GetSrcObjectCount(); // number of children in hierarchy
            fbxStatistics += "\n\t  of which Nodes:     " + std::to_string(scene->GetSrcObjectCount<FbxNode>());
            fbxStatistics += "\n\t  of which Meshes:    " + std::to_string(scene->GetSrcObjectCount<FbxMesh>());
            fbxStatistics += "\n\t  of which Materials: " + std::to_string(scene->GetSrcObjectCount<FbxSurfaceMaterial>());
            fbxStatistics += "\n\t  we don't care about Evaluators, FbxCamera, FbxLight, ...";

            FbxGlobalSettings& settings = scene->GetGlobalSettings();
            fbxStatistics += "\nRelevant Global Settings: (actual/original)";
            fbxStatistics += "\n\tSystem Unit:      " + stringFromSystemUnit(settings.GetSystemUnit()) + '/' +
                             stringFromSystemUnit(settings.GetOriginalSystemUnit());
            fbxStatistics += "\n\tOriginal Up Axis: " + stringFromUpAxis(settings.GetOriginalUpAxis());
            fbxStatistics += "\n\tCurrent Axis System: ";
            fbxStatistics += stringFromAxisSystem(settings.GetAxisSystem());

            ctx.trace("{}", std::make_tuple(fbxStatistics));
        }

        void throwUnsupportedMappingMode(FbxLayerElement::EMappingMode                            mappingMode,
                                         std::unordered_set<FbxLayerElement::EMappingMode> const& supported,
                                         char const*                                              layerName)
        {
            constexpr auto stringFromMappingMode = [](FbxLayerElement::EMappingMode mm) -> std::string {
                switch (mm)
                {
                    case FbxLayerElement::eNone: return "eNone";
                    case FbxLayerElement::eByControlPoint: return "eByControlPoint";
                    case FbxLayerElement::eByPolygonVertex: return "eByPolygonVertex";
                    case FbxLayerElement::eByPolygon: return "eByPolygon";
                    case FbxLayerElement::eByEdge: return "eByEdge";
                    case FbxLayerElement::eAllSame: return "eAllSame";
                    default: return "";
                }
            };
            std::string message = "Unsupported mapping mode for";
            message += layerName;
            message += " Layer Element: " + stringFromMappingMode(mappingMode);
            if (supported.size())
            {
                message += " (must be one of ";
                for (auto const mp : supported)
                    message += stringFromMappingMode(mp) + ", ";
                message.resize(message.size() - 2);
                message += ").";
            }
            throw std::runtime_error(message);
        }

        std::unordered_map<int, std::vector<int>> createControlPointToPolygonIndexMapping(FbxMesh const* mesh)
        {
            std::unordered_map<int, std::vector<int>> mapping;

            int const* polygonVertices = mesh->GetPolygonVertices();
            int const  polygonCount    = mesh->GetPolygonCount();

            for (int polygonIdx = 0; polygonIdx < polygonCount; ++polygonIdx)
            {
                int const indexCount = mesh->GetPolygonSize(polygonIdx);
                int const start      = mesh->GetPolygonVertexIndex(polygonIdx);
                for (int i = 0; i < indexCount; ++i)
                {
                    int controlPointIndex  = polygonVertices[start + i];
                    int polygonVertexIndex = start + i;
                    mapping[controlPointIndex].push_back(polygonVertexIndex);
                }
            }
            return mapping;
        }

        std::unordered_map<int, int> createPolygonIndexToControlPointMapping(FbxMesh const* mesh)
        {
            std::unordered_map<int, int> mapping;

            int const* polygonVertices    = mesh->GetPolygonVertices();
            int const  polygonVertexCount = mesh->GetPolygonVertexCount();

            for (int cornerIdx = 0; cornerIdx < polygonVertexCount; ++cornerIdx)
            {
                int const controlPointIndex = polygonVertices[cornerIdx];
                mapping[cornerIdx]          = controlPointIndex;
            }
            return mapping;
        }

        template <typename Vec, typename Layer>
            requires(std::is_base_of_v<FbxLayerElementTemplate<Vec>, Layer> &&
                     (std::is_same_v<Vec, FbxVector2> || std::is_same_v<Vec, FbxVector4>))
        Vec getLayerElementValue(FbxLayerElementTemplate<Vec> const*   layerElement,
                                 FbxLayerElement::EReferenceMode const referenceMode,
                                 int const                             index)
        {
            FbxLayerElementArrayTemplate<Vec> const& directArray = layerElement->GetDirectArray();
            if (referenceMode == FbxLayerElement::EReferenceMode::eDirect)
                return directArray[index];

            FbxLayerElementArrayTemplate<int> const& indexArray = layerElement->GetIndexArray();
            return directArray[indexArray[index]];
        };

        template <typename Layer, typename Vec, typename Func>
            requires(std::is_base_of_v<FbxLayerElementTemplate<Vec>, Layer> &&
                     (std::is_same_v<FbxVector4, Vec> || std::is_same_v<FbxVector2, Vec>) &&
                     std::is_invocable_r_v<Vec, Func, Vec>)
        std::vector<Vec> processLayerElement(
            FbxMesh const*                                           mesh,
            Layer const*                                             layerElement,
            Func&&                                                   func,
            std::unordered_map<int, std::vector<int>> const&         controlPointToPolygonIndexMapping,
            std::unordered_set<FbxLayerElement::EMappingMode> const& supportedMappingModes,
            char const*                                              layerName)
        {
            using EMappingMode   = FbxLayerElement::EMappingMode;
            using EReferenceMode = FbxLayerElement::EReferenceMode;

            int const        polygonVertexCount = mesh->GetPolygonVertexCount();
            std::vector<Vec> result(polygonVertexCount);

            EMappingMode const   mappingMode   = layerElement->GetMappingMode();
            EReferenceMode const referenceMode = layerElement->GetReferenceMode();

            if (!supportedMappingModes.contains(mappingMode))
                throwUnsupportedMappingMode(mappingMode, supportedMappingModes, layerName);

            // ------------------------------------------------------------
            // eByPolygonVertex
            // ------------------------------------------------------------
            if (mappingMode == EMappingMode::eByPolygonVertex)
            {
                for (int pv = 0; pv < polygonVertexCount; ++pv)
                {
                    result[pv] = func(getLayerElementValue<Vec, Layer>(layerElement, referenceMode, pv));
                }
            }

            // ------------------------------------------------------------
            // eByControlPoint
            // ------------------------------------------------------------
            else if (mappingMode == EMappingMode::eByControlPoint)
            {
                int const controlPointCount = referenceMode == EReferenceMode::eDirect
                                                  ? layerElement->GetDirectArray().GetCount()
                                                  : layerElement->GetIndexArray().GetCount();

                for (int cp = 0; cp < controlPointCount; ++cp)
                {
                    auto it = controlPointToPolygonIndexMapping.find(cp);
                    if (it == controlPointToPolygonIndexMapping.end())
                        continue; // control point unused by mesh

                    Vec value = func(getLayerElementValue<Vec, Layer>(layerElement, referenceMode, cp));

                    for (int corner : it->second)
                        result[corner] = value;
                }
            }

            // ------------------------------------------------------------
            // eByPolygon
            // ------------------------------------------------------------
            else if (mappingMode == EMappingMode::eByPolygon)
            {
                int const polygonCount = mesh->GetPolygonCount();

                for (int p = 0; p < polygonCount; ++p)
                {
                    Vec value = func(getLayerElementValue<Vec, Layer>(layerElement, referenceMode, p));

                    int const start = mesh->GetPolygonVertexIndex(p);
                    int const size  = mesh->GetPolygonSize(p);

                    for (int i = 0; i < size; ++i)
                        result[start + i] = value;
                }
            }

            return result;
        }

        std::vector<FbxVector4> processNormalLayerElement(
            FbxMesh const*                                   mesh,
            FbxLayerElementNormal const*                     normalsLayerElement,
            FbxAMatrix const&                                normalMatrix,
            std::unordered_map<int, std::vector<int>> const& controlPointToPolygonIndexMapping)
        {
            using EMappingMode = FbxLayerElement::EMappingMode;
            static std::unordered_set const supportedMappingModes{EMappingMode::eByControlPoint,
                                                                  EMappingMode::eByPolygon,
                                                                  EMappingMode::eByPolygonVertex};
            return processLayerElement<FbxLayerElementNormal, FbxVector4>( //
                mesh,
                normalsLayerElement,
                [&normalMatrix](FbxVector4 const& val) { return normalMatrix.MultT(val); },
                controlPointToPolygonIndexMapping,
                supportedMappingModes,
                "LayerElementNormal");
        }

        std::vector<FbxVector2> processUVLayerElement(
            FbxMesh const*                                   mesh,
            FbxLayerElementUV const*                         uvLayerElement,
            std::unordered_map<int, std::vector<int>> const& controlPointToPolygonIndexMapping)
        {
            using EMappingMode = FbxLayerElement::EMappingMode;
            static std::unordered_set const supportedMappingModes{EMappingMode::eByPolygonVertex};
            return processLayerElement<FbxLayerElementUV, FbxVector2>( //
                mesh,
                uvLayerElement,
                [](FbxVector2 const& val) { return val; },
                controlPointToPolygonIndexMapping,
                supportedMappingModes,
                "LayerElementUV");
        }

        // first node depth first (recursion flattened with stack)
        FbxNode* getFirstMeshNodeFromFbxScene(FbxScene const* scene)
        {
            assert(scene);
            FbxNode* root = scene->GetRootNode();
            if (!root)
                return nullptr;
            std::vector<FbxNode*> nodeStack{root};
            nodeStack.reserve(64);
            while (!nodeStack.empty())
            {
                FbxNode* node = nodeStack.back();
                nodeStack.pop_back();

                // reverse order insertion of children
                for (int idx = node->GetChildCount() - 1; idx >= 0; --idx)
                    nodeStack.push_back(node->GetChild(idx));
                // instead of having to iterate through attributes of a node, FBX SDK defines
                // "shortcut" methods for standard attribute types like mesh
                if (FbxMesh const* mesh = node->GetMesh(); mesh)
                {
                    // silently discard empty meshes
                    if (mesh->GetPolygonCount() > 0)
                        return node;
                }
            }
            return nullptr;
        }

        void processMeshNode(Context& ctx, FbxNode* meshNode, TriangleMesh& outMesh)
        {
            using namespace std::string_literals;
            FbxMesh const* mesh = meshNode->GetMesh();

            // we need to apply the world space transform
            // https://help.autodesk.com/view/FBX/2020/ENU/?guid=FBX_Developer_Help_nodes_and_scene_graph_fbx_nodes_transformation_data_html
            // global transform: the node transform
            // geometric transform: how the attribute differs from the node
            FbxAMatrix const& globalTransform    = meshNode->EvaluateGlobalTransform();
            FbxAMatrix const  geometricTransform = [meshNode]() {
                FbxAMatrix t;
                t.SetT(meshNode->GetGeometricTranslation(FbxNode::eSourcePivot));
                t.SetR(meshNode->GetGeometricRotation(FbxNode::eSourcePivot));
                t.SetS(meshNode->GetGeometricScaling(FbxNode::eSourcePivot));
                return t;
            }();
            FbxAMatrix const finalTransform = geometricTransform * globalTransform;

            // for normals and tangents: matrix, remove translation, compute inverse-transpose
            FbxAMatrix const normalMatrix = [&finalTransform]() {
                FbxAMatrix result = finalTransform;
                result.SetT(FbxVector4(0, 0, 0, 1));
                result = result.Inverse().Transpose();
                return result;
            }();

            std::unordered_map<int, std::vector<int>> const controlPointToPolygonIndexMapping = createControlPointToPolygonIndexMapping(
                mesh);
            std::unordered_map<int, int> const polygonIndexToControlPointMapping = createPolygonIndexToControlPointMapping(
                mesh);

            // enumerate number of layers, if zero, throw, if bigger than one, warning
            int const layerCount = mesh->GetLayerCount();
            if (layerCount == 0)
                throw std::runtime_error("Mesh '"s + mesh->GetName() + "' has no layers");
            if (layerCount > 1)
                ctx.warn("Mesh '{}' has more than one layer. Considering only layer 0", std::make_tuple(mesh->GetName()));

            // add control points to the mesh
            for (int i = 0; i < mesh->GetControlPointsCount(); ++i)
            {
                // in the ASCII FBX file, there's no w coord. just to be safe, debugbreak
                FbxVector4 objPos = mesh->GetControlPointAt(i);
#if defined(DMT_OS_WINDOWS) && defined(DMT_DEBUG)
                if (objPos.mData[3] != 1. && objPos.mData[3] != 0.)
                    __debugbreak();
#endif
                // Note: we support static mesh and bake to global
                FbxVector4 const pos = finalTransform.MultT(objPos);
                outMesh.addPosition(
                    {static_cast<float>(pos.mData[0]), static_cast<float>(pos.mData[1]), static_cast<float>(pos.mData[2])});
            }

            std::unordered_set<size_t> const positionIndicesDistinct = [mesh] {
                std::unordered_set<size_t> theSet;
                theSet.reserve(mesh->GetPolygonVertexCount() * 3);
                int const* polygonVertices = mesh->GetPolygonVertices();
                int const  vertexCount     = mesh->GetPolygonVertexCount();
                for (int i = 0; i < vertexCount; ++i)
                {
                    theSet.insert(polygonVertices[i]);
                }
                return theSet;
            }();

            for (int p = 0; p < mesh->GetPolygonCount(); ++p)
            {
                if (mesh->GetPolygonSize(p) != 3)
                    throw std::runtime_error("Mesh '"s + mesh->GetName() + "', polygon[" + std::to_string(p) + "' has " +
                                             std::to_string(mesh->GetPolygonSize(p)) + " indices, expected 3");
            }

            // process layer elements of layer zero (only normals[0] and uvs[0], emit warning
            // on ignored things and crash on their absence)
            FbxLayer const* layer = mesh->GetLayer(0);
            ctx.warn("FBX Import: Examining Layer 0 for normal and uv only, ignoring the rest", {});
            FbxLayerElementNormal const* normalsLayerElement = layer->GetNormals();
            if (!normalsLayerElement)
                throw std::runtime_error("Mesh '"s + mesh->GetName() + "' has no Normal Layer");
            FbxLayerElementUV const* uvLayerElement = layer->GetUVs();
            if (!uvLayerElement)
                throw std::runtime_error("Mesh '"s + mesh->GetName() + "' has no UV layer element");
            // per corner vectors
            std::vector<FbxVector4> const normalsPerCorner = processNormalLayerElement(mesh,
                                                                                       normalsLayerElement,
                                                                                       normalMatrix,
                                                                                       controlPointToPolygonIndexMapping);
            std::vector<FbxVector2> const uvsPerCorner = processUVLayerElement(mesh, uvLayerElement, controlPointToPolygonIndexMapping);
            assert(normalsPerCorner.size() == uvsPerCorner.size());
            // push per corner vectors in output mesh
            for (size_t index = 0; index < normalsPerCorner.size(); ++index)
            {
                FbxVector4 const& normal = normalsPerCorner[index];
                FbxVector2 const& uv     = uvsPerCorner[index];
                outMesh.addNormal({static_cast<float>(normal.mData[0]),
                                   static_cast<float>(normal.mData[1]),
                                   static_cast<float>(normal.mData[2])});
                outMesh.addUV({static_cast<float>(uv.mData[0]), static_cast<float>(uv.mData[1])});
            }

            // now finally add the indexed triangles
            auto const matToVertexIndex = [&](size_t const index) -> VertexIndex {
                int const    index32     = index;
                size_t const positionIdx = static_cast<size_t>(polygonIndexToControlPointMapping.at(index32));
                return {.positionIdx = positionIdx, .normalIdx = index, .uvIdx = index};
            };

            for (int p = 0; p < mesh->GetPolygonCount(); ++p)
            {
                assert(mesh->GetPolygonSize(p) == 3);
                int const index = mesh->GetPolygonVertexIndex(p);
                outMesh.addIndexedTriangle(matToVertexIndex(index + 0),
                                           matToVertexIndex(index + 1),
                                           matToVertexIndex(index + 2),
                                           -1);
            }
            ctx.log("Mesh '{}' has been processed", std::make_tuple(mesh->GetName()));
        }
    } // namespace
} // namespace dmt::detail

namespace dmt {
    using namespace dmt::detail;

    class MeshFbxParserImpl
    {
    public:
        MeshFbxParserImpl();

        template <typename OnParse, typename OnError>
            requires(std::is_invocable_r_v<void, OnParse, FbxImporter*, FbxScene*> &&
                     std::is_invocable_r_v<void, OnError, std::string>)
        void openSceneImporter(char const* fileName, OnParse&& onParse, OnError&& onError);

    private:
        // Utilities
        static Handle<FbxIOSettings> createMeshMinimalIOSettings();
        static Handle<FbxImporter>   createFbxImporter(char const* name);
        static std::string      importFile(FbxImporter* importer, char const* fullName, FbxIOSettings* importSettings);
        static Handle<FbxScene> importScene(FbxImporter* importer, char const* name, std::string& error);
        static FbxIOPluginRegistry* getIOPluginRegistry() { return s_fbxManager->GetIOPluginRegistry(); }

        // since it's a singleton, we don't care when it's actually destroyed. It'll
        // live till the end of the program
        static inline FbxManager* s_fbxManager = FbxManager::Create();

        // private state
        Handle<FbxIOSettings> m_defaultSettings;
    };

    template <typename OnParse, typename OnError>
        requires(std::is_invocable_r_v<void, OnParse, FbxImporter*, FbxScene*> &&
                 std::is_invocable_r_v<void, OnError, std::string>)
    void MeshFbxParserImpl::openSceneImporter(char const* fileName, OnParse&& onParse, OnError&& onError)
    {
        Handle<FbxImporter> importer = createFbxImporter(fileName);
        if (!importer)
        {
            onError("Couldn't allocate FbxImporter");
            return;
        }
        if (std::string const error = importFile(importer.get(), fileName, m_defaultSettings.get()); !error.empty())
        {
            onError(error);
            return;
        }
        std::string      error;
        Handle<FbxScene> scene = importScene(importer.get(), fileName, error);
        if (!error.empty())
        {
            onError(error);
            return;
        }

        onParse(importer.get(), scene.get());
    }

    MeshFbxParserImpl::MeshFbxParserImpl() : m_defaultSettings(createMeshMinimalIOSettings()) {}

    Handle<FbxIOSettings> MeshFbxParserImpl::createMeshMinimalIOSettings()
    {
        FbxIOSettings* importSettings = FbxIOSettings::Create(s_fbxManager, IOSROOT);
        if (!importSettings)
            return nullptr;

        // https://help.autodesk.com/cloudhelp/2018/ENU/FBX-Developer-Help/importing_and_exporting_a_scene/io_settings.html
        // Note: true is the default value, but we'll be explicit about what we use
        importSettings->SetBoolProp(IMP_FBX_MATERIAL, true);
        importSettings->SetBoolProp(IMP_FBX_GLOBAL_SETTINGS, true);
        importSettings->SetBoolProp(IMP_FBX_NORMAL, true);

        importSettings->SetBoolProp(IMP_FBX_TEXTURE, false);
        importSettings->SetBoolProp(IMP_FBX_LINK, false);
        importSettings->SetBoolProp(IMP_FBX_SHAPE, false);
        importSettings->SetBoolProp(IMP_FBX_GOBO, false);
        importSettings->SetBoolProp(IMP_FBX_ANIMATION, false);

        return makeFbxHandle<FbxIOSettings>(importSettings);
    }

    Handle<FbxImporter> MeshFbxParserImpl::createFbxImporter(char const* name)
    {
        FbxImporter* importer = FbxImporter::Create(s_fbxManager, name);
        return makeFbxHandle<FbxImporter>(importer);
    }

    std::string MeshFbxParserImpl::importFile(FbxImporter* importer, char const* fullName, FbxIOSettings* importSettings)
    {
        assert(importer && fullName);
        if (importer->Initialize(fullName, -1, importSettings))
        {
            if (Context ctx; ctx.isValid() && ctx.isTraceEnabled())
                printFileFormatAndVersion(ctx, getIOPluginRegistry(), importer);
            return "";
        }
        return importer->GetStatus().GetErrorString();
    }

    Handle<FbxScene> MeshFbxParserImpl::importScene(FbxImporter* importer, char const* name, std::string& error)
    {
        FbxScene* scene = FbxScene::Create(s_fbxManager, name);
        if (!scene)
        {
            error += "Failed to allocate FbxScene Object '";
            error += name;
            error += "'.";
            return nullptr;
        }
        if (!importer->Import(scene))
        {
            error += "FBX Scene Import Failed: ";
            error += importer->GetStatus().GetErrorString();
            return nullptr;
        }
        if (Context ctx; ctx.isValid() && ctx.isTraceEnabled())
            printSceneObjects(ctx, scene);
        return makeFbxHandle<FbxScene>(scene);
    }

    MeshFbxParser::MeshFbxParser() : m_pimpl(std::make_unique<MeshFbxParserImpl>()) {}
    MeshFbxParser::~MeshFbxParser() = default;

    bool MeshFbxParser::ImportFBX(char const* fileName, TriangleMesh* outMesh) const
    {
        assert(outMesh && fileName);
        Context ctx;
        assert(ctx.isValid() && "platform Context was not initialized");

        os::Path const fbxDirectory = os::Path::fromString(fileName, true);
        if (!fbxDirectory.isValid() || !fbxDirectory.isFile())
            return false;

        m_pimpl->openSceneImporter(fileName, [&](FbxImporter* importer, FbxScene* scene) {
            ctx.log("Importing first mesh from scene {}", std::make_tuple(scene->GetName()));
            // Create Our System
            FbxAxisSystem const axisSystem(FbxAxisSystem::EUpVector::eZAxis,
                                           FbxAxisSystem::EFrontVector::eParityOdd,
                                           FbxAxisSystem::ECoordSystem::eLeftHanded);
            // convert everything in centimetres (TODO check if correct. if not, switch to metres)
            if (scene->GetGlobalSettings().GetSystemUnit() != FbxSystemUnit::cm)
            {
                // according to docs, mConvertRrsNodes can cause problems on scaling, so keep it false
                constexpr FbxSystemUnit::ConversionOptions opts{.mConvertRrsNodes               = false,
                                                                .mConvertLimits                 = true,
                                                                .mConvertClusters               = true,
                                                                .mConvertLightIntensity         = true,
                                                                .mConvertPhotometricLProperties = true,
                                                                .mConvertCameraClipPlanes       = true};
                FbxSystemUnit::cm.ConvertScene(scene, opts);
            }
            // convert to our coordinate system (no-op if already correct). "Shallow convert" as only the root node
            // sees its transform changed
            // We'd need to extract scene transform from root and apply it to nodes. Fundamental,
            // as our convertion operations are applied only to the root node.
            // The method `EvaluateGlobalTransform` for `FbxNode` does that for us
            axisSystem.ConvertScene(scene);
            if (ctx.isTraceEnabled())
                printSceneObjects(ctx, scene);
            // if there are more mesh objects, emit a warning
            int const meshCount = scene->GetSrcObjectCount<FbxMesh>();
            if (meshCount <= 0)
                throw std::runtime_error("No meshes to import");
            if (meshCount > 1)
                ctx.warn(
                    "[FBX Import] warning: Only first mesh of this FBX file will be read."
                    "Found {} meshes",
                    std::make_tuple(meshCount));
            FbxNode* firstMeshNode = getFirstMeshNodeFromFbxScene(scene);
            // if it has an animation, issue warning that we are discarding that information (what happens to post-rotation?)
            if (!firstMeshNode)
                throw std::runtime_error("No meshes could be found inside the scene");

            processMeshNode(ctx, firstMeshNode, *outMesh);
        }, [](std::string const& errorMsg) { throw std::runtime_error(errorMsg); });

        if (ctx.isValid() && ctx.isTraceEnabled())
        {
            std::string meshStats = "  Triangle Count: " + std::to_string(outMesh->triCount()) +
                                    "\n  UVs: " + std::to_string(outMesh->uvCount()) +
                                    "\n  Normals: " + std::to_string(outMesh->normalCount()) +
                                    "\n  Positions: " + std::to_string(outMesh->positionCount());
            ctx.trace("\n{}", std::make_tuple(meshStats));
        }

        return true;
    }

} // namespace dmt
