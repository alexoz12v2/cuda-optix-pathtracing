#include "core-mesh-parser.h"

#include "fbxsdk.h"

namespace dmt {

    // Instantiate FBX SDK memory management object-> FbxManager
    //import a contents of an FBX file -> FbxIOSettings, FbxImporter, FbxScene
    //traverse hierarchy FbxScene, FbxNode, FbxNodeAttribute
    //elements information -> FbxNode, FbxNodeAttribute, FbxString
    //os::Path imageDirectory = os::Path::executableDir();
    //        imageDirectory /= "tex";
    //        os::Path const diffuse = imageDirectory / "white_sandstone_bricks_03_diff_4k.exr";
    //toUnderlying()
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

    static FbxManager*    getManager(dFbxManager& inst) { return reinterpret_cast<FbxManager*>(inst.get()); }
    static FbxIOSettings* getIoSettings(dFbxIOSettings& inst) { return reinterpret_cast<FbxIOSettings*>(inst.get()); }

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

    MeshFbxPasser::~MeshFbxPasser() {}
} // namespace dmt
