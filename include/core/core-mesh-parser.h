#pragma once

#include "core/core-macros.h"
#include "core/core-cudautils-cpubuild.h"
#include "core/core-trianglemesh.h"


namespace dmt {
    class DMT_CORE_API FbxDeleter
    {
    public:
        void operator()(void* raw) const;
    };

    class DMT_CORE_API FbxSettingsDeleter
    {
    public:
        void operator()(void* raw) const;
    };

    class DMT_CORE_API FbxSceneDeleter
    {
    public:
        void operator()(void* raw) const;
    };

    using dFbxManager    = std::unique_ptr<void, FbxDeleter>;
    using dFbxIOSettings = std::unique_ptr<void, FbxSettingsDeleter>;
    using dFbxScene      = std::unique_ptr<void, FbxSceneDeleter>;


    dFbxManager DMT_CORE_API createFBXInstance();

#if 0
    struct Settings
    {
        int32_t importMaterial        : 1;
        int32_t importTexture         : 1;
        int32_t importLink            : 1;
        int32_t importShape           : 1;
        int32_t importGobo            : 1;
        int32_t importAnimation       : 1;
        int32_t importGlobal_settings : 1;
        int32_t importNormal          : 1;
    };
#endif

    class DMT_CORE_API MeshFbxPasser
    {

    public:
        MeshFbxPasser(dFbxManager& mng, char* fileName);
        ~MeshFbxPasser();

        bool ImportFBX(dFbxManager& inst, char* fileName);

    private:
        std::pmr::string m_fileName;
        dFbxIOSettings   m_settings;
        dFbxScene        m_scene;
    };
} // namespace dmt
