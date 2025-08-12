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

    using dFbxManger     = std::unique_ptr<void, FbxDeleter>;
    using dFbxIOSettings = std::unique_ptr<void, FbxSettingsDeleter>;

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
        MeshFbx(dFbxManager& mng, char* fileName);
        ~MeshFbx();

    private:
        std::pmr::string m_fileName;
        dFbxIOSettings   m_settings;
    };
} // namespace dmt
