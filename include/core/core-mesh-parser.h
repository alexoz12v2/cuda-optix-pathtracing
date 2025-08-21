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

    struct FbxResources
    {
        void* manager  = nullptr;
        void* settings = nullptr;
    };

    class DMT_CORE_API FbxDeleterResources
    {
    public:
        void operator()(void* raw) const;
    };

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

    class DMT_CORE_API MeshFbxParser
    {

    public:
        MeshFbxParser();
        ~MeshFbxParser();

        bool        ImportFBX(char const* fileName, TriangleMesh* outMesh);
        char const* GetMeshName();

    private:
        void InitFbxManager();

        std::pmr::string                                  m_fileName;
        std::pmr::string                                  m_meshName;
        std::pmr::unordered_map<char const*, char const*> m_ChannelsTexPath;
        FbxResources                                      m_res;
    };
} // namespace dmt
