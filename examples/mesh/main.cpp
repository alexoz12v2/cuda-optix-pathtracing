#define DMT_ENTRY_POINT
#define DMT_WINDOWS_CLI
#include "platform/platform.h"
#include "core/core-bvh-builder.h"
#include "core/core-cudautils-cpubuild.h"
#include "core/core-mesh-parser.h"

namespace dmt {

    static void fbxImportTester()
    {

        dmt::Context ctx;
        assert(ctx.isValid() && "Invalid Context");
        //memory
        std::unique_ptr<unsigned char[]> bufferPtr    = std::make_unique<unsigned char[]>(2048);
        auto                             bufferMemory = std::pmr::monotonic_buffer_resource(bufferPtr.get(), 2048);
        ctx.log("FBXImport test", {});

        dmt::MeshFbxPasser fbxPasser{};

        std::pmr::string fbxBunny     = "stanford-bunny.fbx";
        std::pmr::string fbxBoxFlat   = "BoxFlat.fbx";
        std::pmr::string fbxBoxSmooth = "BoxSmooth.fbx";
        std::pmr::string fbxBoxDown   = "DownSysBox.fbx";
        std::pmr::string fbxBoxRight  = "RightSysBox.fbx";
        std::pmr::string fbxBoxFlatTetured  = "BoxFlatTextured.fbx";

        TriangleMesh mesh;
        bool         r = fbxPasser.ImportFBX(fbxBunny.c_str(), &mesh, std::pmr::get_default_resource());


        if (r)
        {
            ctx.log("Mesh name: {}", std::make_tuple(fbxPasser.GetMeshName()));
        }

        ctx.log("End test", {});
    }
} // namespace dmt

int32_t guardedMain()
{
    dmt::Ctx::init();
    class Janitor
    {
    public:
        ~Janitor() { dmt::Ctx::destroy(); }
    } j;

    {
        dmt::fbxImportTester();
    }
    return 0;
}
