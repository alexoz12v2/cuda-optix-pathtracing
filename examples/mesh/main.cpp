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

        ctx.log("FBXImport test", {});

        dmt::MeshFbxPasser fbxPasser{};

        std::pmr::string fbxName = "stanford-bunny.fbx";

        bool r = fbxPasser.ImportFBX(fbxName.c_str());


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
