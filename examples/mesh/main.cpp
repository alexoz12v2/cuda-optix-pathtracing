#define DMT_ENTRY_POINT
#define DMT_WINDOWS_CLI
#include "platform/platform.h"
#include "core/core-bvh-builder.h"
#include "core/core-cudautils-cpubuild.h"
#include "core/core-mesh-parser.h"

namespace dmt {
}

int32_t guardedMain()
{
    dmt::Ctx::init();
    class Janitor
    {
    public:
        ~Janitor() { dmt::Ctx::destroy(); }
    } j;

    {
    }
}
