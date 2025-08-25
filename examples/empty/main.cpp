#define DMT_ENTRY_POINT

#include "platform/platform.h"

namespace /* static */ {
    void entryPoint()
    {
        dmt::Context ctx;

        float            pi   = 3.14f;
        int              i    = 0;
        void*            addr = &i;
        std::string_view str  = "sdfdsf";

        ctx.log("{} Up and {} {} {} {{sss}}", std::make_tuple(pi, i, addr, str));
    }
} // namespace

int guardedMain()
{
    dmt::Ctx::init();
    struct Janitor
    {
        ~Janitor() noexcept { dmt::Ctx::destroy(); }
    } j;

    entryPoint();

    return 0;
}
