#include <cstdint>

import platform;
import middleware;

int32_t main()
{
    dmt::Platform platform;
    auto&         actx = platform.ctx();
    actx.log("Hello darkness my old friend, {}", {sizeof(dmt::Options)});
}