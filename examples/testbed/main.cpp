#include <fmt/core.h>

import platform;

int main()
{
    dmt::Platform platform;
    fmt::print("platform size : {}", platform.getSize());
}