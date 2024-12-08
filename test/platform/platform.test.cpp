module;

#include "dmtutils.h"

#include <catch2/catch_test_macros.hpp>

module platform;

TEST_CASE("[platform] Test Case for the platform module")
{
    SECTION("Platform::getSize() should return 4096")
    {
        CHECK(dmt::Platform{}.getSize() == 4096);
    }
}