#include "dmtutils.h"

#include <catch2/catch_test_macros.hpp>

import testdmt;

TEST_CASE("[projtest] Test Case for testing")
{
    SECTION("Main stuff")
    {
        STATIC_CHECK(3 == 3);
        CHECK(dmt::TestMath::add(2, 3) == 5);
        CHECK(dmt::testWork() == 4);
    }
}