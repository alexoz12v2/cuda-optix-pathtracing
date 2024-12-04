module;

#include "dmtutils.h"

#include <catch2/catch_test_macros.hpp>

#include <fff/fff.h>

module testdmtpartitions;
namespace dmt
{
FAKE_VALUE_FUNC(float, myCos, float);
}

void resetMyCos()
{
    RESET_FAKE(dmt::myCos);
}

TEST_CASE("[testdmtpartitions:moretrig] Test case")
{
    SECTION("someFunc should call myCos")
    {
        dmt::resetMockHistory();
        dmt::someFunc(3);
        CHECK(dmt::myCos_fake.call_count > 0);
        CALL_RESET(dmt::myCos);
        CHECK(dmt::myCos_fake.call_count == 0);
    }
}