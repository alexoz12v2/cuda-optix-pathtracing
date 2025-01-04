module;

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <numbers>

module testdmtpartitions;

TEST_CASE("[testdmtpartitions] My Sin")
{
    float tolerance = 10 * std::numeric_limits<float>::epsilon();
    SECTION("sin of pi is zero") { CHECK(std::abs(dmt::mySin(static_cast<float>(std::numbers::pi))) < tolerance); }
    SECTION("sin of pi/2 is 1")
    {
        CHECK(std::abs(dmt::mySin(static_cast<float>(std::numbers::pi / 2)) - 1.f) < tolerance);
    }
}