module;

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <numbers>

module testdmtpartitions; // implementation units for partitions do not redeclare the partition

TEST_CASE("[testdmtpartitions:trig] Test Case")
{
    SECTION("should return 0 on pi/2")
    {
        float tol = 10 * std::numeric_limits<float>::epsilon();
        CHECK(std::abs(dmt::myCos(static_cast<float>(std::numbers::pi) / 2)) < tol);
    }
}