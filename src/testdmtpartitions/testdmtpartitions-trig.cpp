/**
 * @file testdmtpartitions-trig.cpp
 * @brief Implementation unit for the `trig` partition.
 */

module;

#include <cmath>

module testdmtpartitions; // implementation units for partitions do not redeclare the partition

namespace dmt {
    float myCos(float x) { return std::cosf(x); }
} // namespace dmt