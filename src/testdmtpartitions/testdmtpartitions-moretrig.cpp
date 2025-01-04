/**
 * @file testdmtpartitions-moretrig.cpp
 * @brief Implementation unit for the `moretrig` partition.
 */

module testdmtpartitions; // implementation units for partitions do not redeclare the partition
import :trig;             // intra module dependencies

namespace dmt {
    float someFunc(float x) { return myCos(x); }
} // namespace dmt