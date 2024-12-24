/**
 * @file testdmtpartitions-implonly.cpp
 * @brief Implementation unit for the `implonly` partition.
 */

// you can also declare some module partitions as only implementation units, if they will be fully exported
// by the primary interface module
module;

#include <iostream>

module testdmtpartitions; // implementation units for partitions do not redeclare the partition

namespace dmt {
    void implOnlyHidden()
    {
        std::cout << "you can't see mee";
    }

    void implOnly()
    {
        implOnlyHidden();
        std::cout << "Basilare, impl only wow" << std::endl;
    }
} // namespace dmt
