/**
* @file testdmtpartitions.cppm
* @brief Primary interface unit for the `testdmtpartitions` module.
*
* This module provides advanced mathematical utilities with partitioned design.
*
* @defgroup testdmtpartitions TestDMTPartitions Module
* @{
*/

// example of primary module interface unit
module;

#include <cmath> // limited to this file alone, not implementation units (not used here)

/**
 * @brief Module `testdmtpartitions`.
 */
export module testdmtpartitions;

// the primary interface unit imports exports all partitions
export import :trig;

// you can import another module partition, and export only part of it from the primary module interface, while
// this means that, while module implementation units can use everything that a partition exports, only
// the part which is exposed by the primary interface unit is visible to the outside world
import :implonly;

// you can either export a namespace, class, variable or function
/*export*/ namespace dmt
{
/**
 * @brief Computes the sine of a given value.
 * @param x The input value (in radians).
 * @return The sine of `x`.
 */
export float mySin(float x);

// no doc as it is redeclared
export void implOnly(); // redeclare the part of the partition you want to export (gives readability warning)
} // namespace dmt
/** @} */ // End of group testdmtpartitions