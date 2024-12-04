/**
* @file testdmtpartitions-trig.cppm
* @brief Partition Interface Unit `trig` containing trigonometric utilities in the `testdmtpartitions` module.
*/

export module testdmtpartitions:trig;
export import :moretrig;

namespace dmt
{
/**
 * @brief Computes the cosine of a given value.
 * @param x The input value (in radians).
 * @return The cosine of `x`.
 */
export float myCos(float x);
} // namespace dmt