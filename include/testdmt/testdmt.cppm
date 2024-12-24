/**
 * @file testdmt.cppm
 * @brief Primary interface unit for the `testdmt` module.
 *
 * This module provides basic mathematical operations and utilities.
 *
 * @defgroup testdmt TestDMT Module
 * @{
 */

// example of a primary module interface unit
module;

#include <iostream>

/**
 * @brief Module `testdmt`.
 */
export module testdmt;

namespace dmt {
    /**
     * @class TestMath
     * @brief Provides mathematical utilities.
     */
    export class TestMath
    {
    public:
        /**
         * @brief Adds two integers.
         * @param a First integer.
         * @param b Second integer.
         * @return The sum of `a` and `b`.
         */
        static int add(int a, int b);

        /**
         * @brief Prints a predefined message.
         * Displays "Hello darkness my old Friend" to the standard output.
         */
        static void printStuff()
        {
            std::cout << "Hello darkness my old Friend\n";
        }
    };
} // namespace dmt
/** @} */ // End of group testdmt