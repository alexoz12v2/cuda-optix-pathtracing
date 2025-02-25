/**
 * @file testdmtpartitions-implonly.cppm
 * @brief Partition Interface Unit `implonly` for the `testdmtpartitions` module.
 */

export module testdmtpartitions:implonly;

export namespace dmt {

    /**
     * @brief A hidden implementation detail function.
     * Not exported in the primary interface.
     */
    void implOnlyHidden();

    /**
     * @brief A function to demonstrate implementation-only exports.
     */
    void implOnly();
} // namespace dmt