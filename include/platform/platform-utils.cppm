/**
 * @file platform-utils.cppm
 * @brief file containing any class/function needed within the module
 *
 * @defgroup platform platform Module
 * @{
 */
module;

#include <source_location>
#include <string_view>

#include <cstdint>
#include <cstdio>

export module platform:utils;

import :logging;

export namespace dmt
{

/**
 * Convenience pointer type to pass around systems of the application to get an enriched/targeted
 * interface to the platform
 * @warning I don't like the fact that we are redunding the logger interface, in particular the
 * redundant `m_level`
 */
class PlatformContext : public BaseLogger<PlatformContext>
{
public:
    // -- Types --
    /**
     * Required stuff from the `BaseLogger`
     */
    struct Traits
    {
        static constexpr ELogDisplay displayType = ELogDisplay::Forward;
    };

    /**
     * Function pointer table stored elsewhere. It is supposed to outlive the context.
     * Meant for functions which are not called often, therefore they can afford the
     * double indirection
     */
    struct Table
    {
        void (*changeLevel)(void* data, ELogLevel level) = [](void* data, ELogLevel level) {};
    };

    /**
     * Function pointer table stored inline here. Meant for functions which are called
     * often
     */
    struct InlineTable
    {
        void (*write)(void* data, ELogLevel level, std::string_view const& str, std::source_location const& loc) =
            [](void* data, ELogLevel level, std::string_view const& str, std::source_location const& loc) {};
        void (*writeArgs)(void*                                data,
                          ELogLevel                            level,
                          std::string_view const&              str,
                          std::initializer_list<StrBuf> const& list,
                          std::source_location const&          loc) =
            [](void*                                data,
               ELogLevel                            level,
               std::string_view const&              str,
               std::initializer_list<StrBuf> const& list,
               std::source_location const&          loc) {};
        bool (*checkLevel)(void* data, ELogLevel level) = [](void* data, ELogLevel level) { return false; };
    };

    PlatformContext(void* data, Table const* pTable, InlineTable const& inlineTable) :
    m_table(pTable),
    m_inlineTable(inlineTable),
    m_data(data)
    {
    }

    /**
     * Setter for the `m_level`
     * @param level new level
     * @warning Purposefully name hiding the `BaseLogger`
     */
    void setLevel(ELogLevel level)
    {
        m_table->changeLevel(m_data, level);
    }

    /**
     * Write function mandated by the CRTP pattern of the class `BaseLogger`
     * @param level log level
     * @param str string to output
     * @param loc location of the log
     */
    void write(ELogLevel level, std::string_view const& str, std::source_location const& loc)
    {
        m_inlineTable.write(m_data, level, str, loc);
    }

    /**
     * Write function mandated by the CRTP pattern of the class `BaseLogger`
     * @param level log level
     * @param str format string
     * @param list list of arguments which will be used to create the final string
     * @param loc location of the log
     */
    void write(ELogLevel                            level,
               std::string_view const&              str,
               std::initializer_list<StrBuf> const& list,
               std::source_location const&          loc)
    {
        m_inlineTable.writeArgs(m_data, level, str, list, loc);
    }

    /**
     * CRTP overridden function to check if the true underlying logger is enabled on the log level
     * @param level log level requested
     * @return bool signaling whether the requested log level is enabled
     */
    bool enabled(ELogLevel level)
    {
        return m_inlineTable.checkLevel(m_data, level);
    }

private:
    /**
     * table of infrequent functions offloaded, stored stored elsewhere
     */
    Table const* m_table;

    /**
     * Table of frequent functions stored here
     */
    InlineTable m_inlineTable;

    /**
     * Pointer to a type erased class, which can be casted back in the function pointer
     * functions, like `Platform`
     */
    void* m_data = nullptr;
};
}; // namespace dmt
/** @} */
