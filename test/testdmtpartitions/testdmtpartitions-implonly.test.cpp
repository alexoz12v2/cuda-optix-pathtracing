// you can also declare some module partitions as only implementation units, if they will be fully exported
// by the primary interface module
module;

#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <sstream>

module testdmtpartitions; // implementation units for partitions do not redeclare the partition

class OutputRedirect
{
public:
    OutputRedirect() : m_originalCoutBuffer(std::cout.rdbuf()) { std::cout.rdbuf(m_mockCout.rdbuf()); }
    OutputRedirect(OutputRedirect const&)            = delete;
    OutputRedirect(OutputRedirect&&)                 = delete;
    OutputRedirect& operator=(OutputRedirect const&) = delete;
    OutputRedirect& operator=(OutputRedirect&&)      = delete;
    ~OutputRedirect() { std::cout.rdbuf(m_originalCoutBuffer); }

    std::string getContent()
    {
        m_mockCout << std::flush;
        return m_mockCout.str();
    }

private:
    std::ostringstream m_mockCout;
    std::streambuf*    m_originalCoutBuffer = nullptr;
};

TEST_CASE("[testdmtpartitions:implonly]")
{
    SECTION("implOnlyHidden should print text")
    {
        auto              var      = OutputRedirect();
        std::string const expected = "you can't see mee";
        dmt::implOnlyHidden();
        std::cout << std::flush;
        CHECK(var.getContent() == expected);
    }
    SECTION("implOnly should print some text")
    {
        auto var = OutputRedirect();
        dmt::implOnly();
        std::cout << std::flush;
        CHECK(var.getContent().length() > 10);
    }
}
