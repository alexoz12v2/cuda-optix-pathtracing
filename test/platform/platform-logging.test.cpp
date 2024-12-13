module;

#include "dmtutils.h"

#include <catch2/catch_test_macros.hpp>

#include <numbers>
#include <source_location>

module platform;

namespace dmt
{
}

TEST_CASE("[platform-logging] General properties of logging utility")
{
    using namespace std::string_view_literals;
    SECTION("Log levels should be in ascending importance ")
    {
        STATIC_CHECK(dmt::ELogLevel::TRACE < dmt::ELogLevel::LOG);
        STATIC_CHECK(dmt::ELogLevel::LOG < dmt::ELogLevel::WARN);
        STATIC_CHECK(dmt::ELogLevel::WARN < dmt::ELogLevel::ERROR);
        STATIC_CHECK(dmt::ELogLevel::ERROR < dmt::ELogLevel::NONE);
    }

    SECTION("Log Levels should have correct string representatiohn")
    {
        STATIC_CHECK(dmt::stringFromLevel(dmt::ELogLevel::TRACE) == "TRACE"sv);
        STATIC_CHECK(dmt::stringFromLevel(dmt::ELogLevel::LOG) == "LOG  "sv);
        STATIC_CHECK(dmt::stringFromLevel(dmt::ELogLevel::WARN) == "WARN "sv);
        STATIC_CHECK(dmt::stringFromLevel(dmt::ELogLevel::ERROR) == "ERROR"sv);
    }

    SECTION("Log colors should be correctly assigned")
    {
        STATIC_CHECK(dmt::logcolor::colorFromLevel(dmt::ELogLevel::TRACE) == dmt::logcolor::greyGreen);
        STATIC_CHECK(dmt::logcolor::colorFromLevel(dmt::ELogLevel::LOG) == dmt::logcolor::brightWhite);
        STATIC_CHECK(dmt::logcolor::colorFromLevel(dmt::ELogLevel::WARN) == dmt::logcolor::brightYellow);
        STATIC_CHECK(dmt::logcolor::colorFromLevel(dmt::ELogLevel::ERROR) == dmt::logcolor::red);
    }
}

TEST_CASE("[platform-logging] StrBuf Basic Functionality", "[StrBuf]")
{
    SECTION("CString Constructor")
    {
        dmt::StrBuf buf("Hello", 5);
        REQUIRE(std::string_view(buf.str, static_cast<uint32_t>(buf.len)) == "Hello");
    }

    SECTION("Zero-Terminated CString Constructor")
    {
        dmt::StrBuf buf("Hello");
        REQUIRE(std::string_view(buf.str, static_cast<uint32_t>(buf.len)) == "Hello");
    }

    SECTION("String View Constructor")
    {
        std::string_view view = "Hello";
        dmt::StrBuf      buf(view);
        REQUIRE(std::string_view(buf.str, static_cast<uint32_t>(buf.len)) == "Hello");
    }

    SECTION("Boolean Constructor")
    {
        dmt::StrBuf trueBuf(true);
        REQUIRE(std::string_view(trueBuf.str, static_cast<uint32_t>(trueBuf.len)) == "true");

        dmt::StrBuf falseBuf(false);
        REQUIRE(std::string_view(falseBuf.str, static_cast<uint32_t>(falseBuf.len)) == "false");
    }

    SECTION("Floating Point Constructor")
    {
        dmt::StrBuf buf(std::numbers::pi_v<float>, "%.2f");
        REQUIRE(std::string_view(buf.buf, static_cast<uint32_t>(-buf.len)) == "3.14");
    }

    SECTION("Integral Constructor")
    {
        dmt::StrBuf buf(42, "%d");
        REQUIRE(std::string_view(buf.buf, static_cast<uint32_t>(-buf.len)) == "42");
    }
}


TEST_CASE("[platform-logging] CircularOStringStream Formatting", "[CircularOStringStream]") {
    using namespace std::string_view_literals;

    SECTION("Format string with arguments") {
        dmt::CircularOStringStream stream;
        stream.logInitList("{} {}", {"Hello"sv, "World"sv});

        REQUIRE(stream.str() == "Hello World");
    }

    SECTION("Multiple format arguments") {
        dmt::CircularOStringStream stream;
        stream.logInitList("{} {} {}", {"One"sv, "Two"sv, "Three"sv});

        REQUIRE(stream.str() == "One Two Three");
    }
}


/*
TEST_CASE("[platform-logging] ConsoleLogger Basic Functionality", "[ConsoleLogger]")
{
    using namespace std::string_view_literals;
    // Create an instance of ConsoleLogger
    dmt::ConsoleLogger logger(dmt::ELogLevel::LOG);

    // Test writing a simple log message
    SECTION("Write simple log message")
    {
        std::string message = "Test log message";
        logger.write(dmt::ELogLevel::LOG, message, std::source_location::current());

        // Verify the message is written correctly (Mock or check with the output)
        // Since we don't have access to the actual console, consider mocking the output stream.
    }

    // Test writing a formatted log message
    SECTION("Write formatted log message")
    {
        std::string format = "Formatted log message: {} {}";
        logger.write(dmt::ELogLevel::LOG, format, {"arg1"sv, "arg2"sv}, std::source_location::current());

        // Verify the formatted message is written correctly
    }

    // Test invalid log level
    SECTION("Ignore lower log levels")
    {
        std::string message = "This message should not appear";
        logger.write(dmt::ELogLevel::TRACE, message, std::source_location::current());

        // Verify that the message is not written
    }

    // Test timestamp generation
    SECTION("Timestamp generation")
    {
        auto timestamp = logger.write;

        REQUIRE_FALSE(timestamp.empty());
        // Add more checks to validate the format if necessary
    }

    // Test relative file name carving
    SECTION("Relative file name carving")
    {
    }
}

TEST_CASE("[platform-logging]ConsoleLogger Thread Safety", "[ConsoleLogger][Multithreading]")
{
    ConsoleLogger logger(ELogLevel::LOG);

    SECTION("Concurrent writes")
    {
        size_t const threadCount = 10;
        size_t const iterations  = 100;

        std::vector<std::thread> threads;
        for (size_t i = 0; i < threadCount; ++i)
        {
            threads.emplace_back(
                [&logger, iterations, i]()
                {
                    for (size_t j = 0; j < iterations; ++j)
                    {
                        logger.write(ELogLevel::LOG,
                                     "Thread " + std::to_string(i) + " message " + std::to_string(j),
                                     std::source_location::current());
                    }
                });
        }

        for (auto& thread : threads)
        {
            thread.join();
        }

        // Verify that all messages are logged without corruption (requires mocking output)
    }
}
 */