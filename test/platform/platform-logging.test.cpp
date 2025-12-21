
#include "dmtutils.h"

#include <catch2/catch_test_macros.hpp>

#include <fff/fff.h>
#include <iostream>
#include <numbers>
#include <source_location>
#include <stdcapture/stdcapture.h>
#include <string>
#include <vector>

#if defined(DMT_OS_LINUX)
#  include <unistd.h>
#elif defined(DMT_OS_WINDOWS)
#  include <windows.h>
#endif
#include <platform/platform-logging.h>

TEST_CASE("[platform-logging] General properties of logging utility") {
  using namespace std::string_view_literals;
  SECTION("Log levels should be in ascending importance ") {
    STATIC_CHECK(dmt::ELogLevel::TRACE < dmt::ELogLevel::LOG);
    STATIC_CHECK(dmt::ELogLevel::LOG < dmt::ELogLevel::WARNING);
    STATIC_CHECK(dmt::ELogLevel::WARNING < dmt::ELogLevel::ERR);
    STATIC_CHECK(dmt::ELogLevel::ERR < dmt::ELogLevel::NONE);
  }

  SECTION("Log Levels should have correct string representatiohn") {
    STATIC_CHECK(dmt::stringFromLevel(dmt::ELogLevel::TRACE) == "TRACE"sv);
    STATIC_CHECK(dmt::stringFromLevel(dmt::ELogLevel::LOG) == "LOG  "sv);
    STATIC_CHECK(dmt::stringFromLevel(dmt::ELogLevel::WARNING) == "WARN "sv);
    STATIC_CHECK(dmt::stringFromLevel(dmt::ELogLevel::ERR) == "ERROR"sv);
  }
}

TEST_CASE("[platform-logging] StrBuf Basic Functionality", "[StrBuf]") {
  SECTION("CString Constructor") {
    dmt::StrBuf buf("Hello", 5);
    REQUIRE(std::string_view(buf.str, static_cast<uint32_t>(buf.len)) ==
            "Hello");
  }

  SECTION("Zero-Terminated CString Constructor") {
    dmt::StrBuf buf("Hello");
    REQUIRE(std::string_view(buf.str, static_cast<uint32_t>(buf.len)) ==
            "Hello");
  }

  SECTION("String View Constructor") {
    std::string_view view = "Hello";
    dmt::StrBuf buf(view);
    REQUIRE(std::string_view(buf.str, static_cast<uint32_t>(buf.len)) ==
            "Hello");
  }

  SECTION("Boolean Constructor") {
    dmt::StrBuf trueBuf(true);
    REQUIRE(std::string_view(trueBuf.str, static_cast<uint32_t>(trueBuf.len)) ==
            "true");

    dmt::StrBuf falseBuf(false);
    REQUIRE(std::string_view(falseBuf.str,
                             static_cast<uint32_t>(falseBuf.len)) == "false");
  }

  SECTION("Floating Point Constructor") {
    dmt::StrBuf buf(std::numbers::pi_v<float>, "%.2f");
    REQUIRE(std::string_view(buf.buf, static_cast<uint32_t>(-buf.len)) ==
            "3.14");
  }

  SECTION("Integral Constructor") {
    dmt::StrBuf buf(42, "%d");
    REQUIRE(std::string_view(buf.buf, static_cast<uint32_t>(-buf.len)) == "42");
  }
}

TEST_CASE("[platform-logging] CircularOStringStream Formatting",
          "[CircularOStringStream]") {
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
