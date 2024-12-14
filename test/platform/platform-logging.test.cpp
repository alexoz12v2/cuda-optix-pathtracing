module;

#include "dmtutils.h"

#include <catch2/catch_test_macros.hpp>

#include <fff/fff.h>
#include <numbers>
#include <source_location>
#include <stdcapture/stdcapture.h>
#include <string>
#include <vector>

#if defined(DMT_OS_LINUX)
#include <unistd.h>
#elif defined(DMT_OS_WINDOWS)
#include <windows.h>
#endif

module platform;

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


TEST_CASE("[platform-logging] CircularOStringStream Formatting", "[CircularOStringStream]")
{
    using namespace std::string_view_literals;

    SECTION("Format string with arguments")
    {
        dmt::CircularOStringStream stream;
        stream.logInitList("{} {}", {"Hello"sv, "World"sv});

        REQUIRE(stream.str() == "Hello World");
    }

    SECTION("Multiple format arguments")
    {
        dmt::CircularOStringStream stream;
        stream.logInitList("{} {} {}", {"One"sv, "Two"sv, "Three"sv});

        REQUIRE(stream.str() == "One Two Three");
    }
}


class alignas(8) MockAsyncIOManager : public dmt::BaseAsyncIOManager
{
public:
    uint32_t findFirstFreeBlocking()
    {
        ++m_.findFirstFreeBlockingCnt;
        return m_.findFirstFreeBlockingReturn;
    }

    char* operator[](uint32_t idx)
    {
        ++m_.subscriptCnt;
        return m_.buffers->operator[](idx).data();
    }

    char* get(uint32_t idx)
    {
        return m_.buffers->operator[](idx).data();
    }

    bool enqueue(uint32_t idx, size_t sz)
    {
        ++m_.enqueueCnt;
        return m_.enqueueReturn;
    }

    void setDestructorCalled(bool* p)
    {
        m_.destructorCalled = p;
    }

    ~MockAsyncIOManager()
    {
        if (m_.destructorCalled)
        {
            *m_.destructorCalled = true;
        }
        delete m_.buffers;
    }

    struct T
    {
        std::array<std::array<char, 4096>, numAios>* buffers = new std::array<std::array<char, 4096>, numAios>({});
        uint32_t                                     findFirstFreeBlockingReturn = 0;
        bool                                         enqueueReturn               = false;
        uint8_t                                      enqueueCnt                  = 0;
        uint8_t                                      findFirstFreeBlockingCnt    = 0;
        uint8_t                                      subscriptCnt                = 0;
        bool*                                        destructorCalled            = nullptr;
    };
    T             m_;
    unsigned char m_padding[dmt::asyncIOClassSize - sizeof(m_)];
};
static_assert(dmt::AsyncIOManager<MockAsyncIOManager>);

#if defined(DMT_OS_LINUX)
#elif defined(DMT_OS_WINDOWS)
#endif

#define DMT_ASYNC 0

TEST_CASE("[platform-logging] Console Logger public functionality", "[ConsoleLogger]")
{
    SECTION("when loglevel not enabled, it shouldn't perform any logging")
    {
        auto                logger  = dmt::ConsoleLogger::create<MockAsyncIOManager>(dmt::ELogLevel::WARN);
        MockAsyncIOManager& manager = logger.getInteralAs<MockAsyncIOManager>();
#if DMT_ASYNC && ((defined(DMT_OS_LINUX) && defined(_POSIX_ASYNCHRONOUS_IO)) || defined(DMT_OS_WINDOWS)) // logMessageAsync
        logger.log("fdsjkafhdslafjl");
        CHECK(manager.m_.findFirstFreeBlockingCnt == 0);
#else // logMessage
        std::string captureBuffer;
        {
            capture::CaptureStdout cap([&](char const* buf, size_t sz) { captureBuffer += std::string(buf, sz); });
            logger.log("fdsjkafhdslafjl");
        }
        CHECK(captureBuffer.empty());
#endif
    }

    SECTION("should perform logging with no arguments")
    {
        auto                logger  = dmt::ConsoleLogger::create<MockAsyncIOManager>();
        MockAsyncIOManager& manager = logger.getInteralAs<MockAsyncIOManager>();
        char const*         ptr     = "aaaaaaa";
#if DMT_ASYNC && ((defined(DMT_OS_LINUX) && defined(_POSIX_ASYNCHRONOUS_IO)) || defined(DMT_OS_WINDOWS)) // logMessageAsync
        logger.log(ptr);
        CHECK(std::strcmp(manager[manager.m_.findFirstFreeBlockingReturn], ptr) == 0);
#else // logMessage
        std::string captureBuffer;
        {
            capture::CaptureStdout cap([&](char const* buf, size_t sz) { captureBuffer += std::string(buf, sz); });
            logger.log("fdsjkafhdslafjl");
        }
        CHECK(captureBuffer == "fdsjkafhdslafjl");
#endif
    }

    SECTION("should perform logging with arguments")
    {
        auto                logger  = dmt::ConsoleLogger::create<MockAsyncIOManager>();
        MockAsyncIOManager& manager = logger.getInteralAs<MockAsyncIOManager>();
#if DMT_ASYNC && ((defined(DMT_OS_LINUX) && defined(_POSIX_ASYNCHRONOUS_IO)) || defined(DMT_OS_WINDOWS)) // logMessageAsync
        logger.log("LOG {}", {"log"});
        CHECK(std::strcmp(manager[manager.m_.findFirstFreeBlockingReturn], "LOG log") == 0);
#else // logMessage
        std::string captureBuffer;
        {
            capture::CaptureStdout cap([&](char const* buf, size_t sz) { captureBuffer += std::string(buf, sz); });
            logger.log("LOG {}", {"log"});
        }
        CHECK(captureBuffer == "LOG log");
#endif
    }

    SECTION("should call the internal class' destructor")
    {
        bool value = false;
        {
            auto                logger  = dmt::ConsoleLogger::create<MockAsyncIOManager>(dmt::ELogLevel::WARN);
            MockAsyncIOManager& manager = logger.getInteralAs<MockAsyncIOManager>();
            manager.setDestructorCalled(&value);
        }
        CHECK(value);
    }
}