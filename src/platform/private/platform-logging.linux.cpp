#include "platform-logging.h"

#include <unistd.h>
#include <termios.h>
#include <signal.h>

#include <atomic>

namespace /*static*/ {
    std::atomic<bool> g_terminalColorReset{false};

    void restoreTerminalColor()
    {
        if (!g_terminalColorReset.exchange(true))
        {
            // Reset terminal color
            char const* reset = "\033[0m";
            write(STDOUT_FILENO, reset, 4);
        }
    }

    void signalHandler(int)
    {
        restoreTerminalColor();
        _Exit(1); // terminate immediately
    }

    void setupSignalHandlers()
    {
        static bool initialized = false;
        if (!initialized)
        {
            initialized = true;
            signal(SIGINT, signalHandler);
            signal(SIGTERM, signalHandler);
            signal(SIGABRT, signalHandler);
            signal(SIGSEGV, signalHandler);
            signal(SIGFPE, signalHandler);
            signal(SIGILL, signalHandler);
        }
    }
} // namespace

namespace dmt {
    bool createConsoleHandler(LogHandler& _out, LogHandlerAllocate _alloc, LogHandlerDeallocate _dealloc)
    {
        setupSignalHandlers();

        _out.hostAllocate   = _alloc;
        _out.hostDeallocate = _dealloc;
        _out.hostFlush      = [](void* _data) { tcdrain(STDOUT_FILENO); };
        _out.hostFilter     = [](void* _data, LogRecord const&) -> bool { return true; };

        _out.hostCallback = [](void*, LogRecord const& record) {
            // --- Determine color and level prefix ---
            char const* color       = "\033[0m";
            char const* levelPrefix = "";

            switch (record.level)
            {
                case ELogLevel::TRACE:
                    color       = "\033[36m";
                    levelPrefix = "[TRACE] ";
                    break; // cyan
                case ELogLevel::LOG:
                    color       = "\033[0m";
                    levelPrefix = "[INFO ] ";
                    break; // green
                case ELogLevel::WARNING:
                    color       = "\033[33m";
                    levelPrefix = "[WARN ] ";
                    break; // yellow
                case ELogLevel::ERR:
                    color       = "\033[31m";
                    levelPrefix = "[ERROR] ";
                    break; // red
                default:
                    color       = "\033[0m";
                    levelPrefix = "[UNKWN] ";
                    break;
            }

            // --- Shorten file path ---
            std::string_view file(record.srcLoc.file_name());
            size_t           lastSlash = file.find_last_of("/\\");
            if (lastSlash != std::string_view::npos)
            {
                file = file.substr(lastSlash + 1);
            }

            // --- Compose prefix ---
            char header[256];
            int  headerLen = std::snprintf(header,
                                          sizeof(header),
                                          "[%s:%u %s] ",
                                          file.data(),
                                          record.srcLoc.line(),
                                          record.srcLoc.function_name());

            // --- Write color, header, level prefix, message ---
            write(STDOUT_FILENO, color, strlen(color));
            write(STDOUT_FILENO, header, headerLen);
            write(STDOUT_FILENO, levelPrefix, strlen(levelPrefix));
            write(STDOUT_FILENO, record.data, record.numBytes);

            // --- Restore terminal color ---
            restoreTerminalColor();
        };

        _out.hostCleanup = [](LogHandlerDeallocate, void*) { restoreTerminalColor(); };

        return true;
    }
} // namespace dmt