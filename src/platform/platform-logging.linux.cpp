#include "platform-logging.h"

#include <unistd.h>
#include <termios.h>

namespace dmt {
    // LOGGING 2.0 ----------------------------------------------------------------------------------------------------
    bool createConsoleHandler(LogHandler& _out, LogHandlerAllocate _alloc, LogHandlerDeallocate _dealloc)
    {
        _out.hostAllocate   = _alloc;
        _out.hostDeallocate = _dealloc;
        _out.hostFlush      = [](void* _data) { tcdrain(STDOUT_FILENO); };
        _out.hostFilter     = [](void* _data, LogRecord const& record) -> bool { return true; };
        _out.hostCallback   = [](void* _data, LogRecord const& record) {
            //std::string_view str{record.data, record.numBytes};
            //write(STDOUT_FILENO, str.data(), str.size());
        };
        _out.hostCleanup = [](LogHandlerDeallocate _dealloc, void* _data) {};
        return true;
    }

} // namespace dmt
