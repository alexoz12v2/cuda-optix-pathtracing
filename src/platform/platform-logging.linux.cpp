#include "platform-logging.h"

#include <unistd.h>
#if defined(_POSIX_ASYNCHRONOUS_IO) // TODO remove in favour of another stragegy
    #include <aio.h> // https://man7.org/linux/man-pages/man7/aio.7.html https://www.gnu.org/software/libc/manual/html_node/Asynchronous-I_002fO.html
#else
    #error "Only supported implementation uses aio.h"
#endif

namespace dmt {
    // LOGGING 2.0 ----------------------------------------------------------------------------------------------------
    bool createConsoleHandler(LogHandler& _out, LogHandlerAllocate _alloc, LogHandlerDeallocate _dealloc)
    {
        _out.hostAllocate   = _alloc;
        _out.hostDeallocate = _dealloc;
        _out.hostFlush      = [](void* _data) {};
        _out.hostFilter     = [](void* _data, LogRecord const& record) -> bool { return true; };
        _out.hostCallback = [](void* _data, LogRecord const& record) { puts(std::bit_cast<char const*>(record.data)); };
        _out.hostCleanup  = [](LogHandlerDeallocate _dealloc, void* _data) {};
        return true;
    }

} // namespace dmt
