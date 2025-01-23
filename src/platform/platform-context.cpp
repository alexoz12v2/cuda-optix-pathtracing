#include "platform-context.h"

// TODO rename this platfom-context.win32.cpp
#if !defined(DMT_OS_WINDOWS)
#error "What"
#endif

namespace dmt {
    ContextImpl::ContextImpl() {}

    ContextImpl::~ContextImpl() {}

    bool ContextImpl::addHandler(LogHandler handler)
    {
        if (common.numHandlers >= maxHandlers)
            return false;
        common.handlers[common.numHandlers++] = handler;
        return true;
    }
} // namespace dmt