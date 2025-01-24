#include "platform-context.h"

// TODO rename this platfom-context.win32.cpp
#if !defined(DMT_OS_WINDOWS)
#error "What"
#endif

namespace dmt {
    ContextImpl::ContextImpl() {}

    ContextImpl::~ContextImpl() {}

    LogHandler* ContextImpl::addHandler()
    {
        if (common.numHandlers >= maxHandlers)
            return nullptr;
        common.handlers[common.numHandlers++] = {};
        return &common.handlers[common.numHandlers - 1];
    }
} // namespace dmt