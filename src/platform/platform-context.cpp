#include "platform-context.h"

namespace dmt {
    ContextImpl::ContextImpl() {}

    ContextImpl::~ContextImpl()
    {
        for (uint32_t i = 0; i < common.numHandlers; ++i)
            common.handlers[i].hostCleanup(common.handlers[i].hostDeallocate, common.handlers[i].data);
    }

    void Context::flush()
    {
        for (uint32_t i = 0; i < m_pimpl->common.numHandlers; ++i)
        {
            if (m_pimpl->handlerEnabled(i))
                m_pimpl->common.handlers[i].hostFlush(m_pimpl->common.handlers[i].data);
        }
    }

} // namespace dmt