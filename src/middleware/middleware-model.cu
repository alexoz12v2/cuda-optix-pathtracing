#define DMT_INTERFACE_AS_HEADER
#undef DMT_NEEDS_MODULE
#include "dmtmacros.h"
#include "middleware-model.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace dmt::model {
    using namespace dmt;
    void test(AppContext& ctx) { ctx.log("AAAAAAAAAAAAAAAAAAAAAAAAAAAA"); }
} // namespace dmt::model

namespace dmt::model::soa {
    using namespace dmt;
}