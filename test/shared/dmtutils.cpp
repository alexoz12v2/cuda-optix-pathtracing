#include "dmtutils.h"

DEFINE_FFF_GLOBALS;

namespace dmt
{
int testWork()
{
    return 4;
}

void resetMockHistory()
{
    FFF_RESET_HISTORY();
}
} // namespace dmt