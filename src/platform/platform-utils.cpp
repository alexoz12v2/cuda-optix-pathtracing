module;

#if defined(DMT_OS_WINDOWS)
#pragma comment(lib, "mincore")
#include <AclAPI.h>
#include <Windows.h>
#include <securitybaseapi.h>
#include <sysinfoapi.h>
#endif

module platform;

namespace dmt {
} // namespace dmt
