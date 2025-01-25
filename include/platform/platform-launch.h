#pragma once

//#if defined(DMT_OS_WINDOWS)
//extern "C" int __stdcall wWinMain(struct HINSTANCE__*, struct HINSTANCE__*, wchar_t*, int);
//#else
//// argv is actually UTF-8 (char8_t)
//int main(int argc, char* argv[]);
//#endif

// to be defined by executables
// TODO takes environment (what's needed), command line and more...
// it can even create a context and set up initial loggign handlers
int guardedMain();

// to be defined before including this only once, on the entry point
#if defined(DMT_ENTRY_POINT)
#if !defined(DMT_WINDOWS_INCLUDED)
#define DMT_WINDOWS_INCLUDED
#if defined(DMT_OS_WINDOWS)
#include <Windows.h>
#endif
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow)
{
#if defined(DMT_DEBUG)
    // Before resuming the process
    MessageBox(nullptr, L"Attach debugger to child process now.", L"Debug Pause", MB_OK);
#endif
    int returnCode = guardedMain();
    return returnCode;
}
#else
// argv is actually UTF-8 (char8_t)
int main(int argc, char* argv[])
{
    int returnCode = guardedMain();
    return returnCode;
}
#endif

#endif
