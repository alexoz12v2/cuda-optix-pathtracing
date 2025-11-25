#ifndef DMT_PLATFORM_PUBLIC_PLATFORM_LAUNCH_H
#define DMT_PLATFORM_PUBLIC_PLATFORM_LAUNCH_H

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
    #if defined(DMT_OS_WINDOWS)
        #include <Windows.h>
        #if !defined(DMT_WINDOWS_CLI)
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
int wmain(int argc, wchar_t* argv[], wchar_t* envp[])
{
    int returnCode = guardedMain();
    return returnCode;
}
        #endif
    #else
// argv is actually UTF-8 (char8_t)
int main(int argc, char* argv[])
{
    int returnCode = guardedMain();
    return returnCode;
}
    #endif
#endif
#endif // DMT_PLATFORM_PUBLIC_PLATFORM_LAUNCH_H
