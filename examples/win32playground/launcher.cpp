#include <windows.h>
#include <iostream>
#include <string>

#include <strsafe.h>

void ErrorExit(wchar_t const* lpszFunction);

struct PipePair
{
    HANDLE hChildStd_IN_Rd;
    HANDLE hChildStd_IN_Wr;
    HANDLE hChildStd_OUT_Rd;
    HANDLE hChildStd_OUT_Wr;
};

struct Janitor
{
    Janitor() : pipes(reinterpret_cast<PipePair*>(HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(PipePair))))
    {
        if (pipes == nullptr)
            ErrorExit(L"Couldn't allocate Heap Memory for Pipes struct");
    }
    Janitor(Janitor const&)                = delete;
    Janitor(Janitor&&) noexcept            = delete;
    Janitor& operator=(Janitor const&)     = delete;
    Janitor& operator=(Janitor&&) noexcept = delete;
    ~Janitor() noexcept
    {
        if (pipes)
            HeapFree(GetProcessHeap(), 0, pipes);
    }

    PipePair* pipes = nullptr;
};

DWORD WINAPI stdinPipeThread(void* params)
{
    wchar_t buffer[1024]{};
    auto*   pipes  = reinterpret_cast<PipePair*>(params);
    HANDLE  hStdin = GetStdHandle(STD_INPUT_HANDLE);
    while (true) // TODO until handles are not closed
    {
        DWORD numBytes = 0;
        if (!ReadFile(hStdin, buffer, 1024, &numBytes, nullptr))
            break;
        if (!WriteFile(pipes->hChildStd_IN_Wr, buffer, numBytes, nullptr, nullptr))
            break;
    }
    // TODO close the handle so that the child stops reading

    ExitThread(0);
    return 0;
}

DWORD WINAPI stdoutPipeThread(void* params)
{
    wchar_t buffer[1024]{};
    auto*   pipes    = reinterpret_cast<PipePair*>(params);
    HANDLE  hStdout  = GetStdHandle(STD_OUTPUT_HANDLE);
    HANDLE  hStderr  = GetStdHandle(STD_ERROR_HANDLE);
    DWORD   numBytes = 0;
    // TODO integrate stderr
    while (ReadFile(pipes->hChildStd_OUT_Rd, buffer, 1024, &numBytes, nullptr))
    {
        WriteFile(hStdout, buffer, numBytes, nullptr, nullptr);
    }

    ExitThread(0);
    return 0;
}

int main()
{
    static constexpr uint32_t BUFSIZE = 4096;
    TCHAR                     moduleName[MAX_PATH];
    GetModuleFileName(nullptr, moduleName, MAX_PATH);

    // Convert `.com` to `.exe`
    std::wstring modulePath(moduleName);
    size_t       dotComPos = modulePath.find_last_of(L'.');
    if (dotComPos == std::wstring::npos || modulePath.substr(dotComPos) != L".com")
    {
        std::wcerr << L"ERROR: Launcher was not invoked as a .com file." << std::endl;
        return 1;
    }
    modulePath.replace(dotComPos, 4, L".exe"); // Replace `.com` with `.exe`

    // Prepare command line
    std::wstring commandLine = L"\"" + modulePath + L"\" ";
    commandLine += GetCommandLine(); // Append all arguments

    PROCESS_INFORMATION processInfo = {0};

    Janitor    j;
    auto const ReadFromPipe = [&]() {
        DWORD  dwRead, dwWritten;
        CHAR   chBuf[BUFSIZE];
        BOOL   bSuccess      = FALSE;
        HANDLE hParentStdOut = GetStdHandle(STD_OUTPUT_HANDLE);

        for (;;)
        {
            bSuccess = ReadFile(j.pipes->hChildStd_OUT_Rd, chBuf, BUFSIZE, &dwRead, NULL);
            if (!bSuccess || dwRead == 0)
                break;

            bSuccess = WriteFile(hParentStdOut, chBuf, dwRead, &dwWritten, NULL);
            if (!bSuccess)
                break;
        }
    };

    SECURITY_ATTRIBUTES saAttr;
    saAttr.nLength              = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle       = TRUE; // This makes pipe handles to be inherited
    saAttr.lpSecurityDescriptor = nullptr;

    // create a pipe for the child process's STDOUT and STDIN
    if (!CreatePipe(&j.pipes->hChildStd_OUT_Rd, &j.pipes->hChildStd_OUT_Wr, &saAttr, 0))
        ErrorExit(L"Failed to create STDOUT Pipe");

    if (!CreatePipe(&j.pipes->hChildStd_IN_Rd, &j.pipes->hChildStd_IN_Wr, &saAttr, 0))
        ErrorExit(L"Failed to create STDIN Pipe");

    // Make sure that child cannot read from stdout and write to stdin
    if (!SetHandleInformation(j.pipes->hChildStd_OUT_Rd, HANDLE_FLAG_INHERIT, 0))
        ErrorExit(L"Couldn't Turn off inheritance of the Read end of the STDOUT Pipe");

    if (!SetHandleInformation(j.pipes->hChildStd_IN_Wr, HANDLE_FLAG_INHERIT, 0))
        ErrorExit(L"Couldn't turn off inheriance of the Write end of the STDIN Pipe");

    STARTUPINFO startupInfo = {};
    startupInfo.cb          = sizeof(STARTUPINFO);
    startupInfo.dwFlags     = STARTF_USESTDHANDLES; // Use redirected stdin/stdout
    startupInfo.hStdError   = j.pipes->hChildStd_OUT_Wr;
    startupInfo.hStdOutput  = j.pipes->hChildStd_OUT_Wr;
    startupInfo.hStdInput   = j.pipes->hChildStd_IN_Rd;

    // Create process in suspended state
    if (!CreateProcessW(nullptr,
                        const_cast<LPWSTR>(commandLine.c_str()),
                        nullptr,
                        nullptr,
                        TRUE, // Inherit handles
                        CREATE_SUSPENDED,
                        nullptr,
                        nullptr,
                        &startupInfo,
                        &processInfo))
        ErrorExit(L"ERROR: Failed to create process");

    // Start IO Listener threads
    DWORD  tids[2];
    HANDLE hThreads[2];
    hThreads[0] = CreateThread(nullptr, 0, stdoutPipeThread, j.pipes, 0, &tids[0]);
    if (!hThreads[0])
        ErrorExit(L"Couldn't create stdout listener thread");

    hThreads[1] = CreateThread(nullptr, 0, stdinPipeThread, j.pipes, 0, &tids[1]);
    if (!hThreads[1])
        ErrorExit(L"Couldn't create stdin listener thread");

    // Resume the suspended process
    ResumeThread(processInfo.hThread);

    // Wait for the process to complete
    WaitForSingleObject(processInfo.hProcess, INFINITE);

    // Retrieve the exit code
    DWORD exitCode;
    GetExitCodeProcess(processInfo.hProcess, &exitCode);

    // wait for io threads (for now this hangs)
    WaitForMultipleObjects(2, hThreads, true, INFINITE);

    // Clean up
    CloseHandle(processInfo.hThread);
    CloseHandle(processInfo.hProcess);

    return static_cast<int>(exitCode);
}


// Format a readable error message, display a message box,
// and exit from the application.
void ErrorExit(wchar_t const* lpszFunction)
{
    wchar_t* lpMsgBuf;
    wchar_t* lpDisplayBuf;
    DWORD    dw = GetLastError();

    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                  NULL,
                  dw,
                  MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                  reinterpret_cast<wchar_t*>(&lpMsgBuf),
                  0,
                  NULL);

    DWORD const numBytes = (wcslen(lpMsgBuf) + wcslen(lpszFunction) + 40) * sizeof(wchar_t);
    lpDisplayBuf         = reinterpret_cast<wchar_t*>(LocalAlloc(LMEM_ZEROINIT, numBytes));

    StringCchPrintf(lpDisplayBuf,
                    LocalSize(lpDisplayBuf) / sizeof(wchar_t),
                    L"%s failed with error %d: %s",
                    lpszFunction,
                    dw,
                    lpMsgBuf);
    MessageBox(NULL, lpDisplayBuf, L"Error", MB_OK);

    LocalFree(lpMsgBuf);
    LocalFree(lpDisplayBuf);
    ExitProcess(1);
}
