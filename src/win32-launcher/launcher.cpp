#include <windows.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <strsafe.h>

[[noreturn]] void ErrorExit(wchar_t const* lpszFunction);

//#define USE_NAMED_PIPES
// TODO: Switch from console API to pseudoconsole https://learn.microsoft.com/en-us/windows/console/creating-a-pseudoconsole-session
// https://devblogs.microsoft.com/commandline/windows-command-line-introducing-the-windows-pseudo-console-conpty/

#if !defined(USE_NAMED_PIPES)
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

    if (!CloseHandle(pipes->hChildStd_IN_Wr))
        ErrorExit(L"Couldn't close write end of the stdin pipe");

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
#else
struct StdNamedPipes
{
    StdNamedPipes(DWORD pid)
    {
        static constexpr DWORD CONNECTIMEOUT = 1000;
        static constexpr DWORD nameSize      = 256;
        void*                  memory        = HeapAlloc(GetProcessHeap(), 0, 3ULL * nameSize * sizeof(wchar_t));
        if (!memory)
            ErrorExit(L"Failed to allocate memory to store Names of the Pipes");
        nameStdOutPipe = reinterpret_cast<wchar_t*>(memory);
        nameStdInPipe  = reinterpret_cast<wchar_t*>(memory) + nameSize;
        nameStdErrPipe = reinterpret_cast<wchar_t*>(memory) + 2ULL * nameSize;

        int32_t numBytes = _snwprintf(nameStdOutPipe, nameSize, L"\\\\.\\pipe\\%dcout", pid);
        hStdOutPipe      = CreateNamedPipe(nameStdOutPipe,
                                      PIPE_ACCESS_INBOUND,
                                      PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                                      1,
                                      1024,
                                      1024,
                                      CONNECTIMEOUT,
                                      nullptr);
        if (hStdOutPipe == INVALID_HANDLE_VALUE)
            ErrorExit(L"Failed To Create Named Pipe for STDOUT");

        numBytes   = _snwprintf(nameStdInPipe, nameSize, L"\\\\.\\pipe\\%dcin", pid);
        hStdInPipe = CreateNamedPipe(nameStdInPipe,
                                     PIPE_ACCESS_OUTBOUND,
                                     PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                                     1,
                                     1024,
                                     1024,
                                     CONNECTIMEOUT,
                                     nullptr);
        if (hStdInPipe == INVALID_HANDLE_VALUE)
            ErrorExit(L"Failed To Create Named Pipe for STDIN");

        numBytes    = _snwprintf(nameStdErrPipe, nameSize, L"\\\\.\\pipe\\%dcerr", pid);
        hStdErrPipe = CreateNamedPipe(nameStdErrPipe,
                                      PIPE_ACCESS_INBOUND,
                                      PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                                      1,
                                      1024,
                                      1024,
                                      CONNECTIMEOUT,
                                      nullptr);

        if (hStdErrPipe == INVALID_HANDLE_VALUE)
            ErrorExit(L"Failed To Create Named Pipe for STDERR");
    }
    StdNamedPipes(StdNamedPipes const&)                = delete;
    StdNamedPipes(StdNamedPipes&&) noexcept            = delete;
    StdNamedPipes& operator=(StdNamedPipes const&)     = delete;
    StdNamedPipes& operator=(StdNamedPipes&&) noexcept = delete;
    ~StdNamedPipes() noexcept
    {
        Sleep(100); //Give pipes a chance to flush
        CloseHandle(hStdOutPipe);
        CloseHandle(hStdInPipe);
        CloseHandle(hStdErrPipe);
        HeapFree(GetProcessHeap(), 0, nameStdOutPipe);
    }

    bool connectPipes()
    {
        return ConnectNamedPipe(hStdOutPipe, nullptr) && ConnectNamedPipe(hStdInPipe, nullptr) &&
               ConnectNamedPipe(hStdErrPipe, nullptr);
    }

    void disconnectPipes()
    {
        DisconnectNamedPipe(hStdOutPipe);
        DisconnectNamedPipe(hStdInPipe);
        DisconnectNamedPipe(hStdErrPipe);
    }

    wchar_t* nameStdOutPipe;
    wchar_t* nameStdInPipe;
    wchar_t* nameStdErrPipe;
    HANDLE   hStdOutPipe;
    HANDLE   hStdInPipe;
    HANDLE   hStdErrPipe;
};

struct Data
{
    Data(DWORD pid) :
    pipes(pid),
    hStdOut_Wr(INVALID_HANDLE_VALUE),
    hStdErr_Wr(INVALID_HANDLE_VALUE),
    hStdIn_Rd(INVALID_HANDLE_VALUE)
    {
        //pipes.connectPipes();
        createFiles();
    }
    Data(Data const&)                = delete;
    Data(Data&&) noexcept            = delete;
    Data& operator=(Data const&)     = delete;
    Data& operator=(Data&&) noexcept = delete;
    ~Data()
    {
        closeHandles();
        pipes.disconnectPipes();
    }

    void createFiles()
    {
        SECURITY_ATTRIBUTES saAttr;
        saAttr.nLength              = sizeof(SECURITY_ATTRIBUTES);
        saAttr.bInheritHandle       = true;
        saAttr.lpSecurityDescriptor = nullptr;
        hStdOut_Wr = CreateFileW(pipes.nameStdOutPipe, GENERIC_WRITE, 0, &saAttr, OPEN_EXISTING, 0, nullptr);
        if (hStdOut_Wr == INVALID_HANDLE_VALUE)
            ErrorExit(L"Error opening STDOUT in write mode");
        hStdErr_Wr = CreateFileW(pipes.nameStdErrPipe, GENERIC_WRITE, 0, &saAttr, OPEN_EXISTING, 0, nullptr);
        if (hStdErr_Wr == INVALID_HANDLE_VALUE)
            ErrorExit(L"Error opening STDERR in write mode");
        hStdIn_Rd = CreateFileW(pipes.nameStdInPipe, GENERIC_READ, 0, &saAttr, OPEN_EXISTING, 0, nullptr);
        if (hStdIn_Rd == INVALID_HANDLE_VALUE)
            ErrorExit(L"Error opening STDIN in read mode");
    }

    void closeHandles()
    {
        if (hStdOut_Wr != INVALID_HANDLE_VALUE)
            CloseHandle(hStdOut_Wr);
        if (hStdErr_Wr != INVALID_HANDLE_VALUE)
            CloseHandle(hStdErr_Wr);
        if (hStdIn_Rd != INVALID_HANDLE_VALUE)
            CloseHandle(hStdIn_Rd);
    }

    StdNamedPipes pipes;
    HANDLE        hStdOut_Wr;
    HANDLE        hStdErr_Wr;
    HANDLE        hStdIn_Rd;
};

struct Janitor
{
    Janitor()
    {
        DWORD pid = GetCurrentProcessId();
        data      = reinterpret_cast<Data*>(HeapAlloc(GetProcessHeap(), 0, sizeof(Data)));
        if (!data)
            ErrorExit(L"Failed to allocate memory for named pipes");

        std::construct_at(data, pid);
    }
    Janitor(Janitor const&)                = delete;
    Janitor(Janitor&&) noexcept            = delete;
    Janitor& operator=(Janitor const&)     = delete;
    Janitor& operator=(Janitor&&) noexcept = delete;
    ~Janitor() noexcept
    {
        if (data)
        {
            std::destroy_at(data);
            HeapFree(GetProcessHeap(), 0, data);
        }
    }

    Data* data;
};

DWORD WINAPI stdinPipeThread(void* params)
{
    wchar_t buffer[1024];
    auto*   data   = reinterpret_cast<Data*>(params);
    HANDLE  hStdIn = GetStdHandle(STD_INPUT_HANDLE);
    //if (!ConnectNamedPipe(data->pipes.hStdInPipe, nullptr))
    //    ErrorExit(L"Couldn't connect to named pipe hStdInPipe");
    while (true)
    {
        DWORD numBytes = 0;
        if (!ReadFile(hStdIn, buffer, 1024, &numBytes, nullptr))
            break;

        if (!WriteFile(data->pipes.hStdInPipe, buffer, numBytes, nullptr, nullptr))
            ExitThread(1);
    }

    return 0;
}

DWORD WINAPI stdoutPipeThread(void* params)
{
    wchar_t buffer[1024];
    auto*   data     = reinterpret_cast<Data*>(params);
    HANDLE  hStdOut  = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD   numBytes = 0;
    //if (!ConnectNamedPipe(data->pipes.hStdOutPipe, nullptr))
    //    ErrorExit(L"Couldn't connect to named pipe hStdOutPipe");

    while (true)
    {
        if (ReadFile(data->pipes.hStdOutPipe, buffer, 1024, &numBytes, nullptr))
            WriteFile(hStdOut, buffer, numBytes, nullptr, nullptr);
        else
            break;
    }
    return 0;
}
#endif

static std::unique_ptr<wchar_t[]> copyEnvironmentBlock()
{
    // String of variables of type L"name=value\0"
    //  A Unicode environment block is terminated by four zero bytes:
    //  two for the last string, two more to terminate the block.
    wchar_t* localEnv = GetEnvironmentStringsW();
    if (!localEnv)
        ErrorExit(L"Coudn't get environement variables");

    // Find the size of the environment block
    wchar_t* ptr  = localEnv;
    size_t   size = 0;
    while (*ptr != L'\0' || *(ptr + 1) != L'\0')
        ++ptr;
    size = (ptr - localEnv) + 2; // Include the final L"\0\0"

    // Allocate a unique_ptr to copy the environment block
    std::unique_ptr<wchar_t[]> envCopy(new wchar_t[size]);

    // Copy the environment block
    memcpy(envCopy.get(), localEnv, size * sizeof(wchar_t));

    // Free the original environment block
    FreeEnvironmentStringsW(localEnv);

    return envCopy;
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
    SECURITY_ATTRIBUTES saAttr;
    saAttr.nLength              = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle       = TRUE; // This makes pipe handles to be inherited
    saAttr.lpSecurityDescriptor = nullptr;

#if !defined(USE_NAMED_PIPES)
    Janitor j;

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
#else

    Janitor     j;
    STARTUPINFO startupInfo = {};
    startupInfo.cb          = sizeof(STARTUPINFO);
    startupInfo.dwFlags     = STARTF_USESTDHANDLES; // Use redirected stdin/stdout
    startupInfo.hStdError   = j.data->hStdErr_Wr;
    startupInfo.hStdOutput  = j.data->hStdOut_Wr;
    startupInfo.hStdInput   = j.data->hStdIn_Rd;
#endif
    DWORD const creationFlags = CREATE_SUSPENDED              // use `ResumeThread` to manally start
                                | CREATE_UNICODE_ENVIRONMENT; // lpEnvironment will use unicode characters

    auto pEnv = copyEnvironmentBlock();

    // Create process in suspended state
    if (!CreateProcessW(nullptr,
                        const_cast<LPWSTR>(commandLine.c_str()),
                        nullptr, // process security attributes
                        nullptr, // main thread security attributes
                        true,    // Inherit handles
                        creationFlags,
                        pEnv.get(),
                        nullptr, // same current directory as launcher
                        &startupInfo,
                        &processInfo))
        ErrorExit(L"ERROR: Failed to create process");

#if defined(USE_NAMED_PIPES)
    j.data->pipes.connectPipes();
#endif

    // Start IO Listener threads
    DWORD  tids[2];
    HANDLE hThreads[2];
#if !defined(USE_NAMED_PIPES)
    hThreads[0] = CreateThread(nullptr, 0, stdoutPipeThread, j.pipes, 0, &tids[0]);
#else
    hThreads[0]             = CreateThread(nullptr, 0, stdoutPipeThread, j.data, 0, &tids[0]);
#endif
    if (!hThreads[0])
        ErrorExit(L"Couldn't create stdout listener thread");
#if !defined(USE_NAMED_PIPES)
    hThreads[1] = CreateThread(nullptr, 0, stdinPipeThread, j.pipes, 0, &tids[1]);
#else
    hThreads[1]             = CreateThread(nullptr, 0, stdinPipeThread, j.data, 0, &tids[1]);
#endif
    if (!hThreads[1])
        ErrorExit(L"Couldn't create stdin listener thread");

    // Resume the suspended process
    ResumeThread(processInfo.hThread);

    // Wait for the process to complete
    WaitForSingleObject(processInfo.hProcess, INFINITE);

    // wait for io threads (for now this hangs)
    // WaitForMultipleObjects(2, hThreads, true, INFINITE);
    TerminateThread(hThreads[0], 0);
    TerminateThread(hThreads[1], 0);

#if defined(USE_NAMED_PIPES)
    j.data->closeHandles();
#endif

    // Retrieve the exit code
    DWORD exitCode;
    GetExitCodeProcess(processInfo.hProcess, &exitCode);

    // Clean up
    CloseHandle(processInfo.hThread);
    CloseHandle(processInfo.hProcess);

    return static_cast<int>(exitCode);
}


// Format a readable error message, display a message box,
// and exit from the application.
[[noreturn]] void ErrorExit(wchar_t const* lpszFunction)
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
