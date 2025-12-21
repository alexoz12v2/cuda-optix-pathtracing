#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <windows.h>
#include <strsafe.h>
#include <tlhelp32.h>
#include <shellapi.h>

#include <atomic>
#include <cassert>

[[noreturn]] void ErrorExit(wchar_t const* lpszFunction);

static constexpr nullptr_t defaultSecurityAttr = nullptr;

static std::atomic_bool s_terminate = false;

// TODO: Switch from console API to pseudoconsole
// https://learn.microsoft.com/en-us/windows/console/creating-a-pseudoconsole-session
// https://devblogs.microsoft.com/commandline/windows-command-line-introducing-the-windows-pseudo-console-conpty/

struct StdNamedPipes {
  StdNamedPipes(DWORD pid) {
    static constexpr DWORD CONNECTIMEOUT = 1000;
    static constexpr DWORD nameSize = 256;
    void* memory =
        HeapAlloc(GetProcessHeap(), 0, 3ULL * nameSize * sizeof(wchar_t));
    if (!memory)
      ErrorExit(L"Failed to allocate memory to store Names of the Pipes");
    nameStdOutPipe = reinterpret_cast<wchar_t*>(memory);
    nameStdInPipe = reinterpret_cast<wchar_t*>(memory) + nameSize;
    nameStdErrPipe = reinterpret_cast<wchar_t*>(memory) + 2ULL * nameSize;

    int32_t numBytes =
        _snwprintf(nameStdOutPipe, nameSize, L"\\\\.\\pipe\\%dcout", pid);
    hStdOutPipe = CreateNamedPipeW(
        nameStdOutPipe, PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED,
        // PIPE_ACCESS_INBOUND,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 2, 8192, 8192,
        CONNECTIMEOUT, defaultSecurityAttr);
    if (hStdOutPipe == INVALID_HANDLE_VALUE) {
      cleanup();
      ErrorExit(L"Failed To Create Named Pipe for STDOUT");
    }

    numBytes = _snwprintf(nameStdInPipe, nameSize, L"\\\\.\\pipe\\%dcin", pid);
    hStdInPipe = CreateNamedPipeW(
        nameStdInPipe, PIPE_ACCESS_OUTBOUND | FILE_FLAG_OVERLAPPED,
        // PIPE_ACCESS_OUTBOUND,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 2, 1024, 1024,
        CONNECTIMEOUT, defaultSecurityAttr);
    if (hStdInPipe == INVALID_HANDLE_VALUE) {
      cleanup();
      ErrorExit(L"Failed To Create Named Pipe for STDIN");
    }

    numBytes =
        _snwprintf(nameStdErrPipe, nameSize, L"\\\\.\\pipe\\%dcerr", pid);
    hStdErrPipe = CreateNamedPipeW(
        nameStdErrPipe, PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED,
        // PIPE_ACCESS_INBOUND,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 2, 8192, 8192,
        CONNECTIMEOUT, defaultSecurityAttr);

    if (hStdErrPipe == INVALID_HANDLE_VALUE) {
      cleanup();
      ErrorExit(L"Failed To Create Named Pipe for STDERR");
    }

    if (!SetEnvironmentVariableW(L"CHILD_CONOUT", nameStdOutPipe) ||
        !SetEnvironmentVariableW(L"CHILD_CONIN", nameStdInPipe) ||
        !SetEnvironmentVariableW(L"CHILD_CONERR", nameStdErrPipe)) {
      cleanup();
      ErrorExit(L"Couldn't set environment variables");
    }
  }
  StdNamedPipes(StdNamedPipes const&) = delete;
  StdNamedPipes(StdNamedPipes&&) noexcept = delete;
  StdNamedPipes& operator=(StdNamedPipes const&) = delete;
  StdNamedPipes& operator=(StdNamedPipes&&) noexcept = delete;
  ~StdNamedPipes() noexcept { cleanup(); }

  void cleanup() noexcept {
    Sleep(100);  // Give pipes a chance to flush
    if (hStdOutPipe != INVALID_HANDLE_VALUE) CloseHandle(hStdOutPipe);
    if (hStdInPipe != INVALID_HANDLE_VALUE) CloseHandle(hStdInPipe);
    if (hStdErrPipe != INVALID_HANDLE_VALUE) CloseHandle(hStdErrPipe);
    if (nameStdOutPipe) HeapFree(GetProcessHeap(), 0, nameStdOutPipe);
  }

  bool connectPipes() {
    return ConnectNamedPipe(hStdOutPipe, nullptr) &&
           ConnectNamedPipe(hStdInPipe, nullptr) &&
           ConnectNamedPipe(hStdErrPipe, nullptr);
  }

  void disconnectPipes() {
    DisconnectNamedPipe(hStdOutPipe);
    DisconnectNamedPipe(hStdInPipe);
    DisconnectNamedPipe(hStdErrPipe);
  }

  wchar_t* nameStdOutPipe;
  wchar_t* nameStdInPipe;
  wchar_t* nameStdErrPipe;
  HANDLE hStdOutPipe = INVALID_HANDLE_VALUE;
  HANDLE hStdInPipe = INVALID_HANDLE_VALUE;
  HANDLE hStdErrPipe = INVALID_HANDLE_VALUE;
};

struct Data {
  Data(DWORD pid)
      : pipes(pid),
        hStdOut_Wr(INVALID_HANDLE_VALUE),
        hStdErr_Wr(INVALID_HANDLE_VALUE),
        hStdIn_Rd(INVALID_HANDLE_VALUE) {
    createFiles(pid);
    handleClosed = false;
    // after construction `connectPipes` should be called
  }
  Data(Data const&) = delete;
  Data(Data&&) noexcept = delete;
  Data& operator=(Data const&) = delete;
  Data& operator=(Data&&) noexcept = delete;
  ~Data() {
    closeHandles();
    pipes.disconnectPipes();
  }

  void createFiles(DWORD pid) {
    SECURITY_ATTRIBUTES saAttr;
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = true;
    saAttr.lpSecurityDescriptor = nullptr;
    hStdOut_Wr = CreateFileW(pipes.nameStdOutPipe, GENERIC_WRITE, 0, &saAttr,
                             OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
    if (hStdOut_Wr == INVALID_HANDLE_VALUE)
      ErrorExit(L"Error opening STDOUT in write mode");
    hStdErr_Wr = CreateFileW(pipes.nameStdErrPipe, GENERIC_WRITE, 0, &saAttr,
                             OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
    if (hStdErr_Wr == INVALID_HANDLE_VALUE)
      ErrorExit(L"Error opening STDERR in write mode");
    hStdIn_Rd = CreateFileW(pipes.nameStdInPipe, GENERIC_READ, 0, &saAttr,
                            OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
    if (hStdIn_Rd == INVALID_HANDLE_VALUE)
      ErrorExit(L"Error opening STDIN in read mode");

    mailslotName = reinterpret_cast<wchar_t*>(
        HeapAlloc(GetProcessHeap(), 0, 256 * sizeof(wchar_t)));
    if (!mailslotName)
      ErrorExit(L"Couldn't allocate nemory to store the name of the mailbox");
    int numChars =
        _snwprintf(mailslotName, 256, L"\\\\.\\mailslot\\%dmailslot", pid);
    mailslotName[numChars >= 256 ? 255 : numChars] = L'\0';
    static DWORD constexpr noMaxMsgSize = 0;
    hMailslot = CreateMailslotW(mailslotName, noMaxMsgSize,
                                MAILSLOT_WAIT_FOREVER, &saAttr);
    if (hMailslot == INVALID_HANDLE_VALUE)
      ErrorExit(L"Couldn't create mailbox");
    // advertise to child process through the environment
    if (!SetEnvironmentVariableW(L"CHILD_MAILBOX", mailslotName))
      ErrorExit(L"Couldn't set environment variable `CHILD_MAILBOX`");

    // create mutex to protect mailbox
    mutexName = reinterpret_cast<wchar_t*>(
        HeapAlloc(GetProcessHeap(), 0, 256 * sizeof(wchar_t)));
    if (!mutexName) ErrorExit(L"Coutdn't allocate mmeory for Mutex Name");
    numChars = _snwprintf(mutexName, 256, L"%d::Mutex", pid);
    mutexName[numChars >= 256 ? 255 : numChars] = L'\0';
    hMutex = CreateMutexW(defaultSecurityAttr, false /*no initial owner*/,
                          mutexName);
    if (!hMutex) ErrorExit(L"CreateMutexW");
    if (!SetEnvironmentVariableW(L"CHILD_MUTEX", mutexName))
      ErrorExit(L"SetEnvironmentVariableW CHILD_MUTEX");
  }

  void closeHandles() {
    if (!handleClosed) {
      if (hStdOut_Wr != INVALID_HANDLE_VALUE) CloseHandle(hStdOut_Wr);
      if (hStdErr_Wr != INVALID_HANDLE_VALUE) CloseHandle(hStdErr_Wr);
      if (hStdIn_Rd != INVALID_HANDLE_VALUE) CloseHandle(hStdIn_Rd);
      if (hMailslot != INVALID_HANDLE_VALUE) CloseHandle(hMailslot);
      handleClosed = true;
    }
  }

  StdNamedPipes pipes;
  HANDLE hStdOut_Wr;
  HANDLE hStdErr_Wr;
  HANDLE hStdIn_Rd;
  HANDLE hMailslot = INVALID_HANDLE_VALUE;
  HANDLE hMutex = nullptr;
  wchar_t* mutexName = nullptr;
  wchar_t* mailslotName = nullptr;
  bool handleClosed = true;
};

struct Janitor {
  Janitor() {
    DWORD pid = GetCurrentProcessId();
    data =
        reinterpret_cast<Data*>(HeapAlloc(GetProcessHeap(), 0, sizeof(Data)));
    if (!data) ErrorExit(L"Failed to allocate memory for named pipes");

    std::construct_at(data, pid);
  }
  Janitor(Janitor const&) = delete;
  Janitor(Janitor&&) noexcept = delete;
  Janitor& operator=(Janitor const&) = delete;
  Janitor& operator=(Janitor&&) noexcept = delete;
  ~Janitor() noexcept {
    if (data) {
      std::destroy_at(data);
      HeapFree(GetProcessHeap(), 0, data);
    }
  }

  Data* data;
};

DWORD WINAPI stdinPipeThread(void* params) {
  wchar_t buffer[1024];
  auto* data = reinterpret_cast<Data*>(params);
  HANDLE hStdIn = CreateFileW(L"CONIN$", GENERIC_READ, FILE_SHARE_WRITE,
                              nullptr, OPEN_EXISTING, FILE_FLAG_OVERLAPPED,
                              nullptr);  // GetStdHandle(STD_INPUT_HANDLE);
  if (hStdIn == INVALID_HANDLE_VALUE) ErrorExit(L"CreateFileW");
  OVERLAPPED ovs{};
  ovs.hEvent = CreateEventW(nullptr, false, true, nullptr);
  if (!ovs.hEvent) ErrorExit(L"Couldn't create Event");
  OVERLAPPED ovi{};
  ovi.hEvent = CreateEventW(nullptr, false, true, nullptr);
  if (!ovi.hEvent) ErrorExit(L"CreateEventW");

  while (!s_terminate.load()) {
    DWORD numBytes = 0;
    bool completed = ReadFile(hStdIn, buffer, 1024, &numBytes, &ovs);
    if (!completed) {
      if (DWORD err = GetLastError(); err != ERROR_IO_PENDING) {
        SetLastError(err);
        ErrorExit(L"ReadFile OVERLAPPED");
      } else {
        if (!WaitForSingleObject(ovs.hEvent, 100)) continue;
      }
    }

    completed =
        WriteFile(data->pipes.hStdInPipe, buffer, numBytes, nullptr, &ovi);
    if (!completed) {
      if (!GetOverlappedResult(data->pipes.hStdInPipe, &ovi, &numBytes, true))
        ErrorExit(L"GetOverlappedResult StdIn");
    }
  }

  CloseHandle(ovs.hEvent);
  CloseHandle(ovi.hEvent);

  return 0;
}

static uint32_t countNulTerminators(wchar_t const* str, uint32_t byteLength) {
  // Calculate the number of wchar_t characters in the string
  uint32_t charCount = byteLength / sizeof(wchar_t);
  uint32_t nullCount = 0;

  // Iterate through the string and count null terminators
  for (size_t i = 0; i < charCount; ++i) {
    if (str[i] == L'\0') ++nullCount;
  }

  return nullCount;
}

struct MutexJanitor {
  MutexJanitor(HANDLE _hMutex) : hMutex(_hMutex) {}
  MutexJanitor(MutexJanitor const&) = delete;
  MutexJanitor(MutexJanitor&&) noexcept = delete;
  MutexJanitor& operator=(MutexJanitor const&) = delete;
  MutexJanitor& operator=(MutexJanitor&&) noexcept = delete;
  ~MutexJanitor() {
    if (hMutex && hMutex != INVALID_HANDLE_VALUE) ReleaseMutex(hMutex);
  }
  HANDLE hMutex;
};

static constexpr WORD consoleColorMask =
    FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY;
static constexpr WORD white =
    FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE;
static constexpr WORD red = FOREGROUND_RED | FOREGROUND_INTENSITY;
static constexpr WORD yellow =
    FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY;
static constexpr WORD olive =
    FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY;

DWORD WINAPI stdoutPipeThread(void* params) {
  static constexpr uint32_t BufferBytes = 8192;
  std::unique_ptr<wchar_t[]> buffer = std::make_unique<wchar_t[]>(BufferBytes);
  auto* data = reinterpret_cast<Data*>(params);
  HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD numBytes = 0;
  // https://learn.microsoft.com/en-us/windows/console/console-screen-buffers#span-idwin32characterattributesspanspan-idwin32characterattributesspancharacter-attributes
  CONSOLE_SCREEN_BUFFER_INFO consoleInfo{};
  if (!GetConsoleScreenBufferInfo(hStdOut, &consoleInfo))
    ErrorExit(L"Couldn't get Console screen buffer info");
  OVERLAPPED ovs{};
  ovs.hEvent = CreateEventW(nullptr, false, true, nullptr);
  if (!ovs.hEvent) ErrorExit(L"fadsjkfads");

  bool last = true;
  while (true) {
    // acquire mutex for mailbox
    DWORD const waitResult = WaitForSingleObject(data->hMutex, INFINITE);
    if (waitResult != WAIT_OBJECT_0) ErrorExit(L"WaitForSingleObject Mutex");
    MutexJanitor jMutex{data->hMutex};

    DWORD numBytesNextMsg = 0, numMsgs = 0;
    if (!GetMailslotInfo(data->hMailslot, nullptr, &numBytesNextMsg, &numMsgs,
                         nullptr))
      ErrorExit(L"Failed GetMailslotInfo");

    if (s_terminate.load()) {
      if (!last && numBytesNextMsg == MAILSLOT_NO_MESSAGE) break;
      last = false;
    }
    if (numBytesNextMsg == MAILSLOT_NO_MESSAGE) continue;

    // each message should end with L'\0'
    bool completed = ReadFile(data->pipes.hStdOutPipe, buffer.get(),
                              BufferBytes, &numBytes, &ovs);
    if (!completed) {
      if (auto err = GetLastError(); err != ERROR_IO_PENDING) {
        SetLastError(err);
        ErrorExit(L"Error while reading pipe");
      } else {
        if (!GetOverlappedResult(data->pipes.hStdOutPipe, &ovs, &numBytes,
                                 true))
          ErrorExit(L"GetOverlappedResult");
      }
    }

#if 0
        wchar_t const* buf        = buffer.get();
        uint32_t       numStrings = countNulTerminators(buffer.get(), numBytes);
        while (numMsgs != 0 && numStrings != 0)
        {
            --numStrings;
            // fetch current message data (level)
            uint8_t c = 0;
            assert(numBytesNextMsg == sizeof(uint8_t));
            if (!ReadFile(data->hMailslot, &c, sizeof(uint8_t), nullptr, nullptr))
                ErrorExit(L"ReadFile Mailbos");

            // Write current message
            FlushFileBuffers(hStdOut);
            WORD const textAttribute = [attrs = consoleInfo.wAttributes, c = c]() -> WORD {
                switch (c)
                {
                    case 0: return (attrs & ~consoleColorMask) | olive;
                    case 1: return (attrs & ~consoleColorMask) | white;
                    case 2: return (attrs & ~consoleColorMask) | yellow;
                    case 3: return (attrs & ~consoleColorMask) | red;
                }
                return 0;
            }();
            if (!textAttribute) // todo remove
                continue;
            if (!SetConsoleTextAttribute(hStdOut, textAttribute))
                ErrorExit(L"Couldn't set console attributes to the proper color");

            // estimate message length by finding L'\0'
            DWORD const numCharsCurrent = static_cast<DWORD>(wcsnlen_s(buf, numBytes));

            WriteConsoleW(hStdOut, buf, numCharsCurrent, nullptr, nullptr);
            //WriteFile(hStdOut, buf, numCharsCurrent << 1, nullptr, nullptr);
            buf += numCharsCurrent + 1;

            // fetch next message
            if (!GetMailslotInfo(data->hMailslot, nullptr, &numBytesNextMsg, &numMsgs, nullptr))
                ErrorExit(L"Failed GetMailslotInfo");
        }
#else
    wchar_t* buf = buffer.get();
    assert((numMsgs == 0 || numMsgs == 1) &&
           "Too many messages in stdout pipe mailbox");
    if (numMsgs != 0) {
      // fetch current message data (level)
      uint8_t c = 0;
      assert(numBytesNextMsg == sizeof(uint8_t));
      if (!ReadFile(data->hMailslot, &c, sizeof(uint8_t), nullptr, nullptr))
        ErrorExit(L"ReadFile Mailbos");

      // Write current message
      FlushFileBuffers(hStdOut);
      WORD const textAttribute = [attrs = consoleInfo.wAttributes,
                                  c = c]() -> WORD {
        switch (c) {
          case 0:
            return (attrs & ~consoleColorMask) | olive;
          case 1:
            return (attrs & ~consoleColorMask) | white;
          case 2:
            return (attrs & ~consoleColorMask) | yellow;
          case 3:
            return (attrs & ~consoleColorMask) | red;
        }
        return 0;
      }();
      if (!textAttribute)  // todo remove
        continue;
      if (!SetConsoleTextAttribute(hStdOut, textAttribute))
        ErrorExit(L"Couldn't set console attributes to the proper color");

      // estimate message length by finding L'\0'
      DWORD numCharsCurrent = static_cast<DWORD>(wcsnlen_s(buf, numBytes));

      // Ensure the last visible character is L'\n'
      if (numCharsCurrent > 0 && buf[numCharsCurrent - 1] != L'\n') {
        // If there's space, insert L'\n' before null terminator
        if (numCharsCurrent + 1 < numBytes)  // +1 for '\n', +1 for '\0'
        {
          buf[numCharsCurrent] = L'\n';
          buf[numCharsCurrent + 1] = L'\0';
          ++numCharsCurrent;  // account for added newline
        } else {
          // Handle error or truncation case if no space to insert '\n'
          ErrorExit(L"Message too long to insert newline");
        }
      }

      WriteConsoleW(hStdOut, buf, numCharsCurrent, nullptr, nullptr);
      // WriteFile(hStdOut, buf, numCharsCurrent << 1, nullptr, nullptr);
      buf += numCharsCurrent + 1;

      // fetch next message
      if (!GetMailslotInfo(data->hMailslot, nullptr, &numBytesNextMsg, &numMsgs,
                           nullptr))
        ErrorExit(L"Failed GetMailslotInfo");
    }
#endif
  }

  if (!SetConsoleTextAttribute(
          hStdOut, (consoleInfo.wAttributes & ~consoleColorMask) | white))
    ErrorExit(L"SetConsoleTextAttribute to original");

  CloseHandle(ovs.hEvent);

  return 0;
}

// Helper function to get child processes of a parent process
// https://learn.microsoft.com/en-us/windows/win32/procthread/process-enumeration
std::vector<DWORD> GetChildProcesses(DWORD parentPID) {
  std::vector<DWORD> childPIDs;
  childPIDs.reserve(32);

  // Take a snapshot of all processes
  HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (hSnapshot == INVALID_HANDLE_VALUE) {
    return childPIDs;
  }

  PROCESSENTRY32 pe;
  pe.dwSize = sizeof(PROCESSENTRY32);

  if (Process32First(hSnapshot, &pe)) {
    do {
      if (pe.th32ParentProcessID == parentPID) {
        childPIDs.push_back(pe.th32ProcessID);
      }
    } while (Process32Next(hSnapshot, &pe));
  }

  CloseHandle(hSnapshot);
  return childPIDs;
}

// Function to terminate all child processes
void TerminateChildProcesses(DWORD parentPID) {
  std::vector<DWORD> childPIDs = GetChildProcesses(parentPID);
  std::unique_ptr<wchar_t[]> buffer = std::make_unique<wchar_t[]>(256);

  for (DWORD pid : childPIDs) {
    HANDLE hProcess = OpenProcess(PROCESS_TERMINATE, FALSE, pid);
    if (hProcess != NULL) {
      int32_t const cc = _snwprintf(
          buffer.get(), 256, L"Terminating child process with PID: %5X\n", pid);
      WriteConsoleW(GetStdHandle(STD_OUTPUT_HANDLE), buffer.get(), cc, nullptr,
                    nullptr);
      TerminateProcess(hProcess, 1);  // Terminate the child process
      CloseHandle(hProcess);
    } else {
      int32_t const cc = _snwprintf(
          buffer.get(), 256, L"Failed to open process with PID: %5X", pid);
      WriteConsoleW(GetStdHandle(STD_ERROR_HANDLE), buffer.get(), cc, nullptr,
                    nullptr);
    }
  }
}

static CONSOLE_SCREEN_BUFFER_INFO s_consoleInfo{};

BOOL WINAPI ctrlHandler(DWORD fdwCtrlType) {
  if (!SetConsoleTextAttribute(
          GetStdHandle(STD_OUTPUT_HANDLE),
          (s_consoleInfo.wAttributes & ~consoleColorMask) | white))
    ErrorExit(L"SetConsoleTextAttribute to original");
  std::wcout << L"CONSOLE HANDLER RUNNING" << std::endl;
  switch (fdwCtrlType) {
    case CTRL_C_EVENT:
      printf("Ctrl-C event\n\n");
      TerminateChildProcesses(GetCurrentProcessId());
      return TRUE;

    case CTRL_CLOSE_EVENT:
      printf("Ctrl-Close event\n\n");
      TerminateChildProcesses(GetCurrentProcessId());
      return TRUE;

    case CTRL_BREAK_EVENT:
      printf("Ctrl-Break event\n\n");
      TerminateChildProcesses(GetCurrentProcessId());
      return FALSE;

    case CTRL_LOGOFF_EVENT:
      printf("Ctrl-Logoff event\n\n");
      TerminateChildProcesses(GetCurrentProcessId());
      return FALSE;

    case CTRL_SHUTDOWN_EVENT:
      printf("Ctrl-Shutdown event\n\n");
      TerminateChildProcesses(GetCurrentProcessId());
      return FALSE;

    default:
      return FALSE;
  }
}

LONG WINAPI TopLevelExceptionHandler(EXCEPTION_POINTERS* ExceptionInfo) {
  s_terminate.store(true);
  TerminateChildProcesses(GetCurrentProcessId());
  // SetEvent(hShutdownEvent);
  // RestoreConsole();
  return EXCEPTION_EXECUTE_HANDLER;
}

static std::unique_ptr<wchar_t[]> copyEnvironmentBlock() {
  // String of variables of type L"name=value\0"
  //  A Unicode environment block is terminated by four zero bytes:
  //  two for the last string, two more to terminate the block.
  wchar_t* localEnv = GetEnvironmentStringsW();
  if (!localEnv) ErrorExit(L"Coudn't get environement variables");

  // Find the size of the environment block
  wchar_t* ptr = localEnv;
  size_t size = 0;
  while (*ptr != L'\0' || *(ptr + 1) != L'\0') ++ptr;
  size = (ptr - localEnv) + 2;  // Include the final L"\0\0"

  // Allocate a unique_ptr to copy the environment block
  std::unique_ptr<wchar_t[]> envCopy(new wchar_t[size]);

  // Copy the environment block
  memcpy(envCopy.get(), localEnv, size * sizeof(wchar_t));

  // Free the original environment block
  FreeEnvironmentStringsW(localEnv);

  return envCopy;
}

int main() {
  if (!SetConsoleCP(CP_UTF8)) ErrorExit(L"SetConsoleCP");
  if (!SetConsoleOutputCP(CP_UTF8)) ErrorExit(L"SetConsoleOutputCP");
// #define test_UNICODE_PRINT
#if defined(test_UNICODE_PRINT)
  // apparently, write file uses the current code point of the console (UTF-8),
  // while write console uses the usual UTF-16 LE the code is already written to
  // convert to UTF-16, so I don't care
  wchar_t const emoji[2] = {0xD83D, 0xDE0A};
  char8_t const utf8[4] = {0xF0, 0x9F, 0x98, 0x8A};
  WriteConsoleW(GetStdHandle(STD_OUTPUT_HANDLE), &emoji[0], 2, nullptr,
                nullptr);
  WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), &utf8[0], 4, nullptr, nullptr);
#endif
  SetUnhandledExceptionFilter(TopLevelExceptionHandler);

  static constexpr uint32_t BUFSIZE = 4096;
  TCHAR moduleName[MAX_PATH];
  GetModuleFileName(nullptr, moduleName, MAX_PATH);

  // recover default console screen buffer values
  if (!GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE),
                                  &s_consoleInfo))
    ErrorExit(L"Couldn't get Console screen buffer info");

  // Convert `.com` to `.exe`
  std::wstring modulePath(moduleName);
  size_t dotComPos = modulePath.find_last_of(L'.');
  if (dotComPos == std::wstring::npos ||
      modulePath.substr(dotComPos) != L".com") {
    std::wcerr << L"ERROR: Launcher was not invoked as a .com file."
               << std::endl;
    return 1;
  }
  modulePath.replace(dotComPos, 4, L".exe");  // Replace `.com` with `.exe`

  // create job object
  // Create job object to handle termination
  HANDLE hJob = CreateJobObject(nullptr, nullptr);
  if (hJob == nullptr) {
    ErrorExit(L"Failed to create job object.");
  }

  // Set the job object to terminate all processes when the parent terminates
  JOBOBJECT_BASIC_LIMIT_INFORMATION jobLimit = {};
  jobLimit.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION jobExtLimit = {};
  jobExtLimit.BasicLimitInformation = jobLimit;

  if (!SetInformationJobObject(hJob, JobObjectExtendedLimitInformation,
                               &jobExtLimit, sizeof(jobExtLimit))) {
    ErrorExit(L"Failed to set job object information.");
  }

  // Prepare command line
  std::wstring commandLine = L"\"" + modulePath + L"\" ";
  int32_t numArgs = 0;
  wchar_t** args =
      CommandLineToArgvW(GetCommandLineW(), &numArgs);  // Append all arguments
  for (int32_t i = 1; i < numArgs; ++i) {
    commandLine += L' ';
    commandLine += args[i];
  }

  PROCESS_INFORMATION processInfo = {0};
  SECURITY_ATTRIBUTES saAttr;
  saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
  saAttr.bInheritHandle = TRUE;  // This makes pipe handles to be inherited
  saAttr.lpSecurityDescriptor = nullptr;

  Janitor j;
  STARTUPINFO startupInfo = {};
  startupInfo.cb = sizeof(STARTUPINFO);
  startupInfo.dwFlags = STARTF_USESTDHANDLES;  // Use redirected stdin/stdout
  startupInfo.hStdError = j.data->hStdErr_Wr;
  startupInfo.hStdOutput = j.data->hStdOut_Wr;
  startupInfo.hStdInput = j.data->hStdIn_Rd;
  DWORD const creationFlags =
      CREATE_SUSPENDED  // use `ResumeThread` to manally start
      |
      CREATE_UNICODE_ENVIRONMENT;  // lpEnvironment will use unicode characters

  // this must be after named pipes creation, as we are passing it with the
  // environment the names of the pipes
  auto pEnv = copyEnvironmentBlock();

  // Create process in suspended state
  if (!CreateProcessW(nullptr, const_cast<LPWSTR>(commandLine.c_str()),
                      nullptr,  // process security attributes
                      nullptr,  // main thread security attributes
                      true,     // Inherit handles
                      creationFlags, pEnv.get(),
                      nullptr,  // same current directory as launcher
                      &startupInfo, &processInfo))
    ErrorExit(L"ERROR: Failed to create process");

  // Assign child process to job object
  if (!AssignProcessToJobObject(hJob, processInfo.hProcess)) {
    ErrorExit(L"Failed to assign process to job object.");
  }

  atexit([]() { TerminateChildProcesses(GetCurrentProcessId()); });

  j.data->pipes.connectPipes();

  if (!SetConsoleCtrlHandler(ctrlHandler, true))
    ErrorExit(L"SetConsoleCtrlHandler");

  // Start IO Listener threads
  DWORD tids[2];
  HANDLE hThreads[2];
  hThreads[0] = CreateThread(nullptr, 0, stdoutPipeThread, j.data, 0, &tids[0]);
  if (!hThreads[0]) ErrorExit(L"Couldn't create stdout listener thread");
  hThreads[1] = CreateThread(nullptr, 0, stdinPipeThread, j.data, 0, &tids[1]);
  if (!hThreads[1]) ErrorExit(L"Couldn't create stdin listener thread");

  // Resume the suspended process
  ResumeThread(processInfo.hThread);

  // Wait for the process to complete
  WaitForSingleObject(processInfo.hProcess, INFINITE);

  // wait for io threads (for now this hangs)
  s_terminate.store(true);
  if (WaitForMultipleObjects(2, hThreads, false, 100)) {
    TerminateThread(hThreads[0], 0);
    TerminateThread(hThreads[1], 0);
  }

  j.data->closeHandles();

  // Retrieve the exit code
  DWORD exitCode;
  GetExitCodeProcess(processInfo.hProcess, &exitCode);

  // Clean up
  CloseHandle(processInfo.hThread);
  CloseHandle(processInfo.hProcess);
  CloseHandle(hJob);  // Close job object handle

  return static_cast<int>(exitCode);
}

// TO REMOVE
// Format a readable error message, display a message box,
// and exit from the application.
[[noreturn]] void ErrorExit(wchar_t const* lpszFunction) {
  wchar_t* lpMsgBuf;
  wchar_t* lpDisplayBuf;
  DWORD dw = GetLastError();

  FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                reinterpret_cast<wchar_t*>(&lpMsgBuf), 0, NULL);

  DWORD const numBytes =
      (wcslen(lpMsgBuf) + wcslen(lpszFunction) + 40) * sizeof(wchar_t);
  lpDisplayBuf =
      reinterpret_cast<wchar_t*>(LocalAlloc(LMEM_ZEROINIT, numBytes));

  StringCchPrintf(lpDisplayBuf, LocalSize(lpDisplayBuf) / sizeof(wchar_t),
                  L"%s failed with error %d: %s", lpszFunction, dw, lpMsgBuf);
  MessageBox(NULL, lpDisplayBuf, L"Error", MB_OK);

  LocalFree(lpMsgBuf);
  LocalFree(lpDisplayBuf);
  ExitProcess(1);
}

// TODO: restore the original Terminal color at exit
// TODO: Rework to not use `ExitProcess` and similiar, because it skips
// destructors