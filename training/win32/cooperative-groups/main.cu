// std stuff
#include <iostream>

// Windows Stuff
#include "Windows.h"
#include "ShlObj.h"

// cuda stuff
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define ANSI_RED "\033[31m"
#define ANSI_YLW "\033[93m"
#define ANSI_RST "\033[0m"

int wmain()
{
    // - Setup console properly such that ANSI escape codes work
    for (HANDLE out : {GetStdHandle(STD_OUTPUT_HANDLE), GetStdHandle(STD_ERROR_HANDLE)})
    {
        DWORD mode = 0;
        GetConsoleMode(out, &mode);
        mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        mode |= DISABLE_NEWLINE_AUTO_RETURN;
        SetConsoleMode(out, mode);
    }
    std::ios::sync_with_stdio();

    // - Print some colored stuff
    std::cout << ANSI_RED "Hello Beautiful World" ANSI_RST << std::endl;
}
