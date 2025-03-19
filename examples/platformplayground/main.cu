#define DMT_ENTRY_POINT
#include "platform/platform.h"
#include "platform/platform-logging-default-formatters.h"

// windwoos only
#include <Psapi.h>
#include <iostream>

static uint32_t utf8FromUtf16le(wchar_t const* wideStr, char* output)
{
    if (!wideStr)
        return 0;

    // Get required buffer size
    DWORD utf8Size = WideCharToMultiByte(CP_UTF8, 0, wideStr, -1, NULL, 0, NULL, NULL);
    if (output == nullptr)
        return utf8Size;
    else // Convert to UTF-8
    {
        // TODO better
        if (WideCharToMultiByte(CP_UTF8, 0, wideStr, -1, output, utf8Size, NULL, NULL) == 0)
            return 0;

        return 0;
    }
}


namespace dmt {
    void listLoadedDLLs()
    {
        HANDLE       hProcess = GetCurrentProcess();
        HMODULE      hMods[1024];
        DWORD        needed = 0;
        dmt::Context ctx;

        if (EnumProcessModules(hProcess, hMods, sizeof(hMods), &needed))
        {
            wchar_t* modName  = new wchar_t[1024];
            char*    uModName = new char[1024];
            for (int32_t i = 0; i < (needed / sizeof(HMODULE)); ++i)
            {
                if (DWORD len = GetModuleFileNameExW(hProcess, hMods[i], modName, 1024); len > 0)
                {
                    utf8FromUtf16le(modName, uModName);
                    ctx.log("Loaded DLL: {}", std::make_tuple(uModName));
                }
            }
            delete[] modName;
            delete[] uModName;
        }
    }
} // namespace dmt

int32_t guardedMain()
{
    using namespace std::string_view_literals;
    auto res = dmt::ctx::addContext(true);
    if (res != dmt::ctx::ECtxReturn::eCreatedOnManaged)
        std::abort();
    dmt::ctx::cs->setActive(0);

    dmt::Context ctx;
    ctx.impl()->addHandler([](dmt::LogHandler& _out) { dmt::createConsoleHandler(_out); });
    ctx.log("Hello World", {});
    dmt::listLoadedDLLs();

    return 0;
}
