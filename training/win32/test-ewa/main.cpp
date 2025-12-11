#define DMT_ENTRY_POINT
#define DMT_WINDOWS_CLI

#include "platform-launch.h"
#include "platform-context.h"
#include "platform.h"
#include "core-texture.h"
#include "core-parser.h"

// library
#include <cstdint>
#include <stdexcept>
#include <array>
#include <string>

// windows
#include <Windows.h>

using namespace dmt;

// Quantization + Normalization gives the best result on a single sample
// How to apply to EWA Filtering + Lerp on 2 Selected Mip Levels?
// -> Apply to the final, filtered normal, because quantization/normalization breaks linearity

namespace {
    void execute()
    {
        Context ctx;

        // open a normal texture file
        auto const thePath = os::Path::executableDir() / "Paint_Chipped_1K_normal.png";
        if (!thePath.isValid() || !thePath.isFile())
            throw std::exception("Image file doesn't exist");
        ImageTexturev2 textureObject{};
        if (auto const res = parse_helpers::tempTexObj(thePath, true, &textureObject);
            res != parse_helpers::TempTexObjResult::eOk)
            throw std::exception(parse_helpers::tempTexObjResultToString(res).data());
        class ImageJanitor
        {
        public:
            explicit ImageJanitor(ImageTexturev2& theObj) : m_theObj(theObj) {}
            ~ImageJanitor() { parse_helpers::freeTempTexObj(m_theObj); }

        private:
            ImageTexturev2& m_theObj;
        } const j{textureObject};

        // the texture should contain a 0,0,1 vector in these coordinates at mip 0
        constexpr uint32_t row = 542;
        constexpr uint32_t col = 425;
        // convert to OpenGL Style Normal
        textureObject.isNormal = true;
        Vector3f const normal                    = textureObject.at(col, row, 0).asVec();
        Vector3f const normalQuantized           = map(normal, [](float const f) { return fl::quantize(f, 4); });
        Vector3f const normalNormalized          = normalize(normal);
        Vector3f const normalQuantizedNormalized = normalize(normalQuantized);
        ctx.log("OpenGL Style Normal Read: {} {} {}", std::make_tuple(normal.x, normal.y, normal.z));
        ctx.log("Normalized: {} {} {}", std::make_tuple(normalNormalized.x, normalNormalized.y, normalNormalized.z));
        ctx.log("Quantized: {} {} {}", std::make_tuple(normalQuantized.x, normalQuantized.y, normalQuantized.z));
        ctx.log("Quantized Normalized: {} {} {}",
                std::make_tuple(normalQuantizedNormalized.x, normalQuantizedNormalized.y, normalQuantizedNormalized.z));
        if (!fl::nearZero(normal.x) || !fl::nearZero(normal.y) || !fl::near(normal.z, 1.f))
            throw std::exception("Unexpected normal");
    }
} // namespace

int32_t guardedMain()
{
    // usual windows preamble for console apps
    {
        SetConsoleCP(CP_UTF8);
        SetConsoleOutputCP(CP_UTF8);
        std::array<DWORD, 2> const fds{STD_OUTPUT_HANDLE, STD_ERROR_HANDLE};
        for (DWORD const fd : fds)
        {
            DWORD consoleMode = 0;
            if (HANDLE const handle = GetStdHandle(fd);
                handle != INVALID_HANDLE_VALUE && GetConsoleMode(handle, &consoleMode))
            {
                consoleMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
                SetConsoleMode(handle, consoleMode);
            }
        }
    }

    Ctx::init();
    struct CtxJanitor
    {
        ~CtxJanitor() { Ctx::destroy(); }
    } const j;

    try
    {
        execute();
        return 0;
    } catch (...)
    {
        try
        {
            if (std::current_exception() != nullptr)
                std::rethrow_exception(std::current_exception());
        } catch (std::exception const& e)
        {
            std::cerr << e.what() << std::endl;
        }
        return 1;
    }
}