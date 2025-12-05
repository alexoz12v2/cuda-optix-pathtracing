#define DMT_ENTRY_POINT
#define DMT_WINDOWS_CLI
#include "platform.h"

#include "core-render.h"
#include "core-parser.h"

#include "CLIManager.h"
#include "platform-math.h"

#include <memory_resource>
#include <string>
#include <regex>

using namespace dmt;

namespace /*static*/ {

    void printNothing()
    {
        Context ctx;
        ctx.log("No Input file given. See --help for more information\n", {});
    }

    bool isValidPathComponent(std::string_view name, bool isLast)
    {
#ifdef DMT_OS_WINDOWS
        // Forbidden characters
        static constexpr std::string_view badChars = R"(< > : " / \ | ? *)";

        if (name.find_first_of(badChars) != std::string_view::npos)
            return false;

        // Control characters
        for (size_t i = 0; i < name.size(); i++)
            if (name[i] < 32)
                return false;

        // Cannot end with space or dot
        if (!name.empty() && (name.back() == ' ' || (name.back() == '.' && isLast)))
            return false;

        // Extract basename (before dot)
        auto             dotPos = name.find('.');
        std::string_view base   = (dotPos == std::string_view::npos) ? name : name.substr(0, dotPos);

        // Reserved names
        static constexpr std::string_view reserved[] = {"CON",  "PRN",  "AUX",  "NUL",  "COM1", "COM2", "COM3", "COM4",
                                                        "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3",
                                                        "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"};

        for (auto r : reserved)
        {
            if (base.size() == r.size())
            {
                bool match = true;
                for (size_t i = 0; i < r.size(); ++i)
                {
                    if (std::toupper(static_cast<unsigned char>(base[i])) != r[i])
                    {
                        match = false;
                        break;
                    }
                }
                if (match)
                    return false;
            }
        }

#elif defined(DMT_OS_LINUX)
        // Forbidden on Linux: slash and NUL
        for (size_t i = 0; i < name.size(); ++i)
            if (name[i] == '/' || name[i] == '\0')
                return false;
#endif

        return true;
    }


    bool isValidStringPath(std::string_view path)
    {
        if (path.empty())
            return false;

        // Accept both separators on Windows; always accept '/' on Linux
        size_t           start     = 0, end;
        std::string_view component = "";
        while (start <= path.size())
        {
            end = path.find_first_of("/\\", start);

            component = (end == std::string_view::npos) ? path.substr(start) : path.substr(start, end - start);

            // Skip empty components caused by leading/duplicate slashes
            // ("/usr/bin", "//server/share", etc.)
            if (!component.empty())
            {
                if (!isValidPathComponent(component, false))
                    return false;
            }

            if (end == std::string_view::npos)
                break;

            start = end + 1;
        }

        if (!isValidPathComponent(component, true))
            return false;

        return true;
    }
} // namespace

int32_t guardedMain()
{
    Ctx::init();
    class Janitor
    {
    public:
        ~Janitor() { Ctx::destroy(); }
    } j;

    {
        Context      ctx;
        ParsedObject parsedObject;
        ctx.trace("Hello Cruel World", {});
        //get the command line arguments
        std::vector<std::string> const argsCmdLine = os::cmdLine();

        if (argsCmdLine.empty())
        {
            printNothing();
            return 0;
        }

        ArgParser                     argParser{};
        std::vector<std::string_view> viewArgs;
        viewArgs.reserve(argsCmdLine.size());
        for (auto& cmd : argsCmdLine)
        {
            viewArgs.push_back(cmd);
        }

        if (!argParser.parse(viewArgs))
        {
            return -2;
        }

        if (argParser.hasFlag("help"))
        {
            ArgParser::printHelp("dmt-tracer");
            return 0;
        }

        enum class ERenderDevice
        {
            eCPU = 0,
            eGPU = 1
        } device = ERenderDevice::eGPU;

        if (auto const& [opt, res] = argParser.getOption("device");
            res == OptionEnum::eValue || res == OptionEnum::eDefaultValue)
        {
            if (opt == "gpu")
            {
                device = ERenderDevice::eGPU;
                ctx.error("TODO", {});
                return -1;
            }
            else if (opt == "cpu")
            {
                device = ERenderDevice::eCPU;
            }
            else
            {
                ctx.error("Invalid Device Value. One of 'cpu', 'gpu'", {});
                return -1;
            }
        }
        else
        {
            ctx.log("No device was specified, defaulting to cpu", {});
            device = ERenderDevice::eCPU;
        }

        alignas(32) char                    buf[256];
        std::pmr::monotonic_buffer_resource stackBuf(buf, 256, std::pmr::null_memory_resource());

        UniqueRef<os::Path> path{nullptr, PmrDeleter::create<os::Path>(&stackBuf)};
        if (auto const& [scene, res] = argParser.getOption("scene"); res == OptionEnum::eValue)
        {
            path = makeUniqueRef<os::Path>(&stackBuf, os::Path::fromString(scene, true));
            if (!path->isValid() || !path->isFile())
            {
                ctx.error(
                    "--scene option should be a valid path,"
                    " '{}' doesn't exist",
                    std::make_tuple(scene));
                return -1;
            }
        }
        else
        {
            ctx.error("--scene option required", {});
            return -1;
        }

        if (auto const& [outPathStr, res] = argParser.getOption("out"); res == OptionEnum::eValue)
        {
            // check whether, as a string, it looks like a file
            if (!isValidStringPath(outPathStr))
            {
                ctx.error("Path '{}' is invalid", std::make_tuple(outPathStr));
                return -1;
            }
            // construct normalized path object
            auto outPath = os::Path::fromString(outPathStr, false);
            {
                auto const parent = outPath.parent();
                if (parent.isDirectory())
                    parsedObject.params.imagePath = std::move(outPath);
                else
                    ctx.warn(
                        "Specified output path '{}' uses a non existing directory component. Using '{}' as output path "
                        "instead",
                        std::make_tuple(outPath.toUnderlying(), parsedObject.params.imagePath.toUnderlying()));
            }
        }
        else
        {
            ctx.log("--out/-o option not found, using '{}' as output file",
                    std::make_tuple(parsedObject.params.imagePath.toUnderlying()));
        }

        if (Parser parser{*path}; !parser.parse(parsedObject))
        {
            ctx.error("Failure parsing file, terminating", {});
            return -1;
        }

        if (device == ERenderDevice::eCPU)
        {
            Renderer renderer{std::move(parsedObject.params),
                              std::move(parsedObject.scene),
                              std::move(parsedObject.texCacheFiles)};
            renderer.startRenderThread();
        }
        else
        {
            // TODO
        }
    }

    return 0;
}
