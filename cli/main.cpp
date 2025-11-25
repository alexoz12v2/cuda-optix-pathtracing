#define DMT_ENTRY_POINT
#define DMT_WINDOWS_CLI
#include "platform.h"

#include "core-render.h"
#include "core-parser.h"

#include "CLIManager.h"

#include <memory_resource>
#include <string>

namespace /*static*/ {
    using namespace dmt;

    void printNothing() {
        Context ctx;
        ctx.log("No Input file given. See --help for more information\n", {});
    }
} // namespace

int32_t guardedMain()
{
    dmt::Ctx::init();
    class Janitor
    {
    public:
        ~Janitor() { dmt::Ctx::destroy(); }
    } j;

    {
        dmt::Context ctx;
        ctx.trace("Hello Cruel World", {});
        //get the command line arguments
        std::vector<std::string> argsCmdLine = dmt::os::cmdLine();

        if (argsCmdLine.size() <= 0)
        {
            printNothing();
            return 0;
        }

        dmt::ArgParser                argParser{};
        std::vector<std::string_view> viewArgs;
        viewArgs.reserve(argsCmdLine.size());
        for (auto& cmd : argsCmdLine)
        {
            viewArgs.push_back(cmd);
        }

        argParser.parse(viewArgs);
        if (argParser.hasFlag("help"))
        {
            argParser.printHelp("dmt-tracer");
            return 0;
        }

        int device = -1;
        if (auto const& [opt, res] = argParser.getOption("device"); res == OptionEnum::eValue || res == OptionEnum::eDefaultValue)
        {
            if (opt == "gpu")
            {
                ctx.error("TODO", {});
                return -1;
            }

            if (opt != "cpu")
            {
                ctx.error("Invalid Device Value. One of 'cpu', 'gpu'", {});
                return -1;
            }
        }

        alignas(32) char                    buf[256];
        std::pmr::monotonic_buffer_resource stackBuf(buf, 256, std::pmr::null_memory_resource());

        dmt::UniqueRef<dmt::os::Path> path{nullptr, dmt::PmrDeleter::create<dmt::os::Path>(&stackBuf)};
        if (auto const& [scene, res] = argParser.getOption("scene"); res == OptionEnum::eValue)
        {
            path = dmt::makeUniqueRef<dmt::os::Path>(&stackBuf, dmt::os::Path::fromString(scene));
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

        if (device == -1)
        {
            dmt::Renderer renderer;
            dmt::Parser parser{*path, &renderer};

            if (!parser.parse())
            {
                ctx.error("Failure parsing file, terminating", {});
                return -1;
            }

            renderer.startRenderThread();
        }
        else
        {
            // TODO
        }
    }

    return 0;
}
