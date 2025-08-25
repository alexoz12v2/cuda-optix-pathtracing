#include "CLIManager.h"
#include "platform-context.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <tuple>

namespace dmt {
    static std::vector<Option> const optionTable =
        {{.names       = {"--device", "-d"},
          .description = "Device used for the rendering. It can be cpu or gpu",
          .requiresVal = true,
          .defaultVal  = "cpu",
          .allowedVals = {"cpu", "gpu"}},
         {.names       = {"--scene", "-s"},
          .description = "JSON file used to describe the scene",
          .requiresVal = true,
          .defaultVal  = "",
          .allowedVals = {}},
         {.names       = {"--help", "-h"},
          .description = "Get Help For Command Line Interface Usage",
          .requiresVal = false,
          .defaultVal  = "",
          .allowedVals = {}},
         {.names       = {"--time", "-t"},
          .description = " Measure and report the execution times of key rendering operations.",
          .requiresVal = false,
          .defaultVal  = "",
          .allowedVals = {}}};

    bool ArgParser::parse(std::span<std::string_view> argv)
    {
        Context ctx;

        for (uint32_t i = 1; i < argv.size(); i++)
        {
            std::string_view arg = argv[i];

            //handle long option
            if (arg.starts_with("--"))
            {
                //find the long option in the optionTable
                auto it = std::find_if(optionTable.begin(), optionTable.end(), [&](Option const& opt) {
                    return opt.names[0] == arg;
                });

                //error
                if (it == optionTable.end())
                {
                    ctx.error("Unknown option.", {});
                    return false;
                }

                std::string_view const key{it->names[0].data() + 2, it->names[0].data() + it->names.size()};

                if (it->requiresVal)
                {
                    if (i + 1 < argv.size() && !std::string(argv[i + 1]).starts_with("--"))
                    {
                        std::string_view opVal = argv[++i];

                        if (!it->allowedVals.empty())
                        {
                            auto itVal = std::find_if(it->allowedVals.begin(),
                                                      it->allowedVals.end(),
                                                      [&](std::string const& val) { return val == opVal; });

                            if (itVal == it->allowedVals.end())
                            {
                                ctx.error("{}: wrong argument is not allowed.", std::make_tuple(arg));
                                printHelp(arg);
                                return false;
                            }
                        }

                        m_options[key] = opVal;
                    }
                    else
                    {
                        if (it->defaultVal.empty())
                        {
                            ctx.error("{}: missed argument is mandatory.", std::make_tuple(arg));
                            printHelp(arg);
                            return false;
                        }

                        m_options[key] = it->defaultVal;
                    }
                }
                else
                {
                    m_flags[key] = true;
                }
            }
            else if (arg.starts_with("-"))
            {
                //find the long option in the optionTable
                auto it = std::find_if(optionTable.begin(), optionTable.end(), [&](Option const& opt) {
                    return opt.names[1] == arg;
                });

                //error
                if (it == optionTable.end())
                {
                    ctx.error("Unknown option.", {});
                    return false;
                }

                std::string_view const key{it->names[0].data() + 2, it->names[0].data() + it->names.size()};

                if (it->requiresVal)
                {
                    if (i + 1 < argv.size() && !std::string(argv[i + 1]).starts_with("-"))
                    {
                        std::string_view opVal = argv[++i];

                        if (!it->allowedVals.empty())
                        {
                            auto itVal = std::find_if(it->allowedVals.begin(),
                                                      it->allowedVals.end(),
                                                      [&](std::string const& val) { return val == opVal; });

                            if (itVal == it->allowedVals.end())
                            {
                                ctx.error("{}: argument is not allowed.", std::make_tuple(arg));
                                printHelp(arg);
                                return false;
                            }
                        }

                        m_options[key] = opVal;
                    }
                    else
                    {
                        if (it->defaultVal.empty())
                        {
                            ctx.error("{}: missed argument is mandatory.", std::make_tuple(arg));
                            printHelp(arg);
                            return false;
                        }

                        m_options[key] = it->defaultVal;
                    }
                }
                else
                {
                    m_flags[key] = true;
                }
            }
            else
            {
                m_positionals.push_back(arg);
            }
        }

        return true;
    }

    void ArgParser::printHelp(std::string_view progName) const
    {
        Context            ctx;
        std::ostringstream str;
        str << "\nUsage: " << progName << " [--device <device>] --scene <JSON Path>\n\n";

        // Find the maximum name length for alignment
        size_t maxNameLength = 0;
        for (auto const& opt : optionTable)
        {
            if (std::find(opt.names.begin(), opt.names.end(), "--help") != opt.names.end())
            {
                continue; // Skip the help option itself
            }

            size_t currentLength = 0;
            for (auto const& name : opt.names)
            {
                currentLength += name.length() + 2; // +2 for ", "
            }
            if (!opt.names.empty())
            {
                currentLength -= 2; // remove the last ", "
            }
            maxNameLength = std::max(currentLength, maxNameLength);
        }

        str << "Options:\n";
        for (auto const& opt : optionTable)
        {
            if (std::find(opt.names.begin(), opt.names.end(), "--help") != opt.names.end())
            {
                continue; // Skip the help option itself
            }

            std::string names;
            for (size_t i = 0; i < opt.names.size(); ++i)
            {
                names += opt.names[i];
                if (i < opt.names.size() - 1)
                {
                    names += ", ";
                }
            }

            str << "  " << std::left << std::setw(static_cast<int>(maxNameLength)) << names;
            str << "  " << opt.description;

            if (opt.requiresVal)
            {
                str << " (requires value)";
            }

            if (!opt.defaultVal.empty())
            {
                str << " [default: " << opt.defaultVal << "]";
            }

            if (!opt.allowedVals.empty())
            {
                str << " [allowed: ";
                for (size_t i = 0; i < opt.allowedVals.size(); ++i)
                {
                    str << opt.allowedVals[i];
                    if (i < opt.allowedVals.size() - 1)
                    {
                        str << ", ";
                    }
                }
                str << "]";
            }

            str << "\n";
        }

        ctx.log("{}", std::make_tuple(str.str()));
    }

    std::optional<std::string> ArgParser::getOption(std::string const& name, bool required, std::string const& defaultValue) const
    {
        if (auto it = m_options.find(name); it != m_options.end())
        {
            return std::make_optional(it->second);
        }

        if (required)
            return std::nullopt;

        return std::make_optional(defaultValue);
    }

    bool ArgParser::hasFlag(std::string const& name) const
    {
        if (auto it = m_flags.find(name); it != m_flags.end())
        {
            return it->second;
        }
        return false;
    }

    std::vector<std::string_view> const& ArgParser::getPositionals() const { return m_positionals; }

}; // namespace dmt