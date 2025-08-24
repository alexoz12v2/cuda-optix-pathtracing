#include "CLIManager.h"
#include "platform-context.h"

#include <string_view>

namespace dmt
{
    bool ArgParser::parse(int argc, std::span<std::string_view> argv)
    {
        Context ctx;

        for(int i = 1; i < argc; i++)
        {
            std::string_view arg = argv[i];

            //handle long option
            if(arg.starts_with("--"))
            {
                //find the long option in the optionTable
                auto it = std::find_if(optionTable.begin(), optionTable.end(),
                [&](const Option& opt){
                    return opt.names[1] == arg;
                });

                //error
                if(it == optionTable.end())
                {
                        ctx.error("Unknown option.", {});
                        return false;
                }
                
                if(it->requiresVal)
                {
                    if(i + 1 < argc && !std::string(argv[static_cast<size_t>(i + 1)]).starts_with("--"))
                    {
                        std::string_view opVal = argv[static_cast<size_t>(++i)];
                        
                        if(!it->allowedVals.empty())
                        {
                            auto itVal = std::find_if(it->allowedVals.begin(), it->allowedVals.end(),
                            [&](const std::string& val){
                                return val == opVal;
                            });

                            if(itVal == it->allowedVals.end())
                            {
                                ctx.error("{}: wrong argument is not allowed.", std::make_tuple(arg));
                                printHelp(arg);
                                return false;
                            }    
                        }

                        m_options[arg] = opVal;
                    }
                    else
                    {
                        if(it->defaultVal.empty())
                        {
                            ctx.error("{}: missed argument is mandatory.", std::make_tuple(arg));
                            printHelp(arg);
                            return false;
                        }
                        
                        m_options[arg] = it->defaultVal;
                    }
                    
                }
                else 
                {
                    m_flags[arg] = true;
                }
                
            }
            else if(arg.starts_with("-"))
            {
                //find the long option in the optionTable
                auto it = std::find_if(optionTable.begin(), optionTable.end(),
                [&](const Option& opt){
                    return opt.names[0] == arg;
                });

                //error
                if(it == optionTable.end())
                {
                        ctx.error("Unknown option.", {});
                        return false;
                }

                if(it->requiresVal)
                {
                    if(i + 1 < argc && !std::string(argv[static_cast<size_t>(i + 1)]).starts_with("-"))
                    {
                       std::string_view opVal = argv[static_cast<size_t>(++i)];
                        
                        if(!it->allowedVals.empty())
                        {
                            auto itVal = std::find_if(it->allowedVals.begin(), it->allowedVals.end(),
                            [&](const std::string& val){
                                return val == opVal;
                            });

                            if(itVal == it->allowedVals.end())
                            {
                                ctx.error("{}: argument is not allowed.", std::make_tuple(arg));
                                printHelp(arg);
                                return false;
                            }    
                        }

                        m_options[arg] = opVal;
                    }
                    else
                    {
                        if(it->defaultVal.empty())
                        {
                            ctx.error("{}: missed argument is mandatory.", std::make_tuple(arg));
                            printHelp(arg);
                            return false;
                        }
                        
                        m_options[arg] = it->defaultVal;
                    }
                }
                else 
                {
                    m_flags[arg] = true;
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
        Context ctx;

        for (const auto& opt : optionTable)
        {
             
            auto it = std::find_if(opt.names.begin(), opt.names.end(),
            [&](const std::string& name){
                return name == progName;
            });

            if(it != opt.names.end())
            {
                
                ctx.log("[Option]:", {});
                //names:
                for(const auto& name : opt.names)
                {
                    ctx.log("{}", std::make_tuple(name));
                }
                ctx.log("Description: ", std::make_tuple(opt.description));
                
                if(opt.requiresVal)
                {
                    if(!opt.defaultVal.empty())
                    {
                        ctx.log("Default values", {});

                        for(const auto& val : opt.defaultVal)
                        {
                            ctx.log("{}", std::make_tuple(val));
                        }

                        ctx.log("Allowed values:", {});
                    } 
                    else{ctx.log("Mandatory allowed values:", {});}
                    
                    //allowedVal:
                    for(const auto& val : opt.allowedVals)
                    {
                        ctx.log("{}", std::make_tuple(val));
                    }
                }
            }
        }
    }

    std::string ArgParser::getOption(const std::string& name, const std::string& defaultValue) const 
    {
        if (auto it = m_options.find(name); it != m_options.end()) 
        {
            return it->second;
        }
        return defaultValue;
    }

    bool ArgParser::hasFlag(const std::string& name) const 
    {
        if (auto it = m_flags.find(name); it != m_flags.end()) 
        {
            return it->second;
        }
        return false;
    }

    const std::vector<std::string_view>& ArgParser::getPositionals() const 
    {
        return m_positionals;
    }

};