#include "CLIManager.h"

namespace dmt
{
    ArgParser::ArgParser(int argc, char* argv[])
    {
        for(int i = 1; i < argc; i++)
        {
            std::string arg = argv[i];

            //handle long option
            if(arg.starts_with("--"))
            {
                if(i + 1 < argc && !std::string(argv[i + 1]).starts_with("--"))
                {
                    m_options[arg] = argv[++i];
                }
                else 
                {
                    m_flags[arg] = true;
                }
            }
            else if(arg.starts_with("-"))
            {
                if(i + 1 < argc && !std::string(argv[i + 1]).starts_with("-"))
                {
                    m_options[arg] = argv[++i];
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

    const std::vector<std::string>& ArgParser::getPositionals() const 
    {
        return m_positionals;
    }

};