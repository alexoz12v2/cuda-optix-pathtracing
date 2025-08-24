#pragma once

#include <iostream>
#include <string_view>
#include <unordered_map>
#include <span>

namespace dmt {

    struct Option
    {
        std::vector<std::string> names;
        std::string              description;
        bool                     requiresVal;
        std::string              defaultVal;
        std::vector<std::string> allowedVals;
    };

    //-device cpu/gpu
    //-scene path
    //-time

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
         {.names       = {"--time", "-t"},
          .description = " Measure and report the execution times of key rendering operations.",
          .requiresVal = false,
          .defaultVal  = "",
          .allowedVals = {}}};


    class ArgParser
    {
    public:
        ArgParser() = default;

        std::string getOption(std::string const& name, std::string const& defaultValue = "") const;

        bool hasFlag(std::string const& name) const;

        std::vector<std::string_view> const& getPositionals() const;

        void printHelp(std::string_view progName) const;

        bool parse(int argc, std::span<std::string_view> argv);
        

    private:
        std::unordered_map<std::string_view, std::string> m_options;
        std::unordered_map<std::string_view, bool>        m_flags;
        std::vector<std::string_view>                     m_positionals;
    };
} // namespace dmt