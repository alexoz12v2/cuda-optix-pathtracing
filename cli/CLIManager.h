#pragma once

#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <span>
#include <vector>

namespace dmt {

    struct Option
    {
        std::vector<std::string> names;
        std::string              description;
        bool                     requiresVal;
        std::string              defaultVal;
        std::vector<std::string> allowedVals;
    };

    enum class OptionEnum
    {
        eNotPresent = 0,
        eRequiredNotPresent,
        eValue,
        eDefaultValue,
        Count
    };

    struct OptionResult
    {
        std::string value;
        OptionEnum  result;
    };

    //-device cpu/gpu
    //-scene path
    //-time

    class ArgParser
    {
    public:
        ArgParser() = default;

        [[nodiscard]] OptionResult getOption(std::string const& name,
                                             bool               required     = false,
                                             std::string const* defaultValue = nullptr) const;

        [[nodiscard]] bool hasFlag(std::string const& name) const;

        [[nodiscard]] std::vector<std::string_view> const& getPositionals() const;

        static void printHelp(std::string_view progName);

        [[nodiscard]] bool parse(std::span<std::string_view> argv);


    private:
        std::unordered_map<std::string_view, std::string> m_options;
        std::unordered_map<std::string_view, bool>        m_flags;
        std::vector<std::string_view>                     m_positionals;
    };
} // namespace dmt