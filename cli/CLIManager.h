#include <iostream>
#include <unordered_map>

namespace dmt
{
    class ArgParser
    {
        public:
        ArgParser(int argc, char* argv[]);

        std::string getOption(const std::string& name, const std::string& defaultValue = "") const;

        bool hasFlag(const std::string& name) const;

        const std::vector<std::string>& getPositionals() const;

        private:
        std::unordered_map<std::string, std::string> m_options;
        std::unordered_map<std::string, bool> m_flags;
        std::vector<std::string> m_positionals;
    };
}