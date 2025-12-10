#ifndef DMT_TRAINING_WIN32_COOPERATIVE_GROUPS_SAMPLE_UTILS_H_
#define DMT_TRAINING_WIN32_COOPERATIVE_GROUPS_SAMPLE_UTILS_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <functional>
#include <type_traits>
#include <optional>

namespace dmt {
    struct CommandMetadata
    {
        // command recognition. Use one of: Position, Aliases. If aliases size is 0, then position is positive
        // each command occupies a unique position. Position management is implicit
        std::unordered_set<std::wstring> Aliases;
        int64_t                          Position;

        // is it required
        bool Required = false;

        // custom validation function to run on each value
        std::function<std::wstring(std::wstring_view)> Validator = [](auto view) { return L""; };

        // Multivalue management: a value is accumulated if: The same option appears multiple times or if more values
        // are concatenated with the specified Separator. Is Multivalue requires has value
        // positional commands don't have a value
        bool    HasValue            = false;
        bool    IsMultivalue        = false;
        wchar_t Separator           = L',';
        wchar_t AliasValueSeparator = L':';
    };

    class TrainingSampleParser;
    class CommandBuilder
    {
    public:
        explicit CommandBuilder(TrainingSampleParser* parser, std::wstring_view const tag) :
        m_Tag(tag),
        m_Parser(parser)
        {
        }
        CommandBuilder(CommandBuilder const&)                = delete;
        CommandBuilder(CommandBuilder&&) noexcept            = delete;
        CommandBuilder& operator=(CommandBuilder const&)     = delete;
        CommandBuilder& operator=(CommandBuilder&&) noexcept = delete;
        ~CommandBuilder();

        // Removes position if present
        CommandBuilder& AddAlias(std::wstring_view alias);
        // requires that an alias is present, otherwise build will fail
        CommandBuilder& WithValue(bool isMultiValue = false, wchar_t separator = L',', wchar_t aliasValueSeparator = L':');
        template <typename V>
            requires std::is_invocable_r_v<std::wstring, V, std::wstring_view>
        CommandBuilder& WithValidator(V&& func);
        // Empties Aliases
        CommandBuilder& SetPositional();

        CommandBuilder& SetRequired();

        bool Build();

    private:
        std::wstring          m_Tag;
        CommandMetadata       m_Metadata{};
        TrainingSampleParser* m_Parser;
    };

    template <typename V>
        requires std::is_invocable_r_v<std::wstring, V, std::wstring_view>
    CommandBuilder& CommandBuilder::WithValidator(V&& func)
    {
        m_Metadata.Validator = std::forward<decltype(m_Metadata.Validator)>(func);
        return *this;
    }

    struct KeyedCommandInstance
    {
        std::vector<std::wstring> values;
    };

    struct CommandData
    {
        // map position to tag
        std::unordered_map<int64_t, std::wstring> Positionals;
        // map tag to values
        std::unordered_map<std::wstring, KeyedCommandInstance> KeyedCmds;
    };

    class TrainingSampleParser
    {
        friend class CommandBuilder;

    public:
        bool           SetKeyedCommandPrefix(wchar_t prefix);
        CommandBuilder AddCommand(std::wstring_view tag);
        // only positional commands increment the running position
        // returns list of errors
        std::vector<std::wstring> Parse(std::vector<std::wstring> const& args, CommandData& outCommandData);

    private:
        using CommandMap = std::unordered_map<std::wstring, CommandMetadata>;
        // utils for Parse
        std::optional<int64_t>           FindPositionalWithTag(std::wstring const& tag);
        CommandMetadata const*           FindKeyedByAlias(std::wstring const& alias); // remove prefix!
        std::unordered_set<std::wstring> GatherRequired();
        std::unordered_set<std::wstring> GatherSingleValueKeyed();
        std::wstring                     AnyRequiredMissing(CommandData const& commandData);
        void                             MergeAndSplitBySeparatorUntilNextPrefix(
                                        std::vector<std::wstring> const& args,
                                        size_t                           startOffset,
                                        wchar_t                          separator,
                                        std::vector<std::wstring>&       outValues,
                                        size_t&                          outAdvancement) const;

        static bool MetadataCanRepeat(CommandMetadata const& metadata);

        // called by builder
        bool AddCommandInternal(CommandMetadata&& metadata, std::wstring&& tag);

        // KeyType = Tag, useful only for debugging for keyed, used for matching in positional
        CommandMap m_Commands;
        int64_t    m_NextPosition       = 0;
        wchar_t    m_KeyedCommandPrefix = '-';
    };
} // namespace dmt

#endif // DMT_TRAINING_WIN32_COOPERATIVE_GROUPS_SAMPLE_UTILS_H_
