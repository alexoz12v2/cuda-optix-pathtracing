#include "sample-utils.h"

// std
#include "cudautils/cudautils-float.cuh"


#include <cassert>
#include <cwctype>
#include <ranges>

static bool isValidSeparator(wchar_t const c)
{
    switch (c)
    {
        case L' ':
        case L',':
        case L';':
        case L':':
        case L'\t':
        case L'|':
        case L'/':
        case L'\\':
        case L'-': return true;
        default: break;
    }

    // Optionally allow any Unicode whitespace character
    if (std::iswspace(c))
        return true;

    return false;
}

static bool isValidPrefix(wchar_t const c)
{
    switch (c)
    {
        case L'-':
        case L'/': return true;
        default: break;
    }
    return false;
}

namespace dmt {
    CommandBuilder::~CommandBuilder()
    {
        if (m_Parser)
        {
            Build();
        }
    }

    CommandBuilder& CommandBuilder::AddAlias(std::wstring_view alias)
    {
        m_Metadata.Position = -1;
        if (!alias.empty())
        {
            m_Metadata.Aliases.emplace(alias);
        }
        return *this;
    }

    CommandBuilder& CommandBuilder::WithValue(bool isMultiValue, wchar_t separator, wchar_t aliasValueSeparator)
    {
        m_Metadata.HasValue     = true;
        m_Metadata.IsMultivalue = isMultiValue;
        if (isValidSeparator(separator))
            m_Metadata.Separator = separator;
        if (isValidSeparator(aliasValueSeparator))
            m_Metadata.AliasValueSeparator = aliasValueSeparator;
        m_Metadata.Position = -1;
        return *this;
    }

    CommandBuilder& CommandBuilder::SetPositional()
    {
        assert(m_Parser);
        m_Metadata.Aliases.clear();
        m_Metadata.Position = m_Parser->m_NextPosition;
        return *this;
    }

    CommandBuilder& CommandBuilder::SetRequired()
    {
        m_Metadata.Required = true;
        return *this;
    }

    bool CommandBuilder::Build()
    {
        // If it has value, then it should have at least an alias and position at -1
        if (m_Metadata.HasValue && (m_Metadata.Position >= 0 || m_Metadata.Aliases.empty()))
            return false;
        // separator should be a valid character
        if (m_Metadata.HasValue && m_Metadata.IsMultivalue)
        {
            if (!isValidSeparator(m_Metadata.Separator))
                return false;
        }
        if (m_Metadata.IsMultivalue && !m_Metadata.HasValue)
            return false;
        // if valid, return true, insert into parser, assign parser to nullptr
        bool const ret = m_Parser->AddCommandInternal(std::move(m_Metadata), std::move(m_Tag));
        m_Parser       = nullptr;
        return ret;
    }

    bool TrainingSampleParser::SetKeyedCommandPrefix(wchar_t const prefix)
    {
        if (isValidPrefix(prefix))
        {
            m_KeyedCommandPrefix = prefix;
            return true;
        }
        return false;
    }

    CommandBuilder TrainingSampleParser::AddCommand(std::wstring_view const tag) { return CommandBuilder{this, tag}; }

    bool TrainingSampleParser::AddCommandInternal(CommandMetadata&& metadata, std::wstring&& tag)
    {
        // increment position if positional
        if (metadata.Position >= 0)
        {
            if (metadata.Position != m_NextPosition)
                return false;
            ++m_NextPosition;
        }

        auto const& [it, wasInserted] = m_Commands.try_emplace(std::move(tag), std::move(metadata));
        return wasInserted;
    }

    std::vector<std::wstring> TrainingSampleParser::Parse(std::vector<std::wstring> const& args, CommandData& outCommandData)
    {
        using namespace std::string_literals;
        std::vector<std::wstring> errors;
        errors.reserve(16);

        std::unordered_set<std::wstring> keyedSingleValue = GatherSingleValueKeyed();

        int64_t runningPosition = 0;
        for (size_t index = 0; index < args.size(); /*inside the body*/)
        {
            size_t const        current = index;
            std::wstring const& arg     = args[current];
            // try to match a position first
            if (auto maybePos = FindPositionalWithTag(arg); maybePos.has_value())
            {
                if (*maybePos != runningPosition)
                {
                    errors.push_back(
                        L"[Parse Error] argument '"s + arg + L"' should be in position " + std::to_wstring(*maybePos) +
                        L" but was found in position " + std::to_wstring(runningPosition));
                    return errors;
                }
                ++runningPosition;
                outCommandData.Positionals.try_emplace(*maybePos, arg);

                // The index!
                ++index;
            }
            // then try to match one of the aliases of the running commands
            else
            {
                // if it's a keyed, it should start with the prefix
                if (!arg.starts_with(m_KeyedCommandPrefix))
                {
                    errors.push_back(std::wstring(L"[Parser Error] keyed argument should with prefix '") +
                                     m_KeyedCommandPrefix + L"'");
                    return errors;
                }
                // if unrecognized emit an error
                CommandMetadata const* keyedMetadata = FindKeyedByAlias(arg);
                if (!keyedMetadata)
                {
                    errors.push_back(L"[Parser Error] argument '" + arg + L"' unrecognized");
                    return errors;
                }
                assert(keyedMetadata->Position < 0);
                // if not multivalue it should be found once
                if (!MetadataCanRepeat(*keyedMetadata))
                {
                    // remove the prefix. \warning: Assuming prefix is of size 1
                    if (keyedSingleValue.contains(arg.substr(1)))
                        keyedSingleValue.erase(arg);
                    else
                    {
                        errors.push_back(
                            L"[Parser Error] argument '" + arg + L"' Was found multiple times and it's not allowed");
                        return errors;
                    }
                }
                // if it has value, we need an inner loop
                if (!keyedMetadata->HasValue)
                {
                    outCommandData.KeyedCmds.try_emplace(arg, KeyedCommandInstance{});
                }
                else
                {
                    if (!keyedMetadata->IsMultivalue)
                    {
                        KeyedCommandInstance keyedCommandInstance;
                        index = current + 1;
                        // if the value starts with prefix, then invalid
                        std::wstring const& theValue = args[current + 1];
                        if (theValue.starts_with(m_KeyedCommandPrefix))
                        {
                            errors.push_back(L"[Parser Error] value '" + theValue +
                                             L"' starts with a flag prefix. (Flag '" + arg + L"'");
                            return errors;
                        }
                        // run validator on value
                        if (std::wstring valErr = keyedMetadata->Validator(theValue); !valErr.empty())
                        {
                            errors.push_back(L"[Parser Error] value '" + theValue + L"' is invalid: " + valErr);
                            return errors;
                        }
                        keyedCommandInstance.values.push_back(theValue);
                        // remove the prefix. \warning: Assuming prefix is of size 1
                        outCommandData.KeyedCmds.try_emplace(arg.substr(1), keyedCommandInstance);

                        // The Index!
                        ++index;
                    }
                    else
                    {
                        // try to see if there's an existing instance, otherwise emplace a new
                        // remove the prefix. \warning: Assuming prefix is of size 1
                        KeyedCommandInstance& keyedCommandInstance = outCommandData.KeyedCmds[arg.substr(1)];

                        // loop until you find a spacing character different from the separator
                        // or if prefix is found. If the separator is a whitespace, we need to look ahead multiple times
                        std::vector<std::wstring> values;
                        size_t                    advancement = 0;
                        MergeAndSplitBySeparatorUntilNextPrefix(args, current + 1, keyedMetadata->Separator, values, advancement);

                        for (std::wstring const& value : values)
                            keyedCommandInstance.values.push_back(value);

                        // The Index!
                        index += advancement;
                    }
                }
            }
        }

        if (std::wstring const str = AnyRequiredMissing(outCommandData); !str.empty())
            errors.push_back(str);

        return errors;
    }

    std::optional<int64_t> TrainingSampleParser::FindPositionalWithTag(std::wstring const& tag)
    {
        auto const it = m_Commands.find(tag);
        if (it == m_Commands.cend())
            return std::nullopt;
        if (it->second.Position < 0)
            return std::nullopt;

        return it->second.Position;
    }

    CommandMetadata const* TrainingSampleParser::FindKeyedByAlias(std::wstring const& alias)
    {
        using RetT    = std::unordered_map<std::wstring, CommandMetadata>::const_iterator;
        RetT const it = std::ranges::find_if(std::as_const(m_Commands), [alias](auto const& pair) {
            // Remove prefix!
            return pair.second.Position < 0 && pair.second.Aliases.contains(alias.substr(1, alias.size()));
        });
        if (it == m_Commands.cend())
            return nullptr;
        return &it->second;
    }

    std::unordered_set<std::wstring> TrainingSampleParser::GatherRequired()
    {
        std::unordered_set<std::wstring> theSet;
        theSet.reserve(m_Commands.size());
        for (auto const& [tag, meta] : m_Commands)
        {
            if (meta.Required)
                theSet.insert(tag);
        }
        return theSet;
    }

    std::unordered_set<std::wstring> TrainingSampleParser::GatherSingleValueKeyed()
    {
        std::unordered_set<std::wstring> theSet;
        theSet.reserve(m_Commands.size());
        for (auto const& [tag, meta] : m_Commands)
        {
            if (!MetadataCanRepeat(meta))
                theSet.insert(tag);
        }
        return theSet;
    }

    std::wstring TrainingSampleParser::AnyRequiredMissing(CommandData const& commandData)
    {
        std::unordered_set<std::wstring> required = GatherRequired();

        // check positionals
        for (std::wstring const& tag : commandData.Positionals | std::views::values)
        {
            if (required.empty())
                break;
            if (required.contains(tag))
                required.erase(tag);
        }

        // check keyed
        for (auto const& tag : commandData.KeyedCmds | std::views::keys)
        {
            if (required.empty())
                break;
            if (required.contains(tag))
                required.erase(tag);
        }

        if (required.empty())
            return L"";
        return L"'" + *required.begin() + L"' is Required But Missing";
    }

    void TrainingSampleParser::MergeAndSplitBySeparatorUntilNextPrefix(
        std::vector<std::wstring> const& args,
        size_t                           offset,
        wchar_t                          separator,
        std::vector<std::wstring>&       outValues,
        size_t&                          outAdvancement) const
    {
        outValues.clear();
        outAdvancement                 = 0;
        bool const whiteSpaceSeparator = std::iswspace(separator);

        // join into a single string
        std::wstring joined;
        joined.reserve(64);
        for (size_t index = offset; index < args.size(); ++index)
        {
            ++outAdvancement;
            std::wstring copy = args[index];
            // if the copy starts with prefix, break
            if (copy.starts_with(m_KeyedCommandPrefix))
                break;
            joined += copy;
            // if the separator is not the whitespace, then we're done
            if (!whiteSpaceSeparator)
                break;
        }

        // now split by separator the grouped arguments
        size_t prev = 0;
        for (size_t pos = 0; (pos = joined.find_first_of(separator, prev)) != std::wstring::npos; /**/)
        {
            outValues.push_back(joined.substr(prev, pos - prev));
            prev = pos + 1;
        }
        if (prev < joined.size())
            outValues.push_back(joined.substr(prev));
    }

    bool TrainingSampleParser::MetadataCanRepeat(CommandMetadata const& metadata)
    {
        return metadata.HasValue && metadata.IsMultivalue;
    }

} // namespace dmt