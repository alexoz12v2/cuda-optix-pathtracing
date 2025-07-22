#pragma once

#include "core/core-macros.h"

#include <type_traits>
#include <optional>

namespace dstd {
    template <std::copyable T, typename A, typename UnaryPredicate>
        requires(std::is_invocable_r_v<bool, UnaryPredicate, T &&>)
    std::optional<T> move_to_back_and_pop_if(std::vector<T, A>& vec, UnaryPredicate pred)
    {
        auto it = std::find_if(vec.begin(), vec.end(), pred);
        if (it != vec.end())
        {
            if (it != vec.end() - 1)
            {
                std::iter_swap(it, vec.end() - 1); // move found element to back
            }
            T res = vec.back();
            vec.pop_back(); // remove it
            return res;
        }
        return std::nullopt; // no match
    }

    template <typename In, typename Out, typename Func>
    class transform_span
    {
    public:
        using value_type = Out;
        using size_type  = std::size_t;

        transform_span(In* data, size_type size, Func func) : _data(data), _size(size), _func(std::move(func)) {}

        value_type operator[](size_type i) const { return _func(_data[i]); }

        size_type size() const noexcept { return _size; }

        bool empty() const noexcept { return _size == 0; }

        In* data() const noexcept { return _data; }

    private:
        In*       _data;
        size_type _size;
        Func      _func;
    };
} // namespace dstd