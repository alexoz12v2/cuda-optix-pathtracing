#ifndef DMT_CORE_PUBLIC_CORE_DSTD_H
#define DMT_CORE_PUBLIC_CORE_DSTD_H

#include "core-macros.h"

#include "platform-memory.h"

#include <algorithm>
#include <type_traits>
#include <optional>
#include <span>

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

    /**
     * @brief represent a heap allocated, fixed length, buffer treated as a 2D array,
     * stored in row-major order (xs near each other)
     */
    template <typename N>
        requires(std::is_integral_v<N> || std::is_floating_point_v<N>)
    class Array2D
    {
    public:
        Array2D(uint32_t xSize, uint32_t ySize, std::pmr::memory_resource* memory = std::pmr::get_default_resource()) :
        m_buffer(dmt::makeUniqueRef<N[]>(memory, xSize * ySize)),
        m_xSize(xSize),
        m_ySize(ySize)
        {
            assert(m_buffer);
        }

        DMT_FORCEINLINE float& operator()(int32_t x, int32_t y)
        {
            assert(x >= 0 && x < static_cast<int32_t>(m_xSize));
            assert(y >= 0 && y < static_cast<int32_t>(m_ySize));
            return m_buffer[x + m_xSize * y];
        }

        DMT_FORCEINLINE float const& operator()(int32_t x, int32_t y) const
        {
            assert(x >= 0 && x < static_cast<int32_t>(m_xSize));
            assert(y >= 0 && y < static_cast<int32_t>(m_ySize));
            return m_buffer[x + m_xSize * y];
        }

        DMT_FORCEINLINE uint32_t xSize() const { return m_xSize; }
        DMT_FORCEINLINE uint32_t ySize() const { return m_ySize; }

        DMT_FORCEINLINE N*       data() { return m_buffer.get(); }
        DMT_FORCEINLINE N const* data() const { return m_buffer.get(); }

        DMT_FORCEINLINE std::span<N> rowSpan(int32_t y)
        {
            assert(y >= 0 && y < static_cast<int32_t>(m_ySize));
            return std::span<N>(m_buffer.get() + y * m_xSize, m_xSize);
        }

        DMT_FORCEINLINE std::span<N const> rowSpan(int32_t y) const
        {
            assert(y >= 0 && y < static_cast<int32_t>(m_ySize));
            return std::span<N const>(m_buffer.get() + y * m_xSize, m_xSize);
        }

    private:
        dmt::UniqueRef<N[]> m_buffer;
        uint32_t            m_xSize;
        uint32_t            m_ySize;
    };

    // Shared base template — used internally
    template <typename T>
    class Array2DViewBase
    {
    public:
        using value_type = T;

        Array2DViewBase(T* data, uint32_t xSize, uint32_t ySize, uint32_t stride) :
        m_data(data),
        m_xSize(xSize),
        m_ySize(ySize),
        m_stride(stride)
        {
            assert(data != nullptr);
        }

        T const& operator()(int32_t x, int32_t y) const
        {
            assert(x >= 0 && x < static_cast<int32_t>(m_xSize));
            assert(y >= 0 && y < static_cast<int32_t>(m_ySize));
            return m_data[x + y * m_stride];
        }

        uint32_t xSize() const { return m_xSize; }
        uint32_t ySize() const { return m_ySize; }

        T const* data() const { return m_data; }

    protected:
        T*       m_data;
        uint32_t m_xSize;
        uint32_t m_ySize;
        uint32_t m_stride;
    };

    // Mutable view
    template <typename N>
        requires(std::is_integral_v<N> || std::is_floating_point_v<N>)
    class Array2DView : public Array2DViewBase<N>
    {
    public:
        using Base = Array2DViewBase<N>;
        using Base::Base;

        N& operator()(int32_t x, int32_t y)
        {
            assert(x >= 0 && x < static_cast<int32_t>(this->m_xSize));
            assert(y >= 0 && y < static_cast<int32_t>(this->m_ySize));
            return this->m_data[x + y * this->m_stride];
        }

        N* data() { return this->m_data; }

        Array2DView<N> subspan(uint32_t xOffset, uint32_t yOffset, uint32_t tileXSize, uint32_t tileYSize) const
        {
            assert(xOffset + tileXSize <= this->m_xSize);
            assert(yOffset + tileYSize <= this->m_ySize);

            N* subData = this->m_data + xOffset + yOffset * this->m_stride;
            return Array2DView<N>(subData, tileXSize, tileYSize, this->m_stride);
        }

        // Constructor from Array2D<N>
        Array2DView(class Array2D<N>& array) : Array2DView(array.data(), array.xSize(), array.ySize(), array.xSize()) {}
    };

    // Const view
    template <typename N>
        requires(std::is_integral_v<N> || std::is_floating_point_v<N>)
    class Array2DView<N const> : public Array2DViewBase<N const>
    {
    public:
        using Base = Array2DViewBase<N const>;
        using Base::Base;

        Array2DView<N const> subspan(uint32_t xOffset, uint32_t yOffset, uint32_t tileXSize, uint32_t tileYSize) const
        {
            assert(xOffset + tileXSize <= this->m_xSize);
            assert(yOffset + tileYSize <= this->m_ySize);

            N const* subData = this->m_data + xOffset + yOffset * this->m_stride;
            return Array2DView<N const>(subData, tileXSize, tileYSize, this->m_stride);
        }

        // Constructor from const Array2D<N>
        Array2DView(const class Array2D<N>& array) :
        Array2DView(array.data(), array.xSize(), array.ySize(), array.xSize())
        {
        }
    };
} // namespace dstd
#endif // DMT_CORE_PUBLIC_CORE_DSTD_H
