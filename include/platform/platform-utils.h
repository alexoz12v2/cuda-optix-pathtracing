#pragma once

#include <platform/platform-macros.h>

#include <bit>
#include <charconv> // For std::from_chars
#include <concepts>
#include <iterator>
#include <memory>
#include <memory_resource>
#include <source_location>
#include <shared_mutex>
#include <map>
#include <string_view>
#include <type_traits>
#include <vector>
#include <string>

#include <cassert>
#include <cctype>
#include <cstdint>

namespace dmt::os {
    DMT_PLATFORM_API uint64_t processId();
    DMT_PLATFORM_API uint64_t threadId();
    DMT_PLATFORM_API void*    reserveVirtualAddressSpace(size_t size);
    DMT_PLATFORM_API size_t   systemAlignment();
    DMT_PLATFORM_API bool     commitPhysicalMemory(void* address, size_t size);
    DMT_PLATFORM_API bool     freeVirtualAddressSpace(void* address, size_t size);
    DMT_PLATFORM_API void     decommitPage(void* pageAddress, size_t pageSize);

    // use C runtime standard allocation functions
    DMT_PLATFORM_API void* allocate(size_t _bytes, size_t _align);
    DMT_PLATFORM_API void  deallocate(void* ptr, [[maybe_unused]] size_t _bytes, [[maybe_unused]] size_t _align);

    // for debugging purposes only, so we don't care about memory
    DMT_PLATFORM_API std::pmr::vector<std::pair<std::pmr::string, std::pmr::string>> getEnv(
        std::pmr::memory_resource* resource = std::pmr::get_default_resource());

    class DMT_PLATFORM_API Path
    {
    public:
        static Path home(std::pmr::memory_resource* resource = std::pmr::get_default_resource());
        static Path cwd(std::pmr::memory_resource* resource = std::pmr::get_default_resource());
        static Path invalid(std::pmr::memory_resource* resource = std::pmr::get_default_resource());

        // disk designator ignored in everything different than windows
        static Path root(char const*                diskDesignator,
                         std::pmr::memory_resource* resource = std::pmr::get_default_resource());

        static Path executableDir(std::pmr::memory_resource* resource = std::pmr::get_default_resource());

        void parent_();
        Path parent() const;

        // shouldn't contain any path separator. case insensitive on windows
        void operator/=(char const* pathComponent);

        // uses the same memory reosurce
        Path operator/(char const* pathComponent) const;

        bool isDirectory() const { return m_isDir; }
        bool isFile() const { return !m_isDir; }
        // synonym to "is valid and exists in the filesystem"
        bool isValid() const { return m_valid; }

        // if nullptr, use the current resource
        std::pmr::string toUnderlying(std::pmr::memory_resource* resource = nullptr) const;

        void const* internalData() const { return m_data; }
        // number of byte pairs (utf16) or number of bytes (utf8)
        uint32_t dataLength() const { return m_dataSize; }

        // implement also a lazy iterator to enumerate all children, like python pathlib's iterdir
        // mkdir, rmdir, newFile

        Path(Path const&);
        Path(Path&&) noexcept;
        Path& operator=(Path const&);
        Path& operator=(Path&&) noexcept;
        ~Path() noexcept;

    private:
        Path(std::pmr::memory_resource* resource, void* content, uint32_t capacity, uint32_t size);

        std::pmr::memory_resource* m_resource;

        // this is a wchar_t* in windows, char (utf8) in linux. this is always normalized. this always starts
        // with "\\?\" (long path) on windows (unless the disk designator is a UCN name)
        // since the path is always normalized, the constructor should check the content
        void* m_data;
        // both length and size are count of bytes
        uint32_t m_capacity;
        uint32_t m_dataSize;
        bool     m_isDir;
        bool     m_valid;
    };

    class DMT_PLATFORM_API LibraryLoader
    {
    public:
        static constexpr uint32_t defaultInitialCapacity = 32;

    public:
        LibraryLoader(bool                       canGrow,
                      std::pmr::memory_resource* resource        = std::pmr::get_default_resource(),
                      uint32_t                   initialCapacity = defaultInitialCapacity);
        LibraryLoader(LibraryLoader const&)                = delete;
        LibraryLoader(LibraryLoader&&) noexcept            = delete;
        LibraryLoader& operator=(LibraryLoader const&)     = delete;
        LibraryLoader& operator=(LibraryLoader&&) noexcept = delete;
        ~LibraryLoader() noexcept;

        bool isValid() const;
        bool pushSearchPath(Path const& path);
        bool popSearchPath();

        void* loadLibrary(std::string_view name, bool useSystemPaths = false, Path const* pathOverride = nullptr) const;
        bool  unloadLibrary(void* library) const;

    private:
        bool tryGrow();

        std::pmr::memory_resource* m_resource            = std::pmr::get_default_resource();
        Path*                      m_searchPaths         = nullptr;
        uint32_t                   m_searchPathsLen      = 0;
        uint32_t                   m_searchPathsCapacity = 0;
        bool                       m_canGrow;
    };

    namespace lib {
        DMT_PLATFORM_API void* getFunc(void* library, char const* funcName);
    }
} // namespace dmt::os

namespace dmt {
    inline int32_t strlen_mb(std::string_view const s)
    {
        std::mblen(nullptr, 0); // reset the conversion state
        int32_t     result = 0;
        char const* ptr    = s.data();
        for (char const* const end = ptr + s.size(); ptr < end; ++result)
        {
            int const next = std::mblen(ptr, end - ptr);
            if (next == -1)
                // throw std::runtime_error("strlen_mb(): conversion error");
                return -1;
            ptr += next;
        }
        return result;
    }

    using sid_t = uint64_t;
    template <typename T>
    struct PmrDeleter
    {
        explicit PmrDeleter(std::pmr::memory_resource* pRes) : resource(pRes) {}

        void operator()(T* ptr) const
        {
            if (ptr)
                resource->deallocate(ptr, sizeof(T), alignof(T));
        }

        std::pmr::memory_resource* resource;
    };

    template <typename Enum>
        requires(std::is_enum_v<Enum>)
    inline constexpr std::underlying_type_t<Enum> toUnderlying(Enum e)
    {
        return static_cast<std::underlying_type_t<Enum>>(e);
    }

    template <typename E>
        requires((std::is_enum_v<E>) && requires() { E::eCount; })
    inline constexpr E fromUnderlying(std::underlying_type_t<E> i)
    {
        if (i < toUnderlying(E::eCount))
        {
            return static_cast<E>(i);
        }

        return static_cast<E>(0);
    }

    template <std::integral I, std::integral I2>
        requires(sizeof(I) <= 8 && std::is_unsigned_v<I> && std::is_unsigned_v<I2> && sizeof(I) == sizeof(I2))
    inline constexpr I roundUpToNextMultipleOf(I value, I2 divisor)
    {
        if (divisor == 0)
        {
            return 0; // Avoid division by zero
        }
        return ((value + divisor - 1) / divisor) * divisor;
    }

    template <std::integral I>
        requires(sizeof(I) <= 8 && std::is_unsigned_v<I>)
    inline constexpr I nextPOT(I value)
    {
        if constexpr (sizeof(I) == 8)
        {
            // Handle 64-bit integers
            if (value == 0)
            {
                return 1;
            }
            --value;
            value |= value >> 1;
            value |= value >> 2;
            value |= value >> 4;
            value |= value >> 8;
            value |= value >> 16;
            value |= value >> 32;
            return ++value;
        }
        else if constexpr (sizeof(I) == 4)
        {
            // Handle 32-bit integers
            if (value == 0)
            {
                return 1;
            }
            --value;
            value |= value >> 1;
            value |= value >> 2;
            value |= value >> 4;
            value |= value >> 8;
            value |= value >> 16;
            return ++value;
        }
        else if constexpr (sizeof(I) == 2)
        {
            // Handle 16-bit integers
            if (value == 0)
            {
                return 1;
            }
            --value;
            value |= value >> 1;
            value |= value >> 2;
            value |= value >> 4;
            value |= value >> 8;
            return ++value;
        }
        else if constexpr (sizeof(I) == 1)
        {
            // Handle 8-bit integers
            if (value == 0)
            {
                return 1;
            }
            --value;
            value |= value >> 1;
            value |= value >> 2;
            value |= value >> 4;
            return ++value;
        }
    }

    inline constexpr uint32_t smallestPOTMask(uint32_t value)
    {
        // If value is 0, smallest POT is 1 (mask = 0x0000'0001)
        if (value == 0)
            return 1;

        // Calculate the smallest power of 2 >= value
        --value;
        value |= value >> 1;
        value |= value >> 2;
        value |= value >> 4;
        value |= value >> 8;
        value |= value >> 16;
        ++value;

        // Return the power of 2 as the mask
        return value - 1;
    }

    template <std::integral T>
    constexpr T clamp(T val, T min_val, T max_val) noexcept
    {
        if (val < min_val)
            return min_val;
        if (val > max_val)
            return max_val;
        return val;
    }

    template <typename Enum>
        requires(std::is_enum_v<Enum>)
    inline constexpr bool isAnyEnum(Enum e, std::initializer_list<Enum> const& values)
    {
        for (auto const& value : values)
        {
            if (e == value)
            {
                return true;
            }
        }
        return false;
    }

    template <std::integral I, std::forward_iterator It>
    inline constexpr bool oneOf(I value, It const& begin, It const& end)
    {
        for (auto it = begin; it != end; ++it)
        {
            auto const& item = *it;
            if (value == item)
            {
                return true;
            }
        }
        return false;
    }

    inline constexpr bool parseFloat(std::string_view input, float& outValue)
    {
        // Remove leading and trailing whitespace
        while (!input.empty() && std::isspace(input.front()))
        {
            input.remove_prefix(1);
        }
        while (!input.empty() && std::isspace(input.back()))
        {
            input.remove_suffix(1);
        }

        if (input.empty())
        {
            return false; // Empty input is not a valid float
        }

        // Parse the float
        char const* begin = input.data();
        char const* end   = input.data() + input.size();

        auto result = std::from_chars(begin, end, outValue);
        if (result.ec != std::errc() || result.ptr != end)
        {
            return false; // Error in parsing or extra characters
        }

        return true;
    }

    template <std::integral I>
    inline constexpr bool parseInt(std::string_view str, I& outValue)
    {
        if (str.empty())
        {
            return false;
        }

        auto const [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), outValue);
        return ec == std::errc{};
    }

    inline constexpr bool endsWithAny(std::string_view str, std::initializer_list<std::string_view> const& suffixes)
    {
        for (auto const& suffix : suffixes)
        {
            if (str.ends_with(suffix))
            {
                return true;
            }
        }
        return false;
    }

    template <std::integral T>
    inline constexpr T popCount(T v)
    {
        v   = v - ((v >> 1) & (T) ~(T)0 / 3);                             // temp
        v   = (v & (T) ~(T)0 / 15 * 3) + ((v >> 2) & (T) ~(T)0 / 15 * 3); // temp
        v   = (v + (v >> 4)) & (T) ~(T)0 / 255 * 15;                      // temp
        T c = (T)(v * ((T) ~(T)0 / 255)) >> (sizeof(T) - 1) * CHAR_BIT;   // count
        return c;
    }

    inline constexpr uint32_t countTrailingZeros(uint32_t v)
    {
        uint32_t c = 32; // c will be the number of zero bits on the right
        v &= -signed(v);
        if (v)
            c--;
        if (v & 0x0000FFFF)
            c -= 16;
        if (v & 0x00FF00FF)
            c -= 8;
        if (v & 0x0F0F0F0F)
            c -= 4;
        if (v & 0x33333333)
            c -= 2;
        if (v & 0x55555555)
            c -= 1;
        return c;
    }

    inline constexpr size_t findFirstWhitespace(std::string_view str)
    {
        for (size_t i = 0; i < str.size(); ++i)
        {
            if (std::isspace(static_cast<unsigned char>(str[i])))
            {
                return i;
            }
        }
        return std::string_view::npos;
    }

    inline constexpr std::string_view trimStartWhitespace(std::string_view str)
    {
        size_t start = 0;
        while (start < str.size() && std::isspace(static_cast<unsigned char>(str[start])))
        {
            ++start;
        }
        return str.substr(start);
    }

    inline constexpr bool startsWithEndsWith(std::string_view str, char start, char end)
    {
        return str.starts_with(start) && str.ends_with(end);
    }

    inline constexpr std::string_view dequoteString(std::string_view str)
    {
        if (str.size() >= 2 && (str.front() == '"' || str.front() == '\'') && str.back() == str.front())
        {
            return str.substr(1, str.size() - 2); // Remove the first and last characters
        }
        return str; // Return as-is if not quoted
    }

    inline constexpr uint32_t countTrailingZeros(uint64_t v)
    {
        uint32_t c = 64; // c will be the number of zero bits on the right
        v &= -signed(v); // Isolate the lowest set bit
        if (v)
            c--;
        if (v & 0x00000000FFFFFFFF) // Check the lower 32 bits
            c -= 32;
        if (v & 0x0000FFFF0000FFFF) // Check the lower 16 bits in each 32-bit half
            c -= 16;
        if (v & 0x00FF00FF00FF00FF) // Check the lower 8 bits in each 16-bit half
            c -= 8;
        if (v & 0x0F0F0F0F0F0F0F0F) // Check the lower 4 bits in each 8-bit half
            c -= 4;
        if (v & 0x3333333333333333) // Check the lower 2 bits in each 4-bit half
            c -= 2;
        if (v & 0x5555555555555555) // Check the lower 1 bit in each 2-bit half
            c -= 1;

        return c;
    }

    // Primary template (matches nothing by default)
    template <template <typename...> class Template, typename T>
    struct is_template_instantiation : std::false_type
    {
    };

    // Specialization for template instantiations
    template <template <typename...> class Template, typename... Args>
    struct is_template_instantiation<Template, Template<Args...>> : std::true_type
    {
    };

    // Helper variable template
    template <template <typename...> class Template, typename T>
    inline constexpr bool is_template_instantiation_v = is_template_instantiation<Template, T>::value;

    // Concept using the helper
    template <template <typename...> class Template, typename T>
    concept TemplateInstantiationOf = is_template_instantiation_v<Template, T>;

    // TODO move to cpp all not inline
    inline constexpr char pathSeparator()
    {
#if defined(DMT_OS_WINDOWS)
        return '\\';
#else
        return '/';
#endif
    }

    inline void* alignTo(void* address, size_t alignment)
    {
        // Ensure alignment is a power of two (required for bitwise operations).
        size_t const mask = alignment - 1;
        assert((alignment & mask) == 0 && "Alignment must be a power of two.");

        uintptr_t addr        = reinterpret_cast<uintptr_t>(address);
        uintptr_t alignedAddr = (addr + mask) & ~mask;

        return reinterpret_cast<void*>(alignedAddr);
    }

    inline constexpr uintptr_t alignToAddr(uintptr_t address, size_t alignment)
    {
        // Ensure alignment is a power of two (required for bitwise operations).
        size_t const mask = alignment - 1;
        assert((alignment & mask) == 0 && "Alignment must be a power of two.");

        uintptr_t alignedAddr = (address + mask) & ~mask;

        return alignedAddr;
    }

    inline void* alignToBackward(void* address, size_t alignment)
    {
        // Ensure alignment is a power of two (required for bitwise operations).
        size_t const mask = alignment - 1;
        assert((alignment & mask) == 0 && "Alignment must be a power of two.");

        uintptr_t addr        = reinterpret_cast<uintptr_t>(address);
        uintptr_t alignedAddr = addr & ~mask;

        return reinterpret_cast<void*>(alignedAddr);
    }

#if !defined(DMT_ARCH_X86_64)
#error "Pointer Tagging relies heavily on x86_64's virtual addreess format"
#endif
    /** Class managing a pointer aligned to a 32 byte boundary, embedding a 12 bits tag split among its 7 high bits and 5 low bits
     * - x86_64 systems actually use 48 bits for virtual addresses. Actually, scratch that, with the
     *   latest PML5 (https://en.wikipedia.org/wiki/Intel_5-level_paging) extended virtual adderesses
     *   to 57 bits. This means that the high 7 bits of a memory address are unused, and we can make good use of them
     * - adding to the fact that minimum block size is 32 Bytes, hence aligned to a 32 Byte boundary, we have an additional
     *   5 bits free to use
     * hence, our tagged pointers can exploit 12 bits of information in total
     * Remember: it holds only for host addresses, and to regain access to the original address, you need to mask out
     * the low bits (5), and sign extend from bit 56 to bit 63
     * Reference test code:
     *   alignas(32) int data  = 42; // Ensure alignment
     *   uint16_t      trueTag = (1u << 12u) - 1;
     *   TaggedPointer tp(&data, trueTag);
     *   std::cout << "True Tag 0x" << std::hex << trueTag << std::dec << '\n';
     *   std::cout << "Raw pointer: " << tp.getPointer() << "\n";
     *   std::cout << "True Pointer: " << &data << '\n';
     *   std::cout << "Tag: 0x" << std::hex << tp.getTag() << "\n";
     *   std::cout << "Dereferenced value: " << std::dec << tp.operator* <int>() << "\n";
     * TODO: we can template this class on the number of low bits we expect to be zeroed out
     */
    class DMT_PLATFORM_API TaggedPointer
    {
    public:
        // Constructor
        constexpr TaggedPointer(std::nullptr_t null = nullptr) : m_taggedPtr(0) {}

        constexpr TaggedPointer(void* ptr, uint16_t tag = 0) { set(std::bit_cast<uintptr_t>(ptr), tag); }

        // Set pointer and tag
        constexpr void set(uintptr_t ptr, uint16_t tag)
        {
            uintptr_t address = ptr;
            assert((address & 0b11111) == 0 && "Pointer must be aligned to 32 bytes");
            assert(tag <= 0xFFF && "Tag must fit in 12 bits");
            uintptr_t lowTag  = tag & lowBitsMask_;
            uintptr_t highTag = (static_cast<uintptr_t>(tag) & ~lowBitsMask_) << (numVirtAddressBits - numLowBits);
            // Store pointer and tag in the m_taggedPtr
            m_taggedPtr = (address & addressMask_) | highTag | lowTag;
        }

        // Get the raw pointer (removing tag bits and restoring original address)
        template <typename T = void>
        constexpr T* pointer() const
        {
            uintptr_t address = m_taggedPtr & addressMask_;
            // Sign extend from bit 56
            if (address & (1ULL << (numVirtAddressBits - 2)))
            {
                address |= highBitsMask_;
            }
            return std::bit_cast<T*>(address);
        }

        constexpr uintptr_t address() const
        {
            uintptr_t address = m_taggedPtr & addressMask_;
            // Sign extend from bit 56
            if (address & (1ULL << (numVirtAddressBits - 2)))
            {
                address |= highBitsMask_;
            }
            return address;
        }

        constexpr bool operator==(TaggedPointer other) const { return m_taggedPtr == other.m_taggedPtr; }

        template <typename T>
        constexpr bool operator==(T* other) const
        {
            void* ptr = pointer();
            return ptr == other;
        }

        constexpr bool operator==(std::nullptr_t null) const
        {
            void* ptr = pointer();
            return ptr == null;
        }

        // Get the tag
        constexpr uint16_t tag() const
        {
            uint16_t highTag = static_cast<uint16_t>((m_taggedPtr & ~addressMask_) >> (numVirtAddressBits - numLowBits));
            uint16_t lowTag = m_taggedPtr & lowBitsMask_;
            return (highTag | lowTag);
        }
        // Dereference operator
        template <typename T>
        constexpr T& operator*() const
        {
            return *reinterpret_cast<T*>(pointer());
        }

        // Arrow operator
        template <typename T>
        constexpr T* operator->() const
        {
            return reinterpret_cast<T*>(pointer());
        }

    private:
        uintptr_t                  m_taggedPtr        = 0; // Stores the tagged pointer
        static constexpr uint32_t  numLowBits         = 5u;
        static constexpr uint32_t  numHighBits        = 7u;
        static constexpr uint32_t  numVirtAddressBits = 57;
        static constexpr uintptr_t lowBitsMask_       = (1ULL << numLowBits) - 1;
        static constexpr uintptr_t addressMask_  = 0x00FFFFFFFFFFFFFFULL & ~lowBitsMask_; // Low 56 bits for the address
        static constexpr uintptr_t highBitsMask_ = 0xFF00000000000000ULL; // High bits for sign extension
    };
    static_assert(sizeof(void*) == sizeof(TaggedPointer) && alignof(TaggedPointer) == alignof(void*));
    inline constexpr TaggedPointer taggedNullptr;

    template <std::integral I>
    inline constexpr I ceilDiv(I num, I den)
    {
        return (num + den - 1) / den;
    }

    namespace detail {
        struct CtxCtrlBlock
        {
            uintptr_t         ctx = 0;
            std::shared_mutex slk;
        };
        extern DMT_PLATFORM_API std::map<uint64_t, CtxCtrlBlock> g_ctxMap;
        extern DMT_PLATFORM_API std::shared_mutex g_slk;
    } // namespace detail

    template <typename T>
    struct ArrayView
    {
        T*       data;
        uint32_t length; // count of elements
    };

    /**
     * Simple class akin to Python3's <a href="https://docs.python.org/3.10/library/stdtypes.html#range">`range()`</a>
     * Basically `std::views::iota`, but can also be used in Device code
     * @warning device link might give headaches
     */
    template <typename T>
        requires(std::integral<T> || std::floating_point<T>)
    struct DMT_PLATFORM_API Range
    {
    public:
        struct DMT_PLATFORM_API End
        {
            T end;
        };
        struct DMT_PLATFORM_API Iterator
        {
        public:
            using difference_type = std::ptrdiff_t;
            using value_type      = T;
            DMT_CPU_GPU Iterator(T _current, T _step) : m_current(_current), m_step(_step)
            {
                assert(_step != static_cast<T>(0));
            }

            inline DMT_CPU_GPU value_type operator*() const { return m_current; }
            inline DMT_CPU_GPU Iterator&  operator++()
            {
                m_current += m_step;
                return *this;
            }
            inline void operator++(int) { ++*this; }

            inline DMT_CPU_GPU bool operator==(End end) const
            {
                if (m_step < static_cast<T>(0))
                    return m_current <= end.end;
                else
                    return m_current >= end.end;
            }

        private:
            T m_current, m_step;
        };

    public:
        DMT_CPU_GPU Range(T _start, T _end, T _step = static_cast<T>(1)) : m_start(_start), m_end(_end), m_step(_step)
        {
        }

        inline DMT_CPU_GPU Iterator begin() const { return Iterator(m_start, m_step); }
        inline DMT_CPU_GPU End      end() const { return {m_end}; }

    private:
        T m_start, m_end, m_step;
    };
    using Rangef = Range<float>;
    using Rangei = Range<int32_t>;
    static_assert(std::input_iterator<Range<int>::Iterator>, "Failed");

    // https://rigtorp.se/spinlock/
    // should be usable with lock_guard
    // TODO move implememntation to cpp
    // implements the NamedRequireemnt BasicLockable https://en.cppreference.com/w/cpp/named_req/BasicLockable
    struct DMT_PLATFORM_API SpinLock
    {
        bool lock_ = 0;

        void lock() noexcept;
        bool try_lock() noexcept;
        void unlock() noexcept;
    };

    /// @brief @Deprecated Don't use this. This should be removed
    struct DMT_PLATFORM_API CudaSharedMutex
    {
        int lock_ = 0; // 0 means unlocked, >0 for shared locks, -1 for exclusive lock

        // Exclusive lock
        inline DMT_CPU_GPU void lock() noexcept;
        inline DMT_CPU_GPU bool try_lock() noexcept;
        inline DMT_CPU_GPU void unlock() noexcept;

        // Shared lock
        inline DMT_CPU_GPU void lock_shared() noexcept;
        inline DMT_CPU_GPU bool try_lock_shared() noexcept;
        inline DMT_CPU_GPU void unlock_shared() noexcept;
    };

    /// @brief This should be removed
    namespace atomic {
        inline DMT_CPU_GPU int exchange(int* addr, int val) noexcept
        {
#ifdef __CUDA_ARCH__
            return atomicExch(addr, val);
#else
            std::atomic_ref<int> ref(*addr);
            return ref.exchange(val, std::memory_order_acquire);
#endif
        }

        // Utility function for atomic load
        inline DMT_CPU_GPU int load(int* addr) noexcept
        {
#ifdef __CUDA_ARCH__
            return atomicAdd(addr, 0); // Atomic add with 0 for load
#else
            std::atomic_ref<int> ref(*addr);
            return ref.load(std::memory_order_relaxed);
#endif
        }

        // Utility function for atomic compare-and-swap
        inline DMT_CPU_GPU bool compare_exchange(int* addr, int expected, int desired) noexcept
        {
#ifdef __CUDA_ARCH__
            return atomicCAS(addr, expected, desired) == expected;
#else
            std::atomic_ref<int> ref(*addr);
            return ref.compare_exchange_strong(expected, desired, std::memory_order_acquire);
#endif
        }

        // Increment function
        inline DMT_CPU_GPU int increment(int* addr) noexcept
        {
#ifdef __CUDA_ARCH__
            return atomicAdd(addr, 1); // CUDA atomic increment
#else
            std::atomic_ref<int> ref(*addr);
            return ref.fetch_add(1, std::memory_order_acquire);
#endif
        }

        // Decrement function
        inline DMT_CPU_GPU int decrement(int* addr) noexcept
        {
#ifdef __CUDA_ARCH__
            return atomicAdd(addr, -1); // CUDA atomic decrement
#else
            std::atomic_ref<int> ref(*addr);
            return ref.fetch_sub(1, std::memory_order_acquire);
#endif
        }
    } // namespace atomic

    inline DMT_CPU_GPU void CudaSharedMutex::lock() noexcept
    {
        for (;;)
        {
            if (atomic::compare_exchange(&lock_, 0, -1))
            {
                return; // Successfully acquired exclusive lock
            }
            while (atomic::load(&lock_) != 0)
            {
            }
        }
    }

    inline DMT_CPU_GPU bool CudaSharedMutex::try_lock() noexcept { return atomic::compare_exchange(&lock_, 0, -1); }

    inline DMT_CPU_GPU void CudaSharedMutex::unlock() noexcept
    {
        atomic::exchange(&lock_, 0); // Release exclusive lock
    }

    inline DMT_CPU_GPU void CudaSharedMutex::lock_shared() noexcept
    {
        for (;;)
        {
            int current_lock = atomic::load(&lock_);
            if (current_lock >= 0 && atomic::compare_exchange(&lock_, current_lock, current_lock + 1))
            {
                return; // Successfully acquired shared lock
            }
            while (atomic::load(&lock_) < 0)
            {
            }
        }
    }

    inline DMT_CPU_GPU bool CudaSharedMutex::try_lock_shared() noexcept
    {
        int current_lock = atomic::load(&lock_);
        return current_lock >= 0 && atomic::compare_exchange(&lock_, current_lock, current_lock + 1);
    }

    inline DMT_CPU_GPU void CudaSharedMutex::unlock_shared() noexcept
    {
        for (;;)
        {
            int current_lock = atomic::load(&lock_);
            if (current_lock > 0 && atomic::compare_exchange(&lock_, current_lock, current_lock - 1))
            {
                return; // Successfully released shared lock
            }
        }
    }
} // namespace dmt
