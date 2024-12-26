#pragma once
#if defined(DMT_COMPILER_MSVC)
#pragma warning(disable : 4005 5106) // SHut SAL macro redefinition up
#endif

#include "dmtmacros.h"

#include <concepts>
#include <iterator>
#include <source_location>

#include <cmath>
#include <cstdint>
#include <cstdio>

#if defined(DMT_OS_WINDOWS)
// clang-format off
#pragma comment(lib, "mincore")
#define NO_SAL
#include <Windows.h>
#include <AclAPI.h>
#include <securitybaseapi.h>
#include <sysinfoapi.h>
// clang-format on
#elif defined(DMT_OS_LINUX)
#endif

#if defined(DMT_INTERFACE_AS_HEADER)
// Keep in sync with .cppm
#include <platform/platform-logging.h>
#else
import <platform/platform-logging.h>;
#endif

DMT_MODULE_EXPORT dmt {
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

    void* alignTo(void* address, size_t alignment)
    {
        // Ensure alignment is a power of two (required for bitwise operations).
        size_t const mask = alignment - 1;
        assert((alignment & mask) == 0 && "Alignment must be a power of two.");

        uintptr_t addr        = reinterpret_cast<uintptr_t>(address);
        uintptr_t alignedAddr = (addr + mask) & ~mask;

        return reinterpret_cast<void*>(alignedAddr);
    }

    uintptr_t alignToAddr(uintptr_t address, size_t alignment)
    {
        // Ensure alignment is a power of two (required for bitwise operations).
        size_t const mask = alignment - 1;
        assert((alignment & mask) == 0 && "Alignment must be a power of two.");

        uintptr_t alignedAddr = (address + mask) & ~mask;

        return alignedAddr;
    }

    void* alignToBackward(void* address, size_t alignment)
    {
        // Ensure alignment is a power of two (required for bitwise operations).
        size_t const mask = alignment - 1;
        assert((alignment & mask) == 0 && "Alignment must be a power of two.");

        uintptr_t addr        = reinterpret_cast<uintptr_t>(address);
        uintptr_t alignedAddr = addr & ~mask;

        return reinterpret_cast<void*>(alignedAddr);
    }

// - x86_64 systems actually use 48 bits for virtual addresses. Actually, scratch that, with the
//   latest PML5 (https://en.wikipedia.org/wiki/Intel_5-level_paging) extended virtual adderesses
//   to 57 bits. This means that the high 7 bits of a memory address are unused, and we can make good use of them
// - adding to the fact that minimum block size is 32 Bytes, hence aligned to a 32 Byte boundary, we have an additional
//   5 bits free to use
// hence, our tagged pointers can exploit 12 bits of information in total
// Remember: it holds only for host addresses, and to regain access to the original address, you need to mask out
// the low bits (5), and sign extend from bit 56 to bit 63
// Reference test code:
//   alignas(32) int data  = 42; // Ensure alignment
//   uint16_t      trueTag = (1u << 12u) - 1;
//   TaggedPointer tp(&data, trueTag);
//   std::cout << "True Tag 0x" << std::hex << trueTag << std::dec << '\n';
//   std::cout << "Raw pointer: " << tp.getPointer() << "\n";
//   std::cout << "True Pointer: " << &data << '\n';
//   std::cout << "Tag: 0x" << std::hex << tp.getTag() << "\n";
//   std::cout << "Dereferenced value: " << std::dec << tp.operator* <int>() << "\n";
// TODO: we can template this class on the number of low bits we expect to be zeroed out
#if !defined(DMT_ARCH_X86_64)
#error "Pointer Tagging relies heavily on x86_64's virtual addreess format"
#endif
    class TaggedPointer
    {
    public:
        // Constructor
        constexpr TaggedPointer(std::nullptr_t null = nullptr) : m_taggedPtr(0)
        {
        }

        constexpr TaggedPointer(void* ptr, uint16_t tag = 0)
        {
            set(std::bit_cast<uintptr_t>(ptr), tag);
        }

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

        constexpr bool operator==(TaggedPointer other) const
        {
            return m_taggedPtr == other.m_taggedPtr;
        }

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

    /**
     * Convenience pointer type to pass around systems of the application to get an enriched/targeted
     * interface to the platform
     * @warning I don't like the fact that we are redunding the logger interface, in particular the
     * redundant `m_level`
     */
    class PlatformContext : public BaseLogger<PlatformContext>
    {
    public:
        // -- Types --
        /**
         * Required stuff from the `BaseLogger`
         */
        struct Traits
        {
            static constexpr ELogDisplay displayType = ELogDisplay::Forward;
        };

        /**
         * Function pointer table stored elsewhere. It is supposed to outlive the context.
         * Meant for functions which are not called often, therefore they can afford the
         * double indirection
         */
        struct Table
        {
            void (*changeLevel)(void* data, ELogLevel level) = [](void* data, ELogLevel level) {};
        };

        /**
         * Function pointer table stored inline here. Meant for functions which are called
         * often
         */
        struct InlineTable
        {
            void (*write)(void* data, ELogLevel level, std::string_view const& str, std::source_location const& loc) =
                [](void* data, ELogLevel level, std::string_view const& str, std::source_location const& loc) {};
            void (*writeArgs)(void*                                data,
                              ELogLevel                            level,
                              std::string_view const&              str,
                              std::initializer_list<StrBuf> const& list,
                              std::source_location const&          loc) =
                [](void*                                data,
                   ELogLevel                            level,
                   std::string_view const&              str,
                   std::initializer_list<StrBuf> const& list,
                   std::source_location const&          loc) {};
            bool (*checkLevel)(void* data, ELogLevel level) = [](void* data, ELogLevel level) { return false; };
        };

        PlatformContext(void* data, Table const* pTable, InlineTable const& inlineTable) :
        m_table(pTable),
        m_inlineTable(inlineTable),
        m_data(data)
        {
        }

        /**
         * Setter for the `m_level`
         * @param level new level
         * @warning Purposefully name hiding the `BaseLogger`
         */
        void setLevel(ELogLevel level)
        {
            m_table->changeLevel(m_data, level);
        }

        /**
         * Write function mandated by the CRTP pattern of the class `BaseLogger`
         * @param level log level
         * @param str string to output
         * @param loc location of the log
         */
        void write(ELogLevel level, std::string_view const& str, std::source_location const& loc)
        {
            m_inlineTable.write(m_data, level, str, loc);
        }

        /**
         * Write function mandated by the CRTP pattern of the class `BaseLogger`
         * @param level log level
         * @param str format string
         * @param list list of arguments which will be used to create the final string
         * @param loc location of the log
         */
        void write(ELogLevel                            level,
                   std::string_view const&              str,
                   std::initializer_list<StrBuf> const& list,
                   std::source_location const&          loc)
        {
            m_inlineTable.writeArgs(m_data, level, str, list, loc);
        }

        /**
         * CRTP overridden function to check if the true underlying logger is enabled on the log level
         * @param level log level requested
         * @return bool signaling whether the requested log level is enabled
         */
        bool enabled(ELogLevel level)
        {
            return m_inlineTable.checkLevel(m_data, level);
        }

    private:
        /**
         * table of infrequent functions offloaded, stored stored elsewhere
         */
        Table const* m_table;

        /**
         * Table of frequent functions stored here
         */
        InlineTable m_inlineTable;

        /**
         * Pointer to a type erased class, which can be casted back in the function pointer
         * functions, like `Platform`
         */
        void* m_data = nullptr;
    };

    // TODO remove this templating tagged pointer
    class alignas(32) ChunkedFileReader
    {
    public:
        struct ChunkInfo
        {
            void*    buffer;
            uint32_t numBytesRead;
            uint32_t chunkNum;
        };

    private:
        struct EndSentinel
        {
        };
        struct InputIterator
        {
        public:
            using difference_type = std::ptrdiff_t;
            using value_type      = ChunkInfo;

            InputIterator(void* pData) : m_pData(pData)
            {
            }

            ChunkInfo operator*() const;

            bool           operator==(EndSentinel const&) const;
            InputIterator& operator++();
            void           operator++(int)
            {
                ++*this;
            }

        private:
            void*    m_pData;
            uint32_t m_chunkNum;
        };

        static_assert(std::input_iterator<InputIterator>);
        friend struct InputIterator;

    public:
        static constexpr uint32_t maxNumBuffers = 72;
        static constexpr uint32_t size          = 64;
        static constexpr uint32_t alignment     = 8;
        ChunkedFileReader(PlatformContext& pctx, char const* filePath, uint32_t chunkSize);
        ChunkedFileReader(PlatformContext& pctx, char const* filePath, uint32_t chunkSize, uint8_t numBuffers, uintptr_t* pBuffers);
        ChunkedFileReader(ChunkedFileReader const&)                = delete;
        ChunkedFileReader(ChunkedFileReader&&) noexcept            = delete;
        ChunkedFileReader& operator=(ChunkedFileReader const&)     = delete;
        ChunkedFileReader& operator=(ChunkedFileReader&&) noexcept = delete;
        ~ChunkedFileReader() noexcept;

        bool     requestChunk(PlatformContext& pctx, void* chunkBuffer, uint32_t chunkNum);
        bool     waitForPendingChunk(PlatformContext& pctx, uint32_t timeoutMillis);
        uint32_t lastNumBytesRead();

        static size_t computeAlignedChunkSize(size_t chunkSize);

    private:
        alignas(alignment) unsigned char m_data[size];
    };
}

// module-private bits of functionality. Should not be exported by the primary interface unit
// note: since this is to be visible to all implementation units, it cannot be put into a .cpp file, as
// implementation units are not linked together. It needs to stay here. It should not be included in the
// binary interface, hence it is fine
namespace dmt {
    template <std::integral I>
    constexpr I ceilDiv(I num, I den)
    {
        return (num + den - 1) / den;
    }

#if defined(DMT_OS_WINDOWS)
    void* reserveVirtualAddressSpace(size_t size)
    {
        void* address = VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_READWRITE);
        return address; // to check whether it is different than nullptr
    }

    size_t systemAlignment()
    {
        SYSTEM_INFO sysInfo{};
        GetSystemInfo(&sysInfo);
        return static_cast<size_t>(sysInfo.dwAllocationGranularity);
    }

    bool commitPhysicalMemory(void* address, size_t size)
    {
        void* committed = VirtualAlloc(address, size, MEM_COMMIT, PAGE_READWRITE);
        return committed != nullptr;
    }

    bool freeVirtualAddressSpace(void* address, size_t size) // true if success
    {
        return VirtualFree(address, 0, MEM_RELEASE);
    }

    void decommitPage(void* pageAddress, size_t pageSize)
    {
        VirtualFree(pageAddress, pageSize, MEM_DECOMMIT);
    }

    namespace win32 {

        uint32_t getLastErrorAsString(char* buffer, uint32_t maxSize)
        {
            //Get the error message ID, if any.
            DWORD errorMessageID = ::GetLastError();
            if (errorMessageID == 0)
            {
                buffer[0] = '\n';
                return 0;
            }
            else
            {

                LPSTR messageBuffer = nullptr;

                //Ask Win32 to give us the string version of that message ID.
                //The parameters we pass in, tell Win32 to create the buffer that holds the message for us (because we don't yet know how long the message string will be).
                size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                                                 FORMAT_MESSAGE_IGNORE_INSERTS,
                                             NULL,
                                             errorMessageID,
                                             MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                             (LPSTR)&messageBuffer,
                                             0,
                                             NULL);

//Copy the error message into a std::string.
#undef min
                size_t actual = std::min(static_cast<size_t>(maxSize - 1), size);
                std::memcpy(buffer, messageBuffer, actual);
                buffer[actual] = '\0';

                //Free the Win32's string's buffer.
                LocalFree(messageBuffer);
                return actual;
            }
        }

        constexpr bool luidCompare(LUID const& luid0, LUID const& luid1)
        {
            return luid0.HighPart == luid1.HighPart && luid1.LowPart == luid0.LowPart;
        }

    } // namespace win32
#elif defined(DMT_OS_LINUX)
    void* reserveVirtualAddressSpace(size_t size)
    {
        void* address = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (address == MAP_FAILED)
        {
            return nullptr;
        }
        return address;
    }

    bool commitPhysicalMemory(void* address, size_t size)
    {
        int result = mprotect(address, size, PROT_READ | PROT_WRITE);
        return result == 0;
    }

    bool freeVirtualAddressSpace(void* address, size_t size)
    {
        return !munmap(address, size);
    }

    void decommitPage(void* pageAddress, size_t pageSize)
    {
        mprotect(pageAddress, pageSize, PROT_NONE);
        madvise(pageAddress, pageSize, MADV_DONTNEED); // Optional: Release physical memory
    }

    size_t systemAlignment()
    {
        // TODO
        return 0;
    }

    namespace linux {
    }
#endif
} // namespace dmt
