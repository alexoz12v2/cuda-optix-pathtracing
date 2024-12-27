#pragma once

#include "dmtmacros.h"

#include <bit>
#include <concepts>
#include <iterator>
#include <source_location>

#include <cassert>
#include <cmath>
#include <cstdint>

namespace dmt {
    void* reserveVirtualAddressSpace(size_t size);

    size_t systemAlignment();

    bool commitPhysicalMemory(void* address, size_t size);

    bool freeVirtualAddressSpace(void* address, size_t size);

    void decommitPage(void* pageAddress, size_t pageSize);
} // namespace dmt

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

    template <std::integral I>
    constexpr I ceilDiv(I num, I den)
    {
        return (num + den - 1) / den;
    }
}
