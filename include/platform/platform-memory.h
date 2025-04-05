/**
 * @file platform-memory.h
 * @brief header file for the platform-memory.cppm module partition interface
 * @defgroup platform platform Module
 * @{
 */
#pragma once

#include "dmtmacros.h"
#include <platform/platform-macros.h>

#include <platform/platform-logging.h>

#include <array>
#include <bit>
#include <iterator>
#include <map>
#include <memory_resource>
#include <string_view>
#include <thread>

#include <cassert>
#include <compare>
#include <cstdint>

namespace dmt {
    /**
     * Memory tags for meaningful object allocation tracking. Whenever you add something here, make sure to add a string representation
     * in the `memoryTagStr` function
     */
    enum class EMemoryTag : uint16_t
    {
        eUnknown = 0, /** temporary tag when you are unsure of what to tag */
        eDebug,       /** tag for anything which goes into debug memory */
        eEngine,      /** generic tag for whatever buffer is allocated by a system of the application CPU side */
        eHashTable,   /** generic tag associated with an hash table */
        eBuffer,      /** generic tag associated with a dynamically allocated array */
        eBlob,        /** generic tag associated with I/O operations */
        eJob,         /** generic tag associated with job data */
        eQueue,       /** generic tag assiciated with Queues to be read by the GPU */
        eScene,       /** generic tag for whatever data is associated to the scene. Prefer specific ones */
        eAccelerationStructure, /** tag associated to acceleration structures such as BVH nodes */
        eGeometry,              /** tag associated with triangular meshes and other types of geo */
        eMaterial,              /** tag associated with material information */
        eTextures,              /** tag associated with image textures */
        eSpectrum,              /** tag associated for a sampled light spectrum */
        eCount,
    };

    inline constexpr std::strong_ordering operator<=>(EMemoryTag a, EMemoryTag b)
    {
        return toUnderlying(a) <=> toUnderlying(b);
    }

    DMT_PLATFORM_API char const* memoryTagStr(EMemoryTag tag);

    /**
     * Compile time lookup table used to perform the CRC algorithm on a string_view to get an id
     */
    inline constexpr std::array<uint64_t, 256>
        crc64Table{0x0000000000000000uLL, 0x42f0e1eba9ea3693uLL, 0x85e1c3d753d46d26uLL, 0xc711223cfa3e5bb5uLL,
                   0x493366450e42ecdfuLL, 0x0bc387aea7a8da4cuLL, 0xccd2a5925d9681f9uLL, 0x8e224479f47cb76auLL,
                   0x9266cc8a1c85d9beuLL, 0xd0962d61b56fef2duLL, 0x17870f5d4f51b498uLL, 0x5577eeb6e6bb820buLL,
                   0xdb55aacf12c73561uLL, 0x99a54b24bb2d03f2uLL, 0x5eb4691841135847uLL, 0x1c4488f3e8f96ed4uLL,
                   0x663d78ff90e185efuLL, 0x24cd9914390bb37cuLL, 0xe3dcbb28c335e8c9uLL, 0xa12c5ac36adfde5auLL,
                   0x2f0e1eba9ea36930uLL, 0x6dfeff5137495fa3uLL, 0xaaefdd6dcd770416uLL, 0xe81f3c86649d3285uLL,
                   0xf45bb4758c645c51uLL, 0xb6ab559e258e6ac2uLL, 0x71ba77a2dfb03177uLL, 0x334a9649765a07e4uLL,
                   0xbd68d2308226b08euLL, 0xff9833db2bcc861duLL, 0x388911e7d1f2dda8uLL, 0x7a79f00c7818eb3buLL,
                   0xcc7af1ff21c30bdeuLL, 0x8e8a101488293d4duLL, 0x499b3228721766f8uLL, 0x0b6bd3c3dbfd506buLL,
                   0x854997ba2f81e701uLL, 0xc7b97651866bd192uLL, 0x00a8546d7c558a27uLL, 0x4258b586d5bfbcb4uLL,
                   0x5e1c3d753d46d260uLL, 0x1cecdc9e94ace4f3uLL, 0xdbfdfea26e92bf46uLL, 0x990d1f49c77889d5uLL,
                   0x172f5b3033043ebfuLL, 0x55dfbadb9aee082cuLL, 0x92ce98e760d05399uLL, 0xd03e790cc93a650auLL,
                   0xaa478900b1228e31uLL, 0xe8b768eb18c8b8a2uLL, 0x2fa64ad7e2f6e317uLL, 0x6d56ab3c4b1cd584uLL,
                   0xe374ef45bf6062eeuLL, 0xa1840eae168a547duLL, 0x66952c92ecb40fc8uLL, 0x2465cd79455e395buLL,
                   0x3821458aada7578fuLL, 0x7ad1a461044d611cuLL, 0xbdc0865dfe733aa9uLL, 0xff3067b657990c3auLL,
                   0x711223cfa3e5bb50uLL, 0x33e2c2240a0f8dc3uLL, 0xf4f3e018f031d676uLL, 0xb60301f359dbe0e5uLL,
                   0xda050215ea6c212fuLL, 0x98f5e3fe438617bcuLL, 0x5fe4c1c2b9b84c09uLL, 0x1d14202910527a9auLL,
                   0x93366450e42ecdf0uLL, 0xd1c685bb4dc4fb63uLL, 0x16d7a787b7faa0d6uLL, 0x5427466c1e109645uLL,
                   0x4863ce9ff6e9f891uLL, 0x0a932f745f03ce02uLL, 0xcd820d48a53d95b7uLL, 0x8f72eca30cd7a324uLL,
                   0x0150a8daf8ab144euLL, 0x43a04931514122dduLL, 0x84b16b0dab7f7968uLL, 0xc6418ae602954ffbuLL,
                   0xbc387aea7a8da4c0uLL, 0xfec89b01d3679253uLL, 0x39d9b93d2959c9e6uLL, 0x7b2958d680b3ff75uLL,
                   0xf50b1caf74cf481fuLL, 0xb7fbfd44dd257e8cuLL, 0x70eadf78271b2539uLL, 0x321a3e938ef113aauLL,
                   0x2e5eb66066087d7euLL, 0x6cae578bcfe24beduLL, 0xabbf75b735dc1058uLL, 0xe94f945c9c3626cbuLL,
                   0x676dd025684a91a1uLL, 0x259d31cec1a0a732uLL, 0xe28c13f23b9efc87uLL, 0xa07cf2199274ca14uLL,
                   0x167ff3eacbaf2af1uLL, 0x548f120162451c62uLL, 0x939e303d987b47d7uLL, 0xd16ed1d631917144uLL,
                   0x5f4c95afc5edc62euLL, 0x1dbc74446c07f0bduLL, 0xdaad56789639ab08uLL, 0x985db7933fd39d9buLL,
                   0x84193f60d72af34fuLL, 0xc6e9de8b7ec0c5dcuLL, 0x01f8fcb784fe9e69uLL, 0x43081d5c2d14a8fauLL,
                   0xcd2a5925d9681f90uLL, 0x8fdab8ce70822903uLL, 0x48cb9af28abc72b6uLL, 0x0a3b7b1923564425uLL,
                   0x70428b155b4eaf1euLL, 0x32b26afef2a4998duLL, 0xf5a348c2089ac238uLL, 0xb753a929a170f4abuLL,
                   0x3971ed50550c43c1uLL, 0x7b810cbbfce67552uLL, 0xbc902e8706d82ee7uLL, 0xfe60cf6caf321874uLL,
                   0xe224479f47cb76a0uLL, 0xa0d4a674ee214033uLL, 0x67c58448141f1b86uLL, 0x253565a3bdf52d15uLL,
                   0xab1721da49899a7fuLL, 0xe9e7c031e063acecuLL, 0x2ef6e20d1a5df759uLL, 0x6c0603e6b3b7c1cauLL,
                   0xf6fae5c07d3274cduLL, 0xb40a042bd4d8425euLL, 0x731b26172ee619ebuLL, 0x31ebc7fc870c2f78uLL,
                   0xbfc9838573709812uLL, 0xfd39626eda9aae81uLL, 0x3a28405220a4f534uLL, 0x78d8a1b9894ec3a7uLL,
                   0x649c294a61b7ad73uLL, 0x266cc8a1c85d9be0uLL, 0xe17dea9d3263c055uLL, 0xa38d0b769b89f6c6uLL,
                   0x2daf4f0f6ff541acuLL, 0x6f5faee4c61f773fuLL, 0xa84e8cd83c212c8auLL, 0xeabe6d3395cb1a19uLL,
                   0x90c79d3fedd3f122uLL, 0xd2377cd44439c7b1uLL, 0x15265ee8be079c04uLL, 0x57d6bf0317edaa97uLL,
                   0xd9f4fb7ae3911dfduLL, 0x9b041a914a7b2b6euLL, 0x5c1538adb04570dbuLL, 0x1ee5d94619af4648uLL,
                   0x02a151b5f156289cuLL, 0x4051b05e58bc1e0fuLL, 0x87409262a28245bauLL, 0xc5b073890b687329uLL,
                   0x4b9237f0ff14c443uLL, 0x0962d61b56fef2d0uLL, 0xce73f427acc0a965uLL, 0x8c8315cc052a9ff6uLL,
                   0x3a80143f5cf17f13uLL, 0x7870f5d4f51b4980uLL, 0xbf61d7e80f251235uLL, 0xfd913603a6cf24a6uLL,
                   0x73b3727a52b393ccuLL, 0x31439391fb59a55fuLL, 0xf652b1ad0167feeauLL, 0xb4a25046a88dc879uLL,
                   0xa8e6d8b54074a6aduLL, 0xea16395ee99e903euLL, 0x2d071b6213a0cb8buLL, 0x6ff7fa89ba4afd18uLL,
                   0xe1d5bef04e364a72uLL, 0xa3255f1be7dc7ce1uLL, 0x64347d271de22754uLL, 0x26c49cccb40811c7uLL,
                   0x5cbd6cc0cc10fafcuLL, 0x1e4d8d2b65facc6fuLL, 0xd95caf179fc497dauLL, 0x9bac4efc362ea149uLL,
                   0x158e0a85c2521623uLL, 0x577eeb6e6bb820b0uLL, 0x906fc95291867b05uLL, 0xd29f28b9386c4d96uLL,
                   0xcedba04ad0952342uLL, 0x8c2b41a1797f15d1uLL, 0x4b3a639d83414e64uLL, 0x09ca82762aab78f7uLL,
                   0x87e8c60fded7cf9duLL, 0xc51827e4773df90euLL, 0x020905d88d03a2bbuLL, 0x40f9e43324e99428uLL,
                   0x2cffe7d5975e55e2uLL, 0x6e0f063e3eb46371uLL, 0xa91e2402c48a38c4uLL, 0xebeec5e96d600e57uLL,
                   0x65cc8190991cb93duLL, 0x273c607b30f68faeuLL, 0xe02d4247cac8d41buLL, 0xa2dda3ac6322e288uLL,
                   0xbe992b5f8bdb8c5cuLL, 0xfc69cab42231bacfuLL, 0x3b78e888d80fe17auLL, 0x7988096371e5d7e9uLL,
                   0xf7aa4d1a85996083uLL, 0xb55aacf12c735610uLL, 0x724b8ecdd64d0da5uLL, 0x30bb6f267fa73b36uLL,
                   0x4ac29f2a07bfd00duLL, 0x08327ec1ae55e69euLL, 0xcf235cfd546bbd2buLL, 0x8dd3bd16fd818bb8uLL,
                   0x03f1f96f09fd3cd2uLL, 0x41011884a0170a41uLL, 0x86103ab85a2951f4uLL, 0xc4e0db53f3c36767uLL,
                   0xd8a453a01b3a09b3uLL, 0x9a54b24bb2d03f20uLL, 0x5d45907748ee6495uLL, 0x1fb5719ce1045206uLL,
                   0x919735e51578e56cuLL, 0xd367d40ebc92d3ffuLL, 0x1476f63246ac884auLL, 0x568617d9ef46bed9uLL,
                   0xe085162ab69d5e3cuLL, 0xa275f7c11f7768afuLL, 0x6564d5fde549331auLL, 0x279434164ca30589uLL,
                   0xa9b6706fb8dfb2e3uLL, 0xeb46918411358470uLL, 0x2c57b3b8eb0bdfc5uLL, 0x6ea7525342e1e956uLL,
                   0x72e3daa0aa188782uLL, 0x30133b4b03f2b111uLL, 0xf7021977f9cceaa4uLL, 0xb5f2f89c5026dc37uLL,
                   0x3bd0bce5a45a6b5duLL, 0x79205d0e0db05dceuLL, 0xbe317f32f78e067buLL, 0xfcc19ed95e6430e8uLL,
                   0x86b86ed5267cdbd3uLL, 0xc4488f3e8f96ed40uLL, 0x0359ad0275a8b6f5uLL, 0x41a94ce9dc428066uLL,
                   0xcf8b0890283e370cuLL, 0x8d7be97b81d4019fuLL, 0x4a6acb477bea5a2auLL, 0x089a2aacd2006cb9uLL,
                   0x14dea25f3af9026duLL, 0x562e43b4931334feuLL, 0x913f6188692d6f4buLL, 0xd3cf8063c0c759d8uLL,
                   0x5dedc41a34bbeeb2uLL, 0x1f1d25f19d51d821uLL, 0xd80c07cd676f8394uLL, 0x9afce626ce85b507uLL};
    /**
     * the CRC algorithm is an iterative one and needs a starting value.
     */
    inline constexpr uint64_t initialCrc64 = crc64Table[107];

    inline constexpr uint64_t hashCRC64(std::string_view str)
    {
        uint64_t crc = initialCrc64;
        for (char ch : str)
        {
            crc = crc64Table[static_cast<uint8_t>((crc >> 56) ^ static_cast<uint64_t>(ch))] ^ (crc << 8);
        }
        return crc;
    }

    /**
     * Operator to convert an ASCII string literal to its string id form with CRC64 algorithm
     * to be preferred when you need constant time evaluation. If you need to intern the string, use `StringTable`
     * @param str '\0' terminated string
     * @param sz unused, required for the operator to work
     * @return `sid_t` string id
     */
    sid_t consteval operator""_side(char const* str, [[maybe_unused]] uint64_t sz) { return hashCRC64({str, sz}); }

    struct SText
    {
        constexpr SText() : sid(0) {}
        constexpr SText(std::string_view s) : str(s), sid(hashCRC64(str.data())) {}

        std::string_view str;
        sid_t            sid;
    };

    /**
     * Class responsible to intern a given string and give back a unique string identifier `sid_t` for the string, obtained
     * with the CRC64 algorithm
     */
    class alignas(8) StringTable
    {
    public:
        static constexpr uint32_t MAX_SID_LEN = 256;

        DMT_PLATFORM_API StringTable(std::pmr::memory_resource* resource = std::pmr::get_default_resource());
        StringTable(StringTable const&)                = delete;
        StringTable(StringTable&&) noexcept            = delete;
        StringTable& operator=(StringTable const&)     = delete;
        StringTable& operator=(StringTable&&) noexcept = delete;
        ~StringTable() noexcept                        = default;

        /**
         * Method to store the association sid -> string into the `m_stringTable` Red Black Tree
         * @param str string to be interned
         * @return `sid_t` string id
         */
        DMT_PLATFORM_API sid_t intern(std::string_view str);

        /**
         * Method to store the association sid -> string into the `m_stringTable` Red Black Tree
         * @param str string to be interned (not necessarely '\0' terminated)
         * @param sz  number of characters (1 byte each, it's supposed to be ASCII)
         * @return `sid_t` string id
         */
        DMT_PLATFORM_API sid_t intern(char const* str, uint64_t sz);

        /**
         * Method to return an interned string from its sid. If not found, the string `"NOT FOUND"` is returned
         * @param sid string id
         * @return `std::string_view` string associated to the sid
         */
        DMT_PLATFORM_API std::string_view lookup(sid_t sid) const;

    private:
        std::pmr::memory_resource* m_resource;

        /**
         * Red Black Tree storing pairs of associated string ids and ASCII strings. They are stored `'\0'` terminated
         */
        std::pmr::map<sid_t, std::array<char, MAX_SID_LEN>> m_stringTable;

        /**
         * Synchronization primitive to ensure emplaces into the map work in a multithreaded environment
         */
        std::mutex m_mtx;
    };

    /**
     * Enum class containing the possible sizes for the `PageAllocator` class
     */
    enum class EPageSize : uint32_t
    {
        e4KB  = 1u << 12u,
        e2MB  = 1u << 21u,
        e1GB  = 1u << 30u,
        Count = 3
    };

    /**
     * Boilerplate to compare strong enum types
     */
    constexpr auto operator<=>(EPageSize lhs, EPageSize rhs) noexcept
    {
        return toUnderlying(lhs) <=> toUnderlying(rhs);
    }

    // Custom deleter that uses std::pmr::memory_resource
    template <typename T>
        requires(!std::is_array_v<T> || std::is_trivially_destructible_v<std::remove_extent_t<T>>)
    struct PmrDeleter
    {
        std::pmr::memory_resource* resource;
        std::size_t                size = 1; // Default to single object; for arrays, this stores the element count

        void operator()(std::conditional_t<std::is_array_v<T>, std::decay_t<T>, T*> ptr) const
        {
            if (!ptr)
                return;

            if constexpr (!std::is_array_v<T>)
            { // Single object
                ptr->~T();
                resource->deallocate(ptr, sizeof(T), alignof(T));
            }
            else
            { // Array case
                resource->deallocate(ptr, sizeof(std::remove_extent_t<T>) * size, alignof(std::remove_extent_t<T>));
            }
        }
    };

    template <typename T>
        requires(!std::is_array_v<T> || std::is_trivially_destructible_v<std::remove_extent_t<T>>)
    using UniqueRef = std::unique_ptr<T, PmrDeleter<T>>;

    // clang-format off
    struct NullptrTag { };
    inline constexpr NullptrTag nullptrTag;
    // clang-format on

    // For a single object (with construction)
    template <typename T, typename... Args>
        requires(!std::is_array_v<T> && (!std::is_same_v<Args, NullptrTag> && ...))
    UniqueRef<T> makeUniqueRef(std::pmr::memory_resource* resource, Args&&... args)
    {
        void* mem = resource->allocate(sizeof(T), alignof(T));
        T*    obj = new (mem) T(std::forward<Args>(args)...);
        return UniqueRef<T>(obj, PmrDeleter<T>{resource});
    }

    // For an array (without construction)
    template <typename T>
        requires(std::is_trivially_destructible_v<std::remove_extent_t<T>> && std::is_array_v<T>)
    std::unique_ptr<T, PmrDeleter<T>> makeUniqueRef(std::pmr::memory_resource* resource, std::size_t size)
    {
        using ElementType = std::remove_extent_t<T>;
        void*        mem  = resource->allocate(sizeof(ElementType) * size, alignof(ElementType));
        ElementType* arr  = static_cast<ElementType*>(mem); // Memory allocated, but elements NOT constructed

        return UniqueRef<T>(arr, PmrDeleter<T>{resource, size});
    }

    template <typename T>
        requires(!std::is_array_v<T>)
    UniqueRef<T> makeUniqueRef(std::pmr::memory_resource* resource, NullptrTag const)
    {
        return UniqueRef<T>(nullptr, PmrDeleter<T>{resource});
    }

    // For an array (without construction)
    template <typename T>
        requires(std::is_trivially_destructible_v<std::remove_extent_t<T>> && std::is_array_v<T>)
    std::unique_ptr<T, PmrDeleter<T>> makeUniqueRef(std::pmr::memory_resource* resource, std::size_t size, NullptrTag const)
    {
        return UniqueRef<T>(nullptr, PmrDeleter<T>{resource, size});
    }

    namespace os {
        // TODO query functions
        DMT_PLATFORM_API void* reserveVirtualAddressSpace(size_t size);
        DMT_PLATFORM_API bool  commitPhysicalMemory(void* address, size_t size);
        DMT_PLATFORM_API void  decommitPhysicalMemory(void* pageAddress, size_t size);
        DMT_PLATFORM_API bool  freeVirtualAddressSpace(void* address, size_t size);

        DMT_PLATFORM_API void* allocateLockedLargePages(size_t    size,
                                                        EPageSize pageSize     = EPageSize::e2MB,
                                                        bool      skipAclCheck = false);
        DMT_PLATFORM_API void  deallocateLockedLargePages(void*                      address,
                                                          size_t                     size,
                                                          [[maybe_unused]] EPageSize pageSize = EPageSize::e2MB);
    } // namespace os

    enum class EBlockSize : uint32_t
    {
        e64B  = 64u,
        e128B = 128u,
        e256B = 256u,
        e512B = 512u
    };

    class DMT_PLATFORM_API SyncPoolAllocator : public std::pmr::memory_resource
    {
    public:
        SyncPoolAllocator(EMemoryTag tag,
                          size_t     reservedSize,
                          uint32_t   numInitialBlocks = 32,
                          EBlockSize blockSize        = EBlockSize::e256B);
        SyncPoolAllocator(SyncPoolAllocator const&)                = delete;
        SyncPoolAllocator(SyncPoolAllocator&&) noexcept            = delete;
        SyncPoolAllocator& operator=(SyncPoolAllocator const&)     = delete;
        SyncPoolAllocator& operator=(SyncPoolAllocator&&) noexcept = delete;
        ~SyncPoolAllocator() noexcept;

        bool     isValid() const;
        uint32_t numBlocks() const;

    protected:
        void* do_allocate(size_t _Bytes, size_t _Align) override;
        void* NewFunction(uint32_t& blkNum, size_t& blkStart, size_t numBlocksRequired, bool& retFlag);
        void  do_deallocate(void* _Ptr, size_t _Bytes, size_t _Align) override;
        bool  do_is_equal(memory_resource const& _That) const noexcept override;

    private:
        static constexpr uint8_t bitmapStateFree         = 0;
        static constexpr uint8_t bitmapStateOccupied     = 2;
        static constexpr uint8_t bitmapStateLastOccupied = 1;
        static constexpr uint8_t bitmapMask              = 0b11;

        size_t bitmapReservedSize() const;
        size_t bitmapCommittedSize() const;
        struct S
        {
            size_t  offset;
            uint8_t shamt;
        };
        S       bitPairOffsetAndMaskFromBlock(size_t blkIdx) const;
        uint8_t extractBitPairState(size_t blkIdx) const;
        void    setBitPairState(size_t blkIdx, uint8_t state);
        bool    grow(size_t additionalMemory, size_t additionalBitmap);
        void*   scrollAndTagBlocks(size_t startBlkIdx, size_t numBlocksRequired);

    private:
        size_t           m_committedSize;
        size_t           m_reservedSize;
        void*            m_bitmap; // 0 = free, 1 = occupied, end, 2 = occupied, continues on next
        void*            m_memory;
        EMemoryTag       m_tag;
        uint32_t         m_blockSize;
        mutable SpinLock m_mtx;
    };
} // namespace dmt

/** @} */
