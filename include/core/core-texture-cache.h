#pragma once

#include "core/core-macros.h"
#include "core/core-texture.h"

#include "platform/platform-memory.h"

#include <gx/shared_mutex.h>

#include <list>

// mipc file description
// - header
// - content
// knowing width,height,mipLevels,texFormat, we need no index to compute the offset for a given level, assuming
// we are storing for increasing index of LOD

namespace dmt::mipc {
    inline constexpr uint32_t MAGIC   = 0x4d495043; // MIPC
    inline constexpr uint32_t VERSION = 100;
#pragma pack(push, 1)
    struct Header
    {
        uint32_t  magic;
        uint32_t  version;
        uint64_t  key; /// CRC64 of the file path
        uint64_t  effectiveFileSize;
        uint32_t  width;
        uint32_t  height;
        uint32_t  mipLevels;
        TexFormat texFormat;
    };
#pragma pack(pop)
} // namespace dmt::mipc

namespace dmt {
    DMT_FORCEINLINE uint64_t baseKeyFromPath(os::Path const& path)
    {
        static char                         buffer[1024];
        std::pmr::monotonic_buffer_resource scratch{buffer, 1024}; // default upstream on purpose, avoid crash
        return hashCRC64(path.toUnderlying(&scratch));
    }

    DMT_FORCEINLINE uint64_t baseKeyFromPath(std::string_view path) { return hashCRC64(path); }

    /// - its static method creates, from the `ImageTexturev2` object, a temporary file which will be deleted
    ///   upon process termination. Such a file stores a header of metadata about the uncompressed image, such as
    ///    - pixel format (and size)
    ///    - resolution, number of mip levels
    ///   Followed by the data
    /// - its `readMip` method takes the mip level and outputs a `void*` buffer of the specific pixel format, provided a
    ///   long enough buffer (a call with `nullptr` can estimate its size)
    /// - It is thread safe without any explicit synchronization except for the fact that, if a `MipCacheFile` instance
    ///   exists in any thread, the file is locked and cannot be modified/deleted by anything else in the system
    class MipCacheFile
    {
    public:
        explicit DMT_CORE_API MipCacheFile(std::pmr::memory_resource* mapMemory = std::pmr::get_default_resource());
        MipCacheFile(MipCacheFile const&)                = delete;
        MipCacheFile(MipCacheFile&&) noexcept            = delete;
        MipCacheFile& operator=(MipCacheFile const&)     = delete;
        MipCacheFile& operator=(MipCacheFile&&) noexcept = delete;
        /// closes all handles in the m_keyfdmap, which means deletes all files
        DMT_CORE_API ~MipCacheFile() noexcept;

        // TODO add overload which takes the cache directory
        bool DMT_CORE_API createCacheFile(uint64_t baseKey, ImageTexturev2 const& tex);

        /// if buffer is not nullptr, then inOutOffset and inOutBytes will be used to avoid reading the header twice
        /// when you use nullptr, the function returns the startOffset from the data returned in the end where you actually
        /// see image data. The start is just filler given by the fact we are reading with a file offset multiple of sector size
        /// (we are disabling file buffering)
        /// note: inOutBytes being uint32 means assuming all mips occupy max 4 GB
        /// Uses a static, common buffer to read the header -> *NOT* thread safe (TextureCache should use exclusive lock)
        /// when called with nullptr, it also returns the texFormat
        int32_t DMT_CORE_API copyMipOfKey(uint64_t   fileKey,
                                          int32_t    level,
                                          void*      outBuffer,
                                          uint32_t*  inOutBytes,
                                          size_t*    inOutOffset,
                                          TexFormat* outFormat) const;

    private:
        /// associate the file key to its file descriptor/HANDLE
        std::pmr::unordered_map<uint64_t, uintptr_t> m_keyfdmap;
        size_t                                       m_sectorSize;
    };
    static_assert(alignof(MipCacheFile) == 8);

    /// LRU cache which maintains
    /// - a hash table which associates {file path, mip level} -> address of allocated buffer,
    ///   where the allocated buffer is the desired mip of a texture. Such a buffer is managed by taking a large portion of
    ///   memory, whose size is predefined at start, and creates a pool allocator whose base element is a tile of 32x32 bytes,
    ///   since we know we are caching large texture
    /// - a singly linked list, allocated in an arena, whose maximum capacity is determined by the maximum number of entries
    ///   the cache can have (computed as a worst case scenario from maximum bytes and tile size)
    ///   This list keeps track of the LRU order, such that the least recently used can be deleted with a `pop_front`
    /// - a shared mutex allows for shared access for trying to get an element from the cache in read only mode,
    ///   while exclusive access is required when an insertion or a lookup requires you to modify the cache
    class TextureCache
    {
    public:
        DMT_CORE_API TextureCache(size_t                     cacheSize = 1ull << 30, // 1 GB
                                  std::pmr::memory_resource* cacheMem  = std::pmr::get_default_resource(),
                                  std::pmr::memory_resource* listMem   = std::pmr::get_default_resource());

        struct Entry
        {
            uint64_t  baseKey;
            void*     data;
            int32_t   startOffset;
            int32_t   miplevel;
            uint32_t  numBytes; // assuming each mip is less than 4 GB
            TexFormat texFormat;
        };

    public:
        DMT_CORE_API void const* getOrInsert(uint64_t baseKey, int32_t mipLevel, uint32_t& outBytes, TexFormat& outTexFormat);

    public:
        /// only one thread, the one parsing the scene before kicking jobs onto workers, should interact with this
        MipCacheFile MipcFiles;

    private:
        /// store associations {path, miplevel} -> pointer to buffer inside texMemory
        /// the key is composed with hashCRC64(path) someBitwiseOp mipLevel
        std::pmr::unordered_map<uint64_t, Entry> m_cache;
        /// store a list of keys in LRU order, such that pop_front gives you the key to evict from the cache and therefore
        /// the address and size of the buffer to deallocate in texMemory
        std::pmr::list<uint64_t> m_lruKeyList;
        /// subclass of memory_resource and used to allocate/deallocate mips
        /// MAYBE TODO: swap this for an allocator which tries to allocate large pages.
        SyncPoolAllocator m_texMemory;
        /// mutex used for cache reads in shared access, exclusive access when inserting/evicting from the cache
        mutable gx::shared_mutex m_shmtx;
        /// track approximate available size in texMemory
        size_t m_available;
    };
} // namespace dmt
