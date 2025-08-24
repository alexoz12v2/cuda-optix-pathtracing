#include "core-texture-cache.h"
#include <memory_resource>

namespace dmt {
    TextureCache::TextureCache(size_t cacheSize, std::pmr::memory_resource* cacheMem, std::pmr::memory_resource* listMem) :
    MipcFiles{cacheMem},
    m_cache{cacheMem},
    m_lruKeyList{listMem},
#if defined(DMT_USE_SYNC_ALLOCATOR)
    m_texMemory{EMemoryTag::eTextures, cacheSize},
#else
    m_texMemory{std::pmr::pool_options{.max_blocks_per_chunk = 32, .largest_required_pool_block = 256}},
#endif
    m_available{cacheSize}
    {
        m_cache.reserve(256);
    }

    void const* TextureCache::getOrInsert(uint64_t baseKey, int32_t mipLevel, uint32_t& outBytes, TexFormat& outTexFormat)
    {
        // 1. shared lock key path
        uint64_t           key = baseKey ^ static_cast<uint64_t>(mipLevel) << 33;
        gx::shareable_lock lk(m_shmtx, gx::lock_mode::shared);
        assert(m_cache.size() == m_lruKeyList.size());
        if (auto it = m_cache.find(key); it != m_cache.cend())
        {
            outBytes     = it->second.numBytes;
            outTexFormat = it->second.texFormat;
            // swap lruKey to last
            auto listit = std::find(m_lruKeyList.begin(), m_lruKeyList.end(), key);
            if (listit == m_lruKeyList.end())
            {
                assert(false && "How did List and HashMap get out of sync");
                return nullptr;
            }
            if (std::next(listit) != m_lruKeyList.end()) // if not last swap to last
            {
                lk.upgrade_to_exclusive();
                m_lruKeyList.splice(m_lruKeyList.end(), m_lruKeyList, listit);
                lk.downgrade_to_shared();
            }

            return reinterpret_cast<unsigned char const*>(it->second.data) + it->second.startOffset;
        }

        // 2. handle cache miss
        lk.upgrade_to_exclusive();

        // upgrade leaves the lock for a moment, therefore someone else might have inserted the object
        if (auto it = m_cache.find(key); it != m_cache.cend())
        {
            outBytes     = it->second.numBytes;
            outTexFormat = it->second.texFormat;
            // swap lruKey to last
            auto listit = std::find(m_lruKeyList.begin(), m_lruKeyList.end(), key);
            if (listit == m_lruKeyList.end())
            {
                assert(false && "How did List and HashMap get out of sync");
                return nullptr;
            }
            if (std::next(listit) != m_lruKeyList.end()) // if not last swap to last
            {
                m_lruKeyList.splice(m_lruKeyList.end(), m_lruKeyList, listit);
                lk.downgrade_to_shared();
            }

            return reinterpret_cast<unsigned char const*>(it->second.data) + it->second.startOffset;
        }

        uint32_t  mipSize            = 0;
        size_t    mipOffset          = 0;
        int32_t   mipUnalignedOffset = 0;
        TexFormat texFormat{};
        if (mipUnalignedOffset = MipcFiles.copyMipOfKey(baseKey, mipLevel, nullptr, &mipSize, &mipOffset, &texFormat);
            mipUnalignedOffset < 0)
        {
            assert(false && "you shouldn't request something which doesn't exist");
            return nullptr; // no file
        }

        // check memory availability
        while (m_available < mipSize && !m_cache.empty())
        {
            assert(!m_lruKeyList.empty());
            // handle eviction and LRU list update
            uint64_t const keyToDel = m_lruKeyList.front();
            auto           it       = m_cache.find(keyToDel);
            if (it == m_cache.end())
            {
                assert(false && "How did List and HashMap get out of sync");
                return nullptr;
            }
            uint32_t const numBytes = it->second.numBytes;
            m_texMemory.deallocate(it->second.data, numBytes, MipcFiles.sectorSize());

            bool const deleted = m_cache.erase(keyToDel);
            assert(deleted);
            m_lruKeyList.pop_front();
            m_available += numBytes;
        }

        if (m_available < mipSize)
        {
            assert(false && "Even when empty the cache cannot accomodate the requested mip");
            return nullptr;
        }

        void* buffer = m_texMemory.allocate(mipSize, MipcFiles.sectorSize());
        if (!buffer)
            return nullptr; // memory failure

        // create entry in LRU map and push_back onto list the key
        auto const& [it, wasInserted] = m_cache.try_emplace(key, baseKey, buffer, mipUnalignedOffset, mipLevel, mipSize, texFormat);

        if (!wasInserted || MipcFiles.copyMipOfKey(baseKey, mipLevel, it->second.data, &mipSize, &mipOffset, nullptr) < 0)
        {
            assert(false && "how did we get here?");
            m_texMemory.deallocate(buffer, mipSize, MipcFiles.sectorSize());
            return nullptr;
        }

        m_lruKeyList.push_back(key);
        assert(m_cache.size() == m_lruKeyList.size());

        // update m_available
        m_available -= mipSize;

        outBytes     = mipSize;
        outTexFormat = it->second.texFormat;
        return reinterpret_cast<unsigned char const*>(it->second.data) + it->second.startOffset;
    }
} // namespace dmt