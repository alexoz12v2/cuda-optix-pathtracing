#include "platform-utils.h"

#include <array>
#include <bit>
#include <concepts>
#include <shared_mutex>
#include <limits>
#include <thread>

#include <cassert>

namespace dmt {
    namespace detail {
        std::map<uint64_t, CtxCtrlBlock> g_ctxMap;
        std::shared_mutex                g_slk;
    } // namespace detail

    // shared member functions inside platform-specific classes -------------------------------------------------------
    namespace os {
        LibraryLoader::LibraryLoader(bool canGrow, std::pmr::memory_resource* resource, uint32_t initialCapacity) :
        m_resource(resource != nullptr ? resource : std::pmr::get_default_resource()),
        m_searchPathsCapacity(initialCapacity),
        m_canGrow(canGrow)
        {
            if (m_searchPathsCapacity > 0)
            {
                m_searchPaths = reinterpret_cast<Path*>(m_resource->allocate(m_searchPathsCapacity * sizeof(Path)));
            }
            assert(m_resource);
        }

        bool LibraryLoader::isValid() const
        {
            return m_searchPathsCapacity == 0 || (m_resource != nullptr && m_searchPaths != nullptr);
        }

        LibraryLoader::~LibraryLoader() noexcept
        {
            if (m_resource != nullptr && m_searchPaths != nullptr)
            {
                assert(m_searchPathsCapacity > 0);
                m_resource->deallocate(m_searchPaths, m_searchPathsCapacity * sizeof(Path));
            }
        }

        bool LibraryLoader::tryGrow()
        {
            if (!m_canGrow || m_searchPathsCapacity > (std::numeric_limits<uint32_t>::max() / 2))
                return false;

            Path* tmp = reinterpret_cast<Path*>(m_resource->allocate((m_searchPathsCapacity << 1) * sizeof(Path)));
            if (!tmp)
                return false;

            for (uint32_t i = 0; i < m_searchPathsLen; ++i)
            {
                std::construct_at(tmp + i, std::move(m_searchPaths[i]));
                std::destroy_at(m_searchPaths + i);
            }

            m_resource->deallocate(m_searchPaths, m_searchPathsCapacity * sizeof(Path));
            m_searchPaths = tmp;
            m_searchPathsCapacity <<= 1;
            return true;
        }

        bool LibraryLoader::pushSearchPath(Path const& path)
        {
            assert(m_searchPathsLen <= m_searchPathsCapacity && m_resource);
            if (m_searchPathsLen == m_searchPathsCapacity)
            {
                if (!tryGrow())
                    return false;
            }

            std::construct_at(m_searchPaths + m_searchPathsLen, path);
            if (m_searchPathsLen < m_searchPathsCapacity)
            {
                std::construct_at(m_searchPaths + m_searchPathsLen, path);
                if (m_searchPaths[m_searchPathsLen].isValid())
                {
                    ++m_searchPathsLen;
                    return true;
                }
                else
                {
                    std::destroy_at(m_searchPaths + m_searchPathsLen);
                }
            }
            return false;
        }

        bool LibraryLoader::popSearchPath()
        {
            assert(m_searchPathsLen <= m_searchPathsCapacity);
            if (m_searchPathsLen == 0)
                return false;

            std::destroy_at(m_searchPaths + m_searchPathsLen - 1);
            --m_searchPathsLen;
            return true;
        }
    } // namespace os

    // SpinLock -------------------------------------------------------------------------------------------------------
    void SpinLock::lock() noexcept
    {
        for (;;)
        {
            std::atomic_ref lk{lock_};
            // Optimistically assume the lock is free on the first try
            if (!lk.exchange(true, std::memory_order_acquire))
            {
                return;
            }
            // Wait for lock to be released without generating cache misses
            while (lk.load(std::memory_order_relaxed))
            {
                // Issue X86 PAUSE or ARM YIELD instruction to reduce contention between
                // hyper-threads
                //#if defined(DMT_COMPILER_GCC) || defined(DMT_COMPILER_CLANG)
                //                __builtin_ia32_pause();
                //#elif defined(DMT_OS_WINDOWS)
                //                YieldProcessor();
                //#endif
                std::this_thread::yield();
            }
        }
    }

    bool SpinLock::try_lock() noexcept
    {
        // First do a relaxed load to check if lock is free in order to prevent
        // unnecessary cache misses if someone does while(!try_lock())
        std::atomic_ref lk{lock_};
        return !lk.load(std::memory_order_relaxed) && !lk.exchange(true, std::memory_order_acquire);
    }

    void SpinLock::unlock() noexcept
    {
        std::atomic_ref lk{lock_};
        lk.store(false, std::memory_order_release);
    }


} // namespace dmt
