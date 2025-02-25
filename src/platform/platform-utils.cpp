#include "platform-utils.h"

#include <array>
#include <bit>
#include <concepts>
#include <shared_mutex>
#include <limits>

#include <cassert>

namespace dmt {
    namespace detail {
        std::map<uint64_t, CtxCtrlBlock> g_ctxMap;
        std::shared_mutex                g_slk;
    } // namespace detail
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
