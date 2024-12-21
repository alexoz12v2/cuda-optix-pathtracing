/**
 * @file platform.cppm
 * @brief Primary interface unit for all utilities necessary to prepare the platform, such as memory allocators,
 * thread pools, ...
 *
 * @defgroup platform platform Module
 * @{
 */

// in the primary module, we need to support both header inclusion and module exports, driven by a macro which
// should be defined by the tranlation unit only when including the interface as an header
#if !defined(DMT_INTERFACE_AS_HEADER)
module;
#endif

#include <cstdint>

#if !defined(DMT_INTERFACE_AS_HEADER)
/**
 * @brief Module `platform`
 */
export module platform;
export import :threadPool;
import :logging;
#else
#include <platform-logging.cppm>
#include <platform-threadPool.cppm>
#endif

#if !defined(DMT_INTERFACE_AS_HEADER)
#define DMT_MODULE_EXPORT export
#else
#define DMT_MODULE_EXPORT
#endif

DMT_MODULE_EXPORT namespace dmt
{

    struct StrBuf;
    class ConsoleLogger;

    /**
 * @class Platform
 * @brief Class whose constructor initializes all the necessary objects to bootstrap the application
 */
    class Platform
    {
    public:
        Platform();
        Platform(Platform const&) = delete;
        Platform(Platform&&) noexcept;
        Platform& operator=(Platform const&) = delete;
        Platform& operator=(Platform&&) noexcept;
        ~Platform() noexcept;

        [[nodiscard]] uint64_t getSize() const;

    private:
        // Threadpool m_threadpool
        // Display m_display
        // ...
        unsigned char* m_buffer = nullptr;
        uint64_t       m_size;
    };

} // namespace dmt
/** @} */