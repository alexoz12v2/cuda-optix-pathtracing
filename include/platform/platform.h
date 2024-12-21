#pragma once

#include <cstdint>

#if defined(DMT_INTERFACE_AS_HEADER)
// Keep in sync with .cppm
#include <platform/platform-logging.h>
#include <platform/platform-threadPool.h>

#endif

// TODO move this in an header grouping commonly used macros
#if !defined(DMT_INTERFACE_AS_HEADER)
#define DMT_MODULE_EXPORT export namespace
#else
#define DMT_MODULE_EXPORT namespace
#endif

DMT_MODULE_EXPORT dmt {
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