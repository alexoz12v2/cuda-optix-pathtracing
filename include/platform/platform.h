#pragma once

#include "dmtmacros.h"

#include <cstdint>

#if defined(DMT_INTERFACE_AS_HEADER)
// Keep in sync with .cppm
#include <platform/platform-logging.h>

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