/**
 * @file platform.cppm
 * @brief Primary interface unit for all utilities necessary to prepare the platform, such as memory allocators,
 * thread pools, ...
 *
 * @defgroup platform platform Module
 * @{
 */
module;

#include <cstdint>

/**
 * @brief Module `platform`
 */
export module platform;

import :logging;

export namespace dmt
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