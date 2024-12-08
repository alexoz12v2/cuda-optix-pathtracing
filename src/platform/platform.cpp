module;

#include <utility>

#include <cstdint>
#include <cstdlib>

module platform;

namespace dmt
{

Platform::Platform() : m_size(4096)
{
    m_buffer = static_cast<decltype(m_buffer)>(std::malloc(m_size));
    if (!m_buffer)
    {
        std::abort();
    }
}

Platform::Platform(Platform&& other) noexcept : m_buffer(std::exchange(other.m_buffer, nullptr))
{
}

Platform& Platform::operator=(Platform&& other) noexcept
{
    std::swap(other.m_buffer, m_buffer);
    return *this;
}

Platform::~Platform() noexcept
{
    free(m_buffer);
}

uint64_t Platform::getSize() const
{
    return m_size;
}

} // namespace dmt