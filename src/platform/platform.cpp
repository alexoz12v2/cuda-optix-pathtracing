module;

#include <utility>

#include <cstdint>
#include <cstdlib>

module platform;

namespace dmt
{

Platform::Platform()
{
}

Platform::Platform(Platform&&) noexcept
{
}

Platform& Platform::operator=(Platform&&) noexcept
{
    return *this;
}

Platform::~Platform() noexcept
{
}

} // namespace dmt