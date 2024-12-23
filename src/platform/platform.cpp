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

uint64_t Platform::getSize() const
{
    return 4096;
}

Platform& Platform::operator=(Platform&&) noexcept
{
    return *this;
}

Platform::~Platform() noexcept
{
}

} // namespace dmt