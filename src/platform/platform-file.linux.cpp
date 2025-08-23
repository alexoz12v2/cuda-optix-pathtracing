#include "platform-file.h"
#include "platform-context.h"

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <cassert>
#include <cstring>
#include <cerrno>
#include <system_error>
#include <cstdlib>

namespace dmt::os {

    struct LinuxChunkedFileReader
    {
        int      fd        = -1;
        size_t   fileSize  = 0;
        uint32_t chunkSize = 0;
        uint32_t numChunks = 0;
    };

    ChunkedFileReader::ChunkedFileReader(char const* filePath, uint32_t chunkSize, std::pmr::memory_resource* resource) :
    m_data(),
    m_resource(resource)
    {
        assert(m_resource);
        dmt::Context            ctx;
        LinuxChunkedFileReader& data = *reinterpret_cast<LinuxChunkedFileReader*>(&m_data);

        if (chunkSize < 512 || (chunkSize & (chunkSize - 1)) != 0)
        {
            if (ctx.isValid())
                ctx.error("Invalid Chunk Size. Must be power of 2 >= 512", {});
            data.fd = -1;
            return;
        }

        int fd = ::open(filePath, O_RDONLY | O_DIRECT);
        if (fd < 0)
        {
            if (ctx.isValid())
                ctx.error("open failed: {}", std::make_tuple(strerror(errno)));
            data.fd = -1;
            return;
        }

        struct stat st{};
        if (fstat(fd, &st) < 0)
        {
            if (ctx.isValid())
                ctx.error("fstat failed: {}", std::make_tuple(strerror(errno)));
            ::close(fd);
            data.fd = -1;
            return;
        }

        data.fd        = fd;
        data.fileSize  = static_cast<size_t>(st.st_size);
        data.chunkSize = chunkSize;
        data.numChunks = static_cast<uint32_t>((data.fileSize + chunkSize - 1) / chunkSize);
    }

    // Destructor
    ChunkedFileReader::~ChunkedFileReader() noexcept
    {
        LinuxChunkedFileReader& data = *reinterpret_cast<LinuxChunkedFileReader*>(&m_data);
        if (data.fd >= 0)
        {
            ::close(data.fd);
            data.fd = -1;
        }
    }

    // Synchronous chunk read
    bool ChunkedFileReader::requestChunk(void* chunkBuffer, uint32_t chunkNum)
    {
        LinuxChunkedFileReader& data = *reinterpret_cast<LinuxChunkedFileReader*>(&m_data);
        dmt::Context            ctx;

        assert(chunkNum < data.numChunks);

        off_t   offset    = static_cast<off_t>(chunkNum) * data.chunkSize;
        ssize_t bytesRead = ::pread(data.fd, chunkBuffer, data.chunkSize, offset);

        if (bytesRead < 0)
        {
            if (ctx.isValid())
                ctx.error("pread failed: {}", std::make_tuple(strerror(errno)));
            return false;
        }

        // store last bytes read in the dummy udata field
        data.chunkSize = static_cast<uint32_t>(bytesRead);
        return true;
    }

    // Wait does nothing in sync mode
    bool ChunkedFileReader::waitForPendingChunk(uint32_t) { return true; }
    bool ChunkedFileReader::waitForPendingChunk() { return true; }

    // Last bytes read
    uint32_t ChunkedFileReader::lastNumBytesRead()
    {
        LinuxChunkedFileReader& data = *reinterpret_cast<LinuxChunkedFileReader*>(&m_data);
        return data.chunkSize;
    }

    ChunkedFileReader::operator bool() const
    {
        LinuxChunkedFileReader const& data = *reinterpret_cast<LinuxChunkedFileReader const*>(&m_data);
        return data.fd >= 0;
    }

    uint32_t ChunkedFileReader::numChunks() const
    {
        LinuxChunkedFileReader const& data = *reinterpret_cast<LinuxChunkedFileReader const*>(&m_data);
        return data.numChunks;
    }

    size_t ChunkedFileReader::computeAlignedChunkSize(size_t chunkSize)
    {
        static size_t const alignment = 512; // O_DIRECT alignment
        return ((chunkSize + alignment - 1) / alignment) * alignment;
    }

} // namespace dmt::os