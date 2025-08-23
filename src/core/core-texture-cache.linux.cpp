#include "core-texture-cache.h"

#include "platform/platform-utils.h"

#include <memory_resource>
#include <string>

#include <fcntl.h> // stat
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

namespace /*static*/ {
    class SectorBufferedWriter
    {
    public:
        SectorBufferedWriter(int fd, size_t sectorSize, size_t bufferMultiplier) :
        m_fd(fd),
        m_sectorSize(sectorSize),
        m_bufferSize(sectorSize * bufferMultiplier)
        {
            // allocate aligned buffer for O_DIRECT
            posix_memalign(reinterpret_cast<void**>(&m_buffer), sectorSize, m_bufferSize);
        }

        ~SectorBufferedWriter()
        {
            flush();
            free(m_buffer);
        }

        bool write(void const* data, size_t bytes)
        {
            unsigned char const* src          = reinterpret_cast<unsigned char const*>(data);
            size_t               writtenTotal = 0;

            while (writtenTotal < bytes)
            {
                size_t space  = m_bufferSize - m_offset;
                size_t toCopy = std::min(space, bytes - writtenTotal);

                std::memcpy(m_buffer + m_offset, src + writtenTotal, toCopy);
                m_offset += toCopy;
                writtenTotal += toCopy;

                if (m_offset == m_bufferSize)
                {
                    if (!flush())
                        return false;
                }
            }
            return true;
        }

        bool flush(bool padToSector = true)
        {
            if (m_offset == 0)
                return true;

            size_t writeSize = m_offset;
            if (padToSector)
            {
                writeSize = (writeSize + m_sectorSize - 1) & ~(m_sectorSize - 1);
            }
            if (writeSize > m_bufferSize)
                return false;

            if (writeSize > m_offset)
            {
                std::memset(m_buffer + m_offset, 0, writeSize - m_offset);
            }

            ssize_t written = pwrite(m_fd, m_buffer, writeSize, m_fileOffset);
            if (written != static_cast<ssize_t>(writeSize))
                return false;

            m_fileOffset += written;
            m_offset = 0;
            return true;
        }

    private:
        int            m_fd;
        size_t         m_sectorSize;
        size_t         m_bufferSize;
        size_t         m_offset     = 0;
        off_t          m_fileOffset = 0;
        unsigned char* m_buffer     = nullptr;
    };

} // namespace

namespace dmt {
    MipCacheFile::MipCacheFile(std::pmr::memory_resource* mapMemory) : m_keyfdmap{mapMemory}, m_sectorSize{0} {}

    bool MipCacheFile::createCacheFile(uint64_t baseKey, ImageTexturev2 const& tex)
    {
        std::pmr::monotonic_buffer_resource mem{1024};

        std::pmr::string tempMIPCDirectory = os::env::get("DMT_LINUX_MIPC_CACHE_LOCATION", &mem);
        if (tempMIPCDirectory.empty())
        {
            tempMIPCDirectory = "~/.cache/dmt-mipc/";
        }
        if (!tempMIPCDirectory.ends_with('/'))
            tempMIPCDirectory.push_back('/');

        tempMIPCDirectory.append(std::to_string(getpid()));

        // if directory doesn't exist, create it (with parents)
        struct stat sb{};
        int         fd = open(tempMIPCDirectory.c_str(), O_RDWR);
        if (fd < 0)
            return false;

        if (fstat(fd, &sb))
        {
            // doesn't exist, make directory
            if (mkdir(tempMIPCDirectory.c_str(), 0644))
                return false;
            close(fd);
            fd = open(tempMIPCDirectory.c_str(), O_RDWR);
        }
        else if (!S_ISDIR(sb.st_mode))
        {
            // exists and is not directory, therefore error
            close(fd);
            return false;
        }
        close(fd);

        // create new .mipc file by asssembling its name
        tempMIPCDirectory.push_back('/');
        tempMIPCDirectory.append(std::to_string(baseKey));

        // create file and mark it such that it is closed when this class is destructed or when the process
        // is forcefully terminated
        // O_CREAT | O_EXCL -> fail if file already exists
        // S_IRUSR | S_IWURS -> only owner can read and write
        // Requirements for O_DIRECT (unbuffered I/O):
        // - File offsets and buffer addresses must be aligned to the disk sector (usually 512 or 4096 bytes).
        // - File size and writes should be multiples of sector size.
        fd = open(tempMIPCDirectory.c_str(), O_RDWR | O_CREAT, O_EXCL, S_IRUSR | S_IWUSR);
        if (fd < 0)
            return false;
        // delete directory entry immediately. The fill will live until this file descriptor remains open
        unlink(tempMIPCDirectory.c_str());

        // 1. Determine sector size
        struct statvfs vfs{};
        if (fstatvfs(fd, &vfs) != 0)
            return false;
        size_t sectorSize = vfs.f_frsize; // filesystem block size

        SectorBufferedWriter writer(fd, sectorSize, 4); // buffer = 4 sectors

        // 2. Write header
        mipc::Header const header{.magic   = mipc::MAGIC,
                                  .version = mipc::VERSION,
                                  .key     = baseKey,
                                  .effectiveFileSize = sizeof(mipc::Header) + mipChainPixelCount(tex.width, tex.height) *
                                                                                  bytesPerPixel(tex.texFormat),
                                  .width     = static_cast<uint32_t>(tex.width),
                                  .height    = static_cast<uint32_t>(tex.height),
                                  .mipLevels = static_cast<uint32_t>(tex.mipLevels),
                                  .texFormat = tex.texFormat};
        if (!writer.write(&header, sizeof(header)))
            return false;

        // 3. Write mip levels sequentially
        size_t pixelSize = bytesPerPixel(tex.texFormat);
        for (int32_t level = 0; level < tex.mipLevels; ++level)
        {
            uint32_t             mipSize = mipBytes(tex.width, tex.height, level, tex.texFormat);
            unsigned char const* mipData = reinterpret_cast<unsigned char const*>(tex.data) +
                                           mortonLevelOffset(tex.width, tex.height, level) * pixelSize;

            if (!writer.write(mipData, mipSize))
                return false;
        }

        // 4. Flush remaining
        if (!writer.flush())
            return false;


        // add it to fdmap
        auto const& [it, wasInserted] = m_keyfdmap.try_emplace(baseKey, static_cast<uintptr_t>(fd));
        if (!wasInserted)
            close(fd);

        return wasInserted;
    }

    int32_t MipCacheFile::copyMipOfKey(uint64_t   fileKey,
                                       int32_t    level,
                                       void*      outBuffer,
                                       uint32_t*  inOutBytes,
                                       size_t*    inOutOffset,
                                       TexFormat* outFormat) const
    {
        if (!m_keyfdmap.contains(fileKey))
            return -1;

        int fd = static_cast<int>(m_keyfdmap.at(fileKey));
        if (fd < 0)
            return -1;

        assert(inOutBytes && inOutOffset);

        if (!outBuffer)
        {
            // --- Read header to compute offsets ---
            mipc::Header header{};
            size_t       headerAlignedSize = (sizeof(header) + m_sectorSize - 1) & ~(m_sectorSize - 1);

            unsigned char* headerBuffer = nullptr;
            posix_memalign(reinterpret_cast<void**>(&headerBuffer), m_sectorSize, headerAlignedSize);
            if (!headerBuffer)
                return -1;

            ssize_t bytesRead = pread(fd, headerBuffer, headerAlignedSize, 0);
            if (bytesRead != static_cast<ssize_t>(headerAlignedSize))
            {
                free(headerBuffer);
                return -1;
            }

            std::memcpy(&header, headerBuffer, sizeof(header));
            free(headerBuffer);

            // --- Compute mip offset and size ---
            size_t mipOffsetUnaligned = sizeof(header) + mortonLevelOffset(header.width, header.height, level) *
                                                             bytesPerPixel(header.texFormat);
            uint32_t mipSizeUnaligned = mipBytes(header.width, header.height, level, header.texFormat);

            // --- Align offset down, size up ---
            size_t   offsetAligned = (mipOffsetUnaligned / m_sectorSize) * m_sectorSize;
            uint32_t sizeAligned   = static_cast<uint32_t>(
                ((mipOffsetUnaligned + mipSizeUnaligned + m_sectorSize - 1) / m_sectorSize) * m_sectorSize - offsetAligned);

            *inOutBytes  = sizeAligned;
            *inOutOffset = offsetAligned;
            if (outFormat)
                *outFormat = header.texFormat;

            return static_cast<int32_t>(mipOffsetUnaligned - offsetAligned);
        }
        else
        {
            // --- Read actual mip into provided buffer ---
            ssize_t bytesRead = pread(fd, outBuffer, *inOutBytes, *inOutOffset);
            if (bytesRead != static_cast<ssize_t>(*inOutBytes))
                return -1;

            return 0;
        }
    }

    MipCacheFile::~MipCacheFile() noexcept
    {
        for (auto const& [key, value] : m_keyfdmap)
        {
            close(static_cast<int>(value));
        }
    }
} // namespace dmt