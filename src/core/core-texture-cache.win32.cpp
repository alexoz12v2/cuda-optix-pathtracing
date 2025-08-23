#include "core-texture-cache.h"

#include <Windows.h>

#include <fileapi.h>
#include <processthreadsapi.h>
#include <WinBase.h> // file flags
#include <sddl.h> // https://learn.microsoft.com/en-us/windows/win32/api/sddl/nf-sddl-convertstringsecuritydescriptortosecuritydescriptorw

namespace dmt {

    inline constexpr uint32_t staticBytes = 8192;
    static unsigned char      s_tempBuffer[staticBytes]{};

    static uint32_t pidUpTo5WChars(wchar_t* out)
    {
        uint32_t id = GetCurrentProcessId();

        // Store digits in a temporary buffer in reverse order
        wchar_t  buf[5];
        uint32_t count = 0;
        do
        {
            buf[count++] = L'0' + (id % 10);
            id /= 10;
        } while (id != 0 && count < 5);

        // Reverse into the output buffer
        for (uint32_t i = 0; i < count; ++i)
        {
            out[i] = buf[count - 1 - i];
        }

        return count;
    }

    static UniqueRef<wchar_t[]> tempMIPCDirectory(std::pmr::memory_resource* tmp, size_t* total, size_t* len)
    {
        static constexpr size_t PathMaxCharCount = nextPOT<size_t>(MAX_PATH);
        static constexpr size_t MipcacheNumel    = 10; // ".mipcache_"
        static wchar_t const s_mipcacheBaseDir[MipcacheNumel] = {L'.', L'm', L'i', L'p', L'c', L'a', L'c', L'h', L'e', L'_'};

        auto tempDirectory = makeUniqueRef<wchar_t[]>(tmp, PathMaxCharCount);

        // Prefix with \\?\ for extended-length paths
        tempDirectory[0] = L'\\';
        tempDirectory[1] = L'\\';
        tempDirectory[2] = L'?';
        tempDirectory[3] = L'\\';

        *total = PathMaxCharCount;

        if (!GetTempPath2W(PathMaxCharCount, tempDirectory.get() + 4))
            return {}; // return empty on error

        size_t lenTempDirectory = wcsnlen(tempDirectory.get(), PathMaxCharCount);
        assert(lenTempDirectory < PathMaxCharCount - 24);

        // Append ".mipcache_"
        std::memcpy(tempDirectory.get() + lenTempDirectory, s_mipcacheBaseDir, MipcacheNumel * sizeof(wchar_t));
        lenTempDirectory += MipcacheNumel;

        // Append PID
        lenTempDirectory += pidUpTo5WChars(tempDirectory.get() + lenTempDirectory);

        // Null-terminate
        tempDirectory[lenTempDirectory] = L'\0';

        // Try creating the directory or resolve collision
        for (int suffix = 0; suffix <= 99; ++suffix)
        {
            DWORD attrs = GetFileAttributesW(tempDirectory.get());
            if (attrs == INVALID_FILE_ATTRIBUTES)
            {
                // Doesn't exist -> try to create
                if (CreateDirectoryW(tempDirectory.get(), nullptr))
                {
                    *len = lenTempDirectory;
                    return tempDirectory; // success
                }
            }
            else if (attrs & FILE_ATTRIBUTE_DIRECTORY)
            {
                *len = lenTempDirectory;
                return tempDirectory; // directory already exists
            }

            // Collision: append "~<n>" and try again
            if (suffix == 99)
                break; // stop after ~99
            lenTempDirectory += swprintf(tempDirectory.get() + lenTempDirectory,
                                         PathMaxCharCount - lenTempDirectory,
                                         L"~%d",
                                         suffix + 1);
        }

        return {}; // failed to create a usable directory
    }

    static DMT_FORCEINLINE bool makeTempFilePath(uint64_t key, size_t& len, UniqueRef<wchar_t[]>& tempDirectory)
    {

        tempDirectory[len++] = L'\\';
        auto keyStr          = std::to_wstring(key); // should fit in SBO buffer
        std::memcpy(tempDirectory.get() + len, keyStr.data(), keyStr.size() * sizeof(wchar_t));
        len += keyStr.size();
        tempDirectory[len++] = L'.', tempDirectory[len++] = L'm', tempDirectory[len++] = L'i';
        tempDirectory[len++] = L'p', tempDirectory[len++] = L'c', tempDirectory[len] = L'\0';

        // if collision, fail miserably
        DWORD attrs = GetFileAttributesW(tempDirectory.get());
        if (attrs != INVALID_FILE_ATTRIBUTES) // if it exists
            return false;
        return true;
    }

    static DMT_FORCEINLINE bool fillDriveOrUNC(wchar_t* dst, wchar_t const* src)
    {
        // copy from source until 1) you encounter L':' and then L'\\', or you encounter three times L'\\'
        int            backslashCount = 0;
        wchar_t const* src_start      = src;

        // Loop through the source string
        while (*src != L'\0')
        {
            // Check for the drive letter case: ':' followed by '\'
            if (*src == L':' && *(src + 1) == L'\\')
            {
                // Copy up to and including the backslash
                std::wcsncpy(dst, src_start, (src - src_start) + 2);
                dst[(src - src_start) + 2] = L'\0'; // Null-terminate the destination string
                return true;
            }

            // Count backslashes for UNC path case
            if (*src == L'\\')
            {
                backslashCount++;
                if (backslashCount == 3)
                {
                    // Copy up to and including the third backslash
                    std::wcsncpy(dst, src_start, (src - src_start) + 1);
                    dst[(src - src_start) + 1] = L'\0'; // Null-terminate the destination string
                    return true;
                }
            }

            // Move to the next character
            src++;
        }

        // If neither condition is met
        return false;
    }

    MipCacheFile::MipCacheFile(std::pmr::memory_resource* mapMemory) : m_keyfdmap{mapMemory}, m_sectorSize{0} {}

    class SectorBufferedWriter
    {
    public:
        SectorBufferedWriter(HANDLE file, size_t sectorSize, size_t bufferMultiplier, std::pmr::memory_resource* tmp) :
        m_file(file),
        m_sectorSize(sectorSize),
        m_bufferSize(sectorSize * bufferMultiplier),
        m_buffer(makeUniqueRef<unsigned char[]>(tmp, m_bufferSize)),
        m_offset(0)
        {
        }

        bool write(void const* data, size_t bytes)
        {
            unsigned char const* src     = reinterpret_cast<unsigned char const*>(data);
            size_t               written = 0;

            while (written < bytes)
            {
                size_t space  = m_bufferSize - m_offset;
                size_t toCopy = std::min(space, bytes - written);
                std::memcpy(m_buffer.get() + m_offset, src + written, toCopy);
                m_offset += toCopy;
                written += toCopy;

                if (m_offset == m_bufferSize)
                {
                    if (!flushBuffer())
                        return false;
                }
            }
            return true;
        }

        bool flush(bool padToSector = true)
        {
            if (m_offset == 0)
                return true; // nothing to flush

            size_t writeSize = m_offset;
            if (padToSector)
                writeSize = (writeSize + m_sectorSize - 1) & ~(m_sectorSize - 1);

            if (writeSize > m_bufferSize)
            {
                assert(false); // should never happen
                return false;
            }

            if (writeSize > m_offset)
                std::memset(m_buffer.get() + m_offset, 0, writeSize - m_offset);

            DWORD bytesWritten = 0;
            if (!WriteFile(m_file, m_buffer.get(), static_cast<DWORD>(writeSize), &bytesWritten, nullptr))
                return false;
            assert(bytesWritten == writeSize);
            m_offset = 0;
            return true;
        }

    private:
        bool flushBuffer() { return flush(true); }

        HANDLE                     m_file;
        size_t                     m_sectorSize;
        size_t                     m_bufferSize;
        UniqueRef<unsigned char[]> m_buffer;
        size_t                     m_offset;
    };

    bool MipCacheFile::createCacheFile(uint64_t baseKey, ImageTexturev2 const& tex)
    {
        std::pmr::monotonic_buffer_resource scratch{s_tempBuffer, staticBytes, std::pmr::null_memory_resource()};
        // check that directory reserved for this process, in temp, exists => .mipcache_<pid>
        size_t total = 0, len = 0;
        auto   tempDirectory = tempMIPCDirectory(&scratch, &total, &len);
        if (!tempDirectory)
            return false;

        // 18,446,744,073,709,551,615 -> 20 chars, plus ".mipc" (5 chars) + \0, "\\" -> 27
        // TODO better resize and copy stuff
        assert(total - len >= 27);

        if (!makeTempFilePath(baseKey, len, tempDirectory))
            return false;

        if (m_keyfdmap.contains(baseKey))
            return false;

        // create file
        // reference caching: https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew#caching_behavior

        // setup security attribute such that only current process user can manipulate this file
        SECURITY_ATTRIBUTES sa{};
        sa.nLength        = sizeof(sa);
        sa.bInheritHandle = FALSE;

        // --- Build security descriptor so only current user + admins have full access ---
        // SDDL format: D: = DACL, (A;;FA;;;<SID>) means allow full access to SID
        // We use "BA" (built-in admins) and "OW" (owner), which is the current user when file is created
        LPCWSTR              sddl = L"D:(A;;FA;;;BA)(A;;FA;;;OW)";
        PSECURITY_DESCRIPTOR pSD  = nullptr;

        if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, SDDL_REVISION_1, &pSD, nullptr))
        {
            return false; // Could not build SD
        }
        sa.lpSecurityDescriptor = pSD;
        HANDLE hFile            = CreateFileW(tempDirectory.get(),
                                   GENERIC_READ | GENERIC_WRITE,
                                   0,          // no sharing
                                   &sa,        // only current process can access
                                   CREATE_NEW, // fail if exists
                                   FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH | FILE_FLAG_DELETE_ON_CLOSE,
                                   nullptr);
        LocalFree(pSD);
        if (hFile == INVALID_HANDLE_VALUE)
            return false;

        // delete file if machine reboots while application is running
        MoveFileExW(tempDirectory.get(), nullptr, MOVEFILE_DELAY_UNTIL_REBOOT);

        // create header
        size_t const headerSize  = sizeof(mipc::Header);
        size_t const contentSize = mipChainPixelCount(tex.width, tex.height) * bytesPerPixel(tex.texFormat);
        size_t const totalSize   = headerSize + contentSize;

        // FILE_FLAG_NO_BUFFERING requires that:
        // - File size changes are aligned to sector size(usually 512 or 4096 bytes)
        // - Reads / writes must also be aligned and multiples of that size.
        if (!m_sectorSize)
        {
            DWORD   sectorsPerCluster, bytesPerSector, freeClusters, totalClusters;
            wchar_t driveLetter[12]{};
            if (!fillDriveOrUNC(driveLetter, tempDirectory.get() + 4))
                return false;
            if (!GetDiskFreeSpaceW(driveLetter, &sectorsPerCluster, &bytesPerSector, &freeClusters, &totalClusters))
                return false;
            uint64_t const sectorSize = bytesPerSector; // Usually 512 or 4096

            // Assumes sector size doesn't change during execution
            m_sectorSize = sectorSize;
        }

        uint64_t const alignedSize = (totalSize + (m_sectorSize - 1)) & ~(m_sectorSize - 1);

        mipc::Header const header{.magic             = mipc::MAGIC,
                                  .version           = mipc::VERSION,
                                  .key               = baseKey,
                                  .effectiveFileSize = totalSize,
                                  .width             = static_cast<uint32_t>(tex.width),
                                  .height            = static_cast<uint32_t>(tex.height),
                                  .mipLevels         = static_cast<uint32_t>(tex.mipLevels),
                                  .texFormat         = tex.texFormat};

        // resize file immediately
        LARGE_INTEGER const FileBytes{.QuadPart = static_cast<LONGLONG>(alignedSize)};
        if (!SetFilePointerEx(hFile, FileBytes, nullptr, FILE_BEGIN) || !SetEndOfFile(hFile))
        {
            CloseHandle(hFile);
            return false;
        }

        if (!SetFilePointerEx(hFile, {0ull}, nullptr, FILE_BEGIN))
        {
            CloseHandle(hFile);
            return false;
        }

        SectorBufferedWriter writer(hFile, m_sectorSize, 4, &scratch);

        // write header
        if (!writer.write(&header, sizeof(header)))
        {
            CloseHandle(hFile);
            return false;
        }

        // write mips
        size_t const pixelSize = bytesPerPixel(header.texFormat);
        for (uint32_t level = 0; level < tex.mipLevels; ++level)
        {
            uint32_t const mipSize = mipBytes(tex.width, tex.height, level, tex.texFormat);
            auto const     mipData = reinterpret_cast<unsigned char const*>(tex.data) +
                                 mortonLevelOffset(tex.width, tex.height, level) * pixelSize;
            if (!writer.write(mipData, mipSize))
            {
                CloseHandle(hFile);
                return false;
            }
        }

        // flush remaining
        if (!writer.flush())
        {
            CloseHandle(hFile);
            return false;
        }

        // note: not returning the file handle, hence we are keeping it open. When the process terminates, it will be
        // deleted and FILE_FLAG_DELETE_ON_CLOSE will kick in
        static_assert(sizeof(HANDLE) == sizeof(uintptr_t));
        m_keyfdmap.insert_or_assign(baseKey, std::bit_cast<uintptr_t>(hFile));
        return m_keyfdmap.contains(baseKey);
    }

    int32_t MipCacheFile::copyMipOfKey(uint64_t   fileKey,
                                       int32_t    level,
                                       void*      outBuffer,
                                       uint32_t*  inOutBytes,
                                       size_t*    inOutOffset,
                                       TexFormat* outFormat) const
    {
        if (!m_keyfdmap.contains(fileKey))
            return false;

        HANDLE hFile     = std::bit_cast<HANDLE>(m_keyfdmap.at(fileKey));
        DWORD  bytesRead = 0;

        assert(inOutBytes && inOutOffset);
        uint64_t const sectorSize = m_sectorSize; // must be > 0

        if (!outBuffer)
        {
            assert(outFormat);
            std::pmr::monotonic_buffer_resource scratch{s_tempBuffer, staticBytes, std::pmr::null_memory_resource()};
            // read header
            mipc::Header   header{};
            uint32_t const headerAlignedSize = (sizeof(header) + m_sectorSize - 1) & ~(m_sectorSize - 1);
            auto           headerBuffer      = makeUniqueRef<unsigned char[]>(&scratch, headerAlignedSize);
            assert(headerBuffer);

            if (!SetFilePointerEx(hFile, {0ull}, nullptr, FILE_BEGIN) ||
                !ReadFile(hFile, headerBuffer.get(), headerAlignedSize, &bytesRead, nullptr))
                return false;
            std::memcpy(&header, headerBuffer.get(), sizeof(mipc::Header));

            // compute unaligned offset and size
            size_t const mipOffsetUnaligned = sizeof(header) + mortonLevelOffset(static_cast<int32_t>(header.width),
                                                                                 static_cast<int32_t>(header.height),
                                                                                 level) *
                                                                   bytesPerPixel(header.texFormat);
            uint32_t const mipSizeUnaligned = mipBytes(static_cast<int32_t>(header.width),
                                                       static_cast<int32_t>(header.height),
                                                       level,
                                                       header.texFormat);

            // align offset down and size up to sector size
            size_t const   offsetAligned = mipOffsetUnaligned / sectorSize * sectorSize;
            uint32_t const sizeAligned   = static_cast<uint32_t>(
                ((mipOffsetUnaligned + mipSizeUnaligned + sectorSize - 1) / sectorSize) * sectorSize - offsetAligned);

            *inOutBytes  = sizeAligned;   // caller allocates this much
            *inOutOffset = offsetAligned; // aligned read start
            *outFormat   = header.texFormat;

            return mipOffsetUnaligned - offsetAligned;
        }
        else
        {
            // move file pointer to sector-aligned offset
            LARGE_INTEGER offsetAligned{};
            offsetAligned.QuadPart = static_cast<LONGLONG>(*inOutOffset);
            assert(*inOutBytes > 0);

            if (!SetFilePointerEx(hFile, offsetAligned, nullptr, FILE_BEGIN) ||
                !ReadFile(hFile, outBuffer, *inOutBytes, &bytesRead, nullptr))
                return false;

            // optionally, you could return the relative "start of actual data" inside buffer
            // but your caller already knows offset difference = actualOffset - alignedOffset

            return true;
        }
    }

    MipCacheFile::~MipCacheFile() noexcept
    {
        for (auto const& [key, value] : m_keyfdmap)
        {
            HANDLE hFile = std::bit_cast<HANDLE>(value);
            CloseHandle(hFile);
        }
    }
} // namespace dmt
