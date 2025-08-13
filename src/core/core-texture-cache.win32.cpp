#include "core-texture-cache.h"

#include <Windows.h>

#include <fileapi.h>
#include <processthreadsapi.h>
#include <WinBase.h> // file flags
#include <sddl.h> // https://learn.microsoft.com/en-us/windows/win32/api/sddl/nf-sddl-convertstringsecuritydescriptortosecuritydescriptorw

namespace dmt {
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

    static uint32_t mipBytes(int32_t width, int32_t height, int32_t level, TexFormat texFormat)
    {
        uint32_t const pixelBytes = bytesPerPixel(texFormat);
        while (level > 0)
        {
            width  = std::max<int32_t>(1, width >> 1);
            height = std::max<int32_t>(1, height >> 1);
        }

        size_t bytes = static_cast<size_t>(pixelBytes) * width * height;
        assert(static_cast<uint32_t>(bytes) == bytes && "A mip level is occupying more than 4 GB");
        return static_cast<uint32_t>(bytes);
    }

    static UniqueRef<wchar_t[]> tempMIPCDirectory(std::pmr::memory_resource* tmp, size_t* total, size_t* len)
    {
        static constexpr size_t PathMaxCharCount = 2ull * MAX_PATH;
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

    MipCacheFile::MipCacheFile(std::pmr::memory_resource* mapMemory) : m_keyfdmap{mapMemory} {}

    bool MipCacheFile::createCacheFile(uint64_t baseKey, ImageTexturev2 const& tex, std::pmr::memory_resource* tmp)
    {
        // check that directory reserved for this process, in temp, exists => .mipcache_<pid>
        size_t total = 0, len = 0;
        auto   tempDirectory = tempMIPCDirectory(tmp, &total, &len);
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
        DWORD   sectorsPerCluster, bytesPerSector, freeClusters, totalClusters;
        wchar_t driveLetter[12]{};
        if (!fillDriveOrUNC(driveLetter, tempDirectory.get() + 4))
            return false;
        if (!GetDiskFreeSpaceW(driveLetter, &sectorsPerCluster, &bytesPerSector, &freeClusters, &totalClusters))
            return false;
        uint64_t const sectorSize  = bytesPerSector; // Usually 512 or 4096
        uint64_t const alignedSize = (totalSize + (sectorSize - 1)) & ~(sectorSize - 1);

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

        // copy header on a buffer multiple of sector size
        size_t const headerBufferSize = (headerSize + (sectorSize - 1)) & ~(sectorSize - 1);
        DWORD        written          = 0;
        assert(static_cast<DWORD>(headerBufferSize) == headerBufferSize);
        {
            auto headerRef = makeUniqueRef<unsigned char[]>(tmp, headerBufferSize);
            std::memcpy(headerRef.get(), &header, headerSize);
            if (size_t diff = headerBufferSize - headerSize; diff != 0)
                std::memset(headerRef.get() + headerSize, 0, diff);

            // write header
            if (!SetFilePointerEx(hFile, {0ull}, nullptr, FILE_BEGIN) ||
                !WriteFile(hFile, headerRef.get(), static_cast<DWORD>(headerBufferSize), &written, nullptr))
            {
                CloseHandle(hFile);
                return false;
            }
            assert(written == headerBufferSize);
        }

        // write mips one by one, assuming each is not more than 4 GB
        LARGE_INTEGER offset{.QuadPart = static_cast<LONGLONG>(headerSize)};
        for (uint32_t level = 0; level < tex.mipLevels; ++level)
        {
            uint32_t const mipSize = mipBytes(tex.width, tex.height, level, tex.texFormat);
            void const*    mipData = reinterpret_cast<unsigned char const*>(tex.data) +
                                  mortonLevelOffset(tex.width, tex.height, level);
            if (!SetFilePointerEx(hFile, offset, nullptr, FILE_BEGIN) ||
                WriteFile(hFile, mipData, mipSize, &written, nullptr))
            {
                CloseHandle(hFile);
                return false;
            }
            assert(mipSize == written);
            assert(mipSize % sectorSize == 0 && "fkdlsfdjlfkjasdlkfjlsdjfadlksfjlk");
            offset.QuadPart += mipSize;
        }

        // TODO zero out the unused part of the file?

        // note: not returning the file handle, hence we are keeping it open. When the process terminates, it will be
        // deleted and FILE_FLAG_DELETE_ON_CLOSE will kick in
        static_assert(sizeof(HANDLE) == sizeof(uintptr_t));
        m_keyfdmap.insert_or_assign(baseKey, std::bit_cast<uintptr_t>(hFile));
        return m_keyfdmap.contains(baseKey);
    }

    bool MipCacheFile::copyMipOfKey(uint64_t fileKey, int32_t level, void* outBuffer, uint32_t* inOutBytes, size_t* inOutOffset) const
    {
        if (!m_keyfdmap.contains(fileKey))
            return false;
        HANDLE hFile     = std::bit_cast<HANDLE>(m_keyfdmap.at(fileKey));
        DWORD  bytesRead = 0;

        assert(inOutBytes && inOutOffset);
        if (!outBuffer)
        {
            // read header
            mipc::Header header{};
            if (!SetFilePointerEx(hFile, {0ull}, nullptr, FILE_BEGIN) ||
                !ReadFile(hFile, &header, static_cast<DWORD>(sizeof(header)), &bytesRead, nullptr))
                return false;

            // compute offset and size of mip and return them
            LARGE_INTEGER const offset{
                .QuadPart = static_cast<LONGLONG>(
                    sizeof(header) +
                    mortonLevelOffset(static_cast<int32_t>(header.width), static_cast<int32_t>(header.height), level))};
            uint32_t const mipSize = mipBytes(static_cast<int32_t>(header.width),
                                              static_cast<int32_t>(header.height),
                                              level,
                                              header.texFormat);
            // if you called to compute the size needed for the buffer, return it together with offset
            *inOutBytes  = mipSize;
            *inOutOffset = offset.QuadPart;
            return true;
        }
        else
        {
            // move file pointer to mip offset and copy mip
            LARGE_INTEGER const offset{.QuadPart = static_cast<LONGLONG>(*inOutOffset)};
            assert(*inOutBytes > 0);
            if (!SetFilePointerEx(hFile, offset, nullptr, FILE_BEGIN) ||
                !ReadFile(hFile, outBuffer, *inOutBytes, &bytesRead, nullptr))
                return false;

            return true;
        }
    }


    TextureCache::TextureCache(size_t cacheSize, std::pmr::memory_resource* cacheMem, std::pmr::memory_resource* listMem) :
    MipcFiles{cacheMem},
    m_cache{cacheMem},
    m_lruKeyList{listMem},
    m_texMemory{EMemoryTag::eTextures, cacheSize},
    m_available{cacheSize}
    {
        m_cache.reserve(256);
    }

    void const* TextureCache::getOrInsert(uint64_t baseKey, int32_t mipLevel, uint32_t& outBytes)
    {
        assert(m_cache.size() == m_lruKeyList.size());
        // 1. shared lock key path
        uint64_t           key = baseKey ^ static_cast<uint64_t>(mipLevel) << 33;
        gx::shareable_lock lk(m_shmtx, gx::lock_mode::shared);
        if (auto it = m_cache.find(key); it != m_cache.cend())
        {
            outBytes = it->second.numBytes;
            // swap lruKey to last
            auto listit = std::find(m_lruKeyList.begin(), m_lruKeyList.end(), key);
            if (listit == m_lruKeyList.end())
            {
                assert(false && "How did List and HashMap get out of sync");
                return nullptr;
            }
            if (std::next(listit) != m_lruKeyList.end()) // if not last swap to last
            {
                lk.upgrade_to_exclusive();
                m_lruKeyList.splice(m_lruKeyList.end(), m_lruKeyList, listit);
                lk.downgrade_to_shared();
            }

            return it->second.data;
        }

        // 2. handle cache miss
        lk.upgrade_to_exclusive();
        uint32_t mipSize   = 0;
        size_t   mipOffset = 0;
        if (!MipcFiles.copyMipOfKey(baseKey, mipLevel, nullptr, &mipSize, &mipOffset))
        {
            assert(false && "you shouldn't request something which doesn't exist");
            return nullptr; // no file
        }

        // check memory availability
        while (m_available < mipSize && !m_cache.empty())
        {
            assert(!m_lruKeyList.empty());
            // handle eviction and LRU list update
            uint64_t const keyToDel = m_lruKeyList.front();
            auto           it       = m_cache.find(keyToDel);
            if (it == m_cache.end())
            {
                assert(false && "How did List and HashMap get out of sync");
                return nullptr;
            }
            uint32_t const numBytes = it->second.numBytes;
            m_texMemory.deallocate(it->second.data, numBytes);

            bool const deleted = m_cache.erase(keyToDel);
            assert(deleted);
            m_lruKeyList.pop_front();
            m_available += numBytes;
        }

        if (m_available < mipSize)
        {
            assert(false && "Even when empty the cache cannot accomodate the requested mip");
            return nullptr;
        }

        void* buffer = m_texMemory.allocate(mipSize);
        if (!buffer)
            return nullptr; // memory failure

        // create entry in LRU map and push_back onto list the key
        auto const& [it, wasInserted] = m_cache.try_emplace(key, baseKey, buffer, mipLevel, mipSize);

        if (!wasInserted || !MipcFiles.copyMipOfKey(baseKey, mipLevel, it->second.data, &mipSize, &mipOffset))
        {
            assert(false && "how did we get here?");
            m_texMemory.deallocate(buffer, mipSize);
            return nullptr;
        }

        m_lruKeyList.push_back(key);
        assert(m_cache.size() == m_lruKeyList.size());

        // update m_available
        m_available -= mipSize;

        outBytes = mipSize;
        return it->second.data;
    }
} // namespace dmt
