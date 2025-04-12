#define DMT_ENTRY_POINT
#include "platform/platform.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <bit>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

#include <cassert>
#include <cstdint>
#include <cstdio>

#if defined(_WIN32)
#include <windows.h>
#define STRICT_TYPED_ITEMIDS // Better type safety for IDLists
#include <commdlg.h>
#include <shobjidl.h> // For IFileDialog
#include <shlobj.h>   // For SHGetPathFromIDListW
#endif

static void worker(void* unused)
{
    dmt::Context ctx;
    for (int i = 0; i < 5; ++i)
    {
        ctx.log("Thread running... {}", std::make_tuple(i));
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

#if defined(_WIN32)
static std::string WideToUtf8(std::wstring const& wide)
{
    if (wide.empty())
        return {};

    int sizeNeeded = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (sizeNeeded == 0)
        return {};

    std::string result(sizeNeeded - 1, '\0'); // -1 to remove null terminator
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, result.data(), sizeNeeded, nullptr, nullptr);
    return result;
}

// https://github.com/MicrosoftDocs/win32/blob/docs/desktop-src/shell/common-file-dialog.md
static HRESULT showFolderSelectDialog(std::wstring& outFolderPath)
{
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    if (FAILED(hr))
        return hr;

    IFileDialog* pFileDialog = nullptr;
    hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pFileDialog));
    if (FAILED(hr))
        return hr;

    // Set the dialog options
    DWORD dwOptions;
    hr = pFileDialog->GetOptions(&dwOptions);
    if (SUCCEEDED(hr))
    {
        hr = pFileDialog->SetOptions(dwOptions | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);
        if (FAILED(hr))
        {
            pFileDialog->Release();
            return hr;
        }
    }

    // Show the dialog
    hr = pFileDialog->Show(NULL);
    if (SUCCEEDED(hr))
    {
        IShellItem* pItem = nullptr;
        hr                = pFileDialog->GetResult(&pItem);
        if (SUCCEEDED(hr))
        {
            PWSTR pszFolderPath = nullptr;
            hr                  = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFolderPath);
            if (SUCCEEDED(hr))
            {
                outFolderPath = pszFolderPath;
                CoTaskMemFree(pszFolderPath);
            }
            pItem->Release();
        }
    }

    pFileDialog->Release();
    CoUninitialize();
    return hr;
}
#endif

struct Color
{
    uint8_t r, g, b;
};
static_assert(sizeof(Color) == 3 * sizeof(uint8_t));

enum class GradientDirection
{
    Horizontal,
    Vertical,
    DiagonalDown, // top-left → bottom-right
    DiagonalUp    // bottom-left → top-right
};

static Color gradientColor(float rowNorm, float colNorm, GradientDirection dir, Color start, Color end)
{
    // Clamp inputs
    rowNorm = std::clamp(rowNorm, 0.0f, 1.0f);
    colNorm = std::clamp(colNorm, 0.0f, 1.0f);

    float t = 0.0f;

    // Determine interpolation factor based on direction
    switch (dir)
    {
        case GradientDirection::Horizontal: t = colNorm; break;
        case GradientDirection::Vertical: t = rowNorm; break;
        case GradientDirection::DiagonalDown: t = (rowNorm + colNorm) / 2.0f; break;
        case GradientDirection::DiagonalUp: t = (1.0f - rowNorm + colNorm) / 2.0f; break;
    }

    // Linear interpolation between start and end color
    auto lerp = [](uint8_t a, uint8_t b, float t) -> uint8_t { return static_cast<uint8_t>(a + t * (b - a)); };

    return Color{lerp(start.r, end.r, t), lerp(start.g, end.g, t), lerp(start.b, end.b, t)};
}

namespace dmt::img {
    // Deinterleave bits: extract x or y from morton index
    uint32_t decodeMorton2D(uint32_t morton)
    {
        uint32_t x = morton;
        x &= 0x55555555; // mask out even bits
        x = (x | (x >> 1)) & 0x33333333;
        x = (x | (x >> 2)) & 0x0F0F0F0F;
        x = (x | (x >> 4)) & 0x00FF00FF;
        x = (x | (x >> 8)) & 0x0000FFFF;
        return x;
    }

    uint32_t decodeMortonX(uint32_t morton) { return decodeMorton2D(morton); }

    uint32_t decodeMortonY(uint32_t morton) { return decodeMorton2D(morton >> 1); }

    void mortonToLinear(uint8_t* DMT_RESTRICT dstLinear, uint8_t const* DMT_RESTRICT srcMorton, int width, int height, int channels)
    {
        assert((width & (width - 1)) == 0); // power of 2
        assert((height & (height - 1)) == 0);

        size_t numPixels = width * height;
        for (uint32_t i = 0; i < numPixels; ++i)
        {
            uint32_t x = decodeMortonX(i);
            uint32_t y = decodeMortonY(i);
            if (x >= width || y >= height)
                continue;

            size_t linearIdx = (y * width + x) * channels;
            size_t mortonIdx = i * channels;

            std::memcpy(&dstLinear[linearIdx], &srcMorton[mortonIdx], channels);
        }
    }

    // Interleave the bits of x and y
    uint32_t encodeMorton2D(uint32_t x, uint32_t y)
    {
        auto part1by1 = [](uint32_t n) -> uint32_t {
            n &= 0x0000ffff;
            n = (n | (n << 8)) & 0x00FF00FF;
            n = (n | (n << 4)) & 0x0F0F0F0F;
            n = (n | (n << 2)) & 0x33333333;
            n = (n | (n << 1)) & 0x55555555;
            return n;
        };

        return (part1by1(y) << 1) | part1by1(x);
    }

    uint8_t* accessMortonPixel(uint8_t* mortonData, int width, int height, int channels, int row, int col)
    {
        assert((width & (width - 1)) == 0); // power of 2
        assert((height & (height - 1)) == 0);

        if (row >= height || col >= width)
            return nullptr;

        uint32_t mortonIdx = encodeMorton2D(col, row); // x = col, y = row
        return &mortonData[mortonIdx * channels];
    }
} // namespace dmt::img

namespace dmt::jobs {
    struct JobDataColorTile
    {
        Color*                mortonImage;
        std::atomic<uint32_t> tilesDone;
        uint32_t              width;
        uint32_t              height;
        uint32_t              tileWidth;
        uint32_t              tileHeight;
        Color                 start;
        Color                 end;
    };

    void colorTile(uintptr_t _data, uint32_t tid)
    {
        dmt::Context ctx;
        auto*        data = reinterpret_cast<JobDataColorTile*>(_data);

        uint32_t tilesPerRow = dmt::ceilDiv(data->width, data->tileWidth);
        uint32_t tilesPerCol = dmt::ceilDiv(data->height, data->tileHeight);
        uint32_t totalTiles  = tilesPerRow * tilesPerCol;

        uint32_t tileIdx = data->tilesDone.fetch_add(1, std::memory_order_acquire);
        ctx.trace("TID: {}, Tile Index: {}", std::make_tuple(tid, tileIdx));
        if (tileIdx >= totalTiles)
            return;

        uint32_t tileRow = tileIdx / tilesPerRow;
        uint32_t tileCol = tileIdx % tilesPerRow;

        uint32_t rowStart = tileRow * data->tileHeight;
        uint32_t colStart = tileCol * data->tileWidth;

        for (uint32_t r = 0; r < data->tileHeight; ++r)
        {
            uint32_t imgRow = rowStart + r;
            if (imgRow >= data->height)
                continue;

            float rowNorm = float(imgRow) / float(data->height - 1);

            for (uint32_t c = 0; c < data->tileWidth; ++c)
            {
                uint32_t imgCol = colStart + c;
                if (imgCol >= data->width)
                    continue;

                float colNorm = float(imgCol) / float(data->width - 1);

                Color pixel = gradientColor(rowNorm, colNorm, GradientDirection::DiagonalDown, data->start, data->end);

                uint32_t idx           = dmt::img::encodeMorton2D(imgRow, imgCol);
                data->mortonImage[idx] = pixel;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        ctx.log("Wrote tile {{{}, {}}}", std::make_tuple(tileRow, tileCol));
    }

} // namespace dmt::jobs

int guardedMain()
{
    using namespace std::string_view_literals;
    dmt::Ctx::init();
    struct Janitor
    {
        ~Janitor() { dmt::Ctx::destroy(); }
    } j;

    {
        dmt::Context ctx;
        ctx.log("Wrote tile {{{}, {}}}", std::make_tuple(3u, 3u));
        ctx.log("Starting memory management tests...", {});

        static constexpr size_t testSize      = dmt::toUnderlying(dmt::EPageSize::e2MB);
        static constexpr size_t largePageSize = dmt::toUnderlying(dmt::EPageSize::e1GB);

        // Test: Reserve Virtual Address Space
        void* reservedAddress = dmt::os::reserveVirtualAddressSpace(testSize);
        if (reservedAddress)
        {
            ctx.log("Successfully reserved virtual address space", {});
        }
        else
        {
            ctx.error("Failed to reserve virtual address space", {});
        }

        // Test: Commit Physical Memory
        bool commitSuccess = dmt::os::commitPhysicalMemory(reservedAddress, testSize);
        if (commitSuccess)
        {
            ctx.log("Successfully committed physical memory", {});
        }
        else
        {
            ctx.error("Failed to commit physical memory", {});
        }

        // Test: Decommit Physical Memory
        dmt::os::decommitPhysicalMemory(reservedAddress, testSize);
        ctx.log("Decommitted physical memory", {});

        // Test: Free Virtual Address Space
        bool freeSuccess = dmt::os::freeVirtualAddressSpace(reservedAddress, testSize);
        if (freeSuccess)
        {
            ctx.log("Successfully freed virtual address space", {});
        }
        else
        {
            ctx.error("Failed to free virtual address space", {});
        }

        // Test: Allocate Locked Large Pages (2MB)
        void* largePageMemory2MB = dmt::os::allocateLockedLargePages(testSize, dmt::EPageSize::e2MB, false);
        if (largePageMemory2MB)
        {
            ctx.log("Successfully allocated locked large pages (2MB)", {});
            dmt::os::deallocateLockedLargePages(largePageMemory2MB, testSize, dmt::EPageSize::e2MB);
            ctx.log("Successfully deallocated locked large pages (2MB)", {});
        }
        else
        {
            ctx.error("Failed to allocate locked large pages (2MB)", {});
        }

        // Test: Allocate Locked Large Pages (1GB)
        void* largePageMemory1GB = dmt::os::allocateLockedLargePages(largePageSize, dmt::EPageSize::e1GB, false);
        if (largePageMemory1GB)
        {
            ctx.log("Successfully allocated locked large pages (1GB)", {});
            dmt::os::deallocateLockedLargePages(largePageMemory1GB, largePageSize, dmt::EPageSize::e1GB);
            ctx.log("Successfully deallocated locked large pages (1GB)", {});
        }
        else
        {
            ctx.error("Failed to allocate locked large pages (1GB)", {});
        }

        ctx.log("Memory management tests completed.", {});

        dmt::os::Thread t{worker};
        ctx.log("Running thread {}", std::make_tuple(t.id()));
        t.start();

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        ctx.log("Joining Thread {}\0", std::make_tuple(t.id()));
        t.join();
        ctx.log("Thread Joined", {});

        ctx.log("Starting Pool Allocator memory tests", {});

        // Create a SyncPoolAllocator instance
        dmt::SyncPoolAllocator allocator(dmt::EMemoryTag::eUnknown, 1024 * 1024, 64, dmt::EBlockSize::e256B); // 1MB reserved, 256B block size, 64 initial blocks

        // Check if allocator is valid
        if (!allocator.isValid())
        {
            ctx.error("Allocator is not valid!", {});
        }
        else
        {
            // Allocate memory: Try to allocate 512 bytes
            void* ptr = allocator.allocate(512);
            if (ptr != nullptr)
                ctx.log("Successfully allocated 512 bytes.", {});
            else
                ctx.error("Allocation failed!", {});

            // Allocate more memory: Try to allocate another 128 bytes
            void* ptr2 = allocator.allocate(128);
            if (ptr2 != nullptr)
                ctx.log("Successfully allocated another 128 bytes.", {});
            else
                ctx.error("Second allocation failed!", {});

            // Deallocate memory: Free the first pointer (512 bytes)
            allocator.deallocate(ptr, 512);
            ctx.log("Successfully deallocated 512 bytes.", {});

            // Deallocate second memory block (128 bytes)
            allocator.deallocate(ptr2, 128);
            ctx.log("Successfully deallocated 128 bytes.", {});

            // Attempting allocation after deallocation
            void* ptr3 = allocator.allocate(256);
            if (ptr3 != nullptr)
                ctx.log("Successfully allocated 256 bytes after deallocation.", {});
            else
                ctx.error("Allocation after deallocation failed!", {});

            // Check number of blocks in the allocator
            uint32_t numBlocks = allocator.numBlocks();
            ctx.log("Number of blocks in the allocator: {}", std::make_tuple(numBlocks));
        }

        // Threadpool test
        dmt::ThreadPoolV2 threadPool{};
#if defined(_WIN32)
        std::wstring wstr;
        while (wstr.empty() /* and is not a valid directory*/)
            showFolderSelectDialog(wstr);

        std::string str = WideToUtf8(wstr);
        ctx.log("Selected Folder: {{}} \"{}\"", std::make_tuple(str));
        str += "\\image.png";


        int32_t width = 512, height = 512, channels = 3;

        // clang-format off
        dmt::UniqueRef<uint8_t[]> dataMorton   = dmt::makeUniqueRef<uint8_t[]>(
            std::pmr::get_default_resource(), width * height * channels);
        dmt::UniqueRef<uint8_t[]> dataRowMajor = dmt::makeUniqueRef<uint8_t[]>(
            std::pmr::get_default_resource(), width * height * channels);
        // clang-format on

        dmt::jobs::JobDataColorTile jobData{
            .mortonImage = reinterpret_cast<Color*>(dataMorton.get()),
            .tilesDone   = 0,
            .width       = static_cast<uint32_t>(width),
            .height      = static_cast<uint32_t>(height),
            .tileWidth   = 32,
            .tileHeight  = 32,
            .start{255, 0, 0},
            .end{0, 255, 0},
        };

        uint32_t const numTiles = dmt::ceilDiv(static_cast<uint32_t>(width * height), 32u * 32u);
        for (uint32_t i = 0; i < numTiles; ++i)
        {
            threadPool.addJob({dmt::jobs::colorTile, reinterpret_cast<uintptr_t>(&jobData)}, dmt::EJobLayer::eDefault);
        }

        threadPool.kickJobs();
        for (uint32_t tilesDone = jobData.tilesDone.load(std::memory_order_acquire); tilesDone < numTiles;
             tilesDone          = jobData.tilesDone.load(std::memory_order_acquire))
        {
            //ctx.log("Progress: {} of {} tiles", std::make_tuple(tilesDone, numTiles));
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        threadPool.pauseJobs();

        dmt::img::mortonToLinear(dataRowMajor.get(), dataMorton.get(), width, height, channels);

        ctx.log("Saving grey image to {}", std::make_tuple(str));
        if (!::stbi_write_png(str.c_str(), width, height, channels, dataRowMajor.get(), width * channels))
            ctx.error("Failed save", {});
#endif
    }

    return 0;
}
