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

        ctx.log("Joining Thread {}", std::make_tuple(t.id()));
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


        int32_t                   width = 512, height = 512, channels = 3;
        dmt::UniqueRef<uint8_t[]> data = dmt::makeUniqueRef<uint8_t[]>(std::pmr::get_default_resource(),
                                                                       width * height * channels);
        for (size_t i = 0; i < width * height; ++i)
        {
            data[i * channels]     = 128;
            data[i * channels + 1] = 128;
            data[i * channels + 2] = 128;
        }

        str += "\\image.png";
        ctx.log("Saving grey image to {}", std::make_tuple(str));
        if (!::stbi_write_png(str.c_str(), width, height, channels, data.get(), width * channels))
            ctx.error("Failed save", {});
#endif
    }

    return 0;
}
