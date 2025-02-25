#include "platform-memory.h"

// Link against `VirtualAlloc2` Manually
// https://learn.microsoft.com/en-us/answers/questions/129563/how-to-use-virtualalloc2-memory-api
#pragma comment(lib, "mincore")
#include <AclAPI.h>
#include <Windows.h>
#include <securitybaseapi.h>
#include <sysinfoapi.h>
#include <PsApi.h>

#include "platform-os-utils.win32.h"

namespace dmt::tp {
    inline constexpr uint32_t                                           tokenPrivilegesBytes = 2048;
    static thread_local std::array<unsigned char, tokenPrivilegesBytes> sTokenPrivileges;
    using PVirtualAlloc = PVOID (*)(
        HANDLE                  Process,
        PVOID                   BaseAddress,
        SIZE_T                  Size,
        ULONG                   AllocationType,
        ULONG                   PageProtection,
        MEM_EXTENDED_PARAMETER* ExtendedParameters,
        ULONG                   ParameterCount);
    static constexpr uint32_t                              sErrorBufferSize = 256;
    static thread_local std::array<char, sErrorBufferSize> sErrorBuffer{};
} // namespace dmt::tp

namespace dmt {
    class HooksJanitor
    {
    public:
        HooksJanitor(LoggingContext& ctx, bool alloc) : m_ctx(ctx), m_alloc(alloc) {}
        HooksJanitor(HooksJanitor const&)                = delete;
        HooksJanitor(HooksJanitor&&) noexcept            = delete;
        HooksJanitor& operator=(HooksJanitor const&)     = delete;
        HooksJanitor& operator=(HooksJanitor&&) noexcept = delete;
        ~HooksJanitor() noexcept
        {
            if (pHooks && pAlloc)
            {
                if (m_alloc)
                {
                    pHooks->allocHook(pHooks->data, m_ctx, *pAlloc);
                }
                else
                {
                    pHooks->freeHook(pHooks->data, m_ctx, *pAlloc);
                }
            }
        }

        PageAllocatorHooks* pHooks = nullptr;
        PageAllocation*     pAlloc = nullptr;

    private:
        LoggingContext& m_ctx;
        bool            m_alloc;
    };

    class Janitor
    {
    public:
        Janitor()                              = default;
        Janitor(Janitor const&)                = delete;
        Janitor(Janitor&&) noexcept            = delete;
        Janitor& operator=(Janitor const&)     = delete;
        Janitor& operator=(Janitor&&) noexcept = delete;
        ~Janitor() noexcept
        {
            if (hProcessToken)
                CloseHandle(hProcessToken);
            if (mallocatedMem)
                std::free(mallocatedMem);
            if (pSecDescriptor)
                LocalFree(pSecDescriptor);
            if (bRevertToSelf)
                RevertToSelf();
            if (hImpersonationToken)
                CloseHandle(hImpersonationToken);
        }

        HANDLE               hProcessToken       = nullptr;
        void*                mallocatedMem       = nullptr;
        PSECURITY_DESCRIPTOR pSecDescriptor      = nullptr;
        bool                 bRevertToSelf       = false;
        HANDLE               hImpersonationToken = nullptr;
    };

    bool PageAllocator::checkAndAdjustPrivileges(LoggingContext& ctx,
                                                 void*           phProcessToken,
                                                 void const*     pseLockMemoryPrivilegeLUID,
                                                 void*           pData)
    {
        HANDLE      hProcessToken             = reinterpret_cast<HANDLE>(phProcessToken);
        Janitor&    janitor                   = *reinterpret_cast<Janitor*>(pData);
        LUID const& seLockMemoryPrivilegeLUID = *reinterpret_cast<LUID const*>(pseLockMemoryPrivilegeLUID);
        // 1. Get count of bytes for the TokenPrivileges TOKEN_INFORMATION_CLASS for this token handle
        DWORD requiredSize = 0;
        if (!GetTokenInformation(hProcessToken, TokenPrivileges, nullptr, 0, &requiredSize) &&
            GetLastError() != ERROR_INSUFFICIENT_BUFFER)
        {
            uint32_t length = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
            std::string_view view{tp::sErrorBuffer.data(), length};
            ctx.error(
                "Could not get the required size {} for the "
                "`TokenPrivileges` Token Information, error: {}",
                {requiredSize, view});
            return false;
        }

        // hoping we fit into the statically allocated buffer
        // note: we need to account for an additional LUID_AND_ATTRIBUTE in case the SeLockMemoryPrivilege
        // is not found. `requiredSize + sizeof(LUID_AND_ATTRIBUTES)` should respect the alignment
        // of LUID_AND_ATTRIBUTES because requiredSize is a (multiple of it + DWORD)
        static_assert(alignof(LUID_AND_ATTRIBUTES) == 4);
        void* pTokenPrivilegesInformation = nullptr;
        bool  mallocated                  = requiredSize > tp::tokenPrivilegesBytes - sizeof(LUID_AND_ATTRIBUTES);
        if (mallocated)
        {
            ctx.warn("Allocating on the Heap token information");
            pTokenPrivilegesInformation = std::malloc(requiredSize + sizeof(LUID_AND_ATTRIBUTES));
            if (pTokenPrivilegesInformation)
            {
                janitor.mallocatedMem = pTokenPrivilegesInformation;
            }
        }
        else
        {
            pTokenPrivilegesInformation = tp::sTokenPrivileges.data();
        }

        if (!pTokenPrivilegesInformation)
        {
            ctx.error("Couldn't reserve memory to hold the token information");
            return false;
        }

        // 2. actually get the TokenPrivileges TOKEN_INFORMATION_CLASS
        if (!GetTokenInformation(hProcessToken, TokenPrivileges, pTokenPrivilegesInformation, requiredSize, &requiredSize))
        {
            uint32_t length = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
            std::string_view view{tp::sErrorBuffer.data(), length};
            ctx.error("Could not get the `TokenPrivileges` Token Information, error: {}", {view});
            return false;
        }

        // 3. Iterate over the list of TOKEN_PRIVILEGES and find the one with the SeLockMemoryPrivilege LUID
        TOKEN_PRIVILEGES& tokenPrivilegeStruct = *reinterpret_cast<TOKEN_PRIVILEGES*>(&tp::sTokenPrivileges[0]);
        auto*             pPrivilegeLUIDs      = reinterpret_cast<LUID_AND_ATTRIBUTES*>(
            tp::sTokenPrivileges.data() + offsetof(TOKEN_PRIVILEGES, Privileges));
        bool    seLockMemoryPrivilegeEnabled = false;
        int64_t seLockMemoryPrivilegeIndex   = -1;
        for (uint32_t i = 0; i < tokenPrivilegeStruct.PrivilegeCount; ++i)
        {
            LUID  luid       = pPrivilegeLUIDs[i].Luid;
            DWORD attributes = pPrivilegeLUIDs[i].Attributes;
            if (os::win32::luidCompare(luid, seLockMemoryPrivilegeLUID))
            {
                // possible attributes: E_PRIVILEGE_ENABLED_BY_DEFAULT, SE_PRIVILEGE_ENABLED,
                // SE_PRIVILEGE_REMOVED, SE_PRIVILEGE_USED_FOR_ACCESS
                if ((attributes & SE_PRIVILEGE_ENABLED) != 0)
                    seLockMemoryPrivilegeEnabled = true;

                seLockMemoryPrivilegeIndex = i;
                break;
            }
        }

        // If the SeLockMemoryPrivilege is not enabled, then try to enable it
        if (!seLockMemoryPrivilegeEnabled)
        {
            return enableLockPrivilege(ctx, hProcessToken, pseLockMemoryPrivilegeLUID, seLockMemoryPrivilegeIndex, pData);
        }

        return true;
    }

    bool PageAllocator::checkVirtualAlloc2InKernelbaseDll(LoggingContext& ctx)
    {
        HMODULE hKernel32Dll = LoadLibraryA("kernelbase.dll");
        if (!hKernel32Dll)
        {
            uint32_t length = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
            std::string_view view{tp::sErrorBuffer.data(), length};
            ctx.error("Could not load the kernelbase.dll into an HMODULE, error: {}", {view});
            return false;
        }

        auto* functionPointer = reinterpret_cast<tp::PVirtualAlloc>(GetProcAddress(hKernel32Dll, "VirtualAlloc2"));
        if (!functionPointer)
        {
            uint32_t length = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
            std::string_view view{tp::sErrorBuffer.data(), length};
            ctx.error("Could not load the `VirtualAlloc2` from kernelbase.dll, using VirtualAlloc, error: {}", {view});
            ctx.error(
                "If you are unsure whether you support `VirtualAlloc2` "
                "or not, you can check the command `dumpbin /EXPORTS C:\\Windows\\System32\\kernelbase.dll | findstr "
                "VirtualAlloc2`");
            return false;
        }

        return true;
    }

    bool PageAllocator::enableLockPrivilege(LoggingContext& ctx,
                                            void*           phProcessToken,
                                            void const*     pseLockMemoryPrivilegeLUID,
                                            int64_t         seLockMemoryPrivilegeIndex,
                                            void*           pData)
    {
        HANDLE      hProcessToken             = reinterpret_cast<HANDLE>(phProcessToken);
        LUID const& seLockMemoryPrivilegeLUID = *reinterpret_cast<LUID const*>(pseLockMemoryPrivilegeLUID);
        // write into a new entry if we didn't find it at all
        // we are basically preparing the `NewState` parameter for the `AdjustTokenPrivileges`
        if (seLockMemoryPrivilegeIndex < 0)
        {
            // also HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management registry can
            // tell if you disabled it
            ctx.error(
                "SeLockMemoryPrivilege is absent in the current user token. "
                "Try to run as admin, check `secpol.msc` or use cudaAllocHost");
            return false;
        }
        else
        {
            TOKEN_PRIVILEGES privs; // Assuming ANYSIZE_ARRAY = 1
            privs.PrivilegeCount           = 1;
            privs.Privileges[0].Luid       = seLockMemoryPrivilegeLUID;
            privs.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
            if (!AdjustTokenPrivileges(hProcessToken, false, &privs, sizeof(TOKEN_PRIVILEGES), nullptr, nullptr))
            {
                uint32_t length = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
                std::string_view view{tp::sErrorBuffer.data(), length};
                ctx.error("Could not add SeLockMemoryPrivilege to the user token, error: {}", {view});
                return false;
            }

            int32_t errCode = GetLastError();
            if (errCode == ERROR_NOT_ALL_ASSIGNED)
            {
                // Sanity check
                PRIVILEGE_SET privilegeSet;
                privilegeSet.PrivilegeCount          = 1;
                privilegeSet.Control                 = PRIVILEGE_SET_ALL_NECESSARY;
                privilegeSet.Privilege[0].Luid       = seLockMemoryPrivilegeLUID;
                privilegeSet.Privilege[0].Attributes = SE_PRIVILEGE_ENABLED;
                BOOL result                          = false;
                if (!PrivilegeCheck(hProcessToken, &privilegeSet, &result))
                {
                    uint32_t length = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
                    std::string_view view{tp::sErrorBuffer.data(), length};
                    ctx.error("Call to `PrivilegeCheck` failed, error: {}", {view});
                    return false;
                }
                if (!result)
                {
                    ctx.error(
                        "Even though you called `AdjustTokenPrivileges`, the SeLockMemoryPrivilege is still nowhere to "
                        "be found");
                    return false;
                }

                return true;
            }
            else if (errCode == ERROR_SUCCESS)
            {
                return true;
            }
            else
            {
                ctx.error("`AdjustTokenPrivilege` returned an unexpected error code: {}", {errCode});
                return false;
            }
        }
    }

    void* PageAllocator::createImpersonatingThreadToken(LoggingContext& ctx, void* phProcessToken, void* pData)
    {
        HANDLE   hProcessToken       = reinterpret_cast<HANDLE>(phProcessToken);
        Janitor& janitor             = *reinterpret_cast<Janitor*>(pData);
        HANDLE   hImpersonationToken = nullptr;
        if (!DuplicateToken(hProcessToken, SecurityImpersonation, &hImpersonationToken))
        {
            uint32_t length = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
            std::string_view view{tp::sErrorBuffer.data(), length};
            ctx.error("Failed call to `OpenThreadToken`, error: {}", {view});
            return INVALID_HANDLE_VALUE;
        }
        janitor.hImpersonationToken = hImpersonationToken;
        HANDLE hThread              = GetCurrentThread();
        if (!SetThreadToken(&hThread, hImpersonationToken))
        {
            uint32_t length = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
            std::string_view view{tp::sErrorBuffer.data(), length};
            ctx.error("Failed call to `SetThreadToken`, error: {}", {view});
            return INVALID_HANDLE_VALUE;
        }
        janitor.bRevertToSelf = true;
        return hImpersonationToken;
    }

    // TODO error handing with the Janitor Pattern
    // the Process Security Descriptor, the Token -> Close
    PageAllocator::PageAllocator(LoggingContext& ctx, PageAllocatorHooks const& hooks) : PageAllocator(hooks)
    {
        Janitor janitor;

        // Get some of the system information relevant to memory allocation, eg
        // - when `VirtualAlloc` is called with MEM_RESERVE, the allocation is aligned to the `allocation granularity`
        // - when `VirtualAlloc` it called with MEM_COMMIT, the allocation is aligned to a page boundary
        SYSTEM_INFO sysInfo{};
        GetSystemInfo(&sysInfo);
        m_systemPageSize        = sysInfo.dwPageSize;
        m_allocationGranularity = sysInfo.dwAllocationGranularity;
        if (m_systemPageSize == 0)
        {
            ctx.error("The current system does not support large pages for some reason");
            return;
        }

        // Retrieve the LUID associated with the SE_LOCK_MEMORY_NAME = "SeLockMemoryPrivilege"
        LUID seLockMemoryPrivilegeLUID{};
        if (!LookupPrivilegeValue(nullptr /*on the local system*/, SE_LOCK_MEMORY_NAME, &seLockMemoryPrivilegeLUID))
        {
            uint32_t length = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
            std::string_view view{tp::sErrorBuffer.data(), length};
            ctx.error("Could not retrieve the LUID for the SeLockMemoryPrivilege. Error: {}", {view});
            return;
        }

        // get the pseudo handle (fixed to -1) of the current process (you call a
        // function anyways for compatibility with the future)
        // pseudo handles need not to be `Closehandle`d
        HANDLE hCurrentProc = GetCurrentProcess();

        // Retrieve the user access token associated to the user of the current process.
        // Open it in DesiredAccessMode = TOKEN_ADJUST_PRIVILEDGES, NOT TOKEN_QUERY, such that, if we need
        // we can add some access control entries into it
        // TOKEN_DUPLICATE and TOKEN_IMPOERSONATION for AccessCheck, as they allow me
        // to duplicate the process token to impersonate the user with a thread token
        // see https://learn.microsoft.com/en-us/windows/win32/secauthz/access-rights-for-access-token-objects
        HANDLE hProcessToken   = nullptr;
        DWORD  tokenAccessMode = TOKEN_QUERY | TOKEN_ADJUST_PRIVILEGES | TOKEN_DUPLICATE | TOKEN_IMPERSONATE;
        if (!OpenProcessToken(hCurrentProc, tokenAccessMode, &hProcessToken))
        {
            uint32_t length = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
            std::string_view view{tp::sErrorBuffer.data(), length};
            ctx.error(
                "Couldn't open in TOKEN_ADJUST_PRIVILEDGES "
                "mode the user access token. Error: {}",
                {view});
            // TODO error handling
        }
        janitor.hProcessToken = hProcessToken;

        // iterate over the existing priviledges on the user token, and if you find SE_LOCK_MEMORY_NAME with
        // attribute SE_PRIVILEGE_ENABLED, then you are good to go
        bool seLockMemoryPrivilegeEnabled = checkAndAdjustPrivileges(ctx, hProcessToken, &seLockMemoryPrivilegeLUID, (void*)&janitor);

        // still not enabled? Fail.
        if (!seLockMemoryPrivilegeEnabled)
        {
            ctx.error("Found SeLockMemoryPrivilege not enabled, hence no large page allocation");
            return;
        }

        // Phase 2: Retrieve the minimum large page the processor supports. if 0, then the processor doesn't support them
        // since we are on x86_64, it should always be different than 0
        size_t minimumPageSize = GetLargePageMinimum();
        if (minimumPageSize == 0)
        {
            ctx.error("For some reason, the current CPU architecture doesn't support large TLB entries");
            return;
        }

        // The size and alignment must be a multiple of the large-page minimum
        if (static_cast<size_t>(toUnderlying(EPageSize::e2MB)) % minimumPageSize != 0)
        {
            ctx.error("Page Size we support (2MB) is not a multiple of the MinimumPageSize, {} B", {minimumPageSize});
            return;
        }

        // ------------ ENABLE LARGE PAGES ------------------------------------------------------------------
        // TODO check miminum page size better
        m_largePageEnabled = true;

        // At this point, we know that we can use Large Page Allocation. We need to decide among 2MB and 1GB
        // - 2MB is allowed by default
        // - 1GB requires 1) support for `VirtualAlloc2`, 2) `PROCESS_VM_OPERATION` Access Right
        // 1. Check whether we support VirtualAlloc2 (Windows 10 and above)
        // to seach for a module specifically, take a look to AddDllDirectory. Usually you can use this
        // only on `LoadLibrary` DLLs. If you didn't allocate the library, don't free it (hence why it is called "Get")
        bool bVirtualAlloc2Supported = checkVirtualAlloc2InKernelbaseDll(ctx);
        if (!bVirtualAlloc2Supported)
        {
            return;
        }

        // 2. Look into the DACL of the current process to see whether you have the `PROCESS_VM_OPERATION` access right
        // Docs: https://learn.microsoft.com/en-us/windows/win32/procthread/process-security-and-access-rights
        // 2.1 First retrieve the Process Security Descriptor (to then free With LocalFree)
        PSECURITY_DESCRIPTOR securityDescriptor = nullptr;

        // left comments for future reference
        SECURITY_INFORMATION securityInfo = OWNER_SECURITY_INFORMATION | GROUP_SECURITY_INFORMATION |
                                            DACL_SECURITY_INFORMATION /*| SACL_SECURITY_INFORMATION|
                                            LABEL_SECURITY_INFORMATION | ATTRIBUTE_SECURITY_INFORMATION |
                                            SCOPE_SECURITY_INFORMATION | PROCESS_TRUST_LABEL_SECURITY_INFORMATION |
                                            ACCESS_FILTER_SECURITY_INFORMATION | BACKUP_SECURITY_INFORMATION
                                            */
            ;
        DWORD status = GetSecurityInfo(hCurrentProc,     // the current process HANDLE
                                       SE_KERNEL_OBJECT, // a process is classified as a kernel object
                                       securityInfo, // bits of the info to retrieve. we want process specific (discretionary)
                                       nullptr, // Owner SID
                                       nullptr, // Group SID
                                       nullptr, // DACL
                                       nullptr, // SACL
                                       &securityDescriptor);

        if (status != ERROR_SUCCESS)
        {
            ctx.error("Could not retrieve the Process Security Descriptor, error code {}", {status});
            return;
        }

        janitor.pSecDescriptor = securityDescriptor;

        if (!IsValidSecurityDescriptor(securityDescriptor))
        {
            ctx.error("The retrieved security descriptor at {} is not valid", {securityDescriptor});
            return;
        }

        // GENERIC_MAPPING = r?w?x?. Each member is an int, ACCESS_MASK, https://learn.microsoft.com/en-us/windows/win32/secauthz/access-mask
        GENERIC_MAPPING genericMapping = {
            PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, // GENERIC_READ
            PROCESS_VM_WRITE | PROCESS_VM_OPERATION,     // GENERIC_WRITE
            PROCESS_CREATE_THREAD,                       // GENERIC_EXECUTE
            PROCESS_ALL_ACCESS                           // GENERIC_ALL
        };

        ACCESS_MASK   outAccessMask = 0;
        DWORD         desiredAccess = PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION;
        BOOL          bAccessStatus = false;
        PRIVILEGE_SET privilegeSet;
        DWORD         privilegeSetSize = sizeof(PRIVILEGE_SET);
        // https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-mapgenericmask
        MapGenericMask(&desiredAccess, &genericMapping);

        // `AccessCheck` function to see whether the process security descriptor has a predefined
        // set of access rights
        // `AccessCheck` requires a *Client Token*, which is a token associated to some entity (local user or client-server), derived from a primary token
        // But `OpenProcessToken` returns a *Primary Token*, which represents the user account under which the process is running
        // To Impersonate a client token from a primary token, use `ImpersonateSelf`
        // source: https://stackoverflow.com/questions/35027524/whats-the-difference-between-a-primary-token-and-an-impersonation-token
        // basically, AccessCheck works with thread tokens, not process tokens, so we need to
        // fetch the association user - process and map it onto the current thread
        // source: book "Programming Windows Security"
        //if (!ImpersonateSelf(SecurityImpersonation))
        //{
        //    uint32_t         length = ::dmt::os::win32::getLastErrorAsString(sErrorBuffer.data(), sErrorBufferSize);
        //    std::string_view view{sErrorBuffer.data(), length};
        //    ctx.error("Failed call to `ImpersonateSelf`, error: {}", {view});
        //    return;
        //}
        //janitor.bRevertToSelf = true;
        // source: https://blog.aaronballman.com/2011/08/how-to-check-access-rights/
        HANDLE hImpersonationToken = reinterpret_cast<HANDLE>(
            createImpersonatingThreadToken(ctx, hProcessToken, (void*)&janitor));
        if (hImpersonationToken == INVALID_HANDLE_VALUE)
        {
            return;
        }

        if (!AccessCheck(securityDescriptor,  // security descriptor against which access is checked
                         hImpersonationToken, // impersonation token representing the user attempting the access
                         desiredAccess,       // desired access rights
                         &genericMapping,
                         &privilegeSet,
                         &privilegeSetSize,
                         &outAccessMask,
                         &bAccessStatus))
        {
            uint32_t length = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
            std::string_view view{tp::sErrorBuffer.data(), length};
            ctx.error("Failed call to AccessCheck, error: {}", {view});
            return;
        }

        if (!bAccessStatus)
        {
            uint32_t length = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
            std::string_view view{tp::sErrorBuffer.data(), length};
            ctx.error(
                "The Process doesn't own the"
                " PROCESS_VM_OPERATION access rights, using 2MB large pages. Error: {}",
                {view});
            return;
        }

        m_largePage1GB = true;
    }

    enum PageAllocationFlags : uint32_t
    {
        DMT_OS_WINDOWS_SMALL_PAGE    = 0,
        DMT_OS_WINDOWS_LARGE_PAGE    = 1 << 0, // Use bit 0
        DMT_OS_WINDOWS_VIRTUALALLOC2 = 1 << 1  // Use bit 1
    };

    // TODO promote to class static protected, shared between windows and linux
    void PageAllocator::addAllocInfo(LoggingContext& ctx, bool isLargePageAlloc, PageAllocation& ret)
    {
        static constexpr uint32_t log4KB      = 12u;
        uint32_t                  errorLength = 0;
        // if you allocated the page with MEM_LARGE_PAGES, then it is locked to memory, and hence, like AWE,
        // is NOT part of the working set of the process.
        // Therefore you can get the physical frame nubmber with `QueryWorkingSetEx`
        // if instead you didn't allocate with MEM_LARGE_PAGES, then the allocated block is part of
        // the working set, and therefore you can use `QueryWorkingSet`
        // (Requires PROCESS_QUERY_INFORMATION and PROCESS_VM_READ access right to the process)
        MEMORY_BASIC_INFORMATION memoryInfo{};
        size_t                   memoryInfoBytes = ret.count * sizeof(MEMORY_BASIC_INFORMATION);

        // VirtualAddress = input, VirtualAttributes = output
        PSAPI_WORKING_SET_EX_INFORMATION input{};
        input.VirtualAddress = ret.address;
        uint32_t inputSize   = sizeof(PSAPI_WORKING_SET_EX_INFORMATION);
        if (!QueryWorkingSetEx(GetCurrentProcess(), &input, inputSize))
        {
            errorLength = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
            std::string_view view{tp::sErrorBuffer.data(), errorLength};
            ctx.error("Call to `QueryWorkingSetEx` failed, hence cannot check page status, error: {}", {view});
        }
        else if (!input.VirtualAttributes.Valid)
        {
            ctx.error("Call to `QueryWorkingSetEx` succeeded, but for some reason attributes are invalid, {}",
                      {input.VirtualAttributes.Flags});
        }
        else
        {
            // POSSIBLE TODO = use `VirtualLock` to lock a page even if not large
            bool effectivelyLarge = input.VirtualAttributes.LargePage;
            bool locked           = input.VirtualAttributes.Locked;
            if (isLargePageAlloc != effectivelyLarge)
            {
                ctx.error("Allocation should be large? {}. But found value {}", {isLargePageAlloc, effectivelyLarge});
            }

            if (isLargePageAlloc && !locked)
            {
                ctx.error("Allocation should be large, but its not locked");
            }
        }

        size_t numBytes = VirtualQuery(ret.address, &memoryInfo, memoryInfoBytes);
        if (numBytes == 0)
        {
            errorLength = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
            std::string_view view{tp::sErrorBuffer.data(), errorLength};
            ctx.error("Call to `VirtualQuery` failed, hence cannot acquire virtual page number, error: {}", {view});
        }
        else
        {
            size_t   expectedSize = toUnderlying(ret.pageSize);
            void*    pRegion      = memoryInfo.BaseAddress;
            uint64_t region       = (uint64_t)pRegion;
            uint64_t mask         = toUnderlying(EPageSize::e4KB) - 1;
            if (expectedSize != memoryInfo.RegionSize)
            {
                ctx.error("Expected region at {} size to be {} B but found {} B",
                          {pRegion, expectedSize, memoryInfo.RegionSize});
            }

            if ((memoryInfo.State & MEM_COMMIT) == 0)
            {
                ctx.error("Expected memory region at {} to be committed, but it's not", {pRegion});
            }

            if ((region & mask) != 0)
            {
                ctx.error("Expected memory region at {} to be aligned to a 4KB boundary. It's not.", {pRegion});
            }
            else
            {
                ret.pageNum = region >> log4KB;
            }
        }

        m_hooks.allocHook(m_hooks.data, ctx, ret);
    }

    // NOTE: Only Pinned memory can be mapped to CUDA device memory, hence you need to
    // check that the page size is not 4KB. Mapping is carried out with cudaHostRegister
    // reserve = take virtual address space. commit = when you write to it, it will be backed by physical memory
    // TODO see Address Windowing Extension pages (AWE)
    // TODO integrate MEM_WRITE_WATCH
    PageAllocation PageAllocator::allocatePage(LoggingContext& ctx, EPageSize sizeOverride)
    {
        HooksJanitor janitor{ctx, true};
        janitor.pHooks = &m_hooks;

        static constexpr uint32_t log4KB = 12u;
        PageAllocation            ret{};
        ret.pageNum               = -1;
        uint32_t  errorLength     = 0;
        DWORD     allocationFlags = MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES;
        DWORD     protectionFlags = PAGE_READWRITE;
        EPageSize pageSize        = sizeOverride;

        if (m_largePageEnabled && pageSize >= EPageSize::e2MB)
        {
            if (m_largePage1GB && sizeOverride == EPageSize::e1GB)
            {
                // necessary additional parameter to perform 1GB allocation attempt
                MEM_EXTENDED_PARAMETER extended{};
                extended.Type    = MemExtendedParameterAttributeFlags;
                extended.ULong64 = MEM_EXTENDED_PARAMETER_NONPAGED_HUGE;

                ret.address = VirtualAlloc2( //
                    nullptr /*current process*/,
                    nullptr /*no base hint*/,
                    toUnderlying(pageSize),
                    allocationFlags,
                    protectionFlags,
                    &extended,
                    1);
                ret.bits    = DMT_OS_WINDOWS_VIRTUALALLOC2;
                if (!ret.address)
                {
                    errorLength = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
                    std::string_view view{tp::sErrorBuffer.data(), errorLength};
                    ctx.error("Failed call to `VirtualAlloc2`, trying `VirtualAlloc`, error: {}", {view});
                }
            }

            // If you are allowed to perform 1GB large page allocation, but failed for some
            // reason, fall back to 2MB allocation
            if (!ret.address)
            { // https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc
                pageSize = EPageSize::e2MB;

                // call VirtualAlloc. How is this memory aligned? See this link
                // https://stackoverflow.com/questions/20023446/is-virtualalloc-alignment-consistent-with-size-of-allocation
                ret.address = VirtualAlloc(nullptr, toUnderlying(pageSize), allocationFlags, protectionFlags);
                ret.bits    = DMT_OS_WINDOWS_LARGE_PAGE;
                if (!ret.address)
                {
                    errorLength = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
                    std::string_view view{tp::sErrorBuffer.data(), errorLength};
                    ctx.error("`VirtualAlloc` with MEM_LARGE_PAGES, trying without it, error: {}", {view});
                }
            }
        }

        if (!ret.address)
        {
            allocationFlags &= ~MEM_LARGE_PAGES;
            pageSize    = EPageSize::e4KB;
            ret.address = VirtualAlloc(nullptr, toUnderlying(pageSize), allocationFlags, protectionFlags);
            ret.bits    = DMT_OS_WINDOWS_SMALL_PAGE;
            if (!ret.address)
            {
                errorLength = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
                std::string_view view{tp::sErrorBuffer.data(), errorLength};
                ctx.error("`VirtualAlloc` failed, couldn't allocate memory, error: {}", {view});
            }
        }

        if (ret.address)
        {
            janitor.pAlloc = &ret;
            // touch first byte to make sure committed memory is backed to physical memory
            reinterpret_cast<unsigned char*>(ret.address)[0] = 0;

            // bookkeeping
            ret.count             = 1;
            ret.pageSize          = pageSize;
            bool isLargePageAlloc = (allocationFlags & MEM_LARGE_PAGES) != 0;

            ctx.trace(
                "Called allocatePage, allocated "
                "at {} page of {}",
                {ret.address, StrBuf{toUnderlying(ret.pageSize), "0x%zx"}});
            ctx.dbgTraceStackTrace();
        }
        else
        {
            ctx.dbgErrorStackTrace();
        }

        return ret;
    }

    bool PageAllocator::allocate2MB(LoggingContext& ctx, PageAllocation& out)
    {
        HooksJanitor janitor{ctx, true};
        janitor.pHooks               = &m_hooks;
        DWORD        allocationFlags = MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES;
        DWORD const  protectionFlags = PAGE_READWRITE;
        size_t const size            = toUnderlying(EPageSize::e2MB);
        out.address                  = nullptr;
        bool     isLargePage         = false;
        uint32_t errorLength         = 0;
        if (m_largePageEnabled)
        {
            out.address = VirtualAlloc(nullptr, size, allocationFlags, protectionFlags);
            if (!out.address)
            {
                errorLength = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
                std::string_view view{tp::sErrorBuffer.data(), errorLength};
                ctx.error("`VirtualAlloc` with MEM_LARGE_PAGES, trying without it, error: {}", {view});
            }
            else
            {
                out.count    = 1;
                out.pageSize = EPageSize::e2MB;
                out.bits     = DMT_OS_WINDOWS_LARGE_PAGE;
                isLargePage  = true;
            }
        }

        if (!out.address)
        {
            allocationFlags &= ~MEM_LARGE_PAGES;
            out.address = VirtualAlloc(nullptr, size, allocationFlags, protectionFlags);
            if (!out.address)
            {
                errorLength = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
                std::string_view view{tp::sErrorBuffer.data(), errorLength};
                ctx.error("`VirtualAlloc` failed, error: {}", {view});
                return false;
            }
            else
            {
                out.count    = num4KBIn2MB;
                out.pageSize = EPageSize::e4KB;
                out.bits     = DMT_OS_WINDOWS_SMALL_PAGE;
            }
        }

        janitor.pAlloc = &out;

        return true;
    }

    void PageAllocator::deallocatePage(LoggingContext& ctx, PageAllocation& alloc)
    {
        // the page allocation might be part of the memory portion itself, we need to copy it
        PageAllocation allocCopy = alloc;

        // decommit = get out of memory, but leave the virtual address space reserved
        // POSSIBLE TODO. The other type, release, will decommit + free up the virtual address space
        DWORD freeType = MEM_RELEASE;
        if (!VirtualFree(alloc.address, 0 /*size 0 if MEM_RELEASE*/, freeType))
        {
            uint32_t errorLength = ::dmt::os::win32::getLastErrorAsString(tp::sErrorBuffer.data(), tp::sErrorBufferSize);
            std::string_view view{tp::sErrorBuffer.data(), errorLength};
            ctx.error("Failed to Free memory at {}, error {}", {allocCopy.address, view});
            ctx.dbgErrorStackTrace();
        }
        else
        {
            ctx.trace("Deallocated memory at {} size {}", {allocCopy.address, toUnderlying(allocCopy.pageSize)});
            ctx.dbgTraceStackTrace();
        }
    }

    PageAllocator::~PageAllocator() {}
} // namespace dmt
