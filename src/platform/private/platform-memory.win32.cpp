#include "platform-memory.h"

#include "platform-context.h"

// Link against `VirtualAlloc2` Manually
// https://learn.microsoft.com/en-us/answers/questions/129563/how-to-use-virtualalloc2-memory-api
#pragma comment(lib, "mincore")
#include <AclAPI.h>
#include <Windows.h>
#include <securitybaseapi.h>
#include <sysinfoapi.h>
#include <PsApi.h>

#include "platform-os-utils.win32.h"

static constexpr uint32_t errorBufferSize = 1024;
static thread_local char s_errorBuffer[errorBufferSize];

static std::string_view getLastWin32Error() {
  uint32_t const length =
      ::dmt::os::win32::getLastErrorAsString(s_errorBuffer, errorBufferSize);
  std::string_view view{s_errorBuffer, length};
  return view;
}

static bool systemSupportsLargePages() {
  SYSTEM_INFO sysInfo{};
  GetSystemInfo(&sysInfo);
  return sysInfo.dwPageSize > 0 && GetLargePageMinimum() > 0;
}

namespace detail {

static bool enableLockPrivilege(dmt::Context& ctx, HANDLE hProcessToken,
                                LUID seLockMemoryPrivilegeLUID,
                                int64_t seLockMemoryPrivilegeIndex) {
  // write into a new entry if we didn't find it at all
  // we are basically preparing the `NewState` parameter for the
  // `AdjustTokenPrivileges`
  if (seLockMemoryPrivilegeIndex < 0) {
    // also HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session
    // Manager\Memory Management registry can tell if you disabled it
    if (ctx.isValid())
      ctx.error(
          "SeLockMemoryPrivilege is absent in the current user token. "
          "Try to run as admin, check `secpol.msc` or use cudaAllocHost",
          {});
    return false;
  } else {
    TOKEN_PRIVILEGES privs;  // Assuming ANYSIZE_ARRAY = 1
    privs.PrivilegeCount = 1;
    privs.Privileges[0].Luid = seLockMemoryPrivilegeLUID;
    privs.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    if (!AdjustTokenPrivileges(hProcessToken, false, &privs,
                               sizeof(TOKEN_PRIVILEGES), nullptr, nullptr)) {
      if (ctx.isValid())
        ctx.error(
            "Could not add SeLockMemoryPrivilege to the user token, error: {}",
            std::make_tuple(getLastWin32Error()));
      return false;
    }

    int32_t errCode = GetLastError();
    if (errCode == ERROR_NOT_ALL_ASSIGNED) {
      // Sanity check
      PRIVILEGE_SET privilegeSet;
      privilegeSet.PrivilegeCount = 1;
      privilegeSet.Control = PRIVILEGE_SET_ALL_NECESSARY;
      privilegeSet.Privilege[0].Luid = seLockMemoryPrivilegeLUID;
      privilegeSet.Privilege[0].Attributes = SE_PRIVILEGE_ENABLED;
      BOOL result = false;
      if (!PrivilegeCheck(hProcessToken, &privilegeSet, &result)) {
        if (ctx.isValid())
          ctx.error("Call to `PrivilegeCheck` failed, error: {}",
                    std::make_tuple(getLastWin32Error()));
        return false;
      }
      if (!result) {
        if (ctx.isValid())
          ctx.error(
              "Even though you called `AdjustTokenPrivileges`, the "
              "SeLockMemoryPrivilege is still "
              "nowhere to "
              "be found",
              {});
        return false;
      }

      return true;
    } else if (errCode == ERROR_SUCCESS) {
      return true;
    } else {
      if (ctx.isValid())
        ctx.error(
            "`AdjustTokenPrivilege` returned an unexpected error code: {}",
            std::make_tuple(errCode));
      return false;
    }
  }
}

// note: now it uses the default allocator
static bool checkAndAdjustPrivileges(dmt::Context& ctx, HANDLE hProcessToken,
                                     LUID seLockMemoryPrivilegeLUID) {
  struct Janitor {
    ~Janitor() {
      assert(memory);
      if (pTokenPrivilegesInformation)
        memory->deallocate(pTokenPrivilegesInformation,
                           requiredSize + sizeof(LUID_AND_ATTRIBUTES));
    }

    std::pmr::memory_resource* memory = std::pmr::get_default_resource();

    DWORD requiredSize = 0;
    void* pTokenPrivilegesInformation = nullptr;
  } j;

  // 1. Get count of bytes for the TokenPrivileges TOKEN_INFORMATION_CLASS for
  // this token handle
  if (!GetTokenInformation(hProcessToken, TokenPrivileges, nullptr, 0,
                           &j.requiredSize) &&
      GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
    if (ctx.isValid())
      ctx.error(
          "Could not get the required size {} for the "
          "`TokenPrivileges` Token Information, error: {}",
          std::make_tuple(j.requiredSize, getLastWin32Error()));
    return false;
  }

  // hoping we fit into the statically allocated buffer
  // note: we need to account for an additional LUID_AND_ATTRIBUTE in case the
  // SeLockMemoryPrivilege is not found. `requiredSize +
  // sizeof(LUID_AND_ATTRIBUTES)` should respect the alignment of
  // LUID_AND_ATTRIBUTES because requiredSize is a (multiple of it + DWORD)
  static_assert(alignof(LUID_AND_ATTRIBUTES) == 4);
  j.pTokenPrivilegesInformation =
      j.memory->allocate(j.requiredSize + sizeof(LUID_AND_ATTRIBUTES));
  if (!j.pTokenPrivilegesInformation) {
    ctx.error("Couldn't reserve memory to hold the token information", {});
    return false;
  }

  // 2. actually get the TokenPrivileges TOKEN_INFORMATION_CLASS
  if (!GetTokenInformation(hProcessToken, TokenPrivileges,
                           j.pTokenPrivilegesInformation, j.requiredSize,
                           &j.requiredSize)) {
    if (ctx.isValid())
      ctx.error(
          "Could not get the `TokenPrivileges` Token Information, error: {}",
          std::make_tuple(getLastWin32Error()));
    return false;
  }

  // 3. Iterate over the list of TOKEN_PRIVILEGES and find the one with the
  // SeLockMemoryPrivilege LUID
  TOKEN_PRIVILEGES& tokenPrivilegeStruct =
      *reinterpret_cast<TOKEN_PRIVILEGES*>(j.pTokenPrivilegesInformation);
  bool seLockMemoryPrivilegeEnabled = false;
  int64_t seLockMemoryPrivilegeIndex = -1;
  for (uint32_t i = 0; i < tokenPrivilegeStruct.PrivilegeCount; ++i) {
    LUID luid = tokenPrivilegeStruct.Privileges[i].Luid;
    DWORD attributes = tokenPrivilegeStruct.Privileges[i].Attributes;
    if (dmt::os::win32::luidCompare(luid, seLockMemoryPrivilegeLUID)) {
      // possible attributes: E_PRIVILEGE_ENABLED_BY_DEFAULT,
      // SE_PRIVILEGE_ENABLED, SE_PRIVILEGE_REMOVED,
      // SE_PRIVILEGE_USED_FOR_ACCESS
      if ((attributes & SE_PRIVILEGE_ENABLED) != 0)
        seLockMemoryPrivilegeEnabled = true;

      seLockMemoryPrivilegeIndex = i;
      break;
    }
  }

  // If the SeLockMemoryPrivilege is not enabled, then try to enable it
  if (!seLockMemoryPrivilegeEnabled) {
    return enableLockPrivilege(ctx, hProcessToken, seLockMemoryPrivilegeLUID,
                               seLockMemoryPrivilegeIndex);
  }

  return true;
}

struct HandlePair {
  HANDLE procImpersonationToken;
  bool shouldRevert;
};

// NOTE: Caller cleans up the two handles
static HandlePair createImpersonatingThreadToken(dmt::Context& ctx,
                                                 HANDLE hProcessToken) {
  HANDLE hImpersonationToken = nullptr;
  if (!DuplicateToken(hProcessToken, SecurityImpersonation,
                      &hImpersonationToken)) {
    if (ctx.isValid())
      ctx.error("Failed call to `OpenThreadToken`, error: {}",
                std::make_tuple(getLastWin32Error()));
    return {INVALID_HANDLE_VALUE, false};
  }
  HANDLE hThread = GetCurrentThread();
  if (!SetThreadToken(&hThread, hImpersonationToken)) {
    if (ctx.isValid())
      ctx.error("Failed call to `SetThreadToken`, error: {}",
                std::make_tuple(getLastWin32Error()));
    return {hImpersonationToken, false};
  }

  // TODO caller cleanup
  // janitor.hImpersonationToken = hImpersonationToken;
  // janitor.bRevertToSelf = true;
  return {hImpersonationToken, true};
}

static void cleanupImpersonatingThreadToken(HandlePair pair) {
  if (pair.procImpersonationToken != INVALID_HANDLE_VALUE) {
    if (pair.shouldRevert) RevertToSelf();
    CloseHandle(pair.procImpersonationToken);
  }
}
}  // namespace detail

static HANDLE checkForSeMemoryLockPrivilegeAndGetProcessToken(
    dmt::Context& ctx) {
  LUID seLockMemoryPrivilegeLUID{};
  if (!LookupPrivilegeValue(nullptr /*on the local system*/,
                            SE_LOCK_MEMORY_NAME, &seLockMemoryPrivilegeLUID)) {
    if (ctx.isValid())
      ctx.error(
          "Could not retrieve the LUID for the SeLockMemoryPrivilege. Error: "
          "{}",
          std::make_tuple(getLastWin32Error()));
    return INVALID_HANDLE_VALUE;
  }

  // get the pseudo handle (fixed to -1) of the current process (you call a
  // function anyways for compatibility with the future)
  // pseudo handles need not to be `Closehandle`d
  HANDLE hCurrentProc = GetCurrentProcess();

  // Retrieve the user access token associated to the user of the current
  // process. Open it in DesiredAccessMode = TOKEN_ADJUST_PRIVILEDGES, NOT
  // TOKEN_QUERY, such that, if we need we can add some access control entries
  // into it TOKEN_DUPLICATE and TOKEN_IMPOERSONATION for AccessCheck, as they
  // allow me to duplicate the process token to impersonate the user with a
  // thread token see
  // https://learn.microsoft.com/en-us/windows/win32/secauthz/access-rights-for-access-token-objects
  DWORD tokenAccessMode = TOKEN_QUERY | TOKEN_ADJUST_PRIVILEGES |
                          TOKEN_DUPLICATE | TOKEN_IMPERSONATE;
  HANDLE hProcessToken = INVALID_HANDLE_VALUE;
  if (!OpenProcessToken(hCurrentProc, tokenAccessMode, &hProcessToken)) {
    if (ctx.isValid())
      ctx.error(
          "Couldn't open in TOKEN_ADJUST_PRIVILEDGES "
          "mode the user access token. Error: {}",
          std::make_tuple(getLastWin32Error()));
    return INVALID_HANDLE_VALUE;
  }

  // iterate over the existing priviledges on the user token, and if you find
  // SE_LOCK_MEMORY_NAME with attribute SE_PRIVILEGE_ENABLED, then you are good
  // to go
  bool seLockMemoryPrivilegeEnabled = detail::checkAndAdjustPrivileges(
      ctx, hProcessToken, seLockMemoryPrivilegeLUID);

  // still not enabled? Fail.
  if (!seLockMemoryPrivilegeEnabled) {
    ctx.error(
        "Found SeLockMemoryPrivilege not enabled, hence no large page "
        "allocation",
        {});
    CloseHandle(hProcessToken);
    return INVALID_HANDLE_VALUE;
  }

  return hProcessToken;
}

static bool checkVirtualAlloc2InKernelbaseDll(dmt::Context& ctx) {
  // we don't need to unload it, as its probably already there
  HMODULE hKernel32Dll = LoadLibraryW(L"kernelbase.dll");
  if (!hKernel32Dll) {
    if (ctx.isValid())
      ctx.error("Could not load the kernelbase.dll into an HMODULE, error: {}",
                std::make_tuple(getLastWin32Error()));
    return false;
  }

  void* functionPointer = GetProcAddress(hKernel32Dll, "VirtualAlloc2");
  if (!functionPointer) {
    if (ctx.isValid()) {
      ctx.error(
          "Could not load the `VirtualAlloc2` from kernelbase.dll, using "
          "VirtualAlloc, error: {}",
          std::make_tuple(getLastWin32Error()));
      ctx.error(
          "If you are unsure whether you support `VirtualAlloc2` "
          "or not, you can check the command `dumpbin /EXPORTS "
          "C:\\Windows\\System32\\kernelbase.dll | findstr "
          "VirtualAlloc2`",
          {});
    }
    return false;
  }

  return true;
}

static bool checkProcessHasVmOperationInDACL(dmt::Context& ctx,
                                             HANDLE hProcessToken) {
  struct Janitor {
    ~Janitor() {
      detail::cleanupImpersonatingThreadToken(pair);
      if (securityDescriptor) LocalFree(securityDescriptor);
    }

    detail::HandlePair pair{INVALID_HANDLE_VALUE, false};
    PSECURITY_DESCRIPTOR securityDescriptor = nullptr;
  } j;
  // - 1GB requires 1) support for `VirtualAlloc2`, 2) `PROCESS_VM_OPERATION`
  // Access Right Look into the DACL of the current process to see whether you
  // have the `PROCESS_VM_OPERATION` access right Docs:
  // https://learn.microsoft.com/en-us/windows/win32/procthread/process-security-and-access-rights
  // 2.1 First retrieve the Process Security Descriptor (to then free With
  // LocalFree)

  // left comments for future reference
  SECURITY_INFORMATION securityInfo =
      OWNER_SECURITY_INFORMATION | GROUP_SECURITY_INFORMATION |
      DACL_SECURITY_INFORMATION /*| SACL_SECURITY_INFORMATION|
      LABEL_SECURITY_INFORMATION | ATTRIBUTE_SECURITY_INFORMATION |
      SCOPE_SECURITY_INFORMATION | PROCESS_TRUST_LABEL_SECURITY_INFORMATION |
      ACCESS_FILTER_SECURITY_INFORMATION | BACKUP_SECURITY_INFORMATION
      */
      ;
  DWORD status = GetSecurityInfo(
      GetCurrentProcess(),  // the current process HANDLE
      SE_KERNEL_OBJECT,     // a process is classified as a kernel object
      securityInfo,  // bits of the info to retrieve. we want process specific
                     // (discretionary)
      nullptr,       // Owner SID
      nullptr,       // Group SID
      nullptr,       // DACL
      nullptr,       // SACL
      &j.securityDescriptor);

  if (status != ERROR_SUCCESS) {
    if (ctx.isValid())
      ctx.error(
          "Could not retrieve the Process Security Descriptor, error code {}",
          std::make_tuple(getLastWin32Error()));
    return false;
  }

  if (!IsValidSecurityDescriptor(j.securityDescriptor)) {
    if (ctx.isValid())
      ctx.error("The retrieved security descriptor at {} is not valid",
                std::make_tuple(j.securityDescriptor));
    return false;
  }

  // GENERIC_MAPPING = r?w?x?. Each member is an int, ACCESS_MASK,
  // https://learn.microsoft.com/en-us/windows/win32/secauthz/access-mask
  GENERIC_MAPPING genericMapping = {
      PROCESS_VM_READ | PROCESS_QUERY_INFORMATION,  // GENERIC_READ
      PROCESS_VM_WRITE | PROCESS_VM_OPERATION,      // GENERIC_WRITE
      PROCESS_CREATE_THREAD,                        // GENERIC_EXECUTE
      PROCESS_ALL_ACCESS                            // GENERIC_ALL
  };

  ACCESS_MASK outAccessMask = 0;
  DWORD desiredAccess = PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION;
  BOOL bAccessStatus = false;
  PRIVILEGE_SET privilegeSet;
  DWORD privilegeSetSize = sizeof(PRIVILEGE_SET);
  // https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-mapgenericmask
  MapGenericMask(&desiredAccess, &genericMapping);

  // `AccessCheck` function to see whether the process security descriptor has a
  // predefined set of access rights `AccessCheck` requires a *Client Token*,
  // which is a token associated to some entity (local user or client-server),
  // derived from a primary token But `OpenProcessToken` returns a *Primary
  // Token*, which represents the user account under which the process is
  // running To Impersonate a client token from a primary token, use
  // `ImpersonateSelf` source:
  // https://stackoverflow.com/questions/35027524/whats-the-difference-between-a-primary-token-and-an-impersonation-token
  // basically, AccessCheck works with thread tokens, not process tokens, so we
  // need to fetch the association user - process and map it onto the current
  // thread source: book "Programming Windows Security" source:
  // https://blog.aaronballman.com/2011/08/how-to-check-access-rights/
  j.pair = detail::createImpersonatingThreadToken(ctx, hProcessToken);
  if (j.pair.procImpersonationToken == INVALID_HANDLE_VALUE) return false;

  if (!AccessCheck(
          j.securityDescriptor,  // security descriptor against which access is
                                 // checked
          j.pair.procImpersonationToken,  // impersonation token representing
                                          // the user attempting the access
          desiredAccess,                  // desired access rights
          &genericMapping, &privilegeSet, &privilegeSetSize, &outAccessMask,
          &bAccessStatus)) {
    if (ctx.isValid())
      ctx.error("Failed call to AccessCheck, error: {}",
                std::make_tuple(getLastWin32Error()));
    return false;
  }

  if (!bAccessStatus) {
    if (ctx.isValid())
      ctx.error(
          "The Process doesn't own the"
          " PROCESS_VM_OPERATION access rights, using 2MB large pages. Error: "
          "{}",
          std::make_tuple(getLastWin32Error()));
    return false;
  }

  return true;
}

namespace dmt::os {
void* reserveVirtualAddressSpace(size_t size) {
  // - when `VirtualAlloc` is called with MEM_RESERVE, the allocation is aligned
  // to the `allocation granularity`
  void* address = VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_READWRITE);
  return address;
}

bool commitPhysicalMemory(void* address, size_t size) {
  // - when `VirtualAlloc` it called with MEM_COMMIT, the allocation is aligned
  // to a page boundary
  void* committed = VirtualAlloc(address, size, MEM_COMMIT, PAGE_READWRITE);
  return committed != nullptr;
}

void decommitPhysicalMemory(void* pageAddress, size_t pageSize) {
  VirtualFree(pageAddress, pageSize, MEM_DECOMMIT);
}

bool freeVirtualAddressSpace(void* address, size_t size) {
  return VirtualFree(address, 0, MEM_RELEASE);
}

void* allocateLockedLargePages(size_t size, EPageSize pageSize,
                               bool skipAclCheck) {
  Context ctx;
  if (pageSize == EPageSize::e2MB || pageSize == EPageSize::e1GB) {
    if (!systemSupportsLargePages() || size % toUnderlying(pageSize) != 0)
      return nullptr;

    if (pageSize == EPageSize::e1GB && !checkVirtualAlloc2InKernelbaseDll(ctx))
      return nullptr;

    if (!skipAclCheck) {
      HANDLE hProcessToken = INVALID_HANDLE_VALUE;
      if (hProcessToken = checkForSeMemoryLockPrivilegeAndGetProcessToken(ctx);
          hProcessToken == INVALID_HANDLE_VALUE)
        return nullptr;
      if (pageSize == EPageSize::e1GB &&
          !checkProcessHasVmOperationInDACL(ctx, hProcessToken)) {
        CloseHandle(hProcessToken);
        return nullptr;
      }

      CloseHandle(hProcessToken);
    }

    void* result = nullptr;
    if (pageSize == EPageSize::e1GB) {
      MEM_ADDRESS_REQUIREMENTS requirement{};
      requirement.LowestStartingAddress = NULL;
      requirement.HighestEndingAddress = NULL;
      requirement.Alignment =
          4ULL * 1024 * 1024 * 1024;  // align to 4GB boundary

      MEM_EXTENDED_PARAMETER xp[2]{};
      xp[0].Type = MemExtendedParameterAddressRequirements;
      xp[0].Pointer = &requirement;

      xp[1].Type = MemExtendedParameterAttributeFlags;
      xp[1].ULong64 =
          MEM_EXTENDED_PARAMETER_NONPAGED_HUGE;  // 1GB pages required
      if (result = VirtualAlloc2(nullptr, nullptr, size,
                                 MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                                 PAGE_READWRITE, xp, 2);
          !result) {
        if (ctx.isValid())
          ctx.error(
              "VirtualAlloc2 Failed: {}\t(Incorrect parameter means that "
              "either huge pages are not "
              "supported on this version of windows or there aren't enough "
              "pooled system resources)",
              std::make_tuple(getLastWin32Error()));
        return nullptr;
      }
    } else  // 2MB
    {
      if (result = VirtualAlloc(nullptr, size,
                                MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                                PAGE_READWRITE);
          !result) {
        if (ctx.isValid())
          ctx.error("VirtualAlloc Failed: {}",
                    std::make_tuple(getLastWin32Error()));
        return nullptr;
      }
    }

    return result;
  } else
    return nullptr;
}

void deallocateLockedLargePages(void* address, size_t size,
                                EPageSize pageSize) {
  Context ctx;
  if (!VirtualFree(address, 0, MEM_RELEASE)) {
    if (ctx.isValid())
      ctx.error("VirtualFree Error: {}", std::make_tuple(getLastWin32Error()));
  }
}
}  // namespace dmt::os
