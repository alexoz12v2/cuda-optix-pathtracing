# Streaming Scenes to GPU
## Scene Data Structure
The scene typically consists of various elements such as:
- Geometry (meshes, triangles, etc.)
- Textures (images, procedural textures)
- Lights (light sources)
- Materials (shaders, BRDFs)
- Acceleration structures (BVH, grids)

You need to consider how to partition these elements into manageable chunks that can be streamed from disk and 
loaded into memory incrementally.

## Disk Streaming Strategy
You can stream the scene in a way that allows you to load parts of it as needed, based on the 
path tracing algorithm's progress. The general approach would involve:

- Chunking the scene: Divide the scene into smaller parts 
  (e.g., individual objects, clusters of objects, or regions of the scene).
- Lazy loading: Load scene data only when it's required by the renderer 
  (e.g., when a ray intersects a particular object or region).
- Asynchronous loading: Use asynchronous I/O to load scene data in parallel with rendering to 
  avoid stalling the GPU.
## Asynchronous Loading
Since loading scene data from disk is relatively slow compared to GPU computations, you can use 
asynchronous loading to ensure that the CPU is not blocked while the GPU is rendering.
- Asynchronous I/O: Use multi-threading or CUDA streams to load scene data asynchronously from disk.
This can be done using standard OS file I/O or CUDA-specific asynchronous memory copy functions.
- CUDA Streams: Use CUDA streams to manage asynchronous operations. For example, you can copy data 
to the GPU in one stream while rendering in another.
- Memory Paging: If the scene data is too large to fit in memory, implement a paging system to 
swap data in and out of memory as needed.

## Loading and Caching Scene Data
You can implement a caching system to keep the most recently used scene data in memory. 
For example:
- LRU Cache: Use a least-recently-used (LRU) cache to store recently accessed scene data in system memory. When data is needed, first check the cache; if it's not found, load it from disk.
- Memory Pooling: Use a memory pool to pre-allocate a block of memory for scene data. When an object or region of the scene is needed, load it into the pool and reuse it.

## Loading Acceleration Structures
The BVH (Bounding Volume Hierarchy) or other acceleration structures are critical for 
ray tracing performance. These structures can also be too large to fit in memory, so they need 
to be streamed efficiently:

- Incremental BVH Construction: If you cannot load the entire BVH at once, 
  consider using an incremental approach to build the BVH on the fly. This means 
  loading only the parts of the BVH that are needed for the current rays being traced.
- Streaming BVH: Store the BVH in a format that allows it to be streamed in parts. 
  You can load and process the BVH nodes as they are needed during the path tracing process.
4. Path Tracing with Streaming
   During path tracing, you'll need to determine which parts of the scene are required for each ray:

Ray Intersection: For each ray, determine which objects it intersects. If the object is not yet in memory, load it from disk.
Lazy Loading: Load the geometry, textures, and materials of the object only when needed. This ensures that you're not loading unnecessary data.
Async Loading: While tracing rays, use asynchronous I/O to load additional scene data in parallel with the rendering process. For example, while one ray is being traced, another part of the scene can be loaded.
5. Example Workflow
   Here’s an example of how the process could work:

Initialization:

Load the scene index (e.g., a list of object bounding boxes, textures, and materials).
Set up a cache for the scene data (geometry, materials, textures).
Rendering:

For each ray, check if the object it intersects is already in memory.
If not, initiate an asynchronous disk read to load the required data into memory (system or GPU memory).
Once the data is loaded, continue the ray tracing process.
Post-Processing:

After rendering, if you need to reload data for further processing (e.g., for another frame or for post-processing), repeat the process of streaming the scene.
6. Challenges and Optimizations
   Latency: Disk I/O can introduce latency, which may affect performance. To mitigate this, you can pre-fetch data or use double buffering to load scene data in parallel with rendering.
   Memory Management: Managing memory between system RAM and GPU memory can be tricky. Consider using unified memory or explicit memory management techniques to ensure that the data is available when needed.
   Data Compression: You can compress scene data on disk to reduce I/O time, though this adds additional overhead during decompression.
   Summary
   To stream a large scene from disk in a CUDA OptiX path tracing application, you need to:

Partition the scene into smaller chunks (objects, regions, or BVH nodes).
Lazy load the scene data from disk as needed, using asynchronous I/O and efficient memory management.
Cache frequently accessed data to avoid redundant disk reads.
Use CUDA streams to overlap GPU computation with disk I/O, ensuring that the GPU is not idle while waiting for data.
This approach allows you to handle large scenes that cannot fit entirely in memory, enabling efficient path tracing even with complex and large-scale scenes.

## Linux Huge Pages
Guide: [Link](https://www.baeldung.com/linux/huge-pages-management)
Relevant is the folder `/sys/kernel/mm/hugepages`, containing various text files which give us information about the configuration of the huge page allocation
There are two ways in which we can allocate huge pages in linux (2MB and 1GB)
- Explicit Huge Pages with `mmap`
- Transparent Huge Pages with `madvise`
### Commands in which we find relevant information
- Virtual Memory Parameters from `/proc/vmstat` (Linux >2.6)
  ```sh 
  cat /proc/vmstat | grep -e 'thp' -e 'huge*page'
  ```
  ```
  nr_shmem_hugepages 0
  nr_file_hugepages 0
  nr_anon_transparent_hugepages 12
  thp_migration_success 0
  thp_migration_fail 0
  thp_migration_split 0
  thp_fault_alloc 79
  thp_fault_fallback 0
  thp_fault_fallback_charge 0
  thp_collapse_alloc 6
  thp_collapse_alloc_failed 0
  thp_file_alloc 0
  thp_file_fallback 0
  thp_file_fallback_charge 0
  thp_file_mapped 0
  thp_split_page 10
  thp_split_page_failed 0
  thp_deferred_split_page 13
  thp_split_pmd 28
  thp_split_pud 0
  thp_zero_page_alloc 1
  thp_zero_page_alloc_failed 0
  thp_swpout 0
  thp_swpout_fallback 0
  ```
- Huge Page data utilization with `/proc/meminfo`
  ```sh 
  cat /proc/meminfo | grep --ignore-case -e 'huge' -e 'filepmd' | grep --invert-match 'Shmem'
  ```
  ``` 
  AnonHugePages:     10666 kB
  FileHugePages:         0 kB
  FilePmdMapped:         0 kB
  HugePages_Total:       0
  HugePages_Free:        0
  HugePages_Rsvd:        0
  HugePages_Surp:        0
  Hugepagesize:       2048 kB
  Hugetlb:               0 kB
  ```
- CPU parameters in `/proc/cpuinfo`, in particular `pse`(2M page support) and `pdpe1gb` (1G page support)
  This prints information for each logical core, and we want that `flags` contains 
  (CPU support should be present on most x86_64 CPUs, so this step is not that relevant)
  ```sh 
  cat /proc/cpuinfo | grep flag
  ```
- OS configuration relevant to huge pages with the `sysctl` command
  ```sh 
  sudo sysctl -a | grep 'huge*page'
  ```
  ```
  vm.nr_hugepages = 0
  vm.nr_hugepages_mempolicy = 0
  vm.nr_overcommit_hugepages = 0
  ```
- To check how many huge page related settings are in the pseudo file system `/sys` (pseudo 
  because if you check `cat /proc/mounts` you'll see it is a `sysfs`, hence they are actually parameters/memory stuff)
  ```sh
  sudo find /sys -name '*huge*page*'
  ```
  ``` 
  /sys/kernel/mm/hugepages/hugepages-2048kB
  /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages
  /sys/kernel/mm/hugepages/hugepages-2048kB/resv_hugepages
  /sys/kernel/mm/hugepages/hugepages-2048kB/surplus_hugepages
  /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages_mempolicy
  /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
  /sys/kernel/mm/hugepages/hugepages-2048kB/nr_overcommit_hugepages
  ...
  ```
  Note that there exist settings for each hugepage size, in particular
  ```sh
  sudo tree /sys/kernel/mm/hugepages /sys/kernel/mm/transparent_hugepage
  ```
  ``` 
  /sys/kernel/mm/hugepages
  ├── hugepages-1048576kB
  │   ├── free_hugepages
  │   ├── nr_hugepages
  │   ├── nr_hugepages_mempolicy
  │   ├── nr_overcommit_hugepages
  │   ├── resv_hugepages
  │   └── surplus_hugepages
  └── hugepages-2048kB
       ├── free_hugepages
       ├── nr_hugepages
       ├── nr_hugepages_mempolicy
       ├── nr_overcommit_hugepages
       ├── resv_hugepages
       └── surplus_hugepages
  /sys/kernel/mm/transparent_hugepage
  ├── defrag
  ├── enabled
  ├── hpage_pmd_size
  ├── khugepaged
  │   ├── alloc_sleep_millisecs
  │   ├── defrag
  │   ├── full_scans
  │   ├── max_ptes_none
  │   ├── max_ptes_shared
  │   ├── max_ptes_swap
  │   ├── pages_collapsed
  │   ├── pages_to_scan
  │   └── scan_sleep_millisecs
  ├── shmem_enabled
  └── use_zero_page
  ```
### Standard Huge Pages and Transparent Huge Pages
Notes before starting:
- the kernel needs to be built with the CONFIG_HUGETLBFS and CONFIG_HUGETLB_PAGE options
- standard huge pages (HP) pre-allocate memory at startup
- transparent huge pages (THP) need dynamic memory allocation at runtime via the khugepaged kernel thread
- THP currently only works for anonymous memory mappings and tmpfs or shmem
- best practices in many production environments, such as database servers, suggest disabling THP, but leaving standard HP
### Explicit Huge Pages
[Link to Documentation](https://www.kernel.org/doc/html/v5.1/admin-guide/mm/hugetlbpage.html)

- Check CPU support with the flags `pse` and/or `pdpe1gb` or by using `cat /proc/filesystems | grep hugetlbfs`
- Command to check relevant information from `/proc/meminfo` (with an example output)
  ```shell
  cat /proc/meminfo | grep --ignore-case -e 'huge' -e 'filepmd' | grep --invert-match 'Shmem'
  AnonHugePages:     10666 kB
  FileHugePages:         0 kB
  FilePmdMapped:         0 kB
  HugePages_Total:       0
  HugePages_Free:        0
  HugePages_Rsvd:        0
  HugePages_Surp:        0
  Hugepagesize:       2048 kB
  Hugetlb:               0 kB
  ```
  In particular, we want `HugePages_Total` to be a positive integer
- check if you have huge pages enabled, you can by checking
  - `cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages` for 2MB pages
  - `cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages` for 1GB pages
  - `cat /proc/meminfo | grep Huge` to see the state of explicit huge pages
- for each supported page size you have a folder on `hugepages` folder in which we take
  interest on the files `nr_hugepages` and `nr_overcommit_hugepages`. The system will 
  be allowed to allocated between `nr_hugepages` and `nr_hugepages+nr_overcommit_hugepages`
  frames.
  this configuration can be manipulated before booting the application with an `sudo echo` 

If, with the `/proc/meminfo` output, we see `HugePages_Total` to 0, then we can enable it with either
```
sudo echo 128 > /proc/sys/vm/nr_hugepages
```
or 
```
sudo sysctl -w vm.nr_hugepages=128
```
(using 128 hugepages as an example). The `vm` hugepages are of the size indicated by 
`HugePagessize` in the `/proc/meminfo` output.
After that, check again the `/proc/meminfo` to confirm that you allocated some huge pages
- To free the memory, you need to set the number back to zero.
- To select another huge page size, you need to change your boot configuration, which 
  varies depending on your boot loader and therefore depending on your linux distribution.
  More information on [Link](https://www.intel.com/content/www/us/en/docs/programmable/683840/1-2-1/enabling-hugepages.html)
- To make the huge page allocation persistent over reboots, you need to change the 
  boot configuration

### Transparent Huge Pages
*They Require Standard Huge Pages to be enabled to be used*
Useful information about thp on `cat /proc/vmstat`, in particular
```sh
cat /proc/vmstat | grep -e 'thp' -e 'huge*page'
```
gives us the information about THP.

To check whether transparent huge pages are enabled or not, we need to check the file
`/sys/kernel/mm/transparent_hugepage/enabled` and see that its "[]" marked setting is
either `always` or `madvice`
```sh 
cat /sys/kernel/mm/transparent_hugepage/enabled
```
``` 
[always] madvise never
```
## Windows Huge Pages
- You can obtain the handle to the current process with `GetCurrentProcess`
  ```c
  HANDLE GetCurrentProcess();
  ```
- To use large page allocation with `VirtualAlloc2`, a process must have the `PROCESS_VM_OPERATION` right.
  When a User logs it, windows collects all its right in a structure called *Access Token*, which describes
  what processes spawned by that user are allowed to do.
  From the token associated to a user, you can spawn a process by specifying the token as parameter, and that 
  will determine the struct *Process Security Descriptor* (token can be the default one of the user or a subset 
  of its rights).
  To retrieve a process security descriptor, call the `GetSecurityInfo` function
  ```c
  // ERROR_SUCCESS if no problems
  DWORD GetSecurityInfo(
    [in]            HANDLE               handle,                 // (process handle)
    [in]            SE_OBJECT_TYPE       ObjectType,             // (SE_KERNEL_OBJECT for processes)
    [in]            SECURITY_INFORMATION SecurityInfo,           // (DACL_SECURITY_INFORMATION)
    [out, optional] PSID                 *ppsidOwner,
    [out, optional] PSID                 *ppsidGroup,
    [out, optional] PACL                 *ppDacl,               // can you take the DACL directly from here?
    [out, optional] PACL                 *ppSacl,
    [out, optional] PSECURITY_DESCRIPTOR *ppSecurityDescriptor   // (output we want, has to be freed with LocalFree)
  );
  ```
- Once you retrieved the process security descriptor, you need to get the list of the access rights associated
  to the process security descriptor. If it contains `PSECURITY_DESCRIPTOR`,  we are happy
  ```c
  // success if 0, else call GetLastError
  BOOL GetSecurityDescriptorDacl(
    [in]  PSECURITY_DESCRIPTOR pSecurityDescriptor,            // security descriptor from before
    [out] LPBOOL               lpbDaclPresent,                 // check if dacl is present
    [out] PACL                 *pDacl,                         // then retrieve the list
    [out] LPBOOL               lpbDaclDefaulted                // if false, the dacl was specified by the user
  );
  ```
- iterate over the DACL to find whether or not you have the `PROCESS_VM_OPERATION` *access control entry* (ACE)
  ```c
  for (DWORD i = 0; i < pDACL->AceCount; i++) {
    PACE_HEADER pAce = NULL;
    if (GetAce(pDACL, i, (LPVOID*)&pAce)) {
      if (HasProcessVmOperationPermission(pAce)) {
        LocalFree(pSD);
        return true; // PROCESS_VM_OPERATION permission is granted
      }
    }
  }

  // Function to check if the ACE grants PROCESS_VM_OPERATION permission
  bool HasProcessVmOperationPermission(PACE_HEADER pAce) {
    // The ACCESS_MASK for PROCESS_VM_OPERATION permission is 0x0008
    // Check if the ACE grants the necessary permission
    DWORD dwAccessMask = ((ACCESS_ALLOWED_ACE*)pAce)->Mask;
    return (dwAccessMask & PROCESS_VM_OPERATION) != 0;
  }
  ```

- If you have the `PROCESS_VM_OPERATION`, then you are allowed to use `VirtualAlloc2`, which is the only one that
  can allocate 1GB pages. Otherwise, you are stuck with 2MB Large Pages.

- the `VirtualAlloc2` function is supported from Windows 10. To check its support you need to check whether the 
  Kernel32.dll (specified by [docs.](https://learn.microsoft.com/it-it/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc)) contains the `VirtualAlloc2`, we can use the `GetProcAddress` function
  ```c
  typedef LPVOID(WINAPI* pVirtualAlloc2)(HANDLE, LPVOID, SIZE_T, DWORD, DWORD, SIZE_T, DWORD);
  bool IsVirtualAlloc2Available() {
    // Get the function pointer for VirtualAlloc2
    HMODULE hKernel32 = GetModuleHandle(L"kernel32.dll");
    if (!hKernel32) {
      return false;
    }

    pVirtualAlloc2 pAlloc = (pVirtualAlloc2)GetProcAddress(hKernel32, "VirtualAlloc2");
    return pAlloc != nullptr;
  }
  ```
- Onto the steps to add the priviledge to allocate large pages
  1. Obtain the `"SeLockMemoryPrivilege"` by calling the `AdjustTokenPrivileges` function (large pages are locked to 
     memory by windows, they cannot be swapped).
     The string can be taken by the macro `SE_LOCK_MEMORY_NAME`
     ```c 
     // returns ERROR_SUCCESS if success, otherwise 0, and GetLastError tells you what went wrong
     BOOL AdjustTokenPrivileges(
       [in]            HANDLE            TokenHandle,          // token handle of the current process (see next step)
       [in]            BOOL              DisableAllPrivileges, // false
       [in, optional]  PTOKEN_PRIVILEGES NewState,         // pointer TOKEN_PRIVILEDGES struct = count+VLA (see below)
       [in]            DWORD             BufferLength,         // length of buffer pointed by PreviousState (0 if NULL)
       [out, optional] PTOKEN_PRIVILEGES PreviousState,        // NULL
       [out, optional] PDWORD            ReturnLength          // returns length of Previous State (<= BufferLength>)
     );

    // newState = list of { privilege, enableOrDisable } we want to add(SE_PRIVIEGE_ENABLED)/remove(SE_PRIVIEGE_REMOVED)
    // reference to TOKEN_PRIVILEGES
    typedef struct _TOKEN_PRIVILEGES {
      DWORD               PrivilegeCount;
      LUID_AND_ATTRIBUTES Privileges[ANYSIZE_ARRAY];
    } TOKEN_PRIVILEGES, *PTOKEN_PRIVILEGES;

    // each LUID_AND_ATTRIBUTE is a pair LUID, Attributes, in which the attributes meaning depends on LUID
    typedef struct _LUID_AND_ATTRIBUTES {
      LUID  Luid;
      DWORD Attributes;
    } LUID_AND_ATTRIBUTES, *PLUID_AND_ATTRIBUTES;

    // LUID = locally unique identifier = 64 pit number, split into 2 32 bit parts (predates win 64 bit)
    typedef struct _LUID {
      DWORD LowPart;
      LONG  HighPart;
    } LUID, *PLUID;
    // the LUID is used to represent privileges in the PRIVILEGE_SET structure
     ```
- You can check whether an *access token* has a certain `PRIVILEGE_SET` by using the `PrivilegeCheck` functiom
  ```c
  BOOL PrivilegeCheck(
    [in]      HANDLE         ClientToken, 
    [in, out] PPRIVILEGE_SET RequiredPrivileges,
    [out]     LPBOOL         pfResult
  );

  typedef struct _PRIVILEGE_SET {
    DWORD               PrivilegeCount; // count
    DWORD               Control;       // PRIVILEGE_SET_ALL_NECESSARY
    LUID_AND_ATTRIBUTES Privilege[ANYSIZE_ARRAY];
  } PRIVILEGE_SET, *PPRIVILEGE_SET;
  ```
  Example of token retrieval
  ```c
  #include <windows.h>
  #include <iostream>

  int main() {
      HANDLE hToken = NULL;

      // Get a handle to the current process
      HANDLE hProcess = GetCurrentProcess();

      // Open the access token associated with the current process
      // second parameter is the access rights, ie what you want to be able to do with the token
      // TOKEN_QUERY or TOKEN_ADJUST_PRIVILEGES
      if (!OpenProcessToken(hProcess, TOKEN_QUERY, &hToken)) { 
          std::cerr << "Failed to open process token. Error: " << GetLastError() << std::endl;
          return 1;
      }

      std::cout << "Successfully retrieved the access token." << std::endl;

      // Use the token (e.g., query information)...

      // Close the token handle when done
      CloseHandle(hToken);

      return 0;
  }
  ```
- Therefore, you 
  - get the *token handle* associated to the *current process*, 
  - open such token in `TOKEN_ADJUST_PRIVILEDGES` DesiredAccessMode,
  - check whether it already contains a `PRIVILEGE_SET` containing `SeLockMemoryPrivilege`, by using
    the function `GetTokenInformation` with the enum value `TokenPrivileges` (enum name = `TOKEN_INFORMATION_CLASS`).
    - with `TokenPrivileges`, the output untyped buffer is actually a pointer to `TOKEN_PRIVILEGES` struct
    - You actually need to call the function twice, the first time to get the required size of the outuput buffer, 
      the second time to actually do something (Vulkan Style)
      ```c
      #include <windows.h>
      #include <iostream>

      void PrintPrivilegeAttributes(DWORD attributes) {
        if (attributes & SE_PRIVILEGE_ENABLED) {
          std::cout << "Privilege is enabled." << std::endl;
        } else {
          std::cout << "Privilege is disabled." << std::endl;
        }

        if (attributes & SE_PRIVILEGE_ENABLED_BY_DEFAULT) {
          std::cout << "Privilege is enabled by default." << std::endl;
        }

        if (attributes & SE_PRIVILEGE_REMOVED) {
          std::cout << "Privilege has been removed." << std::endl;
        }
      }

      int main() {
        HANDLE hToken = NULL;

        // Open the current process token
        if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &hToken)) {
          std::cerr << "Failed to open process token. Error: " << GetLastError() << std::endl;
          return 1;
        }

        // Get the size of the buffer needed for privileges
        DWORD dwSize = 0;
        GetTokenInformation(hToken, TokenPrivileges, NULL, 0, &dwSize);
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
          std::cerr << "Failed to get token privileges size. Error: " << GetLastError() << std::endl;
          CloseHandle(hToken);
          return 1;
        }

        // Allocate buffer for privileges
        TOKEN_PRIVILEGES* pTokenPrivileges = (TOKEN_PRIVILEGES*)malloc(dwSize);
        if (!pTokenPrivileges) {
          std::cerr << "Memory allocation failed." << std::endl;
          CloseHandle(hToken);
          return 1;
        }

        // Retrieve the privileges
        if (!GetTokenInformation(hToken, TokenPrivileges, pTokenPrivileges, dwSize, &dwSize)) {
          std::cerr << "Failed to get token privileges. Error: " << GetLastError() << std::endl;
          free(pTokenPrivileges);
          CloseHandle(hToken);
          return 1;
        }

        // Iterate through the privileges
        for (DWORD i = 0; i < pTokenPrivileges->PrivilegeCount; ++i) {
          LUID luid = pTokenPrivileges->Privileges[i].Luid;
          DWORD attributes = pTokenPrivileges->Privileges[i].Attributes;

          // Lookup the privilege name
          char name[256];
          DWORD nameLen = sizeof(name);
          if (LookupPrivilegeNameA(NULL, &luid, name, &nameLen)) {
              std::cout << "Privilege: " << name << std::endl;
          } else {
              std::cerr << "Failed to lookup privilege name. Error: " << GetLastError() << std::endl;
          }

          // Print the privilege attributes
          PrintPrivilegeAttributes(attributes);
        }

        // Clean up
        free(pTokenPrivileges);
        CloseHandle(hToken);

        return 0;
      }
      ```
    - `LookupPriviegeValue(nullptr/*on local system*/, privName, &outLUID)` to get the privilege LUID
    - from the LUID, you can construct a `TOKEN_PRIVILEGES` struct 
      ```c
      TOKEN_PRIVILEGES tp;
      tp.PrivilegeCount = 1;
      tp.Privileges[0].Luid = luid;
      tp.Privileges[0].Attributes = SE_PRIVIEGE_ENABLED; // 0 if you wnat to remove the privilege
      ```
  - if not, retrieve (`PreviousState`) the current `PRIVILEGE_SET` of the token

### Retrieving the SID (Security IDentifier) from a token
```c
#include <windows.h>
#include <iostream>

int main() {
    HANDLE hToken = NULL;
    DWORD dwSize = 0;

    // Get a handle to the current process
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &hToken)) {
        std::cerr << "Failed to open process token. Error: " << GetLastError() << std::endl;
        return 1;
    }

    // Get the size of the buffer required for the token information
    GetTokenInformation(hToken, TokenUser, NULL, 0, &dwSize);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        std::cerr << "Failed to get token information size. Error: " << GetLastError() << std::endl;
        CloseHandle(hToken);
        return 1;
    }

    // Allocate buffer for token information
    TOKEN_USER* pTokenUser = (TOKEN_USER*)malloc(dwSize);
    if (!pTokenUser) {
        std::cerr << "Memory allocation failed." << std::endl;
        CloseHandle(hToken);
        return 1;
    }

    // Retrieve the token information
    if (!GetTokenInformation(hToken, TokenUser, pTokenUser, dwSize, &dwSize)) {
        std::cerr << "Failed to get token information. Error: " << GetLastError() << std::endl;
        free(pTokenUser);
        CloseHandle(hToken);
        return 1;
    }

    // Convert the SID to a string
    char* sidString = NULL;
    if (!ConvertSidToStringSidA(pTokenUser->User.Sid, &sidString)) {
        std::cerr << "Failed to convert SID to string. Error: " << GetLastError() << std::endl;
    } else {
        std::cout << "User SID: " << sidString << std::endl;
        LocalFree(sidString);
    }

    // Clean up
    free(pTokenUser);
    CloseHandle(hToken);

    return 0;
}
```

### Windows Exaples
[Link](https://learn.microsoft.com/it-it/windows/win32/Memory/reserving-and-committing-memory)

### Windows Terms
- access control entry
  (ACE) An entry in an access control list (ACL). An ACE contains a set of access rights and a security identifier 
  (SID) that identifies a trustee for whom the rights are allowed, denied, or audited.

- access control list
  (ACL) A list of security protections that applies to an object. (An object can be a file, process, event, or 
  anything else having a security descriptor.) An entry in an access control list (ACL) is an access control 
  entry (ACE). There are two types of access control list, discretionary and system.

### Windows Considerations on Large Pages
- Large-page memory regions may be difficult to obtain after the system has been running for a long time because 
  the physical space for each large page must be contiguous, but the memory may have become fragmented. 
  Allocating large pages under these conditions can significantly affect system performance. Therefore, 
  applications should avoid making repeated large-page allocations and instead *allocate all large pages one time*, 
  at startup.
- The memory is always read/write and nonpageable (*always resident in physical memory*).
- The memory is part of the process private bytes but not part of the working set, because the working set 
  by definition contains only pageable memory.
- Large-page allocations are not subject to job limits.
- Large-page memory must be reserved and committed as a single operation. In other words, 
  *large pages cannot be used to commit a previously reserved range of memory*.

### Windows Tracing Page Allocations
The memory tracing utilities in windows are explained at 
[Process Status API Link](https://learn.microsoft.com/en-us/windows/win32/psapi/collecting-memory-usage-information-for-a-process).

Let us examine the `VirtualQuery` function, [Link](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualquery)
```c
// zero if fails
SIZE_T VirtualQuery(
  [in, optional] LPCVOID                   lpAddress, // pointer to the beginning of a page (or of a range of pages)
  [out]          PMEMORY_BASIC_INFORMATION lpBuffer,  // pointer to MEMORY_BASIC_INFORMATION struct
  [in]           SIZE_T                    dwLength   // size of the `lpBuffer` (numPages * sizeof(MEMORY_BASIC_INFORMATION))
);
```

### Windows Memory Terms
- *Committed Memory* = memory allocated which is currently backed by physical RAM memory (a frame).
                       Memory can First be *Reserved* (virtual address space reservation) and then *committed*
- *Resident Memory* = Refers to non locked pages (hence not large) currently in physical memory

### If the `SeLockMemoryPrivileges` isn't found on the token, even as Administrator
- Check Group Policy Settings
  - `WIN+R` and write "`secpol.msc`"
  - Navigate to `Security Settings` -> `Local Policies` -> `User Rights Assigmnent`
    (italian `Impostazioni Sicurezza` -> `Criteri Locali` -> `Asegnazione diritti utente`) 
  - Locate `Lock pages in memory` (italian `Blocco di pagine in memoria`)
  - Add the group `Administrators` and your own user to the list of allowed users
- Open "edit group policy"
  - Navigate to `Conpuuter Configuration` -> `Windows Settings` 
    -> `Security Settings` -> `Local Policies` -> `User Rights Assignment` 
    and check that you find the edits you did in `secpol.msc`
- `WIN+R` and write "`regedit`"
  - Navigate to 
    ```
    HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\MemoryManagement
    ```
    to check for more memory management parameters (this is the equivalent of the `/sys/kernel/mm` and `/proc/meminfo`)

### checking support for VIrtualAlloc2 (basically if you are on windows or not)
Either
  - check windows version and make sure it's the correct one
  - check that the `C:\Windows\System32\kernel.dll` (`extern C`) exports that function
    first you can check yourself if you have that function by running on cmd
    ```cmd
    dumpbin /EXPORTS C:\Windows\System32\kernel32.dll | findstr VirtualAlloc2
    dumpbin /EXPORTS C:\Windows\System32\kernelbase.dll | findstr VirtualAlloc2
    ```
  - When you first try to use `VirtualAlloc2`, most likely it will fail with error
    ```red
    insufficient system resources exist to complete the requested service
    ```
    If, from the `Task Manager` you verify that you indeed have sufficient free memory to complete the allocation, then
    this has to do with the Memory Management configuration in the registry. Therefore, open
    ```
    HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\MemoryManagement
    ```
    We are interested in the `PagedPoolSize` (which should be there) and `PoolUsageMaximum` (which is not 
    there by default.).
    ([Reference link](https://support.arcserve.com/s/article/202808835?language=en_US)). We need to fiddle with 
    these values (and then reboot) to force windows to allocate large pages.
    - To make sure the computer doesn't explode, it is advised to enable Swapping and increase the swapfile size
      [Link](https://danigta.de/tutorial/operating/system/windows/setup-swap-file)
    - setting `PagedPoolSize` to -1 will make windows allocate whatever it can to memory
    - The safest option is to just set `PoolUsageMaximum` (`DWORD`) to some percentage, like 60, as [Docs reccomend](https://learn.microsoft.com/it-it/troubleshoot/windows-server/performance/unable-allocate-memory-system-paged-pool)
