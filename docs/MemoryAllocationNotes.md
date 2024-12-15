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