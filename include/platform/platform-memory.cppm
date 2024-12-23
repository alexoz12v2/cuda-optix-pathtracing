/**
 * @file platform-memory.cppm
 * @brief partition inteface unit for the platform module implementing basic local memory allocation
 * strategies plus the necessary infrastructure for tracing memory allocations and keeping all the
 * necessary information to plot memory utilization (eg. implot).
 * We can identify 3 types of memory allocations which are needed by the project
 * 1. Scene data, persistent throughout the whole application's execution
 * 2. Queues, Buffers, anything our systems/classes need to function correctly. Their lifetime
 *    spans multiple render loop iterations, but they can be deallocated and allocated dynamically
 *    during the program's execution
 * 3. Transient data, like queues sent to the "CUDA kernel" execution. Such memory is needed only
 *    during the current render loop iteration, and can be wiped on the next one (without calling
 *    any destructor. Transient data is supposed to be trivially destructible)
 * 4. Debug memory allocations, ie allocations which are present in debug builds, but should
 *    be stripped out of the build if we are not in a debug build
 * For each of these memory allocation types, we can adapt 3 different allocation strategies
 * 3. Transient data can be mamaged by a stack allocator strategy, meaning that a class manaages
 *    a collection of stacks, keeping a pointer for each of them, and whenever an allocation is
 *    requested with a given (size, alignment), you just move the pointer. If there's no space,
 *    use the next stack.
 *    All the additional stack allocated during execution should be freed upon N render cycles of
 *    inactivity
 * 2. System Required data should be allocated with a pool allocator fashion, as they tend to
 *    be similiarly sized and such strategy supports deallocation as well, differently from the stack
 *    allocator
 * 1. Scene data can be thought os an acyclic graph (instancing, otherwise tree), in which theres
 *    some *structured metadata* (type of object, material description) which points to *unstructured
 *    data* (vertex buffer, textures).
 *    This data can use a chunked allocation strategy, which is similiar to the pool allocation
 *    strategy in the sense that a given buffer is broken up into chunks, but such chunks are linked
 *    together in an embedded linked list fashion. For performance, chunks should be as uniform as
 *    possible, eg all metadata for a 3d mesh should fit into a single chunk, vertex data chunks and
 *    texture data chunks should be near each other as much as possible
 * 4. we don't care too much about performance for this one, as it will probably contain only the
 *    ASCII string table anyways. this means that is can use a pool allocator from the std library
 *    in particular the `std::pmr::synchronized_pool_resource`
 * About Thread Safety
 * - Debug memory is synchronized by the standard library
 * - There will ve 1 Transient data allocator for each thread, as for random number generation
 *   so it can be unsynchronized
 * - since level data will probably use multithreaded, asynchronous IO from files, it should be
 *   synchronized properly. Each thread will open one part of the file
 *   NOTE: this is not definitive as I'll try to reuse as much as possible from pbrt's code, since
 *   our scenes will follow the pbrt-v4 file format
 *   Beyond its first allocation the data will be used read-only, hence during the application
 *   execution there's no need for synchronization
 * - system data allocator (pool allocator) should be fully synchronized
 * OS Details regarding paging
 * - In order to increase performance, the critical data we require in our application should be
 *   allocated on a large page size, in order to minimize the probability of TLB miss and
 *   should be pinned to memory, if possible. Summary of such details
 *   <table>
 *   <caption id="multi_row">Page Size support</caption>
 *   <tr><td>Feature</td>            <td>Linux Huge Page</td><td>Windows Large Page</td></tr>
 *   <tr><td>Size</td>               <td>2 MB/1GB</td>       <td>2 MB, 1 GB(<- Win10 onwards)</td></tr>
 *   <tr><td>Pinned</td>             <td>Yes</td>            <td>Yes</td></tr>
 *   <tr><td>Privilege Required</td> <td>No</td>             <td>Yes</td></tr>
 *   </table>
 *   Resources on the matter:
 *   <ul>
 *   <li><a href="https://stackoverflow.com/questions/35566176/windows-large-page-support-other-than-2mb#:~:text=The%20large%20page%20size%20is,System%20Architecture%20(volume%203A).">Are 1GB supported on Windows</a></li>
 *   <li><a href="https://mazzo.li/posts/check-huge-page.html">linux guide</a></li>
 *   <li><a href="https://www.lukas-barth.net/blog/linux-allocating-huge-pages/">another linux guide</a></li>
 *   <li><a href="https://learn.microsoft.com/en-us/windows/win32/memory/large-page-support">Windows Documentation</a></li>
 *   </ul>
 * What if scene files do not fit into memory all at once?
 * - There should be a limit on how much memory you can add to the resident set from the chunked scene allocator.
 *   Such limit should be set considering both the amount of device memory present in the system, and
 *   on the CUDA capable device (if any)
 * - If, during the loading process, such limit is crossed, then the scene cannot be fully loaded
 *   on disk.
 * - Therefore, during the parsing Process, we can
 *   1. Load to memory chunk 1 the global scene configuration (camera, film, filter, integrator, color space)
 *   2. Build a spatial-aware, binary format, file from the `.pbrt`  World Block (section in which the geometry
 *      of the scene is described)
 *      Such a custom made file format should be a binary image of the BVH built from the text file
 *      This step can be multithreaded on the Object blocks (maybe): World Block = list of Object Blocks
 *   3. Build an in-memory index of the generated binary file. this shoulc be a tree of file offsets,
 *      relative to the parent node (Tree impl: 1. Van embde boas tree 2. Judy arrays)
 *   4. Load up to the memory limit chunks of the acceleration structure, level-wise, such that
 *      the ray-scene intersection algorithm shouldn't block immediately
 *   5. when you hit a bvh node, load its content, if it is not already available. If the bvh node
 *      is a leaf (geometry), try to load all at once both vertex buffer and textures associated
 *      with the material. IDEA = prefetching: try to predict (random) and load additional bvh children nodes
 *  From this, we understand that the chunk allocator acts as a cache, whose eviction should maintain
 *  referential integrity: to understand this, consider a material
 *   - if the system recognizes the material as "loaded to memory", then all its dependencies, namely,
 *     textures, are also loaded to memory
 *  = should the custom binary include the unstructured data too?
 *  = the chunk allocator memory should be mapped onto global memory permanently
 *  = unless the scene is huge, the BVH should probably fit all in one chunk
 * - links:
 *   - https://forums.developer.nvidia.com/t/move-data-to-gpu-from-hard-drive/14602/5
 *   - the file in the `docs` directory
 * Tracing Allocations
 * - whenever we allocate something, we supply to the allocator (size, alignment, string id, tag)
 * - if we are in debug build, then the string id also has a corresponding string representation
 *   in the globl string table (shared reads, exclusive writes), in debug memory, which can be used
 *   to give a textual
 *   representation to the allocation, which can be conditionally shown to screen
 * - We would like to
 *   - keep a sliding window of the count of bytes allocated in each render cycle
 *   - each data point for a render cycle corresponds to the mean value of the total allocated memory
 *     during the render cycle execution (or the max)
 *   - for each data point, we should be able to inspect a render cycle (a "frame") to see, for
 *     each allocation, the `source_location` from which it comes from, and the allocations should
 *     also be timestamped, with the high_resolution_clock or by using the x86_64 instruction
 *     rdtsc.
 * - hence, for each allocation we keep track of
 *   - address, page frame number
 *   - size, alignment
 *   - string id, which in debug gives us a textual tag for the allocation
 *   - allocation timestamp and deallocation timestamp, in milliseconds from the beginning of the app
 *   - source location from which the allocation was requested
 * - we will therefore have an array, as big as we want the sliding window to be. Each element contains
 *   - running mean (max) of the allocations for the current render cycle
 *   - hash table associating string ids to an allocation tracking data block
 *     such hash table should live inside an instance of a stack allocator. This means that
 *     our allocator classes should follow the scoped, composable model specified by C++17
 *     (not necessarely adhering to the std::memory_resource interface)
 * - all of this Should be included also in release builds, hence cannot tracking information cannot
 *   reside in debug memory,
 * - this tracking system should be configurable from the command line
 * Notes
 *   - we can use the tag to store an allocation differently, example
 *     - we can use the stack allocator from both ends if we are clustering tags into 2 categories
 *   - if we want to synchronize allocations, then the allocator should wrap a mutex and define
 *     move semantics
 *
 * @defgroup platform platform Module
 * @{
 */
module;

export module platform:memory;

export import :utils;

export import <platform-memory.h>

/** @} */
