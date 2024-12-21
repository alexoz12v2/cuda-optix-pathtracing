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

#include <array>
#include <string_view>

#include <cassert>
#include <compare>
#include <cstdint>

export module platform:memory;

import :utils;

export namespace dmt
{

enum class EMemoryTag : uint16_t
{
    Unknown,               /** temporary tag when you are unsure of what to tag */
    Debug,                 /** tag for anything which goes into debug memory */
    Engine,                /** generic tag for whatever buffer is allocated by a system of the application CPU side */
    HashTable,             /** generic tag associated with an hash table */
    Buffer,                /** generic tag associated with a dynamically allocated array */
    Blob,                  /** generic tag associated with I/O operations */
    Job,                   /** generic tag associated with job data */
    Queue,                 /** generic tag assiciated with Queues to be read by the GPU */
    Scene,                 /** generic tag for whatever data is associated to the scene. Prefer specific ones */
    AccelerationStructure, /** tag associated to acceleration structures such as BVH nodes */
    Geometry,              /** tag associated with triangular meshes and other types of geo */
    Material,              /** tag associated with material information */
    Textures,              /** tag associated with image textures */
    Spectrum,              /** tag associated for a sampled light spectrum */
    Count,
};

using sid_t = uint64_t;

} // namespace dmt

// not exported
namespace dmt
{

// Primary template (matches nothing by default)
template <template <typename...> class Template, typename T>
struct is_template_instantiation : std::false_type
{
};

// Specialization for template instantiations
template <template <typename...> class Template, typename... Args>
struct is_template_instantiation<Template, Template<Args...>> : std::true_type
{
};

// Helper variable template
template <template <typename...> class Template, typename T>
inline constexpr bool is_template_instantiation_v = is_template_instantiation<Template, T>::value;

// Concept using the helper
template <template <typename...> class Template, typename T>
concept TemplateInstantiationOf = is_template_instantiation_v<Template, T>;

void* alignTo(void* address, size_t alignment)
{
    // Ensure alignment is a power of two (required for bitwise operations).
    size_t const mask = alignment - 1;
    assert((alignment & mask) == 0 && "Alignment must be a power of two.");

    uintptr_t addr = reinterpret_cast<uintptr_t>(address);
    uintptr_t alignedAddr = (addr + mask) & ~mask;

    return reinterpret_cast<void*>(alignedAddr);
}

void* alignToBackward(void* address, size_t alignment)
{
    // Ensure alignment is a power of two (required for bitwise operations).
    size_t const mask = alignment - 1;
    assert((alignment & mask) == 0 && "Alignment must be a power of two.");

    uintptr_t addr        = reinterpret_cast<uintptr_t>(address);
    uintptr_t alignedAddr = addr & ~mask;

    return reinterpret_cast<void*>(alignedAddr);
}

/**
 * Compile time lookup table used to perform the CRC algorithm on a string_view to get an id
 */
inline constexpr std::array<uint64_t, 256> crc64Table{
    0x0000000000000000uLL, 0x42f0e1eba9ea3693uLL, 0x85e1c3d753d46d26uLL, 0xc711223cfa3e5bb5uLL, 0x493366450e42ecdfuLL,
    0x0bc387aea7a8da4cuLL, 0xccd2a5925d9681f9uLL, 0x8e224479f47cb76auLL, 0x9266cc8a1c85d9beuLL, 0xd0962d61b56fef2duLL,
    0x17870f5d4f51b498uLL, 0x5577eeb6e6bb820buLL, 0xdb55aacf12c73561uLL, 0x99a54b24bb2d03f2uLL, 0x5eb4691841135847uLL,
    0x1c4488f3e8f96ed4uLL, 0x663d78ff90e185efuLL, 0x24cd9914390bb37cuLL, 0xe3dcbb28c335e8c9uLL, 0xa12c5ac36adfde5auLL,
    0x2f0e1eba9ea36930uLL, 0x6dfeff5137495fa3uLL, 0xaaefdd6dcd770416uLL, 0xe81f3c86649d3285uLL, 0xf45bb4758c645c51uLL,
    0xb6ab559e258e6ac2uLL, 0x71ba77a2dfb03177uLL, 0x334a9649765a07e4uLL, 0xbd68d2308226b08euLL, 0xff9833db2bcc861duLL,
    0x388911e7d1f2dda8uLL, 0x7a79f00c7818eb3buLL, 0xcc7af1ff21c30bdeuLL, 0x8e8a101488293d4duLL, 0x499b3228721766f8uLL,
    0x0b6bd3c3dbfd506buLL, 0x854997ba2f81e701uLL, 0xc7b97651866bd192uLL, 0x00a8546d7c558a27uLL, 0x4258b586d5bfbcb4uLL,
    0x5e1c3d753d46d260uLL, 0x1cecdc9e94ace4f3uLL, 0xdbfdfea26e92bf46uLL, 0x990d1f49c77889d5uLL, 0x172f5b3033043ebfuLL,
    0x55dfbadb9aee082cuLL, 0x92ce98e760d05399uLL, 0xd03e790cc93a650auLL, 0xaa478900b1228e31uLL, 0xe8b768eb18c8b8a2uLL,
    0x2fa64ad7e2f6e317uLL, 0x6d56ab3c4b1cd584uLL, 0xe374ef45bf6062eeuLL, 0xa1840eae168a547duLL, 0x66952c92ecb40fc8uLL,
    0x2465cd79455e395buLL, 0x3821458aada7578fuLL, 0x7ad1a461044d611cuLL, 0xbdc0865dfe733aa9uLL, 0xff3067b657990c3auLL,
    0x711223cfa3e5bb50uLL, 0x33e2c2240a0f8dc3uLL, 0xf4f3e018f031d676uLL, 0xb60301f359dbe0e5uLL, 0xda050215ea6c212fuLL,
    0x98f5e3fe438617bcuLL, 0x5fe4c1c2b9b84c09uLL, 0x1d14202910527a9auLL, 0x93366450e42ecdf0uLL, 0xd1c685bb4dc4fb63uLL,
    0x16d7a787b7faa0d6uLL, 0x5427466c1e109645uLL, 0x4863ce9ff6e9f891uLL, 0x0a932f745f03ce02uLL, 0xcd820d48a53d95b7uLL,
    0x8f72eca30cd7a324uLL, 0x0150a8daf8ab144euLL, 0x43a04931514122dduLL, 0x84b16b0dab7f7968uLL, 0xc6418ae602954ffbuLL,
    0xbc387aea7a8da4c0uLL, 0xfec89b01d3679253uLL, 0x39d9b93d2959c9e6uLL, 0x7b2958d680b3ff75uLL, 0xf50b1caf74cf481fuLL,
    0xb7fbfd44dd257e8cuLL, 0x70eadf78271b2539uLL, 0x321a3e938ef113aauLL, 0x2e5eb66066087d7euLL, 0x6cae578bcfe24beduLL,
    0xabbf75b735dc1058uLL, 0xe94f945c9c3626cbuLL, 0x676dd025684a91a1uLL, 0x259d31cec1a0a732uLL, 0xe28c13f23b9efc87uLL,
    0xa07cf2199274ca14uLL, 0x167ff3eacbaf2af1uLL, 0x548f120162451c62uLL, 0x939e303d987b47d7uLL, 0xd16ed1d631917144uLL,
    0x5f4c95afc5edc62euLL, 0x1dbc74446c07f0bduLL, 0xdaad56789639ab08uLL, 0x985db7933fd39d9buLL, 0x84193f60d72af34fuLL,
    0xc6e9de8b7ec0c5dcuLL, 0x01f8fcb784fe9e69uLL, 0x43081d5c2d14a8fauLL, 0xcd2a5925d9681f90uLL, 0x8fdab8ce70822903uLL,
    0x48cb9af28abc72b6uLL, 0x0a3b7b1923564425uLL, 0x70428b155b4eaf1euLL, 0x32b26afef2a4998duLL, 0xf5a348c2089ac238uLL,
    0xb753a929a170f4abuLL, 0x3971ed50550c43c1uLL, 0x7b810cbbfce67552uLL, 0xbc902e8706d82ee7uLL, 0xfe60cf6caf321874uLL,
    0xe224479f47cb76a0uLL, 0xa0d4a674ee214033uLL, 0x67c58448141f1b86uLL, 0x253565a3bdf52d15uLL, 0xab1721da49899a7fuLL,
    0xe9e7c031e063acecuLL, 0x2ef6e20d1a5df759uLL, 0x6c0603e6b3b7c1cauLL, 0xf6fae5c07d3274cduLL, 0xb40a042bd4d8425euLL,
    0x731b26172ee619ebuLL, 0x31ebc7fc870c2f78uLL, 0xbfc9838573709812uLL, 0xfd39626eda9aae81uLL, 0x3a28405220a4f534uLL,
    0x78d8a1b9894ec3a7uLL, 0x649c294a61b7ad73uLL, 0x266cc8a1c85d9be0uLL, 0xe17dea9d3263c055uLL, 0xa38d0b769b89f6c6uLL,
    0x2daf4f0f6ff541acuLL, 0x6f5faee4c61f773fuLL, 0xa84e8cd83c212c8auLL, 0xeabe6d3395cb1a19uLL, 0x90c79d3fedd3f122uLL,
    0xd2377cd44439c7b1uLL, 0x15265ee8be079c04uLL, 0x57d6bf0317edaa97uLL, 0xd9f4fb7ae3911dfduLL, 0x9b041a914a7b2b6euLL,
    0x5c1538adb04570dbuLL, 0x1ee5d94619af4648uLL, 0x02a151b5f156289cuLL, 0x4051b05e58bc1e0fuLL, 0x87409262a28245bauLL,
    0xc5b073890b687329uLL, 0x4b9237f0ff14c443uLL, 0x0962d61b56fef2d0uLL, 0xce73f427acc0a965uLL, 0x8c8315cc052a9ff6uLL,
    0x3a80143f5cf17f13uLL, 0x7870f5d4f51b4980uLL, 0xbf61d7e80f251235uLL, 0xfd913603a6cf24a6uLL, 0x73b3727a52b393ccuLL,
    0x31439391fb59a55fuLL, 0xf652b1ad0167feeauLL, 0xb4a25046a88dc879uLL, 0xa8e6d8b54074a6aduLL, 0xea16395ee99e903euLL,
    0x2d071b6213a0cb8buLL, 0x6ff7fa89ba4afd18uLL, 0xe1d5bef04e364a72uLL, 0xa3255f1be7dc7ce1uLL, 0x64347d271de22754uLL,
    0x26c49cccb40811c7uLL, 0x5cbd6cc0cc10fafcuLL, 0x1e4d8d2b65facc6fuLL, 0xd95caf179fc497dauLL, 0x9bac4efc362ea149uLL,
    0x158e0a85c2521623uLL, 0x577eeb6e6bb820b0uLL, 0x906fc95291867b05uLL, 0xd29f28b9386c4d96uLL, 0xcedba04ad0952342uLL,
    0x8c2b41a1797f15d1uLL, 0x4b3a639d83414e64uLL, 0x09ca82762aab78f7uLL, 0x87e8c60fded7cf9duLL, 0xc51827e4773df90euLL,
    0x020905d88d03a2bbuLL, 0x40f9e43324e99428uLL, 0x2cffe7d5975e55e2uLL, 0x6e0f063e3eb46371uLL, 0xa91e2402c48a38c4uLL,
    0xebeec5e96d600e57uLL, 0x65cc8190991cb93duLL, 0x273c607b30f68faeuLL, 0xe02d4247cac8d41buLL, 0xa2dda3ac6322e288uLL,
    0xbe992b5f8bdb8c5cuLL, 0xfc69cab42231bacfuLL, 0x3b78e888d80fe17auLL, 0x7988096371e5d7e9uLL, 0xf7aa4d1a85996083uLL,
    0xb55aacf12c735610uLL, 0x724b8ecdd64d0da5uLL, 0x30bb6f267fa73b36uLL, 0x4ac29f2a07bfd00duLL, 0x08327ec1ae55e69euLL,
    0xcf235cfd546bbd2buLL, 0x8dd3bd16fd818bb8uLL, 0x03f1f96f09fd3cd2uLL, 0x41011884a0170a41uLL, 0x86103ab85a2951f4uLL,
    0xc4e0db53f3c36767uLL, 0xd8a453a01b3a09b3uLL, 0x9a54b24bb2d03f20uLL, 0x5d45907748ee6495uLL, 0x1fb5719ce1045206uLL,
    0x919735e51578e56cuLL, 0xd367d40ebc92d3ffuLL, 0x1476f63246ac884auLL, 0x568617d9ef46bed9uLL, 0xe085162ab69d5e3cuLL,
    0xa275f7c11f7768afuLL, 0x6564d5fde549331auLL, 0x279434164ca30589uLL, 0xa9b6706fb8dfb2e3uLL, 0xeb46918411358470uLL,
    0x2c57b3b8eb0bdfc5uLL, 0x6ea7525342e1e956uLL, 0x72e3daa0aa188782uLL, 0x30133b4b03f2b111uLL, 0xf7021977f9cceaa4uLL,
    0xb5f2f89c5026dc37uLL, 0x3bd0bce5a45a6b5duLL, 0x79205d0e0db05dceuLL, 0xbe317f32f78e067buLL, 0xfcc19ed95e6430e8uLL,
    0x86b86ed5267cdbd3uLL, 0xc4488f3e8f96ed40uLL, 0x0359ad0275a8b6f5uLL, 0x41a94ce9dc428066uLL, 0xcf8b0890283e370cuLL,
    0x8d7be97b81d4019fuLL, 0x4a6acb477bea5a2auLL, 0x089a2aacd2006cb9uLL, 0x14dea25f3af9026duLL, 0x562e43b4931334feuLL,
    0x913f6188692d6f4buLL, 0xd3cf8063c0c759d8uLL, 0x5dedc41a34bbeeb2uLL, 0x1f1d25f19d51d821uLL, 0xd80c07cd676f8394uLL,
    0x9afce626ce85b507uLL};
/**
 * the CRC algorithm is an iterative one and needs a starting value.
 */
inline constexpr uint64_t initialCrc64 = crc64Table[107];

constexpr uint64_t hashCRC64(char const* str)
{
    uint64_t crc = initialCrc64;
    while (*str)
    {
        crc = crc64Table[static_cast<uint8_t>((crc >> 56) ^ static_cast<uint64_t>(*str++))] ^ (crc << 8);
    }
    return crc;
}

} // namespace dmt

export namespace dmt
{

/**
 * Operator to convert an ASCII string literal to its string id form with CRC64 algorithm
 * to be preferred when you need constant time evaluation, eg in switch cases
 * @param str '\0' terminated string
 * @param sz unused, required for the operator to work
 * @return `sid_t` string id
 */
sid_t consteval operator""_side(char const* str, [[maybe_unused]] uint64_t sz)
{
    return hashCRC64(str);
}

/**
 * Operator to convert an ASCII string literal to its string id form with CRC64 algorithm
 * When building a `Debug` or `RelWithDebInfo` build, this will also register a mapping in the
 * currently used string table
 * Prefer `_side` when you need constant time evaluation
 * @param str '\0' terminated string
 * @param sz unused, required for the operator to work
 * @return `sid_t` string id
 */
#if defined(DMT_DEBUG)
inline constexpr uint32_t maxTaglength = 128;
void                      internStringToCurrent(sid_t sid, char const* str, uint64_t sz);
sid_t                     operator""_sid(char const* str, [[maybe_unused]] uint64_t sz)
{
    assert(sz < maxTaglength);
    sid_t sid = hashCRC64(str);
    internStringToCurrent(sid, str, sz);
    return sid;
}
#else
sid_t operator""_sid(char const* str, [[maybe_unused]] uint64_t sz)
{
    return hashCRC64(str);
}
#endif

std::string_view lookupInternedStr(sid_t sid);

enum class EPageSize : uint32_t
{
    e4KB  = 1u << 12u,
    e2MB  = 1u << 21u,
    e1GB  = 1u << 30u,
    Count = 3
};

constexpr uint32_t toUnderlying(EPageSize ePageSize)
{
    return static_cast<uint32_t>(ePageSize);
}

constexpr auto operator<=>(EPageSize lhs, EPageSize rhs) noexcept
{
    return toUnderlying(lhs) <=> toUnderlying(rhs);
}

EPageSize scaleDownPageSize(EPageSize pageSize)
{
    switch (pageSize)
    {
        case EPageSize::e1GB:
            return EPageSize::e2MB;
        case EPageSize::e2MB:
            return EPageSize::e4KB;
        default:
            return EPageSize::e4KB;
    }
}

enum class EPageAllocationQueryOptions : uint8_t
{
    eNone = 0,
    eNo1GB,
    eForce4KB,
    Count,
};

#if defined(_MSC_VER)
#pragma pack(1)
#endif
struct alignas(8) PageAllocation
{
    void*     address;
    int64_t   pageNum; // in linux, 55 bits, windows void pointer, but it should be at least 4KB aligned
    EPageSize pageSize;
    uint32_t  bits  : 24; /** OS specific information, eg was this allocated with mmap or aligned_alloc? */
    uint32_t  count : 8;
};
#if defined(_MSC_VER)
#pragma pack()
#endif
static_assert(sizeof(PageAllocation) == 24 && alignof(PageAllocation) == 8);

struct AllocatePageForBytesResult
{
    size_t   numBytes;
    uint32_t numPages;
};

struct PageAllocatorHooks
{
    void (*allocHook)(void* data, PlatformContext& ctx, PageAllocation const& alloc);
    void (*freeHook)(void* data, PlatformContext& ctx, PageAllocation const& alloc);
    void* data;
};

// TODO make OS specific protected functions which handle parts of the logic, such
// TODO that we can test them individually by subclassing this
class alignas(8) PageAllocator
{
public:
    // -- Types --

    // -- Construtors/Copy Control --
    PageAllocator(PlatformContext& ctx, PageAllocatorHooks const &hooks);
    PageAllocator(PageAllocator const&)            = default;
    PageAllocator(PageAllocator&&) noexcept        = default;
    PageAllocator& operator=(PageAllocator const&) = default;
    PageAllocator& operator=(PageAllocator&&)      = default;
    ~PageAllocator();

    // -- Functions --
    // there's no alignment requirement being passed because we are allocating pages
    [[nodiscard]] AllocatePageForBytesResult allocatePagesForBytes(
        PlatformContext& ctx,
        size_t           numBytes,
        PageAllocation*  pOut,
        uint32_t         inNum,
        EPageSize        pageSize);
    EPageSize      allocatePagesForBytesQuery(PlatformContext&            ctx,
                                              size_t                      numBytes,
                                              uint32_t&                   inOutNum,
                                              EPageAllocationQueryOptions opts = EPageAllocationQueryOptions::eForce4KB);
    bool           checkPageSizeAvailability(PlatformContext& ctx, EPageSize pageSize);
    PageAllocation allocatePage(PlatformContext& ctx, EPageSize sizeOverride = EPageSize::e1GB);
    static void    deallocatePage(PlatformContext& ctx, PageAllocation& alloc);
    void           deallocPage(PlatformContext& ctx, PageAllocation& alloc);

    // TODO should test drive design?

protected:
#if defined(DMT_OS_WINDOWS)
    /**
     * Check whether the current process token has "SeLockMemoryPrivilege". If it has it, 
     * then enabled it if not enabled. If there's no privilege or cannot enable it, false
     */
    static bool  checkAndAdjustPrivileges(PlatformContext& ctx,
                                          void*            hProcessToken,
                                          void const*      seLockMemoryPrivilegeLUID,
                                          void*            pData);
    static bool  enableLockPrivilege(PlatformContext& ctx,
                                     void*            hProcessToken,
                                     void const*      seLockMemoryPrivilegeLUID,
                                     int64_t          seLockMemoryPrivilegeIndex,
                                     void*            pData);
    static bool  checkVirtualAlloc2InKernelbaseDll(PlatformContext& ctx);
    static void* createImpersonatingThreadToken(PlatformContext& ctx, void* hProcessToken, void* pData);
#endif

private:
    explicit PageAllocator(PageAllocatorHooks const& hooks) : m_hooks(hooks) {}

    //-- Members --
#if defined(DMT_OS_LINUX)
    int32_t   m_mmap2MBHugepages   = 0;
    int32_t   m_mmap1GBHugepages   = 0;
    EPageSize m_enabledPageSize    = EPageSize::e4KB;
    uint32_t  m_thpSize            = 0;
    bool      m_thpEnabled         = false;
    bool      m_mmapHugeTlbEnabled = false;
    bool      m_hugePageEnabled    = false;
#elif defined(DMT_OS_WINDOWS)
    uint32_t      m_allocationGranularity = 0;
    uint32_t      m_systemPageSize        = 0;
    bool          m_largePage1GB          = false;
    bool          m_largePageEnabled      = false;
    unsigned char m_padding[14];
#endif
    PageAllocatorHooks m_hooks;
};

static_assert(alignof(PageAllocator) == 8 && sizeof(PageAllocator) == 48);

struct alignas(8) AllocationInfo 
{
    void*      address;
    uint64_t allocTime; // millieconds from start of app
    uint64_t freeTime;  // 0 means not freed yet
    size_t   size;
    sid_t    sid;
    uint32_t alignment;
    EMemoryTag tag;
    //TODO handle source location
    unsigned char padding[8];
};
static_assert(sizeof(AllocationInfo) == 56 && alignof(AllocationInfo) == 8);

template <typename T>
    requires(std::is_standard_layout_v<T> && std::is_trivially_destructible_v<T>)
union Node 
{
    using DType = T;
    struct TrackData
    {
        T alloc;
        Node* next;
    };
    struct Free
    {
        uint64_t magic;
        Node* nextFree;
    };
    TrackData data;
    Free      free;
};

template union Node<PageAllocation>;
template union Node<AllocationInfo>;
using PageNode = Node<PageAllocation>;
using AllocNode = Node<AllocationInfo>;
static_assert(sizeof(PageNode) == 32 && alignof(PageNode) == 8);
static_assert(sizeof(AllocNode) == 64 && alignof(AllocNode) == 8);
static_assert(TemplateInstantiationOf<Node, PageNode>);
static_assert(TemplateInstantiationOf<Node, AllocNode>);


template <typename NodeType>
    requires(TemplateInstantiationOf<Node, NodeType>)
class FreeList
{
    friend class PageAllocationsTracker;

public:
    static constexpr uint64_t theMagic = static_cast<uint64_t>(-1);

    class Iterator
    {
    public:
        explicit Iterator(NodeType* node) : m_current(node) { }

        Iterator& operator++()
        {
            if (m_current)
                m_current = m_current->data.next;

            return *this;
        }

        bool operator!=(Iterator const& other) const
        {
            return m_current != other.m_current;
        }

        NodeType& operator*() const
        {
            return *m_current;
        }

        NodeType* operator->() const
        {
            return m_current;
        }

    private:
        NodeType* m_current = nullptr;
    };

    Iterator beginAllocated()
    {
        return Iterator(m_occupiedHead);
    }
    Iterator endAllocated()
    {
        return Iterator(nullptr);
    }

    // Add a node to the occupied list from the free list
    void addNode(typename NodeType::DType const& data)
    {
        assert(m_freeHead && "unexpected node state");

        // Allocate node from free list
        NodeType* node = m_freeHead;
        m_freeHead     = getNextFree(node);

        // Initialize the node with data and link to the occupied list
        node->data.alloc = data;
        setNextOccupied(node, m_occupiedHead);
        m_occupiedHead = node;
    }

    // Remove a node from the occupied list and return it to the free list
    bool removeNode(typename NodeType::DType const& data)
    {
        NodeType* prev = nullptr;
        NodeType* curr = m_occupiedHead;

        // Locate the node in the occupied list
        while (curr)
        {
            if (curr->data.alloc.address == data.address)
            {
                // Unlink from occupied list
                if (prev)
                    setNextOccupied(prev, getNextOccupied(curr));
                else
                    m_occupiedHead = getNextOccupied(curr);

                // Return node to free list
                setFreeNode(curr);
                return true;
            }
            prev = curr;
            curr = getNextOccupied(curr);
        }
        return false; // Node not found
    }

    // Reset the free list to its initial state
    void reset()
    {
        NodeType* curr = m_freeHead;
        for (uint32_t i = 0; i < m_freeSize - 1; ++i)
        {
            NodeType* next = m_growBackwards ? curr - 1 : curr + 1;
            setNextFree(curr, next);
            curr->free.magic = theMagic;
            curr             = next;
        }

        if (curr)
        {
            setNextFree(curr, nullptr); // Last node
            curr->free.magic = theMagic;
        }
    }

    // Grow the free list with new nodes
    void growList(uint32_t newNodes, NodeType* start)
    {
        NodeType* curr = start;

        for (uint32_t i = 0; i < newNodes - 1; ++i)
        {
            NodeType* next = m_growBackwards ? curr - 1 : curr + 1;
            setNextFree(curr, next);
            curr->free.magic = theMagic;
            curr             = next;
        }

        if (curr)
        {
            setNextFree(curr, m_freeHead); // Link new list to existing free list
            curr->free.magic = theMagic;
        }

        m_freeHead = start;
        m_freeSize += newNodes;
    }

private:
    NodeType* m_freeHead      = nullptr; // Head of the free list
    NodeType* m_occupiedHead  = nullptr; // Head of the occupied list
    uint32_t  m_capacity      = 0;       // Total capacity of the list
    uint32_t  m_freeSize      = 0;       // Number of free nodes available
    bool      m_growBackwards = false;   // Growth direction

    // Helper functions for managing free and occupied nodes
    NodeType* getNextFree(NodeType* node) const
    {
        return reinterpret_cast<NodeType*>(node->free.nextFree);
    }

    void setNextFree(NodeType* node, NodeType* next) const
    {
        node->free.nextFree = reinterpret_cast<NodeType*>(next);
    }

    NodeType* getNextOccupied(NodeType* node) const
    {
        return reinterpret_cast<NodeType*>(node->data.next);
    }

    void setNextOccupied(NodeType* node, NodeType* next) const
    {
        node->data.next = reinterpret_cast<NodeType*>(next);
    }

    void setFreeNode(NodeType* node)
    {
        setNextFree(node, m_freeHead);
        node->free.magic = theMagic;
        m_freeHead       = node;
    }
};
template class FreeList<PageNode>;
template class FreeList<AllocNode>;

// reserve a large portion of the virtual address to memory tracking and treat it like a stack
// on one end we'll track page allocations, on the other object allocation. Since we are not committing
// all memory at once
// TODO = for now it works with 1 page, then write a per-platform method to reserve virtual address space and
// commit on usage
// with a single 4KB bootstrap page, you can allocate 128 nodes (ie pages), which corresponds to 512 KB if all 
// pages are of 4KB. Theferore we need more memory. A good strategy would be to allocate a big enough portion
    // of the virtual address space.
class alignas(8) PageAllocationsTracker
{
public:
    struct PageAllocationView 
    {
        PageAllocationView(FreeList<PageNode>* pageTracking) : m_pageTracking(pageTracking) { }
        auto begin()
        {
            return m_pageTracking->beginAllocated();
        }
        auto end()
        {
            return m_pageTracking->endAllocated();
        }

    private:
        FreeList<PageNode> *m_pageTracking;
    };
    struct AllocationView 
    {
        AllocationView(FreeList<AllocNode>* allocTracking) : m_allocTracking(allocTracking) { }
        auto begin()
        {
            return m_allocTracking->beginAllocated();
        }
        auto end()
        {
            return m_allocTracking->endAllocated();
        }
    private:
        FreeList<AllocNode> *m_allocTracking;
    };

    // Forward iterator for occupied list
    PageAllocationsTracker(PlatformContext& ctx, uint32_t pageTrackCapacity, uint32_t allocTrackCapacity);
    ~PageAllocationsTracker() noexcept;

    PageAllocationView pageAllocations()
    {
        return {&m_pageTracking};
    }

    AllocationView allocations()
    {
        return {&m_allocTracking};
    }

    // start simple: Handle embedded free list inside bootstrap page with allocation and deallocation functions
    void track(PlatformContext& ctx, PageAllocation const &alloc);
    void untrack(PlatformContext& ctx, PageAllocation const &alloc);
    void track(PlatformContext& ctx, AllocationInfo const &alloc);
    void untrack(PlatformContext& ctx, AllocationInfo const &alloc);

private:
    static constexpr uint32_t initialNodeNum = 128;
    template <typename NodeType>
        requires(TemplateInstantiationOf<Node, NodeType>)
    static void growFreeList(PlatformContext& ctx, FreeList<NodeType> &freeList, void* base)
    {
        // Check if we can commit more memory
        if (freeList.m_freeSize >= freeList.m_capacity)
        {
            ctx.error("Buffer capacity exceeded, cannot grow the free list further.");
            std::abort();
        }

        // Calculate how many new nodes to commit (doubling strategy)
        uint32_t newNodes = std::min(initialNodeNum, freeList.m_capacity - freeList.m_freeSize);
        if (newNodes == 0)
        {
            ctx.error("Cannot grow free list: Capacity fully utilized.");
            std::abort();
        }

        // Determine where to start the new free list
        void* newBuffer = nullptr;
        if (freeList.m_growBackwards)
        {
            newBuffer = reinterpret_cast<void*>(
                reinterpret_cast<uintptr_t>(base) - (freeList.m_freeSize + newNodes) * sizeof(NodeType));
        }
        else
        {
            newBuffer = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + freeList.m_freeSize * sizeof(NodeType));
        }

        // Commit additional memory
        if (!commitPhysicalMemory(newBuffer, newNodes * sizeof(NodeType)))
        {
            ctx.error("Failed to commit additional memory for {} nodes.", {newNodes});
            std::abort();
        }

        // Add newly committed nodes to the free list
        NodeType* newFreeListStart = reinterpret_cast<NodeType*>(newBuffer);
        freeList.growList(newNodes, newFreeListStart);
    }

    FreeList<PageNode> m_pageTracking;
    FreeList<AllocNode> m_allocTracking;
    void* m_base   = nullptr;
    void* m_buffer = nullptr;
    size_t m_bufferBytes;
    void*               m_pageBase;
    void*               m_allocBase;
    unsigned char       m_padding[24];
};
static_assert(sizeof(PageAllocationsTracker) == 128 && alignof(PageAllocationsTracker) == 8);

class StackAllocator
{
private:
    struct alignas(8) StackHeader
    {
        uintptr_t bp;
        uintptr_t sp;
        size_t    sz;
        StackHeader* next;
    };
    static_assert(sizeof(StackHeader) == 32 && alignof(StackHeader) == 8);

    // cleanup: until last is different than first, give back the buffer to the page tracker, which will be
    // allocated in the "bootstrap memory page", which is a minimuum (4KB) page containing whathever data the 
    // application needs to track in order to start
    // the bootstrap page should contain, as last member, the allocation tracking data, because it is variable length
    // there might be the possibility we have to track 2 variable length arrays, eg

    StackHeader *m_pFirst;
    StackHeader *m_pLast; 
};

} // namespace dmt
/** @} */
