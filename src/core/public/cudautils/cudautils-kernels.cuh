#ifndef DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_KERNELS_CUH
#define DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_KERNELS_CUH

#include "core/cudautils/cudautils-macro.cuh"

#include <cstdint>

namespace dmt {

    //Take trace of the allocations in the global memory
    template <typename T>
    struct DeviceBuffer
    {
        T*       ptr      = nullptr;
        uint32_t capacity = 0;
    };

    // ---------- Simple queue of indices ----------
    struct IndexQueue
    {
        uint32_t* items    = nullptr; // indices into RayPool/ShadowPool/etc.
        uint32_t* tail     = nullptr; // atomic append pointer
        uint32_t  capacity = 0;
    };


    // ---- Slot decode helpers: adapt to your packing scheme ----
    DMT_GPU inline bool     slot_is_valid(uint64_t slot) { return (slot != 0ull); }
    DMT_GPU inline bool     slot_is_leaf(uint64_t slot) { return ((slot >> 24) & 0x1ull) != 0ull; }
    DMT_GPU inline uint32_t slot_index(uint64_t slot) { return uint32_t(slot & 0x00FFFFFFu); }

    inline __device__ uint32_t lane_id() { return threadIdx.x & 31; }
    inline __device__ uint32_t warp_id_in_block() { return threadIdx.x >> 5; }

    // ---------- Warp bulk enqueue ----------
    // Active lanes call with valid=true and provide 'val' to enqueue to q.
    // Returns the global position written by each active lane.
    DMT_GPU uint32_t warp_enqueue(IndexQueue q, bool valid, uint32_t val);

    // ===================================================
    // Kernel 1: RayGen ? fill RayPool and enqueue indices
    // Each warp generates one ray; we map one ray per warp here,
    // but to keep occupancy, we *allow* each thread to handle one ray
    // and rely on warp size being a multiple of 32.
    // ===================================================
    __global__ void kRayGen(DeviceCamera cam, int tileStartX, int tileStartY, int tileW, int tileH, int spp, RayPool rayPool, IndexQueue rayQ);
    /*
     * trace_kernel_warp
     *
     * Warp-per-ray ray tracing kernel for wide-fanout BVH (BVHWiVeCluster).
     *
     * This kernel assigns **one warp per ray**:
     * - Each warp cooperatively traverses the BVH for its assigned ray.
     * - Lane 0 of the warp owns the "master" ray data (origin, direction, tMax),
     *   which is broadcast to the rest of the warp using __shfl_sync.
     *
     * Traversal overview:
     * 1. Initialize a per-warp stack (in shared memory) with the root node.
     * 2. While the stack is not empty:
     *    a) Pop a node index (root or inner) from the stack.
     *    b) Load the BVHWiVeCluster at that index.
     *    c) Each lane tests one child bounding box (up to 8 children) for intersection:
     *       - Use a warp-wide ballot (__ballot_sync) to gather which children are hit.
     *       - Decode slotEntries to get child index and leaf/inner flags.
     *    d) For each hit child:
     *       i) If it is a leaf:
     *          - Cast the child to BVHWiVeLeaf.
     *          - Distribute triangles among warp lanes (one triangle per lane).
     *          - Intersect triangles using intersect_tri_mt.
     *          - Keep track of the closest hit in the warp using bestT.
     *       ii) If it is an inner node:
     *          - Lane 0 pushes the child index onto the warp stack.
     * 3. Repeat until the stack is empty (all potential intersections tested).
     *
     * After traversal:
     * - Lane 0 writes the closest hit information for this ray:
     *   - t value (hit distance)
     *   - triangle index
     *   - instance index
     *
     * Key warp-level operations:
     * - __shfl_sync(mask, value, srcLane) : broadcast values across lanes.
     * - __ballot_sync(mask, predicate)    : returns a mask of lanes where predicate is true.
     * - __ffs(mask)                       : finds first active lane in a mask.
     * - __syncwarp(mask)                  : synchronize lanes in a warp for cooperative work.
     *
     */
    __global__ void trace_kernel_warp(Ray const* __restrict__ rays,
                                      int numRays,
                                      BVHWiVeCluster const* __restrict__ nodes,
                                      int                   nodeCount,
                                      GpuHaltonOwenSampler* samplers,
                                      HitOut* __restrict__ hitOut);

    // ===================================================
    // Kernel 2: Intersect ? miss adds env to film; hit ? HitPool + ShadeQ
    // ===================================================
    __global__ void kIntersect(DeviceBVH    bvh,
                               DeviceLights lights,
                               FilmSOA      film,
                               RayPool      rays,
                               HitPool      hits,
                               IndexQueue   inRayQ,
                               IndexQueue   shadeQ);

    __global__ void kShade(DeviceBVH       bvh,
                           DeviceLights    lights,
                           DeviceMaterials mats,
                           FilmSOA         film,
                           RayPool         rays,
                           HitPool         hits,
                           int             maxDepth,
                           IndexQueue      shadeQ,
                           IndexQueue      nextRayQ);
} // namespace dmt
#endif // DMT_CORE_PUBLIC_CUDAUTILS_CUDAUTILS_KERNELS_CUH
