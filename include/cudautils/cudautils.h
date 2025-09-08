#pragma once

#include <cstdint>
#include "cudautils/cudautils-macro.h"

#include "cudautils/cudautils-enums.h"
#include "cudautils/cudautils-float.h"
#include "cudautils/cudautils-vecmath.h"
#include "cudautils/cudautils-light.h"
#include "cudautils/cudautils-transform.h"
#include "cudautils/cudautils-camera.h"
#include "cudautils/cudautils-media.h"
#include "cudautils/cudautils-lightsampler.h"
#include "cudautils/cudautils-texture.h"
#include "cudautils/cudautils-material.h"
#include "cudautils/cudautils-sampler.h"
#include "cudautils/cudautils-film.h"
#include "cudautils/cudautils-filter.h"
#include "cudautils/cudautils-bxdf.h"
#include "cudautils/cudautils-spectrum.h"
#include "cudautils/cudautils-color.h"
#include "cudautils/cudautils-numbers.h"
#include "cudautils/cudautils-image.h"
#include "cudautils/cudautils-bvh.h"

#include "cuda-wrappers/cuda-wrappers-utils.h"


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

    struct float3
    {
        float x, y, z;
    };
    DMT_GPU inline float3 make_f3(float x, float y, float z) { return {x, y, z}; }
    DMT_GPU inline float3 operator-(float3 const& a, float3 const& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    DMT_GPU inline float  dot3(float3 const& a, float3 const& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    DMT_GPU inline float3 cross3(float3 const& a, float3 const& b)
    {
        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    }
    DMT_GPU inline float3 mul3(float3 const& a, float s) { return {a.x * s, a.y * s, a.z * s}; }
    DMT_GPU inline float3 add3(float3 const& a, float3 const& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
    DMT_GPU inline float3 normalize3(float3 const& a)
    {
        float inv = rsqrtf(dot3(a, a));
        return mul3(a, inv);
    }

    struct float2
    {
        float x, y;
    };
    DMT_GPU inline float2 make_float2(float x, float y) { return {x, y}; }

    // ---- Slot decode helpers: adapt to your packing scheme ----
    DMT_GPU inline bool     slot_is_valid(uint64_t slot) { return (slot != 0ull); }
    DMT_GPU inline bool     slot_is_leaf(uint64_t slot) { return ((slot >> 24) & 0x1ull) != 0ull; }
    DMT_GPU inline uint32_t slot_index(uint64_t slot) { return uint32_t(slot & 0x00FFFFFFu); }

    static char const* lane_id          = R"a(
    static __inline__ __device__ uint32_t lane_id() { return threadIdx.x & 31; }
)a";
    static char const* warp_id_in_block = R"a(
    static __inline__ __device__ uint32_t warp_id_in_block() { return threadIdx.x >> 5; }
)a";

    // ---------- Warp bulk enqueue ----------
    // Active lanes call with valid=true and provide 'val' to enqueue to q.
    // Returns the global position written by each active lane.
    static char const* warp_enqueue = R"a(
static DMT_FORCEINLINE DMT_GPU uint32_t warp_enqueue(IndexQueue q, bool valid, uint32_t val)
    {
        unsigned mask  = __ballot_sync(0xffffffff, valid);
        int      count = __popc(mask);
        if (count == 0)
            return 0u;
        int lane = lane_id();
        int rank = __popc(mask & ((1u << lane) - 1));

        uint32_t base = 0;
        if (rank == 0)
        { // first active lane reserves space
            base = atomicAdd(q.tail, count);
        }
        base = __shfl_sync(0xffffffff, base, __ffs(mask) - 1);
        if (valid)
        {
            uint32_t pos = base + rank;
            if (pos < q.capacity)
                q.items[pos] = val;
            return pos;
        }
        return 0u;
    }
)a";


}; // namespace dmt