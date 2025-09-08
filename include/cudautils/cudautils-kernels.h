#pragma once

#include "cudautils/cudautils-macro.h"

namespace dmt {
    // ===================================================
    // Kernel 1: RayGen ? fill RayPool and enqueue indices
    // Each warp generates one ray; we map one ray per warp here,
    // but to keep occupancy, we *allow* each thread to handle one ray
    // and rely on warp size being a multiple of 32.
    // ===================================================
    static char const* kRayGen = R"a(
    __global__ void kRayGen(DeviceCamera cam, int tileStartX, int tileStartY, int tileW, int tileH, int spp, RayPool rayPool, IndexQueue rayQ)
    {
        // One thread == one pixel-sample; warps will cooperatively enqueue
        int tid     = blockIdx.x * blockDim.x + threadIdx.x;
        int threads = gridDim.x * blockDim.x;

        int total = tileW * tileH * spp;

        for (int i = tid; i < total; i += threads)
        {
            int s  = i % spp;
            int p  = i / spp;
            int tx = p % tileW;
            int ty = p / tileW;
            int px = tileStartX + tx;
            int py = tileStartY + ty;

            // RNG
            uint32_t seed = 0x12345678u ^ (px * 9781u + py * 6271u + s * 13007u);
            float    u1   = rand01(seed);
            float    u2   = rand01(seed);

            float3 ro, rd;
            generate_camera_ray(cam, px, py, u1, u2, ro, rd);

            // Claim a ray slot = i (re-using linear index) or use atomic counter
            uint32_t rayIdx = i; // safe because total <= capacity and unique
            if (rayIdx >= rayPool.capacity)
                continue;

            rayPool.ox[rayIdx]   = ro.x;
            rayPool.oy[rayIdx]   = ro.y;
            rayPool.oz[rayIdx]   = ro.z;
            rayPool.dx[rayIdx]   = rd.x;
            rayPool.dy[rayIdx]   = rd.y;
            rayPool.dz[rayIdx]   = rd.z;
            rayPool.tmin[rayIdx] = 0.0f;
            rayPool.tmax[rayIdx] = INF_F;

            rayPool.beta_r[rayIdx]    = 1.f;
            rayPool.beta_g[rayIdx]    = 1.f;
            rayPool.beta_b[rayIdx]    = 1.f;
            rayPool.lastPdf[rayIdx]   = 1.f;
            rayPool.depth[rayIdx]     = 0;
            rayPool.pixX[rayIdx]      = px;
            rayPool.pixY[rayIdx]      = py;
            rayPool.sampleIdx[rayIdx] = s;
            rayPool.rng0[rayIdx]      = seed ^ 0xa5a5a5a5u;
            rayPool.rng1[rayIdx]      = seed ^ 0x3c3c3c3cu;

            // Enqueue
            warp_enqueue(rayQ, true, rayIdx);
        }
    } 
)a";
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
    static char const* trace_kernel = R"a(
    __global__ void trace_kernel_warp(const Ray* __restrict__ rays,
                                  int numRays,
                                  const BVHWiVeCluster* __restrict__ nodes,
                                  int nodeCount,
                                  GpuHaltonOwenSampler*              samplers,
                                  HitOut* __restrict__ hitOut)
{
    int warpId = (blockDim.x * blockIdx.x + threadIdx.x) / warpSize;
    if (warpId >= numRays) return;

    int laneId = threadIdx.x % warpSize;
    unsigned mask = 0xFFFFFFFF;

    // Lane 0 loads the ray
    Ray ray;
    if (laneId == 0) ray = rays[warpId];
    ray.o = __shfl_sync(mask, ray.o, 0);
    ray.d = __shfl_sync(mask, ray.d, 0);
    ray.invd = __shfl_sync(mask, ray.invd, 0);

    float bestT = 1e30f;
    int bestTri = -1, bestInst = -1;

    // Per-warp stack in shared memory
    __shared__ int warpStack[64 * 4]; // 64 depth * 4 warps per block
    int* stack = &warpStack[(threadIdx.x / warpSize) * 64];
    int sp = 0;
    if (laneId == 0) stack[sp++] = 0; // root index
    sp = __shfl_sync(mask, sp, 0);

    while (sp > 0) {
        int nodeIdx;
        if (laneId == 0) nodeIdx = stack[--sp];
        nodeIdx = __shfl_sync(mask, nodeIdx, 0);
        sp = __shfl_sync(mask, sp, 0);

        const BVHWiVeCluster& node = nodes[nodeIdx];

        // Each of 8 lanes checks one child AABB
        int c = laneId;
        bool hit = false;
        uint32_t childIdx = 0;
        bool isLeaf = false;
        float tmin;

        if (c < SIMDWidth) {
            uint64_t slot = node.slotEntries[c];
            if (slot != 0ull) {
                childIdx = slot_index(slot);
                isLeaf = slot_is_leaf(slot);
                hit = aabb_intersect_local(ray,
                                           node.bxmin[c], node.bxmax[c],
                                           node.bymin[c], node.bymax[c],
                                           node.bzmin[c], node.bzmax[c],
                                           bestT, tmin);
            }
        }

        unsigned hitMask = __ballot_sync(mask, hit);

        // Process hit children one by one
        while (hitMask) {
            int childLane = __ffs(hitMask) - 1; // first set bit
            hitMask &= ~(1u << childLane);

            uint32_t ci = __shfl_sync(mask, childIdx, childLane);
            bool leaf  = __shfl_sync(mask, isLeaf, childLane);

            if (leaf) {
                // reinterpret as leaf
                const BVHWiVeLeaf* leafNode = reinterpret_cast<const BVHWiVeLeaf*>(&nodes[ci]);
                int triCount = leafNode->triCount;

                // distribute triangles
                if (laneId < triCount) {
                    float tHit;
                    if (intersect_tri_mt(ray,
                                         leafNode->v0s[laneId],
                                         leafNode->v1s[laneId],
                                         leafNode->v2s[laneId],
                                         bestT, tHit)) {
                        if (tHit < bestT) {
                            bestT = tHit;
                            bestTri = leafNode->triIdx[laneId];
                            bestInst = leafNode->instanceIdx;
                        }
                    }
                }
                // sync warp before next child
                __syncwarp(mask);
            } else {
                if (laneId == 0) {
                    stack[sp++] = ci;
                }
                sp = __shfl_sync(mask, sp, 0);
            }
        }
    }

    if (laneId == 0) {
        hitOut[warpId].t = bestT;
        hitOut[warpId].triIdx = bestTri;
        hitOut[warpId].instanceIdx = bestInst;
    }
}
)a";

    // ===================================================
    // Kernel 2: Intersect ? miss adds env to film; hit ? HitPool + ShadeQ
    // ===================================================
    static char const* kIntersect = R"a(
__global__ void kIntersect(DeviceBVH bvh, DeviceLights lights, FilmSOA film, RayPool rays, HitPool hits, IndexQueue inRayQ, IndexQueue shadeQ)
    {
        int      tid       = blockIdx.x * blockDim.x + threadIdx.x;
        int      threads   = gridDim.x * blockDim.x;
        uint32_t queueSize = *inRayQ.tail; // snapshot

        for (uint32_t qIdx = tid; qIdx < queueSize; qIdx += threads)
        {
            uint32_t rayIdx = inRayQ.items[qIdx];
            // load ray
            float3 ro   = {rays.ox[rayIdx], rays.oy[rayIdx], rays.oz[rayIdx]};
            float3 rd   = {rays.dx[rayIdx], rays.dy[rayIdx], rays.dz[rayIdx]};
            float  tmin = rays.tmin[rayIdx], tmax = rays.tmax[rayIdx];

            // traverse
            int    inst = -1, tri = -1;
            float  t = INF_F, u = 0.f, v = 0.f;
            float3 ng  = {0, 0, 1};
            bool   hit = bvh_traverse(bvh, ro, rd, tmin, tmax, inst, tri, t, u, v, ng);

            if (!hit)
            {
                // env MIS (simple add)
                float Lr, Lg, Lb, pdfL;
                env_eval(lights, rd, Lr, Lg, Lb, pdfL);
                int px = rays.pixX[rayIdx], py = rays.pixY[rayIdx];
                if (px >= 0 && px < film.width && py >= 0 && py < film.height)
                {
                    int pix = py * film.width + px;
                    // beta * env
                    float br = rays.beta_r[rayIdx], bg = rays.beta_g[rayIdx], bb = rays.beta_b[rayIdx];
                    atomicAdd(&film.r[pix], br * Lr);
                    atomicAdd(&film.g[pix], bg * Lg);
                    atomicAdd(&film.b[pix], bb * Lb);
                    atomicAdd(&film.w[pix], 1.0f);
                }
                continue;
            }

            // Write hit
            hits.t[rayIdx]      = t;
            hits.u[rayIdx]      = u;
            hits.v[rayIdx]      = v;
            hits.instId[rayIdx] = inst;
            hits.triId[rayIdx]  = tri;
            hits.ngx[rayIdx]    = ng.x;
            hits.ngy[rayIdx]    = ng.y;
            hits.ngz[rayIdx]    = ng.z;

            // enqueue for shading
            warp_enqueue(shadeQ, true, rayIdx);
        }
    }
)a";


    // ===================================================
    // Kernel 3: Shade ? direct light (optional) + BSDF sample ? enqueue nextRay
    // (Shadow rays omitted here for brevity; add a ShadowQ pass later.)
    // ===================================================
    static char const* kShade = R"a(
    __global__ void kShade(DeviceBVH       bvh,
                           DeviceLights    lights,
                           DeviceMaterials mats,
                           FilmSOA         film,
                           RayPool         rays,
                           HitPool         hits,
                           int             maxDepth,
                           IndexQueue      shadeQ,
                           IndexQueue      nextRayQ)
    {
        int      tid       = blockIdx.x * blockDim.x + threadIdx.x;
        int      threads   = gridDim.x * blockDim.x;
        uint32_t queueSize = *shadeQ.tail;

        for (uint32_t qIdx = tid; qIdx < queueSize; qIdx += threads)
        {
            uint32_t rayIdx = shadeQ.items[qIdx];

            int depth = rays.depth[rayIdx];
            if (depth >= maxDepth)
                continue;

            // Load hit data
            float  t  = hits.t[rayIdx];
            float3 ro = {rays.ox[rayIdx], rays.oy[rayIdx], rays.oz[rayIdx]};
            float3 rd = {rays.dx[rayIdx], rays.dy[rayIdx], rays.dz[rayIdx]};
            float3 p  = {ro.x + rd.x * t, ro.y + rd.y * t, ro.z + rd.z * t};
            float3 ng = {hits.ngx[rayIdx], hits.ngy[rayIdx], hits.ngz[rayIdx]};

            // Simple diffuse lobe as placeholder BSDF
            uint32_t s0 = rays.rng0[rayIdx];
            float    u1 = rand01(s0), u2 = rand01(s0);

            // Cosine hemisphere sample (very rough)
            float  phi   = 2.f * 3.14159265f * u1;
            float  r     = sqrtf(u2);
            float  z     = sqrtf(1.f - u2);
            float3 local = {r * cosf(phi), r * sinf(phi), z};

            // Build ONB from ng (naive)
            float3 w  = norm(ng);
            float3 a  = fabsf(w.x) > 0.5f ? make_f3(0, 1, 0) : make_f3(1, 0, 0);
            float3 v  = norm({w.y * a.z - w.z * a.y, w.z * a.x - w.x * a.z, w.x * a.y - w.y * a.x});
            float3 u  = {v.y * w.z - v.z * w.y, v.z * w.x - v.x * w.z, v.x * w.y - v.y * w.x};
            float3 wi = norm({u.x * local.x + v.x * local.y + w.x * local.z,
                              u.y * local.x + v.y * local.y + w.y * local.z,
                              u.z * local.x + v.z * local.y + w.z * local.z});

            // Update throughput beta (Lambert with albedo=0.8)
            float albedo = 0.8f;
            float cosNI  = fmaxf(0.f, wi.x * ng.x + wi.y * ng.y + wi.z * ng.z);
            float pdf    = local.z * (1.0f / 3.14159265f); // cosine / pi
            if (pdf <= 1e-8f)
                continue;
            float3 f = make_f3(albedo / 3.14159265f, albedo / 3.14159265f, albedo / 3.14159265f);

            float br = rays.beta_r[rayIdx], bg = rays.beta_g[rayIdx], bb = rays.beta_b[rayIdx];
            br = br * f.x * cosNI / pdf;
            bg = bg * f.y * cosNI / pdf;
            bb = bb * f.z * cosNI / pdf;

            // Russian roulette
            float maxc = fmaxf(br, fmaxf(bg, bb));
            if (depth > 2)
            {
                float q   = fminf(0.95f, maxc);
                float uRR = rand01(s0);
                if (uRR > q)
                { // terminate
                    rays.rng0[rayIdx] = s0;
                    continue;
                }
                float inv = 1.f / q;
                br *= inv;
                bg *= inv;
                bb *= inv;
            }

            // Spawn next ray
            float eps         = 1e-4f; // offset
            rays.ox[rayIdx]   = p.x + wi.x * eps;
            rays.oy[rayIdx]   = p.y + wi.y * eps;
            rays.oz[rayIdx]   = p.z + wi.z * eps;
            rays.dx[rayIdx]   = wi.x;
            rays.dy[rayIdx]   = wi.y;
            rays.dz[rayIdx]   = wi.z;
            rays.tmin[rayIdx] = 0.0f;
            rays.tmax[rayIdx] = INF_F;

            rays.beta_r[rayIdx]  = br;
            rays.beta_g[rayIdx]  = bg;
            rays.beta_b[rayIdx]  = bb;
            rays.depth[rayIdx]   = depth + 1;
            rays.lastPdf[rayIdx] = pdf;
            rays.rng0[rayIdx]    = s0;

            // Enqueue for next intersect pass
            warp_enqueue(nextRayQ, true, rayIdx);
        }
    }
)a";
} // namespace dmt
