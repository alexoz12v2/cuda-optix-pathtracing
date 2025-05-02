# Vectorized Path Tracing

## AVX/AVX2 Fundamentals

- **[Advanced Vector Extensions](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)**

## Embree Notes

- [Embree](https://github.com/RenderKit/embree/blob/master/kernels/bvh/bvh.h)

  - BVH has 8 children in an AVX2 enabled implementation
  - Intersection path

    - 1: [`rtcIntersect8`](https://github.com/RenderKit/embree/blob/ffc56d50a319c5eb3f246b3c8623f054514b06e6/kernels/common/rtcore.cpp#L820)
    - 2: call intersector given from BVH factory ([example](https://github.com/RenderKit/embree/blob/ffc56d50a319c5eb3f246b3c8623f054514b06e6/kernels/bvh/bvh8_factory.cpp#L232))
      we are interested in the `bvh_intersector_hybrid.h/cpp`, capable of switching between packet and single ray traversal (assume K = 8, N = 8).

## Paper 1: [Vectorized Production Path Tracing](https://stg-research.dreamworks.com/wp-content/uploads/2018/07/Vectorized_Production_Path_Tracing_DWA_2017.pdf)

*Queueing* is used to keep all vector lanes full and improve data coherency. It it implemented using a C extensions called 
[ISPC](https://ispc.github.io/ispc.html#getting-started-with-ispc) (which basically is a compiler which autovectorizes the code given some intrinsics).
Hence, the path tracer here includes a C++ scalar version and a ISPC vectorized breadth-first wavefront version (look also into wavefront integrator of
PBRT for reference)

The followed approach is uni-directional path tracing [Kajiya 1986] with next event prediction [Pharr et al. 2016].

Path tracing is an inherently parallel, and therefore, it should exploit vectorized instructions and multiple cores.

- How do we access the vector hardware efficiently?
- How do we gather batches of work to keep the vector hardware busy?
- How do we avoid scatter gather and keep memory accesses coherent?

  - One step is transitioning from Array of Structs (AoS) to Struct of Arrays (SoA)

**Coherent Rendering** for ray tracing is necessary. Rendering systems therefore rely on spacial sorting, level of detail selection and streaming to efficiently pre-compute
spherical occlusions of out-of-core scenes

![Vectorized Path Tracing](Papers-Resources/vector.png)

Image generation is separated into 2 distinct phases:

1. **Preparation Phase**: load all assets and build the ray-tracing acceleration structure
2. **Rendering Phase**: execute the Monte Carlo asmpling and integration. *Here we focus our vectorization*

Overview of rendering phase:

- Partition image into small square pixel buckets, and sample primary rays from each
- Trace camera rays through the scene and trigger the execution of a surface shader (or inifinite light sampling if ray escapes)
- Compute BSDF closure, importance sample BSDF value
- *Next Event Estimation*: accumulate direct illumination + secondary ray, determined with Multiple Importance Sampling and Russian Roulette (path splitting). unimportant rays
  are culled.

The vectorized path is a feedforward breadth-first pipeline.

Multithreading here uses Intel's **Thread Building Blocks** (TBB) library.

- Memory allocations are confined to *thread local memory arenas* or pre allocated memory pools, to avoid synchronization stalls due to memory allocation
- Each Queue is split in multiple **Thread Local Storage** buffers, such that, each thread will use shared data in read-only (no synchronization needed) and its own TLS storage

Key to vectorization is **Queue Handling** (5.1)...TODO

## Paper 2: [Accelerated Single Ray Tracing for Wide Vector Units](https://web.cs.ucdavis.edu/~hamann/FuetterlingLojewskiPfreundtHamannEbertHPG2017PaperFinal06222017.pdf)

## Paper 3: [Getting Rid of Packets](https://graphics.stanford.edu/~boulos/papers/multi_rt08.pdf)

## Paper 4: [Local Shading Coherence Extraction for SIMD-Efficient Path Tracing on CPU](https://www.embree.org/papers/2016-HPG-shading.pdf)

## Paper 5: [Packet-based Whitted and Distribution Ray Tracing](https://graphics.stanford.edu/~boulos/papers/cook_gi07.pdf)

## Paper 4: [Understanding the efficiency of Ray Traversal on GPUs](https://research.nvidia.com/sites/default/files/pubs/2009-08_Understanding-the-Efficiency/aila2009hpg_paper.pdf)

## Paper 5: [Coherent Path Tracing](http://graphics.ucsd.edu/~henrik/papers/coherent_path_tracing.pdf)

## Paper 6: [GPU RaySorting](https://meistdan.github.io/publications/raysorting/paper.pdf)

## Notes with chatGPT

Vectorized Path tracing makes use of work queues to group work by similarity, such that clusters of work can be dispatched with vectorized instructions.

Rays, in particular, can be inserted in a given queue and sorted by origin, direction and other characteristics.

Ray packets are efficient if rays behave similiarly. THat is, if they traverse the same BVH nodes and hit similiar surfaces

- If rays in a packet hit different surfaces, or branch differently (reflection, transmission, scattering), they can *diverge*
- Divergence means masked warp execution on a CUDA GPU, and
  [conditional operations on AVX2](https://stackoverflow.com/questions/74454057/how-to-do-mask-conditional-branchless-arithmetic-operations-in-avx2)

1. **Use coherent rays whenever possible**: Camera rays and shadow rays are often coherent, while secondary rays (reflection, transmission) are not. meaning you need to
   implement a fallback
2. **Packet Tracing with Masked Execution**: use a mask to know which rays are active. For branching, evaluate different branches with a mask
3. Sort/Group rays by Direction/Material (as done by [embree](https://github.com/search?q=repo%3ARenderKit%2Fembree+__AVX+path%3A%2F%5Ekernels%5C%2Fbvh%5C%2F%2F&type=code))
