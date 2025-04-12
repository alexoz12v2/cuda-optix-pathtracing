# Introduction

## Glossary

- **Program**: executable device come, termed shader by Graphics APIs like Vulkan (I'll call them shaders anyways)

Programs in the Ray Tracing Pipeline defined by OptiX

- **Ray Generation**: Entry point to the ray tracing pipeline, invoked in parallel for each pixel, sample or other user-defined work assignment (like Rays) (page 107)
- **Intersection**: Implements a ray-primitive intersection test, invoked dureing traversal (page 118)
- **Any Hit**: Called when a traced ray finds a new, potentially closest, intersection point, such as for shadow computation (page 118)
- **Closest hit**: Called when a traced ray finds the closest intersection point, such as for material shading (page 74)
- **Miss**: Called when a traced ray misses all scene geometry (perfect place for light at infinity evaluation) (page 127)
- **Exception**: Defines an exception handler for erroneous situations (page 127) 
- **Direct Callables**: (page 133) 
- **Continuation Callables**: Executed by a scheduler (perfect for Accumulating statistics about the rendering and the image) (page 133)

![Ray Tracing Pipeline](resources/ray-tracing-pipeline.png) Ray tracing pipeline

**Shader Binding Table**: Connects geometric data to programs. A *Record* a is a component of the table which is selected at runtime by using 
offsets specified when acceleration structures are created. A record is broken up into

- *Record Header*: Used to identify programmatic behaviour. A primitive, for example, would identify an intersection
- *Record Data*: Any data you need at program execution (example, a color)

```cpp
// 0. OptixDeviceContext context already created
// 1. Create program groups for each type of program you need
OptixProgramGroup raygen_progGroup = nullptr;

// then describe what each program is supposed to be
OptixProgramGroupOptions options{};
OptixProgramGroupDesc raygen_progGroupDescriptor = {};
raygen_progGroupDescriptor.kind = ::OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
raygen_progGroupDescriptor.raygen.module = // CUmodule which contains the raygen function
raygen_progGroupDescriptor.raygen.entryFunctionName = // PTX .entry of __global__ fun (void(*)())
optixProgramGroupCreate(
    optixContext, 
    &raygen_progGroupDescriptor, 1/*count*/, 
    &options, 
    log, logSize, 
    &raygen_progGroup
);

// 2. Create shader binding tables
OptixShaderBindingTable sbt = {}; // see below, end of step 2
CUdeviceptr d_raygenRecord = 0;
cuMemAlloc(&d_raygenRecord, sizeof(RayGenSbtRecord)); // sbt = shader binding table
RayGenSbtRecord rg_sbt;

// I want to use this inside all raygen shaders (prepare the header on the host)
optixSbtRecordPackHeader(raygen_progGroup, &rg_sbt);
// arbitrary data. A color, in this example (prepare the data on the host)
rg_sbt.data = { 0.2f, 0.2f, 0.0f };
cuMemCpyHtoD(d_raygenRecord, &rg_sbt, sizeof(RayGenSbtRecord));
sbt.raygenRecord = d_raygenRecord;
```

**Ray Payload**: used to pass data between programs during ray traversal (page 129). Can be passed in both global memory or stack

**Primitive Attributes**: (same as vertex attributes in the graphics api) Used to pass arbitrary data to
*Any Hit* and *Closest Hit* shaders. The predefined Triangle intersection routine pre-defines The triangle's
barycentric coordinates (U, V).

**Buffer**: Optix's lingo for a `CUdeviceptr`

**Acceleration Structures**: Bounding Volume Hierarchy. There are 2 types

- *Geometry Acceleration Structure*: Built over primitive (triangle, curve, sphere or user-defined)
(called bottom-level acceleration structure in Vulkan)
- *Instance Acceleration Structure*: Built over Motion Transform Nodes (which reference other geometry), allowing for instancing
(called top-level acceleration structure in Vulkan)

**Opacity Micromaps**: records opacity information for a triangle (page 41)

## Traversing the Scene Graph

Optix Defines a scene as a Graph whose nodes are called *Traversables*. There are 5 types of Traversables:

- *Instance Acceleration Structure*
- *Geometry Acceleration Structure* (can only have 1 child of type geometry (page 33))
- *Static Transform* (There can be only one in any path, and their effect is nullified when there are motion transforms?)
- *Matrix Motion Transform*
- *Scaling, rotation, translation (SRT) Motion Transform*
Transform nodes are applied to all children

![Scene Graph Example](resources/scene_graph.png)

## Accessing the OptiX Libary

Accessing any OptiX function is done through `OptixFunctionTable`, which is recovered by looking at some DLL (`nvoptix.dll` windows, `libnvoptix.so.1` on linux). On
linux, the file must be present in the search path, while on windows you can look into the OpenGL related registry values. Anyways, you shouldn't do this yourself, OptiX
ships with the function `optixInit` in the header `optix_stubs.h`, which

- contains the `optixInit` function to ease the loading of the optix library
- Contains inlined function which directly call their counterparts from the global function table

```cpp
inline OptixResult optixDeviceContextCreate( CUcontext fromContext, const OptixDeviceContextOptions* options, OptixDeviceContext* context )
{
    return g_optixFunctionTable.optixDeviceContextCreate( fromContext, options, context );
}
```

## Context (page 21)

Manages a single GPU. Created with `optixDeviceContextCreate` and `optixDeviceContextDestroy`.

We can register a *log callback*, of type

```cpp
void(*OptixLogCallback)(uint32_t level, char const* tag, char const* message, void* data);
```

with the function `optixDeviceContextSetLogCallback`. This function is multithreaded and therefore must be thread-safe.

- TODO: Look in the debugger which thread calls this callback

**Compilation Caching**: When creating an `OptixModule`, `OptixProgramGruop`, `OptixPipeline`, their artifacts will be cached on disk. Functions
to control the cache's behaviour

- `optixDeviceContextSetCacheEnabled`: Enables or disables caching on disk (*Lock the directory if enabled*). Note: The Environment Variable `OPTIX_CACHE_MAXSIZE`,
if set to 0, will effectively disable the cache, overriding this function
- `optixDeviceContextSetCacheLocation`: Sets the directory for the cache. Can be overridden by the environment variable `OPTIX_CACHE_PATH`
- `optixDeviceContextSetCacheDatabaseSizes`: sets low and high watermarks for *disk cache garbage collection*. Whenever, after an entry insertion in the cache, its
size goes beyond the high watermark value, then the library evicts cache entries until the low watermark is reached. The high and low watermark are overridden by
the environment variable `OPTIX_CACHE_MAXSIZE`, which, if set, it is used as high watermark, while the low watermark is set as (high watermark / 2)

**Validation Mode**: Validation Layers are additional, opt-in, controls done in the library routines. They reduce performance but help catching errors during tests
and debug. They are set by enabling them at context creation

```cpp
OptixDeviceContextOptions options = {}
options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
```

Validation mode implicitly adds an Exception Program which reports all exceptions.

## Acceleration Structures (page 25)

There are 5 functions of acceleration structure

- `optixAccelComputeMemoryUsage`: Compute the required memory usage to build an acceleration structure. In particular, it inquires `outputSizeInBytes`, bytes needed by the uncompacted produced by build,
`tempSizeInBytes`, bytes used by the build procedure by Optix as a scratch buffer, and finally `tempUpdateSizeInBytes`, bytes needed in a build update operation
- `optixAccelBuild`: take an array of `OptixBuildInput` and the result from the memory computation usage to perform a streamed build operation
- `optixAccelRelocate`: Copy acceleration structure from a device memory position to another. Useful to propagate the acceleration structure among different devices.
- `optixConvertPointerToTraverableHandle`: The instances in a IAS can have their `CUdeviceptr` be casted into these handles, 64 byte aligned, which can be then traversed

Types:

- `OPTIX_BUILD_INPUT_TYPE_INSANCES`, `OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS`: *Instance Acceleration Structures* (IAS)
- `OPTIX_BUILD_INPUT_TYPE_TRIANGLES`: *Geometry Acceleration Structure* containing triangles (GAS)
- `OPTIX_BUILD_INPUT_TYPE_CURVES`: *Geometry Acceleration Structure* containing curve primitives
- `OPTIX_BUILD_INPUT_TYPE_SPHERES`: *Geometry Acceleration Structure* containing built-in spheres
- `OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES`: *Geometry Acceleration Structure* containing custom primitives

Note: While GAS buffers are independent from input buffers, IAS builds will refer to other IAS/GAS structures and transform nodes

### Triangle Build Input

References an array of vertex buffer for each Motion Blur key (1 vertex buffer if there are no animations), plus an optional, single, Index buffer

```cpp
OptixBuildInputTriangleArray& buildInput = buildInputs[0].triangleArray;
buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
buildInput.vertexBuffers = &d_vertexBuffer;
buildInput.numVertices = numVertices;
buildInput.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
buildInput.vertexStrideInBytes = sizeof(float3);
buildInput.indexBuffer = d_indexBuffer;
buildInput.numIndexTriplets = numTriangles;
buildInput.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
buildInput.indexStrideInBytes = sizeof(int3);
buildInput.preTransform = 0; 
```

where `preTransform` is a **16 bytes aligned, row-major**, 3x4 Transform matrix applied to all vertices at build time.

Each build input maps to one or more consecutive records in the **Shader Binding Table** (SBT), which controls program dispatch (page 77).

```cpp
buildInput.numSbtRecord = 2;
buildInput.sbtIndexOffsetBuffer = d_sbtIndexOffsetBuffer;
buildInput.sbtIndexOffsetSizeInBytes = sizeof(int);
buildInput.sbtIndexOffsetStrideInBytes = sizeof(int);

buildInput.flags = flagsPerSBTRecord;
```

Each build input specifies an array of `OptixGeometryFlags` (`unsigned int`), one for each SBT record. The following flags are supported

- `OPTIX_GEOMETRY_FLAG_NONE`: default behaviour
- `OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL`: By default, the anyhit shader may be called multiple times for each intersected primitive for the sake of more cheap intersection algorithms. Using this
ensures that the anyhit shader is called exactly once for each intersected primitive. There's a performance cost, but it is necessary for *Transparent* objects.
- `OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT`: don't invoke the anyhit shader, even if one or more are present inside the SBT

### Curve Build Input

### Sphere Build Input

### Instance Build Input

### Dynamic Updates, Relocation, Compaction

### Traversable Objects

### Motion Blur

### Opacity MicroMaps (OMM)

## Program Pipeline Creation

- `optixModuleCreate`
- `optixModuleDestroy`
- `optixProgramGroupCreate`
- `optixPipelineCreate`
- `optixPipelineDestroy`
- `optixPipelineSetStackSize`

Programs are compiled into **modules** of type `OptixModule`. One or more modules are combined to create a **program group** (`OptixProgramGroup`), then linked into a **Pipeline** (`OptixPipeline`) on the GPU 
selected with the `OptixContext`.

Each of these three have their own creation function which can fill a compilation/linking log buffer.

Symbols in each `OptixModule` may be unresolved, therefore they may invoke `extern __device__` functions whose definition is elsewhere. Linking errors arise during Pipeline Creation.

The *pipeline* contains all programs required for the ray-tracing algorithm.

Optix programs are either encoded in **OptiX-IR** (proprietary intermediate format), or **PTX**. The former can be created by `nvcc` by passing the `-optix-ir` flag or with `NVRTC` using `nvrtcGetOptiXIR`,
while the latter can be created by `nvcc` with `--ptx` flag or by the `NVRTC` using `nvrtcGetPTX`. OptiX-IR is more reccomended as it provides with more features like symbols debugging 
(while PTX is easier to generate), while PTX is human readable.

There are some flags requirements while compiling modules to be used with OptiX

- at least SM 5.0 `--gpu-architecture=compute_50` (Maxwell architecture)
- `-m64` (64-bit code)
- output type either `--optix-ir` or `--ptx`
- Generate Device Debug symbols (`-G`) is supported only by `--optix-ir`
- enable relocatable device code `-rdc`
- (Optional) `--use_fast_math` will trigger usage of `.approx` math instructions, which are faster but less numerically accurate, and avoids accidental usage of double precision floating points
- To profile code with **Nsight Compute**, enable `--generate-line-info` and specify `debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE` in your `OptixModuleCompileOptions`

*Programming Model*:
