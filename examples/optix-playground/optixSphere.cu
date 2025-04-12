#include <optix.h>

struct Params
{
    uchar4*                image;
    unsigned int           image_width;
    unsigned int           image_height;
    int                    origin_x;
    int                    origin_y;
    OptixTraversableHandle handle;
};

struct RayGenData
{
    float3 cam_eye;
    float3 camera_u, camera_v, camera_w;
};

struct MissData
{
    float r, g, b;
};

struct HitGroupData
{
    // no data needed
};

extern "C"
{
    __constant__ Params params;
}

static __forceinline__ __device__ void trace(
    OptixTraversableHandle handle,
    float3                 rayOrigin,
    float3                 rayDirection,
    float                  tMin,
    float                  tMax,
    float3*                prd)
{
    unsigned int p0 = __float_as_uint(prd->x), p1 = __float_as_uint(prd->y), p2 = __float_as_uint(prd->z);
    // clang-format off
    optixTrace(handle,
               rayOrigin, rayDirection,
               tMin, tMax,
               0.f /*rayTime*/,
               OptixVisibilityMask(1),
               OPTIX_RAY_FLAG_NONE,
               0 /*SBT offset*/,
               0 /*SBT stride*/,
               0, /*Miss SBT index*/
               p0, p1, p2 // variadic templated payload (handled by __miss__, __closesthit__, __anyhit__, )
    );
    // clang-format on
    prd->x = __uint_as_float(p0);
    prd->y = __uint_as_float(p1);
    prd->z = __uint_as_float(p2);
}

static __forceinline__ __device__ void setPayload(float3 p)
{
    optixSetPayload_0(__float_as_uint(p.x));
    optixSetPayload_1(__float_as_uint(p.y));
    optixSetPayload_2(__float_as_uint(p.z));
}

static __forceinline__ __device__ float3 getPayload()
{
    return make_float3(__uint_as_float(optixGetPayload_0()),
                       __uint_as_float(optixGetPayload_1()),
                       __uint_as_float(optixGetPayload_2()));
}

// TODO define vector device operators for convenience (https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h)
__forceinline__ __device__ float3 operator*(float f, float3 v) { return make_float3(f * v.x, f * v.y, f * v.z); }

__forceinline__ __device__ float3 operator*(float3 v, float f) { return make_float3(f * v.x, f * v.y, f * v.z); }

__forceinline__ __device__ float3 operator+(float3 v, float f) { return make_float3(f + v.x, f + v.y, f + v.z); }

__forceinline__ __device__ float3 operator+(float3 v0, float3 v1)
{
    return make_float3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}

__forceinline__ __device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__forceinline__ __device__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return invLen * v;
}

__forceinline__ __device__ float clamp(float x, float a, float b) { return fmaxf(a, fminf(x, b)); }

__forceinline__ __device__ float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

__forceinline__ __device__ float3 toSRGB(float3 const& c)
{
    float  invGamma = 1.0f / 2.4f;
    float3 powed    = make_float3(powf(c.x, invGamma), powf(c.y, invGamma), powf(c.z, invGamma));
    return make_float3(c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
                       c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
                       c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f);
}

__forceinline__ __device__ unsigned char quantizeUnsigned8Bits(float x)
{
    x = clamp(x, 0.0f, 1.0f);
    enum
    {
        N   = (1 << 8) - 1,
        Np1 = (1 << 8)
    };
    return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

__forceinline__ __device__ uchar4 make_color(float3 const& c)
{
    // first apply gamma, then convert to unsigned char
    float3 srgb = toSRGB(clamp(c, 0.0f, 1.0f));
    return make_uchar4(quantizeUnsigned8Bits(srgb.x), quantizeUnsigned8Bits(srgb.y), quantizeUnsigned8Bits(srgb.z), 255u);
}

__forceinline__ __device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }

__forceinline__ __device__ float3 operator/(float3 v, float f) { return make_float3(v.x / f, v.y / f, v.z / f); }

// __raygen__ is responsible to write generated rays in the shader binding table
extern "C" __global__ void __raygen__rg()
{
    // variables maintained by OptiX which are akin to CUDA's intrinsic grid kernel launch variables, but they track position and
    // dimension within the scope of a ray tracing context. Furthermore, OptiX uses a flat 3D grid, not a hierarchical one
    uint3 const idx = optixGetLaunchIndex();
    uint3 const dim = optixGetLaunchDimensions();

    // retrieve first shader binding table racord for raygen shaders
    RayGenData const* rtData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());

    // Camera Frame
    float3 const U = rtData->camera_u;
    float3 const V = rtData->camera_v;
    float3 const W = rtData->camera_w;

    // camera sample coordinates normalized
    float2 d = make_float2(static_cast<float>(idx.x) / static_cast<float>(dim.x),
                           static_cast<float>(idx.y) / static_cast<float>(dim.y));
    d.x -= 1.f;
    d.y -= 1.f;

    // copute direction and origin of a single ray
    float3 const origin    = rtData->cam_eye;
    float3 const direction = normalize(d.x * U + d.y * V + W);

    float3 payload_rgb = make_float3(0.5f, 0.5f, 0.5f);
    trace(params.handle, origin, direction, 0.f, 1e16f, &payload_rgb);
    params.image[idx.y * params.image_width + idx.x] = make_color(payload_rgb);
}

extern "C" __global__ void __miss__ms()
{
    MissData* rtData  = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    float3    payload = getPayload();
    setPayload(make_float3(rtData->r, rtData->g, rtData->b));
}

extern "C" __global__ void __closesthit__ch()
{
    float tHit = optixGetRayTmax();

    float3 const rayOrig = optixGetWorldRayOrigin();
    float3 const rayDir  = optixGetWorldRayDirection();

    unsigned int const           primIdx     = optixGetPrimitiveIndex();
    OptixTraversableHandle const gas         = optixGetGASTraversableHandle();
    unsigned int const           sbtGASIndex = optixGetSbtGASIndex();

    float4 q;
    optixGetSphereData(gas, primIdx, sbtGASIndex, 0.f, &q);

    float3 const worldRaypos = rayOrig + tHit * rayDir;
    float3 const objRaypos   = optixTransformPointFromWorldToObjectSpace(worldRaypos);
    float3 const objNormal   = (objRaypos - make_float3(q.x, q.y, q.z)) / q.w;
    float3 const worldNormal = normalize(optixTransformNormalFromObjectToWorldSpace(objNormal));

    setPayload(worldNormal * 0.5f + 0.5f);
}
