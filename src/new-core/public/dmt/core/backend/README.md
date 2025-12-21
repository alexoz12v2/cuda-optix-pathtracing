# Backend

this is where CPU/GPU specific headers for data structures live. In particular, it's reserved for the
backend-specific (where backend = host or device) representation of the scene, different from the common parsed format

```c++
// backend/cpu/cpu_scene.h
struct CpuScene {
  CpuBVH8           bvh;
  CpuTextureCache   textures;
  CpuLightArray     lights;
};
CpuScene build_cpu_scene(const scene::Scene&);

// backend/cuda/cuda_scene.h
struct CudaScene {
  DeviceBVH         bvh;
  DeviceLightSOA    lights;
  DeviceTextures    textures;
};
CudaScene build_cuda_scene(const scene::Scene&);
```

Sometimes CPU can use parsed representation directly, and that's fine, hence it can copy the struct
