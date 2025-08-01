# Cycles / Blender Materials

## Material System

The `Scene` class contains various instances of the `Shader` class

```cpp
class Scene : public NodeOwner {
    // ...
    unique_ptr_vector<Shader> shaders;
    // ...
    Shader* default_surface;
    Shader* default_volume;
    Shader* default_background;
    Shader* default_empty;
    // ...
};
```

Where the `Shader` is basically a pointer type to a `ShaderGraph` plus some metadata related to rendering features selection.
The Shader Graph is a graph like data structure which manages a material specified as a graph of nodes, in which at least the `OutputNode` must be
present. All node classes are contained inside `shader_nodes.h`

### Where is material evaluation inserted in the main render loop (`Session::run_main_render_loop`)

- Rendering Start: `Session::run_main_render_loop -> PathTrace::render -> PathTrace::render_pipeline -> PathTrace::path_trace`

  - Where pipeline outlines all steps made in a render step

- Rendering samples: `PathTrace::path_trace, parallel_for -> PathTraceWork(CPU|GPU)::render_samples, parallel_for -> PathTraceWorkCPU::render_samples_full_pipeline`

  - Render samples full pipeline steps:

    1. `CPUKernels::integrator_init_from_camera`
    2. `CPUKernels::integrator_megakernel` <- CPU uses a megakernel strategy, of course

- Integrator Megakernel (`megakernel.h`):

  ```cpp
  ccl_device void integrator_megakernel(
    KernelGlobals kg, IntegratorState state, ccl_global float *ccl_restrict render_buffer) {
    // while there are kernels queued for execution
    while (true) {
        uint32_t const shadow_queued_kernel = ((&state->shadow)->shadow_path.queued_kernel);
        if (shadow_queued_kernel) { /*handle shadow ray*/ }

        uint32_t const ao_queued_kernel = ((&state->ao)->shadow_path.queued_kernel);
        if (ao_queued_kernel) { /* handle AO paths */ }

        uint32_t const queued_kernel = ((state)->path.queued_kernel);
        if (queued_kernel) { /* handle regular paths */ }
    }
  }
  ```

  The `/* handle regular paths */` has a `switch` followed by a `continue`. Such switch differenciate among all possible kernel works enqueued,
  such as `DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE`, which is enqueued under the following conditions:

  - `DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST` finds an intersection, and enqueues a `DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME` to account for
    media transmission from the ray origin to the intersection position (here MIS sampling for next and trace?)
  - `DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME` volume contribution doesn't kill the ray (ie the radiance is not completely absorbed, just attenuated),
    then, based on material, enqueue a shade surface kernel. Assume the simplest, which is `DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE`

    ```cpp
    // vcxproj: cycles_kernel, file: intersect_closest.h
    if (use_caustics) {
      integrator_path_next_sorted(
          kg, state, current_kernel, DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_MNEE, shader);
    } else if (use_raytrace_kernel) {
      integrator_path_next_sorted(
          kg, state, current_kernel, DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE, shader);
    } else { // assume this
      integrator_path_next_sorted( // this function writes a number on `state->path.queued_kernel`
          kg, state, current_kernel, DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE, shader);
    }
    ```

    Here's the (simplified) shade kernel:

    ```cpp
    template<
      uint node_feature_mask = KERNEL_FEATURE_NODE_MASK_SURFACE & ~KERNEL_FEATURE_NODE_RAYTRACE,
      DeviceKernel current_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE>
    ccl_device_forceinline void integrator_shade_surface(
        KernelGlobals kg, IntegratorState state, ccl_global float *ccl_restrict render_buffer) {

      // write state and say if path tracing needs to continue or not
      const int continue_path_label = integrate_surface<node_feature_mask>(kg, state, render_buffer);
      if (continue_path_label == LABEL_NONE) {
        integrator_path_terminate(kg, state, current_kernel);
        return;
      }

      // continue path tracing
      integrator_shade_surface_next_kernel<current_kernel>(kg, state);
    }
    ```

  - The `integrate_surface` method is the main deal. Its first step, `integrate_surface_shader_setup`, builds up a `ShaderData` structure, a
    ~6KB struct holding information about the **intersection**, and then, if the mesh isn't volume only or if the current ray is not evaluating
    sub surface, we proceed with the `surface_shader_eval<node_feature_mask>(kg, state, &shaderData, render_buffer, state->path.flag))`, which
    enriches the `ShaderData` structure (_inout_)

### `surface_shader_eval`

Note: **shader closures** are building blocks of surface shading that describe how light interacts with a surface.
They are not actual colors or outputs, but rather descriptions of physical light behavior like reflection, refraction, emission, etc.

Therefore, they loosely correspond to the formulas employed by "Green Nodes" (Shaders Nodes) in the shader editor in blender

```cpp
enum ClosureType {
  /* Diffuse */
  CLOSURE_BSDF_DIFFUSE_ID, CLOSURE_BSDF_OREN_NAYAR_ID, CLOSURE_BSDF_BURLEY_ID, CLOSURE_BSDF_DIFFUSE_RAMP_ID, CLOSURE_BSDF_SHEEN_ID,
  CLOSURE_BSDF_DIFFUSE_TOON_ID, CLOSURE_BSDF_TRANSLUCENT_ID,

  /* Glossy */
  CLOSURE_BSDF_PHYSICAL_CONDUCTOR, /* virtual closure */
  CLOSURE_BSDF_F82_CONDUCTOR,      /* virtual closure */
  CLOSURE_BSDF_MICROFACET_GGX_ID, CLOSURE_BSDF_MICROFACET_BECKMANN_ID,
  CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID, /* virtual closure */
  CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID, CLOSURE_BSDF_ASHIKHMIN_VELVET_ID, CLOSURE_BSDF_PHONG_RAMP_ID,
  CLOSURE_BSDF_GLOSSY_TOON_ID, CLOSURE_BSDF_HAIR_REFLECTION_ID,

  /* Transmission */
  CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID, CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID,
  CLOSURE_BSDF_HAIR_TRANSMISSION_ID,

  /* Glass */
  CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID,  /* virtual closure */
  CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID,       /* virtual closure */
  CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID, /* virtual closure */
  CLOSURE_BSDF_HAIR_CHIANG_ID, CLOSURE_BSDF_HAIR_HUANG_ID,

  /* Special cases */
  CLOSURE_BSDF_RAY_PORTAL_ID, CLOSURE_BSDF_TRANSPARENT_ID,

  /* Volume */
  CLOSURE_VOLUME_ID, CLOSURE_VOLUME_ABSORPTION_ID, CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID,
  CLOSURE_VOLUME_MIE_ID, /* virtual closure */
  CLOSURE_VOLUME_FOURNIER_FORAND_ID, CLOSURE_VOLUME_RAYLEIGH_ID, CLOSURE_VOLUME_DRAINE_ID,

  CLOSURE_BSDF_PRINCIPLED_ID,
  /* ... */
};

struct ShaderClosure {
    float3 weight;
    ClosureType type;
    float sample_weight;
    float3 N;
};
```

We'll focus on the following closures

- Diffuse: _Oren-Nayar_
- Glossy: _Microfacet GGX_
- Transmission: _Microfacet GGX_
- Glass: _Microfacet GGX_
- (Maybe) Volume: _Henyey Greenstein_
- (Maybe) Hair (glossy and transmission): _Chiang_

**Going Back to `surface_shader_eval`**: compute remaining number of closures to compute, set them onto the state, and call `svm_eval_nodes`.

```cpp
  sd->num_closure = 0;
  sd->num_closure_left = kg.max_closures; // if not caustics (skipped)
  sd->closure_transparent_extinction = zero_spectrum(); // RGB(0,0,0)

  svm_eval_nodes<node_feature_mask, SHADER_TYPE_SURFACE>(kg, state, sd, buffer, path_flag);
```

Where SVM stands for **Shader Virtual Machine**.

- Shaders in Cycles are built as node graphs (e.g. BSDFs, textures, math operations).
- Rather than compiling every shader into native GPU code, Cycles converts shaders into a bytecode format.
- This bytecode is executed by the Shader Virtual Machine (SVM) in the rendering kernel.

Purpose of the SVM:

- Portability: Same shader code runs across CPU and GPU.
- Efficiency: Avoids compiling hundreds of shader permutations for GPU.
- Compact representation: Uses instructions (like NODE_CLOSURE_BSDF, NODE_TEX_IMAGE, etc.) that are interpreted in this function.

Therefore, the function `svm_eval_nodes` is a loop over nodes, and switches over all possible 16 bytes bytecode instructions.

Focus on the opcode = 2 (NODE_CLOSURE_BSDF), which calls
`svm_node_closure_bsdf<node_feature_mask, SHADER_TYPE_SURFACE>(kd, sd, stack, closure_weight, node, path_flag, offset)`.

- First, unpack BSDF parameters from instruction (`closure_type`)
- If `closure_type` != `CLOSURED_BSDF_PRINCIPLED_ID` and `has_emission`, skip
- Extract parameters from the stack (`N`)

Then there's a switch on the BSDF type. Let us examine the following:

- `CLOSURE_BSDF_DIFFUSE_ID` -> Uses Oren Nayar
- `CLOSURE_BSDF_MICROFACED_GGX_ID` -> Uses a microfacet model with GGX Normal Distribution Function
- `CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID` -> Uses a microfacet model with GGX Normal Distribution Function, assuming refraction
- `CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID` -> variation of ggx model to support glass material
- `CLOSURE_BSDF_PHYSICAL_CONDUCTOR` -> variation of ggx model to support conductor materials

To see how these BSDF objects are constructed, see `closure.h` file

And in the end, it pushes the compiled parametrized BSDF back onto a closure list. Then, a closure is chosen in a function called after
`svm_eval_nodes`, which is `integrate_surface_bsdf_bssrdf_bounce`, which

- `surface_shader_bsdf_bssrdf_pick` chooses a bsdf from the ones we picked in the shader graph
- if not bssdrd and ray portal, call `surface_shader_bsdf_sample_closure`, Which performs **Multiple Importance Sampling**

```cpp
// call site, shade_surface.h
float    bsdf_pdf = 0.0f;
float    unguided_bsdf_pdf = 0.0f;
BsdfEval bsdf_eval{};
float3   bsdf_wo{};
float2   bsdf_sampled_roughness = make_float2(1.0f, 1.0f);
float    bsdf_eta = 1.0f;
float    mis_pdf = 1.0f;
// kg -> kernel globals, sd -> shader data, sc -> shader closure picked, rand_bsdf 3 uniformly distrib f32

int label = surface_shader_bsdf_sample_closure(
  kg, sd, sc, state->path.flag, rand_bsdf, &bsdf_eval, &bsdf_wo, &bsdf_pdf, &bsdf_sampled_roughness, &bsdf_eta);

// caller, surface_hader.h
/* Sample direction for picked BSDF, and return evaluation and pdf for all
 * BSDFs combined using MIS. */
ccl_device int surface_shader_bsdf_sample_closure(KernelGlobals kg,
                                                  ccl_private ShaderData *sd,
                                                  const ccl_private ShaderClosure *sc,
                                                  const int path_flag,
                                                  const float3 rand_bsdf,
                                                  ccl_private BsdfEval *bsdf_eval,
                                                  ccl_private float3 *wo,
                                                  ccl_private float *pdf,
                                                  ccl_private float2 *sampled_roughness,
                                                  ccl_private float *eta)
{
  /* BSSRDF should already have been handled elsewhere. */
  kernel_assert(CLOSURE_IS_BSDF(sc->type));

  int label;
  Spectrum eval = zero_spectrum();

  *pdf = 0.0f;
  label = bsdf_sample(kg, sd, sc, path_flag, rand_bsdf, &eval, wo, pdf, sampled_roughness, eta);

  if (*pdf != 0.0f) {
    bsdf_eval_init(bsdf_eval, sc, *wo, eval * sc->weight);

    if (sd->num_closure > 1) {
      const float sweight = sc->sample_weight;
      *pdf = _surface_shader_bsdf_eval_mis(kg, sd, *wo, sc, bsdf_eval, *pdf * sweight, sweight, 0);
    }
  }
  else {
    bsdf_eval_init(bsdf_eval, zero_spectrum());
  }

  return label;
}

// light_passes.h
ccl_device_inline void bsdf_eval_init(ccl_private BsdfEval *eval, Spectrum value) {
  eval->diffuse = zero_spectrum();
  eval->glossy = zero_spectrum();
  eval->sum = value;
}

/* --------------------------------------------------------------------
 * BSDF Evaluation
 *
 * BSDF evaluation result, split between diffuse and glossy. This is used to
 * accumulate render passes separately. Note that reflection, transmission
 * and volume scattering are written to different render passes, but we assume
 * that only one of those can happen at a bounce, and so do not need to accumulate
 * them separately. */
ccl_device_inline void bsdf_eval_init(ccl_private BsdfEval *eval,
                                      const ccl_private ShaderClosure *sc,
                                      const float3 wo,
                                      Spectrum value) {
  eval->diffuse = zero_spectrum();
  eval->glossy = zero_spectrum();

  if (CLOSURE_IS_BSDF_DIFFUSE(sc->type)) {
    eval->diffuse = value;
  }
  else if (CLOSURE_IS_BSDF_GLOSSY(sc->type)) {
    eval->glossy = value;
  }
  else if (CLOSURE_IS_GLASS(sc->type)) {
    /* Glass can count as glossy or transmission, depending on which side we end up on. */
    if (dot(sc->N, wo) > 0.0f) {
      eval->glossy = value;
    }
  }

  eval->sum = value;
}
```

- `bsdf_sample` is a switch over ShaderClosure::type
- `bsdf_eval_init` maintains 3 waits for glossy, diffuse, sum (weighted), separately. Note how glass is handled
- `_surface_shader_bsdf_eval_mis` loops over all closures
  
  ```cpp
  ccl_device_inline float _surface_shader_bsdf_eval_mis(KernelGlobals kg,
                                                        ccl_private ShaderData *sd,
                                                        const float3 wo,
                                                        const ccl_private ShaderClosure *skip_sc,
                                                        ccl_private BsdfEval *result_eval,
                                                        float sum_pdf,
                                                        float sum_sample_weight,
                                                        const uint light_shader_flags) {
    /* This is the veach one-sample model with balance heuristic,
     * some PDF factors drop out when using balance heuristic weighting. */
    for (int i = 0; i < sd->num_closure; i++) {
      const ccl_private ShaderClosure *sc = &sd->closure[i];

      if (sc == skip_sc) {
        continue;
      }

      if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
        if (CLOSURE_IS_BSDF(sc->type) && !_surface_shader_exclude(sc->type, light_shader_flags)) {
          float bsdf_pdf = 0.0f;
          const Spectrum eval = bsdf_eval(kg, sd, sc, wo, &bsdf_pdf);

          if (bsdf_pdf != 0.0f) {
            bsdf_eval_accum(result_eval, sc, wo, eval * sc->weight);
            sum_pdf += bsdf_pdf * sc->sample_weight;
          }
        }

        sum_sample_weight += sc->sample_weight;
      }
    }

    return (sum_sample_weight > 0.0f) ? sum_pdf / sum_sample_weight : 0.0f;
  }
  ```

  - `bsdf_eval_accum` add last parameter `value` to `sum` and to diffuse or glossy depending on the same switch specified for the `bsdf_eval_init`
  - `bsdf_eval` -> like `bsdf_sample`, switch on all possible closures

### `CLOSURE_BSDF_DIFFUSE_ID` (Oren Nayar)

Look formulas for `bsdf_sample` and `bsdf_eval` for the Oren Nayar branch, present in `bsdf_oern_nayar.h`

### `CLOSURE_BSDF_MICROFACED_GGX_ID`

### `CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID`

### `CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID`

### `CLOSURE_BSDF_PHYSICAL_CONDUCTOR`

## TODO

### Helper functions

- Tranfrom from Global Space to "Tangent Space" (origin = intersection point, +z = outgoing normal, +x = tangent, +y = bitangent, left
  handed, see mesh preview in UE4)

  - all models work in shading space when computing their closure

- Reflection and Refraction functions

- Sample between reflection and refraction in dielectric and conductor

  - sampling with a given normal distribution function for a microfacet model. Example: Oren-Nayar uses gaussian distibuted normals

- Fresnel Equation

  - Dielectric, const real IOR over spectrum (1 float)
  - Conductor, Spectral complex IOR (3 floats for real part, 3 floats for complex part (SSE))

- Microfacet

### Reflection: Torrance Sparrow Model (Conductor)

- given view direction (outgoing direction) and sampled microfacet normal
- reflection is computed with specular reflection law and fresnel equations, which output
  
  1. incident direction
  2. reflection coefficient of the light carried by the path

- scattered light is scaled by masking function computed on the incident direction to account for masking

  - integral change of variables in normal distribution $\omega_m$ -> $\omega_i$, such that you can insert it in the rendering equation,
    which is $\frac{\mathrm{d}\omega_m}{\mathrm{d}\omega_i} = \frac{1}{4 (\omega_o\cdot\omega_m)}$

### Reflection + Transmission: Generalized Torrance Sparrow (Glossy Dielectric Materials)

- given view direction (outgoing direction) and sampled microfacet normal
- compute fresnel reflectance and transmittance given IOR (real)
- compute reflection probability and refraction probabiltiy (a uniformly distributed number is given, `uc`)

  - reflection if `uc < R / (R + T)`

- compute reflect direction and apply torrance sparrow BRDF _if_ reflection chosen

  - handle not same hemisphere -> discard sample and kill ray?

- compute transmission direction and apply generalized torrance sparrow BTDF _if_ transmission chosen

  - handle same hemisphere or internal reflection or // surface -> discard sample and kill ray?
  - change of variables (as before) different formula
    $$\frac{\mathrm{d}\omega_m}{\mathrm{d}\omega_i} =
      \frac{\eta_i^2 |\omega_i\cdot\omega_m|}{((\omega_i\cdot\omega_m)+(\omega_o\cdot\omega_m))/\eta)^2}
    $$
    Knowning that, you can compute the PDF of transmission (given for the single sample monte carlo integration formula)
    $$\int_\mathcal{D} f(x)\mathrm{d}x \approx \frac{f(x_{\mathrm{sample}})}{p(x_{\mathrm{sample}})},\;\;x_{\mathrm{sample}}\in \mathcal{D}$$

  - Evaluate BTDF formula
  - Compensate for non-symmetry of BTDF during transmission $\eta_o^2f_t(p,\omega_o,\omega_i)=\eta_i^2f_t(p,\omega_i,\omega_o)$ (different from BRDF,
    which is commutative on the two directions)

    - means divide by square of relative IOR

### Reflection: Oren-Nayar (Diffuse Dielectric Materials)

Same framework as Conductor, but

- Use Dielectric formulas for Fresnel Reflectance
- Use a Gaussian Distribution as Normal Distribution Function, and sample $\omega_m$ with same procedure
- Compute the Oren-Nayar BRDF
