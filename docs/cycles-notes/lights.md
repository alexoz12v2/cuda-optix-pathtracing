# Light Sources Notes

## What we need to do

- All light functions are implemented in terms of their own local coordinate space

- To sample lights, we need to know the "light sample context"

  - current position we are in
  - Shading and Geoemtric Normal

- A light sample contains

  - radiance
  - ray connecting interection point and light sampled point
  - PDF for position, PDF for direction

    - Note: for non area lights, PDF cam be computed starting from a ray only, not starting from an intersection and direction

To take a "Light sample" (ie sample incident radiance coming from the light) means choosing a direction upon which sampling direct illlumination
which is proportional to the distribution depending on the light source, as it may be that it illuminates only along certain directions coming from
certain positions

For each light model, we need to be able to

- Sample Li, incoming radiance to a point for direct illumination estimation
- (BONUS) compute the PDF for the Li Sample
- Compute the Power (Radiant Flux) of a given light for power-based light sampling

### Light Modeling (Point Light)

Emitted light from a point light at distance $r$ from the illuminated point is computed by:

$$
L_e = \frac{I_e}{\pi r^2}
$$

Where the constant $\frac{1}{\pi}$ can be tweaked. Note that attenuation cannot be modified without losing physical accuracy, if it matters. Otherwise,
if we prioritize the visual result, the attenuation function of $\frac{1}{r^2}$ can be arbitrarely tweaked

### During the Path Tracing Routine

- If there are no intersections, then we should choose an infinite light (if more than one, use power heuristic, but we assume one), and
  add its Emitted radiance contribution to the Total Radiance $L$ accumulated by theray contribution to the $\beta$ weighted by the Power Heuristic and 
  $\beta$

    ```cpp
    if (!intersection)
    {
        foreach (infiniteLight) {
            Le = //...
            if (depth == 0 || lastBounceSpecular)
                L += beta * Le;
            else
            {
                // compute MIS weights
                L += beta * w_b * Le;
            }
        }
    }
    else 
    {
        if (primitive has emission)
        {
            // same formula as infinite light, but Le the area light
        }

        if (no specular bounces so bar) regularize BSDF
        if (max depth reached) break
        if (BSDF not specular)
        {
            sample direct illumination
            L += beta * Ld;
        }
        // BSDF sampling, update state variables, spawn new ray, perform russian roulette
    }
    ```

## Light Sampling Theory

References:

- PBRT (book and source code)
- Cycles (source code)
- <https://psychopath.io/post/2020_04_20_light_trees>

In Radiometry, holds the principle of superposition of effects, allowing us to compute contributions from each light to a point on a surface
separately. **If lights are too many, how should we proceed?**

Usually a path tracer goes like this when you compute lighting on a surface (for each sample)

- sample lights in the scene to compute **direct lighting**
- sample an indirect ray to compute **indirect lighting**

Here we are focusing on the **direct lighting** part, which requires us to choose from all the lights in the scene

- Using all of them gives us a good result but can be taxing
- Sampling one light works, but it's noisy (even if you select light appropriately, eg constructing a PMF over their emitting radiance flux, which
  is what `PowerLightSampler` in PBRT does)

What we want to achieve is to sample the lights in proportion to their contribution in the **current shading point**. Therefore, we need a spatial
data structure, which gets us to **Light Trees** (Called `BVHLightSampler` in PBRT)

## In PBRT (Light Tree)

- For how the `PathIntegrator` manages MIS sampling and direct illumination see above or on code. Here we focus on **Light Trees**
- First paper on Light Trees (also called *Lightcuts*): <https://www.graphics.cornell.edu/~bjw/lightcuts.pdf>
- powerpoint siggraph 2005 on lightcuts: <https://www.graphics.cornell.edu/~bjw/lightcutSigTalk05.pdf>
- paper of the light tree implemented by cycles: <https://fpsunflower.github.io/ckulla/data/many-lights-hpg2018.pdf>

See Light Trees notes

## In Cycles (Direct Illumination and Light Trees)

Remember the call hierarchy of rendering in cycles, which starts from the rendering thread in the `Session` class

- `Session::thread_render` -> `Session::run_main_render_loop` -> `PathTrace::render(renderWork)` -> `PathTrace::render_pipeline(renderWork)`
  -> `PathTrace::path_trace(renderWork)`: parallel for with
- `PathTraceWorkCPU::render_samples(stats, startSampleIdx, numSamples, sampleOffset)`: TBB parallel execute for each pixel in tile
- `PathTraceWorkCPU::render_samples_full_pipeline(kernelGlobals, workTile, sampleNum)` -> `integrator_megakernel(kernelGlobals, state, renderBuffer)`

The megakernel itself is a loop which continues until there are items in the execution queue to process. The work items can be of three types,
checked in the following order

- Shadow Kernels
- AO Kernels (ignore those)
- Path Kernels

Among the "Normal" Path Kernels there are also the intersection routines to process BVH intersections with a surface and account for partecipating media
scattering/absorption

There are 3 main kernel on which we'll focus

- (shadow) `DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW`
- (shadow) `DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW`
- (path) `DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT`

### Who pushes our kernels of interest onto the state?

Apparently, shadow paths are inserted when calling `integrate_direct_light_shadow_init_common`, which appear after intersecting a surface, as detailed
later.

To comfortably search for work item insertion in the megakernel implementation, you can search for call to the `integrator_path_next()` function.

- If the hit primitive in `integrator_intersect_next_kernel` is a `PRIMITIVE_LAMP`, then enqueue `DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT`, otherwise
  handle normal surface interaction

    - Note: Primitive Lamp = An object which has an area light attached to it, but not a BSDF. PBRT handles primitives which posses both (probably cycles
        can also do that, but `PRIMITIVE_LAMP` is an exception)

- `shadow_linking_intersect` and `shadow_linking_schedule_intersection_kernel` are the equivalent of `SampleLd` in PBRT,
  ie checks if there's a need to evaluate direct illumination, which apparently cycles calls "Dedicated Light". Therefore, this function pushes 

  - `DEVICE_KERNEL_INTEGRATOR_INTERSECT_DEDICATED_LIGHT`
  - `DEVICE_KERNEL_INTEGRATOR_SHADE_DEDICATED_LIGHT`

- Apparently, the function `integratotr_shade_light` itself can also schedule an additional `DEVICE_KERNEL_INTEGRATOR_SHADE_DEDICATED_LIGHT`. Don't care
  for now

Let us focus on "Shadow Linking".

- During `shade_surface.h:integrator_shade_surface`, executed when a `DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE` is scheduled,
  The `shadow_linking_schedule_intersection_kernel` is called when processing a material, after roussian roulette, before BSDF estimation for next bounce

    ```cpp
    template<uint node_feature_mask = KERNEL_FEATURE_NODE_MASK_SURFACE & ~KERNEL_FEATURE_NODE_RAYTRACE,
             DeviceKernel current_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE>
    ccl_device_forceinline void integrator_shade_surface(KernelGlobals kg,
                                                         IntegratorState state,
                                                         ccl_global float *ccl_restrict render_buffer)
    {
      const int continue_path_label = integrate_surface<node_feature_mask>(kg, state, render_buffer);
      if (continue_path_label == LABEL_NONE) {
        integrator_path_terminate(kg, state, current_kernel);
        return;
      }
    
    #ifdef __SHADOW_LINKING__
      /* No need to cast shadow linking rays at a transparent bounce: the lights will be accumulated
       * via the main path in this case. */
      if ((continue_path_label & LABEL_TRANSPARENT) == 0) {
        if (shadow_linking_schedule_intersection_kernel<current_kernel>(kg, state)) {
          return;
        }
      }
    #endif
    
      integrator_shade_surface_next_kernel<current_kernel>(kg, state);
    }
    ```

- `intersect_dedicated_light.h:shadow_linking_intersect` is called by `intersect_dedicated_light.h:integrator_intersect_dedicated_light()`, and 
  therefore only if `DEVICE_KERNEL_INTEGRATOR_INTERSECT_DEDICATED_LIGHT` is scheduled and executed.

  Such kernel is scheduled only in `shadow_linking_schedule_intersection_kernel`

This means that everything starts 

- either from an intersection with a primitive with a light but no material
- or from a surface intersection with a need for direct lighting estimation (done for not completely transparent materials only)

Therefore, skipping the edge case of "Primitive Lamp" surfaces, "Shadow Linking" starts in `integrator_intersect_dedicated_light` when direct lighting
estimation is requested for a surface. Such method calls `shadow_linking_intersect`, which recovers the ray object from the state, including
the intersected primitive, which is kept track of to avoid **Self Intersections**, and calls `shadow_linking_pick_light_intersection`

```cpp
Ray ray{}; intergrator_state_read_ray(state, &ray);
ray.self.prim = state->isect.prim;
ray.self.object = state->isect.object;

Intersection isect{};
if (!shadow_linking_pick_light_intersection(kg, state, &ray, &isect))
    return false; // no light hit. No need for extra shadow ray for direct light

// copy primitives needed by main path (so that you can restore state for main path)
shadow_linking_store_last_primitives(state);

// write intersection result in state
integrator_state_write_isect(state, &isect);

integrator_path_next(kg, state, /*current*/DEVICE_KERNEL_INTEGRATOR_INTERSECT_DEDICATED_LIGHT, /*next*/DEVICE_KERNEL_INTEGRATOR_SHADE_DEDICATED_LIGHT);
return true;
```

We are interested in `shadow_linking_pick_light_intersection`, which basically does 2 things (We assume we have a sampler for shadow ray directions)

- try to intersect an emissive surface

  - sample 1 or more emissive meshes and tag them as chosen for the next shade linked light execution (ie assigned to **Shadow set**)

- try to intersect a non-primitive light source

  - for each light which is part of the **Shadow set** (depends on the "cast shadows" tickbox), first call `light_link_light_match` to see if 
    the ight and receiver are eligible for direct illumination, then, once you sampled a shadow ray direction (done before), call a type-specific
    intersection routine, eg `point_light_intersect`

Let us first look into `light_link_light_match(kg, receiver, light)`, which checks if light can cast shadows and receiver can receive radiance from
the given light (light set membership). **We don't care about this detail**. Once this method is finished we know the **nearest light intersection**

- `point_light_intersect` does a ray-sphere intersection algorithm between the shadow ray and the maximum radius of influence of the point light
  *which cam be computed by its given radiant intensity by using a threashold considered as a meaningful radiance contribution* (or conversely,
  give radius and get radiant intensity out, assuming quadratic decay governed by a *scale*)

At the end of its execution, `shadow_linking_pick_light_intersection` registers into the state `state->shadow_link.dedicated_light_weight = num_hits`

The **Next Kernel** to be executed in shadow linking is `integrator_shade_dedicated_light`:

```cpp
shadow_linking_shade(kg, state, render_buffer);
// restore self-intersection primtives in the main state and then return in the intersect_closest state with the indirect ray
```

Create an 326 bytes emission storage and, if the intersection is a light (again, `PRIMITIVE_LAMP`) then call `shadow_linking_shade_light`,
otherwise call `shadow_linking_shade_surface_emission`

- These 2 function output `bsdf_spectrum`, `mis_weight`, `light_group` (ignore the last one) and write the emission shader data.
- Then, if `bsdf_spectrum` is not zero, create a shadow ray (`shadow_linking_setup_ray_from_intersection`) and 
  **branch off the shadow kernel** by calling `integrate_direct_shadow_init_common`, which

    - Inserts `DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW` in the shadow queue, which was just initialized, and writes some parameters to it, such as
      the number of max bounces, `pass_diffuse_weight` and `pass_glossy_weight` copied from the integrator state into the shadow state

Before moving into the shadow kernel, let us see how cycles samples a `PRIMITIVE_LAMP` in `shadow_linking_shade_light`

- `shadow_linking_light_sample_from_intersection` to create a `LightSample` struct, if nothing sampled return `false`
- `light_sample_shader_eval` to compute spectrum from sample (if there is a light sample). if zero spectrum return `false`
- if not `is_light_shader_visible_to_path` return `false`
- compute MIS weight (`shadow_linking_light_sample_mis_weight`) and multiply the evaluated light spectrum by it to get the final `bsdf_spectrum`

More in detail

- **`shadow_linking_light_sample_from_intersection`**, calls a method for lights at infinity if that's the light we sampled,
  or another method if we sampled a light with a finite position (`distant_light_sample_from_intersection` vs `light_sample_from_intersection`).
  Assume finite position

  - `light_sample_from_intersection` calls a type-specific sampling method. assume we selected only 1 point light, `point_light_sample_from_intersection`
    will be called
  - `point_light_sample_from_intersection`: 

    - normal of light sample = difference between receiver point and center of the light `ray_P - klight->co`
    - PDF for MIS: equal to `sphere_light_pdf(distanceSquared, radiusSquared, N, rayD, ...)`
    - Texture coordinates (why?) applies the inverse transpose of the inverse transform matrix of hte light to the sampled direction
      (**get to light local space**), and maps them to spherical coordinates (discard radius == 1) and then convert them to
      barycentric coordinates: `u = uv.y` and `v = 1.f - uv.x - uv.y`
    - note: always returns `true`

- **`light_sample_shader_eval`**: Assume the considered light doesn't use a shader graph, hence its parameters are constant, therefore the code will
  first call and return `true` from `surface_shader_constant_expression`, and then return its spectrum from `klight->strength` after multiplying the 
  returned light spectrum by `ls.eval_fac_`. Some details about `surface_shader_constant_expression` (note that `ls.Ng` gets reversed if *dot(ng, d) > 0*)

  - return constant emission from point light struct 

- **`shadow_linking_light_sample_mis_weight`**: again, this function differs depending on whether you sampled a distant light or a finite position one.
  assume a finite position one, and therefore `light_sample_mis_weight_forward_lamp` (vs `light_sample_mis_weight_forward_distant`)

  - take a constant weight from the state called `state->path.mis_ray_pdf`, used later for final weight computation (`light_sample_mis_weight_forward`)
  - Assume the integrator uses **Light Trees**
    
    ```cpp
    float pdf = lightSample->pdf;
    pdf *= light_tree_pdf(kg, P, staet->path.mis_origin_n, state->ray.previous_dt, path_flag, 0, light_to_tree[ls->lamp], state->path.mis_ray_object);
    ```
