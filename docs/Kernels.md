    //start 
    copy the parameters of the scene to the global  memory
    value the AOS SOA for the mesh data.
    ---------------------------------------
    kernel: build the BVH -> tree binary -> rebuild to groups the leaf

    Mitchel filter -> constant memory 
                   -> Initialization on CPU
                   -> Function device: Sample 
                   -> Function device: Evaluate
                   -> Function device: PDF
    
    camera differiential rays -> CPU it is enough
    
    kernel: build the light tree
    
    HaltonOwen -> global memory -> states: index, dim each warp/thread that generates the numbers

    Reset HaltonOwen for the current sample
    Get a point on the current pixel and the weight -> getCameraSample
    Generate ray

    Initialize RGB: the radiance L and the beta (throughput)  

    main loop bsdf -> check the intersection ray bvh -> no intersection -> check light -> no light/envLight -> black
            -> yep -> importance sampling (4 and 11 cap pbrt) -> 1 bounce
                   -> retrieve the info about the triagle, material, transform from model to world sys, application of the barycnetric interpolation, compute dv, du in Texture space and transform to the frame space and stimate the dx, dy least squares.
                   -> estimate the hit error, the error could bring inner the surface, so is useful place the point on the surface of the sphere (radious error) along the normal estimated through the intersection.
                   -> Choose a random ligth -> shadow ray casts and verify the intersection through the bvh and evaluate the the light contribution.
                   -> once choose the light get the PMFlight(pdfLight) and the sample on the light( pdf eval.pdf) 
                   -> Kernel: material sample -> kernel foreach material
                   -> check the intersection ray bvh
                   -> update state: beta->montecarlo
                   -> trasmission inner the surface
                   -> russian roulette
    
    RGBFilm on device -> non-caching load
                      -> add the sample
    
    loop Opengl
    Kernel for reading the buffer# Kernels
    ---------------------------------------

least square method (differental "context" for texture sampling)

- differentials minimum ray differential / raster - (pos, dir)
- differentials intersection point / uv
-> differentials intersetion point / raster
