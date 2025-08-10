# Texture Notes

The fundamental steps to be able to sample a texture to apply it to a surface occupying a patch of the screen space are

- estimate texture sampling rate -> *Avoid Aliasing Artifacts* (eg. Moire Pattern)

    - To be done by estimating differential of intersection points with respect to change in the texture coordinates $\frac{\del p}{\del u}$, $\frac{\del p}{\del v}$
    - Estimating them for camera rays uses Ray Differentials or an approximation of ray differentials (first with respect to x and y in image plane space flipped, then uv)
    - estimate such differentials for indirect rays (Reflection Transmission) (**LATER**, chapter 10.1.3), where only specular reflection/transmission can use
      true reflected/refracted ray differentials, while diffuse reflection/trasmission again tries to approximate that

    We'll go with an approximation only approach

- Downsample the texture to whatever sampling rate you need (only a problem if the texture's source comes from an image, otherwise, if it's a function, no big deal)

    - Texture Filtering for images proceeds by pre-generating all downsampled versions of the initial image texture (**MipMapping**), then using some kind of algorithm
      (easiest is Trilinear filtering, hardest is EWA) to sample the texture

- Furthermore, we need a way to generate texture coordinates

    - pre-specified mapping which computes $(u,v)$ from object-space or world-space coordinates
    - pre-specified texture coordinates inside the model itself (UV Maps)

