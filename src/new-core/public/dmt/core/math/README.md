# math

Math module containing generic (_not_ rendering specific) mathematical functions and types.

Each implementation of these should be architecture specific, and the only place in which we use macros for conditional
compilation are the headers

- this prohibits us to use `__host__ __device__` functions, but to specify two distinct functions and implement them into
  different translation units

What about shared functions? _public header has an inline shim implementation which delegates a backend specific one_

```c++
// math/mat4.h
#pragma once

struct Mat4 {
  float m[16];
};

struct Mat4Impl {
  static Mat4 mul_host(const Mat4&, const Mat4&);
  static Mat4 mul_device(const Mat4&, const Mat4&);
};

__host__ __device__
inline Mat4 operator*(const Mat4& a, const Mat4& b) {
#if defined(__CUDA_ARCH__)
  return Mat4Impl::mul_device(a, b);
#else
  return Mat4Impl::mul_host(a, b);
#endif
}
```

And in the host-side implementation you can further forward to another implementation if you need to specialize by ISA

```c++
// math/cpu/mat4_mul.cpp
#include "math/mat4.h"
// include ISA specific implementation here here

Mat4 Mat4Impl::mul_host(const Mat4& a, const Mat4& b) {
#if defined(PT_USE_AVX)
  // AVX implementation
#elif defined(PT_USE_SSE)
  // SSE implementation
#else
  // scalar
#endif
}
```

Example layout:

```text
pathtracer/
  public/math/
    vec4.h
    mat4.h
    quat.h
    ray.h
    aabb.h
    math_constants.h

  private/math/
    cpu/
      vec4_scalar.cpp
      vec4_sse.cpp
      vec4_avx.cpp

    cuda/
      vec4.cu
```
