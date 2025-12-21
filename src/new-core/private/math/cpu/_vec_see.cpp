#include "_vec_sse.h"

#include <immintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>

namespace dmt::arch {

void add4_f32(float* r, float const* a, float const* b) {
  __m128 const va = _mm_loadu_ps(a);
  __m128 const vb = _mm_loadu_ps(b);
  __m128 const vc = _mm_add_ps(va, vb);
  _mm_storeu_ps(r, vc);
}
void sub4_f32(float* r, float const* a, float const* b) {
  __m128 const va = _mm_loadu_ps(a);
  __m128 const vb = _mm_loadu_ps(b);
  __m128 const vc = _mm_sub_ps(va, vb);
  _mm_storeu_ps(r, vc);
}
void mul4_f32(float* r, float const* a, float const* b) {
  __m128 const va = _mm_loadu_ps(a);
  __m128 const vb = _mm_loadu_ps(b);
  __m128 const vc = _mm_mul_ps(va, vb);
  _mm_storeu_ps(r, vc);
}
void div4_f32(float* r, float const* a, float const* b) {
  __m128 const va = _mm_loadu_ps(a);
  __m128 const vb = _mm_loadu_ps(b);
  __m128 const vc = _mm_div_ps(va, vb);
  _mm_storeu_ps(r, vc);
}
void max4_f32(float* r, float const* a, float const* b) {
  __m128 const va = _mm_loadu_ps(a);
  __m128 const vb = _mm_loadu_ps(b);
  __m128 const vc = _mm_max_ps(va, vb);
  _mm_storeu_ps(r, vc);
}
void min4_f32(float* r, float const* a, float const* b) {
  __m128 const va = _mm_loadu_ps(a);
  __m128 const vb = _mm_loadu_ps(b);
  __m128 const vc = _mm_min_ps(va, vb);
  _mm_storeu_ps(r, vc);
}
void normalize4_f32(float* r, float const* a) {
  __m128 v = _mm_loadu_ps(a);
  v = _mm_div_ps(_mm_loadu_ps(a), _mm_rsqrt_ps(v));
  _mm_storeu_ps(r, v);
}
void abs4_f32(float* r, float const* a) {
  __m128 const vval = _mm_loadu_ps(a);
  __m128 const vsign = _mm_set1_ps(-0);
  __m128 const vres = _mm_andnot_ps(vsign, vval);
  _mm_storeu_ps(r, vres);
}
void sqrt4_f32(float* r, float const* a) {
  __m128 v = _mm_loadu_ps(a);
  v = _mm_sqrt_ps(v);
  _mm_storeu_ps(r, v);
}
void ceil4_f32(float* r, float const* a) {
  __m128 v = _mm_loadu_ps(a);
  v = _mm_ceil_ps(v);
  _mm_storeu_ps(r, v);
}
void floor4_f32(float* r, float const* a) {
  __m128 v = _mm_loadu_ps(a);
  v = _mm_floor_ps(v);
  _mm_storeu_ps(r, v);
}
void fma4_f32(float* r, float const* a, float const* b, float const* c) {
  __m128 const vmult0 = _mm_loadu_ps(a);
  __m128 const vmult1 = _mm_loadu_ps(b);
  __m128 const vadd = _mm_loadu_ps(c);
  __m128 const vres = _mm_fmadd_ps(vmult0, vmult1, vadd);
  _mm_storeu_ps(r, vres);
}

void add4_i32(int32_t* r, int32_t const* a, int32_t const* b) {
  __m128i const va = _mm_loadu_epi32(a);
  __m128i const vb = _mm_loadu_epi32(b);
  __m128i const vc = _mm_add_epi32(va, vb);
  _mm_storeu_epi32(r, vc);
}
void sub4_i32(int32_t* r, int32_t const* a, int32_t const* b) {
  __m128i const va = _mm_loadu_epi32(a);
  __m128i const vb = _mm_loadu_epi32(b);
  __m128i const vc = _mm_sub_epi32(va, vb);
  _mm_storeu_epi32(r, vc);
}
void mul4_i32(int32_t* r, int32_t const* a, int32_t const* b) {
  __m128i const va = _mm_loadu_epi32(a);
  __m128i const vb = _mm_loadu_epi32(b);
  __m128i const vc = _mm_mul_epi32(va, vb);
  _mm_storeu_epi32(r, vc);
}
void div4_i32(int32_t* r, int32_t const* a, int32_t const* b) {
  __m128i const va = _mm_loadu_epi32(a);
  __m128i const vb = _mm_loadu_epi32(b);
  __m128i const vc = _mm_div_epi32(va, vb);
  _mm_storeu_epi32(r, vc);
}
void max4_i32(int32_t* r, int32_t const* a, int32_t const* b) {
  __m128i const va = _mm_loadu_epi32(a);
  __m128i const vb = _mm_loadu_epi32(b);
  __m128i const vc = _mm_max_epi32(va, vb);
  _mm_storeu_epi32(r, vc);
}
void min4_i32(int32_t* r, int32_t const* a, int32_t const* b) {
  __m128i const va = _mm_loadu_epi32(a);
  __m128i const vb = _mm_loadu_epi32(b);
  __m128i const vc = _mm_min_epi32(va, vb);
  _mm_storeu_epi32(r, vc);
}
void abs4_i32(int32_t* r, int32_t const* a) {
  __m128i const v = _mm_loadu_epi32(a);
  __m128i const vres = _mm_abs_epi32(v);
  _mm_storeu_epi32(r, vres);
}

}  // namespace dmt::arch
