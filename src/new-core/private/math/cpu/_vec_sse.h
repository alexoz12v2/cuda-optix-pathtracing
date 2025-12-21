#pragma once

#include <cstdint>

namespace dmt::arch {

void add4_f32(float* r, float const* a, float const* b);
void sub4_f32(float* r, float const* a, float const* b);
void mul4_f32(float* r, float const* a, float const* b);
void div4_f32(float* r, float const* a, float const* b);
void max4_f32(float* r, float const* a, float const* b);
void min4_f32(float* r, float const* a, float const* b);
void normalize4_f32(float* r, float const* a);
void abs4_f32(float* r, float const* a);
void sqrt4_f32(float* r, float const* a);
void ceil4_f32(float* r, float const* a);
void floor4_f32(float* r, float const* a);
void fma4_f32(float* r, float const* a, float const* b, float const* c);

void add4_i32(int32_t* r, int32_t const* a, int32_t const* b);
void sub4_i32(int32_t* r, int32_t const* a, int32_t const* b);
void mul4_i32(int32_t* r, int32_t const* a, int32_t const* b);
void div4_i32(int32_t* r, int32_t const* a, int32_t const* b);
void max4_i32(int32_t* r, int32_t const* a, int32_t const* b);
void min4_i32(int32_t* r, int32_t const* a, int32_t const* b);
void abs4_i32(int32_t* r, int32_t const* a);

}  // namespace dmt::arch