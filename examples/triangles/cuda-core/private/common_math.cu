#include "common_math.cuh"

__host__ __device__ Transform::Transform() {
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      mInv[i * 4 + j] = 0.f;
      m[i * 4 + j] = 0.f;
    }
  }
  for (int j = 0; j < 4; ++j) {
    mInv[j * 4 + j] = 1.f;
    m[j * 4 + j] = 1.f;
  }
}

__host__ __device__ Transform::Transform(float const* _m) {
  for (int i = 0; i < 16; ++i) {
    m[i] = _m[i];
  }

  mInv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
            m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];

  mInv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
            m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];

  mInv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
            m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];

  mInv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
             m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

  mInv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
            m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];

  mInv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
            m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];

  mInv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
            m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];

  mInv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
             m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

  mInv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
            m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];

  mInv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
            m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];

  mInv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
             m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];

  mInv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
             m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

  mInv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
            m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];

  mInv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
            m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];

  mInv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
             m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];

  mInv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
             m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

  float det =
      m[0] * mInv[0] + m[1] * mInv[4] + m[2] * mInv[8] + m[3] * mInv[12];

  // assert(det != 0.0f); // non-singular assumption

  float mInvDet = 1.0f / det;
  for (float& v : mInv) {
    v *= mInvDet;
  }
}

__host__ __device__ float3 Transform::applyDirection(float3 v) const {
  float const x = m[0] * v.x + m[4] * v.y + m[8] * v.z;
  float const y = m[1] * v.x + m[5] * v.y + m[9] * v.z;
  float const z = m[2] * v.x + m[6] * v.y + m[10] * v.z;
  return make_float3(x, y, z);
}

// Point transform (w = 1)
__host__ __device__ float3 Transform::apply(float3 p) const {
  float x = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
  float y = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
  float z = m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14];
  float w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];

  if (w != 1.0f && w != 0.0f) {
    float invW = 1.0f / w;
    x *= invW;
    y *= invW;
    z *= invW;
  }

  return make_float3(x, y, z);
}

__host__ __device__ float3 Transform::applyInverse(float3 p) const {
  float x = mInv[0] * p.x + mInv[4] * p.y + mInv[8] * p.z + mInv[12];
  float y = mInv[1] * p.x + mInv[5] * p.y + mInv[9] * p.z + mInv[13];
  float z = mInv[2] * p.x + mInv[6] * p.y + mInv[10] * p.z + mInv[14];
  float w = mInv[3] * p.x + mInv[7] * p.y + mInv[11] * p.z + mInv[15];

  if (w != 1.0f && w != 0.0f) {
    float invW = 1.0f / w;
    x *= invW;
    y *= invW;
    z *= invW;
  }

  return make_float3(x, y, z);
}

__host__ __device__ float3 Transform::applyTranspose(float3 n) const {
  return make_float3(m[0] * n.x + m[1] * n.y + m[2] * n.z,
                     m[4] * n.x + m[5] * n.y + m[6] * n.z,
                     m[8] * n.x + m[9] * n.y + m[10] * n.z);
}

__host__ __device__ float3 Transform::applyInverseTranspose(float3 n) const {
  return make_float3(mInv[0] * n.x + mInv[1] * n.y + mInv[2] * n.z,
                     mInv[4] * n.x + mInv[5] * n.y + mInv[6] * n.z,
                     mInv[8] * n.x + mInv[9] * n.y + mInv[10] * n.z);
}
