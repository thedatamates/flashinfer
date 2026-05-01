#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer::attention::blackwell::sm120_nvfp4 {

__device__ __forceinline__ bool finite_f32(float x) {
  return x == x && fabsf(x) != INFINITY;
}

__device__ __forceinline__ uint8_t fp32_to_e4m3_byte(float x) {
  if (!(x > 0.0f) || !finite_f32(x)) {
    x = 1.0e-8f;
  }
  __nv_fp8_e4m3 y = static_cast<__nv_fp8_e4m3>(x);
  return y.__x;
}

__device__ __forceinline__ float e4m3_byte_to_fp32(uint8_t x) {
  __nv_fp8_e4m3 y;
  y.__x = x;
  return static_cast<float>(y);
}

__device__ __forceinline__ uint8_t nearest_e2m1_code(float x) {
  constexpr float values[16] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  x = fminf(fmaxf(finite_f32(x) ? x : 0.0f, -6.0f), 6.0f);
  float best_dist = fabsf(x - values[0]);
  uint8_t best = 0;
#pragma unroll
  for (uint8_t i = 1; i < 16; ++i) {
    const float dist = fabsf(x - values[i]);
    if (dist < best_dist) {
      best_dist = dist;
      best = i;
    }
  }
  return best;
}

__global__ void quantize_q_rowmajor_kernel(const __nv_bfloat16* q,
                                           uint8_t* q_packed,
                                           uint8_t* q_scales,
                                           int rows,
                                           int head_dim,
                                           int packed_head_dim,
                                           int scale_cols) {
  const int row = blockIdx.x;
  const int scale_col = threadIdx.x;
  if (row >= rows || scale_col >= scale_cols) {
    return;
  }

  const int base = row * head_dim + scale_col * 16;
  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    max_abs = fmaxf(max_abs, fabsf(__bfloat162float(q[base + i])));
  }

  const uint8_t scale_byte = fp32_to_e4m3_byte(fmaxf(max_abs / 6.0f, 1.0e-8f));
  q_scales[row * scale_cols + scale_col] = scale_byte;
  const float scale = fmaxf(e4m3_byte_to_fp32(scale_byte), 1.0e-8f);

#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const float x0 = __bfloat162float(q[base + 2 * pair]) / scale;
    const float x1 = __bfloat162float(q[base + 2 * pair + 1]) / scale;
    const uint8_t c0 = nearest_e2m1_code(x0);
    const uint8_t c1 = nearest_e2m1_code(x1);
    q_packed[row * packed_head_dim + scale_col * 8 + pair] =
        static_cast<uint8_t>(c0 | (c1 << 4));
  }
}

inline cudaError_t quantize_q_rowmajor_raw(const __nv_bfloat16* q,
                                           uint8_t* q_packed,
                                           uint8_t* q_scales,
                                           int rows,
                                           int head_dim,
                                           cudaStream_t stream) {
  if (rows <= 0 || head_dim <= 0 || (head_dim % 16) != 0) {
    return cudaErrorInvalidValue;
  }
  const int packed_head_dim = head_dim / 2;
  const int scale_cols = head_dim / 16;
  quantize_q_rowmajor_kernel<<<rows, scale_cols, 0, stream>>>(
      q, q_packed, q_scales, rows, head_dim, packed_head_dim, scale_cols);
  return cudaGetLastError();
}

}  // namespace flashinfer::attention::blackwell::sm120_nvfp4
