#pragma once

#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_quantization.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer::attention::blackwell::sm120_nvfp4 {

struct Sm120Nvfp4PagedKvLoadParams {
  const uint8_t* k_pages = nullptr;
  const uint8_t* k_scales = nullptr;
  const uint8_t* v_pages = nullptr;
  const uint8_t* v_scales = nullptr;
  const int32_t* block_table = nullptr;
  int64_t block_table_stride = 0;
  int64_t k_stride_page = 0;
  int64_t k_stride_dim1 = 0;
  int64_t k_stride_dim2 = 0;
  int64_t k_stride_dim3 = 0;
  int64_t k_scale_stride_page = 0;
  int64_t k_scale_stride_dim1 = 0;
  int64_t k_scale_stride_dim2 = 0;
  int64_t k_scale_stride_dim3 = 0;
  int64_t v_stride_page = 0;
  int64_t v_stride_dim1 = 0;
  int64_t v_stride_dim2 = 0;
  int64_t v_stride_dim3 = 0;
  int64_t v_scale_stride_page = 0;
  int64_t v_scale_stride_dim1 = 0;
  int64_t v_scale_stride_dim2 = 0;
  int64_t v_scale_stride_dim3 = 0;
  int kv_head = 0;
  int page_size = 16;
  int packed_dim = 0;
  int scale_dim = 0;
  int kv_layout_hnd = 0;
  int v_cache_uses_pv_layout = 0;
  int v_scales_trtllm_interleaved = 0;
  float v_global_scale = 6.0f * 448.0f;

  __device__ __forceinline__ bool enabled() const {
    return block_table != nullptr;
  }
};

__device__ __forceinline__ int trtllm_v_scale_offset(int token_offset,
                                                     int scale_col,
                                                     int scale_dim) {
  const int scale_group = scale_dim / 4;
  const int swizzled_token = (token_offset / 4) * 4 + (scale_col / scale_group);
  const int swizzled_scale = (scale_col % scale_group) * 4 + (token_offset % 4);
  return swizzled_token * scale_dim + swizzled_scale;
}

__device__ __forceinline__ uint8_t sm120_nvfp4_paged_k_code(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int logical_token,
    int dim) {
  const int logical_page = logical_token / params.page_size;
  const int page_offset = logical_token - logical_page * params.page_size;
  const int physical_page = params.block_table[logical_page];
  const int packed_col = dim >> 1;
  const int nibble_shift = (dim & 1) * 4;
  const int64_t src =
      params.kv_layout_hnd
          ? (static_cast<int64_t>(physical_page) * params.k_stride_page +
             static_cast<int64_t>(params.kv_head) * params.k_stride_dim1 +
             static_cast<int64_t>(page_offset) * params.k_stride_dim2 +
             static_cast<int64_t>(packed_col) * params.k_stride_dim3)
          : (static_cast<int64_t>(physical_page) * params.k_stride_page +
             static_cast<int64_t>(page_offset) * params.k_stride_dim1 +
             static_cast<int64_t>(params.kv_head) * params.k_stride_dim2 +
             static_cast<int64_t>(packed_col) * params.k_stride_dim3);
  const uint8_t byte = params.k_pages[src];
  return static_cast<uint8_t>((byte >> nibble_shift) & 0x0f);
}

__device__ __forceinline__ uint8_t sm120_nvfp4_paged_k_scale(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int logical_token,
    int scale_col) {
  const int logical_page = logical_token / params.page_size;
  const int page_offset = logical_token - logical_page * params.page_size;
  const int physical_page = params.block_table[logical_page];
  const int64_t src =
      params.kv_layout_hnd
          ? (static_cast<int64_t>(physical_page) * params.k_scale_stride_page +
             static_cast<int64_t>(params.kv_head) * params.k_scale_stride_dim1 +
             static_cast<int64_t>(page_offset) * params.k_scale_stride_dim2 +
             static_cast<int64_t>(scale_col) * params.k_scale_stride_dim3)
          : (static_cast<int64_t>(physical_page) * params.k_scale_stride_page +
             static_cast<int64_t>(page_offset) * params.k_scale_stride_dim1 +
             static_cast<int64_t>(params.kv_head) * params.k_scale_stride_dim2 +
             static_cast<int64_t>(scale_col) * params.k_scale_stride_dim3);
  return params.k_scales[src];
}

__device__ __forceinline__ uint8_t sm120_nvfp4_paged_v_code(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int logical_token,
    int dim) {
  const int logical_page = logical_token / params.page_size;
  const int page_offset = logical_token - logical_page * params.page_size;
  const int physical_page = params.block_table[logical_page];
  const int packed_col = dim >> 1;
  const int nibble_shift = (dim & 1) * 4;
  const int64_t src =
      params.kv_layout_hnd
          ? (static_cast<int64_t>(physical_page) * params.v_stride_page +
             static_cast<int64_t>(params.kv_head) * params.v_stride_dim1 +
             static_cast<int64_t>(page_offset) * params.v_stride_dim2 +
             static_cast<int64_t>(packed_col) * params.v_stride_dim3)
          : (static_cast<int64_t>(physical_page) * params.v_stride_page +
             static_cast<int64_t>(page_offset) * params.v_stride_dim1 +
             static_cast<int64_t>(params.kv_head) * params.v_stride_dim2 +
             static_cast<int64_t>(packed_col) * params.v_stride_dim3);
  const uint8_t byte = params.v_pages[src];
  return static_cast<uint8_t>((byte >> nibble_shift) & 0x0f);
}

__device__ __forceinline__ uint8_t sm120_nvfp4_paged_v_pv_scale(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int logical_page,
    int dim) {
  const int physical_page = params.block_table[logical_page];
  const int scale_row = dim / params.scale_dim;
  const int scale_col = dim - scale_row * params.scale_dim;
  const int64_t src =
      static_cast<int64_t>(physical_page) * params.v_scale_stride_page +
      static_cast<int64_t>(scale_row) * params.v_scale_stride_dim1 +
      static_cast<int64_t>(params.kv_head) * params.v_scale_stride_dim2 +
      static_cast<int64_t>(scale_col) * params.v_scale_stride_dim3;
  return params.v_scales[src];
}

__device__ __forceinline__ uint8_t sm120_nvfp4_paged_v_scale(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int logical_token,
    int scale_col) {
  const int logical_page = logical_token / params.page_size;
  const int page_offset = logical_token - logical_page * params.page_size;
  const int physical_page = params.block_table[logical_page];
  int scale_offset = page_offset * params.scale_dim + scale_col;
  if (params.v_scales_trtllm_interleaved) {
    scale_offset =
        trtllm_v_scale_offset(page_offset, scale_col, params.scale_dim);
  }
  const int scale_t = scale_offset / params.scale_dim;
  const int scale_s = scale_offset - scale_t * params.scale_dim;
  const int64_t src =
      params.kv_layout_hnd
          ? (static_cast<int64_t>(physical_page) * params.v_scale_stride_page +
             static_cast<int64_t>(params.kv_head) *
                 params.v_scale_stride_dim1 +
             static_cast<int64_t>(scale_t) * params.v_scale_stride_dim2 +
             static_cast<int64_t>(scale_s) * params.v_scale_stride_dim3)
          : (static_cast<int64_t>(physical_page) * params.v_scale_stride_page +
             static_cast<int64_t>(scale_t) * params.v_scale_stride_dim1 +
             static_cast<int64_t>(params.kv_head) *
                 params.v_scale_stride_dim2 +
             static_cast<int64_t>(scale_s) * params.v_scale_stride_dim3);
  return params.v_scales[src];
}

__device__ __forceinline__ float sm120_nvfp4_paged_v_standard_value(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int logical_token,
    int dim) {
  const uint8_t code = sm120_nvfp4_paged_v_code(params, logical_token, dim);
  const uint8_t scale =
      sm120_nvfp4_paged_v_scale(params, logical_token, dim >> 4);
  return e2m1_code_to_fp32(code) * e4m3_byte_to_fp32(scale) *
         params.v_global_scale;
}

__device__ __forceinline__ uint8_t sm120_nvfp4_paged_v_standard_pv_scale(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int logical_page,
    int dim,
    int real_kv_len) {
  float max_abs = 0.0f;
#pragma unroll
  for (int t = 0; t < 16; ++t) {
    const int token = logical_page * params.page_size + t;
    const float value =
        token < real_kv_len
            ? sm120_nvfp4_paged_v_standard_value(params, token, dim)
            : 0.0f;
    max_abs = fmaxf(max_abs, fabsf(value));
  }
  return fp32_to_e4m3_byte(
      fmaxf(max_abs / fmaxf(6.0f * params.v_global_scale, 1.0e-20f),
            1.0e-8f));
}

__device__ __forceinline__ uint8_t sm120_nvfp4_paged_v_standard_pv_code(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int logical_token,
    int dim,
    uint8_t pv_scale_byte,
    int real_kv_len) {
  if (logical_token >= real_kv_len) {
    return 0;
  }
  const float value =
      sm120_nvfp4_paged_v_standard_value(params, logical_token, dim);
  const float quant_scale =
      fmaxf(e4m3_byte_to_fp32(pv_scale_byte) * params.v_global_scale, 1.0e-8f);
  return nearest_e2m1_code(value / quant_scale);
}

}  // namespace flashinfer::attention::blackwell::sm120_nvfp4
