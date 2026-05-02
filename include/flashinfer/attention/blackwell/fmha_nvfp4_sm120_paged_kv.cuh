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

  __device__ __forceinline__ bool enabled() const {
    return block_table != nullptr;
  }
};

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

__device__ __forceinline__ const uint32_t* sm120_nvfp4_paged_k_word_ptr(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int logical_token,
    int dim) {
  const int logical_page = logical_token / params.page_size;
  const int page_offset = logical_token - logical_page * params.page_size;
  const int physical_page = params.block_table[logical_page];
  const int packed_col = dim >> 1;
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
  return reinterpret_cast<const uint32_t*>(params.k_pages + src);
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

}  // namespace flashinfer::attention::blackwell::sm120_nvfp4
