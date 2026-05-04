#pragma once

#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_quantization.cuh>

#include <cuda_bf16.h>
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
  int v_scale_layout = 0;
  const __nv_bfloat16* q_bf16 = nullptr;
  int64_t q_stride_token = 0;
  int64_t q_stride_head = 0;
  int64_t q_stride_dim = 0;
  int64_t q_stride_row = 0;
  int q_is_3d = 1;

  __device__ __forceinline__ bool enabled() const {
    return block_table != nullptr;
  }
};

__device__ __forceinline__ float sm120_nvfp4_paged_q_bf16_value(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int q_begin,
    int local_row,
    int group_size,
    int num_kv_heads,
    bool all_kv_heads,
    int dim) {
  const int token_offset = local_row / group_size;
  const int group_offset = local_row - token_offset * group_size;
  if (params.q_is_3d) {
    const int head =
        (all_kv_heads ? params.kv_head * group_size : 0) + group_offset;
    const int token = q_begin + token_offset;
    const int64_t src =
        static_cast<int64_t>(token) * params.q_stride_token +
        static_cast<int64_t>(head) * params.q_stride_head +
        static_cast<int64_t>(dim) * params.q_stride_dim;
    return __bfloat162float(params.q_bf16[src]);
  }
  const int row =
      all_kv_heads
          ? (q_begin + token_offset) * (num_kv_heads * group_size) +
                params.kv_head * group_size + group_offset
          : q_begin * group_size + local_row;
  const int64_t src = static_cast<int64_t>(row) * params.q_stride_row +
                      static_cast<int64_t>(dim) * params.q_stride_dim;
  return __bfloat162float(params.q_bf16[src]);
}

__device__ __forceinline__ int64_t sm120_nvfp4_paged_q_bf16_row_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int q_begin,
    int local_row,
    int group_size,
    int num_kv_heads,
    bool all_kv_heads) {
  const int token_offset = local_row / group_size;
  const int group_offset = local_row - token_offset * group_size;
  if (params.q_is_3d) {
    const int head =
        (all_kv_heads ? params.kv_head * group_size : 0) + group_offset;
    const int token = q_begin + token_offset;
    return static_cast<int64_t>(token) * params.q_stride_token +
           static_cast<int64_t>(head) * params.q_stride_head;
  }
  const int row =
      all_kv_heads
          ? (q_begin + token_offset) * (num_kv_heads * group_size) +
                params.kv_head * group_size + group_offset
          : q_begin * group_size + local_row;
  return static_cast<int64_t>(row) * params.q_stride_row;
}

__device__ __forceinline__ float sm120_nvfp4_paged_q_bf16_value_from_row_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int64_t row_base,
    int dim) {
  const int64_t src =
      row_base + static_cast<int64_t>(dim) * params.q_stride_dim;
  return __bfloat162float(params.q_bf16[src]);
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

__device__ __forceinline__ int64_t sm120_nvfp4_paged_k_scale_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int physical_page) {
  return params.kv_layout_hnd
             ? (static_cast<int64_t>(physical_page) * params.k_scale_stride_page +
                static_cast<int64_t>(params.kv_head) * params.k_scale_stride_dim1)
             : (static_cast<int64_t>(physical_page) * params.k_scale_stride_page +
                static_cast<int64_t>(params.kv_head) * params.k_scale_stride_dim2);
}

__device__ __forceinline__ uint8_t sm120_nvfp4_paged_k_scale_from_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int64_t scale_page_base,
    int page_offset,
    int scale_col) {
  const int64_t src =
      scale_page_base +
      static_cast<int64_t>(page_offset) *
          (params.kv_layout_hnd ? params.k_scale_stride_dim2
                                : params.k_scale_stride_dim1) +
      static_cast<int64_t>(scale_col) * params.k_scale_stride_dim3;
  return params.k_scales[src];
}

__device__ __forceinline__ int sm120_nvfp4_linear_scale_token(
    int token, int scale_col, int scale_dim, int scale_layout) {
  if (scale_layout == 0) {
    const int scale_group = scale_dim / 4;
    return (token / 4) * 4 + (scale_col / scale_group);
  }
  return token;
}

__device__ __forceinline__ int sm120_nvfp4_linear_scale_col(
    int token, int scale_col, int scale_dim, int scale_layout) {
  if (scale_layout == 0) {
    const int scale_group = scale_dim / 4;
    return (scale_col % scale_group) * 4 + (token & 3);
  }
  return scale_col;
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
      static_cast<int64_t>(physical_page) * params.v_stride_page +
      static_cast<int64_t>(page_offset) * params.v_stride_dim1 +
      static_cast<int64_t>(params.kv_head) * params.v_stride_dim2 +
      static_cast<int64_t>(packed_col) * params.v_stride_dim3;
  const uint8_t byte = params.v_pages[src];
  return static_cast<uint8_t>((byte >> nibble_shift) & 0x0f);
}

__device__ __forceinline__ int64_t sm120_nvfp4_paged_v_data_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int physical_page) {
  return params.kv_layout_hnd
             ? (static_cast<int64_t>(physical_page) * params.v_stride_page +
                static_cast<int64_t>(params.kv_head) * params.v_stride_dim1)
             : (static_cast<int64_t>(physical_page) * params.v_stride_page +
                static_cast<int64_t>(params.kv_head) * params.v_stride_dim2);
}

__device__ __forceinline__ int64_t sm120_nvfp4_paged_v_scale_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int physical_page) {
  return params.kv_layout_hnd
             ? (static_cast<int64_t>(physical_page) * params.v_scale_stride_page +
                static_cast<int64_t>(params.kv_head) * params.v_scale_stride_dim1)
             : (static_cast<int64_t>(physical_page) * params.v_scale_stride_page +
                static_cast<int64_t>(params.kv_head) * params.v_scale_stride_dim2);
}

__device__ __forceinline__ uint8_t sm120_nvfp4_paged_v_code_pair_from_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int64_t page_base,
    int page_offset,
    int packed_col) {
  const int64_t src =
      page_base +
      static_cast<int64_t>(page_offset) *
          (params.kv_layout_hnd ? params.v_stride_dim2 : params.v_stride_dim1) +
      static_cast<int64_t>(packed_col) * params.v_stride_dim3;
  return params.v_pages[src];
}

__device__ __forceinline__ uint32_t sm120_nvfp4_paged_v_word_from_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int64_t page_base,
    int page_offset,
    int packed_col) {
  const int64_t src =
      page_base +
      static_cast<int64_t>(page_offset) *
          (params.kv_layout_hnd ? params.v_stride_dim2 : params.v_stride_dim1) +
      static_cast<int64_t>(packed_col) * params.v_stride_dim3;
  return *reinterpret_cast<const uint32_t*>(params.v_pages + src);
}

static __device__ __noinline__ uint32_t sm120_nvfp4_linear_v_requant_transposed_word(
    uint32_t row_word,
    uint32_t row_scale_byte,
    uint8_t output_scale_byte,
    unsigned subgroup_mask,
    int subgroup_base_lane,
    int dim_lane) {
  const float output_scale =
      fmaxf(e4m3_byte_to_fp32(output_scale_byte), 1.0e-8f);
  const float inv_output_scale = 1.0f / output_scale;
  float vals[8];
#pragma unroll
  for (int src_lane = 0; src_lane < 8; ++src_lane) {
    const uint32_t peer_word =
        __shfl_sync(subgroup_mask, row_word, subgroup_base_lane + src_lane);
    const uint32_t peer_scale_byte =
        __shfl_sync(subgroup_mask, row_scale_byte,
                    subgroup_base_lane + src_lane) & 0xffu;
    const uint8_t code =
        static_cast<uint8_t>((peer_word >> (4 * dim_lane)) & 0x0fu);
    vals[src_lane] =
        e2m1_code_to_fp32(code) *
        e4m3_byte_to_fp32(static_cast<uint8_t>(peer_scale_byte)) *
        inv_output_scale;
  }
  return static_cast<uint32_t>(fp32_pair_to_e2m1_byte(vals[0], vals[1])) |
         (static_cast<uint32_t>(fp32_pair_to_e2m1_byte(vals[2], vals[3]))
          << 8) |
         (static_cast<uint32_t>(fp32_pair_to_e2m1_byte(vals[4], vals[5]))
          << 16) |
         (static_cast<uint32_t>(fp32_pair_to_e2m1_byte(vals[6], vals[7]))
          << 24);
}

__device__ __forceinline__ uint8_t sm120_nvfp4_paged_v_code_from_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int64_t page_base,
    int page_offset,
    int dim) {
  const uint8_t packed = sm120_nvfp4_paged_v_code_pair_from_page_base(
      params, page_base, page_offset, dim >> 1);
  return static_cast<uint8_t>((packed >> ((dim & 1) * 4)) & 0x0f);
}

__device__ __forceinline__ uint8_t sm120_nvfp4_paged_v_linear_code_pair(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int logical_token,
    int packed_col) {
  const int logical_page = logical_token / params.page_size;
  const int page_offset = logical_token - logical_page * params.page_size;
  const int physical_page = params.block_table[logical_page];
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
  return params.v_pages[src];
}

__device__ __forceinline__ uint8_t sm120_nvfp4_paged_v_linear_scale_from_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int64_t scale_page_base,
    int page_offset,
    int scale_col) {
  const int stored_token = sm120_nvfp4_linear_scale_token(
      page_offset, scale_col, params.scale_dim, params.v_scale_layout);
  const int stored_scale_col = sm120_nvfp4_linear_scale_col(
      page_offset, scale_col, params.scale_dim, params.v_scale_layout);
  const int64_t src =
      scale_page_base +
      static_cast<int64_t>(stored_token) *
          (params.kv_layout_hnd ? params.v_scale_stride_dim2
                                : params.v_scale_stride_dim1) +
      static_cast<int64_t>(stored_scale_col) * params.v_scale_stride_dim3;
  return params.v_scales[src];
}

__device__ __forceinline__ uint8_t sm120_nvfp4_paged_v_linear_scale(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int logical_token,
    int scale_col) {
  const int logical_page = logical_token / params.page_size;
  const int page_offset = logical_token - logical_page * params.page_size;
  const int physical_page = params.block_table[logical_page];
  const int stored_token = sm120_nvfp4_linear_scale_token(
      page_offset, scale_col, params.scale_dim, params.v_scale_layout);
  const int stored_scale_col = sm120_nvfp4_linear_scale_col(
      page_offset, scale_col, params.scale_dim, params.v_scale_layout);
  const int64_t src =
      params.kv_layout_hnd
          ? (static_cast<int64_t>(physical_page) * params.v_scale_stride_page +
             static_cast<int64_t>(params.kv_head) * params.v_scale_stride_dim1 +
             static_cast<int64_t>(stored_token) * params.v_scale_stride_dim2 +
             static_cast<int64_t>(stored_scale_col) * params.v_scale_stride_dim3)
          : (static_cast<int64_t>(physical_page) * params.v_scale_stride_page +
             static_cast<int64_t>(stored_token) * params.v_scale_stride_dim1 +
             static_cast<int64_t>(params.kv_head) * params.v_scale_stride_dim2 +
             static_cast<int64_t>(stored_scale_col) * params.v_scale_stride_dim3);
  return params.v_scales[src];
}

__device__ __forceinline__ float sm120_nvfp4_paged_v_linear_value_from_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int64_t data_page_base,
    int64_t scale_page_base,
    int page_offset,
    int dim) {
  const int packed_col = dim >> 1;
  const uint8_t packed = sm120_nvfp4_paged_v_code_pair_from_page_base(
      params, data_page_base, page_offset, packed_col);
  const uint8_t code =
      static_cast<uint8_t>((packed >> ((dim & 1) * 4)) & 0x0f);
  const uint8_t scale_byte =
      sm120_nvfp4_paged_v_linear_scale_from_page_base(
          params, scale_page_base, page_offset, dim >> 4);
  return e2m1_code_to_fp32(code) * e4m3_byte_to_fp32(scale_byte);
}

__device__ __forceinline__ float sm120_nvfp4_paged_v_linear_value(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int logical_token,
    int dim) {
  const int packed_col = dim >> 1;
  const uint8_t packed =
      sm120_nvfp4_paged_v_linear_code_pair(params, logical_token, packed_col);
  const uint8_t code =
      static_cast<uint8_t>((packed >> ((dim & 1) * 4)) & 0x0f);
  const uint8_t scale_byte =
      sm120_nvfp4_paged_v_linear_scale(params, logical_token, dim >> 4);
  return e2m1_code_to_fp32(code) * e4m3_byte_to_fp32(scale_byte);
}

__device__ __forceinline__ uint8_t sm120_nvfp4_paged_v_pv_scale_from_physical_page(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int physical_page,
    int dim) {
  const int scale_row = dim / params.scale_dim;
  const int scale_col = dim - scale_row * params.scale_dim;
  const int64_t src =
      static_cast<int64_t>(physical_page) * params.v_scale_stride_page +
      static_cast<int64_t>(scale_row) * params.v_scale_stride_dim1 +
      static_cast<int64_t>(params.kv_head) * params.v_scale_stride_dim2 +
      static_cast<int64_t>(scale_col) * params.v_scale_stride_dim3;
  return params.v_scales[src];
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
