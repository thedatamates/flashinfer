#pragma once

#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_quantization.cuh>
#include <flashinfer/mma.cuh>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#ifndef FLASHINFER_SM120_NVFP4_DEBUG_TRAPS
#define FLASHINFER_SM120_NVFP4_DEBUG_TRAPS 0
#endif

#if FLASHINFER_SM120_NVFP4_DEBUG_TRAPS
#define SM120_NVFP4_DEBUG_TRAP() asm volatile("trap;\n")
#else
#define SM120_NVFP4_DEBUG_TRAP() ((void)0)
#endif

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
  const uint8_t* v_linear_scale_cache = nullptr;
  int v_linear_scale_cache_groups = 0;
  int64_t v_linear_scale_cache_head_stride = 0;
  int64_t v_linear_scale_cache_batch_stride = 0;
  const uint8_t* v_linear_data_cache = nullptr;
  int64_t v_linear_data_cache_head_stride = 0;
  int64_t v_linear_data_cache_batch_stride = 0;
  int64_t v_linear_data_cache_physical_kv_len = 0;

  __device__ __forceinline__ bool enabled() const {
    return block_table != nullptr;
  }
};

__device__ __forceinline__ int64_t sm120_nvfp4_paged_v_data_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int physical_page);

__device__ __forceinline__ int64_t sm120_nvfp4_paged_v_scale_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int physical_page);

__device__ __forceinline__ uint8_t sm120_nvfp4_paged_v_code_pair_from_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int64_t page_base,
    int page_offset,
    int packed_col);

__device__ __forceinline__ uint8_t sm120_nvfp4_paged_v_linear_scale_from_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int64_t scale_page_base,
    int page_offset,
    int scale_col);

template <bool kCoalescedLayout>
__device__ __forceinline__ uint8_t sm120_nvfp4_linear_v_scale_cache_load(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int batch_idx,
    int kv_head,
    int dim,
    int token_group) {
  int64_t idx =
      static_cast<int64_t>(batch_idx) * params.v_linear_scale_cache_batch_stride +
      static_cast<int64_t>(kv_head) * params.v_linear_scale_cache_head_stride;
  if constexpr (kCoalescedLayout) {
    idx += static_cast<int64_t>(token_group) * params.packed_dim * 2 + dim;
  } else {
    idx += static_cast<int64_t>(dim) * params.v_linear_scale_cache_groups +
           token_group;
  }
  return params.v_linear_scale_cache[idx];
}

template <bool kCoalescedLayout>
__device__ __forceinline__ uint32_t sm120_nvfp4_linear_v_data_cache_word(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int batch_idx,
    int kv_head,
    int token,
    int packed_col) {
#if FLASHINFER_SM120_NVFP4_DEBUG_TRAPS
  if ((packed_col & 3) != 0) {
    SM120_NVFP4_DEBUG_TRAP();
  }
#endif
  int64_t idx =
      static_cast<int64_t>(batch_idx) * params.v_linear_data_cache_batch_stride +
      static_cast<int64_t>(kv_head) * params.v_linear_data_cache_head_stride;
  if constexpr (kCoalescedLayout) {
    const int word_col = packed_col >> 2;
    idx +=
        (static_cast<int64_t>(word_col) *
             params.v_linear_data_cache_physical_kv_len +
         token) *
        4;
  } else {
    idx += static_cast<int64_t>(token) * params.packed_dim + packed_col;
  }
  return *reinterpret_cast<const uint32_t*>(params.v_linear_data_cache + idx);
}

template <bool kCoalescedLayout>
static __global__ void sm120_nvfp4_linear_v_scale_cache_kernel(
    Sm120Nvfp4PagedKvLoadParams params,
    uint8_t* cache,
    const int32_t* kv_lens,
    int batch_size,
    int num_kv_heads,
    int head_dim,
    int physical_kv_len) {
  const int token_groups = physical_kv_len / 16;
  const int dim_pairs = head_dim / 2;
  const int64_t total =
      static_cast<int64_t>(batch_size) * num_kv_heads * dim_pairs *
      token_groups;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t t = idx;
    int token_group;
    int dim_pair;
    if constexpr (kCoalescedLayout) {
      dim_pair = static_cast<int>(t % dim_pairs);
      t /= dim_pairs;
      token_group = static_cast<int>(t % token_groups);
      t /= token_groups;
    } else {
      token_group = static_cast<int>(t % token_groups);
      t /= token_groups;
      dim_pair = static_cast<int>(t % dim_pairs);
      t /= dim_pairs;
    }
    const int kv_head = static_cast<int>(t % num_kv_heads);
    const int batch_idx = static_cast<int>(t / num_kv_heads);

    const int token_group_start = token_group * 16;
    const int kv_len_tokens =
        kv_lens != nullptr ? kv_lens[batch_idx] : physical_kv_len;
    uint8_t sf0 = 0x38u;
    uint8_t sf1 = 0x38u;
    if (token_group_start < kv_len_tokens) {
      Sm120Nvfp4PagedKvLoadParams local = params;
      local.block_table =
          params.block_table +
          static_cast<int64_t>(batch_idx) * params.block_table_stride;
      local.kv_head = kv_head;
      const int logical_page = token_group;
      const int physical_page = local.block_table[logical_page];
      const int64_t data_page_base =
          sm120_nvfp4_paged_v_data_page_base(local, physical_page);
      const int64_t scale_page_base =
          sm120_nvfp4_paged_v_scale_page_base(local, physical_page);
      const int dim0 = dim_pair * 2;
      const int packed_col = dim0 >> 1;
      const int scale_col0 = dim0 >> 4;
      const int scale_col1 = (dim0 + 1) >> 4;
      const int shift0 = (dim0 & 1) * 4;
      const int shift1 = ((dim0 + 1) & 1) * 4;
      float max_abs0 = 0.0f;
      float max_abs1 = 0.0f;
#pragma unroll 1
      for (int offset = 0; offset < 16; ++offset) {
        const int token = token_group_start + offset;
        if (token < kv_len_tokens) {
          const uint8_t packed =
              sm120_nvfp4_paged_v_code_pair_from_page_base(
                  local, data_page_base, offset, packed_col);
          const uint8_t scale_byte0 =
              sm120_nvfp4_paged_v_linear_scale_from_page_base(
                  local, scale_page_base, offset, scale_col0);
          const uint8_t scale_byte1 =
              scale_col1 == scale_col0
                  ? scale_byte0
                  : sm120_nvfp4_paged_v_linear_scale_from_page_base(
                        local, scale_page_base, offset, scale_col1);
          const float scale0 = e4m3_byte_to_fp32(scale_byte0);
          const float scale1 = e4m3_byte_to_fp32(scale_byte1);
          const float val0 =
              e2m1_code_to_fp32(
                  static_cast<uint8_t>((packed >> shift0) & 0x0f)) *
              scale0;
          const float val1 =
              e2m1_code_to_fp32(
                  static_cast<uint8_t>((packed >> shift1) & 0x0f)) *
              scale1;
          max_abs0 = fmaxf(max_abs0, fabsf(val0));
          max_abs1 = fmaxf(max_abs1, fabsf(val1));
        }
      }
      sf0 = fp32_to_e4m3_byte(max_abs0 > 0.0f ? max_abs0 / 6.0f : 1.0f);
      sf1 = fp32_to_e4m3_byte(max_abs1 > 0.0f ? max_abs1 / 6.0f : 1.0f);
    }

    const int64_t head_base =
        static_cast<int64_t>(batch_idx) *
            static_cast<int64_t>(num_kv_heads) * head_dim * token_groups +
        static_cast<int64_t>(kv_head) * head_dim * token_groups;
    const int64_t cache_base =
        kCoalescedLayout
            ? head_base + static_cast<int64_t>(token_group) * head_dim +
                  dim_pair * 2
            : head_base + static_cast<int64_t>(dim_pair * 2) * token_groups +
                  token_group;
    cache[cache_base] = sf0;
    cache[cache_base + (kCoalescedLayout ? 1 : token_groups)] = sf1;
  }
}

inline size_t sm120_nvfp4_linear_v_scale_cache_bytes(
    int batch_size,
    int num_kv_heads,
    int head_dim,
    int physical_kv_len) {
  return static_cast<size_t>(batch_size) * static_cast<size_t>(num_kv_heads) *
         static_cast<size_t>(head_dim) *
         static_cast<size_t>(physical_kv_len / 16);
}

inline cudaError_t sm120_nvfp4_prepare_linear_v_scale_cache(
    Sm120Nvfp4PagedKvLoadParams& params,
    uint8_t* cache,
    const int32_t* kv_lens,
    int batch_size,
    int num_kv_heads,
    int head_dim,
    int physical_kv_len,
    cudaStream_t stream) {
  const int token_groups = physical_kv_len / 16;
  params.v_linear_scale_cache = cache;
  params.v_linear_scale_cache_groups = token_groups;
  params.v_linear_scale_cache_head_stride =
      static_cast<int64_t>(head_dim) * token_groups;
  params.v_linear_scale_cache_batch_stride =
      static_cast<int64_t>(num_kv_heads) *
      params.v_linear_scale_cache_head_stride;
  const int64_t total_pairs =
      static_cast<int64_t>(batch_size) * num_kv_heads * (head_dim / 2) *
      token_groups;
  const bool coalesced_layout = (head_dim == 256);
  constexpr int kThreads = 256;
  int blocks = static_cast<int>((total_pairs + kThreads - 1) / kThreads);
  if (blocks > 65535) {
    blocks = 65535;
  }
  if (coalesced_layout) {
    sm120_nvfp4_linear_v_scale_cache_kernel<true>
        <<<blocks, kThreads, 0, stream>>>(params, cache, kv_lens, batch_size,
                                          num_kv_heads, head_dim,
                                          physical_kv_len);
  } else {
    sm120_nvfp4_linear_v_scale_cache_kernel<false>
        <<<blocks, kThreads, 0, stream>>>(params, cache, kv_lens, batch_size,
                                          num_kv_heads, head_dim,
                                          physical_kv_len);
  }
  return cudaGetLastError();
}

template <bool kCoalescedLayout>
static __global__ void sm120_nvfp4_linear_v_data_cache_kernel(
    Sm120Nvfp4PagedKvLoadParams params,
    uint8_t* cache,
    const int32_t* kv_lens,
    int batch_size,
    int num_kv_heads,
    int head_dim,
    int physical_kv_len) {
  const int packed_dim = head_dim / 2;
  const int packed_word_cols = packed_dim / 4;
  const int64_t total =
      static_cast<int64_t>(batch_size) * num_kv_heads * packed_word_cols *
      physical_kv_len * 4;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t t = idx;
    int token;
    int packed_col;
    if constexpr (kCoalescedLayout) {
      const int byte_in_word = static_cast<int>(t & 3);
      t >>= 2;
      token = static_cast<int>(t % physical_kv_len);
      t /= physical_kv_len;
      const int packed_word_col = static_cast<int>(t % packed_word_cols);
      t /= packed_word_cols;
      packed_col = packed_word_col * 4 + byte_in_word;
    } else {
      packed_col = static_cast<int>(t % packed_dim);
      t /= packed_dim;
      token = static_cast<int>(t % physical_kv_len);
      t /= physical_kv_len;
    }
    const int kv_head = static_cast<int>(t % num_kv_heads);
    const int batch_idx = static_cast<int>(t / num_kv_heads);

    uint8_t packed_out = 0u;
    const int kv_len_tokens =
        kv_lens != nullptr ? kv_lens[batch_idx] : physical_kv_len;
    if (token < kv_len_tokens) {
      Sm120Nvfp4PagedKvLoadParams local = params;
      local.block_table =
          params.block_table +
          static_cast<int64_t>(batch_idx) * params.block_table_stride;
      local.kv_head = kv_head;
      const int logical_page = token >> 4;
      const int page_offset = token & 15;
      const int physical_page = local.block_table[logical_page];
      const int64_t data_page_base =
          sm120_nvfp4_paged_v_data_page_base(local, physical_page);
      const int64_t scale_page_base =
          sm120_nvfp4_paged_v_scale_page_base(local, physical_page);
      const int dim0 = packed_col * 2;
      const int scale_col = dim0 >> 4;
      const uint8_t packed_in =
          sm120_nvfp4_paged_v_code_pair_from_page_base(
              local, data_page_base, page_offset, packed_col);
      const uint8_t input_scale_byte =
          sm120_nvfp4_paged_v_linear_scale_from_page_base(
              local, scale_page_base, page_offset, scale_col);
      const float input_scale = e4m3_byte_to_fp32(input_scale_byte);
      const int token_group = token >> 4;
      const uint8_t sf0_byte =
          sm120_nvfp4_linear_v_scale_cache_load<kCoalescedLayout>(
              params, batch_idx, kv_head, dim0, token_group);
      const uint8_t sf1_byte =
          sm120_nvfp4_linear_v_scale_cache_load<kCoalescedLayout>(
              params, batch_idx, kv_head, dim0 + 1, token_group);
      const float inv_sf0 = 1.0f / fmaxf(e4m3_byte_to_fp32(sf0_byte), 1.0e-8f);
      const float inv_sf1 = 1.0f / fmaxf(e4m3_byte_to_fp32(sf1_byte), 1.0e-8f);
      const float x0 =
          e2m1_code_to_fp32(static_cast<uint8_t>(packed_in & 0x0fu)) *
          input_scale * inv_sf0;
      const float x1 =
          e2m1_code_to_fp32(static_cast<uint8_t>((packed_in >> 4) & 0x0fu)) *
          input_scale * inv_sf1;
      packed_out = fp32_pair_to_e2m1_byte(x0, x1);
    }
    cache[idx] = packed_out;
  }
}

inline size_t sm120_nvfp4_linear_v_data_cache_bytes(
    int batch_size,
    int num_kv_heads,
    int head_dim,
    int physical_kv_len) {
  return static_cast<size_t>(batch_size) * static_cast<size_t>(num_kv_heads) *
         static_cast<size_t>(physical_kv_len) *
         static_cast<size_t>(head_dim / 2);
}

inline cudaError_t sm120_nvfp4_prepare_linear_v_data_cache(
    Sm120Nvfp4PagedKvLoadParams& params,
    uint8_t* cache,
    const int32_t* kv_lens,
    int batch_size,
    int num_kv_heads,
    int head_dim,
    int physical_kv_len,
    cudaStream_t stream) {
  const int packed_dim = head_dim / 2;
  const bool coalesced_layout = (head_dim == 256);
  if ((packed_dim & 3) != 0) {
    return cudaErrorInvalidValue;
  }
  params.v_linear_data_cache = cache;
  params.v_linear_data_cache_physical_kv_len = physical_kv_len;
  params.v_linear_data_cache_head_stride =
      static_cast<int64_t>(physical_kv_len) * packed_dim;
  params.v_linear_data_cache_batch_stride =
      static_cast<int64_t>(num_kv_heads) *
      params.v_linear_data_cache_head_stride;
  const int64_t total =
      static_cast<int64_t>(batch_size) * num_kv_heads * physical_kv_len *
      packed_dim;
  constexpr int kThreads = 256;
  int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  if (blocks > 65535) {
    blocks = 65535;
  }
  if (coalesced_layout) {
    sm120_nvfp4_linear_v_data_cache_kernel<true>
        <<<blocks, kThreads, 0, stream>>>(params, cache, kv_lens, batch_size,
                                          num_kv_heads, head_dim,
                                          physical_kv_len);
  } else {
    sm120_nvfp4_linear_v_data_cache_kernel<false>
        <<<blocks, kThreads, 0, stream>>>(params, cache, kv_lens, batch_size,
                                          num_kv_heads, head_dim,
                                          physical_kv_len);
  }
  return cudaGetLastError();
}

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
    int kv_head,
    bool all_kv_heads) {
  const int token_offset = local_row / group_size;
  const int group_offset = local_row - token_offset * group_size;
  if (params.q_is_3d) {
    const int head =
        (all_kv_heads ? kv_head * group_size : 0) + group_offset;
    const int token = q_begin + token_offset;
    return static_cast<int64_t>(token) * params.q_stride_token +
           static_cast<int64_t>(head) * params.q_stride_head;
  }
  const int row =
      all_kv_heads
          ? (q_begin + token_offset) * (num_kv_heads * group_size) +
                kv_head * group_size + group_offset
          : q_begin * group_size + local_row;
  return static_cast<int64_t>(row) * params.q_stride_row;
}

__device__ __forceinline__ int64_t sm120_nvfp4_paged_q_bf16_row_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int q_begin,
    int local_row,
    int group_size,
    int num_kv_heads,
    bool all_kv_heads) {
  return sm120_nvfp4_paged_q_bf16_row_base(
      params, q_begin, local_row, group_size, num_kv_heads, params.kv_head,
      all_kv_heads);
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
    const int32_t* block_table,
    int kv_head,
    int logical_token,
    int dim) {
  const int logical_page = logical_token / params.page_size;
  const int page_offset = logical_token - logical_page * params.page_size;
  const int physical_page = block_table[logical_page];
  const int packed_col = dim >> 1;
  const int64_t src =
      params.kv_layout_hnd
          ? (static_cast<int64_t>(physical_page) * params.k_stride_page +
             static_cast<int64_t>(kv_head) * params.k_stride_dim1 +
             static_cast<int64_t>(page_offset) * params.k_stride_dim2 +
             static_cast<int64_t>(packed_col) * params.k_stride_dim3)
          : (static_cast<int64_t>(physical_page) * params.k_stride_page +
             static_cast<int64_t>(page_offset) * params.k_stride_dim1 +
             static_cast<int64_t>(kv_head) * params.k_stride_dim2 +
             static_cast<int64_t>(packed_col) * params.k_stride_dim3);
  return reinterpret_cast<const uint32_t*>(params.k_pages + src);
}

__device__ __forceinline__ int64_t sm120_nvfp4_paged_k_data_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int kv_head,
    int physical_page) {
  return params.kv_layout_hnd
             ? (static_cast<int64_t>(physical_page) * params.k_stride_page +
                static_cast<int64_t>(kv_head) * params.k_stride_dim1)
             : (static_cast<int64_t>(physical_page) * params.k_stride_page +
                static_cast<int64_t>(kv_head) * params.k_stride_dim2);
}

__device__ __forceinline__ const uint32_t* sm120_nvfp4_paged_k_word_ptr_from_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int64_t page_base,
    int page_offset,
    int dim) {
  const int packed_col = dim >> 1;
  const int64_t src =
      page_base +
      static_cast<int64_t>(page_offset) *
          (params.kv_layout_hnd ? params.k_stride_dim2
                                : params.k_stride_dim1) +
      static_cast<int64_t>(packed_col) * params.k_stride_dim3;
  return reinterpret_cast<const uint32_t*>(params.k_pages + src);
}

__device__ __forceinline__ const uint32_t* sm120_nvfp4_paged_k_word_ptr(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int logical_token,
    int dim) {
  return sm120_nvfp4_paged_k_word_ptr(params, params.block_table,
                                      params.kv_head, logical_token, dim);
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
    int kv_head,
    int physical_page) {
  return params.kv_layout_hnd
             ? (static_cast<int64_t>(physical_page) * params.k_scale_stride_page +
                static_cast<int64_t>(kv_head) * params.k_scale_stride_dim1)
             : (static_cast<int64_t>(physical_page) * params.k_scale_stride_page +
                static_cast<int64_t>(kv_head) * params.k_scale_stride_dim2);
}

__device__ __forceinline__ int64_t sm120_nvfp4_paged_k_scale_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int physical_page) {
  return sm120_nvfp4_paged_k_scale_page_base(params, params.kv_head,
                                             physical_page);
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

__device__ __forceinline__ const uint32_t*
sm120_nvfp4_paged_k_scale_word_ptr_from_page_base(
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
  return reinterpret_cast<const uint32_t*>(params.k_scales + src);
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
    int kv_head,
    int physical_page) {
  return params.kv_layout_hnd
             ? (static_cast<int64_t>(physical_page) * params.v_stride_page +
                static_cast<int64_t>(kv_head) * params.v_stride_dim1)
             : (static_cast<int64_t>(physical_page) * params.v_stride_page +
                static_cast<int64_t>(kv_head) * params.v_stride_dim2);
}

__device__ __forceinline__ int64_t sm120_nvfp4_paged_v_data_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int physical_page) {
  return sm120_nvfp4_paged_v_data_page_base(params, params.kv_head,
                                            physical_page);
}

__device__ __forceinline__ int64_t sm120_nvfp4_paged_v_scale_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int kv_head,
    int physical_page) {
  return params.kv_layout_hnd
             ? (static_cast<int64_t>(physical_page) * params.v_scale_stride_page +
                static_cast<int64_t>(kv_head) * params.v_scale_stride_dim1)
             : (static_cast<int64_t>(physical_page) * params.v_scale_stride_page +
                static_cast<int64_t>(kv_head) * params.v_scale_stride_dim2);
}

__device__ __forceinline__ int64_t sm120_nvfp4_paged_v_scale_page_base(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int physical_page) {
  return sm120_nvfp4_paged_v_scale_page_base(params, params.kv_head,
                                             physical_page);
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
  return flashinfer::mma::float8_to_e2m1x8(vals[0], vals[1], vals[2], vals[3],
                                           vals[4], vals[5], vals[6], vals[7]);
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
    int kv_head,
    int physical_page,
    int dim) {
  const int scale_row = dim / params.scale_dim;
  const int scale_col = dim - scale_row * params.scale_dim;
  const int64_t src =
      static_cast<int64_t>(physical_page) * params.v_scale_stride_page +
      static_cast<int64_t>(scale_row) * params.v_scale_stride_dim1 +
      static_cast<int64_t>(kv_head) * params.v_scale_stride_dim2 +
      static_cast<int64_t>(scale_col) * params.v_scale_stride_dim3;
  return params.v_scales[src];
}

template <int kScaleDim>
__device__ __forceinline__ uint8_t
sm120_nvfp4_paged_v_pv_scale_from_physical_page_static(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int kv_head,
    int physical_page,
    int dim) {
  static_assert((kScaleDim & (kScaleDim - 1)) == 0,
                "Static PV scale dimension must be a power of two.");
  const int scale_row = dim / kScaleDim;
  const int scale_col = dim - scale_row * kScaleDim;
  const int64_t src =
      static_cast<int64_t>(physical_page) * params.v_scale_stride_page +
      static_cast<int64_t>(scale_row) * params.v_scale_stride_dim1 +
      static_cast<int64_t>(kv_head) * params.v_scale_stride_dim2 +
      static_cast<int64_t>(scale_col) * params.v_scale_stride_dim3;
  return params.v_scales[src];
}

__device__ __forceinline__ uint8_t sm120_nvfp4_paged_v_pv_scale_from_physical_page(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int physical_page,
    int dim) {
  return sm120_nvfp4_paged_v_pv_scale_from_physical_page(
      params, params.kv_head, physical_page, dim);
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
