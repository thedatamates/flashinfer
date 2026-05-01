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
  int native_k = 1;
  int native_v = 1;
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

static __global__ void gather_k_pages_kernel(
    const uint8_t* k_pages,
    const uint8_t* k_sf_pages,
    const int32_t* block_table,
    uint8_t* k_dense,
    uint8_t* k_sf_dense,
    int kv_len,
    int page_size,
    int num_kv_heads,
    int kv_head,
    int packed_dim,
    int scale_dim,
    int64_t k_stride_page,
    int64_t k_stride_dim1,
    int64_t k_stride_dim2,
    int64_t k_stride_dim3,
    int64_t k_sf_stride_page,
    int64_t k_sf_stride_dim1,
    int64_t k_sf_stride_dim2,
    int64_t k_sf_stride_dim3,
    bool kv_layout_hnd) {
  const int token = int(blockIdx.x);
  const int col = int(threadIdx.x);
  if (token >= kv_len) {
    return;
  }
  const int logical_page = token / page_size;
  const int page_offset = token - logical_page * page_size;
  const int physical_page = block_table[logical_page];

  if (col < packed_dim) {
    const int64_t src =
        kv_layout_hnd
            ? (static_cast<int64_t>(physical_page) * k_stride_page +
               static_cast<int64_t>(kv_head) * k_stride_dim1 +
               static_cast<int64_t>(page_offset) * k_stride_dim2 +
               static_cast<int64_t>(col) * k_stride_dim3)
            : (static_cast<int64_t>(physical_page) * k_stride_page +
               static_cast<int64_t>(page_offset) * k_stride_dim1 +
               static_cast<int64_t>(kv_head) * k_stride_dim2 +
               static_cast<int64_t>(col) * k_stride_dim3);
    k_dense[token * packed_dim + col] = k_pages[src];
  }
  if (col < scale_dim) {
    const int64_t src =
        kv_layout_hnd
            ? (static_cast<int64_t>(physical_page) * k_sf_stride_page +
               static_cast<int64_t>(kv_head) * k_sf_stride_dim1 +
               static_cast<int64_t>(page_offset) * k_sf_stride_dim2 +
               static_cast<int64_t>(col) * k_sf_stride_dim3)
            : (static_cast<int64_t>(physical_page) * k_sf_stride_page +
               static_cast<int64_t>(page_offset) * k_sf_stride_dim1 +
               static_cast<int64_t>(kv_head) * k_sf_stride_dim2 +
               static_cast<int64_t>(col) * k_sf_stride_dim3);
    k_sf_dense[token * scale_dim + col] = k_sf_pages[src];
  }
}

static __global__ void gather_v_pv_pages_kernel(
    const uint8_t* v_pages,
    const uint8_t* v_sf_pages,
    const int32_t* block_table,
    uint8_t* v_pv_dense,
    uint8_t* v_pv_sf_dense,
    int kv_len,
    int page_size,
    int num_kv_heads,
    int kv_head,
    int packed_dim,
    int scale_dim,
    int head_dim,
    int64_t v_stride_page,
    int64_t v_stride_dim1,
    int64_t v_stride_dim2,
    int64_t v_stride_dim3,
    int64_t v_sf_stride_page,
    int64_t v_sf_stride_dim1,
    int64_t v_sf_stride_dim2,
    int64_t v_sf_stride_dim3) {
  const int pair = int(blockIdx.x);
  const int d = int(blockIdx.y * blockDim.x + threadIdx.x);
  if (pair >= kv_len / 2 || d >= head_dim) {
    return;
  }

  const int token0 = pair * 2;
  const int token1 = token0 + 1;
  const int logical_page0 = token0 / page_size;
  const int logical_page1 = token1 / page_size;
  const int physical_page0 = block_table[logical_page0];
  const int physical_page1 = block_table[logical_page1];
  const int page_offset0 = token0 - logical_page0 * page_size;
  const int page_offset1 = token1 - logical_page1 * page_size;
  const int packed_col = d >> 1;
  const int nibble_shift = (d & 1) * 4;

  const int64_t src0 =
      static_cast<int64_t>(physical_page0) * v_stride_page +
      static_cast<int64_t>(page_offset0) * v_stride_dim1 +
      static_cast<int64_t>(kv_head) * v_stride_dim2 +
      static_cast<int64_t>(packed_col) * v_stride_dim3;
  const int64_t src1 =
      static_cast<int64_t>(physical_page1) * v_stride_page +
      static_cast<int64_t>(page_offset1) * v_stride_dim1 +
      static_cast<int64_t>(kv_head) * v_stride_dim2 +
      static_cast<int64_t>(packed_col) * v_stride_dim3;
  const uint8_t nib0 = (v_pages[src0] >> nibble_shift) & 0x0f;
  const uint8_t nib1 = (v_pages[src1] >> nibble_shift) & 0x0f;
  v_pv_dense[d * (kv_len / 2) + pair] =
      static_cast<uint8_t>(nib0 | (nib1 << 4));

  if ((pair & ((page_size / 2) - 1)) == 0) {
    const int scale_col = pair / (page_size / 2);
    const int scale_page = block_table[scale_col];
    const int scale_row_in_page = d / scale_dim;
    const int scale_col_in_page = d - scale_row_in_page * scale_dim;
    const int64_t scale_src =
        static_cast<int64_t>(scale_page) * v_sf_stride_page +
        static_cast<int64_t>(scale_row_in_page) * v_sf_stride_dim1 +
        static_cast<int64_t>(kv_head) * v_sf_stride_dim2 +
        static_cast<int64_t>(scale_col_in_page) * v_sf_stride_dim3;
    v_pv_sf_dense[d * (kv_len / page_size) + scale_col] =
        v_sf_pages[scale_src];
  }
}

static __global__ void gather_v_normal_pages_to_dense_pv_kernel(
    const uint8_t* v_pages,
    const uint8_t* v_sf_pages,
    const int32_t* block_table,
    uint8_t* v_pv_dense,
    uint8_t* v_pv_sf_dense,
    int real_kv_len,
    int physical_kv_len,
    int page_size,
    int num_kv_heads,
    int kv_head,
    int packed_dim,
    int scale_dim,
    int head_dim,
    float v_global_scale,
    int64_t v_stride_page,
    int64_t v_stride_dim1,
    int64_t v_stride_dim2,
    int64_t v_stride_dim3,
    int64_t v_sf_stride_page,
    int64_t v_sf_stride_dim1,
    int64_t v_sf_stride_dim2,
    int64_t v_sf_stride_dim3,
    bool kv_layout_hnd,
    bool trtllm_interleaved_v_scales) {
  const int logical_page = int(blockIdx.x);
  const int d = int(blockIdx.y * blockDim.x + threadIdx.x);
  if (logical_page >= physical_kv_len / page_size || d >= head_dim) {
    return;
  }

  const int physical_page = block_table[logical_page];
  const int packed_col = d >> 1;
  const int nibble_shift = (d & 1) * 4;
  const int scale_col = d >> 4;

  float values[16];
  float max_abs = 0.0f;
#pragma unroll
  for (int t = 0; t < 16; ++t) {
    const int token = logical_page * page_size + t;
    float value = 0.0f;
    if (token < real_kv_len && t < page_size) {
      const int64_t data_src =
          kv_layout_hnd
              ? (static_cast<int64_t>(physical_page) * v_stride_page +
                 static_cast<int64_t>(kv_head) * v_stride_dim1 +
                 static_cast<int64_t>(t) * v_stride_dim2 +
                 static_cast<int64_t>(packed_col) * v_stride_dim3)
              : (static_cast<int64_t>(physical_page) * v_stride_page +
                 static_cast<int64_t>(t) * v_stride_dim1 +
                 static_cast<int64_t>(kv_head) * v_stride_dim2 +
                 static_cast<int64_t>(packed_col) * v_stride_dim3);
      const uint8_t packed = v_pages[data_src];
      const uint8_t code = (packed >> nibble_shift) & 0x0f;

      int scale_offset = t * scale_dim + scale_col;
      if (trtllm_interleaved_v_scales) {
        scale_offset = trtllm_v_scale_offset(t, scale_col, scale_dim);
      }
      const int scale_t = scale_offset / scale_dim;
      const int scale_s = scale_offset - scale_t * scale_dim;
      const int64_t scale_src =
          kv_layout_hnd
              ? (static_cast<int64_t>(physical_page) * v_sf_stride_page +
                 static_cast<int64_t>(kv_head) * v_sf_stride_dim1 +
                 static_cast<int64_t>(scale_t) * v_sf_stride_dim2 +
                 static_cast<int64_t>(scale_s) * v_sf_stride_dim3)
              : (static_cast<int64_t>(physical_page) * v_sf_stride_page +
                 static_cast<int64_t>(scale_t) * v_sf_stride_dim1 +
                 static_cast<int64_t>(kv_head) * v_sf_stride_dim2 +
                 static_cast<int64_t>(scale_s) * v_sf_stride_dim3);
      value = e2m1_code_to_fp32(code) * e4m3_byte_to_fp32(v_sf_pages[scale_src]) *
              v_global_scale;
    }
    values[t] = value;
    max_abs = fmaxf(max_abs, fabsf(value));
  }

  const uint8_t scale_byte = fp32_to_e4m3_byte(
      fmaxf(max_abs / fmaxf(6.0f * v_global_scale, 1.0e-20f), 1.0e-8f));
  const float quant_scale =
      fmaxf(e4m3_byte_to_fp32(scale_byte) * v_global_scale, 1.0e-8f);

  v_pv_sf_dense[d * (physical_kv_len / page_size) + logical_page] =
      scale_byte;

  const int pair_base = logical_page * (page_size / 2);
#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const uint8_t code0 = nearest_e2m1_code(values[2 * pair] / quant_scale);
    const uint8_t code1 = nearest_e2m1_code(values[2 * pair + 1] / quant_scale);
    v_pv_dense[d * (physical_kv_len / 2) + pair_base + pair] =
        static_cast<uint8_t>(code0 | (code1 << 4));
  }
}

inline cudaError_t gather_paged_kv_to_dense_pv_raw(
    const uint8_t* k_pages,
    const uint8_t* k_sf_pages,
    const uint8_t* v_pages_pv,
    const uint8_t* v_sf_pages_pv,
    const int32_t* block_table_ptr,
    uint8_t* k_dense,
    uint8_t* k_sf_dense,
    uint8_t* v_pv_dense,
    uint8_t* v_pv_sf_dense,
    int kv_head,
    int kv_len,
    int page_size,
    int num_kv_heads,
    int packed_dim,
    int scale_dim,
    int64_t k_stride_page,
    int64_t k_stride_dim1,
    int64_t k_stride_dim2,
    int64_t k_stride_dim3,
    int64_t k_sf_stride_page,
    int64_t k_sf_stride_dim1,
    int64_t k_sf_stride_dim2,
    int64_t k_sf_stride_dim3,
    int64_t v_stride_page,
    int64_t v_stride_dim1,
    int64_t v_stride_dim2,
    int64_t v_stride_dim3,
    int64_t v_sf_stride_page,
    int64_t v_sf_stride_dim1,
    int64_t v_sf_stride_dim2,
    int64_t v_sf_stride_dim3,
    bool kv_layout_hnd,
    cudaStream_t stream) {
  if (kv_layout_hnd) {
    return cudaErrorInvalidValue;
  }
  const int head_dim = packed_dim * 2;
  constexpr int kThreads = 256;
  gather_k_pages_kernel<<<kv_len, kThreads, 0, stream>>>(
      k_pages, k_sf_pages, block_table_ptr, k_dense, k_sf_dense, kv_len,
      page_size, num_kv_heads, kv_head, packed_dim, scale_dim,
      k_stride_page, k_stride_dim1, k_stride_dim2, k_stride_dim3,
      k_sf_stride_page, k_sf_stride_dim1, k_sf_stride_dim2, k_sf_stride_dim3,
      kv_layout_hnd);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  dim3 grid_v(static_cast<unsigned>(kv_len / 2),
              static_cast<unsigned>((head_dim + kThreads - 1) / kThreads));
  gather_v_pv_pages_kernel<<<grid_v, kThreads, 0, stream>>>(
      v_pages_pv, v_sf_pages_pv, block_table_ptr, v_pv_dense,
      v_pv_sf_dense, kv_len, page_size, num_kv_heads, kv_head, packed_dim,
      scale_dim, head_dim,
      v_stride_page, v_stride_dim1, v_stride_dim2, v_stride_dim3,
      v_sf_stride_page, v_sf_stride_dim1, v_sf_stride_dim2,
      v_sf_stride_dim3);
  return cudaGetLastError();
}

inline cudaError_t gather_paged_v_to_dense_pv_raw(
    const uint8_t* v_pages_pv,
    const uint8_t* v_sf_pages_pv,
    const int32_t* block_table_ptr,
    uint8_t* v_pv_dense,
    uint8_t* v_pv_sf_dense,
    int kv_head,
    int kv_len,
    int page_size,
    int num_kv_heads,
    int packed_dim,
    int scale_dim,
    int64_t v_stride_page,
    int64_t v_stride_dim1,
    int64_t v_stride_dim2,
    int64_t v_stride_dim3,
    int64_t v_sf_stride_page,
    int64_t v_sf_stride_dim1,
    int64_t v_sf_stride_dim2,
    int64_t v_sf_stride_dim3,
    bool kv_layout_hnd,
    cudaStream_t stream) {
  if (kv_layout_hnd) {
    return cudaErrorInvalidValue;
  }
  const int head_dim = packed_dim * 2;
  constexpr int kThreads = 256;
  dim3 grid_v(static_cast<unsigned>(kv_len / 2),
              static_cast<unsigned>((head_dim + kThreads - 1) / kThreads));
  gather_v_pv_pages_kernel<<<grid_v, kThreads, 0, stream>>>(
      v_pages_pv, v_sf_pages_pv, block_table_ptr, v_pv_dense,
      v_pv_sf_dense, kv_len, page_size, num_kv_heads, kv_head, packed_dim,
      scale_dim, head_dim, v_stride_page, v_stride_dim1, v_stride_dim2,
      v_stride_dim3, v_sf_stride_page, v_sf_stride_dim1,
      v_sf_stride_dim2, v_sf_stride_dim3);
  return cudaGetLastError();
}

inline cudaError_t gather_paged_kv_normal_v_to_dense_pv_raw(
    const uint8_t* k_pages,
    const uint8_t* k_sf_pages,
    const uint8_t* v_pages,
    const uint8_t* v_sf_pages,
    const int32_t* block_table_ptr,
    uint8_t* k_dense,
    uint8_t* k_sf_dense,
    uint8_t* v_pv_dense,
    uint8_t* v_pv_sf_dense,
    int kv_head,
    int real_kv_len,
    int physical_kv_len,
    int page_size,
    int num_kv_heads,
    int packed_dim,
    int scale_dim,
    float v_global_scale,
    int64_t k_stride_page,
    int64_t k_stride_dim1,
    int64_t k_stride_dim2,
    int64_t k_stride_dim3,
    int64_t k_sf_stride_page,
    int64_t k_sf_stride_dim1,
    int64_t k_sf_stride_dim2,
    int64_t k_sf_stride_dim3,
    int64_t v_stride_page,
    int64_t v_stride_dim1,
    int64_t v_stride_dim2,
    int64_t v_stride_dim3,
    int64_t v_sf_stride_page,
    int64_t v_sf_stride_dim1,
    int64_t v_sf_stride_dim2,
    int64_t v_sf_stride_dim3,
    bool kv_layout_hnd,
    bool trtllm_interleaved_v_scales,
    cudaStream_t stream) {
  const int head_dim = packed_dim * 2;
  constexpr int kThreads = 256;
  gather_k_pages_kernel<<<physical_kv_len, kThreads, 0, stream>>>(
      k_pages, k_sf_pages, block_table_ptr, k_dense, k_sf_dense,
      physical_kv_len, page_size, num_kv_heads, kv_head, packed_dim,
      scale_dim, k_stride_page, k_stride_dim1, k_stride_dim2,
      k_stride_dim3, k_sf_stride_page, k_sf_stride_dim1,
      k_sf_stride_dim2, k_sf_stride_dim3, kv_layout_hnd);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  dim3 grid_v(static_cast<unsigned>(physical_kv_len / page_size),
              static_cast<unsigned>((head_dim + kThreads - 1) / kThreads));
  gather_v_normal_pages_to_dense_pv_kernel<<<grid_v, kThreads, 0, stream>>>(
      v_pages, v_sf_pages, block_table_ptr, v_pv_dense, v_pv_sf_dense,
      real_kv_len, physical_kv_len, page_size, num_kv_heads, kv_head,
      packed_dim, scale_dim, head_dim, v_global_scale,
      v_stride_page, v_stride_dim1, v_stride_dim2, v_stride_dim3,
      v_sf_stride_page, v_sf_stride_dim1, v_sf_stride_dim2, v_sf_stride_dim3,
      kv_layout_hnd, trtllm_interleaved_v_scales);
  return cudaGetLastError();
}

inline cudaError_t gather_paged_normal_v_to_dense_pv_raw(
    const uint8_t* v_pages,
    const uint8_t* v_sf_pages,
    const int32_t* block_table_ptr,
    uint8_t* v_pv_dense,
    uint8_t* v_pv_sf_dense,
    int kv_head,
    int real_kv_len,
    int physical_kv_len,
    int page_size,
    int num_kv_heads,
    int packed_dim,
    int scale_dim,
    float v_global_scale,
    int64_t v_stride_page,
    int64_t v_stride_dim1,
    int64_t v_stride_dim2,
    int64_t v_stride_dim3,
    int64_t v_sf_stride_page,
    int64_t v_sf_stride_dim1,
    int64_t v_sf_stride_dim2,
    int64_t v_sf_stride_dim3,
    bool kv_layout_hnd,
    bool trtllm_interleaved_v_scales,
    cudaStream_t stream) {
  const int head_dim = packed_dim * 2;
  constexpr int kThreads = 256;
  dim3 grid_v(static_cast<unsigned>(physical_kv_len / page_size),
              static_cast<unsigned>((head_dim + kThreads - 1) / kThreads));
  gather_v_normal_pages_to_dense_pv_kernel<<<grid_v, kThreads, 0, stream>>>(
      v_pages, v_sf_pages, block_table_ptr, v_pv_dense, v_pv_sf_dense,
      real_kv_len, physical_kv_len, page_size, num_kv_heads, kv_head,
      packed_dim, scale_dim, head_dim, v_global_scale, v_stride_page,
      v_stride_dim1, v_stride_dim2, v_stride_dim3, v_sf_stride_page,
      v_sf_stride_dim1, v_sf_stride_dim2, v_sf_stride_dim3, kv_layout_hnd,
      trtllm_interleaved_v_scales);
  return cudaGetLastError();
}

}  // namespace flashinfer::attention::blackwell::sm120_nvfp4
