#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer::attention::blackwell::sm120_nvfp4 {

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
    int scale_dim) {
  const int token = int(blockIdx.x);
  const int col = int(threadIdx.x);
  if (token >= kv_len) {
    return;
  }
  const int logical_page = token / page_size;
  const int page_offset = token - logical_page * page_size;
  const int physical_page = block_table[logical_page];

  if (col < packed_dim) {
    const int src =
        (((physical_page * page_size + page_offset) * num_kv_heads + kv_head) *
             packed_dim +
         col);
    k_dense[token * packed_dim + col] = k_pages[src];
  }
  if (col < scale_dim) {
    const int src =
        (((physical_page * page_size + page_offset) * num_kv_heads + kv_head) *
             scale_dim +
         col);
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
    int head_dim) {
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

  const int src0 =
      (((physical_page0 * page_size + page_offset0) * num_kv_heads + kv_head) *
           packed_dim +
       packed_col);
  const int src1 =
      (((physical_page1 * page_size + page_offset1) * num_kv_heads + kv_head) *
           packed_dim +
       packed_col);
  const uint8_t nib0 = (v_pages[src0] >> nibble_shift) & 0x0f;
  const uint8_t nib1 = (v_pages[src1] >> nibble_shift) & 0x0f;
  v_pv_dense[d * (kv_len / 2) + pair] =
      static_cast<uint8_t>(nib0 | (nib1 << 4));

  if ((pair & ((page_size / 2) - 1)) == 0) {
    const int scale_col = pair / (page_size / 2);
    const int scale_page = block_table[scale_col];
    const int scale_row_in_page = d / scale_dim;
    const int scale_col_in_page = d - scale_row_in_page * scale_dim;
    const int scale_src =
        (((scale_page * page_size + scale_row_in_page) * num_kv_heads +
          kv_head) *
             scale_dim +
         scale_col_in_page);
    v_pv_sf_dense[d * (kv_len / page_size) + scale_col] =
        v_sf_pages[scale_src];
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
    cudaStream_t stream) {
  const int head_dim = packed_dim * 2;
  constexpr int kThreads = 256;
  gather_k_pages_kernel<<<kv_len, kThreads, 0, stream>>>(
      k_pages, k_sf_pages, block_table_ptr, k_dense, k_sf_dense, kv_len,
      page_size, num_kv_heads, kv_head, packed_dim, scale_dim);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  dim3 grid_v(static_cast<unsigned>(kv_len / 2),
              static_cast<unsigned>((head_dim + kThreads - 1) / kThreads));
  gather_v_pv_pages_kernel<<<grid_v, kThreads, 0, stream>>>(
      v_pages_pv, v_sf_pages_pv, block_table_ptr, v_pv_dense,
      v_pv_sf_dense, kv_len, page_size, num_kv_heads, kv_head, packed_dim,
      scale_dim, head_dim);
  return cudaGetLastError();
}

}  // namespace flashinfer::attention::blackwell::sm120_nvfp4
