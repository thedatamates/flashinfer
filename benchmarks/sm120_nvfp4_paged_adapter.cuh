#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>

namespace sm120_nvfp4_paged_adapter {

inline void check_byte_tensor(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.element_size() == 1, name, " must have 1-byte elements");
}

__global__ void gather_k_pages_kernel(
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

__global__ void gather_v_pv_pages_kernel(
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

inline void gather_paged_kv_to_dense_pv_from_block_table_ptr(
    torch::Tensor k_pages,
    torch::Tensor k_sf_pages,
    torch::Tensor v_pages_pv,
    torch::Tensor v_sf_pages_pv,
    const int32_t* block_table_ptr,
    torch::Tensor k_dense,
    torch::Tensor k_sf_dense,
    torch::Tensor v_pv_dense,
    torch::Tensor v_pv_sf_dense,
    int64_t kv_head,
    int64_t kv_len) {
  check_byte_tensor(k_pages, "k_pages");
  check_byte_tensor(k_sf_pages, "k_sf_pages");
  check_byte_tensor(v_pages_pv, "v_pages_pv");
  check_byte_tensor(v_sf_pages_pv, "v_sf_pages_pv");
  check_byte_tensor(k_dense, "k_dense");
  check_byte_tensor(k_sf_dense, "k_sf_dense");
  check_byte_tensor(v_pv_dense, "v_pv_dense");
  check_byte_tensor(v_pv_sf_dense, "v_pv_sf_dense");
  TORCH_CHECK(block_table_ptr != nullptr, "block_table pointer must be valid");
  TORCH_CHECK(k_pages.dim() == 4,
              "k_pages must have shape [num_pages, page_size, H_kv, D/2]");
  TORCH_CHECK(k_sf_pages.dim() == 4,
              "k_sf_pages must have shape [num_pages, page_size, H_kv, D/16]");
  TORCH_CHECK(v_pages_pv.sizes() == k_pages.sizes(),
              "v_pages_pv must match k_pages shape");
  TORCH_CHECK(v_sf_pages_pv.sizes() == k_sf_pages.sizes(),
              "v_sf_pages_pv must match k_sf_pages shape");

  const int64_t num_pages = k_pages.size(0);
  const int64_t page_size = k_pages.size(1);
  const int64_t num_kv_heads = k_pages.size(2);
  const int64_t packed_dim = k_pages.size(3);
  const int64_t scale_dim = k_sf_pages.size(3);
  const int64_t head_dim = packed_dim * 2;
  TORCH_CHECK(page_size == 16,
              "paged adapter currently supports page_size=16 only");
  TORCH_CHECK(k_sf_pages.size(0) == num_pages &&
                  k_sf_pages.size(1) == page_size &&
                  k_sf_pages.size(2) == num_kv_heads,
              "k_sf_pages leading dimensions must match k_pages");
  TORCH_CHECK(scale_dim * 16 == head_dim,
              "scale dimension must equal D/16");
  TORCH_CHECK(kv_head >= 0 && kv_head < num_kv_heads,
              "kv_head out of range");
  TORCH_CHECK(kv_len > 0 && kv_len % 128 == 0,
              "kv_len must be a positive multiple of 128 for the fused kernel");
  TORCH_CHECK(k_dense.sizes() == torch::IntArrayRef({kv_len, packed_dim}),
              "k_dense must have shape [kv_len, D/2]");
  TORCH_CHECK(k_sf_dense.sizes() == torch::IntArrayRef({kv_len, scale_dim}),
              "k_sf_dense must have shape [kv_len, D/16]");
  TORCH_CHECK(v_pv_dense.sizes() ==
                  torch::IntArrayRef({head_dim, kv_len / 2}),
              "v_pv_dense must have shape [D, kv_len/2]");
  TORCH_CHECK(v_pv_sf_dense.sizes() ==
                  torch::IntArrayRef({head_dim, kv_len / page_size}),
              "v_pv_sf_dense must have shape [D, kv_len/page_size]");

  const int k_threads = 256;
  gather_k_pages_kernel<<<kv_len, k_threads, 0,
                          at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(k_pages.data_ptr()),
      reinterpret_cast<const uint8_t*>(k_sf_pages.data_ptr()),
      block_table_ptr,
      reinterpret_cast<uint8_t*>(k_dense.data_ptr()),
      reinterpret_cast<uint8_t*>(k_sf_dense.data_ptr()),
      static_cast<int>(kv_len),
      static_cast<int>(page_size), static_cast<int>(num_kv_heads),
      static_cast<int>(kv_head), static_cast<int>(packed_dim),
      static_cast<int>(scale_dim));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 grid_v(static_cast<unsigned>(kv_len / 2),
              static_cast<unsigned>((head_dim + k_threads - 1) / k_threads));
  gather_v_pv_pages_kernel<<<grid_v, k_threads, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(v_pages_pv.data_ptr()),
      reinterpret_cast<const uint8_t*>(v_sf_pages_pv.data_ptr()),
      block_table_ptr,
      reinterpret_cast<uint8_t*>(v_pv_dense.data_ptr()),
      reinterpret_cast<uint8_t*>(v_pv_sf_dense.data_ptr()),
      static_cast<int>(kv_len),
      static_cast<int>(page_size), static_cast<int>(num_kv_heads),
      static_cast<int>(kv_head), static_cast<int>(packed_dim),
      static_cast<int>(scale_dim), static_cast<int>(head_dim));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

inline void gather_paged_kv_to_dense_pv(
    torch::Tensor k_pages,
    torch::Tensor k_sf_pages,
    torch::Tensor v_pages_pv,
    torch::Tensor v_sf_pages_pv,
    torch::Tensor block_table,
    torch::Tensor k_dense,
    torch::Tensor k_sf_dense,
    torch::Tensor v_pv_dense,
    torch::Tensor v_pv_sf_dense,
    int64_t kv_head,
    int64_t kv_len) {
  TORCH_CHECK(block_table.is_cuda(), "block_table must be CUDA");
  TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32,
              "block_table must be int32");
  const int64_t page_size = k_pages.size(1);
  TORCH_CHECK(block_table.numel() >= (kv_len + page_size - 1) / page_size,
              "block_table does not cover kv_len");
  gather_paged_kv_to_dense_pv_from_block_table_ptr(
      k_pages, k_sf_pages, v_pages_pv, v_sf_pages_pv,
      block_table.data_ptr<int32_t>(), k_dense, k_sf_dense, v_pv_dense,
      v_pv_sf_dense, kv_head, kv_len);
}

}  // namespace sm120_nvfp4_paged_adapter
