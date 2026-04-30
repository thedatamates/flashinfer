#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>

#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_paged_adapter.cuh>

namespace sm120_nvfp4_paged_adapter {

inline void check_byte_tensor(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.element_size() == 1, name, " must have 1-byte elements");
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

  C10_CUDA_CHECK(
      flashinfer::attention::blackwell::sm120_nvfp4::
          gather_paged_kv_to_dense_pv_raw(
      reinterpret_cast<const uint8_t*>(k_pages.data_ptr()),
      reinterpret_cast<const uint8_t*>(k_sf_pages.data_ptr()),
      reinterpret_cast<const uint8_t*>(v_pages_pv.data_ptr()),
      reinterpret_cast<const uint8_t*>(v_sf_pages_pv.data_ptr()),
      block_table_ptr,
      reinterpret_cast<uint8_t*>(k_dense.data_ptr()),
      reinterpret_cast<uint8_t*>(k_sf_dense.data_ptr()),
      reinterpret_cast<uint8_t*>(v_pv_dense.data_ptr()),
      reinterpret_cast<uint8_t*>(v_pv_sf_dense.data_ptr()),
      static_cast<int>(kv_head), static_cast<int>(kv_len),
      static_cast<int>(page_size), static_cast<int>(num_kv_heads),
      static_cast<int>(packed_dim), static_cast<int>(scale_dim),
      at::cuda::getCurrentCUDAStream()));
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
