#pragma once

#include <torch/extension.h>

#include <cstdint>

#include "sm120_nvfp4_paged_adapter.cuh"

namespace sm120_nvfp4_paged_bridge {

inline int64_t round_up_multiple(int64_t x, int64_t multiple) {
  return ((x + multiple - 1) / multiple) * multiple;
}

inline int64_t tensor_i64_at(const torch::Tensor& t, int64_t index) {
  if (t.scalar_type() == torch::kInt32) {
    return static_cast<int64_t>(t.data_ptr<int32_t>()[index]);
  }
  if (t.scalar_type() == torch::kInt64) {
    return t.data_ptr<int64_t>()[index];
  }
  TORCH_CHECK(false, "expected int32 or int64 tensor");
}

inline void check_int_tensor(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.scalar_type() == torch::kInt32 ||
                  t.scalar_type() == torch::kInt64,
              name, " must be int32 or int64");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

template <typename DenseRunner>
void paged_qkv_attention_single(
    torch::Tensor q_packed,
    torch::Tensor q_scales,
    torch::Tensor k_pages,
    torch::Tensor k_sf_pages,
    torch::Tensor v_pages_pv,
    torch::Tensor v_sf_pages_pv,
    torch::Tensor block_table,
    torch::Tensor k_dense_scratch,
    torch::Tensor k_sf_dense_scratch,
    torch::Tensor v_pv_dense_scratch,
    torch::Tensor v_pv_sf_dense_scratch,
    torch::Tensor partial,
    torch::Tensor split_m,
    torch::Tensor split_l,
    torch::Tensor out,
    torch::Tensor workspace,
    double qk_alpha,
    double pv_alpha,
    int64_t kv_head,
    int64_t split_kv_tiles,
    int64_t q_len,
    int64_t group_size,
    int64_t kv_len_tokens,
    bool causal,
    int64_t sliding_window,
    double logits_soft_cap,
    int64_t output_group_span,
    DenseRunner&& run_dense) {
  TORCH_CHECK(block_table.is_cuda(), "block_table must be CUDA");
  TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32,
              "block_table must be int32");
  const int64_t page_size = k_pages.size(1);
  const int64_t physical_kv_len = k_dense_scratch.size(0);
  TORCH_CHECK(physical_kv_len >= kv_len_tokens,
              "K/V scratch must cover logical KV length");
  TORCH_CHECK(physical_kv_len % 128 == 0,
              "K/V scratch length must be a multiple of 128");
  TORCH_CHECK(block_table.numel() >=
                  (physical_kv_len + page_size - 1) / page_size,
              "block_table does not cover physical K/V scratch length");

  sm120_nvfp4_paged_adapter::gather_paged_kv_to_dense_pv_from_block_table_ptr(
      k_pages, k_sf_pages, v_pages_pv, v_sf_pages_pv,
      block_table.data_ptr<int32_t>(), k_dense_scratch, k_sf_dense_scratch,
      v_pv_dense_scratch, v_pv_sf_dense_scratch, kv_head, physical_kv_len);

  run_dense(q_packed, q_scales, k_dense_scratch, k_sf_dense_scratch,
            v_pv_dense_scratch, v_pv_sf_dense_scratch, partial, split_m,
            split_l, out, workspace, qk_alpha, pv_alpha, split_kv_tiles,
            q_len, group_size, kv_len_tokens, causal, sliding_window,
            logits_soft_cap, output_group_span);
}

template <typename DenseRunner>
void varlen_paged_qkv_attention(
    torch::Tensor q_packed,
    torch::Tensor q_scales,
    torch::Tensor k_pages,
    torch::Tensor k_sf_pages,
    torch::Tensor v_pages_pv,
    torch::Tensor v_sf_pages_pv,
    torch::Tensor block_tables,
    torch::Tensor cu_seqlens_q,
    torch::Tensor kv_lens,
    torch::Tensor out,
    torch::Tensor workspace,
    double qk_alpha,
    double pv_alpha,
    int64_t kv_head,
    int64_t group_size,
    int64_t split_kv_tiles,
    bool causal,
    int64_t sliding_window,
    double logits_soft_cap,
    int64_t output_group_span,
    DenseRunner&& run_dense) {
  sm120_nvfp4_paged_adapter::check_byte_tensor(q_packed, "q_packed");
  sm120_nvfp4_paged_adapter::check_byte_tensor(q_scales, "q_scales");
  sm120_nvfp4_paged_adapter::check_byte_tensor(k_pages, "k_pages");
  sm120_nvfp4_paged_adapter::check_byte_tensor(k_sf_pages, "k_sf_pages");
  sm120_nvfp4_paged_adapter::check_byte_tensor(v_pages_pv, "v_pages_pv");
  sm120_nvfp4_paged_adapter::check_byte_tensor(v_sf_pages_pv,
                                               "v_sf_pages_pv");
  TORCH_CHECK(block_tables.is_cuda(), "block_tables must be CUDA");
  TORCH_CHECK(block_tables.is_contiguous(), "block_tables must be contiguous");
  TORCH_CHECK(block_tables.scalar_type() == torch::kInt32,
              "block_tables must be int32");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() &&
                  out.scalar_type() == torch::kBFloat16,
              "out must be contiguous CUDA BF16");
  TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous() &&
                  workspace.scalar_type() == torch::kUInt8,
              "workspace must be contiguous CUDA uint8");
  check_int_tensor(cu_seqlens_q, "cu_seqlens_q");
  check_int_tensor(kv_lens, "kv_lens");
  TORCH_CHECK(q_packed.dim() == 2, "q_packed must be 2D");
  TORCH_CHECK(q_scales.dim() == 2, "q_scales must be 2D");
  TORCH_CHECK(block_tables.dim() == 2,
              "block_tables must have shape [batch, max_pages]");
  TORCH_CHECK(k_pages.dim() == 4,
              "k_pages must have shape [num_pages, page_size, H_kv, D/2]");

  const int64_t batch = block_tables.size(0);
  TORCH_CHECK(cu_seqlens_q.numel() == batch + 1,
              "cu_seqlens_q must have batch + 1 entries");
  TORCH_CHECK(kv_lens.numel() == batch,
              "kv_lens must have batch entries");
  const int64_t page_size = k_pages.size(1);
  const int64_t packed_dim = q_packed.size(1);
  const int64_t head_dim = packed_dim * 2;
  const int64_t scale_dim = head_dim / 16;
  TORCH_CHECK(q_scales.size(0) >= q_packed.size(0) &&
                  q_scales.size(1) == scale_dim,
              "q_scales must have shape [>= q_rows, D/16]");
  TORCH_CHECK(out.size(0) == q_packed.size(0) && out.size(1) == head_dim,
              "out must have shape [q_rows, D]");

  auto cu_q_cpu = cu_seqlens_q.to(torch::kCPU).contiguous();
  auto kv_lens_cpu = kv_lens.to(torch::kCPU).contiguous();
  auto byte_options = q_packed.options().dtype(torch::kUInt8);
  auto bf16_options = out.options().dtype(torch::kBFloat16);
  auto f32_options = out.options().dtype(torch::kFloat32);

  for (int64_t b = 0; b < batch; ++b) {
    const int64_t q_begin = tensor_i64_at(cu_q_cpu, b);
    const int64_t q_end = tensor_i64_at(cu_q_cpu, b + 1);
    const int64_t q_len = q_end - q_begin;
    TORCH_CHECK(q_len >= 0, "cu_seqlens_q must be non-decreasing");
    if (q_len == 0) {
      continue;
    }
    const int64_t q_row_begin = q_begin * group_size;
    const int64_t q_rows = q_len * group_size;
    const int64_t padded_q_rows = round_up_multiple(q_rows, 128);
    const int64_t kv_len_tokens = tensor_i64_at(kv_lens_cpu, b);
    const int64_t physical_kv_len = round_up_multiple(kv_len_tokens, 128);
    TORCH_CHECK(kv_len_tokens > 0, "kv_lens entries must be positive");
    TORCH_CHECK(q_row_begin + q_rows <= q_packed.size(0),
                "q sequence exceeds q_packed rows");
    TORCH_CHECK(physical_kv_len / page_size <= block_tables.size(1),
                "block_tables row does not cover padded KV length");

    auto q_seq = q_packed.narrow(0, q_row_begin, q_rows);
    auto q_sf_seq = q_scales.narrow(0, q_row_begin, q_rows);
    torch::Tensor q_phys = q_seq;
    torch::Tensor q_sf_phys = q_sf_seq;
    if (padded_q_rows != q_rows) {
      q_phys = torch::zeros({padded_q_rows, packed_dim}, byte_options);
      q_sf_phys = torch::zeros({padded_q_rows, scale_dim}, byte_options);
      q_phys.narrow(0, 0, q_rows).copy_(q_seq);
      q_sf_phys.narrow(0, 0, q_rows).copy_(q_sf_seq);
    }

    auto k_dense = torch::empty({physical_kv_len, packed_dim}, byte_options);
    auto k_sf_dense = torch::empty({physical_kv_len, scale_dim}, byte_options);
    auto v_pv_dense = torch::empty({head_dim, physical_kv_len / 2},
                                   byte_options);
    auto v_pv_sf_dense = torch::empty({head_dim, physical_kv_len / page_size},
                                      byte_options);
    const int32_t* block_table_ptr =
        block_tables.data_ptr<int32_t>() + b * block_tables.stride(0);
    sm120_nvfp4_paged_adapter::gather_paged_kv_to_dense_pv_from_block_table_ptr(
        k_pages, k_sf_pages, v_pages_pv, v_sf_pages_pv, block_table_ptr,
        k_dense, k_sf_dense, v_pv_dense, v_pv_sf_dense, kv_head,
        physical_kv_len);

    const int64_t total_kv_tiles = physical_kv_len / 128;
    const int64_t num_splits =
        (total_kv_tiles + split_kv_tiles - 1) / split_kv_tiles;
    auto partial = torch::empty({num_splits, padded_q_rows, head_dim},
                                bf16_options);
    auto split_m = torch::empty({num_splits, padded_q_rows}, f32_options);
    auto split_l = torch::empty({num_splits, padded_q_rows}, f32_options);
    auto out_phys = padded_q_rows == q_rows
                        ? out.narrow(0, q_row_begin, q_rows)
                        : torch::empty({padded_q_rows, head_dim}, bf16_options);

    run_dense(q_phys, q_sf_phys, k_dense, k_sf_dense, v_pv_dense,
              v_pv_sf_dense, partial, split_m, split_l, out_phys, workspace,
              qk_alpha, pv_alpha, split_kv_tiles, q_len, group_size,
              kv_len_tokens, causal, sliding_window, logits_soft_cap,
              output_group_span);

    if (padded_q_rows != q_rows) {
      out.narrow(0, q_row_begin, q_rows)
          .copy_(out_phys.narrow(0, 0, q_rows));
    }
  }
}

}  // namespace sm120_nvfp4_paged_bridge
