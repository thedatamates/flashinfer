/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_d256.cuh>

#include "fmha_nvfp4_sm120_paged_common.cuh"

namespace flashinfer {
namespace {

struct Sm120Nvfp4D256PagedKernel {
  static constexpr int kHeadDim = 256;
  static constexpr int kTileM =
      attention::blackwell::sm120_nvfp4::d256::kCutlassTileM;
  static constexpr int kOutputGroupSpan = 2;
  using PagedParams =
      attention::blackwell::sm120_nvfp4::d256::Sm120Nvfp4PagedKvLoadParams;

  static cudaError_t run(
      uint8_t* q_packed, uint8_t* q_scales, uint8_t* k_packed,
      uint8_t* k_scales, uint8_t* v_pv_packed, uint8_t* v_pv_scales,
      __nv_bfloat16* partial, float* split_m, float* split_l,
      __nv_bfloat16* out, uint8_t* workspace, size_t workspace_bytes,
      float qk_alpha, float pv_alpha, int split_kv_tiles, int q_len,
      int group_size, int kv_len_tokens, bool causal, int sliding_window,
      float logits_soft_cap, int q_rows, int kv_len, cudaStream_t stream,
      PagedParams paged_params, const int32_t* qo_indptr,
      const int32_t* kv_lens, int batch_size, int q_tiles_per_sequence,
      bool skip_internal_combine) {
    return attention::blackwell::sm120_nvfp4::d256::
        sm120_nvfp4_qkv_online_register_q_splitkv_full_grid_raw<2, true>(
            q_packed, q_scales, k_packed, k_scales, v_pv_packed, v_pv_scales,
            partial, split_m, split_l, out, workspace, workspace_bytes,
            qk_alpha, pv_alpha, split_kv_tiles, q_len, group_size,
            kv_len_tokens, causal, sliding_window, logits_soft_cap, q_rows,
            kHeadDim, kv_len, stream, paged_params, qo_indptr, kv_lens,
            batch_size, q_tiles_per_sequence, skip_internal_combine);
  }
};

}  // namespace

void SM120Nvfp4QuantizeQ(TensorView q, TensorView q_packed,
                         TensorView q_scales) {
  sm120_nvfp4_paged::QuantizeQImpl<Sm120Nvfp4D256PagedKernel>(
      q, q_packed, q_scales);
}

void SM120Nvfp4FmhaRunPagedBatch(
    TensorView q_packed, TensorView q_scales, TensorView k_pages,
    TensorView k_sf_pages, TensorView v_pages_pv, TensorView v_sf_pages_pv,
    TensorView block_tables, TensorView qo_indptr, TensorView kv_lens,
    TensorView q_packed_scratch, TensorView q_scales_scratch,
    TensorView partial, TensorView split_m, TensorView split_l,
    TensorView out_scratch, TensorView out, TensorView workspace,
    int64_t max_physical_kv_len, double qk_alpha, double pv_alpha,
    int64_t kv_head, int64_t split_kv_tiles, int64_t group_size, bool causal,
    int64_t sliding_window, double logits_soft_cap,
    int64_t output_group_span, bool v_cache_uses_pv_layout,
    bool normal_v_scales_are_trtllm_interleaved, bool kv_layout_hnd) {
  sm120_nvfp4_paged::RunPagedBatchImpl<Sm120Nvfp4D256PagedKernel>(
      q_packed, q_scales, k_pages, k_sf_pages, v_pages_pv, v_sf_pages_pv,
      block_tables, qo_indptr, kv_lens, q_packed_scratch, q_scales_scratch,
      partial, split_m, split_l, out_scratch, out, workspace,
      max_physical_kv_len, qk_alpha, pv_alpha, kv_head, split_kv_tiles,
      group_size, causal, sliding_window, logits_soft_cap, output_group_span,
      v_cache_uses_pv_layout, normal_v_scales_are_trtllm_interleaved,
      kv_layout_hnd);
}

}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_q, flashinfer::SM120Nvfp4QuantizeQ);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_paged_batch,
                              flashinfer::SM120Nvfp4FmhaRunPagedBatch);
