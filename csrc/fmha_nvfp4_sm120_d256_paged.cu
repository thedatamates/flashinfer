/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "fmha_nvfp4_sm120_config.inc"

#if !SM120_NVFP4_USE_SLIDING_WINDOW_PREPROC
#define FLASHINFER_SM120_NVFP4_D256_TILE_M 128
#define FLASHINFER_SM120_NVFP4_D256_LOAD_WARPS 10
#endif

#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_d256.cuh>

#include "fmha_nvfp4_sm120_paged_common.cuh"

namespace flashinfer {
namespace {

namespace d256 = attention::blackwell::sm120_nvfp4::d256;
using attention::blackwell::sm120_nvfp4::Sm120Nvfp4PagedKvLoadParams;
using sm120_nvfp4_paged::CheckCudaTypeLastDimContiguous;
using sm120_nvfp4_paged::PagedKernelConfig;
using sm120_nvfp4_paged::PagedParams;

Sm120Nvfp4PagedKvLoadParams ToD256PagedParams(PagedParams params) {
  Sm120Nvfp4PagedKvLoadParams out;
  out.k_pages = params.k_pages;
  out.k_scales = params.k_scales;
  out.v_pages = params.v_pages;
  out.v_scales = params.v_scales;
  out.block_table = params.block_table;
  out.block_table_stride = params.block_table_stride;
  out.k_stride_page = params.k_stride_page;
  out.k_stride_dim1 = params.k_stride_dim1;
  out.k_stride_dim2 = params.k_stride_dim2;
  out.k_stride_dim3 = params.k_stride_dim3;
  out.k_scale_stride_page = params.k_scale_stride_page;
  out.k_scale_stride_dim1 = params.k_scale_stride_dim1;
  out.k_scale_stride_dim2 = params.k_scale_stride_dim2;
  out.k_scale_stride_dim3 = params.k_scale_stride_dim3;
  out.v_stride_page = params.v_stride_page;
  out.v_stride_dim1 = params.v_stride_dim1;
  out.v_stride_dim2 = params.v_stride_dim2;
  out.v_stride_dim3 = params.v_stride_dim3;
  out.v_scale_stride_page = params.v_scale_stride_page;
  out.v_scale_stride_dim1 = params.v_scale_stride_dim1;
  out.v_scale_stride_dim2 = params.v_scale_stride_dim2;
  out.v_scale_stride_dim3 = params.v_scale_stride_dim3;
  out.kv_head = params.kv_head;
  out.page_size = params.page_size;
  out.packed_dim = params.packed_dim;
  out.scale_dim = params.scale_dim;
  out.kv_layout_hnd = params.kv_layout_hnd;
  out.v_scale_layout = params.v_scale_layout;
  out.q_bf16 = params.q_bf16;
  out.q_stride_token = params.q_stride_token;
  out.q_stride_head = params.q_stride_head;
  out.q_stride_dim = params.q_stride_dim;
  out.q_stride_row = params.q_stride_row;
  out.q_is_3d = params.q_is_3d;
  return out;
}

cudaError_t Sm120Nvfp4D256RunPagedRaw(
    uint8_t* q_packed, uint8_t* q_scales, uint8_t* k_packed,
    uint8_t* k_scales, uint8_t* v_pv_packed, uint8_t* v_pv_scales,
    __nv_bfloat16* partial, float* split_m, float* split_l,
    __nv_bfloat16* out, uint8_t* workspace, size_t workspace_bytes,
    float qk_alpha, float pv_alpha, int split_kv_tiles, int q_len,
    int group_size, int kv_len_tokens, bool causal, int sliding_window,
    float logits_soft_cap, int q_rows, int kv_len, cudaStream_t stream,
    PagedParams paged_params, const int32_t* qo_indptr,
    const int32_t* kv_lens, int batch_size, int q_tiles_per_sequence,
    int num_kv_heads, bool all_kv_heads, bool skip_internal_combine) {
  return d256::sm120_nvfp4_qkv_online_register_q_splitkv_full_grid_raw<
      2, true, SM120_NVFP4_CAUSAL, SM120_NVFP4_USE_SLIDING_WINDOW,
      SM120_NVFP4_USE_LOGITS_SOFT_CAP, SM120_NVFP4_USE_PV_LAYOUT_V>(
      q_packed, q_scales, k_packed, k_scales, v_pv_packed, v_pv_scales,
      partial, split_m, split_l, out, workspace, workspace_bytes, qk_alpha,
      pv_alpha, split_kv_tiles, q_len, group_size, kv_len_tokens, causal,
      sliding_window, logits_soft_cap, q_rows, d256::kHeadDim, kv_len, stream,
      ToD256PagedParams(paged_params), qo_indptr, kv_lens, batch_size,
      q_tiles_per_sequence, num_kv_heads, all_kv_heads,
      skip_internal_combine);
}

PagedKernelConfig Sm120Nvfp4D256PagedKernelConfig() {
  return {d256::kHeadDim,
          d256::kCutlassTileM,
          2,
          SM120_NVFP4_CAUSAL,
          SM120_NVFP4_USE_SLIDING_WINDOW,
          SM120_NVFP4_USE_LOGITS_SOFT_CAP,
          Sm120Nvfp4D256RunPagedRaw};
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
    int64_t output_group_span, bool kv_layout_hnd, int64_t v_scale_layout,
    int64_t stream_handle) {
  const auto kernel = Sm120Nvfp4D256PagedKernelConfig();
  sm120_nvfp4_paged::RunPagedBatchImpl(
      kernel, q_packed, q_scales, k_pages, k_sf_pages, v_pages_pv,
      v_sf_pages_pv, block_tables, qo_indptr, kv_lens, q_packed_scratch,
      q_scales_scratch, partial, split_m, split_l, out_scratch, out,
      workspace, max_physical_kv_len, qk_alpha, pv_alpha, kv_head,
      split_kv_tiles, group_size, causal, sliding_window, logits_soft_cap,
      output_group_span, kv_layout_hnd, v_scale_layout, stream_handle);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(paged_run, SM120Nvfp4FmhaRunPagedBatch);

void SM120Nvfp4FmhaRunPagedBatchBf16Q(
    TensorView q, TensorView q_packed, TensorView q_scales,
    TensorView k_pages, TensorView k_sf_pages, TensorView v_pages_pv,
    TensorView v_sf_pages_pv, TensorView block_tables, TensorView qo_indptr,
    TensorView kv_lens, TensorView q_packed_scratch,
    TensorView q_scales_scratch, TensorView partial, TensorView split_m,
    TensorView split_l, TensorView out_scratch, TensorView out,
    TensorView workspace, int64_t max_physical_kv_len, double qk_alpha,
    double pv_alpha, int64_t kv_head, int64_t split_kv_tiles,
    int64_t group_size, bool causal, int64_t sliding_window,
    double logits_soft_cap, int64_t output_group_span, bool kv_layout_hnd,
    int64_t v_scale_layout, int64_t stream_handle) {
  const auto kernel = Sm120Nvfp4D256PagedKernelConfig();
  sm120_nvfp4_paged::RunPagedBatchBf16QImpl(
      kernel, q, q_packed, q_scales, k_pages, k_sf_pages, v_pages_pv,
      v_sf_pages_pv, block_tables, qo_indptr, kv_lens, q_packed_scratch,
      q_scales_scratch, partial, split_m, split_l, out_scratch, out,
      workspace, max_physical_kv_len, qk_alpha, pv_alpha, kv_head,
      split_kv_tiles, group_size, causal, sliding_window, logits_soft_cap,
      output_group_span, kv_layout_hnd, v_scale_layout, stream_handle);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(paged_run_bf16_q,
                              SM120Nvfp4FmhaRunPagedBatchBf16Q);

}  // namespace
}  // namespace flashinfer
