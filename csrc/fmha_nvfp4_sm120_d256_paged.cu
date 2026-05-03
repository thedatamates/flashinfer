/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_d256.cuh>

#include "fmha_nvfp4_sm120_config.inc"
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

void SM120Nvfp4FmhaD256DebugProducerSmem(
    TensorView k_pages, TensorView k_sf_pages, TensorView v_pages,
    TensorView v_sf_pages, TensorView block_tables, TensorView k_smem_b,
    TensorView k_smem_sfb, TensorView v_smem_b, TensorView v_smem_sfb,
    TensorView sizes, int64_t kv_len_tokens, int64_t kv_head,
    int64_t kv_tile, int64_t k_outer, int64_t out_group_idx,
    bool kv_layout_hnd, int64_t v_scale_layout, int64_t stream_handle) {
  CheckCudaTypeLastDimContiguous(k_pages, dl_uint8, "k_pages");
  CheckCudaTypeLastDimContiguous(k_sf_pages, dl_uint8, "k_sf_pages");
  CheckCudaTypeLastDimContiguous(v_pages, dl_uint8, "v_pages");
  CheckCudaTypeLastDimContiguous(v_sf_pages, dl_uint8, "v_sf_pages");
  CHECK_INPUT_AND_TYPE(block_tables, dl_int32);
  CHECK_INPUT_AND_TYPE(k_smem_b, dl_uint8);
  CHECK_INPUT_AND_TYPE(k_smem_sfb, dl_uint8);
  CHECK_INPUT_AND_TYPE(v_smem_b, dl_uint8);
  CHECK_INPUT_AND_TYPE(v_smem_sfb, dl_uint8);
  CHECK_INPUT_AND_TYPE(sizes, dl_int32);
  CHECK_DIM(4, k_pages);
  CHECK_DIM(4, k_sf_pages);
  CHECK_DIM(4, v_pages);
  CHECK_DIM(4, v_sf_pages);
  CHECK_DIM(2, block_tables);
  CHECK_DIM(1, k_smem_b);
  CHECK_DIM(1, k_smem_sfb);
  CHECK_DIM(1, v_smem_b);
  CHECK_DIM(1, v_smem_sfb);
  CHECK_DIM(1, sizes);
  TVM_FFI_ICHECK_GE(sizes.size(0), 4);
  TVM_FFI_ICHECK(v_scale_layout == 0 || v_scale_layout == 1)
      << "v_scale_layout must be 0 (trtllm_interleaved) or 1 (linear)";
  ffi::CUDADeviceGuard device_guard(k_pages.device().device_id);
  const cudaStream_t stream = stream_from_handle(stream_handle);
  const int64_t page_size = kv_layout_hnd ? k_pages.size(2) : k_pages.size(1);
  const int64_t packed_dim = k_pages.size(3);
  const int64_t scale_dim = k_sf_pages.size(3);
  TVM_FFI_ICHECK_EQ(page_size, 16);
  TVM_FFI_ICHECK_EQ(packed_dim * 2, d256::kHeadDim);
  TVM_FFI_ICHECK_EQ(scale_dim * 16, d256::kHeadDim);
  TVM_FFI_ICHECK_GE(kv_len_tokens, 0);
  TVM_FFI_ICHECK_GE(kv_head, 0);

  PagedParams params{};
  params.k_pages = static_cast<const uint8_t*>(k_pages.data_ptr());
  params.k_scales = static_cast<const uint8_t*>(k_sf_pages.data_ptr());
  params.v_pages = static_cast<const uint8_t*>(v_pages.data_ptr());
  params.v_scales = static_cast<const uint8_t*>(v_sf_pages.data_ptr());
  params.block_table = static_cast<const int32_t*>(block_tables.data_ptr());
  params.block_table_stride = block_tables.stride(0);
  params.k_stride_page = k_pages.stride(0);
  params.k_stride_dim1 = k_pages.stride(1);
  params.k_stride_dim2 = k_pages.stride(2);
  params.k_stride_dim3 = k_pages.stride(3);
  params.k_scale_stride_page = k_sf_pages.stride(0);
  params.k_scale_stride_dim1 = k_sf_pages.stride(1);
  params.k_scale_stride_dim2 = k_sf_pages.stride(2);
  params.k_scale_stride_dim3 = k_sf_pages.stride(3);
  params.v_stride_page = v_pages.stride(0);
  params.v_stride_dim1 = v_pages.stride(1);
  params.v_stride_dim2 = v_pages.stride(2);
  params.v_stride_dim3 = v_pages.stride(3);
  params.v_scale_stride_page = v_sf_pages.stride(0);
  params.v_scale_stride_dim1 = v_sf_pages.stride(1);
  params.v_scale_stride_dim2 = v_sf_pages.stride(2);
  params.v_scale_stride_dim3 = v_sf_pages.stride(3);
  params.kv_head = static_cast<int>(kv_head);
  params.page_size = static_cast<int>(page_size);
  params.packed_dim = static_cast<int>(packed_dim);
  params.scale_dim = static_cast<int>(scale_dim);
  params.kv_layout_hnd = kv_layout_hnd ? 1 : 0;
  params.v_scale_layout = static_cast<int>(v_scale_layout);

  auto status = d256::sm120_nvfp4_d256_debug_producer_smem_raw<
      SM120_NVFP4_USE_PV_LAYOUT_V>(
      ToD256PagedParams(params), static_cast<int>(kv_len_tokens),
      static_cast<int>(kv_tile), static_cast<int>(k_outer),
      static_cast<int>(out_group_idx),
      static_cast<uint8_t*>(k_smem_b.data_ptr()),
      static_cast<uint8_t*>(k_smem_sfb.data_ptr()),
      static_cast<uint8_t*>(v_smem_b.data_ptr()),
      static_cast<uint8_t*>(v_smem_sfb.data_ptr()),
      static_cast<int32_t*>(sizes.data_ptr()), stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "SM120 NVFP4 D256 producer diagnostic failed: "
      << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(debug_producer_smem,
                              SM120Nvfp4FmhaD256DebugProducerSmem);

void SM120Nvfp4FmhaD256DebugStageRun(
    TensorView q, TensorView q_packed_scratch, TensorView q_scales_scratch,
    TensorView k_pages, TensorView k_sf_pages, TensorView v_pages,
    TensorView v_sf_pages, TensorView block_tables, TensorView qo_indptr,
    TensorView kv_lens, TensorView partial, TensorView split_m,
    TensorView split_l, TensorView out_scratch, TensorView workspace,
    TensorView debug_logits, TensorView debug_p_smem, TensorView debug_p_sfa,
    TensorView debug_o_smem, TensorView debug_out_store, TensorView debug_stats,
    TensorView debug_sizes,
    int64_t max_physical_kv_len, double qk_alpha, double pv_alpha,
    int64_t split_kv_tiles, int64_t group_size, bool causal,
    int64_t sliding_window, double logits_soft_cap, int64_t output_group_span,
    bool kv_layout_hnd, int64_t v_scale_layout, int64_t debug_block_x,
    int64_t debug_block_y, int64_t debug_block_z, int64_t debug_tile,
    int64_t stream_handle) {
  CHECK_INPUT_AND_TYPE(q, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(q_packed_scratch, dl_uint8);
  CHECK_INPUT_AND_TYPE(q_scales_scratch, dl_uint8);
  CheckCudaTypeLastDimContiguous(k_pages, dl_uint8, "k_pages");
  CheckCudaTypeLastDimContiguous(k_sf_pages, dl_uint8, "k_sf_pages");
  CheckCudaTypeLastDimContiguous(v_pages, dl_uint8, "v_pages");
  CheckCudaTypeLastDimContiguous(v_sf_pages, dl_uint8, "v_sf_pages");
  CHECK_INPUT_AND_TYPE(block_tables, dl_int32);
  CHECK_INPUT_AND_TYPE(qo_indptr, dl_int32);
  CHECK_INPUT_AND_TYPE(kv_lens, dl_int32);
  CHECK_INPUT_AND_TYPE(partial, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(split_m, dl_float32);
  CHECK_INPUT_AND_TYPE(split_l, dl_float32);
  CHECK_INPUT_AND_TYPE(out_scratch, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_INPUT_AND_TYPE(debug_logits, dl_uint8);
  CHECK_INPUT_AND_TYPE(debug_p_smem, dl_uint8);
  CHECK_INPUT_AND_TYPE(debug_p_sfa, dl_uint8);
  CHECK_INPUT_AND_TYPE(debug_o_smem, dl_uint8);
  CHECK_INPUT_AND_TYPE(debug_out_store, dl_uint8);
  CHECK_INPUT_AND_TYPE(debug_stats, dl_float32);
  CHECK_INPUT_AND_TYPE(debug_sizes, dl_int32);
  TVM_FFI_ICHECK(q.ndim() == 2 || q.ndim() == 3)
      << "q must have shape [rows, D] or [q_len, heads, D]";
  CheckCudaTypeLastDimContiguous(q, dl_bfloat16, "q");
  CHECK_DIM(2, q_packed_scratch);
  CHECK_DIM(2, q_scales_scratch);
  CHECK_DIM(4, k_pages);
  CHECK_DIM(4, k_sf_pages);
  CHECK_DIM(4, v_pages);
  CHECK_DIM(4, v_sf_pages);
  CHECK_DIM(2, block_tables);
  CHECK_DIM(1, qo_indptr);
  CHECK_DIM(1, kv_lens);
  CHECK_DIM(3, partial);
  CHECK_DIM(2, split_m);
  CHECK_DIM(2, split_l);
  CHECK_DIM(2, out_scratch);
  CHECK_DIM(1, workspace);
  CHECK_DIM(1, debug_logits);
  CHECK_DIM(1, debug_p_smem);
  CHECK_DIM(1, debug_p_sfa);
  CHECK_DIM(1, debug_o_smem);
  CHECK_DIM(1, debug_out_store);
  CHECK_DIM(1, debug_stats);
  CHECK_DIM(1, debug_sizes);
  TVM_FFI_ICHECK_GE(debug_logits.size(0), d256::kSm120Nvfp4LogitsBytes);
  TVM_FFI_ICHECK_GE(debug_p_smem.size(0), d256::kSm120Nvfp4PvPStageBytes);
  TVM_FFI_ICHECK_GE(debug_p_sfa.size(0),
                    d256::kSm120Nvfp4PvLogicalScaleStageElems);
  TVM_FFI_ICHECK_GE(debug_o_smem.size(0),
                    2 * d256::kCutlassTileM * d256::kOutputTileN *
                        static_cast<int64_t>(sizeof(__nv_bfloat16)));
  TVM_FFI_ICHECK_GE(debug_out_store.size(0),
                    2 * d256::kCutlassTileM * d256::kOutputTileN *
                        static_cast<int64_t>(sizeof(__nv_bfloat16)));
  TVM_FFI_ICHECK_GE(debug_stats.size(0), 2 * d256::kCutlassTileM);
  TVM_FFI_ICHECK_GE(debug_sizes.size(0), 9);
  TVM_FFI_ICHECK_EQ(output_group_span, 2);
  TVM_FFI_ICHECK_EQ(causal, SM120_NVFP4_CAUSAL)
      << "debug_stage_run causal mode must match generated module";
  TVM_FFI_ICHECK_EQ(sliding_window > 0, SM120_NVFP4_USE_SLIDING_WINDOW)
      << "debug_stage_run sliding-window mode must match generated module";
  TVM_FFI_ICHECK_EQ(logits_soft_cap > 0.0,
                    SM120_NVFP4_USE_LOGITS_SOFT_CAP)
      << "debug_stage_run soft-cap mode must match generated module";
  TVM_FFI_ICHECK(v_scale_layout == 0 || v_scale_layout == 1)
      << "v_scale_layout must be 0 (trtllm_interleaved) or 1 (linear)";

  ffi::CUDADeviceGuard device_guard(q_packed_scratch.device().device_id);
  const cudaStream_t stream = stream_from_handle(stream_handle);
  const int64_t batch = block_tables.size(0);
  const int64_t page_size = kv_layout_hnd ? k_pages.size(2) : k_pages.size(1);
  const int64_t num_kv_heads =
      kv_layout_hnd ? k_pages.size(1) : k_pages.size(2);
  const int64_t packed_dim = k_pages.size(3);
  const int64_t scale_dim = k_sf_pages.size(3);
  TVM_FFI_ICHECK_EQ(page_size, 16);
  TVM_FFI_ICHECK_EQ(packed_dim * 2, d256::kHeadDim);
  TVM_FFI_ICHECK_EQ(scale_dim * 16, d256::kHeadDim);
  const int64_t q_rows = q.ndim() == 3 ? q.size(0) * q.size(1) : q.size(0);
  TVM_FFI_ICHECK_EQ(q.ndim() == 3 ? q.size(2) : q.size(1), d256::kHeadDim);
  TVM_FFI_ICHECK_EQ(q_rows % group_size, 0);
  TVM_FFI_ICHECK_EQ(q_packed_scratch.size(1), packed_dim);
  TVM_FFI_ICHECK_EQ(q_scales_scratch.size(0), q_packed_scratch.size(0));
  TVM_FFI_ICHECK_EQ(q_scales_scratch.size(1), scale_dim);
  TVM_FFI_ICHECK_EQ(qo_indptr.size(0), batch + 1);
  TVM_FFI_ICHECK_EQ(kv_lens.size(0), batch);
  TVM_FFI_ICHECK(q_packed_scratch.size(0) % (batch * num_kv_heads) == 0)
      << "q scratch rows must be divisible by batch*num_kv_heads";
  const int64_t padded_q_rows_per_seq =
      q_packed_scratch.size(0) / (batch * num_kv_heads);
  TVM_FFI_ICHECK(padded_q_rows_per_seq > 0 &&
                 padded_q_rows_per_seq % d256::kCutlassTileM == 0)
      << "per-sequence q scratch rows must be a positive multiple of D256 tile_m";
  const int64_t q_tiles_per_sequence =
      padded_q_rows_per_seq / d256::kCutlassTileM;
  TVM_FFI_ICHECK_EQ(partial.size(1), q_packed_scratch.size(0));
  TVM_FFI_ICHECK_EQ(partial.size(2), d256::kHeadDim);
  TVM_FFI_ICHECK_EQ(split_m.size(0), partial.size(0));
  TVM_FFI_ICHECK_EQ(split_m.size(1), q_packed_scratch.size(0));
  TVM_FFI_ICHECK_EQ(split_l.size(0), partial.size(0));
  TVM_FFI_ICHECK_EQ(split_l.size(1), q_packed_scratch.size(0));
  TVM_FFI_ICHECK_EQ(out_scratch.size(0), q_packed_scratch.size(0));
  TVM_FFI_ICHECK_EQ(out_scratch.size(1), d256::kHeadDim);
  TVM_FFI_ICHECK(max_physical_kv_len > 0 && max_physical_kv_len % 128 == 0);

  PagedParams params{};
  params.k_pages = static_cast<const uint8_t*>(k_pages.data_ptr());
  params.k_scales = static_cast<const uint8_t*>(k_sf_pages.data_ptr());
  params.v_pages = static_cast<const uint8_t*>(v_pages.data_ptr());
  params.v_scales = static_cast<const uint8_t*>(v_sf_pages.data_ptr());
  params.block_table = static_cast<const int32_t*>(block_tables.data_ptr());
  params.block_table_stride = block_tables.stride(0);
  params.k_stride_page = k_pages.stride(0);
  params.k_stride_dim1 = k_pages.stride(1);
  params.k_stride_dim2 = k_pages.stride(2);
  params.k_stride_dim3 = k_pages.stride(3);
  params.k_scale_stride_page = k_sf_pages.stride(0);
  params.k_scale_stride_dim1 = k_sf_pages.stride(1);
  params.k_scale_stride_dim2 = k_sf_pages.stride(2);
  params.k_scale_stride_dim3 = k_sf_pages.stride(3);
  params.v_stride_page = v_pages.stride(0);
  params.v_stride_dim1 = v_pages.stride(1);
  params.v_stride_dim2 = v_pages.stride(2);
  params.v_stride_dim3 = v_pages.stride(3);
  params.v_scale_stride_page = v_sf_pages.stride(0);
  params.v_scale_stride_dim1 = v_sf_pages.stride(1);
  params.v_scale_stride_dim2 = v_sf_pages.stride(2);
  params.v_scale_stride_dim3 = v_sf_pages.stride(3);
  params.kv_head = 0;
  params.page_size = static_cast<int>(page_size);
  params.packed_dim = static_cast<int>(packed_dim);
  params.scale_dim = static_cast<int>(scale_dim);
  params.kv_layout_hnd = kv_layout_hnd ? 1 : 0;
  params.v_scale_layout = static_cast<int>(v_scale_layout);
  params.q_bf16 = static_cast<const __nv_bfloat16*>(q.data_ptr());
  params.q_is_3d = q.ndim() == 3 ? 1 : 0;
  params.q_stride_token = q.ndim() == 3 ? q.stride(0) : 0;
  params.q_stride_head = q.ndim() == 3 ? q.stride(1) : 0;
  params.q_stride_dim = q.stride(q.ndim() - 1);
  params.q_stride_row = q.ndim() == 2 ? q.stride(0) : 0;

  d256::Sm120Nvfp4D256StageDebugParams debug{};
  debug.logits = static_cast<uint8_t*>(debug_logits.data_ptr());
  debug.p_smem = static_cast<uint8_t*>(debug_p_smem.data_ptr());
  debug.p_sfa = static_cast<uint8_t*>(debug_p_sfa.data_ptr());
  debug.o_smem = static_cast<uint8_t*>(debug_o_smem.data_ptr());
  debug.out_store = static_cast<uint8_t*>(debug_out_store.data_ptr());
  debug.stats = static_cast<float*>(debug_stats.data_ptr());
  debug.sizes = static_cast<int32_t*>(debug_sizes.data_ptr());
  debug.block_x = static_cast<int>(debug_block_x);
  debug.block_y = static_cast<int>(debug_block_y);
  debug.block_z = static_cast<int>(debug_block_z);
  debug.tile = static_cast<int>(debug_tile);

  auto status = d256::sm120_nvfp4_qkv_online_register_q_splitkv_full_grid_raw<
      2, true, SM120_NVFP4_CAUSAL, SM120_NVFP4_USE_SLIDING_WINDOW,
      SM120_NVFP4_USE_LOGITS_SOFT_CAP, SM120_NVFP4_USE_PV_LAYOUT_V>(
      static_cast<uint8_t*>(q_packed_scratch.data_ptr()),
      static_cast<uint8_t*>(q_scales_scratch.data_ptr()),
      static_cast<uint8_t*>(k_pages.data_ptr()),
      static_cast<uint8_t*>(k_sf_pages.data_ptr()),
      static_cast<uint8_t*>(v_pages.data_ptr()),
      static_cast<uint8_t*>(v_sf_pages.data_ptr()),
      static_cast<__nv_bfloat16*>(partial.data_ptr()),
      static_cast<float*>(split_m.data_ptr()),
      static_cast<float*>(split_l.data_ptr()),
      static_cast<__nv_bfloat16*>(out_scratch.data_ptr()),
      static_cast<uint8_t*>(workspace.data_ptr()),
      static_cast<size_t>(workspace.size(0)),
      static_cast<float>(qk_alpha), static_cast<float>(pv_alpha),
      static_cast<int>(split_kv_tiles), 0, static_cast<int>(group_size), 0,
      causal, static_cast<int>(sliding_window),
      static_cast<float>(logits_soft_cap),
      static_cast<int>(q_packed_scratch.size(0)), d256::kHeadDim,
      static_cast<int>(max_physical_kv_len), stream, ToD256PagedParams(params),
      static_cast<const int32_t*>(qo_indptr.data_ptr()),
      static_cast<const int32_t*>(kv_lens.data_ptr()), static_cast<int>(batch),
      static_cast<int>(q_tiles_per_sequence), static_cast<int>(num_kv_heads),
      true, true, debug);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "SM120 NVFP4 D256 stage diagnostic failed: "
      << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(debug_stage_run,
                              SM120Nvfp4FmhaD256DebugStageRun);

}  // namespace
}  // namespace flashinfer
