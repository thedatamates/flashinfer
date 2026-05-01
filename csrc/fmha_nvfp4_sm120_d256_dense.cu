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
using sm120_nvfp4_paged::DenseKernelConfig;

cudaError_t Sm120Nvfp4D256RunDenseRaw(
    uint8_t* q_packed, uint8_t* q_scales, uint8_t* k_packed,
    uint8_t* k_scales, uint8_t* v_pv_packed, uint8_t* v_pv_scales,
    __nv_bfloat16* partial, float* split_m, float* split_l,
    __nv_bfloat16* out, uint8_t* workspace, size_t workspace_bytes,
    float qk_alpha, float pv_alpha, int split_kv_tiles, int q_len,
    int group_size, int kv_len_tokens, bool causal, int sliding_window,
    float logits_soft_cap, int q_rows, int kv_len, cudaStream_t stream) {
  return d256::sm120_nvfp4_qkv_online_register_q_splitkv_full_grid_raw<
      2, false, SM120_NVFP4_CAUSAL, SM120_NVFP4_USE_SLIDING_WINDOW,
      SM120_NVFP4_USE_LOGITS_SOFT_CAP, false>(
      q_packed, q_scales, k_packed, k_scales, v_pv_packed, v_pv_scales,
      partial, split_m, split_l, out, workspace, workspace_bytes, qk_alpha,
      pv_alpha, split_kv_tiles, q_len, group_size, kv_len_tokens, causal,
      sliding_window, logits_soft_cap, q_rows, d256::kHeadDim, kv_len, stream,
      {}, nullptr, nullptr, 1, 0, true);
}

DenseKernelConfig Sm120Nvfp4D256DenseKernelConfig() {
  return {d256::kHeadDim,
          d256::kCutlassTileM,
          2,
          SM120_NVFP4_CAUSAL,
          SM120_NVFP4_USE_SLIDING_WINDOW,
          SM120_NVFP4_USE_LOGITS_SOFT_CAP,
          Sm120Nvfp4D256RunDenseRaw};
}

void SM120Nvfp4FmhaRunDense(
    TensorView q_packed, TensorView q_scales, TensorView k_packed,
    TensorView k_scales, TensorView v_pv_packed, TensorView v_pv_scales,
    TensorView partial, TensorView split_m, TensorView split_l,
    TensorView out, TensorView workspace, double qk_alpha, double pv_alpha,
    int64_t split_kv_tiles, int64_t q_len, int64_t group_size,
    int64_t kv_len_tokens, bool causal, int64_t sliding_window,
    double logits_soft_cap, int64_t output_group_span) {
  const auto kernel = Sm120Nvfp4D256DenseKernelConfig();
  sm120_nvfp4_paged::RunDenseImpl(
      kernel, q_packed, q_scales, k_packed, k_scales, v_pv_packed,
      v_pv_scales, partial, split_m, split_l, out, workspace, qk_alpha,
      pv_alpha, split_kv_tiles, q_len, group_size, kv_len_tokens, causal,
      sliding_window, logits_soft_cap, output_group_span);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dense_run, SM120Nvfp4FmhaRunDense);

}  // namespace
}  // namespace flashinfer
