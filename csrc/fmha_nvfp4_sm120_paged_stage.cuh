/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_paged_kv.cuh>

namespace flashinfer {
namespace sm120_nvfp4_paged {

using PagedParams =
    attention::blackwell::sm120_nvfp4::Sm120Nvfp4PagedKvLoadParams;

using RawPagedRunFn = cudaError_t (*)(
    uint8_t* q_packed, uint8_t* q_scales, uint8_t* k_packed,
    uint8_t* k_scales, uint8_t* v_pv_packed, uint8_t* v_pv_scales,
    __nv_bfloat16* partial, float* split_m, float* split_l,
    __nv_bfloat16* out, uint8_t* workspace, size_t workspace_bytes,
    float qk_alpha, float pv_alpha, int split_kv_tiles, int q_len,
    int group_size, int kv_len_tokens, bool causal, int sliding_window,
    float logits_soft_cap, int q_rows, int kv_len, cudaStream_t stream,
    PagedParams paged_params, const int32_t* qo_indptr,
    const int32_t* kv_lens, int batch_size, int q_tiles_per_sequence,
    bool skip_internal_combine);

using RawDenseRunFn = cudaError_t (*)(
    uint8_t* q_packed, uint8_t* q_scales, uint8_t* k_packed,
    uint8_t* k_scales, uint8_t* v_pv_packed, uint8_t* v_pv_scales,
    __nv_bfloat16* partial, float* split_m, float* split_l,
    __nv_bfloat16* out, uint8_t* workspace, size_t workspace_bytes,
    float qk_alpha, float pv_alpha, int split_kv_tiles, int q_len,
    int group_size, int kv_len_tokens, bool causal, int sliding_window,
    float logits_soft_cap, int q_rows, int kv_len, cudaStream_t stream);

struct PagedKernelConfig {
  int head_dim;
  int tile_m;
  int output_group_span;
  bool causal;
  bool use_sliding_window;
  bool use_logits_soft_cap;
  RawPagedRunFn run;
};

struct DenseKernelConfig {
  int head_dim;
  int tile_m;
  int output_group_span;
  bool causal;
  bool use_sliding_window;
  bool use_logits_soft_cap;
  RawDenseRunFn run;
};

}  // namespace sm120_nvfp4_paged
}  // namespace flashinfer
