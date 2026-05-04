/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_quantization.cuh>

#include "fmha_nvfp4_sm120_paged_stage.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace sm120_nvfp4_paged {

namespace sm120 = attention::blackwell::sm120_nvfp4;

static __global__ void CopyQToPaddedBatchKernel(
    const uint8_t* q_packed, const uint8_t* q_scales,
    const int32_t* qo_indptr, uint8_t* q_packed_scratch,
    uint8_t* q_scales_scratch, int batch_size, int group_size,
    int num_kv_heads, bool all_kv_heads, int padded_q_rows_per_seq,
    int packed_dim, int scale_dim) {
  const int global_row = static_cast<int>(blockIdx.x);
  if (padded_q_rows_per_seq <= 0) {
    return;
  }
  const int rows_per_kv_head = batch_size * padded_q_rows_per_seq;
  const int kv_head = all_kv_heads ? global_row / rows_per_kv_head : 0;
  const int head_row =
      all_kv_heads ? global_row - kv_head * rows_per_kv_head : global_row;
  const int batch_idx = head_row / padded_q_rows_per_seq;
  const int local_row = head_row - batch_idx * padded_q_rows_per_seq;
  if (batch_idx >= batch_size || kv_head >= num_kv_heads) {
    return;
  }
  const int q_begin = qo_indptr[batch_idx];
  const int q_end = qo_indptr[batch_idx + 1];
  const int q_rows = (q_end - q_begin) * group_size;
  const int token_offset = local_row / group_size;
  const int group_offset = local_row - token_offset * group_size;
  const int src_row =
      all_kv_heads
          ? (q_begin + token_offset) * (num_kv_heads * group_size) +
                kv_head * group_size + group_offset
          : q_begin * group_size + local_row;
  for (int col = static_cast<int>(threadIdx.x); col < packed_dim;
       col += static_cast<int>(blockDim.x)) {
    q_packed_scratch[global_row * packed_dim + col] =
        local_row < q_rows ? q_packed[src_row * packed_dim + col] : 0;
  }
  for (int col = static_cast<int>(threadIdx.x); col < scale_dim;
       col += static_cast<int>(blockDim.x)) {
    q_scales_scratch[global_row * scale_dim + col] =
        local_row < q_rows ? q_scales[src_row * scale_dim + col] : 0;
  }
}

static __global__ void QuantizeQToPaddedBatchKernel(
    const __nv_bfloat16* q, const int32_t* qo_indptr,
    uint8_t* q_packed_scratch, uint8_t* q_scales_scratch, int batch_size,
    int group_size, int num_kv_heads, bool all_kv_heads,
    int padded_q_rows_per_seq, int head_dim, int packed_dim, int scale_dim,
    bool q_is_3d, int64_t q_stride_token, int64_t q_stride_head,
    int64_t q_stride_dim, int64_t q_stride_row) {
  const int global_row = static_cast<int>(blockIdx.x);
  const int scale_col = static_cast<int>(threadIdx.x);
  if (padded_q_rows_per_seq <= 0 || scale_col >= scale_dim) {
    return;
  }

  const int rows_per_kv_head = batch_size * padded_q_rows_per_seq;
  const int kv_head = all_kv_heads ? global_row / rows_per_kv_head : 0;
  const int head_row =
      all_kv_heads ? global_row - kv_head * rows_per_kv_head : global_row;
  const int batch_idx = head_row / padded_q_rows_per_seq;
  const int local_row = head_row - batch_idx * padded_q_rows_per_seq;
  if (batch_idx >= batch_size || kv_head >= num_kv_heads) {
    return;
  }

  const int q_begin = qo_indptr[batch_idx];
  const int q_end = qo_indptr[batch_idx + 1];
  const int q_rows = (q_end - q_begin) * group_size;
  const int token_offset = local_row / group_size;
  const int group_offset = local_row - token_offset * group_size;
  const int num_qo_heads =
      (all_kv_heads ? num_kv_heads : 1) * group_size;
  const bool valid_row = local_row < q_rows;
  const int src_row =
      all_kv_heads
          ? (q_begin + token_offset) * num_qo_heads + kv_head * group_size +
                group_offset
          : q_begin * group_size + local_row;
  const int dim_base = scale_col * 16;
  const int token = all_kv_heads ? q_begin + token_offset : src_row / group_size;
  const int head =
      all_kv_heads ? kv_head * group_size + group_offset : src_row % group_size;
  const int64_t src_base =
      q_is_3d
          ? static_cast<int64_t>(token) * q_stride_token +
                static_cast<int64_t>(head) * q_stride_head +
                static_cast<int64_t>(dim_base) * q_stride_dim
          : static_cast<int64_t>(src_row) * q_stride_row +
                static_cast<int64_t>(dim_base) * q_stride_dim;
  const int dst_scale_idx = global_row * scale_dim + scale_col;
  const int dst_packed_base = global_row * packed_dim + scale_col * 8;

  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float value =
        valid_row
            ? __bfloat162float(
                  q[src_base + static_cast<int64_t>(i) * q_stride_dim])
            : 0.0f;
    max_abs = fmaxf(max_abs, fabsf(value));
  }

  const uint8_t scale_byte =
      sm120::fp32_to_e4m3_byte(fmaxf(max_abs / 6.0f, 1.0e-8f));
  q_scales_scratch[dst_scale_idx] = scale_byte;
  const float scale =
      fmaxf(sm120::e4m3_byte_to_fp32(scale_byte), 1.0e-8f);

#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const float x0 =
        valid_row
            ? __bfloat162float(
                  q[src_base + static_cast<int64_t>(2 * pair) * q_stride_dim]) /
                  scale
            : 0.0f;
    const float x1 = valid_row
                         ? __bfloat162float(
                               q[src_base + static_cast<int64_t>(2 * pair + 1) *
                                                q_stride_dim]) /
                               scale
                         : 0.0f;
    q_packed_scratch[dst_packed_base + pair] =
        sm120::fp32_pair_to_e2m1_byte(x0, x1);
  }
}

static __global__ void CopyPaddedBatchOutKernel(
    const __nv_bfloat16* out_scratch, __nv_bfloat16* out,
    const int32_t* qo_indptr, int batch_size, int group_size,
    int num_kv_heads, bool all_kv_heads, int padded_q_rows_per_seq,
    int head_dim) {
  const int global_row = static_cast<int>(blockIdx.x);
  if (padded_q_rows_per_seq <= 0) {
    return;
  }
  const int rows_per_kv_head = batch_size * padded_q_rows_per_seq;
  const int kv_head = all_kv_heads ? global_row / rows_per_kv_head : 0;
  const int head_row =
      all_kv_heads ? global_row - kv_head * rows_per_kv_head : global_row;
  const int batch_idx = head_row / padded_q_rows_per_seq;
  const int local_row = head_row - batch_idx * padded_q_rows_per_seq;
  if (batch_idx >= batch_size || kv_head >= num_kv_heads) {
    return;
  }
  const int q_begin = qo_indptr[batch_idx];
  const int q_end = qo_indptr[batch_idx + 1];
  const int q_rows = (q_end - q_begin) * group_size;
  if (local_row >= q_rows) {
    return;
  }
  const int token_offset = local_row / group_size;
  const int group_offset = local_row - token_offset * group_size;
  const int out_row =
      all_kv_heads
          ? (q_begin + token_offset) * (num_kv_heads * group_size) +
                kv_head * group_size + group_offset
          : q_begin * group_size + local_row;
  for (int col = static_cast<int>(threadIdx.x); col < head_dim;
       col += static_cast<int>(blockDim.x)) {
    out[out_row * head_dim + col] =
        out_scratch[global_row * head_dim + col];
  }
}

static __global__ void Sm120Nvfp4SplitKvCombineBatchKernel(
    const __nv_bfloat16* partial,
    const float* split_m,
    const float* split_l,
    __nv_bfloat16* out,
    const int32_t* qo_indptr,
    const int32_t* kv_lens,
    int batch_size,
    int group_size,
    int num_kv_heads,
    bool all_kv_heads,
    int padded_q_rows_per_seq,
    int split_kv_tiles,
    int max_splits,
    int head_dim) {
  const int global_row = static_cast<int>(blockIdx.x);
  if (padded_q_rows_per_seq <= 0) {
    return;
  }
  const int rows_per_kv_head = batch_size * padded_q_rows_per_seq;
  const int kv_head = all_kv_heads ? global_row / rows_per_kv_head : 0;
  const int head_row =
      all_kv_heads ? global_row - kv_head * rows_per_kv_head : global_row;
  const int batch_idx = head_row / padded_q_rows_per_seq;
  const int local_row = head_row - batch_idx * padded_q_rows_per_seq;
  if (batch_idx >= batch_size || kv_head >= num_kv_heads) {
    return;
  }
  const int q_begin = qo_indptr[batch_idx];
  const int q_end = qo_indptr[batch_idx + 1];
  const int q_rows = (q_end - q_begin) * group_size;
  if (local_row >= q_rows) {
    return;
  }
  const int kv_len_tokens = kv_lens[batch_idx];
  const int total_kv_tiles = (kv_len_tokens + 127) / 128;
  const int num_splits =
      (total_kv_tiles + split_kv_tiles - 1) / split_kv_tiles;
  if (num_splits <= 0 || num_splits > max_splits) {
    return;
  }

  extern __shared__ float split_weights[];
  const int total_padded_rows =
      (all_kv_heads ? num_kv_heads : 1) * batch_size * padded_q_rows_per_seq;
  if (threadIdx.x == 0) {
    float global_m = -INFINITY;
#pragma unroll 1
    for (int split = 0; split < num_splits; ++split) {
      global_m =
          fmaxf(global_m, split_m[split * total_padded_rows + global_row]);
    }

    float global_l = 0.0f;
#pragma unroll 1
    for (int split = 0; split < num_splits; ++split) {
      const int stats_idx = split * total_padded_rows + global_row;
      const float correction =
          sm120::finite_f32(global_m) ? __expf(split_m[stats_idx] - global_m)
                                      : 0.0f;
      split_weights[split] = correction;
      global_l += correction * split_l[stats_idx];
    }
    const float inv_global_l = global_l > 0.0f ? 1.0f / global_l : 0.0f;
#pragma unroll 1
    for (int split = 0; split < num_splits; ++split) {
      split_weights[split] *= inv_global_l;
    }
  }
  __syncthreads();

  const int token_offset = local_row / group_size;
  const int group_offset = local_row - token_offset * group_size;
  const int out_row =
      all_kv_heads
          ? (q_begin + token_offset) * (num_kv_heads * group_size) +
                kv_head * group_size + group_offset
          : q_begin * group_size + local_row;
  for (int col = static_cast<int>(threadIdx.x); col < head_dim;
       col += static_cast<int>(blockDim.x)) {
    float acc = 0.0f;
#pragma unroll 1
    for (int split = 0; split < num_splits; ++split) {
      const int partial_idx =
          (split * total_padded_rows + global_row) * head_dim + col;
      acc += split_weights[split] * __bfloat162float(partial[partial_idx]);
    }
    out[out_row * head_dim + col] = __float2bfloat16(acc);
  }
}

static void CheckCudaTypeLastDimContiguous(TensorView tensor, DLDataType dtype,
                                           const char* name) {
  CHECK_CUDA(tensor);
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dtype)
      << "Inconsistency of Tensor type: " << name;
  TVM_FFI_ICHECK_EQ(tensor.stride(tensor.ndim() - 1), 1)
      << name << " must be contiguous in the last dimension";
}

static void CheckDenseRunTensors(
    const DenseKernelConfig& kernel, TensorView q_packed, TensorView q_scales,
    TensorView k_packed, TensorView k_scales, TensorView v_pv_packed,
    TensorView v_pv_scales, TensorView partial, TensorView split_m,
    TensorView split_l, TensorView out, TensorView workspace,
    int64_t split_kv_tiles, int64_t q_len, int64_t group_size,
    int64_t kv_len_tokens, int64_t output_group_span, bool causal,
    int64_t sliding_window, double logits_soft_cap) {
  CHECK_INPUT_AND_TYPE(q_packed, dl_uint8);
  CHECK_INPUT_AND_TYPE(q_scales, dl_uint8);
  CHECK_INPUT_AND_TYPE(k_packed, dl_uint8);
  CHECK_INPUT_AND_TYPE(k_scales, dl_uint8);
  CHECK_INPUT_AND_TYPE(v_pv_packed, dl_uint8);
  CHECK_INPUT_AND_TYPE(v_pv_scales, dl_uint8);
  CHECK_INPUT_AND_TYPE(partial, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(split_m, dl_float32);
  CHECK_INPUT_AND_TYPE(split_l, dl_float32);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_DIM(2, q_packed);
  CHECK_DIM(2, q_scales);
  CHECK_DIM(2, k_packed);
  CHECK_DIM(2, k_scales);
  CHECK_DIM(2, v_pv_packed);
  CHECK_DIM(2, v_pv_scales);
  CHECK_DIM(3, partial);
  CHECK_DIM(2, split_m);
  CHECK_DIM(2, split_l);
  CHECK_DIM(2, out);
  CHECK_DIM(1, workspace);

  const int64_t q_rows = q_packed.size(0);
  const int64_t head_dim = q_packed.size(1) * 2;
  const int64_t scale_cols = head_dim / 16;
  const int64_t kv_len = k_packed.size(0);
  TVM_FFI_ICHECK_EQ(head_dim, kernel.head_dim)
      << "SM120 NVFP4 module/head_dim mismatch";
  TVM_FFI_ICHECK_EQ(causal, kernel.causal)
      << "SM120 NVFP4 module dense causal mode mismatch";
  TVM_FFI_ICHECK_EQ(sliding_window > 0, kernel.use_sliding_window)
      << "SM120 NVFP4 module dense sliding-window mode mismatch";
  TVM_FFI_ICHECK_EQ(logits_soft_cap > 0.0, kernel.use_logits_soft_cap)
      << "SM120 NVFP4 module dense logits-soft-cap mode mismatch";
  TVM_FFI_ICHECK(q_rows > 0 && q_rows % kernel.tile_m == 0)
      << "q rows must be a positive multiple of tile_m=" << kernel.tile_m;
  TVM_FFI_ICHECK(kv_len > 0 && kv_len % 128 == 0)
      << "KV length must be a positive multiple of 128";
  TVM_FFI_ICHECK(kv_len <= 262144) << "KV length exceeds 256K";
  TVM_FFI_ICHECK(split_kv_tiles > 0) << "split_kv_tiles must be positive";
  TVM_FFI_ICHECK(q_len > 0) << "q_len must be positive";
  TVM_FFI_ICHECK(group_size > 0) << "group_size must be positive";
  TVM_FFI_ICHECK(q_len * group_size <= q_rows)
      << "q_len * group_size must not exceed q_packed rows";
  TVM_FFI_ICHECK(kv_len_tokens > 0 && kv_len_tokens <= kv_len)
      << "kv_len_tokens must be positive and not exceed physical KV length";
  TVM_FFI_ICHECK_EQ(output_group_span, kernel.output_group_span)
      << "dense output_group_span does not match generated SM120 NVFP4 module";
  TVM_FFI_ICHECK_EQ(q_scales.size(0), q_rows);
  TVM_FFI_ICHECK_EQ(q_scales.size(1), scale_cols);
  TVM_FFI_ICHECK_EQ(k_packed.size(1), head_dim / 2);
  TVM_FFI_ICHECK_EQ(k_scales.size(0), kv_len);
  TVM_FFI_ICHECK_EQ(k_scales.size(1), scale_cols);
  TVM_FFI_ICHECK_EQ(v_pv_packed.size(0), head_dim);
  TVM_FFI_ICHECK_EQ(v_pv_packed.size(1), kv_len / 2);
  TVM_FFI_ICHECK_EQ(v_pv_scales.size(0), head_dim);
  TVM_FFI_ICHECK_EQ(v_pv_scales.size(1), kv_len / 16);
  const int64_t total_kv_tiles = kv_len / 128;
  const int64_t num_splits =
      (total_kv_tiles + split_kv_tiles - 1) / split_kv_tiles;
  TVM_FFI_ICHECK_EQ(partial.size(0), num_splits);
  TVM_FFI_ICHECK_EQ(partial.size(1), q_rows);
  TVM_FFI_ICHECK_EQ(partial.size(2), head_dim);
  TVM_FFI_ICHECK_EQ(split_m.size(0), num_splits);
  TVM_FFI_ICHECK_EQ(split_m.size(1), q_rows);
  TVM_FFI_ICHECK_EQ(split_l.size(0), num_splits);
  TVM_FFI_ICHECK_EQ(split_l.size(1), q_rows);
  TVM_FFI_ICHECK_EQ(out.size(0), q_rows);
  TVM_FFI_ICHECK_EQ(out.size(1), head_dim);
}

static __global__ void DenseSplitKvCombineKernel(const __nv_bfloat16* partial,
                                                 const float* split_m,
                                                 const float* split_l,
                                                 __nv_bfloat16* out,
                                                 int num_splits, int q_rows,
                                                 int head_dim) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= q_rows) {
    return;
  }
  extern __shared__ float split_weights[];

  if (threadIdx.x == 0) {
    float global_m = -INFINITY;
#pragma unroll 1
    for (int split = 0; split < num_splits; ++split) {
      global_m = fmaxf(global_m, split_m[split * q_rows + row]);
    }

    float global_l = 0.0f;
#pragma unroll 1
    for (int split = 0; split < num_splits; ++split) {
      const int stats_idx = split * q_rows + row;
      const float correction =
          sm120::finite_f32(global_m) ? __expf(split_m[stats_idx] - global_m)
                                      : 0.0f;
      split_weights[split] = correction;
      global_l += correction * split_l[stats_idx];
    }
    const float inv_global_l = global_l > 0.0f ? 1.0f / global_l : 0.0f;
#pragma unroll 1
    for (int split = 0; split < num_splits; ++split) {
      split_weights[split] *= inv_global_l;
    }
  }
  __syncthreads();

  for (int col = static_cast<int>(threadIdx.x); col < head_dim;
       col += static_cast<int>(blockDim.x)) {
    float acc = 0.0f;
#pragma unroll 1
    for (int split = 0; split < num_splits; ++split) {
      const int partial_idx = (split * q_rows + row) * head_dim + col;
      acc += split_weights[split] * __bfloat162float(partial[partial_idx]);
    }
    out[row * head_dim + col] = __float2bfloat16(acc);
  }
}

static void RunDenseImpl(
    const DenseKernelConfig& kernel, TensorView q_packed, TensorView q_scales,
    TensorView k_packed, TensorView k_scales, TensorView v_pv_packed,
    TensorView v_pv_scales, TensorView partial, TensorView split_m,
    TensorView split_l, TensorView out, TensorView workspace, double qk_alpha,
    double pv_alpha, int64_t split_kv_tiles, int64_t q_len,
    int64_t group_size, int64_t kv_len_tokens, bool causal,
    int64_t sliding_window, double logits_soft_cap,
    int64_t output_group_span, int64_t stream_handle) {
  CheckDenseRunTensors(kernel, q_packed, q_scales, k_packed, k_scales,
                       v_pv_packed, v_pv_scales, partial, split_m, split_l,
                       out, workspace, split_kv_tiles, q_len, group_size,
                       kv_len_tokens, output_group_span, causal,
                       sliding_window, logits_soft_cap);
  ffi::CUDADeviceGuard device_guard(q_packed.device().device_id);
  const cudaStream_t stream =
      stream_handle != 0 ? stream_from_handle(stream_handle)
                         : get_stream(q_packed.device());
  const int q_rows = static_cast<int>(q_packed.size(0));
  const int kv_len = static_cast<int>(k_packed.size(0));
  const int head_dim = kernel.head_dim;
  cudaError_t status = kernel.run(
      static_cast<uint8_t*>(q_packed.data_ptr()),
      static_cast<uint8_t*>(q_scales.data_ptr()),
      static_cast<uint8_t*>(k_packed.data_ptr()),
      static_cast<uint8_t*>(k_scales.data_ptr()),
      static_cast<uint8_t*>(v_pv_packed.data_ptr()),
      static_cast<uint8_t*>(v_pv_scales.data_ptr()),
      static_cast<__nv_bfloat16*>(partial.data_ptr()),
      static_cast<float*>(split_m.data_ptr()),
      static_cast<float*>(split_l.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<uint8_t*>(workspace.data_ptr()),
      workspace.size(0) * get_element_size(workspace),
      static_cast<float>(qk_alpha), static_cast<float>(pv_alpha),
      static_cast<int>(split_kv_tiles), static_cast<int>(q_len),
      static_cast<int>(group_size), static_cast<int>(kv_len_tokens), causal,
      static_cast<int>(sliding_window), static_cast<float>(logits_soft_cap),
      q_rows, kv_len, stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "SM120 NVFP4 FMHA dense run failed: " << cudaGetErrorString(status);

  const int total_kv_tiles = kv_len / 128;
  const int num_splits =
      (total_kv_tiles + static_cast<int>(split_kv_tiles) - 1) /
      static_cast<int>(split_kv_tiles);
  if (num_splits <= 1) {
    return;
  }
  DenseSplitKvCombineKernel<<<q_rows, 256,
                              static_cast<size_t>(num_splits) * sizeof(float),
                              stream>>>(
      static_cast<const __nv_bfloat16*>(partial.data_ptr()),
      static_cast<const float*>(split_m.data_ptr()),
      static_cast<const float*>(split_l.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()), num_splits, q_rows,
      head_dim);
  status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "SM120 NVFP4 FMHA dense combine failed: " << cudaGetErrorString(status);
}

static void RunPagedBatchImpl(
    const PagedKernelConfig& kernel,
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
    int64_t stream_handle,
    bool q_scratch_prepared = false, const __nv_bfloat16* q_bf16 = nullptr,
    bool q_is_3d = true, int64_t q_stride_token = 0,
    int64_t q_stride_head = 0, int64_t q_stride_dim = 0,
    int64_t q_stride_row = 0) {
  CHECK_INPUT_AND_TYPE(q_packed, dl_uint8);
  CHECK_INPUT_AND_TYPE(q_scales, dl_uint8);
  CheckCudaTypeLastDimContiguous(k_pages, dl_uint8, "k_pages");
  CheckCudaTypeLastDimContiguous(k_sf_pages, dl_uint8, "k_sf_pages");
  CheckCudaTypeLastDimContiguous(v_pages_pv, dl_uint8, "v_pages");
  CheckCudaTypeLastDimContiguous(v_sf_pages_pv, dl_uint8, "v_sf_pages");
  CHECK_INPUT_AND_TYPE(block_tables, dl_int32);
  CHECK_INPUT_AND_TYPE(q_packed_scratch, dl_uint8);
  CHECK_INPUT_AND_TYPE(q_scales_scratch, dl_uint8);
  CHECK_INPUT_AND_TYPE(partial, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(split_m, dl_float32);
  CHECK_INPUT_AND_TYPE(split_l, dl_float32);
  CHECK_INPUT_AND_TYPE(out_scratch, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_DIM(2, q_packed);
  CHECK_DIM(2, q_scales);
  CHECK_DIM(4, k_pages);
  CHECK_DIM(4, k_sf_pages);
  CHECK_DIM(4, v_pages_pv);
  CHECK_DIM(4, v_sf_pages_pv);
  CHECK_DIM(2, block_tables);
  CHECK_DIM(2, q_packed_scratch);
  CHECK_DIM(2, q_scales_scratch);
  CHECK_DIM(3, partial);
  CHECK_DIM(2, split_m);
  CHECK_DIM(2, split_l);
  CHECK_DIM(2, out_scratch);
  CHECK_DIM(2, out);
  CHECK_DIM(1, workspace);
  CHECK_INPUT_AND_TYPE(qo_indptr, dl_int32);
  CHECK_INPUT_AND_TYPE(kv_lens, dl_int32);
  CHECK_DIM(1, qo_indptr);
  CHECK_DIM(1, kv_lens);

  ffi::CUDADeviceGuard device_guard(q_packed.device().device_id);
  const cudaStream_t stream =
      stream_handle != 0 ? stream_from_handle(stream_handle)
                         : get_stream(q_packed.device());

  const int64_t batch = block_tables.size(0);
  TVM_FFI_ICHECK_EQ(qo_indptr.size(0), batch + 1);
  TVM_FFI_ICHECK_EQ(kv_lens.size(0), batch);
  const int64_t page_size = kv_layout_hnd ? k_pages.size(2) : k_pages.size(1);
  const int64_t num_kv_heads =
      kv_layout_hnd ? k_pages.size(1) : k_pages.size(2);
  const int64_t packed_dim = k_pages.size(3);
  const int64_t scale_dim = k_sf_pages.size(3);
  const int64_t head_dim = packed_dim * 2;
  TVM_FFI_ICHECK_EQ(head_dim, kernel.head_dim)
      << "SM120 NVFP4 module/head_dim mismatch";
  TVM_FFI_ICHECK_EQ(causal, kernel.causal)
      << "SM120 NVFP4 module was generated for causal=" << kernel.causal
      << " but paged_run received causal=" << causal;
  TVM_FFI_ICHECK_EQ(sliding_window > 0, kernel.use_sliding_window)
      << "SM120 NVFP4 module sliding-window mode mismatch";
  TVM_FFI_ICHECK_EQ(logits_soft_cap > 0.0, kernel.use_logits_soft_cap)
      << "SM120 NVFP4 module logits-soft-cap mode mismatch";
  TVM_FFI_ICHECK_EQ(page_size, 16)
      << "SM120 NVFP4 paged run currently supports page_size=16";
  TVM_FFI_ICHECK_EQ(k_sf_pages.size(kv_layout_hnd ? 2 : 1), page_size);
  TVM_FFI_ICHECK_EQ(k_sf_pages.size(kv_layout_hnd ? 1 : 2), num_kv_heads);
  TVM_FFI_ICHECK_EQ(v_pages_pv.size(kv_layout_hnd ? 2 : 1), page_size);
  TVM_FFI_ICHECK_EQ(v_pages_pv.size(kv_layout_hnd ? 1 : 2), num_kv_heads);
  TVM_FFI_ICHECK_EQ(v_pages_pv.size(3), packed_dim);
  TVM_FFI_ICHECK_EQ(v_sf_pages_pv.size(kv_layout_hnd ? 2 : 1), page_size);
  TVM_FFI_ICHECK_EQ(v_sf_pages_pv.size(kv_layout_hnd ? 1 : 2), num_kv_heads);
  TVM_FFI_ICHECK_EQ(v_sf_pages_pv.size(3), scale_dim);
  TVM_FFI_ICHECK(v_scale_layout == 0 || v_scale_layout == 1)
      << "v_scale_layout must be 0 (trtllm_interleaved) or 1 (linear)";
  TVM_FFI_ICHECK_EQ(scale_dim * 16, head_dim);
  const bool all_kv_heads = kv_head < 0;
  const int64_t run_kv_heads = all_kv_heads ? num_kv_heads : 1;
  TVM_FFI_ICHECK(all_kv_heads || kv_head < num_kv_heads)
      << "kv_head out of range";
  TVM_FFI_ICHECK(split_kv_tiles > 0) << "split_kv_tiles must be positive";
  TVM_FFI_ICHECK(group_size > 0) << "group_size must be positive";
  TVM_FFI_ICHECK_EQ(q_packed.size(0) % group_size, 0)
      << "q_packed rows must be divisible by group_size";
  if (all_kv_heads) {
    TVM_FFI_ICHECK_EQ(q_packed.size(0) % (num_kv_heads * group_size), 0)
        << "all-head paged_run expects Q rows laid out as "
           "[tokens, num_kv_heads * group_size]";
  }
  TVM_FFI_ICHECK_EQ(output_group_span, kernel.output_group_span)
      << "non-default output_group_span is not compiled into the production "
         "SM120 NVFP4 paged module";
  TVM_FFI_ICHECK_EQ(q_packed.size(1), packed_dim);
  TVM_FFI_ICHECK_EQ(q_scales.size(1), scale_dim);
  TVM_FFI_ICHECK_EQ(out.size(0), q_packed.size(0));
  TVM_FFI_ICHECK_EQ(out.size(1), head_dim);
  TVM_FFI_ICHECK_EQ(q_packed_scratch.size(1), packed_dim);
  TVM_FFI_ICHECK_EQ(q_scales_scratch.size(1), scale_dim);
  TVM_FFI_ICHECK_EQ(partial.size(1), q_packed_scratch.size(0));
  TVM_FFI_ICHECK_EQ(partial.size(2), head_dim);
  TVM_FFI_ICHECK_EQ(split_m.size(0), partial.size(0));
  TVM_FFI_ICHECK_EQ(split_m.size(1), q_packed_scratch.size(0));
  TVM_FFI_ICHECK_EQ(split_l.size(0), partial.size(0));
  TVM_FFI_ICHECK_EQ(split_l.size(1), q_packed_scratch.size(0));
  TVM_FFI_ICHECK_EQ(out_scratch.size(0), q_packed_scratch.size(0));
  TVM_FFI_ICHECK_EQ(out_scratch.size(1), head_dim);

  const size_t workspace_bytes =
      static_cast<size_t>(workspace.size(0) * get_element_size(workspace));
  uint8_t* q_scratch_ptr = static_cast<uint8_t*>(q_packed_scratch.data_ptr());
  uint8_t* q_sf_scratch_ptr =
      static_cast<uint8_t*>(q_scales_scratch.data_ptr());
  auto* partial_ptr = static_cast<__nv_bfloat16*>(partial.data_ptr());
  float* split_m_ptr = static_cast<float*>(split_m.data_ptr());
  float* split_l_ptr = static_cast<float*>(split_l.data_ptr());
  auto* out_scratch_ptr = static_cast<__nv_bfloat16*>(out_scratch.data_ptr());
  uint8_t* workspace_ptr = static_cast<uint8_t*>(workspace.data_ptr());
  TVM_FFI_ICHECK(batch > 0) << "batch must be positive";
  TVM_FFI_ICHECK(q_packed_scratch.size(0) % (batch * run_kv_heads) == 0)
      << "q scratch rows must be divisible by batch size and active KV heads";
  const int64_t padded_q_rows_per_seq =
      q_packed_scratch.size(0) / (batch * run_kv_heads);
  TVM_FFI_ICHECK(padded_q_rows_per_seq > 0 &&
                 padded_q_rows_per_seq % kernel.tile_m == 0)
      << "per-sequence q scratch rows must be a positive multiple of tile_m";
  const int64_t total_padded_q_rows = q_packed_scratch.size(0);
  const int64_t q_tiles_per_sequence = padded_q_rows_per_seq / kernel.tile_m;
  const int64_t physical_kv_len = max_physical_kv_len;
  TVM_FFI_ICHECK(physical_kv_len > 0 && physical_kv_len % 128 == 0)
      << "max physical KV length must be a positive multiple of 128";
  const int64_t max_total_kv_tiles = physical_kv_len / 128;
  const int64_t max_splits =
      (max_total_kv_tiles + split_kv_tiles - 1) / split_kv_tiles;
  TVM_FFI_ICHECK(max_splits > 0 && partial.size(0) >= max_splits)
      << "partial scratch does not cover split count";
  TVM_FFI_ICHECK(block_tables.size(1) >= physical_kv_len / page_size)
      << "block_tables rows do not cover max padded KV length";

  const int32_t* qo_indptr_ptr =
      static_cast<const int32_t*>(qo_indptr.data_ptr());
  const int32_t* kv_lens_ptr =
      static_cast<const int32_t*>(kv_lens.data_ptr());
  cudaError_t status = cudaSuccess;
  if (!q_scratch_prepared) {
    CopyQToPaddedBatchKernel<<<static_cast<unsigned>(total_padded_q_rows), 256,
                               0, stream>>>(
        static_cast<const uint8_t*>(q_packed.data_ptr()),
        static_cast<const uint8_t*>(q_scales.data_ptr()), qo_indptr_ptr,
        q_scratch_ptr, q_sf_scratch_ptr, static_cast<int>(batch),
        static_cast<int>(group_size), static_cast<int>(num_kv_heads),
        all_kv_heads, static_cast<int>(padded_q_rows_per_seq),
        static_cast<int>(packed_dim), static_cast<int>(scale_dim));
    status = cudaGetLastError();
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "SM120 NVFP4 batch Q pad copy failed: "
        << cudaGetErrorString(status);
  }

  PagedParams paged_params{};
  paged_params.k_pages = static_cast<const uint8_t*>(k_pages.data_ptr());
  paged_params.k_scales = static_cast<const uint8_t*>(k_sf_pages.data_ptr());
  paged_params.v_pages = static_cast<const uint8_t*>(v_pages_pv.data_ptr());
  paged_params.v_scales =
      static_cast<const uint8_t*>(v_sf_pages_pv.data_ptr());
  paged_params.block_table =
      static_cast<const int32_t*>(block_tables.data_ptr());
  paged_params.block_table_stride = block_tables.stride(0);
  paged_params.k_stride_page = k_pages.stride(0);
  paged_params.k_stride_dim1 = k_pages.stride(1);
  paged_params.k_stride_dim2 = k_pages.stride(2);
  paged_params.k_stride_dim3 = k_pages.stride(3);
  paged_params.k_scale_stride_page = k_sf_pages.stride(0);
  paged_params.k_scale_stride_dim1 = k_sf_pages.stride(1);
  paged_params.k_scale_stride_dim2 = k_sf_pages.stride(2);
  paged_params.k_scale_stride_dim3 = k_sf_pages.stride(3);
  paged_params.v_stride_page = v_pages_pv.stride(0);
  paged_params.v_stride_dim1 = v_pages_pv.stride(1);
  paged_params.v_stride_dim2 = v_pages_pv.stride(2);
  paged_params.v_stride_dim3 = v_pages_pv.stride(3);
  paged_params.v_scale_stride_page = v_sf_pages_pv.stride(0);
  paged_params.v_scale_stride_dim1 = v_sf_pages_pv.stride(1);
  paged_params.v_scale_stride_dim2 = v_sf_pages_pv.stride(2);
  paged_params.v_scale_stride_dim3 = v_sf_pages_pv.stride(3);
  paged_params.kv_head = all_kv_heads ? 0 : static_cast<int>(kv_head);
  paged_params.page_size = static_cast<int>(page_size);
  paged_params.packed_dim = static_cast<int>(packed_dim);
  paged_params.scale_dim = static_cast<int>(scale_dim);
  paged_params.kv_layout_hnd = kv_layout_hnd ? 1 : 0;
  paged_params.v_scale_layout = static_cast<int>(v_scale_layout);
  paged_params.q_bf16 = q_bf16;
  paged_params.q_is_3d = q_is_3d ? 1 : 0;
  paged_params.q_stride_token = q_stride_token;
  paged_params.q_stride_head = q_stride_head;
  paged_params.q_stride_dim = q_stride_dim;
  paged_params.q_stride_row = q_stride_row;

  status = kernel.run(
      q_scratch_ptr, q_sf_scratch_ptr,
      static_cast<uint8_t*>(k_pages.data_ptr()),
      static_cast<uint8_t*>(k_sf_pages.data_ptr()),
      static_cast<uint8_t*>(v_pages_pv.data_ptr()),
      static_cast<uint8_t*>(v_sf_pages_pv.data_ptr()), partial_ptr,
      split_m_ptr, split_l_ptr, out_scratch_ptr, workspace_ptr,
      workspace_bytes, static_cast<float>(qk_alpha),
      static_cast<float>(pv_alpha), static_cast<int>(split_kv_tiles), 0,
      static_cast<int>(group_size), 0, causal,
      static_cast<int>(sliding_window), static_cast<float>(logits_soft_cap),
      static_cast<int>(total_padded_q_rows),
      static_cast<int>(physical_kv_len), stream, paged_params, qo_indptr_ptr,
      kv_lens_ptr, static_cast<int>(batch),
      static_cast<int>(q_tiles_per_sequence), static_cast<int>(num_kv_heads),
      all_kv_heads, true);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "SM120 NVFP4 FMHA batch failed: " << cudaGetErrorString(status);

  if (max_splits == 1) {
    CopyPaddedBatchOutKernel<<<static_cast<unsigned>(total_padded_q_rows), 256,
                               0, stream>>>(
        out_scratch_ptr, static_cast<__nv_bfloat16*>(out.data_ptr()),
        qo_indptr_ptr, static_cast<int>(batch), static_cast<int>(group_size),
        static_cast<int>(num_kv_heads), all_kv_heads,
        static_cast<int>(padded_q_rows_per_seq), static_cast<int>(head_dim));
  } else {
    Sm120Nvfp4SplitKvCombineBatchKernel<<<
        static_cast<unsigned>(total_padded_q_rows), 256,
        static_cast<size_t>(max_splits) * sizeof(float), stream>>>(
        partial_ptr, split_m_ptr, split_l_ptr,
        static_cast<__nv_bfloat16*>(out.data_ptr()), qo_indptr_ptr,
        kv_lens_ptr, static_cast<int>(batch), static_cast<int>(group_size),
        static_cast<int>(num_kv_heads), all_kv_heads,
        static_cast<int>(padded_q_rows_per_seq),
        static_cast<int>(split_kv_tiles), static_cast<int>(max_splits),
        static_cast<int>(head_dim));
  }
  status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "SM120 NVFP4 batch output combine/copy failed: "
      << cudaGetErrorString(status);
}

static void RunPagedBatchBf16QImpl(
    const PagedKernelConfig& kernel,
    TensorView q, TensorView q_packed, TensorView q_scales, TensorView k_pages,
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
  CHECK_CUDA(q);
  TVM_FFI_ICHECK_EQ(q.dtype(), dl_bfloat16)
      << "Inconsistency of Tensor type: q";
  CHECK_INPUT_AND_TYPE(q_packed_scratch, dl_uint8);
  CHECK_INPUT_AND_TYPE(q_scales_scratch, dl_uint8);
  CHECK_INPUT_AND_TYPE(block_tables, dl_int32);
  CHECK_INPUT_AND_TYPE(qo_indptr, dl_int32);
  CheckCudaTypeLastDimContiguous(k_pages, dl_uint8, "k_pages");
  TVM_FFI_ICHECK(q.ndim() == 2 || q.ndim() == 3)
      << "q must have shape [rows, D] or [q_len, heads, D]";
  TVM_FFI_ICHECK_EQ(q.stride(q.ndim() - 1), 1)
      << "q must be contiguous in the last dimension";
  CHECK_DIM(2, q_packed_scratch);
  CHECK_DIM(2, q_scales_scratch);
  CHECK_DIM(2, block_tables);
  CHECK_DIM(1, qo_indptr);
  const int64_t q_rows = q.ndim() == 3 ? q.size(0) * q.size(1) : q.size(0);
  const int64_t head_dim = q.ndim() == 3 ? q.size(2) : q.size(1);
  TVM_FFI_ICHECK_EQ(head_dim, kernel.head_dim)
      << "SM120 NVFP4 module/head_dim mismatch";
  TVM_FFI_ICHECK_EQ(q_packed.ndim(), 2);
  TVM_FFI_ICHECK_EQ(q_packed.size(0), q_rows);
  TVM_FFI_ICHECK_EQ(q_packed.size(1), head_dim / 2);
  TVM_FFI_ICHECK_EQ(q_scales.ndim(), 2);
  TVM_FFI_ICHECK_EQ(q_scales.size(0), q_rows);
  TVM_FFI_ICHECK_EQ(q_scales.size(1), head_dim / 16);

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream =
      stream_handle != 0 ? stream_from_handle(stream_handle)
                         : get_stream(q.device());
  const int64_t batch = block_tables.size(0);
  const int64_t num_kv_heads =
      kv_layout_hnd ? k_pages.size(1) : k_pages.size(2);
  const bool all_kv_heads = kv_head < 0;
  const int64_t run_kv_heads = all_kv_heads ? num_kv_heads : 1;
  TVM_FFI_ICHECK(batch > 0) << "batch must be positive";
  TVM_FFI_ICHECK(run_kv_heads > 0) << "active KV heads must be positive";
  TVM_FFI_ICHECK(q_packed_scratch.size(0) % (batch * run_kv_heads) == 0)
      << "q scratch rows must be divisible by batch size and active KV heads";
  const int64_t padded_q_rows_per_seq =
      q_packed_scratch.size(0) / (batch * run_kv_heads);
  TVM_FFI_ICHECK_EQ(q_packed_scratch.size(1), head_dim / 2);
  TVM_FFI_ICHECK_EQ(q_scales_scratch.size(0), q_packed_scratch.size(0));
  TVM_FFI_ICHECK_EQ(q_scales_scratch.size(1), head_dim / 16);
  if (q.ndim() == 3) {
    const int64_t expected_heads =
        (all_kv_heads ? num_kv_heads : 1) * group_size;
    TVM_FFI_ICHECK_EQ(q.size(1), expected_heads)
        << "BF16 Q head count does not match active KV heads and group_size";
  }

  QuantizeQToPaddedBatchKernel<<<
      static_cast<unsigned>(q_packed_scratch.size(0)),
      static_cast<unsigned>(head_dim / 16), 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q.data_ptr()),
      static_cast<const int32_t*>(qo_indptr.data_ptr()),
      static_cast<uint8_t*>(q_packed_scratch.data_ptr()),
      static_cast<uint8_t*>(q_scales_scratch.data_ptr()),
      static_cast<int>(batch), static_cast<int>(group_size),
      static_cast<int>(num_kv_heads), all_kv_heads,
      static_cast<int>(padded_q_rows_per_seq), static_cast<int>(head_dim),
      static_cast<int>(head_dim / 2), static_cast<int>(head_dim / 16),
      q.ndim() == 3, q.ndim() == 3 ? q.stride(0) : 0,
      q.ndim() == 3 ? q.stride(1) : 0, q.stride(q.ndim() - 1),
      q.ndim() == 2 ? q.stride(0) : 0);
  auto status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "SM120 NVFP4 BF16 Q padded quantization failed: "
      << cudaGetErrorString(status);

  RunPagedBatchImpl(kernel, q_packed, q_scales, k_pages, k_sf_pages,
                    v_pages_pv, v_sf_pages_pv, block_tables, qo_indptr,
                    kv_lens, q_packed_scratch, q_scales_scratch, partial,
                    split_m, split_l, out_scratch, out, workspace,
                    max_physical_kv_len, qk_alpha, pv_alpha, kv_head,
                    split_kv_tiles, group_size, causal, sliding_window,
                    logits_soft_cap, output_group_span, kv_layout_hnd,
                    v_scale_layout, stream_handle, true);
}

}  // namespace sm120_nvfp4_paged
}  // namespace flashinfer
