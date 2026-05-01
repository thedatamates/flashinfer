/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_d128.cuh>
#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_d256.cuh>
#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_d512.cuh>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace {

int64_t tile_m_for_head_dim(int64_t head_dim) {
  if (head_dim == 128) {
    return attention::blackwell::sm120_nvfp4::d128::kCutlassTileM;
  }
  if (head_dim == 256) {
    return attention::blackwell::sm120_nvfp4::d256::kCutlassTileM;
  }
  if (head_dim == 512) {
    return attention::blackwell::sm120_nvfp4::d512::kCutlassTileM;
  }
  TVM_FFI_ICHECK(false) << "unsupported SM120 NVFP4 head_dim " << head_dim;
  return 0;
}

void CheckDenseRunTensors(TensorView q_packed, TensorView q_scales,
                          TensorView k_packed, TensorView k_scales,
                          TensorView v_pv_packed, TensorView v_pv_scales,
                          TensorView partial, TensorView split_m,
                          TensorView split_l, TensorView out,
                          TensorView workspace, int64_t split_kv_tiles,
                          int64_t q_len, int64_t group_size,
                          int64_t kv_len_tokens,
                          int64_t output_group_span) {
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
  const int64_t tile_m = tile_m_for_head_dim(head_dim);

  TVM_FFI_ICHECK(head_dim == 128 || head_dim == 256 || head_dim == 512);
  TVM_FFI_ICHECK(q_rows > 0 && q_rows % tile_m == 0)
      << "q rows must be a positive multiple of tile_m=" << tile_m;
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
  TVM_FFI_ICHECK(output_group_span == 1 || output_group_span == 2 ||
                output_group_span == 4)
      << "output_group_span must be 1, 2, or 4";
  TVM_FFI_ICHECK(head_dim != 128 || output_group_span == 1)
      << "D128 specialization only supports output_group_span=1";
  TVM_FFI_ICHECK(head_dim % (output_group_span * 128) == 0)
      << "head_dim must be divisible by output_group_span * 128";
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

__global__ void DenseSplitKvCombineKernel(const __nv_bfloat16* partial,
                                          const float* split_m,
                                          const float* split_l,
                                          __nv_bfloat16* out,
                                          int num_splits,
                                          int q_rows,
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
          isfinite(global_m) ? __expf(split_m[stats_idx] - global_m) : 0.0f;
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
      const int partial_idx =
          (split * q_rows + row) * head_dim + col;
      acc += split_weights[split] * __bfloat162float(partial[partial_idx]);
    }
    out[row * head_dim + col] = __float2bfloat16(acc);
  }
}

template <int HeadDim, int OutputGroupSpan>
cudaError_t RunDenseRawForHeadDim(
    uint8_t* q_packed, uint8_t* q_scales, uint8_t* k_packed,
    uint8_t* k_scales, uint8_t* v_pv_packed, uint8_t* v_pv_scales,
    __nv_bfloat16* partial, float* split_m, float* split_l,
    __nv_bfloat16* out, uint8_t* workspace, size_t workspace_bytes,
    float qk_alpha, float pv_alpha, int split_kv_tiles, int q_len,
    int group_size, int kv_len_tokens, bool causal, int sliding_window,
    float logits_soft_cap, int q_rows, int kv_len, cudaStream_t stream) {
  if constexpr (HeadDim == 128) {
    return attention::blackwell::sm120_nvfp4::d128::
        sm120_nvfp4_qkv_online_register_q_splitkv_full_grid_raw<
            OutputGroupSpan>(
            q_packed, q_scales, k_packed, k_scales, v_pv_packed, v_pv_scales,
            partial, split_m, split_l, out, workspace, workspace_bytes,
            qk_alpha, pv_alpha, split_kv_tiles, q_len, group_size,
            kv_len_tokens, causal, sliding_window, logits_soft_cap, q_rows,
            HeadDim, kv_len, stream, {}, nullptr, nullptr, 1, 0, true);
  } else if constexpr (HeadDim == 256) {
    return attention::blackwell::sm120_nvfp4::d256::
        sm120_nvfp4_qkv_online_register_q_splitkv_full_grid_raw<
            OutputGroupSpan>(
            q_packed, q_scales, k_packed, k_scales, v_pv_packed, v_pv_scales,
            partial, split_m, split_l, out, workspace, workspace_bytes,
            qk_alpha, pv_alpha, split_kv_tiles, q_len, group_size,
            kv_len_tokens, causal, sliding_window, logits_soft_cap, q_rows,
            HeadDim, kv_len, stream, {}, nullptr, nullptr, 1, 0, true);
  } else {
    return attention::blackwell::sm120_nvfp4::d512::
        sm120_nvfp4_qkv_online_register_q_splitkv_full_grid_raw<
            OutputGroupSpan>(
            q_packed, q_scales, k_packed, k_scales, v_pv_packed, v_pv_scales,
            partial, split_m, split_l, out, workspace, workspace_bytes,
            qk_alpha, pv_alpha, split_kv_tiles, q_len, group_size,
            kv_len_tokens, causal, sliding_window, logits_soft_cap, q_rows,
            HeadDim, kv_len, stream, {}, nullptr, nullptr, 1, 0, true);
  }
}

template <int HeadDim, int OutputGroupSpan>
cudaError_t RunDenseForHeadDim(TensorView q_packed, TensorView q_scales,
                               TensorView k_packed, TensorView k_scales,
                               TensorView v_pv_packed, TensorView v_pv_scales,
                               TensorView partial, TensorView split_m,
                               TensorView split_l, TensorView out,
                               TensorView workspace, float qk_alpha,
                               float pv_alpha, int split_kv_tiles, int q_len,
                               int group_size, int kv_len_tokens, bool causal,
                               int sliding_window, float logits_soft_cap,
                               cudaStream_t stream) {
  const int head_dim = static_cast<int>(q_packed.size(1) * 2);
  const int q_rows = static_cast<int>(q_packed.size(0));
  const int kv_len = static_cast<int>(k_packed.size(0));
  cudaError_t status = RunDenseRawForHeadDim<HeadDim, OutputGroupSpan>(
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
      workspace.size(0) * get_element_size(workspace), qk_alpha, pv_alpha,
      split_kv_tiles, q_len, group_size, kv_len_tokens, causal,
      sliding_window, logits_soft_cap, q_rows, kv_len, stream);
  if (status != cudaSuccess) {
    return status;
  }
  const int total_kv_tiles = kv_len / 128;
  const int num_splits =
      (total_kv_tiles + split_kv_tiles - 1) / split_kv_tiles;
  if (num_splits <= 1) {
    return cudaSuccess;
  }
  DenseSplitKvCombineKernel<<<q_rows, 256,
                              static_cast<size_t>(num_splits) * sizeof(float),
                              stream>>>(
      static_cast<const __nv_bfloat16*>(partial.data_ptr()),
      static_cast<const float*>(split_m.data_ptr()),
      static_cast<const float*>(split_l.data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()), num_splits, q_rows,
      head_dim);
  return cudaGetLastError();
}

template <int HeadDim>
cudaError_t RunDenseSpanDispatch(int64_t output_group_span, TensorView q_packed,
                                 TensorView q_scales, TensorView k_packed,
                                 TensorView k_scales, TensorView v_pv_packed,
                                 TensorView v_pv_scales, TensorView partial,
                                 TensorView split_m, TensorView split_l,
                                 TensorView out, TensorView workspace,
                                 float qk_alpha, float pv_alpha,
                                 int split_kv_tiles, int q_len,
                                 int group_size, int kv_len_tokens,
                                 bool causal, int sliding_window,
                                 float logits_soft_cap,
                                 cudaStream_t stream) {
  if (output_group_span == 1) {
    return RunDenseForHeadDim<HeadDim, 1>(
        q_packed, q_scales, k_packed, k_scales, v_pv_packed, v_pv_scales,
        partial, split_m, split_l, out, workspace, qk_alpha, pv_alpha,
        split_kv_tiles, q_len, group_size, kv_len_tokens, causal,
        sliding_window, logits_soft_cap, stream);
  }
  if (output_group_span == 2) {
    return RunDenseForHeadDim<HeadDim, 2>(
        q_packed, q_scales, k_packed, k_scales, v_pv_packed, v_pv_scales,
        partial, split_m, split_l, out, workspace, qk_alpha, pv_alpha,
        split_kv_tiles, q_len, group_size, kv_len_tokens, causal,
        sliding_window, logits_soft_cap, stream);
  }
  return RunDenseForHeadDim<HeadDim, 4>(
      q_packed, q_scales, k_packed, k_scales, v_pv_packed, v_pv_scales,
      partial, split_m, split_l, out, workspace, qk_alpha, pv_alpha,
      split_kv_tiles, q_len, group_size, kv_len_tokens, causal,
      sliding_window, logits_soft_cap, stream);
}

}  // namespace

void SM120Nvfp4FmhaRunDense(TensorView q_packed, TensorView q_scales,
                            TensorView k_packed, TensorView k_scales,
                            TensorView v_pv_packed, TensorView v_pv_scales,
                            TensorView partial, TensorView split_m,
                            TensorView split_l, TensorView out,
                            TensorView workspace, double qk_alpha,
                            double pv_alpha, int64_t split_kv_tiles,
                            int64_t q_len, int64_t group_size,
                            int64_t kv_len_tokens, bool causal,
                            int64_t sliding_window, double logits_soft_cap,
                            int64_t output_group_span) {
  CheckDenseRunTensors(q_packed, q_scales, k_packed, k_scales, v_pv_packed,
                       v_pv_scales, partial, split_m, split_l, out, workspace,
                       split_kv_tiles, q_len, group_size, kv_len_tokens,
                       output_group_span);
  ffi::CUDADeviceGuard device_guard(q_packed.device().device_id);
  const cudaStream_t stream = get_stream(q_packed.device());
  const int64_t head_dim = q_packed.size(1) * 2;
  cudaError_t status = cudaSuccess;
  if (head_dim == 128) {
    status = RunDenseForHeadDim<128, 1>(
        q_packed, q_scales, k_packed, k_scales, v_pv_packed, v_pv_scales,
        partial, split_m, split_l, out, workspace, static_cast<float>(qk_alpha),
        static_cast<float>(pv_alpha), static_cast<int>(split_kv_tiles),
        static_cast<int>(q_len), static_cast<int>(group_size),
        static_cast<int>(kv_len_tokens), causal,
        static_cast<int>(sliding_window), static_cast<float>(logits_soft_cap),
        stream);
  } else if (head_dim == 256) {
    status = RunDenseSpanDispatch<256>(
        output_group_span, q_packed, q_scales, k_packed, k_scales,
        v_pv_packed, v_pv_scales, partial, split_m, split_l, out, workspace,
        static_cast<float>(qk_alpha), static_cast<float>(pv_alpha),
        static_cast<int>(split_kv_tiles), static_cast<int>(q_len),
        static_cast<int>(group_size), static_cast<int>(kv_len_tokens), causal,
        static_cast<int>(sliding_window), static_cast<float>(logits_soft_cap),
        stream);
  } else {
    status = RunDenseSpanDispatch<512>(
        output_group_span, q_packed, q_scales, k_packed, k_scales,
        v_pv_packed, v_pv_scales, partial, split_m, split_l, out, workspace,
        static_cast<float>(qk_alpha), static_cast<float>(pv_alpha),
        static_cast<int>(split_kv_tiles), static_cast<int>(q_len),
        static_cast<int>(group_size), static_cast<int>(kv_len_tokens), causal,
        static_cast<int>(sliding_window), static_cast<float>(logits_soft_cap),
        stream);
  }
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "SM120 NVFP4 FMHA dense run failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_dense, flashinfer::SM120Nvfp4FmhaRunDense);
