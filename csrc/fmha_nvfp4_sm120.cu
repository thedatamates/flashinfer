/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_d128.cuh>
#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_d256.cuh>
#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_d512.cuh>
#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_paged_adapter.cuh>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace {

__global__ void CopyQToPaddedKernel(const uint8_t* q_packed,
                                    const uint8_t* q_scales,
                                    uint8_t* q_packed_scratch,
                                    uint8_t* q_scales_scratch, int q_rows,
                                    int padded_q_rows, int packed_dim,
                                    int scale_dim) {
  const int row = static_cast<int>(blockIdx.x);
  const int col = static_cast<int>(threadIdx.x);
  if (row >= padded_q_rows) {
    return;
  }
  if (col < packed_dim) {
    q_packed_scratch[row * packed_dim + col] =
        row < q_rows ? q_packed[row * packed_dim + col] : 0;
  }
  if (col < scale_dim) {
    q_scales_scratch[row * scale_dim + col] =
        row < q_rows ? q_scales[row * scale_dim + col] : 0;
  }
}

__global__ void CopyPaddedOutKernel(const __nv_bfloat16* out_scratch,
                                    __nv_bfloat16* out, int q_rows,
                                    int head_dim) {
  const int row = static_cast<int>(blockIdx.x);
  const int col = static_cast<int>(threadIdx.x);
  if (row < q_rows && col < head_dim) {
    out[row * head_dim + col] = out_scratch[row * head_dim + col];
  }
}

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

int64_t RoundUpMultiple(int64_t x, int64_t multiple) {
  return ((x + multiple - 1) / multiple) * multiple;
}

std::vector<int64_t> ReadIndexTensorToHost(TensorView tensor, const char* name,
                                           cudaStream_t stream) {
  TVM_FFI_ICHECK(tensor.ndim() == 1) << name << " must be a 1D tensor";
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
  TVM_FFI_ICHECK(tensor.dtype() == dl_int32 || tensor.dtype() == dl_int64)
      << name << " must be int32 or int64";
  std::vector<int64_t> host(tensor.size(0));
  if (tensor.device().device_type == kDLCPU) {
    if (tensor.dtype() == dl_int32) {
      const int32_t* src = static_cast<const int32_t*>(tensor.data_ptr());
      for (int64_t i = 0; i < tensor.size(0); ++i) {
        host[i] = static_cast<int64_t>(src[i]);
      }
    } else {
      std::memcpy(host.data(), tensor.data_ptr(),
                  static_cast<size_t>(tensor.size(0)) * sizeof(int64_t));
    }
    return host;
  }
  CHECK_CUDA(tensor);
  if (tensor.dtype() == dl_int32) {
    std::vector<int32_t> tmp(tensor.size(0));
    cudaError_t status =
        cudaMemcpyAsync(tmp.data(), tensor.data_ptr(),
                        static_cast<size_t>(tensor.size(0)) * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "failed to copy " << name << " to host: "
        << cudaGetErrorString(status);
    status = cudaStreamSynchronize(stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "failed to synchronize " << name << " copy: "
        << cudaGetErrorString(status);
    for (int64_t i = 0; i < tensor.size(0); ++i) {
      host[i] = static_cast<int64_t>(tmp[i]);
    }
  } else {
    cudaError_t status =
        cudaMemcpyAsync(host.data(), tensor.data_ptr(),
                        static_cast<size_t>(tensor.size(0)) * sizeof(int64_t),
                        cudaMemcpyDeviceToHost, stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "failed to copy " << name << " to host: "
        << cudaGetErrorString(status);
    status = cudaStreamSynchronize(stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "failed to synchronize " << name << " copy: "
        << cudaGetErrorString(status);
  }
  return host;
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
            HeadDim, kv_len, stream);
  } else if constexpr (HeadDim == 256) {
    return attention::blackwell::sm120_nvfp4::d256::
        sm120_nvfp4_qkv_online_register_q_splitkv_full_grid_raw<
            OutputGroupSpan>(
            q_packed, q_scales, k_packed, k_scales, v_pv_packed, v_pv_scales,
            partial, split_m, split_l, out, workspace, workspace_bytes,
            qk_alpha, pv_alpha, split_kv_tiles, q_len, group_size,
            kv_len_tokens, causal, sliding_window, logits_soft_cap, q_rows,
            HeadDim, kv_len, stream);
  } else {
    return attention::blackwell::sm120_nvfp4::d512::
        sm120_nvfp4_qkv_online_register_q_splitkv_full_grid_raw<
            OutputGroupSpan>(
            q_packed, q_scales, k_packed, k_scales, v_pv_packed, v_pv_scales,
            partial, split_m, split_l, out, workspace, workspace_bytes,
            qk_alpha, pv_alpha, split_kv_tiles, q_len, group_size,
            kv_len_tokens, causal, sliding_window, logits_soft_cap, q_rows,
            HeadDim, kv_len, stream);
  }
}

template <int HeadDim>
cudaError_t RunDenseRawSpanDispatch(
    int64_t output_group_span, uint8_t* q_packed, uint8_t* q_scales,
    uint8_t* k_packed, uint8_t* k_scales, uint8_t* v_pv_packed,
    uint8_t* v_pv_scales, __nv_bfloat16* partial, float* split_m,
    float* split_l, __nv_bfloat16* out, uint8_t* workspace,
    size_t workspace_bytes, float qk_alpha, float pv_alpha,
    int split_kv_tiles, int q_len, int group_size, int kv_len_tokens,
    bool causal, int sliding_window, float logits_soft_cap, int q_rows,
    int kv_len, cudaStream_t stream) {
  if (output_group_span == 1) {
    return RunDenseRawForHeadDim<HeadDim, 1>(
        q_packed, q_scales, k_packed, k_scales, v_pv_packed, v_pv_scales,
        partial, split_m, split_l, out, workspace, workspace_bytes, qk_alpha,
        pv_alpha, split_kv_tiles, q_len, group_size, kv_len_tokens, causal,
        sliding_window, logits_soft_cap, q_rows, kv_len, stream);
  }
  if (output_group_span == 2) {
    return RunDenseRawForHeadDim<HeadDim, 2>(
        q_packed, q_scales, k_packed, k_scales, v_pv_packed, v_pv_scales,
        partial, split_m, split_l, out, workspace, workspace_bytes, qk_alpha,
        pv_alpha, split_kv_tiles, q_len, group_size, kv_len_tokens, causal,
        sliding_window, logits_soft_cap, q_rows, kv_len, stream);
  }
  return RunDenseRawForHeadDim<HeadDim, 4>(
      q_packed, q_scales, k_packed, k_scales, v_pv_packed, v_pv_scales,
      partial, split_m, split_l, out, workspace, workspace_bytes, qk_alpha,
      pv_alpha, split_kv_tiles, q_len, group_size, kv_len_tokens, causal,
      sliding_window, logits_soft_cap, q_rows, kv_len, stream);
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
  return RunDenseRawForHeadDim<HeadDim, OutputGroupSpan>(
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
        static_cast<int>(kv_len_tokens), causal, static_cast<int>(sliding_window),
        static_cast<float>(logits_soft_cap), stream);
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

void SM120Nvfp4FmhaRunPagedSingle(
    TensorView q_packed, TensorView q_scales, TensorView k_pages,
    TensorView k_sf_pages, TensorView v_pages_pv, TensorView v_sf_pages_pv,
    TensorView block_table, TensorView k_dense_scratch,
    TensorView k_sf_dense_scratch, TensorView v_pv_dense_scratch,
    TensorView v_pv_sf_dense_scratch, TensorView partial, TensorView split_m,
    TensorView split_l, TensorView out, TensorView workspace, double qk_alpha,
    double pv_alpha, int64_t kv_head, int64_t split_kv_tiles, int64_t q_len,
    int64_t group_size, int64_t kv_len_tokens, bool causal,
    int64_t sliding_window, double logits_soft_cap,
    int64_t output_group_span) {
  CHECK_INPUT_AND_TYPE(k_pages, dl_uint8);
  CHECK_INPUT_AND_TYPE(k_sf_pages, dl_uint8);
  CHECK_INPUT_AND_TYPE(v_pages_pv, dl_uint8);
  CHECK_INPUT_AND_TYPE(v_sf_pages_pv, dl_uint8);
  CHECK_INPUT_AND_TYPE(block_table, dl_int32);
  CHECK_INPUT_AND_TYPE(k_dense_scratch, dl_uint8);
  CHECK_INPUT_AND_TYPE(k_sf_dense_scratch, dl_uint8);
  CHECK_INPUT_AND_TYPE(v_pv_dense_scratch, dl_uint8);
  CHECK_INPUT_AND_TYPE(v_pv_sf_dense_scratch, dl_uint8);
  CHECK_DIM(1, block_table);
  CHECK_DIM(4, k_pages);
  CHECK_DIM(4, k_sf_pages);
  CHECK_DIM(4, v_pages_pv);
  CHECK_DIM(4, v_sf_pages_pv);
  CHECK_DIM(2, k_dense_scratch);
  CHECK_DIM(2, k_sf_dense_scratch);
  CHECK_DIM(2, v_pv_dense_scratch);
  CHECK_DIM(2, v_pv_sf_dense_scratch);
  CHECK_SHAPE(k_pages, v_pages_pv);
  CHECK_SHAPE(k_sf_pages, v_sf_pages_pv);

  const int64_t num_pages = k_pages.size(0);
  const int64_t page_size = k_pages.size(1);
  const int64_t num_kv_heads = k_pages.size(2);
  const int64_t packed_dim = k_pages.size(3);
  const int64_t scale_dim = k_sf_pages.size(3);
  const int64_t head_dim = packed_dim * 2;
  TVM_FFI_ICHECK_EQ(page_size, 16)
      << "SM120 NVFP4 paged run currently supports page_size=16";
  TVM_FFI_ICHECK_EQ(k_sf_pages.size(0), num_pages);
  TVM_FFI_ICHECK_EQ(k_sf_pages.size(1), page_size);
  TVM_FFI_ICHECK_EQ(k_sf_pages.size(2), num_kv_heads);
  TVM_FFI_ICHECK_EQ(scale_dim * 16, head_dim);
  TVM_FFI_ICHECK(kv_head >= 0 && kv_head < num_kv_heads)
      << "kv_head out of range";
  const int64_t physical_kv_len = k_dense_scratch.size(0);
  TVM_FFI_ICHECK(physical_kv_len >= kv_len_tokens);
  TVM_FFI_ICHECK(physical_kv_len % 128 == 0);
  TVM_FFI_ICHECK(block_table.size(0) >=
                 (physical_kv_len + page_size - 1) / page_size)
      << "block_table does not cover physical scratch KV length";
  TVM_FFI_ICHECK_EQ(k_dense_scratch.size(1), packed_dim);
  TVM_FFI_ICHECK_EQ(k_sf_dense_scratch.size(0), physical_kv_len);
  TVM_FFI_ICHECK_EQ(k_sf_dense_scratch.size(1), scale_dim);
  TVM_FFI_ICHECK_EQ(v_pv_dense_scratch.size(0), head_dim);
  TVM_FFI_ICHECK_EQ(v_pv_dense_scratch.size(1), physical_kv_len / 2);
  TVM_FFI_ICHECK_EQ(v_pv_sf_dense_scratch.size(0), head_dim);
  TVM_FFI_ICHECK_EQ(v_pv_sf_dense_scratch.size(1), physical_kv_len / page_size);

  ffi::CUDADeviceGuard device_guard(k_pages.device().device_id);
  const cudaStream_t stream = get_stream(k_pages.device());
  cudaError_t gather_status =
      attention::blackwell::sm120_nvfp4::gather_paged_kv_to_dense_pv_raw(
          static_cast<const uint8_t*>(k_pages.data_ptr()),
          static_cast<const uint8_t*>(k_sf_pages.data_ptr()),
          static_cast<const uint8_t*>(v_pages_pv.data_ptr()),
          static_cast<const uint8_t*>(v_sf_pages_pv.data_ptr()),
          static_cast<const int32_t*>(block_table.data_ptr()),
          static_cast<uint8_t*>(k_dense_scratch.data_ptr()),
          static_cast<uint8_t*>(k_sf_dense_scratch.data_ptr()),
          static_cast<uint8_t*>(v_pv_dense_scratch.data_ptr()),
          static_cast<uint8_t*>(v_pv_sf_dense_scratch.data_ptr()),
          static_cast<int>(kv_head), static_cast<int>(physical_kv_len),
          static_cast<int>(k_pages.size(1)), static_cast<int>(k_pages.size(2)),
          static_cast<int>(packed_dim), static_cast<int>(scale_dim), stream);
  TVM_FFI_ICHECK_EQ(gather_status, cudaSuccess)
      << "SM120 NVFP4 paged gather failed: "
      << cudaGetErrorString(gather_status);
  SM120Nvfp4FmhaRunDense(
      q_packed, q_scales, k_dense_scratch, k_sf_dense_scratch,
      v_pv_dense_scratch, v_pv_sf_dense_scratch, partial, split_m, split_l,
      out, workspace, qk_alpha, pv_alpha, split_kv_tiles, q_len, group_size,
      kv_len_tokens, causal, sliding_window, logits_soft_cap,
      output_group_span);
}

void SM120Nvfp4FmhaRunPagedBatch(
    TensorView q_packed, TensorView q_scales, TensorView k_pages,
    TensorView k_sf_pages, TensorView v_pages_pv, TensorView v_sf_pages_pv,
    TensorView block_tables, TensorView qo_indptr, TensorView kv_lens,
    TensorView q_packed_scratch, TensorView q_scales_scratch,
    TensorView k_dense_scratch, TensorView k_sf_dense_scratch,
    TensorView v_pv_dense_scratch, TensorView v_pv_sf_dense_scratch,
    TensorView partial, TensorView split_m, TensorView split_l,
    TensorView out_scratch, TensorView out, TensorView workspace,
    double qk_alpha, double pv_alpha, int64_t kv_head,
    int64_t split_kv_tiles, int64_t group_size, bool causal,
    int64_t sliding_window, double logits_soft_cap,
    int64_t output_group_span) {
  CHECK_INPUT_AND_TYPE(q_packed, dl_uint8);
  CHECK_INPUT_AND_TYPE(q_scales, dl_uint8);
  CHECK_INPUT_AND_TYPE(k_pages, dl_uint8);
  CHECK_INPUT_AND_TYPE(k_sf_pages, dl_uint8);
  CHECK_INPUT_AND_TYPE(v_pages_pv, dl_uint8);
  CHECK_INPUT_AND_TYPE(v_sf_pages_pv, dl_uint8);
  CHECK_INPUT_AND_TYPE(block_tables, dl_int32);
  CHECK_INPUT_AND_TYPE(q_packed_scratch, dl_uint8);
  CHECK_INPUT_AND_TYPE(q_scales_scratch, dl_uint8);
  CHECK_INPUT_AND_TYPE(k_dense_scratch, dl_uint8);
  CHECK_INPUT_AND_TYPE(k_sf_dense_scratch, dl_uint8);
  CHECK_INPUT_AND_TYPE(v_pv_dense_scratch, dl_uint8);
  CHECK_INPUT_AND_TYPE(v_pv_sf_dense_scratch, dl_uint8);
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
  CHECK_DIM(2, k_dense_scratch);
  CHECK_DIM(2, k_sf_dense_scratch);
  CHECK_DIM(2, v_pv_dense_scratch);
  CHECK_DIM(2, v_pv_sf_dense_scratch);
  CHECK_DIM(3, partial);
  CHECK_DIM(2, split_m);
  CHECK_DIM(2, split_l);
  CHECK_DIM(2, out_scratch);
  CHECK_DIM(2, out);
  CHECK_DIM(1, workspace);
  CHECK_SHAPE(k_pages, v_pages_pv);
  CHECK_SHAPE(k_sf_pages, v_sf_pages_pv);
  if (qo_indptr.device().device_type == kDLCUDA) {
    CHECK_DEVICE(q_packed, qo_indptr);
  } else {
    CHECK_CPU(qo_indptr);
  }
  if (kv_lens.device().device_type == kDLCUDA) {
    CHECK_DEVICE(q_packed, kv_lens);
  } else {
    CHECK_CPU(kv_lens);
  }

  ffi::CUDADeviceGuard device_guard(q_packed.device().device_id);
  const cudaStream_t stream = get_stream(q_packed.device());
  std::vector<int64_t> qo_host =
      ReadIndexTensorToHost(qo_indptr, "qo_indptr", stream);
  std::vector<int64_t> kv_host =
      ReadIndexTensorToHost(kv_lens, "kv_lens", stream);

  const int64_t batch = block_tables.size(0);
  TVM_FFI_ICHECK_EQ(static_cast<int64_t>(qo_host.size()), batch + 1);
  TVM_FFI_ICHECK_EQ(static_cast<int64_t>(kv_host.size()), batch);
  const int64_t page_size = k_pages.size(1);
  const int64_t num_kv_heads = k_pages.size(2);
  const int64_t packed_dim = k_pages.size(3);
  const int64_t scale_dim = k_sf_pages.size(3);
  const int64_t head_dim = packed_dim * 2;
  const int64_t tile_m = tile_m_for_head_dim(head_dim);
  TVM_FFI_ICHECK(head_dim == 128 || head_dim == 256 || head_dim == 512);
  TVM_FFI_ICHECK_EQ(page_size, 16)
      << "SM120 NVFP4 paged run currently supports page_size=16";
  TVM_FFI_ICHECK_EQ(k_sf_pages.size(1), page_size);
  TVM_FFI_ICHECK_EQ(k_sf_pages.size(2), num_kv_heads);
  TVM_FFI_ICHECK_EQ(scale_dim * 16, head_dim);
  TVM_FFI_ICHECK(kv_head >= 0 && kv_head < num_kv_heads)
      << "kv_head out of range";
  TVM_FFI_ICHECK(split_kv_tiles > 0) << "split_kv_tiles must be positive";
  TVM_FFI_ICHECK(group_size > 0) << "group_size must be positive";
  TVM_FFI_ICHECK(output_group_span == 1 || output_group_span == 2 ||
                output_group_span == 4)
      << "output_group_span must be 1, 2, or 4";
  TVM_FFI_ICHECK(head_dim != 128 || output_group_span == 1)
      << "D128 specialization only supports output_group_span=1";
  TVM_FFI_ICHECK(head_dim % (output_group_span * 128) == 0)
      << "head_dim must be divisible by output_group_span * 128";
  TVM_FFI_ICHECK_EQ(q_packed.size(1), packed_dim);
  TVM_FFI_ICHECK_EQ(q_scales.size(1), scale_dim);
  TVM_FFI_ICHECK_EQ(out.size(0), q_packed.size(0));
  TVM_FFI_ICHECK_EQ(out.size(1), head_dim);
  TVM_FFI_ICHECK_EQ(q_packed_scratch.size(1), packed_dim);
  TVM_FFI_ICHECK_EQ(q_scales_scratch.size(1), scale_dim);
  TVM_FFI_ICHECK_EQ(k_dense_scratch.size(1), packed_dim);
  TVM_FFI_ICHECK_EQ(k_sf_dense_scratch.size(1), scale_dim);
  TVM_FFI_ICHECK_EQ(v_pv_dense_scratch.size(0), head_dim);
  TVM_FFI_ICHECK_EQ(v_pv_sf_dense_scratch.size(0), head_dim);
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
  uint8_t* k_scratch_ptr = static_cast<uint8_t*>(k_dense_scratch.data_ptr());
  uint8_t* k_sf_scratch_ptr =
      static_cast<uint8_t*>(k_sf_dense_scratch.data_ptr());
  uint8_t* v_scratch_ptr =
      static_cast<uint8_t*>(v_pv_dense_scratch.data_ptr());
  uint8_t* v_sf_scratch_ptr =
      static_cast<uint8_t*>(v_pv_sf_dense_scratch.data_ptr());
  __nv_bfloat16* partial_ptr =
      static_cast<__nv_bfloat16*>(partial.data_ptr());
  float* split_m_ptr = static_cast<float*>(split_m.data_ptr());
  float* split_l_ptr = static_cast<float*>(split_l.data_ptr());
  __nv_bfloat16* out_scratch_ptr =
      static_cast<__nv_bfloat16*>(out_scratch.data_ptr());
  uint8_t* workspace_ptr = static_cast<uint8_t*>(workspace.data_ptr());

  for (int64_t b = 0; b < batch; ++b) {
    const int64_t q_begin = qo_host[b];
    const int64_t q_end = qo_host[b + 1];
    TVM_FFI_ICHECK(q_end >= q_begin)
        << "qo_indptr must be non-decreasing";
    const int64_t q_len = q_end - q_begin;
    if (q_len == 0) {
      continue;
    }
    const int64_t q_row_begin = q_begin * group_size;
    const int64_t q_rows = q_len * group_size;
    const int64_t padded_q_rows = RoundUpMultiple(q_rows, tile_m);
    const int64_t kv_len_tokens = kv_host[b];
    TVM_FFI_ICHECK(kv_len_tokens > 0) << "kv_lens entries must be positive";
    const int64_t physical_kv_len = RoundUpMultiple(kv_len_tokens, 128);
    const int64_t total_kv_tiles = physical_kv_len / 128;
    const int64_t num_splits =
        (total_kv_tiles + split_kv_tiles - 1) / split_kv_tiles;
    TVM_FFI_ICHECK(q_row_begin + q_rows <= q_packed.size(0))
        << "q sequence exceeds q_packed rows";
    TVM_FFI_ICHECK(q_row_begin + q_rows <= out.size(0))
        << "q sequence exceeds out rows";
    TVM_FFI_ICHECK(padded_q_rows <= q_packed_scratch.size(0))
        << "q scratch is too small for padded sequence";
    TVM_FFI_ICHECK(physical_kv_len <= k_dense_scratch.size(0))
        << "K scratch is too small for padded KV length";
    TVM_FFI_ICHECK(physical_kv_len <= k_sf_dense_scratch.size(0))
        << "K scale scratch is too small for padded KV length";
    TVM_FFI_ICHECK(v_pv_dense_scratch.size(1) >= physical_kv_len / 2)
        << "V scratch is too small for padded KV length";
    TVM_FFI_ICHECK(v_pv_sf_dense_scratch.size(1) >= physical_kv_len / page_size)
        << "V scale scratch is too small for padded KV length";
    TVM_FFI_ICHECK(num_splits <= partial.size(0))
        << "partial scratch does not cover split count";
    TVM_FFI_ICHECK(block_tables.size(1) >= physical_kv_len / page_size)
        << "block_tables row does not cover padded KV length";

    const int32_t* block_table_ptr =
        static_cast<const int32_t*>(block_tables.data_ptr()) +
        b * block_tables.stride(0);
    cudaError_t status =
        attention::blackwell::sm120_nvfp4::gather_paged_kv_to_dense_pv_raw(
            static_cast<const uint8_t*>(k_pages.data_ptr()),
            static_cast<const uint8_t*>(k_sf_pages.data_ptr()),
            static_cast<const uint8_t*>(v_pages_pv.data_ptr()),
            static_cast<const uint8_t*>(v_sf_pages_pv.data_ptr()),
            block_table_ptr, k_scratch_ptr, k_sf_scratch_ptr, v_scratch_ptr,
            v_sf_scratch_ptr, static_cast<int>(kv_head),
            static_cast<int>(physical_kv_len), static_cast<int>(page_size),
            static_cast<int>(num_kv_heads), static_cast<int>(packed_dim),
            static_cast<int>(scale_dim), stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "SM120 NVFP4 paged gather failed: "
        << cudaGetErrorString(status);

    CopyQToPaddedKernel<<<static_cast<unsigned>(padded_q_rows), 256, 0,
                          stream>>>(
        static_cast<const uint8_t*>(q_packed.data_ptr()) +
            q_row_begin * packed_dim,
        static_cast<const uint8_t*>(q_scales.data_ptr()) +
            q_row_begin * scale_dim,
        q_scratch_ptr, q_sf_scratch_ptr, static_cast<int>(q_rows),
        static_cast<int>(padded_q_rows), static_cast<int>(packed_dim),
        static_cast<int>(scale_dim));
    status = cudaGetLastError();
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "SM120 NVFP4 Q pad copy failed: " << cudaGetErrorString(status);

    if (head_dim == 128) {
      status = RunDenseRawForHeadDim<128, 1>(
          q_scratch_ptr, q_sf_scratch_ptr, k_scratch_ptr, k_sf_scratch_ptr,
          v_scratch_ptr, v_sf_scratch_ptr, partial_ptr, split_m_ptr,
          split_l_ptr, out_scratch_ptr, workspace_ptr, workspace_bytes,
          static_cast<float>(qk_alpha), static_cast<float>(pv_alpha),
          static_cast<int>(split_kv_tiles), static_cast<int>(q_len),
          static_cast<int>(group_size), static_cast<int>(kv_len_tokens), causal,
          static_cast<int>(sliding_window), static_cast<float>(logits_soft_cap),
          static_cast<int>(padded_q_rows), static_cast<int>(physical_kv_len),
          stream);
    } else if (head_dim == 256) {
      status = RunDenseRawSpanDispatch<256>(
          output_group_span, q_scratch_ptr, q_sf_scratch_ptr, k_scratch_ptr,
          k_sf_scratch_ptr, v_scratch_ptr, v_sf_scratch_ptr, partial_ptr,
          split_m_ptr, split_l_ptr, out_scratch_ptr, workspace_ptr,
          workspace_bytes, static_cast<float>(qk_alpha),
          static_cast<float>(pv_alpha), static_cast<int>(split_kv_tiles),
          static_cast<int>(q_len), static_cast<int>(group_size),
          static_cast<int>(kv_len_tokens), causal,
          static_cast<int>(sliding_window), static_cast<float>(logits_soft_cap),
          static_cast<int>(padded_q_rows), static_cast<int>(physical_kv_len),
          stream);
    } else {
      status = RunDenseRawSpanDispatch<512>(
          output_group_span, q_scratch_ptr, q_sf_scratch_ptr, k_scratch_ptr,
          k_sf_scratch_ptr, v_scratch_ptr, v_sf_scratch_ptr, partial_ptr,
          split_m_ptr, split_l_ptr, out_scratch_ptr, workspace_ptr,
          workspace_bytes, static_cast<float>(qk_alpha),
          static_cast<float>(pv_alpha), static_cast<int>(split_kv_tiles),
          static_cast<int>(q_len), static_cast<int>(group_size),
          static_cast<int>(kv_len_tokens), causal,
          static_cast<int>(sliding_window), static_cast<float>(logits_soft_cap),
          static_cast<int>(padded_q_rows), static_cast<int>(physical_kv_len),
          stream);
    }
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "SM120 NVFP4 FMHA batch sequence failed: "
        << cudaGetErrorString(status);

    CopyPaddedOutKernel<<<static_cast<unsigned>(q_rows), 1024, 0, stream>>>(
        out_scratch_ptr,
        static_cast<__nv_bfloat16*>(out.data_ptr()) + q_row_begin * head_dim,
        static_cast<int>(q_rows), static_cast<int>(head_dim));
    status = cudaGetLastError();
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "SM120 NVFP4 output copy failed: " << cudaGetErrorString(status);
  }
}

}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_dense, flashinfer::SM120Nvfp4FmhaRunDense);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_paged_single,
                              flashinfer::SM120Nvfp4FmhaRunPagedSingle);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_paged_batch,
                              flashinfer::SM120Nvfp4FmhaRunPagedBatch);
