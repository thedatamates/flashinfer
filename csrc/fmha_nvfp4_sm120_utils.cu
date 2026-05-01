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

#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_paged_adapter.cuh>
#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_quantization.cuh>

#include "tvm_ffi_utils.h"

namespace flashinfer {

void SM120Nvfp4QuantizeQ(TensorView q, TensorView q_packed, TensorView q_scales) {
  CHECK_INPUT_AND_TYPE(q, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(q_packed, dl_uint8);
  CHECK_INPUT_AND_TYPE(q_scales, dl_uint8);
  TVM_FFI_ICHECK(q.ndim() == 2 || q.ndim() == 3)
      << "q must have shape [rows, D] or [q_len, group, D]";
  const int64_t q_rows = q.ndim() == 3 ? q.size(0) * q.size(1) : q.size(0);
  const int64_t head_dim = q.ndim() == 3 ? q.size(2) : q.size(1);
  TVM_FFI_ICHECK(head_dim == 128 || head_dim == 256 || head_dim == 512)
      << "SM120 NVFP4 Q quantizer supports D128/D256/D512, got D" << head_dim;
  TVM_FFI_ICHECK_EQ(q_packed.ndim(), 2);
  TVM_FFI_ICHECK_EQ(q_packed.size(0), q_rows);
  TVM_FFI_ICHECK_EQ(q_packed.size(1), head_dim / 2);
  TVM_FFI_ICHECK_EQ(q_scales.ndim(), 2);
  TVM_FFI_ICHECK(q_scales.size(0) >= q_rows);
  TVM_FFI_ICHECK_EQ(q_scales.size(1), head_dim / 16);

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  cudaError_t status =
      attention::blackwell::sm120_nvfp4::quantize_q_rowmajor_raw(
          static_cast<const __nv_bfloat16*>(q.data_ptr()),
          static_cast<uint8_t*>(q_packed.data_ptr()),
          static_cast<uint8_t*>(q_scales.data_ptr()),
          static_cast<int>(q_rows), static_cast<int>(head_dim), stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "SM120 NVFP4 Q quantization failed: " << cudaGetErrorString(status);
}

void SM120Nvfp4GatherPagedKvToDensePv(TensorView k_pages, TensorView k_sf_pages,
                                      TensorView v_pages_pv, TensorView v_sf_pages_pv,
                                      TensorView block_table, TensorView k_dense,
                                      TensorView k_sf_dense, TensorView v_pv_dense,
                                      TensorView v_pv_sf_dense, int64_t kv_head,
                                      int64_t kv_len) {
  CHECK_INPUT_AND_TYPE(k_pages, dl_uint8);
  CHECK_INPUT_AND_TYPE(k_sf_pages, dl_uint8);
  CHECK_INPUT_AND_TYPE(v_pages_pv, dl_uint8);
  CHECK_INPUT_AND_TYPE(v_sf_pages_pv, dl_uint8);
  CHECK_INPUT_AND_TYPE(block_table, dl_int32);
  CHECK_INPUT_AND_TYPE(k_dense, dl_uint8);
  CHECK_INPUT_AND_TYPE(k_sf_dense, dl_uint8);
  CHECK_INPUT_AND_TYPE(v_pv_dense, dl_uint8);
  CHECK_INPUT_AND_TYPE(v_pv_sf_dense, dl_uint8);
  CHECK_DIM(4, k_pages);
  CHECK_DIM(4, k_sf_pages);
  CHECK_SHAPE(k_pages, v_pages_pv);
  CHECK_SHAPE(k_sf_pages, v_sf_pages_pv);

  const int64_t num_pages = k_pages.size(0);
  const int64_t page_size = k_pages.size(1);
  const int64_t num_kv_heads = k_pages.size(2);
  const int64_t packed_dim = k_pages.size(3);
  const int64_t scale_dim = k_sf_pages.size(3);
  const int64_t head_dim = packed_dim * 2;
  TVM_FFI_ICHECK_EQ(page_size, 16)
      << "SM120 NVFP4 paged adapter currently supports page_size=16";
  TVM_FFI_ICHECK_EQ(k_sf_pages.size(0), num_pages);
  TVM_FFI_ICHECK_EQ(k_sf_pages.size(1), page_size);
  TVM_FFI_ICHECK_EQ(k_sf_pages.size(2), num_kv_heads);
  TVM_FFI_ICHECK_EQ(scale_dim * 16, head_dim);
  TVM_FFI_ICHECK(kv_head >= 0 && kv_head < num_kv_heads) << "kv_head out of range";
  TVM_FFI_ICHECK(kv_len > 0 && (kv_len % 128) == 0)
      << "kv_len must be a positive multiple of 128";
  TVM_FFI_ICHECK(block_table.size(0) >= (kv_len + page_size - 1) / page_size)
      << "block_table does not cover kv_len";
  TVM_FFI_ICHECK_EQ(k_dense.ndim(), 2);
  TVM_FFI_ICHECK_EQ(k_dense.size(0), kv_len);
  TVM_FFI_ICHECK_EQ(k_dense.size(1), packed_dim);
  TVM_FFI_ICHECK_EQ(k_sf_dense.ndim(), 2);
  TVM_FFI_ICHECK_EQ(k_sf_dense.size(0), kv_len);
  TVM_FFI_ICHECK_EQ(k_sf_dense.size(1), scale_dim);
  TVM_FFI_ICHECK_EQ(v_pv_dense.ndim(), 2);
  TVM_FFI_ICHECK_EQ(v_pv_dense.size(0), head_dim);
  TVM_FFI_ICHECK_EQ(v_pv_dense.size(1), kv_len / 2);
  TVM_FFI_ICHECK_EQ(v_pv_sf_dense.ndim(), 2);
  TVM_FFI_ICHECK_EQ(v_pv_sf_dense.size(0), head_dim);
  TVM_FFI_ICHECK_EQ(v_pv_sf_dense.size(1), kv_len / page_size);

  ffi::CUDADeviceGuard device_guard(k_pages.device().device_id);
  const cudaStream_t stream = get_stream(k_pages.device());
  cudaError_t status =
      attention::blackwell::sm120_nvfp4::gather_paged_kv_to_dense_pv_raw(
          static_cast<const uint8_t*>(k_pages.data_ptr()),
          static_cast<const uint8_t*>(k_sf_pages.data_ptr()),
          static_cast<const uint8_t*>(v_pages_pv.data_ptr()),
          static_cast<const uint8_t*>(v_sf_pages_pv.data_ptr()),
          static_cast<const int32_t*>(block_table.data_ptr()),
          static_cast<uint8_t*>(k_dense.data_ptr()),
          static_cast<uint8_t*>(k_sf_dense.data_ptr()),
          static_cast<uint8_t*>(v_pv_dense.data_ptr()),
          static_cast<uint8_t*>(v_pv_sf_dense.data_ptr()),
          static_cast<int>(kv_head), static_cast<int>(kv_len),
          static_cast<int>(page_size), static_cast<int>(num_kv_heads),
          static_cast<int>(packed_dim), static_cast<int>(scale_dim), stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "SM120 NVFP4 paged gather failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_q, flashinfer::SM120Nvfp4QuantizeQ);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gather_paged_kv_to_dense_pv,
                              flashinfer::SM120Nvfp4GatherPagedKvToDensePv);
