/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once

#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_quantization.cuh>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace sm120_nvfp4_quantize {

namespace sm120 = attention::blackwell::sm120_nvfp4;

static void QuantizeQImpl(int head_dim_expected, TensorView q,
                          TensorView q_packed, TensorView q_scales) {
  CHECK_INPUT_AND_TYPE(q, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(q_packed, dl_uint8);
  CHECK_INPUT_AND_TYPE(q_scales, dl_uint8);
  TVM_FFI_ICHECK(q.ndim() == 2 || q.ndim() == 3)
      << "q must have shape [rows, D] or [q_len, group, D]";
  const int64_t q_rows = q.ndim() == 3 ? q.size(0) * q.size(1) : q.size(0);
  const int64_t head_dim = q.ndim() == 3 ? q.size(2) : q.size(1);
  TVM_FFI_ICHECK_EQ(head_dim, head_dim_expected)
      << "SM120 NVFP4 module/head_dim mismatch";
  TVM_FFI_ICHECK_EQ(q_packed.ndim(), 2);
  TVM_FFI_ICHECK_EQ(q_packed.size(0), q_rows);
  TVM_FFI_ICHECK_EQ(q_packed.size(1), head_dim / 2);
  TVM_FFI_ICHECK_EQ(q_scales.ndim(), 2);
  TVM_FFI_ICHECK(q_scales.size(0) >= q_rows);
  TVM_FFI_ICHECK_EQ(q_scales.size(1), head_dim / 16);

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  cudaError_t status = sm120::quantize_q_rowmajor_raw(
      static_cast<const __nv_bfloat16*>(q.data_ptr()),
      static_cast<uint8_t*>(q_packed.data_ptr()),
      static_cast<uint8_t*>(q_scales.data_ptr()), static_cast<int>(q_rows),
      static_cast<int>(head_dim), stream);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "SM120 NVFP4 Q quantization failed: " << cudaGetErrorString(status);
}

}  // namespace sm120_nvfp4_quantize
}  // namespace flashinfer
