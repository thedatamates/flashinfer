/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "fmha_nvfp4_sm120_quantize_common.cuh"

namespace flashinfer {
namespace {

void SM120Nvfp4QuantizeQ(TensorView q, TensorView q_packed,
                         TensorView q_scales, int64_t stream_handle) {
  sm120_nvfp4_quantize::QuantizeQImpl(128, q, q_packed, q_scales,
                                      stream_handle);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_q, SM120Nvfp4QuantizeQ);

}  // namespace
}  // namespace flashinfer
