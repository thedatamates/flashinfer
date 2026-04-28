/*
 * Copyright (c) 2025 by FlashInfer team.
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

// Fused row-wise softmax and NVFP4 quantization for Blackwell FP4 attention
// prototypes. The scale output uses the same 128x4 swizzled E4M3 layout as
// fp4_quantize(..., SfLayout.layout_128x4), so it can feed CUTLASS FP4 GEMM.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "tvm_ffi_utils.h"

constexpr int NVFP4_BLOCK_SIZE = 16;

__host__ __device__ __forceinline__ int pad_up(int x, int y) {
  return ((x + y - 1) / y) * y;
}

__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float to_float(half x) { return __half2float(x); }

// Given a row/column in the unswizzled scale-factor matrix, compute the offset
// in TensorRT-LLM/CUTLASS' 128x4 swizzled scale-factor layout.
__device__ __forceinline__ int compute_sf_index_swizzled_128x4(int row_idx, int col_idx,
                                                               int total_cols) {
  constexpr int kColumnGroup0Size = 4;
  constexpr int kRowGroup0Size = 32;
  constexpr int kRowGroup1Size = kRowGroup0Size * 4;

  int padded_column = pad_up(total_cols, 4);

  int column_idx_in_group0 = col_idx % kColumnGroup0Size;
  int column_group_idx = col_idx / kColumnGroup0Size;
  constexpr int column_group_stride = kColumnGroup0Size * kRowGroup1Size;

  int row_idx_in_group0 = row_idx % kRowGroup0Size;
  int row_idx_in_group1 = row_idx % kRowGroup1Size / kRowGroup0Size;
  int row_group_idx = row_idx / kRowGroup1Size;
  constexpr int row_group1_stride = kColumnGroup0Size;
  constexpr int row_group0_stride = kColumnGroup0Size * row_group1_stride;
  int row_group_stride = kRowGroup1Size * padded_column;

  return column_idx_in_group0 + column_group_idx * column_group_stride +
         row_idx_in_group0 * row_group0_stride + row_idx_in_group1 * row_group1_stride +
         row_group_idx * row_group_stride;
}

__device__ __forceinline__ uint8_t fp32_pair_to_e2m1(float x, float y) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint16_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "mov.b16 %0, {byte0, 0};\n"
      "}"
      : "=h"(val)
      : "f"(x), "f"(y));
  return static_cast<uint8_t>(val);
#else
  __trap();
  return 0;
#endif
}

template <typename InType, int NUM_THREADS = 256>
__global__ void nvfp4_softmax_quant_kernel(const InType* __restrict__ logits,
                                           const float* __restrict__ global_scale_ptr,
                                           uint8_t* __restrict__ fp4_output,
                                           uint8_t* __restrict__ block_scales, int M, int N) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= M) return;

  extern __shared__ float smem[];
  const InType* row_logits = logits + static_cast<int64_t>(row) * N;

  float thread_max = -INFINITY;
  for (int col = tid; col < N; col += NUM_THREADS) {
    thread_max = fmaxf(thread_max, to_float(row_logits[col]));
  }
  smem[tid] = thread_max;
  __syncthreads();

  for (int stride = NUM_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }
  float row_max = smem[0];

  float thread_sum = 0.0f;
  for (int col = tid; col < N; col += NUM_THREADS) {
    thread_sum += __expf(to_float(row_logits[col]) - row_max);
  }
  smem[tid] = thread_sum;
  __syncthreads();

  for (int stride = NUM_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  float inv_sum = reciprocal_approximate_ftz(smem[0]);
  float global_scale = *global_scale_ptr;

  int num_scale_cols = N / NVFP4_BLOCK_SIZE;
  uint8_t* row_fp4 = fp4_output + static_cast<int64_t>(row) * (N / 2);

  for (int scale_col = tid; scale_col < num_scale_cols; scale_col += NUM_THREADS) {
    int base_col = scale_col * NVFP4_BLOCK_SIZE;
    float probs[NVFP4_BLOCK_SIZE];
    float vec_max = 0.0f;

#pragma unroll
    for (int i = 0; i < NVFP4_BLOCK_SIZE; ++i) {
      float p = __expf(to_float(row_logits[base_col + i]) - row_max) * inv_sum;
      probs[i] = p;
      vec_max = fmaxf(vec_max, p);
    }

    float sf_value = global_scale * (vec_max * reciprocal_approximate_ftz(6.0f));
    __nv_fp8_e4m3 fp8_scale = __nv_fp8_e4m3(sf_value);
    uint8_t fp8_scale_val = fp8_scale.__x;
    sf_value = static_cast<float>(fp8_scale);

    float output_scale = 0.0f;
    if (vec_max != 0.0f && sf_value != 0.0f) {
      output_scale = reciprocal_approximate_ftz(sf_value * reciprocal_approximate_ftz(global_scale));
    }

    int sf_idx = compute_sf_index_swizzled_128x4(row, scale_col, num_scale_cols);
    block_scales[sf_idx] = fp8_scale_val;

    uint8_t* packed = row_fp4 + base_col / 2;
#pragma unroll
    for (int i = 0; i < NVFP4_BLOCK_SIZE / 2; ++i) {
      packed[i] = fp32_pair_to_e2m1(probs[2 * i] * output_scale,
                                    probs[2 * i + 1] * output_scale);
    }
  }
}

template <int NUM_THREADS>
void launch_nvfp4_softmax_quant(TensorView logits, TensorView global_scale, TensorView fp4_output,
                                TensorView block_scales, int M, int N, cudaStream_t stream) {
  dim3 grid(M);
  dim3 block(NUM_THREADS);
  size_t smem_size = NUM_THREADS * sizeof(float);

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(logits.dtype(), c_type, [&] {
    nvfp4_softmax_quant_kernel<c_type, NUM_THREADS>
        <<<grid, block, smem_size, stream>>>(static_cast<const c_type*>(logits.data_ptr()),
                                             static_cast<const float*>(global_scale.data_ptr()),
                                             static_cast<uint8_t*>(fp4_output.data_ptr()),
                                             static_cast<uint8_t*>(block_scales.data_ptr()), M, N);
    return true;
  });
}

void nvfp4_softmax_quant(TensorView logits, TensorView global_scale, TensorView fp4_output,
                         TensorView block_scales, int64_t num_threads) {
  CHECK_INPUT(logits);
  CHECK_CUDA(global_scale);
  CHECK_INPUT(fp4_output);
  CHECK_INPUT(block_scales);
  CHECK_INPUT_TYPE(global_scale, dl_float32);
  CHECK_INPUT_TYPE(fp4_output, dl_uint8);
  CHECK_INPUT_TYPE(block_scales, dl_uint8);

  TVM_FFI_ICHECK(logits.ndim() == 2) << "logits must be 2D";
  TVM_FFI_ICHECK(global_scale.ndim() == 1 && global_scale.size(0) == 1)
      << "global_scale must have shape [1]";

  const int M = logits.size(0);
  const int N = logits.size(1);
  TVM_FFI_ICHECK(N % NVFP4_BLOCK_SIZE == 0)
      << "N dimension must be divisible by " << NVFP4_BLOCK_SIZE;

  const int scale_cols = N / NVFP4_BLOCK_SIZE;
  const int padded_scale_rows = pad_up(M, 128);
  const int padded_scale_cols = pad_up(scale_cols, 4);

  TVM_FFI_ICHECK(fp4_output.ndim() == 2) << "fp4_output must be 2D";
  TVM_FFI_ICHECK(fp4_output.size(0) == M) << "fp4_output row count mismatch";
  TVM_FFI_ICHECK(fp4_output.size(1) == N / 2) << "fp4_output column count mismatch";
  TVM_FFI_ICHECK(block_scales.ndim() == 2) << "block_scales must be 2D";
  TVM_FFI_ICHECK(block_scales.size(0) == padded_scale_rows)
      << "block_scales row count mismatch";
  TVM_FFI_ICHECK(block_scales.size(1) == padded_scale_cols)
      << "block_scales column count mismatch";

  CHECK_DEVICE(logits, global_scale);
  CHECK_DEVICE(logits, fp4_output);
  CHECK_DEVICE(logits, block_scales);

  ffi::CUDADeviceGuard device_guard(logits.device().device_id);
  cudaStream_t stream = get_stream(logits.device());

  cudaMemsetAsync(block_scales.data_ptr(), 0, block_scales.size(0) * block_scales.size(1), stream);

  if (num_threads == 128) {
    launch_nvfp4_softmax_quant<128>(logits, global_scale, fp4_output, block_scales, M, N, stream);
  } else if (num_threads == 256) {
    launch_nvfp4_softmax_quant<256>(logits, global_scale, fp4_output, block_scales, M, N, stream);
  } else if (num_threads == 512) {
    launch_nvfp4_softmax_quant<512>(logits, global_scale, fp4_output, block_scales, M, N, stream);
  } else {
    TVM_FFI_ICHECK(false) << "num_threads must be 128, 256, or 512";
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_softmax_quant, nvfp4_softmax_quant);
