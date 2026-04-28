#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>

#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cutlass/float8.h>
#include <cutlass/float_subbyte.h>

#include <flashinfer/mma.cuh>

namespace {

constexpr int kQLen = 512;
constexpr int kGroup = 8;
constexpr int kKvLen = 32768;
constexpr int kHeadDim = 512;
constexpr int kPackedHeadDim = kHeadDim / 2;
constexpr int kScaleCols = kHeadDim / 16;
constexpr int kQRows = kQLen * kGroup;
constexpr int kDebugHead = 0;
constexpr int kDebugTileM = 16;
constexpr int kDebugTileN = 16;
constexpr int kQBlocks = kQLen / kDebugTileM;
#ifndef SM120_NVFP4_FUSED_KV_TILE
#define SM120_NVFP4_FUSED_KV_TILE 64
#endif
constexpr int kFusedKvTile = SM120_NVFP4_FUSED_KV_TILE;
static_assert(kFusedKvTile == 64 || kFusedKvTile == 128,
              "SM120_NVFP4_FUSED_KV_TILE must be 64 or 128");
constexpr int kFusedConsumerWarps = 16;
constexpr int kFusedWarps = 1 + kFusedConsumerWarps;
constexpr int kFusedThreads = kFusedWarps * 32;
constexpr int kPipelinedProducerWarps = kFusedKvTile / kDebugTileN;
constexpr int kPipelinedWarps = kPipelinedProducerWarps + kFusedConsumerWarps;
constexpr int kPipelinedThreads = kPipelinedWarps * 32;
#ifndef SM120_NVFP4_SPLIT_KV_LEN
#define SM120_NVFP4_SPLIT_KV_LEN 2048
#endif
constexpr int kSplitKvLen = SM120_NVFP4_SPLIT_KV_LEN;
static_assert(kKvLen % kSplitKvLen == 0,
              "SM120_NVFP4_SPLIT_KV_LEN must divide kKvLen");
constexpr int kNumSplits = kKvLen / kSplitKvLen;
constexpr int kHeadDimBlocks64 = kHeadDim / 64;
constexpr int kKvTiles16 = kKvLen / 16;
constexpr int kKvBlocks64 = kKvLen / 64;
constexpr int kOutTiles16 = kHeadDim / 16;
constexpr int kProbPackedCols = kKvLen / 2;
constexpr int kProbScaleCols = kKvLen / 16;
constexpr float kProbGlobalScale = 6.0f * 448.0f;
constexpr float kQkScale = 0.044194173824159216f;  // 1 / sqrt(512)

using Fp4MmaAtom =
    cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<cutlass::float_e2m1_t,
                                                  cutlass::float_e2m1_t, float,
                                                  cutlass::float_ue4m3_t, 16>;

__device__ __forceinline__ bool finite_f32(float x) {
  return x == x && fabsf(x) != INFINITY;
}

__device__ __forceinline__ uint8_t fp32_to_e4m3_byte(float x) {
  if (!(x > 0.0f) || !finite_f32(x)) {
    x = 1.0e-8f;
  }
  __nv_fp8_e4m3 y = static_cast<__nv_fp8_e4m3>(x);
  return y.__x;
}

__device__ __forceinline__ float e4m3_byte_to_fp32(uint8_t x) {
  __nv_fp8_e4m3 y;
  y.__x = x;
  return static_cast<float>(y);
}

__device__ __forceinline__ uint8_t nearest_e2m1_code(float x) {
  constexpr float values[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  x = fminf(fmaxf(finite_f32(x) ? x : 0.0f, -6.0f), 6.0f);
  float best_dist = fabsf(x - values[0]);
  uint8_t best = 0;
#pragma unroll
  for (uint8_t i = 1; i < 16; ++i) {
    const float dist = fabsf(x - values[i]);
    if (dist < best_dist) {
      best_dist = dist;
      best = i;
    }
  }
  return best;
}

__device__ __forceinline__ float reciprocal_approximate_ftz(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint8_t fp32_pair_to_e2m1_byte(float x, float y) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint16_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
      "mov.b16 %0, {byte0, 0};\n"
      "}"
      : "=h"(val)
      : "f"(x), "f"(y));
  return static_cast<uint8_t>(val);
#else
  return 0;
#endif
}

__global__ void zero_output_kernel(__nv_bfloat16* out) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(kQLen) * kGroup * kHeadDim;
  if (idx < total) {
    out[idx] = __float2bfloat16(0.0f);
  }
}

__global__ void quantize_q_rowmajor_kernel(const __nv_bfloat16* q, uint8_t* q_packed,
                                           uint8_t* q_scales) {
  const int row = blockIdx.x;
  const int scale_col = threadIdx.x;
  if (row >= kQRows || scale_col >= kScaleCols) {
    return;
  }

  const int base = row * kHeadDim + scale_col * 16;
  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    max_abs = fmaxf(max_abs, fabsf(__bfloat162float(q[base + i])));
  }
  const uint8_t scale_byte = fp32_to_e4m3_byte(fmaxf(max_abs / 6.0f, 1.0e-8f));
  q_scales[row * kScaleCols + scale_col] = scale_byte;
  const float scale = fmaxf(e4m3_byte_to_fp32(scale_byte), 1.0e-8f);

#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const float x0 = __bfloat162float(q[base + 2 * pair]) / scale;
    const float x1 = __bfloat162float(q[base + 2 * pair + 1]) / scale;
    const uint8_t c0 = nearest_e2m1_code(x0);
    const uint8_t c1 = nearest_e2m1_code(x1);
    q_packed[row * kPackedHeadDim + scale_col * 8 + pair] =
        static_cast<uint8_t>(c0 | (c1 << 4));
  }
}

__device__ __forceinline__ uint8_t get_nvfp4_code(const uint8_t* packed, int row,
                                                  int k) {
  const uint8_t byte = packed[row * kPackedHeadDim + (k >> 1)];
  return static_cast<uint8_t>((k & 1) ? ((byte >> 4) & 0x0f) : (byte & 0x0f));
}

__device__ __forceinline__ uint8_t get_prob_code(const uint8_t* packed, int row,
                                                 int kv) {
  const uint8_t byte = packed[row * kProbPackedCols + (kv >> 1)];
  return static_cast<uint8_t>((kv & 1) ? ((byte >> 4) & 0x0f) : (byte & 0x0f));
}

__device__ __forceinline__ uint32_t pack_e2m1_codes8(const uint8_t (&codes)[8]) {
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    packed |= static_cast<uint32_t>(codes[i] & 0x0f) << (4 * i);
  }
  return packed;
}

__device__ __forceinline__ int q_flat_row_for_debug_tile(int tile_row) {
  return tile_row * kGroup + kDebugHead;
}

__device__ __forceinline__ int q_flat_row_for_tile(int q_token_base, int head,
                                                   int tile_row) {
  return (q_token_base + tile_row) * kGroup + head;
}

__device__ __forceinline__ uint32_t make_q_scale_reg_from_rowmajor_tile(
    const uint8_t* q_scales, int k_base, int q_token_base, int head) {
  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::SFALayout layout;
  uint8_t s[4];
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    const int linear = int(layout(lane, slot));
    const int row = linear % kDebugTileM;
    const int k_group = linear / kDebugTileM;
    const int q_row = q_flat_row_for_tile(q_token_base, head, row);
    s[slot] = q_scales[q_row * kScaleCols + (k_base >> 4) + k_group];
  }
  return flashinfer::mma::pack_e4m3_scale_reg(s[0], s[1], s[2], s[3]);
}

__device__ __forceinline__ uint32_t make_q_scale_reg_from_rowmajor(
    const uint8_t* q_scales, int k_base) {
  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::SFALayout layout;
  uint8_t s[4];
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    const int linear = int(layout(lane, slot));
    const int row = linear % kDebugTileM;
    const int k_group = linear / kDebugTileM;
    const int q_row = q_flat_row_for_debug_tile(row);
    s[slot] = q_scales[q_row * kScaleCols + (k_base >> 4) + k_group];
  }
  return flashinfer::mma::pack_e4m3_scale_reg(s[0], s[1], s[2], s[3]);
}

__device__ __forceinline__ uint32_t make_k_scale_reg_from_rowmajor(
    const uint8_t* k_scales, int k_base, int atom_col_offset, int kv_base) {
  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::SFBLayout layout;
  uint8_t s[4];
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    const int linear = int(layout(lane, slot));
    const int col = kv_base + atom_col_offset + (linear % 8);
    const int k_group = linear / 8;
    s[slot] = k_scales[col * kScaleCols + (k_base >> 4) + k_group];
  }
  return flashinfer::mma::pack_e4m3_scale_reg(s[0], s[1], s[2], s[3]);
}

__device__ __forceinline__ void make_q_frag_from_rowmajor(const uint8_t* q_packed,
                                                          int k_base,
                                                          uint32_t (&frag)[4]) {
  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::ALayout layout;
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    uint8_t codes[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int linear = int(layout(lane, 8 * reg + i));
      const int row = linear % kDebugTileM;
      const int k = linear / kDebugTileM;
      codes[i] = get_nvfp4_code(q_packed, q_flat_row_for_debug_tile(row),
                                k_base + k);
    }
    frag[reg] = pack_e2m1_codes8(codes);
  }
}

__device__ __forceinline__ void make_q_frag_from_rowmajor_tile(
    const uint8_t* q_packed, int k_base, int q_token_base, int head,
    uint32_t (&frag)[4]) {
  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::ALayout layout;
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    uint8_t codes[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int linear = int(layout(lane, 8 * reg + i));
      const int row = linear % kDebugTileM;
      const int k = linear / kDebugTileM;
      codes[i] = get_nvfp4_code(q_packed, q_flat_row_for_tile(q_token_base, head, row),
                                k_base + k);
    }
    frag[reg] = pack_e2m1_codes8(codes);
  }
}

__device__ __forceinline__ void make_k_frag_from_rowmajor(const uint8_t* k_packed,
                                                          int k_base,
                                                          int atom_col_offset,
                                                          int kv_base,
                                                          uint32_t* frag) {
  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::BLayout layout;
#pragma unroll
  for (int reg = 0; reg < 2; ++reg) {
    uint8_t codes[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int linear = int(layout(lane, 8 * reg + i));
      const int col = kv_base + atom_col_offset + (linear % 8);
      const int k = linear / 8;
      codes[i] = get_nvfp4_code(k_packed, col, k_base + k);
    }
    frag[reg] = pack_e2m1_codes8(codes);
  }
}

__global__ void qk_tile_mma_debug_kernel(const uint8_t* q_packed,
                                         const uint8_t* q_scales,
                                         const uint8_t* k_packed,
                                         const uint8_t* k_scales,
                                         float* out_tile) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0.0f;
  }

#pragma unroll
  for (int k_base = 0; k_base < kHeadDim; k_base += 64) {
    uint32_t q_frag[4];
    uint32_t k_frag[4];
    const uint32_t q_scale = make_q_scale_reg_from_rowmajor(q_scales, k_base);
    const uint32_t k_scale0 = make_k_scale_reg_from_rowmajor(k_scales, k_base, 0, 0);
    const uint32_t k_scale1 = make_k_scale_reg_from_rowmajor(k_scales, k_base, 8, 0);
    make_q_frag_from_rowmajor(q_packed, k_base, q_frag);
    make_k_frag_from_rowmajor(k_packed, k_base, 0, 0, k_frag);
    make_k_frag_from_rowmajor(k_packed, k_base, 8, 0, k_frag + 2);

    if (k_base == 0) {
      flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32<
          flashinfer::mma::MMAMode::kInit>(acc, q_frag, k_frag, q_scale,
                                           k_scale0, k_scale1);
    } else {
      flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(
          acc, q_frag, k_frag, q_scale, k_scale0, k_scale1);
    }
  }

  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::CLayout c_layout;
#pragma unroll
  for (int value_idx = 0; value_idx < 4; ++value_idx) {
    const int linear = int(c_layout(lane, value_idx));
    const int row = linear % kDebugTileM;
    const int col = linear / kDebugTileM;
    out_tile[row * kDebugTileN + col] = acc[value_idx];
    out_tile[row * kDebugTileN + 8 + col] = acc[4 + value_idx];
  }
#else
  if (threadIdx.x == 0) {
    out_tile[0] = -1.0f;
  }
#endif
}

__global__ void qk_full_mma_debug_kernel(const uint8_t* q_packed,
                                         const uint8_t* q_scales,
                                         const uint8_t* k_packed,
                                         const uint8_t* k_scales,
                                         float* out_scores) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  const int kv_base = blockIdx.x * kDebugTileN;
  if (kv_base >= kKvLen) {
    return;
  }

  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0.0f;
  }

#pragma unroll
  for (int k_base = 0; k_base < kHeadDim; k_base += 64) {
    uint32_t q_frag[4];
    uint32_t k_frag[4];
    const uint32_t q_scale = make_q_scale_reg_from_rowmajor(q_scales, k_base);
    const uint32_t k_scale0 =
        make_k_scale_reg_from_rowmajor(k_scales, k_base, 0, kv_base);
    const uint32_t k_scale1 =
        make_k_scale_reg_from_rowmajor(k_scales, k_base, 8, kv_base);
    make_q_frag_from_rowmajor(q_packed, k_base, q_frag);
    make_k_frag_from_rowmajor(k_packed, k_base, 0, kv_base, k_frag);
    make_k_frag_from_rowmajor(k_packed, k_base, 8, kv_base, k_frag + 2);

    if (k_base == 0) {
      flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32<
          flashinfer::mma::MMAMode::kInit>(acc, q_frag, k_frag, q_scale,
                                           k_scale0, k_scale1);
    } else {
      flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(
          acc, q_frag, k_frag, q_scale, k_scale0, k_scale1);
    }
  }

  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::CLayout c_layout;
#pragma unroll
  for (int value_idx = 0; value_idx < 4; ++value_idx) {
    const int linear = int(c_layout(lane, value_idx));
    const int row = linear % kDebugTileM;
    const int col = linear / kDebugTileM;
    out_scores[row * kKvLen + kv_base + col] = acc[value_idx];
    out_scores[row * kKvLen + kv_base + 8 + col] = acc[4 + value_idx];
  }
#else
  if (threadIdx.x == 0) {
    out_scores[0] = -1.0f;
  }
#endif
}

__global__ void qk_all_mma_debug_kernel(const uint8_t* q_packed,
                                        const uint8_t* q_scales,
                                        const uint8_t* k_packed,
                                        const uint8_t* k_scales,
                                        float* out_scores) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  const int kv_base = blockIdx.x * kDebugTileN;
  const int q_block_head = blockIdx.y;
  const int q_block = q_block_head / kGroup;
  const int head = q_block_head - q_block * kGroup;
  const int q_token_base = q_block * kDebugTileM;
  if (kv_base >= kKvLen || q_token_base >= kQLen) {
    return;
  }

  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0.0f;
  }

#pragma unroll
  for (int k_base = 0; k_base < kHeadDim; k_base += 64) {
    uint32_t q_frag[4];
    uint32_t k_frag[4];
    const uint32_t q_scale =
        make_q_scale_reg_from_rowmajor_tile(q_scales, k_base, q_token_base, head);
    const uint32_t k_scale0 =
        make_k_scale_reg_from_rowmajor(k_scales, k_base, 0, kv_base);
    const uint32_t k_scale1 =
        make_k_scale_reg_from_rowmajor(k_scales, k_base, 8, kv_base);
    make_q_frag_from_rowmajor_tile(q_packed, k_base, q_token_base, head, q_frag);
    make_k_frag_from_rowmajor(k_packed, k_base, 0, kv_base, k_frag);
    make_k_frag_from_rowmajor(k_packed, k_base, 8, kv_base, k_frag + 2);

    if (k_base == 0) {
      flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32<
          flashinfer::mma::MMAMode::kInit>(acc, q_frag, k_frag, q_scale,
                                           k_scale0, k_scale1);
    } else {
      flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(
          acc, q_frag, k_frag, q_scale, k_scale0, k_scale1);
    }
  }

  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::CLayout c_layout;
#pragma unroll
  for (int value_idx = 0; value_idx < 4; ++value_idx) {
    const int linear = int(c_layout(lane, value_idx));
    const int row = linear % kDebugTileM;
    const int col = linear / kDebugTileM;
    const int out_row = q_flat_row_for_tile(q_token_base, head, row);
    out_scores[static_cast<int64_t>(out_row) * kKvLen + kv_base + col] =
        acc[value_idx];
    out_scores[static_cast<int64_t>(out_row) * kKvLen + kv_base + 8 + col] =
        acc[4 + value_idx];
  }
#else
  if (threadIdx.x == 0) {
    out_scores[0] = -1.0f;
  }
#endif
}

__global__ void softmax_quant_p_debug_kernel(const float* scores, uint8_t* p_packed,
                                             uint8_t* p_scales) {
  constexpr int kThreads = 256;
  static_assert(kKvLen % 16 == 0);
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= kDebugTileM) {
    return;
  }

  __shared__ float reduce[kThreads];
  const float* row_scores = scores + static_cast<int64_t>(row) * kKvLen;

  float thread_max = -INFINITY;
  for (int col = tid; col < kKvLen; col += kThreads) {
    thread_max = fmaxf(thread_max, row_scores[col] * kQkScale);
  }
  reduce[tid] = thread_max;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float row_max = reduce[0];

  float thread_sum = 0.0f;
  for (int col = tid; col < kKvLen; col += kThreads) {
    thread_sum += __expf(row_scores[col] * kQkScale - row_max);
  }
  reduce[tid] = thread_sum;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum = reciprocal_approximate_ftz(reduce[0]);

  uint8_t* row_packed = p_packed + static_cast<int64_t>(row) * kProbPackedCols;
  uint8_t* row_scales = p_scales + static_cast<int64_t>(row) * kProbScaleCols;
  for (int scale_col = tid; scale_col < kProbScaleCols; scale_col += kThreads) {
    const int base_col = scale_col * 16;
    float probs[16];
    float vec_max = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float p = __expf(row_scores[base_col + i] * kQkScale - row_max) * inv_sum;
      probs[i] = p;
      vec_max = fmaxf(vec_max, p);
    }

    const float requested_scale = fmaxf(kProbGlobalScale * vec_max / 6.0f, 1.0e-8f);
    const uint8_t scale_byte = fp32_to_e4m3_byte(requested_scale);
    row_scales[scale_col] = scale_byte;
    const float scale = fmaxf(e4m3_byte_to_fp32(scale_byte), 1.0e-8f);
    const float output_scale = kProbGlobalScale / scale;
    uint8_t* packed = row_packed + base_col / 2;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      packed[i] = fp32_pair_to_e2m1_byte(probs[2 * i] * output_scale,
                                         probs[2 * i + 1] * output_scale);
    }
  }
}

__global__ void softmax_quant_p_all_debug_kernel(const float* scores, uint8_t* p_packed,
                                                 uint8_t* p_scales) {
  constexpr int kThreads = 256;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= kQRows) {
    return;
  }

  __shared__ float reduce[kThreads];
  const float* row_scores = scores + static_cast<int64_t>(row) * kKvLen;

  float thread_max = -INFINITY;
  for (int col = tid; col < kKvLen; col += kThreads) {
    thread_max = fmaxf(thread_max, row_scores[col] * kQkScale);
  }
  reduce[tid] = thread_max;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float row_max = reduce[0];

  float thread_sum = 0.0f;
  for (int col = tid; col < kKvLen; col += kThreads) {
    thread_sum += __expf(row_scores[col] * kQkScale - row_max);
  }
  reduce[tid] = thread_sum;
  __syncthreads();

  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum = reciprocal_approximate_ftz(reduce[0]);

  uint8_t* row_packed = p_packed + static_cast<int64_t>(row) * kProbPackedCols;
  uint8_t* row_scales = p_scales + static_cast<int64_t>(row) * kProbScaleCols;
  for (int scale_col = tid; scale_col < kProbScaleCols; scale_col += kThreads) {
    const int base_col = scale_col * 16;
    float probs[16];
    float vec_max = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float p = __expf(row_scores[base_col + i] * kQkScale - row_max) * inv_sum;
      probs[i] = p;
      vec_max = fmaxf(vec_max, p);
    }

    const float requested_scale = fmaxf(kProbGlobalScale * vec_max / 6.0f, 1.0e-8f);
    const uint8_t scale_byte = fp32_to_e4m3_byte(requested_scale);
    row_scales[scale_col] = scale_byte;
    const float scale = fmaxf(e4m3_byte_to_fp32(scale_byte), 1.0e-8f);
    const float output_scale = kProbGlobalScale / scale;
    uint8_t* packed = row_packed + base_col / 2;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      packed[i] = fp32_pair_to_e2m1_byte(probs[2 * i] * output_scale,
                                         probs[2 * i + 1] * output_scale);
    }
  }
}

__device__ __forceinline__ uint32_t make_p_scale_reg_from_rowmajor(
    const uint8_t* p_scales, int kv_base) {
  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::SFALayout layout;
  uint8_t s[4];
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    const int linear = int(layout(lane, slot));
    const int row = linear % kDebugTileM;
    const int kv_group = linear / kDebugTileM;
    s[slot] = p_scales[row * kProbScaleCols + (kv_base >> 4) + kv_group];
  }
  return flashinfer::mma::pack_e4m3_scale_reg(s[0], s[1], s[2], s[3]);
}

__device__ __forceinline__ uint32_t make_v_pv_scale_reg_from_rowmajor(
    const uint8_t* v_pv_scales, int kv_base, int atom_col_offset,
    int out_col_base) {
  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::SFBLayout layout;
  uint8_t s[4];
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    const int linear = int(layout(lane, slot));
    const int col = out_col_base + atom_col_offset + (linear % 8);
    const int kv_group = linear / 8;
    s[slot] = v_pv_scales[col * kProbScaleCols + (kv_base >> 4) + kv_group];
  }
  return flashinfer::mma::pack_e4m3_scale_reg(s[0], s[1], s[2], s[3]);
}

__device__ __forceinline__ void make_p_frag_from_rowmajor(const uint8_t* p_packed,
                                                          int kv_base,
                                                          uint32_t (&frag)[4]) {
  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::ALayout layout;
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    uint8_t codes[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int linear = int(layout(lane, 8 * reg + i));
      const int row = linear % kDebugTileM;
      const int kv = linear / kDebugTileM;
      codes[i] = get_prob_code(p_packed, row, kv_base + kv);
    }
    frag[reg] = pack_e2m1_codes8(codes);
  }
}

__device__ __forceinline__ uint32_t make_p_scale_reg_from_rowmajor_tile(
    const uint8_t* p_scales, int kv_base, int q_token_base, int head) {
  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::SFALayout layout;
  uint8_t s[4];
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    const int linear = int(layout(lane, slot));
    const int row = linear % kDebugTileM;
    const int kv_group = linear / kDebugTileM;
    const int p_row = q_flat_row_for_tile(q_token_base, head, row);
    s[slot] = p_scales[p_row * kProbScaleCols + (kv_base >> 4) + kv_group];
  }
  return flashinfer::mma::pack_e4m3_scale_reg(s[0], s[1], s[2], s[3]);
}

__device__ __forceinline__ void make_p_frag_from_rowmajor_tile(
    const uint8_t* p_packed, int kv_base, int q_token_base, int head,
    uint32_t (&frag)[4]) {
  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::ALayout layout;
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    uint8_t codes[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int linear = int(layout(lane, 8 * reg + i));
      const int row = linear % kDebugTileM;
      const int kv = linear / kDebugTileM;
      codes[i] =
          get_prob_code(p_packed, q_flat_row_for_tile(q_token_base, head, row),
                        kv_base + kv);
    }
    frag[reg] = pack_e2m1_codes8(codes);
  }
}

__device__ __forceinline__ uint8_t get_p_smem_code(const uint8_t* p_packed,
                                                   int row, int kv_local) {
  const uint8_t byte = p_packed[row * (kFusedKvTile / 2) + (kv_local >> 1)];
  return static_cast<uint8_t>((kv_local & 1) ? ((byte >> 4) & 0x0f) : (byte & 0x0f));
}

__device__ __forceinline__ uint32_t make_p_scale_reg_from_smem(
    const uint8_t* p_scales, int kv_local_base = 0) {
  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::SFALayout layout;
  uint8_t s[4];
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    const int linear = int(layout(lane, slot));
    const int row = linear % kDebugTileM;
    const int kv_group = linear / kDebugTileM;
    s[slot] =
        p_scales[row * (kFusedKvTile / 16) + (kv_local_base / 16) + kv_group];
  }
  return flashinfer::mma::pack_e4m3_scale_reg(s[0], s[1], s[2], s[3]);
}

__device__ __forceinline__ void make_p_frag_from_smem(const uint8_t* p_packed,
                                                      uint32_t (&frag)[4],
                                                      int kv_local_base = 0) {
  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::ALayout layout;
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    uint8_t codes[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int linear = int(layout(lane, 8 * reg + i));
      const int row = linear % kDebugTileM;
      const int kv = linear / kDebugTileM;
      codes[i] = get_p_smem_code(p_packed, row, kv_local_base + kv);
    }
    frag[reg] = pack_e2m1_codes8(codes);
  }
}

__device__ __forceinline__ void make_v_pv_frag_from_rowmajor(
    const uint8_t* v_pv_packed, int kv_base, int atom_col_offset,
    int out_col_base, uint32_t* frag) {
  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::BLayout layout;
#pragma unroll
  for (int reg = 0; reg < 2; ++reg) {
    uint8_t codes[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int linear = int(layout(lane, 8 * reg + i));
      const int col = out_col_base + atom_col_offset + (linear % 8);
      const int kv = linear / 8;
      codes[i] = get_prob_code(v_pv_packed, col, kv_base + kv);
    }
    frag[reg] = pack_e2m1_codes8(codes);
  }
}

__global__ void pv_tile_mma_debug_kernel(const uint8_t* p_packed,
                                         const uint8_t* p_scales,
                                         const uint8_t* v_pv_packed,
                                         const uint8_t* v_pv_scales,
                                         float* out_tile) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  constexpr int kOutColBase = 0;
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0.0f;
  }

  for (int kv_base = 0; kv_base < kKvLen; kv_base += 64) {
    uint32_t p_frag[4];
    uint32_t v_frag[4];
    const uint32_t p_scale = make_p_scale_reg_from_rowmajor(p_scales, kv_base);
    const uint32_t v_scale0 =
        make_v_pv_scale_reg_from_rowmajor(v_pv_scales, kv_base, 0, kOutColBase);
    const uint32_t v_scale1 =
        make_v_pv_scale_reg_from_rowmajor(v_pv_scales, kv_base, 8, kOutColBase);
    make_p_frag_from_rowmajor(p_packed, kv_base, p_frag);
    make_v_pv_frag_from_rowmajor(v_pv_packed, kv_base, 0, kOutColBase, v_frag);
    make_v_pv_frag_from_rowmajor(v_pv_packed, kv_base, 8, kOutColBase, v_frag + 2);

    if (kv_base == 0) {
      flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32<
          flashinfer::mma::MMAMode::kInit>(acc, p_frag, v_frag, p_scale,
                                           v_scale0, v_scale1);
    } else {
      flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(
          acc, p_frag, v_frag, p_scale, v_scale0, v_scale1);
    }
  }

  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::CLayout c_layout;
#pragma unroll
  for (int value_idx = 0; value_idx < 4; ++value_idx) {
    const int linear = int(c_layout(lane, value_idx));
    const int row = linear % kDebugTileM;
    const int col = linear / kDebugTileM;
    out_tile[row * kDebugTileN + col] = acc[value_idx];
    out_tile[row * kDebugTileN + 8 + col] = acc[4 + value_idx];
  }
#else
  if (threadIdx.x == 0) {
    out_tile[0] = -1.0f;
  }
#endif
}

__global__ void pv_full_mma_debug_kernel(const uint8_t* p_packed,
                                         const uint8_t* p_scales,
                                         const uint8_t* v_pv_packed,
                                         const uint8_t* v_pv_scales,
                                         float* out_block) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  const int out_col_base = blockIdx.x * kDebugTileN;
  if (out_col_base >= kHeadDim) {
    return;
  }

  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0.0f;
  }

  for (int kv_base = 0; kv_base < kKvLen; kv_base += 64) {
    uint32_t p_frag[4];
    uint32_t v_frag[4];
    const uint32_t p_scale = make_p_scale_reg_from_rowmajor(p_scales, kv_base);
    const uint32_t v_scale0 =
        make_v_pv_scale_reg_from_rowmajor(v_pv_scales, kv_base, 0, out_col_base);
    const uint32_t v_scale1 =
        make_v_pv_scale_reg_from_rowmajor(v_pv_scales, kv_base, 8, out_col_base);
    make_p_frag_from_rowmajor(p_packed, kv_base, p_frag);
    make_v_pv_frag_from_rowmajor(v_pv_packed, kv_base, 0, out_col_base, v_frag);
    make_v_pv_frag_from_rowmajor(v_pv_packed, kv_base, 8, out_col_base, v_frag + 2);

    if (kv_base == 0) {
      flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32<
          flashinfer::mma::MMAMode::kInit>(acc, p_frag, v_frag, p_scale,
                                           v_scale0, v_scale1);
    } else {
      flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(
          acc, p_frag, v_frag, p_scale, v_scale0, v_scale1);
    }
  }

  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::CLayout c_layout;
#pragma unroll
  for (int value_idx = 0; value_idx < 4; ++value_idx) {
    const int linear = int(c_layout(lane, value_idx));
    const int row = linear % kDebugTileM;
    const int col = linear / kDebugTileM;
    out_block[row * kHeadDim + out_col_base + col] = acc[value_idx];
    out_block[row * kHeadDim + out_col_base + 8 + col] = acc[4 + value_idx];
  }
#else
  if (threadIdx.x == 0) {
    out_block[0] = -1.0f;
  }
#endif
}

__global__ void pv_all_mma_debug_kernel(const uint8_t* p_packed,
                                        const uint8_t* p_scales,
                                        const uint8_t* v_pv_packed,
                                        const uint8_t* v_pv_scales,
                                        float* out_rows) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  const int out_col_base = blockIdx.x * kDebugTileN;
  const int q_block_head = blockIdx.y;
  const int q_block = q_block_head / kGroup;
  const int head = q_block_head - q_block * kGroup;
  const int q_token_base = q_block * kDebugTileM;
  if (out_col_base >= kHeadDim || q_token_base >= kQLen) {
    return;
  }

  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0.0f;
  }

  for (int kv_base = 0; kv_base < kKvLen; kv_base += 64) {
    uint32_t p_frag[4];
    uint32_t v_frag[4];
    const uint32_t p_scale =
        make_p_scale_reg_from_rowmajor_tile(p_scales, kv_base, q_token_base, head);
    const uint32_t v_scale0 =
        make_v_pv_scale_reg_from_rowmajor(v_pv_scales, kv_base, 0, out_col_base);
    const uint32_t v_scale1 =
        make_v_pv_scale_reg_from_rowmajor(v_pv_scales, kv_base, 8, out_col_base);
    make_p_frag_from_rowmajor_tile(p_packed, kv_base, q_token_base, head, p_frag);
    make_v_pv_frag_from_rowmajor(v_pv_packed, kv_base, 0, out_col_base, v_frag);
    make_v_pv_frag_from_rowmajor(v_pv_packed, kv_base, 8, out_col_base, v_frag + 2);

    if (kv_base == 0) {
      flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32<
          flashinfer::mma::MMAMode::kInit>(acc, p_frag, v_frag, p_scale,
                                           v_scale0, v_scale1);
    } else {
      flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(
          acc, p_frag, v_frag, p_scale, v_scale0, v_scale1);
    }
  }

  const int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Fp4MmaAtom>::CLayout c_layout;
#pragma unroll
  for (int value_idx = 0; value_idx < 4; ++value_idx) {
    const int linear = int(c_layout(lane, value_idx));
    const int row = linear % kDebugTileM;
    const int col = linear / kDebugTileM;
    const int out_row = q_flat_row_for_tile(q_token_base, head, row);
    out_rows[out_row * kHeadDim + out_col_base + col] = acc[value_idx];
    out_rows[out_row * kHeadDim + out_col_base + 8 + col] = acc[4 + value_idx];
  }
#else
  if (threadIdx.x == 0) {
    out_rows[0] = -1.0f;
  }
#endif
}

__global__ void prepack_k_fragments_kernel(const uint8_t* k_packed,
                                           const uint8_t* k_scales,
                                           int32_t* k_frag_pre,
                                           int32_t* k_scale_pre) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  const int tile16 = blockIdx.x;
  const int k_block = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int kv_base = tile16 * 16;
  const int k_base = k_block * 64;
  uint32_t frag[4];
  make_k_frag_from_rowmajor(k_packed, k_base, 0, kv_base, frag);
  make_k_frag_from_rowmajor(k_packed, k_base, 8, kv_base, frag + 2);
  const uint32_t scale0 = make_k_scale_reg_from_rowmajor(k_scales, k_base, 0, kv_base);
  const uint32_t scale1 = make_k_scale_reg_from_rowmajor(k_scales, k_base, 8, kv_base);
  int64_t frag_base =
      (((static_cast<int64_t>(tile16) * kHeadDimBlocks64 + k_block) * 32 + lane) * 4);
  int64_t scale_base =
      (((static_cast<int64_t>(tile16) * kHeadDimBlocks64 + k_block) * 32 + lane) * 2);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    k_frag_pre[frag_base + i] = static_cast<int32_t>(frag[i]);
  }
  k_scale_pre[scale_base] = static_cast<int32_t>(scale0);
  k_scale_pre[scale_base + 1] = static_cast<int32_t>(scale1);
#endif
}

__global__ void prepack_v_fragments_kernel(const uint8_t* v_pv_packed,
                                           const uint8_t* v_pv_scales,
                                           int32_t* v_frag_pre,
                                           int32_t* v_scale_pre) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  const int out_tile = blockIdx.x;
  const int kv_block = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int out_col_base = out_tile * 16;
  const int kv_base = kv_block * 64;
  uint32_t frag[4];
  make_v_pv_frag_from_rowmajor(v_pv_packed, kv_base, 0, out_col_base, frag);
  make_v_pv_frag_from_rowmajor(v_pv_packed, kv_base, 8, out_col_base, frag + 2);
  const uint32_t scale0 =
      make_v_pv_scale_reg_from_rowmajor(v_pv_scales, kv_base, 0, out_col_base);
  const uint32_t scale1 =
      make_v_pv_scale_reg_from_rowmajor(v_pv_scales, kv_base, 8, out_col_base);
  int64_t frag_base =
      (((static_cast<int64_t>(out_tile) * kKvBlocks64 + kv_block) * 32 + lane) * 4);
  int64_t scale_base =
      (((static_cast<int64_t>(out_tile) * kKvBlocks64 + kv_block) * 32 + lane) * 2);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v_frag_pre[frag_base + i] = static_cast<int32_t>(frag[i]);
  }
  v_scale_pre[scale_base] = static_cast<int32_t>(scale0);
  v_scale_pre[scale_base + 1] = static_cast<int32_t>(scale1);
#endif
}

__device__ __forceinline__ void load_prepacked_k(const int32_t* k_frag_pre,
                                                 const int32_t* k_scale_pre,
                                                 int tile16, int k_block,
                                                 uint32_t (&frag)[4],
                                                 uint32_t& scale0,
                                                 uint32_t& scale1) {
  const int lane = threadIdx.x & 31;
  const int64_t frag_base =
      (((static_cast<int64_t>(tile16) * kHeadDimBlocks64 + k_block) * 32 + lane) * 4);
  const int64_t scale_base =
      (((static_cast<int64_t>(tile16) * kHeadDimBlocks64 + k_block) * 32 + lane) * 2);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    frag[i] = static_cast<uint32_t>(k_frag_pre[frag_base + i]);
  }
  scale0 = static_cast<uint32_t>(k_scale_pre[scale_base]);
  scale1 = static_cast<uint32_t>(k_scale_pre[scale_base + 1]);
}

__device__ __forceinline__ void load_prepacked_v(const int32_t* v_frag_pre,
                                                 const int32_t* v_scale_pre,
                                                 int out_col_base, int kv_base,
                                                 uint32_t (&frag)[4],
                                                 uint32_t& scale0,
                                                 uint32_t& scale1) {
  const int lane = threadIdx.x & 31;
  const int out_tile = out_col_base / 16;
  const int kv_block = kv_base / 64;
  const int64_t frag_base =
      (((static_cast<int64_t>(out_tile) * kKvBlocks64 + kv_block) * 32 + lane) * 4);
  const int64_t scale_base =
      (((static_cast<int64_t>(out_tile) * kKvBlocks64 + kv_block) * 32 + lane) * 2);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    frag[i] = static_cast<uint32_t>(v_frag_pre[frag_base + i]);
  }
  scale0 = static_cast<uint32_t>(v_scale_pre[scale_base]);
  scale1 = static_cast<uint32_t>(v_scale_pre[scale_base + 1]);
}

__launch_bounds__(kFusedThreads, 1) __global__ void fused_attention_all_debug_kernel(
    const uint8_t* q_packed, const uint8_t* q_scales, const uint8_t* k_packed,
    const uint8_t* k_scales, const uint8_t* v_pv_packed,
    const uint8_t* v_pv_scales, float* out_rows) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  __shared__ float scores[kDebugTileM][kFusedKvTile];
  __shared__ uint8_t p_packed_smem[kDebugTileM][kFusedKvTile / 2];
  __shared__ uint8_t p_scales_smem[kDebugTileM][kFusedKvTile / 16];
  __shared__ float row_m[kDebugTileM];
  __shared__ float row_l[kDebugTileM];
  __shared__ float row_alpha[kDebugTileM];

  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int q_block_head = blockIdx.x;
  const int q_block = q_block_head / kGroup;
  const int head = q_block_head - q_block * kGroup;
  const int q_token_base = q_block * kDebugTileM;

  if (warp_id == 0 && lane < kDebugTileM) {
    row_m[lane] = -INFINITY;
    row_l[lane] = 0.0f;
    row_alpha[lane] = 0.0f;
  }

  float acc_a[8];
  float acc_b[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc_a[i] = 0.0f;
    acc_b[i] = 0.0f;
  }

  const int consumer_idx = warp_id - 1;
  const int out_col_a = consumer_idx * kDebugTileN;
  const int out_col_b = (consumer_idx + kFusedConsumerWarps) * kDebugTileN;

  __syncthreads();

  for (int kv_base = 0; kv_base < kKvLen; kv_base += kFusedKvTile) {
    if (warp_id == 0) {
#pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
        float qk_acc[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          qk_acc[i] = 0.0f;
        }

#pragma unroll
        for (int k_base = 0; k_base < kHeadDim; k_base += 64) {
          uint32_t q_frag[4];
          uint32_t k_frag[4];
          const int sub_kv_base = kv_base + sub * kDebugTileN;
          const uint32_t q_scale =
              make_q_scale_reg_from_rowmajor_tile(q_scales, k_base, q_token_base, head);
          const uint32_t k_scale0 =
              make_k_scale_reg_from_rowmajor(k_scales, k_base, 0, sub_kv_base);
          const uint32_t k_scale1 =
              make_k_scale_reg_from_rowmajor(k_scales, k_base, 8, sub_kv_base);
          make_q_frag_from_rowmajor_tile(q_packed, k_base, q_token_base, head, q_frag);
          make_k_frag_from_rowmajor(k_packed, k_base, 0, sub_kv_base, k_frag);
          make_k_frag_from_rowmajor(k_packed, k_base, 8, sub_kv_base, k_frag + 2);

          if (k_base == 0) {
            flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32<
                flashinfer::mma::MMAMode::kInit>(qk_acc, q_frag, k_frag,
                                                 q_scale, k_scale0, k_scale1);
          } else {
            flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(
                qk_acc, q_frag, k_frag, q_scale, k_scale0, k_scale1);
          }
        }

        typename cute::MMA_Traits<Fp4MmaAtom>::CLayout c_layout;
#pragma unroll
        for (int value_idx = 0; value_idx < 4; ++value_idx) {
          const int linear = int(c_layout(lane, value_idx));
          const int row = linear % kDebugTileM;
          const int col = linear / kDebugTileM;
          scores[row][sub * kDebugTileN + col] = qk_acc[value_idx];
          scores[row][sub * kDebugTileN + 8 + col] = qk_acc[4 + value_idx];
        }
      }

      if (lane < kDebugTileM) {
        const int row = lane;
        float tile_max = -INFINITY;
#pragma unroll
        for (int col = 0; col < kFusedKvTile; ++col) {
          tile_max = fmaxf(tile_max, scores[row][col] * kQkScale);
        }
        const float old_m = row_m[row];
        const float old_l = row_l[row];
        const float new_m = fmaxf(old_m, tile_max);
        const float alpha = __expf(old_m - new_m);
        float beta_sum = 0.0f;
#pragma unroll
        for (int col = 0; col < kFusedKvTile; ++col) {
          const float b = __expf(scores[row][col] * kQkScale - new_m);
          scores[row][col] = b;
          beta_sum += b;
        }
        row_alpha[row] = alpha;
        row_l[row] = old_l * alpha + beta_sum;
        row_m[row] = new_m;

#pragma unroll
        for (int group = 0; group < kFusedKvTile / 16; ++group) {
          float vec_max = 0.0f;
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            const int col = group * 16 + i;
            vec_max = fmaxf(vec_max, scores[row][col]);
          }
          const float requested_scale =
              fmaxf(kProbGlobalScale * vec_max / 6.0f, 1.0e-8f);
          const uint8_t scale_byte = fp32_to_e4m3_byte(requested_scale);
          p_scales_smem[row][group] = scale_byte;
          const float scale = fmaxf(e4m3_byte_to_fp32(scale_byte), 1.0e-8f);
          const float output_scale = kProbGlobalScale / scale;
#pragma unroll
          for (int pair = 0; pair < 8; ++pair) {
            const int col0 = group * 16 + 2 * pair;
            const int col1 = col0 + 1;
            p_packed_smem[row][group * 8 + pair] =
                fp32_pair_to_e2m1_byte(scores[row][col0] * output_scale,
                                       scores[row][col1] * output_scale);
          }
        }
      }
    }

    __syncthreads();

    if (warp_id > 0) {
      typename cute::MMA_Traits<Fp4MmaAtom>::CLayout c_layout;
#pragma unroll
      for (int value_idx = 0; value_idx < 4; ++value_idx) {
        const int linear = int(c_layout(lane, value_idx));
        const int row = linear % kDebugTileM;
        const float alpha = row_alpha[row];
        acc_a[value_idx] *= alpha;
        acc_a[4 + value_idx] *= alpha;
        acc_b[value_idx] *= alpha;
        acc_b[4 + value_idx] *= alpha;
      }

      uint32_t p_frag[4];
      uint32_t v_frag[4];
      const uint32_t p_scale =
          make_p_scale_reg_from_smem(&p_scales_smem[0][0]);
      make_p_frag_from_smem(&p_packed_smem[0][0], p_frag);

      uint32_t v_scale0 =
          make_v_pv_scale_reg_from_rowmajor(v_pv_scales, kv_base, 0, out_col_a);
      uint32_t v_scale1 =
          make_v_pv_scale_reg_from_rowmajor(v_pv_scales, kv_base, 8, out_col_a);
      make_v_pv_frag_from_rowmajor(v_pv_packed, kv_base, 0, out_col_a, v_frag);
      make_v_pv_frag_from_rowmajor(v_pv_packed, kv_base, 8, out_col_a, v_frag + 2);
      flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(
          acc_a, p_frag, v_frag, p_scale, v_scale0, v_scale1);

      v_scale0 =
          make_v_pv_scale_reg_from_rowmajor(v_pv_scales, kv_base, 0, out_col_b);
      v_scale1 =
          make_v_pv_scale_reg_from_rowmajor(v_pv_scales, kv_base, 8, out_col_b);
      make_v_pv_frag_from_rowmajor(v_pv_packed, kv_base, 0, out_col_b, v_frag);
      make_v_pv_frag_from_rowmajor(v_pv_packed, kv_base, 8, out_col_b, v_frag + 2);
      flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(
          acc_b, p_frag, v_frag, p_scale, v_scale0, v_scale1);
    }

    __syncthreads();
  }

  if (warp_id > 0) {
    typename cute::MMA_Traits<Fp4MmaAtom>::CLayout c_layout;
#pragma unroll
    for (int value_idx = 0; value_idx < 4; ++value_idx) {
      const int linear = int(c_layout(lane, value_idx));
      const int row = linear % kDebugTileM;
      const int col = linear / kDebugTileM;
      const int out_row = q_flat_row_for_tile(q_token_base, head, row);
      const float denom = fmaxf(row_l[row] * kProbGlobalScale, 1.0e-20f);
      out_rows[out_row * kHeadDim + out_col_a + col] = acc_a[value_idx] / denom;
      out_rows[out_row * kHeadDim + out_col_a + 8 + col] = acc_a[4 + value_idx] / denom;
      out_rows[out_row * kHeadDim + out_col_b + col] = acc_b[value_idx] / denom;
      out_rows[out_row * kHeadDim + out_col_b + 8 + col] = acc_b[4 + value_idx] / denom;
    }
  }
#else
  if (threadIdx.x == 0) {
    out_rows[0] = -1.0f;
  }
#endif
}

__launch_bounds__(kFusedThreads, 1) __global__ void fused_attention_split_partial_kernel(
    const uint8_t* q_packed, const uint8_t* q_scales, const int32_t* k_frag_pre,
    const int32_t* k_scale_pre, const int32_t* v_frag_pre,
    const int32_t* v_scale_pre, float* partial_o, float* partial_m, float* partial_l) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  __shared__ float scores[kDebugTileM][kFusedKvTile];
  __shared__ uint8_t p_packed_smem[kDebugTileM][kFusedKvTile / 2];
  __shared__ uint8_t p_scales_smem[kDebugTileM][kFusedKvTile / 16];
  __shared__ float row_m[kDebugTileM];
  __shared__ float row_l[kDebugTileM];
  __shared__ float row_alpha[kDebugTileM];

  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int q_block_head = blockIdx.x;
  const int split_id = blockIdx.y;
  const int q_block = q_block_head / kGroup;
  const int head = q_block_head - q_block * kGroup;
  const int q_token_base = q_block * kDebugTileM;
  const int kv_start = split_id * kSplitKvLen;
  const int kv_stop = kv_start + kSplitKvLen;

  if (warp_id == 0 && lane < kDebugTileM) {
    row_m[lane] = -INFINITY;
    row_l[lane] = 0.0f;
    row_alpha[lane] = 0.0f;
  }

  float acc_a[8];
  float acc_b[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc_a[i] = 0.0f;
    acc_b[i] = 0.0f;
  }

  const int consumer_idx = warp_id - 1;
  const int out_col_a = consumer_idx * kDebugTileN;
  const int out_col_b = (consumer_idx + kFusedConsumerWarps) * kDebugTileN;

  __syncthreads();

  for (int kv_base = kv_start; kv_base < kv_stop; kv_base += kFusedKvTile) {
    if (warp_id == 0) {
#pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
        float qk_acc[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          qk_acc[i] = 0.0f;
        }

#pragma unroll
        for (int k_base = 0; k_base < kHeadDim; k_base += 64) {
          uint32_t q_frag[4];
          uint32_t k_frag[4];
          const int sub_kv_base = kv_base + sub * kDebugTileN;
          const uint32_t q_scale =
              make_q_scale_reg_from_rowmajor_tile(q_scales, k_base, q_token_base, head);
          uint32_t k_scale0;
          uint32_t k_scale1;
          make_q_frag_from_rowmajor_tile(q_packed, k_base, q_token_base, head, q_frag);
          load_prepacked_k(k_frag_pre, k_scale_pre, sub_kv_base / 16, k_base / 64,
                           k_frag, k_scale0, k_scale1);

          if (k_base == 0) {
            flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32<
                flashinfer::mma::MMAMode::kInit>(qk_acc, q_frag, k_frag,
                                                 q_scale, k_scale0, k_scale1);
          } else {
            flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(
                qk_acc, q_frag, k_frag, q_scale, k_scale0, k_scale1);
          }
        }

        typename cute::MMA_Traits<Fp4MmaAtom>::CLayout c_layout;
#pragma unroll
        for (int value_idx = 0; value_idx < 4; ++value_idx) {
          const int linear = int(c_layout(lane, value_idx));
          const int row = linear % kDebugTileM;
          const int col = linear / kDebugTileM;
          scores[row][sub * kDebugTileN + col] = qk_acc[value_idx];
          scores[row][sub * kDebugTileN + 8 + col] = qk_acc[4 + value_idx];
        }
      }

      if (lane < kDebugTileM) {
        const int row = lane;
        float tile_max = -INFINITY;
#pragma unroll
        for (int col = 0; col < kFusedKvTile; ++col) {
          tile_max = fmaxf(tile_max, scores[row][col] * kQkScale);
        }
        const float old_m = row_m[row];
        const float old_l = row_l[row];
        const float new_m = fmaxf(old_m, tile_max);
        const float alpha = __expf(old_m - new_m);
        float beta_sum = 0.0f;
#pragma unroll
        for (int col = 0; col < kFusedKvTile; ++col) {
          const float b = __expf(scores[row][col] * kQkScale - new_m);
          scores[row][col] = b;
          beta_sum += b;
        }
        row_alpha[row] = alpha;
        row_l[row] = old_l * alpha + beta_sum;
        row_m[row] = new_m;

#pragma unroll
        for (int group = 0; group < kFusedKvTile / 16; ++group) {
          float vec_max = 0.0f;
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            const int col = group * 16 + i;
            vec_max = fmaxf(vec_max, scores[row][col]);
          }
          const float requested_scale =
              fmaxf(kProbGlobalScale * vec_max / 6.0f, 1.0e-8f);
          const uint8_t scale_byte = fp32_to_e4m3_byte(requested_scale);
          p_scales_smem[row][group] = scale_byte;
          const float scale = fmaxf(e4m3_byte_to_fp32(scale_byte), 1.0e-8f);
          const float output_scale = kProbGlobalScale / scale;
#pragma unroll
          for (int pair = 0; pair < 8; ++pair) {
            const int col0 = group * 16 + 2 * pair;
            const int col1 = col0 + 1;
            p_packed_smem[row][group * 8 + pair] =
                fp32_pair_to_e2m1_byte(scores[row][col0] * output_scale,
                                       scores[row][col1] * output_scale);
          }
        }
      }
    }

    __syncthreads();

    if (warp_id > 0) {
      typename cute::MMA_Traits<Fp4MmaAtom>::CLayout c_layout;
#pragma unroll
      for (int value_idx = 0; value_idx < 4; ++value_idx) {
        const int linear = int(c_layout(lane, value_idx));
        const int row = linear % kDebugTileM;
        const float alpha = row_alpha[row];
        acc_a[value_idx] *= alpha;
        acc_a[4 + value_idx] *= alpha;
        acc_b[value_idx] *= alpha;
        acc_b[4 + value_idx] *= alpha;
      }

      uint32_t p_frag[4];
      uint32_t v_frag[4];
      const uint32_t p_scale =
          make_p_scale_reg_from_smem(&p_scales_smem[0][0]);
      make_p_frag_from_smem(&p_packed_smem[0][0], p_frag);

      uint32_t v_scale0;
      uint32_t v_scale1;
      load_prepacked_v(v_frag_pre, v_scale_pre, out_col_a, kv_base, v_frag,
                       v_scale0, v_scale1);
      flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(
          acc_a, p_frag, v_frag, p_scale, v_scale0, v_scale1);

      load_prepacked_v(v_frag_pre, v_scale_pre, out_col_b, kv_base, v_frag,
                       v_scale0, v_scale1);
      flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(
          acc_b, p_frag, v_frag, p_scale, v_scale0, v_scale1);
    }

    __syncthreads();
  }

  if (warp_id == 0 && lane < kDebugTileM) {
    const int out_row = q_flat_row_for_tile(q_token_base, head, lane);
    partial_m[split_id * kQRows + out_row] = row_m[lane];
    partial_l[split_id * kQRows + out_row] = row_l[lane];
  }

  if (warp_id > 0) {
    typename cute::MMA_Traits<Fp4MmaAtom>::CLayout c_layout;
#pragma unroll
    for (int value_idx = 0; value_idx < 4; ++value_idx) {
      const int linear = int(c_layout(lane, value_idx));
      const int row = linear % kDebugTileM;
      const int col = linear / kDebugTileM;
      const int out_row = q_flat_row_for_tile(q_token_base, head, row);
      float* partial_row = partial_o +
                           (static_cast<int64_t>(split_id) * kQRows + out_row) *
                               kHeadDim;
      partial_row[out_col_a + col] = acc_a[value_idx];
      partial_row[out_col_a + 8 + col] = acc_a[4 + value_idx];
      partial_row[out_col_b + col] = acc_b[value_idx];
      partial_row[out_col_b + 8 + col] = acc_b[4 + value_idx];
    }
  }
#else
  if (threadIdx.x == 0) {
    partial_o[0] = -1.0f;
  }
#endif
}

__launch_bounds__(kPipelinedThreads, 1) __global__ void
fused_attention_split_partial_pipelined_kernel(
    const uint8_t* q_packed, const uint8_t* q_scales, const int32_t* k_frag_pre,
    const int32_t* k_scale_pre, const int32_t* v_frag_pre,
    const int32_t* v_scale_pre, float* partial_o, float* partial_m,
    float* partial_l) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  __shared__ float scores[kDebugTileM][kFusedKvTile];
  __shared__ uint8_t p_packed_smem[2][kDebugTileM][kFusedKvTile / 2];
  __shared__ uint8_t p_scales_smem[2][kDebugTileM][kFusedKvTile / 16];
  __shared__ float row_alpha_smem[2][kDebugTileM];
  __shared__ float row_m[kDebugTileM];
  __shared__ float row_l[kDebugTileM];
  __shared__ int tile_start[2];
  __shared__ int producer_done[2];
  __shared__ int tile_ready[2];
  __shared__ int tile_done[2];

  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int q_block_head = blockIdx.x;
  const int split_id = blockIdx.y;
  const int q_block = q_block_head / kGroup;
  const int head = q_block_head - q_block * kGroup;
  const int q_token_base = q_block * kDebugTileM;
  const int kv_start = split_id * kSplitKvLen;
  constexpr int kTilesPerSplit = kSplitKvLen / kFusedKvTile;

  if (threadIdx.x == 0) {
    tile_start[0] = -1;
    tile_start[1] = -1;
    producer_done[0] = 0;
    producer_done[1] = 0;
    tile_ready[0] = -1;
    tile_ready[1] = -1;
    tile_done[0] = kFusedConsumerWarps;
    tile_done[1] = kFusedConsumerWarps;
  }
  if (warp_id == 0 && lane < kDebugTileM) {
    row_m[lane] = -INFINITY;
    row_l[lane] = 0.0f;
  }

  float acc_a[8];
  float acc_b[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc_a[i] = 0.0f;
    acc_b[i] = 0.0f;
  }

  const int consumer_idx = warp_id - kPipelinedProducerWarps;
  const int out_col_a = consumer_idx * kDebugTileN;
  const int out_col_b = (consumer_idx + kFusedConsumerWarps) * kDebugTileN;

  __syncthreads();

  if (warp_id < kPipelinedProducerWarps) {
#pragma unroll
    for (int tile = 0; tile < kTilesPerSplit; ++tile) {
      const int buf = tile & 1;
      if (warp_id == 0 && lane == 0) {
        volatile int* done_view = tile_done;
        while (done_view[buf] != kFusedConsumerWarps) {
          __nanosleep(32);
        }
        tile_done[buf] = 0;
        producer_done[buf] = 0;
        __threadfence_block();
        tile_start[buf] = tile;
      }
      __syncwarp();
      if (lane == 0) {
        volatile int* start_view = tile_start;
        while (start_view[buf] != tile) {
          __nanosleep(32);
        }
      }
      __syncwarp();

      const int kv_base = kv_start + tile * kFusedKvTile;
      const int sub = warp_id;
      float qk_acc[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        qk_acc[i] = 0.0f;
      }

#pragma unroll
      for (int k_base = 0; k_base < kHeadDim; k_base += 64) {
        uint32_t q_frag[4];
        uint32_t k_frag[4];
        const int sub_kv_base = kv_base + sub * kDebugTileN;
        const uint32_t q_scale =
            make_q_scale_reg_from_rowmajor_tile(q_scales, k_base, q_token_base, head);
        uint32_t k_scale0;
        uint32_t k_scale1;
        make_q_frag_from_rowmajor_tile(q_packed, k_base, q_token_base, head, q_frag);
        load_prepacked_k(k_frag_pre, k_scale_pre, sub_kv_base / 16, k_base / 64,
                         k_frag, k_scale0, k_scale1);

        if (k_base == 0) {
          flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32<
              flashinfer::mma::MMAMode::kInit>(qk_acc, q_frag, k_frag,
                                               q_scale, k_scale0, k_scale1);
        } else {
          flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(
              qk_acc, q_frag, k_frag, q_scale, k_scale0, k_scale1);
        }
      }

      typename cute::MMA_Traits<Fp4MmaAtom>::CLayout c_layout;
#pragma unroll
      for (int value_idx = 0; value_idx < 4; ++value_idx) {
        const int linear = int(c_layout(lane, value_idx));
        const int row = linear % kDebugTileM;
        const int col = linear / kDebugTileM;
        scores[row][sub * kDebugTileN + col] = qk_acc[value_idx];
        scores[row][sub * kDebugTileN + 8 + col] = qk_acc[4 + value_idx];
      }

      __threadfence_block();
      __syncwarp();
      if (lane == 0) {
        atomicAdd(&producer_done[buf], 1);
      }
      __syncwarp();

      if (warp_id == 0 && lane == 0) {
        volatile int* producer_done_view = producer_done;
        while (producer_done_view[buf] != kPipelinedProducerWarps) {
          __nanosleep(32);
        }
      }
      __syncwarp();

      if (warp_id == 0 && lane < kDebugTileM) {
        const int row = lane;
        float tile_max = -INFINITY;
#pragma unroll
        for (int col = 0; col < kFusedKvTile; ++col) {
          tile_max = fmaxf(tile_max, scores[row][col] * kQkScale);
        }
        const float old_m = row_m[row];
        const float old_l = row_l[row];
        const float new_m = fmaxf(old_m, tile_max);
        const float alpha = __expf(old_m - new_m);
        float beta_sum = 0.0f;
#pragma unroll
        for (int col = 0; col < kFusedKvTile; ++col) {
          const float b = __expf(scores[row][col] * kQkScale - new_m);
          scores[row][col] = b;
          beta_sum += b;
        }
        row_alpha_smem[buf][row] = alpha;
        row_l[row] = old_l * alpha + beta_sum;
        row_m[row] = new_m;

#pragma unroll
        for (int group = 0; group < kFusedKvTile / 16; ++group) {
          float vec_max = 0.0f;
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            const int col = group * 16 + i;
            vec_max = fmaxf(vec_max, scores[row][col]);
          }
          const float requested_scale =
              fmaxf(kProbGlobalScale * vec_max / 6.0f, 1.0e-8f);
          const uint8_t scale_byte = fp32_to_e4m3_byte(requested_scale);
          p_scales_smem[buf][row][group] = scale_byte;
          const float scale = fmaxf(e4m3_byte_to_fp32(scale_byte), 1.0e-8f);
          const float output_scale = kProbGlobalScale / scale;
#pragma unroll
          for (int pair = 0; pair < 8; ++pair) {
            const int col0 = group * 16 + 2 * pair;
            const int col1 = col0 + 1;
            p_packed_smem[buf][row][group * 8 + pair] =
                fp32_pair_to_e2m1_byte(scores[row][col0] * output_scale,
                                       scores[row][col1] * output_scale);
          }
        }
      }

      if (warp_id == 0) {
        __threadfence_block();
      }
      __syncwarp();
      if (warp_id == 0 && lane == 0) {
        tile_ready[buf] = tile;
      }
      __syncwarp();
      if (lane == 0) {
        volatile int* ready_view = tile_ready;
        while (ready_view[buf] != tile) {
          __nanosleep(32);
        }
      }
      __syncwarp();
    }
  } else {
#pragma unroll
    for (int tile = 0; tile < kTilesPerSplit; ++tile) {
      const int buf = tile & 1;
      if (lane == 0) {
        volatile int* ready_view = tile_ready;
        while (ready_view[buf] != tile) {
          __nanosleep(32);
        }
      }
      __syncwarp();

      typename cute::MMA_Traits<Fp4MmaAtom>::CLayout c_layout;
#pragma unroll
      for (int value_idx = 0; value_idx < 4; ++value_idx) {
        const int linear = int(c_layout(lane, value_idx));
        const int row = linear % kDebugTileM;
        const float alpha = row_alpha_smem[buf][row];
        acc_a[value_idx] *= alpha;
        acc_a[4 + value_idx] *= alpha;
        acc_b[value_idx] *= alpha;
        acc_b[4 + value_idx] *= alpha;
      }

      const int kv_base = kv_start + tile * kFusedKvTile;
      uint32_t p_frag[4];
      uint32_t v_frag[4];
#pragma unroll
      for (int pv_k = 0; pv_k < kFusedKvTile; pv_k += 64) {
        const uint32_t p_scale =
            make_p_scale_reg_from_smem(&p_scales_smem[buf][0][0], pv_k);
        make_p_frag_from_smem(&p_packed_smem[buf][0][0], p_frag, pv_k);

        uint32_t v_scale0;
        uint32_t v_scale1;
        load_prepacked_v(v_frag_pre, v_scale_pre, out_col_a, kv_base + pv_k,
                         v_frag, v_scale0, v_scale1);
        flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(
            acc_a, p_frag, v_frag, p_scale, v_scale0, v_scale1);

        load_prepacked_v(v_frag_pre, v_scale_pre, out_col_b, kv_base + pv_k,
                         v_frag, v_scale0, v_scale1);
        flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(
            acc_b, p_frag, v_frag, p_scale, v_scale0, v_scale1);
      }

      __syncwarp();
      if (lane == 0) {
        atomicAdd(&tile_done[buf], 1);
      }
    }
  }

  __syncthreads();

  if (warp_id == 0 && lane < kDebugTileM) {
    const int out_row = q_flat_row_for_tile(q_token_base, head, lane);
    partial_m[split_id * kQRows + out_row] = row_m[lane];
    partial_l[split_id * kQRows + out_row] = row_l[lane];
  }

  if (warp_id > 0) {
    typename cute::MMA_Traits<Fp4MmaAtom>::CLayout c_layout;
#pragma unroll
    for (int value_idx = 0; value_idx < 4; ++value_idx) {
      const int linear = int(c_layout(lane, value_idx));
      const int row = linear % kDebugTileM;
      const int col = linear / kDebugTileM;
      const int out_row = q_flat_row_for_tile(q_token_base, head, row);
      float* partial_row = partial_o +
                           (static_cast<int64_t>(split_id) * kQRows + out_row) *
                               kHeadDim;
      partial_row[out_col_a + col] = acc_a[value_idx];
      partial_row[out_col_a + 8 + col] = acc_a[4 + value_idx];
      partial_row[out_col_b + col] = acc_b[value_idx];
      partial_row[out_col_b + 8 + col] = acc_b[4 + value_idx];
    }
  }
#else
  if (threadIdx.x == 0) {
    partial_o[0] = -1.0f;
  }
#endif
}

__global__ void fused_attention_split_weights_kernel(const float* partial_m,
                                                     const float* partial_l,
                                                     float* weights) {
  const int row = blockIdx.x;
  if (row >= kQRows || threadIdx.x != 0) {
    return;
  }
  float final_m = -INFINITY;
#pragma unroll
  for (int split = 0; split < kNumSplits; ++split) {
    final_m = fmaxf(final_m, partial_m[split * kQRows + row]);
  }
  float final_l = 0.0f;
#pragma unroll
  for (int split = 0; split < kNumSplits; ++split) {
    const float scale = __expf(partial_m[split * kQRows + row] - final_m);
    final_l += partial_l[split * kQRows + row] * scale;
  }
  const float inv_denom = reciprocal_approximate_ftz(final_l * kProbGlobalScale);
#pragma unroll
  for (int split = 0; split < kNumSplits; ++split) {
    weights[split * kQRows + row] =
        __expf(partial_m[split * kQRows + row] - final_m) * inv_denom;
  }
}

__global__ void fused_attention_split_reduce_kernel(const float* partial_o,
                                                    const float* weights,
                                                    float* out_rows) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(kQRows) * kHeadDim;
  if (idx >= total) {
    return;
  }
  const int row = static_cast<int>(idx / kHeadDim);
  const int col = static_cast<int>(idx - static_cast<int64_t>(row) * kHeadDim);
  float acc = 0.0f;
#pragma unroll
  for (int split = 0; split < kNumSplits; ++split) {
    acc += partial_o[(static_cast<int64_t>(split) * kQRows + row) * kHeadDim + col] *
           weights[split * kQRows + row];
  }
  out_rows[idx] = acc;
}

void check_tensor(const torch::Tensor& tensor, const char* name, torch::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has unexpected dtype");
}

}  // namespace

void attention_b1_d512_g8_q512_kv32768(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                       torch::Tensor k_scales, torch::Tensor v_scales,
                                       torch::Tensor out) {
  check_tensor(q, "q", torch::kBFloat16);
  check_tensor(k, "k", torch::kUInt8);
  check_tensor(v, "v", torch::kUInt8);
  check_tensor(k_scales, "k_scales", torch::kUInt8);
  check_tensor(v_scales, "v_scales", torch::kUInt8);
  check_tensor(out, "out", torch::kBFloat16);

  TORCH_CHECK(q.sizes() == torch::IntArrayRef({kQLen, kGroup, kHeadDim}),
              "q must have shape [512, 8, 512]");
  TORCH_CHECK(k.sizes() == torch::IntArrayRef({kKvLen, kPackedHeadDim}),
              "k must have shape [32768, 256]");
  TORCH_CHECK(v.sizes() == torch::IntArrayRef({kKvLen, kPackedHeadDim}),
              "v must have shape [32768, 256]");
  TORCH_CHECK(k_scales.sizes() == torch::IntArrayRef({kKvLen, kScaleCols}),
              "k_scales must have shape [32768, 32]");
  TORCH_CHECK(v_scales.sizes() == torch::IntArrayRef({kKvLen, kScaleCols}),
              "v_scales must have shape [32768, 32]");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({kQLen, kGroup, kHeadDim}),
              "out must have shape [512, 8, 512]");

  auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  const int64_t total = static_cast<int64_t>(kQLen) * kGroup * kHeadDim;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  zero_output_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(out_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void quantize_q_rowmajor(torch::Tensor q, torch::Tensor q_packed, torch::Tensor q_scales) {
  check_tensor(q, "q", torch::kBFloat16);
  check_tensor(q_packed, "q_packed", torch::kUInt8);
  check_tensor(q_scales, "q_scales", torch::kUInt8);
  TORCH_CHECK(q.sizes() == torch::IntArrayRef({kQLen, kGroup, kHeadDim}),
              "q must have shape [512, 8, 512]");
  TORCH_CHECK(q_packed.sizes() == torch::IntArrayRef({kQRows, kPackedHeadDim}),
              "q_packed must have shape [4096, 256]");
  TORCH_CHECK(q_scales.sizes() == torch::IntArrayRef({kQRows, kScaleCols}),
              "q_scales must have shape [4096, 32]");

  const auto* q_ptr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
  auto* packed_ptr = q_packed.data_ptr<uint8_t>();
  auto* scale_ptr = q_scales.data_ptr<uint8_t>();
  quantize_q_rowmajor_kernel<<<kQRows, kScaleCols, 0, at::cuda::getCurrentCUDAStream()>>>(
      q_ptr, packed_ptr, scale_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qk_tile_mma_debug(torch::Tensor q_packed, torch::Tensor q_scales, torch::Tensor k,
                       torch::Tensor k_scales, torch::Tensor out_tile) {
  check_tensor(q_packed, "q_packed", torch::kUInt8);
  check_tensor(q_scales, "q_scales", torch::kUInt8);
  check_tensor(k, "k", torch::kUInt8);
  check_tensor(k_scales, "k_scales", torch::kUInt8);
  check_tensor(out_tile, "out_tile", torch::kFloat32);
  TORCH_CHECK(q_packed.sizes() == torch::IntArrayRef({kQRows, kPackedHeadDim}),
              "q_packed must have shape [4096, 256]");
  TORCH_CHECK(q_scales.sizes() == torch::IntArrayRef({kQRows, kScaleCols}),
              "q_scales must have shape [4096, 32]");
  TORCH_CHECK(k.sizes() == torch::IntArrayRef({kKvLen, kPackedHeadDim}),
              "k must have shape [32768, 256]");
  TORCH_CHECK(k_scales.sizes() == torch::IntArrayRef({kKvLen, kScaleCols}),
              "k_scales must have shape [32768, 32]");
  TORCH_CHECK(out_tile.sizes() == torch::IntArrayRef({kDebugTileM, kDebugTileN}),
              "out_tile must have shape [16, 16]");

  qk_tile_mma_debug_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      q_packed.data_ptr<uint8_t>(), q_scales.data_ptr<uint8_t>(),
      k.data_ptr<uint8_t>(), k_scales.data_ptr<uint8_t>(),
      out_tile.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qk_full_mma_debug(torch::Tensor q_packed, torch::Tensor q_scales, torch::Tensor k,
                       torch::Tensor k_scales, torch::Tensor out_scores) {
  check_tensor(q_packed, "q_packed", torch::kUInt8);
  check_tensor(q_scales, "q_scales", torch::kUInt8);
  check_tensor(k, "k", torch::kUInt8);
  check_tensor(k_scales, "k_scales", torch::kUInt8);
  check_tensor(out_scores, "out_scores", torch::kFloat32);
  TORCH_CHECK(q_packed.sizes() == torch::IntArrayRef({kQRows, kPackedHeadDim}),
              "q_packed must have shape [4096, 256]");
  TORCH_CHECK(q_scales.sizes() == torch::IntArrayRef({kQRows, kScaleCols}),
              "q_scales must have shape [4096, 32]");
  TORCH_CHECK(k.sizes() == torch::IntArrayRef({kKvLen, kPackedHeadDim}),
              "k must have shape [32768, 256]");
  TORCH_CHECK(k_scales.sizes() == torch::IntArrayRef({kKvLen, kScaleCols}),
              "k_scales must have shape [32768, 32]");
  TORCH_CHECK(out_scores.sizes() == torch::IntArrayRef({kDebugTileM, kKvLen}),
              "out_scores must have shape [16, 32768]");

  qk_full_mma_debug_kernel<<<kKvLen / kDebugTileN, 32, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
      q_packed.data_ptr<uint8_t>(), q_scales.data_ptr<uint8_t>(),
      k.data_ptr<uint8_t>(), k_scales.data_ptr<uint8_t>(),
      out_scores.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qk_all_mma_debug(torch::Tensor q_packed, torch::Tensor q_scales, torch::Tensor k,
                      torch::Tensor k_scales, torch::Tensor out_scores) {
  check_tensor(q_packed, "q_packed", torch::kUInt8);
  check_tensor(q_scales, "q_scales", torch::kUInt8);
  check_tensor(k, "k", torch::kUInt8);
  check_tensor(k_scales, "k_scales", torch::kUInt8);
  check_tensor(out_scores, "out_scores", torch::kFloat32);
  TORCH_CHECK(q_packed.sizes() == torch::IntArrayRef({kQRows, kPackedHeadDim}),
              "q_packed must have shape [4096, 256]");
  TORCH_CHECK(q_scales.sizes() == torch::IntArrayRef({kQRows, kScaleCols}),
              "q_scales must have shape [4096, 32]");
  TORCH_CHECK(k.sizes() == torch::IntArrayRef({kKvLen, kPackedHeadDim}),
              "k must have shape [32768, 256]");
  TORCH_CHECK(k_scales.sizes() == torch::IntArrayRef({kKvLen, kScaleCols}),
              "k_scales must have shape [32768, 32]");
  TORCH_CHECK(out_scores.sizes() == torch::IntArrayRef({kQRows, kKvLen}),
              "out_scores must have shape [4096, 32768]");

  dim3 grid(kKvLen / kDebugTileN, kQBlocks * kGroup);
  qk_all_mma_debug_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      q_packed.data_ptr<uint8_t>(), q_scales.data_ptr<uint8_t>(),
      k.data_ptr<uint8_t>(), k_scales.data_ptr<uint8_t>(),
      out_scores.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void softmax_quant_p_debug(torch::Tensor scores, torch::Tensor p_packed,
                           torch::Tensor p_scales) {
  check_tensor(scores, "scores", torch::kFloat32);
  check_tensor(p_packed, "p_packed", torch::kUInt8);
  check_tensor(p_scales, "p_scales", torch::kUInt8);
  TORCH_CHECK(scores.sizes() == torch::IntArrayRef({kDebugTileM, kKvLen}),
              "scores must have shape [16, 32768]");
  TORCH_CHECK(p_packed.sizes() == torch::IntArrayRef({kDebugTileM, kProbPackedCols}),
              "p_packed must have shape [16, 16384]");
  TORCH_CHECK(p_scales.sizes() == torch::IntArrayRef({kDebugTileM, kProbScaleCols}),
              "p_scales must have shape [16, 2048]");

  softmax_quant_p_debug_kernel<<<kDebugTileM, 256, 0,
                                 at::cuda::getCurrentCUDAStream()>>>(
      scores.data_ptr<float>(), p_packed.data_ptr<uint8_t>(),
      p_scales.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void softmax_quant_p_all_debug(torch::Tensor scores, torch::Tensor p_packed,
                               torch::Tensor p_scales) {
  check_tensor(scores, "scores", torch::kFloat32);
  check_tensor(p_packed, "p_packed", torch::kUInt8);
  check_tensor(p_scales, "p_scales", torch::kUInt8);
  TORCH_CHECK(scores.sizes() == torch::IntArrayRef({kQRows, kKvLen}),
              "scores must have shape [4096, 32768]");
  TORCH_CHECK(p_packed.sizes() == torch::IntArrayRef({kQRows, kProbPackedCols}),
              "p_packed must have shape [4096, 16384]");
  TORCH_CHECK(p_scales.sizes() == torch::IntArrayRef({kQRows, kProbScaleCols}),
              "p_scales must have shape [4096, 2048]");

  softmax_quant_p_all_debug_kernel<<<kQRows, 256, 0,
                                     at::cuda::getCurrentCUDAStream()>>>(
      scores.data_ptr<float>(), p_packed.data_ptr<uint8_t>(),
      p_scales.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pv_tile_mma_debug(torch::Tensor p_packed, torch::Tensor p_scales,
                       torch::Tensor v_pv, torch::Tensor v_pv_scales,
                       torch::Tensor out_tile) {
  check_tensor(p_packed, "p_packed", torch::kUInt8);
  check_tensor(p_scales, "p_scales", torch::kUInt8);
  check_tensor(v_pv, "v_pv", torch::kUInt8);
  check_tensor(v_pv_scales, "v_pv_scales", torch::kUInt8);
  check_tensor(out_tile, "out_tile", torch::kFloat32);
  TORCH_CHECK(p_packed.sizes() == torch::IntArrayRef({kDebugTileM, kProbPackedCols}),
              "p_packed must have shape [16, 16384]");
  TORCH_CHECK(p_scales.sizes() == torch::IntArrayRef({kDebugTileM, kProbScaleCols}),
              "p_scales must have shape [16, 2048]");
  TORCH_CHECK(v_pv.sizes() == torch::IntArrayRef({kHeadDim, kProbPackedCols}),
              "v_pv must have shape [512, 16384]");
  TORCH_CHECK(v_pv_scales.sizes() == torch::IntArrayRef({kHeadDim, kProbScaleCols}),
              "v_pv_scales must have shape [512, 2048]");
  TORCH_CHECK(out_tile.sizes() == torch::IntArrayRef({kDebugTileM, kDebugTileN}),
              "out_tile must have shape [16, 16]");

  pv_tile_mma_debug_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      p_packed.data_ptr<uint8_t>(), p_scales.data_ptr<uint8_t>(),
      v_pv.data_ptr<uint8_t>(), v_pv_scales.data_ptr<uint8_t>(),
      out_tile.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pv_all_mma_debug(torch::Tensor p_packed, torch::Tensor p_scales,
                      torch::Tensor v_pv, torch::Tensor v_pv_scales,
                      torch::Tensor out_rows) {
  check_tensor(p_packed, "p_packed", torch::kUInt8);
  check_tensor(p_scales, "p_scales", torch::kUInt8);
  check_tensor(v_pv, "v_pv", torch::kUInt8);
  check_tensor(v_pv_scales, "v_pv_scales", torch::kUInt8);
  check_tensor(out_rows, "out_rows", torch::kFloat32);
  TORCH_CHECK(p_packed.sizes() == torch::IntArrayRef({kQRows, kProbPackedCols}),
              "p_packed must have shape [4096, 16384]");
  TORCH_CHECK(p_scales.sizes() == torch::IntArrayRef({kQRows, kProbScaleCols}),
              "p_scales must have shape [4096, 2048]");
  TORCH_CHECK(v_pv.sizes() == torch::IntArrayRef({kHeadDim, kProbPackedCols}),
              "v_pv must have shape [512, 16384]");
  TORCH_CHECK(v_pv_scales.sizes() == torch::IntArrayRef({kHeadDim, kProbScaleCols}),
              "v_pv_scales must have shape [512, 2048]");
  TORCH_CHECK(out_rows.sizes() == torch::IntArrayRef({kQRows, kHeadDim}),
              "out_rows must have shape [4096, 512]");

  dim3 grid(kHeadDim / kDebugTileN, kQBlocks * kGroup);
  pv_all_mma_debug_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      p_packed.data_ptr<uint8_t>(), p_scales.data_ptr<uint8_t>(),
      v_pv.data_ptr<uint8_t>(), v_pv_scales.data_ptr<uint8_t>(),
      out_rows.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prepack_k_fragments_debug(torch::Tensor k, torch::Tensor k_scales,
                               torch::Tensor k_frag_pre,
                               torch::Tensor k_scale_pre) {
  check_tensor(k, "k", torch::kUInt8);
  check_tensor(k_scales, "k_scales", torch::kUInt8);
  check_tensor(k_frag_pre, "k_frag_pre", torch::kInt32);
  check_tensor(k_scale_pre, "k_scale_pre", torch::kInt32);
  TORCH_CHECK(k.sizes() == torch::IntArrayRef({kKvLen, kPackedHeadDim}),
              "k must have shape [32768, 256]");
  TORCH_CHECK(k_scales.sizes() == torch::IntArrayRef({kKvLen, kScaleCols}),
              "k_scales must have shape [32768, 32]");
  TORCH_CHECK(
      k_frag_pre.sizes() ==
          torch::IntArrayRef({kKvTiles16, kHeadDimBlocks64, 32, 4}),
      "k_frag_pre must have shape [2048, 8, 32, 4]");
  TORCH_CHECK(
      k_scale_pre.sizes() ==
          torch::IntArrayRef({kKvTiles16, kHeadDimBlocks64, 32, 2}),
      "k_scale_pre must have shape [2048, 8, 32, 2]");
  dim3 grid(kKvTiles16, kHeadDimBlocks64);
  prepack_k_fragments_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      k.data_ptr<uint8_t>(), k_scales.data_ptr<uint8_t>(),
      k_frag_pre.data_ptr<int32_t>(), k_scale_pre.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prepack_v_fragments_debug(torch::Tensor v_pv, torch::Tensor v_pv_scales,
                               torch::Tensor v_frag_pre,
                               torch::Tensor v_scale_pre) {
  check_tensor(v_pv, "v_pv", torch::kUInt8);
  check_tensor(v_pv_scales, "v_pv_scales", torch::kUInt8);
  check_tensor(v_frag_pre, "v_frag_pre", torch::kInt32);
  check_tensor(v_scale_pre, "v_scale_pre", torch::kInt32);
  TORCH_CHECK(v_pv.sizes() == torch::IntArrayRef({kHeadDim, kProbPackedCols}),
              "v_pv must have shape [512, 16384]");
  TORCH_CHECK(v_pv_scales.sizes() == torch::IntArrayRef({kHeadDim, kProbScaleCols}),
              "v_pv_scales must have shape [512, 2048]");
  TORCH_CHECK(
      v_frag_pre.sizes() ==
          torch::IntArrayRef({kOutTiles16, kKvBlocks64, 32, 4}),
      "v_frag_pre must have shape [32, 512, 32, 4]");
  TORCH_CHECK(
      v_scale_pre.sizes() ==
          torch::IntArrayRef({kOutTiles16, kKvBlocks64, 32, 2}),
      "v_scale_pre must have shape [32, 512, 32, 2]");
  dim3 grid(kOutTiles16, kKvBlocks64);
  prepack_v_fragments_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      v_pv.data_ptr<uint8_t>(), v_pv_scales.data_ptr<uint8_t>(),
      v_frag_pre.data_ptr<int32_t>(), v_scale_pre.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_attention_all_debug(torch::Tensor q_packed, torch::Tensor q_scales,
                               torch::Tensor k, torch::Tensor k_scales,
                               torch::Tensor v_pv, torch::Tensor v_pv_scales,
                               torch::Tensor out_rows) {
  check_tensor(q_packed, "q_packed", torch::kUInt8);
  check_tensor(q_scales, "q_scales", torch::kUInt8);
  check_tensor(k, "k", torch::kUInt8);
  check_tensor(k_scales, "k_scales", torch::kUInt8);
  check_tensor(v_pv, "v_pv", torch::kUInt8);
  check_tensor(v_pv_scales, "v_pv_scales", torch::kUInt8);
  check_tensor(out_rows, "out_rows", torch::kFloat32);
  TORCH_CHECK(q_packed.sizes() == torch::IntArrayRef({kQRows, kPackedHeadDim}),
              "q_packed must have shape [4096, 256]");
  TORCH_CHECK(q_scales.sizes() == torch::IntArrayRef({kQRows, kScaleCols}),
              "q_scales must have shape [4096, 32]");
  TORCH_CHECK(k.sizes() == torch::IntArrayRef({kKvLen, kPackedHeadDim}),
              "k must have shape [32768, 256]");
  TORCH_CHECK(k_scales.sizes() == torch::IntArrayRef({kKvLen, kScaleCols}),
              "k_scales must have shape [32768, 32]");
  TORCH_CHECK(v_pv.sizes() == torch::IntArrayRef({kHeadDim, kProbPackedCols}),
              "v_pv must have shape [512, 16384]");
  TORCH_CHECK(v_pv_scales.sizes() == torch::IntArrayRef({kHeadDim, kProbScaleCols}),
              "v_pv_scales must have shape [512, 2048]");
  TORCH_CHECK(out_rows.sizes() == torch::IntArrayRef({kQRows, kHeadDim}),
              "out_rows must have shape [4096, 512]");

  fused_attention_all_debug_kernel<<<kQBlocks * kGroup, kFusedThreads, 0,
                                     at::cuda::getCurrentCUDAStream()>>>(
      q_packed.data_ptr<uint8_t>(), q_scales.data_ptr<uint8_t>(),
      k.data_ptr<uint8_t>(), k_scales.data_ptr<uint8_t>(),
      v_pv.data_ptr<uint8_t>(), v_pv_scales.data_ptr<uint8_t>(),
      out_rows.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_attention_split_partial_debug(torch::Tensor q_packed, torch::Tensor q_scales,
                                         torch::Tensor k_frag_pre,
                                         torch::Tensor k_scale_pre,
                                         torch::Tensor v_frag_pre,
                                         torch::Tensor v_scale_pre,
                                         torch::Tensor partial_o,
                                         torch::Tensor partial_m,
                                         torch::Tensor partial_l) {
  check_tensor(q_packed, "q_packed", torch::kUInt8);
  check_tensor(q_scales, "q_scales", torch::kUInt8);
  check_tensor(k_frag_pre, "k_frag_pre", torch::kInt32);
  check_tensor(k_scale_pre, "k_scale_pre", torch::kInt32);
  check_tensor(v_frag_pre, "v_frag_pre", torch::kInt32);
  check_tensor(v_scale_pre, "v_scale_pre", torch::kInt32);
  check_tensor(partial_o, "partial_o", torch::kFloat32);
  check_tensor(partial_m, "partial_m", torch::kFloat32);
  check_tensor(partial_l, "partial_l", torch::kFloat32);
  TORCH_CHECK(q_packed.sizes() == torch::IntArrayRef({kQRows, kPackedHeadDim}),
              "q_packed must have shape [4096, 256]");
  TORCH_CHECK(q_scales.sizes() == torch::IntArrayRef({kQRows, kScaleCols}),
              "q_scales must have shape [4096, 32]");
  TORCH_CHECK(
      k_frag_pre.sizes() ==
          torch::IntArrayRef({kKvTiles16, kHeadDimBlocks64, 32, 4}),
      "k_frag_pre must have shape [2048, 8, 32, 4]");
  TORCH_CHECK(
      k_scale_pre.sizes() ==
          torch::IntArrayRef({kKvTiles16, kHeadDimBlocks64, 32, 2}),
      "k_scale_pre must have shape [2048, 8, 32, 2]");
  TORCH_CHECK(
      v_frag_pre.sizes() ==
          torch::IntArrayRef({kOutTiles16, kKvBlocks64, 32, 4}),
      "v_frag_pre must have shape [32, 512, 32, 4]");
  TORCH_CHECK(
      v_scale_pre.sizes() ==
          torch::IntArrayRef({kOutTiles16, kKvBlocks64, 32, 2}),
      "v_scale_pre must have shape [32, 512, 32, 2]");
  TORCH_CHECK(partial_o.sizes() == torch::IntArrayRef({kNumSplits, kQRows, kHeadDim}),
              "partial_o must have shape [32, 4096, 512]");
  TORCH_CHECK(partial_m.sizes() == torch::IntArrayRef({kNumSplits, kQRows}),
              "partial_m must have shape [32, 4096]");
  TORCH_CHECK(partial_l.sizes() == torch::IntArrayRef({kNumSplits, kQRows}),
              "partial_l must have shape [32, 4096]");

  dim3 grid(kQBlocks * kGroup, kNumSplits);
  fused_attention_split_partial_kernel<<<grid, kFusedThreads, 0,
                                         at::cuda::getCurrentCUDAStream()>>>(
      q_packed.data_ptr<uint8_t>(), q_scales.data_ptr<uint8_t>(),
      k_frag_pre.data_ptr<int32_t>(), k_scale_pre.data_ptr<int32_t>(),
      v_frag_pre.data_ptr<int32_t>(), v_scale_pre.data_ptr<int32_t>(),
      partial_o.data_ptr<float>(), partial_m.data_ptr<float>(),
      partial_l.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_attention_split_partial_pipelined_debug(
    torch::Tensor q_packed, torch::Tensor q_scales, torch::Tensor k_frag_pre,
    torch::Tensor k_scale_pre, torch::Tensor v_frag_pre,
    torch::Tensor v_scale_pre, torch::Tensor partial_o, torch::Tensor partial_m,
    torch::Tensor partial_l) {
  check_tensor(q_packed, "q_packed", torch::kUInt8);
  check_tensor(q_scales, "q_scales", torch::kUInt8);
  check_tensor(k_frag_pre, "k_frag_pre", torch::kInt32);
  check_tensor(k_scale_pre, "k_scale_pre", torch::kInt32);
  check_tensor(v_frag_pre, "v_frag_pre", torch::kInt32);
  check_tensor(v_scale_pre, "v_scale_pre", torch::kInt32);
  check_tensor(partial_o, "partial_o", torch::kFloat32);
  check_tensor(partial_m, "partial_m", torch::kFloat32);
  check_tensor(partial_l, "partial_l", torch::kFloat32);
  TORCH_CHECK(q_packed.sizes() == torch::IntArrayRef({kQRows, kPackedHeadDim}),
              "q_packed must have shape [4096, 256]");
  TORCH_CHECK(q_scales.sizes() == torch::IntArrayRef({kQRows, kScaleCols}),
              "q_scales must have shape [4096, 32]");
  TORCH_CHECK(
      k_frag_pre.sizes() ==
          torch::IntArrayRef({kKvTiles16, kHeadDimBlocks64, 32, 4}),
      "k_frag_pre must have shape [2048, 8, 32, 4]");
  TORCH_CHECK(
      k_scale_pre.sizes() ==
          torch::IntArrayRef({kKvTiles16, kHeadDimBlocks64, 32, 2}),
      "k_scale_pre must have shape [2048, 8, 32, 2]");
  TORCH_CHECK(
      v_frag_pre.sizes() ==
          torch::IntArrayRef({kOutTiles16, kKvBlocks64, 32, 4}),
      "v_frag_pre must have shape [32, 512, 32, 4]");
  TORCH_CHECK(
      v_scale_pre.sizes() ==
          torch::IntArrayRef({kOutTiles16, kKvBlocks64, 32, 2}),
      "v_scale_pre must have shape [32, 512, 32, 2]");
  TORCH_CHECK(partial_o.sizes() == torch::IntArrayRef({kNumSplits, kQRows, kHeadDim}),
              "partial_o must have shape [32, 4096, 512]");
  TORCH_CHECK(partial_m.sizes() == torch::IntArrayRef({kNumSplits, kQRows}),
              "partial_m must have shape [32, 4096]");
  TORCH_CHECK(partial_l.sizes() == torch::IntArrayRef({kNumSplits, kQRows}),
              "partial_l must have shape [32, 4096]");

  dim3 grid(kQBlocks * kGroup, kNumSplits);
  fused_attention_split_partial_pipelined_kernel<<<
      grid, kPipelinedThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      q_packed.data_ptr<uint8_t>(), q_scales.data_ptr<uint8_t>(),
      k_frag_pre.data_ptr<int32_t>(), k_scale_pre.data_ptr<int32_t>(),
      v_frag_pre.data_ptr<int32_t>(), v_scale_pre.data_ptr<int32_t>(),
      partial_o.data_ptr<float>(), partial_m.data_ptr<float>(),
      partial_l.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_attention_split_reduce_debug(torch::Tensor partial_o,
                                        torch::Tensor partial_m,
                                        torch::Tensor partial_l,
                                        torch::Tensor weights,
                                        torch::Tensor out_rows) {
  check_tensor(partial_o, "partial_o", torch::kFloat32);
  check_tensor(partial_m, "partial_m", torch::kFloat32);
  check_tensor(partial_l, "partial_l", torch::kFloat32);
  check_tensor(weights, "weights", torch::kFloat32);
  check_tensor(out_rows, "out_rows", torch::kFloat32);
  TORCH_CHECK(partial_o.sizes() == torch::IntArrayRef({kNumSplits, kQRows, kHeadDim}),
              "partial_o must have shape [32, 4096, 512]");
  TORCH_CHECK(partial_m.sizes() == torch::IntArrayRef({kNumSplits, kQRows}),
              "partial_m must have shape [32, 4096]");
  TORCH_CHECK(partial_l.sizes() == torch::IntArrayRef({kNumSplits, kQRows}),
              "partial_l must have shape [32, 4096]");
  TORCH_CHECK(weights.sizes() == torch::IntArrayRef({kNumSplits, kQRows}),
              "weights must have shape [32, 4096]");
  TORCH_CHECK(out_rows.sizes() == torch::IntArrayRef({kQRows, kHeadDim}),
              "out_rows must have shape [4096, 512]");

  fused_attention_split_weights_kernel<<<kQRows, 1, 0,
                                         at::cuda::getCurrentCUDAStream()>>>(
      partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
      weights.data_ptr<float>());
  const int64_t total = static_cast<int64_t>(kQRows) * kHeadDim;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  fused_attention_split_reduce_kernel<<<blocks, threads, 0,
                                        at::cuda::getCurrentCUDAStream()>>>(
      partial_o.data_ptr<float>(), weights.data_ptr<float>(),
      out_rows.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pv_full_mma_debug(torch::Tensor p_packed, torch::Tensor p_scales,
                       torch::Tensor v_pv, torch::Tensor v_pv_scales,
                       torch::Tensor out_block) {
  check_tensor(p_packed, "p_packed", torch::kUInt8);
  check_tensor(p_scales, "p_scales", torch::kUInt8);
  check_tensor(v_pv, "v_pv", torch::kUInt8);
  check_tensor(v_pv_scales, "v_pv_scales", torch::kUInt8);
  check_tensor(out_block, "out_block", torch::kFloat32);
  TORCH_CHECK(p_packed.sizes() == torch::IntArrayRef({kDebugTileM, kProbPackedCols}),
              "p_packed must have shape [16, 16384]");
  TORCH_CHECK(p_scales.sizes() == torch::IntArrayRef({kDebugTileM, kProbScaleCols}),
              "p_scales must have shape [16, 2048]");
  TORCH_CHECK(v_pv.sizes() == torch::IntArrayRef({kHeadDim, kProbPackedCols}),
              "v_pv must have shape [512, 16384]");
  TORCH_CHECK(v_pv_scales.sizes() == torch::IntArrayRef({kHeadDim, kProbScaleCols}),
              "v_pv_scales must have shape [512, 2048]");
  TORCH_CHECK(out_block.sizes() == torch::IntArrayRef({kDebugTileM, kHeadDim}),
              "out_block must have shape [16, 512]");

  pv_full_mma_debug_kernel<<<kHeadDim / kDebugTileN, 32, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
      p_packed.data_ptr<uint8_t>(), p_scales.data_ptr<uint8_t>(),
      v_pv.data_ptr<uint8_t>(), v_pv_scales.data_ptr<uint8_t>(),
      out_block.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("split_kv_len") = kSplitKvLen;
  m.attr("num_splits") = kNumSplits;
  m.attr("fused_kv_tile") = kFusedKvTile;
  m.def("attention_b1_d512_g8_q512_kv32768", &attention_b1_d512_g8_q512_kv32768,
        "Fixed-shape SM120 NVFP4 reference attention launcher");
  m.def("quantize_q_rowmajor", &quantize_q_rowmajor,
        "Fixed-shape BF16 Q to row-major NVFP4 quantization");
  m.def("qk_tile_mma_debug", &qk_tile_mma_debug,
        "Fixed-shape SM120 block-scaled FP4 QK 16x16 debug tile");
  m.def("qk_full_mma_debug", &qk_full_mma_debug,
        "Fixed-shape SM120 block-scaled FP4 QK debug scores for one 16-query block");
  m.def("qk_all_mma_debug", &qk_all_mma_debug,
        "Fixed-shape SM120 block-scaled FP4 QK debug scores for all query/head rows");
  m.def("softmax_quant_p_debug", &softmax_quant_p_debug,
        "Fixed-shape softmax plus NVFP4 probability quantization debug path");
  m.def("softmax_quant_p_all_debug", &softmax_quant_p_all_debug,
        "Fixed-shape softmax plus NVFP4 probability quantization for all query/head rows");
  m.def("pv_tile_mma_debug", &pv_tile_mma_debug,
        "Fixed-shape SM120 block-scaled FP4 PV 16x16 debug tile");
  m.def("pv_full_mma_debug", &pv_full_mma_debug,
        "Fixed-shape SM120 block-scaled FP4 PV debug block for all output columns");
  m.def("pv_all_mma_debug", &pv_all_mma_debug,
        "Fixed-shape SM120 block-scaled FP4 PV debug output for all query/head rows");
  m.def("prepack_k_fragments_debug", &prepack_k_fragments_debug,
        "Prepack row-major K into SM120 FP4 MMA fragment layout");
  m.def("prepack_v_fragments_debug", &prepack_v_fragments_debug,
        "Prepack PV-oriented V into SM120 FP4 MMA fragment layout");
  m.def("fused_attention_all_debug", &fused_attention_all_debug,
        "Fixed-shape fused SM120 block-scaled FP4 attention reference kernel");
  m.def("fused_attention_split_partial_debug", &fused_attention_split_partial_debug,
        "Fixed-shape split-KV fused SM120 block-scaled FP4 attention partial kernel");
  m.def("fused_attention_split_partial_pipelined_debug",
        &fused_attention_split_partial_pipelined_debug,
        "Fixed-shape split-KV pipelined producer/consumer FP4 attention partial kernel");
  m.def("fused_attention_split_reduce_debug", &fused_attention_split_reduce_debug,
        "Fixed-shape split-KV fused SM120 block-scaled FP4 attention reduction kernels");
}
