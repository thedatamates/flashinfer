/*
 * Native SM120 NVFP4 QK fragment probe.
 *
 * This isolates the proposed FA2 structural change for Shape A:
 * quantize the non-KV Q operand to E2M1 + UE4M3 scale registers and feed Q/K
 * directly to the Blackwell block-scaled FP4 MMA atom. It does not use the FA2
 * smem loader or scheduler; the point is to validate the fragment/register math
 * and quantify the raw cost before changing the fused attention kernel.
 */

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cutlass/float8.h>
#include <cutlass/float_subbyte.h>

#include <flashinfer/mma.cuh>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                        \
    cudaError_t status = (expr);                                              \
    if (status != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(status));                               \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

using Atom = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
    cutlass::float_ue4m3_t, 16>;

__host__ __device__ __forceinline__ float q_value(int row, int k) {
  int raw = ((row * 5) ^ (k * 3) ^ (k >> 2)) & 15;
  return 0.125f * float(raw - 7);
}

__host__ __device__ __forceinline__ float k_value(int k, int col) {
  int raw = ((k * 7) ^ (col * 5) ^ (k >> 3)) & 15;
  return 0.125f * float(raw - 7);
}

__host__ __device__ __forceinline__ float finite_or_zero_probe(float x) {
  return (x == x && fabsf(x) < 1.0e20f) ? x : 0.f;
}

__host__ __device__ __forceinline__ float clamp_e2m1_probe(float x) {
  x = finite_or_zero_probe(x);
  return fminf(fmaxf(x, -6.f), 6.f);
}

__host__ __device__ __forceinline__ float round_e2m1_probe(float x) {
  x = clamp_e2m1_probe(x);
  float sign = x < 0.f ? -1.f : 1.f;
  float ax = fabsf(x);
  float mag = ax < 0.25f     ? 0.f
              : ax < 0.75f   ? 0.5f
              : ax < 1.25f   ? 1.f
              : ax < 1.75f   ? 1.5f
              : ax < 2.5f    ? 2.f
              : ax < 3.5f    ? 3.f
              : ax < 5.f     ? 4.f
                              : 6.f;
  return sign * mag;
}

__device__ __forceinline__ uint8_t fp32_to_ue4m3_byte(float x) {
  __nv_fp8_e4m3 y = static_cast<__nv_fp8_e4m3>(x);
  return y.__x;
}

__device__ __forceinline__ float ue4m3_byte_to_fp32(uint8_t x) {
  __nv_fp8_e4m3 y;
  y.__x = x;
  return static_cast<float>(y);
}

__device__ __forceinline__ uint32_t q_scale_idx_for_k(uint32_t k) {
  return k >> 4;
}

__device__ __forceinline__ uint8_t choose_scale_byte16(const float (&values)[16]) {
  float max_abs = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    max_abs = fmaxf(max_abs, fabsf(finite_or_zero_probe(values[i])));
  }
  if (!(max_abs > 0.f)) {
    return fp32_to_ue4m3_byte(1.f);
  }
  return fp32_to_ue4m3_byte(max_abs / 6.f);
}

__device__ __forceinline__ uint8_t q_scale_byte(int row, int k_base, int k_group) {
  float vals[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    vals[i] = q_value(row, k_base + 16 * k_group + i);
  }
  return choose_scale_byte16(vals);
}

__device__ __forceinline__ uint8_t k_scale_byte(int k_base, int col, int k_group) {
  float vals[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    vals[i] = k_value(k_base + k_group * 16 + i, col);
  }
  return choose_scale_byte16(vals);
}

__device__ __forceinline__ uint32_t make_q_scale_reg(int k_base) {
  int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Atom>::SFALayout layout;
  uint8_t s[4];
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    int linear = int(layout(lane, slot));
    int row = linear % 16;
    int k_group = linear / 16;
    s[slot] = q_scale_byte(row, k_base, k_group);
  }
  return flashinfer::mma::pack_e4m3_scale_reg(s[0], s[1], s[2], s[3]);
}

__device__ __forceinline__ uint32_t make_k_scale_reg(int k_base, int atom_col_offset) {
  int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Atom>::SFBLayout layout;
  uint8_t s[4];
#pragma unroll
  for (int slot = 0; slot < 4; ++slot) {
    int linear = int(layout(lane, slot));
    int col = atom_col_offset + (linear % 8);
    int k_group = linear / 8;
    s[slot] = k_scale_byte(k_base, col, k_group);
  }
  return flashinfer::mma::pack_e4m3_scale_reg(s[0], s[1], s[2], s[3]);
}

__device__ __forceinline__ uint8_t get_q_scale_byte(uint32_t scale_reg, uint32_t row,
                                                    uint32_t k) {
  uint32_t owner_lane = 4 * (row & 0x7u) + (row >> 3);
  uint32_t slot = q_scale_idx_for_k(k);
  uint32_t owner_scale_reg = __shfl_sync(0xffffffff, scale_reg, owner_lane);
  return static_cast<uint8_t>((owner_scale_reg >> (8 * slot)) & 0xffu);
}

__device__ __forceinline__ uint8_t get_k_scale_byte(uint32_t scale_reg, uint32_t col,
                                                    uint32_t k) {
  uint32_t col_local = col & 0x7u;
  uint32_t owner_lane = 4u * col_local;
  uint32_t slot = k >> 4;
  uint32_t owner_scale_reg = __shfl_sync(0xffffffff, scale_reg, owner_lane);
  return static_cast<uint8_t>((owner_scale_reg >> (8 * slot)) & 0xffu);
}

__device__ __forceinline__ void make_q_frag_fp4(int k_base, uint32_t scale_reg,
                                                uint32_t (&frag)[4]) {
  int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Atom>::ALayout layout;
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    float vals[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int linear = int(layout(lane, 8 * reg + i));
      uint32_t row = linear % 16;
      uint32_t k = linear / 16;
      float scale = ue4m3_byte_to_fp32(get_q_scale_byte(scale_reg, row, k));
      vals[i] = clamp_e2m1_probe(scale > 0.f ? q_value(row, k_base + k) / scale : 0.f);
    }
    frag[reg] = flashinfer::mma::float8_to_e2m1x8(vals[0], vals[1], vals[2], vals[3],
                                                  vals[4], vals[5], vals[6], vals[7]);
  }
}

__device__ __forceinline__ void make_k_frag_fp4(int k_base, int atom_col_offset,
                                                uint32_t scale_reg, uint32_t* frag) {
  int lane = threadIdx.x & 31;
  typename cute::MMA_Traits<Atom>::BLayout layout;
#pragma unroll
  for (int reg = 0; reg < 2; ++reg) {
    float vals[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int linear = int(layout(lane, 8 * reg + i));
      uint32_t col = linear % 8;
      uint32_t k = linear / 8;
      float scale = ue4m3_byte_to_fp32(get_k_scale_byte(scale_reg, atom_col_offset + col, k));
      vals[i] = clamp_e2m1_probe(scale > 0.f ? k_value(k_base + k, atom_col_offset + col) / scale
                                             : 0.f);
    }
    frag[reg] = flashinfer::mma::float8_to_e2m1x8(vals[0], vals[1], vals[2], vals[3],
                                                  vals[4], vals[5], vals[6], vals[7]);
  }
}

__global__ void qk_native_kernel(float* out, int iters, int prepacked_k) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0.f;
  }

  for (int iter = 0; iter < iters; ++iter) {
#pragma unroll
    for (int k_group64 = 0; k_group64 < 4; ++k_group64) {
      uint32_t q_frag[4];
      uint32_t k_frag[4];
      int k_base = k_group64 * 64;
      uint32_t q_scale = make_q_scale_reg(k_base);
      make_q_frag_fp4(k_base, q_scale, q_frag);
      uint32_t k_scale0;
      uint32_t k_scale1;
      if (prepacked_k) {
        k_frag[0] = 0x22222222u;
        k_frag[1] = 0x22222222u;
        k_frag[2] = 0x22222222u;
        k_frag[3] = 0x22222222u;
        k_scale0 = 0x38383838u;
        k_scale1 = 0x38383838u;
      } else {
        k_scale0 = make_k_scale_reg(k_base, 0);
        k_scale1 = make_k_scale_reg(k_base, 8);
        make_k_frag_fp4(k_base, 0, k_scale0, k_frag);
        make_k_frag_fp4(k_base, 8, k_scale1, k_frag + 2);
      }
      if (iter == 0 && k_group64 == 0) {
        flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32<flashinfer::mma::MMAMode::kInit>(
            acc, q_frag, k_frag, q_scale, k_scale0, k_scale1);
      } else {
        flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(acc, q_frag, k_frag, q_scale,
                                                            k_scale0, k_scale1);
      }
    }
  }

  int lane = threadIdx.x & 31;
  int base = blockIdx.x * 32 * 8 + lane * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[base + i] = acc[i];
  }
#else
  if (threadIdx.x == 0) {
    out[0] = -1.f;
  }
#endif
}

__global__ void qk_dequant_reference_kernel(float* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 16 * 16) {
    return;
  }
  int row = idx / 16;
  int col = idx % 16;
  float sum = 0.f;
  for (int k_base = 0; k_base < 256; k_base += 64) {
#pragma unroll
    for (int k_local = 0; k_local < 64; ++k_local) {
      int kg = k_local / 16;
      float q_scale = ue4m3_byte_to_fp32(q_scale_byte(row, k_base, kg));
      float k_scale = ue4m3_byte_to_fp32(k_scale_byte(k_base, col, kg));
      float q_deq = round_e2m1_probe(q_value(row, k_base + k_local) / q_scale) * q_scale;
      float k_deq = round_e2m1_probe(k_value(k_base + k_local, col) / k_scale) * k_scale;
      sum += q_deq * k_deq;
    }
  }
  out[idx] = sum;
}

int main(int argc, char** argv) {
  int repeat_iters = argc > 1 ? std::atoi(argv[1]) : 1;
  int bench_iters = argc > 2 ? std::atoi(argv[2]) : 4096;
  int bench_blocks = argc > 3 ? std::atoi(argv[3]) : 4096;
  int bench_prepacked_k = argc > 4 ? std::atoi(argv[4]) : 1;
  float* out = nullptr;
  float* dequant_ref = nullptr;
  CUDA_CHECK(cudaMalloc(&out, static_cast<size_t>(bench_blocks) * 32 * 8 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dequant_ref, 16 * 16 * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, static_cast<size_t>(bench_blocks) * 32 * 8 * sizeof(float)));

  qk_native_kernel<<<1, 32>>>(out, repeat_iters, 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float host[32 * 8];
  CUDA_CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
  qk_dequant_reference_kernel<<<1, 256>>>(dequant_ref);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float dequant_host[16 * 16];
  CUDA_CHECK(cudaMemcpy(dequant_host, dequant_ref, sizeof(dequant_host), cudaMemcpyDeviceToHost));

  typename cute::MMA_Traits<Atom>::CLayout c_layout;
  float tile[16 * 16] = {};
  for (int lane = 0; lane < 32; ++lane) {
    for (int value_idx = 0; value_idx < 4; ++value_idx) {
      int linear = int(c_layout(lane, value_idx));
      tile[(linear % 16) * 16 + (linear / 16)] = host[lane * 8 + value_idx];
      tile[(linear % 16) * 16 + 8 + (linear / 16)] = host[lane * 8 + 4 + value_idx];
    }
  }

  float max_abs = 0.f;
  float mean_abs = 0.f;
  float max_abs_quant = 0.f;
  float mean_abs_quant = 0.f;
  float ref_norm = 0.f;
  float dot = 0.f;
  float out_norm = 0.f;
  float qref_norm = 0.f;
  float qdot = 0.f;
  for (int m = 0; m < 16; ++m) {
    for (int n = 0; n < 16; ++n) {
      float ref = 0.f;
      for (int k = 0; k < 256; ++k) {
        ref += q_value(m, k) * k_value(k, n);
      }
      float got = tile[m * 16 + n] / float(repeat_iters);
      float err = std::fabs(got - ref);
      float qref = dequant_host[m * 16 + n];
      float qerr = std::fabs(got - qref);
      max_abs = std::max(max_abs, err);
      max_abs_quant = std::max(max_abs_quant, qerr);
      mean_abs += err;
      mean_abs_quant += qerr;
      ref_norm += ref * ref;
      qref_norm += qref * qref;
      out_norm += got * got;
      dot += ref * got;
      qdot += qref * got;
    }
  }
  mean_abs /= 256.f;
  mean_abs_quant /= 256.f;
  float cosine = dot / std::sqrt(std::max(ref_norm * out_norm, 1.0e-20f));
  float quant_cosine = qdot / std::sqrt(std::max(qref_norm * out_norm, 1.0e-20f));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  qk_native_kernel<<<bench_blocks, 32>>>(out, bench_iters, bench_prepacked_k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(dequant_ref));

  double flops = double(bench_blocks) * double(bench_iters) * 2.0 * 16.0 * 16.0 * 256.0;
  double tflops = flops / (double(ms) * 1.0e9);
  std::printf(
      "repeat_iters=%d bench_iters=%d bench_blocks=%d bench_prepacked_k=%d ms=%.6f qk_tflops=%.2f ref_max_abs=%.6g ref_mean_abs=%.6g ref_cosine=%.8f qref_max_abs=%.6g qref_mean_abs=%.6g qref_cosine=%.8f sample=[%.6g,%.6g,%.6g,%.6g]\n",
      repeat_iters, bench_iters, bench_blocks, bench_prepacked_k, ms, tflops, max_abs, mean_abs,
      cosine, max_abs_quant, mean_abs_quant, quant_cosine, tile[0], tile[1], tile[16], tile[17]);
  return 0;
}
