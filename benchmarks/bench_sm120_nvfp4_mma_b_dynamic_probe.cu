/*
 * Probe dynamic B fragments for SM120 block-scaled FP4 MMA.
 *
 * The attention PV path uses dynamic per-lane A and B fragments plus per-lane
 * UE4M3 scale registers. Existing probes covered dynamic A and constant B; this
 * covers the B-side packing and scale-lane ownership used by the native PV path.
 */

#include <cuda_runtime.h>

#include <math_constants.h>

#include <cstdio>
#include <cstdlib>

#include <flashinfer/mma.cuh>

#define CUDA_CHECK(expr)                                                   \
  do {                                                                    \
    cudaError_t status = (expr);                                          \
    if (status != cudaSuccess) {                                          \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(status));                           \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

__device__ __forceinline__ float finite_or_zero(float x) {
  return (x == x && fabsf(x) != CUDART_INF_F) ? x : 0.f;
}

__device__ __forceinline__ float clamp_e2m1_finite(float x) {
  x = finite_or_zero(x);
  return fminf(fmaxf(x, -6.f), 6.f);
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

__device__ __forceinline__ float synthetic_v_value(uint32_t k, uint32_t n) {
  int raw = int((k * 19 + n * 7) % 23) - 11;
  return 0.125f * float(raw);
}

__device__ __forceinline__ uint8_t quantized_v_scale_byte(uint32_t n, uint32_t k_group) {
  float max_abs = 0.f;
#pragma unroll
  for (uint32_t i = 0; i < 16; ++i) {
    max_abs = fmaxf(max_abs, fabsf(finite_or_zero(synthetic_v_value(k_group * 16 + i, n))));
  }
  return fp32_to_ue4m3_byte(max_abs > 0.f ? max_abs / 6.f : 1.f);
}

__device__ __forceinline__ uint32_t make_v_scale_reg(uint32_t atom_n_offset) {
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t k_group = (lane >> 2) & 0x3u;
  uint8_t scale_bytes[4];
#pragma unroll
  for (uint32_t scale_idx = 0; scale_idx < 4; ++scale_idx) {
    const uint32_t n = atom_n_offset + 2 * scale_idx + ((lane >> 4) & 0x1u);
    scale_bytes[scale_idx] = quantized_v_scale_byte(n, k_group);
  }
  return flashinfer::mma::pack_e4m3_scale_reg(scale_bytes[0], scale_bytes[1],
                                              scale_bytes[2], scale_bytes[3]);
}

__device__ __forceinline__ uint8_t get_v_scale_byte_from_reg(uint32_t scale_reg, uint32_t n,
                                                             uint32_t k_group) {
  const uint32_t n_local = n & 0x7u;
  const uint32_t owner_lane = ((n_local & 0x1u) << 4) | (k_group << 2);
  const uint32_t scale_idx = n_local >> 1;
  const uint32_t owner_scale_reg = __shfl_sync(0xffffffff, scale_reg, owner_lane);
  return static_cast<uint8_t>((owner_scale_reg >> (8 * scale_idx)) & 0xffu);
}

__device__ __forceinline__ void make_v_frag_fp4(uint32_t atom_n_offset, uint32_t scale_reg,
                                                uint32_t* b_frag) {
  const uint32_t lane = threadIdx.x & 31;
#pragma unroll
  for (uint32_t reg = 0; reg < 2; ++reg) {
    float vals[8];
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      const uint32_t value_idx = reg * 8 + i;
      const uint32_t n = (lane & 0x3u) + 4 * (value_idx >> 3);
      const uint32_t k = ((lane >> 2) & 0x7u) + 8 * (value_idx & 0x7u);
      const float scale = ue4m3_byte_to_fp32(get_v_scale_byte_from_reg(scale_reg, n, k / 16));
      vals[i] = clamp_e2m1_finite(scale > 0.f ? synthetic_v_value(k, atom_n_offset + n) / scale
                                               : 0.f);
    }
    b_frag[reg] = flashinfer::mma::float8_to_e2m1x8(vals[0], vals[1], vals[2], vals[3],
                                                    vals[4], vals[5], vals[6], vals[7]);
  }
}

__global__ void dynamic_b_probe_kernel(float* out, int iters) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  uint32_t a[4] = {0x22222222u, 0x22222222u, 0x22222222u, 0x22222222u};
  uint32_t b[4];
  uint32_t scale_b0 = make_v_scale_reg(0);
  uint32_t scale_b1 = make_v_scale_reg(8);
  make_v_frag_fp4(0, scale_b0, b);
  make_v_frag_fp4(8, scale_b1, b + 2);
  constexpr uint32_t scale_a = 0x38383838u;
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0.f;
  }
  for (int i = 0; i < iters; ++i) {
    flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(acc, a, b, scale_a, scale_b0,
                                                        scale_b1);
  }
  const int linear_tid = threadIdx.x + blockDim.x * threadIdx.y;
  const int threads_per_block = blockDim.x * blockDim.y;
  const int base = (blockIdx.x * threads_per_block + linear_tid) * 8;
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

int main(int argc, char** argv) {
  int blocks = argc > 1 ? std::atoi(argv[1]) : 1024;
  int iters = argc > 2 ? std::atoi(argv[2]) : 1024;
  int warps = argc > 3 ? std::atoi(argv[3]) : 2;
  size_t out_elems = static_cast<size_t>(blocks) * 32 * warps * 8;
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, out_elems * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, out_elems * sizeof(float)));

  dim3 block(32, warps, 1);
  dynamic_b_probe_kernel<<<blocks, block>>>(out, 4);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  dynamic_b_probe_kernel<<<blocks, block>>>(out, iters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  float sample[8];
  CUDA_CHECK(cudaMemcpy(sample, out, sizeof(sample), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));

  double flops = static_cast<double>(blocks) * warps * iters * 2.0 * 16.0 * 16.0 * 64.0;
  double tflops = flops / (static_cast<double>(ms) * 1.0e9);
  std::printf("blocks=%d warps=%d iters=%d ms=%.6f atom_tflops=%.2f sample=[", blocks,
              warps, iters, ms, tflops);
  for (int i = 0; i < 8; ++i) {
    std::printf("%s%g", i ? "," : "", sample[i]);
  }
  std::printf("]\n");
  return 0;
}
