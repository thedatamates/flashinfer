/*
 * Probe dynamic per-lane FP4 fragments for the SM120 block-scaled MMA atom.
 *
 * This distinguishes native-MMA/converter issues from attention-kernel
 * fragment-layout issues. It does not use FlashAttention state; each lane
 * constructs different E2M1 A registers from finite FP32 values and feeds them
 * to the same FlashInfer native FP4 MMA wrapper used by the prefill PV path.
 */

#include <cuda_runtime.h>

#include <math_constants.h>

#include <cstdio>
#include <cstdlib>

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

__device__ __forceinline__ float probe_value(int lane, int reg, int idx,
                                             int pattern) {
  if (pattern == 0) {
    return 1.f;
  }
  if (pattern == 1) {
    return ((lane + reg + idx) & 1) ? -1.f : 1.f;
  }
  if (pattern == 2) {
    return (float(((lane * 17 + reg * 5 + idx * 3) % 13) - 6)) * 0.75f;
  }
  return (idx & 1) ? 6.f : -6.f;
}

__device__ __forceinline__ float finite_or_zero_probe(float x) {
  return (x == x && fabsf(x) != CUDART_INF_F) ? x : 0.f;
}

__device__ __forceinline__ float clamp_e2m1_probe(float x) {
  x = finite_or_zero_probe(x);
  return fminf(fmaxf(x, -6.f), 6.f);
}

__device__ __forceinline__ float simulated_s_value(const float (&s_frag)[4][8],
                                                   int row, int col) {
  const int mma_kv = col / 16;
  const int col_in_mma = col % 16;
  const int row_group = row / 8;
  const int owner_lane = 4 * (row % 8) + ((col_in_mma % 8) / 2);
  const int reg_id = (col_in_mma / 8) * 4 + row_group * 2 + (col_in_mma % 2);
  const float local = ((threadIdx.x & 31) == owner_lane) ? s_frag[mma_kv][reg_id] : 0.f;
  return __shfl_sync(0xffffffff, local, owner_lane);
}

__device__ __forceinline__ uint32_t make_simulated_s_frag_fp4(int pattern) {
  const int lane = threadIdx.x & 31;
  float s_frag[4][8];
#pragma unroll
  for (int mma_kv = 0; mma_kv < 4; ++mma_kv) {
#pragma unroll
    for (int reg = 0; reg < 8; ++reg) {
      s_frag[mma_kv][reg] = 0.03125f *
                             float(((lane + 3 * mma_kv + 5 * reg + pattern) % 17) - 8);
    }
  }

  float vals[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int value_idx = i;
    const int row = 2 * (lane & 0x3) + ((value_idx >> 2) & 0x1);
    const int col = ((lane >> 2) & 0x7) + 16 * (value_idx & 0x3) +
                    8 * ((value_idx >> 3) & 0x1);
    float max_abs = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      max_abs = fmaxf(max_abs, fabsf(finite_or_zero_probe(
                                      simulated_s_value(s_frag, row, (col / 16) * 16 + j))));
    }
    const float scale = max_abs > 0.f ? max_abs / 6.f : 1.f;
    vals[i] = clamp_e2m1_probe(simulated_s_value(s_frag, row, col) / scale);
  }
  return flashinfer::mma::float8_to_e2m1x8(vals[0], vals[1], vals[2], vals[3], vals[4],
                                           vals[5], vals[6], vals[7]);
}

__global__ void dynamic_mma_probe_kernel(float* out, int iters, int pattern) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  const int lane = threadIdx.x & 31;
  uint32_t a[4];
  uint32_t b[4] = {0x22222222u, 0x22222222u, 0x22222222u, 0x22222222u};
  float acc[8];

  if (pattern == 4) {
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
      a[reg] = make_simulated_s_frag_fp4(pattern + reg);
    }
  } else if (pattern == 5) {
    const bool active_1 =
        lane == 0 || lane == 4 || lane == 9 || lane == 13 || lane == 18 || lane == 22 ||
        lane == 27 || lane == 31;
    const bool active_0 = lane == 0 || lane == 4;
    const uint32_t word = active_1 ? (active_0 ? 0x00070001u : 0x00070007u) : 0x00000000u;
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
      a[reg] = word;
    }
  } else {
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
      float v[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        v[i] = probe_value(lane, reg, i, pattern);
      }
      a[reg] = flashinfer::mma::float8_to_e2m1x8(v[0], v[1], v[2], v[3], v[4], v[5],
                                                 v[6], v[7]);
    }
  }

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0.f;
  }

  constexpr uint32_t scale = 0x38383838u;
  for (int i = 0; i < iters; ++i) {
    flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(acc, a, b, scale, scale, scale);
  }

  const int linear_tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
  const int base = (blockIdx.x * threads_per_block + linear_tid) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[base + i] = acc[i];
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = -1.f;
  }
#endif
}

int main(int argc, char** argv) {
  int blocks = argc > 1 ? std::atoi(argv[1]) : 4096;
  int iters = argc > 2 ? std::atoi(argv[2]) : 1024;
  int pattern = argc > 3 ? std::atoi(argv[3]) : 2;
  int threads = argc > 4 ? std::atoi(argv[4]) : 32;
  bool use_2d = argc > 5 ? std::atoi(argv[5]) != 0 : false;
  size_t out_elems = static_cast<size_t>(blocks) * threads * 8;

  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, out_elems * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, out_elems * sizeof(float)));

  dim3 block = use_2d ? dim3(32, threads / 32, 1) : dim3(threads, 1, 1);
  dynamic_mma_probe_kernel<<<blocks, block>>>(out, 4, pattern);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  dynamic_mma_probe_kernel<<<blocks, block>>>(out, iters, pattern);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  float sample[8];
  CUDA_CHECK(cudaMemcpy(sample, out, sizeof(sample), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));

  double flops = static_cast<double>(blocks) * iters * 2.0 * 16.0 * 16.0 * 64.0;
  double tflops = flops / (static_cast<double>(ms) * 1.0e9);
  std::printf("blocks=%d threads=%d iters=%d pattern=%d ms=%.6f atom_tflops=%.2f sample=[",
              blocks, threads, iters, pattern, ms, tflops);
  for (int i = 0; i < 8; ++i) {
    std::printf("%s%g", i ? "," : "", sample[i]);
  }
  std::printf("]\n");
  return 0;
}
