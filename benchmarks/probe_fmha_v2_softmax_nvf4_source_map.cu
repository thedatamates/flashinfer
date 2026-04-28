/*
 * Map Softmax storage columns to decoded Blackwell NVFP4 A-fragment K columns.
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <fmha/fragment.h>
#include <fmha/kernel_traits.h>
#include <fmha/softmax.h>
#include <fmha/traits.h>
#include <fmha/utils.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                        \
    cudaError_t status = (expr);                                              \
    if (status != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(status));                               \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

using Traits = fmha::Blackwell_mma_nvf4_fp32_traits;
using KernelTraits = fmha::Kernel_traits_v2_bf16_q_nvf4_paged_kv_cache<
    Traits, 64, 512, 0, 64, 4, 1, 1, 0x5022u | 0x200u | 0x4000u>;
using CtaTile = KernelTraits::Cta_tile_p;
using MmaTile = Traits::Mma_tile<CtaTile>;
using Softmax = fmha::Softmax<Traits, CtaTile, KernelTraits>;

struct Params {
  uint32_t scale_bmm1;
  uint32_t* scale_bmm1_d;
};

__global__ void probe_kernel(float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  extern __shared__ char smem[];
  int const active_col = blockIdx.x;
  int const lane = threadIdx.x & 31;
  Params params{0x3f800000u, nullptr};
  Softmax softmax(params, smem, 0, threadIdx.x);

#pragma unroll
  for (int row = 0; row < Softmax::ROWS_PER_THREAD; ++row) {
#pragma unroll
    for (int idx = 0; idx < MmaTile::MMAS_N * 4; ++idx) {
      int const ni = idx / 4;
      int const jj = idx & 3;
      int const logical_row = (lane >> 2) + row * 8;
      int const storage_col =
          ni * MmaTile::N_PER_MMA_PER_CTA + 2 * (lane & 3) + (jj & 1) +
          (jj & 2) * 4;
      softmax.elt_[row][idx] =
          (logical_row == 0 && storage_col == active_col) ? 1.f : 0.f;
    }
  }

  fmha::Fragment_a<Traits, fmha::Row> frag_p[KernelTraits::TOTAL_BMM2_MMAS_K]
                                                [MmaTile::MMAS_M];
  softmax.pack(frag_p);

  float* tile = out + active_col * 64;
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    uint32_t const packed = frag_p[0][0].reg(reg);
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      int const logical_m = (lane >> 2) + 8 * (reg & 1);
      int const logical_k = 16 * (lane & 3) + 8 * (reg >> 1) + jj;
      if (logical_m == 0) {
        uint32_t const code = (packed >> (4 * jj)) & 0xfu;
        tile[logical_k] = fmha::e2m1_to_float(code);
      }
    }
  }
#endif
}

int main() {
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, 64 * 64 * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, 64 * 64 * sizeof(float)));
  probe_kernel<<<64, CtaTile::THREADS_PER_CTA, 4096>>>(out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float host[64 * 64];
  CUDA_CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));
  for (int src = 0; src < 64; ++src) {
    std::printf("src %02d ->", src);
    for (int k = 0; k < 64; ++k) {
      if (host[src * 64 + k] != 0.f) {
        std::printf(" %02d:%.1f", k, host[src * 64 + k]);
      }
    }
    std::printf("\n");
  }
  return 0;
}
