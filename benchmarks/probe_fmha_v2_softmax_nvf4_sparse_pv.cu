/*
 * Isolate Softmax<Blackwell NVFP4>::pack through the native PV MMA.
 *
 * For each source K column, the probe writes a one-hot P row through
 * Softmax::pack, builds a V fragment with only the same K row nonzero, runs
 * one SM120 FP4 MMA, and decodes the output tile. Correct P/V K alignment gives
 * row0 = 1 across all 16 output columns and every other row = 0.
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <cute/atom/mma_traits_sm120.hpp>

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

__device__ __forceinline__ uint32_t onehot_v_reg(int active_k, int reg) {
  int const lane = threadIdx.x & 31;
  uint32_t packed = 0u;
#pragma unroll
  for (int jj = 0; jj < 8; ++jj) {
    int const logical_k = 16 * (lane & 3) + 8 * (reg & 1) + jj;
    // All N columns are 1 for the selected K row, otherwise zero.
    uint32_t const code = logical_k == active_k ? 0x2u : 0x0u;
    packed |= code << (4 * jj);
  }
  return packed;
}

__global__ void probe_kernel(float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  extern __shared__ char smem[];
  int const active_k = static_cast<int>(blockIdx.x);
  int const lane = threadIdx.x & 31;
  int const warp_m = (threadIdx.x >> 5) % CtaTile::WARPS_M;

  Params params{0x3f800000u, nullptr};
  Softmax softmax(params, smem, 0, threadIdx.x);

#pragma unroll
  for (int row = 0; row < Softmax::ROWS_PER_THREAD; ++row) {
#pragma unroll
    for (int idx = 0; idx < MmaTile::MMAS_N * 4; ++idx) {
      int const ni = idx / 4;
      int const jj = idx & 3;
      int const logical_row =
          warp_m * MmaTile::M_PER_MMA + (lane >> 2) + row * 8;
      int const storage_col =
          ni * MmaTile::N_PER_MMA_PER_CTA + 2 * (lane & 3) + (jj & 1) +
          (jj & 2) * 4;
      softmax.elt_[row][idx] =
          (logical_row == 0 && storage_col == active_k) ? 1.f : 0.f;
    }
  }

  fmha::Fragment_a<Traits, fmha::Row> frag_p[KernelTraits::TOTAL_BMM2_MMAS_K]
                                                [MmaTile::MMAS_M];
  softmax.pack(frag_p);

  fmha::Fragment_b<Traits, fmha::Col> frag_v;
  frag_v.reg(0) = onehot_v_reg(active_k, 0);
  frag_v.reg(1) = onehot_v_reg(active_k, 1);
  frag_v.reg(2) = onehot_v_reg(active_k, 2);
  frag_v.reg(3) = onehot_v_reg(active_k, 3);
  frag_v.scale_reg[0] = fmha::make_ue4m3_scale_reg(1.f, 1.f, 1.f, 1.f);
  frag_v.scale_reg[1] = fmha::make_ue4m3_scale_reg(1.f, 1.f, 1.f, 1.f);

  fmha::Fragment_accumulator<Traits> acc;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc.elt(i) = 0.f;
  }
  acc.mma(frag_p[0][0], frag_v);

  float* tile = out + active_k * 16 * 16;
#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    int const logical_m =
        warp_m * MmaTile::M_PER_MMA + (lane >> 2) + 8 * ((elem % 4) / 2);
    int const logical_n = 2 * (lane & 3) + (elem & 1) + 8 * (elem / 4);
    tile[logical_m * 16 + logical_n] = acc.elt(elem) / (6.f * 448.f);
  }
#else
  if (threadIdx.x == 0) {
    out[0] = -1.f;
  }
#endif
}

int main() {
  constexpr int kPatterns = 64;
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, kPatterns * 16 * 16 * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, kPatterns * 16 * 16 * sizeof(float)));
  // This probe decodes only one logical m16n16 MMA tile. Launch one warp so
  // higher M-warps do not write outside the diagnostic tile.
  probe_kernel<<<kPatterns, CtaTile::THREADS_PER_WARP, 4096>>>(out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float host[kPatterns * 16 * 16];
  CUDA_CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));

  int total_bad = 0;
  for (int active_k = 0; active_k < kPatterns; ++active_k) {
    float row0_min = host[active_k * 16 * 16];
    float row0_max = row0_min;
    float other_max_abs = 0.f;
    for (int m = 0; m < 16; ++m) {
      for (int n = 0; n < 16; ++n) {
        float const v = host[active_k * 16 * 16 + m * 16 + n];
        if (m == 0) {
          row0_min = v < row0_min ? v : row0_min;
          row0_max = v > row0_max ? v : row0_max;
        } else {
          float const av = v < 0.f ? -v : v;
          other_max_abs = av > other_max_abs ? av : other_max_abs;
        }
      }
    }
    bool const bad =
        row0_min < 0.999f || row0_max > 1.001f || other_max_abs > 0.001f;
    total_bad += bad ? 1 : 0;
    if (bad || active_k < 16 || active_k % 8 == 0) {
      std::printf("k %02d row0 [%.6f, %.6f] other_abs %.6f%s\n",
                  active_k, row0_min, row0_max, other_max_abs,
                  bad ? " BAD" : "");
      if (active_k == 2 || active_k == 7 || active_k == 16) {
        std::printf("  row0:");
        for (int n = 0; n < 16; ++n) {
          std::printf(" %.1f", host[active_k * 16 * 16 + n]);
        }
        std::printf("\n");
      }
    }
  }
  std::printf("total_bad %d\n", total_bad);
  return total_bad == 0 ? 0 : 1;
}
