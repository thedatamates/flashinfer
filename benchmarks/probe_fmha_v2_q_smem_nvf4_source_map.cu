/*
 * Map the Blackwell NVFP4 Q shared-memory store/load path.
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <fmha/kernel_traits.h>
#include <fmha/smem_tile.h>
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
using GmemQ = KernelTraits::Gmem_tile_q;
using SmemQ = KernelTraits::Smem_tile_q;

__global__ void probe_kernel(float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  extern __shared__ char smem[];
  int const active_row = blockIdx.x;
  int const active_col = blockIdx.y;
  int const tidx = threadIdx.x;
  SmemQ smem_q(smem, tidx);

  uint4 fetch[GmemQ::LDGS];
#pragma unroll
  for (int ii = 0; ii < GmemQ::LDGS; ++ii) {
    fetch[ii] = make_uint4(0u, 0u, 0u, 0u);
  }

  int const row_base = tidx / GmemQ::THREADS_PER_ROW;
  int const col_base = (tidx % GmemQ::THREADS_PER_ROW) * GmemQ::ELEMENTS_PER_LDG;
#pragma unroll
  for (int ii = 0; ii < GmemQ::LDGS; ++ii) {
    int const logical_row = row_base + ii * GmemQ::ROWS_PER_LDG;
    int const local_col = active_col - col_base;
    if (logical_row == active_row && local_col >= 0 &&
        local_col < GmemQ::ELEMENTS_PER_LDG) {
      uint32_t* regs = reinterpret_cast<uint32_t*>(&fetch[ii]);
      regs[local_col / 8] = 0x7u << (4 * (local_col & 7));
    }
  }

  smem_q.store(fetch);
  __syncthreads();

  fmha::Fragment_a<Traits, fmha::Row> frag[MmaTile::MMAS_M];
  smem_q.load(frag, 0);

  int const lane = threadIdx.x & 31;
  typename cute::MMA_Traits<typename fmha::Fragment_accumulator<Traits>::Mma_atom>::ALayout
      a_layout;
#pragma unroll
  for (int mi = 0; mi < MmaTile::MMAS_M; ++mi) {
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
      uint32_t const packed = frag[mi].reg(reg);
#pragma unroll
      for (int nib = 0; nib < 8; ++nib) {
        int const value_idx = reg * 8 + nib;
        int const a_linear = static_cast<int>(a_layout(lane, value_idx));
        int const logical_row =
            mi * MmaTile::M_PER_MMA_PER_CTA + a_linear / MmaTile::K_PER_MMA;
        int const logical_col = a_linear % MmaTile::K_PER_MMA;
        uint32_t const code = (packed >> (4 * nib)) & 0xfu;
        if (code != 0u) {
          out[(active_row * 64 + active_col) * CtaTile::M * 64 +
              logical_row * 64 + logical_col] = fmha::e2m1_to_float(code);
        }
      }
    }
  }
#endif
}

int main() {
  constexpr int rows = CtaTile::M;
  constexpr int cols = 64;
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, rows * cols * rows * cols * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, rows * cols * rows * cols * sizeof(float)));
  probe_kernel<<<dim3(rows, cols), CtaTile::THREADS_PER_CTA, SmemQ::BYTES_PER_TILE>>>(
      out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float* host = new float[rows * cols * rows * cols];
  CUDA_CHECK(cudaMemcpy(host, out, rows * cols * rows * cols * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));
  for (int src_row = 0; src_row < 16; ++src_row) {
    for (int src_col = 0; src_col < 16; ++src_col) {
      std::printf("src (%02d,%02d) ->", src_row, src_col);
      int const base = (src_row * cols + src_col) * rows * cols;
      for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 64; ++col) {
          float const v = host[base + row * cols + col];
          if (v != 0.f) {
            std::printf(" (%02d,%02d):%.1f", row, col, v);
          }
        }
      }
      std::printf("\n");
    }
  }
  delete[] host;
  return 0;
}
