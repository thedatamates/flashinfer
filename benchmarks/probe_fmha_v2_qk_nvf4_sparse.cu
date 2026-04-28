/*
 * Probe the FMHAv2 Blackwell NVFP4 QK path through the production Q/K
 * shared-memory tiles.
 *
 * Each block creates one-hot Q(row0, dim) and one-hot K(token, dim), stores
 * both through Smem_tile_q/k, loads fragments, runs native SM120 FP4 MMA, and
 * decodes the score tile. Correct Q/K data and scale layout gives exactly one
 * nonzero score at (query row 0, target token).
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <cute/atom/mma_traits_sm120.hpp>

#include <fmha/fragment.h>
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
using GmemK = KernelTraits::Gmem_tile_k;
using SmemQ = KernelTraits::Smem_tile_q;
using SmemK = KernelTraits::Smem_tile_k;

template <typename GmemTile>
__device__ __forceinline__ void fill_onehot_fetch(uint4 (&fetch)[GmemTile::LDGS],
                                                  int active_row, int active_col,
                                                  float value) {
#pragma unroll
  for (int ii = 0; ii < GmemTile::LDGS; ++ii) {
    fetch[ii] = make_uint4(0u, 0u, 0u, 0u);
  }

  int const tidx = threadIdx.x;
  int const row_base = tidx / GmemTile::THREADS_PER_ROW;
  int const col_base =
      (tidx % GmemTile::THREADS_PER_ROW) * GmemTile::ELEMENTS_PER_LDG;
#pragma unroll
  for (int ii = 0; ii < GmemTile::LDGS; ++ii) {
    int const logical_row = row_base + ii * GmemTile::ROWS_PER_LDG;
    int const local_col = active_col - col_base;
    if (logical_row == active_row && local_col >= 0 &&
        local_col < GmemTile::ELEMENTS_PER_LDG) {
      uint32_t* regs = reinterpret_cast<uint32_t*>(&fetch[ii]);
      int const reg = local_col / 8;
      int const idx = local_col & 7;
      float vals[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      vals[idx] = value;
      regs[reg] = fmha::float8_to_e2m1x8(vals[0], vals[1], vals[2], vals[3],
                                          vals[4], vals[5], vals[6], vals[7]);
    }
  }
}

__global__ void probe_kernel(float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  extern __shared__ char smem[];
  int const active_token = static_cast<int>(blockIdx.x);
  int const active_dim = static_cast<int>(blockIdx.y);
  int const active_row = static_cast<int>(blockIdx.z);
  int const tidx = threadIdx.x;

  SmemQ smem_q(smem, tidx);
  SmemK smem_k(smem + SmemQ::BYTES_PER_TILE, tidx);

  uint4 fetch_q[GmemQ::LDGS];
  uint4 fetch_k[GmemK::LDGS];
  fill_onehot_fetch<GmemQ>(fetch_q, active_row, active_dim, 1.f);
  fill_onehot_fetch<GmemK>(fetch_k, active_token, active_dim, 1.f);
  smem_q.store(fetch_q);
  smem_k.store(fetch_k);

  uint32_t const scale_word = fmha::make_ue4m3_scale_reg(1.f, 1.f, 1.f, 1.f);
  for (int row = tidx; row < CtaTile::M; row += CtaTile::THREADS_PER_CTA) {
    smem_q.store_q_scale(row, scale_word);
  }
  for (int row = tidx; row < CtaTile::N; row += CtaTile::THREADS_PER_CTA) {
    smem_k.store_k_scale_word(row, scale_word);
  }
  __syncthreads();

  int const lane = threadIdx.x & 31;
  fmha::Fragment_a<Traits, fmha::Row> frag_q[MmaTile::MMAS_M];
  fmha::Fragment_b<Traits, fmha::Col> frag_k[MmaTile::MMAS_N];
#if defined(USE_BASE_LOAD)
  smem_q.SmemQ::Base::load(frag_q, 0);
  smem_k.SmemK::Base::load(frag_k, 0);
#elif defined(USE_DIRECT_FRAGMENT)
  int const direct_warp_m = (threadIdx.x >> 5) % CtaTile::WARPS_M;
#pragma unroll
  for (int mi = 0; mi < MmaTile::MMAS_M; ++mi) {
    frag_q[mi].scale_reg = scale_word;
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
      uint32_t packed = 0u;
#pragma unroll
      for (int nib = 0; nib < 8; ++nib) {
        int const logical_m = direct_warp_m * MmaTile::M_PER_MMA +
                              (lane >> 2) + 8 * (reg & 1);
        int const logical_k = 16 * (lane & 3) + 8 * (reg >> 1) + nib;
        if (logical_m == active_row && logical_k == active_dim) {
          packed |= 0x2u << (4 * nib);
        }
      }
      frag_q[mi].reg(reg) = packed;
    }
  }
#pragma unroll
  for (int ni = 0; ni < MmaTile::MMAS_N; ++ni) {
    frag_k[ni].scale_reg[0] = scale_word;
    frag_k[ni].scale_reg[1] = scale_word;
#pragma unroll
    for (int reg = 0; reg < 4; ++reg) {
      uint32_t packed = 0u;
#pragma unroll
      for (int nib = 0; nib < 8; ++nib) {
        int const logical_n = ni * MmaTile::N_PER_MMA_PER_CTA +
                              (lane >> 2) + 8 * (reg >> 1);
        int const logical_k = 16 * (lane & 3) + 8 * (reg & 1) + nib;
        if (logical_n == active_token && logical_k == active_dim) {
          packed |= 0x2u << (4 * nib);
        }
      }
      frag_k[ni].reg(reg) = packed;
    }
  }
#else
  smem_q.load(frag_q, 0);
  smem_k.load(frag_k, 0);
#endif

  int const warp_m = (threadIdx.x >> 5) % CtaTile::WARPS_M;
  float* tile = out + ((active_row * 64 + active_token) * 64 + active_dim) *
                          CtaTile::M * CtaTile::N;
#pragma unroll
  for (int ni = 0; ni < MmaTile::MMAS_N; ++ni) {
    fmha::Fragment_accumulator<Traits> acc;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc.elt(i) = 0.f;
    }
    acc.mma(frag_q[0], frag_k[ni]);
#pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
        int const logical_m = warp_m * MmaTile::M_PER_MMA +
                              (lane >> 2) + 8 * ((elem % 4) / 2);
        int const logical_n = ni * MmaTile::N_PER_MMA_PER_CTA +
                              2 * (lane & 3) + (elem & 1) +
                              8 * (elem / 4);
        tile[logical_m * CtaTile::N + logical_n] = acc.elt(elem);
    }
  }
#else
  if (threadIdx.x == 0) {
    out[0] = -1.f;
  }
#endif
}

int main() {
  constexpr int kTokens = 64;
  constexpr int kDims = 64;
  size_t const elems =
      static_cast<size_t>(CtaTile::M) * kTokens * kDims * CtaTile::M *
      CtaTile::N;
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, elems * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, elems * sizeof(float)));
  probe_kernel<<<dim3(kTokens, kDims, CtaTile::M), CtaTile::THREADS_PER_CTA,
                 SmemQ::BYTES_PER_TILE + SmemK::BYTES_PER_TILE>>>(out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float* host = new float[elems];
  CUDA_CHECK(cudaMemcpy(host, out, elems * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));

  int total_bad = 0;
  for (int row = 0; row < CtaTile::M; ++row) {
  for (int token = 0; token < kTokens; ++token) {
    for (int dim = 0; dim < kDims; ++dim) {
      float target =
          host[(((row * kTokens + token) * kDims + dim) * CtaTile::M + row) *
                   CtaTile::N +
               token];
      float other_max_abs = 0.f;
      int max_m = -1;
      int max_n = -1;
      for (int m = 0; m < CtaTile::M; ++m) {
        for (int n = 0; n < CtaTile::N; ++n) {
          if (m == row && n == token) {
            continue;
          }
          float const v =
              host[(((row * kTokens + token) * kDims + dim) * CtaTile::M + m) *
                       CtaTile::N +
                   n];
          float const av = v < 0.f ? -v : v;
          if (av > other_max_abs) {
            other_max_abs = av;
            max_m = m;
            max_n = n;
          }
        }
      }
      bool const bad = target < 0.999f || target > 1.001f || other_max_abs > 0.001f;
      total_bad += bad ? 1 : 0;
      if (bad && total_bad <= 64) {
        std::printf("row %02d token %02d dim %02d target %.6f other_abs %.6f at m %02d n %02d BAD\n",
                    row, token, dim, target, other_max_abs, max_m, max_n);
      }
    }
  }
  }
  std::printf("total_bad %d / %d\n", total_bad, CtaTile::M * kTokens * kDims);
  delete[] host;
  return total_bad == 0 ? 0 : 1;
}
