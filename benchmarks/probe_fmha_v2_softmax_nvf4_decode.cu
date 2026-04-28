/*
 * Decode Softmax<Blackwell NVFP4>::pack back to a logical P tile.
 *
 * This checks the P-side FP4 data layout and SFA scale layout without going
 * through PV MMA or output-fragment layout.
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <fmha/fragment.h>
#include <fmha/kernel_traits.h>
#include <fmha/mask.h>
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

struct BlockInfo {
  int actual_seqlen;
  int actual_q_seqlen;
  int actual_kv_seqlen;
  int bidn;
};

__global__ void probe_kernel(float* out, float* raw) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  extern __shared__ char smem[];
  int const valid = static_cast<int>(blockIdx.x) + 1;
  Params params{0x3f800000u, nullptr};
  BlockInfo binfo{valid, 16, valid, 0};
  Softmax softmax(params, smem, 0, threadIdx.x);
  fmha::Mask_dispatcher<Traits, CtaTile, 2, false, true> mask(params, binfo, threadIdx.x);
  mask.load(0);
  mask.move_to_offset(0);

  fmha::Fragment_accumulator<Traits> acc_p[MmaTile::MMAS_M][MmaTile::MMAS_N];
#pragma unroll
  for (int mi = 0; mi < MmaTile::MMAS_M; ++mi) {
#pragma unroll
    for (int ni = 0; ni < MmaTile::MMAS_N; ++ni) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        acc_p[mi][ni].elt(e) = 0.f;
      }
    }
  }
  softmax.unpack(acc_p);
  softmax.apply_mask(mask);
  float max_vals[Softmax::ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < Softmax::ROWS_PER_THREAD; ++i) {
    max_vals[i] = 0.f;
  }
  softmax.template apply_exp_with_mask<true>(max_vals);
  if (valid == 49 && threadIdx.x == 0) {
    raw[0] = softmax.elt_[0][12];
    raw[1] = softmax.elt_[0][13];
    raw[2] = softmax.elt_[0][14];
    raw[3] = softmax.elt_[0][15];
  }
  float dbg_amax = 0.f;
#pragma unroll
  for (int jj = 0; jj < 4; ++jj) {
    dbg_amax = fmaxf(dbg_amax, fabsf(softmax.elt_[0][12 + jj]));
  }
  dbg_amax = fmaxf(dbg_amax, __shfl_xor_sync(uint32_t(-1), dbg_amax, 1));
  dbg_amax = fmaxf(dbg_amax, __shfl_xor_sync(uint32_t(-1), dbg_amax, 2));
  float const dbg_sf = dbg_amax > 0.f ? fminf(dbg_amax * 448.f, 448.f) : 1.f;
  float const dbg_inv_sf = dbg_amax > 0.f ? 1.f / dbg_sf : 0.f;
  float const dbg_val = __shfl_sync(uint32_t(-1), softmax.elt_[0][13], 0);
  if (valid == 49 && threadIdx.x == 4) {
    raw[0] = dbg_val;
    raw[1] = dbg_amax;
    raw[2] = dbg_inv_sf;
    raw[3] = dbg_val * 6.f * 448.f * dbg_inv_sf;
  }

  fmha::Fragment_a<Traits, fmha::Row> frag_p[KernelTraits::TOTAL_BMM2_MMAS_K]
                                                [MmaTile::MMAS_M];
  softmax.pack(frag_p);
  if (valid == 49 && threadIdx.x == 4) {
    raw[0] = raw[0];
    raw[1] = raw[1];
    raw[2] = static_cast<float>(frag_p[0][0].reg(0));
    raw[3] = static_cast<float>((frag_p[0][0].reg(0) >> 12) & 0xfu);
  }

  int const lane = threadIdx.x & 31;
  typename cute::MMA_Traits<typename fmha::Fragment_accumulator<Traits>::Mma_atom>::ALayout
      a_layout;
  float* tile = out + blockIdx.x * 16 * 64;
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    uint32_t const packed = frag_p[0][0].reg(reg);
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      int const value_idx = reg * 8 + jj;
      int const a_linear = static_cast<int>(a_layout(lane, value_idx));
      int const logical_m = a_linear % MmaTile::M_PER_MMA;
      int const logical_k = a_linear / MmaTile::M_PER_MMA;
      int const kg = logical_k >> 4;
      int const scale_lane =
          logical_m < 8 ? 4 * logical_m : 4 * (logical_m - 8) + 1;
      uint32_t const scale_reg =
          __shfl_sync(uint32_t(-1), frag_p[0][0].scale_reg, scale_lane);
      uint8_t const scale_byte =
          static_cast<uint8_t>((scale_reg >> (8 * kg)) & 0xffu);
      uint32_t const code = (packed >> (4 * jj)) & 0xfu;
      tile[logical_m * 64 + logical_k] =
          fmha::e2m1_to_float(code) * fmha::e4m3_byte_to_float(scale_byte) /
          (6.f * 448.f);
    }
  }
#endif
}

int main() {
  constexpr int kBlocks = 64;
  float* out = nullptr;
  float* raw = nullptr;
  CUDA_CHECK(cudaMalloc(&out, kBlocks * 16 * 64 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&raw, 4 * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, kBlocks * 16 * 64 * sizeof(float)));
  CUDA_CHECK(cudaMemset(raw, 0, 4 * sizeof(float)));
  probe_kernel<<<kBlocks, CtaTile::THREADS_PER_CTA, 4096>>>(out, raw);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float host[kBlocks * 16 * 64];
  float raw_host[4];
  CUDA_CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(raw_host, raw, sizeof(raw_host), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(raw));
  std::printf("valid 49 raw lane0 elt[12..15]: %.1f %.1f %.1f %.1f\n",
              raw_host[0], raw_host[1], raw_host[2], raw_host[3]);

  for (int valid = 1; valid <= 64; ++valid) {
    float mn = 1e30f;
    float mx = -1e30f;
    for (int row = 0; row < 16; ++row) {
      float sum = 0.f;
      for (int k = 0; k < 64; ++k) {
        sum += host[(valid - 1) * 16 * 64 + row * 64 + k];
      }
      mn = sum < mn ? sum : mn;
      mx = sum > mx ? sum : mx;
    }
    if (valid <= 16 || valid % 8 == 0) {
      std::printf("valid %02d decoded row_sum range [%.6f, %.6f] expected %d\n",
                  valid, mn, mx, valid);
    }
    if (valid == 49 || valid == 59) {
      std::printf("valid %02d row0 nonzero:", valid);
      for (int k = 0; k < 64; ++k) {
        float v = host[(valid - 1) * 16 * 64 + k];
        if (v != 0.f) {
          std::printf(" %d:%.1f", k, v);
        }
      }
      std::printf("\n");
    }
  }
  return 0;
}
