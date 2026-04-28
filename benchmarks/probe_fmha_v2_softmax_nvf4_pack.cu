/*
 * Isolate Softmax<Blackwell NVFP4>::pack for the P operand of PV.
 *
 * The probe creates zero logits, applies the normal padding mask for a chosen
 * KV length, packs P to NVFP4, and multiplies by V=ones using the native SM120
 * FP4 MMA. The expected raw output is valid_kv_len before the flash-attention
 * final normalization.
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

__global__ void probe_kernel(float* out) {
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

  fmha::Fragment_a<Traits, fmha::Row> frag_p[KernelTraits::TOTAL_BMM2_MMAS_K]
                                                [MmaTile::MMAS_M];
  softmax.pack(frag_p);

  fmha::Fragment_b<Traits, fmha::Col> frag_v;
  frag_v.reg(0) = 0x22222222u;
  frag_v.reg(1) = 0x22222222u;
  frag_v.reg(2) = 0x22222222u;
  frag_v.reg(3) = 0x22222222u;
  frag_v.scale_reg[0] = fmha::make_ue4m3_scale_reg(1.f, 1.f, 1.f, 1.f);
  frag_v.scale_reg[1] = fmha::make_ue4m3_scale_reg(1.f, 1.f, 1.f, 1.f);

  fmha::Fragment_accumulator<Traits> acc;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    acc.elt(e) = 0.f;
  }
  acc.mma(frag_p[0][0], frag_v);

  int const lane = threadIdx.x & 31;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    out[blockIdx.x * 32 * 8 + lane * 8 + e] = acc.elt(e) / (6.f * 448.f);
  }
#else
  if (threadIdx.x == 0) {
    out[0] = -1.f;
  }
#endif
}

int main() {
  constexpr int kBlocks = 64;
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, kBlocks * 32 * 8 * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, kBlocks * 32 * 8 * sizeof(float)));
  probe_kernel<<<kBlocks, CtaTile::THREADS_PER_CTA, 4096>>>(out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float host[kBlocks * 32 * 8];
  CUDA_CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));

  for (int valid = 1; valid <= 64; ++valid) {
    float mn = host[(valid - 1) * 32 * 8];
    float mx = mn;
    for (int i = 0; i < 32 * 8; ++i) {
      float v = host[(valid - 1) * 32 * 8 + i];
      mn = v < mn ? v : mn;
      mx = v > mx ? v : mx;
    }
    if (valid <= 16 || valid % 8 == 0) {
      std::printf("valid %02d range [%.1f, %.1f] ratio %.6f\n", valid, mn, mx,
                  mx / static_cast<float>(valid));
    }
  }
  return 0;
}
