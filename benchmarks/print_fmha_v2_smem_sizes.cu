#include <algorithm>
#include <cstdio>

#include <fmha/kernel_traits.h>

template <typename KTraits>
void print_traits(const char* name) {
  std::printf("%s\n", name);
  std::printf("  CTA_P_TILE_M/N/K: %d %d %d\n", KTraits::CTA_P_TILE_M,
              KTraits::CTA_P_TILE_N, KTraits::CTA_P_TILE_K);
  std::printf("  CTA_O_TILE_M/N/K: %d %d %d\n", KTraits::CTA_O_TILE_M,
              KTraits::CTA_O_TILE_N, KTraits::CTA_O_TILE_K);
  std::printf("  buffers Q/K/V: %d %d %d\n", KTraits::BUFFERS_PER_TILE_SMEM_Q,
              KTraits::BUFFERS_PER_TILE_SMEM_K, KTraits::BUFFERS_PER_TILE_SMEM_V);
  std::printf("  smem Q:   %6d\n", KTraits::Smem_tile_q::BYTES_PER_TILE);
  std::printf("  smem K:   %6d\n", KTraits::Smem_tile_k::BYTES_PER_TILE);
  std::printf("  smem V:   %6d\n", KTraits::Smem_tile_v::BYTES_PER_TILE);
  std::printf("  smem O:   %6d\n", KTraits::Smem_tile_o::BYTES_PER_TILE);
  std::printf("  smem QK:  %6d\n", KTraits::BYTES_PER_SMEM_QK);
  std::printf("  smem QKV: %6d\n", KTraits::BYTES_PER_SMEM_QKV);
  std::printf("  smem QO:  %6d\n", KTraits::BYTES_PER_SMEM_QO);
  std::printf("  smem max: %6d\n", KTraits::BYTES_PER_SMEM);
}

int main() {
  constexpr uint32_t kBf16D512Flags =
      0x1u |   // ldgsts_q
      0x20u |  // non-interleaved heads
      0x1000u |  // granular tiled
      0x200u;    // no-loop

  constexpr uint32_t kNvfp4D512Flags =
      0x2u |   // ldgsts_k
      0x20u |  // non-interleaved heads
      0x1000u |  // granular tiled
      0x200u;    // no-loop

  using Bf16D512 = fmha::Kernel_traits_v2_paged_kv_cache<
      fmha::Ampere_hmma_bf16_traits,
      64,   // kv_loop_step
      512,  // head_dim
      512,  // head_dim_v
      64,   // q_loop_step
      4,    // warps_m
      1,    // warps_n
      1,    // ctas_per_head
      kBf16D512Flags,
      3,     // causal
      true,  // bmm2 bf16 epilogue
      fmha::bf16_t>;

  using Nvfp4D512 = fmha::Kernel_traits_v2_bf16_q_nvf4_paged_kv_cache<
      fmha::Blackwell_mma_nvf4_fp32_traits,
      64,   // kv_loop_step
      512,  // head_dim
      512,  // head_dim_v
      64,   // q_loop_step
      4,    // warps_m
      1,    // warps_n
      1,    // ctas_per_head
      kNvfp4D512Flags,
      3,     // causal
      true,  // bmm2 bf16 epilogue
      fmha::bf16_t>;

  print_traits<Bf16D512>("BF16 paged D512");
  print_traits<Nvfp4D512>("NVFP4 paged D512");
  return 0;
}
