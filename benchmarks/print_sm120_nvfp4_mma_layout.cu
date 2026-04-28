#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cutlass/float_subbyte.h>
#include <cutlass/float8.h>

#include <cstdio>

int main() {
  using Atom = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
      cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
      cutlass::float_ue4m3_t, 16>;
  using Traits = cute::MMA_Traits<Atom>;
  auto a_layout = typename Traits::ALayout{};
  auto b_layout = typename Traits::BLayout{};
  auto sfa_layout = typename Traits::SFALayout{};
  auto sfb_layout = typename Traits::SFBLayout{};
  auto c_layout = typename Traits::CLayout{};

  std::printf("A layout, lane 0..31 value 0..31 -> linear m*64+k\n");
  for (int t = 0; t < 32; ++t) {
    std::printf("lane %02d:", t);
    for (int v = 0; v < 32; ++v) {
      std::printf(" %d", int(a_layout(t, v)));
    }
    std::printf("\n");
  }
  std::printf("B layout, lane 0..31 value 0..15 -> linear n*64+k\n");
  for (int t = 0; t < 32; ++t) {
    std::printf("lane %02d:", t);
    for (int v = 0; v < 16; ++v) {
      std::printf(" %d", int(b_layout(t, v)));
    }
    std::printf("\n");
  }
  std::printf("SFA layout, lane 0..31 value 0..3 -> linear m*4+kg\n");
  for (int t = 0; t < 32; ++t) {
    std::printf("lane %02d:", t);
    for (int v = 0; v < 4; ++v) {
      std::printf(" %d", int(sfa_layout(t, v)));
    }
    std::printf("\n");
  }
  std::printf("SFB layout, lane 0..31 value 0..3 -> linear n*4+kg\n");
  for (int t = 0; t < 32; ++t) {
    std::printf("lane %02d:", t);
    for (int v = 0; v < 4; ++v) {
      std::printf(" %d", int(sfb_layout(t, v)));
    }
    std::printf("\n");
  }
  std::printf("C layout, lane 0..31 value 0..3 -> linear m*8+n\n");
  for (int t = 0; t < 32; ++t) {
    std::printf("lane %02d:", t);
    for (int v = 0; v < 4; ++v) {
      std::printf(" %d", int(c_layout(t, v)));
    }
    std::printf("\n");
  }
}
