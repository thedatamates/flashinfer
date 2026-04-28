/*
 * Print CUTE's SM120 FP4 CLayout as logical (m,n) coordinates.
 */

#include <cstdio>

#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cutlass/float8.h>
#include <cutlass/float_subbyte.h>

using Atom = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
    cutlass::float_ue4m3_t, 16>;

int main() {
  typename cute::MMA_Traits<Atom>::CLayout layout;
  for (int lane = 0; lane < 32; ++lane) {
    std::printf("lane %02d:", lane);
    for (int value_idx = 0; value_idx < 4; ++value_idx) {
      int const linear = static_cast<int>(layout(lane, value_idx));
      std::printf(" %02d:(m%02d,n%02d)", value_idx, linear % 16,
                  linear / 16);
    }
    std::printf("\n");
  }
  return 0;
}
