/*
 * Probe SM120 FP4 MMA A-scale register interpretation.
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cutlass/float8.h>
#include <cutlass/float_subbyte.h>

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

using Atom = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
    cutlass::float_ue4m3_t, 16>;

template <int ACTIVE_KG, int SCALE_SLOT>
__device__ __forceinline__ uint32_t make_a_reg(int reg) {
  typename cute::MMA_Traits<Atom>::ALayout layout;
  int const lane = threadIdx.x & 31;
  uint32_t word = 0;
#pragma unroll
  for (int nib = 0; nib < 8; ++nib) {
    int const value_idx = reg * 8 + nib;
    int const linear = int(layout(lane, value_idx));
    int const logical_k = linear / 16;
    uint32_t const code = logical_k / 16 == ACTIVE_KG ? 2u : 0u;
    word |= code << (4 * nib);
  }
  return word;
}

template <int ACTIVE_KG, int SCALE_SLOT>
__global__ void probe(float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  uint32_t a[4] = {make_a_reg<ACTIVE_KG, SCALE_SLOT>(0),
                   make_a_reg<ACTIVE_KG, SCALE_SLOT>(1),
                   make_a_reg<ACTIVE_KG, SCALE_SLOT>(2),
                   make_a_reg<ACTIVE_KG, SCALE_SLOT>(3)};
  uint32_t b[4] = {0x22222222u, 0x22222222u, 0x22222222u, 0x22222222u};
  float scales[4] = {1.f, 1.f, 1.f, 1.f};
  scales[SCALE_SLOT] = 2.f;
  uint32_t const scale_a = fmha::make_ue4m3_scale_reg(scales[0], scales[1],
                                                      scales[2], scales[3]);
  uint32_t const scale_b = fmha::make_ue4m3_scale_reg(1.f, 1.f, 1.f, 1.f);
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  Atom::fma(acc[0], acc[1], acc[2], acc[3], a[0], a[1], a[2], a[3], b[0],
            b[1], acc[0], acc[1], acc[2], acc[3], scale_a, scale_b);
  Atom::fma(acc[4], acc[5], acc[6], acc[7], a[0], a[1], a[2], a[3], b[2],
            b[3], acc[4], acc[5], acc[6], acc[7], scale_a, scale_b);
  int const lane = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[blockIdx.x * 32 * 8 + lane * 8 + i] = acc[i];
  }
#endif
}

int main() {
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, 16 * 32 * 8 * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, 16 * 32 * 8 * sizeof(float)));
  probe<0, 0><<<1, 32>>>(out + 0 * 32 * 8);
  probe<0, 1><<<1, 32>>>(out + 1 * 32 * 8);
  probe<0, 2><<<1, 32>>>(out + 2 * 32 * 8);
  probe<0, 3><<<1, 32>>>(out + 3 * 32 * 8);
  probe<1, 0><<<1, 32>>>(out + 4 * 32 * 8);
  probe<1, 1><<<1, 32>>>(out + 5 * 32 * 8);
  probe<1, 2><<<1, 32>>>(out + 6 * 32 * 8);
  probe<1, 3><<<1, 32>>>(out + 7 * 32 * 8);
  probe<2, 0><<<1, 32>>>(out + 8 * 32 * 8);
  probe<2, 1><<<1, 32>>>(out + 9 * 32 * 8);
  probe<2, 2><<<1, 32>>>(out + 10 * 32 * 8);
  probe<2, 3><<<1, 32>>>(out + 11 * 32 * 8);
  probe<3, 0><<<1, 32>>>(out + 12 * 32 * 8);
  probe<3, 1><<<1, 32>>>(out + 13 * 32 * 8);
  probe<3, 2><<<1, 32>>>(out + 14 * 32 * 8);
  probe<3, 3><<<1, 32>>>(out + 15 * 32 * 8);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float host[16 * 32 * 8];
  CUDA_CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));
  for (int kg = 0; kg < 4; ++kg) {
    for (int slot = 0; slot < 4; ++slot) {
      int const case_idx = kg * 4 + slot;
      float mn = host[case_idx * 32 * 8], mx = mn;
    for (int i = 0; i < 32 * 8; ++i) {
        float v = host[case_idx * 32 * 8 + i];
      mn = v < mn ? v : mn;
      mx = v > mx ? v : mx;
    }
      std::printf("kg %d slot %d range [%.1f, %.1f] lane0:", kg, slot, mn,
                  mx);
    for (int i = 0; i < 8; ++i) {
        std::printf(" %.1f", host[case_idx * 32 * 8 + i]);
    }
    std::printf("\\n");
  }
  }
  return 0;
}
