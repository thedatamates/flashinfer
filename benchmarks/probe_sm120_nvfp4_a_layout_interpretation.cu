/*
 * Compare plausible interpretations of CUTE's SM120 FP4 ALayout.
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

__device__ __forceinline__ float k_value(int k) {
  constexpr float vals[7] = {0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return vals[k % 7];
}

__device__ __forceinline__ uint32_t e2m1_positive_code(float x) {
  if (x <= 0.5f) return 1u;
  if (x <= 1.f) return 2u;
  if (x <= 1.5f) return 3u;
  if (x <= 2.f) return 4u;
  if (x <= 3.f) return 5u;
  if (x <= 4.f) return 6u;
  return 7u;
}

template <int INTERPRETATION>
__device__ __forceinline__ uint32_t make_a_reg(int reg) {
  typename cute::MMA_Traits<Atom>::ALayout layout;
  int const lane = threadIdx.x & 31;
  uint32_t word = 0;
#pragma unroll
  for (int nib = 0; nib < 8; ++nib) {
    int const value_idx = reg * 8 + nib;
    int const linear = int(layout(lane, value_idx));
    int k;
    if constexpr (INTERPRETATION == 0) {
      k = linear % 64;
    } else {
      k = linear / 16;
    }
    uint32_t const code = e2m1_positive_code(k_value(k));
    word |= code << (4 * nib);
  }
  return word;
}

template <int INTERPRETATION>
__global__ void probe(float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  uint32_t a[4] = {make_a_reg<INTERPRETATION>(0), make_a_reg<INTERPRETATION>(1),
                   make_a_reg<INTERPRETATION>(2), make_a_reg<INTERPRETATION>(3)};
  uint32_t b[4] = {0x22222222u, 0x22222222u, 0x22222222u, 0x22222222u};
  uint32_t const scale = fmha::make_ue4m3_scale_reg(1.f, 1.f, 1.f, 1.f);
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  Atom::fma(acc[0], acc[1], acc[2], acc[3], a[0], a[1], a[2], a[3], b[0],
            b[1], acc[0], acc[1], acc[2], acc[3], scale, scale);
  Atom::fma(acc[4], acc[5], acc[6], acc[7], a[0], a[1], a[2], a[3], b[2],
            b[3], acc[4], acc[5], acc[6], acc[7], scale, scale);
  int const lane = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[lane * 8 + i] = acc[i];
  }
#endif
}

int main() {
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, 32 * 8 * sizeof(float)));
  for (int interp = 0; interp < 2; ++interp) {
    CUDA_CHECK(cudaMemset(out, 0, 32 * 8 * sizeof(float)));
    if (interp == 0) {
      probe<0><<<1, 32>>>(out);
    } else {
      probe<1><<<1, 32>>>(out);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float host[32 * 8];
    CUDA_CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
    float mn = host[0], mx = host[0];
    for (float v : host) {
      mn = v < mn ? v : mn;
      mx = v > mx ? v : mx;
    }
    std::printf("interpretation %d range=[%.1f, %.1f] sample:", interp, mn, mx);
    for (int i = 0; i < 8; ++i) std::printf(" %.1f", host[i]);
    std::printf("\\n");
  }
  CUDA_CHECK(cudaFree(out));
  return 0;
}
