/*
 * Direct SM120 FP4 MMA probe for B-fragment K ordering.
 *
 * A is all ones. B is a function of K only and is constant across N. Every
 * output column must therefore be identical if B fragment packing matches the
 * native MMA's K-axis layout.
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <flashinfer/mma.cuh>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                        \
    cudaError_t status = (expr);                                              \
    if (status != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(status));                               \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

__device__ __forceinline__ float e2m1_row_value(uint32_t k) {
  // Positive E2M1-representable values, repeated every 7 rows.
  constexpr float vals[7] = {0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return vals[k % 7];
}

__device__ __forceinline__ uint32_t make_b_frag_row_ramp(uint32_t atom_n_offset,
                                                         uint32_t reg) {
  const uint32_t lane = threadIdx.x & 31;
  float vals[8];
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const uint32_t value_idx = i * 2 + reg;
    const uint32_t k = ((lane >> 2) & 0x7u) + 8 * (value_idx & 0x7u);
    vals[i] = e2m1_row_value(k);
  }
  return flashinfer::mma::float8_to_e2m1x8(vals[0], vals[1], vals[2], vals[3], vals[4],
                                           vals[5], vals[6], vals[7]);
}

__global__ void b_row_ramp_probe_kernel(float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  uint32_t a[4] = {0x22222222u, 0x22222222u, 0x22222222u, 0x22222222u};
  uint32_t b[4] = {
      make_b_frag_row_ramp(0, 0),
      make_b_frag_row_ramp(0, 1),
      make_b_frag_row_ramp(8, 0),
      make_b_frag_row_ramp(8, 1),
  };
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  constexpr uint32_t scale = 0x38383838u;
  flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32<flashinfer::mma::MMAMode::kInit>(
      acc, a, b, scale, scale, scale);
  const uint32_t lane = threadIdx.x & 31;
  for (int i = 0; i < 8; ++i) {
    out[lane * 8 + i] = acc[i];
  }
#else
  if (threadIdx.x == 0) {
    out[0] = -1.f;
  }
#endif
}

int main() {
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, 32 * 8 * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, 32 * 8 * sizeof(float)));
  b_row_ramp_probe_kernel<<<1, 32>>>(out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float host[32 * 8];
  CUDA_CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));
  for (int lane = 0; lane < 32; ++lane) {
    std::printf("lane %02d:", lane);
    for (int i = 0; i < 8; ++i) {
      std::printf(" %.1f", host[lane * 8 + i]);
    }
    std::printf("\n");
  }
  return 0;
}
