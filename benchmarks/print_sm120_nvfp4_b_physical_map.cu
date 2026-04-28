/*
 * Print physical B nibble -> C output map for SM120 block-scaled FP4 MMA.
 *
 * A is all ones. For each physical B nibble position, exactly one FP4 value is
 * set to +1. The nonzero C columns show which N column that physical position
 * feeds. This validates the subbyte packing order, not just CUTE's logical
 * BLayout.
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

__device__ __forceinline__ void set_nibble(uint32_t* regs, uint32_t physical_pos,
                                           uint32_t code) {
  const uint32_t reg = physical_pos / 8;
  const uint32_t nib = physical_pos % 8;
  regs[reg] |= code << (4 * nib);
}

__global__ void b_physical_map_kernel(float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  const uint32_t target_lane = blockIdx.x;
  const uint32_t p = blockIdx.y;
  const uint32_t lane = threadIdx.x & 31;
  uint32_t a[4] = {0x22222222u, 0x22222222u, 0x22222222u, 0x22222222u};
  uint32_t b[4] = {0u, 0u, 0u, 0u};
  if (lane == target_lane) {
    set_nibble(b, p, 0x2u);  // +1.0
  }
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  constexpr uint32_t scale = 0x38383838u;
  flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32<flashinfer::mma::MMAMode::kInit>(
      acc, a, b, scale, scale, scale);
  const uint32_t block = blockIdx.y * 32 + blockIdx.x;
  const uint32_t base = (block * 32 + lane) * 8;
  for (int i = 0; i < 8; ++i) {
    out[base + i] = acc[i];
  }
#else
  if (threadIdx.x == 0) {
    out[0] = -1.f;
  }
#endif
}

int main() {
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, 32 * 32 * 32 * 8 * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, 32 * 32 * 32 * 8 * sizeof(float)));
  b_physical_map_kernel<<<dim3(32, 32, 1), 32>>>(out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float host[32 * 32 * 32 * 8];
  CUDA_CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));
  for (int target_lane = 0; target_lane < 32; ++target_lane) {
    for (int p = 0; p < 32; ++p) {
    std::printf("src L%02d phys %02d:", target_lane, p);
    int printed = 0;
    int block = p * 32 + target_lane;
    for (int lane = 0; lane < 32; ++lane) {
      for (int r = 0; r < 8; ++r) {
        const float v = host[(block * 32 + lane) * 8 + r];
        if (v != 0.f) {
          std::printf(" L%02dR%d=%.0f", lane, r, v);
          ++printed;
          if (printed >= 8) {
            break;
          }
        }
      }
      if (printed >= 8) {
        break;
      }
    }
    std::printf("\n");
    }
  }
  return 0;
}
