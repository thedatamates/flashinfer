/*
 * Print the nibble order produced by flashinfer::mma::float8_to_e2m1x8.
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

__global__ void pack_order_kernel(uint32_t* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  out[0] = flashinfer::mma::float8_to_e2m1x8(0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.5f);
#else
  out[0] = 0xffffffffu;
#endif
}

int main() {
  uint32_t* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, sizeof(uint32_t)));
  pack_order_kernel<<<1, 1>>>(out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t host = 0;
  CUDA_CHECK(cudaMemcpy(&host, out, sizeof(host), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));
  std::printf("word=0x%08x nibbles:", host);
  for (int i = 0; i < 8; ++i) {
    std::printf(" %x", (host >> (4 * i)) & 0xf);
  }
  std::printf("\n");
  return 0;
}
