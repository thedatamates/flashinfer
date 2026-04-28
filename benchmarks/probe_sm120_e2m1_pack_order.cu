#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <flashinfer/mma.cuh>

#define CUDA_CHECK(expr)                                                        \
  do {                                                                         \
    cudaError_t status = (expr);                                               \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(status));                                \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__global__ void probe(uint32_t* out) {
  int i = threadIdx.x;
  float x[8] = {};
  x[i] = 1.f;
  out[i] = flashinfer::mma::float8_to_e2m1x8(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
}

int main() {
  uint32_t* d = nullptr;
  uint32_t h[8] = {};
  CUDA_CHECK(cudaMalloc(&d, sizeof(h)));
  probe<<<1, 8>>>(d);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d));
  for (int i = 0; i < 8; ++i) {
    std::printf("arg%d -> 0x%08x nibbles:", i, h[i]);
    for (int n = 0; n < 8; ++n) {
      std::printf(" %x", (h[i] >> (4 * n)) & 0xf);
    }
    std::printf("\n");
  }
}
