#include <cstdio>
#include <cuda_runtime.h>

#include "fmha/utils.h"

namespace {

__global__ void pack_kernel(uint32_t* e2m1_out, uint32_t* scale_out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float base = 0.125f * static_cast<float>((tid & 7) + 1);

  e2m1_out[tid] = fmha::float8_to_e2m1x8(base, base * 2.f, base * 3.f, base * 4.f,
                                         base * 5.f, base * 6.f, base * 7.f, base * 8.f);
  scale_out[tid] = fmha::make_ue4m3_scale_reg(1.f, 0.5f, 0.25f, 0.125f);
}

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

}  // namespace

int main() {
  constexpr int kItems = 256;
  uint32_t *e2m1 = nullptr, *scale = nullptr;
  check(cudaMalloc(&e2m1, kItems * sizeof(uint32_t)), "cudaMalloc e2m1");
  check(cudaMalloc(&scale, kItems * sizeof(uint32_t)), "cudaMalloc scale");

  pack_kernel<<<1, kItems>>>(e2m1, scale);
  check(cudaGetLastError(), "pack_kernel launch");
  check(cudaDeviceSynchronize(), "pack_kernel sync");

  uint32_t h_e2m1 = 0, h_scale = 0;
  check(cudaMemcpy(&h_e2m1, e2m1, sizeof(uint32_t), cudaMemcpyDeviceToHost), "copy e2m1");
  check(cudaMemcpy(&h_scale, scale, sizeof(uint32_t), cudaMemcpyDeviceToHost), "copy scale");
  std::printf("e2m1=0x%08x scale=0x%08x\n", h_e2m1, h_scale);

  check(cudaFree(e2m1), "free e2m1");
  check(cudaFree(scale), "free scale");
  return 0;
}

