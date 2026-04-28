#include <cute/arch/mma_sm120.hpp>
#include <cutlass/float8.h>
#include <cutlass/float_subbyte.h>

#include <cstdio>

__device__ inline uint32_t fp8e4m3x4(float x0, float x1, float x2, float x3) {
  uint32_t res;
  asm volatile(
      "{\n"
      ".reg .b16 lo;\n"
      ".reg .b16 hi;\n"
      "cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n"
      "mov.b32 %0, {lo, hi};\n"
      "}"
      : "=r"(res)
      : "f"(x0), "f"(x1), "f"(x2), "f"(x3));
  return res;
}

__global__ void constant_mma(float* out) {
  using Atom = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
      cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
      cutlass::float_ue4m3_t, 16>;

  // E2M1 code 0x2 is +1.0. Fill every A/B operand slot with +1.0 and
  // every block scale with +1.0. Each C element should be sum_k(1*1)=64.
  uint32_t const fp4_ones = 0x22222222u;
  uint32_t const scale_ones = fp8e4m3x4(1.f, 1.f, 1.f, 1.f);
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  Atom::fma(d0, d1, d2, d3,
            fp4_ones, fp4_ones, fp4_ones, fp4_ones,
            fp4_ones, fp4_ones,
            d0, d1, d2, d3,
            scale_ones, scale_ones);
  int const lane = threadIdx.x & 31;
  out[lane * 4 + 0] = d0;
  out[lane * 4 + 1] = d1;
  out[lane * 4 + 2] = d2;
  out[lane * 4 + 3] = d3;
}

int main() {
  float* out = nullptr;
  cudaMalloc(&out, 32 * 4 * sizeof(float));
  constant_mma<<<1, 32>>>(out);
  cudaDeviceSynchronize();
  float host[32 * 4];
  cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost);
  cudaFree(out);
  for (int lane = 0; lane < 32; ++lane) {
    std::printf("lane %02d: %.1f %.1f %.1f %.1f\n",
                lane, host[lane * 4 + 0], host[lane * 4 + 1],
                host[lane * 4 + 2], host[lane * 4 + 3]);
  }
}
