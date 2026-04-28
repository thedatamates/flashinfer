/*
 * Scale-semantics probe for the SM120 native NVFP4 MMA atom.
 *
 * This is intentionally small: it feeds constant E2M1 operands and constant
 * UE4M3 scale registers into the exact Fragment_accumulator path used by the
 * FMHAv2 bring-up. The first output lane should make scale mistakes obvious.
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <fmha/fragment.h>
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

__global__ void nvfp4_scale_probe_kernel(float* out, float a_scale,
                                         float b_scale, float a_value,
                                         float b_value, int distinct_a_scales,
                                         int distinct_b_scales) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  fmha::Fragment_a<fmha::Blackwell_mma_nvf4_fp32_traits, fmha::Row> a;
  fmha::Fragment_b<fmha::Blackwell_mma_nvf4_fp32_traits, fmha::Col> b;
  fmha::Fragment_accumulator<fmha::Blackwell_mma_nvf4_fp32_traits> acc;

  uint32_t const a_reg = fmha::float8_to_e2m1x8(a_value, a_value, a_value,
                                                a_value, a_value, a_value,
                                                a_value, a_value);
  uint32_t const b_reg = fmha::float8_to_e2m1x8(b_value, b_value, b_value,
                                                b_value, b_value, b_value,
                                                b_value, b_value);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a.reg(i) = a_reg;
    b.reg(i) = b_reg;
  }
  a.scale_reg = distinct_a_scales
                    ? fmha::make_ue4m3_scale_reg(a_scale, a_scale * 2.f,
                                                 a_scale * 4.f, a_scale * 8.f)
                    : fmha::make_ue4m3_scale_reg(a_scale, a_scale, a_scale,
                                                 a_scale);
  b.scale_reg[0] = distinct_b_scales
                       ? fmha::make_ue4m3_scale_reg(b_scale, b_scale * 2.f,
                                                    b_scale * 4.f, b_scale * 8.f)
                       : fmha::make_ue4m3_scale_reg(b_scale, b_scale, b_scale,
                                                    b_scale);
  b.scale_reg[1] = b.scale_reg[0];

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc.elt(i) = 0.f;
  }
  acc.mma(a, b);

  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      out[i] = acc.elt(i);
    }
  }
#else
  if (threadIdx.x == 0) {
    out[0] = -1.f;
  }
#endif
}

int main(int argc, char** argv) {
  float a_scale = argc > 1 ? std::atof(argv[1]) : 1.f;
  float b_scale = argc > 2 ? std::atof(argv[2]) : 1.f;
  float a_value = argc > 3 ? std::atof(argv[3]) : 1.f;
  float b_value = argc > 4 ? std::atof(argv[4]) : 1.f;
  int distinct_a_scales = argc > 5 ? std::atoi(argv[5]) : 0;
  int distinct_b_scales = argc > 6 ? std::atoi(argv[6]) : 0;

  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, 8 * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, 8 * sizeof(float)));
  nvfp4_scale_probe_kernel<<<1, 32>>>(out, a_scale, b_scale, a_value, b_value,
                                      distinct_a_scales, distinct_b_scales);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float h[8];
  CUDA_CHECK(cudaMemcpy(h, out, sizeof(h), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));

  std::printf("a_scale=%g b_scale=%g a_value=%g b_value=%g distinct_a=%d distinct_b=%d out=[",
              a_scale, b_scale, a_value, b_value, distinct_a_scales,
              distinct_b_scales);
  for (int i = 0; i < 8; ++i) {
    std::printf("%s%g", i ? "," : "", h[i]);
  }
  std::printf("]\n");
  return 0;
}
