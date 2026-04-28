#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <flashinfer/attention/prefill.cuh>

#define CUDA_CHECK(expr)                                                        \
  do {                                                                         \
    cudaError_t status = (expr);                                               \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(status));                                \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace fi = flashinfer;

using TestAttention = fi::DefaultAttention<false, false, false, false>;
using TestTraits =
    fi::KernelTraits<fi::MaskMode::kNone, 32, 1, 4, 32, 16, 2, 1, fi::PosEncodingMode::kNone,
                     __nv_bfloat16, __nv_fp4x2_e2m1, __nv_bfloat16, float, int32_t,
                     TestAttention>;

__global__ void probe(uint32_t onehot_col, uint32_t* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  float s_frag[TestTraits::NUM_MMA_Q][TestTraits::NUM_MMA_KV][8];
  float row_boost[2] = {1.f, 1.f};
  uint32_t a_frag[4] = {};
  uint32_t scale = 0;

  for (uint32_t mma_kv = 0; mma_kv < TestTraits::NUM_MMA_KV; ++mma_kv) {
    for (uint32_t reg = 0; reg < 8; ++reg) {
      const uint32_t col =
          mma_kv * 16 + 2 * (threadIdx.x & 3) + 8 * (reg / 4) + (reg & 1);
      s_frag[0][mma_kv][reg] = col == onehot_col ? 1.f : 0.f;
    }
  }

  scale = fi::make_s_scale_reg<TestTraits>(s_frag, 0, row_boost);
  fi::make_s_frag_fp4<TestTraits>(s_frag, 0, scale, row_boost, a_frag);

  if (threadIdx.x < 8) {
    const uint32_t base = threadIdx.x * 5;
    out[base + 0] = scale;
    out[base + 1] = a_frag[0];
    out[base + 2] = a_frag[1];
    out[base + 3] = a_frag[2];
    out[base + 4] = a_frag[3];
  }
#endif
}

int main(int argc, char** argv) {
  uint32_t onehot_col = argc > 1 ? static_cast<uint32_t>(std::atoi(argv[1])) : 2;
  uint32_t* d = nullptr;
  uint32_t h[8 * 5] = {};
  CUDA_CHECK(cudaMalloc(&d, sizeof(h)));
  CUDA_CHECK(cudaMemset(d, 0, sizeof(h)));
  probe<<<1, 32>>>(onehot_col, d);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d));
  std::printf("onehot_col=%u\n", onehot_col);
  for (int lane = 0; lane < 8; ++lane) {
    uint32_t* p = h + lane * 5;
    std::printf("lane%d scale=%08x a=%08x %08x %08x %08x\n", lane, p[0], p[1], p[2], p[3],
                p[4]);
  }
}
