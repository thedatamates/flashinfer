#include <cstdio>
#include <cuda_runtime.h>

#include "fmha/softmax.h"

namespace {

struct TestCtaTile {
  enum {
    M = 16,
    N = 64,
    K = 64,
    VALID_N = 64,
    VALID_K = 64,
    WARPS_M = 1,
    WARPS_N = 1,
    WARPS_K = 1,
    THREADS_PER_WARP = 32,
  };
};

struct TestKernelTraits {
  enum { VERSION = 1, CAUSAL_MASK = 0 };
};

struct TestParams {
  uint32_t scale_bmm1;
  uint32_t const* scale_bmm1_d;
};

__global__ void softmax_pack_kernel(uint32_t* regs, uint32_t* scales) {
  extern __shared__ float smem[];
  using Traits = fmha::Blackwell_mma_nvf4_fp32_traits;
  using Softmax = fmha::Softmax<Traits, TestCtaTile, TestKernelTraits>;
  using MmaTile = typename Traits::template Mma_tile<TestCtaTile>;

  float scale = 1.f;
  TestParams params{reinterpret_cast<uint32_t&>(scale), nullptr};
  Softmax softmax(params, smem, 0, threadIdx.x);

  fmha::Fragment_accumulator<Traits> acc[MmaTile::MMAS_M][MmaTile::MMAS_N];
#pragma unroll
  for (int ni = 0; ni < MmaTile::MMAS_N; ++ni) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[0][ni].elt(i) = static_cast<float>((threadIdx.x & 3) + ni + i) * 0.125f;
    }
  }

  float row_max[MmaTile::MMAS_M * 2] = {4.f, 4.f};
  softmax.unpack(acc);
  softmax.template apply_exp_with_mask<false>(row_max);

  fmha::Fragment_a<Traits, fmha::Row> frag[1][MmaTile::MMAS_M];
  softmax.pack(frag);

  int tid = threadIdx.x;
  regs[tid * 4 + 0] = frag[0][0].reg(0);
  regs[tid * 4 + 1] = frag[0][0].reg(1);
  regs[tid * 4 + 2] = frag[0][0].reg(2);
  regs[tid * 4 + 3] = frag[0][0].reg(3);
  scales[tid] = frag[0][0].scale_reg;
}

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

}  // namespace

int main() {
  uint32_t *regs = nullptr, *scales = nullptr;
  check(cudaMalloc(&regs, 32 * 4 * sizeof(uint32_t)), "cudaMalloc regs");
  check(cudaMalloc(&scales, 32 * sizeof(uint32_t)), "cudaMalloc scales");

  softmax_pack_kernel<<<1, 32, 1024>>>(regs, scales);
  check(cudaGetLastError(), "softmax_pack launch");
  check(cudaDeviceSynchronize(), "softmax_pack sync");

  uint32_t h_regs[4] = {};
  uint32_t h_scale = 0;
  check(cudaMemcpy(h_regs, regs, sizeof(h_regs), cudaMemcpyDeviceToHost), "copy regs");
  check(cudaMemcpy(&h_scale, scales, sizeof(h_scale), cudaMemcpyDeviceToHost), "copy scale");
  std::printf("regs=[0x%08x,0x%08x,0x%08x,0x%08x] scale=0x%08x\n", h_regs[0], h_regs[1],
              h_regs[2], h_regs[3], h_scale);

  check(cudaFree(regs), "free regs");
  check(cudaFree(scales), "free scales");
  return 0;
}

