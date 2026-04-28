/*
 * Standalone smoke benchmark for the native SM120 NVFP4 MMA atom.
 *
 * This intentionally bypasses the CUTLASS GEMM wrapper and calls the
 * block-scaled mma.sync instruction directly through CUTE. It is a small
 * bring-up target for the fused attention kernel path.
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <flashinfer/mma.cuh>

#define CUDA_CHECK(expr)                                                              \
  do {                                                                               \
    cudaError_t status = (expr);                                                     \
    if (status != cudaSuccess) {                                                     \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(status));                                      \
      std::exit(1);                                                                  \
    }                                                                                \
  } while (0)

__global__ void nvfp4_mma_atom_kernel(float* out, int iters) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  uint32_t a[4];
  uint32_t b[4];
  float acc[8];

  a[0] = 0x22222222u;
  a[1] = 0x22222222u;
  a[2] = 0x22222222u;
  a[3] = 0x22222222u;
  b[0] = 0x22222222u;
  b[1] = 0x22222222u;
  b[2] = 0x22222222u;
  b[3] = 0x22222222u;

  // UE4M3 scale factor byte 0x38 encodes 1.0. VS=16 consumes four scale
  // factors per packed register for the 4X block-scaled MMA mode.
  uint32_t const a_scale_reg = 0x38383838u;
  uint32_t const b_scale_reg0 = 0x38383838u;
  uint32_t const b_scale_reg1 = 0x38383838u;

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc[i] = 0.f;
  }

  if (iters > 0) {
    flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32<flashinfer::mma::MMAMode::kInit>(
        acc, a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], a_scale_reg, b_scale_reg0,
        b_scale_reg1);
  }
  for (int i = 1; i < iters; ++i) {
    flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(
        acc, a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], a_scale_reg, b_scale_reg0,
        b_scale_reg1);
  }

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[idx * 8 + i] = acc[i];
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = -1.f;
  }
#endif
}

int main(int argc, char** argv) {
  int blocks = argc > 1 ? std::atoi(argv[1]) : 4096;
  int iters = argc > 2 ? std::atoi(argv[2]) : 1024;
  int threads = 32;
  size_t out_elems = static_cast<size_t>(blocks) * threads * 8;

  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, out_elems * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, out_elems * sizeof(float)));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  nvfp4_mma_atom_kernel<<<blocks, threads>>>(out, 4);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  nvfp4_mma_atom_kernel<<<blocks, threads>>>(out, iters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  float sample[4];
  CUDA_CHECK(cudaMemcpy(sample, out, sizeof(sample), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));

  // One warp issues two native m16n8k64 atoms, i.e. a 16x16x64 tile.
  double flops =
      static_cast<double>(blocks) * iters * 2.0 * 16.0 * 16.0 * 64.0;
  double tflops = flops / (static_cast<double>(ms) * 1.0e9);

  std::printf("blocks=%d threads=%d iters=%d ms=%.6f atom_tflops=%.2f sample=[%g,%g,%g,%g]\n",
              blocks, threads, iters, ms, tflops, sample[0], sample[1], sample[2], sample[3]);
  return 0;
}
