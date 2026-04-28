/*
 * Probe Blackwell NVFP4 V SMEM column mapping through the PV MMA.
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <fmha/fragment.h>
#include <fmha/smem_tile_v.h>
#include <fmha/traits.h>
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

using Traits = fmha::Blackwell_mma_nvf4_fp32_traits;
using CtaTile = Traits::Cta_tile_extd</*M=*/64, /*N=*/256, /*K=*/64,
                                      /*VALID_N=*/256, /*VALID_K=*/64,
                                      /*WARPS_M=*/4, /*WARPS_N=*/1,
                                      /*WARPS_K=*/1>;
using SmemTile = fmha::Smem_tile_v<Traits, CtaTile, 1>;
using MmaTile = Traits::Mma_tile<CtaTile>;

__global__ void probe_kernel(float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  extern __shared__ char smem[];
  int const active_col = static_cast<int>(blockIdx.x);
  int const lane = threadIdx.x & 31;
  SmemTile smem_v(smem, threadIdx.x);

  for (int byte_idx = threadIdx.x; byte_idx < CtaTile::K * (CtaTile::N / 2);
       byte_idx += CtaTile::THREADS_PER_CTA) {
    int const row = byte_idx / (CtaTile::N / 2);
    int const byte_col = byte_idx - row * (CtaTile::N / 2);
    int const col0 = byte_col * 2;
    uint8_t packed = 0u;
    if (row == 0) {
      if (col0 == active_col) {
        packed |= 0x2u;
      }
      if (col0 + 1 == active_col) {
        packed |= 0x20u;
      }
    }
    smem_v.store_v_data_byte(row, col0, packed);
  }
  for (int idx = threadIdx.x; idx < CtaTile::N * (CtaTile::K / Traits::NVFP4_SCALE_VEC_SIZE);
       idx += CtaTile::THREADS_PER_CTA) {
    int const col = idx / (CtaTile::K / Traits::NVFP4_SCALE_VEC_SIZE);
    int const scale_group = idx - col * (CtaTile::K / Traits::NVFP4_SCALE_VEC_SIZE);
    smem_v.store_v_scale(col, scale_group, fmha::float_to_e4m3_byte(1.f));
  }
  __syncthreads();

  fmha::Fragment_a<Traits, fmha::Row> a;
  a.reg(0) = lane == 0 ? 0x2u : 0u;
  a.reg(1) = 0u;
  a.reg(2) = 0u;
  a.reg(3) = 0u;
  a.scale_reg = fmha::make_ue4m3_scale_reg(1.f, 1.f, 1.f, 1.f);

  typename SmemTile::Fragment b[MmaTile::VALID_MMAS_N];
  smem_v.load(b, 0);

  int const ni = active_col / MmaTile::N_PER_MMA_PER_CTA;
  fmha::Fragment_accumulator<Traits> acc;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc.elt(i) = 0.f;
  }
  acc.mma(a, b[ni]);

#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    int const row = (lane >> 2) + 8 * ((elem % 4) / 2);
    int const col = ni * MmaTile::N_PER_MMA_PER_CTA +
                    2 * (lane & 3) + (elem & 1) + 8 * (elem / 4);
    if (row == 0) {
      out[active_col * CtaTile::N + col] = acc.elt(elem);
    }
  }
#endif
}

int main() {
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, CtaTile::N * CtaTile::N * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, CtaTile::N * CtaTile::N * sizeof(float)));
  probe_kernel<<<CtaTile::N, CtaTile::THREADS_PER_CTA, SmemTile::BYTES_PER_TILE>>>(out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float* host = new float[CtaTile::N * CtaTile::N];
  CUDA_CHECK(cudaMemcpy(host, out, CtaTile::N * CtaTile::N * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));

  int bad = 0;
  for (int src = 0; src < 64; ++src) {
    int argmax = 0;
    float max_val = host[src * CtaTile::N];
    for (int col = 1; col < 64; ++col) {
      float const val = host[src * CtaTile::N + col];
      if (val > max_val) {
        max_val = val;
        argmax = col;
      }
    }
    bool const ok = argmax == src && max_val > 0.9f;
    bad += ok ? 0 : 1;
    std::printf("src %02d -> argmax %02d %.1f%s\n", src, argmax, max_val,
                ok ? "" : " BAD");
  }
  delete[] host;
  std::printf("bad %d\n", bad);
  return bad == 0 ? 0 : 1;
}
