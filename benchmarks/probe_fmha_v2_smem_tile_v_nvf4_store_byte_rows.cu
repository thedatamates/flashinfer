/*
 * Probe Smem_tile_v::store_v_data_byte for all logical K rows.
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
  int const active_row = static_cast<int>(blockIdx.x);
  SmemTile smem_v(smem, threadIdx.x);

  for (int byte_idx = threadIdx.x; byte_idx < CtaTile::K * (CtaTile::N / 2);
       byte_idx += CtaTile::THREADS_PER_CTA) {
    int const row = byte_idx / (CtaTile::N / 2);
    int const byte_col = byte_idx - row * (CtaTile::N / 2);
    smem_v.store_v_data_byte(row, byte_col * 2,
                             row == active_row ? 0x22u : 0x00u);
  }
  for (int idx = threadIdx.x; idx < CtaTile::N * (CtaTile::K / Traits::NVFP4_SCALE_VEC_SIZE);
       idx += CtaTile::THREADS_PER_CTA) {
    int const col = idx / (CtaTile::K / Traits::NVFP4_SCALE_VEC_SIZE);
    int const scale_group = idx - col * (CtaTile::K / Traits::NVFP4_SCALE_VEC_SIZE);
    smem_v.store_v_scale(col, scale_group, fmha::float_to_e4m3_byte(1.f));
  }
  __syncthreads();

  fmha::Fragment_a<Traits, fmha::Row> a;
  a.reg(0) = 0x22222222u;
  a.reg(1) = 0x22222222u;
  a.reg(2) = 0x22222222u;
  a.reg(3) = 0x22222222u;
  a.scale_reg = fmha::make_ue4m3_scale_reg(1.f, 1.f, 1.f, 1.f);

  typename SmemTile::Fragment b[MmaTile::VALID_MMAS_N];
  smem_v.load(b, 0);

  fmha::Fragment_accumulator<Traits> acc;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc.elt(i) = 0.f;
  }
  acc.mma(a, b[0]);

  int const lane = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[active_row * 32 * 8 + lane * 8 + i] = acc.elt(i);
  }
#endif
}

int main() {
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, 64 * 32 * 8 * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, 64 * 32 * 8 * sizeof(float)));
  probe_kernel<<<64, CtaTile::THREADS_PER_CTA, SmemTile::BYTES_PER_TILE>>>(out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float host[64 * 32 * 8];
  CUDA_CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));

  int bad = 0;
  for (int row = 0; row < 64; ++row) {
    float mn = host[row * 32 * 8];
    float mx = mn;
    for (int i = 0; i < 32 * 8; ++i) {
      float const v = host[row * 32 * 8 + i];
      mn = v < mn ? v : mn;
      mx = v > mx ? v : mx;
    }
    bool const ok = mn == 1.f && mx == 1.f;
    bad += ok ? 0 : 1;
    if (!ok || row < 16 || row % 8 == 0) {
      std::printf("row %02d range [%.1f, %.1f]%s\n", row, mn, mx,
                  ok ? "" : " BAD");
    }
  }
  std::printf("bad %d\n", bad);
  return bad == 0 ? 0 : 1;
}
