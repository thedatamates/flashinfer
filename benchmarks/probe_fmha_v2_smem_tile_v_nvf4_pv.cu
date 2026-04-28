/*
 * Isolate the FMHAv2 Blackwell NVFP4 V shared-memory tile.
 *
 * This bypasses softmax and paged-KV wrappers. It writes a logical V tile
 * through Smem_tile_v::store(), reads it back through Smem_tile_v::load(), and
 * feeds the resulting B fragments to the native SM120 FP4 MMA with A = 1.
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

__host__ __device__ __forceinline__ float row_value(int row) {
  constexpr float vals[7] = {0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return vals[row % 7];
}

__device__ __forceinline__ uint32_t e2m1_positive_code(float x) {
  if (x <= 0.5f) {
    return 1u;
  }
  if (x <= 1.f) {
    return 2u;
  }
  if (x <= 1.5f) {
    return 3u;
  }
  if (x <= 2.f) {
    return 4u;
  }
  if (x <= 3.f) {
    return 5u;
  }
  if (x <= 4.f) {
    return 6u;
  }
  return 7u;
}

__device__ __forceinline__ uint8_t packed_pair_for_code(uint8_t code) {
  return static_cast<uint8_t>(code | (code << 4));
}

__global__ void probe_kernel(float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  extern __shared__ char smem[];
  SmemTile smem_v(smem, threadIdx.x);

  typename SmemTile::Store_type data[SmemTile::STS];
  uint32_t* words = reinterpret_cast<uint32_t*>(data);
#pragma unroll
  for (int i = 0; i < SmemTile::STS * 4; ++i) {
    words[i] = 0u;
  }

  int const row_base = threadIdx.x / SmemTile::THREADS_PER_ROW;
  int const col_byte_base =
      (threadIdx.x % SmemTile::THREADS_PER_ROW) * SmemTile::BYTES_PER_STS;
#pragma unroll
  for (int si = 0; si < SmemTile::STS; ++si) {
    int const row = row_base + si * SmemTile::ROWS_PER_STS;
    uint8_t code;
    if (blockIdx.x == 0) {
      code = static_cast<uint8_t>(e2m1_positive_code(row_value(row)));
    } else {
      code = row == static_cast<int>(blockIdx.x - 1) ? 2u : 0u;
    }
    uint8_t const packed = packed_pair_for_code(code);
    uint32_t word = static_cast<uint32_t>(packed) |
                    (static_cast<uint32_t>(packed) << 8) |
                    (static_cast<uint32_t>(packed) << 16) |
                    (static_cast<uint32_t>(packed) << 24);
    uint32_t* dst = reinterpret_cast<uint32_t*>(&data[si]);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      int const logical_byte = col_byte_base + w * 4;
      dst[w] = logical_byte < SmemTile::BYTES_PER_ROW_BEFORE_PACKING ? word : 0u;
    }
  }

  smem_v.store(data);

  for (int idx = threadIdx.x; idx < CtaTile::N * (CtaTile::K / Traits::NVFP4_SCALE_VEC_SIZE);
       idx += CtaTile::THREADS_PER_CTA) {
    int const logical_col = idx / (CtaTile::K / Traits::NVFP4_SCALE_VEC_SIZE);
    int const scale_group = idx - logical_col * (CtaTile::K / Traits::NVFP4_SCALE_VEC_SIZE);
    smem_v.store_v_scale(logical_col, scale_group, fmha::float_to_e4m3_byte(1.f));
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
    out[blockIdx.x * 32 * 8 + lane * 8 + i] = acc.elt(i);
  }
#else
  if (threadIdx.x == 0) {
    out[0] = -1.f;
  }
#endif
}

int main() {
  float expected = 0.f;
  for (int row = 0; row < 64; ++row) {
    expected += row_value(row);
  }
  std::printf("expected %.1f\n", expected);

  float* out = nullptr;
  constexpr int kBlocks = 65;
  CUDA_CHECK(cudaMalloc(&out, kBlocks * 32 * 8 * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, kBlocks * 32 * 8 * sizeof(float)));
  probe_kernel<<<kBlocks, CtaTile::THREADS_PER_CTA, SmemTile::BYTES_PER_TILE>>>(out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float host[kBlocks * 32 * 8];
  CUDA_CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));

  float mn = host[0];
  float mx = host[0];
  for (int i = 0; i < 32 * 8; ++i) {
    float v = host[i];
    mn = v < mn ? v : mn;
    mx = v > mx ? v : mx;
  }
  std::printf("range [%.1f, %.1f]\n", mn, mx);
  for (int lane = 0; lane < 8; ++lane) {
    std::printf("lane %02d:", lane);
    for (int i = 0; i < 8; ++i) {
      std::printf(" %.1f", host[lane * 8 + i]);
    }
    std::printf("\n");
  }
  std::printf("onehot:");
  for (int row = 0; row < 64; ++row) {
    float const v = host[(row + 1) * 32 * 8];
    if (v != 1.f) {
      std::printf(" r%d=%.1f", row, v);
    }
  }
  std::printf("\n");
  return 0;
}
