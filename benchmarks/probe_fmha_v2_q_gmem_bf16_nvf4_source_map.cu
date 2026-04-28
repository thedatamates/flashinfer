/*
 * Probe the production BF16-Q -> NVFP4 shared-memory path used by the
 * Blackwell FMHAv2 D=512 grouped-M kernel.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <fmha/kernel_traits.h>
#include <fmha/smem_tile.h>
#include <fmha/traits.h>
#include <fmha/utils.h>
#include <fused_multihead_attention.h>
#include <fused_multihead_attention_kernel.h>

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
using KernelTraits = fmha::Kernel_traits_v2_bf16_q_nvf4_paged_kv_cache<
    Traits, 64, 512, 0, 64, 4, 1, 1, 0x5022u | 0x200u | 0x4000u>;
using CtaTile = KernelTraits::Cta_tile_p;
using MmaTile = Traits::Mma_tile<CtaTile>;
using GmemQ = KernelTraits::Gmem_tile_q;
using SmemQ = KernelTraits::Smem_tile_q;

__global__ void probe_kernel(bert::Fused_multihead_attention_params_v2 params,
                             int active_row, int active_col, float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  extern __shared__ char smem[];
  int const q_col_tile = active_col / CtaTile::K;
  int const tidx = threadIdx.x;

  auto binfo = fused_multihead_attention::Single_cta<2>(params, 0, 0, 0, tidx);
  GmemQ gmem_q(params, 0, binfo, tidx);
  SmemQ smem_q(smem, tidx);
  gmem_q.move_col(q_col_tile);
  gmem_q.load(smem_q);
  gmem_q.commit(smem_q);
  __syncthreads();

  float* tile = out;
  for (int idx = tidx; idx < CtaTile::M * CtaTile::K; idx += CtaTile::THREADS_PER_CTA) {
    int const logical_row = idx / CtaTile::K;
    int const local_col = idx - logical_row * CtaTile::K;
    uint8_t byte = smem_q.load_q_data_byte(logical_row, local_col);
    uint32_t const code = (local_col & 1) ? ((byte >> 4) & 0x0fu) : (byte & 0x0fu);
    if (code != 0u) {
      tile[idx] = fmha::e2m1_to_float(code);
    }
  }
  if (tidx < CtaTile::M) {
    uint32_t scale_reg = smem_q.load_q_scale(tidx, 0);
    uint8_t const* scale_bytes = reinterpret_cast<uint8_t const*>(&scale_reg);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      float scale = fmha::e4m3_byte_to_float(scale_bytes[s]);
      if (scale != 0.f) {
        tile[CtaTile::M * CtaTile::K + tidx * 4 + s] = scale;
      }
    }
  }
#endif
}

int main() {
  constexpr int q_len = 8;
  constexpr int num_q_heads = 8;
  constexpr int head_dim = 512;
  __nv_bfloat16* q = nullptr;
  int* cu_q = nullptr;
  int* cu_kv = nullptr;
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&q, q_len * num_q_heads * head_dim * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemset(q, 0, q_len * num_q_heads * head_dim * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&cu_q, 2 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&cu_kv, 2 * sizeof(int)));
  int h_cu_q[2] = {0, q_len};
  int h_cu_kv[2] = {0, 64};
  CUDA_CHECK(cudaMemcpy(cu_q, h_cu_q, sizeof(h_cu_q), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(cu_kv, h_cu_kv, sizeof(h_cu_kv), cudaMemcpyHostToDevice));

  int const rows_to_probe[] = {0, 7, 8, 15, 16, 23, 24, 31, 32, 39, 40, 47, 48, 55, 56, 63};
  int const cols_to_probe[] = {0, 5, 17, 63, 64, 127, 128, 191, 255, 256, 321, 384, 447, 511};
  constexpr int num_rows_to_probe = sizeof(rows_to_probe) / sizeof(rows_to_probe[0]);
  constexpr int num_cols_to_probe = sizeof(cols_to_probe) / sizeof(cols_to_probe[0]);
  size_t const per_case_elems = CtaTile::M * CtaTile::K + CtaTile::M * 4;
  size_t const out_elems =
      static_cast<size_t>(num_rows_to_probe) * num_cols_to_probe * per_case_elems;
  CUDA_CHECK(cudaMalloc(&out, out_elems * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, out_elems * sizeof(float)));

  __nv_bfloat16 one = __float2bfloat16(6.f);
  for (int row_idx = 0; row_idx < num_rows_to_probe; ++row_idx) {
    int const row = rows_to_probe[row_idx];
    for (int col_idx = 0; col_idx < num_cols_to_probe; ++col_idx) {
      int const col = cols_to_probe[col_idx];
      CUDA_CHECK(cudaMemcpy(q + (row / num_q_heads) * num_q_heads * head_dim +
                                (row % num_q_heads) * head_dim + col,
                            &one, sizeof(one), cudaMemcpyHostToDevice));
      bert::Fused_multihead_attention_params_v2 params{};
      params.q_ptr = q;
      params.q_stride_in_bytes = num_q_heads * head_dim * sizeof(__nv_bfloat16);
      params.cu_q_seqlens = cu_q;
      params.cu_kv_seqlens = cu_kv;
      params.h = num_q_heads;
      params.h_kv = 1;
      params.h_q_per_kv = num_q_heads;
      params.d = head_dim;
      params.dv = head_dim;
      params.num_grouped_heads = num_q_heads;
      probe_kernel<<<1, CtaTile::THREADS_PER_CTA, SmemQ::BYTES_PER_TILE>>>(
          params, row, col,
          out + (row_idx * num_cols_to_probe + col_idx) * per_case_elems);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      __nv_bfloat16 zero = __float2bfloat16(0.f);
      CUDA_CHECK(cudaMemcpy(q + (row / num_q_heads) * num_q_heads * head_dim +
                                (row % num_q_heads) * head_dim + col,
                            &zero, sizeof(zero), cudaMemcpyHostToDevice));
    }
  }

  float* host = new float[out_elems];
  CUDA_CHECK(cudaMemcpy(host, out, out_elems * sizeof(float), cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int row_idx = 0; row_idx < num_rows_to_probe; ++row_idx) {
    int const row = rows_to_probe[row_idx];
    for (int col_idx = 0; col_idx < num_cols_to_probe; ++col_idx) {
      int const col = cols_to_probe[col_idx];
      int const base = (row_idx * num_cols_to_probe + col_idx) * per_case_elems;
      float target = host[base + row * CtaTile::K + (col % CtaTile::K)];
      float other = 0.f;
      int other_row = -1, other_col = -1;
      for (int r = 0; r < CtaTile::M; ++r) {
        for (int c = 0; c < CtaTile::K; ++c) {
          if (r == row && c == (col % CtaTile::K)) {
            continue;
          }
          float v = host[base + r * CtaTile::K + c];
          float av = v < 0.f ? -v : v;
          if (av > other) {
            other = av;
            other_row = r;
            other_col = c;
          }
        }
      }
      bool is_bad = target < 5.9f || target > 6.1f || other > 0.1f;
      if (is_bad && bad < 80) {
        std::printf("src (%02d,%03d) target %.3f other %.3f at (%02d,%02d) BAD\n",
                    row, col, target, other, other_row, other_col);
      }
      bad += is_bad ? 1 : 0;
    }
  }
  std::printf("bad %d / %d\n", bad, num_rows_to_probe * num_cols_to_probe);

  delete[] host;
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(cu_kv));
  CUDA_CHECK(cudaFree(cu_q));
  CUDA_CHECK(cudaFree(q));
  return bad == 0 ? 0 : 1;
}
