/*
 * Probe BMM1 through the production persistent-Q and paged-K NVFP4 loaders.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <fmha/fragment.h>
#include <fmha/gemm.h>
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
using GmemK = KernelTraits::Gmem_tile_k;
using PersistentSmemQ = KernelTraits::Persistent_smem_tile_q;
using SmemK = KernelTraits::Smem_tile_k;

__global__ void probe_kernel(bert::Fused_multihead_attention_params_v2 params,
                             int active_row, int active_token, int active_col,
                             float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  extern __shared__ char smem[];
  int const tidx = threadIdx.x;
  int const q_col_tile = active_col / CtaTile::K;

  auto binfo = fused_multihead_attention::Single_cta<2>(params, 0, 0, 0, tidx);
  GmemQ gmem_q(params, 0, binfo, tidx, 0);
  GmemQ persistent_gmem_q = gmem_q;
#pragma unroll
  for (int tile = 0; tile < KernelTraits::PERSISTENT_Q_COL_TILES; ++tile) {
    PersistentSmemQ smem_q_tile(
        &smem[tile * PersistentSmemQ::BYTES_PER_TILE], tidx);
    persistent_gmem_q.load(smem_q_tile);
    persistent_gmem_q.commit(smem_q_tile);
    if (tile + 1 < KernelTraits::PERSISTENT_Q_COL_TILES) {
      persistent_gmem_q.move_col();
    }
  }

  GmemK gmem_k(params, 1, binfo, tidx);
  SmemK smem_k(&smem[KernelTraits::BYTES_PER_SMEM_Q], tidx);
#pragma unroll 1
  for (int tile = 0; tile < q_col_tile; ++tile) {
    gmem_k.move_col();
  }
  gmem_k.load(smem_k);
  gmem_k.commit(smem_k);
  __syncthreads();

  PersistentSmemQ smem_q(
      &smem[q_col_tile * PersistentSmemQ::BYTES_PER_TILE], tidx);
  typename PersistentSmemQ::Fragment frag_q[MmaTile::MMAS_M];
  typename SmemK::Fragment frag_k[MmaTile::MMAS_N];
  smem_q.load(frag_q, 0);
  smem_k.load(frag_k, 0);

  fmha::Fragment_accumulator<Traits> acc[MmaTile::MMAS_M][MmaTile::MMAS_N];
  fmha::Clear_accumulator<typename Traits::Accumulator_type, CtaTile::WARPS_K>::apply(acc);
  fmha::gemm(acc, frag_q, frag_k);

  int const lane = tidx & 31;
  int const warp = tidx / CtaTile::THREADS_PER_WARP;
  int const warp_m = warp % CtaTile::WARPS_M;
  float* tile_out = out;
#pragma unroll
  for (int mi = 0; mi < MmaTile::MMAS_M; ++mi) {
#pragma unroll
    for (int ni = 0; ni < MmaTile::MMAS_N; ++ni) {
#pragma unroll
      for (int elem = 0; elem < 8; ++elem) {
        int const row = (mi * CtaTile::WARPS_M + warp_m) *
                            MmaTile::M_PER_MMA +
                        (lane >> 2) + 8 * ((elem % 4) / 2);
        int const col = ni * MmaTile::N_PER_MMA_PER_CTA +
                        2 * (lane & 3) + (elem & 1) + 8 * (elem / 4);
        tile_out[row * CtaTile::N + col] = acc[mi][ni].elt(elem);
      }
    }
  }
#endif
}

int main() {
  constexpr int q_len = 8;
  constexpr int kv_len = 64;
  constexpr int num_q_heads = 8;
  constexpr int num_kv_heads = 1;
  constexpr int head_dim = 512;
  constexpr int tokens_per_block = 16;
  constexpr int max_blocks_per_seq = kv_len / tokens_per_block;
  constexpr int bytes_per_kv_block =
      tokens_per_block * num_kv_heads * (head_dim / 2);
  constexpr int scale_groups = head_dim / Traits::NVFP4_SCALE_VEC_SIZE;
  constexpr int scale_page_stride = tokens_per_block * num_kv_heads * scale_groups;
  constexpr uint8_t e2m1_code = 0x6u;
  constexpr uint8_t e4m3_one = 0x38u;

  __nv_bfloat16* q = nullptr;
  uint8_t* pool = nullptr;
  uint8_t* k_scale = nullptr;
  int32_t* block_offsets = nullptr;
  int* cu_q = nullptr;
  int* cu_kv = nullptr;
  float* out = nullptr;

  CUDA_CHECK(cudaMalloc(&q, q_len * num_q_heads * head_dim * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemset(q, 0, q_len * num_q_heads * head_dim * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&pool, 2 * max_blocks_per_seq * bytes_per_kv_block));
  CUDA_CHECK(cudaMemset(pool, 0, 2 * max_blocks_per_seq * bytes_per_kv_block));
  CUDA_CHECK(cudaMalloc(&k_scale, max_blocks_per_seq * scale_page_stride));
  CUDA_CHECK(cudaMemset(k_scale, e4m3_one, max_blocks_per_seq * scale_page_stride));
  CUDA_CHECK(cudaMalloc(&block_offsets, 2 * max_blocks_per_seq * sizeof(int32_t)));
  int32_t h_offsets[2 * max_blocks_per_seq] = {0, 2, 4, 6, 1, 3, 5, 7};
  CUDA_CHECK(cudaMemcpy(block_offsets, h_offsets, sizeof(h_offsets), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&cu_q, 2 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&cu_kv, 2 * sizeof(int)));
  int h_cu_q[2] = {0, q_len};
  int h_cu_kv[2] = {0, kv_len};
  CUDA_CHECK(cudaMemcpy(cu_q, h_cu_q, sizeof(h_cu_q), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(cu_kv, h_cu_kv, sizeof(h_cu_kv), cudaMemcpyHostToDevice));

  int const rows_to_probe[] = {0, 7, 8, 15, 16, 23, 24, 31,
                               32, 39, 40, 47, 48, 55, 56, 63};
  int const tokens_to_probe[] = {0, 7, 15, 16, 31, 32, 47, 63};
  int const cols_to_probe[] = {0, 5, 17, 63, 64, 127, 128, 191,
                               255, 256, 321, 384, 447, 511};
  constexpr int num_rows = sizeof(rows_to_probe) / sizeof(rows_to_probe[0]);
  constexpr int num_tokens = sizeof(tokens_to_probe) / sizeof(tokens_to_probe[0]);
  constexpr int num_cols = sizeof(cols_to_probe) / sizeof(cols_to_probe[0]);
  size_t const per_case_elems = CtaTile::M * CtaTile::N;
  size_t const out_elems =
      static_cast<size_t>(num_rows) * num_tokens * num_cols * per_case_elems;
  CUDA_CHECK(cudaMalloc(&out, out_elems * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, out_elems * sizeof(float)));

  bert::Fused_multihead_attention_params_v2 params{};
  params.q_ptr = q;
  params.q_stride_in_bytes = num_q_heads * head_dim * sizeof(__nv_bfloat16);
  params.cu_q_seqlens = cu_q;
  params.cu_kv_seqlens = cu_kv;
  params.h = num_q_heads;
  params.h_kv = num_kv_heads;
  params.h_q_per_kv = num_q_heads / num_kv_heads;
  params.d = head_dim;
  params.dv = head_dim;
  params.num_grouped_heads = params.h_q_per_kv;
  params.k_stride_in_bytes = head_dim / 2;
  params.k_stride_in_bytes_2 = head_dim / 2;
  params.k_scale_ptr = k_scale;
  params.k_scale_page_stride_in_bytes = scale_page_stride;
  params.k_scale_head_stride_in_bytes = tokens_per_block * scale_groups;
  params.k_scale_token_stride_in_bytes = scale_groups;
  params.k_scale_vec_stride_in_bytes = 1;
  params.paged_kv_cache =
      fmha::Kv_block_array(1, max_blocks_per_seq, tokens_per_block, bytes_per_kv_block, pool);
  params.paged_kv_cache.mBlockOffsets = block_offsets;

  __nv_bfloat16 q_value = __float2bfloat16(128.f);
  for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
    int const row = rows_to_probe[row_idx];
    int const q_token = row / num_q_heads;
    int const q_head = row % num_q_heads;
    for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
      int const token = tokens_to_probe[token_idx];
      int const logical_page = token / tokens_per_block;
      int const row_in_page = token % tokens_per_block;
      int const physical_page = h_offsets[logical_page];
      for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
        int const col = cols_to_probe[col_idx];
        __nv_bfloat16* q_ptr =
            q + (q_token * num_q_heads + q_head) * head_dim + col;
        CUDA_CHECK(cudaMemcpy(q_ptr, &q_value, sizeof(q_value), cudaMemcpyHostToDevice));
        uint8_t packed = (col & 1) ? static_cast<uint8_t>(e2m1_code << 4) : e2m1_code;
        uint8_t* k_ptr = pool + physical_page * bytes_per_kv_block +
                         row_in_page * (head_dim / 2) + col / 2;
        CUDA_CHECK(cudaMemcpy(k_ptr, &packed, sizeof(packed), cudaMemcpyHostToDevice));

        size_t const case_idx =
            (static_cast<size_t>(row_idx) * num_tokens + token_idx) * num_cols + col_idx;
        probe_kernel<<<1, CtaTile::THREADS_PER_CTA,
                       KernelTraits::BYTES_PER_SMEM_Q + SmemK::BYTES_PER_TILE>>>(
            params, row, token, col, out + case_idx * per_case_elems);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        __nv_bfloat16 q_zero = __float2bfloat16(0.f);
        uint8_t k_zero = 0u;
        CUDA_CHECK(cudaMemcpy(q_ptr, &q_zero, sizeof(q_zero), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(k_ptr, &k_zero, sizeof(k_zero), cudaMemcpyHostToDevice));
      }
    }
  }

  float* host = new float[out_elems];
  CUDA_CHECK(cudaMemcpy(host, out, out_elems * sizeof(float), cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
    int const row = rows_to_probe[row_idx];
    for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
      int const token = tokens_to_probe[token_idx];
      for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
        int const col = cols_to_probe[col_idx];
        size_t const case_idx =
            (static_cast<size_t>(row_idx) * num_tokens + token_idx) * num_cols + col_idx;
        float const* tile = host + case_idx * per_case_elems;
        float target = tile[row * CtaTile::N + token];
        float other = 0.f;
        int other_row = -1, other_col = -1;
        for (int r = 0; r < CtaTile::M; ++r) {
          for (int c = 0; c < CtaTile::N; ++c) {
            if (r == row && c == token) {
              continue;
            }
            float const v = tile[r * CtaTile::N + c];
            float const av = v < 0.f ? -v : v;
            if (av > other) {
              other = av;
              other_row = r;
              other_col = c;
            }
          }
        }
        bool const is_bad = target < 100.f || other > 1e-3f;
        if (is_bad && bad < 120) {
          std::printf(
              "row %02d token %02d col %03d target %.3f other %.3f at "
              "(%02d,%02d) BAD\n",
              row, token, col, target, other, other_row, other_col);
        }
        bad += is_bad ? 1 : 0;
      }
    }
  }
  std::printf("bad %d / %d\n", bad, num_rows * num_tokens * num_cols);

  delete[] host;
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(cu_kv));
  CUDA_CHECK(cudaFree(cu_q));
  CUDA_CHECK(cudaFree(block_offsets));
  CUDA_CHECK(cudaFree(k_scale));
  CUDA_CHECK(cudaFree(pool));
  CUDA_CHECK(cudaFree(q));
  return bad == 0 ? 0 : 1;
}
