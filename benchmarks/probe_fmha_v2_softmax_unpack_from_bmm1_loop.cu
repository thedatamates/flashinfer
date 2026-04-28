/*
 * Probe Softmax::unpack() after the full production BMM1 D-tile loop.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <fmha/fragment.h>
#include <fmha/gemm.h>
#include <fmha/kernel_traits.h>
#include <fmha/smem_tile.h>
#include <fmha/softmax.h>
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
using Softmax = fmha::Softmax<Traits, CtaTile, KernelTraits>;

__global__ void probe_kernel(bert::Fused_multihead_attention_params_v2 params,
                             float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  extern __shared__ char smem[];
  int const tidx = threadIdx.x;

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
  gmem_k.load(smem_k);
  gmem_k.commit(smem_k);

  typename PersistentSmemQ::Fragment frag_q[MmaTile::MMAS_K][MmaTile::MMAS_M];
  typename SmemK::Fragment frag_k[MmaTile::MMAS_K][MmaTile::MMAS_N];
  fmha::Fragment_accumulator<Traits> acc[MmaTile::MMAS_M][MmaTile::MMAS_N];
  fmha::Clear_accumulator<typename Traits::Accumulator_type, CtaTile::WARPS_K>::apply(acc);

  constexpr int BMM1_VALID_MMAS_K = MmaTile::VALID_MMAS_K;
  constexpr int BMM1_TAIL_MMAS_K_BOUND =
      BMM1_VALID_MMAS_K % MmaTile::MMAS_K ? BMM1_VALID_MMAS_K % MmaTile::MMAS_K
                                          : MmaTile::MMAS_K;
  constexpr int BMM1_MAIN_MMAS_K_BOUND =
      BMM1_VALID_MMAS_K - BMM1_TAIL_MMAS_K_BOUND;

  for (int bmm1_k = 0; bmm1_k < BMM1_MAIN_MMAS_K_BOUND; bmm1_k += MmaTile::MMAS_K) {
    gmem_k.move_col();
    smem_k.move_to_next_write_buffer();
    gmem_k.load(smem_k);
    gmem_k.commit(smem_k);
    __syncthreads();
#pragma unroll
    for (int ki = 0; ki < MmaTile::MMAS_K; ++ki) {
      int const q_col_tile = bmm1_k + ki;
      PersistentSmemQ smem_q(
          &smem[q_col_tile * PersistentSmemQ::BYTES_PER_TILE], tidx);
      smem_q.load(frag_q[ki], 0);
      smem_k.load(frag_k[ki], ki);
      fmha::gemm(acc, frag_q[ki], frag_k[ki]);
    }
    __syncthreads();
    smem_k.move_to_next_read_buffer();
  }

  {
    __syncthreads();
#pragma unroll
    for (int ki = 0; ki < MmaTile::MMAS_K; ++ki) {
      int const q_col_tile = BMM1_MAIN_MMAS_K_BOUND + ki;
      PersistentSmemQ smem_q(
          &smem[q_col_tile * PersistentSmemQ::BYTES_PER_TILE], tidx);
      smem_q.load(frag_q[ki], 0);
      smem_k.load(frag_k[ki], ki);
      if (CtaTile::VALID_K % CtaTile::K == 0 || ki < BMM1_TAIL_MMAS_K_BOUND) {
        fmha::gemm(acc, frag_q[ki], frag_k[ki]);
      }
    }
  }

  Softmax softmax(params, &smem[KernelTraits::BYTES_PER_SMEM_Q], 0, tidx);
  softmax.unpack(acc);

  int const warp = tidx / CtaTile::THREADS_PER_WARP;
  int const warp_m = warp % CtaTile::WARPS_M;
  int const lane = tidx & 31;
  int const row_in_warp = lane >> 2;
#pragma unroll
  for (int row_slot = 0; row_slot < 2; ++row_slot) {
    int const local_row = row_in_warp + row_slot * 8;
    int const global_row = warp_m * 16 + local_row;
#pragma unroll
    for (int col = 0; col < CtaTile::N; ++col) {
      float const value = softmax.get_value(0, local_row, col);
      out[global_row * CtaTile::N + col] = value;
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
  int const cases[][2] = {
      {0, 0}, {7, 5}, {15, 17}, {16, 63},
      {31, 128}, {32, 255}, {47, 321}, {63, 511},
  };
  constexpr int num_cases = sizeof(cases) / sizeof(cases[0]);

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
  CUDA_CHECK(cudaMalloc(&out, CtaTile::M * CtaTile::N * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, CtaTile::M * CtaTile::N * sizeof(float)));

  __nv_bfloat16 q_value = __float2bfloat16(128.f);
  for (int q_token = 0; q_token < q_len; ++q_token) {
    int const col = cases[q_token][1];
    for (int q_head = 0; q_head < num_q_heads; ++q_head) {
      __nv_bfloat16* q_ptr =
          q + (q_token * num_q_heads + q_head) * head_dim + col;
      CUDA_CHECK(cudaMemcpy(q_ptr, &q_value, sizeof(q_value), cudaMemcpyHostToDevice));
    }
  }
  for (int i = 0; i < num_cases; ++i) {
    int const token = cases[i][0];
    int const col = cases[i][1];
    int const logical_page = token / tokens_per_block;
    int const row_in_page = token % tokens_per_block;
    int const physical_page = h_offsets[logical_page];
    uint8_t packed = (col & 1) ? static_cast<uint8_t>(e2m1_code << 4) : e2m1_code;
    uint8_t* k_ptr = pool + physical_page * bytes_per_kv_block +
                     row_in_page * (head_dim / 2) + col / 2;
    CUDA_CHECK(cudaMemcpy(k_ptr, &packed, sizeof(packed), cudaMemcpyHostToDevice));
  }

  float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
  uint32_t scale_bits = reinterpret_cast<uint32_t const&>(scale);
  bert::Fused_multihead_attention_params_v2 params{};
  params.scale_bmm1 = scale_bits;
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

  probe_kernel<<<1, CtaTile::THREADS_PER_CTA,
                 KernelTraits::BYTES_PER_SMEM_Q + SmemK::BYTES_PER_TILE + 4096>>>(params, out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float host[CtaTile::M * CtaTile::N];
  CUDA_CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int q_token = 0; q_token < q_len; ++q_token) {
    int const target = cases[q_token][0];
    for (int q_head = 0; q_head < num_q_heads; ++q_head) {
      int const row = q_token * num_q_heads + q_head;
      float target_val = host[row * CtaTile::N + target];
      float other = -1e30f;
      int other_col = -1;
      for (int col = 0; col < CtaTile::N; ++col) {
        if (col == target) {
          continue;
        }
        float const v = host[row * CtaTile::N + col];
        if (v > other) {
          other = v;
          other_col = col;
        }
      }
      bool const is_bad = target_val < 10.f || other > 1e-3f;
      if (is_bad) {
        std::printf("q_token %d head %d row %02d target %02d logit %.3f other %.3f at %02d BAD\n",
                    q_token, q_head, row, target, target_val, other, other_col);
      }
      bad += is_bad ? 1 : 0;
    }
  }
  std::printf("bad %d / %d\n", bad, q_len * num_q_heads);

  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(cu_kv));
  CUDA_CHECK(cudaFree(cu_q));
  CUDA_CHECK(cudaFree(block_offsets));
  CUDA_CHECK(cudaFree(k_scale));
  CUDA_CHECK(cudaFree(pool));
  CUDA_CHECK(cudaFree(q));
  return bad == 0 ? 0 : 1;
}
