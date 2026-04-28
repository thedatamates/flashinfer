/*
 * Probe the production paged-K NVFP4 global-memory -> shared-memory path used
 * by the Blackwell FMHAv2 D=512 grouped-M kernel.
 */

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
using GmemK = KernelTraits::Gmem_tile_k;
using SmemK = KernelTraits::Smem_tile_k;

__global__ void probe_kernel(bert::Fused_multihead_attention_params_v2 params,
                             int active_token, int active_col, float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  extern __shared__ char smem[];
  int const tidx = threadIdx.x;
  int const q_col_tile = active_col / CtaTile::K;

  auto binfo = fused_multihead_attention::Single_cta<2>(params, 0, 0, 0, tidx);
  GmemK gmem_k(params, 1, binfo, tidx);
  SmemK smem_k(smem, tidx);

#pragma unroll 1
  for (int tile = 0; tile < q_col_tile; ++tile) {
    gmem_k.move_col();
  }
  gmem_k.load(smem_k);
  __syncthreads();

  float* tile_out = out;
  for (int idx = tidx; idx < CtaTile::N * CtaTile::K; idx += CtaTile::THREADS_PER_CTA) {
    int const logical_row = idx / CtaTile::K;
    int const local_col = idx - logical_row * CtaTile::K;
    uint8_t byte = smem_k.load_k_data_byte(logical_row, local_col);
    uint32_t const code = (local_col & 1) ? ((byte >> 4) & 0x0fu) : (byte & 0x0fu);
    if (code != 0u) {
      tile_out[idx] = static_cast<float>(code);
    }
  }
  if (tidx < CtaTile::N) {
    int const token_row = tidx;
    int const scale_base = q_col_tile * (CtaTile::K / Traits::NVFP4_SCALE_VEC_SIZE);
#pragma unroll
    for (int s = 0; s < CtaTile::K / Traits::NVFP4_SCALE_VEC_SIZE; ++s) {
      uint8_t scale = smem_k.load_k_scale_byte(token_row, scale_base + s);
      tile_out[CtaTile::N * CtaTile::K + token_row * 4 + s] =
          static_cast<float>(scale);
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

  uint8_t* pool = nullptr;
  uint8_t* k_scale = nullptr;
  int32_t* block_offsets = nullptr;
  int* cu_q = nullptr;
  int* cu_kv = nullptr;
  float* out = nullptr;

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

  int const tokens_to_probe[] = {0, 7, 15, 16, 31, 32, 47, 63};
  int const cols_to_probe[] = {0, 5, 17, 63, 64, 127, 128, 191, 255, 256, 321, 384, 447, 511};
  constexpr int num_tokens_to_probe = sizeof(tokens_to_probe) / sizeof(tokens_to_probe[0]);
  constexpr int num_cols_to_probe = sizeof(cols_to_probe) / sizeof(cols_to_probe[0]);
  size_t const per_case_elems = CtaTile::N * CtaTile::K + CtaTile::N * 4;
  size_t const out_elems =
      static_cast<size_t>(num_tokens_to_probe) * num_cols_to_probe * per_case_elems;
  CUDA_CHECK(cudaMalloc(&out, out_elems * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, out_elems * sizeof(float)));

  bert::Fused_multihead_attention_params_v2 params{};
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

  for (int token_idx = 0; token_idx < num_tokens_to_probe; ++token_idx) {
    int const token = tokens_to_probe[token_idx];
    int const logical_page = token / tokens_per_block;
    int const row_in_page = token % tokens_per_block;
    int const physical_page = h_offsets[logical_page];
    for (int col_idx = 0; col_idx < num_cols_to_probe; ++col_idx) {
      int const col = cols_to_probe[col_idx];
      uint8_t packed = (col & 1) ? static_cast<uint8_t>(e2m1_code << 4) : e2m1_code;
      uint8_t* ptr = pool + physical_page * bytes_per_kv_block +
                     row_in_page * (head_dim / 2) + col / 2;
      CUDA_CHECK(cudaMemcpy(ptr, &packed, sizeof(packed), cudaMemcpyHostToDevice));
      probe_kernel<<<1, CtaTile::THREADS_PER_CTA, SmemK::BYTES_PER_TILE>>>(
          params, token, col,
          out + (token_idx * num_cols_to_probe + col_idx) * per_case_elems);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      uint8_t zero = 0u;
      CUDA_CHECK(cudaMemcpy(ptr, &zero, sizeof(zero), cudaMemcpyHostToDevice));
    }
  }

  float* host = new float[out_elems];
  CUDA_CHECK(cudaMemcpy(host, out, out_elems * sizeof(float), cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int token_idx = 0; token_idx < num_tokens_to_probe; ++token_idx) {
    int const token = tokens_to_probe[token_idx];
    for (int col_idx = 0; col_idx < num_cols_to_probe; ++col_idx) {
      int const col = cols_to_probe[col_idx];
      int const base = (token_idx * num_cols_to_probe + col_idx) * per_case_elems;
      int const local_col = col % CtaTile::K;
      float target = host[base + token * CtaTile::K + local_col];
      float other = 0.f;
      int other_row = -1, other_col = -1;
      for (int row = 0; row < CtaTile::N; ++row) {
        for (int c = 0; c < CtaTile::K; ++c) {
          if (row == token && c == local_col) {
            continue;
          }
          float v = host[base + row * CtaTile::K + c];
          if (v > other) {
            other = v;
            other_row = row;
            other_col = c;
          }
        }
      }
      bool scales_ok = true;
      int const q_col_tile = col / CtaTile::K;
      int const scale_base = q_col_tile * (CtaTile::K / Traits::NVFP4_SCALE_VEC_SIZE);
      for (int s = 0; s < CtaTile::K / Traits::NVFP4_SCALE_VEC_SIZE; ++s) {
        float scale = host[base + CtaTile::N * CtaTile::K + token * 4 + s];
        scales_ok &= static_cast<uint8_t>(scale) == e4m3_one;
      }
      bool is_bad = static_cast<uint8_t>(target) != e2m1_code || other != 0.f || !scales_ok;
      if (is_bad && bad < 80) {
        std::printf(
            "src (%02d,%03d) target %.0f other %.0f at (%02d,%02d) "
            "scale_base %d scales_ok %d BAD\n",
            token, col, target, other, other_row, other_col, scale_base,
            scales_ok ? 1 : 0);
      }
      bad += is_bad ? 1 : 0;
    }
  }
  std::printf("bad %d / %d\n", bad, num_tokens_to_probe * num_cols_to_probe);

  delete[] host;
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(cu_kv));
  CUDA_CHECK(cudaFree(cu_q));
  CUDA_CHECK(cudaFree(block_offsets));
  CUDA_CHECK(cudaFree(k_scale));
  CUDA_CHECK(cudaFree(pool));
  return bad == 0 ? 0 : 1;
}
