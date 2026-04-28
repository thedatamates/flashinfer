/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <fmha/gemm.h>
#include <fmha/kernel_traits.h>
#include <fmha/utils.h>
#include <fused_multihead_attention_kernel.h>

namespace fused_multihead_attention {

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_KV_STAGING)
#define FLASHINFER_FMHA_V2_PROFILE_SKIP_K_STAGING
#define FLASHINFER_FMHA_V2_PROFILE_SKIP_V_STAGING
#endif
#if defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_K_STAGING)
#define FLASHINFER_FMHA_V2_PROFILE_SKIP_K_GMEM_STAGING
#define FLASHINFER_FMHA_V2_PROFILE_SKIP_K_SMEM_LOAD
#endif
#if defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_V_STAGING)
#define FLASHINFER_FMHA_V2_PROFILE_SKIP_V_GMEM_STAGING
#define FLASHINFER_FMHA_V2_PROFILE_SKIP_V_SMEM_LOAD
#endif

template <typename Kernel_traits, typename Params>
inline __device__ void device_flash_attention_nl_tiled(Params const& params) {
  // The instruction traits.
  using Traits_p = typename Kernel_traits::Traits_p;
  using Traits_o = typename Kernel_traits::Traits_o;

  // The description of the CTA tile for the 1st batched GEMM.
  using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
  // The description of the CTA tile for the 2nd batched GEMM.
  using Cta_tile_o = typename Kernel_traits::Cta_tile_o;

  // The MMA tile for the 1st GEMM.
  using Mma_tile_p = typename Traits_p::template Mma_tile<Cta_tile_p>;
  // The MMA tile for the 2nd GEMM.
  using Mma_tile_o = typename Traits_o::template Mma_tile<Cta_tile_o>;

  // The global memory tile to load Q.
  using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;
  // The shared memory tile to swizzle Q.
  using Smem_tile_q = typename Kernel_traits::Smem_tile_q;
  using Persistent_smem_tile_q = typename Kernel_traits::Persistent_smem_tile_q;

  // The global memory tile to load K.
  using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;
  // The shared memory tile to swizzle K.
  using Smem_tile_k = typename Kernel_traits::Smem_tile_k;

  // The global memory tile to load V.
  using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;
  // The shared memory tile to swizzle V.
  using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

  // The global memory tile to store O.
  using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;
  // The shared memory tile to swizzle O.
  using Smem_tile_o = typename Kernel_traits::Smem_tile_o;

  // Do we use LDGSTS for Q, K or V?
  enum { USE_LDGSTS_Q = Kernel_traits::USE_LDGSTS_Q };

  enum { USE_LDGSTS_K = Kernel_traits::USE_LDGSTS_K };

  enum { USE_LDGSTS_V = Kernel_traits::USE_LDGSTS_V };

  // Do we use LDGSTS for any of the 3 input matrices.
  enum { USE_LDGSTS = USE_LDGSTS_Q || USE_LDGSTS_K || USE_LDGSTS_V };

  // TODO ANT: assertions
  static_assert(USE_LDGSTS, "Supports only USE_LDGSTS = true");
  static_assert(!Kernel_traits::SHARE_SMEM_FOR_K_AND_V, "");

  // If either K or V uses LDGSTS, they cannot share a buffer.
  static_assert(!(USE_LDGSTS_K || USE_LDGSTS_V) || !Kernel_traits::SHARE_SMEM_FOR_K_AND_V, "");

  // Fragment double buffer (reduce register pressure)
  enum {
    FRAGMENT_QK_SIZE_IN_K_DIM = (Kernel_traits::LIMIT_QK_FRAGMENTS) * 2 +
                                !(Kernel_traits::LIMIT_QK_FRAGMENTS)*Mma_tile_p::MMAS_K
  };

  static_assert(!(Kernel_traits::SHARE_SMEM_FOR_K_AND_V &&
                  (Kernel_traits::LIMIT_QK_FRAGMENTS || Kernel_traits::LIMIT_V_FRAGMENTS)),
                "");

  enum {
    FRAGMENT_V_SIZE_IN_K_DIM = (Kernel_traits::LIMIT_V_FRAGMENTS) * 2 +
                               !(Kernel_traits::LIMIT_V_FRAGMENTS)*Mma_tile_o::MMAS_K
  };

  // Do we need to check if there are negative inf for softmax row_max ?
  enum { CHECK_NEG_INF = Kernel_traits::SLIDING_WINDOW_ATTENTION || Kernel_traits::CUSTOM_MASK };

  // Shared memory.
  extern __shared__ char smem_[];

  // The loop -- each CTA works on a different loop iteration.
  int const ctas_per_o_row = (Kernel_traits::VALID_DV + Cta_tile_o::N - 1) / Cta_tile_o::N;
  int const q_loop = blockIdx.x / ctas_per_o_row;
  // Each CTA works on a specific row partition of O.
  int const o_part = blockIdx.x % ctas_per_o_row;
  // The block index for the batch. Optional split-KV mode expands grid.z by
  // num_kv_splits and writes one partial state per split.
  int const num_kv_splits = params.num_kv_splits > 0 ? params.num_kv_splits : 1;
  int const bidb = blockIdx.z / num_kv_splits;
  int const kv_split_idx = blockIdx.z - bidb * num_kv_splits;
  // The block index for the head.
  int const bidh = blockIdx.y;
  // The thread index.
  int const tidx = threadIdx.x;

  // The block info.
  Single_cta<Kernel_traits::VERSION> const binfo(params, bidb, bidh, 0, tidx);
  constexpr bool group_q_heads_in_m = Kernel_traits::GROUP_Q_HEADS_IN_M;
  constexpr bool grouped_m_rows = Kernel_traits::IS_MTP || Kernel_traits::GROUP_Q_HEADS_IN_M;
  int const q_row_loop = q_loop * Gmem_tile_q::ROWS;
  int const q_rows =
      group_q_heads_in_m ? binfo.actual_q_seqlen * params.num_grouped_heads
                         : binfo.actual_q_seqlen;
  // The local sequence offset of Q.
  int q_sequence_start =
      grouped_m_rows ? q_row_loop / params.num_grouped_heads : q_row_loop;
  // Consider the past sequence length.
  q_sequence_start += binfo.actual_kv_seqlen - binfo.actual_q_seqlen;
  if (q_row_loop >= q_rows) {
    return;
  }

#if defined(FLASHINFER_FMHA_V2_PROFILE_RETURN_AFTER_ROW_CHECK)
  return;
#endif

  // Create the object to control the masks.
  fmha::Mask_dispatcher<Traits_p, Cta_tile_p, Kernel_traits::MASK_VERSION, Kernel_traits::IS_MTP,
                        Kernel_traits::GROUP_Q_HEADS_IN_M>
      mask(params, binfo, tidx);

  // Allocate the global memory tile loader for Q.
  Gmem_tile_q gmem_q(params, 0, binfo, tidx, q_row_loop);
  // Allocate the shared memory tile loader for Q.
  Smem_tile_q smem_q(&smem_[0], tidx);

  // Allocate the global memory tile loader for K.
  Gmem_tile_k gmem_k(params, 1, binfo, tidx);
  // Allocate the shared memory tile loader for K.
  Smem_tile_k smem_k(&smem_[Kernel_traits::BYTES_PER_SMEM_Q], tidx);

  // Allocate the global memory tile loader for V.
  enum { GMEM_V_CTA_COL_BYTES = (Cta_tile_o::N * Traits_o::BITS_PER_ELEMENT_B + 7) / 8 };
  Gmem_tile_v gmem_v(params, 2, binfo, tidx, 0,
                     o_part * GMEM_V_CTA_COL_BYTES);
  // Allocate the shared memory tile loader for V.
  Smem_tile_v smem_v(&smem_[Kernel_traits::BYTES_PER_SMEM_QK], tidx);

  // With chunked attention, the q_start_seqlen might not be multiple of Cta_tile_p::M.
  int const kv_mask_loop_start = int(q_sequence_start / Cta_tile_p::N) * Cta_tile_p::N;

  // The start/end step of kv loops.
  // Do we need to mask out the tokens that is not in the sliding window.
  bool const mask_sliding_window = Kernel_traits::SLIDING_WINDOW_ATTENTION &&
                                   binfo.actual_kv_seqlen > params.sliding_window_size;
  int const q_tile_tokens =
      grouped_m_rows ? (Cta_tile_p::M + params.num_grouped_heads - 1) /
                           params.num_grouped_heads
                     : Cta_tile_p::M;
  int const valid_seqlen = Kernel_traits::CAUSAL_MASK
                               ? min(q_sequence_start + q_tile_tokens, binfo.actual_kv_seqlen)
                               : binfo.actual_kv_seqlen;

  int kv_loop_end = ((valid_seqlen + Cta_tile_p::N - 1) / Cta_tile_p::N) * Cta_tile_p::N;
  int kv_loop_start =
      mask_sliding_window
          ? (max(0, q_sequence_start + 1 - params.sliding_window_size) / Cta_tile_p::N) *
                Cta_tile_p::N
          : 0;
  if (num_kv_splits > 1) {
    int const split_start = kv_split_idx * params.kv_split_size;
    int const split_end = min(split_start + params.kv_split_size, binfo.actual_kv_seqlen);
    kv_loop_start = max(kv_loop_start, split_start);
    kv_loop_end = min(kv_loop_end, split_end);
  }
  int const sliding_window_mask_end =
      mask_sliding_window ? (max(0, q_sequence_start + q_tile_tokens - params.sliding_window_size) /
                             Cta_tile_p::N) *
                                Cta_tile_p::N
                          : 0;

  // Move K and V tiles.
  // We need offset here since we split single k loops into finer granularity.
  gmem_k.move_by_offset(kv_loop_start);
  gmem_v.move_by_offset(kv_loop_start);

  if constexpr (Kernel_traits::USE_PERSISTENT_Q) {
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_Q_STAGING)
    static_assert(Mma_tile_p::MMAS_K == 1, "");
    static_assert(Kernel_traits::PERSISTENT_Q_COL_TILES == Mma_tile_p::VALID_MMAS_K, "");
    Gmem_tile_q persistent_gmem_q = gmem_q;
#pragma unroll
    for (int q_col_tile = 0; q_col_tile < Kernel_traits::PERSISTENT_Q_COL_TILES; ++q_col_tile) {
      Persistent_smem_tile_q persistent_smem_q(
          &smem_[q_col_tile * Persistent_smem_tile_q::BYTES_PER_TILE], tidx);
      persistent_gmem_q.load(persistent_smem_q);
      persistent_gmem_q.commit(persistent_smem_q);
      if (q_col_tile + 1 < Kernel_traits::PERSISTENT_Q_COL_TILES) {
        persistent_gmem_q.move_col();
      }
    }
#endif
  } else {
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_Q_STAGING)
    // Trigger the loads for Q.
    gmem_q.load(smem_q);
#endif
  }
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_K_GMEM_STAGING)
  // Trigger the loads for K.
  gmem_k.load(smem_k);
#endif

  // Push the LDGDEPBAR instruction after the loads for Q, K
  fmha::ldgdepbar<USE_LDGSTS>();

  // Commit the data for Q and K to shared memory.
  if constexpr (!Kernel_traits::USE_PERSISTENT_Q) {
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_Q_STAGING)
    gmem_q.commit(smem_q);
#endif
  }
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_K_GMEM_STAGING)
  gmem_k.commit(smem_k);
#endif

  // Declare the fragments for Q/K/V.
  typename Smem_tile_q::Fragment frag_q[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_M];
  typename Smem_tile_k::Fragment frag_k[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_N];
  typename Smem_tile_v::Fragment frag_v[Mma_tile_o::MMAS_K][Mma_tile_o::VALID_MMAS_N];

  // Store/load P to/from memory (for debugging).
#if defined(STORE_P)
  enum { BITS_PER_ELT_P = sizeof(typename Traits_p::Accumulator_type) * 8 };

  using Gmem_tile_p = fmha::Gmem_tile_p<Traits_p, Cta_tile_p, BITS_PER_ELT_P>;
  Gmem_tile_p gmem_p(params.p_ptr, params.p_stride_in_bytes, params.scale_bmm1, tidx,
                     Kernel_traits::NO_LOOP ? q_loop * Cta_tile_p::M : 0);
#endif

  // Store S to memory (for debugging). NOTE: We use A_type as C_type is int32 for IMMA???
#if defined(STORE_S)
  enum { BITS_PER_ELT_S = sizeof(typename Traits_p::A_type) * 8 };

  using Gmem_tile_s = fmha::Gmem_tile_s<Traits_p, Cta_tile_p, BITS_PER_ELT_S>;
  Gmem_tile_s gmem_s(params.s_ptr, params.s_stride_in_bytes, params.scale_softmax, tidx,
                     Kernel_traits::NO_LOOP ? q_loop * Cta_tile_p::M : 0);
#endif

  // Create the object to do the softmax.
  using Softmax = fmha::Softmax<Traits_p, Cta_tile_p, Kernel_traits>;
  Softmax softmax(params, &smem_[Kernel_traits::BYTES_PER_SMEM_Q], bidb, tidx);

  // Prefetch next kv buffer to share memory
  enum {
    PREFETCH_K_BUFFER_TO_SMEM = !Kernel_traits::LIMIT_QK_FRAGMENTS && !Softmax::USE_SHARED_MEMORY
  };

  enum { PREFETCH_V_BUFFER_TO_SMEM = !Kernel_traits::SHARE_SMEM_FOR_K_AND_V };

  enum { PREFETCH_KV_BUFFER_TO_SMEM = PREFETCH_K_BUFFER_TO_SMEM && PREFETCH_V_BUFFER_TO_SMEM };

  // The number of threads per row.
  enum { THREADS_PER_ROW = 32 };

  // Load the mask for that iteration.
  mask.load(Kernel_traits::CUSTOM_MASK || Kernel_traits::IS_MTP ||
                    Kernel_traits::GROUP_Q_HEADS_IN_M
                ? q_row_loop
                : q_sequence_start);

  // Declare the accumulators for the 2nd gemm.
  fmha::Fragment_accumulator<Traits_o> acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::VALID_MMAS_N];
  using Acc_type_o = typename Traits_o::Accumulator_type;
  fmha::Clear_accumulator<Acc_type_o, Cta_tile_o::WARPS_K>::apply(acc_o);

  // Flash attention updater
  fmha::Tile_o_normalizer<Traits_o, Cta_tile_o> acc_o_normalizer(params, binfo);
  float global_max[Softmax::ROWS_PER_THREAD];
  float global_sum[Softmax::ROWS_PER_THREAD];
#if defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_SOFTMAX)
#pragma unroll
  for (int i = 0; i < Softmax::ROWS_PER_THREAD; ++i) {
    global_max[i] = 0.f;
    global_sum[i] = 1.f;
  }
#endif

  // BMM1_MAIN_MMAS_K_BOUND is the number of MMAs in the K dimension to execute in the loop and
  // BMM1_TAIL_MMAS_K_BOUND is the number of MMAs in the K dimension in the residual of the loop.
  //
  // When VALID_MMAS_K is a multiple of MMAS_K, we compute:
  //
  //   BMM1_MAIN_MMAS_K_BOUND = VALID_MMAS_K - MMAS_K
  //   BMM1_TAIL_MMAS_K_BOUND = MMAS_K
  //
  // When VALID_MASK_K is not a multiple of MMAS_K, we compute:
  //
  //   BMM1_MAIN_MMAS_K_BOUND = VALID_MMAS_K / MMAS_K * MMAS_K
  //   BMM1_TAIL_MMAS_K_BOUND = VALID_MMAS_K % MMAS_K
  constexpr int BMM1_VALID_MMAS_K = Mma_tile_p::VALID_MMAS_K;
  constexpr int BMM1_TAIL_MMAS_K_BOUND = BMM1_VALID_MMAS_K % Mma_tile_p::MMAS_K
                                             ? BMM1_VALID_MMAS_K % Mma_tile_p::MMAS_K
                                             : Mma_tile_p::MMAS_K;
  constexpr int BMM1_MAIN_MMAS_K_BOUND = BMM1_VALID_MMAS_K - BMM1_TAIL_MMAS_K_BOUND;

  // BMM2_MAIN_MMAS_K_BOUND is the number of MMAs in the K dimension to execute in the loop and
  // BMM2_TAIL_MMAS_K_BOUND is the number of MMAs in the K dimension in the residual of the loop.
  constexpr int BMM2_TAIL_MMAS_K_BOUND = Mma_tile_o::MMAS_K;
  constexpr int BMM2_MAIN_MMAS_K_BOUND = Kernel_traits::TOTAL_BMM2_MMAS_K - BMM2_TAIL_MMAS_K_BOUND;

  for (int kv_loop = kv_loop_start; kv_loop < kv_loop_end; kv_loop += Cta_tile_p::N) {
    bool const first_step = (kv_loop == kv_loop_start);
    // It is possible that all tokens are masked out (sliding-window-attention).
    bool const apply_sliding_window_mask =
        (mask_sliding_window && kv_loop <= sliding_window_mask_end);
    bool const apply_mask =
        params.has_alibi || (kv_loop >= kv_mask_loop_start) || apply_sliding_window_mask;

    // Declare the accumulators for the 1st gemm.
    fmha::Fragment_accumulator<Traits_p> acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
    using Acc_type_p = typename Traits_p::Accumulator_type;
    fmha::Clear_accumulator<Acc_type_p, Cta_tile_p::WARPS_K>::apply(acc_p);

    // Move mask offset.
    // Pre-load packed mask if it has.
    mask.move_to_offset(kv_loop);

    // BMM1 main loop
    // MMAS_K is the padded D, VALID_MMAS_K is D length
    // now MMAS_K is tiling size in D, VALID_MMAS_K is D length
    for (int bmm1_k = 0; bmm1_k < BMM1_MAIN_MMAS_K_BOUND; bmm1_k += Mma_tile_p::MMAS_K) {
      // Trigger the load for the next Q/K values
      if constexpr (!Kernel_traits::USE_PERSISTENT_Q) {
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_Q_STAGING)
        if (Kernel_traits::RELOAD_Q || first_step) {
          gmem_q.move_col();
          smem_q.move_to_next_write_buffer();
          gmem_q.load(smem_q);
        }
#endif
      }

#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_K_GMEM_STAGING)
      gmem_k.move_col();
      smem_k.move_to_next_write_buffer();
      gmem_k.load(smem_k);
#endif

      // Push the LDGDEPBAR instruction after the loads for QK.
      fmha::ldgdepbar<USE_LDGSTS>();

      if constexpr (!Kernel_traits::USE_PERSISTENT_Q) {
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_Q_STAGING)
        if (Kernel_traits::RELOAD_Q || first_step) {
          gmem_q.commit(smem_q);
        }
#endif
      }

#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_K_GMEM_STAGING)
      gmem_k.commit(smem_k);
#endif

      // Double SMEM buffer: make sure we are done writing the data from last stage (smem).
      // Leave 1 outstanding batch loadings
      fmha::depbar_<USE_LDGSTS, 1>();

// Do this part of P^T = (Q * K^T)^T.
#pragma unroll
      for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki) {
        // Load the fragments for Q/K inner tile
        if constexpr (Kernel_traits::USE_PERSISTENT_Q) {
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_Q_STAGING)
          int const q_col_tile = bmm1_k + ki;
          Persistent_smem_tile_q persistent_smem_q(
              &smem_[q_col_tile * Persistent_smem_tile_q::BYTES_PER_TILE], tidx);
          persistent_smem_q.load(frag_q[ki], 0);
#endif
        } else {
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_Q_STAGING)
          smem_q.load(frag_q[ki], ki);
#endif
        }
        if constexpr (!Kernel_traits::GROUP_Q_HEADS_IN_M) {
          smem_k.load(frag_k[ki], ki);
        } else {
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_K_SMEM_LOAD)
          smem_k.load(frag_k[ki], ki);
#endif
        }

        // Do the math for the values already in registers.
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_BMM1_GEMM)
        fmha::gemm(acc_p, frag_q[ki], frag_k[ki]);
#endif
      }

      // Make sure we are done reading the data (smem).
      __syncthreads();

      if constexpr (!Kernel_traits::USE_PERSISTENT_Q) {
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_Q_STAGING)
        smem_q.move_to_next_read_buffer();
#endif
      }
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_K_SMEM_LOAD)
      smem_k.move_to_next_read_buffer();
#endif
    }

    // BMM1 tail loop
    {
      // Trigger the load for next V values
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_V_GMEM_STAGING)
      if (kv_loop > kv_loop_start) {
        gmem_v.move();
        smem_v.move_to_next_write_buffer();
      }
      gmem_v.load(smem_v);
#endif

      fmha::ldgdepbar<USE_LDGSTS>();

#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_V_GMEM_STAGING)
      gmem_v.commit(smem_v);
#endif

      // Wait for Q/K to finish loading
      fmha::depbar_<USE_LDGSTS, 1>();

// Do this part of P^T = (Q * K^T)^T.
#pragma unroll
      for (int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki) {
        // Load the fragments for Q/K inner tile
        if constexpr (Kernel_traits::USE_PERSISTENT_Q) {
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_Q_STAGING)
          int const q_col_tile = BMM1_MAIN_MMAS_K_BOUND + ki;
          Persistent_smem_tile_q persistent_smem_q(
              &smem_[q_col_tile * Persistent_smem_tile_q::BYTES_PER_TILE], tidx);
          persistent_smem_q.load(frag_q[ki], 0);
#endif
        } else {
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_Q_STAGING)
          smem_q.load(frag_q[ki], ki);
#endif
        }
        if constexpr (!Kernel_traits::GROUP_Q_HEADS_IN_M) {
          smem_k.load(frag_k[ki], ki);
        } else {
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_K_SMEM_LOAD)
          smem_k.load(frag_k[ki], ki);
#endif
        }

        // Do the math for the values already in registers.
        if (Cta_tile_p::VALID_K % Cta_tile_p::K == 0 || ki < BMM1_TAIL_MMAS_K_BOUND) {
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_BMM1_GEMM)
          fmha::gemm(acc_p, frag_q[ki], frag_k[ki]);
#endif
        }
      }
      if constexpr (!Kernel_traits::USE_PERSISTENT_Q) {
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_Q_STAGING)
        smem_q.move_to_next_read_buffer();
#endif
      }
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_K_SMEM_LOAD)
      smem_k.move_to_next_read_buffer();
#endif
    }  // end BMM1 tail loop

    // Store the P matrix.
#if defined(STORE_P)
    gmem_p.store(acc_p);
#endif

#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_SOFTMAX)
    // Convert from the accumulator type to FP32 for Softmax.
#if defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_SOFTMAX_UNPACK)
    softmax.clear();
#else
    softmax.unpack(acc_p);
#endif

    // Apply the mask.
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_SOFTMAX_MASK)
    if (apply_mask) {
      if (params.has_alibi) {
        softmax.apply_mask_alibi(mask, bidh, params.alibi_params);
      } else {
        softmax.apply_mask(mask);
      }
    }
#endif

    // Make sure we are done reading the data (smem_v).
    if ((Kernel_traits::SHARE_SMEM_FOR_K_AND_V || Kernel_traits::LIMIT_QK_FRAGMENTS) &&
        Softmax::USE_SHARED_MEMORY) {
      __syncthreads();
    }

    // First step of the flash attention
    if (first_step) {
      // Compute the max.
#if defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_SOFTMAX_REDUCE_MAX)
#pragma unroll
      for (int i = 0; i < Softmax::ROWS_PER_THREAD; ++i) {
        global_max[i] = 0.f;
      }
#else
      softmax.template reduce<fmha::Max_>(global_max);
#endif

      if (Softmax::USE_SHARED_MEMORY) {
        // Make sure we are done reading shared memory.
        __syncthreads();
      }

      // It is possible that all elts are -FLT_MAX with sliding_window_causal or custom_mask.
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_SOFTMAX_EXP)
      softmax.template apply_exp_with_mask<CHECK_NEG_INF>(global_max);
#endif

      // Compute the sum.
#if defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_SOFTMAX_REDUCE_SUM)
#pragma unroll
      for (int i = 0; i < Softmax::ROWS_PER_THREAD; ++i) {
        global_sum[i] = 1.f;
      }
#else
      softmax.template reduce<fmha::Sum_>(global_sum);
#endif
    } else {
      float tmp[Softmax::ROWS_PER_THREAD];

#pragma unroll
      for (int i = 0; i < Softmax::ROWS_PER_THREAD; i++) {
        tmp[i] = global_max[i];
      }

      // Compute the max.
#if defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_SOFTMAX_REDUCE_MAX)
#pragma unroll
      for (int i = 0; i < Softmax::ROWS_PER_THREAD; ++i) {
        global_max[i] = 0.f;
      }
#else
      softmax.template reduce<fmha::Max_>(global_max);
#endif

      if (Softmax::USE_SHARED_MEMORY) {
        // Make sure we are done reading shared memory.
        __syncthreads();
      }

      // Update last step's acc_o.
      acc_o_normalizer.update(acc_o, global_max, tmp, global_sum);
      // Apply expf of softmax.
      // It is possible that all elts are -FLT_MAX with sliding_window_causal or custom_mask.
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_SOFTMAX_EXP)
      softmax.template apply_exp_with_mask<CHECK_NEG_INF>(global_max);
#endif

// Update the global sum.
// TODO Can we just zero out tmp and reduce into that
#pragma unroll
      for (int i = 0; i < Softmax::ROWS_PER_THREAD; i++) {
        tmp[i] = global_sum[i];
        global_sum[i] = 0.f;
      }

      // Compute the sum.
#if defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_SOFTMAX_REDUCE_SUM)
#pragma unroll
      for (int i = 0; i < Softmax::ROWS_PER_THREAD; ++i) {
        global_sum[i] = 1.f;
      }
#else
      softmax.template reduce<fmha::Sum_>(global_sum);
#endif

#pragma unroll
      for (int i = 0; i < Softmax::ROWS_PER_THREAD; i++) {
        global_sum[i] += tmp[i];
      }
    }
#endif

    // Store the P matrix.
#if defined(STORE_S)
    softmax.store(gmem_s);
#endif

#if defined(STORE_P)
    gmem_p.move_n();
#endif

#if defined(STORE_S)
    gmem_s.move();
#endif

#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_BMM2)
    // Repack for the next BMM.
    fmha::Fragment_a<Traits_p, fmha::Row> frag_p[Kernel_traits::TOTAL_BMM2_MMAS_K]
                                                [Mma_tile_o::MMAS_M];
#if defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_SOFTMAX_PACK)
#pragma unroll
    for (int p_ki = 0; p_ki < Kernel_traits::TOTAL_BMM2_MMAS_K; ++p_ki) {
#pragma unroll
      for (int mi = 0; mi < Mma_tile_o::MMAS_M; ++mi) {
        frag_p[p_ki][mi].set(0);
        if constexpr (std::is_same<Traits_p, fmha::Blackwell_mma_nvf4_fp32_traits>::value) {
          frag_p[p_ki][mi].scale_reg =
              fmha::make_ue4m3_scale_reg(1.f, 1.f, 1.f, 1.f);
        }
      }
    }
#else
    softmax.pack(frag_p);
#endif

    // BMM2 main loop
    for (int bmm2_k = 0; bmm2_k < BMM2_MAIN_MMAS_K_BOUND; bmm2_k += Mma_tile_o::MMAS_K) {
      // Trigger the load for next V values
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_V_GMEM_STAGING)
      gmem_v.move();
      smem_v.move_to_next_write_buffer();
      gmem_v.load(smem_v);
#endif

      // Push the LDGDEPBAR instruction after the loads for K.
      fmha::ldgdepbar<USE_LDGSTS>();
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_V_GMEM_STAGING)
      gmem_v.commit(smem_v);
#endif

      // Leave 1 outstanding batch loadings
      fmha::depbar_<USE_LDGSTS, 1>();

#pragma unroll
      for (int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki) {
        int p_ki = bmm2_k + ki;
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_V_SMEM_LOAD)
        smem_v.load(frag_v[ki], ki);
#endif
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_BMM2_GEMM)
        fmha::gemm(acc_o, frag_p[p_ki], frag_v[ki]);
#endif
      }

      // Make sure we are done reading the data (smem).
      __syncthreads();

#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_V_SMEM_LOAD)
      smem_v.move_to_next_read_buffer();
#endif
    }
#endif

    // BMM2 tail loop
    {
      if (kv_loop + Cta_tile_p::N < kv_loop_end) {
        // Trigger the load for the next Q/K values
        // Advance K rows and rewind QK cols
        int col_steps = fmha::div_up((int)Kernel_traits::VALID_D, (int)Kernel_traits::CTA_P_TILE_K);
        // Enforce warp-uniform value. Removing this one line can contribute to 10% perf drop
        // due to register spills
        col_steps = __shfl_sync(0xffffffff, col_steps, 0);

        if constexpr (Kernel_traits::RELOAD_Q && !Kernel_traits::USE_PERSISTENT_Q) {
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_Q_STAGING)
          gmem_q.move_col();
          gmem_q.rewind_col(col_steps);
          smem_q.move_to_next_write_buffer();
          gmem_q.load(smem_q);
#endif
        }
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_K_STAGING)
        gmem_k.move();
        gmem_k.move_col();
        gmem_k.rewind_col(col_steps);
        smem_k.move_to_next_write_buffer();
        gmem_k.load(smem_k);
#endif

        // Push the LDGDEPBAR instruction after the loads for K.
        fmha::ldgdepbar<USE_LDGSTS>();

        if constexpr (Kernel_traits::RELOAD_Q && !Kernel_traits::USE_PERSISTENT_Q) {
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_Q_STAGING)
          gmem_q.commit(smem_q);
#endif
        }
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_K_STAGING)
        gmem_k.commit(smem_k);
#endif

        // Wait for V to finish loading
        fmha::depbar_<USE_LDGSTS, 1>();
      } else {
        fmha::depbar_<USE_LDGSTS, 0>();
        __syncthreads();
      }

// Do this part of O = P^T * V^T.
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_BMM2)
#pragma unroll
      for (int ki = 0; ki < BMM2_TAIL_MMAS_K_BOUND; ++ki) {
        int p_ki = BMM2_MAIN_MMAS_K_BOUND + ki;
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_V_SMEM_LOAD)
        smem_v.load(frag_v[ki], ki);
#endif
#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_BMM2_GEMM)
        fmha::gemm(acc_o, frag_p[p_ki], frag_v[ki]);
#endif
      }
#endif

#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_V_SMEM_LOAD)
      smem_v.move_to_next_read_buffer();
#endif
    }  // end BMM2 tail loop

  }  // Inner loop over the key/value sequence length.

  // Update the sum if attention sinks are used.
  acc_o_normalizer.update_sum(global_max, global_sum);
  // Update acc_o of flash attention
  acc_o_normalizer.final_update(acc_o, global_sum);

  // If kv_loop breaks prematurely in case of causal masking, make sure there is no data in-flight
  if (Kernel_traits::CAUSAL_MASK) {
    fmha::depbar_<USE_LDGSTS, 0>();
  }
  // Wait for last round of LDS K/V to finish
  __syncthreads();

#if defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_OUTPUT)
  return;
#endif

  if constexpr (std::is_same<Traits_o, fmha::Blackwell_mma_nvf4_fp32_traits>::value &&
                !Kernel_traits::USE_SMEM_O) {
    int const lane = tidx & 31;
    int const warp = tidx / Cta_tile_o::THREADS_PER_WARP;
    int const warp_m = warp % Cta_tile_o::WARPS_M;
    int const warp_k = warp / (Cta_tile_o::WARPS_M * Cta_tile_o::WARPS_N);
    uint32_t const scale_bits = params.scale_bmm2_d ? *params.scale_bmm2_d : params.scale_bmm2;
    float const scale_bmm2 = reinterpret_cast<float const&>(scale_bits);
    char* o_base = reinterpret_cast<char*>(params.o_ptr);
    if (num_kv_splits > 1) {
      o_base += static_cast<int64_t>(kv_split_idx) * params.split_o_stride_in_bytes;
    }
    int const valid_bytes_per_row = params.dv * static_cast<int>(sizeof(fmha::bf16_t));
    constexpr int REDUCE_ELTS_PER_THREAD = 8;
    constexpr int REDUCE_STRIDE =
        Cta_tile_o::THREADS_PER_CTA * REDUCE_ELTS_PER_THREAD;
    constexpr int REDUCE_TILE_STRIDE =
        Mma_tile_o::VALID_MMAS_N * REDUCE_STRIDE;
    constexpr bool USE_MI_OUTPUT_REDUCTION =
        Cta_tile_o::WARPS_K > 1 &&
        REDUCE_TILE_STRIDE * static_cast<int>(sizeof(float)) <=
            Kernel_traits::BYTES_PER_SMEM;
    float* reduce_smem = reinterpret_cast<float*>(smem_);

    if constexpr (Cta_tile_o::WARPS_K > 1) {
      static_assert(Cta_tile_o::THREADS_PER_CTA * REDUCE_ELTS_PER_THREAD *
                            sizeof(float) <=
                        Kernel_traits::BYTES_PER_SMEM,
                    "");
    }
    bool const full_output_tile =
        q_row_loop + Cta_tile_o::M <= q_rows &&
        o_part * Cta_tile_o::N + Cta_tile_o::N <= params.dv;

#pragma unroll
    for (int mi = 0; mi < Mma_tile_o::MMAS_M; ++mi) {
      if constexpr (USE_MI_OUTPUT_REDUCTION) {
#pragma unroll
        for (int ni = 0; ni < Mma_tile_o::VALID_MMAS_N; ++ni) {
          int const reduce_base =
              ni * REDUCE_STRIDE + tidx * REDUCE_ELTS_PER_THREAD;
#pragma unroll
          for (int e = 0; e < REDUCE_ELTS_PER_THREAD; ++e) {
            reduce_smem[reduce_base + e] = acc_o[mi][ni].elt(e);
          }
        }
        __syncthreads();
      }

      int const row0 =
          q_row_loop + (mi * Cta_tile_o::WARPS_M + warp_m) *
                           Mma_tile_o::M_PER_MMA +
          (lane >> 2);
      int const row1 = row0 + 8;
      int const out_token_row0 =
          group_q_heads_in_m ? row0 / params.num_grouped_heads : row0;
      int const out_token_row1 =
          group_q_heads_in_m ? row1 / params.num_grouped_heads : row1;
      int const out_head0 =
          group_q_heads_in_m ? binfo.bidh * params.h_q_per_kv +
                                   row0 % params.num_grouped_heads
                             : binfo.bidh;
      int const out_head1 =
          group_q_heads_in_m ? binfo.bidh * params.h_q_per_kv +
                                   row1 % params.num_grouped_heads
                             : binfo.bidh;
      char* row0_base =
          o_base + static_cast<int64_t>(binfo.sum_s + out_token_row0) *
                       params.o_stride_in_bytes +
          static_cast<int64_t>(out_head0) * valid_bytes_per_row;
      char* row1_base =
          o_base + static_cast<int64_t>(binfo.sum_s + out_token_row1) *
                       params.o_stride_in_bytes +
          static_cast<int64_t>(out_head1) * valid_bytes_per_row;

#pragma unroll
      for (int ni = 0; ni < Mma_tile_o::VALID_MMAS_N; ++ni) {
        float out_frag[REDUCE_ELTS_PER_THREAD];
#pragma unroll
        for (int e = 0; e < REDUCE_ELTS_PER_THREAD; ++e) {
          out_frag[e] = acc_o[mi][ni].elt(e);
        }
        if constexpr (Cta_tile_o::WARPS_K > 1) {
          int const reduce_base =
              (USE_MI_OUTPUT_REDUCTION ? ni * REDUCE_STRIDE : 0) +
              tidx * REDUCE_ELTS_PER_THREAD;
          if constexpr (!USE_MI_OUTPUT_REDUCTION) {
#pragma unroll
            for (int e = 0; e < REDUCE_ELTS_PER_THREAD; ++e) {
              reduce_smem[reduce_base + e] = out_frag[e];
            }
            __syncthreads();
          }

          if (warp_k == 0) {
            constexpr int WARP_K_THREAD_STRIDE =
                Cta_tile_o::WARPS_M * Cta_tile_o::WARPS_N *
                Cta_tile_o::THREADS_PER_WARP;
#pragma unroll
            for (int kk = 1; kk < Cta_tile_o::WARPS_K; ++kk) {
#pragma unroll
              for (int e = 0; e < REDUCE_ELTS_PER_THREAD; ++e) {
                out_frag[e] +=
                    reduce_smem[reduce_base +
                                kk * WARP_K_THREAD_STRIDE *
                                    REDUCE_ELTS_PER_THREAD +
                                e];
            }
          }
          }
          if constexpr (!USE_MI_OUTPUT_REDUCTION) {
            __syncthreads();
          }
        }
        int const col0 = o_part * Cta_tile_o::N +
                         ni * Mma_tile_o::N_PER_MMA_PER_CTA +
                         2 * (lane & 3);
        int const col1 = col0 + 8;
        bool const write_output = Cta_tile_o::WARPS_K == 1 || warp_k == 0;
        if (write_output) {
          uint32_t const out00 = fmha::float2_to_16bit_2<fmha::bf16_t>(
              out_frag[0] * scale_bmm2, out_frag[1] * scale_bmm2);
          uint32_t const out10 = fmha::float2_to_16bit_2<fmha::bf16_t>(
              out_frag[2] * scale_bmm2, out_frag[3] * scale_bmm2);
          uint32_t const out01 = fmha::float2_to_16bit_2<fmha::bf16_t>(
              out_frag[4] * scale_bmm2, out_frag[5] * scale_bmm2);
          uint32_t const out11 = fmha::float2_to_16bit_2<fmha::bf16_t>(
              out_frag[6] * scale_bmm2, out_frag[7] * scale_bmm2);
          if (full_output_tile) {
            fmha::stg(row0_base + static_cast<int64_t>(col0) *
                                      sizeof(fmha::bf16_t),
                      out00);
            fmha::stg(row1_base + static_cast<int64_t>(col0) *
                                      sizeof(fmha::bf16_t),
                      out10);
            fmha::stg(row0_base + static_cast<int64_t>(col1) *
                                      sizeof(fmha::bf16_t),
                      out01);
            fmha::stg(row1_base + static_cast<int64_t>(col1) *
                                      sizeof(fmha::bf16_t),
                      out11);
          } else {
            if (row0 < q_rows && col0 + 1 < params.dv) {
              fmha::stg(row0_base + static_cast<int64_t>(col0) *
                                        sizeof(fmha::bf16_t),
                        out00);
            }
            if (row1 < q_rows && col0 + 1 < params.dv) {
              fmha::stg(row1_base + static_cast<int64_t>(col0) *
                                        sizeof(fmha::bf16_t),
                        out10);
            }
            if (row0 < q_rows && col1 + 1 < params.dv) {
              fmha::stg(row0_base + static_cast<int64_t>(col1) *
                                        sizeof(fmha::bf16_t),
                        out01);
            }
            if (row1 < q_rows && col1 + 1 < params.dv) {
              fmha::stg(row1_base + static_cast<int64_t>(col1) *
                                        sizeof(fmha::bf16_t),
                        out11);
            }
          }
        }
      }
      if constexpr (USE_MI_OUTPUT_REDUCTION) {
        __syncthreads();
      }
    }
    if (params.softmax_stats_ptr != nullptr) {
      using Mma_tile = typename Traits_p::template Mma_tile<Cta_tile_o>;
      fmha::Softmax_saver<Cta_tile_o, Mma_tile> saver(params, binfo);
      saver.store(q_loop, global_sum, global_max);
    }
    return;
  } else {

  // Allocate the global memory tile loader for O.
  Gmem_tile_o gmem_o(params, binfo, tidx, q_row_loop,
                     o_part * Cta_tile_o::N * Gmem_tile_o::BYTES_PER_ELEMENT);
  // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
  Smem_tile_o smem_o(&smem_[Kernel_traits::NO_LOOP ? 0 : Smem_tile_q::BYTES_PER_TILE], tidx);

// Loop over MMAS_M.
#pragma unroll
  for (int ii = 0; ii < Gmem_tile_o::LOOPS; ++ii) {
    // Swizzle the elements and do the final reduction.
    smem_o.store(acc_o, ii);

    // Make sure the data is in shared memory.
    __syncthreads();

    // Load from shared memory.
    uint4 out[Gmem_tile_o::STGS_PER_LOOP];
    smem_o.load(out);

    // Make sure the data was read from shared memory.
    if (ii < Gmem_tile_o::LOOPS - 1) {
      __syncthreads();
    }

    // Output the values.
    gmem_o.store(out, ii);
  }
    if (params.softmax_stats_ptr != nullptr) {
      using Mma_tile = typename Traits_p::template Mma_tile<Cta_tile_o>;
      fmha::Softmax_saver<Cta_tile_o, Mma_tile> saver(params, binfo);
      saver.store(q_loop, global_sum, global_max);
    }
  }
}  // device_flash_attention_1xN

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fused_multihead_attention
