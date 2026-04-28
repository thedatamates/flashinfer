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
#include <cuda_bf16.h>

#include <type_traits>
#include <utility>

#include <fmha/traits.h>
#include <fused_multihead_attention.h>

namespace fmha {
namespace v2 {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class Has_v_scale_write_buffer {
  template <typename U>
  static auto test(int) -> decltype(
      std::declval<U const&>().load_v_scale_byte_from_write_buffer(0, 0), std::true_type{});

  template <typename>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T>
class Has_k_data_write_buffer {
  template <typename U>
  static auto test(int) -> decltype(
      std::declval<U&>().store_k_data_byte(0, 0, uint8_t{0}), std::true_type{});

  template <typename>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int USE_LDGSTS>
struct Ldgsts_helper {
  template <typename This, typename Smem_tile, int LDGS>
  static inline __device__ void load(This* this_, Smem_tile& smem_tile, void const* (&ptrs)[LDGS],
                                     uint32_t (&preds)[LDGS]) {
    fmha::pack_predicates(this_->preds_, preds);
    smem_tile.store(ptrs, this_->preds_);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Ldgsts_helper<0> {
  template <typename This, typename Smem_tile, int LDGS>
  static inline __device__ void load(This* this_, Smem_tile& smem_tile, void const* (&ptrs)[LDGS],
                                     uint32_t (&preds)[LDGS]) {
#if 0
        fmha::pack_predicates(this_->preds_, preds);
        fmha::ldg(this_->fetch_, ptrs, this_->preds_);
#else
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      this_->fetch_[ii] = make_uint4(0u, 0u, 0u, 0u);
    }
    // not packing predicates removes restrictions (e.g. FP16 384, 4 warps)
    Ldg_functor<uint4, LDGS> fct(this_->fetch_, ptrs);
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      fct.ldgsts(ii, preds[ii]);
    }
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bits per element.
    int BITS_PER_ELEMENT_,
    // The number of rows of Q, K or V loaded by this tile.
    int ROWS_,
    // The number of columns (padded, e.g 64).
    int COLS,
    // The actual number of columns (unpadded, e.g 40)
    int VALID_COLS_,
    // Do we use LDGSTS?
    bool USE_LDGSTS_,
    // Are attention heads interleaved?
    bool HEADS_INTERLEAVED,
    // The number of matrices
    int NUM_MATS = 3,
    // Is sliding window attention used ?
    bool SLIDING_WINDOW_ATTENTION = false>
struct Gmem_tile_qkv {
  // The size of each LDG.
  enum { BYTES_PER_LDG = 16 };

  // The number of bits/bytes of element
  enum { BITS_PER_ELEMENT = BITS_PER_ELEMENT_ };

  enum { BYTES_PER_ELEMENT = BITS_PER_ELEMENT_ >= 8 ? BITS_PER_ELEMENT_ / 8 : 0 };

  // The size of a row in bytes.
  enum { BYTES_PER_ROW = (COLS * BITS_PER_ELEMENT + 7) / 8 };

  // The number of threads to load a "row" of the matrix.
  enum { THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDG };

  // The number of logical elements loaded per thread.
  enum { ELEMENTS_PER_LDG = BYTES_PER_LDG * 8 / BITS_PER_ELEMENT };

  // The valid size of a row in bytes (without paddings).
  enum { VALID_COLS = VALID_COLS_ };

  // The amount of bytes that are valid per row.
  enum { VALID_BYTES_PER_ROW = (VALID_COLS * BITS_PER_ELEMENT + 7) / 8 };

  static inline __host__ __device__ int64_t bytes_for_elements(int64_t elements) {
    return elements * BITS_PER_ELEMENT / 8;
  }

  // The number of "rows" loaded per LDG.
  enum { ROWS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };

  // The number of rows.
  enum { ROWS = ROWS_ };

  // The number of LDGs needed to load a chunk of the Q matrix.
  enum { LDGS = fmha::Div_up<ROWS, ROWS_PER_LDG>::VALUE };

  // The number of predicate registers.
  enum { PRED_REGS = fmha::Compute_number_of_pred_regs<LDGS>::VALUE };

  // Is it Hopper?
  enum {
    IS_HOPPER = std::is_same<typename Traits::Gpu_arch, typename fmha::Hopper>::value == true
  };

  // Make sure we use a single register to store predicates. Do not throw for Hopper for now.
  static_assert(!USE_LDGSTS_ || PRED_REGS == 1 || IS_HOPPER, "");

  // We do not use LDGSTS (for the moment).
  enum { USE_LDGSTS = USE_LDGSTS_ };

  // Ctor for bert::Fused_multihead_attention_params_v2 class
  template <typename Block_info>
  inline __device__ Gmem_tile_qkv(bert::Fused_multihead_attention_params_v2 const& params,
                                  int qkv_offset, Block_info const& binfo, int tidx,
                                  int cta_row_offset = 0, int cta_col_offset_in_bytes = 0)
      : Gmem_tile_qkv(params.qkv_ptr, params.q_stride_in_bytes, params.d, params.dv, params.h,
                      qkv_offset, binfo, tidx, params.h_kv, cta_row_offset,
                      cta_col_offset_in_bytes) {}

  // Ctor for other param classes (such as Qkv_params in train_ops)
  template <typename Params, typename Block_info>
  inline __device__ Gmem_tile_qkv(Params const& params, int qkv_offset, Block_info const& binfo,
                                  int tidx, int cta_row_offset = 0, int cta_col_offset_in_bytes = 0)
      : Gmem_tile_qkv(params.qkv_ptr, params.q_stride_in_bytes, params.d, params.dv, params.h,
                      qkv_offset, binfo, tidx, cta_row_offset, cta_col_offset_in_bytes) {}

  // Ctor.
  template <typename Block_info>
  inline __device__ Gmem_tile_qkv(void* qkv_ptr, size_t qkv_stride_in_bytes, int d, int dv,
                                  int num_heads, int qkv_offset, Block_info const& binfo, int tidx,
                                  int num_kv_heads = 0, int cta_row_offset = 0,
                                  int cta_col_offset_in_bytes = 0)
      : params_qkv_stride_in_bytes_(qkv_stride_in_bytes),
        actual_seqlen_(binfo.actual_seqlen),
        qkv_ptr_(reinterpret_cast<char*>(qkv_ptr)) {
    // Compute the position in the sequence (within the CTA for the moment).
    int row = tidx / THREADS_PER_ROW;
    // Compute the position of the thread in the row.
    int col = tidx % THREADS_PER_ROW;

    // We must store the value to update the predicates in "load".
    row_ = row;
    // Do not load/store if the thread is in the padded area
    col_in_bytes_ = cta_col_offset_in_bytes + col * BYTES_PER_LDG;

    // The row offset in the batched GEMM. For each seq element, we store QKV in that order.
    int64_t row_offset = (int64_t)(row + cta_row_offset) * params_qkv_stride_in_bytes_;
    // Add the byte index.
    int64_t idx;

    // Both MQA and GQA will use non HEADS_INTERLEAVED layout
    if (num_kv_heads < num_heads) {
      int const head_id = binfo.bidh;
      int const kv_head_id = binfo.bidh / (num_heads / num_kv_heads);
      // QKV layout [b, s, [q_hd, k_h'd, v_h'd]]
      idx = binfo.sum_s * params_qkv_stride_in_bytes_;
      if (qkv_offset == 0) {  // Q tensor
        idx += head_id * VALID_BYTES_PER_ROW;
      } else if (qkv_offset == 1) {  // K tensor
        idx += (num_heads + kv_head_id) * VALID_BYTES_PER_ROW;
      } else if (qkv_offset == 2) {  // V tensor
        /*  When qkv_offset == 2, this is an instance of Gmem_tile_v defined in Kernel_traits:
                using Gmem_tile_v = Gmem_tile_v_<Traits_o,
                        Cta_tile_o,
                        Traits_o::BITS_PER_ELEMENT_B,
                        CTA_O_TILE_K,
                        CTA_O_TILE_N,
                        VALID_DV,   // instead of VALID_D
                        USE_LDGSTS_V,
                        HEADS_INTERLEAVED,
                        3, // NUM_MATS
                        SLIDING_WINDOW_ATTENTION>;
            the 6th template argument is VALID_DV instead of VALID_D.
            Thus, here VALID_COLS equals VALID_DV, and
            VALID_BYTES_PER_ROW equals VALID_DV * BYTES_PER_ELEMENT,
            and `kv_head_id * dv * BYTES_PER_ELEMENT` can be optimized to
            `kv_head_id * VALID_BYTES_PER_ROW`. */
        idx += bytes_for_elements((num_heads + num_kv_heads) * d) +
               kv_head_id * VALID_BYTES_PER_ROW;
      }
    } else if (HEADS_INTERLEAVED) {
      // [b, s, h, [q_d, k_d, v_d]] aka bsh3d
      // bidx = sum_s * params.h + bidh;
      idx = bytes_for_elements(binfo.bidx * (2 * d + dv) + qkv_offset * d);
    } else {
      // [b, s, [q_hd, k_hd, v_hd]] aka bs3hd
      idx = binfo.sum_s * params_qkv_stride_in_bytes_ +
            bytes_for_elements(qkv_offset * num_heads * d) +
            binfo.bidh * VALID_BYTES_PER_ROW;
    }

    // Assemble the final pointer.
    qkv_ptr_ += row_offset + idx + col_in_bytes_;

    // Take the CTA offset to modify the sequence length.
    actual_seqlen_ -= cta_row_offset;

    // Set the initial seq_len and qkv_offset in case of reinterating
    actual_seqlen_init_ = actual_seqlen_;
    qkv_ptr_init_ = qkv_ptr_;
  }

  // Store data to shared memory.
  template <typename Smem_tile>
  inline __device__ void commit(Smem_tile& smem_tile) {
    if (!USE_LDGSTS) {
      smem_tile.store(fetch_);
    }
  }

  // Load data from memory.
  template <typename Smem_tile>
  inline __device__ void load(Smem_tile& smem_tile) {
    uint32_t preds[LDGS];
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      preds[ii] = row_ + ii * (int)ROWS_PER_LDG < min((int)ROWS, actual_seqlen_);
      preds[ii] &= col_in_bytes_ < VALID_BYTES_PER_ROW;
    }

    // Prepare the load pointers.
    void const* ptrs[LDGS];
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      ptrs[ii] = qkv_ptr_ + (int64_t)ii * ROWS_PER_LDG * params_qkv_stride_in_bytes_;
    }

    // Trigger LDGSTS or the LDGs.
    // The predicates protect against out-of-bound access in rows and cols
    Ldgsts_helper<USE_LDGSTS>::load(this, smem_tile, ptrs, preds);
  }

  // Load data from memory.
  inline __device__ void load() {
    uint32_t preds[LDGS];
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      preds[ii] = row_ + ii * (int)ROWS_PER_LDG < min((int)ROWS, actual_seqlen_);
    }

    // Prepare the load pointers.
    void const* ptrs[LDGS];
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      ptrs[ii] = qkv_ptr_ + (int64_t)ii * ROWS_PER_LDG * params_qkv_stride_in_bytes_;
    }

    // Trigger the LDGs.
    if (col_in_bytes_ < VALID_BYTES_PER_ROW) {
      fmha::pack_predicates(preds_, preds);
      fmha::ldg(fetch_, ptrs, preds_);
    } else {
#pragma unroll
      for (int ii = 0; ii < LDGS; ++ii) {
        fetch_[ii] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  }

  // Move the pointer to the next row location.
  inline __device__ void move(int const steps = 1) {
    qkv_ptr_ += (int64_t)ROWS * params_qkv_stride_in_bytes_ * steps;
    actual_seqlen_ -= (int)ROWS * steps;
  }

  // Move the pointer to the next row location by the offset (not step).
  inline __device__ void move_by_offset(int const offset) {
    qkv_ptr_ = qkv_ptr_init_ + (int64_t)offset * params_qkv_stride_in_bytes_;
    actual_seqlen_ = actual_seqlen_init_ - (int)offset;
  }

  // Move the pointer to the next column location
  inline __device__ void move_col(int const steps = 1) {
    qkv_ptr_ += (int64_t)BYTES_PER_ROW * steps;
    // Update col_in_bytes_ to ensure load predicates work
    col_in_bytes_ += THREADS_PER_ROW * BYTES_PER_LDG * steps;
  }

  inline __device__ void reset() {
    qkv_ptr_ = qkv_ptr_init_;
    actual_seqlen_ = actual_seqlen_init_;
  }

  // Rewind the pointer back to previous column location
  inline __device__ void rewind_col(int const steps) {
    qkv_ptr_ -= BYTES_PER_ROW * steps;
    // Update col_in_bytes_ to ensure load predicates work
    col_in_bytes_ -= THREADS_PER_ROW * BYTES_PER_LDG * steps;
  }

  inline __device__ void move_to(int const step) {
    qkv_ptr_ = qkv_ptr_init_ + (int64_t)ROWS * params_qkv_stride_in_bytes_ * step;
    actual_seqlen_ = actual_seqlen_init_ - (int)ROWS * step;
  }

  // Store data to memory.
  inline __device__ void store(uint4 const (&data)[LDGS]) {
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      char* ptr = qkv_ptr_ + (int64_t)ii * ROWS_PER_LDG * params_qkv_stride_in_bytes_;
      if (((row_ + ii * ROWS_PER_LDG) < min(ROWS, actual_seqlen_)) &&
          col_in_bytes_ < VALID_BYTES_PER_ROW /*TODO: double check*/) {
        fmha::stg(ptr, data[ii]);
      }
    }
  }

  // The stride between rows for the QKV matrice.
  int64_t params_qkv_stride_in_bytes_;
  // The pointer.
  char* qkv_ptr_;
  char* qkv_ptr_init_;
  // The register to store predicates.
  uint32_t preds_[PRED_REGS];
  // The fetch registers.
  uint4 fetch_[LDGS];
  // Keep track of the row and col the thread is processing as we move the tile.
  int row_;
  int col_in_bytes_;
  // The sequence length.
  int actual_seqlen_;
  int actual_seqlen_init_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

// We expect the Q/K/V layout to be [B, S, H, D] with variable sequence length support.
template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bits per element.
    int BITS_PER_ELEMENT_,
    // The number of rows of Q, K or V loaded by this tile.
    int ROWS_,
    // The number of columns (padded, e.g 64).
    int COLS,
    // The actual number of columns (unpadded, e.g 40)
    int VALID_COLS_,
    // Do we use LDGSTS?
    bool USE_LDGSTS_,
    // Are attention heads interleaved? (not used)
    bool HEADS_INTERLEAVED = false,
    // The number of matrices (not used)
    int NUM_MATS = 1,
    // Is sliding window attention used ?
    bool SLIDING_WINDOW_ATTENTION = false>
struct Gmem_tile_q_k_v {
  // The size of each LDG.
  enum { BYTES_PER_LDG = 16 };

  // The number of bits/bytes of element
  enum { BITS_PER_ELEMENT = BITS_PER_ELEMENT_ };

  enum { BYTES_PER_ELEMENT = BITS_PER_ELEMENT_ >= 8 ? BITS_PER_ELEMENT_ / 8 : 0 };

  // The size of a row in bytes.
  enum { BYTES_PER_ROW = (COLS * BITS_PER_ELEMENT + 7) / 8 };

  // The number of threads to load a "row" of the matrix.
  enum { THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDG };

  // The number of logical elements loaded per thread.
  enum { ELEMENTS_PER_LDG = BYTES_PER_LDG * 8 / BITS_PER_ELEMENT };

  // The valid size of a row in bytes (without paddings).
  enum { VALID_COLS = VALID_COLS_ };

  // The amount of bytes that are valid per row.
  enum { VALID_BYTES_PER_ROW = (VALID_COLS * BITS_PER_ELEMENT + 7) / 8 };

  static inline __host__ __device__ int64_t bytes_for_elements(int64_t elements) {
    return elements * BITS_PER_ELEMENT / 8;
  }

  // The number of "rows" loaded per LDG.
  enum { ROWS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };

  // The number of rows.
  enum { ROWS = ROWS_ };

  // The number of LDGs needed to load a chunk of the Q matrix.
  enum { LDGS = fmha::Div_up<ROWS, ROWS_PER_LDG>::VALUE };

  // The number of predicate registers.
  enum { PRED_REGS = fmha::Compute_number_of_pred_regs<LDGS>::VALUE };

  // Is it Hopper?
  enum {
    IS_HOPPER = std::is_same<typename Traits::Gpu_arch, typename fmha::Hopper>::value == true
  };

  // Make sure we use a single register to store predicates. Do not throw for Hopper for now.
  static_assert(!USE_LDGSTS_ || PRED_REGS == 1 || IS_HOPPER, "");

  // We do not use LDGSTS (for the moment).
  enum { USE_LDGSTS = USE_LDGSTS_ };

  // Ctor
  // qkv_offset: 0 for Q, 1 for K, 2 for V
  template <typename Block_info>
  inline __device__ Gmem_tile_q_k_v(bert::Fused_multihead_attention_params_v2 const& params,
                                    int qkv_offset, Block_info const& binfo, int tidx,
                                    int cta_row_offset = 0, int cta_col_offset_in_bytes = 0) {
    int seq_offset = 0;
    if (qkv_offset == 0) {
      // Q tensor
      params_q_k_v_stride_in_bytes_ = params.q_stride_in_bytes;
      q_k_v_ptr_ = reinterpret_cast<char*>(params.q_ptr);
      actual_seqlen_ = binfo.actual_q_seqlen;
      seq_offset = binfo.sum_s;
    } else if (qkv_offset == 1) {
      // K tensor
      params_q_k_v_stride_in_bytes_ = params.k_stride_in_bytes;
      q_k_v_ptr_ = reinterpret_cast<char*>(params.k_ptr);
      actual_seqlen_ = binfo.actual_kv_seqlen;
      seq_offset = binfo.sum_s_kv;
    } else if (qkv_offset == 2) {
      // V tensor
      params_q_k_v_stride_in_bytes_ = params.v_stride_in_bytes;
      q_k_v_ptr_ = reinterpret_cast<char*>(params.v_ptr);
      actual_seqlen_ = binfo.actual_kv_seqlen;
      seq_offset = binfo.sum_s_kv;
    }

    // Compute the position in the sequence (within the CTA for the moment).
    int row = tidx / THREADS_PER_ROW;
    // Compute the position of the thread in the row.
    int col = tidx % THREADS_PER_ROW;

    // We must store the value to update the predicates in "load".
    row_ = row;
    // Do not load/store if the thread is in the padded area
    col_in_bytes_ = cta_col_offset_in_bytes + col * BYTES_PER_LDG;

    // The row offset in the batched GEMM, including the sequence offset.
    int64_t row_offset =
        (int64_t)(row + cta_row_offset + seq_offset) * params_q_k_v_stride_in_bytes_;
    // Add the head index.
    int64_t idx = binfo.bidh;

    // Assemble the final pointer.
    q_k_v_ptr_ += row_offset + idx * VALID_BYTES_PER_ROW + col_in_bytes_;

    // Take the CTA offset to modify the sequence length.
    actual_seqlen_ -= cta_row_offset;

    // Set the initial seq_len and qkv_offset in case of reinterating
    actual_seqlen_init_ = actual_seqlen_;
    q_k_v_ptr_init_ = q_k_v_ptr_;
  }

  // Store data to shared memory.
  template <typename Smem_tile>
  inline __device__ void commit(Smem_tile& smem_tile) {
    if (!USE_LDGSTS) {
      smem_tile.store(fetch_);
    }
  }

  // Load data from memory.
  template <typename Smem_tile>
  inline __device__ void load(Smem_tile& smem_tile) {
    uint32_t preds[LDGS];
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      preds[ii] = row_ + ii * (int)ROWS_PER_LDG < min((int)ROWS, actual_seqlen_);
      preds[ii] &= col_in_bytes_ < VALID_BYTES_PER_ROW;
    }

    // Prepare the load pointers.
    void const* ptrs[LDGS];
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      ptrs[ii] = q_k_v_ptr_ + (int64_t)ii * ROWS_PER_LDG * params_q_k_v_stride_in_bytes_;
    }

    // Trigger LDGSTS or the LDGs.
    // The predicates protect against out-of-bound access in rows and cols
    Ldgsts_helper<USE_LDGSTS>::load(this, smem_tile, ptrs, preds);
  }

  // Move the pointer to the next row location.
  inline __device__ void move(int const steps = 1) {
    q_k_v_ptr_ += (int64_t)ROWS * params_q_k_v_stride_in_bytes_ * steps;
    actual_seqlen_ -= (int)ROWS * steps;
  }

  // Move the pointer to the next row location by the offset (not step).
  inline __device__ void move_by_offset(int const offset) {
    q_k_v_ptr_ = q_k_v_ptr_init_ + (int64_t)offset * params_q_k_v_stride_in_bytes_;
    actual_seqlen_ = actual_seqlen_init_ - (int)offset;
  }

  // Move the pointer to the next column location
  inline __device__ void move_col() {
    q_k_v_ptr_ += (int64_t)BYTES_PER_ROW;
    // Update col_in_bytes_ to ensure load predicates work
    col_in_bytes_ += THREADS_PER_ROW * BYTES_PER_LDG;
  }

  // Rewind the pointer back to previous column location
  inline __device__ void rewind_col(int const steps) {
    q_k_v_ptr_ -= BYTES_PER_ROW * steps;
    // Update col_in_bytes_ to ensure load predicates work
    col_in_bytes_ -= THREADS_PER_ROW * BYTES_PER_LDG * steps;
  }

  // Move the pointer to the specified step.
  inline __device__ void move_to(int const step) {
    q_k_v_ptr_ = q_k_v_ptr_init_ + (int64_t)ROWS * params_q_k_v_stride_in_bytes_ * step;
    actual_seqlen_ = actual_seqlen_init_ - (int)ROWS * step;
  }

  inline __device__ void reset() {
    q_k_v_ptr_ = q_k_v_ptr_init_;
    actual_seqlen_ = actual_seqlen_init_;
  }

  // The stride between rows for the Q/K/V matrice.
  int64_t params_q_k_v_stride_in_bytes_;
  // The pointer.
  char* q_k_v_ptr_;
  char* q_k_v_ptr_init_;
  // The register to store predicates.
  uint32_t preds_[PRED_REGS];
  // The fetch registers.
  uint4 fetch_[LDGS];
  // Keep track of the row and col the thread is processing as we move the tile.
  int row_;
  int64_t col_in_bytes_;
  // The sequence length.
  int actual_seqlen_;
  int actual_seqlen_init_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Q for native NVFP4 attention is read as BF16 from global memory, then packed
// into the FP4 shared-memory/MMA layout before the first MMA. It also writes a
// compact per-row scale sidecar consumed by Smem_tile_a when building the
// Blackwell block-scaled MMA fragment.
template <
    typename Traits,
    typename Cta_tile,
    int BITS_PER_ELEMENT_,
    int ROWS_,
    int COLS,
    int VALID_COLS_,
    bool USE_LDGSTS_,
    bool HEADS_INTERLEAVED = false,
    int NUM_MATS = 1,
    bool SLIDING_WINDOW_ATTENTION = false>
struct Gmem_tile_q_bf16 {
  enum { BYTES_PER_LDG = 16 };
  enum { BITS_PER_ELEMENT = BITS_PER_ELEMENT_ };
  enum { BYTES_PER_ELEMENT = 0 };
  enum { BYTES_PER_ROW = (COLS * BITS_PER_ELEMENT + 7) / 8 };
  enum { VALID_COLS = VALID_COLS_ };
  enum { VALID_BYTES_PER_ROW = (VALID_COLS * BITS_PER_ELEMENT + 7) / 8 };
  enum { ELEMENTS_PER_LDG = BYTES_PER_LDG * 8 / BITS_PER_ELEMENT };
  enum { THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDG };
  enum { ROWS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };
  enum { ROWS = ROWS_ };
  enum { LDGS = fmha::Div_up<ROWS, ROWS_PER_LDG>::VALUE };
  enum { PRED_REGS = fmha::Compute_number_of_pred_regs<LDGS>::VALUE };
  enum { USE_LDGSTS = 0 };
  enum { SCALE_GROUPS_PER_ROW = COLS / Traits::NVFP4_SCALE_VEC_SIZE };

  static_assert(std::is_same<Traits, fmha::Blackwell_mma_nvf4_fp32_traits>::value, "");
  static_assert(BITS_PER_ELEMENT == 4, "");
  static_assert(COLS % Traits::NVFP4_SCALE_VEC_SIZE == 0, "");
  static_assert(COLS % Traits::K_PER_MMA == 0, "");
  static_assert(!USE_LDGSTS_, "BF16 Q must be packed to FP4 in registers before SMEM store");

  template <typename Block_info>
  inline __device__ Gmem_tile_q_bf16(bert::Fused_multihead_attention_params_v2 const& params,
                                     int qkv_offset, Block_info const& binfo, int tidx,
                                     int cta_row_offset = 0,
                                     int cta_col_offset_in_bytes = 0)
      : params_q_stride_in_bytes_(params.q_stride_in_bytes),
        params_d_(params.d),
        q_token_base_(binfo.sum_s),
        q_row_offset_(cta_row_offset),
        q_row_offset_init_(cta_row_offset),
        num_grouped_heads_(params.num_grouped_heads),
        q_head_base_(params.num_grouped_heads > 1 ? binfo.bidh * params.h_q_per_kv
                                                  : binfo.bidh),
        actual_seqlen_(params.num_grouped_heads > 1
                           ? binfo.actual_q_seqlen * params.num_grouped_heads - cta_row_offset
                           : binfo.actual_q_seqlen - cta_row_offset),
        q_ptr_(reinterpret_cast<char const*>(params.q_ptr)) {
    assert(qkv_offset == 0);

    int const row = tidx / THREADS_PER_ROW;
    int const col = tidx % THREADS_PER_ROW;
    row_ = row;
    col_offset_elements_ =
        cta_col_offset_in_bytes * 8 / BITS_PER_ELEMENT + col * ELEMENTS_PER_LDG;

    actual_seqlen_init_ = actual_seqlen_;
    col_offset_elements_init_ = col_offset_elements_;
  }

  template <typename Smem_tile>
  inline __device__ void commit(Smem_tile& smem_tile) {
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      int const local_col = col_offset_elements_ % COLS;
      if (row_ + ii * ROWS_PER_LDG < ROWS && col_offset_elements_ < VALID_COLS &&
          local_col % Traits::K_PER_MMA == 0) {
        int const scale_group = local_col / Traits::NVFP4_SCALE_VEC_SIZE;
        smem_tile.store_q_scale(row_ + ii * ROWS_PER_LDG, scale_group,
                                scale_fetch_[ii]);
      }
    }
    smem_tile.store(fetch_);
  }

  template <int IDX>
  static inline __device__ uint16_t get_bf16_bits(uint4 const& data) {
    static_assert(IDX >= 0 && IDX < 8, "");
    uint32_t word;
    if constexpr (IDX < 2) {
      word = data.x;
    } else if constexpr (IDX < 4) {
      word = data.y;
    } else if constexpr (IDX < 6) {
      word = data.z;
    } else {
      word = data.w;
    }
    if constexpr ((IDX & 1) == 0) {
      return static_cast<uint16_t>(word & 0xffffu);
    } else {
      return static_cast<uint16_t>(word >> 16);
    }
  }

  inline __device__ uint4 load_q_bf16_vec(int row, int col) const {
    uint4 data = make_uint4(0u, 0u, 0u, 0u);
    if (row < actual_seqlen_ && col + 7 < VALID_COLS) {
      int const absolute_row = q_row_offset_ + row;
      int const token_row = absolute_row / num_grouped_heads_;
      int const head_in_group = absolute_row - token_row * num_grouped_heads_;
      int const q_head = q_head_base_ + head_in_group;
      int64_t const token_offset =
          static_cast<int64_t>(q_token_base_ + token_row) * params_q_stride_in_bytes_;
      int64_t const head_offset =
          static_cast<int64_t>(q_head) * params_d_ * sizeof(__nv_bfloat16);
      fmha::ldg(data, q_ptr_ + token_offset + head_offset +
                          static_cast<int64_t>(col) * sizeof(__nv_bfloat16));
    }
    return data;
  }

  static inline __device__ float amax_bf16_vec(uint4 const& data) {
    float amax = 0.f;
#pragma unroll
    for (int idx = 0; idx < 8; ++idx) {
      uint16_t bits;
      if (idx == 0) {
        bits = get_bf16_bits<0>(data);
      } else if (idx == 1) {
        bits = get_bf16_bits<1>(data);
      } else if (idx == 2) {
        bits = get_bf16_bits<2>(data);
      } else if (idx == 3) {
        bits = get_bf16_bits<3>(data);
      } else if (idx == 4) {
        bits = get_bf16_bits<4>(data);
      } else if (idx == 5) {
        bits = get_bf16_bits<5>(data);
      } else if (idx == 6) {
        bits = get_bf16_bits<6>(data);
      } else {
        bits = get_bf16_bits<7>(data);
      }
      amax = fmaxf(amax, fabsf(fmha::bf16_to_float(bits)));
    }
    return amax;
  }

  template <int BASE>
  static inline __device__ uint32_t pack_bf16_vec(uint4 const& data, float inv_scale) {
    float v0 = fmha::bf16_to_float(get_bf16_bits<BASE + 0>(data)) * inv_scale;
    float v1 = fmha::bf16_to_float(get_bf16_bits<BASE + 1>(data)) * inv_scale;
    float v2 = fmha::bf16_to_float(get_bf16_bits<BASE + 2>(data)) * inv_scale;
    float v3 = fmha::bf16_to_float(get_bf16_bits<BASE + 3>(data)) * inv_scale;
    float v4 = fmha::bf16_to_float(get_bf16_bits<BASE + 4>(data)) * inv_scale;
    float v5 = fmha::bf16_to_float(get_bf16_bits<BASE + 5>(data)) * inv_scale;
    float v6 = fmha::bf16_to_float(get_bf16_bits<BASE + 6>(data)) * inv_scale;
    float v7 = fmha::bf16_to_float(get_bf16_bits<BASE + 7>(data)) * inv_scale;
    return fmha::float8_to_e2m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
  }

  template <typename Smem_tile>
  inline __device__ void load(Smem_tile&) {
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      int const row = row_ + ii * ROWS_PER_LDG;
      preds_[ii] = row < min(static_cast<int>(ROWS), actual_seqlen_) &&
                   col_offset_elements_ < VALID_COLS;

      uint32_t packed[4];
      uint4 q_vec[4];
      float local_scale[2];
#pragma unroll
      for (int vec = 0; vec < 4; ++vec) {
        q_vec[vec] =
            load_q_bf16_vec(row, col_offset_elements_ + vec * 8);
      }
#pragma unroll
      for (int local_group = 0; local_group < 2; ++local_group) {
        uint4 const lo = q_vec[2 * local_group + 0];
        uint4 const hi = q_vec[2 * local_group + 1];
        float const amax = fmaxf(amax_bf16_vec(lo), amax_bf16_vec(hi));
        local_scale[local_group] = amax > 0.f ? amax / 6.f : 1.f;
      }

      float const local_mma_scale = fmaxf(local_scale[0], local_scale[1]);
      float const inv_local_mma_scale =
          local_mma_scale > 0.f ? 1.f / local_mma_scale : 0.f;
      packed[0] = pack_bf16_vec<0>(q_vec[0], inv_local_mma_scale);
      packed[1] = pack_bf16_vec<0>(q_vec[1], inv_local_mma_scale);
      packed[2] = pack_bf16_vec<0>(q_vec[2], inv_local_mma_scale);
      packed[3] = pack_bf16_vec<0>(q_vec[3], inv_local_mma_scale);

      float const peer_mma_scale = __shfl_xor_sync(0xffffffff, local_mma_scale, 1);
      int const local_group_offset =
          (col_offset_elements_ % COLS) / Traits::NVFP4_SCALE_VEC_SIZE;
      if (local_group_offset == 0) {
        scale_fetch_[ii] =
            fmha::make_ue4m3_scale_reg(local_mma_scale, peer_mma_scale,
                                       local_mma_scale, peer_mma_scale);
      } else {
        scale_fetch_[ii] =
            fmha::make_ue4m3_scale_reg(peer_mma_scale, local_mma_scale,
                                       peer_mma_scale, local_mma_scale);
      }
      fetch_[ii] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
  }

  inline __device__ void move(int const steps = 1) {
    q_row_offset_ += static_cast<int>(ROWS) * steps;
    actual_seqlen_ -= static_cast<int>(ROWS) * steps;
  }

  inline __device__ void move_col(int const steps = 1) {
    col_offset_elements_ += COLS * steps;
  }

  inline __device__ void rewind_col(int const steps) {
    col_offset_elements_ -= COLS * steps;
  }

  inline __device__ void reset() {
    q_row_offset_ = q_row_offset_init_;
    actual_seqlen_ = actual_seqlen_init_;
    col_offset_elements_ = col_offset_elements_init_;
  }

  int64_t params_q_stride_in_bytes_;
  int params_d_;
  int q_token_base_;
  int q_row_offset_;
  int q_row_offset_init_;
  int num_grouped_heads_;
  int q_head_base_;
  char const* q_ptr_;
  uint32_t preds_[LDGS];
  uint4 fetch_[LDGS];
  uint32_t scale_fetch_[LDGS];
  int row_;
  int col_offset_elements_;
  int col_offset_elements_init_;
  int actual_seqlen_;
  int actual_seqlen_init_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Shape [B, S, 2, H, D] where S can be variable sequence length.
template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bits per element.
    int BITS_PER_ELEMENT_,
    // The number of rows of Q, K or V loaded by this tile.
    int ROWS_,
    // The number of columns (padded, e.g 64).
    int COLS,
    // The actual number of columns (unpadded, e.g 40)
    int VALID_COLS_,
    // Do we use LDGSTS?
    bool USE_LDGSTS_,
    // Are attention heads interleaved? (Not used)
    bool HEADS_INTERLEAVED,
    // The number of matrices (Not used)
    int NUM_MATS = 2,
    // Is sliding window attention used ?
    bool SLIDING_WINDOW_ATTENTION = false>
struct Gmem_tile_contiguous_kv {
  // The size of each LDG.
  enum { BYTES_PER_LDG = 16 };

  // The number of bits/bytes of element
  enum { BITS_PER_ELEMENT = BITS_PER_ELEMENT_ };

  enum { BYTES_PER_ELEMENT = BITS_PER_ELEMENT_ >= 8 ? BITS_PER_ELEMENT_ / 8 : 0 };

  // The size of a row in bytes.
  enum { BYTES_PER_ROW = (COLS * BITS_PER_ELEMENT + 7) / 8 };

  // The number of threads to load a "row" of the matrix.
  enum { THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDG };

  // The valid size of a row in bytes (without paddings).
  enum { VALID_COLS = VALID_COLS_ };

  // The amount of bytes that are valid per row.
  enum { VALID_BYTES_PER_ROW = (VALID_COLS * BITS_PER_ELEMENT + 7) / 8 };

  static inline __host__ __device__ int64_t bytes_for_elements(int64_t elements) {
    return elements * BITS_PER_ELEMENT / 8;
  }

  // The number of "rows" loaded per LDG.
  enum { ROWS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };

  // The number of rows.
  enum { ROWS = ROWS_ };

  // The number of LDGs needed to load a chunk of the Q matrix.
  enum { LDGS = fmha::Div_up<ROWS, ROWS_PER_LDG>::VALUE };

  // The number of predicate registers.
  enum { PRED_REGS = fmha::Compute_number_of_pred_regs<LDGS>::VALUE };

  // Is it Hopper?
  enum {
    IS_HOPPER = std::is_same<typename Traits::Gpu_arch, typename fmha::Hopper>::value == true
  };

  // Make sure we use a single register to store predicates. Do not throw for Hopper for now.
  static_assert(!USE_LDGSTS_ || PRED_REGS == 1 || IS_HOPPER, "");

  // We do not use LDGSTS (for the moment).
  enum { USE_LDGSTS = USE_LDGSTS_ };

  // Ctor for bert::Fused_multihead_attention_params_v2 class
  template <typename Block_info>
  inline __device__ Gmem_tile_contiguous_kv(bert::Fused_multihead_attention_params_v2 const& params,
                                            int qkv_offset,  // q = 0, k = 1, v = 2.
                                            Block_info const& binfo, int tidx,
                                            int cta_row_offset = 0, int cta_col_offset_in_bytes = 0)
      : Gmem_tile_contiguous_kv(params.kv_ptr, params.k_stride_in_bytes, params.h_kv,
                                params.h_q_per_kv, qkv_offset, binfo, tidx, cta_row_offset,
                                cta_col_offset_in_bytes) {}

  // Ctor.
  template <typename Block_info>
  inline __device__ Gmem_tile_contiguous_kv(void* kv_ptr, size_t kv_stride_in_bytes,
                                            int num_kv_heads, int head_group_size, int qkv_offset,
                                            Block_info const& binfo, int tidx,
                                            int cta_row_offset = 0, int cta_col_offset_in_bytes = 0)
      : params_kv_stride_in_bytes_(kv_stride_in_bytes),
        actual_seqlen_(binfo.actual_kv_seqlen),
        kv_ptr_(reinterpret_cast<char*>(kv_ptr)) {
    // Compute the position in the sequence (within the CTA for the moment).
    int row = tidx / THREADS_PER_ROW;
    // Compute the position of the thread in the row.
    int col = tidx % THREADS_PER_ROW;

    // We must store the value to update the predicates in "load".
    row_ = row;
    // Do not load/store if the thread is in the padded area
    col_in_bytes_ = cta_col_offset_in_bytes + col * BYTES_PER_LDG;

    // The row offset in the batched GEMM.
    int64_t row_offset = (int64_t)(row + cta_row_offset) * params_kv_stride_in_bytes_;
    // [b, s, 2, h_kv, d].
    int64_t idx =
        (binfo.sum_s_kv * 2 + qkv_offset - 1) * num_kv_heads + (binfo.bidh / head_group_size);

    // Assemble the final pointer.
    kv_ptr_ += row_offset + idx * VALID_BYTES_PER_ROW + col_in_bytes_;

    // Take the CTA offset to modify the sequence length.
    actual_seqlen_ -= cta_row_offset;

    // Set the initial seq_len and qkv_offset in case of reinterating
    actual_seqlen_init_ = actual_seqlen_;
    kv_ptr_init_ = kv_ptr_;
  }

  // Store data to shared memory.
  template <typename Smem_tile>
  inline __device__ void commit(Smem_tile& smem_tile) {
    if (!USE_LDGSTS) {
      smem_tile.store(fetch_);
    }
  }

  // Load data from memory.
  template <typename Smem_tile>
  inline __device__ void load(Smem_tile& smem_tile) {
    uint32_t preds[LDGS];
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      preds[ii] = row_ + ii * (int)ROWS_PER_LDG < min((int)ROWS, actual_seqlen_);
      preds[ii] &= col_in_bytes_ < VALID_BYTES_PER_ROW;
    }

    // Prepare the load pointers.
    void const* ptrs[LDGS];
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      ptrs[ii] = kv_ptr_ + (int64_t)ii * ROWS_PER_LDG * params_kv_stride_in_bytes_;
    }

    // Trigger LDGSTS or the LDGs.
    // The predicates protect against out-of-bound access in rows and cols
    Ldgsts_helper<USE_LDGSTS>::load(this, smem_tile, ptrs, preds);
  }

  // Load data from memory.
  inline __device__ void load() {
    uint32_t preds[LDGS];
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      preds[ii] = row_ + ii * (int)ROWS_PER_LDG < min((int)ROWS, actual_seqlen_);
    }

    // Prepare the load pointers.
    void const* ptrs[LDGS];
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      ptrs[ii] = kv_ptr_ + (int64_t)ii * ROWS_PER_LDG * params_kv_stride_in_bytes_;
    }

    // Trigger the LDGs.
    if (col_in_bytes_ < VALID_BYTES_PER_ROW) {
      fmha::pack_predicates(preds_, preds);
      fmha::ldg(fetch_, ptrs, preds_);
    } else {
#pragma unroll
      for (int ii = 0; ii < LDGS; ++ii) {
        fetch_[ii] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  }

  // Move the pointer to the next row location.
  inline __device__ void move(int const steps = 1) {
    kv_ptr_ += (int64_t)ROWS * params_kv_stride_in_bytes_ * steps;
    actual_seqlen_ -= (int)ROWS * steps;
  }

  // Move the pointer to the next row location by the offset (not step).
  inline __device__ void move_by_offset(int const offset) {
    kv_ptr_ = kv_ptr_init_ + (int64_t)offset * params_kv_stride_in_bytes_;
    actual_seqlen_ = actual_seqlen_init_ - (int)offset;
  }

  // Move the pointer to the next column location
  inline __device__ void move_col(int const steps = 1) {
    kv_ptr_ += (int64_t)BYTES_PER_ROW * steps;
    // Update col_in_bytes_ to ensure load predicates work
    col_in_bytes_ += THREADS_PER_ROW * BYTES_PER_LDG * steps;
  }

  inline __device__ void reset() {
    kv_ptr_ = kv_ptr_init_;
    actual_seqlen_ = actual_seqlen_init_;
  }

  // Rewind the pointer back to previous column location
  inline __device__ void rewind_col(int const steps) {
    kv_ptr_ -= BYTES_PER_ROW * steps;
    // Update col_in_bytes_ to ensure load predicates work
    col_in_bytes_ -= THREADS_PER_ROW * BYTES_PER_LDG * steps;
  }

  inline __device__ void move_to(int const step) {
    kv_ptr_ = kv_ptr_init_ + (int64_t)ROWS * params_kv_stride_in_bytes_ * step;
    actual_seqlen_ = actual_seqlen_init_ - (int)ROWS * step;
  }

  // Store data to memory.
  inline __device__ void store(uint4 const (&data)[LDGS]) {
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      char* ptr = kv_ptr_ + (int64_t)ii * ROWS_PER_LDG * params_kv_stride_in_bytes_;
      if (((row_ + ii * ROWS_PER_LDG) < min(ROWS, actual_seqlen_)) &&
          col_in_bytes_ < VALID_BYTES_PER_ROW /*TODO: double check*/) {
        fmha::stg(ptr, data[ii]);
      }
    }
  }

  // The stride between rows for the QKV matrice.
  int64_t params_kv_stride_in_bytes_;
  // The pointer.
  char* kv_ptr_;
  char* kv_ptr_init_;
  // The register to store predicates.
  uint32_t preds_[PRED_REGS];
  // The fetch registers.
  uint4 fetch_[LDGS];
  // Keep track of the row and col the thread is processing as we move the tile.
  int row_;
  int col_in_bytes_;
  // The sequence length.
  int actual_seqlen_;
  int actual_seqlen_init_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// We expect the paged KV layout to be blocks of indices with shape of [B, 2, Blocks_per_Seq],
// and the indice tells the memory distance to the pool ptr in global memory.

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bits per element.
    int BITS_PER_ELEMENT_,
    // The number of rows of Q, K or V loaded by this tile.
    int ROWS_,
    // The number of columns (padded, e.g 64).
    int COLS,
    // The actual number of columns (unpadded, e.g 40)
    int VALID_COLS_,
    // Do we use LDGSTS?
    bool USE_LDGSTS_,
    // Are attention heads interleaved? (not used)
    bool HEADS_INTERLEAVED = false,
    // The number of matrices (not used)
    int NUM_MATS = 2,
    // Is sliding window attention used ?
    bool SLIDING_WINDOW_ATTENTION_ = false>
struct Gmem_tile_paged_kv {
  // The size of each LDG.
  enum { BYTES_PER_LDG = 16 };

  // The number of bits/bytes of element
  enum { BITS_PER_ELEMENT = BITS_PER_ELEMENT_ };

  enum { BYTES_PER_ELEMENT = BITS_PER_ELEMENT_ >= 8 ? BITS_PER_ELEMENT_ / 8 : 0 };

  // The size of a row in bytes.
  enum { BYTES_PER_ROW = (COLS * BITS_PER_ELEMENT + 7) / 8 };

  // The number of threads to load a "row" of the matrix.
  enum { THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDG };

  // The number of logical elements loaded per thread.
  enum { ELEMENTS_PER_LDG = BYTES_PER_LDG * 8 / BITS_PER_ELEMENT };

  // The valid size of a row in bytes (without paddings).
  enum { VALID_COLS = VALID_COLS_ };

  // The amount of bytes that are valid per row.
  enum { VALID_BYTES_PER_ROW = (VALID_COLS * BITS_PER_ELEMENT + 7) / 8 };

  static inline __host__ __device__ int64_t bytes_for_elements(int64_t elements) {
    return elements * BITS_PER_ELEMENT / 8;
  }

  // The number of "rows" loaded per LDG.
  enum { ROWS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };

  // The number of rows.
  enum { ROWS = ROWS_ };

  // The number of LDGs needed to load a chunk of the Q matrix.
  enum { LDGS = fmha::Div_up<ROWS, ROWS_PER_LDG>::VALUE };

  // The number of predicate registers.
  enum { PRED_REGS = fmha::Compute_number_of_pred_regs<LDGS>::VALUE };

  // Is sliding window attention used ?
  enum { SLIDING_WINDOW_ATTENTION = SLIDING_WINDOW_ATTENTION_ };

  // Is it Hopper?
  enum {
    IS_HOPPER = std::is_same<typename Traits::Gpu_arch, typename fmha::Hopper>::value == true
  };

  // Make sure we use a single register to store predicates. Do not throw for Hopper for now.
  static_assert(!USE_LDGSTS_ || PRED_REGS == 1 || IS_HOPPER, "");

  // We do not use LDGSTS (for the moment).
  enum { USE_LDGSTS = USE_LDGSTS_ };

  // Ctor.
  template <typename Block_info>
  inline __device__ Gmem_tile_paged_kv(bert::Fused_multihead_attention_params_v2 const& params,
                                       int qkv_offset,  // q = 0, k = 1, v = 2.
                                       Block_info const& binfo, int tidx, int cta_row_offset = 0,
                                       int cta_col_offset_in_bytes = 0)
      : actual_seqlen_(binfo.actual_seqlen),
        past_seqlen_(binfo.actual_seqlen - binfo.actual_q_seqlen),
        sliding_window_size_(params.sliding_window_size),
        paged_kv_log2_block_size_(params.paged_kv_cache.mTokensPerBlockLog2),
        paged_kv_block_pool_ptr_(reinterpret_cast<char*>(params.paged_kv_cache.mPoolPtr)),
        paged_kv_global_block_offsets_(params.paged_kv_cache.mBlockOffsets),
        params_kv_block_size_in_bytes_(params.paged_kv_cache.mBytesPerBlock),
        qkv_offset_(qkv_offset),
        nvfp4_scale_ptr_(
            reinterpret_cast<char const*>(qkv_offset == 1 ? params.k_scale_ptr : params.v_scale_ptr)),
        nvfp4_scale_page_stride_in_bytes_(qkv_offset == 1 ? params.k_scale_page_stride_in_bytes
                                                          : params.v_scale_page_stride_in_bytes),
        nvfp4_scale_head_stride_in_bytes_(qkv_offset == 1 ? params.k_scale_head_stride_in_bytes
                                                          : params.v_scale_head_stride_in_bytes),
        nvfp4_scale_token_stride_in_bytes_(qkv_offset == 1 ? params.k_scale_token_stride_in_bytes
                                                           : params.v_scale_token_stride_in_bytes),
        nvfp4_scale_vec_stride_in_bytes_(qkv_offset == 1 ? params.k_scale_vec_stride_in_bytes
                                                         : params.v_scale_vec_stride_in_bytes),
        nvfp4_v_cache_uses_pv_layout_(params.nvfp4_v_cache_uses_pv_layout),
        row_offset_(0),
        col_tile_offset_in_bytes_(cta_col_offset_in_bytes) {
    // Handle Paged KV with shape [S, Dh], by offsetting it to the target batch.
    int32_t const paged_kv_block_offset =
        (binfo.bidb * 2 + qkv_offset - 1) * params.paged_kv_cache.mMaxBlocksPerSeq;
    paged_kv_global_block_offsets_ += paged_kv_block_offset;

    // Compute the position in the sequence (within the CTA for the moment).
    int row = tidx / THREADS_PER_ROW;
    // Compute the position of the thread in the row.
    int col = tidx % THREADS_PER_ROW;

    // We must store the value to update the predicates in "load".
    row_ = row;
    // Do not load/store if the thread is in the padded area
    col_in_bytes_ = cta_col_offset_in_bytes + col * BYTES_PER_LDG;

    // The head stride in bytes.
    int64_t head_stride_in_bytes =
        qkv_offset == 1 ? params.k_stride_in_bytes_2 : params.v_stride_in_bytes_2;
    // In the grouped-M path blockIdx.y is already the KV head. Otherwise
    // blockIdx.y is a Q head and must be mapped back to its KV head.
    int const kv_head_id =
        params.num_grouped_heads > 1 ? binfo.bidh : binfo.bidh / params.h_q_per_kv;
    head_offset_in_bytes_ = kv_head_id * head_stride_in_bytes;
    nvfp4_scale_head_offset_in_bytes_ = kv_head_id * nvfp4_scale_head_stride_in_bytes_;

    // The token stride in bytes.
    token_stride_in_bytes_ = qkv_offset == 1 ? params.k_stride_in_bytes : params.v_stride_in_bytes;

    // Take the CTA offset to modify the sequence length.
    // Actually we don't need that for flash attention.
    actual_seqlen_ -= cta_row_offset;
  }

  // Store data to shared memory.
  template <typename Smem_tile>
  inline __device__ void commit(Smem_tile& smem_tile) {
    if constexpr (std::is_same<Traits, fmha::Blackwell_mma_nvf4_fp32_traits>::value &&
                  BITS_PER_ELEMENT == 4) {
      if constexpr (!USE_LDGSTS) {
        if (qkv_offset_ == 1 && nvfp4_scale_ptr_ != nullptr) {
          return;
        }
        if (qkv_offset_ == 2 && nvfp4_scale_ptr_ != nullptr &&
            nvfp4_v_cache_uses_pv_layout_) {
          return;
        }
        if (qkv_offset_ == 2 && nvfp4_scale_ptr_ != nullptr) {
          smem_tile.store(fetch_);
          return;
        }
      }
    }
    if (!USE_LDGSTS) {
      smem_tile.store(fetch_);
    }
  }

  // Load data from memory.
  template <typename Smem_tile>
  inline __device__ void load(Smem_tile& smem_tile) {
    if constexpr (std::is_same<Traits, fmha::Blackwell_mma_nvf4_fp32_traits>::value &&
                  BITS_PER_ELEMENT == 4) {
      if constexpr (Has_k_data_write_buffer<Smem_tile>::value) {
        if (qkv_offset_ == 1 && nvfp4_scale_ptr_ != nullptr) {
          smem_tile.set_nvfp4_scale_metadata(
              nvfp4_scale_ptr_, nvfp4_scale_head_offset_in_bytes_,
              nvfp4_scale_page_stride_in_bytes_, nvfp4_scale_token_stride_in_bytes_,
              nvfp4_scale_vec_stride_in_bytes_, paged_kv_global_block_offsets_,
              paged_kv_log2_block_size_, row_offset_, col_tile_offset_in_bytes_, actual_seqlen_);
          load_k_scale(smem_tile);
          load_k_data(smem_tile);
          return;
        }
      }
      if constexpr (Has_v_scale_write_buffer<Smem_tile>::value) {
        if (qkv_offset_ == 2 && nvfp4_scale_ptr_ != nullptr) {
          if (nvfp4_v_cache_uses_pv_layout_) {
            load_pv_layout_v(smem_tile);
          } else {
            load_reblocked_v(smem_tile);
          }
          return;
        }
      }
    }

    // Prepare the predicates.
    uint32_t preds[LDGS];
    // Prepare the load pointers.
    void const* ptrs[LDGS];

    // Offset for the new paged kv pointer.
    uint64_t const head_col_in_bytes = head_offset_in_bytes_ + col_in_bytes_;

// Update paged_kv ptr for each LDG (reuse is possible).
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      int row_idx = row_ + ii * (int)ROWS_PER_LDG;
      int paged_kv_block_idx = (row_idx >> paged_kv_log2_block_size_);
      char const* local_kv_ptr = reinterpret_cast<char*>(
          paged_kv_block_pool_ptr_ +
          params_kv_block_size_in_bytes_ * paged_kv_global_block_offsets_[paged_kv_block_idx]);

      // Predicates.
      // TODO: do we need to make sure row_idx < ROWS ?
      preds[ii] = row_idx < actual_seqlen_;
      preds[ii] &= col_in_bytes_ < VALID_BYTES_PER_ROW;

      // Pointers.
      int row_idx_in_block = row_idx & ((1 << paged_kv_log2_block_size_) - 1);
      ptrs[ii] =
          local_kv_ptr + head_col_in_bytes + (int64_t)row_idx_in_block * token_stride_in_bytes_;

    }

    // Trigger LDGSTS or the LDGs.
    // The predicates protect against out-of-bound access in rows and cols
    Ldgsts_helper<USE_LDGSTS>::load(this, smem_tile, ptrs, preds);
  }

  inline __device__ bool get_nvfp4_row_ptrs(int token_row, char const*& kv_row_ptr,
                                            char const*& scale_head_ptr,
                                            int& row_in_page) const {
    if (token_row < 0 || token_row >= actual_seqlen_) {
      kv_row_ptr = nullptr;
      scale_head_ptr = nullptr;
      row_in_page = 0;
      return false;
    }
    int const page_idx = token_row >> paged_kv_log2_block_size_;
    row_in_page = token_row & ((1 << paged_kv_log2_block_size_) - 1);
    int32_t const physical_page = paged_kv_global_block_offsets_[page_idx];
    // The data block table is expanded for separate K/V blocks, while scale
    // tensors are passed as separate logical-page tensors.
    int32_t const scale_page = physical_page / 2;
    char const* local_kv_ptr = reinterpret_cast<char const*>(
        paged_kv_block_pool_ptr_ +
        params_kv_block_size_in_bytes_ * static_cast<int64_t>(physical_page));
    kv_row_ptr =
        local_kv_ptr + head_offset_in_bytes_ + static_cast<int64_t>(row_in_page) * token_stride_in_bytes_;
    scale_head_ptr =
        nvfp4_scale_ptr_ + static_cast<int64_t>(scale_page) * nvfp4_scale_page_stride_in_bytes_ +
        nvfp4_scale_head_offset_in_bytes_;
    return true;
  }

  inline __device__ uint8_t load_v_original_scale_byte(char const* scale_head_ptr,
                                                       int row_in_page, int col) const {
    int const scale_idx = col / fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE;
    int const scale_dim = VALID_COLS / fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE;
    int const scale_group = scale_dim / 4;
    int const swizzled_t = (row_in_page / 4) * 4 + (scale_idx / scale_group);
    int const swizzled_s = (scale_idx % scale_group) * 4 + (row_in_page % 4);
    char const* scale_ptr =
        scale_head_ptr + static_cast<int64_t>(swizzled_t) * nvfp4_scale_token_stride_in_bytes_ +
        static_cast<int64_t>(swizzled_s) * nvfp4_scale_vec_stride_in_bytes_;
    uint8_t scale_byte;
    fmha::ldg(scale_byte, scale_ptr);
    return scale_byte;
  }

  inline __device__ uint8_t load_k_scale_byte(int token_row, int scale_group) const {
    if (token_row < 0 || token_row >= actual_seqlen_ || scale_group < 0) {
      return fmha::float_to_e4m3_byte(1.f);
    }
    constexpr int SCALE_GROUPS_PER_ROW =
        VALID_COLS / fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE;
    if (scale_group >= SCALE_GROUPS_PER_ROW) {
      return fmha::float_to_e4m3_byte(1.f);
    }
    int const page_idx = token_row >> paged_kv_log2_block_size_;
    int const row_in_page = token_row & ((1 << paged_kv_log2_block_size_) - 1);
    int32_t const physical_page = paged_kv_global_block_offsets_[page_idx];
    int32_t const scale_page = physical_page / 2;
    char const* scale_ptr =
        nvfp4_scale_ptr_ + static_cast<int64_t>(scale_page) * nvfp4_scale_page_stride_in_bytes_ +
        nvfp4_scale_head_offset_in_bytes_ +
        static_cast<int64_t>(row_in_page) * nvfp4_scale_token_stride_in_bytes_ +
        static_cast<int64_t>(scale_group) * nvfp4_scale_vec_stride_in_bytes_;
    uint8_t scale_byte;
    fmha::ldg(scale_byte, scale_ptr);
    return scale_byte;
  }

  inline __device__ uint32_t load_k_scale_word4(int token_row, int scale_group) const {
    uint8_t const one = fmha::float_to_e4m3_byte(1.f);
    uint32_t const one_word = static_cast<uint32_t>(one) |
                              (static_cast<uint32_t>(one) << 8) |
                              (static_cast<uint32_t>(one) << 16) |
                              (static_cast<uint32_t>(one) << 24);
    char const* scale_ptr;
    if (!get_k_scale_word4_ptr(token_row, scale_group, scale_ptr)) {
      return one_word;
    }
    uint32_t scale_word;
    fmha::ldg(scale_word, scale_ptr);
    return scale_word;
  }

  inline __device__ bool get_k_scale_word4_ptr(int token_row, int scale_group,
                                               char const*& scale_ptr) const {
    return get_k_scale_ptr(token_row, scale_group, 4, scale_ptr);
  }

  inline __device__ bool get_k_scale_vec16_ptr(int token_row, int scale_group,
                                               char const*& scale_ptr) const {
    if ((scale_group & 15) != 0) {
      scale_ptr = nullptr;
      return false;
    }
    return get_k_scale_ptr(token_row, scale_group, 16, scale_ptr);
  }

  inline __device__ bool get_k_scale_ptr(int token_row, int scale_group, int bytes,
                                         char const*& scale_ptr) const {
    scale_ptr = nullptr;
    if (token_row < 0 || token_row >= actual_seqlen_ || scale_group < 0 ||
        nvfp4_scale_vec_stride_in_bytes_ != 1) {
      return false;
    }
    constexpr int SCALE_GROUPS_PER_ROW =
        VALID_COLS / fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE;
    if (scale_group + bytes - 1 >= SCALE_GROUPS_PER_ROW) {
      return false;
    }
    int const page_idx = token_row >> paged_kv_log2_block_size_;
    int const row_in_page = token_row & ((1 << paged_kv_log2_block_size_) - 1);
    int32_t const physical_page = paged_kv_global_block_offsets_[page_idx];
    int32_t const scale_page = physical_page / 2;
    scale_ptr =
        nvfp4_scale_ptr_ + static_cast<int64_t>(scale_page) * nvfp4_scale_page_stride_in_bytes_ +
        nvfp4_scale_head_offset_in_bytes_ +
        static_cast<int64_t>(row_in_page) * nvfp4_scale_token_stride_in_bytes_ +
        static_cast<int64_t>(scale_group) * nvfp4_scale_vec_stride_in_bytes_;
    return true;
  }

  template <typename Smem_tile>
  inline __device__ void load_nvfp4_row_major_data(Smem_tile& smem_tile) {
    if constexpr (USE_LDGSTS) {
      uint32_t preds[LDGS];
      void const* ptrs[LDGS];
      bool can_use_ldgsts = true;

#pragma unroll
      for (int ii = 0; ii < LDGS; ++ii) {
        int const token_row = row_ + ii * static_cast<int>(ROWS_PER_LDG);
        char const* kv_row_ptr;
        char const* scale_head_ptr;
        int row_in_page;
        bool const valid_row =
            get_nvfp4_row_ptrs(token_row, kv_row_ptr, scale_head_ptr, row_in_page);

        bool const full_vector =
            valid_row && col_in_bytes_ >= 0 &&
            col_in_bytes_ + BYTES_PER_LDG <= VALID_BYTES_PER_ROW;
        preds[ii] = full_vector;
        ptrs[ii] = full_vector ? kv_row_ptr + col_in_bytes_ : nullptr;

        if (valid_row && col_in_bytes_ >= 0 && col_in_bytes_ < VALID_BYTES_PER_ROW &&
            col_in_bytes_ + BYTES_PER_LDG > VALID_BYTES_PER_ROW) {
          can_use_ldgsts = false;
        }
      }

      if (can_use_ldgsts) {
        fmha::pack_predicates(preds_, preds);
        smem_tile.store(ptrs, preds_);
        return;
      }
    }

#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      int const token_row = row_ + ii * static_cast<int>(ROWS_PER_LDG);
      char const* kv_row_ptr;
      char const* scale_head_ptr;
      int row_in_page;
      bool const valid_row =
          get_nvfp4_row_ptrs(token_row, kv_row_ptr, scale_head_ptr, row_in_page);

      uint4 data = make_uint4(0u, 0u, 0u, 0u);
      if (valid_row && col_in_bytes_ >= 0 && col_in_bytes_ < VALID_BYTES_PER_ROW) {
        if (col_in_bytes_ + BYTES_PER_LDG <= VALID_BYTES_PER_ROW) {
          fmha::ldg(data, kv_row_ptr + col_in_bytes_);
        } else {
          uint32_t words[4] = {0u, 0u, 0u, 0u};
#pragma unroll
          for (int byte = 0; byte < BYTES_PER_LDG; ++byte) {
            int64_t const global_byte = col_in_bytes_ + byte;
            uint8_t value = 0u;
            if (global_byte < VALID_BYTES_PER_ROW) {
              fmha::ldg(value, kv_row_ptr + global_byte);
            }
            words[byte / 4] |= static_cast<uint32_t>(value) << (8 * (byte & 3));
          }
          data = make_uint4(words[0], words[1], words[2], words[3]);
        }
      }
      fetch_[ii] = data;
    }
    smem_tile.store(fetch_);
  }

  template <typename Smem_tile>
  inline __device__ void load_k_scale(Smem_tile& smem_tile) {
#if defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_K_SCALE_STAGING)
    return;
#else
    constexpr int SCALE_GROUPS_PER_TILE_ROW =
        COLS / fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE;
    int const col_tile_offset_elements =
        static_cast<int>(col_tile_offset_in_bytes_ * 8 / BITS_PER_ELEMENT);
    int const col_group_offset =
        col_tile_offset_elements / fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE;

    if constexpr (SCALE_GROUPS_PER_TILE_ROW == 4) {
      if (nvfp4_scale_vec_stride_in_bytes_ == 1) {
#pragma unroll 1
        for (int logical_row = threadIdx.x; logical_row < ROWS;
             logical_row += Cta_tile::THREADS_PER_CTA) {
          int const token_row = row_offset_ + logical_row;
          if constexpr (USE_LDGSTS) {
            char const* scale_ptr;
            if (get_k_scale_vec16_ptr(token_row, col_group_offset, scale_ptr)) {
              fmha::ldgsts128(smem_tile.k_scale_row_smem_ptr(logical_row), scale_ptr);
            } else {
              smem_tile.store_k_scale_word(
                  logical_row, load_k_scale_word4(token_row, col_group_offset));
            }
          } else {
            smem_tile.store_k_scale_word(
                logical_row, load_k_scale_word4(token_row, col_group_offset));
          }
        }
        return;
      }
    }

#pragma unroll 1
    for (int scale_idx = threadIdx.x; scale_idx < ROWS * SCALE_GROUPS_PER_TILE_ROW;
         scale_idx += Cta_tile::THREADS_PER_CTA) {
      int const logical_row = scale_idx / SCALE_GROUPS_PER_TILE_ROW;
      int const local_scale_group =
          scale_idx - logical_row * SCALE_GROUPS_PER_TILE_ROW;
      int const token_row = row_offset_ + logical_row;
      int const scale_group = col_group_offset + local_scale_group;
      smem_tile.store_k_scale(logical_row, local_scale_group,
                              load_k_scale_byte(token_row, scale_group));
    }
#endif
  }

  template <typename Smem_tile>
  inline __device__ void load_k_data(Smem_tile& smem_tile) {
    load_nvfp4_row_major_data(smem_tile);
  }

  inline __device__ uint8_t load_v_pv_layout_scale_byte(int token_group_start,
                                                        int col) const {
    if (token_group_start < 0 || token_group_start >= actual_seqlen_ ||
        col < 0 || col >= VALID_COLS) {
      return fmha::float_to_e4m3_byte(1.f);
    }
    int const tokens_per_page = 1 << paged_kv_log2_block_size_;
    if (tokens_per_page < fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE) {
      return fmha::float_to_e4m3_byte(1.f);
    }
    int const page_idx = token_group_start >> paged_kv_log2_block_size_;
    int const row_in_page = token_group_start & (tokens_per_page - 1);
    int const token_group_in_page =
        row_in_page / fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE;
    int const scale_dim = VALID_COLS / fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE;
    int const linear_scale = token_group_in_page * VALID_COLS + col;
    int const scale_row = linear_scale / scale_dim;
    int const scale_vec = linear_scale - scale_row * scale_dim;
    int32_t const physical_page = paged_kv_global_block_offsets_[page_idx];
    // The data block table is expanded for separate K/V blocks, while scale
    // tensors are passed as separate logical-page tensors.
    int32_t const scale_page = physical_page / 2;
    char const* scale_ptr =
        nvfp4_scale_ptr_ + static_cast<int64_t>(scale_page) * nvfp4_scale_page_stride_in_bytes_ +
        nvfp4_scale_head_offset_in_bytes_ +
        static_cast<int64_t>(scale_row) * nvfp4_scale_token_stride_in_bytes_ +
        static_cast<int64_t>(scale_vec) * nvfp4_scale_vec_stride_in_bytes_;
    uint8_t scale_byte;
    fmha::ldg(scale_byte, scale_ptr);
    return scale_byte;
  }

  template <int IDX>
  static inline __device__ uint8_t get_u8(uint4 const& data) {
    static_assert(IDX >= 0 && IDX < 16, "");
    uint32_t word;
    if constexpr (IDX < 4) {
      word = data.x;
    } else if constexpr (IDX < 8) {
      word = data.y;
    } else if constexpr (IDX < 12) {
      word = data.z;
    } else {
      word = data.w;
    }
    return static_cast<uint8_t>((word >> ((IDX & 3) * 8)) & 0xffu);
  }

  template <int IDX, typename Smem_tile>
  static inline __device__ void store_v_scale_vec16_bytes(Smem_tile& smem_tile,
                                                          int logical_col,
                                                          int scale_group,
                                                          uint4 const& scale_vec) {
    smem_tile.store_v_scale(logical_col + IDX, scale_group, get_u8<IDX>(scale_vec));
    if constexpr (IDX + 1 < 16) {
      store_v_scale_vec16_bytes<IDX + 1>(smem_tile, logical_col, scale_group, scale_vec);
    }
  }

  inline __device__ bool load_v_pv_layout_scale_vec16(int token_group_start,
                                                      int col, uint4& scale_vec) const {
    scale_vec = make_uint4(0u, 0u, 0u, 0u);
    if (token_group_start < 0 || token_group_start >= actual_seqlen_ ||
        col < 0 || col + 15 >= VALID_COLS) {
      return false;
    }
    int const tokens_per_page = 1 << paged_kv_log2_block_size_;
    if (tokens_per_page < fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE ||
        nvfp4_scale_vec_stride_in_bytes_ != 1) {
      return false;
    }
    int const page_idx = token_group_start >> paged_kv_log2_block_size_;
    int const row_in_page = token_group_start & (tokens_per_page - 1);
    int const token_group_in_page =
        row_in_page / fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE;
    int const scale_dim = VALID_COLS / fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE;
    int const linear_scale = token_group_in_page * VALID_COLS + col;
    int const scale_row = linear_scale / scale_dim;
    int const scale_col = linear_scale - scale_row * scale_dim;
    if (scale_col + 15 >= scale_dim) {
      return false;
    }
    int32_t const physical_page = paged_kv_global_block_offsets_[page_idx];
    int32_t const scale_page = physical_page / 2;
    char const* scale_ptr =
        nvfp4_scale_ptr_ + static_cast<int64_t>(scale_page) * nvfp4_scale_page_stride_in_bytes_ +
        nvfp4_scale_head_offset_in_bytes_ +
        static_cast<int64_t>(scale_row) * nvfp4_scale_token_stride_in_bytes_ +
        static_cast<int64_t>(scale_col) * nvfp4_scale_vec_stride_in_bytes_;
    fmha::ldg(scale_vec, scale_ptr);
    return true;
  }

  inline __device__ float load_nvfp4_cache_value_from_row(char const* kv_row_ptr,
                                                          char const* scale_head_ptr,
                                                          int row_in_page, int col) const {
    if (col < 0 || col >= VALID_COLS) {
      return 0.f;
    }
    uint8_t packed;
    fmha::ldg(packed, kv_row_ptr + col / 2);
    uint8_t const nibble = (col & 1) ? ((packed >> 4) & 0x0fu) : (packed & 0x0fu);

    uint8_t const scale_byte = load_v_original_scale_byte(scale_head_ptr, row_in_page, col);
    return fmha::e2m1_to_float(nibble) * fmha::e4m3_byte_to_float(scale_byte);
  }

  inline __device__ float load_nvfp4_cache_value_no_global(int token_row, int col) const {
    char const* kv_row_ptr;
    char const* scale_head_ptr;
    int row_in_page;
    if (!get_nvfp4_row_ptrs(token_row, kv_row_ptr, scale_head_ptr, row_in_page)) {
      return 0.f;
    }
    return load_nvfp4_cache_value_from_row(kv_row_ptr, scale_head_ptr, row_in_page, col);
  }

  inline __device__ void compute_v_reblock_scale_pair(int token_group_start, int col,
                                                      uint8_t& sf0_byte,
                                                      uint8_t& sf1_byte) const {
    float amax0 = 0.f;
    float amax1 = 0.f;
#pragma unroll 1
    for (int ii = 0; ii < fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE; ++ii) {
      char const* kv_row_ptr;
      char const* scale_head_ptr;
      int row_in_page;
      bool const valid_row =
          get_nvfp4_row_ptrs(token_group_start + ii, kv_row_ptr, scale_head_ptr, row_in_page);
      if (valid_row && col >= 0 && col < VALID_COLS) {
        uint8_t original_pair;
        fmha::ldg(original_pair, kv_row_ptr + col / 2);
        uint8_t const original_scale_byte =
            load_v_original_scale_byte(scale_head_ptr, row_in_page, col);
        float const original_scale = fmha::e4m3_byte_to_float(original_scale_byte);
        float const val0 = fmha::e2m1_to_float(original_pair & 0x0fu) * original_scale;
        amax0 = fmaxf(amax0, fabsf(val0));
        if (col + 1 < VALID_COLS) {
          float const val1 =
              fmha::e2m1_to_float((original_pair >> 4) & 0x0fu) * original_scale;
          amax1 = fmaxf(amax1, fabsf(val1));
        }
      }
    }
    float const sf0 = amax0 > 0.f ? amax0 / 6.f : 1.f;
    float const sf1 = amax1 > 0.f ? amax1 / 6.f : 1.f;
    sf0_byte = fmha::float_to_e4m3_byte(sf0);
    sf1_byte = fmha::float_to_e4m3_byte(sf1);
    if (sf0_byte == 0) {
      sf0_byte = fmha::float_to_e4m3_byte(1.f);
    }
    if (sf1_byte == 0) {
      sf1_byte = fmha::float_to_e4m3_byte(1.f);
    }
  }

  template <typename Smem_tile>
  inline __device__ void load_reblocked_v(Smem_tile& smem_tile) {
    int const logical_col_base =
        static_cast<int>((col_in_bytes_ - col_tile_offset_in_bytes_) * 8 / BITS_PER_ELEMENT);
    int const global_col_base = static_cast<int>(col_in_bytes_ * 8 / BITS_PER_ELEMENT);
    int const col_tile_offset_elements =
        static_cast<int>(col_tile_offset_in_bytes_ * 8 / BITS_PER_ELEMENT);

#pragma unroll 1
    for (int pair_idx = threadIdx.x;
         pair_idx <
         (ROWS / fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE) * (COLS / 2);
         pair_idx += Cta_tile::THREADS_PER_CTA) {
      int const scale_group = pair_idx / (COLS / 2);
      int const logical_col = (pair_idx - scale_group * (COLS / 2)) * 2;
      int const token_group_start =
          row_offset_ +
          scale_group * fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE;
      int const col = col_tile_offset_elements + logical_col;
      uint8_t sf0_byte;
      uint8_t sf1_byte;
      compute_v_reblock_scale_pair(token_group_start, col, sf0_byte, sf1_byte);
      smem_tile.store_v_scale(logical_col, scale_group, sf0_byte);
      smem_tile.store_v_scale(logical_col + 1, scale_group, sf1_byte);
    }

    __syncthreads();

#pragma unroll 1
    for (int ii = 0; ii < LDGS; ++ii) {
      int const row_idx = row_ + ii * static_cast<int>(ROWS_PER_LDG);
      int const local_row = row_idx - row_offset_;
      int const scale_group =
          local_row / fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE;
      char const* kv_row_ptr;
      char const* scale_head_ptr;
      int row_in_page;
      bool const valid_row = get_nvfp4_row_ptrs(row_idx, kv_row_ptr, scale_head_ptr, row_in_page);

      uint32_t packed[4];
#pragma unroll
      for (int reg = 0; reg < 4; ++reg) {
        float vals[8];
        int const col_base = global_col_base + reg * 8;
        uint8_t original_scale_byte = fmha::float_to_e4m3_byte(1.f);
        if (valid_row && col_base >= 0 && col_base < VALID_COLS) {
          original_scale_byte = load_v_original_scale_byte(scale_head_ptr, row_in_page, col_base);
        }
        float const original_scale = fmha::e4m3_byte_to_float(original_scale_byte);
#pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
          int const col0 = col_base + byte * 2;
          uint8_t original_pair = 0u;
          if (valid_row && col0 >= 0 && col0 < VALID_COLS) {
            fmha::ldg(original_pair, kv_row_ptr + col0 / 2);
          }
          uint8_t const nibble0 = original_pair & 0x0fu;
          uint8_t const nibble1 = (original_pair >> 4) & 0x0fu;
          uint8_t const sf_byte0 =
              smem_tile.load_v_scale_byte_from_write_buffer(logical_col_base + reg * 8 + byte * 2,
                                                            scale_group);
          uint8_t const sf_byte1 =
              smem_tile.load_v_scale_byte_from_write_buffer(
                  logical_col_base + reg * 8 + byte * 2 + 1, scale_group);
          float const sf0 = fmha::e4m3_byte_to_float(sf_byte0);
          float const sf1 = fmha::e4m3_byte_to_float(sf_byte1);
          float const inv_sf0 = sf0 > 0.f ? 1.f / sf0 : 0.f;
          float const inv_sf1 = sf1 > 0.f ? 1.f / sf1 : 0.f;
          vals[byte * 2] = fmha::e2m1_to_float(nibble0) * original_scale * inv_sf0;
          vals[byte * 2 + 1] = fmha::e2m1_to_float(nibble1) * original_scale * inv_sf1;
        }
        packed[reg] = fmha::float8_to_e2m1x8(vals[0], vals[1], vals[2], vals[3],
                                             vals[4], vals[5], vals[6], vals[7]);
      }
      fetch_[ii] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
    if constexpr (USE_LDGSTS) {
      smem_tile.store(fetch_);
    }
  }

  template <typename Smem_tile>
  inline __device__ void load_pv_layout_v(Smem_tile& smem_tile) {
    int const col_tile_offset_elements =
        static_cast<int>(col_tile_offset_in_bytes_ * 8 / BITS_PER_ELEMENT);
    constexpr int SCALE_VEC_BYTES = 16;
    constexpr int SCALE_GROUPS_PER_TILE =
        ROWS / fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE;
    constexpr int SCALE_VECS_PER_GROUP = COLS / SCALE_VEC_BYTES;

#if !defined(FLASHINFER_FMHA_V2_PROFILE_SKIP_V_SCALE_STAGING)
#pragma unroll 1
    for (int scale_vec_idx = threadIdx.x;
         scale_vec_idx < SCALE_GROUPS_PER_TILE * SCALE_VECS_PER_GROUP;
         scale_vec_idx += Cta_tile::THREADS_PER_CTA) {
      int const scale_group = scale_vec_idx / SCALE_VECS_PER_GROUP;
      int const logical_col = (scale_vec_idx - scale_group * SCALE_VECS_PER_GROUP) *
                              SCALE_VEC_BYTES;
      int const token_group_start =
          row_offset_ +
          scale_group * fmha::Blackwell_mma_nvf4_fp32_traits::NVFP4_SCALE_VEC_SIZE;
      int const col = col_tile_offset_elements + logical_col;
      uint4 scale_vec;
      bool const loaded_vec =
          load_v_pv_layout_scale_vec16(token_group_start, col, scale_vec);
      if (loaded_vec) {
        store_v_scale_vec16_bytes<0>(smem_tile, logical_col, scale_group, scale_vec);
        continue;
      }
#pragma unroll
      for (int byte = 0; byte < SCALE_VEC_BYTES; ++byte) {
        smem_tile.store_v_scale(
            logical_col + byte, scale_group,
            load_v_pv_layout_scale_byte(token_group_start, col + byte));
      }
    }
#endif

    load_nvfp4_row_major_data(smem_tile);

  }

  // Move the pointer to the next row location.
  inline __device__ void move() {
    row_ += ROWS;
    row_offset_ += ROWS;
  }

  // Move the pointer to the next row location by the offset (not step).
  inline __device__ void move_by_offset(int const offset) {
    row_ += offset;
    row_offset_ += offset;
  }

  // Move the pointer to the next column location
  inline __device__ void move_col() {
    col_in_bytes_ += THREADS_PER_ROW * BYTES_PER_LDG;
    col_tile_offset_in_bytes_ += THREADS_PER_ROW * BYTES_PER_LDG;
  }

  // Rewind the pointer back to previous column location
  inline __device__ void rewind_col(int const steps) {
    // Update col_in_bytes_ to ensure load predicates work
    col_in_bytes_ -= THREADS_PER_ROW * BYTES_PER_LDG * steps;
    col_tile_offset_in_bytes_ -= THREADS_PER_ROW * BYTES_PER_LDG * steps;
  }

  // The stride between rows for the KV matrice.
  int64_t params_kv_block_size_in_bytes_;
  // The paged cache pool pointer.
  char* paged_kv_block_pool_ptr_;
  // The paged block offsets.
  int32_t* paged_kv_global_block_offsets_;
  // The paged block size.
  int paged_kv_log2_block_size_;
  // The register to store predicates.
  uint32_t preds_[PRED_REGS];
  // The fetch registers.
  uint4 fetch_[LDGS];
  // Keep track of the row and col the thread is processing as we move the tile.
  int row_;
  int64_t col_in_bytes_;
  // Keep track of the head offset.
  int64_t head_offset_in_bytes_;
  // // for DeepSeek MLA, the stride of V tokens != VALID_BYTES_PER_ROW
  int64_t token_stride_in_bytes_;
  // Optional NVFP4 scale-factor tensor metadata for paged K/V.
  int qkv_offset_;
  char const* nvfp4_scale_ptr_;
  int64_t nvfp4_scale_page_stride_in_bytes_;
  int64_t nvfp4_scale_head_stride_in_bytes_;
  int64_t nvfp4_scale_token_stride_in_bytes_;
  int64_t nvfp4_scale_vec_stride_in_bytes_;
  int64_t nvfp4_scale_head_offset_in_bytes_;
  bool nvfp4_v_cache_uses_pv_layout_;
  int row_offset_;
  int64_t col_tile_offset_in_bytes_;
  // The sequence length.
  int actual_seqlen_;
  // The past sequence length (kv_seqlen - q_seqlen) considering chunked context.
  int past_seqlen_;
  // The sliding attention window size.
  int sliding_window_size_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bits per element.
    int BITS_PER_ELEMENT,
    // The number of rows of Q loaded by this tile.
    int ROWS_,
    // The number of columns.
    int COLS,
    // Do we use LDGSTS?
    bool USE_LDGSTS_,
    // Are attention heads interleaved?
    bool HEADS_INTERLEAVED,
    // The number of matrices
    int NUM_MATS = 1>
struct Gmem_tile_q_kv {
  // The size of each LDG.
  enum { BYTES_PER_LDG = 16 };

  // The padded to the next power of 2 number of columns
  enum { COLS_PADDED = Next_power_of_two<COLS>::VALUE };

  // The padded size of a row in bytes.
  enum { BYTES_PER_ROW_PADDED = (COLS_PADDED * BITS_PER_ELEMENT + 7) / 8 };

  // The size of a row in bytes.
  enum { BYTES_PER_ROW = (COLS * BITS_PER_ELEMENT + 7) / 8 };

  // The number of threads to load a padded "row" of the matrix.
  enum { THREADS_PER_ROW_PADDED = BYTES_PER_ROW_PADDED / BYTES_PER_LDG };

  // The number of threads to load a "row" of the matrix.
  enum { THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDG };

  // The number of "rows" loaded per LDG.
  enum { ROWS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW_PADDED };

  // The number of rows.
  enum { ROWS = ROWS_ };

  // The number of LDGs needed to load a chunk of the Q matrix.
  enum { LDGS = fmha::Div_up<ROWS, ROWS_PER_LDG>::VALUE };

  // The number of predicate registers.
  enum { PRED_REGS = fmha::Compute_number_of_pred_regs<LDGS>::VALUE };

  // Is it Hopper?
  enum {
    IS_HOPPER = std::is_same<typename Traits::Gpu_arch, typename fmha::Hopper>::value == true
  };

  // Make sure we use a single register to store predicates. Do not throw for Hopper for now.
  static_assert(!USE_LDGSTS_ || PRED_REGS == 1 || IS_HOPPER, "");

  // We do not use LDGSTS (for the moment).
  enum { USE_LDGSTS = USE_LDGSTS_ };

  // Ctor.
  template <typename Params, typename Block_info>
  inline __device__ Gmem_tile_q_kv(Params const& params, int offset, Block_info const& binfo,
                                   int tidx, int cta_row_offset = 0)
      : params_stride_in_bytes_(params.stride_in_bytes),
        actual_seqlen_(binfo.actual_seqlen),
        ptr_(reinterpret_cast<char*>(params.ptr)) {
    // Compute the position in the sequence (within the CTA for the moment).
    int row = tidx / THREADS_PER_ROW_PADDED;
    // Compute the position of the thread in the row.
    int col = tidx % THREADS_PER_ROW_PADDED;

    // We must store the value to update the predicates in "load".
    row_ = row;
    // Mask for predicate if the channels are in the padded area
    int const bytes_per_row_non_padded = (params.d * BITS_PER_ELEMENT + 7) / 8;
    mask_ = col < bytes_per_row_non_padded / BYTES_PER_LDG;

    // The row offset in the batched GEMM. For each seq element, we store QKV in that order.
    int64_t row_offset = (int64_t)(row + cta_row_offset) * params.stride_in_bytes;
    // Add the block index.
    int64_t idx;
    if (HEADS_INTERLEAVED) {
      idx = binfo.bidx * NUM_MATS + offset;
    } else {
      idx = (binfo.sum_s * NUM_MATS + offset) * params.h + binfo.bidh;
    }
    // Assemble the final pointer.
    ptr_ += row_offset + idx * bytes_per_row_non_padded + col * BYTES_PER_LDG;

    // Take the CTA offset to modify the sequence length.
    actual_seqlen_ -= cta_row_offset;
  }

  // Store data to shared memory.
  template <typename Smem_tile>
  inline __device__ void commit(Smem_tile& smem_tile) {
    if (!USE_LDGSTS) {
      smem_tile.store(fetch_);
    }
  }

  // Load data from memory.
  template <typename Smem_tile>
  inline __device__ void load(Smem_tile& smem_tile) {
    uint32_t preds[LDGS];
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      preds[ii] = (row_ + ii * (int)ROWS_PER_LDG < min((int)ROWS, actual_seqlen_)) && mask_;
    }

    // Prepare the load pointers.
    void const* ptrs[LDGS];
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      ptrs[ii] = ptr_ + (int64_t)ii * ROWS_PER_LDG * params_stride_in_bytes_;
    }

    // Trigger LDGSTS or the LDGs.
    Ldgsts_helper<USE_LDGSTS>::load(this, smem_tile, ptrs, preds);
  }

  inline __device__ void move(int const steps = 1) {
    ptr_ += (int64_t)ROWS * params_stride_in_bytes_ * steps;
    actual_seqlen_ -= (int)ROWS * steps;
  }

  // Store data to memory.
  inline __device__ void store(uint4 const (&data)[LDGS]) {
#pragma unroll
    for (int ii = 0; ii < LDGS; ++ii) {
      char* ptr = ptr_ + (int64_t)ii * ROWS_PER_LDG * params_stride_in_bytes_;
      if ((row_ + ii * ROWS_PER_LDG) < min(ROWS, actual_seqlen_)) {
        fmha::stg(ptr, data[ii]);
      }
    }
  }

  // The stride between rows for the matrix.
  int64_t params_stride_in_bytes_;
  // The pointer.
  char* ptr_;
  // The register to store predicates.
  uint32_t preds_[PRED_REGS];
  // The fetch registers.
  uint4 fetch_[LDGS];
  // Keep track of the row and col the thread is processing as we move the tile.
  int row_;
  // Keep track of predicate state that depends only on the initialization state.
  int mask_;
  // The sequence length.
  int actual_seqlen_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bits per element.
    int BITS_PER_ELEMENT,
    // The number of rows of Q, K or V loaded by this tile.
    int ROWS_,
    // The number of columns.
    int COLS,
    // Do we use LDGSTS?
    bool USE_LDGSTS_>
struct Gmem_tile_qkv_interleaved {
  // The vectorization width for NC/32HW32.
  enum { VEC = 32 };

  // The size of each LDG.
  enum { BYTES_PER_LDG = 16 };

  // The size of a row in bytes.
  enum { BYTES_PER_ROW = (VEC * BITS_PER_ELEMENT + 7) / 8 };

  // DEBUG.
  static_assert(BYTES_PER_ROW == 32, "");

  // END OF DEBUG.

  // The number of threads to load a "row" of the matrix.
  enum { THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDG };

  // DEBUG.
  static_assert(THREADS_PER_ROW == 2, "");

  // END OF DEBUG.

  // The number of "rows" loaded per LDG.
  enum { ROWS_PER_LDG = Cta_tile::THREADS_PER_CTA / THREADS_PER_ROW };

  // The number of slices. It is either 1 for DIM_PER_HEAD == 32 and 2 for DIM_PER_HEAD == 64.
  enum { NUM_SLICES = COLS / VEC };

  // DEBUG.
  static_assert(NUM_SLICES == 1 || NUM_SLICES == 2, "");

  // END OF DEBUG.

  // The number of rows in a slice.
  enum { ROWS = ROWS_ };

  // The number of LDGs needed to load a chunk of the Q matrix.
  enum { LDGS = fmha::Div_up<ROWS * NUM_SLICES, ROWS_PER_LDG>::VALUE };

  // The number of predicate registers.
  enum { PRED_REGS = fmha::Compute_number_of_pred_regs<LDGS>::VALUE };

  // Make sure we use a single register to store predicates.
  static_assert(PRED_REGS == 1, "");

  // Do we use LDGSTS on Ampere?
  enum { USE_LDGSTS = USE_LDGSTS_ };

  // Ctor.
  template <typename Params, typename Block_info>
  inline __device__ Gmem_tile_qkv_interleaved(Params const& params, int qkv_select,
                                              Block_info const& block_info, int tidx,
                                              int cta_row_offset = 0)
      : actual_seqlen_(block_info.actual_seqlen - cta_row_offset),
        total_(params.q_stride_in_bytes),
        kv_ptr_(reinterpret_cast<char const*>(params.qkv_ptr)) {
    int bidh = block_info.bidh;
    int sum_s = block_info.sum_s;

    // We must keep track of the row to repack predicates in load.
    row_ = tidx / THREADS_PER_ROW;
    // The column.
    int col = tidx % THREADS_PER_ROW;

    // h is N
    // d is H
    // we get the data in as: 3 x h x (d/32) x total x 32 (think 3 x h x (d/32)
    // x b x s x 32)

    // Loading qkv: ignore slice for now.
    int qkv_offset = qkv_select * params.h * NUM_SLICES * total_;
    // bidh * GROUPS * B * S + b * S.
    int block_offset = bidh * NUM_SLICES * total_ + sum_s;
    // The row offset.
    int row_offset = (qkv_offset + block_offset + cta_row_offset) * BYTES_PER_ROW;

    // That's the pointer to load from (see "load").
    kv_ptr_ += row_offset + col * BYTES_PER_LDG;

    init_actual_seqlen_ = actual_seqlen_;
    init_kv_ptr_ = kv_ptr_;
  }

  // Store data to shared memory.
  template <typename Smem_tile>
  inline __device__ void commit(Smem_tile& smem_tile) {
    if (!USE_LDGSTS) {
      smem_tile.store(fetch_);
    }
  }

  // Load data from memory.
  template <typename Smem_tile>
  inline __device__ void load(Smem_tile& smem_tile) {
    void const* ptrs[LDGS];
    uint32_t preds[LDGS];

// We precompute slice offsets and predicates
#pragma unroll
    for (int ii = 0; ii < LDGS; ii++) {
      // the next row
      int row_i = row_ + ii * ROWS_PER_LDG;

      // Decompose the current row in slice and original row
      int slice = row_i / ROWS;
      // The position in the slice.
      int row_in_slice = row_i % ROWS;

      // Update the predicate.
      preds[ii] = row_in_slice < min(actual_seqlen_, ROWS);
      // Compute the pointer.
      ptrs[ii] = &kv_ptr_[(slice * total_ + row_in_slice) * BYTES_PER_ROW];
    }

    // Update the predicate register.
    fmha::pack_predicates(preds_, preds);

    // Trigger the loads.
    if (USE_LDGSTS) {
      smem_tile.store(ptrs, preds_);
    } else {
      fmha::ldg(fetch_, ptrs, preds_);
    }
  }

  // Move the pointer to the next location.
  inline __device__ void move(int const steps = 1) {
    kv_ptr_ += (int64_t)ROWS * BYTES_PER_ROW * steps;
    actual_seqlen_ -= ROWS * steps;
  }

  // Reset to the initial location.
  inline __device__ void reset() {
    kv_ptr_ = init_kv_ptr_;
    actual_seqlen_ = init_actual_seqlen_;
  }

  // The pointer.
  char const* kv_ptr_;
  char const* init_kv_ptr_;
  // The register to store predicates.
  uint32_t preds_[PRED_REGS];
  // The fetch registers.
  uint4 fetch_[LDGS];
  // keep track of the row the thread is processing as we move the tile
  int row_;
  // The sequence length.
  int actual_seqlen_;
  int init_actual_seqlen_;
  // The number of rows per slice??
  int total_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace v2
}  // namespace fmha
