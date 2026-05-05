#pragma once

// This file is included from fmha_nvfp4_sm120_d512.cuh inside the D512 namespace,
// after the D512 tile constants and Fp4MmaAtom are defined.

constexpr int kD512ControlledVWordsPerRow = kCutlassTileN / 8;
constexpr int kD512ControlledVB128PerRow = kD512ControlledVWordsPerRow / 4;
constexpr int kD512ControlledVStageB128 =
    kCutlassTileN * kD512ControlledVB128PerRow;
constexpr int kD512ControlledVScaleBytesPerStage =
    (kCutlassTileN / 16) * kCutlassTileN;
constexpr int kD512ControlledVStages = 2;
constexpr int kD512ControlledPVCacheQTileThreshold = 16;
constexpr int64_t kD512ControlledPVCacheStageBytes =
    static_cast<int64_t>(kD512ControlledVStageB128) *
    static_cast<int64_t>(sizeof(flashinfer::b128_t));

using D512ControlledVSmem = flashinfer::smem_t<flashinfer::SwizzleMode::k64B>;

__device__ __forceinline__ uint32_t*
sm120_nvfp4_d512_controlled_v_word_ptr(uint8_t* base,
                                       int stage,
                                       int row,
                                       int word_idx) {
  auto* base_b128 = reinterpret_cast<flashinfer::b128_t*>(base);
  const uint32_t swizzled =
      D512ControlledVSmem::template get_permuted_offset<
          kD512ControlledVB128PerRow>(
          static_cast<uint32_t>(row), static_cast<uint32_t>(word_idx >> 2));
  return reinterpret_cast<uint32_t*>(
             base_b128 + stage * kD512ControlledVStageB128 + swizzled) +
         (word_idx & 3);
}

__device__ __forceinline__ flashinfer::b128_t*
sm120_nvfp4_d512_controlled_v_b128_ptr(uint8_t* base,
                                       int stage,
                                       int row,
                                       int b128_col) {
  auto* base_b128 = reinterpret_cast<flashinfer::b128_t*>(base);
  const uint32_t swizzled =
      D512ControlledVSmem::template get_permuted_offset<
          kD512ControlledVB128PerRow>(
          static_cast<uint32_t>(row), static_cast<uint32_t>(b128_col));
  return base_b128 + stage * kD512ControlledVStageB128 + swizzled;
}

__device__ __forceinline__ const flashinfer::b128_t*
sm120_nvfp4_d512_controlled_v_b128_ptr(const uint8_t* base,
                                       int stage,
                                       int row,
                                       int b128_col) {
  const auto* base_b128 = reinterpret_cast<const flashinfer::b128_t*>(base);
  const uint32_t swizzled =
      D512ControlledVSmem::template get_permuted_offset<
          kD512ControlledVB128PerRow>(
          static_cast<uint32_t>(row), static_cast<uint32_t>(b128_col));
  return base_b128 + stage * kD512ControlledVStageB128 + swizzled;
}

__device__ __forceinline__ const uint32_t*
sm120_nvfp4_d512_controlled_v_word_ptr(const uint8_t* base,
                                       int stage,
                                       int row,
                                       int word_idx) {
  const auto* base_b128 = reinterpret_cast<const flashinfer::b128_t*>(base);
  const uint32_t swizzled =
      D512ControlledVSmem::template get_permuted_offset<
          kD512ControlledVB128PerRow>(
          static_cast<uint32_t>(row), static_cast<uint32_t>(word_idx >> 2));
  return reinterpret_cast<const uint32_t*>(
             base_b128 + stage * kD512ControlledVStageB128 + swizzled) +
         (word_idx & 3);
}

__device__ __forceinline__ uint8_t* sm120_nvfp4_d512_controlled_v_scale_ptr(
    uint8_t* base, int stage, int scale_group, int col) {
  const int k_block = scale_group >> 2;
  const int scale_idx = scale_group & 3;
  return base + stage * kD512ControlledVScaleBytesPerStage +
         (k_block * kCutlassTileN + col) * 4 + scale_idx;
}

__device__ __forceinline__ const uint8_t*
sm120_nvfp4_d512_controlled_v_scale_ptr(const uint8_t* base,
                                        int stage,
                                        int scale_group,
                                        int col) {
  const int k_block = scale_group >> 2;
  const int scale_idx = scale_group & 3;
  return base + stage * kD512ControlledVScaleBytesPerStage +
         (k_block * kCutlassTileN + col) * 4 + scale_idx;
}

__device__ __forceinline__ uint32_t*
sm120_nvfp4_d512_controlled_v_scale_word_ptr(uint8_t* base,
                                             int stage,
                                             int k_block,
                                             int col) {
  return reinterpret_cast<uint32_t*>(
      base + stage * kD512ControlledVScaleBytesPerStage +
      (k_block * kCutlassTileN + col) * 4);
}

__device__ __forceinline__ const uint32_t*
sm120_nvfp4_d512_controlled_v_scale_word_ptr(const uint8_t* base,
                                             int stage,
                                             int k_block,
                                             int col) {
  return reinterpret_cast<const uint32_t*>(
      base + stage * kD512ControlledVScaleBytesPerStage +
      (k_block * kCutlassTileN + col) * 4);
}

inline size_t sm120_nvfp4_d512_controlled_pv_cache_stage_bytes() {
  return static_cast<size_t>(kD512ControlledPVCacheStageBytes);
}

inline size_t sm120_nvfp4_d512_controlled_pv_data_cache_bytes(
    int batch_size,
    int num_kv_heads,
    int total_kv_tiles) {
  return static_cast<size_t>(batch_size) * static_cast<size_t>(num_kv_heads) *
         static_cast<size_t>(total_kv_tiles) *
         static_cast<size_t>(kColumnGroups) *
         sm120_nvfp4_d512_controlled_pv_cache_stage_bytes();
}

inline size_t sm120_nvfp4_d512_controlled_pv_scale_cache_bytes(
    int batch_size,
    int num_kv_heads,
    int total_kv_tiles) {
  return static_cast<size_t>(batch_size) * static_cast<size_t>(num_kv_heads) *
         static_cast<size_t>(total_kv_tiles) *
         static_cast<size_t>(kColumnGroups) *
         static_cast<size_t>(kD512ControlledVScaleBytesPerStage);
}

__device__ __forceinline__ flashinfer::b128_t*
sm120_nvfp4_d512_controlled_pv_cache_b128_ptr(
    uint8_t* cache,
    int64_t batch_stride,
    int64_t head_stride,
    int total_kv_tiles,
    int batch_idx,
    int kv_head,
    int kv_tile,
    int out_group,
    int row,
    int b128_col) {
  uint8_t* head_base =
      cache + static_cast<int64_t>(batch_idx) * batch_stride +
      static_cast<int64_t>(kv_head) * head_stride;
  uint8_t* stage_base =
      head_base +
      (static_cast<int64_t>(kv_tile) * kColumnGroups + out_group) *
          kD512ControlledPVCacheStageBytes;
  return sm120_nvfp4_d512_controlled_v_b128_ptr(stage_base, 0, row, b128_col);
}

__device__ __forceinline__ const flashinfer::b128_t*
sm120_nvfp4_d512_controlled_pv_cache_b128_ptr(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int batch_idx,
    int kv_head,
    int kv_tile,
    int out_group,
    int raw_idx) {
  const int64_t stage_b128 = kD512ControlledVStageB128;
  const int64_t head_base =
      static_cast<int64_t>(batch_idx) * params.v_linear_data_cache_batch_stride +
      static_cast<int64_t>(kv_head) * params.v_linear_data_cache_head_stride;
  const int64_t stage_base =
      head_base +
      (static_cast<int64_t>(kv_tile) * kColumnGroups + out_group) *
          stage_b128 * static_cast<int64_t>(sizeof(flashinfer::b128_t));
  return reinterpret_cast<const flashinfer::b128_t*>(
             params.v_linear_data_cache + stage_base) +
         raw_idx;
}

__device__ __forceinline__ uint32_t*
sm120_nvfp4_d512_controlled_pv_cache_scale_word_ptr(
    uint8_t* cache,
    int64_t batch_stride,
    int64_t head_stride,
    int total_kv_tiles,
    int batch_idx,
    int kv_head,
    int kv_tile,
    int out_group,
    int k_block,
    int col) {
  uint8_t* head_base =
      cache + static_cast<int64_t>(batch_idx) * batch_stride +
      static_cast<int64_t>(kv_head) * head_stride;
  uint8_t* stage_base =
      head_base +
      (static_cast<int64_t>(kv_tile) * kColumnGroups + out_group) *
          kD512ControlledVScaleBytesPerStage;
  return sm120_nvfp4_d512_controlled_v_scale_word_ptr(stage_base, 0, k_block,
                                                      col);
}

__device__ __forceinline__ const uint32_t*
sm120_nvfp4_d512_controlled_pv_cache_scale_word_ptr(
    const Sm120Nvfp4PagedKvLoadParams& params,
    int batch_idx,
    int kv_head,
    int kv_tile,
    int out_group,
    int raw_word_idx) {
  const int64_t head_base =
      static_cast<int64_t>(batch_idx) *
          params.v_linear_scale_cache_batch_stride +
      static_cast<int64_t>(kv_head) * params.v_linear_scale_cache_head_stride;
  const int64_t stage_base =
      head_base +
      (static_cast<int64_t>(kv_tile) * kColumnGroups + out_group) *
          kD512ControlledVScaleBytesPerStage;
  return reinterpret_cast<const uint32_t*>(
             params.v_linear_scale_cache + stage_base) +
         raw_word_idx;
}

static __global__ void sm120_nvfp4_d512_controlled_pv_data_cache_kernel(
    Sm120Nvfp4PagedKvLoadParams params,
    uint8_t* cache,
    const int32_t* kv_lens,
    int batch_size,
    int num_kv_heads,
    int total_kv_tiles,
    int kv_len_tokens,
    int64_t batch_stride,
    int64_t head_stride) {
  const int64_t stage_logical_b128 =
      static_cast<int64_t>(kCutlassTileN) * kD512ControlledVB128PerRow;
  const int64_t total =
      static_cast<int64_t>(batch_size) * num_kv_heads * total_kv_tiles *
      kColumnGroups * stage_logical_b128;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t t = idx;
    const int logical_b128 = static_cast<int>(t % stage_logical_b128);
    t /= stage_logical_b128;
    const int out_group = static_cast<int>(t % kColumnGroups);
    t /= kColumnGroups;
    const int kv_tile = static_cast<int>(t % total_kv_tiles);
    t /= total_kv_tiles;
    const int kv_head = static_cast<int>(t % num_kv_heads);
    const int batch_idx = static_cast<int>(t / num_kv_heads);

    const int row = logical_b128 / kD512ControlledVB128PerRow;
    const int b128_col = logical_b128 - row * kD512ControlledVB128PerRow;
    const int k_block = row >> 6;
    const int row_in_k = row & 63;
    const int reg = row_in_k >> 5;
    const int row_in_reg = row_in_k & 31;
    const int raw_select = row_in_reg >> 3;
    const int token_quad_in_k_block = reg * 8 + (row_in_reg & 7);
    const int local_token_base = k_block * 64 + token_quad_in_k_block * 4;
    const int token_base = kv_tile * kCutlassTileN + local_token_base;
    const int packed_col_local = b128_col * 16 + raw_select * 4;
    const int packed_col = out_group * (kCutlassTileN / 2) + packed_col_local;
    const int live_kv_len =
        kv_lens != nullptr ? kv_lens[batch_idx] : kv_len_tokens;

    uint32_t row_words[4] = {0u, 0u, 0u, 0u};
    if (token_base < live_kv_len) {
      Sm120Nvfp4PagedKvLoadParams local = params;
      local.block_table =
          params.block_table +
          static_cast<int64_t>(batch_idx) * params.block_table_stride;
      local.kv_head = kv_head;
      const int logical_page = token_base / params.page_size;
      const int page_offset0 = token_base - logical_page * params.page_size;
      const int physical_page = local.block_table[logical_page];
      const int64_t data_page_base =
          sm120_nvfp4_paged_v_data_page_base(local, kv_head, physical_page);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (token_base + i < live_kv_len) {
          row_words[i] = sm120_nvfp4_paged_v_word_from_page_base(
              local, data_page_base, page_offset0 + i, packed_col);
        }
      }
    }

    uint32_t packed_words[4];
#pragma unroll
    for (int word_in_b128 = 0; word_in_b128 < 4; ++word_in_b128) {
      uint32_t packed_word = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint8_t pair =
            static_cast<uint8_t>((row_words[i] >> (8 * word_in_b128)) & 0xffu);
        packed_word |= static_cast<uint32_t>(pair & 0x0f) << (4 * i);
        packed_word |= static_cast<uint32_t>((pair >> 4) & 0x0f)
                       << (4 * (i + 4));
      }
      packed_words[word_in_b128] = packed_word;
    }
    *sm120_nvfp4_d512_controlled_pv_cache_b128_ptr(
        cache, batch_stride, head_stride, total_kv_tiles, batch_idx, kv_head,
        kv_tile, out_group, row, b128_col) =
        flashinfer::b128_t{
            packed_words[0], packed_words[1], packed_words[2],
            packed_words[3]};
  }
}

static __global__ void sm120_nvfp4_d512_controlled_pv_scale_cache_kernel(
    Sm120Nvfp4PagedKvLoadParams params,
    uint8_t* cache,
    const int32_t* kv_lens,
    int batch_size,
    int num_kv_heads,
    int total_kv_tiles,
    int kv_len_tokens,
    int64_t batch_stride,
    int64_t head_stride) {
  constexpr int kScaleGroupsPerKBlock = 4;
  constexpr int kScaleKBlocks = kCutlassTileN / 64;
  constexpr int kScaleWordCols = kCutlassTileN / 4;
  const int64_t total =
      static_cast<int64_t>(batch_size) * num_kv_heads * total_kv_tiles *
      kColumnGroups * kScaleKBlocks * kScaleWordCols;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t t = idx;
    const int scale_word_col = static_cast<int>(t % kScaleWordCols);
    t /= kScaleWordCols;
    const int k_block = static_cast<int>(t % kScaleKBlocks);
    t /= kScaleKBlocks;
    const int out_group = static_cast<int>(t % kColumnGroups);
    t /= kColumnGroups;
    const int kv_tile = static_cast<int>(t % total_kv_tiles);
    t /= total_kv_tiles;
    const int kv_head = static_cast<int>(t % num_kv_heads);
    const int batch_idx = static_cast<int>(t / num_kv_heads);

    Sm120Nvfp4PagedKvLoadParams local = params;
    local.block_table =
        params.block_table +
        static_cast<int64_t>(batch_idx) * params.block_table_stride;
    local.kv_head = kv_head;
    const int live_kv_len =
        kv_lens != nullptr ? kv_lens[batch_idx] : kv_len_tokens;
    const int col = scale_word_col * 4;
    const int dim_base = out_group * kCutlassTileN + col;
    uint32_t scale_words[kScaleGroupsPerKBlock] = {
        0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
#pragma unroll
    for (int scale_idx = 0; scale_idx < kScaleGroupsPerKBlock;
         ++scale_idx) {
      const int token_group = k_block * kScaleGroupsPerKBlock + scale_idx;
      const int k0 = token_group * 16;
      const int token = kv_tile * kCutlassTileN + k0;
      if (token < live_kv_len) {
        const int logical_page = token / params.page_size;
        const int physical_page = local.block_table[logical_page];
        scale_words[scale_idx] =
            sm120_nvfp4_paged_v_pv_scale_word_from_physical_page_static<
                kHeadDim / 16>(local, kv_head, physical_page,
                               dim_base);
      }
    }
#pragma unroll
    for (int col_offset = 0; col_offset < 4; ++col_offset) {
      uint32_t consumer_word = 0;
#pragma unroll
      for (int scale_idx = 0; scale_idx < kScaleGroupsPerKBlock;
           ++scale_idx) {
        consumer_word |=
            ((scale_words[scale_idx] >> (8 * col_offset)) & 0xffu)
            << (8 * scale_idx);
      }
      *sm120_nvfp4_d512_controlled_pv_cache_scale_word_ptr(
          cache, batch_stride, head_stride, total_kv_tiles, batch_idx, kv_head,
          kv_tile, out_group, k_block, col + col_offset) = consumer_word;
    }
  }
}

inline cudaError_t sm120_nvfp4_d512_prepare_controlled_pv_cache(
    Sm120Nvfp4PagedKvLoadParams& params,
    uint8_t* data_cache,
    uint8_t* scale_cache,
    const int32_t* kv_lens,
    int batch_size,
    int num_kv_heads,
    int total_kv_tiles,
    int kv_len_tokens,
    cudaStream_t stream) {
  params.v_linear_data_cache = data_cache;
  params.v_linear_data_cache_physical_kv_len = total_kv_tiles;
  params.v_linear_data_cache_head_stride =
      static_cast<int64_t>(total_kv_tiles) * kColumnGroups *
      static_cast<int64_t>(sm120_nvfp4_d512_controlled_pv_cache_stage_bytes());
  params.v_linear_data_cache_batch_stride =
      static_cast<int64_t>(num_kv_heads) *
      params.v_linear_data_cache_head_stride;
  params.v_linear_scale_cache = scale_cache;
  params.v_linear_scale_cache_groups = total_kv_tiles;
  params.v_linear_scale_cache_head_stride =
      static_cast<int64_t>(total_kv_tiles) * kColumnGroups *
      kD512ControlledVScaleBytesPerStage;
  params.v_linear_scale_cache_batch_stride =
      static_cast<int64_t>(num_kv_heads) *
      params.v_linear_scale_cache_head_stride;

  constexpr int kThreads = 256;
  const int64_t data_total =
      static_cast<int64_t>(batch_size) * num_kv_heads * total_kv_tiles *
      kColumnGroups * kCutlassTileN * kD512ControlledVB128PerRow;
  int data_blocks = static_cast<int>((data_total + kThreads - 1) / kThreads);
  if (data_blocks > 65535) {
    data_blocks = 65535;
  }
  sm120_nvfp4_d512_controlled_pv_data_cache_kernel<<<data_blocks, kThreads, 0,
                                                     stream>>>(
      params, data_cache, kv_lens, batch_size, num_kv_heads, total_kv_tiles,
      kv_len_tokens, params.v_linear_data_cache_batch_stride,
      params.v_linear_data_cache_head_stride);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  const int64_t scale_total =
      static_cast<int64_t>(batch_size) * num_kv_heads * total_kv_tiles *
      kColumnGroups * (kCutlassTileN / 64) * (kCutlassTileN / 4);
  int scale_blocks = static_cast<int>((scale_total + kThreads - 1) / kThreads);
  if (scale_blocks > 65535) {
    scale_blocks = 65535;
  }
  sm120_nvfp4_d512_controlled_pv_scale_cache_kernel<<<scale_blocks, kThreads, 0,
                                                      stream>>>(
      params, scale_cache, kv_lens, batch_size, num_kv_heads, total_kv_tiles,
      kv_len_tokens, params.v_linear_scale_cache_batch_stride,
      params.v_linear_scale_cache_head_stride);
  return cudaGetLastError();
}

__device__ __forceinline__ uint8_t sm120_nvfp4_d512_controlled_v_code(
    const uint8_t* base, int stage, int row, int col) {
  const uint32_t word =
      *sm120_nvfp4_d512_controlled_v_word_ptr(base, stage, row, col >> 3);
  return static_cast<uint8_t>((word >> (4 * (col & 7))) & 0x0f);
}

__device__ __forceinline__ uint32_t sm120_nvfp4_d512_pack_e2m1_codes_for_mma(
    uint8_t c0,
    uint8_t c1,
    uint8_t c2,
    uint8_t c3,
    uint8_t c4,
    uint8_t c5,
    uint8_t c6,
    uint8_t c7) {
  return flashinfer::mma::float8_to_e2m1x8(
      e2m1_code_to_fp32(c0), e2m1_code_to_fp32(c1), e2m1_code_to_fp32(c2),
      e2m1_code_to_fp32(c3), e2m1_code_to_fp32(c4), e2m1_code_to_fp32(c5),
      e2m1_code_to_fp32(c6), e2m1_code_to_fp32(c7));
}

__device__ __forceinline__ uint32_t sm120_nvfp4_d512_controlled_v_b_reg_scalar(
    const uint8_t* base,
    int stage,
    int atom_col0,
    int k_block,
    int reg,
    int mma_lane) {
  const int n = mma_lane >> 2;
  const int k_base = k_block * 64 + 8 * (mma_lane & 3) + 32 * reg;
  const int col = atom_col0 + n;
  uint8_t codes[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    codes[i] =
        sm120_nvfp4_d512_controlled_v_code(base, stage, k_base + i, col);
  }
  return sm120_nvfp4_d512_pack_e2m1_codes_for_mma(
      codes[0], codes[1], codes[2], codes[3], codes[4], codes[5], codes[6],
      codes[7]);
}

__device__ __forceinline__ void
sm120_nvfp4_d512_controlled_v_b_regs_ldmatrix(const uint8_t* base,
                                              int stage,
                                              int atom_col0,
                                              int k_block,
                                              int mma_lane,
                                              uint32_t (&b)[2]) {
  auto* base_b128 = const_cast<flashinfer::b128_t*>(
      reinterpret_cast<const flashinfer::b128_t*>(base));
  D512ControlledVSmem smem(base_b128 + stage * kD512ControlledVStageB128);
  const int lane = mma_lane & 31;
  const int raw_select = (atom_col0 >> 3) & 3;
  const int b128_col = atom_col0 >> 5;
#pragma unroll
  for (int reg = 0; reg < 2; ++reg) {
    const uint32_t row = static_cast<uint32_t>(
        k_block * 64 + 32 * reg + (lane & 15) + 16 * (lane >> 4));
    const uint32_t offset =
        D512ControlledVSmem::template get_permuted_offset<
            kD512ControlledVB128PerRow>(row, b128_col);
    uint32_t raw[2];
    if (raw_select < 2) {
      smem.ldmatrix_m8n8x4_trans_left_half(offset, raw);
      b[reg] = raw[raw_select];
    } else {
      smem.ldmatrix_m8n8x4_trans_right_half(offset, raw);
      b[reg] = raw[raw_select - 2];
    }
  }
}

template <class BFragTensor, class CoordTensor>
__device__ __forceinline__ void
sm120_nvfp4_d512_fill_controlled_v_fragment(BFragTensor& b_frag,
                                            CoordTensor const& c_coords,
                                            const uint8_t* controlled_v_base,
                                            int read_stage,
                                            int k_block,
                                            int mma_thread_idx) {
  const int n_count = int(cute::size<1>(b_frag));
  const int mma_lane = mma_thread_idx & 31;
  CUTE_UNROLL
  for (int n = 0; n < n_count; ++n) {
    const int atom_col0 =
        int(cute::get<1>(c_coords(cute::Int<0>{}, 0, n))) & ~7;
    uint32_t b_regs[2];
    sm120_nvfp4_d512_controlled_v_b_regs_ldmatrix(
        controlled_v_base, read_stage, atom_col0, k_block, mma_lane, b_regs);
    auto rB = cute::recast<uint32_t>(b_frag(cute::_, n, k_block));
    CUTE_STATIC_ASSERT_V(cute::size(rB) == cute::Int<2>{});
    rB(0) = b_regs[0];
    rB(1) = b_regs[1];
  }
}

template <class AccumTensor, class AFragTensor, class SfaFragTensor,
          class CoordTensor>
__device__ __forceinline__ void sm120_nvfp4_d512_pv_gemm_controlled_v(
    AccumTensor& accum,
    AFragTensor const& a_frag,
    SfaFragTensor const& a_scale_frag,
    CoordTensor const& c_coords,
    const uint8_t* controlled_v_base,
    const uint8_t* controlled_v_scale_base,
    int read_stage,
    int k_block,
    int mma_thread_idx) {
  const int m_count = int(cute::size<1>(a_frag));
  const int n_count = int(cute::size<2>(accum));
  const int mma_lane = mma_thread_idx & 31;
  CUTE_UNROLL
  for (int n = 0; n < n_count; ++n) {
    const int atom_col0 =
        int(cute::get<1>(c_coords(cute::Int<0>{}, 0, n))) & ~7;
    uint32_t b_regs[2];
    sm120_nvfp4_d512_controlled_v_b_regs_ldmatrix(
        controlled_v_base, read_stage, atom_col0, k_block, mma_lane, b_regs);
    const int scale_col = atom_col0 + (mma_lane >> 2);
    const uint32_t b_scale_reg =
        *sm120_nvfp4_d512_controlled_v_scale_word_ptr(
            controlled_v_scale_base, read_stage, k_block, scale_col);
    CUTE_UNROLL
    for (int m = 0; m < m_count; ++m) {
      auto rD = cute::recast<float>(accum(cute::_, m, n));
      auto rA = cute::recast<uint32_t>(a_frag(cute::_, m));
      auto rSFA =
          cute::recast<uint32_t>(cute::filter_zeros(a_scale_frag(cute::_, m)));
      CUTE_STATIC_ASSERT_V(cute::size(rD) == cute::Int<4>{});
      CUTE_STATIC_ASSERT_V(cute::size(rA) == cute::Int<4>{});
      CUTE_STATIC_ASSERT_V(cute::size(rSFA) == cute::Int<1>{});
      Fp4MmaAtom::fma(rD(0), rD(1), rD(2), rD(3), rA(0), rA(1), rA(2),
                      rA(3), b_regs[0], b_regs[1], rD(0), rD(1), rD(2),
                      rD(3), rSFA(0), b_scale_reg);
    }
  }
}

__device__ __forceinline__ uint32_t sm120_nvfp4_d512_controlled_v_scale_reg(
    const uint8_t* base, int stage, int atom_col0, int k_block, int mma_lane) {
  const int k_group = k_block * 4 + ((mma_lane >> 2) & 3);
  const int n_parity = (mma_lane >> 4) & 1;
  uint8_t scale_bytes[4];
#pragma unroll
  for (int scale_idx = 0; scale_idx < 4; ++scale_idx) {
    const int col = atom_col0 + 2 * scale_idx + n_parity;
    scale_bytes[scale_idx] =
        *sm120_nvfp4_d512_controlled_v_scale_ptr(base, stage, k_group, col);
  }
  return flashinfer::mma::pack_e4m3_scale_reg(
      scale_bytes[0], scale_bytes[1], scale_bytes[2], scale_bytes[3]);
}

template <class AccumTensor, class AFragTensor, class SfaFragTensor,
          class BFragTensor, class SfbFragTensor>
__device__ __forceinline__ void sm120_nvfp4_d512_pv_gemm_controlled_v_fragment(
    AccumTensor& accum,
    AFragTensor const& a_frag,
    SfaFragTensor const& a_scale_frag,
    BFragTensor const& b_frag,
    SfbFragTensor const& b_scale_frag,
    int k_block) {
  const int m_count = int(cute::size<1>(a_frag));
  const int n_count = int(cute::size<2>(accum));
  CUTE_UNROLL
  for (int n = 0; n < n_count; ++n) {
    auto rB = cute::recast<uint32_t>(b_frag(cute::_, n, k_block));
    CUTE_STATIC_ASSERT_V(cute::size(rB) == cute::Int<2>{});
    auto rSFB =
        cute::recast<uint32_t>(cute::filter_zeros(b_scale_frag(cute::_, n)));
    CUTE_STATIC_ASSERT_V(cute::size(rSFB) == cute::Int<1>{});
    CUTE_UNROLL
    for (int m = 0; m < m_count; ++m) {
      auto rD = cute::recast<float>(accum(cute::_, m, n));
      auto rA = cute::recast<uint32_t>(a_frag(cute::_, m));
      auto rSFA =
          cute::recast<uint32_t>(cute::filter_zeros(a_scale_frag(cute::_, m)));
      CUTE_STATIC_ASSERT_V(cute::size(rD) == cute::Int<4>{});
      CUTE_STATIC_ASSERT_V(cute::size(rA) == cute::Int<4>{});
      CUTE_STATIC_ASSERT_V(cute::size(rSFA) == cute::Int<1>{});
      Fp4MmaAtom::fma(rD(0), rD(1), rD(2), rD(3), rA(0), rA(1), rA(2),
                      rA(3), rB(0), rB(1), rD(0), rD(1), rD(2),
                      rD(3), rSFA(0), rSFB(0));
    }
  }
}
