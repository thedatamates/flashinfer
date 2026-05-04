#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cute/arch/mma_sm120.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cutlass/device_kernel.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/float8.h>
#include <cutlass/float_subbyte.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_covered_smem.cuh>
#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_paged_kv.cuh>
#include <flashinfer/cp_async.cuh>
#include <flashinfer/mma.cuh>

namespace flashinfer::attention::blackwell::sm120_nvfp4::d128 {

// SM120 NVFP4 fused attention specialization for D=128 heads.
constexpr int kHeadDim = 128;
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kCutlassTileM = 64;
constexpr int kCutlassTileN = 128;
constexpr int kCutlassTileK = 128;
constexpr int kMinBlocksPerSm = 1;
constexpr int kLogitsRowSkew = 4;
constexpr int kSoftmaxThreadsPerRow = 4;
constexpr int kQkTileN = kCutlassTileN;
constexpr int kOutputTileN = 128;
constexpr int kCutlassTileK128 = 128;
constexpr int kFusedWarpsPerCta = 8;
constexpr int kColumnGroups = kHeadDim / (kTileN * kFusedWarpsPerCta);
constexpr float kProbGlobalScale = 6.0f * 448.0f;

enum class Sm120Nvfp4FmhaRole : int {
  Softmax0 = 0,
  Softmax1 = 1,
  Correction = 2,
  Mma = 3,
  Load = 4,
  Epilogue = 5,
  Empty = 6,
  Count = 7,
};

constexpr int kSm120Nvfp4FmhaNumWarpsSoftmax0 = 0;
constexpr int kSm120Nvfp4FmhaNumWarpsSoftmax1 = 0;
constexpr int kSm120Nvfp4FmhaNumWarpsCorrection = 0;
constexpr int kSm120Nvfp4FmhaNumWarpsMma = 8;
constexpr int kSm120Nvfp4FmhaNumWarpsLoad = 8;
constexpr int kSm120Nvfp4FmhaNumWarpsEpilogue = 1;
constexpr int kSm120Nvfp4FmhaWarpSoftmax0Begin = 0;
constexpr int kSm120Nvfp4FmhaWarpSoftmax1Begin =
    kSm120Nvfp4FmhaWarpSoftmax0Begin + kSm120Nvfp4FmhaNumWarpsSoftmax0;
constexpr int kSm120Nvfp4FmhaWarpCorrectionBegin =
    kSm120Nvfp4FmhaWarpSoftmax1Begin + kSm120Nvfp4FmhaNumWarpsSoftmax1;
constexpr int kSm120Nvfp4FmhaWarpMmaBegin =
    kSm120Nvfp4FmhaWarpCorrectionBegin + kSm120Nvfp4FmhaNumWarpsCorrection;
constexpr int kSm120Nvfp4FmhaWarpLoad =
    kSm120Nvfp4FmhaWarpMmaBegin + kSm120Nvfp4FmhaNumWarpsMma;
constexpr int kSm120Nvfp4FmhaWarpEpilogue =
    kSm120Nvfp4FmhaWarpLoad + kSm120Nvfp4FmhaNumWarpsLoad;
constexpr int kSm120Nvfp4FmhaNumWarps =
    kSm120Nvfp4FmhaWarpEpilogue + kSm120Nvfp4FmhaNumWarpsEpilogue;
constexpr int kSm120Nvfp4FmhaThreadCount =
    kSm120Nvfp4FmhaNumWarps * cutlass::NumThreadsPerWarp;
constexpr int kSm120Nvfp4FmhaOutputThreadCount =
    kSm120Nvfp4FmhaNumWarpsEpilogue * cutlass::NumThreadsPerWarp;
constexpr int kSm120Nvfp4FmhaLoadThreadCount =
    kSm120Nvfp4FmhaNumWarpsLoad * cutlass::NumThreadsPerWarp;
constexpr int kSm120Nvfp4FmhaMmaSoftmaxThreadCount =
    kSm120Nvfp4FmhaNumWarpsMma * cutlass::NumThreadsPerWarp;
constexpr int kSm120Nvfp4FmhaMmaSoftmaxLoadThreadCount =
    kSm120Nvfp4FmhaMmaSoftmaxThreadCount +
    kSm120Nvfp4FmhaNumWarpsLoad * cutlass::NumThreadsPerWarp;
constexpr uint32_t kSm120Nvfp4BarrierLoadGroup =
    static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
__host__ __device__ constexpr Sm120Nvfp4FmhaRole
sm120_nvfp4_fmha_role_for_warp(int warp_idx) {
  if (warp_idx >= kSm120Nvfp4FmhaWarpSoftmax0Begin &&
      warp_idx < kSm120Nvfp4FmhaWarpSoftmax1Begin) {
    return Sm120Nvfp4FmhaRole::Softmax0;
  }
  if (warp_idx >= kSm120Nvfp4FmhaWarpSoftmax1Begin &&
      warp_idx < kSm120Nvfp4FmhaWarpCorrectionBegin) {
    return Sm120Nvfp4FmhaRole::Softmax1;
  }
  if (warp_idx >= kSm120Nvfp4FmhaWarpCorrectionBegin &&
      warp_idx < kSm120Nvfp4FmhaWarpMmaBegin) {
    return Sm120Nvfp4FmhaRole::Correction;
  }
  if (warp_idx >= kSm120Nvfp4FmhaWarpMmaBegin &&
      warp_idx < kSm120Nvfp4FmhaWarpLoad) {
    return Sm120Nvfp4FmhaRole::Mma;
  }
  if (warp_idx >= kSm120Nvfp4FmhaWarpLoad &&
      warp_idx < kSm120Nvfp4FmhaWarpEpilogue) {
    return Sm120Nvfp4FmhaRole::Load;
  }
  if (warp_idx == kSm120Nvfp4FmhaWarpEpilogue) {
    return Sm120Nvfp4FmhaRole::Epilogue;
  }
  return Sm120Nvfp4FmhaRole::Empty;
}

__device__ __forceinline__ int sm120_nvfp4_fmha_mma_thread_idx(
    int thread_idx) {
  const int warp_idx = thread_idx / cutlass::NumThreadsPerWarp;
  const int lane_idx = thread_idx % cutlass::NumThreadsPerWarp;
  return (warp_idx - kSm120Nvfp4FmhaWarpMmaBegin) *
             cutlass::NumThreadsPerWarp +
         lane_idx;
}

using Fp4MmaAtom =
    cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<cutlass::float_e2m1_t,
                                                  cutlass::float_e2m1_t,
                                                  float,
                                                  cutlass::float_ue4m3_t,
                                                  16>;

using CutlassElementAB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using CutlassElementC = void;
using CutlassElementD = cutlass::bfloat16_t;
using CutlassThreadBlockShape =
    cute::Shape<cute::Int<kCutlassTileM>,
                cute::Int<kCutlassTileN>,
                cute::Int<kCutlassTileK>>;
using CutlassClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
using CutlassFusionOperation =
    cutlass::epilogue::fusion::LinearCombination<CutlassElementD, float,
                                                 CutlassElementC, float>;
using CutlassCollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassTensorOp,
        CutlassThreadBlockShape,
        CutlassClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        float,
        float,
        CutlassElementC,
        cutlass::layout::RowMajor,
        128 / cutlass::sizeof_bits<CutlassElementD>::value,
        CutlassElementD,
        cutlass::layout::RowMajor,
        128 / cutlass::sizeof_bits<CutlassElementD>::value,
        cutlass::epilogue::TmaWarpSpecialized,
        CutlassFusionOperation>::CollectiveOp;
using CutlassCollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassBlockScaledTensorOp,
        CutlassElementAB,
        cutlass::layout::RowMajor,
        32,
        CutlassElementAB,
        cutlass::layout::ColumnMajor,
        32,
        float,
        CutlassThreadBlockShape,
        CutlassClusterShape,
        cutlass::gemm::collective::StageCount<2>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using CutlassProblemShape = cute::Shape<int, int, int, int>;
using CutlassGemmKernel =
    cutlass::gemm::kernel::GemmUniversal<CutlassProblemShape,
                                         CutlassCollectiveMainloop,
                                         CutlassCollectiveEpilogue,
                                         cutlass::gemm::StaticPersistentScheduler>;

using CutlassThreadBlockShapeK128 =
    cute::Shape<cute::Int<kCutlassTileM>,
                cute::Int<kOutputTileN>,
                cute::Int<kCutlassTileN>>;
using CutlassCollectiveMainloopK128Stage2 =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassBlockScaledTensorOp,
        CutlassElementAB,
        cutlass::layout::RowMajor,
        32,
        CutlassElementAB,
        cutlass::layout::ColumnMajor,
        32,
        float,
        CutlassThreadBlockShapeK128,
        CutlassClusterShape,
        cutlass::gemm::collective::StageCount<2>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using CutlassGemmKernelK128Stage2 =
    cutlass::gemm::kernel::GemmUniversal<
        CutlassProblemShape,
        CutlassCollectiveMainloopK128Stage2,
        CutlassCollectiveEpilogue,
        cutlass::gemm::StaticPersistentScheduler>;

template <typename GemmKernel>
inline typename GemmKernel::Arguments prepare_sm120_nvfp4_gemm_args(
    void* d,
    void const* a,
    void const* b,
    void const* sfa,
    void const* sfb,
    float const* alpha,
    int m,
    int n,
    int k,
    int batch_count) {
  using Sm1xxBlkScaledConfig =
      typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using ElementC = void;
  using ElementD = typename GemmKernel::ElementD;
  using ElementCompute = float;

  typename GemmKernel::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.epilogue.thread.alpha_ptr =
      static_cast<ElementCompute const*>(alpha);
  args.problem_shape = cute::make_shape(m, n, k, batch_count);

  args.mainloop.ptr_A = static_cast<cutlass::float_e2m1_t const*>(a);
  args.mainloop.ptr_B = static_cast<cutlass::float_e2m1_t const*>(b);
  args.mainloop.ptr_SFA = static_cast<cutlass::float_ue4m3_t const*>(sfa);
  args.mainloop.ptr_SFB = static_cast<cutlass::float_ue4m3_t const*>(sfb);
  args.epilogue.ptr_C = static_cast<ElementC const*>(d);
  args.epilogue.ptr_D = static_cast<ElementD*>(d);

  const int stride_a = batch_count == 1 ? 0 : m * k;
  const int stride_b = batch_count == 1 ? 0 : n * k;
  const int stride_c = batch_count == 1 ? 0 : m * n;

  args.mainloop.dA =
      cute::make_int_tuple_from<typename GemmKernel::StrideA>(k, stride_a);
  args.mainloop.dB =
      cute::make_int_tuple_from<typename GemmKernel::StrideB>(k, stride_b);
  args.epilogue.dC =
      cute::make_int_tuple_from<typename GemmKernel::StrideC>(n, stride_c);
  args.epilogue.dD = args.epilogue.dC;

  args.mainloop.layout_SFA =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(args.problem_shape);
  args.mainloop.layout_SFB =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(args.problem_shape);

  if constexpr (!std::is_const_v<decltype(args.scheduler.max_swizzle_size)>) {
    args.scheduler.max_swizzle_size = 1;
  }
  if constexpr (!std::is_const_v<decltype(args.scheduler.raster_order)>) {
    using EnumT = decltype(args.scheduler.raster_order);
    args.scheduler.raster_order = EnumT::Heuristic;
  }
  args.hw_info.cluster_shape = dim3(1, 1, 1);
  args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
  return args;
}

using Sm120Nvfp4PipelineE = cutlass::PipelineAsync<1>;

struct Sm120Nvfp4MainloopPipelineStorage {
  alignas(16) typename Sm120Nvfp4PipelineE::SharedStorage corr_epi;
};

struct Sm120Nvfp4QkLoadCollectiveStorage {
  alignas(128) typename CutlassCollectiveMainloop::TensorStorage tensors;
  alignas(16) typename CutlassCollectiveMainloop::PipelineStorage
      q_pipeline_storage;
  alignas(16) typename CutlassCollectiveMainloop::PipelineStorage
      k_pipeline_storage;
};

static_assert(sizeof(Sm120Nvfp4QkLoadCollectiveStorage) <= (99u << 10),
              "SM120 Q/K load collective storage must fit SM120 opt-in shared memory");

struct Sm120Nvfp4EmptyStorage {
  uint8_t data[1];

  __host__ __device__ cutlass::float_ue4m3_t* begin() {
    return reinterpret_cast<cutlass::float_ue4m3_t*>(data);
  }
};

template <int kBytes>
struct Sm120Nvfp4ByteStorage {
  uint8_t data[kBytes];
};

template <int kBytes, int kAlignment>
struct alignas(kAlignment) Sm120Nvfp4AlignedByteStorage {
  uint8_t data[kBytes];
};

using Sm120Nvfp4PvSmemAllocA =
    typename CutlassCollectiveMainloopK128Stage2::SmemAllocTypeA;
using Sm120Nvfp4PvSmemAllocB =
    typename CutlassCollectiveMainloopK128Stage2::SmemAllocTypeB;
constexpr int kSm120Nvfp4PvPStageBytes = cutlass::bits_to_bytes(
    cute::cosize_v<typename CutlassCollectiveMainloopK128Stage2::SmemLayoutA> *
    cute::sizeof_bits_v<Sm120Nvfp4PvSmemAllocA>);
constexpr int kSm120Nvfp4PvScaleStageElems =
    cute::cosize_v<typename CutlassCollectiveMainloopK128Stage2::SmemLayoutSFA>;
constexpr int kSm120Nvfp4LogitsBytes =
    kCutlassTileM * kCutlassTileN * static_cast<int>(sizeof(__nv_bfloat16));
constexpr bool kSm120Nvfp4AliasP0InQkA =
    sizeof(decltype(
        ((typename CutlassCollectiveMainloop::TensorStorage*)nullptr)->smem_A)) >=
    kSm120Nvfp4PvPStageBytes;
constexpr bool kSm120Nvfp4AliasP0ScaleInQkSFA =
    sizeof(decltype(
        ((typename CutlassCollectiveMainloop::TensorStorage*)nullptr)->smem_SFA)) >=
    kSm120Nvfp4PvScaleStageElems *
        static_cast<int>(sizeof(cutlass::float_ue4m3_t));
using Sm120Nvfp4LogitsStorage =
    Sm120Nvfp4AlignedByteStorage<kSm120Nvfp4LogitsBytes, 128>;
using Sm120Nvfp4PStorage0 = std::conditional_t<
    kSm120Nvfp4AliasP0InQkA,
    Sm120Nvfp4EmptyStorage,
    Sm120Nvfp4AlignedByteStorage<kSm120Nvfp4PvPStageBytes, 1024>>;

struct Sm120Nvfp4QkvLoadCollectiveStorage {
  alignas(128) typename CutlassCollectiveMainloop::TensorStorage qk_tensors;
  alignas(16) typename CutlassCollectiveMainloop::PipelineStorage
      q_pipeline_storage;
  alignas(16) typename CutlassCollectiveMainloop::PipelineStorage
      k_pipeline_storage;
  alignas(1024) cute::ArrayEngine<
      Sm120Nvfp4PvSmemAllocB,
      cute::cosize_v<typename CutlassCollectiveMainloopK128Stage2::SmemLayoutB>>
      v_smem_B;
  alignas(16) cute::ArrayEngine<
      cutlass::float_ue4m3_t,
      cute::cosize_v<typename CutlassCollectiveMainloopK128Stage2::SmemLayoutSFB>>
      v_smem_SFB;
  alignas(16) typename CutlassCollectiveMainloopK128Stage2::PipelineStorage
      v_pipeline_storage;
  Sm120Nvfp4LogitsStorage logits_smem;
  Sm120Nvfp4PStorage0 p_smem_A0;
  alignas(16) std::conditional_t<
      kSm120Nvfp4AliasP0ScaleInQkSFA,
      Sm120Nvfp4EmptyStorage,
      cute::ArrayEngine<
          cutlass::float_ue4m3_t,
          cute::cosize_v<typename CutlassCollectiveMainloopK128Stage2::SmemLayoutSFA>>>
	      p_smem_SFA0;
  alignas(16) Sm120Nvfp4MainloopPipelineStorage role_pipeline_storage;
  alignas(16) float global_m[kCutlassTileM];
  alignas(16) float global_l[kCutlassTileM];
  alignas(16) float old_scale_stage[2][kCutlassTileM];
};

static_assert(sizeof(Sm120Nvfp4QkvLoadCollectiveStorage) <= (99u << 10),
              "SM120 Q/K/V load collective storage must fit SM120 opt-in shared memory");

__device__ __forceinline__ cutlass::float_ue4m3_t make_ue4m3_raw(uint8_t raw) {
  cutlass::float_ue4m3_t value;
  value.storage = raw;
  return value;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
template <class FrgTensorA, class FrgTensorSFA, class SmemTensorA,
          class SmemTensorSFA>
__device__ __forceinline__ void cutlass_qk_tma_q_register_stage(
    typename CutlassCollectiveMainloop::MainloopPipeline pipeline,
    typename CutlassCollectiveMainloop::PipelineState& smem_pipe_read,
    FrgTensorA& q_frag,
    FrgTensorSFA& q_scale_frag,
    int thread_idx,
    SmemTensorA& sA,
    SmemTensorSFA& sSFA) {
  using namespace cute;

  auto tiled_mma = typename CutlassCollectiveMainloop::TiledMma{};
  CutlassCollectiveMainloop collective;

  auto smem_tiled_copy_A = make_tiled_copy_A(
      typename CutlassCollectiveMainloop::SmemCopyAtomA{}, tiled_mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
  Tensor tCsA = smem_thr_copy_A.partition_S(
      as_position_independent_swizzle_tensor(sA));
  Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(q_frag);

  auto tile_shape_mnk = tile_shape(tiled_mma);
  auto smem_tiled_copy_SFA = make_tiled_copy_impl(
      typename CutlassCollectiveMainloop::SmemCopyAtomSFA{},
      collective.get_layoutSFA_TV(tiled_mma),
      make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
  Tensor tCsSFA = smem_thr_copy_SFA.partition_S(
      as_position_independent_swizzle_tensor(sSFA));
  Tensor tCrSFA_copy_view = smem_thr_copy_SFA.retile_D(q_scale_frag);

  auto K_BLOCK_MAX = size<2>(q_frag);
  const int read_stage = smem_pipe_read.index();
  auto tCsA_stage = tCsA(_, _, _, read_stage);
  auto tCsSFA_stage = tCsSFA(_, _, _, read_stage);

  pipeline.consumer_wait(smem_pipe_read);

  for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
    copy(smem_tiled_copy_A, tCsA_stage(_, _, k_block),
         tCrA_copy_view(_, _, k_block));
    using MMAOp = typename CutlassCollectiveMainloop::TiledMma::MMA_Op;
    fp4_shift_A(MMAOp{}, tCrA_copy_view(_, _, k_block));
    copy(tCsSFA_stage(_, _, k_block), tCrSFA_copy_view(_, _, k_block));
  });

  cutlass::arch::NamedBarrier::sync(
      thr_size(tiled_mma),
      cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
  pipeline.consumer_release(smem_pipe_read);
  ++smem_pipe_read;
}
#endif

template <class AccumTensor, class CoordTensor>
__device__ __forceinline__ void sm120_stage_o_fragment_to_epilogue_smem(
    AccumTensor const& accum,
    CoordTensor const& coords,
    __nv_bfloat16* smem_o,
    const float* global_l,
    float pv_base_scale,
    bool normalize_by_l) {
  for (int i = 0; i < int(cute::size(accum)); ++i) {
    auto coord = coords(i);
    const int row = int(cute::get<0>(coord));
    const int col = int(cute::get<1>(coord));
    if (row < kCutlassTileM && col < kOutputTileN) {
      const float norm = normalize_by_l ? fmaxf(global_l[row], 1.0e-20f) : 1.0f;
      const float row_scale =
          pv_base_scale / norm;
      smem_o[row * kOutputTileN + col] =
          __float2bfloat16(accum(i) * row_scale);
    }
  }
}

__device__ __forceinline__ void sm120_epilogue_store_bf16_tile(
    const __nv_bfloat16* smem_o,
    __nv_bfloat16* out_tile,
    int out_stride_cols,
    int epilogue_thread_idx) {
  for (int idx = epilogue_thread_idx; idx < kCutlassTileM * kOutputTileN;
       idx += kSm120Nvfp4FmhaNumWarpsEpilogue * cutlass::NumThreadsPerWarp) {
    const int row = idx / kOutputTileN;
    const int col = idx - row * kOutputTileN;
    out_tile[row * out_stride_cols + col] = smem_o[idx];
  }
}

template <int kOutputGroupSpan, bool kUsePagedKv, bool kCausal,
          bool kUseSlidingWindow, bool kUseLogitsSoftCap,
          bool kPvLayoutV = true>
	__global__ __launch_bounds__(kSm120Nvfp4FmhaThreadCount,
	                             kMinBlocksPerSm)
	void sm120_nvfp4_qkv_online_register_q_stage_kernel(
    CUTLASS_GRID_CONSTANT typename CutlassGemmKernel::Params const qk_params,
    CUTLASS_GRID_CONSTANT typename CutlassGemmKernelK128Stage2::Params const pv_params,
    __nv_bfloat16* out_group,
    float qk_alpha,
    float pv_alpha,
    int q_tile,
    int kv_tile_start,
    int num_kv_tiles,
    int total_kv_tiles,
    int q_len,
    int group_size,
    int kv_len_tokens,
    int causal,
    int sliding_window,
    float logits_soft_cap,
    int out_group_idx,
    int out_stride_cols,
    float* split_m,
    float* split_l,
    int split_stats_stride_rows,
    int split_output_stride_elems,
    Sm120Nvfp4PagedKvLoadParams paged_kv_params,
    const int32_t* qo_indptr,
    const int32_t* kv_lens,
    int batch_size,
    int q_tiles_per_sequence,
    int num_kv_heads,
    bool all_kv_heads) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  using cute::_;
  static_assert(kOutputGroupSpan == 1,
                "D128 specialization has exactly one 128-wide output group");

  extern __shared__ __align__(128) char smem[];
  auto& storage = *reinterpret_cast<Sm120Nvfp4QkvLoadCollectiveStorage*>(smem);
  const int q_block_idx = q_tile + int(blockIdx.x);
  int head_q_block_idx = q_block_idx;
  int batch_idx = 0;
  int local_q_tile = q_block_idx;
  int effective_q_tile = q_block_idx;
  int effective_q_begin = 0;
  if constexpr (kUsePagedKv) {
    if (all_kv_heads) {
      if (q_tiles_per_sequence <= 0 || batch_size <= 0 ||
          num_kv_heads <= 0) {
        return;
      }
      const int q_tiles_per_kv_head = batch_size * q_tiles_per_sequence;
      const int kv_head = q_block_idx / q_tiles_per_kv_head;
      if (kv_head >= num_kv_heads) {
        return;
      }
      head_q_block_idx = q_block_idx - kv_head * q_tiles_per_kv_head;
      paged_kv_params.kv_head = kv_head;
    }
    const bool varlen_batch = qo_indptr != nullptr && kv_lens != nullptr;
    if (varlen_batch) {
      if (q_tiles_per_sequence <= 0) {
        return;
      }
      batch_idx = head_q_block_idx / q_tiles_per_sequence;
      if (batch_idx >= batch_size) {
        return;
      }
      local_q_tile = head_q_block_idx - batch_idx * q_tiles_per_sequence;
      effective_q_tile = q_block_idx;
      const int q_begin = qo_indptr[batch_idx];
      const int q_end = qo_indptr[batch_idx + 1];
      effective_q_begin = q_begin;
      q_len = q_end - q_begin;
      if (q_len <= 0) {
        return;
      }
      kv_len_tokens = kv_lens[batch_idx];
      if (kv_len_tokens <= 0) {
        return;
      }
      total_kv_tiles = (kv_len_tokens + kCutlassTileN - 1) / kCutlassTileN;
      paged_kv_params.block_table =
          paged_kv_params.block_table +
          static_cast<int64_t>(batch_idx) * paged_kv_params.block_table_stride;
    }
  }
  const int effective_split_idx = int(blockIdx.z);
  const int effective_kv_tile_start =
      kv_tile_start + effective_split_idx * num_kv_tiles;
  const int remaining_kv_tiles = total_kv_tiles - effective_kv_tile_start;
  const int effective_num_kv_tiles =
      remaining_kv_tiles < num_kv_tiles ? remaining_kv_tiles : num_kv_tiles;
  if (effective_num_kv_tiles <= 0) {
    return;
  }
  const int qk_head_chunks = out_stride_cols / kCutlassTileK;
  const float qk_scale = qk_alpha * rsqrtf(static_cast<float>(out_stride_cols));
  const int effective_out_group_base =
      out_group_idx + int(blockIdx.y) * kOutputGroupSpan;
  __nv_bfloat16* out_split_base =
      out_group + effective_split_idx * split_output_stride_elems;
  __nv_bfloat16* out_tile =
      out_split_base + int(blockIdx.x) * kCutlassTileM * out_stride_cols +
      int(blockIdx.y) * kOutputGroupSpan * kOutputTileN;

  const int thread_idx = int(threadIdx.x);
  const int warp_idx = thread_idx / cutlass::NumThreadsPerWarp;
  const int lane_idx = thread_idx % cutlass::NumThreadsPerWarp;
  const Sm120Nvfp4FmhaRole role = sm120_nvfp4_fmha_role_for_warp(warp_idx);
  const bool is_load = role == Sm120Nvfp4FmhaRole::Load;
  const bool is_mma = role == Sm120Nvfp4FmhaRole::Mma;
  const bool is_epilogue = role == Sm120Nvfp4FmhaRole::Epilogue;
  const int load_thread_idx =
      is_load ? thread_idx - kSm120Nvfp4FmhaWarpLoad * cutlass::NumThreadsPerWarp
              : 0;
  const bool load_leader = is_load && load_thread_idx == 0;
  const bool first_load_warp = is_load && load_thread_idx < cutlass::NumThreadsPerWarp;
  auto load_group_sync = [&]() {
    cutlass::arch::NamedBarrier::sync(kSm120Nvfp4FmhaLoadThreadCount,
                                      kSm120Nvfp4BarrierLoadGroup);
  };
  const int qk_mma_thread_idx =
      is_mma ? sm120_nvfp4_fmha_mma_thread_idx(thread_idx) : 0;
  const int pv_mma_thread_idx = qk_mma_thread_idx;
  const int epilogue_thread_idx =
      is_epilogue
          ? thread_idx - kSm120Nvfp4FmhaWarpEpilogue * cutlass::NumThreadsPerWarp
          : 0;
  const int output_thread_idx = is_epilogue ? epilogue_thread_idx : lane_idx;

  for (int row = thread_idx; row < kCutlassTileM; row += blockDim.x) {
    storage.global_m[row] = -INFINITY;
    storage.global_l[row] = 0.0f;
    storage.old_scale_stage[0][row] = 0.0f;
    storage.old_scale_stage[1][row] = 0.0f;
  }
  __syncthreads();

  typename Sm120Nvfp4PipelineE::Params pipeline_corr_epi_params{};
  if (is_mma) {
    pipeline_corr_epi_params.role =
        Sm120Nvfp4PipelineE::ThreadCategory::Producer;
  }
  if (is_epilogue) {
    pipeline_corr_epi_params.role =
        Sm120Nvfp4PipelineE::ThreadCategory::Consumer;
  }
  pipeline_corr_epi_params.producer_arv_count = 1;
  pipeline_corr_epi_params.consumer_arv_count =
      kSm120Nvfp4FmhaOutputThreadCount;
  pipeline_corr_epi_params.initializing_warp = kSm120Nvfp4FmhaWarpLoad;
  Sm120Nvfp4PipelineE pipeline_corr_epi(
      storage.role_pipeline_storage.corr_epi, pipeline_corr_epi_params,
      cute::true_type{});

  typename CutlassCollectiveMainloop::PipelineState k_pipe_release;
  typename Sm120Nvfp4PipelineE::PipelineState pipeline_corr_epi_producer_state =
      cutlass::make_producer_start_state<Sm120Nvfp4PipelineE>();
  typename Sm120Nvfp4PipelineE::PipelineState pipeline_corr_epi_consumer_state;

  if (load_leader) {
    CutlassCollectiveMainloop::prefetch_tma_descriptors(qk_params.mainloop);
    CutlassCollectiveMainloopK128Stage2::prefetch_tma_descriptors(
        pv_params.mainloop);
  }

  using QkPipeline = typename CutlassCollectiveMainloop::MainloopPipeline;
  using VPipeline = typename CutlassCollectiveMainloopK128Stage2::MainloopPipeline;
  typename QkPipeline::Params q_pipeline_params{};
  typename QkPipeline::Params k_pipeline_params{};
  typename VPipeline::Params v_pipeline_params{};
  if (is_load) {
    q_pipeline_params.role = QkPipeline::ThreadCategory::Producer;
    k_pipeline_params.role = QkPipeline::ThreadCategory::Producer;
    v_pipeline_params.role = VPipeline::ThreadCategory::Producer;
  }
  if (is_mma) {
    q_pipeline_params.role = QkPipeline::ThreadCategory::Consumer;
    k_pipeline_params.role = QkPipeline::ThreadCategory::Consumer;
    v_pipeline_params.role = VPipeline::ThreadCategory::Consumer;
  }
  const bool mma_warpgroup_leader =
      is_mma && (qk_mma_thread_idx % cutlass::NumThreadsPerWarpGroup) == 0;
  q_pipeline_params.is_leader = load_leader || mma_warpgroup_leader;
  k_pipeline_params.is_leader = load_leader || mma_warpgroup_leader;
  v_pipeline_params.is_leader = load_leader || mma_warpgroup_leader;
  q_pipeline_params.num_consumers = CutlassCollectiveMainloop::ThreadCount;
  k_pipeline_params.num_consumers = CutlassCollectiveMainloop::ThreadCount;
  v_pipeline_params.num_consumers =
      CutlassCollectiveMainloopK128Stage2::ThreadCount;
  q_pipeline_params.num_producers =
      CutlassCollectiveMainloop::NumProducerThreadEvents;
  k_pipeline_params.num_producers =
      CutlassCollectiveMainloop::NumProducerThreadEvents;
  v_pipeline_params.num_producers =
      CutlassCollectiveMainloopK128Stage2::NumProducerThreadEvents;
  q_pipeline_params.transaction_bytes =
      qk_params.mainloop.tma_transaction_bytes_mk;
  k_pipeline_params.transaction_bytes =
      qk_params.mainloop.tma_transaction_bytes_nk;
  v_pipeline_params.transaction_bytes =
      pv_params.mainloop.tma_transaction_bytes_nk;
  q_pipeline_params.initializing_warp = kSm120Nvfp4FmhaWarpLoad;
  k_pipeline_params.initializing_warp = kSm120Nvfp4FmhaWarpLoad;
  v_pipeline_params.initializing_warp = kSm120Nvfp4FmhaWarpLoad;

  QkPipeline q_pipeline(storage.q_pipeline_storage, q_pipeline_params,
                        CutlassClusterShape{});
  QkPipeline k_pipeline(storage.k_pipeline_storage, k_pipeline_params,
                        CutlassClusterShape{});
  VPipeline v_pipeline(storage.v_pipeline_storage, v_pipeline_params,
                       CutlassClusterShape{});
  typename CutlassCollectiveMainloop::PipelineState q_pipe_read;
  typename CutlassCollectiveMainloop::PipelineState k_pipe_read;
  typename CutlassCollectiveMainloopK128Stage2::PipelineState v_pipe_read;
  typename CutlassCollectiveMainloop::PipelineState q_pipe_write =
      cutlass::make_producer_start_state<QkPipeline>();
  typename CutlassCollectiveMainloop::PipelineState k_pipe_write =
      cutlass::make_producer_start_state<QkPipeline>();
  typename CutlassCollectiveMainloopK128Stage2::PipelineState v_pipe_write =
      cutlass::make_producer_start_state<VPipeline>();

  __syncthreads();

  CutlassCollectiveMainloop qk_collective;
  auto qk_tiled_mma = typename CutlassCollectiveMainloop::TiledMma{};
  auto qk_thread_mma = qk_tiled_mma.get_thread_slice(qk_mma_thread_idx);
  constexpr auto kCoveredSmemInitBarrier =
      cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier;
  using QkSmemAllocA = typename CutlassCollectiveMainloop::SmemAllocTypeA;
  using QkSmemAllocB = typename CutlassCollectiveMainloop::SmemAllocTypeB;
  auto qk_sA_ptr = storage.qk_tensors.smem_A.begin();
  ZeroSmemTile<QkSmemAllocA, typename CutlassCollectiveMainloop::SmemLayoutA,
               decltype(qk_sA_ptr)>
      qk_sA_covered(qk_sA_ptr, typename CutlassCollectiveMainloop::SmemLayoutA{},
                    blockDim.x, kCoveredSmemInitBarrier);
  auto qk_sA = qk_sA_covered.tensor();
  auto qk_sB_ptr = storage.qk_tensors.smem_B.begin();
  ZeroSmemTile<QkSmemAllocB, typename CutlassCollectiveMainloop::SmemLayoutB,
               decltype(qk_sB_ptr)>
      qk_sB_covered(qk_sB_ptr, typename CutlassCollectiveMainloop::SmemLayoutB{},
                    blockDim.x, kCoveredSmemInitBarrier);
  auto qk_sB = qk_sB_covered.tensor();
  // SM120 has no TMEM, so QK logits, softmax probabilities, and PV staging use
  // compact shared-memory aliasing once each producer-consumer phase releases
  // its operand storage.
  __nv_bfloat16* smem_logits0 =
      reinterpret_cast<__nv_bfloat16*>(storage.logits_smem.data);
  __nv_bfloat16* smem_epilogue_o = smem_logits0;
  auto qk_sSFA_ptr = storage.qk_tensors.smem_SFA.begin();
  E4M3OneSmemTile<cutlass::float_ue4m3_t,
                  typename CutlassCollectiveMainloop::SmemLayoutSFA,
                  decltype(qk_sSFA_ptr)>
      qk_sSFA_covered(qk_sSFA_ptr,
                      typename CutlassCollectiveMainloop::SmemLayoutSFA{},
                      blockDim.x, kCoveredSmemInitBarrier);
  auto qk_sSFA = qk_sSFA_covered.tensor();
  auto qk_sSFA_m = [&]() {
    if constexpr (!CutlassCollectiveMainloop::PadSFA_M) {
      return qk_sSFA;
    } else {
      return qk_sSFA(
          cute::make_coord(cute::_, effective_q_tile %
                                       CutlassCollectiveMainloop::SFA_M_Ratio),
          cute::_, cute::_);
    }
  }();
  auto qk_sSFB_ptr = storage.qk_tensors.smem_SFB.begin();
  E4M3OneSmemTile<cutlass::float_ue4m3_t,
                  typename CutlassCollectiveMainloop::SmemLayoutSFB,
                  decltype(qk_sSFB_ptr)>
      qk_sSFB_covered(qk_sSFB_ptr,
                      typename CutlassCollectiveMainloop::SmemLayoutSFB{},
                      blockDim.x, kCoveredSmemInitBarrier);
  auto qk_sSFB = qk_sSFB_covered.tensor();

  CutlassCollectiveMainloopK128Stage2 pv_collective;
  auto pv_tiled_mma = typename CutlassCollectiveMainloopK128Stage2::TiledMma{};
  auto pv_thread_mma = pv_tiled_mma.get_thread_slice(pv_mma_thread_idx);
  using PvSmemAllocB =
      typename CutlassCollectiveMainloopK128Stage2::SmemAllocTypeB;
  auto pv_sB_ptr = storage.v_smem_B.begin();
  ZeroSmemTile<PvSmemAllocB,
               typename CutlassCollectiveMainloopK128Stage2::SmemLayoutB,
               decltype(pv_sB_ptr)>
      pv_sB_covered(pv_sB_ptr,
                    typename CutlassCollectiveMainloopK128Stage2::SmemLayoutB{},
                    blockDim.x, kCoveredSmemInitBarrier);
  auto pv_sB = pv_sB_covered.tensor();
  auto pv_sSFB_ptr = storage.v_smem_SFB.begin();
  E4M3OneSmemTile<cutlass::float_ue4m3_t,
                  typename CutlassCollectiveMainloopK128Stage2::SmemLayoutSFB,
                  decltype(pv_sSFB_ptr)>
      pv_sSFB_covered(
          pv_sSFB_ptr,
          typename CutlassCollectiveMainloopK128Stage2::SmemLayoutSFB{},
          blockDim.x, kCoveredSmemInitBarrier);
  auto pv_sSFB = pv_sSFB_covered.tensor();

  auto qk_problem_shape_mnkl =
      cute::append<4>(qk_params.problem_shape, cute::Int<1>{});
  auto [qk_gA_mkl, qk_gB_nkl, qk_gSFA_mkl, qk_gSFB_nkl] =
      qk_collective.load_init(qk_problem_shape_mnkl, qk_params.mainloop);
  auto qk_block_tma_a = qk_params.mainloop.tma_load_a.get_slice(0);
  auto qk_block_tma_b = qk_params.mainloop.tma_load_b.get_slice(0);
  auto qk_block_tma_sfa = qk_params.mainloop.tma_load_sfa.get_slice(0);
  auto qk_block_tma_sfb = qk_params.mainloop.tma_load_sfb.get_slice(0);
  auto qk_tAsA = qk_block_tma_a.partition_D(qk_sA);
  auto qk_tBsB = qk_block_tma_b.partition_D(qk_sB);
  auto qk_tAsSFA = qk_block_tma_sfa.partition_D(qk_sSFA);
  auto qk_tBsSFB = qk_block_tma_sfb.partition_D(qk_sSFB);

  auto pv_problem_shape_mnkl =
      cute::append<4>(pv_params.problem_shape, cute::Int<1>{});
  auto [pv_gA_mkl, pv_gB_nkl, pv_gSFA_mkl, pv_gSFB_nkl] =
      pv_collective.load_init(pv_problem_shape_mnkl, pv_params.mainloop);
  (void)pv_gA_mkl;
  (void)pv_gSFA_mkl;
  auto pv_block_tma_b = pv_params.mainloop.tma_load_b.get_slice(0);
  auto pv_block_tma_sfb = pv_params.mainloop.tma_load_sfb.get_slice(0);
  auto pv_tBsB = pv_block_tma_b.partition_D(pv_sB);
  auto pv_tBsSFB = pv_block_tma_sfb.partition_D(pv_sSFB);

  auto complete_manual_tma_pipeline_stage = [](auto& pipeline,
                                               auto const& state,
                                               uint32_t transaction_bytes) {
    cutlass::arch::fence_view_shared();
    auto* barrier = pipeline.producer_get_barrier(state);
    cutlass::arch::ClusterTransactionBarrier::complete_transaction(
        barrier, cute::block_rank_in_cluster(), transaction_bytes);
  };

  auto stage_paged_k_tile = [&](int kv_tile, int k_outer, int write_stage) {
    if constexpr (kUsePagedKv) {
      using QkSmemLayoutB = typename CutlassCollectiveMainloop::SmemLayoutB;
      static_assert(decltype(cute::size<1>(QkSmemLayoutB{}))::value >= 8,
                    "Paged K producer requires an inner-K layout extent large "
                    "enough for byte-packed FP4 writes.");
      auto smem_tiled_copy_B = cute::make_tiled_copy_B(
          typename CutlassCollectiveMainloop::SmemCopyAtomB{}, qk_tiled_mma);
      auto cB = cute::make_identity_tensor(
          cute::make_shape(cute::Int<kCutlassTileN>{},
                           cute::Int<kCutlassTileK>{}, cute::Int<1>{}));

      for (int copy_thread = load_thread_idx;
           copy_thread < CutlassCollectiveMainloop::ThreadCount;
           copy_thread += kSm120Nvfp4FmhaLoadThreadCount) {
        auto smem_thr_copy_B =
            smem_tiled_copy_B.get_thread_slice(copy_thread);
        auto tBsB_prod = smem_thr_copy_B.partition_D(qk_sB);
        auto tBcB_prod = smem_thr_copy_B.partition_D(cB);
        auto K_BLOCK_MAX_PROD = cute::size<2>(tBsB_prod);
        cute::for_each(cute::make_int_sequence<K_BLOCK_MAX_PROD>{},
                       [&](auto k_block) {
          auto dst = tBsB_prod(_, _, k_block, write_stage);
          auto coord_tensor = tBcB_prod(_, _, k_block, cute::Int<0>{});
          // K producer emits one 32-bit word for each 8-nibble partition.
          if (int(cute::size(dst)) % 8 != 0) {
            SM120_NVFP4_DEBUG_TRAP();
          }
          for (int i = 0; i < int(cute::size(dst)); i += 8) {
            auto coord0 = coord_tensor(i);
            const int row0 = int(cute::get<0>(coord0));
            const int k0 = int(cute::get<1>(coord0));
            auto ref0 = dst(i);
            uint8_t* dst0 = cute::recast_ptr<uint8_t>(&ref0);
            // K producer requires 4-byte-aligned contiguous dim stride.
            if (((reinterpret_cast<uintptr_t>(dst0) & 3u) != 0u) ||
                ((k0 & 7) != 0) || paged_kv_params.k_stride_dim3 != 1) {
              SM120_NVFP4_DEBUG_TRAP();
            }
            const int token = kv_tile * kCutlassTileN + row0;
            const int dim0 = k_outer * kCutlassTileK + k0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              auto coord = coord_tensor(i + j);
              auto ref = dst(i + j);
              uint8_t* dst_byte = cute::recast_ptr<uint8_t>(&ref);
              const int row = int(cute::get<0>(coord));
              const int k = int(cute::get<1>(coord));
              // CUTLASS B smem must colocate the 8 logical K nibbles in 4 bytes.
              if (row != row0 || k != k0 + j ||
                  dst_byte != dst0 + (j >> 1)) {
                SM120_NVFP4_DEBUG_TRAP();
              }
            }
            const bool in_bounds = token < kv_len_tokens;
            const uint32_t* src =
                in_bounds
                    ? sm120_nvfp4_paged_k_word_ptr(paged_kv_params, token,
                                                   dim0)
                    : reinterpret_cast<const uint32_t*>(paged_kv_params.k_pages);
            cp_async::pred_load_32b<cp_async::SharedMemFillMode::kFillZero>(
                reinterpret_cast<uint32_t*>(dst0), src, in_bounds);
          }
        });
      }

      cp_async::commit_group();

      constexpr int kKScaleCols = kCutlassTileK / 16;
      for (int idx = load_thread_idx; idx < kCutlassTileN * kKScaleCols;
           idx += kSm120Nvfp4FmhaLoadThreadCount) {
        const int row = idx / kKScaleCols;
        const int local_scale_col = idx - row * kKScaleCols;
        const int token = kv_tile * kCutlassTileN + row;
        const int scale_col = k_outer * kKScaleCols + local_scale_col;
        uint8_t scale = 0x38;
        if (token < kv_len_tokens) {
          const int logical_page = token / paged_kv_params.page_size;
          const int page_offset =
              token - logical_page * paged_kv_params.page_size;
          const int physical_page =
              paged_kv_params.block_table[logical_page];
          const int64_t scale_page_base =
              sm120_nvfp4_paged_k_scale_page_base(paged_kv_params,
                                                   physical_page);
          scale = sm120_nvfp4_paged_k_scale_from_page_base(
              paged_kv_params, scale_page_base, page_offset, scale_col);
        }
#pragma unroll
        for (int k_offset = 0; k_offset < 16; k_offset += 2) {
          qk_sSFB(row, local_scale_col * 16 + k_offset, write_stage) =
              make_ue4m3_raw(scale);
        }
      }
      cp_async::wait_group<0>();
    }
  };

  auto stage_paged_v_tile = [&](int kv_tile, int effective_out_group_idx,
                                int write_stage) {
    if constexpr (kUsePagedKv) {
      auto pv_scale_pair_for = [&](int token_group_start, int dim0,
                                   uint8_t& sf0, uint8_t& sf1) {
        float max_abs0 = 0.0f;
        float max_abs1 = 0.0f;
        if constexpr (!kPvLayoutV) {
          if (paged_kv_params.v_linear_scale_cache != nullptr) {
            const int token_group = token_group_start >> 4;
            if (token_group < paged_kv_params.v_linear_scale_cache_groups) {
              sf0 = sm120_nvfp4_linear_v_scale_cache_load(
                  paged_kv_params, batch_idx, paged_kv_params.kv_head, dim0,
                  token_group);
              sf1 = sm120_nvfp4_linear_v_scale_cache_load(
                  paged_kv_params, batch_idx, paged_kv_params.kv_head,
                  dim0 + 1, token_group);
              return;
            }
          }
          if (token_group_start < kv_len_tokens) {
            const int logical_page =
                token_group_start / paged_kv_params.page_size;
            const int page_offset0 =
                token_group_start - logical_page * paged_kv_params.page_size;
            const int physical_page = paged_kv_params.block_table[logical_page];
            const int64_t data_page_base =
                sm120_nvfp4_paged_v_data_page_base(paged_kv_params,
                                                    physical_page);
            const int64_t scale_page_base =
                sm120_nvfp4_paged_v_scale_page_base(paged_kv_params,
                                                     physical_page);
            const int packed_col = dim0 >> 1;
            const int scale_col = dim0 >> 4;
            const int shift0 = (dim0 & 1) * 4;
            const int shift1 = ((dim0 + 1) & 1) * 4;
#pragma unroll 1
            for (int offset = 0; offset < 16; ++offset) {
              const int t = token_group_start + offset;
              if (t < kv_len_tokens) {
                const int page_offset = page_offset0 + offset;
                const uint8_t packed =
                    sm120_nvfp4_paged_v_code_pair_from_page_base(
                        paged_kv_params, data_page_base, page_offset,
                        packed_col);
                const uint8_t scale_byte =
                    sm120_nvfp4_paged_v_linear_scale_from_page_base(
                        paged_kv_params, scale_page_base, page_offset,
                        scale_col);
                const float scale = e4m3_byte_to_fp32(scale_byte);
                const float val0 =
                    e2m1_code_to_fp32(
                        static_cast<uint8_t>((packed >> shift0) & 0x0f)) *
                    scale;
                const float val1 =
                    e2m1_code_to_fp32(
                        static_cast<uint8_t>((packed >> shift1) & 0x0f)) *
                    scale;
                max_abs0 = fmaxf(max_abs0, fabsf(val0));
                max_abs1 = fmaxf(max_abs1, fabsf(val1));
              }
            }
          }
        }
        const float scale0 = max_abs0 > 0.0f ? max_abs0 / 6.0f : 1.0f;
        const float scale1 = max_abs1 > 0.0f ? max_abs1 / 6.0f : 1.0f;
        sf0 = fp32_to_e4m3_byte(scale0);
        sf1 = fp32_to_e4m3_byte(scale1);
      };

      auto pv_scale_for = [&](int token, int dim) {
        const int token_group_start = (token / 16) * 16;
        const int scale_k = token_group_start - kv_tile * kCutlassTileN;
        auto scale_ref = pv_sSFB(dim - effective_out_group_idx * kOutputTileN,
                                 scale_k, write_stage);
        return *cute::recast_ptr<uint8_t>(&scale_ref);
      };

      if constexpr (!kPvLayoutV) {
        constexpr int kTokenScaleGroups = kCutlassTileN / 16;
        for (int idx = load_thread_idx;
             idx < (kOutputTileN / 2) * kTokenScaleGroups;
             idx += kSm120Nvfp4FmhaLoadThreadCount) {
          const int col_pair = idx / kTokenScaleGroups;
          const int token_group = idx - col_pair * kTokenScaleGroups;
          const int col0 = 2 * col_pair;
          const int local_k0 = 16 * token_group;
          const int token_group_start = kv_tile * kCutlassTileN + local_k0;
          const int dim0 = effective_out_group_idx * kOutputTileN + col0;
          uint8_t sf0 = 0x38;
          uint8_t sf1 = 0x38;
          pv_scale_pair_for(token_group_start, dim0, sf0, sf1);
#pragma unroll
          for (int k_offset = 0; k_offset < 16; k_offset += 2) {
            pv_sSFB(col0, local_k0 + k_offset, write_stage) =
                make_ue4m3_raw(sf0);
            pv_sSFB(col0 + 1, local_k0 + k_offset, write_stage) =
                make_ue4m3_raw(sf1);
          }
        }
        load_group_sync();
      }

      if constexpr (kPvLayoutV) {
        if (paged_kv_params.v_stride_dim3 != 1) {
          SM120_NVFP4_DEBUG_TRAP();
        }
        constexpr int kTransposeTokens = 8;
        constexpr int kTransposeDims = 8;
        constexpr int kTokenGroups = kCutlassTileN / kTransposeTokens;
        constexpr int kDimGroups = kOutputTileN / kTransposeDims;
        constexpr int kWarpTransposeGroups = kSm120Nvfp4FmhaLoadThreadCount / 8;
        const int load_subgroup = load_thread_idx >> 3;
        const int subgroup = lane_idx >> 3;
        const int subgroup_lane = lane_idx & 7;
        const int subgroup_base_lane = lane_idx & ~7;
        const unsigned subgroup_mask =
            static_cast<unsigned>(0xffu << (subgroup * 8));
        for (int tile = load_subgroup; tile < kTokenGroups * kDimGroups;
             tile += kWarpTransposeGroups) {
          const int token_group = tile % kTokenGroups;
          const int dim_group = tile / kTokenGroups;
          const int local_k0 = token_group * kTransposeTokens;
          const int local_dim0 = dim_group * kTransposeDims;
          const int token =
              kv_tile * kCutlassTileN + local_k0 + subgroup_lane;
          const int dim_base =
              effective_out_group_idx * kOutputTileN + local_dim0;
          uint32_t row_word = 0;
          if (token < kv_len_tokens) {
            const int logical_page =
                token / paged_kv_params.page_size;
            const int page_offset =
                token - logical_page * paged_kv_params.page_size;
            const int physical_page =
                paged_kv_params.block_table[logical_page];
            const int64_t data_page_base =
                sm120_nvfp4_paged_v_data_page_base(paged_kv_params,
                                                    physical_page);
            row_word = sm120_nvfp4_paged_v_word_from_page_base(
                paged_kv_params, data_page_base, page_offset, dim_base >> 1);
          }

          uint32_t packed_word = 0;
#pragma unroll
          for (int src_lane = 0; src_lane < 8; ++src_lane) {
            const uint32_t peer_word =
                __shfl_sync(subgroup_mask, row_word, subgroup_base_lane + src_lane);
            const uint8_t code = static_cast<uint8_t>(
                (peer_word >> (4 * subgroup_lane)) & 0x0f);
            packed_word |= static_cast<uint32_t>(code) << (4 * src_lane);
          }

          const int local_col = local_dim0 + subgroup_lane;
          auto ref0 = pv_sB(local_col, local_k0, write_stage);
          uint8_t* dst0 = cute::recast_ptr<uint8_t>(&ref0);
          if ((reinterpret_cast<uintptr_t>(dst0) & 3u) != 0u) {
            SM120_NVFP4_DEBUG_TRAP();
          }
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            auto ref = pv_sB(local_col, local_k0 + j, write_stage);
            auto pair_ref =
                pv_sB(local_col, local_k0 + (j ^ 1), write_stage);
            uint8_t* dst_byte = cute::recast_ptr<uint8_t>(&ref);
            uint8_t* pair_byte = cute::recast_ptr<uint8_t>(&pair_ref);
            if (dst_byte < dst0 || dst_byte >= dst0 + 4 ||
                pair_byte != dst_byte || int(dst_byte - dst0) != (j >> 1)) {
              SM120_NVFP4_DEBUG_TRAP();
            }
          }
          *reinterpret_cast<uint32_t*>(dst0) = packed_word;
        }
      } else {
        if (paged_kv_params.v_stride_dim3 != 1 ||
            paged_kv_params.v_scale_stride_dim3 != 1) {
          SM120_NVFP4_DEBUG_TRAP();
        }
        constexpr int kTransposeTokens = 8;
        constexpr int kTransposeDims = 8;
        constexpr int kTokenGroups = kCutlassTileN / kTransposeTokens;
        constexpr int kDimGroups = kOutputTileN / kTransposeDims;
        constexpr int kWarpTransposeGroups = kSm120Nvfp4FmhaLoadThreadCount / 8;
        const int load_subgroup = load_thread_idx >> 3;
        const int subgroup = lane_idx >> 3;
        const int subgroup_lane = lane_idx & 7;
        const int subgroup_base_lane = lane_idx & ~7;
        const unsigned subgroup_mask =
            static_cast<unsigned>(0xffu << (subgroup * 8));
        for (int tile = load_subgroup; tile < kTokenGroups * kDimGroups;
             tile += kWarpTransposeGroups) {
          const int token_group = tile % kTokenGroups;
          const int dim_group = tile / kTokenGroups;
          const int local_k0 = token_group * kTransposeTokens;
          const int local_dim0 = dim_group * kTransposeDims;
          const int token =
              kv_tile * kCutlassTileN + local_k0 + subgroup_lane;
          const int dim_base =
              effective_out_group_idx * kOutputTileN + local_dim0;
          const int scale_col = dim_base >> 4;
          const int local_col = local_dim0 + subgroup_lane;
          const int dim = effective_out_group_idx * kOutputTileN + local_col;
          const int token0 = kv_tile * kCutlassTileN + local_k0;
          uint32_t row_word = 0;
          uint32_t row_scale_byte = 0x38u;
          if (token < kv_len_tokens) {
            if (paged_kv_params.v_linear_data_cache != nullptr) {
              row_word = sm120_nvfp4_linear_v_data_cache_word(
                  paged_kv_params, batch_idx, paged_kv_params.kv_head, token,
                  dim_base >> 1);
            } else {
              const int logical_page =
                  token / paged_kv_params.page_size;
              const int page_offset =
                  token - logical_page * paged_kv_params.page_size;
              const int physical_page =
                  paged_kv_params.block_table[logical_page];
              const int64_t data_page_base =
                  sm120_nvfp4_paged_v_data_page_base(paged_kv_params,
                                                      physical_page);
              const int64_t scale_page_base =
                  sm120_nvfp4_paged_v_scale_page_base(paged_kv_params,
                                                       physical_page);
              row_word = sm120_nvfp4_paged_v_word_from_page_base(
                  paged_kv_params, data_page_base, page_offset, dim_base >> 1);
              row_scale_byte = sm120_nvfp4_paged_v_linear_scale_from_page_base(
                  paged_kv_params, scale_page_base, page_offset, scale_col);
            }
          }

          const uint8_t output_scale =
              token0 < kv_len_tokens ? pv_scale_for(token0, dim) : 0x38;
          uint32_t packed_word = 0;
          if (paged_kv_params.v_linear_data_cache != nullptr) {
#pragma unroll
            for (int src_lane = 0; src_lane < 8; ++src_lane) {
              const uint32_t peer_word =
                  __shfl_sync(subgroup_mask, row_word,
                              subgroup_base_lane + src_lane);
              const uint8_t code = static_cast<uint8_t>(
                  (peer_word >> (4 * subgroup_lane)) & 0x0f);
              packed_word |= static_cast<uint32_t>(code) << (4 * src_lane);
            }
          } else {
            packed_word = sm120_nvfp4_linear_v_requant_transposed_word(
                row_word, row_scale_byte, output_scale, subgroup_mask,
                subgroup_base_lane, subgroup_lane);
          }

          auto ref0 = pv_sB(local_col, local_k0, write_stage);
          uint8_t* dst0 = cute::recast_ptr<uint8_t>(&ref0);
          if ((reinterpret_cast<uintptr_t>(dst0) & 3u) != 0u) {
            SM120_NVFP4_DEBUG_TRAP();
          }
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            auto ref = pv_sB(local_col, local_k0 + j, write_stage);
            auto pair_ref =
                pv_sB(local_col, local_k0 + (j ^ 1), write_stage);
            uint8_t* dst_byte = cute::recast_ptr<uint8_t>(&ref);
            uint8_t* pair_byte = cute::recast_ptr<uint8_t>(&pair_ref);
            if (dst_byte < dst0 || dst_byte >= dst0 + 4 ||
                pair_byte != dst_byte || int(dst_byte - dst0) != (j >> 1)) {
              SM120_NVFP4_DEBUG_TRAP();
            }
          }
          *reinterpret_cast<uint32_t*>(dst0) = packed_word;
        }
      }

      if constexpr (kPvLayoutV) {
        constexpr int kTokenScaleGroups = kCutlassTileN / 16;
        for (int idx = load_thread_idx; idx < kOutputTileN * kTokenScaleGroups;
             idx += kSm120Nvfp4FmhaLoadThreadCount) {
          const int col = idx / kTokenScaleGroups;
          const int token_group = idx - col * kTokenScaleGroups;
          const int k0 = token_group * 16;
          const int token = kv_tile * kCutlassTileN + k0;
          const int dim = effective_out_group_idx * kOutputTileN + col;
          uint8_t scale = 0x38;
          if (token < kv_len_tokens) {
            const int logical_page = token / paged_kv_params.page_size;
            const int physical_page =
                paged_kv_params.block_table[logical_page];
            scale = sm120_nvfp4_paged_v_pv_scale_from_physical_page(
                paged_kv_params, physical_page, dim);
          }
#pragma unroll
          for (int k_offset = 0; k_offset < 16; k_offset += 2) {
            const uint8_t store_scale =
                token + k_offset < kv_len_tokens ? scale : 0x38;
            pv_sSFB(col, k0 + k_offset, write_stage) =
                make_ue4m3_raw(store_scale);
          }
        }
      }
    }
  };

  auto stage_bf16_q_tile = [&](int k_outer, int write_stage) {
    if constexpr (kUsePagedKv) {
      auto q_row_base = [&](int row) {
        const int local_row = local_q_tile * kCutlassTileM + row;
        if (local_row >= q_len * group_size) {
          return int64_t{-1};
        }
        return sm120_nvfp4_paged_q_bf16_row_base(
            paged_kv_params, effective_q_begin, local_row, group_size,
            num_kv_heads, all_kv_heads);
      };
      auto q_value_from_row = [&](int64_t row_base, int dim) {
        if (row_base < 0) {
          return 0.0f;
        }
        return sm120_nvfp4_paged_q_bf16_value_from_row_base(
            paged_kv_params, row_base, dim);
      };
      auto q_scale_byte = [&](int64_t row_base, int scale_col) {
        float max_abs = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const int dim = scale_col * 16 + i;
          max_abs = fmaxf(max_abs, fabsf(q_value_from_row(row_base, dim)));
        }
        return fp32_to_e4m3_byte(fmaxf(max_abs / 6.0f, 1.0e-8f));
      };
      auto q_code = [&](int64_t row_base, int dim, uint8_t scale_byte) {
        const float scale = fmaxf(e4m3_byte_to_fp32(scale_byte), 1.0e-8f);
        return fp32_to_e2m1_code_hw(q_value_from_row(row_base, dim) / scale);
      };

      auto smem_tiled_copy_A = cute::make_tiled_copy_A(
          typename CutlassCollectiveMainloop::SmemCopyAtomA{}, qk_tiled_mma);
      auto cA = cute::make_identity_tensor(
          cute::make_shape(cute::Int<kCutlassTileM>{},
                           cute::Int<kCutlassTileK>{}, cute::Int<1>{}));
      for (int copy_thread = load_thread_idx;
           copy_thread < CutlassCollectiveMainloop::ThreadCount;
           copy_thread += kSm120Nvfp4FmhaLoadThreadCount) {
        auto smem_thr_copy_A =
            smem_tiled_copy_A.get_thread_slice(copy_thread);
        auto tAsA_prod = smem_thr_copy_A.partition_D(qk_sA);
        auto tAcA_prod = smem_thr_copy_A.partition_D(cA);
        auto K_BLOCK_MAX_PROD = cute::size<2>(tAsA_prod);
        cute::for_each(cute::make_int_sequence<K_BLOCK_MAX_PROD>{},
                       [&](auto k_block) {
          auto dst = tAsA_prod(_, _, k_block, write_stage);
          auto coord_tensor = tAcA_prod(_, _, k_block, cute::Int<0>{});
          // Q producer emits one 32-bit word for each 8-nibble partition.
          if (int(cute::size(dst)) % 8 != 0) {
            SM120_NVFP4_DEBUG_TRAP();
          }
          for (int i = 0; i < int(cute::size(dst)); i += 8) {
            auto coord0 = coord_tensor(i);
            const int row0 = int(cute::get<0>(coord0));
            const int k0 = int(cute::get<1>(coord0));
            auto ref0 = dst(i);
            uint8_t* dst0 = cute::recast_ptr<uint8_t>(&ref0);
            // Q producer writes each packed word through a 4-byte smem store.
            if ((reinterpret_cast<uintptr_t>(dst0) & 3u) != 0u ||
                ((k0 & 7) != 0)) {
              SM120_NVFP4_DEBUG_TRAP();
            }
            const int64_t row_base0 = q_row_base(row0);
            const int dim0 = k_outer * kCutlassTileK + k0;
            const uint8_t word_scale = q_scale_byte(row_base0, dim0 >> 4);
            uint32_t packed_word = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              auto coord = coord_tensor(i + j);
              auto ref = dst(i + j);
              auto pair_ref = dst(i + (j ^ 1));
              uint8_t* dst_byte = cute::recast_ptr<uint8_t>(&ref);
              uint8_t* pair_byte = cute::recast_ptr<uint8_t>(&pair_ref);
              const int row = int(cute::get<0>(coord));
              const int k = int(cute::get<1>(coord));
              // Q smem layout must colocate each FP4 pair inside the word.
              if (row != row0 || k != k0 + j || dst_byte < dst0 ||
                  dst_byte >= dst0 + 4 || pair_byte != dst_byte) {
                SM120_NVFP4_DEBUG_TRAP();
              }
              const int dim = k_outer * kCutlassTileK + k;
              const uint8_t code = q_code(row_base0, dim, word_scale);
              const int byte_offset = int(dst_byte - dst0);
              const int nibble_shift = (k & 1) ? 4 : 0;
              packed_word |= static_cast<uint32_t>(code)
                             << (8 * byte_offset + nibble_shift);
            }
            *reinterpret_cast<uint32_t*>(dst0) = packed_word;
          }
        });
      }

      for (int idx = load_thread_idx; idx < kCutlassTileM * kCutlassTileK / 16;
           idx += kSm120Nvfp4FmhaLoadThreadCount) {
        const int row = idx / (kCutlassTileK / 16);
        const int local_scale_col = idx - row * (kCutlassTileK / 16);
        const int k0 = local_scale_col * 16;
        const int scale_col = (k_outer * kCutlassTileK + k0) >> 4;
        const uint8_t scale = q_scale_byte(q_row_base(row), scale_col);
#pragma unroll
        for (int k_offset = 0; k_offset < 16; k_offset += 2) {
          qk_sSFA_m(row, k0 + k_offset, write_stage) =
              make_ue4m3_raw(scale);
        }
      }
    }
  };

  auto load_q_chunk = [&](int k_outer) {
    if constexpr (kUsePagedKv) {
      if (paged_kv_params.q_bf16 != nullptr) {
        if (is_load) {
          if (load_leader) {
            q_pipeline.producer_acquire(q_pipe_write);
          }
          load_group_sync();
          const int write_stage = q_pipe_write.index();
          stage_bf16_q_tile(k_outer, write_stage);
          cutlass::arch::fence_view_shared();
          load_group_sync();
          if (load_leader) {
            complete_manual_tma_pipeline_stage(
                q_pipeline, q_pipe_write,
                qk_params.mainloop.tma_transaction_bytes_mk);
          }
          load_group_sync();
          ++q_pipe_write;
        }
        return;
      }
    }
    if (load_leader) {
      auto gA = qk_gA_mkl(_, _, effective_q_tile, _, 0);
      auto broadcast_m = cute::make_layout(
          cute::make_shape(
              cute::Int<CutlassCollectiveMainloop::SFA_M_Ratio>{},
              cute::Int<cute::numeric_limits<int>::max()>{}),
          cute::make_stride(
              cute::_0{},
              cute::Int<CutlassCollectiveMainloop::SFA_M_Ratio>{}));
      auto gSFA = qk_gSFA_mkl(_, _, broadcast_m(effective_q_tile), _, 0);
      auto tAgA = qk_block_tma_a.partition_S(gA);
      auto tAgSFA = qk_block_tma_sfa.partition_S(gSFA);
      auto k_tile_iter = cute::make_coord_iterator(
          cute::idx2crd(k_outer, cute::shape<3>(qk_gA_mkl)),
          cute::shape<3>(qk_gA_mkl));
      q_pipeline.producer_acquire(q_pipe_write);
      using BarrierType = typename QkPipeline::ProducerBarrierType;
      BarrierType* tma_barrier =
          q_pipeline.producer_get_barrier(q_pipe_write);
      const int write_stage = q_pipe_write.index();
      cute::copy(qk_params.mainloop.tma_load_a.with(*tma_barrier),
                 tAgA(_, _, _, *k_tile_iter),
                 qk_tAsA(_, _, _, write_stage));
      cute::copy(qk_params.mainloop.tma_load_sfa.with(*tma_barrier),
                 tAgSFA(_, _, _, *k_tile_iter),
                 qk_tAsSFA(_, _, _, write_stage));
      ++q_pipe_write;
    }
  };

  auto load_k_chunk = [&](int kv_tile, int k_outer) {
    if constexpr (kUsePagedKv) {
      if (is_load) {
        if (load_leader) {
          k_pipeline.producer_acquire(k_pipe_write);
        }
        load_group_sync();
        const int write_stage = k_pipe_write.index();
        stage_paged_k_tile(kv_tile, k_outer, write_stage);
        cutlass::arch::fence_view_shared();
        load_group_sync();
        if (load_leader) {
          complete_manual_tma_pipeline_stage(
              k_pipeline, k_pipe_write,
              qk_params.mainloop.tma_transaction_bytes_nk);
        }
        load_group_sync();
        ++k_pipe_write;
      }
    } else if (load_leader) {
      auto gB = qk_gB_nkl(_, _, kv_tile, _, 0);
      auto gSFB = qk_gSFB_nkl(_, _, kv_tile, _, 0);
      auto tBgB = qk_block_tma_b.partition_S(gB);
      auto tBgSFB = qk_block_tma_sfb.partition_S(gSFB);
      auto k_tile_iter = cute::make_coord_iterator(
          cute::idx2crd(k_outer, cute::shape<3>(qk_gB_nkl)),
          cute::shape<3>(qk_gB_nkl));
      k_pipeline.producer_acquire(k_pipe_write);
      using BarrierType = typename QkPipeline::ProducerBarrierType;
      BarrierType* tma_barrier =
          k_pipeline.producer_get_barrier(k_pipe_write);
      const int write_stage = k_pipe_write.index();
      cute::copy(qk_params.mainloop.tma_load_b.with(*tma_barrier),
                 tBgB(_, _, _, *k_tile_iter),
                 qk_tBsB(_, _, _, write_stage));
      cute::copy(qk_params.mainloop.tma_load_sfb.with(*tma_barrier),
                 tBgSFB(_, _, _, *k_tile_iter),
                 qk_tBsSFB(_, _, _, write_stage));
      ++k_pipe_write;
    }
  };

  auto load_v_chunk = [&](int kv_tile, int group_offset) {
    const int effective_out_group_idx =
        effective_out_group_base + group_offset;
    if constexpr (kUsePagedKv) {
      if (is_load) {
        if (load_leader) {
          v_pipeline.producer_acquire(v_pipe_write);
        }
        load_group_sync();
        const int write_stage = v_pipe_write.index();
        stage_paged_v_tile(kv_tile, effective_out_group_idx, write_stage);
        cutlass::arch::fence_view_shared();
        load_group_sync();
        if (load_leader) {
          complete_manual_tma_pipeline_stage(
              v_pipeline, v_pipe_write,
              pv_params.mainloop.tma_transaction_bytes_nk);
        }
        load_group_sync();
        ++v_pipe_write;
      }
    } else if (load_leader) {
      auto gB = pv_gB_nkl(_, _, effective_out_group_idx, _, 0);
      auto gSFB = pv_gSFB_nkl(_, _, effective_out_group_idx, _, 0);
      auto tBgB = pv_block_tma_b.partition_S(gB);
      auto tBgSFB = pv_block_tma_sfb.partition_S(gSFB);
      auto k_tile_iter = cute::make_coord_iterator(
          cute::idx2crd(kv_tile, cute::shape<3>(pv_gB_nkl)),
          cute::shape<3>(pv_gB_nkl));
      v_pipeline.producer_acquire(v_pipe_write);
      using BarrierType = typename VPipeline::ProducerBarrierType;
      BarrierType* tma_barrier =
          v_pipeline.producer_get_barrier(v_pipe_write);
      const int write_stage = v_pipe_write.index();
      cute::copy(pv_params.mainloop.tma_load_b.with(*tma_barrier),
                 tBgB(_, _, _, *k_tile_iter),
                 pv_tBsB(_, _, _, write_stage));
      cute::copy(pv_params.mainloop.tma_load_sfb.with(*tma_barrier),
                 tBgSFB(_, _, _, *k_tile_iter),
                 pv_tBsSFB(_, _, _, write_stage));
      ++v_pipe_write;
    }
  };

  auto load_v_group_span = [&](int kv_tile) {
#pragma unroll
    for (int group_offset = 0; group_offset < kOutputGroupSpan;
         ++group_offset) {
      load_v_chunk(kv_tile, group_offset);
    }
  };

  using PvSmemAllocA =
      typename CutlassCollectiveMainloopK128Stage2::SmemAllocTypeA;
  uint8_t* p_smem_a0_bytes = nullptr;
  if constexpr (kSm120Nvfp4AliasP0InQkA) {
    p_smem_a0_bytes =
        cute::recast_ptr<uint8_t>(storage.qk_tensors.smem_A.begin());
  } else {
    p_smem_a0_bytes = storage.p_smem_A0.data;
  }
  cutlass::float_ue4m3_t* p_smem_sfa0 = nullptr;
  if constexpr (kSm120Nvfp4AliasP0ScaleInQkSFA) {
    p_smem_sfa0 = storage.qk_tensors.smem_SFA.begin();
  } else {
    p_smem_sfa0 = storage.p_smem_SFA0.begin();
  }
  auto p_sA0_ptr = cute::recast_ptr<PvSmemAllocA>(p_smem_a0_bytes);
  ZeroSmemTile<PvSmemAllocA,
               typename CutlassCollectiveMainloopK128Stage2::SmemLayoutA,
               decltype(p_sA0_ptr)>
      p_sA0_covered(p_sA0_ptr,
                    typename CutlassCollectiveMainloopK128Stage2::SmemLayoutA{},
                    blockDim.x, kCoveredSmemInitBarrier);
  auto p_sA0 = p_sA0_covered.tensor();
  E4M3OneSmemTile<cutlass::float_ue4m3_t,
                  typename CutlassCollectiveMainloopK128Stage2::SmemLayoutSFA,
                  decltype(p_smem_sfa0)>
      p_sSFA0_covered(
          p_smem_sfa0,
          typename CutlassCollectiveMainloopK128Stage2::SmemLayoutSFA{},
          blockDim.x, kCoveredSmemInitBarrier);
  auto p_sSFA0 = p_sSFA0_covered.tensor();
  auto p_sSFA0_m = [&]() {
    if constexpr (!CutlassCollectiveMainloopK128Stage2::PadSFA_M) {
      return p_sSFA0;
    } else {
      return p_sSFA0(
          cute::make_coord(
              cute::_,
              effective_q_tile % CutlassCollectiveMainloopK128Stage2::SFA_M_Ratio),
          cute::_, cute::_);
    }
  }();

  auto consume_and_store_output_span = [&](int store_thread_idx) {
#pragma unroll
    for (int group_offset = 0; group_offset < kOutputGroupSpan;
         ++group_offset) {
      pipeline_corr_epi.consumer_wait(pipeline_corr_epi_consumer_state);
      sm120_epilogue_store_bf16_tile(
          smem_epilogue_o, out_tile + group_offset * kOutputTileN,
          out_stride_cols, store_thread_idx);
      if (group_offset == 0 && split_m != nullptr && split_l != nullptr) {
        for (int row = store_thread_idx; row < kCutlassTileM;
             row += kSm120Nvfp4FmhaOutputThreadCount) {
          const int global_row = effective_q_tile * kCutlassTileM + row;
          const int stats_idx =
              effective_split_idx * split_stats_stride_rows + global_row;
          split_m[stats_idx] = storage.global_m[row];
          split_l[stats_idx] = storage.global_l[row];
        }
      }
      pipeline_corr_epi.consumer_release(pipeline_corr_epi_consumer_state);
      ++pipeline_corr_epi_consumer_state;
    }
  };

  if (is_load) {
    load_q_chunk(0);
    load_k_chunk(effective_kv_tile_start, 0);
    if (qk_head_chunks > 1) {
      load_q_chunk(1);
    }
    load_v_group_span(effective_kv_tile_start);
    if (qk_head_chunks > 1) {
      load_k_chunk(effective_kv_tile_start, 1);
    }
    if (first_load_warp) {
      qk_collective.load_tail(q_pipeline, q_pipe_write);
    }

    for (int tile = 0; tile < effective_num_kv_tiles; ++tile) {
      if (tile + 1 < effective_num_kv_tiles) {
        const int next_kv_tile = effective_kv_tile_start + tile + 1;
        load_v_group_span(next_kv_tile);
        load_k_chunk(next_kv_tile, 0);
        if (qk_head_chunks > 1) {
          load_k_chunk(next_kv_tile, 1);
        }
      }
    }

    if (first_load_warp) {
      qk_collective.load_tail(k_pipeline, k_pipe_write);
      pv_collective.load_tail(v_pipeline, v_pipe_write);
    }
  } else if (is_mma) {
    auto q_frag0 =
        qk_thread_mma.partition_fragment_A(qk_sA(_, _, cute::Int<0>{}));
    auto q_frag1 =
        qk_thread_mma.partition_fragment_A(qk_sA(_, _, cute::Int<0>{}));
    auto q_scale_frag0 =
        qk_collective.partition_fragment_SFA(qk_sSFA_m(_, _, cute::Int<0>{}),
                                             qk_thread_mma);
    auto q_scale_frag1 =
        qk_collective.partition_fragment_SFA(qk_sSFA_m(_, _, cute::Int<0>{}),
                                             qk_thread_mma);
    auto v_frag =
        pv_thread_mma.partition_fragment_B(pv_sB(_, _, cute::Int<0>{}));
    auto v_scale_frag =
        pv_collective.partition_fragment_SFB(pv_sSFB(_, _, cute::Int<0>{}),
                                             pv_thread_mma);
    auto pv_accum0 = cute::partition_fragment_C(
        pv_tiled_mma, cute::take<0, 2>(CutlassThreadBlockShapeK128{}));
    cutlass_qk_tma_q_register_stage(
        q_pipeline, q_pipe_read, q_frag0, q_scale_frag0, qk_mma_thread_idx,
        qk_sA, qk_sSFA_m);
    if (qk_head_chunks > 1) {
      cutlass_qk_tma_q_register_stage(
          q_pipeline, q_pipe_read, q_frag1, q_scale_frag1, qk_mma_thread_idx,
          qk_sA, qk_sSFA_m);
    }
    cute::clear(pv_accum0);
    k_pipe_release = k_pipe_read;

    auto qk_tCrB =
        qk_thread_mma.partition_fragment_B(qk_sB(_, _, cute::Int<0>{}));
    auto qk_tCrSFB =
        qk_collective.partition_fragment_SFB(qk_sSFB(_, _, cute::Int<0>{}),
                                             qk_thread_mma);
    auto qk_smem_tiled_copy_B = cute::make_tiled_copy_B(
        typename CutlassCollectiveMainloop::SmemCopyAtomB{}, qk_tiled_mma);
    auto qk_smem_thr_copy_B =
        qk_smem_tiled_copy_B.get_thread_slice(qk_mma_thread_idx);
    auto qk_tCsB = qk_smem_thr_copy_B.partition_S(
        cute::as_position_independent_swizzle_tensor(qk_sB));
    auto qk_tCrB_copy_view = qk_smem_thr_copy_B.retile_D(qk_tCrB);
    auto qk_tile_shape_mnk = cute::tile_shape(qk_tiled_mma);
    auto qk_smem_tiled_copy_SFB = cute::make_tiled_copy_impl(
        typename CutlassCollectiveMainloop::SmemCopyAtomSFB{},
        qk_collective.get_layoutSFB_TV(qk_tiled_mma),
        cute::make_shape(cute::size<1>(qk_tile_shape_mnk),
                         cute::size<2>(qk_tile_shape_mnk)));
    auto qk_smem_thr_copy_SFB =
        qk_smem_tiled_copy_SFB.get_thread_slice(qk_mma_thread_idx);
    auto qk_tCsSFB = qk_smem_thr_copy_SFB.partition_S(
        cute::as_position_independent_swizzle_tensor(qk_sSFB));
    auto qk_tCrSFB_copy_view = qk_smem_thr_copy_SFB.retile_D(qk_tCrSFB);

    auto qk_consume_k_stage = [&](auto const& q_frag,
                                  auto const& q_scale_frag, auto& qk_accum) {
      auto qk_K_BLOCK_MAX = cute::size<2>(q_frag);
      const int read_stage = k_pipe_read.index();
      auto qk_tCsB_stage = qk_tCsB(_, _, _, read_stage);
      auto qk_tCsSFB_stage = qk_tCsSFB(_, _, _, read_stage);

      auto qk_copy_kblock = [&](auto k_block) {
        cute::copy(qk_smem_tiled_copy_B, qk_tCsB_stage(_, _, k_block),
                   qk_tCrB_copy_view(_, _, k_block));
        using MMAOp = typename CutlassCollectiveMainloop::TiledMma::MMA_Op;
        fp4_shift_B(MMAOp{}, qk_tCrB_copy_view(_, _, k_block));
        cute::copy(qk_tCsSFB_stage(_, _, k_block),
                   qk_tCrSFB_copy_view(_, _, k_block));
      };

      auto qk_gemm_kblock = [&](auto k_block) {
        cute::gemm(qk_tiled_mma,
                   cute::make_zip_tensor(q_frag(_, _, k_block),
                                          q_scale_frag(_, _, k_block)),
                   cute::make_zip_tensor(qk_tCrB(_, _, k_block),
                                          qk_tCrSFB(_, _, k_block)),
                   qk_accum);
      };

      k_pipeline.consumer_wait(k_pipe_read);
      qk_copy_kblock(cute::_0{});
      cute::for_each(cute::make_int_sequence<qk_K_BLOCK_MAX>{},
                     [&](auto k_block) {
        auto k_block_next =
            ((k_block + 1) == qk_K_BLOCK_MAX) ? 0 : (k_block + 1);
        if (k_block == qk_K_BLOCK_MAX - 1) {
          ++k_pipe_read;
        }
        if (k_block_next > 0) {
          qk_copy_kblock(k_block_next);
        }
        qk_gemm_kblock(k_block);
      });
    };

    auto pv_smem_tiled_copy_B = cute::make_tiled_copy_B(
        typename CutlassCollectiveMainloopK128Stage2::SmemCopyAtomB{},
        pv_tiled_mma);
    auto pv_smem_thr_copy_B =
        pv_smem_tiled_copy_B.get_thread_slice(pv_mma_thread_idx);
    auto pv_tCsB = pv_smem_thr_copy_B.partition_S(
        cute::as_position_independent_swizzle_tensor(pv_sB));
    auto pv_tCrB_copy_view = pv_smem_thr_copy_B.retile_D(v_frag);
    auto pv_tile_shape_mnk = cute::tile_shape(pv_tiled_mma);
    auto pv_smem_tiled_copy_SFB = cute::make_tiled_copy_impl(
        typename CutlassCollectiveMainloopK128Stage2::SmemCopyAtomSFB{},
        pv_collective.get_layoutSFB_TV(pv_tiled_mma),
        cute::make_shape(cute::size<1>(pv_tile_shape_mnk),
                         cute::size<2>(pv_tile_shape_mnk)));
    auto pv_smem_thr_copy_SFB =
        pv_smem_tiled_copy_SFB.get_thread_slice(pv_mma_thread_idx);
    auto pv_tCsSFB = pv_smem_thr_copy_SFB.partition_S(
        cute::as_position_independent_swizzle_tensor(pv_sSFB));
    auto pv_tCrSFB_copy_view = pv_smem_thr_copy_SFB.retile_D(v_scale_frag);

    auto pv_consume_v_stage = [&]() {
      auto pv_K_BLOCK_MAX = cute::size<2>(v_frag);
      const int read_stage = v_pipe_read.index();
      auto pv_tCsB_stage = pv_tCsB(_, _, _, read_stage);
      auto pv_tCsSFB_stage = pv_tCsSFB(_, _, _, read_stage);

      v_pipeline.consumer_wait(v_pipe_read);
      cute::for_each(cute::make_int_sequence<pv_K_BLOCK_MAX>{},
                     [&](auto k_block) {
        cute::copy(pv_smem_tiled_copy_B, pv_tCsB_stage(_, _, k_block),
                   pv_tCrB_copy_view(_, _, k_block));
        using MMAOp =
            typename CutlassCollectiveMainloopK128Stage2::TiledMma::MMA_Op;
        fp4_shift_B(MMAOp{}, pv_tCrB_copy_view(_, _, k_block));
        cute::copy(pv_tCsSFB_stage(_, _, k_block),
                   pv_tCrSFB_copy_view(_, _, k_block));
      });
    };

    auto pv_release_v_stage = [&]() {
      v_pipeline.consumer_release(v_pipe_read);
      ++v_pipe_read;
    };

    auto pv_smem_tiled_copy_A = cute::make_tiled_copy_A(
        typename CutlassCollectiveMainloopK128Stage2::SmemCopyAtomA{},
        pv_tiled_mma);
    auto pv_smem_thr_copy_A =
        pv_smem_tiled_copy_A.get_thread_slice(pv_mma_thread_idx);
    auto pv_smem_tiled_copy_SFA = cute::make_tiled_copy_impl(
        typename CutlassCollectiveMainloopK128Stage2::SmemCopyAtomSFA{},
        pv_collective.get_layoutSFA_TV(pv_tiled_mma),
        cute::make_shape(cute::size<0>(pv_tile_shape_mnk),
                         cute::size<2>(pv_tile_shape_mnk)));
    auto pv_smem_thr_copy_SFA =
        pv_smem_tiled_copy_SFA.get_thread_slice(pv_mma_thread_idx);

    auto pv_copy_p_frag = [&](auto& p_sA_stage, auto& p_sSFA_stage,
                              auto& pv_tCrA, auto& pv_tCrSFA) {
      auto pv_tCsA = pv_smem_thr_copy_A.partition_S(
          cute::as_position_independent_swizzle_tensor(p_sA_stage));
      auto pv_tCrA_copy_view = pv_smem_thr_copy_A.retile_D(pv_tCrA);
      auto pv_tCsSFA = pv_smem_thr_copy_SFA.partition_S(
          cute::as_position_independent_swizzle_tensor(p_sSFA_stage));
      auto pv_tCrSFA_copy_view = pv_smem_thr_copy_SFA.retile_D(pv_tCrSFA);
      auto pv_K_BLOCK_MAX = cute::size<2>(pv_tCrA);
      auto pv_copy_p_kblock = [&](auto k_block) {
        cute::copy(pv_smem_tiled_copy_A,
                   pv_tCsA(_, _, k_block, cute::Int<0>{}),
                   pv_tCrA_copy_view(_, _, k_block));
        using MMAOp =
            typename CutlassCollectiveMainloopK128Stage2::TiledMma::MMA_Op;
        fp4_shift_A(MMAOp{}, pv_tCrA_copy_view(_, _, k_block));
        cute::copy(pv_tCsSFA(_, _, k_block, cute::Int<0>{}),
                   pv_tCrSFA_copy_view(_, _, k_block));
      };

      pv_copy_p_kblock(cute::_0{});
      cute::for_each(cute::make_int_sequence<pv_K_BLOCK_MAX>{},
                     [&](auto k_block) {
        auto k_block_next =
            ((k_block + 1) == pv_K_BLOCK_MAX) ? 0 : (k_block + 1);
        if (k_block_next > 0) {
          pv_copy_p_kblock(k_block_next);
        }
      });
    };

    auto pv_gemm_loaded_p = [&](auto& accum, auto& pv_tCrA,
                                auto& pv_tCrSFA) {
      auto pv_K_BLOCK_MAX = cute::size<2>(pv_tCrA);
      cute::for_each(cute::make_int_sequence<pv_K_BLOCK_MAX>{},
                     [&](auto k_block) {
        cute::gemm(pv_tiled_mma,
                   cute::make_zip_tensor(pv_tCrA(_, _, k_block),
                                          pv_tCrSFA(_, _, k_block)),
                   cute::make_zip_tensor(v_frag(_, _, k_block),
                                          v_scale_frag(_, _, k_block)),
                   accum);
      });
    };

    auto pv_gemm_p_stage = [&](auto& accum, auto& p_sA_stage,
                               auto& p_sSFA_stage) {
      auto pv_tCrA =
          pv_thread_mma.partition_fragment_A(p_sA_stage(_, _, cute::Int<0>{}));
      auto pv_tCrSFA =
          pv_collective.partition_fragment_SFA(
              p_sSFA_stage(_, _, cute::Int<0>{}), pv_thread_mma);
      pv_copy_p_frag(p_sA_stage, p_sSFA_stage, pv_tCrA, pv_tCrSFA);
      pv_gemm_loaded_p(accum, pv_tCrA, pv_tCrSFA);
    };

	    auto pv_cC = cute::make_identity_tensor(
	        cute::take<0, 2>(CutlassThreadBlockShapeK128{}));
	    auto pv_tCcC = pv_thread_mma.partition_C(pv_cC);
	    static_assert(kCutlassTileN == 128,
	                  "D128 MMA-owned softmax currently assumes 128 score columns");
	    static_assert(kSoftmaxThreadsPerRow == 4);
	    static_assert((kCutlassTileN & (kCutlassTileN - 1)) == 0,
	                  "logits row skew assumes power-of-two score tile width");
	    auto logits_smem_index = [](int row, int col) {
	      const int skewed_col =
	          (col + (row & 0x0f) * kLogitsRowSkew) & (kCutlassTileN - 1);
	      return row * kCutlassTileN + skewed_col;
	    };
    auto logits_layout = cute::make_layout(
        cute::make_shape(cute::Int<kCutlassTileM * kCutlassTileN>{}));
    NegInfSmemTile<__nv_bfloat16, decltype(logits_layout)> logits0_covered(
        CoveredSmemNoInit{}, smem_logits0, logits_layout);
    auto logits0 = logits0_covered.tensor();
    ZeroSmemTile<__nv_bfloat16, decltype(logits_layout)> epilogue_o_covered(
        CoveredSmemNoInit{}, smem_epilogue_o, logits_layout);
    auto score_is_valid = [&](int row, int col, int tile) {
      const int global_q_row = local_q_tile * kCutlassTileM + row;
      if (global_q_row >= q_len * group_size) {
        return false;
      }
      const int q_token = global_q_row / group_size;
      const int q_pos = kv_len_tokens - q_len + q_token;
      const int kv_pos =
          (effective_kv_tile_start + tile) * kCutlassTileN + col;
      if (kv_pos >= kv_len_tokens) {
        return false;
      }
      if constexpr (kCausal) {
        if (kv_pos > q_pos) {
          return false;
        }
      }
      if constexpr (kUseSlidingWindow) {
        if (kv_pos < q_pos - sliding_window + 1) {
          return false;
        }
      }
      return true;
    };
    auto transform_score = [&](float logit, int row, int col, int tile) {
      if (!score_is_valid(row, col, tile)) {
        return -INFINITY;
      }
      if constexpr (kUseLogitsSoftCap) {
        logit = logits_soft_cap * tanhf(logit / logits_soft_cap);
      }
      return logit;
    };
	    const bool mma_softmax_row_owner =
	        qk_mma_thread_idx < kCutlassTileM * kSoftmaxThreadsPerRow;
	    const int mma_softmax_row =
	        qk_mma_thread_idx / kSoftmaxThreadsPerRow;
	    const int mma_softmax_row_lane =
	        qk_mma_thread_idx & (kSoftmaxThreadsPerRow - 1);
    float mma_running_m = -INFINITY;
    float mma_running_l = 0.0f;

    auto mma_stage_probability_row = [&](auto& p_sA, auto& p_sSFA,
                                         auto& logits_stage,
                                         int tile, bool final_tile) {
      if (!mma_softmax_row_owner) {
        return;
      }
	      constexpr unsigned kSoftmaxGroupMask = 0xffffffffu;
	      constexpr int kColsPerSoftmaxThread =
	          kCutlassTileN / kSoftmaxThreadsPerRow;
      const int col_begin = mma_softmax_row_lane * kColsPerSoftmaxThread;
      float tile_m_local = -INFINITY;
#pragma unroll
      for (int col_offset = 0; col_offset < kColsPerSoftmaxThread;
           ++col_offset) {
        const int col = col_begin + col_offset;
        const float logit = __bfloat162float(
            logits_stage(logits_smem_index(mma_softmax_row, col)));
        tile_m_local = fmaxf(tile_m_local, logit);
	      }
	      float tile_m = tile_m_local;
	      tile_m = fmaxf(tile_m,
	                     __shfl_xor_sync(kSoftmaxGroupMask, tile_m, 1));
	      tile_m = fmaxf(tile_m,
	                     __shfl_xor_sync(kSoftmaxGroupMask, tile_m, 2));
      const bool tile_has_values = tile_m != -INFINITY;
      const float next_m = tile_has_values ? fmaxf(mma_running_m, tile_m)
                                           : mma_running_m;
      const float old_scale =
          mma_running_l == 0.0f ? 0.0f : __expf(mma_running_m - next_m);
      const float tile_scale = tile_has_values ? __expf(tile_m - next_m) : 0.0f;
      const float safe_tile_m = tile_has_values ? tile_m : 0.0f;
      float tile_l_scaled_local = 0.0f;
#pragma unroll
      for (int scale_group = col_begin / 16;
           scale_group < (col_begin + kColsPerSoftmaxThread) / 16;
           ++scale_group) {
        const int local_col = scale_group * 16;
        float vec_max = 0.0f;
        float p_vals[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const float logit = __bfloat162float(
              logits_stage(logits_smem_index(mma_softmax_row,
                                             local_col + i)));
          const float p_scaled = __expf(logit - safe_tile_m) * tile_scale;
          tile_l_scaled_local += p_scaled;
          vec_max = fmaxf(vec_max, p_scaled);
          p_vals[i] = p_scaled;
        }
        const float scale_value =
            fmaxf(kProbGlobalScale * vec_max / 6.0f, 1.0e-8f);
        const uint8_t scale_byte = fp32_to_e4m3_byte(scale_value);
        p_sSFA(mma_softmax_row, local_col, cute::Int<0>{}) =
            make_ue4m3_raw(scale_byte);
        const float output_scale = kProbGlobalScale / scale_value;
        uint32_t packed_lo = 0;
        uint32_t packed_hi = 0;
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
          const float p0 = p_vals[2 * pair];
          const float p1 = p_vals[2 * pair + 1];
          const uint8_t packed_pair = fp32_pair_to_e2m1_byte(
              p0 * output_scale, p1 * output_scale);
          if (pair < 4) {
            packed_lo |= static_cast<uint32_t>(packed_pair) << (8 * pair);
          } else {
            packed_hi |= static_cast<uint32_t>(packed_pair) << (8 * (pair - 4));
          }
        }
        auto first_ref = p_sA(mma_softmax_row, local_col, cute::Int<0>{});
        auto second_ref = p_sA(mma_softmax_row, local_col + 2, cute::Int<0>{});
        auto third_ref = p_sA(mma_softmax_row, local_col + 4, cute::Int<0>{});
        auto fourth_ref = p_sA(mma_softmax_row, local_col + 6, cute::Int<0>{});
        auto fifth_ref = p_sA(mma_softmax_row, local_col + 8, cute::Int<0>{});
        auto sixth_ref = p_sA(mma_softmax_row, local_col + 10, cute::Int<0>{});
        auto seventh_ref = p_sA(mma_softmax_row, local_col + 12, cute::Int<0>{});
        auto eighth_ref = p_sA(mma_softmax_row, local_col + 14, cute::Int<0>{});
        uint8_t* dst0 = cute::recast_ptr<uint8_t>(&first_ref);
        uint8_t* dst1 = cute::recast_ptr<uint8_t>(&second_ref);
        uint8_t* dst2 = cute::recast_ptr<uint8_t>(&third_ref);
        uint8_t* dst3 = cute::recast_ptr<uint8_t>(&fourth_ref);
        uint8_t* dst4 = cute::recast_ptr<uint8_t>(&fifth_ref);
        uint8_t* dst5 = cute::recast_ptr<uint8_t>(&sixth_ref);
        uint8_t* dst6 = cute::recast_ptr<uint8_t>(&seventh_ref);
        uint8_t* dst7 = cute::recast_ptr<uint8_t>(&eighth_ref);
        const bool contiguous =
            dst1 == dst0 + 1 && dst2 == dst0 + 2 && dst3 == dst0 + 3 &&
            dst4 == dst0 + 4 && dst5 == dst0 + 5 && dst6 == dst0 + 6 &&
            dst7 == dst0 + 7;
        // P staging stores two 32-bit words and requires 8 contiguous bytes.
        if (contiguous && ((reinterpret_cast<uintptr_t>(dst0) & 0x3u) == 0)) {
          *reinterpret_cast<uint32_t*>(dst0) = packed_lo;
          *reinterpret_cast<uint32_t*>(dst0 + 4) = packed_hi;
        } else {
          SM120_NVFP4_DEBUG_TRAP();
        }
	      }
	      float tile_l_scaled = tile_l_scaled_local;
	      tile_l_scaled +=
	          __shfl_xor_sync(kSoftmaxGroupMask, tile_l_scaled, 1);
	      tile_l_scaled +=
	          __shfl_xor_sync(kSoftmaxGroupMask, tile_l_scaled, 2);
      mma_running_l = mma_running_l * old_scale + tile_l_scaled;
      mma_running_m = next_m;
      if (mma_softmax_row_lane == 0) {
        storage.old_scale_stage[tile & 1][mma_softmax_row] = old_scale;
        if (final_tile) {
          storage.global_m[mma_softmax_row] = mma_running_m;
          storage.global_l[mma_softmax_row] = mma_running_l;
        }
      }
    };

    auto mma_stage_softmax_tile = [&](int tile, bool final_tile) {
      cutlass::arch::NamedBarrier::sync(
          CutlassCollectiveMainloop::ThreadCount,
          cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
	      mma_stage_probability_row(p_sA0, p_sSFA0_m, logits0, tile,
	                                final_tile);
      cutlass::arch::NamedBarrier::sync(
          CutlassCollectiveMainloop::ThreadCount,
          cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
    };

    auto release_k_chunk = [&]() {
      k_pipeline.consumer_release(k_pipe_release);
      ++k_pipe_release;
    };

    auto release_consumed_k_chunks = [&]() {
      release_k_chunk();
      if (qk_head_chunks > 1) {
        release_k_chunk();
      }
    };

    auto acquire_output_stage = [&]() {
      pipeline_corr_epi.producer_acquire(pipeline_corr_epi_producer_state);
    };

    auto commit_output_stage = [&]() {
      cutlass::arch::NamedBarrier::sync(
          CutlassCollectiveMainloopK128Stage2::ThreadCount,
          cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
      cutlass::arch::fence_view_async_shared();
      if (qk_mma_thread_idx == 0) {
        pipeline_corr_epi.producer_commit(pipeline_corr_epi_producer_state);
      }
      ++pipeline_corr_epi_producer_state;
    };

    auto run_qk_tile = [&](int tile) {
      auto qk_accum = cute::partition_fragment_C(
          qk_tiled_mma, cute::take<0, 2>(CutlassThreadBlockShape{}));
      cute::clear(qk_accum);
      qk_consume_k_stage(q_frag0, q_scale_frag0, qk_accum);
      if (qk_head_chunks > 1) {
        qk_consume_k_stage(q_frag1, q_scale_frag1, qk_accum);
      }

      auto cC = cute::make_identity_tensor(
          cute::take<0, 2>(CutlassThreadBlockShape{}));
      auto tCcC = qk_thread_mma.partition_C(cC);
      logits0_covered.fill_and_sync(
          kSm120Nvfp4FmhaMmaSoftmaxThreadCount,
          cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier,
          qk_mma_thread_idx);
      for (int i = 0; i < cute::size(qk_accum); ++i) {
        auto coord = tCcC(i);
        const int row = int(cute::get<0>(coord));
        const int col = int(cute::get<1>(coord));
        if (row < kCutlassTileM && col < kCutlassTileN) {
          const float logit =
              transform_score(qk_accum(i) * qk_scale, row, col, tile);
          logits0(logits_smem_index(row, col)) = __float2bfloat16(logit);
        }
      }
    };

    auto run_pv_tile = [&](int tile, bool final_tile, auto& pv_accum) {
      if (final_tile) {
        acquire_output_stage();
      }
      pv_consume_v_stage();
      for (int i = 0; i < cute::size(pv_accum); ++i) {
        if (tile == 0) {
          pv_accum(i) = 0.0f;
          continue;
        }
        auto coord = pv_tCcC(i);
        const int row = int(cute::get<0>(coord));
        if (row < kCutlassTileM) {
          pv_accum(i) *= storage.old_scale_stage[tile & 1][row];
        }
      }
	      pv_gemm_p_stage(pv_accum, p_sA0, p_sSFA0_m);
      pv_release_v_stage();
      if (final_tile) {
        const float pv_base_scale = pv_alpha / kProbGlobalScale;
        epilogue_o_covered.fill_and_sync(
            CutlassCollectiveMainloopK128Stage2::ThreadCount,
            cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier,
            pv_mma_thread_idx);
        sm120_stage_o_fragment_to_epilogue_smem(
            pv_accum, pv_tCcC, smem_epilogue_o, storage.global_l,
            pv_base_scale, split_m == nullptr);
        commit_output_stage();
      }
    };

    auto rescale_pv_accum = [&](int tile, auto& pv_accum) {
      for (int i = 0; i < cute::size(pv_accum); ++i) {
        if (tile == 0) {
          pv_accum(i) = 0.0f;
          continue;
        }
        auto coord = pv_tCcC(i);
        const int row = int(cute::get<0>(coord));
        if (row < kCutlassTileM) {
          pv_accum(i) *= storage.old_scale_stage[tile & 1][row];
        }
      }
    };

    for (int tile = 0; tile < effective_num_kv_tiles; ++tile) {
      run_qk_tile(tile);
      if (tile > 0) {
        run_pv_tile(tile - 1, false, pv_accum0);
      }
      mma_stage_softmax_tile(tile, tile == effective_num_kv_tiles - 1);
      release_consumed_k_chunks();
    }
    run_pv_tile(effective_num_kv_tiles - 1, true, pv_accum0);
  } else if (is_epilogue) {
    consume_and_store_output_span(output_thread_idx);
  }
#else
  if (threadIdx.x == 0) {
    out_group[0] = __float2bfloat16(-1.0f);
  }
#endif
}

template <int kOutputGroupSpan, bool kUsePagedKv, bool kCausal,
          bool kUseSlidingWindow, bool kUseLogitsSoftCap,
          bool kPvLayoutV = true>
cudaError_t sm120_nvfp4_qkv_online_register_q_splitkv_full_grid_raw(
    uint8_t* q_packed,
    uint8_t* q_scales,
    uint8_t* k_packed,
    uint8_t* k_scales,
    uint8_t* v_pv_packed,
    uint8_t* v_pv_scales,
    __nv_bfloat16* partial,
    float* split_m,
    float* split_l,
    __nv_bfloat16* out,
    uint8_t* workspace,
    size_t workspace_bytes,
    float qk_alpha,
    float pv_alpha,
    int split_kv_tiles,
    int q_len,
    int group_size,
    int kv_len_tokens,
    bool causal,
    int sliding_window,
    float logits_soft_cap,
    int q_rows,
    int head_dim,
    int kv_len,
    cudaStream_t stream,
    Sm120Nvfp4PagedKvLoadParams paged_kv_params = {},
    const int32_t* qo_indptr = nullptr,
    const int32_t* kv_lens = nullptr,
    int batch_size = 1,
    int q_tiles_per_sequence = 0,
    int num_kv_heads = 1,
    bool all_kv_heads = false,
    bool skip_internal_combine = false) {
  static_assert(kOutputGroupSpan == 1,
                "D128 split-KV launcher supports span 1 only");
  const int total_kv_tiles = kv_len / kCutlassTileN;
  const int num_splits =
      (total_kv_tiles + split_kv_tiles - 1) / split_kv_tiles;

  float alpha = 1.0f;
  auto qk_args = prepare_sm120_nvfp4_gemm_args<CutlassGemmKernel>(
      nullptr, q_packed, k_packed, q_scales, k_scales, &alpha, q_rows,
      kv_len, head_dim, 1);
  const size_t qk_workspace_size =
      CutlassGemmKernel::get_workspace_size(qk_args);
  constexpr size_t kWorkspaceAlignment = 256;
  auto align_workspace = [](size_t bytes) {
    return (bytes + kWorkspaceAlignment - 1) & ~(kWorkspaceAlignment - 1);
  };
  const size_t qk_workspace_alloc = align_workspace(qk_workspace_size);
  char* workspace_base = reinterpret_cast<char*>(workspace);
  char* qk_workspace = workspace_base;

  auto pv_args = prepare_sm120_nvfp4_gemm_args<
      CutlassGemmKernelK128Stage2>(
      nullptr, q_packed, v_pv_packed, q_scales, v_pv_scales, &alpha,
      kCutlassTileM, head_dim, kv_len, 1);
  const size_t pv_workspace_size =
      CutlassGemmKernelK128Stage2::get_workspace_size(pv_args);
  char* pv_workspace = workspace_base + qk_workspace_alloc;
  const size_t base_required_workspace_bytes =
      qk_workspace_alloc + pv_workspace_size;
  const size_t linear_scale_cache_offset =
      align_workspace(base_required_workspace_bytes);
  const size_t linear_scale_cache_bytes =
      (kUsePagedKv && !kPvLayoutV)
          ? sm120_nvfp4_linear_v_scale_cache_bytes(batch_size, num_kv_heads,
                                                   head_dim, kv_len)
          : 0;
  const size_t linear_data_cache_offset =
      align_workspace(linear_scale_cache_offset + linear_scale_cache_bytes);
  const size_t linear_data_cache_bytes =
      (kUsePagedKv && !kPvLayoutV)
          ? sm120_nvfp4_linear_v_data_cache_bytes(batch_size, num_kv_heads,
                                                  head_dim, kv_len)
          : 0;
  const size_t required_workspace_bytes =
      linear_data_cache_offset + linear_data_cache_bytes;
  if (workspace_bytes < required_workspace_bytes) {
    return cudaErrorInvalidValue;
  }
  constexpr size_t kWorkspaceClearBytes = 32 * 1024 * 1024;
  const size_t workspace_clear_bytes =
      workspace_bytes < kWorkspaceClearBytes ? workspace_bytes
                                             : kWorkspaceClearBytes;
  const size_t workspace_zero_bytes =
      workspace_clear_bytes > base_required_workspace_bytes
          ? workspace_clear_bytes
          : base_required_workspace_bytes;
  // CUTLASS persistent scheduler state is sensitive to allocator residue.
  // Clear a bounded prefix before building the QK/PV argument objects.
  auto workspace_zero_status =
      cudaMemsetAsync(workspace_base, 0, workspace_zero_bytes, stream);
  if (workspace_zero_status != cudaSuccess) {
    return workspace_zero_status;
  }
  auto qk_status = CutlassGemmKernel::initialize_workspace(
      qk_args, qk_workspace, stream);
  if (qk_status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  auto qk_params = CutlassGemmKernel::to_underlying_arguments(
      qk_args, qk_workspace);
  auto pv_status = CutlassGemmKernelK128Stage2::initialize_workspace(
      pv_args, pv_workspace, stream);
  if (pv_status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  auto pv_params = CutlassGemmKernelK128Stage2::to_underlying_arguments(
      pv_args, pv_workspace);

  if constexpr (kUsePagedKv && !kPvLayoutV) {
    uint8_t* linear_scale_cache = reinterpret_cast<uint8_t*>(
        workspace_base + linear_scale_cache_offset);
    auto scale_cache_status = sm120_nvfp4_prepare_linear_v_scale_cache(
        paged_kv_params, linear_scale_cache, kv_lens, batch_size,
        num_kv_heads, head_dim, kv_len, stream);
    if (scale_cache_status != cudaSuccess) {
      return scale_cache_status;
    }
    uint8_t* linear_data_cache = reinterpret_cast<uint8_t*>(
        workspace_base + linear_data_cache_offset);
    auto data_cache_status = sm120_nvfp4_prepare_linear_v_data_cache(
        paged_kv_params, linear_data_cache, kv_lens, batch_size,
        num_kv_heads, head_dim, kv_len, stream);
    if (data_cache_status != cudaSuccess) {
      return data_cache_status;
    }
  }

  constexpr int kSmemBytes =
      static_cast<int>(sizeof(Sm120Nvfp4QkvLoadCollectiveStorage));
  auto stage_kernel =
      sm120_nvfp4_qkv_online_register_q_stage_kernel<
          kOutputGroupSpan, kUsePagedKv, kCausal, kUseSlidingWindow,
          kUseLogitsSoftCap, kPvLayoutV>;
  cudaError_t status = cudaFuncSetAttribute(
      stage_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  if (status != cudaSuccess) {
    return status;
  }

  const bool direct_single_split = num_splits == 1;
  __nv_bfloat16* stage_out = direct_single_split ? out : partial;
  float* stage_split_m = direct_single_split ? nullptr : split_m;
  float* stage_split_l = direct_single_split ? nullptr : split_l;
  const int stage_output_stride = q_rows * head_dim;
  stage_kernel<<<dim3(q_rows / kCutlassTileM,
                      head_dim / (kOutputGroupSpan * kOutputTileN),
                      num_splits),
                 kSm120Nvfp4FmhaThreadCount, kSmemBytes, stream>>>(
      qk_params, pv_params, stage_out, qk_alpha, pv_alpha, 0, 0,
      split_kv_tiles, total_kv_tiles, q_len, group_size, kv_len_tokens,
      causal ? 1 : 0, sliding_window, logits_soft_cap, 0, head_dim,
      stage_split_m, stage_split_l, q_rows, stage_output_stride,
      paged_kv_params, qo_indptr, kv_lens, batch_size, q_tiles_per_sequence,
      num_kv_heads, all_kv_heads);
  status = cudaGetLastError();
  return status;
}

}  // namespace flashinfer::attention::blackwell::sm120_nvfp4::d128
