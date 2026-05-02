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

#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_paged_kv.cuh>
#include <flashinfer/cp_async.cuh>
#include <flashinfer/mma.cuh>

namespace flashinfer::attention::blackwell::sm120_nvfp4::d128 {

// D128 specialization seed. This starts as the D512 scaffold, but lives in a
// separate translation unit so we can shrink storage/pipelines without risking
// the established D512 path.
constexpr int kQLen = 512;
constexpr int kGroup = 2;
constexpr int kKvLen = 32768;
constexpr int kHeadDim = 128;
constexpr int kPackedHeadDim = kHeadDim / 2;
constexpr int kScaleCols = kHeadDim / 16;
constexpr int kQRows = kQLen * kGroup;
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
constexpr int kDebugHead = 0;
constexpr int kProbPackedCols = kKvLen / 2;
constexpr int kProbScaleCols = kKvLen / 16;
constexpr int kShapeBMaxKvLen = 262144;
constexpr int kShapeBMaxKvTiles = kShapeBMaxKvLen / kCutlassTileN;
constexpr int kFusedWarpsPerCta = 8;
constexpr int kSplitKvLen = 1024;
constexpr int kNumKvSplits = kKvLen / kSplitKvLen;
constexpr int kBenchRows = 128;
constexpr int kBenchQTiles = kBenchRows / kTileM;
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
constexpr int kSm120Nvfp4FmhaNumWarpsLoad = 1;
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
constexpr int kSm120Nvfp4FmhaMmaSoftmaxThreadCount =
    kSm120Nvfp4FmhaNumWarpsMma * cutlass::NumThreadsPerWarp;
constexpr int kSm120Nvfp4FmhaMmaSoftmaxLoadThreadCount =
    kSm120Nvfp4FmhaMmaSoftmaxThreadCount +
    kSm120Nvfp4FmhaNumWarpsLoad * cutlass::NumThreadsPerWarp;
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
  if (warp_idx == kSm120Nvfp4FmhaWarpLoad) {
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

__device__ __forceinline__ uint8_t smem_fp4_debug_code(uint8_t code,
                                                       int mode) {
  code &= 0x0f;
  if (mode == 1) {
    return static_cast<uint8_t>(code << 2);
  }
  if (mode == 2) {
    return static_cast<uint8_t>(code << 4);
  }
  if (mode == 3) {
    return static_cast<uint8_t>(code | (code << 4));
  }
  return code;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
template <typename Mainloop, typename ThreadBlockShape, int TileM, int TileN,
          int TileK>
__device__ __forceinline__ void cutlass_smem_atom_gemm_tile_body_impl(
    typename Mainloop::TensorStorage& storage,
    const uint8_t* q_packed,
    const uint8_t* q_scales,
    const uint8_t* k_packed,
    const uint8_t* k_scales,
    float* out_tile,
    int q_row_base,
    int q_col_base,
    int q_packed_cols,
    int q_scale_cols,
    int kv_row_base,
    int kv_col_base,
    int kv_packed_cols,
    int kv_scale_cols,
    int k_tile_limit,
    int out_stride,
    int out_row_base,
    int out_col_base,
    int data_debug_mode,
    int scale_debug_mode) {
  using cute::_;

  const int block_thread_idx = int(threadIdx.x);
  const bool mma_thread_active = block_thread_idx < Mainloop::ThreadCount;
  const int thread_idx = mma_thread_active ? block_thread_idx : 0;
  auto tiled_mma = typename Mainloop::TiledMma{};
  auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
  Mainloop collective;

  auto accum = cute::partition_fragment_C(
      tiled_mma, cute::take<0, 2>(ThreadBlockShape{}));
  cute::clear(accum);

  auto sA = cute::make_tensor(cute::make_smem_ptr(storage.smem_A.begin()),
                              typename Mainloop::SmemLayoutA{});
  auto sB = cute::make_tensor(cute::make_smem_ptr(storage.smem_B.begin()),
                              typename Mainloop::SmemLayoutB{});
  auto sSFA = cute::make_tensor(cute::make_smem_ptr(storage.smem_SFA.begin()),
                                typename Mainloop::SmemLayoutSFA{});
  auto sSFB = cute::make_tensor(cute::make_smem_ptr(storage.smem_SFB.begin()),
                                typename Mainloop::SmemLayoutSFB{});

  auto tCrA = thread_mma.partition_fragment_A(sA(_, _, cute::Int<0>{}));
  auto tCrB = thread_mma.partition_fragment_B(sB(_, _, cute::Int<0>{}));
  auto tCrSFA = collective.partition_fragment_SFA(sSFA(_, _, cute::Int<0>{}),
                                                  thread_mma);
  auto tCrSFB = collective.partition_fragment_SFB(sSFB(_, _, cute::Int<0>{}),
                                                  thread_mma);

  auto smem_tiled_copy_A = cute::make_tiled_copy_A(
      typename Mainloop::SmemCopyAtomA{}, tiled_mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
  auto tCsA = smem_thr_copy_A.partition_S(
      cute::as_position_independent_swizzle_tensor(sA));
  auto tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  auto cA = cute::make_identity_tensor(
      cute::make_shape(cute::Int<TileM>{}, cute::Int<TileK>{},
                       cute::Int<1>{}));
  auto tAsA_prod = smem_thr_copy_A.partition_D(sA);
  auto tAcA_prod = smem_thr_copy_A.partition_D(cA);

  auto smem_tiled_copy_B = cute::make_tiled_copy_B(
      typename Mainloop::SmemCopyAtomB{}, tiled_mma);
  auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
  auto tCsB = smem_thr_copy_B.partition_S(
      cute::as_position_independent_swizzle_tensor(sB));
  auto tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  auto cB = cute::make_identity_tensor(
      cute::make_shape(cute::Int<TileN>{}, cute::Int<TileK>{},
                       cute::Int<1>{}));
  auto tBsB_prod = smem_thr_copy_B.partition_D(sB);
  auto tBcB_prod = smem_thr_copy_B.partition_D(cB);

  auto tile_shape_mnk = cute::tile_shape(tiled_mma);
  auto smem_tiled_copy_SFA = cute::make_tiled_copy_impl(
      typename Mainloop::SmemCopyAtomSFA{},
      collective.get_layoutSFA_TV(tiled_mma),
      cute::make_shape(cute::size<0>(tile_shape_mnk),
                       cute::size<2>(tile_shape_mnk)));
  auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
  auto tCsSFA = smem_thr_copy_SFA.partition_S(
      cute::as_position_independent_swizzle_tensor(sSFA));
  auto tCrSFA_copy_view = smem_thr_copy_SFA.retile_D(tCrSFA);

  auto smem_tiled_copy_SFB = cute::make_tiled_copy_impl(
      typename Mainloop::SmemCopyAtomSFB{},
      collective.get_layoutSFB_TV(tiled_mma),
      cute::make_shape(cute::size<1>(tile_shape_mnk),
                       cute::size<2>(tile_shape_mnk)));
  auto smem_thr_copy_SFB = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
  auto tCsSFB = smem_thr_copy_SFB.partition_S(
      cute::as_position_independent_swizzle_tensor(sSFB));
  auto tCrSFB_copy_view = smem_thr_copy_SFB.retile_D(tCrSFB);

  auto copy_kblock = [&](auto k_block) {
    cute::copy(smem_tiled_copy_A, tCsA(_, _, k_block, cute::Int<0>{}),
               tCrA_copy_view(_, _, k_block));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, k_block, cute::Int<0>{}),
               tCrB_copy_view(_, _, k_block));

    using MMAOp = typename Mainloop::TiledMma::MMA_Op;
    fp4_shift_A(MMAOp{}, tCrA_copy_view(_, _, k_block));
    fp4_shift_B(MMAOp{}, tCrB_copy_view(_, _, k_block));

    cute::copy(tCsSFA(_, _, k_block, cute::Int<0>{}),
               tCrSFA_copy_view(_, _, k_block));
    cute::copy(tCsSFB(_, _, k_block, cute::Int<0>{}),
               tCrSFB_copy_view(_, _, k_block));
  };

  auto gemm_kblock = [&](auto k_block) {
    cute::gemm(tiled_mma,
               cute::make_zip_tensor(tCrA(_, _, k_block),
                                     tCrSFA(_, _, k_block)),
               cute::make_zip_tensor(tCrB(_, _, k_block),
                                     tCrSFB(_, _, k_block)),
               accum);
  };

  constexpr int kTileElements = TileM * TileK;
  const int active_k_tile_limit = (data_debug_mode == 5) ? 1 : k_tile_limit;
  uint8_t* smem_a_bytes = cute::recast_ptr<uint8_t>(storage.smem_A.begin());
  uint8_t* smem_b_bytes = cute::recast_ptr<uint8_t>(storage.smem_B.begin());
  constexpr int kSmemABytes =
      (cute::cosize_v<typename Mainloop::SmemLayoutA> + 1) / 2;
  constexpr int kSmemBBytes =
      (cute::cosize_v<typename Mainloop::SmemLayoutB> + 1) / 2;
  auto write_partitioned_fp4 = [&](auto tDst,
                                   auto tCoord,
                                   const uint8_t* packed,
                                   int source_row_base,
                                   int source_col_base,
                                   int packed_cols,
                                   int k_base,
                                   int mode) {
    for (int i = 0; i < int(cute::size(tDst)); ++i) {
      auto coord = tCoord(i);
      const int row = int(cute::get<0>(coord));
      const int k = int(cute::get<1>(coord));
      const int source_col = source_col_base + k_base + k;
      const uint8_t byte =
          packed[(source_row_base + row) * packed_cols + (source_col >> 1)];
      const uint8_t code = smem_fp4_debug_code(
          static_cast<uint8_t>((source_col & 1) ? ((byte >> 4) & 0x0f)
                                                : (byte & 0x0f)),
          mode);
      tDst(i) = cute::uint4_t(code);
    }
  };
  auto K_BLOCK_MAX_PROD = cute::size<2>(tAsA_prod);
  for (int k_tile = 0; k_tile < active_k_tile_limit; ++k_tile) {
    const int k_base = k_tile * TileK;
    for (int idx = block_thread_idx; idx < kSmemABytes; idx += blockDim.x) {
      smem_a_bytes[idx] = 0;
    }
    for (int idx = block_thread_idx; idx < kSmemBBytes; idx += blockDim.x) {
      smem_b_bytes[idx] = 0;
    }
    __syncthreads();
    if (mma_thread_active) {
      cute::for_each(cute::make_int_sequence<K_BLOCK_MAX_PROD>{}, [&](auto k_block) {
        write_partitioned_fp4(tAsA_prod(_, _, k_block, cute::Int<0>{}),
                              tAcA_prod(_, _, k_block, cute::Int<0>{}),
                              q_packed, q_row_base, q_col_base, q_packed_cols,
                              k_base, data_debug_mode);
        write_partitioned_fp4(tBsB_prod(_, _, k_block, cute::Int<0>{}),
                              tBcB_prod(_, _, k_block, cute::Int<0>{}),
                              k_packed, kv_row_base, kv_col_base,
                              kv_packed_cols, k_base, data_debug_mode);
      });
    }
    for (int idx = block_thread_idx; idx < kTileElements / 2; idx += blockDim.x) {
      const int row = idx / (TileK / 2);
      const int packed_k = idx - row * (TileK / 2);
      const int k0 = 2 * packed_k;
      const int q_scale_col = (q_col_base + k_base + k0) >> 4;
      const int kv_scale_col = (kv_col_base + k_base + k0) >> 4;
      sSFA(row, k0, cute::Int<0>{}) =
          make_ue4m3_raw(q_scales[(q_row_base + row) * q_scale_cols +
                                   q_scale_col]);
      sSFB(row, k0, cute::Int<0>{}) =
          make_ue4m3_raw(k_scales[(kv_row_base + row) * kv_scale_cols +
                                   kv_scale_col]);
    }
    if (data_debug_mode == 4) {
      for (int idx = block_thread_idx;
           idx < cute::cosize_v<typename Mainloop::SmemLayoutA>;
           idx += blockDim.x) {
        storage.smem_A.begin()[idx] = 2;
      }
      for (int idx = block_thread_idx;
           idx < cute::cosize_v<typename Mainloop::SmemLayoutB>;
           idx += blockDim.x) {
        storage.smem_B.begin()[idx] = 2;
      }
    }
    if (data_debug_mode == 6) {
      uint8_t* smem_a_bytes =
          cute::recast_ptr<uint8_t>(storage.smem_A.begin());
      uint8_t* smem_b_bytes =
          cute::recast_ptr<uint8_t>(storage.smem_B.begin());
      constexpr int kSmemABytes =
          (cute::cosize_v<typename Mainloop::SmemLayoutA> + 1) / 2;
      constexpr int kSmemBBytes =
          (cute::cosize_v<typename Mainloop::SmemLayoutB> + 1) / 2;
      for (int idx = block_thread_idx;
           idx < kSmemABytes;
           idx += blockDim.x) {
        smem_a_bytes[idx] = 0x22;
      }
      for (int idx = block_thread_idx;
           idx < kSmemBBytes;
           idx += blockDim.x) {
        smem_b_bytes[idx] = 0x22;
      }
    }
    if (scale_debug_mode == 1) {
      for (int idx = block_thread_idx;
           idx < cute::cosize_v<typename Mainloop::SmemLayoutSFA>;
           idx += blockDim.x) {
        storage.smem_SFA.begin()[idx] = make_ue4m3_raw(0x38);
      }
      for (int idx = block_thread_idx;
           idx < cute::cosize_v<typename Mainloop::SmemLayoutSFB>;
           idx += blockDim.x) {
        storage.smem_SFB.begin()[idx] = make_ue4m3_raw(0x38);
      }
    }
    __syncthreads();

    if (mma_thread_active) {
      auto K_BLOCK_MAX = cute::size<2>(tCrA);
      copy_kblock(cute::Int<0>{});
      cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
        if (k_block_next > 0) {
          copy_kblock(k_block_next);
        }
        gemm_kblock(k_block);
      });
    }
    __syncthreads();
  }

  if (mma_thread_active) {
    auto cC = cute::make_identity_tensor(
        cute::take<0, 2>(ThreadBlockShape{}));
    auto tCcC = thread_mma.partition_C(cC);
    for (int i = 0; i < cute::size(accum); ++i) {
      auto coord = tCcC(i);
      const int row = int(cute::get<0>(coord));
      const int col = int(cute::get<1>(coord));
      if (row < TileM && col < TileN) {
        out_tile[(out_row_base + row) * out_stride + out_col_base + col] =
            accum(i);
      }
    }
  }
}

__device__ __forceinline__ void cutlass_smem_atom_gemm_tile_body(
    typename CutlassCollectiveMainloop::TensorStorage& storage,
    const uint8_t* q_packed,
    const uint8_t* q_scales,
    const uint8_t* k_packed,
    const uint8_t* k_scales,
    float* out_tile,
    int q_row_base,
    int q_col_base,
    int q_packed_cols,
    int q_scale_cols,
    int kv_row_base,
    int kv_col_base,
    int kv_packed_cols,
    int kv_scale_cols,
    int k_tile_limit,
    int out_stride,
    int out_row_base,
    int out_col_base,
    int data_debug_mode,
    int scale_debug_mode) {
  cutlass_smem_atom_gemm_tile_body_impl<CutlassCollectiveMainloop,
                                        CutlassThreadBlockShape,
                                        kCutlassTileM,
                                        kCutlassTileN,
                                        kCutlassTileK>(
      storage, q_packed, q_scales, k_packed, k_scales, out_tile, q_row_base,
      q_col_base, q_packed_cols, q_scale_cols, kv_row_base, kv_col_base,
      kv_packed_cols, kv_scale_cols, k_tile_limit, out_stride, out_row_base,
      out_col_base, data_debug_mode, scale_debug_mode);
}

template <class FrgTensorA, class FrgTensorSFA>
__device__ __forceinline__ void cutlass_qk_tma_q_register_stage(
    typename CutlassCollectiveMainloop::MainloopPipeline pipeline,
    typename CutlassCollectiveMainloop::PipelineState& smem_pipe_read,
    FrgTensorA& q_frag,
    FrgTensorSFA& q_scale_frag,
    int thread_idx,
    int m_coord,
    typename CutlassCollectiveMainloop::TensorStorage& shared_tensors) {
  using namespace cute;

  Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()),
                          typename CutlassCollectiveMainloop::SmemLayoutA{});
  Tensor sSFA_full = make_tensor(
      make_smem_ptr(shared_tensors.smem_SFA.begin()),
      typename CutlassCollectiveMainloop::SmemLayoutSFA{});
  Tensor sSFA = [&]() {
    if constexpr (!CutlassCollectiveMainloop::PadSFA_M) {
      return sSFA_full;
    } else {
      return sSFA_full(
          make_coord(_, m_coord % CutlassCollectiveMainloop::SFA_M_Ratio),
          _, _);
    }
  }();

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
          bool kUseSlidingWindow, bool kUseLogitsSoftCap>
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
    int q_tiles_per_sequence) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  using cute::_;
  static_assert(kOutputGroupSpan == 1,
                "D128 specialization has exactly one 128-wide output group");

  extern __shared__ __align__(128) char smem[];
  auto& storage = *reinterpret_cast<Sm120Nvfp4QkvLoadCollectiveStorage*>(smem);
  int batch_idx = 0;
  int local_q_tile = q_tile + int(blockIdx.x);
  int effective_q_tile = q_tile + int(blockIdx.x);
  if constexpr (kUsePagedKv) {
    const bool varlen_batch = qo_indptr != nullptr && kv_lens != nullptr;
    if (varlen_batch) {
      if (q_tiles_per_sequence <= 0) {
        return;
      }
      batch_idx = int(blockIdx.x) / q_tiles_per_sequence;
      if (batch_idx >= batch_size) {
        return;
      }
      local_q_tile = int(blockIdx.x) - batch_idx * q_tiles_per_sequence;
      effective_q_tile = int(blockIdx.x);
      const int q_begin = qo_indptr[batch_idx];
      const int q_end = qo_indptr[batch_idx + 1];
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
  const bool lane_predicate = lane_idx == 0;
  const Sm120Nvfp4FmhaRole role = sm120_nvfp4_fmha_role_for_warp(warp_idx);
  const bool is_load = role == Sm120Nvfp4FmhaRole::Load;
  const bool is_mma = role == Sm120Nvfp4FmhaRole::Mma;
  const bool is_epilogue = role == Sm120Nvfp4FmhaRole::Epilogue;
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

  if (is_load && lane_predicate) {
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
  q_pipeline_params.is_leader =
      (is_load && lane_predicate) || mma_warpgroup_leader;
  k_pipeline_params.is_leader =
      (is_load && lane_predicate) || mma_warpgroup_leader;
  v_pipeline_params.is_leader =
      (is_load && lane_predicate) || mma_warpgroup_leader;
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
  auto qk_sA = cute::make_tensor(
      cute::make_smem_ptr(storage.qk_tensors.smem_A.begin()),
      typename CutlassCollectiveMainloop::SmemLayoutA{});
  auto qk_sB = cute::make_tensor(
      cute::make_smem_ptr(storage.qk_tensors.smem_B.begin()),
      typename CutlassCollectiveMainloop::SmemLayoutB{});
  // SM120 has no TMEM, so the 77 S/P/O lifetime is represented with compact
  // shared-memory aliasing. B is the transient score/O region while K reuse is
  // blocked; A/SFA are free after Q is resident and hold two compact P stages.
  __nv_bfloat16* smem_logits0 =
      reinterpret_cast<__nv_bfloat16*>(storage.logits_smem.data);
  __nv_bfloat16* smem_epilogue_o = smem_logits0;
  auto qk_sSFA = cute::make_tensor(
      cute::make_smem_ptr(storage.qk_tensors.smem_SFA.begin()),
      typename CutlassCollectiveMainloop::SmemLayoutSFA{});
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
  auto qk_sSFB = cute::make_tensor(
      cute::make_smem_ptr(storage.qk_tensors.smem_SFB.begin()),
      typename CutlassCollectiveMainloop::SmemLayoutSFB{});

  CutlassCollectiveMainloopK128Stage2 pv_collective;
  auto pv_tiled_mma = typename CutlassCollectiveMainloopK128Stage2::TiledMma{};
  auto pv_thread_mma = pv_tiled_mma.get_thread_slice(pv_mma_thread_idx);
  auto pv_sB = cute::make_tensor(
      cute::make_smem_ptr(storage.v_smem_B.begin()),
      typename CutlassCollectiveMainloopK128Stage2::SmemLayoutB{});
  auto pv_sSFB = cute::make_tensor(
      cute::make_smem_ptr(storage.v_smem_SFB.begin()),
      typename CutlassCollectiveMainloopK128Stage2::SmemLayoutSFB{});

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

      for (int copy_thread = lane_idx;
           copy_thread < CutlassCollectiveMainloop::ThreadCount;
           copy_thread += cutlass::NumThreadsPerWarp) {
        auto smem_thr_copy_B =
            smem_tiled_copy_B.get_thread_slice(copy_thread);
        auto tBsB_prod = smem_thr_copy_B.partition_D(qk_sB);
        auto tBcB_prod = smem_thr_copy_B.partition_D(cB);
        auto K_BLOCK_MAX_PROD = cute::size<2>(tBsB_prod);
        cute::for_each(cute::make_int_sequence<K_BLOCK_MAX_PROD>{},
                       [&](auto k_block) {
          auto dst = tBsB_prod(_, _, k_block, write_stage);
          auto coord_tensor = tBcB_prod(_, _, k_block, cute::Int<0>{});
          if (int(cute::size(dst)) % 8 != 0) {
            asm volatile("trap;\n");
          }
          for (int i = 0; i < int(cute::size(dst)); i += 8) {
            auto coord0 = coord_tensor(i);
            const int row0 = int(cute::get<0>(coord0));
            const int k0 = int(cute::get<1>(coord0));
            auto ref0 = dst(i);
            uint8_t* dst0 = cute::recast_ptr<uint8_t>(&ref0);
            if (((reinterpret_cast<uintptr_t>(dst0) & 3u) != 0u) ||
                ((k0 & 7) != 0) || paged_kv_params.k_stride_dim3 != 1) {
              asm volatile("trap;\n");
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
              if (row != row0 || k != k0 + j ||
                  dst_byte != dst0 + (j >> 1)) {
                asm volatile("trap;\n");
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

      for (int idx = lane_idx; idx < kCutlassTileN * kCutlassTileK / 2;
           idx += cutlass::NumThreadsPerWarp) {
        const int row = idx / (kCutlassTileK / 2);
        const int packed_k = idx - row * (kCutlassTileK / 2);
        const int k0 = 2 * packed_k;
        const int token = kv_tile * kCutlassTileN + row;
        const int scale_col = (k_outer * kCutlassTileK + k0) >> 4;
        const uint8_t scale =
            token < kv_len_tokens
                ? sm120_nvfp4_paged_k_scale(paged_kv_params, token, scale_col)
                : 0x38;
        qk_sSFB(row, k0, write_stage) = make_ue4m3_raw(scale);
      }
      cp_async::wait_group<0>();
    }
  };

  auto stage_paged_v_tile = [&](int kv_tile, int effective_out_group_idx,
                                int write_stage) {
    if constexpr (kUsePagedKv) {
      auto smem_tiled_copy_B = cute::make_tiled_copy_B(
          typename CutlassCollectiveMainloopK128Stage2::SmemCopyAtomB{},
          pv_tiled_mma);
      auto cB = cute::make_identity_tensor(
          cute::make_shape(cute::Int<kOutputTileN>{},
                           cute::Int<kCutlassTileN>{}, cute::Int<1>{}));

      auto pv_scale_for = [&](int token, int dim) {
        const int logical_page = token / paged_kv_params.page_size;
        return sm120_nvfp4_paged_v_pv_scale(paged_kv_params, logical_page,
                                            dim);
      };

      auto pv_code_for = [&](int token, int dim, uint8_t scale) {
        (void)scale;
        return token < kv_len_tokens
                   ? sm120_nvfp4_paged_v_code(paged_kv_params, token, dim)
                   : uint8_t{0};
      };

      for (int copy_thread = lane_idx;
           copy_thread < CutlassCollectiveMainloopK128Stage2::ThreadCount;
           copy_thread += cutlass::NumThreadsPerWarp) {
        auto smem_thr_copy_B =
            smem_tiled_copy_B.get_thread_slice(copy_thread);
        auto tBsB_prod = smem_thr_copy_B.partition_D(pv_sB);
        auto tBcB_prod = smem_thr_copy_B.partition_D(cB);
        auto K_BLOCK_MAX_PROD = cute::size<2>(tBsB_prod);
        cute::for_each(cute::make_int_sequence<K_BLOCK_MAX_PROD>{},
                       [&](auto k_block) {
          auto dst = tBsB_prod(_, _, k_block, write_stage);
          auto coord_tensor = tBcB_prod(_, _, k_block, cute::Int<0>{});
          for (int i = 0; i < int(cute::size(dst)); ++i) {
            auto coord = coord_tensor(i);
            const int col = int(cute::get<0>(coord));
            const int k = int(cute::get<1>(coord));
            const int token = kv_tile * kCutlassTileN + k;
            const int dim = effective_out_group_idx * kOutputTileN + col;
            const uint8_t scale = pv_scale_for(token, dim);
            dst(i) = cute::uint4_t(pv_code_for(token, dim, scale));
          }
        });
      }

      for (int idx = lane_idx; idx < kOutputTileN * kCutlassTileN / 2;
           idx += cutlass::NumThreadsPerWarp) {
        const int col = idx / (kCutlassTileN / 2);
        const int packed_k = idx - col * (kCutlassTileN / 2);
        const int k0 = 2 * packed_k;
        const int token = kv_tile * kCutlassTileN + k0;
        const int dim = effective_out_group_idx * kOutputTileN + col;
        const uint8_t scale =
            token < kv_len_tokens ? pv_scale_for(token, dim) : 0x38;
        pv_sSFB(col, k0, write_stage) = make_ue4m3_raw(scale);
      }
    }
  };

  auto load_q_chunk = [&](int k_outer) {
    if (is_load && lane_predicate) {
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
        if (lane_predicate) {
          k_pipeline.producer_acquire(k_pipe_write);
        }
        __syncwarp();
        const int write_stage = k_pipe_write.index();
        stage_paged_k_tile(kv_tile, k_outer, write_stage);
        cutlass::arch::fence_view_shared();
        __syncwarp();
        if (lane_predicate) {
          complete_manual_tma_pipeline_stage(
              k_pipeline, k_pipe_write,
              qk_params.mainloop.tma_transaction_bytes_nk);
          ++k_pipe_write;
        }
      }
    } else if (is_load && lane_predicate) {
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
        if (lane_predicate) {
          v_pipeline.producer_acquire(v_pipe_write);
        }
        __syncwarp();
        const int write_stage = v_pipe_write.index();
        stage_paged_v_tile(kv_tile, effective_out_group_idx, write_stage);
        cutlass::arch::fence_view_shared();
        __syncwarp();
        if (lane_predicate) {
          complete_manual_tma_pipeline_stage(
              v_pipeline, v_pipe_write,
              pv_params.mainloop.tma_transaction_bytes_nk);
          ++v_pipe_write;
        }
      }
    } else if (is_load && lane_predicate) {
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
	  auto p_sA0 = cute::make_tensor(
	      cute::make_smem_ptr(cute::recast_ptr<PvSmemAllocA>(
	          p_smem_a0_bytes)),
	      typename CutlassCollectiveMainloopK128Stage2::SmemLayoutA{});
	  auto p_sSFA0 = cute::make_tensor(
	      cute::make_smem_ptr(p_smem_sfa0),
	      typename CutlassCollectiveMainloopK128Stage2::SmemLayoutSFA{});
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
    qk_collective.load_tail(q_pipeline, q_pipe_write);

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

    qk_collective.load_tail(k_pipeline, k_pipe_write);
    pv_collective.load_tail(v_pipeline, v_pipe_write);
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
        effective_q_tile, storage.qk_tensors);
    if (qk_head_chunks > 1) {
      cutlass_qk_tma_q_register_stage(
          q_pipeline, q_pipe_read, q_frag1, q_scale_frag1, qk_mma_thread_idx,
          effective_q_tile, storage.qk_tensors);
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
                                         const __nv_bfloat16* smem_logits_stage,
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
            smem_logits_stage[logits_smem_index(mma_softmax_row, col)]);
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
              smem_logits_stage[logits_smem_index(mma_softmax_row,
                                                  local_col + i)]);
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
        if (contiguous && ((reinterpret_cast<uintptr_t>(dst0) & 0x3u) == 0)) {
          *reinterpret_cast<uint32_t*>(dst0) = packed_lo;
          *reinterpret_cast<uint32_t*>(dst0 + 4) = packed_hi;
        } else {
          *dst0 = static_cast<uint8_t>(packed_lo);
          *dst1 = static_cast<uint8_t>(packed_lo >> 8);
          *dst2 = static_cast<uint8_t>(packed_lo >> 16);
          *dst3 = static_cast<uint8_t>(packed_lo >> 24);
          *dst4 = static_cast<uint8_t>(packed_hi);
          *dst5 = static_cast<uint8_t>(packed_hi >> 8);
          *dst6 = static_cast<uint8_t>(packed_hi >> 16);
          *dst7 = static_cast<uint8_t>(packed_hi >> 24);
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
	      const __nv_bfloat16* smem_logits_stage = smem_logits0;
	      mma_stage_probability_row(p_sA0, p_sSFA0_m, smem_logits_stage, tile,
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
      __nv_bfloat16* smem_logits_stage = smem_logits0;
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
      for (int i = 0; i < cute::size(qk_accum); ++i) {
        auto coord = tCcC(i);
        const int row = int(cute::get<0>(coord));
        const int col = int(cute::get<1>(coord));
        if (row < kCutlassTileM && col < kCutlassTileN) {
          const float logit =
              transform_score(qk_accum(i) * qk_scale, row, col, tile);
          smem_logits_stage[logits_smem_index(row, col)] =
              __float2bfloat16(logit);
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
          bool kUseSlidingWindow, bool kUseLogitsSoftCap>
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
  if (workspace_bytes < qk_workspace_alloc + pv_workspace_size) {
    return cudaErrorInvalidValue;
  }
  char* pv_workspace = workspace_base + qk_workspace_alloc;
  const size_t required_workspace_bytes = qk_workspace_alloc + pv_workspace_size;
  constexpr size_t kWorkspaceClearBytes = 32 * 1024 * 1024;
  const size_t workspace_clear_bytes =
      workspace_bytes < kWorkspaceClearBytes ? workspace_bytes
                                             : kWorkspaceClearBytes;
  const size_t workspace_zero_bytes =
      workspace_clear_bytes > required_workspace_bytes
          ? workspace_clear_bytes
          : required_workspace_bytes;
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

  constexpr int kSmemBytes =
      static_cast<int>(sizeof(Sm120Nvfp4QkvLoadCollectiveStorage));
  auto stage_kernel =
      sm120_nvfp4_qkv_online_register_q_stage_kernel<
          kOutputGroupSpan, kUsePagedKv, kCausal, kUseSlidingWindow,
          kUseLogitsSoftCap>;
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
      paged_kv_params, qo_indptr, kv_lens, batch_size, q_tiles_per_sequence);
  status = cudaGetLastError();
  return status;
}

}  // namespace flashinfer::attention::blackwell::sm120_nvfp4::d128
