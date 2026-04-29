#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

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

#include <flashinfer/gemm/cutlass_gemm_configs.h>
#include <flashinfer/gemm/fp4_gemm_cutlass.h>
#include <flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h>
#include <flashinfer/mma.cuh>

namespace {

constexpr int kQLen = 512;
constexpr int kGroup = 8;
constexpr int kKvLen = 32768;
constexpr int kHeadDim = 512;
constexpr int kPackedHeadDim = kHeadDim / 2;
constexpr int kScaleCols = kHeadDim / 16;
constexpr int kQRows = kQLen * kGroup;
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kCutlassTileM = 128;
constexpr int kCutlassTileN = 128;
constexpr int kCutlassTileK = 256;
constexpr int kCutlassTileK128 = 128;
constexpr int kDebugHead = 0;
constexpr int kProbPackedCols = kKvLen / 2;
constexpr int kProbScaleCols = kKvLen / 16;
constexpr int kFusedWarpsPerCta = 8;
constexpr int kSplitKvLen = 1024;
constexpr int kNumKvSplits = kKvLen / kSplitKvLen;
constexpr int kBenchRows = 128;
constexpr int kBenchQTiles = kBenchRows / kTileM;
constexpr int kColumnGroups = kHeadDim / (kTileN * kFusedWarpsPerCta);
constexpr float kProbGlobalScale = 6.0f * 448.0f;
constexpr float kQkScale = 0.044194173824159216f;  // 1 / sqrt(512)

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

constexpr int kSm120Nvfp4FmhaNumWarpsSoftmax0 = 2;
constexpr int kSm120Nvfp4FmhaNumWarpsSoftmax1 = 2;
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
constexpr int kSm120Nvfp4FmhaSoftmaxThreadCount =
    (kSm120Nvfp4FmhaNumWarpsSoftmax0 +
     kSm120Nvfp4FmhaNumWarpsSoftmax1) *
    cutlass::NumThreadsPerWarp;
constexpr int kSm120Nvfp4FmhaSoftmaxGroupThreadCount =
    kSm120Nvfp4FmhaNumWarpsSoftmax0 * cutlass::NumThreadsPerWarp;
static_assert(kSm120Nvfp4FmhaNumWarpsSoftmax0 ==
              kSm120Nvfp4FmhaNumWarpsSoftmax1);
constexpr int kSm120Nvfp4FmhaMmaSoftmaxThreadCount =
    kSm120Nvfp4FmhaNumWarpsMma * cutlass::NumThreadsPerWarp +
    kSm120Nvfp4FmhaSoftmaxThreadCount;
constexpr int kSm120Nvfp4FmhaMmaSoftmaxLoadThreadCount =
    kSm120Nvfp4FmhaMmaSoftmaxThreadCount +
    kSm120Nvfp4FmhaNumWarpsLoad * cutlass::NumThreadsPerWarp;
constexpr uint32_t kSm120Nvfp4BarrierSoftmax0Internal = 0;
constexpr uint32_t kSm120Nvfp4BarrierSoftmax1Internal = 1;
constexpr uint32_t kSm120Nvfp4BarrierSoftmaxInternal =
    kSm120Nvfp4BarrierSoftmax0Internal;
constexpr uint32_t kSm120Nvfp4BarrierMmaSoftmax =
    kSm120Nvfp4BarrierSoftmax1Internal;
constexpr uint32_t kSm120Nvfp4BarrierSoftmaxMma = 2;
constexpr uint32_t kSm120Nvfp4BarrierSoftmaxCorrection = 3;
constexpr uint32_t kSm120Nvfp4BarrierCorrectionSoftmax = 4;
constexpr uint32_t kSm120Nvfp4BarrierSoftmax0OnlineReady = 5;
constexpr uint32_t kSm120Nvfp4BarrierSoftmax1OnlineReady = 6;

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

__device__ __forceinline__ int sm120_nvfp4_fmha_softmax_thread_idx(
    int thread_idx) {
  return thread_idx - kSm120Nvfp4FmhaWarpSoftmax0Begin *
                          cutlass::NumThreadsPerWarp;
}

__device__ __forceinline__ int sm120_nvfp4_fmha_softmax_group_thread_idx(
    int thread_idx,
    Sm120Nvfp4FmhaRole role) {
  const int group_warp_begin =
      role == Sm120Nvfp4FmhaRole::Softmax1
          ? kSm120Nvfp4FmhaWarpSoftmax1Begin
          : kSm120Nvfp4FmhaWarpSoftmax0Begin;
  return thread_idx - group_warp_begin * cutlass::NumThreadsPerWarp;
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
using CutlassThreadBlockShape = cute::Shape<cute::_128, cute::_128, cute::_256>;
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
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CutlassCollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using CutlassProblemShape = cute::Shape<int, int, int, int>;
using CutlassGemmKernel =
    cutlass::gemm::kernel::GemmUniversal<CutlassProblemShape,
                                         CutlassCollectiveMainloop,
                                         CutlassCollectiveEpilogue,
                                         cutlass::gemm::StaticPersistentScheduler>;
using CutlassGemm =
    cutlass::gemm::device::GemmUniversalAdapter<CutlassGemmKernel>;

template <typename ThreadBlockShape>
struct Sm120Fp4CollectiveTraits {
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm120,
          cutlass::arch::OpClassTensorOp,
          ThreadBlockShape,
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
  using CollectiveMainloop =
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
          ThreadBlockShape,
          CutlassClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(
                  sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<CutlassProblemShape,
                                           CollectiveMainloop,
                                           CollectiveEpilogue,
                                           cutlass::gemm::StaticPersistentScheduler>;
};

using Sm120Fp4Tile128x128x128 =
    Sm120Fp4CollectiveTraits<cute::Shape<cute::_128, cute::_128, cute::_128>>;
using Sm120Fp4Tile128x128x256 =
    Sm120Fp4CollectiveTraits<cute::Shape<cute::_128, cute::_128, cute::_256>>;
using Sm120Fp4Tile256x128x128 =
    Sm120Fp4CollectiveTraits<cute::Shape<cute::_256, cute::_128, cute::_128>>;
using CutlassThreadBlockShapeK128 =
    cute::Shape<cute::_128, cute::_128, cute::_128>;
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
        typename Sm120Fp4Tile128x128x128::CollectiveEpilogue,
        cutlass::gemm::StaticPersistentScheduler>;
using CutlassGemmK128Stage2 =
    cutlass::gemm::device::GemmUniversalAdapter<CutlassGemmKernelK128Stage2>;

using RunnerConfig = flashinfer::gemm::CutlassGemmConfig;
using RunnerTileConfig = flashinfer::gemm::CutlassTileConfigSM120;
using RunnerMainloopSchedule = flashinfer::gemm::MainloopScheduleType;
using RunnerEpilogueSchedule = flashinfer::gemm::EpilogueScheduleType;
using RunnerClusterShape = flashinfer::gemm::ClusterShape;
using RunnerFp4Type = flashinfer::gemm::FP4GemmType;

using Sm120Nvfp4PipelineS = cutlass::PipelineAsync<1>;
using Sm120Nvfp4PipelineE = cutlass::PipelineAsync<2>;
using Sm120Nvfp4OrderBarrierSoftmax =
    cutlass::OrderedSequenceBarrier<1, 2>;

struct Sm120Nvfp4MainloopPipelineStorage {
  alignas(16) typename Sm120Nvfp4PipelineS::SharedStorage mma_s0;
  alignas(16) typename Sm120Nvfp4PipelineS::SharedStorage mma_s1;
  alignas(16) typename Sm120Nvfp4PipelineE::SharedStorage corr_epi;
  alignas(16) typename Sm120Nvfp4OrderBarrierSoftmax::SharedStorage order_s01;
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

struct Sm120Nvfp4QkvLoadCollectiveStorage {
  alignas(128) typename CutlassCollectiveMainloop::TensorStorage qk_tensors;
  alignas(16) typename CutlassCollectiveMainloop::PipelineStorage
      q_pipeline_storage;
  alignas(16) typename CutlassCollectiveMainloop::PipelineStorage
      k_pipeline_storage;
  alignas(1024) cute::ArrayEngine<
      typename CutlassCollectiveMainloopK128Stage2::SmemAllocTypeB,
      cute::cosize_v<typename CutlassCollectiveMainloopK128Stage2::SmemLayoutB>>
      v_smem_B;
  alignas(16) cute::ArrayEngine<
      cutlass::float_ue4m3_t,
      cute::cosize_v<typename CutlassCollectiveMainloopK128Stage2::SmemLayoutSFB>>
      v_smem_SFB;
  alignas(16) typename CutlassCollectiveMainloopK128Stage2::PipelineStorage
      v_pipeline_storage;
  alignas(16) Sm120Nvfp4MainloopPipelineStorage role_pipeline_storage;
  alignas(16) float global_m[kCutlassTileM];
  alignas(16) float global_l[kCutlassTileM];
  alignas(16) float old_scale_stage[2][kCutlassTileM];
};

static_assert(sizeof(Sm120Nvfp4QkvLoadCollectiveStorage) <= (99u << 10),
              "SM120 Q/K/V load collective storage must fit SM120 opt-in shared memory");
static_assert(
    sizeof(decltype(
        ((typename CutlassCollectiveMainloop::TensorStorage*)nullptr)->smem_B)) >=
        kCutlassTileM * kCutlassTileN * static_cast<int>(sizeof(__nv_bfloat16)),
    "QK B smem must be large enough for the aliased BF16 logits/O tile");
constexpr int kSm120Nvfp4PvPStageBytes = cutlass::bits_to_bytes(
    cute::cosize_v<typename CutlassCollectiveMainloopK128Stage2::SmemLayoutA> *
    cute::sizeof_bits_v<
        typename CutlassCollectiveMainloopK128Stage2::SmemAllocTypeA>);
constexpr int kSm120Nvfp4PvScaleStageElems =
    cute::cosize_v<typename CutlassCollectiveMainloopK128Stage2::SmemLayoutSFA>;
static_assert(
    sizeof(decltype(
        ((typename CutlassCollectiveMainloop::TensorStorage*)nullptr)->smem_A)) >=
        2 * kSm120Nvfp4PvPStageBytes,
    "QK A smem must be large enough for two compact aliased PV P stages");
static_assert(
    sizeof(decltype(
        ((typename CutlassCollectiveMainloop::TensorStorage*)nullptr)->smem_SFA)) >=
        2 * kSm120Nvfp4PvScaleStageElems *
            static_cast<int>(sizeof(cutlass::float_ue4m3_t)),
    "QK SFA smem must be large enough for two compact PV P scale stages");

__device__ __forceinline__ bool finite_f32(float x) {
  return x == x && fabsf(x) != INFINITY;
}

__device__ __forceinline__ uint8_t fp32_to_e4m3_byte(float x) {
  if (!(x > 0.0f) || !finite_f32(x)) {
    x = 1.0e-8f;
  }
  __nv_fp8_e4m3 y = static_cast<__nv_fp8_e4m3>(x);
  return y.__x;
}

__device__ __forceinline__ float e4m3_byte_to_fp32(uint8_t x) {
  __nv_fp8_e4m3 y;
  y.__x = x;
  return static_cast<float>(y);
}

__device__ __forceinline__ uint8_t nearest_e2m1_code(float x) {
  constexpr float values[16] = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  x = fminf(fmaxf(finite_f32(x) ? x : 0.0f, -6.0f), 6.0f);
  float best_dist = fabsf(x - values[0]);
  uint8_t best = 0;
#pragma unroll
  for (uint8_t i = 1; i < 16; ++i) {
    const float dist = fabsf(x - values[i]);
    if (dist < best_dist) {
      best_dist = dist;
      best = i;
    }
  }
  return best;
}

__device__ __forceinline__ uint8_t fp32_pair_to_e2m1_byte(float x, float y) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint16_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
      "mov.b16 %0, {byte0, 0};\n"
      "}"
      : "=h"(val)
      : "f"(x), "f"(y));
  return static_cast<uint8_t>(val);
#else
  return 0;
#endif
}

__device__ __forceinline__ uint8_t fp32_to_e2m1_code_hw(float x) {
  return static_cast<uint8_t>(fp32_pair_to_e2m1_byte(x, x) & 0x0Fu);
}

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

__global__ void quantize_q_rowmajor_kernel(const __nv_bfloat16* q,
                                           uint8_t* q_packed,
                                           uint8_t* q_scales) {
  const int row = blockIdx.x;
  const int scale_col = threadIdx.x;
  if (row >= kQRows || scale_col >= kScaleCols) {
    return;
  }

  const int base = row * kHeadDim + scale_col * 16;
  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    max_abs = fmaxf(max_abs, fabsf(__bfloat162float(q[base + i])));
  }

  const uint8_t scale_byte = fp32_to_e4m3_byte(fmaxf(max_abs / 6.0f, 1.0e-8f));
  q_scales[row * kScaleCols + scale_col] = scale_byte;
  const float scale = fmaxf(e4m3_byte_to_fp32(scale_byte), 1.0e-8f);

#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const float x0 = __bfloat162float(q[base + 2 * pair]) / scale;
    const float x1 = __bfloat162float(q[base + 2 * pair + 1]) / scale;
    const uint8_t c0 = nearest_e2m1_code(x0);
    const uint8_t c1 = nearest_e2m1_code(x1);
    q_packed[row * kPackedHeadDim + scale_col * 8 + pair] =
        static_cast<uint8_t>(c0 | (c1 << 4));
  }
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
    typename CutlassCollectiveMainloop::TensorStorage& shared_tensors) {
  using namespace cute;

  Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()),
                          typename CutlassCollectiveMainloop::SmemLayoutA{});
  Tensor sSFA = make_tensor(
      make_smem_ptr(shared_tensors.smem_SFA.begin()),
      typename CutlassCollectiveMainloop::SmemLayoutSFA{});

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
    float pv_base_scale) {
  for (int i = 0; i < int(cute::size(accum)); ++i) {
    auto coord = coords(i);
    const int row = int(cute::get<0>(coord));
    const int col = int(cute::get<1>(coord));
    if (row < kCutlassTileM && col < kCutlassTileN) {
      const float row_scale =
          pv_base_scale / fmaxf(global_l[row], 1.0e-20f);
      smem_o[row * kCutlassTileN + col] =
          __float2bfloat16(accum(i) * row_scale);
    }
  }
}

__device__ __forceinline__ void sm120_epilogue_store_bf16_tile(
    const __nv_bfloat16* smem_o,
    __nv_bfloat16* out_tile,
    int out_stride_cols,
    int epilogue_thread_idx) {
  for (int idx = epilogue_thread_idx; idx < kCutlassTileM * kCutlassTileN;
       idx += kSm120Nvfp4FmhaNumWarpsEpilogue * cutlass::NumThreadsPerWarp) {
    const int row = idx / kCutlassTileN;
    const int col = idx - row * kCutlassTileN;
    out_tile[row * out_stride_cols + col] = smem_o[idx];
  }
}

__global__ __launch_bounds__(kSm120Nvfp4FmhaThreadCount, 1)
void sm120_nvfp4_qkv_online_register_q_stage_kernel(
    CUTLASS_GRID_CONSTANT typename CutlassGemmKernel::Params const qk_params,
    CUTLASS_GRID_CONSTANT typename CutlassGemmKernelK128Stage2::Params const pv_params,
    __nv_bfloat16* out_group,
    float qk_alpha,
    float pv_alpha,
    int q_tile,
    int kv_tile_start,
    int num_kv_tiles,
    int out_group_idx,
    int out_stride_cols) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  using cute::_;

  extern __shared__ __align__(128) char smem[];
  auto& storage = *reinterpret_cast<Sm120Nvfp4QkvLoadCollectiveStorage*>(smem);
  const int effective_q_tile = q_tile + int(blockIdx.x);
  const int effective_out_group_idx = out_group_idx + int(blockIdx.y);
  __nv_bfloat16* out_tile =
      out_group + int(blockIdx.x) * kCutlassTileM * out_stride_cols +
      int(blockIdx.y) * kCutlassTileN;

  const int thread_idx = int(threadIdx.x);
  const int warp_idx = thread_idx / cutlass::NumThreadsPerWarp;
  const int lane_idx = thread_idx % cutlass::NumThreadsPerWarp;
  const bool lane_predicate = lane_idx == 0;
  const Sm120Nvfp4FmhaRole role = sm120_nvfp4_fmha_role_for_warp(warp_idx);
  const bool is_load = role == Sm120Nvfp4FmhaRole::Load;
  const bool is_mma = role == Sm120Nvfp4FmhaRole::Mma;
  const bool is_softmax0 = role == Sm120Nvfp4FmhaRole::Softmax0;
  const bool is_softmax1 = role == Sm120Nvfp4FmhaRole::Softmax1;
  const bool is_softmax = role == Sm120Nvfp4FmhaRole::Softmax0 ||
                          role == Sm120Nvfp4FmhaRole::Softmax1;
  const bool is_epilogue = role == Sm120Nvfp4FmhaRole::Epilogue;
  const int qk_mma_thread_idx =
      is_mma ? sm120_nvfp4_fmha_mma_thread_idx(thread_idx) : 0;
  const int pv_mma_thread_idx = qk_mma_thread_idx;
  const int softmax_group_thread_idx =
      is_softmax ? sm120_nvfp4_fmha_softmax_group_thread_idx(thread_idx, role)
                 : 0;
  const int epilogue_thread_idx =
      is_epilogue
          ? thread_idx - kSm120Nvfp4FmhaWarpEpilogue * cutlass::NumThreadsPerWarp
          : 0;

  typename Sm120Nvfp4PipelineS::Params pipeline_mma_s0_params{};
  typename Sm120Nvfp4PipelineS::Params pipeline_mma_s1_params{};
  typename Sm120Nvfp4PipelineE::Params pipeline_corr_epi_params{};
  if (is_mma) {
    pipeline_mma_s0_params.role =
        Sm120Nvfp4PipelineS::ThreadCategory::Producer;
    pipeline_mma_s1_params.role =
        Sm120Nvfp4PipelineS::ThreadCategory::Producer;
  }
  if (is_softmax0) {
    pipeline_mma_s0_params.role =
        Sm120Nvfp4PipelineS::ThreadCategory::Consumer;
  }
  if (is_softmax1) {
    pipeline_mma_s1_params.role =
        Sm120Nvfp4PipelineS::ThreadCategory::Consumer;
  }
  if (is_mma) {
    pipeline_corr_epi_params.role =
        Sm120Nvfp4PipelineE::ThreadCategory::Producer;
  }
  if (is_epilogue) {
    pipeline_corr_epi_params.role =
        Sm120Nvfp4PipelineE::ThreadCategory::Consumer;
  }
  pipeline_mma_s0_params.producer_arv_count = 1;
  pipeline_mma_s1_params.producer_arv_count = 1;
  pipeline_mma_s0_params.consumer_arv_count =
      kSm120Nvfp4FmhaSoftmaxGroupThreadCount;
  pipeline_mma_s1_params.consumer_arv_count =
      kSm120Nvfp4FmhaSoftmaxGroupThreadCount;
  pipeline_corr_epi_params.producer_arv_count = 1;
  pipeline_corr_epi_params.consumer_arv_count =
      kSm120Nvfp4FmhaNumWarpsEpilogue * cutlass::NumThreadsPerWarp;
  pipeline_mma_s0_params.initializing_warp = kSm120Nvfp4FmhaWarpLoad;
  pipeline_mma_s1_params.initializing_warp = kSm120Nvfp4FmhaWarpLoad;
  pipeline_corr_epi_params.initializing_warp = kSm120Nvfp4FmhaWarpLoad;
  Sm120Nvfp4PipelineS pipeline_mma_s0(
      storage.role_pipeline_storage.mma_s0, pipeline_mma_s0_params,
      cute::true_type{});
  Sm120Nvfp4PipelineS pipeline_mma_s1(
      storage.role_pipeline_storage.mma_s1, pipeline_mma_s1_params,
      cute::true_type{});
  Sm120Nvfp4PipelineE pipeline_corr_epi(
      storage.role_pipeline_storage.corr_epi, pipeline_corr_epi_params,
      cute::true_type{});
  typename Sm120Nvfp4OrderBarrierSoftmax::Params order_s01_params{};
  order_s01_params.group_id =
      role == Sm120Nvfp4FmhaRole::Softmax1 ? 1 : 0;
  order_s01_params.group_size = kSm120Nvfp4FmhaSoftmaxGroupThreadCount;
  order_s01_params.initializing_warp = kSm120Nvfp4FmhaWarpLoad;
  Sm120Nvfp4OrderBarrierSoftmax order_s01(
      storage.role_pipeline_storage.order_s01, order_s01_params);

  typename CutlassCollectiveMainloop::PipelineState k_pipe_release;
  typename Sm120Nvfp4PipelineS::PipelineState pipeline_mma_s0_producer_state =
      cutlass::make_producer_start_state<Sm120Nvfp4PipelineS>();
  typename Sm120Nvfp4PipelineS::PipelineState pipeline_mma_s0_consumer_state;
  typename Sm120Nvfp4PipelineS::PipelineState pipeline_mma_s1_producer_state =
      cutlass::make_producer_start_state<Sm120Nvfp4PipelineS>();
  typename Sm120Nvfp4PipelineS::PipelineState pipeline_mma_s1_consumer_state;
  typename Sm120Nvfp4PipelineE::PipelineState pipeline_corr_epi_producer_state =
      cutlass::make_producer_start_state<Sm120Nvfp4PipelineE>();
  typename Sm120Nvfp4PipelineE::PipelineState pipeline_corr_epi_consumer_state;
  bool pipeline_mma_s0_acquired = false;
  bool pipeline_mma_s1_acquired = false;

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
      cute::recast_ptr<__nv_bfloat16>(storage.qk_tensors.smem_B.begin());
  __nv_bfloat16* smem_logits1 = smem_logits0;
  __nv_bfloat16* smem_epilogue_o = smem_logits0;
  auto qk_sSFA = cute::make_tensor(
      cute::make_smem_ptr(storage.qk_tensors.smem_SFA.begin()),
      typename CutlassCollectiveMainloop::SmemLayoutSFA{});
  auto qk_sSFB = cute::make_tensor(
      cute::make_smem_ptr(storage.qk_tensors.smem_SFB.begin()),
      typename CutlassCollectiveMainloop::SmemLayoutSFB{});

  auto q_frag0 = qk_thread_mma.partition_fragment_A(qk_sA(_, _, cute::Int<0>{}));
  auto q_frag1 = qk_thread_mma.partition_fragment_A(qk_sA(_, _, cute::Int<0>{}));
  auto q_scale_frag0 =
      qk_collective.partition_fragment_SFA(qk_sSFA(_, _, cute::Int<0>{}),
                                           qk_thread_mma);
  auto q_scale_frag1 =
      qk_collective.partition_fragment_SFA(qk_sSFA(_, _, cute::Int<0>{}),
                                           qk_thread_mma);

  CutlassCollectiveMainloopK128Stage2 pv_collective;
  auto pv_tiled_mma = typename CutlassCollectiveMainloopK128Stage2::TiledMma{};
  auto pv_thread_mma = pv_tiled_mma.get_thread_slice(pv_mma_thread_idx);
  auto pv_sB = cute::make_tensor(
      cute::make_smem_ptr(storage.v_smem_B.begin()),
      typename CutlassCollectiveMainloopK128Stage2::SmemLayoutB{});
  auto pv_sSFB = cute::make_tensor(
      cute::make_smem_ptr(storage.v_smem_SFB.begin()),
      typename CutlassCollectiveMainloopK128Stage2::SmemLayoutSFB{});
  auto v_frag = pv_thread_mma.partition_fragment_B(pv_sB(_, _, cute::Int<0>{}));
  auto v_scale_frag =
      pv_collective.partition_fragment_SFB(pv_sSFB(_, _, cute::Int<0>{}),
                                           pv_thread_mma);
  auto pv_accum = cute::partition_fragment_C(
      pv_tiled_mma, cute::take<0, 2>(CutlassThreadBlockShapeK128{}));

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

  auto load_q_chunk = [&](int k_outer) {
    if (is_load && lane_predicate) {
      auto gA = qk_gA_mkl(_, _, effective_q_tile, _, 0);
      auto gSFA = qk_gSFA_mkl(_, _, effective_q_tile, _, 0);
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
    if (is_load && lane_predicate) {
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

  auto load_v_chunk = [&](int kv_tile) {
    if (is_load && lane_predicate) {
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

  using PvSmemAllocA =
      typename CutlassCollectiveMainloopK128Stage2::SmemAllocTypeA;
  uint8_t* p_smem_a_bytes =
      cute::recast_ptr<uint8_t>(storage.qk_tensors.smem_A.begin());
  cutlass::float_ue4m3_t* p_smem_sfa =
      storage.qk_tensors.smem_SFA.begin();
  auto p_sA0 = cute::make_tensor(
      cute::make_smem_ptr(cute::recast_ptr<PvSmemAllocA>(
          p_smem_a_bytes)),
      typename CutlassCollectiveMainloopK128Stage2::SmemLayoutA{});
  auto p_sA1 = cute::make_tensor(
      cute::make_smem_ptr(cute::recast_ptr<PvSmemAllocA>(
          p_smem_a_bytes + kSm120Nvfp4PvPStageBytes)),
      typename CutlassCollectiveMainloopK128Stage2::SmemLayoutA{});
  auto p_sSFA0 = cute::make_tensor(
      cute::make_smem_ptr(p_smem_sfa),
      typename CutlassCollectiveMainloopK128Stage2::SmemLayoutSFA{});
  auto p_sSFA1 = cute::make_tensor(
      cute::make_smem_ptr(p_smem_sfa + kSm120Nvfp4PvScaleStageElems),
      typename CutlassCollectiveMainloopK128Stage2::SmemLayoutSFA{});

  if (is_load) {
    load_q_chunk(0);
    load_k_chunk(kv_tile_start, 0);
    load_q_chunk(1);
    load_v_chunk(kv_tile_start);
    load_k_chunk(kv_tile_start, 1);
    qk_collective.load_tail(q_pipeline, q_pipe_write);

    for (int tile = 0; tile < num_kv_tiles; ++tile) {
      if (tile + 1 < num_kv_tiles) {
        const int next_kv_tile = kv_tile_start + tile + 1;
        load_k_chunk(next_kv_tile, 0);
        load_k_chunk(next_kv_tile, 1);
        load_v_chunk(next_kv_tile);
      }
    }

    qk_collective.load_tail(k_pipeline, k_pipe_write);
    pv_collective.load_tail(v_pipeline, v_pipe_write);
  } else if (is_mma) {
    cutlass_qk_tma_q_register_stage(
        q_pipeline, q_pipe_read, q_frag0, q_scale_frag0, qk_mma_thread_idx,
        storage.qk_tensors);
    cutlass_qk_tma_q_register_stage(
        q_pipeline, q_pipe_read, q_frag1, q_scale_frag1, qk_mma_thread_idx,
        storage.qk_tensors);
    cute::clear(pv_accum);
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
          cutlass::arch::NamedBarrier::sync(
              cute::thr_size(qk_tiled_mma),
              cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
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

      cutlass::arch::NamedBarrier::sync(
          cute::thr_size(pv_tiled_mma),
          cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
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

    auto pv_gemm_p_stage = [&](auto& accum, auto& p_sA_stage,
                               auto& p_sSFA_stage) {
      auto pv_tCrA =
          pv_thread_mma.partition_fragment_A(p_sA_stage(_, _, cute::Int<0>{}));
      auto pv_tCrSFA =
          pv_collective.partition_fragment_SFA(
              p_sSFA_stage(_, _, cute::Int<0>{}), pv_thread_mma);
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

      auto pv_gemm_kblock = [&](auto k_block) {
        cute::gemm(pv_tiled_mma,
                   cute::make_zip_tensor(pv_tCrA(_, _, k_block),
                                          pv_tCrSFA(_, _, k_block)),
                   cute::make_zip_tensor(v_frag(_, _, k_block),
                                          v_scale_frag(_, _, k_block)),
                   accum);
      };

      pv_copy_p_kblock(cute::_0{});
      cute::for_each(cute::make_int_sequence<pv_K_BLOCK_MAX>{},
                     [&](auto k_block) {
        auto k_block_next =
            ((k_block + 1) == pv_K_BLOCK_MAX) ? 0 : (k_block + 1);
        if (k_block_next > 0) {
          pv_copy_p_kblock(k_block_next);
        }
        pv_gemm_kblock(k_block);
      });
    };

    auto pv_cC = cute::make_identity_tensor(
        cute::take<0, 2>(CutlassThreadBlockShapeK128{}));
    auto pv_tCcC = pv_thread_mma.partition_C(pv_cC);

    auto acquire_score_stages = [&]() {
      if (!pipeline_mma_s0_acquired) {
        pipeline_mma_s0.producer_acquire(pipeline_mma_s0_producer_state);
      }
      if (!pipeline_mma_s1_acquired) {
        pipeline_mma_s1.producer_acquire(pipeline_mma_s1_producer_state);
      }
      pipeline_mma_s0_acquired = false;
      pipeline_mma_s1_acquired = false;
    };

    auto commit_score_stages = [&]() {
      cutlass::arch::NamedBarrier::sync(
          CutlassCollectiveMainloop::ThreadCount,
          cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
      if (qk_mma_thread_idx == 0) {
        pipeline_mma_s0.producer_commit(pipeline_mma_s0_producer_state);
        pipeline_mma_s1.producer_commit(pipeline_mma_s1_producer_state);
      }
      ++pipeline_mma_s0_producer_state;
      ++pipeline_mma_s1_producer_state;
    };

    auto release_k_chunk = [&]() {
      k_pipeline.consumer_release(k_pipe_release);
      ++k_pipe_release;
    };

    auto wait_p_ready_and_release_k = [&]() {
      pipeline_mma_s0.producer_acquire(pipeline_mma_s0_producer_state);
      pipeline_mma_s0_acquired = true;
      release_k_chunk();
      pipeline_mma_s1.producer_acquire(pipeline_mma_s1_producer_state);
      pipeline_mma_s1_acquired = true;
      release_k_chunk();
    };

    auto acquire_output_stage = [&]() {
      pipeline_corr_epi.producer_acquire(pipeline_corr_epi_producer_state);
    };

    auto commit_output_stage = [&](bool final_tile) {
      if (final_tile) {
        cutlass::arch::NamedBarrier::sync(
            CutlassCollectiveMainloopK128Stage2::ThreadCount,
            cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
        cutlass::arch::fence_view_async_shared();
      }
      if (qk_mma_thread_idx == 0) {
        pipeline_corr_epi.producer_commit(pipeline_corr_epi_producer_state);
      }
      ++pipeline_corr_epi_producer_state;
    };

    auto run_qk_tile = [&](int tile) {
      __nv_bfloat16* smem_logits_stage =
          (tile & 1) == 0 ? smem_logits0 : smem_logits1;
      auto qk_accum = cute::partition_fragment_C(
          qk_tiled_mma, cute::take<0, 2>(CutlassThreadBlockShape{}));
      cute::clear(qk_accum);
      qk_consume_k_stage(q_frag0, q_scale_frag0, qk_accum);
      qk_consume_k_stage(q_frag1, q_scale_frag1, qk_accum);

      auto cC = cute::make_identity_tensor(
          cute::take<0, 2>(CutlassThreadBlockShape{}));
      auto tCcC = qk_thread_mma.partition_C(cC);
      for (int i = 0; i < cute::size(qk_accum); ++i) {
        auto coord = tCcC(i);
        const int row = int(cute::get<0>(coord));
        const int col = int(cute::get<1>(coord));
        if (row < kCutlassTileM && col < kCutlassTileN) {
          const float logit = qk_accum(i) * qk_alpha * kQkScale;
          smem_logits_stage[row * kCutlassTileN + col] =
              __float2bfloat16(logit);
        }
      }

      commit_score_stages();
    };

    auto run_pv_tile = [&](int tile, bool final_tile) {
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
      if ((tile & 1) == 0) {
        pv_gemm_p_stage(pv_accum, p_sA0, p_sSFA0);
      } else {
        pv_gemm_p_stage(pv_accum, p_sA1, p_sSFA1);
      }
      if (final_tile) {
        const float pv_base_scale = pv_alpha / kProbGlobalScale;
        sm120_stage_o_fragment_to_epilogue_smem(
            pv_accum, pv_tCcC, smem_epilogue_o, storage.global_l,
            pv_base_scale);
        commit_output_stage(true);
      }
    };

    for (int tile = 0; tile < num_kv_tiles; ++tile) {
      acquire_score_stages();
      run_qk_tile(tile);
      if (tile > 0) {
        run_pv_tile(tile - 1, false);
      }
      wait_p_ready_and_release_k();
    }
    run_pv_tile(num_kv_tiles - 1, true);
  } else if (is_softmax) {
    auto wait_score_stage = [&]() {
      if (is_softmax0) {
        pipeline_mma_s0.consumer_wait(pipeline_mma_s0_consumer_state);
      } else {
        pipeline_mma_s1.consumer_wait(pipeline_mma_s1_consumer_state);
      }
    };

    auto release_p_ready = [&]() {
      if (is_softmax0) {
        pipeline_mma_s0.consumer_release(pipeline_mma_s0_consumer_state);
        ++pipeline_mma_s0_consumer_state;
      } else {
        pipeline_mma_s1.consumer_release(pipeline_mma_s1_consumer_state);
        ++pipeline_mma_s1_consumer_state;
      }
    };

    constexpr int kRowsPerSoftmaxRole = kCutlassTileM / 2;
    const int row_begin = is_softmax1 ? kRowsPerSoftmaxRole : 0;
    const int row_end = row_begin + kRowsPerSoftmaxRole;
    const int owned_row = row_begin + softmax_group_thread_idx;
    const bool owns_row = softmax_group_thread_idx < kRowsPerSoftmaxRole;
    const uint32_t online_barrier_id =
        is_softmax1 ? kSm120Nvfp4BarrierSoftmax1OnlineReady
                    : kSm120Nvfp4BarrierSoftmax0OnlineReady;
    float running_m = -INFINITY;
    float running_l = 0.0f;

    auto stage_probability_row = [&](auto& p_sA, auto& p_sSFA,
                                     const __nv_bfloat16* smem_logits_stage,
                                     int row, float tile_m,
                                     float tile_scale) {
      float tile_l_scaled = 0.0f;
#pragma unroll
      for (int scale_group = 0; scale_group < kCutlassTileN / 16;
           ++scale_group) {
        const int local_col = scale_group * 16;
        float vec_max = 0.0f;
        float p_vals[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const float logit = __bfloat162float(
              smem_logits_stage[row * kCutlassTileN + local_col + i]);
          const float p_scaled = __expf(logit - tile_m) * tile_scale;
          tile_l_scaled += p_scaled;
          vec_max = fmaxf(vec_max, p_scaled);
          p_vals[i] = p_scaled;
        }

        const float scale_value =
            fmaxf(kProbGlobalScale * vec_max / 6.0f, 1.0e-8f);
        const uint8_t scale_byte = fp32_to_e4m3_byte(scale_value);
        p_sSFA(row, local_col, cute::Int<0>{}) = make_ue4m3_raw(scale_byte);
        const float output_scale = kProbGlobalScale / scale_value;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          p_sA(row, local_col + i, cute::Int<0>{}) =
              cute::uint4_t(fp32_to_e2m1_code_hw(p_vals[i] * output_scale));
        }
      }
      return tile_l_scaled;
    };

    for (int tile = 0; tile < num_kv_tiles; ++tile) {
      wait_score_stage();
      const __nv_bfloat16* smem_logits_stage =
          (tile & 1) == 0 ? smem_logits0 : smem_logits1;
      if (owns_row) {
        float tile_m = -INFINITY;
#pragma unroll
        for (int col = 0; col < kCutlassTileN; ++col) {
          const float logit = __bfloat162float(
              smem_logits_stage[owned_row * kCutlassTileN + col]);
          tile_m = fmaxf(tile_m, logit);
        }
        const float next_m = fmaxf(running_m, tile_m);
        const float old_scale =
            running_l == 0.0f ? 0.0f : __expf(running_m - next_m);
        const float tile_scale = __expf(tile_m - next_m);
        const float tile_l_scaled =
            (tile & 1) == 0
                ? stage_probability_row(p_sA0, p_sSFA0, smem_logits0,
                                        owned_row, tile_m, tile_scale)
                : stage_probability_row(p_sA1, p_sSFA1, smem_logits1,
                                        owned_row, tile_m, tile_scale);
        running_l = running_l * old_scale + tile_l_scaled;
        running_m = next_m;
        storage.old_scale_stage[tile & 1][owned_row] = old_scale;
        if (tile == num_kv_tiles - 1) {
          storage.global_m[owned_row] = running_m;
          storage.global_l[owned_row] = running_l;
        }
      }
      cutlass::arch::NamedBarrier::sync(
          kSm120Nvfp4FmhaSoftmaxGroupThreadCount,
          online_barrier_id);
      cutlass::arch::fence_view_async_shared();
      release_p_ready();
    }
  } else if (is_epilogue) {
    pipeline_corr_epi.consumer_wait(pipeline_corr_epi_consumer_state);
    sm120_epilogue_store_bf16_tile(
        smem_epilogue_o, out_tile, out_stride_cols, epilogue_thread_idx);
    pipeline_corr_epi.consumer_release(pipeline_corr_epi_consumer_state);
    ++pipeline_corr_epi_consumer_state;
  }
#else
  if (threadIdx.x == 0) {
    out_group[0] = __float2bfloat16(-1.0f);
  }
#endif
}

__global__ void qk_cutlass_smem_atom_tile_kernel(const uint8_t* q_packed,
                                                 const uint8_t* q_scales,
                                                 const uint8_t* k_packed,
                                                 const uint8_t* k_scales,
                                                 float* out_tile,
                                                 int data_debug_mode,
                                                 int scale_debug_mode) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  using TensorStorage = typename CutlassCollectiveMainloop::TensorStorage;
  extern __shared__ __align__(128) char smem[];
  auto& storage = *reinterpret_cast<TensorStorage*>(smem);
  cutlass_smem_atom_gemm_tile_body(storage, q_packed, q_scales, k_packed,
                                   k_scales, out_tile, 0, 0, kPackedHeadDim,
                                   kScaleCols, 0, 0, kPackedHeadDim, kScaleCols,
                                   kHeadDim / kCutlassTileK, kCutlassTileN, 0,
                                   0, data_debug_mode, scale_debug_mode);
#else
  if (threadIdx.x == 0) {
    out_tile[0] = -1.0f;
  }
#endif
}

__global__ void qk_cutlass_smem_atom_block_kernel(const uint8_t* q_packed,
                                                  const uint8_t* q_scales,
                                                  const uint8_t* k_packed,
                                                  const uint8_t* k_scales,
                                                  float* out_scores) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  const int kv_row_base = int(blockIdx.x) * kCutlassTileN;
  const int q_row_base = int(blockIdx.y) * kCutlassTileM;
  if (kv_row_base >= kKvLen || q_row_base >= kQRows) {
    return;
  }
  using TensorStorage = typename CutlassCollectiveMainloop::TensorStorage;
  extern __shared__ __align__(128) char smem[];
  auto& storage = *reinterpret_cast<TensorStorage*>(smem);
  cutlass_smem_atom_gemm_tile_body(storage, q_packed, q_scales, k_packed,
                                   k_scales, out_scores, q_row_base, 0,
                                   kPackedHeadDim, kScaleCols, kv_row_base, 0,
                                   kPackedHeadDim, kScaleCols,
                                   kHeadDim / kCutlassTileK, kKvLen, q_row_base,
                                   kv_row_base, 0, 0);
#else
  if (threadIdx.x == 0) {
    out_scores[0] = -1.0f;
  }
#endif
}

__global__ void pv_cutlass_smem_atom_tile_kernel(const uint8_t* p_packed,
                                                 const uint8_t* p_scales,
                                                 const uint8_t* v_pv_packed,
                                                 const uint8_t* v_pv_scales,
                                                 float* out_tile,
                                                 int kv_base,
                                                 int out_col_base) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  using TensorStorage = typename CutlassCollectiveMainloop::TensorStorage;
  extern __shared__ __align__(128) char smem[];
  auto& storage = *reinterpret_cast<TensorStorage*>(smem);
  cutlass_smem_atom_gemm_tile_body(storage, p_packed, p_scales, v_pv_packed,
                                   v_pv_scales, out_tile, 0, kv_base,
                                   kProbPackedCols, kProbScaleCols,
                                   out_col_base, kv_base, kProbPackedCols,
                                   kProbScaleCols, 1, kCutlassTileN, 0, 0, 0,
                                   0);
#else
  if (threadIdx.x == 0) {
    out_tile[0] = -1.0f;
  }
#endif
}
__global__ __launch_bounds__(kSm120Nvfp4FmhaThreadCount, 1)
void sm120_nvfp4_role_schedule_smoke_kernel(int32_t* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  const int thread_idx = int(threadIdx.x);
  const int warp_idx = thread_idx / cutlass::NumThreadsPerWarp;
  const int lane_idx = thread_idx % cutlass::NumThreadsPerWarp;
  const Sm120Nvfp4FmhaRole role = sm120_nvfp4_fmha_role_for_warp(warp_idx);

  for (int i = thread_idx; i < 16; i += blockDim.x) {
    out[i] = 0;
  }
  __syncthreads();

  if (lane_idx == 0) {
    atomicAdd(&out[static_cast<int>(role)], 1);
  }
  if (role == Sm120Nvfp4FmhaRole::Mma) {
    const int mma_thread_idx = sm120_nvfp4_fmha_mma_thread_idx(thread_idx);
    if (mma_thread_idx >= 0 && mma_thread_idx < 256) {
      atomicAdd(&out[8], 1);
    }
  }
  if (thread_idx == 0) {
    out[7] = kSm120Nvfp4FmhaNumWarps;
    out[9] = kSm120Nvfp4FmhaThreadCount;
    out[10] = kSm120Nvfp4FmhaWarpMmaBegin;
    out[11] = kSm120Nvfp4FmhaWarpLoad;
    out[12] = kSm120Nvfp4FmhaWarpEpilogue;
  }
#else
  if (threadIdx.x == 0) {
    out[0] = -1;
  }
#endif
}

void check_tensor(const torch::Tensor& t, const char* name, c10::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has unexpected dtype");
}

void sm120_nvfp4_role_schedule_smoke(torch::Tensor out) {
  check_tensor(out, "out", torch::kInt32);
  TORCH_CHECK(out.numel() >= 16, "out must have at least 16 int32 elements");
  auto kernel = sm120_nvfp4_role_schedule_smoke_kernel;
  kernel<<<1, kSm120Nvfp4FmhaThreadCount, 0,
           at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void quantize_q_rowmajor(torch::Tensor q,
                         torch::Tensor q_packed,
                         torch::Tensor q_scales) {
  check_tensor(q, "q", torch::kBFloat16);
  check_tensor(q_packed, "q_packed", torch::kUInt8);
  check_tensor(q_scales, "q_scales", torch::kUInt8);
  TORCH_CHECK(q.sizes() == torch::IntArrayRef({kQLen, kGroup, kHeadDim}),
              "q must have shape [512, 8, 512]");
  TORCH_CHECK(q_packed.sizes() == torch::IntArrayRef({kQRows, kPackedHeadDim}),
              "q_packed must have shape [4096, 256]");
  TORCH_CHECK(q_scales.sizes() == torch::IntArrayRef({kQRows, kScaleCols}),
              "q_scales must have shape [4096, 32]");
  quantize_q_rowmajor_kernel<<<kQRows, kScaleCols, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      q_packed.data_ptr<uint8_t>(),
      q_scales.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qk_cutlass_smem_atom_tile(torch::Tensor q_packed,
                               torch::Tensor q_scales,
                               torch::Tensor k_packed,
                               torch::Tensor k_scales,
                               torch::Tensor out_tile,
                               int64_t data_debug_mode,
                               int64_t scale_debug_mode) {
  check_tensor(q_packed, "q_packed", torch::kUInt8);
  check_tensor(q_scales, "q_scales", torch::kUInt8);
  check_tensor(k_packed, "k_packed", torch::kUInt8);
  check_tensor(k_scales, "k_scales", torch::kUInt8);
  check_tensor(out_tile, "out_tile", torch::kFloat32);
  TORCH_CHECK(q_packed.sizes() == torch::IntArrayRef({kQRows, kPackedHeadDim}),
              "q_packed must have shape [4096, 256]");
  TORCH_CHECK(q_scales.sizes() == torch::IntArrayRef({kQRows, kScaleCols}),
              "q_scales must have shape [4096, 32]");
  TORCH_CHECK(k_packed.sizes() == torch::IntArrayRef({kKvLen, kPackedHeadDim}),
              "k_packed must have shape [32768, 256]");
  TORCH_CHECK(k_scales.sizes() == torch::IntArrayRef({kKvLen, kScaleCols}),
              "k_scales must have shape [32768, 32]");
  TORCH_CHECK(out_tile.sizes() ==
                  torch::IntArrayRef({kCutlassTileM, kCutlassTileN}),
              "out_tile must have shape [128, 128]");

  auto kernel = qk_cutlass_smem_atom_tile_kernel;
  const size_t smem_size =
      sizeof(typename CutlassCollectiveMainloop::TensorStorage);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size)));
  kernel<<<1, CutlassCollectiveMainloop::ThreadCount, smem_size,
           at::cuda::getCurrentCUDAStream()>>>(
      q_packed.data_ptr<uint8_t>(),
      q_scales.data_ptr<uint8_t>(),
      k_packed.data_ptr<uint8_t>(),
      k_scales.data_ptr<uint8_t>(),
      out_tile.data_ptr<float>(),
      static_cast<int>(data_debug_mode),
	      static_cast<int>(scale_debug_mode));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qk_cutlass_smem_atom_block(torch::Tensor q_packed,
                                torch::Tensor q_scales,
                                torch::Tensor k_packed,
                                torch::Tensor k_scales,
                                torch::Tensor out_scores) {
  check_tensor(q_packed, "q_packed", torch::kUInt8);
  check_tensor(q_scales, "q_scales", torch::kUInt8);
  check_tensor(k_packed, "k_packed", torch::kUInt8);
  check_tensor(k_scales, "k_scales", torch::kUInt8);
  check_tensor(out_scores, "out_scores", torch::kFloat32);
  TORCH_CHECK(q_packed.sizes() == torch::IntArrayRef({kQRows, kPackedHeadDim}),
              "q_packed must have shape [4096, 256]");
  TORCH_CHECK(q_scales.sizes() == torch::IntArrayRef({kQRows, kScaleCols}),
              "q_scales must have shape [4096, 32]");
  TORCH_CHECK(k_packed.sizes() == torch::IntArrayRef({kKvLen, kPackedHeadDim}),
              "k_packed must have shape [32768, 256]");
  TORCH_CHECK(k_scales.sizes() == torch::IntArrayRef({kKvLen, kScaleCols}),
              "k_scales must have shape [32768, 32]");
  TORCH_CHECK(out_scores.sizes() == torch::IntArrayRef({kBenchRows, kKvLen}),
              "out_scores must have shape [128, 32768]");

  auto kernel = qk_cutlass_smem_atom_block_kernel;
  const size_t smem_size =
      sizeof(typename CutlassCollectiveMainloop::TensorStorage);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size)));
  dim3 grid(kKvLen / kCutlassTileN, kBenchRows / kCutlassTileM, 1);
  kernel<<<grid, CutlassCollectiveMainloop::ThreadCount, smem_size,
           at::cuda::getCurrentCUDAStream()>>>(
      q_packed.data_ptr<uint8_t>(),
      q_scales.data_ptr<uint8_t>(),
      k_packed.data_ptr<uint8_t>(),
      k_scales.data_ptr<uint8_t>(),
      out_scores.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pv_cutlass_smem_atom_tile(torch::Tensor p_packed,
                               torch::Tensor p_scales,
                               torch::Tensor v_pv_packed,
                               torch::Tensor v_pv_scales,
                               torch::Tensor out_tile,
                               int64_t kv_base,
                               int64_t out_col_base) {
  check_tensor(p_packed, "p_packed", torch::kUInt8);
  check_tensor(p_scales, "p_scales", torch::kUInt8);
  check_tensor(v_pv_packed, "v_pv_packed", torch::kUInt8);
  check_tensor(v_pv_scales, "v_pv_scales", torch::kUInt8);
  check_tensor(out_tile, "out_tile", torch::kFloat32);
  TORCH_CHECK(p_packed.sizes() == torch::IntArrayRef({kBenchRows, kProbPackedCols}),
              "p_packed must have shape [128, 16384]");
  TORCH_CHECK(p_scales.sizes() == torch::IntArrayRef({kBenchRows, kProbScaleCols}),
              "p_scales must have shape [128, 2048]");
  TORCH_CHECK(v_pv_packed.sizes() == torch::IntArrayRef({kHeadDim, kProbPackedCols}),
              "v_pv_packed must have shape [512, 16384]");
  TORCH_CHECK(v_pv_scales.sizes() == torch::IntArrayRef({kHeadDim, kProbScaleCols}),
              "v_pv_scales must have shape [512, 2048]");
  TORCH_CHECK(out_tile.sizes() ==
                  torch::IntArrayRef({kCutlassTileM, kCutlassTileN}),
              "out_tile must have shape [128, 128]");
  TORCH_CHECK(kv_base >= 0 && kv_base + kCutlassTileK <= kKvLen,
              "kv_base out of range");
  TORCH_CHECK(out_col_base >= 0 && out_col_base + kCutlassTileN <= kHeadDim,
              "out_col_base out of range");
  TORCH_CHECK(kv_base % kCutlassTileK == 0,
              "kv_base must be a multiple of 256");
  TORCH_CHECK(out_col_base % kCutlassTileN == 0,
              "out_col_base must be a multiple of 128");

  auto kernel = pv_cutlass_smem_atom_tile_kernel;
  const size_t smem_size =
      sizeof(typename CutlassCollectiveMainloop::TensorStorage);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size)));
  kernel<<<1, CutlassCollectiveMainloop::ThreadCount, smem_size,
           at::cuda::getCurrentCUDAStream()>>>(
      p_packed.data_ptr<uint8_t>(),
      p_scales.data_ptr<uint8_t>(),
      v_pv_packed.data_ptr<uint8_t>(),
      v_pv_scales.data_ptr<uint8_t>(),
      out_tile.data_ptr<float>(),
      static_cast<int>(kv_base),
      static_cast<int>(out_col_base));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm120_nvfp4_qkv_online_register_q_stage(torch::Tensor q_packed,
                                             torch::Tensor q_scales,
                                             torch::Tensor k_packed,
                                             torch::Tensor k_scales,
                                             torch::Tensor v_pv_packed,
                                             torch::Tensor v_pv_scales,
                                             torch::Tensor out_group,
                                             torch::Tensor workspace,
                                             double qk_alpha,
                                             double pv_alpha,
                                             int64_t q_tile,
                                             int64_t kv_tile_start,
                                             int64_t num_kv_tiles,
                                             int64_t out_group_idx) {
  check_tensor(q_packed, "q_packed", torch::kUInt8);
  check_tensor(q_scales, "q_scales", torch::kUInt8);
  check_tensor(k_packed, "k_packed", torch::kUInt8);
  check_tensor(k_scales, "k_scales", torch::kUInt8);
  check_tensor(v_pv_packed, "v_pv_packed", torch::kUInt8);
  check_tensor(v_pv_scales, "v_pv_scales", torch::kUInt8);
  check_tensor(out_group, "out_group", torch::kBFloat16);
  check_tensor(workspace, "workspace", torch::kUInt8);
  TORCH_CHECK(q_packed.sizes() == torch::IntArrayRef({kQRows, kPackedHeadDim}),
              "q_packed must have shape [4096, 256]");
  TORCH_CHECK(q_scales.sizes() == torch::IntArrayRef({kQRows, kScaleCols}),
              "q_scales must have shape [4096, 32]");
  TORCH_CHECK(k_packed.sizes() == torch::IntArrayRef({kKvLen, kPackedHeadDim}),
              "k_packed must have shape [32768, 256]");
  TORCH_CHECK(k_scales.sizes() == torch::IntArrayRef({kKvLen, kScaleCols}),
              "k_scales must have shape [32768, 32]");
  TORCH_CHECK(v_pv_packed.sizes() ==
                  torch::IntArrayRef({kHeadDim, kProbPackedCols}),
              "v_pv_packed must have shape [512, 16384]");
  TORCH_CHECK(v_pv_scales.sizes() ==
                  torch::IntArrayRef({kHeadDim, kProbScaleCols}),
              "v_pv_scales must have shape [512, 2048]");
  TORCH_CHECK(out_group.sizes() ==
                  torch::IntArrayRef({kCutlassTileM, kCutlassTileN}),
              "out_group must have shape [128, 128]");
  TORCH_CHECK(q_tile >= 0 && q_tile < kQRows / kCutlassTileM,
              "q_tile out of range");
  TORCH_CHECK(kv_tile_start >= 0 &&
                  kv_tile_start < kKvLen / kCutlassTileN,
              "kv_tile_start out of range");
  TORCH_CHECK(num_kv_tiles > 0 &&
                  kv_tile_start + num_kv_tiles <= kKvLen / kCutlassTileN,
              "num_kv_tiles out of range");
  TORCH_CHECK(out_group_idx >= 0 && out_group_idx < kHeadDim / kCutlassTileN,
              "out_group_idx out of range");

  float alpha = 1.0f;
  auto qk_args = flashinfer::gemm::prepareGemmArgsImpl<CutlassGemm>(
      nullptr,
      q_packed.data_ptr<uint8_t>(),
      k_packed.data_ptr<uint8_t>(),
      q_scales.data_ptr<uint8_t>(),
      k_scales.data_ptr<uint8_t>(),
      &alpha,
      kQRows,
      kKvLen,
      kHeadDim,
      1);
  CutlassGemm qk_gemm;
  const size_t qk_workspace_size = qk_gemm.get_workspace_size(qk_args);
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(qk_workspace_size),
              "workspace too small for QK: need ", qk_workspace_size,
              " bytes, got ", workspace.numel());
  auto qk_status = qk_gemm.initialize(
      qk_args,
      reinterpret_cast<char*>(workspace.data_ptr<uint8_t>()),
      at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(qk_status == cutlass::Status::kSuccess,
              "failed to initialize CUTLASS QK GEMM params");
  auto qk_params = qk_gemm.params();

  auto pv_args = flashinfer::gemm::prepareGemmArgsImpl<CutlassGemmK128Stage2>(
      nullptr,
      q_packed.data_ptr<uint8_t>(),
      v_pv_packed.data_ptr<uint8_t>(),
      q_scales.data_ptr<uint8_t>(),
      v_pv_scales.data_ptr<uint8_t>(),
      &alpha,
      kCutlassTileM,
      kHeadDim,
      kKvLen,
      1);
  CutlassGemmK128Stage2 pv_gemm;
  const size_t pv_workspace_size = pv_gemm.get_workspace_size(pv_args);
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(pv_workspace_size),
              "workspace too small for PV: need ", pv_workspace_size,
              " bytes, got ", workspace.numel());
  auto pv_status = pv_gemm.initialize(
      pv_args,
      reinterpret_cast<char*>(workspace.data_ptr<uint8_t>()),
      at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(pv_status == cutlass::Status::kSuccess,
              "failed to initialize CUTLASS stage-2 PV GEMM params");
  auto pv_params = pv_gemm.params();

  constexpr int kSmemBytes =
      static_cast<int>(sizeof(Sm120Nvfp4QkvLoadCollectiveStorage));
  auto kernel = sm120_nvfp4_qkv_online_register_q_stage_kernel;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  kernel<<<dim3(1, 1, 1), kSm120Nvfp4FmhaThreadCount, kSmemBytes,
           at::cuda::getCurrentCUDAStream()>>>(
      qk_params, pv_params,
      reinterpret_cast<__nv_bfloat16*>(out_group.data_ptr<at::BFloat16>()),
      static_cast<float>(qk_alpha), static_cast<float>(pv_alpha),
      static_cast<int>(q_tile), static_cast<int>(kv_tile_start),
      static_cast<int>(num_kv_tiles), static_cast<int>(out_group_idx),
      kCutlassTileN);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sm120_nvfp4_qkv_online_register_q_full_grid(torch::Tensor q_packed,
                                                 torch::Tensor q_scales,
                                                 torch::Tensor k_packed,
                                                 torch::Tensor k_scales,
                                                 torch::Tensor v_pv_packed,
                                                 torch::Tensor v_pv_scales,
                                                 torch::Tensor out,
                                                 torch::Tensor workspace,
                                                 double qk_alpha,
                                                 double pv_alpha) {
  check_tensor(q_packed, "q_packed", torch::kUInt8);
  check_tensor(q_scales, "q_scales", torch::kUInt8);
  check_tensor(k_packed, "k_packed", torch::kUInt8);
  check_tensor(k_scales, "k_scales", torch::kUInt8);
  check_tensor(v_pv_packed, "v_pv_packed", torch::kUInt8);
  check_tensor(v_pv_scales, "v_pv_scales", torch::kUInt8);
  check_tensor(out, "out", torch::kBFloat16);
  check_tensor(workspace, "workspace", torch::kUInt8);
  TORCH_CHECK(q_packed.sizes() == torch::IntArrayRef({kQRows, kPackedHeadDim}),
              "q_packed must have shape [4096, 256]");
  TORCH_CHECK(q_scales.sizes() == torch::IntArrayRef({kQRows, kScaleCols}),
              "q_scales must have shape [4096, 32]");
  TORCH_CHECK(k_packed.sizes() == torch::IntArrayRef({kKvLen, kPackedHeadDim}),
              "k_packed must have shape [32768, 256]");
  TORCH_CHECK(k_scales.sizes() == torch::IntArrayRef({kKvLen, kScaleCols}),
              "k_scales must have shape [32768, 32]");
  TORCH_CHECK(v_pv_packed.sizes() ==
                  torch::IntArrayRef({kHeadDim, kProbPackedCols}),
              "v_pv_packed must have shape [512, 16384]");
  TORCH_CHECK(v_pv_scales.sizes() ==
                  torch::IntArrayRef({kHeadDim, kProbScaleCols}),
              "v_pv_scales must have shape [512, 2048]");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({kQRows, kHeadDim}),
              "out must have shape [4096, 512]");

  float alpha = 1.0f;
  auto qk_args = flashinfer::gemm::prepareGemmArgsImpl<CutlassGemm>(
      nullptr,
      q_packed.data_ptr<uint8_t>(),
      k_packed.data_ptr<uint8_t>(),
      q_scales.data_ptr<uint8_t>(),
      k_scales.data_ptr<uint8_t>(),
      &alpha,
      kQRows,
      kKvLen,
      kHeadDim,
      1);
  CutlassGemm qk_gemm;
  const size_t qk_workspace_size = qk_gemm.get_workspace_size(qk_args);
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(qk_workspace_size),
              "workspace too small for QK: need ", qk_workspace_size,
              " bytes, got ", workspace.numel());
  auto qk_status = qk_gemm.initialize(
      qk_args,
      reinterpret_cast<char*>(workspace.data_ptr<uint8_t>()),
      at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(qk_status == cutlass::Status::kSuccess,
              "failed to initialize CUTLASS QK GEMM params");
  auto qk_params = qk_gemm.params();

  auto pv_args = flashinfer::gemm::prepareGemmArgsImpl<CutlassGemmK128Stage2>(
      nullptr,
      q_packed.data_ptr<uint8_t>(),
      v_pv_packed.data_ptr<uint8_t>(),
      q_scales.data_ptr<uint8_t>(),
      v_pv_scales.data_ptr<uint8_t>(),
      &alpha,
      kCutlassTileM,
      kHeadDim,
      kKvLen,
      1);
  CutlassGemmK128Stage2 pv_gemm;
  const size_t pv_workspace_size = pv_gemm.get_workspace_size(pv_args);
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(pv_workspace_size),
              "workspace too small for PV: need ", pv_workspace_size,
              " bytes, got ", workspace.numel());
  auto pv_status = pv_gemm.initialize(
      pv_args,
      reinterpret_cast<char*>(workspace.data_ptr<uint8_t>()),
      at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(pv_status == cutlass::Status::kSuccess,
              "failed to initialize CUTLASS stage-2 PV GEMM params");
  auto pv_params = pv_gemm.params();

  constexpr int kSmemBytes =
      static_cast<int>(sizeof(Sm120Nvfp4QkvLoadCollectiveStorage));
  auto kernel = sm120_nvfp4_qkv_online_register_q_stage_kernel;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  kernel<<<dim3(kQRows / kCutlassTileM, kHeadDim / kCutlassTileN, 1),
           kSm120Nvfp4FmhaThreadCount, kSmemBytes,
           at::cuda::getCurrentCUDAStream()>>>(
      qk_params, pv_params,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<float>(qk_alpha), static_cast<float>(pv_alpha), 0, 0,
      kKvLen / kCutlassTileN, 0, kHeadDim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

RunnerConfig runner_config_from_tactic(int64_t tactic) {
  static std::vector<RunnerConfig> configs =
      flashinfer::gemm::CutlassFp4GemmRunner<
          __nv_bfloat16, RunnerFp4Type::W4A4_NVFP4_NVFP4>{}
          .getConfigs();
  TORCH_CHECK(tactic >= 0 && tactic < static_cast<int64_t>(configs.size()),
              "tactic must be in [0, ", configs.size(), ")");
  return configs[tactic];
}

void cutlass_runner_fp4_gemm(torch::Tensor a_packed,
                             torch::Tensor b_packed_t,
                             torch::Tensor a_scales,
                             torch::Tensor b_scales_t,
                             torch::Tensor alpha,
                             torch::Tensor out,
                             torch::Tensor workspace,
                             int64_t tactic) {
  check_tensor(a_packed, "a_packed", torch::kUInt8);
  check_tensor(b_packed_t, "b_packed_t", torch::kUInt8);
  check_tensor(a_scales, "a_scales", torch::kUInt8);
  check_tensor(b_scales_t, "b_scales_t", torch::kUInt8);
  check_tensor(alpha, "alpha", torch::kFloat32);
  check_tensor(out, "out", torch::kBFloat16);
  check_tensor(workspace, "workspace", torch::kUInt8);
  TORCH_CHECK(a_packed.dim() == 2, "a_packed must be 2D");
  TORCH_CHECK(b_packed_t.dim() == 2, "b_packed_t must be 2D");
  TORCH_CHECK(alpha.numel() == 1, "alpha must contain one float");
  const int m = static_cast<int>(a_packed.size(0));
  const int k_packed = static_cast<int>(a_packed.size(1));
  const int n = static_cast<int>(b_packed_t.size(0));
  TORCH_CHECK(b_packed_t.size(1) == k_packed,
              "b_packed_t.size(1) must match a_packed.size(1)");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({m, n}),
              "out must have shape [a_rows, b_rows]");
  const int k = k_packed * 2;
  auto config = runner_config_from_tactic(tactic);
  flashinfer::gemm::CutlassFp4GemmRunner<
      __nv_bfloat16, RunnerFp4Type::W4A4_NVFP4_NVFP4>
      runner;
  const int64_t required_workspace =
      static_cast<int64_t>(runner.getWorkspaceSize(m, n, k, 1));
  TORCH_CHECK(workspace.numel() >= required_workspace,
              "workspace too small: need ", required_workspace, " bytes, got ",
              workspace.numel());
  runner.gemm(out.data_ptr(), a_packed.data_ptr(), b_packed_t.data_ptr(),
              a_scales.data_ptr(), b_scales_t.data_ptr(),
              alpha.data_ptr<float>(), m, n, k, 1, config,
              reinterpret_cast<char*>(workspace.data_ptr<uint8_t>()),
              static_cast<size_t>(workspace.numel()),
              at::cuda::getCurrentCUDAStream());
}

template <typename Traits>
pybind11::dict cutlass_tile_shape_metadata(const char* name,
                                           int tile_m,
                                           int tile_n,
                                           int tile_k) {
  using Mainloop = typename Traits::CollectiveMainloop;
  using Epilogue = typename Traits::CollectiveEpilogue;
  using Kernel = typename Traits::GemmKernel;
  constexpr int64_t kSm120OptinSmemBytes = 99ll << 10;
  const int64_t tensor_storage =
      static_cast<int64_t>(sizeof(typename Mainloop::TensorStorage));
  const int64_t score_scratch =
      static_cast<int64_t>(tile_m) * tile_n * sizeof(float);
  const int64_t p_packed =
      static_cast<int64_t>(tile_m) * (tile_k / 2);
  const int64_t p_scales =
      static_cast<int64_t>(tile_m) * (tile_k / 16);
  const int64_t row_state =
      5 * static_cast<int64_t>(tile_m) * sizeof(float);
  const int64_t scaffold_storage =
      std::max(tensor_storage, score_scratch) + p_packed + p_scales +
      row_state;

  pybind11::dict d;
  d["name"] = name;
  d["tile_m"] = tile_m;
  d["tile_n"] = tile_n;
  d["tile_k"] = tile_k;
  d["thread_count"] = Mainloop::ThreadCount;
  d["scale_vec_size"] = Mainloop::TiledMma::SFVecSize;
  d["gemm_kernel_block_threads"] =
      static_cast<int64_t>(Kernel::get_block_shape().x);
  d["gemm_kernel_shared_storage_bytes"] =
      static_cast<int64_t>(Kernel::SharedStorageSize);
  d["mainloop_tensor_storage_bytes"] = tensor_storage;
  d["mainloop_shared_storage_bytes"] =
      static_cast<int64_t>(sizeof(typename Mainloop::SharedStorage));
  d["epilogue_shared_storage_bytes"] =
      static_cast<int64_t>(sizeof(typename Epilogue::SharedStorage));
  d["score_scratch_bytes"] = score_scratch;
  d["score_fits_in_tensor_storage"] = tensor_storage >= score_scratch;
  d["p_packed_bytes"] = p_packed;
  d["p_scales_bytes"] = p_scales;
  d["row_state_bytes"] = row_state;
  d["scaffold_storage_min_bytes"] = scaffold_storage;
  d["sm120_optin_smem_bytes"] = kSm120OptinSmemBytes;
  d["scaffold_storage_margin_bytes"] =
      kSm120OptinSmemBytes - scaffold_storage;
  return d;
}

pybind11::dict cutlass_sm120_blockscaled_collective_metadata() {
  pybind11::dict d;
  d["arch"] = "sm120";
  d["operator_class"] = "OpClassBlockScaledTensorOp";
  d["tile_m"] = 128;
  d["tile_n"] = 128;
  d["tile_k"] = 256;
  d["scale_vec_size"] = CutlassCollectiveMainloop::TiledMma::SFVecSize;
  d["thread_count"] = CutlassCollectiveMainloop::ThreadCount;
  d["gemm_kernel_block_threads"] =
      static_cast<int64_t>(CutlassGemmKernel::get_block_shape().x);
  d["gemm_kernel_shared_storage_bytes"] =
      static_cast<int64_t>(CutlassGemmKernel::SharedStorageSize);
  d["mainloop_tensor_storage_bytes"] =
      static_cast<int64_t>(sizeof(typename CutlassCollectiveMainloop::TensorStorage));
  d["mainloop_shared_storage_bytes"] =
      static_cast<int64_t>(sizeof(typename CutlassCollectiveMainloop::SharedStorage));
  d["epilogue_shared_storage_bytes"] =
      static_cast<int64_t>(sizeof(typename CutlassCollectiveEpilogue::SharedStorage));
  d["layout_sfa_bytes"] =
      static_cast<int64_t>(sizeof(typename CutlassCollectiveMainloop::LayoutSFA));
  d["layout_sfb_bytes"] =
      static_cast<int64_t>(sizeof(typename CutlassCollectiveMainloop::LayoutSFB));
  constexpr int64_t kSm120OptinSmemBytes = 99ll << 10;
  const int64_t qk_shared_bytes =
      static_cast<int64_t>(sizeof(typename CutlassCollectiveMainloop::SharedStorage));  d["sm120_optin_smem_bytes"] = kSm120OptinSmemBytes;  d["sm120_qk_load_collective_storage_bytes"] =
      static_cast<int64_t>(sizeof(Sm120Nvfp4QkLoadCollectiveStorage));
  d["sm120_qk_load_collective_storage_margin_bytes"] =
      kSm120OptinSmemBytes -
      static_cast<int64_t>(sizeof(Sm120Nvfp4QkLoadCollectiveStorage));
  d["pv_k128_stage2_tensor_storage_bytes"] =
      static_cast<int64_t>(
          sizeof(typename CutlassCollectiveMainloopK128Stage2::TensorStorage));
  d["pv_k128_stage2_shared_storage_bytes"] =
      static_cast<int64_t>(
          sizeof(typename CutlassCollectiveMainloopK128Stage2::SharedStorage));
  d["pv_k128_stage2_pipeline_storage_bytes"] =
      static_cast<int64_t>(
          sizeof(typename CutlassCollectiveMainloopK128Stage2::PipelineStorage));
  d["sm120_role_pipeline_storage_bytes"] =
      static_cast<int64_t>(sizeof(Sm120Nvfp4MainloopPipelineStorage));
  d["sm120_pipeline_s_storage_bytes"] =
      static_cast<int64_t>(sizeof(typename Sm120Nvfp4PipelineS::SharedStorage));
  d["sm120_pipeline_e_storage_bytes"] =
      static_cast<int64_t>(sizeof(typename Sm120Nvfp4PipelineE::SharedStorage));
  d["sm120_order_s01_storage_bytes"] =
      static_cast<int64_t>(
          sizeof(typename Sm120Nvfp4OrderBarrierSoftmax::SharedStorage));
  d["sm120_qkv_load_collective_storage_bytes"] =
      static_cast<int64_t>(sizeof(Sm120Nvfp4QkvLoadCollectiveStorage));
  d["sm120_qkv_load_collective_storage_margin_bytes"] =
      kSm120OptinSmemBytes -
      static_cast<int64_t>(sizeof(Sm120Nvfp4QkvLoadCollectiveStorage));
  d["bf16_logits_128x128_bytes"] =
      static_cast<int64_t>(kCutlassTileM) * kCutlassTileN *
      static_cast<int64_t>(sizeof(__nv_bfloat16));
  d["pv_p_stage_bytes"] = static_cast<int64_t>(kSm120Nvfp4PvPStageBytes);
  d["pv_p_scale_stage_elems"] =
      static_cast<int64_t>(kSm120Nvfp4PvScaleStageElems);
  d["qk_smem_a_bytes"] = static_cast<int64_t>(sizeof(decltype(
      ((typename CutlassCollectiveMainloop::TensorStorage*)nullptr)->smem_A)));
  d["qk_smem_sfa_bytes"] = static_cast<int64_t>(sizeof(decltype(
      ((typename CutlassCollectiveMainloop::TensorStorage*)nullptr)->smem_SFA)));
  d["bf16_logits_64x128_bytes"] =
      64ll * kCutlassTileN * static_cast<int64_t>(sizeof(__nv_bfloat16));
  d["independent_double_buffered_logits_128x128_margin_bytes"] =
      kSm120OptinSmemBytes -
      static_cast<int64_t>(sizeof(Sm120Nvfp4QkvLoadCollectiveStorage)) -
      2ll * static_cast<int64_t>(kCutlassTileM) * kCutlassTileN *
          static_cast<int64_t>(sizeof(__nv_bfloat16));
  d["independent_double_buffered_logits_64x128_margin_bytes"] =
      kSm120OptinSmemBytes -
      static_cast<int64_t>(sizeof(Sm120Nvfp4QkvLoadCollectiveStorage)) -
      2ll * 64ll * kCutlassTileN * static_cast<int64_t>(sizeof(__nv_bfloat16));
  d["pv_stage2_thread_count"] =
      static_cast<int64_t>(CutlassCollectiveMainloopK128Stage2::ThreadCount);
  pybind11::dict role_schedule;
  role_schedule["total_warps"] =
      static_cast<int64_t>(kSm120Nvfp4FmhaNumWarps);
  role_schedule["total_threads"] =
      static_cast<int64_t>(kSm120Nvfp4FmhaThreadCount);
  role_schedule["softmax0_warps"] =
      static_cast<int64_t>(kSm120Nvfp4FmhaNumWarpsSoftmax0);
  role_schedule["softmax1_warps"] =
      static_cast<int64_t>(kSm120Nvfp4FmhaNumWarpsSoftmax1);
  role_schedule["correction_warps"] =
      static_cast<int64_t>(kSm120Nvfp4FmhaNumWarpsCorrection);
  role_schedule["mma_warps"] =
      static_cast<int64_t>(kSm120Nvfp4FmhaNumWarpsMma);
  role_schedule["mma_threads"] =
      static_cast<int64_t>(kSm120Nvfp4FmhaNumWarpsMma *
                           cutlass::NumThreadsPerWarp);
  role_schedule["load_warps"] =
      static_cast<int64_t>(kSm120Nvfp4FmhaNumWarpsLoad);
  role_schedule["epilogue_warps"] =
      static_cast<int64_t>(kSm120Nvfp4FmhaNumWarpsEpilogue);
  role_schedule["mma_warp_begin"] =
      static_cast<int64_t>(kSm120Nvfp4FmhaWarpMmaBegin);
  role_schedule["load_warp"] =
      static_cast<int64_t>(kSm120Nvfp4FmhaWarpLoad);
  role_schedule["epilogue_warp"] =
      static_cast<int64_t>(kSm120Nvfp4FmhaWarpEpilogue);
  d["sm120_role_schedule"] = role_schedule;
  pybind11::dict tile_variants;
  tile_variants["128x128x128"] =
      cutlass_tile_shape_metadata<Sm120Fp4Tile128x128x128>(
          "128x128x128", 128, 128, 128);
  tile_variants["128x128x256"] =
      cutlass_tile_shape_metadata<Sm120Fp4Tile128x128x256>(
          "128x128x256", 128, 128, 256);
  tile_variants["256x128x128"] =
      cutlass_tile_shape_metadata<Sm120Fp4Tile256x128x128>(
          "256x128x128", 256, 128, 128);
  d["tile_variants"] = tile_variants;
  auto smem_layout_a = typename CutlassCollectiveMainloop::SmemLayoutA{};
  auto smem_layout_b = typename CutlassCollectiveMainloop::SmemLayoutB{};
  pybind11::list a_offsets;
  pybind11::list b_offsets;
  for (int k = 0; k < 16; ++k) {
    a_offsets.append(static_cast<int64_t>(smem_layout_a(0, k, cute::Int<0>{})));
    b_offsets.append(static_cast<int64_t>(smem_layout_b(0, k, cute::Int<0>{})));
  }
  d["smem_a_row0_k0_15_offsets"] = a_offsets;
  d["smem_b_row0_k0_15_offsets"] = b_offsets;
  pybind11::list a_rows;
  pybind11::list b_rows;
  for (int row = 0; row < 8; ++row) {
    pybind11::list a_row;
    pybind11::list b_row;
    for (int k = 0; k < 32; ++k) {
      a_row.append(static_cast<int64_t>(
          smem_layout_a(row, k, cute::Int<0>{})));
      b_row.append(static_cast<int64_t>(
          smem_layout_b(row, k, cute::Int<0>{})));
    }
    a_rows.append(a_row);
    b_rows.append(b_row);
  }
  d["smem_a_rows0_7_k0_31_offsets"] = a_rows;
  d["smem_b_rows0_7_k0_31_offsets"] = b_rows;
  int64_t a_split_pairs = 0;
  int64_t b_split_pairs = 0;
  pybind11::list first_split_pairs;
  for (int row = 0; row < kCutlassTileM; ++row) {
    for (int k = 0; k < kCutlassTileK; k += 2) {
      const int a0 = int(smem_layout_a(row, k, cute::Int<0>{}));
      const int a1 = int(smem_layout_a(row, k + 1, cute::Int<0>{}));
      const int b0 = int(smem_layout_b(row, k, cute::Int<0>{}));
      const int b1 = int(smem_layout_b(row, k + 1, cute::Int<0>{}));
      if ((a0 >> 1) != (a1 >> 1)) {
        ++a_split_pairs;
        if (pybind11::len(first_split_pairs) < 16) {
          pybind11::list entry;
          entry.append("A");
          entry.append(row);
          entry.append(k);
          entry.append(a0);
          entry.append(a1);
          first_split_pairs.append(entry);
        }
      }
      if ((b0 >> 1) != (b1 >> 1)) {
        ++b_split_pairs;
        if (pybind11::len(first_split_pairs) < 16) {
          pybind11::list entry;
          entry.append("B");
          entry.append(row);
          entry.append(k);
          entry.append(b0);
          entry.append(b1);
          first_split_pairs.append(entry);
        }
      }
    }
  }
  d["smem_a_adjacent_k_split_byte_pairs"] = a_split_pairs;
  d["smem_b_adjacent_k_split_byte_pairs"] = b_split_pairs;
  d["smem_first_split_byte_pairs"] = first_split_pairs;
  {
    auto tiled_mma = typename CutlassCollectiveMainloop::TiledMma{};
    auto cC = cute::make_identity_tensor(
        cute::take<0, 2>(CutlassThreadBlockShape{}));
    std::vector<int> c_counts(kCutlassTileM * kCutlassTileN, 0);
    pybind11::list first_c_coords;
    int64_t c_writes = 0;
    for (int thread_idx = 0;
         thread_idx < int(CutlassCollectiveMainloop::ThreadCount);
         ++thread_idx) {
      auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
      auto tCcC = thread_mma.partition_C(cC);
      for (int i = 0; i < int(cute::size(tCcC)); ++i) {
        auto coord = tCcC(i);
        const int row = int(cute::get<0>(coord));
        const int col = int(cute::get<1>(coord));
        if (row >= 0 && row < kCutlassTileM && col >= 0 &&
            col < kCutlassTileN) {
          ++c_counts[row * kCutlassTileN + col];
          ++c_writes;
          if (pybind11::len(first_c_coords) < 32) {
            pybind11::list entry;
            entry.append(thread_idx);
            entry.append(i);
            entry.append(row);
            entry.append(col);
            first_c_coords.append(entry);
          }
        }
      }
    }
    int64_t c_missing = 0;
    int64_t c_duplicate_slots = 0;
    int c_max_count = 0;
    for (int count : c_counts) {
      if (count == 0) {
        ++c_missing;
      }
      if (count > 1) {
        ++c_duplicate_slots;
      }
      c_max_count = std::max(c_max_count, count);
    }
    d["c_fragment_writes"] = c_writes;
    d["c_fragment_missing_slots"] = c_missing;
    d["c_fragment_duplicate_slots"] = c_duplicate_slots;
    d["c_fragment_max_count"] = c_max_count;
    d["c_fragment_first_coords"] = first_c_coords;
  }
  pybind11::list e2m1_values;
  for (int code = 0; code < 16; ++code) {
    e2m1_values.append(
        static_cast<float>(cutlass::float_e2m1_t::bitcast(
            static_cast<uint8_t>(code))));
  }
  d["cutlass_e2m1_code_values"] = e2m1_values;
  d["runner_tactic_count"] = static_cast<int64_t>(
      flashinfer::gemm::CutlassFp4GemmRunner<
          __nv_bfloat16, RunnerFp4Type::W4A4_NVFP4_NVFP4>{}
          .getConfigs()
          .size());
  return d;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_q_rowmajor", &quantize_q_rowmajor,
        "Fixed Shape B BF16 Q to row-major NVFP4 quantization");
  m.def("qk_cutlass_smem_atom_tile", &qk_cutlass_smem_atom_tile,
        "CUTLASS SM120 smem-layout/copy-atom QK 128x128 debug tile");
  m.def("qk_cutlass_smem_atom_block", &qk_cutlass_smem_atom_block,
        "CUTLASS SM120 smem-layout/copy-atom QK for 128x32768 block");
  m.def("pv_cutlass_smem_atom_tile", &pv_cutlass_smem_atom_tile,
        "CUTLASS SM120 smem-layout/copy-atom PV 128x128 debug tile");
  m.def("sm120_nvfp4_role_schedule_smoke",
        &sm120_nvfp4_role_schedule_smoke,
        "SM120 NVFP4 explicit role-schedule smoke for the fused FMHA mainloop");
  m.def("sm120_nvfp4_qkv_online_register_q_stage",
        &sm120_nvfp4_qkv_online_register_q_stage,
        "SM120 NVFP4 online-softmax register-Q smoke with compact Q/K/V storage");
  m.def("sm120_nvfp4_qkv_online_register_q_full_grid",
        &sm120_nvfp4_qkv_online_register_q_full_grid,
        "SM120 NVFP4 online-softmax register-Q full Shape-B tile grid");
  m.def("cutlass_runner_fp4_gemm", &cutlass_runner_fp4_gemm,
        "FlashInfer SM120 CUTLASS FP4 GEMM runner smoke hook");
  m.def("cutlass_sm120_blockscaled_collective_metadata",
        &cutlass_sm120_blockscaled_collective_metadata,
        "Compile-time metadata for the SM120 block-scaled CUTLASS collective");
}
