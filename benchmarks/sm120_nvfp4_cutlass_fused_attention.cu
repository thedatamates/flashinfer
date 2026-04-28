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
using CutlassGemmKernelK128 =
    typename Sm120Fp4Tile128x128x128::GemmKernel;
using CutlassGemmK128 =
    cutlass::gemm::device::GemmUniversalAdapter<CutlassGemmKernelK128>;
using CutlassCollectiveMainloopK128 =
    typename Sm120Fp4Tile128x128x128::CollectiveMainloop;

using RunnerConfig = flashinfer::gemm::CutlassGemmConfig;
using RunnerTileConfig = flashinfer::gemm::CutlassTileConfigSM120;
using RunnerMainloopSchedule = flashinfer::gemm::MainloopScheduleType;
using RunnerEpilogueSchedule = flashinfer::gemm::EpilogueScheduleType;
using RunnerClusterShape = flashinfer::gemm::ClusterShape;
using RunnerFp4Type = flashinfer::gemm::FP4GemmType;

struct CutlassFusedOneTileStorage {
  alignas(128) typename CutlassCollectiveMainloop::TensorStorage tensor;
  alignas(16) uint8_t p_packed[kCutlassTileM * (kCutlassTileK / 2)];
  alignas(16) uint8_t p_scales[kCutlassTileM * (kCutlassTileK / 16)];
  alignas(16) float row_m[kCutlassTileM];
  alignas(16) float row_l[kCutlassTileM];
  alignas(16) float tile_m[kCutlassTileM];
  alignas(16) float tile_l[kCutlassTileM];
  alignas(16) float row_alpha[kCutlassTileM];
};

struct CutlassFusedOneTileStorageK128 {
  alignas(128) typename CutlassCollectiveMainloopK128::TensorStorage tensor;
  alignas(16) uint8_t p_packed[kCutlassTileM * (kCutlassTileK128 / 2)];
  alignas(16) uint8_t p_scales[kCutlassTileM * (kCutlassTileK128 / 16)];
  alignas(16) float row_m[kCutlassTileM];
  alignas(16) float row_l[kCutlassTileM];
  alignas(16) float tile_m[kCutlassTileM];
  alignas(16) float tile_l[kCutlassTileM];
  alignas(16) float row_alpha[kCutlassTileM];
};

struct CutlassFusedTmaQkPv128Storage {
  alignas(128) typename CutlassCollectiveMainloop::SharedStorage qk;
  alignas(16) typename CutlassCollectiveMainloopK128::PipelineStorage
      pv_pipeline_storage;
  alignas(16) uint8_t p_packed[kCutlassTileM * (kCutlassTileK128 / 2)];
  alignas(16) uint8_t p_scales[kCutlassTileM * (kCutlassTileK128 / 16)];
  alignas(16) float row_m[kCutlassTileM];
  alignas(16) float row_l[kCutlassTileM];
  alignas(16) float global_m[kCutlassTileM];
  alignas(16) float global_l[kCutlassTileM];
  alignas(16) float old_scale[kCutlassTileM];
  alignas(16) float tile_scale[kCutlassTileM];
};

static_assert(sizeof(CutlassFusedTmaQkPv128Storage) <= (99u << 10),
              "phased QK/PV owner storage must fit SM120 opt-in shared memory");

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

__global__ void reduce_split_attention_kernel(const float* partial_out,
                                              const float* partial_m,
                                              const float* partial_l,
                                              float* out_block) {
  const int row = blockIdx.x;
  const int col = threadIdx.x;
  if (row >= kTileM || col >= kHeadDim) {
    return;
  }

  float m = -INFINITY;
#pragma unroll
  for (int split = 0; split < kNumKvSplits; ++split) {
    m = fmaxf(m, partial_m[split * kTileM + row]);
  }

  float l = 0.0f;
  float o = 0.0f;
#pragma unroll
  for (int split = 0; split < kNumKvSplits; ++split) {
    const float alpha = __expf(partial_m[split * kTileM + row] - m);
    l += partial_l[split * kTileM + row] * alpha;
    o += partial_out[(split * kTileM + row) * kHeadDim + col] * alpha;
  }

  out_block[row * kHeadDim + col] = o / fmaxf(l, 1.0e-20f);
}

__global__ void reduce_split_attention_128rows_kernel(const float* partial_out,
                                                      const float* partial_m,
                                                      const float* partial_l,
                                                      float* out_block,
                                                      int num_splits) {
  const int q_tile = blockIdx.y;
  const int row_local = blockIdx.x;
  const int col = threadIdx.x;
  if (q_tile >= kBenchQTiles || row_local >= kTileM || col >= kHeadDim) {
    return;
  }

  float m = -INFINITY;
  for (int split = 0; split < num_splits; ++split) {
    m = fmaxf(m, partial_m[(q_tile * num_splits + split) * kTileM +
                           row_local]);
  }

  float l = 0.0f;
  float o = 0.0f;
  for (int split = 0; split < num_splits; ++split) {
    const int partial_base =
        ((q_tile * num_splits + split) * kTileM + row_local) * kHeadDim;
    const float alpha =
        __expf(partial_m[(q_tile * num_splits + split) * kTileM +
                         row_local] -
               m);
    l += partial_l[(q_tile * num_splits + split) * kTileM + row_local] *
         alpha;
    o += partial_out[partial_base + col] * alpha;
  }

  out_block[(q_tile * kTileM + row_local) * kHeadDim + col] =
      o / fmaxf(l, 1.0e-20f);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
__device__ __forceinline__ void qk_cutlass_collective_tile_body(
    typename CutlassGemmKernel::Params const& params,
    typename CutlassCollectiveMainloop::SharedStorage& storage,
    float* out_tile,
    int m_tile,
    int n_tile,
    int out_stride) {
  using cute::_;

  const int thread_idx = int(threadIdx.x);
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
  const int warp_group_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
  const int mma_thread_idx = thread_idx % CutlassCollectiveMainloop::ThreadCount;
  const bool lane_predicate = cute::elect_one_sync();
  const uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

  enum class WarpGroupRole {
    Producer = 0,
    Consumer0 = 1,
    Consumer1 = 2,
  };
  enum class ProducerWarpRole {
    Mainloop = 0,
    Warp1 = 1,
    Epilogue = 2,
    MainloopAux = 3,
  };

  auto warp_group_role =
      WarpGroupRole(cutlass::canonical_warp_group_idx());
  auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);

  if (warp_idx == 0 && lane_predicate) {
    CutlassCollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
  }

  using MainloopPipeline = typename CutlassCollectiveMainloop::MainloopPipeline;
  typename MainloopPipeline::Params pipeline_params;
  if (warp_group_role == WarpGroupRole::Producer &&
      (producer_warp_role == ProducerWarpRole::Mainloop ||
       producer_warp_role == ProducerWarpRole::MainloopAux)) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
  }
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  pipeline_params.is_leader = warp_group_thread_idx == 0;
  pipeline_params.num_consumers = CutlassCollectiveMainloop::ThreadCount;
  pipeline_params.num_producers = CutlassCollectiveMainloop::NumProducerThreadEvents;
  pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;

  MainloopPipeline pipeline(storage.pipeline_storage, pipeline_params,
                            CutlassClusterShape{});
  typename CutlassCollectiveMainloop::PipelineState pipe_read;
  typename CutlassCollectiveMainloop::PipelineState pipe_write =
      cutlass::make_producer_start_state<MainloopPipeline>();

  __syncthreads();

  CutlassCollectiveMainloop collective;
  auto problem_shape_mnkl = cute::append<4>(params.problem_shape, cute::Int<1>{});
  auto load_inputs = collective.load_init(problem_shape_mnkl, params.mainloop);
  auto blk_coord = cute::make_coord(m_tile, n_tile, cute::_, 0);
  const int k_tile_count = kHeadDim / kCutlassTileK;
  auto k_tile_iter = cute::make_coord_iterator(k_tile_count);

  if (warp_group_role == WarpGroupRole::Producer &&
      producer_warp_role == ProducerWarpRole::Mainloop) {
    collective.load(params.mainloop, pipeline, pipe_write, load_inputs,
                    blk_coord, k_tile_iter, k_tile_count,
                    cutlass::canonical_lane_idx(), block_rank_in_cluster,
                    storage.tensors);
  }

  auto tiled_mma = typename CutlassCollectiveMainloop::TiledMma{};
  auto accum = cute::partition_fragment_C(tiled_mma,
                                          cute::take<0, 2>(CutlassThreadBlockShape{}));
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    collective.mma(pipeline, pipe_read, accum, k_tile_count, mma_thread_idx,
                   storage.tensors, params.mainloop);
  }
  if (warp_group_role == WarpGroupRole::Producer &&
      producer_warp_role == ProducerWarpRole::Mainloop) {
    collective.load_tail(pipeline, pipe_write);
  }
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    collective.mma_tail(pipeline, pipe_read, k_tile_count);
  }

  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    auto thread_mma = tiled_mma.get_thread_slice(mma_thread_idx);
    auto cC = cute::make_identity_tensor(
        cute::take<0, 2>(CutlassThreadBlockShape{}));
    auto tCcC = thread_mma.partition_C(cC);
    for (int i = 0; i < cute::size(accum); ++i) {
      auto coord = tCcC(i);
      const int row = int(cute::get<0>(coord));
      const int col = int(cute::get<1>(coord));
      if (row < kCutlassTileM && col < kCutlassTileN) {
        out_tile[row * out_stride + col] = accum(i);
      }
    }
  }
  __syncthreads();
}

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

__device__ __forceinline__ void cutlass_smem_atom_gemm_tile_body_k128(
    typename CutlassCollectiveMainloopK128::TensorStorage& storage,
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
  cutlass_smem_atom_gemm_tile_body_impl<CutlassCollectiveMainloopK128,
                                        CutlassThreadBlockShapeK128,
                                        kCutlassTileM,
                                        kCutlassTileN,
                                        kCutlassTileK128>(
      storage, q_packed, q_scales, k_packed, k_scales, out_tile, q_row_base,
      q_col_base, q_packed_cols, q_scale_cols, kv_row_base, kv_col_base,
      kv_packed_cols, kv_scale_cols, k_tile_limit, out_stride, out_row_base,
      out_col_base, data_debug_mode, scale_debug_mode);
}

__device__ __forceinline__ void cutlass_smem_pv_k128_reuse_p_full_width_body(
    typename CutlassCollectiveMainloopK128::TensorStorage& storage,
    const uint8_t* p_packed,
    const uint8_t* p_scales,
    const uint8_t* v_pv_packed,
    const uint8_t* v_pv_scales,
    float* out,
    int kv_base) {
  using cute::_;

  const int block_thread_idx = int(threadIdx.x);
  const bool mma_thread_active =
      block_thread_idx < CutlassCollectiveMainloopK128::ThreadCount;
  const int thread_idx = mma_thread_active ? block_thread_idx : 0;
  auto tiled_mma = typename CutlassCollectiveMainloopK128::TiledMma{};
  auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
  CutlassCollectiveMainloopK128 collective;

  auto sA = cute::make_tensor(cute::make_smem_ptr(storage.smem_A.begin()),
                              typename CutlassCollectiveMainloopK128::SmemLayoutA{});
  auto sB = cute::make_tensor(cute::make_smem_ptr(storage.smem_B.begin()),
                              typename CutlassCollectiveMainloopK128::SmemLayoutB{});
  auto sSFA = cute::make_tensor(
      cute::make_smem_ptr(storage.smem_SFA.begin()),
      typename CutlassCollectiveMainloopK128::SmemLayoutSFA{});
  auto sSFB = cute::make_tensor(
      cute::make_smem_ptr(storage.smem_SFB.begin()),
      typename CutlassCollectiveMainloopK128::SmemLayoutSFB{});

  auto tCrA = thread_mma.partition_fragment_A(sA(_, _, cute::Int<0>{}));
  auto tCrB = thread_mma.partition_fragment_B(sB(_, _, cute::Int<0>{}));
  auto tCrSFA = collective.partition_fragment_SFA(sSFA(_, _, cute::Int<0>{}),
                                                  thread_mma);
  auto tCrSFB = collective.partition_fragment_SFB(sSFB(_, _, cute::Int<0>{}),
                                                  thread_mma);

  auto smem_tiled_copy_A = cute::make_tiled_copy_A(
      typename CutlassCollectiveMainloopK128::SmemCopyAtomA{}, tiled_mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
  auto tCsA = smem_thr_copy_A.partition_S(
      cute::as_position_independent_swizzle_tensor(sA));
  auto tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  auto cA = cute::make_identity_tensor(
      cute::make_shape(cute::Int<kCutlassTileM>{},
                       cute::Int<kCutlassTileK128>{}, cute::Int<1>{}));
  auto tAsA_prod = smem_thr_copy_A.partition_D(sA);
  auto tAcA_prod = smem_thr_copy_A.partition_D(cA);

  auto smem_tiled_copy_B = cute::make_tiled_copy_B(
      typename CutlassCollectiveMainloopK128::SmemCopyAtomB{}, tiled_mma);
  auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
  auto tCsB = smem_thr_copy_B.partition_S(
      cute::as_position_independent_swizzle_tensor(sB));
  auto tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  auto cB = cute::make_identity_tensor(
      cute::make_shape(cute::Int<kCutlassTileN>{},
                       cute::Int<kCutlassTileK128>{}, cute::Int<1>{}));
  auto tBsB_prod = smem_thr_copy_B.partition_D(sB);
  auto tBcB_prod = smem_thr_copy_B.partition_D(cB);

  auto tile_shape_mnk = cute::tile_shape(tiled_mma);
  auto smem_tiled_copy_SFA = cute::make_tiled_copy_impl(
      typename CutlassCollectiveMainloopK128::SmemCopyAtomSFA{},
      collective.get_layoutSFA_TV(tiled_mma),
      cute::make_shape(cute::size<0>(tile_shape_mnk),
                       cute::size<2>(tile_shape_mnk)));
  auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
  auto tCsSFA = smem_thr_copy_SFA.partition_S(
      cute::as_position_independent_swizzle_tensor(sSFA));
  auto tCrSFA_copy_view = smem_thr_copy_SFA.retile_D(tCrSFA);

  auto smem_tiled_copy_SFB = cute::make_tiled_copy_impl(
      typename CutlassCollectiveMainloopK128::SmemCopyAtomSFB{},
      collective.get_layoutSFB_TV(tiled_mma),
      cute::make_shape(cute::size<1>(tile_shape_mnk),
                       cute::size<2>(tile_shape_mnk)));
  auto smem_thr_copy_SFB = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
  auto tCsSFB = smem_thr_copy_SFB.partition_S(
      cute::as_position_independent_swizzle_tensor(sSFB));
  auto tCrSFB_copy_view = smem_thr_copy_SFB.retile_D(tCrSFB);

  auto write_partitioned_fp4 = [&](auto tDst,
                                   auto tCoord,
                                   const uint8_t* packed,
                                   int source_row_base,
                                   int source_col_base,
                                   int packed_cols) {
    for (int i = 0; i < int(cute::size(tDst)); ++i) {
      auto coord = tCoord(i);
      const int row = int(cute::get<0>(coord));
      const int k = int(cute::get<1>(coord));
      const int source_col = source_col_base + k;
      const uint8_t byte =
          packed[(source_row_base + row) * packed_cols + (source_col >> 1)];
      const uint8_t code =
          static_cast<uint8_t>((source_col & 1) ? ((byte >> 4) & 0x0f)
                                                : (byte & 0x0f));
      tDst(i) = cute::uint4_t(code);
    }
  };

  uint8_t* smem_a_bytes = cute::recast_ptr<uint8_t>(storage.smem_A.begin());
  constexpr int kSmemABytes =
      (cute::cosize_v<typename CutlassCollectiveMainloopK128::SmemLayoutA> + 1) /
      2;
  for (int idx = block_thread_idx; idx < kSmemABytes; idx += blockDim.x) {
    smem_a_bytes[idx] = 0;
  }
  __syncthreads();

  auto K_BLOCK_MAX_PROD = cute::size<2>(tAsA_prod);
  if (mma_thread_active) {
    cute::for_each(cute::make_int_sequence<K_BLOCK_MAX_PROD>{}, [&](auto k_block) {
      write_partitioned_fp4(tAsA_prod(_, _, k_block, cute::Int<0>{}),
                            tAcA_prod(_, _, k_block, cute::Int<0>{}),
                            p_packed, 0, 0, kCutlassTileK128 / 2);
    });
  }
  for (int idx = block_thread_idx;
       idx < kCutlassTileM * kCutlassTileK128 / 2;
       idx += blockDim.x) {
    const int row = idx / (kCutlassTileK128 / 2);
    const int packed_k = idx - row * (kCutlassTileK128 / 2);
    const int k0 = 2 * packed_k;
    const int scale_col = k0 >> 4;
    sSFA(row, k0, cute::Int<0>{}) =
        make_ue4m3_raw(p_scales[row * (kCutlassTileK128 / 16) + scale_col]);
  }
  __syncthreads();

  auto copy_kblock = [&](auto k_block) {
    cute::copy(smem_tiled_copy_A, tCsA(_, _, k_block, cute::Int<0>{}),
               tCrA_copy_view(_, _, k_block));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, k_block, cute::Int<0>{}),
               tCrB_copy_view(_, _, k_block));

    using MMAOp = typename CutlassCollectiveMainloopK128::TiledMma::MMA_Op;
    fp4_shift_A(MMAOp{}, tCrA_copy_view(_, _, k_block));
    fp4_shift_B(MMAOp{}, tCrB_copy_view(_, _, k_block));

    cute::copy(tCsSFA(_, _, k_block, cute::Int<0>{}),
               tCrSFA_copy_view(_, _, k_block));
    cute::copy(tCsSFB(_, _, k_block, cute::Int<0>{}),
               tCrSFB_copy_view(_, _, k_block));
  };

  auto gemm_kblock = [&](auto k_block, auto& accum) {
    cute::gemm(tiled_mma,
               cute::make_zip_tensor(tCrA(_, _, k_block),
                                     tCrSFA(_, _, k_block)),
               cute::make_zip_tensor(tCrB(_, _, k_block),
                                     tCrSFB(_, _, k_block)),
               accum);
  };

  uint8_t* smem_b_bytes = cute::recast_ptr<uint8_t>(storage.smem_B.begin());
  constexpr int kSmemBBytes =
      (cute::cosize_v<typename CutlassCollectiveMainloopK128::SmemLayoutB> + 1) /
      2;

#pragma unroll
  for (int out_group = 0; out_group < kHeadDim / kCutlassTileN; ++out_group) {
    const int out_col_base = out_group * kCutlassTileN;
    for (int idx = block_thread_idx; idx < kSmemBBytes; idx += blockDim.x) {
      smem_b_bytes[idx] = 0;
    }
    __syncthreads();

    if (mma_thread_active) {
      cute::for_each(cute::make_int_sequence<K_BLOCK_MAX_PROD>{}, [&](auto k_block) {
        write_partitioned_fp4(tBsB_prod(_, _, k_block, cute::Int<0>{}),
                              tBcB_prod(_, _, k_block, cute::Int<0>{}),
                              v_pv_packed, out_col_base, kv_base,
                              kProbPackedCols);
      });
    }
    for (int idx = block_thread_idx;
         idx < kCutlassTileN * kCutlassTileK128 / 2;
         idx += blockDim.x) {
      const int row = idx / (kCutlassTileK128 / 2);
      const int packed_k = idx - row * (kCutlassTileK128 / 2);
      const int k0 = 2 * packed_k;
      const int scale_col = (kv_base + k0) >> 4;
      sSFB(row, k0, cute::Int<0>{}) =
          make_ue4m3_raw(v_pv_scales[(out_col_base + row) * kProbScaleCols +
                                     scale_col]);
    }
    __syncthreads();

    auto accum = cute::partition_fragment_C(
        tiled_mma, cute::take<0, 2>(CutlassThreadBlockShapeK128{}));
    cute::clear(accum);
    if (mma_thread_active) {
      auto K_BLOCK_MAX = cute::size<2>(tCrA);
      copy_kblock(cute::Int<0>{});
      cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
        if (k_block_next > 0) {
          copy_kblock(k_block_next);
        }
        gemm_kblock(k_block, accum);
      });

      auto cC = cute::make_identity_tensor(
          cute::take<0, 2>(CutlassThreadBlockShapeK128{}));
      auto tCcC = thread_mma.partition_C(cC);
      for (int i = 0; i < cute::size(accum); ++i) {
        auto coord = tCcC(i);
        const int row = int(cute::get<0>(coord));
        const int col = int(cute::get<1>(coord));
        if (row < kCutlassTileM && col < kCutlassTileN) {
          out[row * kHeadDim + out_col_base + col] = accum(i);
        }
      }
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void cutlass_smem_pv_k128_reuse_p_tma_v_full_width_body(
    typename CutlassGemmKernelK128::Params const& pv_params,
    typename CutlassCollectiveMainloopK128::SharedStorage& shared,
    const uint8_t* p_packed,
    const uint8_t* p_scales,
    float* out,
    int kv_tile_128) {
  using cute::_;

  const int block_thread_idx = int(threadIdx.x);
  const bool mma_thread_active =
      block_thread_idx < CutlassCollectiveMainloopK128::ThreadCount;
  const int thread_idx = mma_thread_active ? block_thread_idx : 0;
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
  const int warp_group_thread_idx =
      block_thread_idx % cutlass::NumThreadsPerWarpGroup;
  const int mma_thread_idx =
      block_thread_idx % CutlassCollectiveMainloopK128::ThreadCount;
  const bool lane_predicate = cute::elect_one_sync();

  enum class WarpGroupRole {
    Producer = 0,
    Consumer0 = 1,
    Consumer1 = 2,
  };
  enum class ProducerWarpRole {
    Mainloop = 0,
    Warp1 = 1,
    Epilogue = 2,
    MainloopAux = 3,
  };

  auto warp_group_role = WarpGroupRole(cutlass::canonical_warp_group_idx());
  auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);

  CutlassCollectiveMainloopK128 collective;
  auto tiled_mma = typename CutlassCollectiveMainloopK128::TiledMma{};
  auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

  auto sA = cute::make_tensor(cute::make_smem_ptr(shared.tensors.smem_A.begin()),
                              typename CutlassCollectiveMainloopK128::SmemLayoutA{});
  auto sSFA = cute::make_tensor(
      cute::make_smem_ptr(shared.tensors.smem_SFA.begin()),
      typename CutlassCollectiveMainloopK128::SmemLayoutSFA{});
  auto sB = cute::make_tensor(cute::make_smem_ptr(shared.tensors.smem_B.begin()),
                              typename CutlassCollectiveMainloopK128::SmemLayoutB{});
  auto sSFB = cute::make_tensor(
      cute::make_smem_ptr(shared.tensors.smem_SFB.begin()),
      typename CutlassCollectiveMainloopK128::SmemLayoutSFB{});

  auto smem_tiled_copy_A = cute::make_tiled_copy_A(
      typename CutlassCollectiveMainloopK128::SmemCopyAtomA{}, tiled_mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
  auto cA = cute::make_identity_tensor(
      cute::make_shape(cute::Int<kCutlassTileM>{},
                       cute::Int<kCutlassTileK128>{}, cute::Int<1>{}));
  auto tAsA_prod = smem_thr_copy_A.partition_D(sA);
  auto tAcA_prod = smem_thr_copy_A.partition_D(cA);

  auto tile_shape_mnk = cute::tile_shape(tiled_mma);
  auto smem_tiled_copy_SFA = cute::make_tiled_copy_impl(
      typename CutlassCollectiveMainloopK128::SmemCopyAtomSFA{},
      collective.get_layoutSFA_TV(tiled_mma),
      cute::make_shape(cute::size<0>(tile_shape_mnk),
                       cute::size<2>(tile_shape_mnk)));
  auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_thread_slice(thread_idx);

  auto write_partitioned_fp4 = [&](auto tDst,
                                   auto tCoord,
                                   const uint8_t* packed,
                                   int source_row_base,
                                   int source_col_base,
                                   int packed_cols) {
    for (int i = 0; i < int(cute::size(tDst)); ++i) {
      auto coord = tCoord(i);
      const int row = int(cute::get<0>(coord));
      const int k = int(cute::get<1>(coord));
      const int source_col = source_col_base + k;
      const uint8_t byte =
          packed[(source_row_base + row) * packed_cols + (source_col >> 1)];
      const uint8_t code =
          static_cast<uint8_t>((source_col & 1) ? ((byte >> 4) & 0x0f)
                                                : (byte & 0x0f));
      tDst(i) = cute::uint4_t(code);
    }
  };

  uint8_t* smem_a_bytes = cute::recast_ptr<uint8_t>(shared.tensors.smem_A.begin());
  constexpr int kSmemABytes =
      (cute::cosize_v<typename CutlassCollectiveMainloopK128::SmemLayoutA> + 1) /
      2;
  for (int idx = block_thread_idx; idx < kSmemABytes; idx += blockDim.x) {
    smem_a_bytes[idx] = 0;
  }
  __syncthreads();

  auto K_BLOCK_MAX_PROD = cute::size<2>(tAsA_prod);
  if (mma_thread_active) {
    cute::for_each(cute::make_int_sequence<K_BLOCK_MAX_PROD>{}, [&](auto k_block) {
      write_partitioned_fp4(tAsA_prod(_, _, k_block, cute::Int<0>{}),
                            tAcA_prod(_, _, k_block, cute::Int<0>{}),
                            p_packed, 0, 0, kCutlassTileK128 / 2);
    });
  }
  for (int idx = block_thread_idx;
       idx < kCutlassTileM * kCutlassTileK128 / 2;
       idx += blockDim.x) {
    const int row = idx / (kCutlassTileK128 / 2);
    const int packed_k = idx - row * (kCutlassTileK128 / 2);
    const int k0 = 2 * packed_k;
    const int scale_col = k0 >> 4;
    sSFA(row, k0, cute::Int<0>{}) =
        make_ue4m3_raw(p_scales[row * (kCutlassTileK128 / 16) + scale_col]);
  }
  __syncthreads();

  if (warp_idx == 0 && lane_predicate) {
    CutlassCollectiveMainloopK128::prefetch_tma_descriptors(
        pv_params.mainloop);
  }
  __syncthreads();

#pragma unroll
  for (int out_group = 0; out_group < kHeadDim / kCutlassTileN; ++out_group) {
    using MainloopPipeline =
        typename CutlassCollectiveMainloopK128::MainloopPipeline;
    typename MainloopPipeline::Params pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer &&
        (producer_warp_role == ProducerWarpRole::Mainloop ||
         producer_warp_role == ProducerWarpRole::MainloopAux)) {
      pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 ||
        warp_group_role == WarpGroupRole::Consumer1) {
      pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = CutlassCollectiveMainloopK128::ThreadCount;
    pipeline_params.num_producers =
        CutlassCollectiveMainloopK128::NumProducerThreadEvents;
    pipeline_params.transaction_bytes =
        pv_params.mainloop.tma_transaction_bytes_nk;

    MainloopPipeline pipeline(shared.pipeline_storage, pipeline_params,
                              CutlassClusterShape{});
    typename CutlassCollectiveMainloopK128::PipelineState pipe_read;
    typename CutlassCollectiveMainloopK128::PipelineState pipe_write =
        cutlass::make_producer_start_state<MainloopPipeline>();
    __syncthreads();

    auto problem_shape_mnkl =
        cute::append<4>(pv_params.problem_shape, cute::Int<1>{});
    auto load_inputs = collective.load_init(problem_shape_mnkl,
                                            pv_params.mainloop);
    auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl] = load_inputs;

    if (warp_group_role == WarpGroupRole::Producer &&
        producer_warp_role == ProducerWarpRole::Mainloop) {
      if (lane_predicate) {
        auto block_tma_b = pv_params.mainloop.tma_load_b.get_slice(0);
        auto block_tma_sfb = pv_params.mainloop.tma_load_sfb.get_slice(0);
        auto gB = gB_nkl(_, _, out_group, _, 0);
        auto gSFB = gSFB_nkl(_, _, out_group, _, 0);
        auto tBgB = block_tma_b.partition_S(gB);
        auto tBsB = block_tma_b.partition_D(sB);
        auto tBgSFB = block_tma_sfb.partition_S(gSFB);
        auto tBsSFB = block_tma_sfb.partition_D(sSFB);

        auto k_tile_iter = cute::make_coord_iterator(
            cute::idx2crd(kv_tile_128, cute::shape<3>(gB_nkl)),
            cute::shape<3>(gB_nkl));
        pipeline.producer_acquire(pipe_write);
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(pipe_write);
        int write_stage = pipe_write.index();
        copy(pv_params.mainloop.tma_load_b.with(*tma_barrier),
             tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, write_stage));
        copy(pv_params.mainloop.tma_load_sfb.with(*tma_barrier),
             tBgSFB(_, _, _, *k_tile_iter), tBsSFB(_, _, _, write_stage));
        ++pipe_write;
      }
      collective.load_tail(pipeline, pipe_write);
    }

    auto accum = cute::partition_fragment_C(
        tiled_mma, cute::take<0, 2>(CutlassThreadBlockShapeK128{}));
    if (warp_group_role == WarpGroupRole::Consumer0 ||
        warp_group_role == WarpGroupRole::Consumer1) {
      collective.mma(pipeline, pipe_read, accum, 1, mma_thread_idx,
                     shared.tensors, pv_params.mainloop);
      collective.mma_tail(pipeline, pipe_read, 1);

      auto output_thread_mma = tiled_mma.get_thread_slice(mma_thread_idx);
      auto cC = cute::make_identity_tensor(
          cute::take<0, 2>(CutlassThreadBlockShapeK128{}));
      auto tCcC = output_thread_mma.partition_C(cC);
      const int out_col_base = out_group * kCutlassTileN;
      for (int i = 0; i < cute::size(accum); ++i) {
        auto coord = tCcC(i);
        const int row = int(cute::get<0>(coord));
        const int col = int(cute::get<1>(coord));
        if (row < kCutlassTileM && col < kCutlassTileN) {
          out[row * kHeadDim + out_col_base + col] = accum(i);
        }
      }
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void cutlass_smem_pv_k128_stage_p_body(
    typename CutlassCollectiveMainloopK128::SharedStorage& shared,
    const uint8_t* p_packed,
    const uint8_t* p_scales,
    const float* row_scale = nullptr,
    float base_scale = 1.0f,
    int smem_stage = 0) {
  using cute::_;

  const int block_thread_idx = int(threadIdx.x);
  const bool mma_thread_active =
      block_thread_idx < CutlassCollectiveMainloopK128::ThreadCount;
  const int thread_idx = mma_thread_active ? block_thread_idx : 0;

  CutlassCollectiveMainloopK128 collective;
  auto tiled_mma = typename CutlassCollectiveMainloopK128::TiledMma{};
  auto sA = cute::make_tensor(cute::make_smem_ptr(shared.tensors.smem_A.begin()),
                              typename CutlassCollectiveMainloopK128::SmemLayoutA{});
  auto sSFA = cute::make_tensor(
      cute::make_smem_ptr(shared.tensors.smem_SFA.begin()),
      typename CutlassCollectiveMainloopK128::SmemLayoutSFA{});

  auto smem_tiled_copy_A = cute::make_tiled_copy_A(
      typename CutlassCollectiveMainloopK128::SmemCopyAtomA{}, tiled_mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
  auto cA = cute::make_identity_tensor(
      cute::make_shape(cute::Int<kCutlassTileM>{},
                       cute::Int<kCutlassTileK128>{}, cute::Int<1>{}));
  auto tAsA_prod = smem_thr_copy_A.partition_D(sA);
  auto tAcA_prod = smem_thr_copy_A.partition_D(cA);

  auto write_partitioned_fp4 = [&](auto tDst,
                                   auto tCoord,
                                   const uint8_t* packed,
                                   int source_row_base,
                                   int source_col_base,
                                   int packed_cols) {
    for (int i = 0; i < int(cute::size(tDst)); ++i) {
      auto coord = tCoord(i);
      const int row = int(cute::get<0>(coord));
      const int k = int(cute::get<1>(coord));
      const int source_col = source_col_base + k;
      const uint8_t byte =
          packed[(source_row_base + row) * packed_cols + (source_col >> 1)];
      const uint8_t code =
          static_cast<uint8_t>((source_col & 1) ? ((byte >> 4) & 0x0f)
                                                : (byte & 0x0f));
      tDst(i) = cute::uint4_t(code);
    }
  };

  uint8_t* smem_a_bytes = cute::recast_ptr<uint8_t>(shared.tensors.smem_A.begin());
  constexpr int kSmemABytes =
      (cute::cosize_v<typename CutlassCollectiveMainloopK128::SmemLayoutA> + 1) /
      2;
  for (int idx = block_thread_idx; idx < kSmemABytes; idx += blockDim.x) {
    smem_a_bytes[idx] = 0;
  }
  __syncthreads();

  auto K_BLOCK_MAX_PROD = cute::size<2>(tAsA_prod);
  if (mma_thread_active) {
    cute::for_each(cute::make_int_sequence<K_BLOCK_MAX_PROD>{}, [&](auto k_block) {
      write_partitioned_fp4(tAsA_prod(_, _, k_block, smem_stage),
                            tAcA_prod(_, _, k_block, cute::Int<0>{}),
                            p_packed, 0, 0, kCutlassTileK128 / 2);
    });
  }

  for (int idx = block_thread_idx;
       idx < kCutlassTileM * kCutlassTileK128 / 2;
       idx += blockDim.x) {
    const int row = idx / (kCutlassTileK128 / 2);
    const int packed_k = idx - row * (kCutlassTileK128 / 2);
    const int k0 = 2 * packed_k;
    const int scale_col = k0 >> 4;
    uint8_t scale_byte = p_scales[row * (kCutlassTileK128 / 16) + scale_col];
    if (row_scale != nullptr) {
      const float scale =
          e4m3_byte_to_fp32(scale_byte) * row_scale[row] * base_scale;
      scale_byte = fp32_to_e4m3_byte(scale);
    }
    sSFA(row, k0, smem_stage) = make_ue4m3_raw(scale_byte);
  }
  __syncthreads();
}

template <class FrgTensorC>
__device__ __forceinline__ void cutlass_smem_pv_k128_scale_or_clear_accum(
    FrgTensorC& accum,
    const float* row_scale,
    int mma_thread_idx,
    int clear_accum) {
  auto tiled_mma = typename CutlassCollectiveMainloopK128::TiledMma{};
  auto thread_mma = tiled_mma.get_thread_slice(mma_thread_idx);
  auto cC = cute::make_identity_tensor(
      cute::take<0, 2>(CutlassThreadBlockShapeK128{}));
  auto tCcC = thread_mma.partition_C(cC);
  for (int i = 0; i < cute::size(accum); ++i) {
    if (clear_accum != 0) {
      accum(i) = 0.0f;
      continue;
    }
    auto coord = tCcC(i);
    const int row = int(cute::get<0>(coord));
    if (row < kCutlassTileM) {
      accum(i) *= row_scale[row];
    }
  }
}

template <class FrgTensorC>
__device__ __forceinline__ void cutlass_smem_pv_k128_mma_no_clear(
    typename CutlassCollectiveMainloopK128::MainloopPipeline pipeline,
    typename CutlassCollectiveMainloopK128::PipelineState& smem_pipe_read,
    FrgTensorC& accum,
    int k_tile_count,
    int thread_idx,
    typename CutlassCollectiveMainloopK128::TensorStorage& shared_tensors) {
  using namespace cute;

  Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()),
                          typename CutlassCollectiveMainloopK128::SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()),
                          typename CutlassCollectiveMainloopK128::SmemLayoutB{});
  Tensor sSFA = make_tensor(
      make_smem_ptr(shared_tensors.smem_SFA.begin()),
      typename CutlassCollectiveMainloopK128::SmemLayoutSFA{});
  Tensor sSFB = make_tensor(
      make_smem_ptr(shared_tensors.smem_SFB.begin()),
      typename CutlassCollectiveMainloopK128::SmemLayoutSFB{});

  auto tiled_mma = typename CutlassCollectiveMainloopK128::TiledMma{};
  CutlassCollectiveMainloopK128 collective;
  auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

  Tensor tCrA = thread_mma.partition_fragment_A(sA(_, _, Int<0>{}));
  Tensor tCrB = thread_mma.partition_fragment_B(sB(_, _, Int<0>{}));
  Tensor tCrSFA =
      collective.partition_fragment_SFA(sSFA(_, _, Int<0>{}), thread_mma);
  Tensor tCrSFB =
      collective.partition_fragment_SFB(sSFB(_, _, Int<0>{}), thread_mma);

  auto smem_tiled_copy_A = make_tiled_copy_A(
      typename CutlassCollectiveMainloopK128::SmemCopyAtomA{}, tiled_mma);
  auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
  Tensor tCsA = smem_thr_copy_A.partition_S(
      as_position_independent_swizzle_tensor(sA));
  Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);

  auto smem_tiled_copy_B = make_tiled_copy_B(
      typename CutlassCollectiveMainloopK128::SmemCopyAtomB{}, tiled_mma);
  auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
  Tensor tCsB = smem_thr_copy_B.partition_S(
      as_position_independent_swizzle_tensor(sB));
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);

  auto tile_shape_mnk = tile_shape(tiled_mma);
  auto smem_tiled_copy_SFA = make_tiled_copy_impl(
      typename CutlassCollectiveMainloopK128::SmemCopyAtomSFA{},
      collective.get_layoutSFA_TV(tiled_mma),
      make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto smem_thr_copy_SFA = smem_tiled_copy_SFA.get_thread_slice(thread_idx);
  Tensor tCsSFA = smem_thr_copy_SFA.partition_S(
      as_position_independent_swizzle_tensor(sSFA));
  Tensor tCrSFA_copy_view = smem_thr_copy_SFA.retile_D(tCrSFA);

  auto smem_tiled_copy_SFB = make_tiled_copy_impl(
      typename CutlassCollectiveMainloopK128::SmemCopyAtomSFB{},
      collective.get_layoutSFB_TV(tiled_mma),
      make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto smem_thr_copy_SFB = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
  Tensor tCsSFB = smem_thr_copy_SFB.partition_S(
      as_position_independent_swizzle_tensor(sSFB));
  Tensor tCrSFB_copy_view = smem_thr_copy_SFB.retile_D(tCrSFB);

  auto K_BLOCK_MAX = size<2>(tCrA);
  int read_stage = smem_pipe_read.index();
  auto tCsA_stage = tCsA(_, _, _, read_stage);
  auto tCsB_stage = tCsB(_, _, _, read_stage);
  auto tCsSFA_stage = tCsSFA(_, _, _, read_stage);
  auto tCsSFB_stage = tCsSFB(_, _, _, read_stage);

  auto copy_kblock = [&](auto k_block) {
    copy(smem_tiled_copy_A, tCsA_stage(_, _, k_block),
         tCrA_copy_view(_, _, k_block));
    copy(smem_tiled_copy_B, tCsB_stage(_, _, k_block),
         tCrB_copy_view(_, _, k_block));
    using MMAOp = typename CutlassCollectiveMainloopK128::TiledMma::MMA_Op;
    fp4_shift_A(MMAOp{}, tCrA_copy_view(_, _, k_block));
    fp4_shift_B(MMAOp{}, tCrB_copy_view(_, _, k_block));
    copy(tCsSFA_stage(_, _, k_block), tCrSFA_copy_view(_, _, k_block));
    copy(tCsSFB_stage(_, _, k_block), tCrSFB_copy_view(_, _, k_block));
  };

  auto gemm_kblock = [&](auto k_block) {
    cute::gemm(tiled_mma,
               make_zip_tensor(tCrA(_, _, k_block), tCrSFA(_, _, k_block)),
               make_zip_tensor(tCrB(_, _, k_block), tCrSFB(_, _, k_block)),
               accum);
  };

  pipeline.consumer_wait(smem_pipe_read);
  copy_kblock(_0{});

  CUTLASS_PRAGMA_NO_UNROLL
  for (; k_tile_count > 1; --k_tile_count) {
    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
      auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
      if (k_block == K_BLOCK_MAX - 1) {
        cutlass::arch::NamedBarrier::sync(
            thr_size(tiled_mma),
            cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
        pipeline.consumer_release(smem_pipe_read);
        ++smem_pipe_read;
        read_stage = smem_pipe_read.index();
        tCsA_stage = tCsA(_, _, _, read_stage);
        tCsB_stage = tCsB(_, _, _, read_stage);
        tCsSFA_stage = tCsSFA(_, _, _, read_stage);
        tCsSFB_stage = tCsSFB(_, _, _, read_stage);
        pipeline.consumer_wait(smem_pipe_read);
      }
      copy_kblock(k_block_next);
      gemm_kblock(k_block);
    });
  }

  for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
    auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
    if (k_block == K_BLOCK_MAX - 1) {
      cutlass::arch::NamedBarrier::sync(
          thr_size(tiled_mma),
          cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
    }
    if (k_block_next > 0) {
      copy_kblock(k_block_next);
    }
    gemm_kblock(k_block);
  });

  cutlass::arch::NamedBarrier::sync(
      thr_size(tiled_mma),
      cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
}

template <class FrgTensorA, class FrgTensorSFA, class FrgTensorC>
__device__ __forceinline__ void cutlass_qk_tma_k_mma_register_q_stage(
    typename CutlassCollectiveMainloop::MainloopPipeline pipeline,
    typename CutlassCollectiveMainloop::PipelineState& smem_pipe_read,
    FrgTensorA const& q_frag,
    FrgTensorSFA const& q_scale_frag,
    FrgTensorC& accum,
    int thread_idx,
    typename CutlassCollectiveMainloop::TensorStorage& shared_tensors) {
  using namespace cute;

  Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()),
                          typename CutlassCollectiveMainloop::SmemLayoutB{});
  Tensor sSFB = make_tensor(
      make_smem_ptr(shared_tensors.smem_SFB.begin()),
      typename CutlassCollectiveMainloop::SmemLayoutSFB{});

  auto tiled_mma = typename CutlassCollectiveMainloop::TiledMma{};
  CutlassCollectiveMainloop collective;
  auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

  Tensor tCrB = thread_mma.partition_fragment_B(sB(_, _, Int<0>{}));
  Tensor tCrSFB =
      collective.partition_fragment_SFB(sSFB(_, _, Int<0>{}), thread_mma);

  auto smem_tiled_copy_B = make_tiled_copy_B(
      typename CutlassCollectiveMainloop::SmemCopyAtomB{}, tiled_mma);
  auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
  Tensor tCsB = smem_thr_copy_B.partition_S(
      as_position_independent_swizzle_tensor(sB));
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);

  auto tile_shape_mnk = tile_shape(tiled_mma);
  auto smem_tiled_copy_SFB = make_tiled_copy_impl(
      typename CutlassCollectiveMainloop::SmemCopyAtomSFB{},
      collective.get_layoutSFB_TV(tiled_mma),
      make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto smem_thr_copy_SFB = smem_tiled_copy_SFB.get_thread_slice(thread_idx);
  Tensor tCsSFB = smem_thr_copy_SFB.partition_S(
      as_position_independent_swizzle_tensor(sSFB));
  Tensor tCrSFB_copy_view = smem_thr_copy_SFB.retile_D(tCrSFB);

  auto K_BLOCK_MAX = size<2>(q_frag);
  int read_stage = smem_pipe_read.index();
  auto tCsB_stage = tCsB(_, _, _, read_stage);
  auto tCsSFB_stage = tCsSFB(_, _, _, read_stage);

  auto copy_kblock = [&](auto k_block) {
    copy(smem_tiled_copy_B, tCsB_stage(_, _, k_block),
         tCrB_copy_view(_, _, k_block));
    using MMAOp = typename CutlassCollectiveMainloop::TiledMma::MMA_Op;
    fp4_shift_B(MMAOp{}, tCrB_copy_view(_, _, k_block));
    copy(tCsSFB_stage(_, _, k_block), tCrSFB_copy_view(_, _, k_block));
  };

  auto gemm_kblock = [&](auto k_block) {
    cute::gemm(tiled_mma,
               make_zip_tensor(q_frag(_, _, k_block),
                               q_scale_frag(_, _, k_block)),
               make_zip_tensor(tCrB(_, _, k_block),
                               tCrSFB(_, _, k_block)),
               accum);
  };

  pipeline.consumer_wait(smem_pipe_read);
  copy_kblock(_0{});

  for_each(make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
    auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
    if (k_block == K_BLOCK_MAX - 1) {
      cutlass::arch::NamedBarrier::sync(
          thr_size(tiled_mma),
          cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
    }
    if (k_block_next > 0) {
      copy_kblock(k_block_next);
    }
    gemm_kblock(k_block);
  });

  cutlass::arch::NamedBarrier::sync(
      thr_size(tiled_mma),
      cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
}

__device__ __forceinline__ void cutlass_smem_pv_k128_tma_v_group_body(
    typename CutlassGemmKernelK128::Params const& pv_params,
    typename CutlassCollectiveMainloopK128::SharedStorage& shared,
    float* out,
    int out_stride,
    int out_col_base,
    int out_group,
    int kv_tile_128,
    const float* old_scale = nullptr,
    const float* tile_scale = nullptr,
    float output_scale = 1.0f,
    int accumulate = 0) {
  using cute::_;

  const int block_thread_idx = int(threadIdx.x);
  const int thread_idx =
      block_thread_idx < CutlassCollectiveMainloopK128::ThreadCount
          ? block_thread_idx
          : 0;
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
  const int warp_group_thread_idx =
      block_thread_idx % cutlass::NumThreadsPerWarpGroup;
  const int mma_thread_idx =
      block_thread_idx % CutlassCollectiveMainloopK128::ThreadCount;
  const bool lane_predicate = cute::elect_one_sync();

  enum class WarpGroupRole {
    Producer = 0,
    Consumer0 = 1,
    Consumer1 = 2,
  };
  enum class ProducerWarpRole {
    Mainloop = 0,
    Warp1 = 1,
    Epilogue = 2,
    MainloopAux = 3,
  };

  auto warp_group_role = WarpGroupRole(cutlass::canonical_warp_group_idx());
  auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);

  CutlassCollectiveMainloopK128 collective;
  auto sB = cute::make_tensor(cute::make_smem_ptr(shared.tensors.smem_B.begin()),
                              typename CutlassCollectiveMainloopK128::SmemLayoutB{});
  auto sSFB = cute::make_tensor(
      cute::make_smem_ptr(shared.tensors.smem_SFB.begin()),
      typename CutlassCollectiveMainloopK128::SmemLayoutSFB{});

  if (warp_idx == 0 && lane_predicate) {
    CutlassCollectiveMainloopK128::prefetch_tma_descriptors(
        pv_params.mainloop);
  }
  __syncthreads();

  using MainloopPipeline =
      typename CutlassCollectiveMainloopK128::MainloopPipeline;
  typename MainloopPipeline::Params pipeline_params;
  if (warp_group_role == WarpGroupRole::Producer &&
      (producer_warp_role == ProducerWarpRole::Mainloop ||
       producer_warp_role == ProducerWarpRole::MainloopAux)) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
  }
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  pipeline_params.is_leader = warp_group_thread_idx == 0;
  pipeline_params.num_consumers = CutlassCollectiveMainloopK128::ThreadCount;
  pipeline_params.num_producers =
      CutlassCollectiveMainloopK128::NumProducerThreadEvents;
  pipeline_params.transaction_bytes =
      pv_params.mainloop.tma_transaction_bytes_nk;

  MainloopPipeline pipeline(shared.pipeline_storage, pipeline_params,
                            CutlassClusterShape{});
  typename CutlassCollectiveMainloopK128::PipelineState pipe_read;
  typename CutlassCollectiveMainloopK128::PipelineState pipe_write =
      cutlass::make_producer_start_state<MainloopPipeline>();
  __syncthreads();

  auto problem_shape_mnkl =
      cute::append<4>(pv_params.problem_shape, cute::Int<1>{});
  auto load_inputs = collective.load_init(problem_shape_mnkl,
                                          pv_params.mainloop);
  auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl] = load_inputs;

  if (warp_group_role == WarpGroupRole::Producer &&
      producer_warp_role == ProducerWarpRole::Mainloop) {
    if (lane_predicate) {
      auto block_tma_b = pv_params.mainloop.tma_load_b.get_slice(0);
      auto block_tma_sfb = pv_params.mainloop.tma_load_sfb.get_slice(0);
      auto gB = gB_nkl(_, _, out_group, _, 0);
      auto gSFB = gSFB_nkl(_, _, out_group, _, 0);
      auto tBgB = block_tma_b.partition_S(gB);
      auto tBsB = block_tma_b.partition_D(sB);
      auto tBgSFB = block_tma_sfb.partition_S(gSFB);
      auto tBsSFB = block_tma_sfb.partition_D(sSFB);

      auto k_tile_iter = cute::make_coord_iterator(
          cute::idx2crd(kv_tile_128, cute::shape<3>(gB_nkl)),
          cute::shape<3>(gB_nkl));
      pipeline.producer_acquire(pipe_write);
      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(pipe_write);
      int write_stage = pipe_write.index();
      copy(pv_params.mainloop.tma_load_b.with(*tma_barrier),
           tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, write_stage));
      copy(pv_params.mainloop.tma_load_sfb.with(*tma_barrier),
           tBgSFB(_, _, _, *k_tile_iter), tBsSFB(_, _, _, write_stage));
      ++pipe_write;
    }
    collective.load_tail(pipeline, pipe_write);
  }

  auto tiled_mma = typename CutlassCollectiveMainloopK128::TiledMma{};
  auto accum = cute::partition_fragment_C(
      tiled_mma, cute::take<0, 2>(CutlassThreadBlockShapeK128{}));
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    collective.mma(pipeline, pipe_read, accum, 1, mma_thread_idx,
                   shared.tensors, pv_params.mainloop);
    collective.mma_tail(pipeline, pipe_read, 1);

    auto thread_mma = tiled_mma.get_thread_slice(mma_thread_idx);
    auto cC = cute::make_identity_tensor(
        cute::take<0, 2>(CutlassThreadBlockShapeK128{}));
    auto tCcC = thread_mma.partition_C(cC);
    for (int i = 0; i < cute::size(accum); ++i) {
      auto coord = tCcC(i);
      const int row = int(cute::get<0>(coord));
      const int col = int(cute::get<1>(coord));
      if (row < kCutlassTileM && col < kCutlassTileN) {
        float value = accum(i);
        if (tile_scale != nullptr) {
          value *= tile_scale[row] * output_scale;
        }
        const int out_idx = row * out_stride + out_col_base + col;
        if (accumulate != 0) {
          value += out[out_idx] * old_scale[row];
        }
        out[out_idx] = value;
      }
    }
  }
  __syncthreads();
}

template <class FrgTensorC>
__device__ __forceinline__ void cutlass_smem_pv_k128_tma_v_group_accum_body(
    typename CutlassGemmKernelK128::Params const& pv_params,
    typename CutlassCollectiveMainloopK128::SharedStorage& shared,
    FrgTensorC& accum,
    int out_group,
    int kv_tile_128,
    typename CutlassCollectiveMainloopK128::PipelineStorage*
        pipeline_storage_override = nullptr) {
  using cute::_;

  const int block_thread_idx = int(threadIdx.x);
  const int thread_idx =
      block_thread_idx < CutlassCollectiveMainloopK128::ThreadCount
          ? block_thread_idx
          : 0;
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
  const int warp_group_thread_idx =
      block_thread_idx % cutlass::NumThreadsPerWarpGroup;
  const int mma_thread_idx =
      block_thread_idx % CutlassCollectiveMainloopK128::ThreadCount;
  const bool lane_predicate = cute::elect_one_sync();

  enum class WarpGroupRole {
    Producer = 0,
    Consumer0 = 1,
    Consumer1 = 2,
  };
  enum class ProducerWarpRole {
    Mainloop = 0,
    Warp1 = 1,
    Epilogue = 2,
    MainloopAux = 3,
  };

  auto warp_group_role = WarpGroupRole(cutlass::canonical_warp_group_idx());
  auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);

  CutlassCollectiveMainloopK128 collective;
  auto sB = cute::make_tensor(cute::make_smem_ptr(shared.tensors.smem_B.begin()),
                              typename CutlassCollectiveMainloopK128::SmemLayoutB{});
  auto sSFB = cute::make_tensor(
      cute::make_smem_ptr(shared.tensors.smem_SFB.begin()),
      typename CutlassCollectiveMainloopK128::SmemLayoutSFB{});

  if (warp_idx == 0 && lane_predicate) {
    CutlassCollectiveMainloopK128::prefetch_tma_descriptors(
        pv_params.mainloop);
  }
  __syncthreads();

  using MainloopPipeline =
      typename CutlassCollectiveMainloopK128::MainloopPipeline;
  typename MainloopPipeline::Params pipeline_params;
  if (warp_group_role == WarpGroupRole::Producer &&
      (producer_warp_role == ProducerWarpRole::Mainloop ||
       producer_warp_role == ProducerWarpRole::MainloopAux)) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
  }
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  pipeline_params.is_leader = warp_group_thread_idx == 0;
  pipeline_params.num_consumers = CutlassCollectiveMainloopK128::ThreadCount;
  pipeline_params.num_producers =
      CutlassCollectiveMainloopK128::NumProducerThreadEvents;
  pipeline_params.transaction_bytes =
      pv_params.mainloop.tma_transaction_bytes_nk;

  auto* pipeline_storage = pipeline_storage_override != nullptr
                               ? pipeline_storage_override
                               : &shared.pipeline_storage;
  MainloopPipeline pipeline(*pipeline_storage, pipeline_params,
                            CutlassClusterShape{});
  typename CutlassCollectiveMainloopK128::PipelineState pipe_read;
  typename CutlassCollectiveMainloopK128::PipelineState pipe_write =
      cutlass::make_producer_start_state<MainloopPipeline>();
  __syncthreads();

  auto problem_shape_mnkl =
      cute::append<4>(pv_params.problem_shape, cute::Int<1>{});
  auto load_inputs = collective.load_init(problem_shape_mnkl,
                                          pv_params.mainloop);
  auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl] = load_inputs;

  if (warp_group_role == WarpGroupRole::Producer &&
      producer_warp_role == ProducerWarpRole::Mainloop) {
    if (lane_predicate) {
      auto block_tma_b = pv_params.mainloop.tma_load_b.get_slice(0);
      auto block_tma_sfb = pv_params.mainloop.tma_load_sfb.get_slice(0);
      auto gB = gB_nkl(_, _, out_group, _, 0);
      auto gSFB = gSFB_nkl(_, _, out_group, _, 0);
      auto tBgB = block_tma_b.partition_S(gB);
      auto tBsB = block_tma_b.partition_D(sB);
      auto tBgSFB = block_tma_sfb.partition_S(gSFB);
      auto tBsSFB = block_tma_sfb.partition_D(sSFB);

      auto k_tile_iter = cute::make_coord_iterator(
          cute::idx2crd(kv_tile_128, cute::shape<3>(gB_nkl)),
          cute::shape<3>(gB_nkl));
      pipeline.producer_acquire(pipe_write);
      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(pipe_write);
      int write_stage = pipe_write.index();
      copy(pv_params.mainloop.tma_load_b.with(*tma_barrier),
           tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, write_stage));
      copy(pv_params.mainloop.tma_load_sfb.with(*tma_barrier),
           tBgSFB(_, _, _, *k_tile_iter), tBsSFB(_, _, _, write_stage));
      ++pipe_write;
    }
    collective.load_tail(pipeline, pipe_write);
  }

  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    cutlass_smem_pv_k128_mma_no_clear(
        pipeline, pipe_read, accum, 1, mma_thread_idx, shared.tensors);
    collective.mma_tail(pipeline, pipe_read, 1);
  }
  __syncthreads();
}

template <class FrgTensorC>
__device__ __forceinline__
void cutlass_smem_pv_k128_tma_v_group_accum_persistent_step(
    typename CutlassGemmKernelK128::Params const& pv_params,
    typename CutlassCollectiveMainloopK128::SharedStorage& shared,
    typename CutlassCollectiveMainloopK128::MainloopPipeline& pipeline,
    typename CutlassCollectiveMainloopK128::PipelineState& pipe_read,
    typename CutlassCollectiveMainloopK128::PipelineState& pipe_write,
    FrgTensorC& accum,
    int out_group,
    int kv_tile_128) {
  using cute::_;

  const int block_thread_idx = int(threadIdx.x);
  const int thread_idx =
      block_thread_idx < CutlassCollectiveMainloopK128::ThreadCount
          ? block_thread_idx
          : 0;
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
  const int warp_group_thread_idx =
      block_thread_idx % cutlass::NumThreadsPerWarpGroup;
  const int mma_thread_idx =
      block_thread_idx % CutlassCollectiveMainloopK128::ThreadCount;
  const bool lane_predicate = cute::elect_one_sync();

  enum class WarpGroupRole {
    Producer = 0,
    Consumer0 = 1,
    Consumer1 = 2,
  };
  enum class ProducerWarpRole {
    Mainloop = 0,
    Warp1 = 1,
    Epilogue = 2,
    MainloopAux = 3,
  };

  auto warp_group_role = WarpGroupRole(cutlass::canonical_warp_group_idx());
  auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);

  CutlassCollectiveMainloopK128 collective;
  auto sB = cute::make_tensor(cute::make_smem_ptr(shared.tensors.smem_B.begin()),
                              typename CutlassCollectiveMainloopK128::SmemLayoutB{});
  auto sSFB = cute::make_tensor(
      cute::make_smem_ptr(shared.tensors.smem_SFB.begin()),
      typename CutlassCollectiveMainloopK128::SmemLayoutSFB{});
  auto problem_shape_mnkl =
      cute::append<4>(pv_params.problem_shape, cute::Int<1>{});
  auto load_inputs = collective.load_init(problem_shape_mnkl,
                                          pv_params.mainloop);
  auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl] = load_inputs;

  if (warp_group_role == WarpGroupRole::Producer &&
      producer_warp_role == ProducerWarpRole::Mainloop) {
    if (lane_predicate) {
      auto block_tma_b = pv_params.mainloop.tma_load_b.get_slice(0);
      auto block_tma_sfb = pv_params.mainloop.tma_load_sfb.get_slice(0);
      auto gB = gB_nkl(_, _, out_group, _, 0);
      auto gSFB = gSFB_nkl(_, _, out_group, _, 0);
      auto tBgB = block_tma_b.partition_S(gB);
      auto tBsB = block_tma_b.partition_D(sB);
      auto tBgSFB = block_tma_sfb.partition_S(gSFB);
      auto tBsSFB = block_tma_sfb.partition_D(sSFB);

      auto k_tile_iter = cute::make_coord_iterator(
          cute::idx2crd(kv_tile_128, cute::shape<3>(gB_nkl)),
          cute::shape<3>(gB_nkl));
      pipeline.producer_acquire(pipe_write);
      using BarrierType =
          typename CutlassCollectiveMainloopK128::MainloopPipeline::
              ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(pipe_write);
      int write_stage = pipe_write.index();
      copy(pv_params.mainloop.tma_load_b.with(*tma_barrier),
           tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, write_stage));
      copy(pv_params.mainloop.tma_load_sfb.with(*tma_barrier),
           tBgSFB(_, _, _, *k_tile_iter), tBsSFB(_, _, _, write_stage));
      ++pipe_write;
    }
  }

  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    cutlass_smem_pv_k128_mma_no_clear(
        pipeline, pipe_read, accum, 1, mma_thread_idx, shared.tensors);
  }
  __syncthreads();
}
#endif

__device__ __forceinline__ void softmax_quant_scores_128(
    const float* scores,
    float qk_alpha,
    uint8_t* p_packed,
    uint8_t* p_scales,
    float* row_m_out,
    float* row_l_out) {
  const int tid = int(threadIdx.x);

  for (int row = tid; row < kCutlassTileM; row += blockDim.x) {
    float row_m = -INFINITY;
#pragma unroll
    for (int col = 0; col < kCutlassTileN; ++col) {
      const float logit =
          scores[row * kCutlassTileN + col] * qk_alpha * kQkScale;
      row_m = fmaxf(row_m, logit);
    }
    float row_l = 0.0f;
#pragma unroll
    for (int col = 0; col < kCutlassTileN; ++col) {
      const float logit =
          scores[row * kCutlassTileN + col] * qk_alpha * kQkScale;
      row_l += __expf(logit - row_m);
    }
    row_m_out[row] = row_m;
    row_l_out[row] = row_l;
  }
  __syncthreads();

  for (int idx = tid; idx < kCutlassTileM * (kCutlassTileN / 16);
       idx += blockDim.x) {
    const int row = idx / (kCutlassTileN / 16);
    const int scale_group = idx - row * (kCutlassTileN / 16);
    const int local_col = scale_group * 16;
    const float row_m = row_m_out[row];
    float probs[16];
    float vec_max = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float logit =
          scores[row * kCutlassTileN + local_col + i] * qk_alpha * kQkScale;
      const float p = __expf(logit - row_m);
      probs[i] = p;
      vec_max = fmaxf(vec_max, p);
    }
    const uint8_t scale_byte =
        fp32_to_e4m3_byte(fmaxf(kProbGlobalScale * vec_max / 6.0f, 1.0e-8f));
    p_scales[row * (kCutlassTileK128 / 16) + scale_group] = scale_byte;
    const float scale = fmaxf(e4m3_byte_to_fp32(scale_byte), 1.0e-8f);
    const float output_scale = kProbGlobalScale / scale;
    uint8_t* packed = p_packed + row * (kCutlassTileK128 / 2) + local_col / 2;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      packed[i] = fp32_pair_to_e2m1_byte(probs[2 * i] * output_scale,
                                         probs[2 * i + 1] * output_scale);
    }
  }
  __syncthreads();
}

__global__ void persistent_mainloop_owner_layout_smoke_kernel(float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  extern __shared__ __align__(128) char smem[];
  auto& storage = *reinterpret_cast<CutlassFusedTmaQkPv128Storage*>(smem);

  const int thread_idx = int(threadIdx.x);
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
  const int warp_group_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;

  enum class WarpGroupRole {
    Producer = 0,
    Consumer0 = 1,
    Consumer1 = 2,
  };
  enum class ProducerWarpRole {
    Mainloop = 0,
    Warp1 = 1,
    Epilogue = 2,
    MainloopAux = 3,
  };

  const auto warp_group_role =
      WarpGroupRole(cutlass::canonical_warp_group_idx());
  const auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);

  using QkPipeline = typename CutlassCollectiveMainloop::MainloopPipeline;
  typename QkPipeline::Params qk_pipeline_params;
  if (warp_group_role == WarpGroupRole::Producer &&
      (producer_warp_role == ProducerWarpRole::Mainloop ||
       producer_warp_role == ProducerWarpRole::MainloopAux)) {
    qk_pipeline_params.role = QkPipeline::ThreadCategory::Producer;
  }
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    qk_pipeline_params.role = QkPipeline::ThreadCategory::Consumer;
  }
  qk_pipeline_params.is_leader = warp_group_thread_idx == 0;
  qk_pipeline_params.num_consumers = CutlassCollectiveMainloop::ThreadCount;
  qk_pipeline_params.num_producers =
      CutlassCollectiveMainloop::NumProducerThreadEvents;
  qk_pipeline_params.transaction_bytes = 0;
  QkPipeline qk_pipeline(storage.qk.pipeline_storage, qk_pipeline_params,
                         CutlassClusterShape{});
  typename CutlassCollectiveMainloop::PipelineState qk_pipe_read;
  typename CutlassCollectiveMainloop::PipelineState qk_pipe_write =
      cutlass::make_producer_start_state<QkPipeline>();

  __syncthreads();

  auto& pv_shared =
      *reinterpret_cast<typename CutlassCollectiveMainloopK128::SharedStorage*>(
          &storage.qk);
  using PvPipeline = typename CutlassCollectiveMainloopK128::MainloopPipeline;
  typename PvPipeline::Params pv_pipeline_params;
  if (warp_group_role == WarpGroupRole::Producer &&
      (producer_warp_role == ProducerWarpRole::Mainloop ||
       producer_warp_role == ProducerWarpRole::MainloopAux)) {
    pv_pipeline_params.role = PvPipeline::ThreadCategory::Producer;
  }
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    pv_pipeline_params.role = PvPipeline::ThreadCategory::Consumer;
  }
  pv_pipeline_params.is_leader = warp_group_thread_idx == 0;
  pv_pipeline_params.num_consumers = CutlassCollectiveMainloopK128::ThreadCount;
  pv_pipeline_params.num_producers =
      CutlassCollectiveMainloopK128::NumProducerThreadEvents;
  pv_pipeline_params.transaction_bytes = 0;
  PvPipeline pv_pipeline(pv_shared.pipeline_storage, pv_pipeline_params,
                         CutlassClusterShape{});
  typename CutlassCollectiveMainloopK128::PipelineState pv_pipe_read;
  typename CutlassCollectiveMainloopK128::PipelineState pv_pipe_write =
      cutlass::make_producer_start_state<PvPipeline>();

  if (thread_idx == 0) {
    out[0] = 1.0f;
    out[1] = float(sizeof(CutlassFusedTmaQkPv128Storage));
    out[2] = float(sizeof(typename CutlassCollectiveMainloop::SharedStorage));
    out[3] = float(sizeof(typename CutlassCollectiveMainloopK128::SharedStorage));
    out[4] = float(qk_pipe_read.index());
    out[5] = float(qk_pipe_write.index());
    out[6] = float(pv_pipe_read.index());
    out[7] = float(pv_pipe_write.index());
  }
#else
  if (threadIdx.x == 0) {
    out[0] = -1.0f;
  }
#endif
}

__global__ void persistent_mainloop_owner_qk_stage_kernel(
    CUTLASS_GRID_CONSTANT typename CutlassGemmKernel::Params const params,
    float* out_tile,
    uint8_t* p_packed_out,
    uint8_t* p_scales_out,
    float* row_m_out,
    float* row_l_out,
    float qk_alpha,
    int q_tile,
    int kv_tile,
    int write_softmax) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  using cute::_;

  extern __shared__ __align__(128) char smem[];
  auto& owner_storage = *reinterpret_cast<CutlassFusedTmaQkPv128Storage*>(smem);
  auto& storage = owner_storage.qk;
  float* scores = reinterpret_cast<float*>(&storage.tensors);

  const int thread_idx = int(threadIdx.x);
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
  const int warp_group_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
  const int mma_thread_idx = thread_idx % CutlassCollectiveMainloop::ThreadCount;
  const bool lane_predicate = cute::elect_one_sync();
  const uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

  enum class WarpGroupRole {
    Producer = 0,
    Consumer0 = 1,
    Consumer1 = 2,
  };
  enum class ProducerWarpRole {
    Mainloop = 0,
    Warp1 = 1,
    Epilogue = 2,
    MainloopAux = 3,
  };

  const auto warp_group_role =
      WarpGroupRole(cutlass::canonical_warp_group_idx());
  const auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);

  if (warp_idx == 0 && lane_predicate) {
    CutlassCollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
  }

  using MainloopPipeline = typename CutlassCollectiveMainloop::MainloopPipeline;
  typename MainloopPipeline::Params pipeline_params;
  if (warp_group_role == WarpGroupRole::Producer &&
      (producer_warp_role == ProducerWarpRole::Mainloop ||
       producer_warp_role == ProducerWarpRole::MainloopAux)) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
  }
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  pipeline_params.is_leader = warp_group_thread_idx == 0;
  pipeline_params.num_consumers = CutlassCollectiveMainloop::ThreadCount;
  pipeline_params.num_producers =
      CutlassCollectiveMainloop::NumProducerThreadEvents;
  pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;

  MainloopPipeline pipeline(storage.pipeline_storage, pipeline_params,
                            CutlassClusterShape{});
  typename CutlassCollectiveMainloop::PipelineState pipe_read;
  typename CutlassCollectiveMainloop::PipelineState pipe_write =
      cutlass::make_producer_start_state<MainloopPipeline>();

  __syncthreads();

  CutlassCollectiveMainloop collective;
  auto problem_shape_mnkl = cute::append<4>(params.problem_shape, cute::Int<1>{});
  auto load_inputs = collective.load_init(problem_shape_mnkl, params.mainloop);
  auto blk_coord = cute::make_coord(q_tile, kv_tile, cute::_, 0);
  const int k_tile_count = kHeadDim / kCutlassTileK;
  auto k_tile_iter = cute::make_coord_iterator(k_tile_count);

  if (warp_group_role == WarpGroupRole::Producer &&
      producer_warp_role == ProducerWarpRole::Mainloop) {
    collective.load(params.mainloop, pipeline, pipe_write, load_inputs,
                    blk_coord, k_tile_iter, k_tile_count,
                    cutlass::canonical_lane_idx(), block_rank_in_cluster,
                    storage.tensors);
  }

  auto tiled_mma = typename CutlassCollectiveMainloop::TiledMma{};
  auto accum = cute::partition_fragment_C(tiled_mma,
                                          cute::take<0, 2>(CutlassThreadBlockShape{}));
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    collective.mma(pipeline, pipe_read, accum, k_tile_count, mma_thread_idx,
                   storage.tensors, params.mainloop);
  }
  if (warp_group_role == WarpGroupRole::Producer &&
      producer_warp_role == ProducerWarpRole::Mainloop) {
    collective.load_tail(pipeline, pipe_write);
  }
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    collective.mma_tail(pipeline, pipe_read, k_tile_count);
  }

  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    auto thread_mma = tiled_mma.get_thread_slice(mma_thread_idx);
    auto cC = cute::make_identity_tensor(
        cute::take<0, 2>(CutlassThreadBlockShape{}));
    auto tCcC = thread_mma.partition_C(cC);
    for (int i = 0; i < cute::size(accum); ++i) {
      auto coord = tCcC(i);
      const int row = int(cute::get<0>(coord));
      const int col = int(cute::get<1>(coord));
      if (row < kCutlassTileM && col < kCutlassTileN) {
        const float value = accum(i);
        if (out_tile != nullptr) {
          out_tile[row * kCutlassTileN + col] = value;
        }
        if (write_softmax != 0) {
          scores[row * kCutlassTileN + col] = value;
        }
      }
    }
  }
  if (write_softmax != 0) {
    __syncthreads();
    softmax_quant_scores_128(scores, qk_alpha, owner_storage.p_packed,
                             owner_storage.p_scales, owner_storage.row_m,
                             owner_storage.row_l);
    const int tid = int(threadIdx.x);
    for (int idx = tid; idx < kCutlassTileM * (kCutlassTileK128 / 2);
         idx += blockDim.x) {
      p_packed_out[idx] = owner_storage.p_packed[idx];
    }
    for (int idx = tid; idx < kCutlassTileM * (kCutlassTileK128 / 16);
         idx += blockDim.x) {
      p_scales_out[idx] = owner_storage.p_scales[idx];
    }
    for (int idx = tid; idx < kCutlassTileM; idx += blockDim.x) {
      row_m_out[idx] = owner_storage.row_m[idx];
      row_l_out[idx] = owner_storage.row_l[idx];
    }
  }
#else
  if (threadIdx.x == 0) {
    if (out_tile != nullptr) {
      out_tile[0] = -1.0f;
    }
    if (row_m_out != nullptr) {
      row_m_out[0] = -1.0f;
    }
  }
#endif
}

__global__ void persistent_mainloop_owner_pv_group_stage_kernel(
    CUTLASS_GRID_CONSTANT typename CutlassGemmKernel::Params const qk_params,
    CUTLASS_GRID_CONSTANT typename CutlassGemmKernelK128::Params const pv_params,
    float* out_group,
    uint8_t* p_packed_out,
    uint8_t* p_scales_out,
    float* row_m_out,
    float* row_l_out,
    float qk_alpha,
    int q_tile,
    int kv_tile,
    int out_group_idx) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  extern __shared__ __align__(128) char smem[];
  auto& owner_storage = *reinterpret_cast<CutlassFusedTmaQkPv128Storage*>(smem);
  float* scores = reinterpret_cast<float*>(&owner_storage.qk.tensors);
  const int tid = int(threadIdx.x);

  qk_cutlass_collective_tile_body(qk_params, owner_storage.qk, scores, q_tile,
                                  kv_tile, kCutlassTileN);
  softmax_quant_scores_128(scores, qk_alpha, owner_storage.p_packed,
                           owner_storage.p_scales, owner_storage.row_m,
                           owner_storage.row_l);

  for (int idx = tid; idx < kCutlassTileM * (kCutlassTileK128 / 2);
       idx += blockDim.x) {
    p_packed_out[idx] = owner_storage.p_packed[idx];
  }
  for (int idx = tid; idx < kCutlassTileM * (kCutlassTileK128 / 16);
       idx += blockDim.x) {
    p_scales_out[idx] = owner_storage.p_scales[idx];
  }
  for (int idx = tid; idx < kCutlassTileM; idx += blockDim.x) {
    row_m_out[idx] = owner_storage.row_m[idx];
    row_l_out[idx] = owner_storage.row_l[idx];
  }
  __syncthreads();

  auto& pv_shared =
      *reinterpret_cast<typename CutlassCollectiveMainloopK128::SharedStorage*>(
          &owner_storage.qk);
  cutlass_smem_pv_k128_stage_p_body(
      pv_shared, owner_storage.p_packed, owner_storage.p_scales);
  cutlass_smem_pv_k128_tma_v_group_body(
      pv_params, pv_shared, out_group, kCutlassTileN, 0, out_group_idx,
      kv_tile);
#else
  if (threadIdx.x == 0) {
    out_group[0] = -1.0f;
    row_m_out[0] = -1.0f;
  }
#endif
}

__global__ void persistent_mainloop_owner_full_tile_stage_kernel(
    CUTLASS_GRID_CONSTANT typename CutlassGemmKernel::Params const qk_params,
    CUTLASS_GRID_CONSTANT typename CutlassGemmKernelK128::Params const pv_params,
    float* out,
    float qk_alpha,
    float pv_alpha,
    int q_tile,
    int kv_tile) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  extern __shared__ __align__(128) char smem[];
  auto& owner_storage = *reinterpret_cast<CutlassFusedTmaQkPv128Storage*>(smem);
  float* scores = reinterpret_cast<float*>(&owner_storage.qk.tensors);
  const int tid = int(threadIdx.x);

  qk_cutlass_collective_tile_body(qk_params, owner_storage.qk, scores, q_tile,
                                  kv_tile, kCutlassTileN);
  softmax_quant_scores_128(scores, qk_alpha, owner_storage.p_packed,
                           owner_storage.p_scales, owner_storage.row_m,
                           owner_storage.row_l);

  auto& pv_shared =
      *reinterpret_cast<typename CutlassCollectiveMainloopK128::SharedStorage*>(
          &owner_storage.qk);
  cutlass_smem_pv_k128_stage_p_body(
      pv_shared, owner_storage.p_packed, owner_storage.p_scales);
#pragma unroll
  for (int out_group_idx = 0; out_group_idx < kHeadDim / kCutlassTileN;
       ++out_group_idx) {
    cutlass_smem_pv_k128_tma_v_group_body(
        pv_params, pv_shared, out, kHeadDim, out_group_idx * kCutlassTileN,
        out_group_idx, kv_tile);
  }

  for (int idx = tid; idx < kCutlassTileM * kHeadDim; idx += blockDim.x) {
    const int row = idx / kHeadDim;
    out[idx] *= pv_alpha /
                (kProbGlobalScale * fmaxf(owner_storage.row_l[row], 1.0e-20f));
  }
#else
  if (threadIdx.x == 0) {
    out[0] = -1.0f;
  }
#endif
}

__global__ __launch_bounds__(384, 1)
void persistent_mainloop_owner_group_online_stage_kernel(
    CUTLASS_GRID_CONSTANT typename CutlassGemmKernel::Params const qk_params,
    CUTLASS_GRID_CONSTANT typename CutlassGemmKernelK128::Params const pv_params,
    float* out_group,
    float qk_alpha,
    float pv_alpha,
    int q_tile,
    int kv_tile_start,
    int num_kv_tiles,
    int out_group_idx) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  extern __shared__ __align__(128) char smem[];
  auto& owner_storage = *reinterpret_cast<CutlassFusedTmaQkPv128Storage*>(smem);
  float* scores = reinterpret_cast<float*>(&owner_storage.qk.tensors);
  const int tid = int(threadIdx.x);
  const float pv_base_scale = pv_alpha / kProbGlobalScale;

  for (int tile = 0; tile < num_kv_tiles; ++tile) {
    const int kv_tile = kv_tile_start + tile;
    qk_cutlass_collective_tile_body(qk_params, owner_storage.qk, scores, q_tile,
                                    kv_tile, kCutlassTileN);
    softmax_quant_scores_128(scores, qk_alpha, owner_storage.p_packed,
                             owner_storage.p_scales, owner_storage.row_m,
                             owner_storage.row_l);

    if (tile == 0) {
      for (int row = tid; row < kCutlassTileM; row += blockDim.x) {
        owner_storage.global_m[row] = owner_storage.row_m[row];
        owner_storage.global_l[row] = owner_storage.row_l[row];
        owner_storage.old_scale[row] = 0.0f;
        owner_storage.tile_scale[row] = 1.0f;
      }
    } else {
      for (int row = tid; row < kCutlassTileM; row += blockDim.x) {
        const float old_m = owner_storage.global_m[row];
        const float tile_m = owner_storage.row_m[row];
        const float new_m = fmaxf(old_m, tile_m);
        const float old_scale = __expf(old_m - new_m);
        const float tile_scale = __expf(tile_m - new_m);
        owner_storage.global_m[row] = new_m;
        owner_storage.global_l[row] =
            owner_storage.global_l[row] * old_scale +
            owner_storage.row_l[row] * tile_scale;
        owner_storage.old_scale[row] = old_scale;
        owner_storage.tile_scale[row] = tile_scale;
      }
    }
    __syncthreads();

    auto& pv_shared =
        *reinterpret_cast<typename CutlassCollectiveMainloopK128::SharedStorage*>(
            &owner_storage.qk);
    cutlass_smem_pv_k128_stage_p_body(
        pv_shared, owner_storage.p_packed, owner_storage.p_scales);
    cutlass_smem_pv_k128_tma_v_group_body(
        pv_params, pv_shared, out_group, kCutlassTileN, 0, out_group_idx,
        kv_tile, owner_storage.old_scale, owner_storage.tile_scale,
        pv_base_scale, tile == 0 ? 0 : 1);
  }

  for (int idx = tid; idx < kCutlassTileM * kCutlassTileN; idx += blockDim.x) {
    const int row = idx / kCutlassTileN;
    out_group[idx] /= fmaxf(owner_storage.global_l[row], 1.0e-20f);
  }
#else
  if (threadIdx.x == 0) {
    out_group[0] = -1.0f;
  }
#endif
}

__global__ __launch_bounds__(384, 1)
void persistent_mainloop_owner_group_online_register_stage_kernel(
    CUTLASS_GRID_CONSTANT typename CutlassGemmKernel::Params const qk_params,
    CUTLASS_GRID_CONSTANT typename CutlassGemmKernelK128::Params const pv_params,
    float* out_group,
    float qk_alpha,
    float pv_alpha,
    int q_tile,
    int kv_tile_start,
    int num_kv_tiles,
    int out_group_idx) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  extern __shared__ __align__(128) char smem[];
  auto& owner_storage = *reinterpret_cast<CutlassFusedTmaQkPv128Storage*>(smem);
  float* scores = reinterpret_cast<float*>(&owner_storage.qk.tensors);
  const int tid = int(threadIdx.x);
  const int block_thread_idx = int(threadIdx.x);
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
  const int warp_group_thread_idx =
      block_thread_idx % cutlass::NumThreadsPerWarpGroup;
  const int warp_group_idx = cutlass::canonical_warp_group_idx();
  const bool is_consumer = warp_group_idx == 1 || warp_group_idx == 2;
  const int mma_thread_idx =
      block_thread_idx % CutlassCollectiveMainloopK128::ThreadCount;
  const bool lane_predicate = cute::elect_one_sync();
  const float pv_base_scale = pv_alpha / kProbGlobalScale;

  enum class WarpGroupRole {
    Producer = 0,
    Consumer0 = 1,
    Consumer1 = 2,
  };
  enum class ProducerWarpRole {
    Mainloop = 0,
    Warp1 = 1,
    Epilogue = 2,
    MainloopAux = 3,
  };

  const auto warp_group_role = WarpGroupRole(warp_group_idx);
  const auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);

  auto tiled_mma = typename CutlassCollectiveMainloopK128::TiledMma{};
  auto accum = cute::partition_fragment_C(
      tiled_mma, cute::take<0, 2>(CutlassThreadBlockShapeK128{}));
  auto& pv_shared =
      *reinterpret_cast<typename CutlassCollectiveMainloopK128::SharedStorage*>(
          &owner_storage.qk);

  if (warp_idx == 0 && lane_predicate) {
    CutlassCollectiveMainloopK128::prefetch_tma_descriptors(
        pv_params.mainloop);
  }

  using PvPipeline = typename CutlassCollectiveMainloopK128::MainloopPipeline;
  typename PvPipeline::Params pv_pipeline_params;
  if (warp_group_role == WarpGroupRole::Producer &&
      (producer_warp_role == ProducerWarpRole::Mainloop ||
       producer_warp_role == ProducerWarpRole::MainloopAux)) {
    pv_pipeline_params.role = PvPipeline::ThreadCategory::Producer;
  }
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    pv_pipeline_params.role = PvPipeline::ThreadCategory::Consumer;
  }
  pv_pipeline_params.is_leader = warp_group_thread_idx == 0;
  pv_pipeline_params.num_consumers = CutlassCollectiveMainloopK128::ThreadCount;
  pv_pipeline_params.num_producers =
      CutlassCollectiveMainloopK128::NumProducerThreadEvents;
  pv_pipeline_params.transaction_bytes =
      pv_params.mainloop.tma_transaction_bytes_nk;
  PvPipeline pv_pipeline(owner_storage.pv_pipeline_storage, pv_pipeline_params,
                         CutlassClusterShape{});
  typename CutlassCollectiveMainloopK128::PipelineState pv_pipe_read;
  typename CutlassCollectiveMainloopK128::PipelineState pv_pipe_write =
      cutlass::make_producer_start_state<PvPipeline>();
  __syncthreads();

  for (int tile = 0; tile < num_kv_tiles; ++tile) {
    const int kv_tile = kv_tile_start + tile;
    qk_cutlass_collective_tile_body(qk_params, owner_storage.qk, scores, q_tile,
                                    kv_tile, kCutlassTileN);
    softmax_quant_scores_128(scores, qk_alpha, owner_storage.p_packed,
                             owner_storage.p_scales, owner_storage.row_m,
                             owner_storage.row_l);

    if (tile == 0) {
      for (int row = tid; row < kCutlassTileM; row += blockDim.x) {
        owner_storage.global_m[row] = owner_storage.row_m[row];
        owner_storage.global_l[row] = owner_storage.row_l[row];
        owner_storage.old_scale[row] = 0.0f;
        owner_storage.tile_scale[row] = 1.0f;
      }
    } else {
      for (int row = tid; row < kCutlassTileM; row += blockDim.x) {
        const float old_m = owner_storage.global_m[row];
        const float tile_m = owner_storage.row_m[row];
        const float new_m = fmaxf(old_m, tile_m);
        const float old_scale = __expf(old_m - new_m);
        const float tile_scale = __expf(tile_m - new_m);
        owner_storage.global_m[row] = new_m;
        owner_storage.global_l[row] =
            owner_storage.global_l[row] * old_scale +
            owner_storage.row_l[row] * tile_scale;
        owner_storage.old_scale[row] = old_scale;
        owner_storage.tile_scale[row] = tile_scale;
      }
    }
    __syncthreads();

    auto& pv_shared =
        *reinterpret_cast<typename CutlassCollectiveMainloopK128::SharedStorage*>(
            &owner_storage.qk);
    const int pv_stage = tile % PvPipeline::Stages;
    if (tile == 0) {
      cutlass_smem_pv_k128_stage_p_body(
          pv_shared, owner_storage.p_packed, owner_storage.p_scales,
          nullptr, 1.0f, pv_stage);
    } else {
      cutlass_smem_pv_k128_stage_p_body(
          pv_shared, owner_storage.p_packed, owner_storage.p_scales,
          owner_storage.tile_scale, 1.0f, pv_stage);
    }
    if (is_consumer) {
      cutlass_smem_pv_k128_scale_or_clear_accum(
          accum, owner_storage.old_scale, mma_thread_idx, tile == 0 ? 1 : 0);
    }
    cutlass_smem_pv_k128_tma_v_group_accum_persistent_step(
        pv_params, pv_shared, pv_pipeline, pv_pipe_read, pv_pipe_write,
        accum, out_group_idx, kv_tile);
  }

  if (warp_group_role == WarpGroupRole::Producer &&
      producer_warp_role == ProducerWarpRole::Mainloop) {
    CutlassCollectiveMainloopK128 collective;
    collective.load_tail(pv_pipeline, pv_pipe_write);
  }

  if (is_consumer) {
    auto thread_mma = tiled_mma.get_thread_slice(mma_thread_idx);
    auto cC = cute::make_identity_tensor(
        cute::take<0, 2>(CutlassThreadBlockShapeK128{}));
    auto tCcC = thread_mma.partition_C(cC);
    for (int i = 0; i < cute::size(accum); ++i) {
      auto coord = tCcC(i);
      const int row = int(cute::get<0>(coord));
      const int col = int(cute::get<1>(coord));
      if (row < kCutlassTileM && col < kCutlassTileN) {
        out_group[row * kCutlassTileN + col] =
            accum(i) * pv_base_scale /
            fmaxf(owner_storage.global_l[row], 1.0e-20f);
      }
    }
  }
#else
  if (threadIdx.x == 0) {
    out_group[0] = -1.0f;
  }
#endif
}

__global__ __launch_bounds__(384, 1)
void persistent_mainloop_owner_group_online_register_q_stage_kernel(
    const uint8_t* q_packed,
    const uint8_t* q_scales,
    CUTLASS_GRID_CONSTANT typename CutlassGemmKernel::Params const qk_params,
    CUTLASS_GRID_CONSTANT typename CutlassGemmKernelK128::Params const pv_params,
    float* out_group,
    float qk_alpha,
    float pv_alpha,
    int q_tile,
    int kv_tile_start,
    int num_kv_tiles,
    int out_group_idx) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  using cute::_;
  extern __shared__ __align__(128) char smem[];
  auto& owner_storage = *reinterpret_cast<CutlassFusedTmaQkPv128Storage*>(smem);
  float* scores = reinterpret_cast<float*>(&owner_storage.qk.tensors);
  const int tid = int(threadIdx.x);
  const int block_thread_idx = int(threadIdx.x);
  const int warp_idx = cutlass::canonical_warp_idx_sync();
  const int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
  const int warp_group_thread_idx =
      block_thread_idx % cutlass::NumThreadsPerWarpGroup;
  const int warp_group_idx = cutlass::canonical_warp_group_idx();
  const bool is_consumer = warp_group_idx == 1 || warp_group_idx == 2;
  const int qk_mma_thread_idx =
      block_thread_idx % CutlassCollectiveMainloop::ThreadCount;
  const int pv_mma_thread_idx =
      block_thread_idx % CutlassCollectiveMainloopK128::ThreadCount;
  const bool lane_predicate = cute::elect_one_sync();
  const float pv_base_scale = pv_alpha / kProbGlobalScale;

  enum class WarpGroupRole {
    Producer = 0,
    Consumer0 = 1,
    Consumer1 = 2,
  };
  enum class ProducerWarpRole {
    Mainloop = 0,
    Warp1 = 1,
    Epilogue = 2,
    MainloopAux = 3,
  };

  const auto warp_group_role = WarpGroupRole(warp_group_idx);
  const auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);

  CutlassCollectiveMainloop qk_collective;
  auto qk_tiled_mma = typename CutlassCollectiveMainloop::TiledMma{};
  auto qk_thread_mma = qk_tiled_mma.get_thread_slice(qk_mma_thread_idx);
  auto qk_sA = cute::make_tensor(
      cute::make_smem_ptr(owner_storage.qk.tensors.smem_A.begin()),
      typename CutlassCollectiveMainloop::SmemLayoutA{});
  auto qk_sB = cute::make_tensor(
      cute::make_smem_ptr(owner_storage.qk.tensors.smem_B.begin()),
      typename CutlassCollectiveMainloop::SmemLayoutB{});
  auto qk_sSFA = cute::make_tensor(
      cute::make_smem_ptr(owner_storage.qk.tensors.smem_SFA.begin()),
      typename CutlassCollectiveMainloop::SmemLayoutSFA{});
  auto qk_sSFB = cute::make_tensor(
      cute::make_smem_ptr(owner_storage.qk.tensors.smem_SFB.begin()),
      typename CutlassCollectiveMainloop::SmemLayoutSFB{});

  auto q_frag0 = qk_thread_mma.partition_fragment_A(
      qk_sA(_, _, cute::Int<0>{}));
  auto q_frag1 = qk_thread_mma.partition_fragment_A(
      qk_sA(_, _, cute::Int<0>{}));
  auto q_scale_frag0 =
      qk_collective.partition_fragment_SFA(qk_sSFA(_, _, cute::Int<0>{}),
                                           qk_thread_mma);
  auto q_scale_frag1 =
      qk_collective.partition_fragment_SFA(qk_sSFA(_, _, cute::Int<0>{}),
                                           qk_thread_mma);

  auto qk_smem_tiled_copy_A = cute::make_tiled_copy_A(
      typename CutlassCollectiveMainloop::SmemCopyAtomA{}, qk_tiled_mma);
  auto qk_smem_thr_copy_A =
      qk_smem_tiled_copy_A.get_thread_slice(qk_mma_thread_idx);
  auto qk_tCsA = qk_smem_thr_copy_A.partition_S(
      cute::as_position_independent_swizzle_tensor(qk_sA));
  auto q_frag0_copy_view = qk_smem_thr_copy_A.retile_D(q_frag0);
  auto q_frag1_copy_view = qk_smem_thr_copy_A.retile_D(q_frag1);
  auto qk_cA = cute::make_identity_tensor(cute::make_shape(
      cute::Int<kCutlassTileM>{}, cute::Int<kCutlassTileK>{},
      cute::Int<1>{}));
  auto qk_tAsA_prod = qk_smem_thr_copy_A.partition_D(qk_sA);
  auto qk_tAcA_prod = qk_smem_thr_copy_A.partition_D(qk_cA);

  auto qk_tile_shape_mnk = cute::tile_shape(qk_tiled_mma);
  auto qk_smem_tiled_copy_SFA = cute::make_tiled_copy_impl(
      typename CutlassCollectiveMainloop::SmemCopyAtomSFA{},
      qk_collective.get_layoutSFA_TV(qk_tiled_mma),
      cute::make_shape(cute::size<0>(qk_tile_shape_mnk),
                       cute::size<2>(qk_tile_shape_mnk)));
  auto qk_smem_thr_copy_SFA =
      qk_smem_tiled_copy_SFA.get_thread_slice(qk_mma_thread_idx);
  auto qk_tCsSFA = qk_smem_thr_copy_SFA.partition_S(
      cute::as_position_independent_swizzle_tensor(qk_sSFA));
  auto q_scale_frag0_copy_view =
      qk_smem_thr_copy_SFA.retile_D(q_scale_frag0);
  auto q_scale_frag1_copy_view =
      qk_smem_thr_copy_SFA.retile_D(q_scale_frag1);

  auto write_partitioned_q_fp4 = [&](auto tDst,
                                     auto tCoord,
                                     int source_col_base) {
    for (int i = 0; i < int(cute::size(tDst)); ++i) {
      auto coord = tCoord(i);
      const int row = int(cute::get<0>(coord));
      const int k = int(cute::get<1>(coord));
      const int source_col = source_col_base + k;
      const uint8_t byte =
          q_packed[(q_tile * kCutlassTileM + row) * kPackedHeadDim +
                   (source_col >> 1)];
      const uint8_t code =
          static_cast<uint8_t>((source_col & 1) ? ((byte >> 4) & 0x0f)
                                                : (byte & 0x0f));
      tDst(i) = cute::uint4_t(code);
    }
  };

  auto stage_q_register_fragment = [&](int source_col_base,
                                       auto& q_frag_copy_view,
                                       auto& q_scale_frag_copy_view) {
    uint8_t* smem_a_bytes =
        cute::recast_ptr<uint8_t>(owner_storage.qk.tensors.smem_A.begin());
    constexpr int kSmemABytes =
        (cute::cosize_v<typename CutlassCollectiveMainloop::SmemLayoutA> + 1) /
        2;
    for (int idx = block_thread_idx; idx < kSmemABytes; idx += blockDim.x) {
      smem_a_bytes[idx] = 0;
    }
    __syncthreads();

    auto K_BLOCK_MAX_PROD = cute::size<2>(qk_tAsA_prod);
    if (is_consumer) {
      cute::for_each(cute::make_int_sequence<K_BLOCK_MAX_PROD>{},
                     [&](auto k_block) {
        write_partitioned_q_fp4(qk_tAsA_prod(_, _, k_block, cute::Int<0>{}),
                                qk_tAcA_prod(_, _, k_block, cute::Int<0>{}),
                                source_col_base);
      });
    }
    for (int idx = block_thread_idx;
         idx < kCutlassTileM * kCutlassTileK / 2;
         idx += blockDim.x) {
      const int row = idx / (kCutlassTileK / 2);
      const int packed_k = idx - row * (kCutlassTileK / 2);
      const int k0 = 2 * packed_k;
      const int q_scale_col = (source_col_base + k0) >> 4;
      qk_sSFA(row, k0, cute::Int<0>{}) =
          make_ue4m3_raw(q_scales[(q_tile * kCutlassTileM + row) *
                                      kScaleCols +
                                  q_scale_col]);
    }
    __syncthreads();

    if (is_consumer) {
      auto K_BLOCK_MAX = cute::size<2>(q_frag_copy_view);
      cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
        cute::copy(qk_smem_tiled_copy_A,
                   qk_tCsA(_, _, k_block, cute::Int<0>{}),
                   q_frag_copy_view(_, _, k_block));
        using MMAOp = typename CutlassCollectiveMainloop::TiledMma::MMA_Op;
        fp4_shift_A(MMAOp{}, q_frag_copy_view(_, _, k_block));
        cute::copy(qk_tCsSFA(_, _, k_block, cute::Int<0>{}),
                   q_scale_frag_copy_view(_, _, k_block));
      });
    }
    __syncthreads();
  };

  stage_q_register_fragment(0, q_frag0_copy_view, q_scale_frag0_copy_view);
  stage_q_register_fragment(kCutlassTileK, q_frag1_copy_view,
                            q_scale_frag1_copy_view);

  if (warp_idx == 0 && lane_predicate) {
    cute::prefetch_tma_descriptor(
        qk_params.mainloop.tma_load_b.get_tma_descriptor());
    cute::prefetch_tma_descriptor(
        qk_params.mainloop.tma_load_sfb.get_tma_descriptor());
    CutlassCollectiveMainloopK128::prefetch_tma_descriptors(
        pv_params.mainloop);
  }

  using QkPipeline = typename CutlassCollectiveMainloop::MainloopPipeline;
  typename QkPipeline::Params qk_pipeline_params;
  if (warp_group_role == WarpGroupRole::Producer &&
      (producer_warp_role == ProducerWarpRole::Mainloop ||
       producer_warp_role == ProducerWarpRole::MainloopAux)) {
    qk_pipeline_params.role = QkPipeline::ThreadCategory::Producer;
  }
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    qk_pipeline_params.role = QkPipeline::ThreadCategory::Consumer;
  }
  qk_pipeline_params.is_leader = warp_group_thread_idx == 0;
  qk_pipeline_params.num_consumers = CutlassCollectiveMainloop::ThreadCount;
  qk_pipeline_params.num_producers =
      CutlassCollectiveMainloop::NumProducerThreadEvents;
  qk_pipeline_params.transaction_bytes =
      qk_params.mainloop.tma_transaction_bytes_nk;
  QkPipeline qk_pipeline(owner_storage.qk.pipeline_storage, qk_pipeline_params,
                         CutlassClusterShape{});
  typename CutlassCollectiveMainloop::PipelineState qk_pipe_read;
  typename CutlassCollectiveMainloop::PipelineState qk_pipe_write =
      cutlass::make_producer_start_state<QkPipeline>();

  using PvPipeline = typename CutlassCollectiveMainloopK128::MainloopPipeline;
  typename PvPipeline::Params pv_pipeline_params;
  if (warp_group_role == WarpGroupRole::Producer &&
      (producer_warp_role == ProducerWarpRole::Mainloop ||
       producer_warp_role == ProducerWarpRole::MainloopAux)) {
    pv_pipeline_params.role = PvPipeline::ThreadCategory::Producer;
  }
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    pv_pipeline_params.role = PvPipeline::ThreadCategory::Consumer;
  }
  pv_pipeline_params.is_leader = warp_group_thread_idx == 0;
  pv_pipeline_params.num_consumers = CutlassCollectiveMainloopK128::ThreadCount;
  pv_pipeline_params.num_producers =
      CutlassCollectiveMainloopK128::NumProducerThreadEvents;
  pv_pipeline_params.transaction_bytes =
      pv_params.mainloop.tma_transaction_bytes_nk;
  PvPipeline pv_pipeline(owner_storage.pv_pipeline_storage, pv_pipeline_params,
                         CutlassClusterShape{});
  typename CutlassCollectiveMainloopK128::PipelineState pv_pipe_read;
  typename CutlassCollectiveMainloopK128::PipelineState pv_pipe_write =
      cutlass::make_producer_start_state<PvPipeline>();
  __syncthreads();

  auto qk_problem_shape_mnkl =
      cute::append<4>(qk_params.problem_shape, cute::Int<1>{});
  auto qk_load_inputs =
      qk_collective.load_init(qk_problem_shape_mnkl, qk_params.mainloop);
  auto [qk_gA_mkl, qk_gB_nkl, qk_gSFA_mkl, qk_gSFB_nkl] = qk_load_inputs;
  auto qk_block_tma_b = qk_params.mainloop.tma_load_b.get_slice(0);
  auto qk_block_tma_sfb = qk_params.mainloop.tma_load_sfb.get_slice(0);
  auto qk_tBsB = qk_block_tma_b.partition_D(qk_sB);
  auto qk_tBsSFB = qk_block_tma_sfb.partition_D(qk_sSFB);

  auto pv_tiled_mma = typename CutlassCollectiveMainloopK128::TiledMma{};
  auto pv_accum = cute::partition_fragment_C(
      pv_tiled_mma, cute::take<0, 2>(CutlassThreadBlockShapeK128{}));
  auto& pv_shared =
      *reinterpret_cast<typename CutlassCollectiveMainloopK128::SharedStorage*>(
          &owner_storage.qk);

  for (int tile = 0; tile < num_kv_tiles; ++tile) {
    const int kv_tile = kv_tile_start + tile;
    auto qk_accum = cute::partition_fragment_C(
        qk_tiled_mma, cute::take<0, 2>(CutlassThreadBlockShape{}));
    if (is_consumer) {
      cute::clear(qk_accum);
    }

    auto load_qk_b_tile = [&](int k_outer) {
      if (warp_group_role == WarpGroupRole::Producer &&
          producer_warp_role == ProducerWarpRole::Mainloop) {
        if (lane_predicate) {
          auto gB = qk_gB_nkl(_, _, kv_tile, _, 0);
          auto gSFB = qk_gSFB_nkl(_, _, kv_tile, _, 0);
          auto qk_tBgB = qk_block_tma_b.partition_S(gB);
          auto qk_tBgSFB = qk_block_tma_sfb.partition_S(gSFB);
          auto k_tile_iter = cute::make_coord_iterator(
              cute::idx2crd(k_outer, cute::shape<3>(qk_gB_nkl)),
              cute::shape<3>(qk_gB_nkl));
          qk_pipeline.producer_acquire(qk_pipe_write);
          using BarrierType = typename QkPipeline::ProducerBarrierType;
          BarrierType* tma_barrier =
              qk_pipeline.producer_get_barrier(qk_pipe_write);
          const int write_stage = qk_pipe_write.index();
          cute::copy(qk_params.mainloop.tma_load_b.with(*tma_barrier),
                     qk_tBgB(_, _, _, *k_tile_iter),
                     qk_tBsB(_, _, _, write_stage));
          cute::copy(qk_params.mainloop.tma_load_sfb.with(*tma_barrier),
                     qk_tBgSFB(_, _, _, *k_tile_iter),
                     qk_tBsSFB(_, _, _, write_stage));
          ++qk_pipe_write;
        }
      }
    };

    load_qk_b_tile(0);
    if (is_consumer) {
      cutlass_qk_tma_k_mma_register_q_stage(
          qk_pipeline, qk_pipe_read, q_frag0, q_scale_frag0, qk_accum,
          qk_mma_thread_idx, owner_storage.qk.tensors);
    }
    load_qk_b_tile(1);
    if (is_consumer) {
      cutlass_qk_tma_k_mma_register_q_stage(
          qk_pipeline, qk_pipe_read, q_frag1, q_scale_frag1, qk_accum,
          qk_mma_thread_idx, owner_storage.qk.tensors);

      auto qk_cC = cute::make_identity_tensor(
          cute::take<0, 2>(CutlassThreadBlockShape{}));
      auto qk_tCcC = qk_thread_mma.partition_C(qk_cC);
      for (int i = 0; i < cute::size(qk_accum); ++i) {
        auto coord = qk_tCcC(i);
        const int row = int(cute::get<0>(coord));
        const int col = int(cute::get<1>(coord));
        if (row < kCutlassTileM && col < kCutlassTileN) {
          scores[row * kCutlassTileN + col] = qk_accum(i);
        }
      }
    }
    __syncthreads();

    softmax_quant_scores_128(scores, qk_alpha, owner_storage.p_packed,
                             owner_storage.p_scales, owner_storage.row_m,
                             owner_storage.row_l);

    if (tile == 0) {
      for (int row = tid; row < kCutlassTileM; row += blockDim.x) {
        owner_storage.global_m[row] = owner_storage.row_m[row];
        owner_storage.global_l[row] = owner_storage.row_l[row];
        owner_storage.old_scale[row] = 0.0f;
        owner_storage.tile_scale[row] = 1.0f;
      }
    } else {
      for (int row = tid; row < kCutlassTileM; row += blockDim.x) {
        const float old_m = owner_storage.global_m[row];
        const float tile_m = owner_storage.row_m[row];
        const float new_m = fmaxf(old_m, tile_m);
        const float old_scale = __expf(old_m - new_m);
        const float tile_scale = __expf(tile_m - new_m);
        owner_storage.global_m[row] = new_m;
        owner_storage.global_l[row] =
            owner_storage.global_l[row] * old_scale +
            owner_storage.row_l[row] * tile_scale;
        owner_storage.old_scale[row] = old_scale;
        owner_storage.tile_scale[row] = tile_scale;
      }
    }
    __syncthreads();

    const int pv_stage = tile % PvPipeline::Stages;
    if (tile == 0) {
      cutlass_smem_pv_k128_stage_p_body(
          pv_shared, owner_storage.p_packed, owner_storage.p_scales, nullptr,
          1.0f, pv_stage);
    } else {
      cutlass_smem_pv_k128_stage_p_body(
          pv_shared, owner_storage.p_packed, owner_storage.p_scales,
          owner_storage.tile_scale, 1.0f, pv_stage);
    }
    if (is_consumer) {
      cutlass_smem_pv_k128_scale_or_clear_accum(
          pv_accum, owner_storage.old_scale, pv_mma_thread_idx,
          tile == 0 ? 1 : 0);
    }
    cutlass_smem_pv_k128_tma_v_group_accum_persistent_step(
        pv_params, pv_shared, pv_pipeline, pv_pipe_read, pv_pipe_write,
        pv_accum, out_group_idx, kv_tile);
  }

  if (warp_group_role == WarpGroupRole::Producer &&
      producer_warp_role == ProducerWarpRole::Mainloop) {
    qk_collective.load_tail(qk_pipeline, qk_pipe_write);
    CutlassCollectiveMainloopK128 pv_collective;
    pv_collective.load_tail(pv_pipeline, pv_pipe_write);
  }

  if (is_consumer) {
    auto pv_thread_mma = pv_tiled_mma.get_thread_slice(pv_mma_thread_idx);
    auto pv_cC = cute::make_identity_tensor(
        cute::take<0, 2>(CutlassThreadBlockShapeK128{}));
    auto pv_tCcC = pv_thread_mma.partition_C(pv_cC);
    for (int i = 0; i < cute::size(pv_accum); ++i) {
      auto coord = pv_tCcC(i);
      const int row = int(cute::get<0>(coord));
      const int col = int(cute::get<1>(coord));
      if (row < kCutlassTileM && col < kCutlassTileN) {
        out_group[row * kCutlassTileN + col] =
            pv_accum(i) * pv_base_scale /
            fmaxf(owner_storage.global_l[row], 1.0e-20f);
      }
    }
  }
#else
  if (threadIdx.x == 0) {
    out_group[0] = -1.0f;
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

__global__ void reduce_full_width_split_kernel(const float* partial_out,
                                               const float* partial_m,
                                               const float* partial_l,
                                               float* out,
                                               int num_splits) {
  const int row = int(blockIdx.x);
  const int q_tile = row / kCutlassTileM;
  const int row_local = row - q_tile * kCutlassTileM;
  const int col = int(threadIdx.x);
  if (col >= kHeadDim) {
    return;
  }
  float m = -INFINITY;
  for (int split = 0; split < num_splits; ++split) {
    const int partial_index = q_tile * num_splits + split;
    m = fmaxf(m, partial_m[partial_index * kCutlassTileM + row_local]);
  }
  float l = 0.0f;
  float o = 0.0f;
  for (int split = 0; split < num_splits; ++split) {
    const int partial_index = q_tile * num_splits + split;
    const float split_m =
        partial_m[partial_index * kCutlassTileM + row_local];
    const float alpha = __expf(split_m - m);
    l += partial_l[partial_index * kCutlassTileM + row_local] * alpha;
    o += partial_out[(static_cast<int64_t>(partial_index) * kCutlassTileM +
                      row_local) *
                         kHeadDim + col] *
         alpha;
  }
  out[row * kHeadDim + col] = o / fmaxf(l, 1.0e-20f);
}

void check_tensor(const torch::Tensor& t, const char* name, c10::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has unexpected dtype");
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

void persistent_mainloop_owner_layout_smoke(torch::Tensor out) {
  check_tensor(out, "out", torch::kFloat32);
  TORCH_CHECK(out.numel() >= 8, "out must contain at least 8 float values");

  constexpr int kSmemBytes =
      static_cast<int>(sizeof(CutlassFusedTmaQkPv128Storage));
  auto kernel = persistent_mainloop_owner_layout_smoke_kernel;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  kernel<<<dim3(1, 1, 1), CutlassGemmKernel::get_block_shape(), kSmemBytes,
           at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void persistent_mainloop_owner_qk_stage(torch::Tensor q_packed,
                                        torch::Tensor q_scales,
                                        torch::Tensor k_packed,
                                        torch::Tensor k_scales,
                                        torch::Tensor out_tile,
                                        torch::Tensor workspace,
                                        int64_t q_tile,
                                        int64_t kv_tile) {
  check_tensor(q_packed, "q_packed", torch::kUInt8);
  check_tensor(q_scales, "q_scales", torch::kUInt8);
  check_tensor(k_packed, "k_packed", torch::kUInt8);
  check_tensor(k_scales, "k_scales", torch::kUInt8);
  check_tensor(out_tile, "out_tile", torch::kFloat32);
  check_tensor(workspace, "workspace", torch::kUInt8);
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
  TORCH_CHECK(q_tile >= 0 && q_tile < kQRows / kCutlassTileM,
              "q_tile out of range");
  TORCH_CHECK(kv_tile >= 0 && kv_tile < kKvLen / kCutlassTileN,
              "kv_tile out of range");

  float alpha = 1.0f;
  auto args = flashinfer::gemm::prepareGemmArgsImpl<CutlassGemm>(
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
  CutlassGemm gemm;
  const size_t workspace_size = gemm.get_workspace_size(args);
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(workspace_size),
              "workspace too small: need ", workspace_size, " bytes, got ",
              workspace.numel());
  auto status = gemm.initialize(
      args,
      reinterpret_cast<char*>(workspace.data_ptr<uint8_t>()),
      at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "failed to initialize CUTLASS QK GEMM params");
  auto params = gemm.params();

  constexpr int kSmemBytes =
      static_cast<int>(sizeof(CutlassFusedTmaQkPv128Storage));
  auto kernel = persistent_mainloop_owner_qk_stage_kernel;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  kernel<<<dim3(1, 1, 1), CutlassGemmKernel::get_block_shape(), kSmemBytes,
           at::cuda::getCurrentCUDAStream()>>>(
      params, out_tile.data_ptr<float>(), nullptr, nullptr, nullptr, nullptr,
      0.0f, static_cast<int>(q_tile), static_cast<int>(kv_tile), 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void persistent_mainloop_owner_softmax_stage(torch::Tensor q_packed,
                                             torch::Tensor q_scales,
                                             torch::Tensor k_packed,
                                             torch::Tensor k_scales,
                                             torch::Tensor p_packed,
                                             torch::Tensor p_scales,
                                             torch::Tensor row_m,
                                             torch::Tensor row_l,
                                             torch::Tensor workspace,
                                             double qk_alpha,
                                             int64_t q_tile,
                                             int64_t kv_tile) {
  check_tensor(q_packed, "q_packed", torch::kUInt8);
  check_tensor(q_scales, "q_scales", torch::kUInt8);
  check_tensor(k_packed, "k_packed", torch::kUInt8);
  check_tensor(k_scales, "k_scales", torch::kUInt8);
  check_tensor(p_packed, "p_packed", torch::kUInt8);
  check_tensor(p_scales, "p_scales", torch::kUInt8);
  check_tensor(row_m, "row_m", torch::kFloat32);
  check_tensor(row_l, "row_l", torch::kFloat32);
  check_tensor(workspace, "workspace", torch::kUInt8);
  TORCH_CHECK(q_packed.sizes() == torch::IntArrayRef({kQRows, kPackedHeadDim}),
              "q_packed must have shape [4096, 256]");
  TORCH_CHECK(q_scales.sizes() == torch::IntArrayRef({kQRows, kScaleCols}),
              "q_scales must have shape [4096, 32]");
  TORCH_CHECK(k_packed.sizes() == torch::IntArrayRef({kKvLen, kPackedHeadDim}),
              "k_packed must have shape [32768, 256]");
  TORCH_CHECK(k_scales.sizes() == torch::IntArrayRef({kKvLen, kScaleCols}),
              "k_scales must have shape [32768, 32]");
  TORCH_CHECK(p_packed.sizes() ==
                  torch::IntArrayRef({kCutlassTileM, kCutlassTileK128 / 2}),
              "p_packed must have shape [128, 64]");
  TORCH_CHECK(p_scales.sizes() ==
                  torch::IntArrayRef({kCutlassTileM, kCutlassTileK128 / 16}),
              "p_scales must have shape [128, 8]");
  TORCH_CHECK(row_m.sizes() == torch::IntArrayRef({kCutlassTileM}),
              "row_m must have shape [128]");
  TORCH_CHECK(row_l.sizes() == torch::IntArrayRef({kCutlassTileM}),
              "row_l must have shape [128]");
  TORCH_CHECK(q_tile >= 0 && q_tile < kQRows / kCutlassTileM,
              "q_tile out of range");
  TORCH_CHECK(kv_tile >= 0 && kv_tile < kKvLen / kCutlassTileN,
              "kv_tile out of range");

  float alpha = 1.0f;
  auto args = flashinfer::gemm::prepareGemmArgsImpl<CutlassGemm>(
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
  CutlassGemm gemm;
  const size_t workspace_size = gemm.get_workspace_size(args);
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(workspace_size),
              "workspace too small: need ", workspace_size, " bytes, got ",
              workspace.numel());
  auto status = gemm.initialize(
      args,
      reinterpret_cast<char*>(workspace.data_ptr<uint8_t>()),
      at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "failed to initialize CUTLASS QK GEMM params");
  auto params = gemm.params();

  constexpr int kSmemBytes =
      static_cast<int>(sizeof(CutlassFusedTmaQkPv128Storage));
  auto kernel = persistent_mainloop_owner_qk_stage_kernel;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  kernel<<<dim3(1, 1, 1), CutlassGemmKernel::get_block_shape(), kSmemBytes,
           at::cuda::getCurrentCUDAStream()>>>(
      params, nullptr, p_packed.data_ptr<uint8_t>(),
      p_scales.data_ptr<uint8_t>(), row_m.data_ptr<float>(),
      row_l.data_ptr<float>(), static_cast<float>(qk_alpha),
      static_cast<int>(q_tile), static_cast<int>(kv_tile), 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void persistent_mainloop_owner_pv_group_stage(torch::Tensor q_packed,
                                              torch::Tensor q_scales,
                                              torch::Tensor k_packed,
                                              torch::Tensor k_scales,
                                              torch::Tensor v_pv_packed,
                                              torch::Tensor v_pv_scales,
                                              torch::Tensor out_group,
                                              torch::Tensor p_packed,
                                              torch::Tensor p_scales,
                                              torch::Tensor row_m,
                                              torch::Tensor row_l,
                                              torch::Tensor workspace,
                                              double qk_alpha,
                                              int64_t q_tile,
                                              int64_t kv_tile,
                                              int64_t out_group_idx) {
  check_tensor(q_packed, "q_packed", torch::kUInt8);
  check_tensor(q_scales, "q_scales", torch::kUInt8);
  check_tensor(k_packed, "k_packed", torch::kUInt8);
  check_tensor(k_scales, "k_scales", torch::kUInt8);
  check_tensor(v_pv_packed, "v_pv_packed", torch::kUInt8);
  check_tensor(v_pv_scales, "v_pv_scales", torch::kUInt8);
  check_tensor(out_group, "out_group", torch::kFloat32);
  check_tensor(p_packed, "p_packed", torch::kUInt8);
  check_tensor(p_scales, "p_scales", torch::kUInt8);
  check_tensor(row_m, "row_m", torch::kFloat32);
  check_tensor(row_l, "row_l", torch::kFloat32);
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
  TORCH_CHECK(p_packed.sizes() ==
                  torch::IntArrayRef({kCutlassTileM, kCutlassTileK128 / 2}),
              "p_packed must have shape [128, 64]");
  TORCH_CHECK(p_scales.sizes() ==
                  torch::IntArrayRef({kCutlassTileM, kCutlassTileK128 / 16}),
              "p_scales must have shape [128, 8]");
  TORCH_CHECK(row_m.sizes() == torch::IntArrayRef({kCutlassTileM}),
              "row_m must have shape [128]");
  TORCH_CHECK(row_l.sizes() == torch::IntArrayRef({kCutlassTileM}),
              "row_l must have shape [128]");
  TORCH_CHECK(q_tile >= 0 && q_tile < kQRows / kCutlassTileM,
              "q_tile out of range");
  TORCH_CHECK(kv_tile >= 0 && kv_tile < kKvLen / kCutlassTileN,
              "kv_tile out of range");
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

  auto pv_args = flashinfer::gemm::prepareGemmArgsImpl<CutlassGemmK128>(
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
  CutlassGemmK128 pv_gemm;
  const size_t pv_workspace_size = pv_gemm.get_workspace_size(pv_args);
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(pv_workspace_size),
              "workspace too small for PV: need ", pv_workspace_size,
              " bytes, got ", workspace.numel());
  auto pv_status = pv_gemm.initialize(
      pv_args,
      reinterpret_cast<char*>(workspace.data_ptr<uint8_t>()),
      at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(pv_status == cutlass::Status::kSuccess,
              "failed to initialize CUTLASS PV GEMM params");
  auto pv_params = pv_gemm.params();

  constexpr int kSmemBytes =
      static_cast<int>(sizeof(CutlassFusedTmaQkPv128Storage));
  auto kernel = persistent_mainloop_owner_pv_group_stage_kernel;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  kernel<<<dim3(1, 1, 1), CutlassGemmKernel::get_block_shape(), kSmemBytes,
           at::cuda::getCurrentCUDAStream()>>>(
      qk_params, pv_params, out_group.data_ptr<float>(),
      p_packed.data_ptr<uint8_t>(), p_scales.data_ptr<uint8_t>(),
      row_m.data_ptr<float>(), row_l.data_ptr<float>(),
      static_cast<float>(qk_alpha), static_cast<int>(q_tile),
      static_cast<int>(kv_tile), static_cast<int>(out_group_idx));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void persistent_mainloop_owner_full_tile_stage(torch::Tensor q_packed,
                                               torch::Tensor q_scales,
                                               torch::Tensor k_packed,
                                               torch::Tensor k_scales,
                                               torch::Tensor v_pv_packed,
                                               torch::Tensor v_pv_scales,
                                               torch::Tensor out,
                                               torch::Tensor workspace,
                                               double qk_alpha,
                                               double pv_alpha,
                                               int64_t q_tile,
                                               int64_t kv_tile) {
  check_tensor(q_packed, "q_packed", torch::kUInt8);
  check_tensor(q_scales, "q_scales", torch::kUInt8);
  check_tensor(k_packed, "k_packed", torch::kUInt8);
  check_tensor(k_scales, "k_scales", torch::kUInt8);
  check_tensor(v_pv_packed, "v_pv_packed", torch::kUInt8);
  check_tensor(v_pv_scales, "v_pv_scales", torch::kUInt8);
  check_tensor(out, "out", torch::kFloat32);
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
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({kCutlassTileM, kHeadDim}),
              "out must have shape [128, 512]");
  TORCH_CHECK(q_tile >= 0 && q_tile < kQRows / kCutlassTileM,
              "q_tile out of range");
  TORCH_CHECK(kv_tile >= 0 && kv_tile < kKvLen / kCutlassTileN,
              "kv_tile out of range");

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

  auto pv_args = flashinfer::gemm::prepareGemmArgsImpl<CutlassGemmK128>(
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
  CutlassGemmK128 pv_gemm;
  const size_t pv_workspace_size = pv_gemm.get_workspace_size(pv_args);
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(pv_workspace_size),
              "workspace too small for PV: need ", pv_workspace_size,
              " bytes, got ", workspace.numel());
  auto pv_status = pv_gemm.initialize(
      pv_args,
      reinterpret_cast<char*>(workspace.data_ptr<uint8_t>()),
      at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(pv_status == cutlass::Status::kSuccess,
              "failed to initialize CUTLASS PV GEMM params");
  auto pv_params = pv_gemm.params();

  constexpr int kSmemBytes =
      static_cast<int>(sizeof(CutlassFusedTmaQkPv128Storage));
  auto kernel = persistent_mainloop_owner_full_tile_stage_kernel;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  kernel<<<dim3(1, 1, 1), CutlassGemmKernel::get_block_shape(), kSmemBytes,
           at::cuda::getCurrentCUDAStream()>>>(
      qk_params, pv_params, out.data_ptr<float>(),
      static_cast<float>(qk_alpha), static_cast<float>(pv_alpha),
      static_cast<int>(q_tile), static_cast<int>(kv_tile));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void persistent_mainloop_owner_group_online_stage(torch::Tensor q_packed,
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
  check_tensor(out_group, "out_group", torch::kFloat32);
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

  auto pv_args = flashinfer::gemm::prepareGemmArgsImpl<CutlassGemmK128>(
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
  CutlassGemmK128 pv_gemm;
  const size_t pv_workspace_size = pv_gemm.get_workspace_size(pv_args);
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(pv_workspace_size),
              "workspace too small for PV: need ", pv_workspace_size,
              " bytes, got ", workspace.numel());
  auto pv_status = pv_gemm.initialize(
      pv_args,
      reinterpret_cast<char*>(workspace.data_ptr<uint8_t>()),
      at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(pv_status == cutlass::Status::kSuccess,
              "failed to initialize CUTLASS PV GEMM params");
  auto pv_params = pv_gemm.params();

  constexpr int kSmemBytes =
      static_cast<int>(sizeof(CutlassFusedTmaQkPv128Storage));
  auto kernel = persistent_mainloop_owner_group_online_stage_kernel;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  kernel<<<dim3(1, 1, 1), CutlassGemmKernel::get_block_shape(), kSmemBytes,
           at::cuda::getCurrentCUDAStream()>>>(
      qk_params, pv_params, out_group.data_ptr<float>(),
      static_cast<float>(qk_alpha), static_cast<float>(pv_alpha),
      static_cast<int>(q_tile), static_cast<int>(kv_tile_start),
           static_cast<int>(num_kv_tiles), static_cast<int>(out_group_idx));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void persistent_mainloop_owner_group_online_register_stage(
    torch::Tensor q_packed,
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
  check_tensor(out_group, "out_group", torch::kFloat32);
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

  auto pv_args = flashinfer::gemm::prepareGemmArgsImpl<CutlassGemmK128>(
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
  CutlassGemmK128 pv_gemm;
  const size_t pv_workspace_size = pv_gemm.get_workspace_size(pv_args);
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(pv_workspace_size),
              "workspace too small for PV: need ", pv_workspace_size,
              " bytes, got ", workspace.numel());
  auto pv_status = pv_gemm.initialize(
      pv_args,
      reinterpret_cast<char*>(workspace.data_ptr<uint8_t>()),
      at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(pv_status == cutlass::Status::kSuccess,
              "failed to initialize CUTLASS PV GEMM params");
  auto pv_params = pv_gemm.params();

  constexpr int kSmemBytes =
      static_cast<int>(sizeof(CutlassFusedTmaQkPv128Storage));
  auto kernel = persistent_mainloop_owner_group_online_register_stage_kernel;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  kernel<<<dim3(1, 1, 1), CutlassGemmKernel::get_block_shape(), kSmemBytes,
           at::cuda::getCurrentCUDAStream()>>>(
      qk_params, pv_params, out_group.data_ptr<float>(),
      static_cast<float>(qk_alpha), static_cast<float>(pv_alpha),
      static_cast<int>(q_tile), static_cast<int>(kv_tile_start),
      static_cast<int>(num_kv_tiles), static_cast<int>(out_group_idx));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void persistent_mainloop_owner_group_online_register_q_stage(
    torch::Tensor q_packed,
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
  check_tensor(out_group, "out_group", torch::kFloat32);
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

  auto pv_args = flashinfer::gemm::prepareGemmArgsImpl<CutlassGemmK128>(
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
  CutlassGemmK128 pv_gemm;
  const size_t pv_workspace_size = pv_gemm.get_workspace_size(pv_args);
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(pv_workspace_size),
              "workspace too small for PV: need ", pv_workspace_size,
              " bytes, got ", workspace.numel());
  auto pv_status = pv_gemm.initialize(
      pv_args,
      reinterpret_cast<char*>(workspace.data_ptr<uint8_t>()),
      at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(pv_status == cutlass::Status::kSuccess,
              "failed to initialize CUTLASS PV GEMM params");
  auto pv_params = pv_gemm.params();

  constexpr int kSmemBytes =
      static_cast<int>(sizeof(CutlassFusedTmaQkPv128Storage));
  auto kernel = persistent_mainloop_owner_group_online_register_q_stage_kernel;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  kernel<<<dim3(1, 1, 1), CutlassGemmKernel::get_block_shape(), kSmemBytes,
           at::cuda::getCurrentCUDAStream()>>>(
      q_packed.data_ptr<uint8_t>(), q_scales.data_ptr<uint8_t>(),
      qk_params, pv_params, out_group.data_ptr<float>(),
      static_cast<float>(qk_alpha), static_cast<float>(pv_alpha),
      static_cast<int>(q_tile), static_cast<int>(kv_tile_start),
      static_cast<int>(num_kv_tiles), static_cast<int>(out_group_idx));
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
      static_cast<int64_t>(sizeof(typename CutlassCollectiveMainloop::SharedStorage));
  const int64_t pv_k128_shared_bytes =
      static_cast<int64_t>(
          sizeof(typename CutlassCollectiveMainloopK128::SharedStorage));
  const int64_t qk_pv_independent_shared_bytes =
      qk_shared_bytes + pv_k128_shared_bytes;
  d["sm120_optin_smem_bytes"] = kSm120OptinSmemBytes;
  d["persistent_independent_qk_pv_shared_bytes"] =
      qk_pv_independent_shared_bytes;
  d["persistent_independent_qk_pv_fits_sm120"] =
      qk_pv_independent_shared_bytes <= kSm120OptinSmemBytes;
  d["persistent_independent_qk_pv_smem_margin_bytes"] =
      kSm120OptinSmemBytes - qk_pv_independent_shared_bytes;
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
  m.def("persistent_mainloop_owner_layout_smoke",
        &persistent_mainloop_owner_layout_smoke,
        "SM120 phased single-storage persistent-mainloop owner layout smoke");
  m.def("persistent_mainloop_owner_qk_stage",
        &persistent_mainloop_owner_qk_stage,
        "SM120 phased single-storage persistent-mainloop owner QK stage smoke");
  m.def("persistent_mainloop_owner_softmax_stage",
        &persistent_mainloop_owner_softmax_stage,
        "SM120 phased single-storage persistent-mainloop owner QK plus P-quant smoke");
  m.def("persistent_mainloop_owner_pv_group_stage",
        &persistent_mainloop_owner_pv_group_stage,
        "SM120 phased single-storage persistent-mainloop owner QK/P-quant/PV one-group smoke");
  m.def("persistent_mainloop_owner_full_tile_stage",
        &persistent_mainloop_owner_full_tile_stage,
        "SM120 phased single-storage persistent-mainloop owner QK/P-quant/PV full-tile smoke");
  m.def("persistent_mainloop_owner_group_online_stage",
        &persistent_mainloop_owner_group_online_stage,
        "SM120 phased owner one-output-group online-softmax multi-KV smoke");
  m.def("persistent_mainloop_owner_group_online_register_stage",
        &persistent_mainloop_owner_group_online_register_stage,
        "SM120 phased owner one-output-group online-softmax register-accum multi-KV smoke");
  m.def("persistent_mainloop_owner_group_online_register_q_stage",
        &persistent_mainloop_owner_group_online_register_q_stage,
        "SM120 phased owner one-output-group online-softmax register-resident-Q smoke");
  m.def("cutlass_runner_fp4_gemm", &cutlass_runner_fp4_gemm,
        "FlashInfer SM120 CUTLASS FP4 GEMM runner smoke hook");
  m.def("cutlass_sm120_blockscaled_collective_metadata",
        &cutlass_sm120_blockscaled_collective_metadata,
        "Compile-time metadata for the SM120 block-scaled CUTLASS collective");
}
