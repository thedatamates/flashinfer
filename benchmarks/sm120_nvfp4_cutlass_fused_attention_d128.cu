#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_d128.cuh>
#include <flashinfer/attention/blackwell/fmha_nvfp4_sm120_quantization.cuh>

namespace {

using namespace flashinfer::attention::blackwell::sm120_nvfp4::d128;

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
  TORCH_CHECK(q.dim() == 2 || q.dim() == 3,
              "q must have shape [rows, D] or [q_len, group, D]");
  const int64_t q_rows64 = q.dim() == 3 ? q.size(0) * q.size(1) : q.size(0);
  const int64_t head_dim64 = q.dim() == 3 ? q.size(2) : q.size(1);
  TORCH_CHECK(head_dim64 == kHeadDim, "q head_dim must be ", kHeadDim);
  TORCH_CHECK(q_packed.sizes() ==
                  torch::IntArrayRef({q_rows64, kPackedHeadDim}),
              "q_packed must have shape [q_rows, D/2]");
  TORCH_CHECK(q_scales.size(0) >= q_rows64 && q_scales.size(1) == kScaleCols,
              "q_scales must have shape [>= q_rows, D/16]");
  C10_CUDA_CHECK(flashinfer::attention::blackwell::sm120_nvfp4::
                     quantize_q_rowmajor_raw(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      q_packed.data_ptr<uint8_t>(), q_scales.data_ptr<uint8_t>(),
      static_cast<int>(q_rows64), kHeadDim, at::cuda::getCurrentCUDAStream()));
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

void pv_cutlass_stage2_atom_tile(torch::Tensor p_packed,
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
  TORCH_CHECK(p_packed.sizes() ==
                  torch::IntArrayRef({kBenchRows, kProbPackedCols}),
              "p_packed must have shape [128, 16384]");
  TORCH_CHECK(p_scales.sizes() ==
                  torch::IntArrayRef({kBenchRows, kProbScaleCols}),
              "p_scales must have shape [128, 2048]");
  TORCH_CHECK(v_pv_packed.sizes() ==
                  torch::IntArrayRef({kHeadDim, kProbPackedCols}),
              "v_pv_packed must have shape [512, 16384]");
  TORCH_CHECK(v_pv_scales.sizes() ==
                  torch::IntArrayRef({kHeadDim, kProbScaleCols}),
              "v_pv_scales must have shape [512, 2048]");
  TORCH_CHECK(out_tile.sizes() ==
                  torch::IntArrayRef({kCutlassTileM, kOutputTileN}),
              "out_tile must have shape [tile_m, output_tile_n]");
  TORCH_CHECK(kv_base >= 0 && kv_base + kCutlassTileN <= kKvLen,
              "kv_base out of range");
  TORCH_CHECK(out_col_base >= 0 && out_col_base + kOutputTileN <= kHeadDim,
              "out_col_base out of range");
  TORCH_CHECK(kv_base % kCutlassTileN == 0,
              "kv_base must be a multiple of tile_n");
  TORCH_CHECK(out_col_base % kOutputTileN == 0,
              "out_col_base must be a multiple of output_tile_n");

  auto kernel = pv_cutlass_stage2_atom_tile_kernel;
  const size_t smem_size =
      sizeof(typename CutlassCollectiveMainloopK128Stage2::TensorStorage);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size)));
  kernel<<<1, CutlassCollectiveMainloopK128Stage2::ThreadCount, smem_size,
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
                  torch::IntArrayRef({kCutlassTileM, kOutputTileN}),
              "out_group must have shape [tile_m, output_tile_n]");
  TORCH_CHECK(q_tile >= 0 && q_tile < kQRows / kCutlassTileM,
              "q_tile out of range");
  TORCH_CHECK(kv_tile_start >= 0 &&
                  kv_tile_start < kKvLen / kCutlassTileN,
              "kv_tile_start out of range");
  TORCH_CHECK(num_kv_tiles > 0 &&
                  kv_tile_start + num_kv_tiles <= kKvLen / kCutlassTileN,
              "num_kv_tiles out of range");
  TORCH_CHECK(out_group_idx >= 0 && out_group_idx < kHeadDim / kOutputTileN,
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
  auto kernel = sm120_nvfp4_qkv_online_register_q_stage_kernel<1>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  kernel<<<dim3(1, 1, 1), kSm120Nvfp4FmhaThreadCount, kSmemBytes,
           at::cuda::getCurrentCUDAStream()>>>(
      qk_params, pv_params,
      reinterpret_cast<__nv_bfloat16*>(out_group.data_ptr<at::BFloat16>()),
      static_cast<float>(qk_alpha), static_cast<float>(pv_alpha),
      static_cast<int>(q_tile), static_cast<int>(kv_tile_start),
      static_cast<int>(num_kv_tiles), kKvLen / kCutlassTileN,
      kQLen, kGroup, kKvLen, 0, -1, 0.0f,
      static_cast<int>(out_group_idx), kOutputTileN, nullptr, nullptr, 0, 0,
      Sm120Nvfp4PagedKvLoadParams{}, nullptr, nullptr, 1, 0);
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
  auto kernel = sm120_nvfp4_qkv_online_register_q_stage_kernel<1>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  kernel<<<dim3(kQRows / kCutlassTileM, kHeadDim / kOutputTileN, 1),
           kSm120Nvfp4FmhaThreadCount, kSmemBytes,
           at::cuda::getCurrentCUDAStream()>>>(
      qk_params, pv_params,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      static_cast<float>(qk_alpha), static_cast<float>(pv_alpha), 0, 0,
      kKvLen / kCutlassTileN, kKvLen / kCutlassTileN,
      kQLen, kGroup, kKvLen, 0, -1, 0.0f,
      0, kHeadDim, nullptr, nullptr, 0, 0,
      Sm120Nvfp4PagedKvLoadParams{}, nullptr, nullptr, 1, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kOutputGroupSpan>
void sm120_nvfp4_qkv_online_register_q_splitkv_full_grid_impl(
    torch::Tensor q_packed,
    torch::Tensor q_scales,
    torch::Tensor k_packed,
    torch::Tensor k_scales,
    torch::Tensor v_pv_packed,
    torch::Tensor v_pv_scales,
    torch::Tensor partial,
    torch::Tensor split_m,
    torch::Tensor split_l,
    torch::Tensor out,
    torch::Tensor workspace,
    double qk_alpha,
    double pv_alpha,
    int64_t split_kv_tiles,
    int64_t q_len,
    int64_t group_size,
    int64_t kv_len_tokens,
    bool causal,
    int64_t sliding_window,
    double logits_soft_cap) {
  static_assert(kOutputGroupSpan == 1,
                "D128 split-KV wrapper supports span 1 only");
  check_tensor(q_packed, "q_packed", torch::kUInt8);
  check_tensor(q_scales, "q_scales", torch::kUInt8);
  check_tensor(k_packed, "k_packed", torch::kUInt8);
  check_tensor(k_scales, "k_scales", torch::kUInt8);
  check_tensor(v_pv_packed, "v_pv_packed", torch::kUInt8);
  check_tensor(v_pv_scales, "v_pv_scales", torch::kUInt8);
  check_tensor(partial, "partial", torch::kBFloat16);
  check_tensor(split_m, "split_m", torch::kFloat32);
  check_tensor(split_l, "split_l", torch::kFloat32);
  check_tensor(out, "out", torch::kBFloat16);
  check_tensor(workspace, "workspace", torch::kUInt8);
  TORCH_CHECK(q_packed.dim() == 2, "q_packed must be 2D");
  TORCH_CHECK(k_packed.dim() == 2, "k_packed must be 2D");
  TORCH_CHECK(v_pv_packed.dim() == 2, "v_pv_packed must be 2D");
  const int64_t q_rows64 = q_packed.size(0);
  const int64_t packed_head_dim64 = q_packed.size(1);
  const int64_t head_dim64 = packed_head_dim64 * 2;
  const int64_t scale_cols64 = head_dim64 / 16;
  const int64_t kv_len64 = k_packed.size(0);
  const int64_t prob_packed_cols64 = kv_len64 / 2;
  const int64_t prob_scale_cols64 = kv_len64 / 16;
  TORCH_CHECK(head_dim64 == 128 || head_dim64 == 256 || head_dim64 == 512,
              "SM120 fused wrapper currently supports D128/D128/D512 only, got D",
              head_dim64);
  TORCH_CHECK(q_rows64 > 0 && q_rows64 % kCutlassTileM == 0,
              "q rows must be a positive multiple of ", kCutlassTileM);
  TORCH_CHECK(kv_len64 > 0 && kv_len64 % kCutlassTileN == 0,
              "KV length must be a positive multiple of ", kCutlassTileN);
  TORCH_CHECK(kv_len64 <= kShapeBMaxKvLen,
              "KV length exceeds Shape B max supported by combine scratch: ",
              kv_len64);
  TORCH_CHECK(k_packed.size(1) == packed_head_dim64,
              "k_packed packed head dim must match q_packed");
  TORCH_CHECK(q_scales.sizes() ==
                  torch::IntArrayRef({q_rows64, scale_cols64}),
              "q_scales must have shape [q_rows, D/16]");
  TORCH_CHECK(k_scales.sizes() ==
                  torch::IntArrayRef({kv_len64, scale_cols64}),
              "k_scales must have shape [kv_len, D/16]");
  TORCH_CHECK(v_pv_packed.sizes() ==
                  torch::IntArrayRef({head_dim64, prob_packed_cols64}),
              "v_pv_packed must have shape [D, kv_len/2]");
  TORCH_CHECK(v_pv_scales.sizes() ==
                  torch::IntArrayRef({head_dim64, prob_scale_cols64}),
              "v_pv_scales must have shape [D, kv_len/16]");
  TORCH_CHECK(split_kv_tiles > 0, "split_kv_tiles must be positive");
  TORCH_CHECK(q_len > 0, "q_len must be positive");
  TORCH_CHECK(group_size > 0, "group_size must be positive");
  TORCH_CHECK(q_len * group_size <= q_rows64,
              "q_len * group_size must not exceed physical q rows");
  TORCH_CHECK(kv_len_tokens > 0 && kv_len_tokens <= kv_len64,
              "kv_len_tokens must be positive and not exceed physical K/V length");
  TORCH_CHECK(sliding_window == -1 || sliding_window > 0,
              "sliding_window must be -1 or positive");
  const int q_rows = static_cast<int>(q_rows64);
  const int head_dim = static_cast<int>(head_dim64);
  const int kv_len = static_cast<int>(kv_len64);
  const int total_kv_tiles = kv_len / kCutlassTileN;
  const int num_splits =
      static_cast<int>((total_kv_tiles + split_kv_tiles - 1) / split_kv_tiles);
  TORCH_CHECK(num_splits <= kShapeBMaxKvTiles,
              "num_splits exceeds Shape B combine scratch");
  TORCH_CHECK(partial.sizes() ==
                  torch::IntArrayRef({num_splits, q_rows64, head_dim64}),
              "partial must have shape [num_splits, q_rows, D]");
  TORCH_CHECK(split_m.sizes() == torch::IntArrayRef({num_splits, q_rows64}),
              "split_m must have shape [num_splits, q_rows]");
  TORCH_CHECK(split_l.sizes() == torch::IntArrayRef({num_splits, q_rows64}),
              "split_l must have shape [num_splits, q_rows]");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({q_rows64, head_dim64}),
              "out must have shape [q_rows, D]");
  TORCH_CHECK(head_dim % (kOutputGroupSpan * kOutputTileN) == 0,
              "head dimension must be divisible by output-group span");

  C10_CUDA_CHECK(sm120_nvfp4_qkv_online_register_q_splitkv_full_grid_raw<
                 kOutputGroupSpan>(
      q_packed.data_ptr<uint8_t>(), q_scales.data_ptr<uint8_t>(),
      k_packed.data_ptr<uint8_t>(), k_scales.data_ptr<uint8_t>(),
      v_pv_packed.data_ptr<uint8_t>(), v_pv_scales.data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(partial.data_ptr<at::BFloat16>()),
      split_m.data_ptr<float>(), split_l.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      workspace.data_ptr<uint8_t>(), static_cast<size_t>(workspace.numel()),
      static_cast<float>(qk_alpha), static_cast<float>(pv_alpha),
      static_cast<int>(split_kv_tiles), static_cast<int>(q_len),
      static_cast<int>(group_size), static_cast<int>(kv_len_tokens), causal,
      static_cast<int>(sliding_window), static_cast<float>(logits_soft_cap),
      q_rows, head_dim, kv_len, at::cuda::getCurrentCUDAStream()));
}

void sm120_nvfp4_qkv_online_register_q_splitkv_full_grid(
    torch::Tensor q_packed,
    torch::Tensor q_scales,
    torch::Tensor k_packed,
    torch::Tensor k_scales,
    torch::Tensor v_pv_packed,
    torch::Tensor v_pv_scales,
    torch::Tensor partial,
    torch::Tensor split_m,
    torch::Tensor split_l,
    torch::Tensor out,
    torch::Tensor workspace,
    double qk_alpha,
    double pv_alpha,
    int64_t split_kv_tiles,
    int64_t q_len,
    int64_t group_size,
    int64_t kv_len_tokens,
    bool causal,
    int64_t sliding_window,
    double logits_soft_cap) {
  sm120_nvfp4_qkv_online_register_q_splitkv_full_grid_impl<1>(
      q_packed, q_scales, k_packed, k_scales, v_pv_packed, v_pv_scales,
      partial, split_m, split_l, out, workspace, qk_alpha, pv_alpha,
      split_kv_tiles, q_len, group_size, kv_len_tokens, causal,
      sliding_window, logits_soft_cap);
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

void cutlass_active_tile_fp4_gemm(torch::Tensor a_packed,
                                  torch::Tensor b_packed_t,
                                  torch::Tensor a_scales,
                                  torch::Tensor b_scales_t,
                                  torch::Tensor alpha,
                                  torch::Tensor out,
                                  torch::Tensor workspace) {
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

  auto args = flashinfer::gemm::prepareGemmArgsImpl<CutlassGemm>(
      nullptr, a_packed.data_ptr<uint8_t>(), b_packed_t.data_ptr<uint8_t>(),
      a_scales.data_ptr<uint8_t>(), b_scales_t.data_ptr<uint8_t>(),
      alpha.data_ptr<float>(), m, n, k, 1);
  CutlassGemm gemm;
  const size_t required_workspace = gemm.get_workspace_size(args);
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(required_workspace),
              "workspace too small: need ", required_workspace, " bytes, got ",
              workspace.numel());
  auto status = gemm.initialize(
      args, reinterpret_cast<char*>(workspace.data_ptr<uint8_t>()),
      at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "failed to initialize active-tile CUTLASS GEMM params");
  status = gemm.run(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "active-tile CUTLASS GEMM run failed");
}

pybind11::dict cutlass_sm120_blockscaled_collective_metadata() {
  pybind11::dict d;
  d["arch"] = "sm120";
  d["operator_class"] = "OpClassBlockScaledTensorOp";
  d["tile_m"] = kCutlassTileM;
  d["tile_n"] = kCutlassTileN;
  d["tile_k"] = kCutlassTileK;
  d["scale_vec_size"] = CutlassCollectiveMainloop::TiledMma::SFVecSize;
  d["active_blockscaled_mma_atom"] = "m16n8k64_mxf4nvf4_ue4m3";
  d["qk_mainloop_stages"] =
      static_cast<int64_t>(CutlassCollectiveMainloop::DispatchPolicy::Stages);
  d["pv_k128_stage2_mainloop_stages"] = static_cast<int64_t>(
      CutlassCollectiveMainloopK128Stage2::DispatchPolicy::Stages);
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
  d["sm120_pipeline_e_storage_bytes"] =
      static_cast<int64_t>(sizeof(typename Sm120Nvfp4PipelineE::SharedStorage));
  d["sm120_qkv_load_collective_storage_bytes"] =
      static_cast<int64_t>(sizeof(Sm120Nvfp4QkvLoadCollectiveStorage));
  d["sm120_qkv_load_collective_storage_margin_bytes"] =
      kSm120OptinSmemBytes -
      static_cast<int64_t>(sizeof(Sm120Nvfp4QkvLoadCollectiveStorage));
  pybind11::dict qkv_storage_layout;
  qkv_storage_layout["qk_tensors_offset"] =
      static_cast<int64_t>(
          offsetof(Sm120Nvfp4QkvLoadCollectiveStorage, qk_tensors));
  qkv_storage_layout["qk_tensors_bytes"] =
      static_cast<int64_t>(
          sizeof(typename CutlassCollectiveMainloop::TensorStorage));
  qkv_storage_layout["q_pipeline_storage_offset"] =
      static_cast<int64_t>(
          offsetof(Sm120Nvfp4QkvLoadCollectiveStorage, q_pipeline_storage));
  qkv_storage_layout["q_pipeline_storage_bytes"] =
      static_cast<int64_t>(
          sizeof(typename CutlassCollectiveMainloop::PipelineStorage));
  qkv_storage_layout["k_pipeline_storage_offset"] =
      static_cast<int64_t>(
          offsetof(Sm120Nvfp4QkvLoadCollectiveStorage, k_pipeline_storage));
  qkv_storage_layout["k_pipeline_storage_bytes"] =
      static_cast<int64_t>(
          sizeof(typename CutlassCollectiveMainloop::PipelineStorage));
  qkv_storage_layout["v_smem_B_offset"] =
      static_cast<int64_t>(
          offsetof(Sm120Nvfp4QkvLoadCollectiveStorage, v_smem_B));
  qkv_storage_layout["v_smem_B_bytes"] =
      static_cast<int64_t>(
          sizeof(decltype(((Sm120Nvfp4QkvLoadCollectiveStorage*)nullptr)
                              ->v_smem_B)));
  qkv_storage_layout["v_smem_SFB_offset"] =
      static_cast<int64_t>(
          offsetof(Sm120Nvfp4QkvLoadCollectiveStorage, v_smem_SFB));
  qkv_storage_layout["v_smem_SFB_bytes"] =
      static_cast<int64_t>(
          sizeof(decltype(((Sm120Nvfp4QkvLoadCollectiveStorage*)nullptr)
                              ->v_smem_SFB)));
  qkv_storage_layout["v_pipeline_storage_offset"] =
      static_cast<int64_t>(
          offsetof(Sm120Nvfp4QkvLoadCollectiveStorage, v_pipeline_storage));
  qkv_storage_layout["v_pipeline_storage_bytes"] =
      static_cast<int64_t>(
          sizeof(typename CutlassCollectiveMainloopK128Stage2::PipelineStorage));
  qkv_storage_layout["logits_smem_offset"] =
      static_cast<int64_t>(
          offsetof(Sm120Nvfp4QkvLoadCollectiveStorage, logits_smem));
  qkv_storage_layout["logits_smem_bytes"] =
      static_cast<int64_t>(
          sizeof(decltype(((Sm120Nvfp4QkvLoadCollectiveStorage*)nullptr)
                              ->logits_smem)));
  qkv_storage_layout["p_smem_A0_offset"] =
      static_cast<int64_t>(
          offsetof(Sm120Nvfp4QkvLoadCollectiveStorage, p_smem_A0));
  qkv_storage_layout["p_smem_A0_bytes"] =
      static_cast<int64_t>(
          sizeof(decltype(((Sm120Nvfp4QkvLoadCollectiveStorage*)nullptr)
                              ->p_smem_A0)));
  qkv_storage_layout["p_smem_SFA0_offset"] =
      static_cast<int64_t>(
          offsetof(Sm120Nvfp4QkvLoadCollectiveStorage, p_smem_SFA0));
  qkv_storage_layout["p_smem_SFA0_bytes"] =
      static_cast<int64_t>(
          sizeof(decltype(((Sm120Nvfp4QkvLoadCollectiveStorage*)nullptr)
	                              ->p_smem_SFA0)));
  qkv_storage_layout["role_pipeline_storage_offset"] =
      static_cast<int64_t>(
          offsetof(Sm120Nvfp4QkvLoadCollectiveStorage,
                   role_pipeline_storage));
  qkv_storage_layout["role_pipeline_storage_bytes"] =
      static_cast<int64_t>(sizeof(Sm120Nvfp4MainloopPipelineStorage));
  qkv_storage_layout["global_m_offset"] =
      static_cast<int64_t>(
          offsetof(Sm120Nvfp4QkvLoadCollectiveStorage, global_m));
  qkv_storage_layout["global_m_bytes"] =
      static_cast<int64_t>(
          sizeof(decltype(((Sm120Nvfp4QkvLoadCollectiveStorage*)nullptr)
                              ->global_m)));
  qkv_storage_layout["global_l_offset"] =
      static_cast<int64_t>(
          offsetof(Sm120Nvfp4QkvLoadCollectiveStorage, global_l));
  qkv_storage_layout["global_l_bytes"] =
      static_cast<int64_t>(
          sizeof(decltype(((Sm120Nvfp4QkvLoadCollectiveStorage*)nullptr)
                              ->global_l)));
  qkv_storage_layout["old_scale_stage_offset"] =
      static_cast<int64_t>(
          offsetof(Sm120Nvfp4QkvLoadCollectiveStorage, old_scale_stage));
  qkv_storage_layout["old_scale_stage_bytes"] =
      static_cast<int64_t>(
          sizeof(decltype(((Sm120Nvfp4QkvLoadCollectiveStorage*)nullptr)
                              ->old_scale_stage)));
  d["sm120_qkv_storage_layout"] = qkv_storage_layout;
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
    pybind11::list row_thread_summary;
    for (int row = 0; row < std::min(kCutlassTileM, 16); ++row) {
      pybind11::list row_entries;
      for (int thread_idx = 0;
           thread_idx < int(CutlassCollectiveMainloop::ThreadCount);
           ++thread_idx) {
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
        auto tCcC = thread_mma.partition_C(cC);
        pybind11::list cols;
        for (int i = 0; i < int(cute::size(tCcC)); ++i) {
          auto coord = tCcC(i);
          const int coord_row = int(cute::get<0>(coord));
          const int col = int(cute::get<1>(coord));
          if (coord_row == row && col >= 0 && col < kCutlassTileN) {
            cols.append(col);
          }
        }
        if (pybind11::len(cols) > 0) {
          pybind11::list entry;
          entry.append(thread_idx);
          entry.append(cols);
          row_entries.append(entry);
        }
      }
      row_thread_summary.append(row_entries);
    }
    d["c_fragment_row_thread_summary_0_15"] = row_thread_summary;
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
  m.def("pv_cutlass_stage2_atom_tile", &pv_cutlass_stage2_atom_tile,
        "CUTLASS SM120 stage-2 PV atom debug tile");
  m.def("sm120_nvfp4_role_schedule_smoke",
        &sm120_nvfp4_role_schedule_smoke,
        "SM120 NVFP4 explicit role-schedule smoke for the fused FMHA mainloop");
  m.def("sm120_nvfp4_qkv_online_register_q_stage",
        &sm120_nvfp4_qkv_online_register_q_stage,
        "SM120 NVFP4 online-softmax register-Q smoke with compact Q/K/V storage");
  m.def("sm120_nvfp4_qkv_online_register_q_full_grid",
        &sm120_nvfp4_qkv_online_register_q_full_grid,
        "SM120 NVFP4 online-softmax register-Q full Shape-B tile grid");
  m.def("sm120_nvfp4_qkv_online_register_q_splitkv_full_grid",
        &sm120_nvfp4_qkv_online_register_q_splitkv_full_grid,
        "SM120 NVFP4 split-KV online-softmax full Shape-B tile grid");
  m.def("cutlass_runner_fp4_gemm", &cutlass_runner_fp4_gemm,
        "FlashInfer SM120 CUTLASS FP4 GEMM runner smoke hook");
  m.def("cutlass_active_tile_fp4_gemm", &cutlass_active_tile_fp4_gemm,
        "Active tile-shape CUTLASS FP4 GEMM smoke hook");
  m.def("cutlass_sm120_blockscaled_collective_metadata",
        &cutlass_sm120_blockscaled_collective_metadata,
        "Compile-time metadata for the SM120 block-scaled CUTLASS collective");
}
