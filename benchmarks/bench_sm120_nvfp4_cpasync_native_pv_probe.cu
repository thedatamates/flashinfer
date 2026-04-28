/*
 * Reduced repeat-launch probe for the D512/VO256 NVFP4 prefill PV path.
 *
 * This keeps FlashInfer's production cp.async V/SF loaders and the native FP4
 * PV helper, but removes QK, masking, scheduling, and Python/FFI. It isolates
 * whether the repeat-launch hang comes from cp.async-loaded shared V data
 * feeding native FP4 MMA.
 */

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include <flashinfer/attention/prefill.cuh>

#define CUDA_CHECK(expr)                                                        \
  do {                                                                         \
    cudaError_t status = (expr);                                               \
    if (status != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(status));                                \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace fi = flashinfer;

using TestAttention = fi::DefaultAttention</*use_custom_mask=*/false,
                                         /*use_sliding_window=*/false,
                                         /*use_logits_soft_cap=*/false,
                                         /*use_alibi=*/false>;
using TestTraits =
    fi::KernelTraits<fi::MaskMode::kNone, /*CTA_TILE_Q=*/32, /*NUM_MMA_Q=*/1,
                     /*NUM_MMA_KV=*/4, /*NUM_MMA_D_QK=*/32, /*NUM_MMA_D_VO=*/16,
                     /*NUM_WARPS_Q=*/2, /*NUM_WARPS_KV=*/1, fi::PosEncodingMode::kNone,
                     __nv_bfloat16, __nv_fp4x2_e2m1, __nv_bfloat16, float, int32_t,
                     TestAttention>;

__global__ void cpasync_native_pv_probe_kernel(__nv_fp4x2_e2m1* v, uint8_t* v_sf,
                                               float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  __shared__ typename TestTraits::SharedStorage storage;
  constexpr uint32_t UPCAST_STRIDE_V = TestTraits::UPCAST_STRIDE_V;
  constexpr uint32_t KV_THR_LAYOUT_ROW = TestTraits::KV_THR_LAYOUT_ROW;
  constexpr uint32_t KV_THR_LAYOUT_COL = TestTraits::KV_THR_LAYOUT_COL;
  constexpr uint32_t CTA_TILE_KV = TestTraits::CTA_TILE_KV;
  constexpr uint32_t packed_cols = TestTraits::HEAD_DIM_VO / 2;
  constexpr uint32_t sf_cols = TestTraits::HEAD_DIM_VO / fi::NVFP4_SF_VEC_SIZE;

  const dim3 tid = threadIdx;
  const uint32_t warp_idx = fi::get_warp_idx<TestTraits>(tid.y, tid.z);
  const uint32_t lane_idx = tid.x;
  const uint32_t linear_tid = lane_idx + 32 * tid.y;
  const uint32_t nthreads = blockDim.x * blockDim.y;

  fi::smem_t<TestTraits::SWIZZLE_MODE_KV> v_smem(storage.v_smem);
  uint32_t v_smem_offset_w = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
      warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
      lane_idx % KV_THR_LAYOUT_COL);

  constexpr uint32_t fp4_pack = 2;
  __nv_fp4x2_e2m1* v_ptr =
      v + (warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL) * packed_cols +
      (lane_idx % KV_THR_LAYOUT_COL) * fi::upcast_size<__nv_fp4x2_e2m1>() / fp4_pack;

  fi::produce_kv<true, fi::SharedMemFillMode::kFillZero, TestTraits>(
      v_smem, &v_smem_offset_w, &v_ptr, packed_cols, 0, CTA_TILE_KV, tid);
  fi::produce_kv_sf<true, TestTraits>(&storage, v_sf, 0, 0, packed_cols, packed_cols, 0,
                                      CTA_TILE_KV, warp_idx, lane_idx);
  fi::cp_async::commit_group();
  fi::cp_async::wait_group<0>();
  __syncthreads();

  float s_frag[TestTraits::NUM_MMA_Q][TestTraits::NUM_MMA_KV][8];
  float o_frag[TestTraits::NUM_MMA_Q][TestTraits::NUM_MMA_D_VO][8];
  float d[TestTraits::NUM_MMA_Q][2];

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < TestTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t row_group = 0; row_group < 2; ++row_group) {
      d[mma_q][row_group] = 0.f;
    }
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < TestTraits::NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t reg = 0; reg < 8; ++reg) {
        const int raw = int((linear_tid * 11 + mma_kv * 7 + reg * 3) % 17) - 8;
        s_frag[mma_q][mma_kv][reg] = 0.03125f * float(raw + 9);
      }
    }
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < TestTraits::NUM_MMA_D_VO; ++mma_d) {
#pragma unroll
      for (uint32_t reg = 0; reg < 8; ++reg) {
        o_frag[mma_q][mma_d][reg] = 0.f;
      }
    }
  }

  fi::compute_sfm_v_native_fp4<TestTraits>(&v_smem, storage.v_sf_smem, s_frag, o_frag, d);

  const uint32_t out_base = (blockIdx.z * nthreads + linear_tid) * 10;
  out[out_base + 0] = d[0][0];
  out[out_base + 1] = d[0][1];
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    out[out_base + 2 + i] = o_frag[0][0][i];
  }
#else
  if (threadIdx.x == 0) {
    out[0] = -1.f;
  }
#endif
}

int main(int argc, char** argv) {
  const int repeat = argc > 1 ? std::atoi(argv[1]) : 1;
  const int blocks_z = argc > 2 ? std::atoi(argv[2]) : 2;
  constexpr int threads = 64;
  constexpr int rows = TestTraits::CTA_TILE_KV;
  constexpr int packed_cols = TestTraits::HEAD_DIM_VO / 2;
  constexpr int sf_cols = TestTraits::HEAD_DIM_VO / fi::NVFP4_SF_VEC_SIZE;

  __nv_fp4x2_e2m1* v = nullptr;
  uint8_t* v_sf = nullptr;
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&v, rows * packed_cols * sizeof(__nv_fp4x2_e2m1)));
  CUDA_CHECK(cudaMalloc(&v_sf, rows * sf_cols));
  CUDA_CHECK(cudaMalloc(&out, blocks_z * threads * 10 * sizeof(float)));
  CUDA_CHECK(cudaMemset(v, 0x22, rows * packed_cols * sizeof(__nv_fp4x2_e2m1)));
  CUDA_CHECK(cudaMemset(v_sf, 0x30, rows * sf_cols));

  for (int i = 0; i < repeat; ++i) {
    CUDA_CHECK(cudaMemset(out, 0, blocks_z * threads * 10 * sizeof(float)));
    cpasync_native_pv_probe_kernel<<<dim3(1, 1, blocks_z), dim3(32, 2, 1)>>>(v, v_sf, out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  float sample[10];
  CUDA_CHECK(cudaMemcpy(sample, out, sizeof(sample), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(v));
  CUDA_CHECK(cudaFree(v_sf));
  CUDA_CHECK(cudaFree(out));
  std::printf("sample=[");
  for (int i = 0; i < 10; ++i) {
    std::printf("%s%g", i ? "," : "", sample[i]);
  }
  std::printf("]\n");
  return 0;
}
