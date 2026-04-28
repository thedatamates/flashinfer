/*
 * Probe FP4 V shared-memory reads after the production paged KV loader.
 */

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <flashinfer/attention/prefill.cuh>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                        \
    cudaError_t status = (expr);                                              \
    if (status != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(status));                               \
      std::exit(1);                                                           \
    }                                                                         \
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

__global__ void page_v_smem_read_probe_kernel(__nv_fp4x2_e2m1* v, float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  __shared__ typename TestTraits::SharedStorage storage;
  constexpr uint32_t UPCAST_STRIDE_V = TestTraits::UPCAST_STRIDE_V;
  constexpr uint32_t KV_THR_LAYOUT_ROW = TestTraits::KV_THR_LAYOUT_ROW;
  constexpr uint32_t KV_THR_LAYOUT_COL = TestTraits::KV_THR_LAYOUT_COL;
  constexpr uint32_t CTA_TILE_KV = TestTraits::CTA_TILE_KV;
  constexpr uint32_t NUM_MMA_KV = TestTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_WARPS_Q = TestTraits::NUM_WARPS_Q;
  constexpr uint32_t packed_cols = TestTraits::HEAD_DIM_VO / 2;
  constexpr uint32_t sf_cols = TestTraits::HEAD_DIM_VO / fi::NVFP4_SF_VEC_SIZE;

  const dim3 tid = threadIdx;
  const uint32_t warp_idx = fi::get_warp_idx<TestTraits>(tid.y, tid.z);
  const uint32_t lane_idx = tid.x;

  fi::smem_t<TestTraits::SWIZZLE_MODE_KV> v_smem(storage.v_smem);
  uint32_t v_smem_offset_w = v_smem.template get_permuted_offset<UPCAST_STRIDE_V>(
      warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
      lane_idx % KV_THR_LAYOUT_COL);

  size_t thr_local_kv_offset[NUM_MMA_KV * KV_THR_LAYOUT_COL / 2 / NUM_WARPS_Q];
#pragma unroll
  for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
    const uint32_t entry_idx =
        warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL +
        KV_THR_LAYOUT_ROW * NUM_WARPS_Q * TestTraits::NUM_WARPS_KV * i;
    thr_local_kv_offset[i] =
        entry_idx * packed_cols +
        (lane_idx % KV_THR_LAYOUT_COL) * fi::upcast_size<__nv_fp4x2_e2m1>() / 2;
  }

  fi::page_produce_kv<true, TestTraits>(&storage, &v_smem_offset_w, v, 0,
                                        thr_local_kv_offset, CTA_TILE_KV, warp_idx, lane_idx);

  const uint32_t linear_tid = lane_idx + 32 * tid.y;
  for (uint32_t idx = linear_tid; idx < CTA_TILE_KV * sf_cols; idx += blockDim.x * blockDim.y) {
    storage.v_sf_smem[idx] = 0x38u;
  }

  fi::cp_async::commit_group();
  fi::cp_async::wait_group<0>();
  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for (uint32_t row = 0; row < CTA_TILE_KV; ++row) {
      for (uint32_t col = 0; col < 16; ++col) {
        out[row * 16 + col] = fi::get_v_value<TestTraits>(&v_smem, storage.v_sf_smem, row, col);
      }
    }
  }
#else
  if (threadIdx.x == 0) {
    out[0] = -1.f;
  }
#endif
}

int main() {
  constexpr int rows = TestTraits::CTA_TILE_KV;
  constexpr int packed_cols = TestTraits::HEAD_DIM_VO / 2;

  std::vector<uint8_t> host_v(rows * packed_cols);
  for (int row = 0; row < rows; ++row) {
    const uint8_t code = static_cast<uint8_t>(1 + (row % 7));
    const uint8_t packed = static_cast<uint8_t>(code | (code << 4));
    for (int col = 0; col < packed_cols; ++col) {
      host_v[row * packed_cols + col] = packed;
    }
  }

  __nv_fp4x2_e2m1* v = nullptr;
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&v, host_v.size()));
  CUDA_CHECK(cudaMalloc(&out, rows * 16 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(v, host_v.data(), host_v.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(out, 0, rows * 16 * sizeof(float)));

  page_v_smem_read_probe_kernel<<<1, dim3(32, 2, 1)>>>(v, out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> host_out(rows * 16);
  CUDA_CHECK(cudaMemcpy(host_out.data(), out, rows * 16 * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(v));
  CUDA_CHECK(cudaFree(out));

  for (int row = 0; row < rows; ++row) {
    std::printf("row %02d:", row);
    for (int col = 0; col < 16; ++col) {
      std::printf(" %.1f", host_out[row * 16 + col]);
    }
    std::printf("\n");
  }
  return 0;
}
