/*
 * Probe FP4 V shared-memory reads used by the native SM120 prefill PV path.
 *
 * The native PV path reads V from smem with get_v_value() instead of ldmatrix.
 * This checks that the logical row/column coordinates read by get_v_value()
 * match the layout written by the production produce_kv()/produce_kv_sf()
 * cp.async loaders.
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

__global__ void v_smem_read_probe_kernel(__nv_fp4x2_e2m1* v, uint8_t* v_sf, float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  __shared__ typename TestTraits::SharedStorage storage;
  constexpr uint32_t UPCAST_STRIDE_V = TestTraits::UPCAST_STRIDE_V;
  constexpr uint32_t KV_THR_LAYOUT_ROW = TestTraits::KV_THR_LAYOUT_ROW;
  constexpr uint32_t KV_THR_LAYOUT_COL = TestTraits::KV_THR_LAYOUT_COL;
  constexpr uint32_t CTA_TILE_KV = TestTraits::CTA_TILE_KV;
  constexpr uint32_t packed_cols = TestTraits::HEAD_DIM_VO / 2;

  const dim3 tid = threadIdx;
  const uint32_t warp_idx = fi::get_warp_idx<TestTraits>(tid.y, tid.z);
  const uint32_t lane_idx = tid.x;

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

  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for (uint32_t row = 0; row < 16; ++row) {
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
  constexpr int sf_cols = TestTraits::HEAD_DIM_VO / fi::NVFP4_SF_VEC_SIZE;

  std::vector<uint8_t> host_v(rows * packed_cols);
  std::vector<uint8_t> host_sf(rows * sf_cols, 0x38);  // UE4M3 1.0
  for (int row = 0; row < rows; ++row) {
    // E2M1 positive values: 0.5, 1, 1.5, 2, 3, 4, 6, then repeat.
    const uint8_t code = static_cast<uint8_t>(1 + (row % 7));
    const uint8_t packed = static_cast<uint8_t>(code | (code << 4));
    for (int col = 0; col < packed_cols; ++col) {
      host_v[row * packed_cols + col] = packed;
    }
  }

  __nv_fp4x2_e2m1* v = nullptr;
  uint8_t* v_sf = nullptr;
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&v, host_v.size()));
  CUDA_CHECK(cudaMalloc(&v_sf, host_sf.size()));
  CUDA_CHECK(cudaMalloc(&out, 16 * 16 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(v, host_v.data(), host_v.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(v_sf, host_sf.data(), host_sf.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(out, 0, 16 * 16 * sizeof(float)));

  v_smem_read_probe_kernel<<<1, dim3(32, 2, 1)>>>(v, v_sf, out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float host_out[16 * 16];
  CUDA_CHECK(cudaMemcpy(host_out, out, sizeof(host_out), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(v));
  CUDA_CHECK(cudaFree(v_sf));
  CUDA_CHECK(cudaFree(out));

  for (int row = 0; row < 16; ++row) {
    std::printf("row %02d:", row);
    for (int col = 0; col < 16; ++col) {
      std::printf(" %.1f", host_out[row * 16 + col]);
    }
    std::printf("\n");
  }
  return 0;
}
