/*
 * Standalone probe for the D512/VO256 native NVFP4 PV helper used by prefill.
 *
 * This isolates compute_sfm_v_native_fp4 from the full FlashAttention prefill
 * kernel so we can distinguish helper-level MMA/fragment issues from scheduler,
 * cp.async, and generated-kernel interactions.
 */

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

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

__device__ __forceinline__ uint8_t fp32_to_ue4m3_byte_probe(float x) {
  __nv_fp8_e4m3 y = static_cast<__nv_fp8_e4m3>(x);
  return y.__x;
}

__global__ void native_pv_helper_probe_kernel(float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  __shared__ typename TestTraits::SharedStorage storage;
  fi::smem_t<TestTraits::SWIZZLE_MODE_KV> v_smem(storage.v_smem);

  constexpr uint32_t rows = TestTraits::CTA_TILE_KV;
  constexpr uint32_t head_dim = TestTraits::HEAD_DIM_VO;
  constexpr uint32_t packed_cols = head_dim / 2;
  constexpr uint32_t sf_cols = head_dim / fi::NVFP4_SF_VEC_SIZE;
  constexpr uint32_t total_v_bytes = rows * packed_cols;
  constexpr uint32_t total_sf_bytes = rows * sf_cols;
  const uint32_t linear_tid = threadIdx.x + blockDim.x * threadIdx.y;
  const uint32_t nthreads = blockDim.x * blockDim.y;

  for (uint32_t idx = linear_tid; idx < total_v_bytes; idx += nthreads) {
    const uint32_t row = idx / packed_cols;
    const uint32_t packed_col = idx % packed_cols;
    const uint32_t smem_col = packed_col / 8;
    const uint32_t byte_in_vec = packed_col % 8;
    const uint32_t offset =
        fi::smem_t<TestTraits::SWIZZLE_MODE_KV>::template get_permuted_offset<
            TestTraits::UPCAST_STRIDE_V>(row, smem_col);
    auto* ptr = reinterpret_cast<uint8_t*>(v_smem.base + offset);
    ptr[byte_in_vec] = static_cast<uint8_t>((idx * 17 + 29) & 0xffu);
  }
  for (uint32_t idx = linear_tid; idx < total_sf_bytes; idx += nthreads) {
    storage.v_sf_smem[idx] = fp32_to_ue4m3_byte_probe(0.25f);
  }
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
  const int blocks_z = argc > 2 ? std::atoi(argv[2]) : 1;
  constexpr int threads = 64;
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, blocks_z * threads * 10 * sizeof(float)));
  for (int i = 0; i < repeat; ++i) {
    CUDA_CHECK(cudaMemset(out, 0, blocks_z * threads * 10 * sizeof(float)));
    native_pv_helper_probe_kernel<<<dim3(1, 1, blocks_z), dim3(32, 2, 1)>>>(out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  float sample[10];
  CUDA_CHECK(cudaMemcpy(sample, out, sizeof(sample), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));
  std::printf("sample=[");
  for (int i = 0; i < 10; ++i) {
    std::printf("%s%g", i ? "," : "", sample[i]);
  }
  std::printf("]\n");
  return 0;
}
