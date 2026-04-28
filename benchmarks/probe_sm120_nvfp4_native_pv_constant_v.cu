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

__global__ void probe_kernel(float* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  __shared__ typename TestTraits::SharedStorage storage;
  fi::smem_t<TestTraits::SWIZZLE_MODE_KV> v_smem(storage.v_smem);
  constexpr uint32_t rows = TestTraits::CTA_TILE_KV;
  constexpr uint32_t packed_cols = TestTraits::HEAD_DIM_VO / 2;
  constexpr uint32_t sf_cols = TestTraits::HEAD_DIM_VO / fi::NVFP4_SF_VEC_SIZE;
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
    ptr[byte_in_vec] = 0x22u;
  }
  for (uint32_t idx = linear_tid; idx < total_sf_bytes; idx += nthreads) {
    storage.v_sf_smem[idx] = 0x38u;
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
        const uint32_t row = (threadIdx.x >> 2) + 8 * ((reg % 4) / 2);
        const uint32_t col = mma_kv * 16 + 2 * (threadIdx.x & 3) + 8 * (reg / 4) + (reg & 1);
        const uint32_t onehot_col = blockIdx.x;
        const float dense =
            0.000001f * float(1 + ((threadIdx.x * 13 + mma_kv * 7 + reg * 5) % 97));
        s_frag[mma_q][mma_kv][reg] = onehot_col < 64 ? (col == onehot_col ? 1.f : 0.f) : dense;
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

  const uint32_t base = (blockIdx.x * nthreads + linear_tid) * 10;
  out[base + 0] = d[0][0];
  out[base + 1] = d[0][1];
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const uint32_t row_group = (i % 4) / 2;
    out[base + 2 + i] = d[0][row_group] != 0.f ? o_frag[0][0][i] / d[0][row_group] : -1.f;
  }
#else
  if (threadIdx.x == 0) {
    out[0] = -1.f;
  }
#endif
}

int main() {
  constexpr int threads = 64;
  constexpr int patterns = 65;
  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, patterns * threads * 10 * sizeof(float)));
  CUDA_CHECK(cudaMemset(out, 0, patterns * threads * 10 * sizeof(float)));
  probe_kernel<<<patterns, dim3(32, 2, 1)>>>(out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float host[patterns * threads * 10];
  CUDA_CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(out));
  for (int pattern = 0; pattern < patterns; ++pattern) {
    int bad = 0;
    float min_ratio = 1.0e20f;
    float max_ratio = -1.0e20f;
    for (int tid = 0; tid < threads; ++tid) {
      const int base = (pattern * threads + tid) * 10;
      for (int i = 0; i < 8; ++i) {
        const float ratio = host[base + 2 + i];
        min_ratio = fminf(min_ratio, ratio);
        max_ratio = fmaxf(max_ratio, ratio);
        bad += fabsf(ratio - 1.f) > 0.001f;
      }
    }
    const int sample_base = pattern * threads * 10;
    std::printf("pattern %02d bad %d min %.6f max %.6f d0 %.6f d1 %.6f o0 %.6f o2 %.6f\n",
                pattern, bad, min_ratio, max_ratio, host[sample_base + 0], host[sample_base + 1],
                host[sample_base + 2], host[sample_base + 4]);
  }
  return 0;
}
