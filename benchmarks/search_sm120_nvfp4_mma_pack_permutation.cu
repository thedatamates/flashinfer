/*
 * Exhaustive register/nibble packing probe for SM120 block-scaled FP4 MMA.
 *
 * CUTE exposes the logical per-lane A/B layouts, but hand-built uint32_t
 * fragments still need the physical register/nibble -> logical value-index
 * mapping. This probe searches small bit-permutation spaces for A and B and
 * scores row/column preservation plus A/B K-axis agreement.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <flashinfer/mma.cuh>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                        \
    cudaError_t status = (expr);                                              \
    if (status != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(status));                               \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

struct Result {
  float max_abs_err;
  float range;
  float sample0;
  float sample1;
  int idx_a;
  int idx_b;
};

__device__ __forceinline__ float e2m1_decode(uint32_t code) {
  code &= 0xfu;
  const float mag = (code == 0 || code == 8)   ? 0.f
                    : (code == 1 || code == 9) ? 0.5f
                    : (code == 2 || code == 10)
                        ? 1.f
                        : (code == 3 || code == 11)
                            ? 1.5f
                            : (code == 4 || code == 12)
                                ? 2.f
                                : (code == 5 || code == 13) ? 3.f : (code == 6 || code == 14) ? 4.f : 6.f;
  return (code & 0x8u) ? -mag : mag;
}

__device__ __forceinline__ uint32_t pattern_a_code(uint32_t k) {
  constexpr uint32_t codes[16] = {
      0x1u, 0xau, 0x3u, 0xcu, 0x5u, 0xeu, 0x7u, 0x9u,
      0x2u, 0xbu, 0x4u, 0xdu, 0x6u, 0xfu, 0x1u, 0x8u};
  return codes[(k * 5u + 3u) & 0xfu];
}

__device__ __forceinline__ uint32_t pattern_b_code(uint32_t k) {
  constexpr uint32_t codes[16] = {
      0x2u, 0x9u, 0x5u, 0xfu, 0x1u, 0xdu, 0x7u, 0xbu,
      0x4u, 0xeu, 0x3u, 0xcu, 0x6u, 0xau, 0x1u, 0x8u};
  return codes[(k * 7u + 1u) & 0xfu];
}

__device__ __forceinline__ uint32_t apply_perm(uint32_t x, const int* perm, uint32_t nbits) {
  uint32_t out = 0;
#pragma unroll
  for (uint32_t bit = 0; bit < 5; ++bit) {
    if (bit < nbits) {
      out |= ((x >> bit) & 1u) << perm[bit];
    }
  }
  return out;
}

__device__ __forceinline__ uint32_t a_logical_k(uint32_t lane, uint32_t v) {
  return ((lane >> 2) & 0x7u) + 16u * (v & 0x3u) + 8u * ((v >> 3) & 0x1u);
}

__device__ __forceinline__ uint32_t b_logical_k(uint32_t lane, uint32_t v) {
  return ((lane >> 2) & 0x7u) + 8u * (v & 0x7u);
}

__device__ __forceinline__ uint32_t pack_a_by_perm(const int* perm_a, bool use_pattern) {
  const uint32_t lane = threadIdx.x & 31u;
  uint32_t regs[4] = {0, 0, 0, 0};
#pragma unroll
  for (uint32_t p = 0; p < 32; ++p) {
    const uint32_t v = apply_perm(p, perm_a, 5);
    const uint32_t code = use_pattern ? pattern_a_code(a_logical_k(lane, v)) : 0x2u;
    regs[p >> 3] |= code << (4u * (p & 0x7u));
  }
  return regs[threadIdx.y & 3u];
}

__device__ __forceinline__ uint32_t make_a_reg_by_perm(uint32_t reg, const int* perm_a,
                                                       bool use_pattern) {
  const uint32_t lane = threadIdx.x & 31u;
  uint32_t word = 0;
#pragma unroll
  for (uint32_t nib = 0; nib < 8; ++nib) {
    const uint32_t p = reg * 8u + nib;
    const uint32_t v = apply_perm(p, perm_a, 5);
    const uint32_t code = use_pattern ? pattern_a_code(a_logical_k(lane, v)) : 0x2u;
    word |= code << (4u * nib);
  }
  return word;
}

__device__ __forceinline__ uint32_t make_b_reg_by_perm(uint32_t reg, const int* perm_b,
                                                       bool use_pattern) {
  const uint32_t lane = threadIdx.x & 31u;
  uint32_t word = 0;
#pragma unroll
  for (uint32_t nib = 0; nib < 8; ++nib) {
    const uint32_t p = reg * 8u + nib;
    const uint32_t v = apply_perm(p, perm_b, 4);
    const uint32_t code = use_pattern ? pattern_b_code(b_logical_k(lane, v)) : 0x2u;
    word |= code << (4u * nib);
  }
  return word;
}

__global__ void score_pack_perms_kernel(const int* perms_a, int num_a, const int* perms_b,
                                        int num_b, Result* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  const int pair = blockIdx.x;
  const int idx_a = pair / num_b;
  const int idx_b = pair - idx_a * num_b;
  const int* perm_a = perms_a + idx_a * 5;
  const int* perm_b = perms_b + idx_b * 4;

  uint32_t a[4];
  uint32_t b[4];
#pragma unroll
  for (uint32_t reg = 0; reg < 4; ++reg) {
    a[reg] = make_a_reg_by_perm(reg, perm_a, true);
  }
#pragma unroll
  for (uint32_t reg = 0; reg < 2; ++reg) {
    b[reg] = make_b_reg_by_perm(reg, perm_b, true);
    b[reg + 2] = b[reg];
  }

  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  constexpr uint32_t scale = 0x38383838u;
  flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32<flashinfer::mma::MMAMode::kInit>(
      acc, a, b, scale, scale, scale);

  float expected = 0.f;
#pragma unroll
  for (uint32_t k = 0; k < 64; ++k) {
    expected += e2m1_decode(pattern_a_code(k)) * e2m1_decode(pattern_b_code(k));
  }

  float local_max_abs_err = 0.f;
  float local_min = acc[0];
  float local_max = acc[0];
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    local_max_abs_err = fmaxf(local_max_abs_err, fabsf(acc[i] - expected));
    local_min = fminf(local_min, acc[i]);
    local_max = fmaxf(local_max, acc[i]);
  }
#pragma unroll
  for (uint32_t mask = 16; mask > 0; mask >>= 1) {
    local_max_abs_err = fmaxf(local_max_abs_err, __shfl_xor_sync(0xffffffff, local_max_abs_err, mask));
    local_min = fminf(local_min, __shfl_xor_sync(0xffffffff, local_min, mask));
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, mask));
  }
  if ((threadIdx.x & 31) == 0) {
    out[pair] = {local_max_abs_err, local_max - local_min, acc[0], acc[1], idx_a, idx_b};
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = {-1.f, -1.f, -1.f, -1.f, 0, 0};
  }
#endif
}

static std::vector<int> make_perms(int nbits) {
  std::vector<int> base(nbits);
  for (int i = 0; i < nbits; ++i) {
    base[i] = i;
  }
  std::vector<int> perms;
  do {
    perms.insert(perms.end(), base.begin(), base.end());
  } while (std::next_permutation(base.begin(), base.end()));
  return perms;
}

int main() {
  std::vector<int> perms_a = make_perms(5);
  std::vector<int> perms_b = make_perms(4);
  const int num_a = static_cast<int>(perms_a.size() / 5);
  const int num_b = static_cast<int>(perms_b.size() / 4);
  const int pairs = num_a * num_b;

  int* d_perms_a = nullptr;
  int* d_perms_b = nullptr;
  Result* d_results = nullptr;
  CUDA_CHECK(cudaMalloc(&d_perms_a, perms_a.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_perms_b, perms_b.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_results, pairs * sizeof(Result)));
  CUDA_CHECK(cudaMemcpy(d_perms_a, perms_a.data(), perms_a.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_perms_b, perms_b.data(), perms_b.size() * sizeof(int),
                        cudaMemcpyHostToDevice));

  score_pack_perms_kernel<<<pairs, 32>>>(d_perms_a, num_a, d_perms_b, num_b, d_results);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<Result> results(pairs);
  CUDA_CHECK(cudaMemcpy(results.data(), d_results, pairs * sizeof(Result),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_perms_a));
  CUDA_CHECK(cudaFree(d_perms_b));
  CUDA_CHECK(cudaFree(d_results));

  std::sort(results.begin(), results.end(), [](const Result& a, const Result& b) {
    if (a.max_abs_err != b.max_abs_err) {
      return a.max_abs_err < b.max_abs_err;
    }
    return a.range < b.range;
  });

  std::printf("num_a=%d num_b=%d pairs=%d\n", num_a, num_b, pairs);
  for (int rank = 0; rank < 20 && rank < pairs; ++rank) {
    const Result& r = results[rank];
    std::printf("rank %02d err=%g range=%g sample=[%g,%g] a=%d perm_a=[",
                rank, r.max_abs_err, r.range, r.sample0, r.sample1, r.idx_a);
    for (int i = 0; i < 5; ++i) {
      std::printf("%s%d", i ? "," : "", perms_a[r.idx_a * 5 + i]);
    }
    std::printf("] b=%d perm_b=[", r.idx_b);
    for (int i = 0; i < 4; ++i) {
      std::printf("%s%d", i ? "," : "", perms_b[r.idx_b * 4 + i]);
    }
    std::printf("]\n");
  }
  return 0;
}
