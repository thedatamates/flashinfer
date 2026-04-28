/*
 * Map SM120 block-scaled FP4 MMA UE4M3 scale-register byte ownership.
 */

#include <cuda_runtime.h>

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

struct ScaleOwner {
  int slot;
  float value;
};

__device__ __forceinline__ uint32_t pack_scale_with_slot(int slot, int lane, int atom) {
  uint32_t reg = 0x38383838u;
  if (slot >= 0) {
    const int slot_atom = slot >> 7;
    const int rem = slot & 127;
    const int slot_lane = rem >> 2;
    const int byte = rem & 3;
    if (slot_atom == atom && slot_lane == lane) {
      reg &= ~(0xffu << (byte * 8));
      reg |= 0x40u << (byte * 8);  // UE4M3 2.0
    }
  }
  return reg;
}

__device__ __forceinline__ void set_one_a(uint32_t* a, int a_idx, int lane) {
  const int a_lane = a_idx >> 5;
  const int pos = a_idx & 31;
  if (lane == a_lane) {
    a[pos >> 3] = 0x2u << (4u * (pos & 7));
  }
}

__device__ __forceinline__ void set_one_b(uint32_t* b, int b_idx, int lane) {
  const int b_lane = b_idx >> 5;
  const int pos = b_idx & 31;
  if (lane == b_lane) {
    b[pos >> 3] = 0x2u << (4u * (pos & 7));
  }
}

__device__ __forceinline__ int a_idx_for_row_k(int row, int k) {
  const int lane = (row & 7) * 4 + (k >> 4);
  const int reg = ((row >> 3) & 1) + 2 * ((k >> 3) & 1);
  const int nib = k & 7;
  return lane * 32 + reg * 8 + nib;
}

__device__ __forceinline__ int b_idx_for_col_k(int col, int k) {
  const int lane = (col & 7) * 4 + (k >> 4);
  const int reg = ((col >> 3) & 1) * 2 + ((k >> 3) & 1);
  const int nib = k & 7;
  return lane * 32 + reg * 8 + nib;
}

__global__ void map_a_scale_kernel(ScaleOwner* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  const int item = blockIdx.x;
  const int a_idx = item >> 7;
  const int scale_slot = item & 127;
  const int lane = threadIdx.x & 31;
  const int a_pos = a_idx & 31;
  const int a_lane = a_idx >> 5;
  const int row = (a_lane >> 2) + ((a_pos >> 3) & 1) * 8;
  const int k = (a_lane & 3) * 16 + ((a_pos >> 4) & 1) * 8 + (a_pos & 7);
  const int b_idx = b_idx_for_col_k(0, k);

  uint32_t a[4] = {0, 0, 0, 0};
  uint32_t b[4] = {0, 0, 0, 0};
  set_one_a(a, a_idx, lane);
  set_one_b(b, b_idx, lane);
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const uint32_t scale_a = pack_scale_with_slot(scale_slot, lane, 0);
  constexpr uint32_t scale_b = 0x38383838u;
  flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32<flashinfer::mma::MMAMode::kInit>(
      acc, a, b, scale_a, scale_b, scale_b);
  float local = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    local = fmaxf(local, acc[i]);
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, mask));
  }
  if (lane == 0 && local > 1.5f) {
    out[a_idx] = {scale_slot, local};
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = {-1, -1.f};
  }
#endif
}

__global__ void map_b_scale_kernel(ScaleOwner* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  const int item = blockIdx.x;
  const int b_idx = item >> 8;
  const int scale_slot = item & 255;
  const int lane = threadIdx.x & 31;
  const int b_pos = b_idx & 31;
  const int b_lane = b_idx >> 5;
  const int col = (b_lane >> 2) + ((b_pos >> 4) & 1) * 8;
  const int k = (b_lane & 3) * 16 + ((b_pos >> 3) & 1) * 8 + (b_pos & 7);
  const int a_idx = a_idx_for_row_k(0, k);

  uint32_t a[4] = {0, 0, 0, 0};
  uint32_t b[4] = {0, 0, 0, 0};
  set_one_a(a, a_idx, lane);
  set_one_b(b, b_idx, lane);
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  constexpr uint32_t scale_a = 0x38383838u;
  const uint32_t scale_b0 = pack_scale_with_slot(scale_slot, lane, 0);
  const uint32_t scale_b1 = pack_scale_with_slot(scale_slot, lane, 1);
  flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32<flashinfer::mma::MMAMode::kInit>(
      acc, a, b, scale_a, scale_b0, scale_b1);
  float local = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    local = fmaxf(local, acc[i]);
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    local = fmaxf(local, __shfl_xor_sync(0xffffffff, local, mask));
  }
  if (lane == 0 && local > 1.5f) {
    out[b_idx] = {scale_slot, local};
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = {-1, -1.f};
  }
#endif
}

int main() {
  constexpr int kSlots = 1024;
  ScaleOwner* d_a = nullptr;
  ScaleOwner* d_b = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, kSlots * sizeof(ScaleOwner)));
  CUDA_CHECK(cudaMalloc(&d_b, kSlots * sizeof(ScaleOwner)));
  CUDA_CHECK(cudaMemset(d_a, 0xff, kSlots * sizeof(ScaleOwner)));
  CUDA_CHECK(cudaMemset(d_b, 0xff, kSlots * sizeof(ScaleOwner)));
  map_a_scale_kernel<<<kSlots * 128, 32>>>(d_a);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  map_b_scale_kernel<<<kSlots * 256, 32>>>(d_b);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<ScaleOwner> a(kSlots), b(kSlots);
  CUDA_CHECK(cudaMemcpy(a.data(), d_a, kSlots * sizeof(ScaleOwner), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(b.data(), d_b, kSlots * sizeof(ScaleOwner), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));

  std::printf("A scale owners first 8 lanes:\n");
  for (int lane = 0; lane < 8; ++lane) {
    std::printf("A lane %02d:\n", lane);
    for (int reg = 0; reg < 4; ++reg) {
      std::printf("  reg%d:", reg);
      for (int nib = 0; nib < 8; ++nib) {
        const int idx = lane * 32 + reg * 8 + nib;
        std::printf(" %d", a[idx].slot);
      }
      std::printf("\n");
    }
  }
  std::printf("B scale owners first 8 lanes:\n");
  for (int lane = 0; lane < 8; ++lane) {
    std::printf("B lane %02d:\n", lane);
    for (int reg = 0; reg < 4; ++reg) {
      std::printf("  reg%d:", reg);
      for (int nib = 0; nib < 8; ++nib) {
        const int idx = lane * 32 + reg * 8 + nib;
        std::printf(" %d", b[idx].slot);
      }
      std::printf("\n");
    }
  }
  return 0;
}
