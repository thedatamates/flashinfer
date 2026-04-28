/*
 * Map physical SM120 FP4 MMA register/nibble slots by one-hot pair probing.
 *
 * One block probes one physical A slot against one physical B slot. A nonzero
 * output means the two slots share the same hidden K lane in the native MMA.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
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

struct Match {
  int c_owner;
  int count;
};

__device__ __forceinline__ int c_linear_from_owner(int owner) {
  const int lane = owner >> 3;
  const int reg = owner & 7;
  const int row = (lane >> 2) + ((reg & 0x2) ? 8 : 0);
  const int col = ((reg >> 2) * 8) + ((lane & 0x3) * 2) + (reg & 0x1);
  return row * 16 + col;
}

__global__ void map_slots_kernel(Match* out) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
  const int pair = blockIdx.x;
  const int a_idx = pair >> 10;
  const int b_idx = pair & 1023;
  const int a_lane = a_idx >> 5;
  const int a_pos = a_idx & 31;
  const int b_lane = b_idx >> 5;
  const int b_pos = b_idx & 31;
  const int lane = threadIdx.x & 31;

  uint32_t a[4] = {0, 0, 0, 0};
  uint32_t b[4] = {0, 0, 0, 0};
  if (lane == a_lane) {
    a[a_pos >> 3] = 0x2u << (4u * (a_pos & 7));
  }
  if (lane == b_lane) {
    b[b_pos >> 3] = 0x2u << (4u * (b_pos & 7));
  }

  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  constexpr uint32_t scale = 0x38383838u;
  flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32<flashinfer::mma::MMAMode::kInit>(
      acc, a, b, scale, scale, scale);

  int local_count = 0;
  int local_first = -1;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (acc[i] > 0.25f) {
      local_first = lane * 8 + i;
      local_count++;
    }
  }

  int total_count = local_count;
  int first = local_first;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    const int other_count = __shfl_xor_sync(0xffffffff, total_count, mask);
    const int other_first = __shfl_xor_sync(0xffffffff, first, mask);
    if (first < 0 || (other_first >= 0 && other_first < first)) {
      first = other_first;
    }
    total_count += other_count;
  }
  if (lane == 0) {
    out[pair] = {first >= 0 ? c_linear_from_owner(first) : -1, total_count};
  }
#else
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[0] = {-1, -1};
  }
#endif
}

static int a_lane(int idx) { return idx >> 5; }
static int a_reg(int idx) { return (idx & 31) >> 3; }
static int a_nib(int idx) { return idx & 7; }
static int b_lane(int idx) { return idx >> 5; }
static int b_reg(int idx) { return (idx & 31) >> 3; }
static int b_nib(int idx) { return idx & 7; }

int main() {
  constexpr int kSlots = 1024;
  constexpr int kPairs = kSlots * kSlots;
  Match* d_matches = nullptr;
  CUDA_CHECK(cudaMalloc(&d_matches, kPairs * sizeof(Match)));
  map_slots_kernel<<<kPairs, 32>>>(d_matches);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<Match> matches(kPairs);
  CUDA_CHECK(cudaMemcpy(matches.data(), d_matches, kPairs * sizeof(Match),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_matches));

  std::vector<std::vector<int>> a_to_b(kSlots);
  std::vector<std::vector<int>> a_to_c(kSlots);
  std::vector<int> a_row(kSlots, -1);
  std::vector<int> b_col(kSlots, -1);
  int bad_count = 0;
  int total = 0;
  int count_hist[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int a = 0; a < kSlots; ++a) {
    for (int b = 0; b < kSlots; ++b) {
      const Match m = matches[a * kSlots + b];
      if (m.count > 0) {
        total++;
        count_hist[m.count < 7 ? m.count : 7]++;
        if (m.count != 1 || m.c_owner < 0) {
          bad_count++;
          continue;
        }
        const int row = m.c_owner / 16;
        const int col = m.c_owner - row * 16;
        if (a_row[a] < 0) {
          a_row[a] = row;
        } else if (a_row[a] != row) {
          bad_count++;
        }
        if (b_col[b] < 0) {
          b_col[b] = col;
        } else if (b_col[b] != col) {
          bad_count++;
        }
        a_to_b[a].push_back(b);
        a_to_c[a].push_back(m.c_owner);
      }
    }
  }

  std::unordered_map<int, int> b_min_to_class;
  int next_class = 0;
  std::vector<int> a_class(kSlots, -1);
  for (int a = 0; a < kSlots; ++a) {
    if (a_to_b[a].empty()) {
      continue;
    }
    std::sort(a_to_b[a].begin(), a_to_b[a].end());
    const int key = a_to_b[a].front();
    auto it = b_min_to_class.find(key);
    if (it == b_min_to_class.end()) {
      it = b_min_to_class.emplace(key, next_class++).first;
    }
    a_class[a] = it->second;
  }

  std::printf("total_matches=%d bad=%d k_classes=%d count_hist:", total, bad_count, next_class);
  for (int i = 0; i < 8; ++i) {
    std::printf(" %d:%d", i, count_hist[i]);
  }
  std::printf("\n");
  for (int idx = 0; idx < 4; ++idx) {
    std::printf("A sample idx %d lane%d reg%d nib%d:", idx, idx >> 5, (idx & 31) >> 3, idx & 7);
    for (size_t j = 0; j < a_to_b[idx].size() && j < 24; ++j) {
      const int b = a_to_b[idx][j];
      std::printf(" b(l%d r%d n%d)->c%d", b >> 5, (b & 31) >> 3, b & 7, a_to_c[idx][j]);
    }
    std::printf("\n");
  }
  for (int lane = 0; lane < 8; ++lane) {
    std::printf("A lane %02d:\n", lane);
    for (int reg = 0; reg < 4; ++reg) {
      std::printf("  reg%d:", reg);
      for (int nib = 0; nib < 8; ++nib) {
        const int idx = lane * 32 + reg * 8 + nib;
        std::printf(" [%02d r%02d k%02d]", nib, a_row[idx], a_class[idx]);
      }
      std::printf("\n");
    }
  }
  for (int lane = 0; lane < 8; ++lane) {
    std::printf("B lane %02d:\n", lane);
    for (int reg = 0; reg < 4; ++reg) {
      std::printf("  reg%d:", reg);
      for (int nib = 0; nib < 8; ++nib) {
        const int idx = lane * 32 + reg * 8 + nib;
        int cls = -1;
        for (int a = 0; a < kSlots && cls < 0; ++a) {
          if (std::binary_search(a_to_b[a].begin(), a_to_b[a].end(), idx)) {
            cls = a_class[a];
          }
        }
        std::printf(" [%02d c%02d k%02d]", nib, b_col[idx], cls);
      }
      std::printf("\n");
    }
  }
  return 0;
}
