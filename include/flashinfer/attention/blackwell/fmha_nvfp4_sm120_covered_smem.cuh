#pragma once

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace flashinfer::attention::blackwell::sm120_nvfp4 {

struct SentinelZero {};
struct SentinelNegInf {};
struct SentinelE4M3One {};
struct CoveredSmemNoInit {};

template <typename Element, typename Sentinel>
struct SmemFillTraits;

template <typename Element>
struct SmemFillTraits<Element, SentinelZero> {
  static constexpr bool kIsBytewise = true;
  __device__ static uint8_t byte_value() { return 0u; }
};

template <typename Element>
struct SmemFillTraits<Element, SentinelE4M3One> {
  static constexpr bool kIsBytewise = true;
  __device__ static uint8_t byte_value() { return 0x38u; }
};

template <>
struct SmemFillTraits<__nv_bfloat16, SentinelNegInf> {
  static constexpr bool kIsBytewise = false;
  __device__ static __nv_bfloat16 value() {
    return __float2bfloat16(-INFINITY);
  }
};

template <>
struct SmemFillTraits<float, SentinelNegInf> {
  static constexpr bool kIsBytewise = false;
  __device__ static float value() { return -INFINITY; }
};

// Constructs a CUTE smem tensor and collectively initializes the full physical
// extent to the consumer's sentinel. Use this for any raw smem region whose
// producer may write only a partition subset before a CUTLASS consumer reads a
// full fragment.
template <typename Element, typename Layout, typename Sentinel,
          typename SmemPointer = Element*>
class CoveredSmemTile {
 public:
  using TensorT = decltype(cute::make_tensor(
      cute::make_smem_ptr(std::declval<SmemPointer>()), Layout{}));
  using FillT = SmemFillTraits<Element, Sentinel>;

  template <typename BarrierId>
  __device__ CoveredSmemTile(SmemPointer smem_ptr, Layout layout,
                             int participating_threads, BarrierId barrier_id)
      : CoveredSmemTile(smem_ptr, layout, participating_threads, barrier_id,
                        static_cast<int>(threadIdx.x)) {}

  template <typename BarrierId>
  __device__ CoveredSmemTile(SmemPointer smem_ptr, Layout layout,
                             int participating_threads, BarrierId barrier_id,
                             int participant_idx)
      : smem_ptr_(smem_ptr),
        tensor_(cute::make_tensor(cute::make_smem_ptr(smem_ptr), layout)) {
    fill(participating_threads, participant_idx);
    cutlass::arch::NamedBarrier::sync(participating_threads,
                                      static_cast<int>(barrier_id));
  }

  __device__ CoveredSmemTile(CoveredSmemNoInit, SmemPointer smem_ptr,
                             Layout layout)
      : smem_ptr_(smem_ptr),
        tensor_(cute::make_tensor(cute::make_smem_ptr(smem_ptr), layout)) {}

  template <typename BarrierId>
  __device__ void fill_and_sync(int participating_threads,
                                BarrierId barrier_id) {
    fill_and_sync(participating_threads, barrier_id,
                  static_cast<int>(threadIdx.x));
  }

  template <typename BarrierId>
  __device__ void fill_and_sync(int participating_threads, BarrierId barrier_id,
                                int participant_idx) {
    fill(participating_threads, participant_idx);
    cutlass::arch::NamedBarrier::sync(participating_threads,
                                      static_cast<int>(barrier_id));
  }

  __device__ TensorT& tensor() { return tensor_; }
  __device__ TensorT const& tensor() const { return tensor_; }

  template <typename... Coords>
  __device__ auto operator()(Coords&&... coords) {
    return tensor_(static_cast<Coords&&>(coords)...);
  }

  template <typename... Coords>
  __device__ auto operator()(Coords&&... coords) const {
    return tensor_(static_cast<Coords&&>(coords)...);
  }

  __device__ auto layout() const { return tensor_.layout(); }
  __device__ auto data() const { return tensor_.data(); }
  __device__ auto shape() const { return tensor_.shape(); }

  __device__ operator TensorT&() { return tensor_; }
  __device__ operator TensorT const&() const { return tensor_; }

 private:
  __device__ void fill(int participating_threads, int participant_idx) {
    constexpr int kCosize = cute::cosize_v<Layout>;
    if constexpr (FillT::kIsBytewise) {
      constexpr int kTotalBytes =
          (kCosize * cute::sizeof_bits_v<Element> + 7) / 8;
      uint8_t* raw = cute::recast_ptr<uint8_t>(smem_ptr_);
      const uint8_t fill_byte = FillT::byte_value();
      for (int idx = participant_idx; idx < kTotalBytes;
           idx += participating_threads) {
        raw[idx] = fill_byte;
      }
    } else {
      static_assert(sizeof(Element) >= 1,
                    "Non-bytewise sentinel requires byte-addressable Element.");
      const Element fill_value = FillT::value();
      for (int idx = participant_idx; idx < kCosize;
           idx += participating_threads) {
        smem_ptr_[idx] = fill_value;
      }
    }
  }

  SmemPointer smem_ptr_;
  TensorT tensor_;
};

template <typename Element, typename Layout, typename SmemPointer = Element*>
using ZeroSmemTile =
    CoveredSmemTile<Element, Layout, SentinelZero, SmemPointer>;

template <typename Element, typename Layout, typename SmemPointer = Element*>
using NegInfSmemTile =
    CoveredSmemTile<Element, Layout, SentinelNegInf, SmemPointer>;

template <typename Element, typename Layout, typename SmemPointer = Element*>
using E4M3OneSmemTile =
    CoveredSmemTile<Element, Layout, SentinelE4M3One, SmemPointer>;

}  // namespace flashinfer::attention::blackwell::sm120_nvfp4
