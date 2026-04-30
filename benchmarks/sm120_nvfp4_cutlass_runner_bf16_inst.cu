// Local BF16 instantiations for the SM120 NVFP4 CUTLASS runner used by
// sm120_nvfp4_cutlass_fused_attention_d512.cu. FlashInfer's normal Python path
// generates these TUs through its JIT cache; this standalone extension needs
// the same concrete launchers linked directly.

#include <cuda_bf16.h>

#include <flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h>

namespace flashinfer {
namespace gemm {

INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 128, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 128, 128, 256, 1, 1, 1, _1SM)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(__nv_bfloat16, 256, 128, 128, 1, 1, 1, _1SM)

}  // namespace gemm
}  // namespace flashinfer
