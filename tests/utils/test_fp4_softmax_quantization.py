import pytest
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer import SfLayout
from flashinfer.fp4_quantization import (
    _select_nvfp4_softmax_quant_threads,
    e2m1_and_ufp8sf_scale_to_float,
)


def _is_nvfp4_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    cuda_version = torch.version.cuda
    return major >= 10 and cuda_version is not None and tuple(
        int(part) for part in cuda_version.split(".")[:2]
    ) >= (12, 8)


@pytest.mark.parametrize("shape", [(64, 1024), (128, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_nvfp4_softmax_quantize_matches_separate_softmax_quantize(shape, dtype):
    if not _is_nvfp4_supported():
        pytest.skip("NVFP4 softmax quantization requires SM100+ and CUDA >= 12.8")

    torch.manual_seed(42)
    logits = torch.randn(shape, device="cuda", dtype=dtype)
    global_scale = torch.tensor([448.0 * 6.0], device="cuda", dtype=torch.float32)

    probs = torch.softmax(logits, dim=-1)
    ref_packed, ref_scale = flashinfer.nvfp4_quantize(
        probs,
        global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    packed, scale = flashinfer.nvfp4_softmax_quantize(logits, global_scale)

    assert packed.shape == ref_packed.shape
    assert scale.shape == ref_scale.shape
    assert packed.dtype == torch.uint8
    assert scale.dtype == torch.uint8

    packed_match = (packed == ref_packed).float().mean().item()
    scale_match = (scale == ref_scale).float().mean().item()
    assert packed_match > 0.98
    assert scale_match > 0.98

    dequantized = e2m1_and_ufp8sf_scale_to_float(
        packed,
        scale,
        1 / global_scale,
        sf_vec_size=16,
        ufp8_type=1,
        is_sf_swizzled_layout=True,
    ).to("cuda")
    ref_dequantized = e2m1_and_ufp8sf_scale_to_float(
        ref_packed,
        ref_scale,
        1 / global_scale,
        sf_vec_size=16,
        ufp8_type=1,
        is_sf_swizzled_layout=True,
    ).to("cuda")

    assert not torch.isnan(dequantized).any()
    assert not torch.isinf(dequantized).any()
    assert (
        F.cosine_similarity(
            dequantized.reshape(-1),
            ref_dequantized.reshape(-1),
            dim=0,
        ).item()
        > 0.999
    )


def test_nvfp4_softmax_quantize_thread_selector():
    assert _select_nvfp4_softmax_quant_threads(64, 8192) == 256
    assert _select_nvfp4_softmax_quant_threads(64, 32768) == 512
    assert _select_nvfp4_softmax_quant_threads(512, 32768) == 128
    assert _select_nvfp4_softmax_quant_threads(2048, 32768) == 256
