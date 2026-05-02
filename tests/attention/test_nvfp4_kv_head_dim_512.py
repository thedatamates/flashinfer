"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

import flashinfer
from flashinfer.fmha_nvfp4_sm120 import BatchPrefillWithPagedKVCacheSM120Nvfp4Wrapper
from flashinfer.fp4_quantization import fp4_quantize, nvfp4_quantize_paged_kv_cache
from flashinfer.utils import get_compute_capability
from tests.test_helpers.utils_fp4 import E2M1_TO_FLOAT32, nvfp4_to_float


NVFP4_GQA_HEAD_DIMS = (128, 256, 512)


def _requires_sm12x_nvfp4():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, minor = get_compute_capability(torch.device("cuda"))
    if (major, minor) not in ((12, 0), (12, 1)):
        pytest.skip("NVFP4 KV cache is supported on SM120/SM121")


def _sharp_attention_cases(head_dim: int):
    if head_dim == 128:
        active_dims = [0, 5, 17, 31, 47, 63, 95, 127]
    elif head_dim == 256:
        active_dims = [0, 5, 17, 63, 127, 191, 223, 255]
    elif head_dim == 512:
        active_dims = [0, 5, 17, 63, 128, 255, 321, 511]
    else:
        raise ValueError(f"Unsupported sharp-token head_dim: {head_dim}")
    target_tokens = [0, 7, 15, 16, 31, 32, 47, 63]
    return list(zip(target_tokens, active_dims))


def _make_nvfp4_paged_kv(
    *,
    num_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
):
    packed_dim = head_dim // 2
    sf_dim = head_dim // 16
    k = torch.randint(
        0,
        256,
        (num_pages, page_size, num_kv_heads, packed_dim),
        device="cuda",
        dtype=torch.uint8,
    )
    v = torch.randint(
        0,
        256,
        (num_pages, page_size, num_kv_heads, packed_dim),
        device="cuda",
        dtype=torch.uint8,
    )
    k_sf = torch.ones(
        (num_pages, page_size, num_kv_heads, sf_dim),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    v_sf = torch.ones(
        (num_pages, page_size, num_kv_heads, sf_dim),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    return (k, v), (k_sf, v_sf)


def _unswizzle_v_sf(v_sf: torch.Tensor) -> torch.Tensor:
    raw = v_sf.view(torch.uint8)
    out = torch.empty_like(raw)
    page_size = raw.shape[1]
    sf_dim = raw.shape[3]
    sf_group = sf_dim // 4
    for t in range(page_size):
        for s in range(sf_dim):
            swizzled_t = (t // 4) * 4 + (s // sf_group)
            swizzled_s = (s % sf_group) * 4 + (t % 4)
            out[:, t, :, s] = raw[:, swizzled_t, :, swizzled_s]
    return out.view(torch.float8_e4m3fn)


def _copy_to_combined_vllm_layout(k_data, v_data, k_sf, v_sf):
    num_pages, page_size, num_kv_heads, packed_dim = k_data.shape
    sf_dim = k_sf.shape[-1]
    full_dim = packed_dim + sf_dim
    combined = torch.empty(
        (num_pages, 2, page_size, num_kv_heads, full_dim),
        device=k_data.device,
        dtype=torch.uint8,
    )

    k_side = combined[:, 0]
    v_side = combined[:, 1]
    k_side[..., :packed_dim].copy_(k_data)
    v_side[..., :packed_dim].copy_(v_data)
    k_side[..., packed_dim:].copy_(k_sf.view(torch.uint8))
    v_side[..., packed_dim:].copy_(v_sf.view(torch.uint8))

    return (
        k_side[..., :packed_dim],
        v_side[..., :packed_dim],
    ), (
        k_side[..., packed_dim:].view(torch.float8_e4m3fn),
        v_side[..., packed_dim:].view(torch.float8_e4m3fn),
    )


def _quantize_v_pv_layout_nhd(v_cache: torch.Tensor, v_global_sf: torch.Tensor):
    num_pages, page_size, num_kv_heads, head_dim = v_cache.shape
    assert page_size == 16

    # Quantize each output column across a 16-token page. This is the V-cache
    # physical layout consumed by FMHAv2's PV MMA path.
    v_by_col = (
        v_cache.permute(0, 2, 3, 1)
        .contiguous()
        .reshape(num_pages * num_kv_heads * head_dim, page_size)
    )
    packed_col_token, sf_col = fp4_quantize(
        v_by_col,
        v_global_sf,
        sf_vec_size=16,
        is_sf_swizzled_layout=False,
    )

    packed_col_token = packed_col_token.view(torch.uint8).reshape(
        num_pages, num_kv_heads, head_dim, page_size // 2
    )
    nibbles = torch.empty(
        (num_pages, num_kv_heads, head_dim, page_size),
        device=v_cache.device,
        dtype=torch.uint8,
    )
    nibbles[..., 0::2] = packed_col_token & 0x0F
    nibbles[..., 1::2] = (packed_col_token >> 4) & 0x0F

    nibbles_by_row = nibbles.permute(0, 3, 1, 2).contiguous()
    v_packed = (
        nibbles_by_row[..., 0::2] | (nibbles_by_row[..., 1::2] << 4)
    ).contiguous()

    scale_dim = head_dim // 16
    sf_col = sf_col.view(torch.uint8).reshape(num_pages, num_kv_heads, head_dim)
    v_sf = torch.empty(
        (num_pages, page_size, num_kv_heads, scale_dim),
        device=v_cache.device,
        dtype=torch.uint8,
    )
    for col in range(head_dim):
        v_sf[:, col // scale_dim, :, col % scale_dim] = sf_col[:, :, col]

    return v_packed, v_sf.view(torch.float8_e4m3fn)


def _dequantize_v_pv_layout_nhd(
    v_packed: torch.Tensor,
    v_sf: torch.Tensor,
    global_scale: float,
) -> torch.Tensor:
    num_pages, page_size, num_kv_heads, packed_dim = v_packed.shape
    head_dim = packed_dim * 2
    scale_dim = head_dim // 16

    nibbles = torch.empty(
        (num_pages, page_size, num_kv_heads, head_dim),
        device=v_packed.device,
        dtype=torch.uint8,
    )
    nibbles[..., 0::2] = v_packed & 0x0F
    nibbles[..., 1::2] = (v_packed >> 4) & 0x0F
    values = torch.tensor(E2M1_TO_FLOAT32, device=v_packed.device)[nibbles.long()]

    raw_sf = v_sf.view(torch.uint8)
    scale_bytes = torch.empty(
        (num_pages, num_kv_heads, head_dim),
        device=v_packed.device,
        dtype=torch.uint8,
    )
    for col in range(head_dim):
        scale_bytes[:, :, col] = raw_sf[:, col // scale_dim, :, col % scale_dim]
    scales = scale_bytes.view(torch.float8_e4m3fn).float()
    return values * scales[:, None, :, :] * global_scale


def _decode_ref(q, k, v, kv_len):
    head_grp_size = q.shape[1] // k.shape[2]
    k_flat = k.reshape(-1, k.shape[2], k.shape[3])[:kv_len].float()
    v_flat = v.reshape(-1, v.shape[2], v.shape[3])[:kv_len].float()
    q_float = q.float()[0]
    out = []
    for qo_head in range(q.shape[1]):
        kv_head = qo_head // head_grp_size
        scores = (k_flat[:, kv_head] @ q_float[qo_head]) / (q.shape[-1] ** 0.5)
        out.append(torch.softmax(scores, dim=0) @ v_flat[:, kv_head])
    return torch.stack(out).to(q.dtype).unsqueeze(0)


def _prefill_ref(q, k, v, q_len, kv_len, causal):
    head_grp_size = q.shape[1] // k.shape[2]
    k_flat = k.reshape(-1, k.shape[2], k.shape[3])[:kv_len].float()
    v_flat = v.reshape(-1, v.shape[2], v.shape[3])[:kv_len].float()
    q_float = q.float()[:q_len]
    out = torch.empty_like(q_float)
    for qo_head in range(q.shape[1]):
        kv_head = qo_head // head_grp_size
        scores = (q_float[:, qo_head] @ k_flat[:, kv_head].T) / (q.shape[-1] ** 0.5)
        if causal:
            q_idx = torch.arange(q_len, device=q.device).unsqueeze(1)
            kv_idx = torch.arange(kv_len, device=q.device).unsqueeze(0)
            scores = scores.masked_fill(q_idx + kv_len - q_len < kv_idx, float("-inf"))
        out[:, qo_head] = torch.softmax(scores, dim=-1) @ v_flat[:, kv_head]
    return out.to(q.dtype)


@pytest.mark.parametrize(
    "head_dim,window_left,q_lens,last_page_lens",
    [
        (128, 1024, [8, 13], [11, 7]),
        (256, 1024, [8, 13], [11, 7]),
        (512, -1, [8, 80], [13, 5]),
    ],
)
def test_nvfp4_paged_prefill_sm12x(
    head_dim, window_left, q_lens, last_page_lens
):
    _requires_sm12x_nvfp4()
    torch.manual_seed(0)

    page_size = 16
    batch_size = len(q_lens)
    num_pages = batch_size
    num_kv_heads = 2
    num_qo_heads = 4
    total_q_len = sum(q_lens)
    q = torch.randn(
        total_q_len, num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv_cache, kv_cache_sf = _make_nvfp4_paged_kv(
        num_pages=num_pages,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    qo_indptr = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()],
        device="cpu",
        dtype=torch.int32,
    )
    paged_kv_indptr = torch.arange(
        batch_size + 1, device="cpu", dtype=torch.int32
    )
    paged_kv_indices = torch.arange(num_pages, device="cuda", dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor(
        last_page_lens, device="cpu", dtype=torch.int32
    )
    out = torch.empty_like(q)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        "NHD",
        backend="auto",
    )
    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        window_left=window_left,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
        o_data_type=torch.bfloat16,
    )
    wrapper.run(
        q,
        kv_cache,
        out=out,
        kv_cache_sf=kv_cache_sf,
        k_scale=1.0,
        v_scale=1.0,
    )
    torch.cuda.synchronize()

    assert out.shape == q.shape
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_nvfp4_paged_prefill_vllm_strided_scale_layout_sm12x():
    _requires_sm12x_nvfp4()
    torch.manual_seed(0)

    page_size = 16
    head_dim = 512
    num_pages = 2
    kv_len = page_size * num_pages - 1
    num_kv_heads = 2
    num_qo_heads = 4

    q = torch.randn(1, num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim,
        device="cuda", dtype=torch.bfloat16,
    ) / 4
    v = torch.randn_like(k) / 4
    (k_data, v_data), (k_sf, v_sf), k_scale, v_scale = (
        nvfp4_quantize_paged_kv_cache(k, v, "NHD")
    )
    kv_cache, kv_cache_sf = _copy_to_combined_vllm_layout(
        k_data, v_data, k_sf, v_sf
    )
    k_scale_tensor = torch.tensor(k_scale, device="cuda", dtype=torch.float32)
    v_scale_tensor = torch.tensor(v_scale, device="cuda", dtype=torch.float32)
    k_ref = nvfp4_to_float(k_data, k_sf.view(torch.uint8), k_scale_tensor)
    v_ref = nvfp4_to_float(
        v_data, _unswizzle_v_sf(v_sf).view(torch.uint8), v_scale_tensor
    )
    out_ref = _decode_ref(q, k_ref, v_ref, kv_len)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        "NHD",
        backend="fa2",
    )
    wrapper.plan(
        torch.tensor([0, 1], device="cpu", dtype=torch.int32),
        torch.tensor([0, num_pages], device="cpu", dtype=torch.int32),
        torch.arange(num_pages, device="cuda", dtype=torch.int32),
        torch.tensor([page_size - 1], device="cpu", dtype=torch.int32),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
        o_data_type=torch.bfloat16,
    )
    out = wrapper.run(
        q,
        kv_cache,
        kv_cache_sf=kv_cache_sf,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), out_ref.float(), rtol=2e-1, atol=2e-1)


@pytest.mark.parametrize("causal", [False, True])
def test_nvfp4_paged_prefill_d512_all_ones_v_matches_torch_sm12x(causal):
    _requires_sm12x_nvfp4()
    torch.manual_seed(0)

    page_size = 16
    q_len = 17
    kv_len = 59
    num_pages = (kv_len + page_size - 1) // page_size
    num_kv_heads = 2
    num_qo_heads = 4
    head_dim = 512

    q = torch.randn(q_len, num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim,
        device="cuda", dtype=torch.bfloat16,
    ) / 8
    v = torch.ones_like(k)
    (k_data, v_data), (k_sf, v_sf), k_scale, v_scale = (
        nvfp4_quantize_paged_kv_cache(k, v, "NHD")
    )
    k_ref = nvfp4_to_float(
        k_data, k_sf.view(torch.uint8),
        torch.tensor(k_scale, device="cuda", dtype=torch.float32),
    )
    v_ref = nvfp4_to_float(
        v_data, _unswizzle_v_sf(v_sf).view(torch.uint8),
        torch.tensor(v_scale, device="cuda", dtype=torch.float32),
    )
    out_ref = _prefill_ref(q, k_ref, v_ref, q_len, kv_len, causal)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        "NHD",
        backend="fa2",
    )
    wrapper.plan(
        torch.tensor([0, q_len], device="cpu", dtype=torch.int32),
        torch.tensor([0, num_pages], device="cpu", dtype=torch.int32),
        torch.arange(num_pages, device="cuda", dtype=torch.int32),
        torch.tensor([(kv_len - 1) % page_size + 1], device="cpu", dtype=torch.int32),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
        o_data_type=torch.bfloat16,
    )
    out = wrapper.run(
        q,
        (k_data, v_data),
        kv_cache_sf=(k_sf, v_sf),
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), out_ref.float(), rtol=2e-1, atol=2e-1)


def test_nvfp4_paged_prefill_d512_random_quantized_matches_torch_sm12x():
    _requires_sm12x_nvfp4()
    torch.manual_seed(0)

    q_len = 17
    kv_len = 59
    page_size = 16
    num_pages = (kv_len + page_size - 1) // page_size
    num_kv_heads = 2
    num_qo_heads = 8
    head_dim = 512

    q = torch.randn(q_len, num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ) / 4
    v = torch.randn_like(k) / 4
    kv_cache, kv_cache_sf, k_scale, v_scale = nvfp4_quantize_paged_kv_cache(
        k, v, "NHD"
    )
    k_ref = nvfp4_to_float(
        kv_cache[0],
        kv_cache_sf[0].view(torch.uint8),
        torch.tensor(k_scale, device="cuda", dtype=torch.float32),
    )
    v_ref = nvfp4_to_float(
        kv_cache[1],
        _unswizzle_v_sf(kv_cache_sf[1]).view(torch.uint8),
        torch.tensor(v_scale, device="cuda", dtype=torch.float32),
    )
    out_ref = _prefill_ref(q, k_ref, v_ref, q_len, kv_len, causal=True)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        "NHD",
        backend="fa2",
    )
    wrapper.plan(
        torch.tensor([0, q_len], device="cpu", dtype=torch.int32),
        torch.tensor([0, num_pages], device="cpu", dtype=torch.int32),
        torch.arange(num_pages, device="cuda", dtype=torch.int32),
        torch.tensor([(kv_len - 1) % page_size + 1], device="cpu", dtype=torch.int32),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
        o_data_type=torch.bfloat16,
    )
    out = wrapper.run(
        q,
        kv_cache,
        kv_cache_sf=kv_cache_sf,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), out_ref.float(), rtol=2e-1, atol=2e-1)


def test_fmha_v2_nvfp4_paged_prefill_pv_layout_matches_reblocked_sm12x():
    _requires_sm12x_nvfp4()
    torch.manual_seed(1)

    batch_size = 2
    q_len = 64
    kv_len = 256
    page_size = 16
    pages_per_seq = kv_len // page_size
    total_pages = batch_size * pages_per_seq
    num_kv_heads = 2
    num_qo_heads = 8
    head_dim = 512

    q = torch.randn(
        batch_size * q_len,
        num_qo_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        total_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ) / 4
    v = torch.randn_like(k) / 4
    (k_data, v_reblocked), (k_sf, v_reblocked_sf), k_scale, v_scale = (
        nvfp4_quantize_paged_kv_cache(k, v, "NHD")
    )
    v_pv, v_pv_sf = _quantize_v_pv_layout_nhd(
        v,
        torch.tensor([1.0 / v_scale], device="cuda", dtype=torch.float32),
    )

    qo_indptr = torch.arange(
        0, (batch_size + 1) * q_len, q_len, device="cpu", dtype=torch.int32
    )
    paged_kv_indptr = torch.arange(
        0,
        (batch_size + 1) * pages_per_seq,
        pages_per_seq,
        device="cpu",
        dtype=torch.int32,
    )
    paged_kv_indices = torch.arange(total_pages, device="cuda", dtype=torch.int32)
    paged_kv_last_page_len = torch.full(
        (batch_size,), page_size, device="cpu", dtype=torch.int32
    )
    block_tables = paged_kv_indices.view(batch_size, pages_per_seq)

    def run(v_cache, v_sf, use_pv_layout):
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8),
            "NHD",
            backend="fmha_v2",
        )
        wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.uint8,
            o_data_type=torch.bfloat16,
            block_tables=block_tables,
        )
        out = wrapper.run(
            q,
            (k_data, v_cache),
            kv_cache_sf=(k_sf, v_sf),
            k_scale=k_scale,
            v_scale=v_scale,
            nvfp4_v_cache_uses_pv_layout=use_pv_layout,
        )
        torch.cuda.synchronize()
        return out

    out_reblocked = run(v_reblocked, v_reblocked_sf, False)
    out_pv = run(v_pv, v_pv_sf, True)

    assert torch.isfinite(out_reblocked.float()).all()
    assert torch.isfinite(out_pv.float()).all()
    diff = (out_pv.float() - out_reblocked.float()).abs()
    assert diff.mean() < 5e-3
    assert torch.quantile(diff.flatten(), 0.99) < 2e-2
    assert diff.max() < 5e-2


@pytest.mark.parametrize("head_dim", NVFP4_GQA_HEAD_DIMS)
def test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x(head_dim):
    _requires_sm12x_nvfp4()
    torch.manual_seed(2)

    q_len = 17
    kv_len = 59
    page_size = 16
    num_pages = (kv_len + page_size - 1) // page_size
    num_kv_heads = 1
    num_qo_heads = 8

    q = torch.randn(q_len, num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ) / 4
    v = torch.randn_like(k) / 4
    (k_data, _v_reblocked), (k_sf, _v_reblocked_sf), k_scale, v_scale = (
        nvfp4_quantize_paged_kv_cache(k, v, "NHD")
    )
    v_pv, v_pv_sf = _quantize_v_pv_layout_nhd(
        v,
        torch.tensor([1.0 / v_scale], device="cuda", dtype=torch.float32),
    )

    k_ref = nvfp4_to_float(
        k_data,
        k_sf.view(torch.uint8),
        torch.tensor(k_scale, device="cuda", dtype=torch.float32),
    )
    v_ref = _dequantize_v_pv_layout_nhd(v_pv, v_pv_sf, v_scale)
    out_ref = _prefill_ref(q, k_ref, v_ref, q_len, kv_len, causal=True)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        "NHD",
        backend="fmha_v2",
    )
    wrapper.plan(
        torch.tensor([0, q_len], device="cpu", dtype=torch.int32),
        torch.tensor([0, num_pages], device="cpu", dtype=torch.int32),
        torch.arange(num_pages, device="cuda", dtype=torch.int32),
        torch.tensor([(kv_len - 1) % page_size + 1], device="cpu", dtype=torch.int32),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
        o_data_type=torch.bfloat16,
        block_tables=torch.arange(num_pages, device="cuda", dtype=torch.int32).view(1, num_pages),
    )
    out = wrapper.run(
        q,
        (k_data, v_pv),
        kv_cache_sf=(k_sf, v_pv_sf),
        k_scale=k_scale,
        v_scale=v_scale,
        nvfp4_v_cache_uses_pv_layout=True,
    )
    torch.cuda.synchronize()

    diff = (out.float() - out_ref.float()).abs()
    assert diff.mean() < 5e-2
    assert torch.quantile(diff.flatten(), 0.99) < 1.6e-1
    assert diff.max() < 3.5e-1


def test_fmha_v2_nvfp4_split_kv_partials_merge_sm12x():
    _requires_sm12x_nvfp4()
    torch.manual_seed(123)

    q_len = 17
    kv_len = 256
    page_size = 16
    num_pages = kv_len // page_size
    num_kv_heads = 1
    num_qo_heads = 8
    head_dim = 512

    q = torch.randn(
        q_len, num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ) / 4
    v = torch.randn_like(k) / 4
    (k_data, _v_reblocked), (k_sf, _v_reblocked_sf), _k_scale, v_scale = (
        nvfp4_quantize_paged_kv_cache(k, v, "NHD")
    )
    v_pv, v_pv_sf = _quantize_v_pv_layout_nhd(
        v,
        torch.tensor([1.0 / v_scale], device="cuda", dtype=torch.float32),
    )

    block_tables = torch.arange(num_pages, device="cuda", dtype=torch.int32).view(
        1, num_pages
    )
    cu_q = torch.tensor([0, q_len], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_len], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([kv_len], device="cuda", dtype=torch.int32)
    workspace = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    common_kwargs = dict(
        input_layout="Q_PAGED_KV_NHD",
        workspace_buffer=workspace,
        seq_lens=seq_lens,
        max_q_len=q_len,
        max_kv_len=kv_len,
        bmm1_scale=head_dim**-0.5,
        bmm2_scale=1.0,
        batch_size=1,
        cum_seq_lens_q=cu_q,
        cum_seq_lens_kv=cu_kv,
        block_tables=block_tables,
        out_dtype=torch.bfloat16,
        mask_mode="causal",
        kv_cache_sf=(k_sf, v_pv_sf),
        nvfp4_v_cache_uses_pv_layout=True,
        save_softmax_stats=True,
    )

    full, _full_stats = flashinfer.prefill.trtllm_fmha_v2_prefill(
        (q, torch.stack((k_data, v_pv), dim=1)),
        **common_kwargs,
    )
    parts, part_stats = flashinfer.prefill.trtllm_fmha_v2_prefill(
        (q, torch.stack((k_data, v_pv), dim=1)),
        kv_split_size=128,
        num_kv_splits=2,
        **common_kwargs,
    )
    part_lse = part_stats[..., 0] + torch.log(part_stats[..., 1].clamp_min(1e-30))
    merged, _merged_lse = flashinfer.merge_states(parts, part_lse)
    torch.cuda.synchronize()

    assert parts.shape == (q_len, 2, num_qo_heads, head_dim)
    assert torch.isfinite(parts.float()).all()
    assert parts[:, 1].float().abs().mean() > 1.0
    diff = ((full.float() - merged.float()) * v_scale).abs()
    assert diff.mean() < 5e-3
    assert torch.quantile(diff.flatten(), 0.99) < 5e-2
    assert diff.max() < 1.5e-1


@pytest.mark.parametrize("head_dim", NVFP4_GQA_HEAD_DIMS)
def test_fmha_v2_grouped_m_nvfp4_paged_prefill_sharp_tokens_sm12x(head_dim):
    _requires_sm12x_nvfp4()

    page_size = 16
    kv_len = 64
    num_pages = kv_len // page_size
    num_kv_heads = 1
    num_qo_heads = 8
    cases = _sharp_attention_cases(head_dim)
    q_len = len(cases)

    q = torch.zeros(
        q_len, num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.zeros(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.zeros_like(k)
    k_flat = k.view(kv_len, num_kv_heads, head_dim)
    v_flat = v.view(kv_len, num_kv_heads, head_dim)
    for row, (target_token, active_dim) in enumerate(cases):
        q[row, :, active_dim] = 128
        k_flat[target_token, 0, active_dim] = 6
        v_flat[target_token, 0, target_token] = 1

    (k_data, _v_reblocked), (k_sf, _v_reblocked_sf), k_scale, v_scale = (
        nvfp4_quantize_paged_kv_cache(k, v, "NHD")
    )
    v_pv, v_pv_sf = _quantize_v_pv_layout_nhd(
        v,
        torch.tensor([1.0 / v_scale], device="cuda", dtype=torch.float32),
    )

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        "NHD",
        backend="fmha_v2",
    )
    wrapper.plan(
        torch.tensor([0, q_len], device="cpu", dtype=torch.int32),
        torch.tensor([0, num_pages], device="cpu", dtype=torch.int32),
        torch.arange(num_pages, device="cuda", dtype=torch.int32),
        torch.tensor([page_size], device="cpu", dtype=torch.int32),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
        o_data_type=torch.bfloat16,
        block_tables=torch.arange(num_pages, device="cuda", dtype=torch.int32).view(
            1, num_pages
        ),
    )
    out = wrapper.run(
        q,
        (k_data, v_pv),
        kv_cache_sf=(k_sf, v_pv_sf),
        k_scale=k_scale,
        v_scale=v_scale,
        nvfp4_v_cache_uses_pv_layout=True,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(out.float()).all()
    for row, (target_token, _active_dim) in enumerate(cases):
        selected = out[row, :, :kv_len].float()
        assert torch.equal(
            selected.argmax(dim=-1),
            torch.full((num_qo_heads,), target_token, device="cuda"),
        )
        assert torch.all(selected[:, target_token] > 0.9)
        selected[:, target_token] = 0
        assert selected.abs().max() < 0.1


def test_nvfp4_paged_prefill_d512_high_page_random_bits_finite_sm12x():
    _requires_sm12x_nvfp4()
    torch.manual_seed(0)

    q_len = 64
    page_size = 16
    pages_per_seq = 64
    total_pages = 2 * pages_per_seq
    num_kv_heads = 2
    num_qo_heads = 8
    head_dim = 512

    q = torch.randn(
        q_len, num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv_cache, kv_cache_sf = _make_nvfp4_paged_kv(
        num_pages=total_pages,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        "NHD",
        backend="fa2",
    )
    wrapper.plan(
        torch.tensor([0, q_len], device="cpu", dtype=torch.int32),
        torch.tensor([0, pages_per_seq], device="cpu", dtype=torch.int32),
        torch.arange(pages_per_seq, total_pages, device="cuda", dtype=torch.int32),
        torch.full((1,), page_size, device="cpu", dtype=torch.int32),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
        o_data_type=torch.bfloat16,
    )
    out = wrapper.run(
        q,
        kv_cache,
        kv_cache_sf=kv_cache_sf,
        k_scale=1.0,
        v_scale=1.0,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(out.float()).all()


def test_nvfp4_paged_prefill_d512_multitile_all_ones_v_sm12x():
    _requires_sm12x_nvfp4()
    torch.manual_seed(123)

    q_len = 64
    kv_len = 256
    page_size = 16
    num_pages = kv_len // page_size
    num_kv_heads = 2
    num_qo_heads = 8
    head_dim = 512

    q = torch.randn(
        q_len, num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    (k_data, v_data), (k_sf, v_sf) = _make_nvfp4_paged_kv(
        num_pages=num_pages,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    v_data.fill_(0x22)
    v_sf.fill_(1.0)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        "NHD",
        backend="fa2",
    )
    wrapper.plan(
        torch.tensor([0, q_len], device="cpu", dtype=torch.int32),
        torch.tensor([0, num_pages], device="cpu", dtype=torch.int32),
        torch.arange(num_pages, device="cuda", dtype=torch.int32),
        torch.tensor([page_size], device="cpu", dtype=torch.int32),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
        o_data_type=torch.bfloat16,
    )
    out = wrapper.run(
        q,
        (k_data, v_data),
        kv_cache_sf=(k_sf, v_sf),
        k_scale=1.0,
        v_scale=1.0,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), torch.ones_like(out.float()), rtol=0, atol=0)


@pytest.mark.parametrize("head_dim,window_left", [(128, 1024), (256, 1024), (512, -1)])
def test_nvfp4_xqa_decode_sm12x(head_dim, window_left):
    _requires_sm12x_nvfp4()
    torch.manual_seed(0)

    batch_size = 2
    page_size = 16
    num_pages = batch_size
    num_kv_heads = 2
    num_qo_heads = 4
    q = torch.randn(
        batch_size, num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv_cache, kv_cache_sf = _make_nvfp4_paged_kv(
        num_pages=num_pages,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    out = torch.empty_like(q)

    flashinfer.decode.xqa_batch_decode_with_kv_cache(
        q,
        kv_cache,
        torch.zeros(32 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        torch.arange(num_pages, device="cuda", dtype=torch.int32).view(
            batch_size, 1
        ),
        torch.tensor([page_size, page_size - 3], device="cuda", dtype=torch.uint32),
        max_seq_len=page_size,
        out=out,
        kv_layout="NHD",
        window_left=window_left,
        kv_cache_sf=kv_cache_sf,
    )
    torch.cuda.synchronize()

    assert out.shape == q.shape
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize("head_dim", NVFP4_GQA_HEAD_DIMS)
def test_nvfp4_xqa_decode_matches_torch_sm12x(head_dim):
    _requires_sm12x_nvfp4()
    torch.manual_seed(3)

    page_size = 16
    kv_len = 59
    num_pages = (kv_len + page_size - 1) // page_size
    num_kv_heads = 1
    num_qo_heads = 8
    q = torch.randn(1, num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ) / 4
    v = torch.randn_like(k) / 4
    kv_cache, kv_cache_sf, k_scale, v_scale = nvfp4_quantize_paged_kv_cache(
        k, v, "NHD"
    )
    k_ref = nvfp4_to_float(
        kv_cache[0],
        kv_cache_sf[0].view(torch.uint8),
        torch.tensor(k_scale, device="cuda", dtype=torch.float32),
    )
    v_ref = nvfp4_to_float(
        kv_cache[1],
        _unswizzle_v_sf(kv_cache_sf[1]).view(torch.uint8),
        torch.tensor(v_scale, device="cuda", dtype=torch.float32),
    )
    out_ref = _decode_ref(q, k_ref, v_ref, kv_len)
    out = torch.empty_like(q)

    flashinfer.decode.xqa_batch_decode_with_kv_cache(
        q,
        kv_cache,
        torch.zeros(64 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        torch.arange(num_pages, device="cuda", dtype=torch.int32).view(1, num_pages),
        torch.tensor([kv_len], device="cuda", dtype=torch.uint32),
        max_seq_len=kv_len,
        bmm1_scale=k_scale / (head_dim**0.5),
        bmm2_scale=v_scale,
        out=out,
        kv_layout="NHD",
        kv_cache_sf=kv_cache_sf,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), out_ref.float(), rtol=2e-1, atol=2e-1)


@pytest.mark.parametrize("head_dim", NVFP4_GQA_HEAD_DIMS)
def test_nvfp4_xqa_decode_sharp_tokens_sm12x(head_dim):
    _requires_sm12x_nvfp4()

    page_size = 16
    kv_len = 64
    num_pages = kv_len // page_size
    num_kv_heads = 1
    num_qo_heads = 8
    cases = _sharp_attention_cases(head_dim)

    q = torch.zeros(1, num_qo_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.zeros_like(k)
    k_flat = k.view(kv_len, num_kv_heads, head_dim)
    v_flat = v.view(kv_len, num_kv_heads, head_dim)
    for head, (target_token, active_dim) in enumerate(cases):
        q[0, head, active_dim] = 128
        k_flat[target_token, 0, active_dim] = 6
        v_flat[target_token, 0, target_token] = 1

    kv_cache, kv_cache_sf, k_scale, v_scale = nvfp4_quantize_paged_kv_cache(
        k, v, "NHD"
    )
    out = torch.empty_like(q)

    flashinfer.decode.xqa_batch_decode_with_kv_cache(
        q,
        kv_cache,
        torch.zeros(64 * 1024 * 1024, device="cuda", dtype=torch.uint8),
        torch.arange(num_pages, device="cuda", dtype=torch.int32).view(1, num_pages),
        torch.tensor([kv_len], device="cuda", dtype=torch.uint32),
        max_seq_len=kv_len,
        bmm1_scale=k_scale / (head_dim**0.5),
        bmm2_scale=v_scale,
        out=out,
        kv_layout="NHD",
        kv_cache_sf=kv_cache_sf,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(out.float()).all()
    for head, (target_token, _active_dim) in enumerate(cases):
        selected = out[0, head, :kv_len].float()
        assert selected.argmax() == target_token
        assert selected[target_token] > 0.9
        selected[target_token] = 0
        assert selected.abs().max() < 0.1


@pytest.mark.parametrize("head_dim,group", [(128, 4), (256, 6), (512, 4)])
def test_sm120_nvfp4_wrapper_multi_kv_matches_single_kv_sm12x(head_dim, group):
    _requires_sm12x_nvfp4()

    device = torch.device("cuda")
    page_size = 16
    q_lens = [128, 256]
    kv_lens = [1024, 1024]
    num_kv_heads = 2
    num_qo_heads = group * num_kv_heads
    pages_per_seq = [(kv_len + page_size - 1) // page_size for kv_len in kv_lens]
    total_pages = sum(pages_per_seq)
    table_width = max(pages_per_seq)

    gen = torch.Generator(device=device)
    gen.manual_seed(120256 + head_dim)
    q = (
        torch.randn(
            (sum(q_lens), num_qo_heads, head_dim), device=device, generator=gen
        )
        / 4
    ).to(torch.bfloat16)
    k = (
        torch.randn(
            (total_pages, page_size, num_kv_heads, head_dim),
            device=device,
            generator=gen,
        )
        / 4
    ).to(torch.bfloat16)
    v = (
        torch.randn(
            (total_pages, page_size, num_kv_heads, head_dim),
            device=device,
            generator=gen,
        )
        / 4
    ).to(torch.bfloat16)

    block_tables = torch.full(
        (len(q_lens), table_width), -1, dtype=torch.int32, device=device
    )
    page_cursor = 0
    for batch_idx, pages in enumerate(pages_per_seq):
        block_tables[batch_idx, :pages] = torch.arange(
            page_cursor, page_cursor + pages, dtype=torch.int32, device=device
        )
        page_cursor += pages

    qo_indptr = torch.tensor(
        [0, q_lens[0], sum(q_lens)], dtype=torch.int32, device="cpu"
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32, device="cpu")
    (k_fp4, v_fp4), (k_sf, v_sf), k_scale, v_scale = (
        nvfp4_quantize_paged_kv_cache(
            k, v, "NHD", v_data_layout="pv", v_scale_layout="pv"
        )
    )

    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithPagedKVCacheSM120Nvfp4Wrapper(workspace)
    wrapper.plan(
        qo_indptr,
        block_tables,
        kv_lens_t,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=True,
        window_left=1024,
        logits_soft_cap=50.0,
        split_kv_len=8192,
    )
    out = wrapper.run(
        q, (k_fp4, v_fp4), (k_sf, v_sf), k_scale=k_scale, v_scale=v_scale
    )

    expected_slices = []
    for kv_head in range(num_kv_heads):
        single_wrapper = BatchPrefillWithPagedKVCacheSM120Nvfp4Wrapper(
            torch.empty_like(workspace)
        )
        single_wrapper.plan(
            qo_indptr,
            block_tables,
            kv_lens_t,
            num_qo_heads=group,
            num_kv_heads=1,
            head_dim=head_dim,
            causal=True,
            window_left=1024,
            logits_soft_cap=50.0,
            split_kv_len=8192,
        )
        q_slice = q[:, kv_head * group : (kv_head + 1) * group, :].contiguous()
        out_slice = single_wrapper.run(
            q_slice,
            (
                k_fp4[:, :, kv_head : kv_head + 1, :].contiguous(),
                v_fp4[:, :, kv_head : kv_head + 1, :].contiguous(),
            ),
            (
                k_sf[:, :, kv_head : kv_head + 1, :].contiguous(),
                v_sf[:, :, kv_head : kv_head + 1, :].contiguous(),
            ),
            k_scale=k_scale,
            v_scale=v_scale,
        )
        expected_slices.append(out_slice)

    expected = torch.cat(expected_slices, dim=1)
    assert torch.isfinite(out.float()).all()
    multi_kv_atol = 3e-3 if head_dim == 512 else 1e-3
    torch.testing.assert_close(out, expected, rtol=0, atol=multi_kv_atol)


@pytest.mark.parametrize("head_dim,group", [(128, 4), (256, 6), (512, 4)])
def test_standard_prefill_wrapper_sm120_nvfp4_backend_matches_direct_wrapper_sm12x(
    head_dim, group
):
    _requires_sm12x_nvfp4()

    device = torch.device("cuda")
    page_size = 16
    q_lens = [128, 256]
    kv_lens = [1024, 768]
    num_kv_heads = 2
    num_qo_heads = group * num_kv_heads
    pages_per_seq = [(kv_len + page_size - 1) // page_size for kv_len in kv_lens]
    total_pages = sum(pages_per_seq)
    table_width = max(pages_per_seq)
    physical_pages = total_pages + 7

    gen = torch.Generator(device=device)
    gen.manual_seed(120512 + head_dim)
    q = (
        torch.randn(
            (sum(q_lens), num_qo_heads, head_dim), device=device, generator=gen
        )
        / 4
    ).to(torch.bfloat16)
    k = (
        torch.randn(
            (physical_pages, page_size, num_kv_heads, head_dim),
            device=device,
            generator=gen,
        )
        / 4
    ).to(torch.bfloat16)
    v = (
        torch.randn(
            (physical_pages, page_size, num_kv_heads, head_dim),
            device=device,
            generator=gen,
        )
        / 4
    ).to(torch.bfloat16)

    page_order = torch.randperm(physical_pages, device=device)[:total_pages].to(
        torch.int32
    )
    block_tables = torch.full(
        (len(q_lens), table_width), -1, dtype=torch.int32, device=device
    )
    offset = 0
    for batch_idx, pages in enumerate(pages_per_seq):
        block_tables[batch_idx, :pages] = page_order[offset : offset + pages]
        offset += pages

    qo_indptr = torch.tensor(
        [0, q_lens[0], sum(q_lens)], dtype=torch.int32, device="cpu"
    )
    paged_kv_indptr = torch.tensor(
        [0, pages_per_seq[0], total_pages], dtype=torch.int32, device="cpu"
    )
    paged_kv_indices = page_order.contiguous()
    paged_kv_last_page_len = torch.full(
        (len(q_lens),), page_size, dtype=torch.int32, device="cpu"
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32, device="cpu")
    (k_fp4, v_fp4), (k_sf, v_sf), k_scale, v_scale = (
        nvfp4_quantize_paged_kv_cache(
            k, v, "NHD", v_data_layout="pv", v_scale_layout="pv"
        )
    )

    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
    standard = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="sm120-nvfp4"
    )
    standard.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        window_left=512,
        logits_soft_cap=50.0,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
        o_data_type=torch.bfloat16,
        seq_lens=kv_lens_t,
        block_tables=block_tables,
        nvfp4_v_cache_uses_pv_layout=True,
    )
    out = standard.run(
        q,
        (k_fp4, v_fp4),
        kv_cache_sf=(k_sf, v_sf),
        k_scale=k_scale,
        v_scale=v_scale,
        nvfp4_v_cache_uses_pv_layout=True,
    )

    direct_workspace = torch.empty_like(workspace)
    direct = BatchPrefillWithPagedKVCacheSM120Nvfp4Wrapper(direct_workspace)
    direct.plan(
        qo_indptr,
        block_tables,
        kv_lens_t,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=True,
        window_left=512,
        logits_soft_cap=50.0,
        split_kv_len=8192,
    )
    expected = direct.run(
        q, (k_fp4, v_fp4), (k_sf, v_sf), k_scale=k_scale, v_scale=v_scale
    )
    torch.cuda.synchronize()

    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out, expected, rtol=0, atol=2e-3)


@pytest.mark.parametrize("head_dim,group", [(128, 4), (256, 6), (512, 4)])
def test_sm120_nvfp4_backend_accepts_normal_v_layout_sm12x(head_dim, group):
    _requires_sm12x_nvfp4()

    device = torch.device("cuda")
    page_size = 16
    q_lens = [128, 128]
    kv_lens = [1024, 1024]
    num_kv_heads = 2
    num_qo_heads = group * num_kv_heads
    pages_per_seq = [(kv_len + page_size - 1) // page_size for kv_len in kv_lens]
    total_pages = sum(pages_per_seq)
    table_width = max(pages_per_seq)

    gen = torch.Generator(device=device)
    gen.manual_seed(130120 + head_dim)
    q = (
        torch.randn(
            (sum(q_lens), num_qo_heads, head_dim), device=device, generator=gen
        )
        / 4
    ).to(torch.bfloat16)
    k = (
        torch.randn(
            (total_pages, page_size, num_kv_heads, head_dim),
            device=device,
            generator=gen,
        )
        / 4
    ).to(torch.bfloat16)
    v = (
        torch.randn(
            (total_pages, page_size, num_kv_heads, head_dim),
            device=device,
            generator=gen,
        )
        / 4
    ).to(torch.bfloat16)

    block_tables = torch.arange(total_pages, device=device, dtype=torch.int32).view(
        len(q_lens), table_width
    )
    qo_indptr = torch.tensor(
        [0, q_lens[0], sum(q_lens)], dtype=torch.int32, device="cpu"
    )
    paged_kv_indptr = torch.tensor(
        [0, pages_per_seq[0], total_pages], dtype=torch.int32, device="cpu"
    )
    paged_kv_indices = torch.arange(total_pages, device=device, dtype=torch.int32)
    paged_kv_last_page_len = torch.full(
        (len(q_lens),), page_size, dtype=torch.int32, device="cpu"
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32, device="cpu")

    (k_normal, v_normal), (k_sf, v_sf_normal), k_scale, v_scale = (
        nvfp4_quantize_paged_kv_cache(k, v, "NHD")
    )
    (k_pv, v_pv), (k_sf_pv, v_sf_pv), k_scale_pv, v_scale_pv = (
        nvfp4_quantize_paged_kv_cache(
            k, v, "NHD", v_data_layout="pv", v_scale_layout="pv"
        )
    )
    assert torch.equal(k_normal, k_pv)
    assert torch.equal(k_sf.view(torch.uint8), k_sf_pv.view(torch.uint8))
    assert k_scale == k_scale_pv
    assert v_scale == v_scale_pv

    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)

    def run(v_cache, v_sf, use_pv_layout):
        run_workspace = torch.empty_like(workspace)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            run_workspace, "NHD", backend="sm120-nvfp4"
        )
        wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            window_left=512,
            logits_soft_cap=50.0,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.uint8,
            o_data_type=torch.bfloat16,
            seq_lens=kv_lens_t,
            block_tables=block_tables,
            nvfp4_v_cache_uses_pv_layout=use_pv_layout,
        )
        out = wrapper.run(
            q,
            (k_normal, v_cache),
            kv_cache_sf=(k_sf, v_sf),
            k_scale=k_scale,
            v_scale=v_scale,
            nvfp4_v_cache_uses_pv_layout=use_pv_layout,
        )
        torch.cuda.synchronize()
        return out

    out_normal = run(v_normal, v_sf_normal, False)
    out_pv = run(v_pv, v_sf_pv, True)
    torch.cuda.synchronize()

    assert torch.isfinite(out_normal.float()).all()
    assert torch.isfinite(out_pv.float()).all()
    diff = (out_normal.float() - out_pv.float()).abs().flatten()
    assert diff.mean() < 3e-3
    assert torch.quantile(diff, 0.99) < 1.2e-2
    assert diff.max() < 2.5e-2

    out_normal_hnd_ref = run(v_normal, v_sf_normal, False)

    k_hnd = k_normal.permute(0, 2, 1, 3).contiguous()
    v_hnd = v_normal.permute(0, 2, 1, 3).contiguous()
    k_sf_hnd = k_sf.permute(0, 2, 1, 3).contiguous()
    v_sf_hnd = v_sf_normal.permute(0, 2, 1, 3).contiguous()
    hnd_workspace = torch.empty_like(workspace)
    wrapper_hnd = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        hnd_workspace, "HND", backend="sm120-nvfp4"
    )
    wrapper_hnd.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        window_left=512,
        logits_soft_cap=50.0,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
        o_data_type=torch.bfloat16,
        seq_lens=kv_lens_t,
        block_tables=block_tables,
        nvfp4_v_cache_uses_pv_layout=False,
    )
    out_hnd = wrapper_hnd.run(
        q,
        (k_hnd, v_hnd),
        kv_cache_sf=(k_sf_hnd, v_sf_hnd),
        k_scale=k_scale,
        v_scale=v_scale,
        nvfp4_v_cache_uses_pv_layout=False,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(out_hnd.float()).all()
    diff_hnd = (out_hnd.float() - out_normal_hnd_ref.float()).abs().flatten()
    assert diff_hnd.mean() < 1e-5
    assert diff_hnd.max() < 1e-3
