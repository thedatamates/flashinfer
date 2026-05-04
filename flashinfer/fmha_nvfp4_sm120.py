"""
Copyright (c) 2026 by FlashInfer team.

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

from __future__ import annotations

import functools
import math
from typing import Optional, Tuple

import torch

from .api_logging import flashinfer_api
from .jit import gen_fmha_nvfp4_sm120_module
from .utils import check_shape_dtype_device


def _tile_m_for_head_dim(head_dim: int) -> int:
    if head_dim in (128, 256):
        return 64
    if head_dim == 512:
        return 128
    raise ValueError("SM120 NVFP4 FMHA supports head_dim in {128, 256, 512}.")


def _paged_tile_m_for_config(head_dim: int, use_sliding_window: bool) -> int:
    if head_dim == 256 and not use_sliding_window:
        return 128
    return _tile_m_for_head_dim(head_dim)


def _default_output_group_span(head_dim: int) -> int:
    if head_dim == 128:
        return 1
    if head_dim == 256:
        return 2
    if head_dim == 512:
        return 4
    raise ValueError("SM120 NVFP4 FMHA supports head_dim in {128, 256, 512}.")


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _auto_split_kv_len(
    *,
    head_dim: int,
    max_q_len: int,
    group_size: int,
    window_left: int,
) -> int:
    if window_left > 0:
        return _round_up(window_left, 128)
    tile_m = _paged_tile_m_for_config(head_dim, False)
    q_tiles = max(1, _round_up(max_q_len * group_size, tile_m) // tile_m)
    if head_dim == 512:
        q_tiles *= 3
    split_tiles = min(max(q_tiles, 8), 96)
    return split_tiles * 128


def _empty_aligned(
    shape: Tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    alignment: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    numel = math.prod(shape)
    elem_size = torch.empty((), dtype=dtype).element_size()
    nbytes = numel * elem_size
    base = torch.empty(nbytes + alignment, dtype=torch.uint8, device=device)
    offset = (-base.data_ptr()) % alignment
    tensor = base[offset : offset + nbytes].view(dtype).view(shape)
    return tensor, base


@functools.cache
def _get_sm120_nvfp4_fmha_module(
    head_dim: int,
    causal: bool,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    v_cache_uses_pv_layout: bool,
):
    return gen_fmha_nvfp4_sm120_module(
        head_dim,
        causal=causal,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
        v_cache_uses_pv_layout=v_cache_uses_pv_layout,
    ).build_and_load()


def _as_uint8_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.uint8:
        return scale
    if scale.dtype == torch.float8_e4m3fn:
        return scale.view(torch.uint8)
    raise TypeError(
        "NVFP4 scale tensors must have dtype torch.uint8 or torch.float8_e4m3fn, "
        f"got {scale.dtype}."
    )


def _check_cuda_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    require_last_dim_contiguous: bool = True,
) -> None:
    if tensor.dtype != dtype or not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA {dtype} tensor.")
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}.")
    if tensor.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension.")
    if require_last_dim_contiguous and tensor.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous in the last dimension.")


class BatchPrefillWithPagedKVCacheSM120Nvfp4Wrapper:
    r"""SM120 NVFP4 paged prefill wrapper for BF16 Q/O and NVFP4 KV.

    This wrapper exposes the productionized SM120 fused FMHA kernels for
    D128/D256/D512. Standard vLLM paged NVFP4 KV uses linear V layout at the
    public API boundary; the attention V producer reblocks it into the PV
    physical layout required by the SM120 block-scaled PV MMA inside the fused
    kernel.

    Callers that already store PV-layout V from
    :func:`flashinfer.nvfp4_quantize_paged_kv_cache` with
    ``v_data_layout="pv"`` and ``v_scale_layout="pv"`` can pass
    ``v_cache_uses_pv_layout=True`` to skip that conversion. In both cases the
    attention kernel stages K/V directly from the paged block table.
    """

    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD") -> None:
        if workspace_buffer.dtype != torch.uint8 or not workspace_buffer.is_cuda:
            raise ValueError("workspace_buffer must be a CUDA torch.uint8 tensor.")
        if kv_layout not in ("NHD", "HND"):
            raise ValueError("kv_layout must be 'NHD' or 'HND'.")
        self._workspace_buffer = workspace_buffer
        self.device = workspace_buffer.device
        self._kv_layout = kv_layout
        self._module = None
        self._planned = False

    def reset_workspace_buffer(self, workspace_buffer: torch.Tensor) -> None:
        if workspace_buffer.dtype != torch.uint8 or not workspace_buffer.is_cuda:
            raise ValueError("workspace_buffer must be a CUDA torch.uint8 tensor.")
        self._workspace_buffer = workspace_buffer
        self.device = workspace_buffer.device

    @flashinfer_api
    def plan(
        self,
        qo_indptr: torch.Tensor,
        block_tables: torch.Tensor,
        kv_lens: torch.Tensor,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int = 16,
        causal: bool = True,
        window_left: int = -1,
        logits_soft_cap: float = 0.0,
        split_kv_len: int = 0,
        output_group_span: Optional[int] = None,
        v_cache_uses_pv_layout: bool = True,
    ) -> None:
        if page_size != 16:
            raise ValueError("SM120 NVFP4 FMHA currently requires page_size=16.")
        if head_dim not in (128, 256, 512):
            raise ValueError("head_dim must be one of {128, 256, 512}.")
        if num_qo_heads % num_kv_heads != 0:
            raise ValueError("num_qo_heads must be divisible by num_kv_heads.")
        if split_kv_len < 0 or split_kv_len % 128 != 0:
            raise ValueError(
                "split_kv_len must be 0 for auto-selection or a positive multiple of 128."
            )
        if window_left < -1 or window_left == 0:
            raise ValueError(
                "window_left must be -1 to disable SWA or a positive window size."
            )
        if output_group_span is None:
            output_group_span = _default_output_group_span(head_dim)
        if output_group_span not in (1, 2, 4):
            raise ValueError("output_group_span must be 1, 2, or 4.")
        if head_dim == 128 and output_group_span != 1:
            raise ValueError("D128 only supports output_group_span=1.")
        if head_dim % (output_group_span * 128) != 0:
            raise ValueError("head_dim must be divisible by output_group_span * 128.")
        if v_cache_uses_pv_layout and self._kv_layout != "NHD":
            raise ValueError("PV-layout V cache is currently supported only with NHD layout.")

        if qo_indptr.dtype != torch.int32:
            raise ValueError("qo_indptr must have dtype torch.int32.")
        if qo_indptr.ndim != 1:
            raise ValueError("qo_indptr must have shape [batch_size + 1].")
        if qo_indptr.numel() < 2:
            raise ValueError("qo_indptr must have shape [batch_size + 1].")
        if block_tables.dtype != torch.int32 or not block_tables.is_cuda:
            raise ValueError("block_tables must be a CUDA torch.int32 tensor.")
        if block_tables.device != self.device:
            raise ValueError(
                f"block_tables must be on device {self.device}, got {block_tables.device}."
            )
        if block_tables.ndim != 2:
            raise ValueError("block_tables must have shape [batch_size, max_pages].")
        if block_tables.stride(1) != 1:
            raise ValueError("block_tables must be contiguous in the page dimension.")
        if kv_lens.dtype != torch.int32:
            raise ValueError("kv_lens must have dtype torch.int32.")
        if kv_lens.ndim != 1:
            raise ValueError("kv_lens must have shape [batch_size].")

        qo_cpu = qo_indptr.to("cpu", dtype=torch.int64)
        kv_cpu = kv_lens.to("cpu", dtype=torch.int64)
        block_cpu = block_tables.to("cpu", dtype=torch.int64)
        batch_size = int(qo_cpu.numel() - 1)
        if int(kv_cpu.numel()) != batch_size:
            raise ValueError("kv_lens must have one entry per sequence.")
        if int(block_cpu.shape[0]) != batch_size:
            raise ValueError("block_tables must have one row per sequence.")

        q_lens = qo_cpu[1:] - qo_cpu[:-1]
        if torch.any(q_lens < 0):
            raise ValueError("qo_indptr must be non-decreasing.")
        if torch.any(kv_cpu <= 0):
            raise ValueError("kv_lens entries must be positive.")
        required_pages = torch.div(
            kv_cpu + page_size - 1, page_size, rounding_mode="floor"
        )
        if torch.any(required_pages > block_cpu.shape[1]):
            raise ValueError("block_tables rows do not cover kv_lens.")
        active_blocks = [
            block_cpu[batch_idx, : int(required_pages[batch_idx].item())]
            for batch_idx in range(batch_size)
        ]
        active_pages = (
            torch.cat(active_blocks) if active_blocks else block_cpu.new_empty((0,))
        )
        if active_pages.numel() == 0 or torch.any(active_pages < 0):
            raise ValueError("block_tables must contain valid page indices for every KV token.")

        self._qo_indptr = qo_indptr
        self._qo_indptr_device = qo_indptr.to(
            device=self.device, dtype=torch.int32
        ).contiguous()
        self._block_tables = block_tables
        self._kv_lens = kv_lens
        self._kv_lens_device = kv_lens.to(
            device=self.device, dtype=torch.int32
        ).contiguous()
        self._batch_size = batch_size
        self._total_q_len = int(qo_cpu[-1].item())
        self._max_q_len = int(q_lens.max().item()) if batch_size > 0 else 0
        self._max_kv_len = int(kv_cpu.max().item()) if batch_size > 0 else 0
        self._max_physical_pages = int(active_pages.max().item()) + 1
        self._num_qo_heads = int(num_qo_heads)
        self._num_kv_heads = int(num_kv_heads)
        self._group_size = int(num_qo_heads // num_kv_heads)
        self._head_dim = int(head_dim)
        self._page_size = int(page_size)
        self._causal = bool(causal)
        self._window_left = int(window_left)
        self._logits_soft_cap = float(logits_soft_cap)
        self._v_cache_uses_pv_layout = bool(v_cache_uses_pv_layout)
        self._output_group_span = int(output_group_span)
        self._module = _get_sm120_nvfp4_fmha_module(
            self._head_dim,
            self._causal,
            self._window_left > 0,
            self._logits_soft_cap > 0.0,
            self._v_cache_uses_pv_layout,
        )

        tile_m = _paged_tile_m_for_config(head_dim, self._window_left > 0)
        resolved_split_kv_len = (
            _auto_split_kv_len(
                head_dim=head_dim,
                max_q_len=self._max_q_len,
                group_size=self._group_size,
                window_left=self._window_left,
            )
            if split_kv_len == 0
            else int(split_kv_len)
        )
        self._split_kv_tiles = int(resolved_split_kv_len // 128)
        max_q_rows_per_kv_head = self._max_q_len * self._group_size
        padded_q_rows_per_seq = _round_up(max_q_rows_per_kv_head, tile_m)
        batch_padded_q_rows = (
            self._num_kv_heads * batch_size * padded_q_rows_per_seq
        )
        physical_kv_len = _round_up(self._max_kv_len, 128)
        self._physical_kv_len = physical_kv_len
        num_splits = math.ceil((physical_kv_len // 128) / self._split_kv_tiles)
        total_q_rows = self._total_q_len * self._num_qo_heads

        self._scratch_bases = []

        def alloc(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            tensor, base = _empty_aligned(shape, dtype=dtype, device=self.device)
            self._scratch_bases.append(base)
            return tensor

        self._q_packed = alloc((total_q_rows, head_dim // 2), torch.uint8)
        self._q_scales = alloc((total_q_rows, head_dim // 16), torch.uint8)
        self._q_packed_scratch = alloc(
            (batch_padded_q_rows, head_dim // 2), torch.uint8
        )
        self._q_scales_scratch = alloc(
            (batch_padded_q_rows, head_dim // 16), torch.uint8
        )
        self._partial = alloc(
            (num_splits, batch_padded_q_rows, head_dim), torch.bfloat16
        )
        self._split_m = alloc((num_splits, batch_padded_q_rows), torch.float32)
        self._split_l = alloc((num_splits, batch_padded_q_rows), torch.float32)
        self._out_scratch = alloc((batch_padded_q_rows, head_dim), torch.bfloat16)
        self._out_group = alloc((total_q_rows, head_dim), torch.bfloat16)
        self._planned = True

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Tuple[torch.Tensor, torch.Tensor],
        kv_cache_sf: Tuple[torch.Tensor, torch.Tensor],
        *,
        k_scale: float,
        v_scale: float,
        v_cache_uses_pv_layout: bool = True,
        v_cache_sf_layout: str = "trtllm_interleaved",
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._planned:
            raise RuntimeError("plan() must be called before run().")
        if self._module is None:
            raise RuntimeError("SM120 NVFP4 module was not initialized by plan().")
        if q.dtype != torch.bfloat16 or not q.is_cuda:
            raise ValueError("q must be a CUDA torch.bfloat16 tensor.")
        if q.device != self.device:
            raise ValueError(f"q must be on device {self.device}, got {q.device}.")
        if q.shape != (
            self._total_q_len,
            self._num_qo_heads,
            self._head_dim,
        ):
            raise ValueError(
                "q must have shape "
                f"({self._total_q_len}, {self._num_qo_heads}, {self._head_dim}), "
                f"got {tuple(q.shape)}."
            )
        if q.stride(-1) != 1:
            raise ValueError("q must be contiguous in the last dimension.")

        k_pages, v_pages_input = paged_kv_cache
        k_sf_pages, v_sf_pages_input = kv_cache_sf
        _check_cuda_tensor(
            k_pages, name="k_pages", dtype=torch.uint8, device=self.device
        )
        _check_cuda_tensor(
            v_pages_input, name="v_pages", dtype=torch.uint8, device=self.device
        )
        k_sf_pages_u8 = _as_uint8_scale(k_sf_pages)
        v_sf_pages_input_u8 = _as_uint8_scale(v_sf_pages_input)
        _check_cuda_tensor(
            k_sf_pages_u8,
            name="k scale pages",
            dtype=torch.uint8,
            device=self.device,
        )
        _check_cuda_tensor(
            v_sf_pages_input_u8,
            name="V scale pages",
            dtype=torch.uint8,
            device=self.device,
        )
        if v_cache_sf_layout not in ("trtllm_interleaved", "linear", "pv"):
            raise ValueError(
                "v_cache_sf_layout must be 'trtllm_interleaved', 'linear', or 'pv'."
            )
        if v_cache_sf_layout == "pv" and not self._v_cache_uses_pv_layout:
            raise ValueError(
                "v_cache_sf_layout='pv' requires v_cache_uses_pv_layout=True."
            )
        if bool(v_cache_uses_pv_layout) != self._v_cache_uses_pv_layout:
            raise ValueError(
                "v_cache_uses_pv_layout must match the value passed to plan()."
            )
        if self._v_cache_uses_pv_layout and self._kv_layout != "NHD":
            raise ValueError("PV-layout V cache is currently supported only with NHD layout.")

        expected_page_shape = (
            (None, self._num_kv_heads, self._page_size, self._head_dim // 2)
            if self._kv_layout == "HND"
            else (None, self._page_size, self._num_kv_heads, self._head_dim // 2)
        )
        if k_pages.dtype != torch.uint8 or v_pages_input.dtype != torch.uint8:
            raise ValueError("paged_kv_cache tensors must have dtype torch.uint8.")
        if k_pages.shape[1:] != expected_page_shape[1:]:
            raise ValueError(
                "k_pages must have shape "
                f"{expected_page_shape}, got {tuple(k_pages.shape)}."
            )
        if v_pages_input.shape != k_pages.shape:
            raise ValueError("V pages must have the same shape as K pages.")
        if k_pages.shape[0] < self._max_physical_pages:
            raise ValueError(
                "paged_kv_cache has fewer physical pages than block_tables reference."
            )
        expected_sf_shape = (
            (k_pages.shape[0], self._num_kv_heads, self._page_size, self._head_dim // 16)
            if self._kv_layout == "HND"
            else (k_pages.shape[0], self._page_size, self._num_kv_heads, self._head_dim // 16)
        )
        if k_sf_pages_u8.shape != expected_sf_shape:
            raise ValueError(
                f"k scale pages must have shape {expected_sf_shape}, "
                f"got {tuple(k_sf_pages_u8.shape)}."
            )
        if v_sf_pages_input_u8.shape != expected_sf_shape:
            raise ValueError(
                f"V scale pages must have shape {expected_sf_shape}, "
                f"got {tuple(v_sf_pages_input_u8.shape)}."
            )

        run_kv_layout_hnd = (
            False if self._v_cache_uses_pv_layout else self._kv_layout == "HND"
        )
        v_scale_layout_code = (
            0
            if self._v_cache_uses_pv_layout
            else 1 if v_cache_sf_layout == "linear" else 0
        )

        if out is None:
            out = torch.zeros_like(q)
        else:
            check_shape_dtype_device(out, q.shape, torch.bfloat16, q.device, "out")

        stream = torch.cuda.current_stream(q.device).cuda_stream
        self._module.paged_run_bf16_q(
            q,
            self._q_packed,
            self._q_scales,
            k_pages,
            k_sf_pages_u8,
            v_pages_input,
            v_sf_pages_input_u8,
            self._block_tables,
            self._qo_indptr_device,
            self._kv_lens_device,
            self._q_packed_scratch,
            self._q_scales_scratch,
            self._partial,
            self._split_m,
            self._split_l,
            self._out_scratch,
            self._out_group,
            self._workspace_buffer,
            self._physical_kv_len,
            float(k_scale),
            float(v_scale),
            -1,
            self._split_kv_tiles,
            self._group_size,
            self._causal,
            self._window_left,
            self._logits_soft_cap,
            self._output_group_span,
            run_kv_layout_hnd,
            v_scale_layout_code,
            stream,
        )
        out.copy_(
            self._out_group.view(
                self._total_q_len, self._num_qo_heads, self._head_dim
            )
        )

        return out


__all__ = ["BatchPrefillWithPagedKVCacheSM120Nvfp4Wrapper"]
