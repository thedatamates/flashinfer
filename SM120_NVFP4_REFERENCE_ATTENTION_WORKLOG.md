# SM120 NVFP4 Reference Attention Worklog

Date started: 2026-04-27

## Objective

Build an unconstrained hand-written fused attention reference kernel for Blackwell SM120 that answers the structural performance question directly:

Can NVFP4 KV attention beat FP8 FA2 by a meaningful margin when the kernel is allowed to use the SM120 block-scaled FP4 tensor-core path and is not constrained by existing FlashInfer FA2 abstractions?

This is a performance-reference kernel first. Correctness comes first, throughput second, maintainability third. Abstractions and production plumbing are explicitly out of scope until the fixed-shape kernel proves the ceiling.

Update: the active target is ceiling seeking, not the fastest incremental ship path. The CUTLASS two-stage results below are now a baseline and a proof that the SM120 block-scaled FP4 tensor path has enough headroom. The next prototype should fuse the two CUTLASS/CuTe block-scaled MMA stages into one attention kernel and should treat the hand-written warp-tile kernels as correctness scaffolding only.

## Target Cell

Gemma4 31B dense, Shape B global-attention production cell:

- Query dtype: BF16
- KV dtype: NVFP4 E2M1 packed bytes
- Scale dtype: FP8 E4M3, one scale per 16 FP4 values
- Output dtype: BF16
- Head dim: D=512
- GQA group: 8
- q_len: 512
- kv_len: 32768
- Batch: 1
- KV layout: contiguous, not paged
- Attention: dense, no causal mask, no sliding window

Raw launcher target:

```cpp
void attention_b1_d512_g8_q512_kv32768(
    const __nv_bfloat16* q,
    const uint8_t* k,
    const uint8_t* v,
    const __nv_fp8_e4m3* k_scales,
    const __nv_fp8_e4m3* v_scales,
    __nv_bfloat16* out);
```

## Dropped Constraints

The first reference kernel intentionally does not support:

- Template parameters
- JIT instantiation
- Variable sequence lengths
- Padding logic
- Causal masking
- Sliding window
- Paged KV
- FlashInfer plan/run wrappers
- Workspace or scheduler abstractions
- CUTLASS abstractions in the kernel body
- Persistent scheduling

Initial launch shape is one CTA per `(q_block, head)` unless measurement proves that is the immediate bottleneck.

## Known Baselines

Shape B q=512/kv=32768:

- Standalone CUTLASS block-scaled FP4 GEMM: 473 TFLOP/s
- Standalone BF16 GEMM: 235 TFLOP/s
- Existing FA2 FP4 attention: 4.8984 ms
- Existing FA2 FP8 attention: 5.0289 ms
- Existing FA2 FP4 / FP8: 1.03x

Interpretation:

The hardware-level FP4 math path has roughly 2x BF16 GEMM headroom at this shape, but existing fused FA2 captures only a small advantage over FP8. The reference kernel is meant to determine whether fused attention can capture that headroom.

## Required Kernel Strategy

The reference kernel must use the SM120 block-scaled FP4 MMA path:

```text
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec.*
```

That implies:

- Q must be quantized to NVFP4 inline before or during the QK path.
- K is already NVFP4 with E4M3 block scales.
- Softmax probabilities P must be quantized to NVFP4 inline before PV.
- V is already NVFP4 with E4M3 block scales.
- QK and PV should use FP32 accumulators.

Existing FA2 software-dequant path is not the target. The point is to remove the separate FP4→BF16 conversion and scale-apply instruction stream from the main MMA path.

## Build Order

1. Build a standalone extension and Python harness for the fixed target cell.
2. Generate deterministic tensors and a PyTorch BF16/FP32 reference.
3. Implement and validate Q load plus Q NVFP4 quantization.
4. Implement and validate K load plus a cp.async pipeline with at least 3 stages.
5. Implement QK via block-scaled FP4 MMA.
6. Implement online softmax.
7. Fuse softmax probability quantization to NVFP4.
8. Implement V load plus scale staging.
9. Implement PV via block-scaled FP4 MMA.
10. Write BF16 output.
11. Profile with Nsight Compute and iterate on smem layout, pipeline depth, register pressure, and SASS instruction mix.

Every checkpoint needs a correctness gate before performance work continues.

## Correctness Gate

The harness must compare against a PyTorch reference for the exact fixed target cell:

```python
scores = (q.float() @ k_dequant.float().transpose(-1, -2)) / sqrt(512)
p = torch.softmax(scores, dim=-1)
out_ref = p @ v_dequant.float()
```

Acceptance threshold starts loose for bring-up and tightens once all pieces are present:

- Bring-up: finite output, no NaNs, shape-correct.
- First complete kernel: relative/absolute error comparable to NVFP4 quantization noise.
- Final reference: stable over multiple random seeds and adversarial scale ranges.

## Performance Gate

Primary target:

```text
reference FP4 wall_ms < FP8 FA2 wall_ms
```

At the target cell, FP8 FA2 baseline is approximately `5.0289 ms`.

Ceiling-seeking target:

```text
fused CUTLASS/CuTe FP4 attention wall_ms < CUTLASS two-stage FP4 wall_ms < FP8 FA2 wall_ms
```

At the target cell, reproduced local baselines are:

```text
q_len=512, kv_len=32768, D=512, group=8
CUTLASS two-stage FP4: 0.634016 ms
FP8 FA2:               5.036832 ms
```

A meaningful win should be much larger than the current FA2 FP4 margin. The standalone GEMM result says the upper bound is not a 3-4% improvement; the fused kernel should determine whether attention can beat the already-fast two-stage CUTLASS baseline by eliminating QK/P HBM round trips and reusing P tiles directly into PV.

## Prior Findings To Carry Forward

- CUDA 13.2 is live and should be used for this track.
- Direct `cvt.rn.bf16x2.e2m1x2` exists and is already useful for software-dequant fallback paths, but it is not the main reference-kernel strategy.
- The bigger lever is SM120 block-scaled FP4 MMA via PTX 9.1+.
- CUTLASS example `3rdparty/cutlass/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu` is the concrete SM120 NVFP4/BF16 reference for block-scaled MMA.
- Existing probe files under `benchmarks/bench_sm120_nvfp4_*` and `benchmarks/probe_sm120_nvfp4_*` map operand layout, scale layout, and physical slot behavior.
- Existing FA2 FP4 path still uses software dequant plus BF16 MMA; it is not the reference for the new kernel.

## Rejected From This Track

Do not spend time on these unless the reference kernel later proves the cost is negligible:

- Paged KV support
- Sliding window support
- General D128/D256/D512 support
- Auto dispatch policy
- Existing FA2 sidecar layout compatibility
- Existing FlashInfer plan/run APIs
- Existing helper abstractions when inline PTX is clearer

## Open Questions

- What CTA tile shape captures the most of the standalone GEMM headroom once softmax and P quantization are included?
- Can P quantization be fused into the online softmax without introducing enough ALU/register pressure to erase the FP4 MMA win?
- Is the one-CTA-per-`(q_block, head)` grid enough for this cell, or does it need split-K/persistent scheduling after correctness?
- Does a 3-stage cp.async pipeline fit cleanly with Q/P/K/V scale state, or is 4-stage needed?
- How close can the fused kernel get to the 473 TFLOP/s standalone FP4 QK/PV GEMM ceiling after attention overhead?

## Current State

Standalone scaffold added:

- CUDA extension: `benchmarks/sm120_nvfp4_ref_attention.cu`
- Python harness: `benchmarks/bench_sm120_nvfp4_ref_attention.py`

Reference-only harness status:

```text
reference_shape: (512, 8, 512)
reference_dtype: torch.bfloat16
reference_finite: True
k_shape: (32768, 256)
scale_shape: (32768, 32)
```

Extension status:

```text
CUDA_HOME=/usr/local/cuda-13.2
TORCH_CUDA_ARCH_LIST=12.0f
extension build: passed
stub launcher: passed
stub_out_shape: (512, 8, 512)
stub_out_dtype: torch.bfloat16
stub_out_finite: True
```

The stub currently zero-fills output, so it is not expected to match the reference. This validates the standalone build path, fixed-shape tensor contract, row-major NVFP4 K/V packing, and PyTorch reference. Next step is Q-load plus Q NVFP4 quantization inside the CUDA extension.

## Q Quantization Bring-Up

Added extension entry point:

```cpp
void quantize_q_rowmajor(torch::Tensor q,
                         torch::Tensor q_packed,
                         torch::Tensor q_scales);
```

Contract:

- Input Q: `[512, 8, 512]`, BF16
- Output packed Q: `[4096, 256]`, uint8, two E2M1 values per byte
- Output scales: `[4096, 32]`, uint8 bytes interpreted as E4M3, one scale per 16 Q values

Validation result:

```text
q_quant_scale_match: True
q_quant_scale_diff: 0
q_quant_packed_match: False
q_quant_packed_diff: 3734
q_quant_dequant_delta_mean: 0.0001318157
q_quant_dequant_delta_max: 0.234375
q_quant_expected_error_mean: 0.0178635847
q_quant_actual_error_mean: 0.0178635847
```

Interpretation:

Scale bytes match exactly. Packed nibble differences are rare threshold/tie differences in E2M1 nearest-value rounding; the dequantized mean delta is far below the quantization error itself, and actual-vs-input quantization error matches the PyTorch reference. This is acceptable for Q quantization bring-up. The next step is wiring the quantized Q fragments into the first block-scaled QK MMA probe inside the reference extension.

## QK Native MMA Tile Bring-Up

Added extension entry point:

```cpp
void qk_tile_mma_debug(torch::Tensor q_packed,
                       torch::Tensor q_scales,
                       torch::Tensor k,
                       torch::Tensor k_scales,
                       torch::Tensor out_tile);
```

Contract:

- Q input: pre-quantized row-major NVFP4 `[4096, 256]` plus E4M3 scales `[4096, 32]`
- K input: row-major NVFP4 `[32768, 256]` plus E4M3 scales `[32768, 32]`
- Debug tile: head 0, query rows 0-15, KV rows 0-15
- Output: raw unscaled QK tile `[16, 16]`, FP32

Implementation uses the SM120 block-scaled FP4 MMA atom through:

```cpp
flashinfer::mma::mma_sync_m16n16k64_row_col_f4f4f32(...)
```

Validation result:

```text
qk_tile_finite: True
qk_tile_mean_abs: 0.0
qk_tile_max_abs: 0.0
qk_tile_cosine: 1.0000001192
qk_tile_sample: [-0.8890686, -1.4307098, 2.2789536, -0.2207489]
qk_ref_sample: [-0.8890686, -1.4307098, 2.2789536, -0.2207489]
```

Interpretation:

The first fixed 16x16 QK tile matches the dequantized NVFP4 PyTorch reference exactly. This verifies the row-major packed Q/K contract, CUTE A/B/SFA/SFB/C fragment layouts, E4M3 scale-register construction, and the SM120 block-scaled FP4 MMA wrapper for QK. The next step is expanding this from one KV tile to the full 32K KV span for one 16-query/head block.

## QK Full-KV Debug Span

Added extension entry point:

```cpp
void qk_full_mma_debug(torch::Tensor q_packed,
                       torch::Tensor q_scales,
                       torch::Tensor k,
                       torch::Tensor k_scales,
                       torch::Tensor out_scores);
```

Contract:

- Same pre-quantized Q/K inputs as the 16x16 tile probe
- Debug output: raw unscaled QK scores `[16, 32768]`, FP32
- Launch shape: one CTA per 16-column KV tile, `32768 / 16 = 2048` CTAs

Validation result:

```text
qk_full_finite: True
qk_full_mean_abs: 0.0
qk_full_max_abs: 0.0
qk_full_cosine: 1.0
qk_full_sample: [-0.8890686, -1.4307098, 2.2789536, -0.2207489]
qk_full_ref_sample: [-0.8890686, -1.4307098, 2.2789536, -0.2207489]
```

Interpretation:

The block-scaled FP4 QK path remains exact against the dequantized NVFP4 reference across the full 32K KV span for one 16-query/head block. This verifies the K row-major addressing and per-KV-row scale-register construction beyond the first tile. The next step is online-softmax bring-up and P quantization to NVFP4 over the 32K scores.

## Softmax And P Quantization Bring-Up

Added extension entry point:

```cpp
void softmax_quant_p_debug(torch::Tensor scores,
                           torch::Tensor p_packed,
                           torch::Tensor p_scales);
```

Contract:

- Input scores: raw QK `[16, 32768]`, FP32
- Softmax scale: `1 / sqrt(512)`
- Probability global scale: `6 * 448 = 2688`
- Output P: row-major NVFP4 `[16, 16384]`
- Output P scales: row-major E4M3 `[16, 2048]`

Validation result:

```text
p_quant_finite: True
p_quant_row_sum_min: 1.0448319
p_quant_row_sum_max: 1.0542938
p_quant_mean_abs: 3.5884434e-06
p_quant_max_abs: 6.5391177e-06
p_quant_cosine: 0.9938061
p_quant_scale_nonzero: 32768
```

Interpretation:

P quantization is finite and uses nonzero scale bytes across all 16x2048 scale groups. Row sums drift to roughly 1.05 after dequantization, which is expected from coarse E2M1 probability quantization at 32K context. This stage is acceptable for PV bring-up because the PV correctness reference should compare against the dequantized P actually consumed by the MMA, not against exact FP32 softmax.

## PV Scale-Orientation Finding

Native block-scaled PV cannot consume the current row-major V cache layout exactly.

Reason:

- QK uses B=K^T, so the MMA K dimension is head_dim. Row-major K scales are grouped across head_dim for each KV row, which matches the B operand scale layout.
- PV uses B=V, so the MMA K dimension is KV positions. Row-major V scales are grouped across head_dim for each KV row, which does not match the B operand scale layout. The hardware expects one B scale per 16 KV positions for each output column group.

Existing FlashInfer PV benchmarks already use a PV-oriented V layout: quantize `value_t` as `[head_dim, kv_len]`, then pass the transposed packed data to the runner. The reference kernel will use the same conceptual layout for PV bring-up:

```text
V_PV packed: [head_dim, kv_len / 2]
V_PV scales: [head_dim, kv_len / 16]
```

This is not a shortcut. It is the scale orientation required by the SM120 block-scaled PV MMA. Any production version that wants native FP4 PV must either store V in a PV-compatible layout or add an explicit reblocking/requantization stage whose cost is measured.

## PV Native MMA Tile Bring-Up

Added extension entry point:

```cpp
void pv_tile_mma_debug(torch::Tensor p_packed,
                       torch::Tensor p_scales,
                       torch::Tensor v_pv,
                       torch::Tensor v_pv_scales,
                       torch::Tensor out_tile);
```

Contract:

- P input: row-major NVFP4 `[16, 16384]` plus E4M3 scales `[16, 2048]`
- V input: PV-oriented NVFP4 `[512, 16384]` plus E4M3 scales `[512, 2048]`
- Debug tile: output columns 0-15
- Output: raw PV tile `[16, 16]`, FP32, divided by `2688` in the harness

Validation result:

```text
pv_tile_finite: True
pv_tile_vs_quant_p_mean_abs: 4.7192295e-10
pv_tile_vs_quant_p_max_abs: 2.7939677e-09
pv_tile_vs_quant_p_cosine: 0.9999998808
pv_tile_vs_exact_p_mean_abs: 0.0001502384
pv_tile_vs_exact_p_max_abs: 0.0005510398
```

Interpretation:

The native block-scaled FP4 PV tile matches the dequantized quantized-P/PV-layout-V reference. The remaining delta versus exact softmax is the expected P quantization loss. This verifies the P fragment layout, PV-oriented V scale layout, and post-PV `1 / PROB_GLOBAL_SCALE` compensation for one output tile. Next step is expanding PV to all 512 output columns for the 16-query/head block.

## PV Full-Width Block Bring-Up

Added extension entry point:

```cpp
void pv_full_mma_debug(torch::Tensor p_packed,
                       torch::Tensor p_scales,
                       torch::Tensor v_pv,
                       torch::Tensor v_pv_scales,
                       torch::Tensor out_block);
```

Contract:

- Same P and PV-oriented V inputs as the PV tile probe
- Output: `[16, 512]`, FP32, divided by `2688` in the harness
- Launch shape: one CTA per 16 output columns, `512 / 16 = 32` CTAs

Validation result:

```text
pv_full_finite: True
pv_full_vs_quant_p_mean_abs: 4.0341208e-10
pv_full_vs_quant_p_max_abs: 2.2118911e-09
pv_full_vs_quant_p_cosine: 0.9999998808
pv_full_vs_exact_p_mean_abs: 0.0001411682
pv_full_vs_exact_p_max_abs: 0.0006754489
```

Interpretation:

Full-width native block-scaled PV is correct against the quantized-P/PV-layout-V reference. QK, P quantization, and PV are now individually validated for the fixed 16-query/head block. The next problem is scheduling, not fragment correctness:

- A single warp can produce a 16x16 output tile.
- A full 16x512 output block needs 32 such N tiles.
- One warp cannot hold the full 16x512 FP32 output accumulator.

The reference fused design therefore needs either:

- a CTA with multiple PV consumer warps covering output-column tiles while sharing a QK/P tile producer, or
- a multi-kernel/staged reference that writes quantized P globally before PV, used only as a correctness/performance lower bound.

The production target remains a fused path. The staged path is useful only to validate math and quantify the cost of materializing P.

## Staged Block Timing

Added `--bench` timing to `benchmarks/bench_sm120_nvfp4_ref_attention.py`.

Command:

```bash
CUDA_HOME=/usr/local/cuda-13.2 \
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv \
TORCH_CUDA_ARCH_LIST=12.0f \
/home/josh/tdm/infer/current/.venv/bin/python \
  benchmarks/bench_sm120_nvfp4_ref_attention.py \
  --device 0 --bench --warmup 5 --repeat 20
```

Timing for one 16-query/head block:

```text
bench_qk_full min_ms: 0.021792
bench_softmax_quant_p min_ms: 0.016256
bench_pv_full min_ms: 0.509632
```

Interpretation:

The PV number is not representative of a full-grid kernel because the one-block debug launch only has 32 CTAs, one per output-column tile, and severely underfills the GPU. It is still useful as a correctness/per-CTA smoke. The full target has 256 query/head blocks, so a full-grid PV phase would launch `256 * 32 = 8192` CTAs and should have a different occupancy profile. Do not use the one-block PV timing as a shipping performance estimate.

The immediate scheduling lesson remains valid: full-width PV needs multiple output-column tiles. A fused production kernel needs to reuse each P tile across multiple PV consumers instead of recomputing or reloading P independently for every output-column tile.

## Full-Cell Staged Reference

Added optional full-cell staged path:

```bash
benchmarks/bench_sm120_nvfp4_ref_attention.py --full-staged
```

This path runs the whole target cell:

- Q rows: `512 * 8 = 4096`
- KV length: `32768`
- Output dim: `512`
- Stages: full QK materialization, full P quantization, full PV

Correctness result:

```text
full_staged_finite: True
full_staged_vs_exact_ref_mean_abs: 0.0001708317
full_staged_vs_exact_ref_max_abs: 0.0010864849
full_staged_first_block_vs_quant_ref_mean_abs: 4.0341208e-10
full_staged_first_block_vs_quant_ref_max_abs: 2.2118911e-09
```

The full staged output is correct against the quantized-P/PV-layout-V reference for the checked head-0 block. The delta versus exact PyTorch attention is dominated by Q/P/V quantization choices, not the block-scaled MMA.

Full-cell staged timing with `--bench --warmup 2 --repeat 5`:

```text
bench_full_qk_all min_ms: 4.592736
bench_full_softmax_quant_p_all min_ms: 0.859584
bench_full_pv_all min_ms: 6.421472
staged_total_min_ms: 11.873792
```

Interpretation:

The fully staged design is slower than the FP8 FA2 target (~5.03 ms). This decisively rules out materializing full QK and P as the shipping approach. The validated pieces are still useful:

- Native QK MMA is correct.
- Native PV MMA is correct when V is in PV-compatible scale layout.
- P quantization is correct enough for attention output.

The performance work now has to move to a fused producer/consumer design that avoids full QK/P HBM round trips and reuses each P tile across output-column PV consumers.

## Next Fused Prototype Shape

The next prototype should not materialize full QK or full P to HBM.

Working design:

- CTA owns one `(q_block, head)` with `M=16`.
- One producer warp computes QK tiles and quantizes P tiles.
- Multiple consumer warps compute PV for different output-column tiles.
- P tile lives in shared memory just long enough for all PV consumers to use it.
- V must be in PV-compatible layout `[head_dim, kv_len / 2]` with scales `[head_dim, kv_len / 16]`.

Key constraint:

```text
P tile reuse is mandatory.
```

If each output-column tile recomputes QK/softmax independently, QK cost is multiplied by 32 and the kernel cannot beat FP8. If full P is materialized globally, HBM traffic pushes the staged path to ~11.87 ms. The fused reference must sit between those two failures:

- no global QK/P materialization
- no per-output-column QK recomputation
- one P tile consumed by all active PV output-column consumers

Initial implementation target:

- Start with one CTA handling a small number of output-column tiles, likely 4 or 8 consumer warps, not all 32.
- Measure the cost of looping over output-column groups versus increasing CTA warp count.
- Keep the fixed target cell only: D=512, group=8, q_len=512, kv_len=32768.

Correctness plan:

1. Reproduce the staged output for one `(q_block, head)` using shared-memory P tiles.
2. Expand output-column coverage until `[16, 512]` is correct.
3. Expand grid to all 256 `(q_block, head)` blocks.
4. Only then optimize warp count, smem layout, pipeline depth, and P tile swizzle.

## Fused Prototype 1: Single CTA Per Query Block

Added:

```cpp
void fused_attention_all_debug(...);
```

Shape:

- one CTA per `(q_block, head)`
- 17 warps per CTA: one QK/P producer warp plus 16 PV consumer warps
- one CTA walks the full 32K KV sequence
- P is kept in shared memory and reused across all PV output-column consumers

Correctness:

```text
fused_finite: True
fused_vs_exact_ref_mean_abs: 0.0001613715
fused_vs_exact_ref_max_abs: 0.0011675542
```

Timing:

```text
bench_fused_attention_all min_ms: 46.369823
```

Interpretation:

Correctness passes, but the shape is far too serial. Only 256 CTAs are launched and each CTA walks all 32K KV tiles. This underfills the GPU and cannot be the performance reference.

## Fused Prototype 2: Split-KV Partials

Added:

```cpp
void fused_attention_split_partial_debug(...);
void fused_attention_split_reduce_debug(...);
```

Shape:

- split KV into 32 chunks of 1024 tokens
- partial kernel grid: `(q_block_head=256, split=32)` = 8192 CTAs
- each CTA keeps QK/P/PV fused within its KV chunk
- stores only partial `O`, `m`, and `l`
- reduction combines partials with standard log-sum-exp rescaling

Correctness:

```text
split_fused_finite: True
split_fused_vs_exact_ref_mean_abs: 0.0001620132
split_fused_vs_exact_ref_max_abs: 0.0010063406
```

Timing:

```text
bench_split_fused_partial min_ms: 37.361153
bench_split_fused_reduce min_ms: 0.182400
```

Resource usage:

```text
fused_attention_split_partial_kernel:
  REG:96 STACK:3048 SHARED:5888
fused_attention_all_debug_kernel:
  REG:96 STACK:5304 SHARED:5888
```

Interpretation:

Split-KV improves occupancy versus the single-CTA version but is still much slower than the staged debug kernels and far slower than FP8 FA2. The reduction is not the bottleneck. The partial kernel is dominated by the fused CTA body.

Two concrete problems:

- The fused CTA structure has high register/stack pressure.
- More importantly, K and V fragments are rebuilt from row-major packed bytes with scalar gather and scale-register construction for every MMA. This measures row-major decode overhead, not the SM120 block-scaled MMA ceiling.

Next direction:

Use an unconstrained prepacked-fragment layout for K/V in the reference kernel:

```text
K fragments: [kv_tile16, k_block64, lane, frag_reg]
K scales:    [kv_tile16, k_block64, lane, scale_reg_pair]
V fragments: [out_col_tile16, kv_block64, lane, frag_reg]
V scales:    [out_col_tile16, kv_block64, lane, scale_reg_pair]
```

This is acceptable for the perf-ceiling lab. It answers whether fused attention can be fast when K/V are already in the MMA-native fragment layout. If it wins, production work can decide whether to store the KV cache that way or add a measured reblocking path.

## Prepacked K/V Fragment Result

Added:

```cpp
void prepack_k_fragments_debug(...);
void prepack_v_fragments_debug(...);
```

K/V fragment layout:

```text
K fragments: [2048, 8, 32, 4] int32
K scales:    [2048, 8, 32, 2] int32
V fragments: [32, 512, 32, 4] int32
V scales:    [32, 512, 32, 2] int32
```

The split partial kernel now consumes these prepacked fragments directly.

Correctness stayed stable:

```text
split_fused_finite: True
split_fused_vs_exact_ref_mean_abs: 0.0001620129
split_fused_vs_exact_ref_max_abs: 0.0010063406
```

Timing:

```text
bench_split_fused_partial min_ms: 12.288832
bench_split_fused_reduce min_ms: 0.181888
bench_prepack_k min_ms: 0.015584
bench_prepack_v min_ms: 0.016288
```

Interpretation:

Prepacking K/V into MMA-native fragment layout is a major improvement:

```text
row-major split partial: 37.36 ms
prepacked split partial: 12.29 ms
```

This confirms row-major fragment decode was a large false bottleneck. However, the prepacked fused partial is still slower than FP8 FA2 and roughly comparable to the fully staged reference. The remaining bottleneck is the CTA execution structure: large 17-warp CTAs, barriers every 64 KV tokens, serial producer/consumer phases, and low residency.

Rejected probe:

```text
__launch_bounds__(kFusedThreads, 2)
```

Result:

```text
bench_split_fused_partial min_ms: 21.512
```

Forcing two resident CTAs caused spills and regressed. Keep `__launch_bounds__(kFusedThreads, 1)` for this shape.

Split-size probe:

```text
KV split 1024: partial 12.29 ms, reduce 0.18 ms
KV split 512:  partial 15.70 ms, reduce 0.35 ms
```

1024-token splits are better. Smaller chunks increase partial traffic and overhead more than they improve CTA occupancy.

## CUTLASS Two-Stage Ceiling Check

Command:

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv \
/home/josh/tdm/infer/current/.venv/bin/python \
  benchmarks/bench_nvfp4_native_attention_gemm.py \
  --m 4096 --n 32768 --d 512 \
  --warmup 2 --repeat 5 --device 0 \
  --softmax-quant-threads 256
```

This matches the full Shape B q_len=512, group=8 cell:

```text
m = q_len * group = 512 * 8 = 4096
n = kv_len = 32768
d = 512
```

Result:

```text
qk_fp4 min_ms: 0.213440
softmax_p_quantize_fused min_ms: 0.265888
pv_fp4_after_fused_p min_ms: 0.155616
two_stage_fp4_fused_softmax_quant min_ms: 0.634016

qk_fp4_tflops_min_ms: 643.9
pv_fp4_tflops_min_ms: 879.9
```

Comparison:

```text
FP8 FA2 target at this cell: ~5.03 ms
CUTLASS two-stage FP4:       ~0.63 ms
```

Interpretation:

The hardware ceiling is not the hand-written warp-tile fused kernel. CUTLASS block-scaled GEMM captures the SM120 NVFP4 tensor path by a wide margin. The current hand-written fused prototypes are slow because they are warp-tile kernels with large CTA barriers, low tensor-pipe utilization, and scalar-heavy softmax/P packing. They are useful correctness scaffolds, not the path to a shippable fast kernel.

This also falsifies one earlier assumption for this production cell: full QK/P materialization is not necessarily too expensive. At q_len=512, kv=32K, D=512, the two-stage CUTLASS path beats FP8 FA2 by roughly 8x despite materializing logits/P. The next production direction should be a CUTLASS-backed two-stage or hybrid policy for prefill cells where materialization fits, plus XQA/decode or another backend for q=1.

Open production questions:

- Include Q quantization and KV-cache layout/reblocking costs in end-to-end timing.
- Sweep Shape B q_len `{512, 2048}` and kv `{8K, 32K, 131K}` for memory and wall time.
- Sweep Shape A D=256/group=2/kv=1024 to see whether the same two-stage path wins or whether FA2 FP8 remains better.
- Decide whether the production KV cache can be stored in CUTLASS/PV-compatible layouts or whether a reblocking path is required.

## CUTLASS Two-Stage Shape B q512 kv8K Check

Command:

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv \
/home/josh/tdm/infer/current/.venv/bin/python \
  benchmarks/bench_nvfp4_native_attention_gemm.py \
  --m 4096 --n 8192 --d 512 \
  --warmup 2 --repeat 5 --device 0 \
  --softmax-quant-threads 256
```

Result:

```text
qk_fp4 min_ms: 0.057152
softmax_p_quantize_fused min_ms: 0.062080
pv_fp4_after_fused_p min_ms: 0.048928
two_stage_fp4_fused_softmax_quant min_ms: 0.155264

qk_fp4_tflops_min_ms: 601.2
pv_fp4_tflops_min_ms: 693.6
```

Interpretation:

The CUTLASS two-stage path remains extremely fast at the shorter Shape B kv=8K point. The two measured q512 cells now are:

```text
D512 group8 q512 kv8K:  0.155 ms
D512 group8 q512 kv32K: 0.634 ms
```

Scaling is roughly linear in KV length, which is what we want to see from a real GEMM-dominated path.

## Fused Producer/Consumer Prototype

Added a split-KV pipelined partial kernel:

```text
fused_attention_split_partial_pipelined_kernel
```

Design:

- Same fixed Shape B cell: D=512, group=8, q_len=512, kv_len=32768.
- Same split layout as the previous split fused kernel: 32 splits of 1024 KV tokens.
- Prepacked K/V fragments are still used, so row-major fragment decode is not the measured bottleneck.
- Producer warp computes QK + online softmax + P quantization into a double-buffered shared-memory P tile.
- Consumer warps consume the previous P tile for PV while the producer starts the next QK/P tile.
- Shared flags replace the full-block barrier between every 64-token tile.

Correctness:

```text
split_fused_finite: True
split_fused_vs_exact_ref_mean_abs: 0.0001620132
split_fused_vs_exact_ref_max_abs: 0.0010063406
```

Timing, same build/run conditions:

```text
non-pipelined split partial min_ms: 12.110592
pipelined split partial min_ms:     8.130048
split reduce min_ms:                0.181344
prepack K min_ms:                   0.015680
prepack V min_ms:                   0.016992
```

Resource usage:

```text
fused_attention_split_partial_kernel:
  REG:96 STACK:1872 SHARED:5888

fused_attention_split_partial_pipelined_kernel:
  REG:96 STACK:1960 SHARED:6544
```

Interpretation:

The producer/consumer overlap is real. It improves the partial kernel by about 33% with essentially the same register footprint. This proves that the previous full-block barrier structure was a material bottleneck.

It does not close the gap to the CUTLASS two-stage ceiling. Even after overlap, the hand-written warp-tile fused path is still roughly 13x slower than CUTLASS two-stage for the same q512/kv32K/D512 cell.

NCU one-launch counter pass on the pipelined partial kernel:

```text
tensor pipe active:       1.91%
warps active:             35.37%
warp latency / issued:    28.49 cycles
global ld inst:           112,459,776
global st inst:           2,113,536
shared ld inst:           737,351,714
shared st inst:           11,689,984
uniform branch targets:   99.62%
```

Comparison against the earlier non-pipelined NCU pass:

```text
non-pipelined tensor pipe active:    1.30%
non-pipelined warp latency/issued:   86.28 cycles
```

The overlap mostly reduced warp issue latency and raised tensor-pipe activity slightly. The remaining problem is still structural: shared/global instruction volume is enormous and tensor-pipe utilization is far below what the CUTLASS GEMM path demonstrates is possible.

Rejected follow-up:

Tried prepacking each P tile into SM120 MMA fragment layout once in the producer warp and having all consumers load the ready fragments directly from shared memory. This preserved correctness but regressed:

```text
pipelined baseline partial min_ms:      8.130048
pipelined P-frag-prepack partial min_ms: 9.226560
```

Conclusion: in this CTA structure, extra producer work and extra shared stores cost more than reducing the consumer-side row-major P gather. Do not retry P-fragment prepack inside this 17-warp CTA structure unless the producer work is moved to a different kernel shape.

## CUTLASS Two-Stage Production Grid Sweep

All runs used:

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv \
/home/josh/tdm/infer/current/.venv/bin/python \
  benchmarks/bench_nvfp4_native_attention_gemm.py \
  --device 0 --softmax-quant-threads 256
```

Shape B maps to `m = q_len * group = q_len * 8`, `d = 512`.

```text
Shape B global, D512 group8

q_len  kv_len   m      n       qk_fp4_ms  softmax_p_ms  pv_fp4_ms  two_stage_ms
512    8K       4096   8192    0.057152   0.062080      0.048928   0.155264
512    32K      4096   32768   0.213440   0.265888      0.155616   0.634016
512    131K     4096   131072  0.877696   2.234016      0.664064   3.717632
2048   8K       16384  8192    0.203808   0.263712      0.137632   0.624640
2048   32K      16384  32768   0.854912   1.134944      0.734176   2.638720
2048   131K     16384  131072  3.439904   8.985984      2.906720   17.169855
```

Interpretation:

- Shape B two-stage scales predictably with `m*n`.
- QK and PV are consistently fast and stay in the several-hundred-TFLOP/s range.
- At long KV and q_len=2048, fused softmax/P quantization becomes the dominant stage: 8.99 ms of the 17.17 ms total.
- This strongly argues for a CUTLASS-backed two-stage or hybrid prefill path for Shape B, with follow-up work focused on softmax/P quantization and memory policy rather than hand-written warp-tile MMA.

Shape A maps to `m = q_len * group = q_len * 2`, `n = 1024`, `d = 256`.

```text
Shape A sliding, D256 group2 kv1024

q_len  m     qk_fp4_ms  softmax_p_ms  pv_fp4_ms  two_stage_ms  bf16_torch_attention_ms
1      2     0.016416   0.017216      0.020256   0.046272     0.031584
512    1024  0.016352   0.017216      0.020288   0.047712     0.032352
2048   4096  0.020704   0.026304      0.020384   0.048384     0.043072
```

Interpretation:

- Shape A is launch/overhead dominated in this two-stage design.
- The FP4 GEMMs do not have enough arithmetic work at `D=256, kv=1024` to amortize the two-stage materialization path.
- This is consistent with the earlier finding that Shape A FP4 FA2 trails FP8 FA2 by about 12-15%.
- Production policy should not force CUTLASS two-stage for Shape A. Shape A needs either the existing FA2/FP8 path, a specialized fused small-KV kernel, or a separate Shape A-specific optimization. The Shape B two-stage win does not transfer to sliding attention.

## FA2 D512 Paged Prefill Baseline Check

Added `fp8` support to `benchmarks/bench_nvfp4_d512_prefill.py` so the FA2 FP8 baseline can be measured in the same local FlashInfer tree instead of relying on old notes.

Command shape:

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv \
/home/josh/tdm/infer/current/.venv/bin/python \
  benchmarks/bench_nvfp4_d512_prefill.py \
  --backend fa2 --batch-size 1 \
  --q-len 512 --kv-len 32768 \
  --num-qo-heads 32 --num-kv-heads 4 \
  --head-dim 512 --workspace-mib 2048 \
  --warmup 2 --repeat 5 --device 0
```

Results:

```text
FA2 FP8 KV:   min_ms 24.838144
FA2 NVFP4 KV: min_ms 23.910368
FA2 BF16 KV:  invalid D512 configuration in this local path
```

The BF16 failure:

```text
Invalid configuration:
NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=1
NUM_WARPS_Q=4 NUM_WARPS_KV=1
```

Equivalent full-layer CUTLASS two-stage comparison:

The per-KV-head Shape B q512/kv32K CUTLASS result is `m=4096,n=32768,d=512` at `0.634 ms`. A rough full-layer equivalent for 4 KV heads is represented by the measured `m=16384,n=32768,d=512` cell:

```text
CUTLASS two-stage full-layer-equivalent: 2.638720 ms
FA2 FP8 paged prefill:                  24.838144 ms
FA2 NVFP4 paged prefill:                23.910368 ms
```

Interpretation:

- In this local FlashInfer path, FA2 D512 paged prefill is not a competitive baseline.
- NVFP4 FA2 is only slightly faster than FP8 FA2 at this shape, and both are much slower than CUTLASS two-stage.
- The old ~5 ms FP8 FA2 target should not be used as authoritative unless reproduced under a matching harness. The local evidence says Shape B prefill should move toward CUTLASS block-scaled GEMM, not FA2.

## Shape B Direct FP8 FA2 Comparison

The correct apples-to-apples FA2 comparison is the per-KV-head grouped GQA shape used by `bench_nvfp4_fmha_v2_gqa_grouped_attention.py`, not the full-layer D512 paged script above.

Command shape:

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv \
/home/josh/tdm/infer/current/.venv/bin/python \
  benchmarks/bench_nvfp4_fmha_v2_gqa_grouped_attention.py \
  --gemma4-shape global \
  --head-dim 512 --group-sizes 8 \
  --fp4-backend fa2 \
  --fp4-v-layout nhd --fp4-v-sf-layout linear \
  --only fp8 --device 0
```

Direct per-group comparison:

```text
Shape B global, D512 group8

q_len  kv_len  CUTLASS two-stage FP4 ms  FP8 FA2 ms   FP4 speedup vs FP8
512    8K      0.155264                  1.323296     8.52x
512    32K     0.634016                  5.036832     7.94x
512    131K    3.717632                  20.048544    5.39x
2048   8K      0.624640                  5.686400     9.10x
2048   32K     2.638720                  24.330240    9.22x
2048   131K    17.169855                 99.509758    5.80x
```

Interpretation:

- CUTLASS two-stage FP4 beats FP8 FA2 on every measured Shape B prefill cell.
- The speedup is largest at 8K/32K and narrows at 131K because the fused softmax/P quantization stage becomes the dominant cost.
- This is now the strongest production signal in the lab: for Gemma4 global-attention prefill on SM120, block-scaled CUTLASS FP4 GEMMs plus a better softmax/P stage are the route to beating FP8 FA2. The hand-written fused CTA reference is useful only as a correctness and scheduling experiment.

## Shape A Direct FP4 / FP8 / BF16 Comparison

Command shape:

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv \
/home/josh/tdm/infer/current/.venv/bin/python \
  benchmarks/bench_nvfp4_fmha_v2_gqa_grouped_attention.py \
  --gemma4-shape sliding \
  --kv-len 1024 --head-dim 256 --group-sizes 2 \
  --fp4-backend fa2 \
  --fp4-v-layout nhd --fp4-v-sf-layout linear \
  --only all --device 0
```

Result:

```text
Shape A sliding, D256 group2 kv1024

q_len  FP4 FA2 ms  FP8 FA2 ms  BF16 FA2 ms  FP4 / FP8
1      0.023744    0.021184    0.160032     0.89x
512    0.032000    0.027968    0.165536     0.87x
2048   0.060160    0.052544    0.160960     0.87x
```

Interpretation:

- Shape A is not a CUTLASS two-stage target and not an FP4 FA2 win today.
- FP8 FA2 is consistently faster than FP4 FA2 for the sliding layers.
- Both quantized paths are much faster than BF16.
- Current production policy should be: Shape B prefill uses the CUTLASS block-scaled FP4 two-stage path; Shape A stays on FP8 FA2 unless a dedicated small-KV FP4 kernel is built.

## Ceiling Direction: Fused CUTLASS/CuTe Block-Scaled Attention

The user explicitly set the target as faster performance and ceiling seeking. That means the next milestone is not another FA2 threshold, sidecar-layout, or scalar warp-tile tweak. The active target is a fused SM12x attention kernel that uses the same CUTLASS/CuTe block-scaled FP4 atom family as the fast two-stage GEMM path.

Concrete reference sources:

```text
3rdparty/cutlass/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu
include/flashinfer/gemm/group_gemm_nvfp4_groupwise_sm120.cuh
include/flashinfer/mma.cuh
3rdparty/cutlass/include/cute/arch/mma_sm120.hpp
```

Relevant implementation facts:

- CUTLASS Example 79a is the SM120/GeForce NVFP4->BF16 GEMM reference.
- It uses `cutlass::arch::Sm120` and `cutlass::arch::OpClassBlockScaledTensorOp`.
- It selects `cutlass::nv_float4_t<cutlass::float_e2m1_t>` for both A and B operands and BF16 output.
- The underlying atom is the SM120 `m16n8k64` block-scaled MMA with UE4M3 scale factors.
- FlashInfer's groupwise SM120 NVFP4 GEMM already wraps this family for production GEMM use.

New success bar for the first fused Shape B prototype:

```text
Shape B q=512 kv=32768 D=512 group=8
fused CUTLASS/CuTe FP4 attention < 0.634016 ms
```

The `0.634016 ms` value is the measured CUTLASS two-stage FP4 baseline for the same per-KV-head grouped-GQA cell. The fused kernel must beat that by avoiding full logits/P materialization and by reusing a P tile directly into PV. Beating FP8 FA2 is already proven by two-stage; the ceiling kernel exists to beat two-stage.

Design target:

- BF16 Q is quantized to NVFP4 inline or in a producer phase using the same scale layout expected by the SM120 block-scaled MMA.
- QK uses CUTLASS/CuTe SM120 block-scaled MMA, not the scalar hand-packed warp-tile loop.
- Online softmax is performed tile-wise with stable row max/sum state.
- P is quantized to NVFP4 tile-wise and fed directly to PV without writing full P to HBM.
- PV uses CUTLASS/CuTe SM120 block-scaled MMA.
- Output is BF16.

Rejected for this next milestone:

- More fmha_v2 sidecar tuning.
- More FA2 threshold tuning.
- Extending the slow 17-warp hand-written CTA as the performance path.
- Paged KV, variable shape, masking, and production dispatch before the fixed fused cell beats the two-stage baseline.

Immediate coding task:

Create a dedicated fused CUTLASS/CuTe prototype file rather than continuing to grow `sm120_nvfp4_ref_attention.cu`. The first compile target should instantiate the SM120 block-scaled atom/collective for the fixed Shape B cell and expose enough debug hooks to validate QK tile correctness before adding online softmax and PV.

## Dedicated CUTLASS/CuTe Prototype Bring-Up

Added:

```text
benchmarks/sm120_nvfp4_cutlass_fused_attention.cu
benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py
```

First compile/run command:

```bash
CUDA_VISIBLE_DEVICES=2 \
CUDA_HOME=/usr/local/cuda-13.2 \
TORCH_CUDA_ARCH_LIST=12.0f \
PYTHONPATH=/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv \
/home/josh/tdm/infer/current/.venv/bin/python \
  benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py --device 0
```

Result:

```text
q_quant_scale_match: True
q_quant_packed_match: False
q_quant_dequant_delta_mean: 0.00013181567192077637
q_quant_dequant_delta_max: 0.234375
qk_tile_finite: True
qk_tile_mean_abs: 0.0
qk_tile_max_abs: 0.0
qk_tile_cosine: 1.0000001192092896
```

Interpretation:

- The dedicated prototype builds cleanly under CUDA 13.2 targeting `sm_120f`.
- The fixed Shape B Q quantization behavior matches the earlier reference path.
- The new prototype's SM120 block-scaled QK atom reproduces the PyTorch dequantized 16x16 QK tile exactly.
- This is the correct starting point for the fused CUTLASS/CuTe path. Next checkpoint is full 16-row QK over all 32768 KV tokens in this dedicated file, then tile-wise online softmax/P quantization, then PV.

Added a full 16-row QK block checkpoint in the same dedicated file:

```text
qk_block_finite: True
qk_block_mean_abs: 0.0
qk_block_max_abs: 0.0
qk_block_cosine: 1.0
```

This validates the block-scaled atom mapping over the entire `kv_len=32768` row for one 16-query tile. The current debug path writes full scores to HBM only as a correctness checkpoint. It is not the intended performance path; the next stage replaces this with online softmax/P quantization inside the fused pipeline.

Added softmax/P quantization and PV checkpoints in the dedicated prototype:

```text
p_quant_finite: True
p_quant_row_sum_min: 1.0448318719863892
p_quant_row_sum_max: 1.0542937517166138
p_quant_mean_abs: 3.588443405533326e-06
p_quant_max_abs: 6.539117748616263e-06
p_quant_cosine: 0.9938060641288757
pv_block_finite: True
pv_block_vs_quant_p_mean_abs: 4.034120815177289e-10
pv_block_vs_quant_p_max_abs: 2.2118911147117615e-09
pv_block_vs_quant_p_cosine: 0.9999998807907104
pv_block_vs_exact_p_mean_abs: 0.00014116820238996297
pv_block_vs_exact_p_max_abs: 0.0006754489149898291
```

Interpretation:

- The materialized debug chain `QK -> softmax/P quant -> PV` is correct for one 16-row Shape B block.
- PV agrees with the quantized-P reference almost exactly after dividing by `PROB_GLOBAL_SCALE`.
- The remaining exact-P error is the expected probability quantization loss.
- The next performance step is to remove the materialized `scores` and `p_packed/p_scales` HBM round trips and reuse P tiles directly in the PV path.

Timing of the dedicated debug stages for one 16-row Q tile:

```text
bench_qk_block min_ms:             0.021504
bench_softmax_quant_p_block min_ms: 0.016768
bench_pv_block min_ms:             0.461184
```

Interpretation:

- These timings are for the debug/scalar-fragment path, not the ceiling path.
- PV is the obvious failure point in this checkpoint design: it computes the 16x512 output block with 32 small CTAs and a scalar fragment loop over all 32768 KV entries.
- The fast CUTLASS two-stage PV baseline for the full Shape B cell is `0.155616 ms`, so this PV debug kernel is only a correctness scaffold.
- The fused ceiling kernel must use CUTLASS's SM120 smem layout/copy atoms and block-scaled tensor-core scheduling for PV, not the scalar fragment path.

## CUTLASS FMHA Architecture Read

Relevant CUTLASS files inspected:

```text
3rdparty/cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp
3rdparty/cutlass/examples/88_hopper_fmha/kernel/fmha_kernel_tma.hpp
3rdparty/cutlass/examples/88_hopper_fmha/kernel/fmha_kernel_builder.hpp
3rdparty/cutlass/examples/88_hopper_fmha/collective/fmha_collective_tma_warpspecialized.hpp
3rdparty/cutlass/examples/88_hopper_fmha/collective/fmha_collective_softmax.hpp
```

Facts:

- The SM120 block-scaled GEMM collective already exposes the useful split points: producer `load(...)`, consumer `mma(...)`, and separate shared storage for A/B/SFA/SFB.
- Its producer uses TMA to load A, B, SFA, and SFB into smem and its consumer copies smem fragments/scales into register fragments before calling the block-scaled MMA.
- Hopper FMHA Example 88 is the right architectural template: QK MMA -> online softmax state -> PV MMA -> epilogue, all inside one kernel instead of returning to a GEMM epilogue.
- Example 88 is not directly reusable as-is for SM120 NVFP4 because it is SM90-oriented and assumes the FMHA element type path, not SM120 NVFP4 E2M1 plus UE4M3 sidecar scales.
- The correct porting unit is therefore not `GemmUniversalAdapter`; it is a custom SM120 attention mainloop that borrows the CUTLASS producer pipeline/layout/fragment copy machinery and swaps the epilogue for online softmax plus P quantization plus PV.

Concrete next design point:

- QK can use a CUTLASS-style SM120 block-scaled producer/consumer over quantized Q and K.
- PV cannot simply call a second GEMM collective with a TMA-loaded A operand, because A is dynamic P generated from QK. P must be quantized into a layout consumable by the SM120 block-scaled PV MMA, ideally in smem/registers, then consumed directly by PV.
- The current dedicated debug chain proves the numerical pieces. The next kernel should merge those pieces into a single custom mainloop, starting with one fixed 16-query tile, then replacing the scalar hand-packed fragments with CUTLASS's smem layout and copy atoms.

Added a compile-time metadata hook for the exact SM120 block-scaled CUTLASS collective in the dedicated extension.

Result:

```text
arch: sm120
operator_class: OpClassBlockScaledTensorOp
tile: 128x128x128
scale_vec_size: 16
thread_count: 256
mainloop_shared_storage_bytes: 74752
epilogue_shared_storage_bytes: 13312
layout_sfa_bytes: 20
layout_sfb_bytes: 20
```

Interpretation:

- The CUTLASS collective needed for the ceiling path compiles cleanly inside the PyTorch extension under CUDA 13.2 `sm_120f`.
- Mainloop plus epilogue shared storage is `88064` bytes before custom attention storage, fitting under SM120's 99 KiB opt-in shared-memory limit.
- The first fused implementation should budget shared memory carefully: there is enough room for the CUTLASS QK mainloop, but not enough to casually add a second full CUTLASS collective plus large P tiles.
- This points toward a custom attention mainloop that reuses one CUTLASS-style pipeline at a time and stores only the P tile/scale state needed for immediate PV consumption.

## Split-Fused Producer/Consumer Prototype

The first on-chip producer/consumer prototype is in
`benchmarks/sm120_nvfp4_ref_attention.cu` under
`fused_attention_split_partial_pipelined_kernel`.

Target cell:

```text
Shape B: D=512, group=8, q_len=512, kv_len=32768
Q/output: BF16
K/V/P: NVFP4 E2M1 plus E4M3 group scale
```

Performance bars:

```text
FP8 FA2 deployed baseline:          ~5.04 ms
Current FA2 FP4 path:               ~4.90 ms
CUTLASS two-stage FP4 ceiling ref:  ~0.634 ms
  QK:      0.213 ms
  P/soft:  0.266 ms
  PV:      0.156 ms
```

Split-fused measurements:

```text
one producer warp, tile=64, split=1024:
  partial: 8.13 ms
  reduce:  0.18 ms
  total:   8.31 ms

one producer warp, tile=64, split=2048:
  partial: 8.14 ms
  reduce:  0.076 ms
  total:   8.22 ms

one producer warp, tile=64, split=4096:
  partial: 9.36 ms
  reduce:  0.016 ms
  total:   9.37 ms

one producer warp, tile=64, split=512:
  partial: 8.36 ms
  reduce:  0.355 ms
  total:   8.71 ms

four producer warps, tile=64, split=2048:
  partial: 6.88 ms
  reduce:  0.077 ms
  total:   6.95 ms

four producer warps, tile=64, split=1024:
  partial: 7.05 ms
  reduce:  0.18 ms
  total:   7.23 ms

four producer warps, tile=64, split=4096:
  partial: 6.75 ms
  reduce:  0.018 ms
  total:   6.77 ms

four producer warps, tile=64, split=8192:
  partial: 7.16 ms
  reduce:  0.011 ms
  total:   7.17 ms

eight producer warps, tile=128, split=4096:
  partial: 7.17 ms
  reduce:  0.016 ms
  total:   7.19 ms
```

Interpretation:

- Split length alone is not the main lever. Increasing split length reduces
  final reduction cost, but the partial kernel dominates.
- Four producer warps are a real win versus one producer warp, proving the
  one-warp QK producer was a bottleneck.
- `tile=128` regresses despite halving the tile loop count. The larger tile
  increases per-tile softmax/P work and PV issue count enough to lose in this
  hand-written pipeline.
- The best hand-written prototype so far is still slower than FP8 FA2:
  `~6.77 ms` versus `~5.04 ms`.
- This prototype is useful for isolating structure, but it is not the final
  ceiling path. The remaining gap to the `0.634 ms` two-stage CUTLASS
  reference is too large for split/tile tuning.

Correctness notes:

- The split-fused path is finite and close to the exact PyTorch reference.
- Comparing split-fused directly to the materialized staged P path is not a
  strict equality test because split-fused quantizes online per split before
  final softmax reduction, while the staged path quantizes final full-softmax
  probabilities.
- The final shippable path should avoid split-local probability quantization if
  it causes unacceptable accuracy drift; the current split path is a
  performance probe, not the final numerical contract.

Rejected or postponed paths:

- Directly embedding the CUTLASS SM120 block-scaled TMA collective compiled but
  hit an illegal instruction under sanitizer in `SM90_TMA_LOAD_3D::copy` from
  `sm120_blockscaled_mma_tma.hpp`. That path likely needs the full CUTLASS
  scheduler/launch protocol or a deeper port, not a small wrapper.
- `tile=128` is not a win in the current producer/consumer implementation.
- More split-length tuning will not reach the ceiling; the best total improved
  only to `~6.77 ms`.

Next structural direction:

- Stop treating the current producer/consumer kernel as the final architecture.
- Either port the CUTLASS block-scaled collective pipeline correctly into an
  attention mainloop, or write the equivalent CuTe/CUTLASS atom path directly
  with proper smem layouts and a persistent work schedule.
- The goal remains fusing the fast CUTLASS FP4 QK and PV phases while keeping
  P on-chip.

## Architecture Pivot: Close The Pure Hand-Written Path

The pure hand-written warp-tile kernel is no longer the active performance
path. It remains useful as a correctness scaffold and as evidence for operand
layouts, but it is not close enough to the two-stage CUTLASS baseline:

```text
best hand-written split-fused prototype: ~6.77 ms
CUTLASS two-stage FP4 baseline:          ~0.634 ms
gap:                                     ~10.7x
```

This is not a tile-size or synchronization tuning problem. The hand-written
prototype lacks the important CUTLASS pieces that make the two-stage baseline
fast:

- TMA-backed block-scaled mainloop.
- CUTLASS/CuTe smem layouts for SM120 NVFP4 operands and scale tensors.
- Persistent scheduler / PDL launch behavior.
- Efficient epilogue/writeback machinery.
- High tensor-pipe utilization from the production mainloop.

The active implementation direction is now:

```text
start from the fast CUTLASS/CuTe SM120 block-scaled FP4 mainloop
then build an FMHA-style kernel around it:
  QK block-scaled MMA
  online row max/sum softmax state
  P quantization to NVFP4 tile/register layout
  PV block-scaled MMA
  BF16 output
```

This is conceptually the CUTLASS Example 88 FMHA architecture, but with SM120
NVFP4 block-scaled MMA instead of SM90 tensor-op GMMA.

Implementation facts from source review:

- FlashInfer's fast two-stage path goes through `CutlassFp4GemmRunner` in
  `include/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h`.
- The runner dispatches SM120 CTA shapes:
  `128x128x128`, `128x128x256`, and `256x128x128`.
- It supports both the default persistent scheduler and StreamK.
- It launches through `GemmUniversalAdapter` with PDL enabled.
- CUTLASS Example 88 shows the correct FMHA control flow:
  separate Q and K/V load pipelines, QK MMA, online softmax, PV MMA, epilogue.

Important constraint:

- Fusing the two CUTLASS phases is not a normal GEMM epilogue fusion. QK tiles
  cover only a slice of `N`, but softmax is row-global over the full KV length.
  The fused kernel therefore has to be an FMHA mainloop with online softmax,
  not a QK GEMM with a custom epilogue.

## FlashInfer Blackwell FMHA Reference

Checked `include/flashinfer/attention/blackwell/fmha_cutlass_sm100.cuh` and
the associated Blackwell FMHA files:

- `include/flashinfer/attention/blackwell/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`
- `include/flashinfer/attention/blackwell/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp`
- `include/flashinfer/attention/blackwell/device/fmha.hpp`

This is a better architectural reference than standalone CUTLASS Example 88
because it is FlashInfer's integrated FMHA wrapper and scheduler path. It
already has the right decomposition:

- Q/K/V load pipelines.
- QK MMA producing score tiles.
- Online softmax state.
- PV MMA consuming probability tiles.
- Correction/rescale.
- Epilogue/writeback.
- Host-precomputed tile scheduling and PDL-style launch plumbing.

The code is not directly portable to SM120 because it is SM100-oriented:

- Uses `cutlass::arch::Sm100`.
- Uses `KernelTmaWarpSpecialized1SmSm100`.
- Uses TMEM allocation and `SM100_TMEM_*` copies.
- Uses UMMA/TCGEN-oriented producer-consumer paths.

Actionable conclusion: use this FlashInfer SM100 FMHA path as the control-flow
and API/scheduler template, but replace the math/load internals with SM120
block-scaled `mma.sync` NVFP4 primitives and SM120-compatible shared/register
storage. Do not keep extending the pure hand-written scaffold.

## Dedicated Extension CUTLASS Runner Hook

## External SM120 CuTe DSL Reference Is Informative, Not The Active Path

2026-04-27T15:55:21-05:00

PR #2598 in `flashinfer-ai/flashinfer` adds an SM120 CuTe DSL attention
backend around a `FlashAttentionForwardSm120` example. It validates BF16/FP16
attention on SM120/SM121 and is useful as a reference for:

- SM120 attention shape choices and validation expectations.
- The fact that SM120 fused attention work is active upstream.
- A separate authoring model that avoids some C++ template friction.

It is not the active implementation path for this work. The active path remains
C++ CUTLASS/CuTe and the existing SM120 block-scaled FP4 CUTLASS primitives,
because the target is NVFP4 KV attention using block-scaled FP4 MMA, and the
existing two-stage C++ CUTLASS runner already proves the hardware FP4 path and
layout are correct and fast for the production cell.

The C++ collective smoke currently still faults with an illegal memory access.
The validated runner hook remains clean, so the fault is in the manual embedded
collective launch boundary, not in the SM120 FP4 runner or tensor layouts.

Fetched and inspected the actual reference sources:

- PR #2598 local worktree:
  `/home/josh/tdm/infer/worktrees/flashinfer-pr2598-cute-dsl-sm120`
- FA4 reference tree:
  `/home/josh/tdm/infer/worktrees/flash-attention-fa4-ref`
- Key files:
  - `flashinfer/cute_dsl_attention.py`
  - `flash_attn/cute/flash_fwd_sm120.py`
  - `flash_attn/cute/flash_fwd.py`
  - `flash_attn/cute/interface.py`

Findings:

- FlashInfer PR #2598 is a thin optional wrapper. It discovers and loads a
  `FlashAttentionForwardSm120` Python/CuTe DSL class, compiles it with
  `cute.compile`, and integrates it into `BatchPrefill`.
- The actual `FlashAttentionForwardSm120` implementation in FA4 subclasses
  `FlashAttentionForwardSm80`. It does not implement a new SM120 mainloop.
- `FlashAttentionForwardSm120.arch = 80` intentionally keeps the SM80 cp.async
  path and SM80-era `mma.sync.aligned.m16n8k16` tensor-core instructions.
- The SM120 subclass only changes the `can_implement` SMEM-capacity check to
  SM120's 99 KiB limit.
- It supports BF16/FP16 only, not NVFP4/block-scaled MMA.
- The FA4 SM120 default tile policy is:
  - `D <= 64`: `tile_m=128`, `tile_n=128`
  - `D > 64`: `tile_m=128`, `tile_n=64`
- For our production D=512 target, the FA4 SM120 path is not directly usable as
  a ceiling path because it is BF16/FP16 and uses a smaller N tile to fit SMEM.

How this applies to the active C++ path:

- Useful reference: cp.async Q/K/V staging, reverse-N traversal, online softmax,
  P materialization in registers, PV using reshaped `rP`, and final output
  epilogue.
- Not useful as a direct implementation target: it does not use SM120
  block-scaled FP4 MMA, CUTLASS SM120 NVFP4 scale layouts, or the fast
  two-stage FP4 runner path that already reaches the target hardware throughput.
- Active implementation remains C++ CUTLASS/CuTe with SM120 block-scaled FP4 MMA.

Added a local BF16 SM120 runner instantiation TU:

```text
benchmarks/sm120_nvfp4_cutlass_runner_bf16_inst.cu
```

It instantiates the three SM120 NVFP4 runner tile shapes used by
`CutlassFp4GemmRunner<__nv_bfloat16, W4A4_NVFP4_NVFP4>`:

```text
128x128x128
128x128x256
256x128x128
```

The dedicated fused-attention extension now exports:

```text
cutlass_runner_fp4_gemm(...)
```

Important layout finding:

- The native runner/module expects B as `[N, K/2]` and B scales as
  `[N, K/16]`.
- The Python `gemm_base.py` runner wrapper expects the caller's `b` argument as
  a non-contiguous transpose view, then calls `b.T` before entering the native
  module. Making that transpose contiguous changes the storage convention and
  produces wrong output.

Validation command:

```bash
CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/usr/local/cuda-13.2 TORCH_CUDA_ARCH_LIST=12.0f \
PYTHONPATH=/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv \
/home/josh/tdm/infer/current/.venv/bin/python \
  benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py \
  --device 0 --runner-check --bench --warmup 3 --repeat 10
```

Result:

```text
qk_runner_vs_official_mean_abs: 0.0
qk_runner_vs_official_max_abs:  0.0
qk_runner_vs_official_cosine:   0.99999994

bench_cutlass_runner_qk_128x32768 min_ms: 0.022368
```

The runner hook also matches `flashinfer.gemm.mm_fp4` exactly. Against a BF16
reference for the same quantized operands, the cosine is about 0.9953; the
remaining error is quantization error, not runner/layout error.

PV-side runner validation:

```text
pv_runner_vs_official_mean_abs: 0.0
pv_runner_vs_official_max_abs:  0.0
pv_runner_vs_official_cosine:   1.0
pv_runner_vs_exact_mean_abs:    0.000161
pv_runner_vs_exact_max_abs:     0.000921
pv_runner_vs_exact_cosine:      0.990829

bench_cutlass_runner_pv_128x512_k32768 min_ms: 0.131008
```

The old atom-debug PV path for the same 128-row slice is much slower:

```text
bench_pv_block min_ms: 0.462304
```

Conclusion: the dedicated extension now has a verified way to call both fast
SM120 CUTLASS FP4 GEMM phases. The atom-debug kernels remain correctness/layout
probes only and must not be used for performance conclusions.

Full-shape runner reproduction of the two-stage ceiling:

```text
full_runner_qk_shape: (4096, 32768)
full_runner_pv_shape: (4096, 512)

bench_cutlass_runner_qk_4096x32768 min_ms:      0.215424
bench_cutlass_runner_pv_4096x512_k32768 min_ms: 0.156640
```

This matches the earlier two-stage CUTLASS baseline:

```text
qk_fp4 min_ms: 0.213440
pv_fp4_after_fused_p min_ms: 0.155616
```

The dedicated extension is now anchored to the same fast SM120 block-scaled
CUTLASS path as the external two-stage benchmark. The next kernel work should
fuse around this primitive, not around the atom-debug or pure hand-written
paths.

## Atom-Level FMHA Pivot

The attempted `CollectiveMainloop` composition path is now rejected as the
active implementation path.

What failed:

- A standalone kernel that passed only `CollectiveMainloop::Params` into a
  custom debug launch faulted at descriptor prefetch.
- Matching the SM120 cooperative GEMM block size corrected one issue:
  `CollectiveMainloop::ThreadCount == 256`, but the full GEMM kernel launches
  384 threads: one producer warpgroup plus two consumer warpgroups.
- A corrected pipeline-init-only gate passes with the 384-thread role split.
- Descriptor prefetch still faults even when params are produced through the
  full `GemmKernel::to_underlying_arguments()` path and the debug kernel takes a
  full `GemmKernel::Params`.

Conclusion: the collective layer is the wrong composition boundary for fused
attention. The fast SM120 FP4 GEMM collective is designed to live under
`kernel::GemmUniversal` with its scheduler, epilogue, full shared-storage
layout, and launch contract. Trying to peel out and compose only
`CollectiveMainloop` is framework friction, not progress toward the fused
attention kernel.

Reference-source findings:

- CUTLASS Example 88 (`3rdparty/cutlass/examples/88_hopper_fmha`) is the right
  conceptual architecture: one FMHA-specific mainloop calls QK MMA, runs online
  softmax inline, then calls PV MMA. The relevant code is in
  `collective/fmha_collective_tma.hpp` around the `gemm_zero_acc` /
  `softmax.step_*` / `cute::gemm` sequence.
- CUTLASS Example 77 (`3rdparty/cutlass/examples/77_blackwell_fmha`) is the
  Blackwell form of the same idea. It reuses CUTLASS builder selection logic
  but recombines atom-level primitives into an FMHA-specific kernel, with
  separate Q/KV, MMA/softmax/correction/epilogue pipelines.
- Neither example fuses two full CUTLASS GEMMs. Both write one custom mainloop
  and call atom-level primitives inside that mainloop.

New active implementation boundary:

```text
device::Gemm / GemmUniversalAdapter       no
kernel::GemmUniversal                     no
collective::CollectiveMainloop/Epilogue   no
cute::TiledMMA + Copy_Atom + fragments    yes
```

The next prototype should be a fixed-shape SM120 atom-level FMHA kernel:

- Use the existing characterized SM120 NVFP4 MMA atom:
  `SM120_16x8x64_TN_VS`.
- Preserve the existing QK/PV atom correctness helpers as operand-layout
  evidence.
- Add one fused CTA/block prototype that computes a small fixed Q tile end to
  end: QK, online softmax, P quantization/register staging, PV, output.
- Once correct, replace scalar global gather in the atom helpers with
  coalesced copy/smem staging modeled after Example 88/77. Performance should
  be judged only after the copy path is no longer scalar-gather.

## Atom-Level Fused Scaffold Plan

The first fused scaffold should not recompute QK once per output tile. It should
use one CTA for the fixed 16-query-row tile and all 512 output columns:

```text
CTA = 8 warps for the first launchable correctness scaffold
  warp 0     QK producer + online softmax/P-quant producer
  warps 0-7  PV consumers, one warp per 16 output columns
```

For each 64-token KV chunk:

1. Warp 0 computes four `16x16x512` QK atom tiles and writes a `16x64` score
   tile to shared memory.
2. Warp 0 updates online row max/sum state, rescales the PV accumulators via a
   per-row alpha, and quantizes the current `16x64` unnormalized P tile into
   shared NVFP4 codes/scales.
3. All 32 warps consume the same shared P tile for PV. Each warp owns a
   distinct 16-column output tile, so P is reused across all 512 output columns
   without HBM round-tripping.
4. After the final chunk, each warp divides its PV accumulator by
   `PROB_GLOBAL_SCALE * row_sum` and writes its output columns.

This scaffold still uses scalar global gathers for Q/K/V and a simple shared
P sidecar. That is intentional for the first correctness step. The performance
path after correctness is replacing those gathers with coalesced
`Copy_Atom`/cp.async or TMA-style staging and CUTLASS-compatible smem layouts.

Important production constraint: Examples 77/88 are not single-producer-warp
kernels. They use balanced cooperative warpgroups, either with multiple
producer and multiple consumer warps or fully cooperative warpgroup structure.
The 1-producer/7-consumer split is therefore not a performance architecture. It
is only a launchable correctness bridge after the 32-warp version exceeded
per-block resources. The next performance prototype must repartition the CTA so
QK, softmax/P quantization, and PV are cooperatively executed by balanced
warpgroups instead of serializing all QK/softmax work through one warp.

First scaffold correctness checkpoint:

```text
32-warp CTA attempt: launch failed, too many resources requested
8-warp CTA attempt:  launch succeeds

fused_atom_finite:                  True
fused_atom_vs_quant_p_mean_abs:     1.372e-4
fused_atom_vs_quant_p_max_abs:      7.600e-4
fused_atom_vs_exact_p_mean_abs:     1.288e-4
fused_atom_vs_exact_p_max_abs:      6.296e-4
fused_atom_vs_exact_p_cosine:       0.99497
```

This validates the end-to-end single-kernel math: QK atom, online max/sum
state, inline P quantization, PV atom, final `1/(PROB_GLOBAL_SCALE * row_sum)`
normalization. The result is close to the staged quantized-P and exact-P
references. This does not validate performance; the known bottleneck is the
single producer warp.

First timing checkpoint:

```text
4 CTAs x 8 warps, each CTA recomputes QK/softmax for a 128-column output slice:
  bench_fused_atom_attention_block: 11.08 ms

1 CTA x 8 warps, each warp carries four PV output tiles so QK/softmax is
computed once:
  bench_fused_atom_attention_block: 12.28 ms
```

The second version removes redundant QK/softmax but increases per-thread
register pressure by carrying four PV accumulator tiles per warp; it regresses.
This confirms the obvious architecture problem: single-warp QK/P production and
fat per-warp PV state are not viable. Next scaffold moves toward the
Example 77/88 balance by splitting QK production over four producer warps
(one warp per 16-column score subtile), parallelizing online softmax row updates
over four warps, and parallelizing P quantization over the same four warps
before all eight warps run PV.

Balanced-producer scaffold timing:

```text
1 CTA x 8 warps, four producer warps for QK/row softmax/P quantization,
all eight warps for PV:
  fused_atom correctness: unchanged
  bench_fused_atom_attention_block: 8.04 ms
```

This is a real improvement over 12.28 ms, but it exposes the next structural
problem: the scaffold has essentially no CTA-level parallelism. One CTA owns a
16-query-row tile and loops serially over all 32K KV tokens. The fast CUTLASS
runner gets its throughput from many CTA tiles over the GEMM surface. A fused
attention kernel must also expose KV and/or Q-block parallelism; otherwise each
CTA is a long serial loop and the GPU is underfilled.

Next architecture step: introduce split-KV partial attention tiles. Each CTA
computes a 16-row x 512-output partial over a KV chunk, writes `(m, l, o)` state,
and a second reducer combines chunks with the standard online-softmax merge:

```text
m = max(m0, m1)
l = l0 * exp(m0 - m) + l1 * exp(m1 - m)
o = o0 * exp(m0 - m) + o1 * exp(m1 - m)
```

This is the minimum path to recover CTA-level parallelism while preserving the
atom-level block-scaled FP4 QK/PV work.

Split-KV scaffold checkpoint:

```text
split count: 32
KV chunk:    1024 tokens
partial:     one 8-warp CTA computes 16 query rows x 512 output columns for one
             1024-token KV chunk
reducer:     merges 32 partial `(m,l,o)` states

split_fused_atom_finite:              True
split_fused_atom_vs_quant_p_mean_abs: 1.317e-4
split_fused_atom_vs_quant_p_max_abs:  5.844e-4
split_fused_atom_vs_exact_p_mean_abs: 1.291e-4
split_fused_atom_vs_exact_p_max_abs:  6.833e-4
split_fused_atom_vs_exact_p_cosine:   0.99492

bench_fused_atom_attention_split_block: 0.274 ms
```

This is a 29x speedup over the single-CTA serial-KV balanced-producer scaffold
(`8.04 ms`) and proves that the missing CTA-level parallelism was structural.
The result is still only for one 16-row query tile. The next comparison must be
against the CUTLASS runner at a comparable row count, starting with 128 query
rows. That requires switching the scaffold from fixed `q_token_base/head` row
addressing to flat row-tile addressing and launching multiple Q tiles.

128-row split-KV checkpoint:

```text
Q rows:       first 128 flat Q rows
Q tiles:      8
KV splits:    32
partial CTAs: 256

split_128_fused_atom_finite:            True
split_128_vs_exact_p_mean_abs:          1.289e-4
split_128_vs_exact_p_max_abs:           6.833e-4
split_128_vs_exact_p_cosine:            0.99503

bench_fused_atom_attention_split_128rows: 0.606 ms
```

Correctness is stable. Performance is not yet competitive at 128 rows because
the scaffold writes and rereads a large partial-output tensor:

```text
8 q_tiles * 32 splits * 16 rows * 512 cols * sizeof(float)
  = 16 MiB of partial O traffic
```

This is useful as an architectural diagnostic, not a ship path. The split-KV
parallelism fixed the one-CTA serial loop, but the reducer/materialization cost
is now a first-order term. Next lever: sweep split granularity to find the
parallelism-vs-partial-HBM knee, then replace the HBM reducer with an on-chip or
cluster-style reduction if the best split still needs reduction.

Split-count sweep:

```text
bench_fused_atom_attention_split_128rows_s4:  1.933 ms
bench_fused_atom_attention_split_128rows_s8:  0.972 ms
bench_fused_atom_attention_split_128rows_s16: 0.491 ms
bench_fused_atom_attention_split_128rows_s32: 0.485 ms
```

The knee is 16-32 splits. Fewer splits underfill the GPU; the reducer/HBM
partial-output traffic is not yet the dominant limiter for this 128-row
prototype.

Attempted 16 warps/CTA to reduce each warp's PV accumulator ownership from four
16-column output tiles to two. The kernel failed at launch with
`cudaErrorLaunchOutOfResources`. The 8-warp CTA is the current launchable shape.
Register pressure must be reduced structurally before adding more consumer
warps.

Column-split probe:

```text
unsplit 8-warp CTA, each warp carries four PV output tiles:
  bench_fused_atom_attention_split_128rows_s32:          0.485 ms

column-split, each CTA handles only 128 output columns and each warp carries one
PV tile, but QK/softmax/P is recomputed for each of four column groups:
  bench_fused_atom_attention_split_128rows_colsplit_s32: 0.957 ms
  bench_fused_atom_attention_split_128rows_colsplit_s16: 0.958 ms
```

Conclusion: recomputing QK/softmax/P four times is much worse than carrying four
PV accumulator tiles per warp. Keep the unsplit 8-warp CTA shape for now. The
next bottleneck is not output-column ownership; it is the scalar global gather /
staging path inside the atom helpers.

## Pre-Test Worklog Review Checkpoint

2026-04-27

Reviewed the prior NVFP4 worklogs before running any additional tests:

- `KV_CACHE_NVFP4_WORKLOG.md`
- `FA2_NVFP4_WORKLOG.md`
- `KV_CACHE_NVFP4_TUNING.md`
- `SM120_NVFP4_REFERENCE_ATTENTION_WORKLOG.md`

Constraints from the logs that should govern the next implementation step:

- Do not retry K/V load folding. `attention_k_eq_v=true` is projection-level
  weight sharing only; K and V are distinct after norm/RoPE at the kernel input.
- Do not keep tuning D128 as a target. Gemma4 31B production shapes are D256
  sliding/group2/kv1024 and D512 global/group8/long-KV, with D512 prioritized.
- Do not compose two full CUTLASS GEMMs or two `collective::` objects. The
  earlier embedded collective smoke faulted at the launch/protocol boundary.
  The correct abstraction level is an FMHA mainloop using CuTe/CUTLASS atoms:
  copy atoms, tiled MMA atoms, softmax state, P quantization, and PV MMA inside
  one kernel.
- Do not keep extending the scalar hand-written scaffold as the final path. It
  is a correctness/layout scaffold only. It is still far from the CUTLASS
  two-stage FP4 ceiling because it lacks the production CUTLASS/CuTe copy/smem
  layouts, scheduler behavior, and mainloop cadence.
- Do not pursue column splitting in the current scaffold. It is correct but
  slower because it recomputes QK/softmax/P for each output-column group.
- Do not use blunt `maxrregcount` as a fix. Prior FMHA-v2 caps regressed the
  already-spilling kernel, and the 128-row scaffold cap probe regressed the
  important s32 shape. Register work has to remove live state structurally.
- Do not repeat buffer-count-only stage-depth probes. Prior stage-3 attempts
  increased shared/register pressure without changing the phase schedule enough
  to expose real overlap.
- Do not retry narrow Shape A sidecar tweaks as the main path. CUDA 13.2, linear
  V-scale layout, 16-byte destination copies, L2 hints, and k64B swizzle probes
  left Shape A below FP8 FA2. Shape A requires a structural layout/consumer path
  if revisited.

Current best local scaffold facts:

```text
best launchable 128-row split scaffold, s32: ~0.485 ms
colsplit s32:                              ~0.957 ms
16-warp CTA:                               launch out-of-resources
CUTLASS runner hook QK 128x32768:          ~0.023 ms
CUTLASS runner hook PV 128x512,k32768:     ~0.131 ms
```

Current diagnostic direction:

```text
The gap is not solved by more split-count or column-ownership tuning.
Next implementation work should replace the scalar gather/staging helpers with
CUTLASS/CuTe-style copy atoms and SM120 NVFP4 shared-memory/scale layouts, then
embed QK MMA, online softmax/P quantization, and PV MMA in one FMHA mainloop.
```

## SM120 CUTLASS Source Review For Next Kernel

2026-04-27

Relevant source files reviewed:

- `include/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h`
- `include/flashinfer/gemm/fp4_gemm_template_sm120.h`
- `3rdparty/cutlass/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu`
- `3rdparty/cutlass/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl`
- `3rdparty/cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp`
- `3rdparty/cutlass/include/cute/arch/mma_sm120.hpp`
- `3rdparty/cutlass/include/cute/atom/mma_traits_sm120.hpp`

Source facts:

- FlashInfer's SM120 FP4 runner dispatches only `128x128x128`,
  `128x128x256`, and `256x128x128` CTA shapes, with 1x1x1 cluster shape and
  both persistent and StreamK schedulers.
- The fast runner uses `cutlass::arch::OpClassBlockScaledTensorOp` with
  `cutlass::nv_float4_t<cutlass::float_e2m1_t>` operands and E4M3 scale
  factors.
- The CUTLASS blockscaled builder selects a CuTe `TiledMma` from
  `rr_blockscaled_op_selector_sm120`, with atom layout `Shape<_4,_2,_1>` for
  the cooperative schedule and a tile like `Tile<128,128,64>` at the MMA layer.
- The blockscaled collective's source-level structure is the useful primitive:
  TMA/cp.async producer staging for A/B and scale tensors, CUTLASS/CuTe smem
  layouts, smem-to-register copy atoms, then `cute::gemm` over zipped
  `(data, scale)` fragments.
- The current scalar scaffold uses the same underlying SM120
  `m16n8k64` NVFP4 MMA instruction, but it hand-gathers register fragments
  from global memory and manually packs scale registers. That bypasses the
  CUTLASS smem layouts and copy atoms that make the runner fast.

Implementation boundary:

```text
Use CUTLASS/CuTe atom-level pieces:
  - TiledMma
  - SmemLayoutA/B and SmemLayoutSFA/SFB
  - SmemCopyAtomA/B and SmemCopyAtomSFA/SFB
  - partition_fragment_A/B/SFA/SFB
  - cute::gemm(zipped data+scale fragments)

Do not use:
  - full GemmUniversalAdapter composition
  - full CollectiveMainloop launch/protocol embedding
  - scalar global-gather fragment helpers as the performance path
```

Next kernel prototype should therefore be a small fixed-shape atom-mainloop
prototype that stages one `128x128x256` QK tile into the CUTLASS SM120 smem
layouts and uses the CUTLASS smem-to-register copy atoms before MMA. Once QK is
correct and close to the runner for one tile, extend the same structure to
online softmax/P quantization and PV.

## CUTLASS Smem/Copy-Atom QK Tile Prototype

2026-04-27

Implemented the first bridge away from scalar global-gather fragments:

```text
benchmarks/sm120_nvfp4_cutlass_fused_attention.cu
  qk_cutlass_smem_atom_tile_kernel(...)

benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py
  --smem-atom-check
```

Scope:

- Fixed Shape B debug tile only: `M=128, N=128, K=512`.
- Inputs are still the row-major debug NVFP4 tensors so the existing PyTorch
  reference path can validate the output.
- The kernel stages one `128x128x256` K-slice at a time into
  `CutlassCollectiveMainloop::SmemLayoutA/B` and `SmemLayoutSFA/SFB`.
- It consumes those tiles with `SmemCopyAtomA/B/SFA/SFB`,
  `partition_fragment_A/B/SFA/SFB`, and `cute::gemm(zipped data+scale)`.

This is not the final performance path yet. It deliberately avoids the rejected
`CollectiveMainloop` launch/protocol boundary while reusing the exact CUTLASS
smem layouts and register-copy atoms needed by the real FMHA mainloop. The next
validation step is compile/correctness for this QK tile before extending the
same structure to online softmax/P quantization and PV.

First compile/correctness result:

```text
CUDA_HOME=/usr/local/cuda-13.2 TORCH_CUDA_ARCH_LIST=12.0f \
  python benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py \
  --device 0 --smem-atom-check-only

qk_smem_atom_finite:   True
qk_smem_atom_mean_abs: 0.85747
qk_smem_atom_max_abs:  4.71082
qk_smem_atom_cosine:   0.65256
```

This proves the CUTLASS smem/copy-atom prototype compiles and executes on
SM120, but it is not yet a correct QK tile. The next debugging target is operand
and scale mapping into `SmemLayoutA/B` and `SmemLayoutSFA/SFB`; the symptom is
layout/scale mismatch, not arithmetic non-finiteness.

## CUTLASS Smem/Copy-Atom QK Tile Current Debug State

2026-04-27

Reviewed the prior worklogs before running further tests. The current
implementation state is the fixed-shape SM120 CuTe atom bridge for a single QK
tile:

```text
M = 128
N = 128
K = 512
dtype = NVFP4 x NVFP4 -> FP32 accumulator / BF16 comparison path
```

Current facts established by the latest bring-up:

- The CUTLASS smem/copy-atom prototype compiles and executes on SM120 with
  CUDA 13.2 and `TORCH_CUDA_ARCH_LIST=12.0f`.
- A direct all-ones fast CUTLASS runner check returns exactly `512`, so the
  SM120 block-scaled MMA numeric convention is correct.
- Raw physical-byte smem fill with FP4 code `0x22` and E4M3 scale byte `0x38`
  returns exactly `512`. This proves the copy atoms and MMA loop can compute the
  expected result when the physical shared-memory bytes match what the atom
  expects.
- Normal packed physical staging now also passes the all-ones invariant:
  output is finite and exactly `512` across the tile.
- Constant-code probes for all E2M1 codes match the expected squared dot
  products. This proves the E2M1 code-value interpretation is correct.
- Random input remains incorrect even when all scales are forced to one. The
  failure therefore is not primarily scale-sidecar magnitude handling.
- Feeding official `flashinfer.nvfp4_quantize(..., SfLayout.layout_128x4)`
  tensors into the smem atom path still fails versus the fast runner. The issue
  is not only the local row-major debug quantizer.
- Reversing both A and B nibbles leaves the random result effectively
  unchanged, while reversing only A or only B destroys cosine. That indicates a
  paired/permuted data-layout issue rather than simple low/high nibble sign or
  value decoding.
- The rejected debug store paths using stripped swizzle/direct subbyte stores
  are worse than the current physical pair staging. Do not replace the current
  path with those modes.

Current representative random failure:

```text
finite:          true
mean_abs:        ~1.05
max_abs:         ~7.94
cosine:          ~0.46
transpose_cos:   ~0.01
norm_ratio:      ~1.06
```

Conclusion:

```text
The current blocker is the A/B FP4 data permutation into the SM120
block-scaled MMA shared-memory layout consumed by the CUTLASS smem copy atoms.
Constants and all-ones are insufficient because they are invariant to many
within-tile permutations. Random data exposes that the physical byte placement
still does not match the atom's logical fragment mapping.
```

Next diagnostic direction:

- Derive the exact logical-to-physical mapping from the CuTe copy atom and
  `TiledMma` fragment coordinates instead of guessing from `layout(row, col)`.
- Prefer a diagnostic kernel that exports the A/B register fragments produced
  by `cute::copy` for patterned inputs, then compare those fragments to
  `thread_mma.partition_fragment_A/B` logical coordinates.
- Once random QK tile correctness matches the fast CUTLASS runner, extend the
  same atom-level structure to online softmax, P quantization, and PV.

## Source-Convention Comparison: SM120 FP4 Smem Is Unpacked U8

2026-04-27

Compared the prototype against the closest working source references before
running another test:

- `include/flashinfer/attention/blackwell/fmha_cutlass_sm100.cuh`
- `include/flashinfer/attention/blackwell/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`
- `include/flashinfer/attention/blackwell/collective/sm100_fmha_load_tma_warpspecialized.hpp`
- `3rdparty/cutlass/examples/88_hopper_fmha/collective/fmha_collective_tma.hpp`
- `3rdparty/cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp`
- `3rdparty/cutlass/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl`
- `3rdparty/cutlass/include/cutlass/gemm/collective/builders/sm120_common.inl`
- `3rdparty/cutlass/include/cute/arch/copy_sm100.hpp`

Important convention first suspected:

```text
SM120 FP4/F6/F8 collective:
  SmemAllocTypeA/B = uint8_t
  TMA internal element for FP4 = float_e2m1_unpacksmem_t
  smem copy atom = SM100_SU4_DU8x16_x4_LDSM_N
```

Initial interpretation, now rejected:

```text
one logical FP4 value per uint8_t shared-memory slot
```

Probe result:

```text
normal packed-pair staging:
  all-ones -> 512

one-code-per-byte staging:
  all-ones -> 256
  random cosine -> ~0.004
```

Conclusion:

```text
The one-code-per-byte interpretation is wrong for the current SM120 SU4 LDSM
source path. The instruction/source layout still needs both FP4 nibbles present
in the packed 64-bit source payload. The prototype was reverted to the prior
packed-pair staging because it preserves the all-ones invariant.
```

The useful source-comparison finding is narrower:

```text
Do not infer the producer write mapping from `SmemLayoutA/B(row,k,stage)` alone.
The working collective writes through TMA `partition_D(sA/sB)` and reads through
`as_position_independent_swizzle_tensor(sA/sB)` plus SU4 LDSM. The next fix
should derive the manual producer mapping from the same TMA destination view or
export copy-atom register fragments for patterned inputs.
```

Follow-up metadata:

```text
SmemLayoutA rows 0..7, k 0..31:
  row0: 0..31
  row1: 288..319
  row2: 576..607
  row3: 864..895
  row4: 1024..1055
  row5: 1312..1343
  row6: 1600..1631
  row7: 1888..1919

SmemLayoutB rows 0..7, k 0..31:
  identical to A
```

This rules out the specific "adjacent K pair crosses unrelated bytes" concern
for the first atom rows: adjacent K values do map to adjacent offsets, so the
manual two-nibble byte store is locally consistent. The remaining mismatch is
deeper than byte ownership in the first visible rows.

Harness note:

```text
Added --smem-atom-unit-scales so source q/k scales and the Python reference can
both be forced to E4M3 1.0. This is the clean data-only diagnostic; the older
--smem-atom-scale-mode=1 only changed smem scale staging inside the kernel and
therefore compared against the wrong reference scale.
```

## Mixed E2M1 Code Sweep: Encoding Ruled Out

2026-04-27

Added two diagnostics before the next layout fix:

- `cutlass_e2m1_code_values` in `cutlass_sm120_blockscaled_collective_metadata()`.
- `--smem-atom-code-sweep` in
  `bench_sm120_nvfp4_cutlass_fused_attention.py`.

The sweep forces unit scales (`0x38`) and fills all A values with one E2M1 code
and all B values with an independently selected E2M1 code. It runs all 16x16
code pairs and checks the QK tile against:

```text
expected = 512 * value(A_code) * value(B_code)
```

Result:

```text
CUTLASS host E2M1 code values:
  [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

mixed-code pairs:      256
mismatch count:        0
max_abs:               0.0
```

Conclusion:

```text
The random QK smem-atom failure is not E2M1 sign/value encoding, nibble order,
or scale magnitude. Constants, including mixed-sign constants, all pass exactly.
The remaining bug is a non-constant logical-to-physical placement mismatch:
some combination of row, K-position, or B operand coordinate mapping is wrong
when values vary within the tile.
```

Next diagnostic:

```text
Use non-constant structured patterns that isolate row and K coordinates:
  - A varies by row, B is all ones.
  - A varies by K, B is all ones.
  - A is all ones, B varies by row/N.
  - A is all ones, B varies by K.

These patterns should identify whether the mismatch is row placement, K
placement, B N-coordinate placement, or copy-atom fragment handoff.
```

## Manual Smem Producer vs CUTLASS Runner

2026-04-27

The unit-scale random-code case was compared against the known-good
`CutlassFp4GemmRunner` path using the same packed data and `0x38` scales:

```text
manual smem atom vs Python:
  mean_abs: 108.78
  max_abs:  1102.5
  cosine:   0.5006

CUTLASS runner vs Python:
  mean_abs: 0.2056
  max_abs:  2.0
  cosine:   0.9999985

manual smem atom vs runner:
  mean_abs: 108.88
  max_abs:  1101.75
  cosine:   0.5006
```

Conclusion:

```text
The SM120 block-scaled MMA hardware path, the reference math, and the packed
NVFP4 code values are correct. The bug is specifically the manual shared-memory
producer used by `qk_cutlass_smem_atom_tile_kernel`.
```

Additional negative results:

```text
adjacent K split-byte pairs:
  A: 0
  B: 0

C fragment write coverage:
  writes:          16384
  missing slots:   0
  duplicate slots: 0
```

Structured probes:

```text
all 16x16 mixed-code constants: pass exactly
A varies by row, B all ones:    pass exactly
A all ones, B varies by N:      pass exactly
A varies by K, B all ones:      pass exactly
A all ones, B varies by K:      pass exactly
checker/full low-rank pattern:  pass exactly
uniform random codes:           fail, cosine ~0.50
```

Interpretation:

```text
The manual writer can satisfy low-rank and one-hot tests while still violating
the CUTLASS copy-atom producer/consumer contract for high-entropy data. Raw
`SmemLayoutA/B(row,k,stage)` indexing is not a sufficient authority for the
manual producer. The write-side mapping must be derived from the same
partitioned copy view used by the consumer (`partition_D` for TMA or the
matching copy-atom partition), not from layout offsets alone.
```

The existing `qk_cutlass_collective_tile_kernel` wrapper is not yet usable as a
canonical tile path: debug mode 3 (return before pipeline setup) works, but
debug mode 1 (enter pipeline setup, return before TMA) faults with an illegal
address. That wrapper needs separate cleanup; it does not change the conclusion
above because the official runner path is already correct.

## QK Smem Atom Producer Rewrite

2026-04-27

The raw shared-memory producer in `qk_cutlass_smem_atom_tile_kernel` was
rewritten to derive its write mapping from the same CuTe copy partition used by
the consumer:

```text
A data writes: SmemCopyAtomA thread slice -> partition_D(sA)
B data writes: SmemCopyAtomB thread slice -> partition_D(sB)
consumer:      partition_S(as_position_independent_swizzle_tensor(sA/sB))
```

This removes the producer's direct dependence on
`SmemLayoutA/B(row,k,stage)` offsets. That direct indexing was the bug: it did
not satisfy the copy-atom producer/consumer contract for high-entropy tiles,
even though constants and low-rank structured patterns passed.

Implementation details that matter:

```text
Identity coordinates must use the logical tile shape:
  A: (M, K, 1)
  B: (N, K, 1)

Using cute::shape(sA/sB) creates hierarchical coordinates from the physical
shared-memory layout, which are not the logical row/K coordinates needed for
loading packed NVFP4 source values.

The producer writes through CuTe's subbyte destination reference:
  tDst(i) = cute::uint4_t(code)

Manual byte/nibble writes are no longer used in this path. The earlier
byte-vs-nibble ambiguity was already resolved: pair-packed staging is correct,
but manually computing byte offsets still bypasses the copy partition and loses
part of the thread-slice mapping.
```

Verification:

```text
unit-scale random Q/K:
  qk_smem_atom_mean_abs: 0.0
  qk_smem_atom_max_abs:  0.0
  qk_smem_atom_cosine:   0.9999998807907104

normal-scale random Q/K:
  qk_smem_atom_mean_abs: 0.0
  qk_smem_atom_max_abs:  0.0
  qk_smem_atom_cosine:   0.9999998807907104
  qk_smem_atom_nonzero:  16384
```

Conclusion:

```text
The QK shared-memory atom bridge is now correct for high-entropy random data.
The confirmed convention is: data producers should write through the
partition_D destination view derived from the same TiledMMA/copy atom that the
consumer uses, not through raw layout offsets.
```

## Offset-Capable QK Block Primitive

2026-04-27

The corrected `partition_D` producer was factored into a reusable tile body
with explicit source and output offsets:

```text
q_row_base:  source Q row offset
kv_row_base: source K row offset
out_stride/out_row_base/out_col_base: destination placement
```

A new launcher, `qk_cutlass_smem_atom_block`, applies that body across the full
Shape B KV axis for the first 128 query rows:

```text
output shape: 128 x 32768
tile grid:    (32768 / 128) x (128 / 128)
tile shape:   128 x 128 x 512
```

Correctness against the PyTorch reference:

```text
qk_smem_atom_block_finite:     True
qk_smem_atom_block_mean_abs:   0.0
qk_smem_atom_block_max_abs:    0.0
qk_smem_atom_block_cosine:     0.9999999403953552
qk_smem_atom_block_norm_ratio: 1.0
```

Conclusion:

```text
The partition_D producer is now valid beyond the single debug tile. It handles
nonzero K/source/output offsets across the full 32k KV axis, which is the QK
producer primitive needed before replacing the old scalar-fragment QK path in
the fused softmax/PV scaffold.
```

Benchmark, same primitive:

```text
bench_qk_smem_atom_block_128x32768:
  min_ms:  0.18806399405002594
  mean_ms: 0.19006240069866182
  max_ms:  0.19491200149059296
```

## PV Smem Atom Tile Primitive

2026-04-27

The QK tile body was generalized to accept packed/scale strides and independent
source column bases. This lets the same CuTe `partition_D` producer feed the PV
MMA:

```text
A operand: P, row-major [128, 32768], source K offset = kv_base
B operand: V^T, row-major [512, 32768], source row offset = out_col_base,
           source K offset = kv_base
tile:      128 x 128 x 256
```

Correctness against a PyTorch dequantized reference for one PV tile:

```text
pv_smem_atom_tile_finite:     True
pv_smem_atom_tile_mean_abs:   0.0
pv_smem_atom_tile_max_abs:    0.0
pv_smem_atom_tile_cosine:     1.0
pv_smem_atom_tile_norm_ratio: 1.0
```

Benchmark:

```text
bench_pv_smem_atom_tile_128x128x256:
  min_ms:  0.05113599821925163
  mean_ms: 0.052057599648833275
  max_ms:  0.05526399984955788
```

Conclusion:

```text
Both QK and PV can now be driven through the same SM120 block-scaled CuTe
producer/consumer convention. The next fused-kernel step is no longer operand
layout discovery; it is mainloop construction: compute QK 128-column tiles,
run online softmax/P quantization, then reuse each P[128 x 256] tile across the
four PV output-column tiles without round-tripping P through global memory.
```

## First On-Chip Fused Scaffold

2026-04-27

Added `fused_cutlass_smem_one_kv_tile`, a single-CTA correctness scaffold for
one Shape B tile:

```text
Q rows:       128
KV tile:      256 tokens
output cols:  128
QK:           two 128-column QK tiles using the partition_D CuTe producer
softmax:      local over the 256-token tile
P:            unnormalized exp(score - row_m), quantized to NVFP4 in shared
PV:           P[128 x 256] x V[128 x 256]^T through the same block-scaled path
normalization: output divided by PROB_GLOBAL_SCALE * row_l
```

Important implementation constraint:

```text
TensorStorage plus a full 128x256 FP32 score tile does not fit in SM120 shared
memory. The scaffold intentionally recomputes the two QK halves for max, row_l,
and P quantization. That is not the final performance design; it is a correct
on-chip producer/softmax/PV scaffold that avoids HBM round-tripping P and proves
the partition_D QK/PV handoff works through a fused attention-shaped flow.
```

Correctness:

```text
fused_smem_one_tile_finite:              True
fused_smem_one_tile_vs_quant_mean_abs:   3.0845566101334043e-09
fused_smem_one_tile_vs_quant_max_abs:    3.3527612686157227e-08
fused_smem_one_tile_vs_quant_cosine:     1.0
fused_smem_one_tile_vs_exact_mean_abs:   0.0014373366720974445
fused_smem_one_tile_vs_exact_max_abs:    0.006850942969322205
```

Benchmark:

```text
bench_fused_smem_one_kv_tile_128x128x256:
  min_ms:  0.5990399718284607
  mean_ms: 0.6029503971338273
  max_ms:  0.6063039898872375
```

Conclusion:

```text
The fused attention-shaped dataflow is now correct for a single 256-token KV
tile. Its timing is expectedly poor because QK is recomputed six times to stay
within shared memory and keep the implementation simple. The next performance
step is the real online mainloop: keep only row_m/row_l and the current P tile,
compute each QK tile once for the active phase where possible, rescale PV
accumulators as row_m changes, and reuse P across the four V output-column
tiles.
```

## Full-KV Online Column Scaffold

2026-04-27

Added `fused_cutlass_smem_online_col_tile`, the first full-KV online-softmax
scaffold using the corrected CuTe block-scaled QK/PV tile primitive:

```text
Q rows:        128
output cols:   128
KV tile:       256 tokens
KV tiles:      runtime argument, tested at 4 and 128
full tested KV: 32768 tokens
```

Per KV tile, the kernel:

```text
1. Computes tile row max over two 128-column QK halves.
2. Updates global row_m and row_alpha = exp(old_m - new_m).
3. Scales the existing PV accumulator by row_alpha.
4. Recomputes QK halves, accumulates tile_l against new row_m, and quantizes
   unnormalized P into shared NVFP4.
5. Runs PV for the current P[128 x 256] tile and one V output-column tile.
6. Adds the PV tile into the rescaled accumulator.
7. Normalizes by PROB_GLOBAL_SCALE * row_l after all KV tiles.
```

Correctness, 4 KV tiles / 1024 tokens:

```text
fused_smem_online_finite:             True
fused_smem_online_vs_quant_mean_abs:  1.1580540970612674e-09
fused_smem_online_vs_quant_max_abs:   9.313225746154785e-09
fused_smem_online_vs_quant_cosine:    0.9999999403953552
fused_smem_online_vs_exact_mean_abs:  0.0007142516551539302
fused_smem_online_vs_exact_max_abs:   0.003437190316617489
```

Correctness, 128 KV tiles / 32768 tokens:

```text
fused_smem_online_finite:             True
fused_smem_online_vs_quant_mean_abs:  3.168883766502262e-10
fused_smem_online_vs_quant_max_abs:   2.7939677238464355e-09
fused_smem_online_vs_quant_cosine:    1.0
fused_smem_online_vs_exact_mean_abs:  0.000130419633933343
fused_smem_online_vs_exact_max_abs:   0.0006509469822049141
```

Benchmark, 128 KV tiles / 32768 tokens, one 128-column output tile:

```text
bench_fused_smem_online_col_tile:
  min_ms:  64.857666015625
  mean_ms: 64.86561889648438
  max_ms:  64.87830352783203
```

Conclusion:

```text
The full-KV online recurrence is now correct for Shape B's 32k global-attention
case for one output-column tile. This is still a correctness scaffold, not a
performance candidate: it recomputes each QK half multiple times, serializes the
entire 32k KV scan in one CTA, and only produces one of four output-column
tiles. The next move is reducing recomputation and increasing CTA-level
parallelism while preserving the now-validated online recurrence.
```

## Full-Width Online Scaffold With P Reuse

2026-04-27

Added `fused_cutlass_smem_online_full_width`, which extends the online
recurrence to the full D=512 output width for one 128-row query tile:

```text
Q rows:       128
output cols:  512
KV tile:      256 tokens
KV tiles:     runtime argument, tested at 4 and 128
```

Per KV tile, the kernel computes QK/online state once, quantizes a single
P[128 x 256] tile, then reuses that P tile across the four V output-column
tiles:

```text
for out_group in 0..3:
  PV: P[128 x 256] x V[out_group*128:(out_group+1)*128, kv:kv+256]^T
  accumulate into the matching 128-column slice
```

Correctness, 4 KV tiles / 1024 tokens:

```text
fused_smem_online_full_width_finite:             True
fused_smem_online_full_width_vs_quant_mean_abs:  1.2128164028624155e-09
fused_smem_online_full_width_vs_quant_max_abs:   1.1175870895385742e-08
fused_smem_online_full_width_vs_quant_cosine:    1.0
fused_smem_online_full_width_vs_exact_mean_abs:  0.0007274801027961075
fused_smem_online_full_width_vs_exact_max_abs:   0.0039443704299628735
```

Correctness, 128 KV tiles / 32768 tokens:

```text
fused_smem_online_full_width_finite:             True
fused_smem_online_full_width_vs_quant_mean_abs:  3.064804798835752e-10
fused_smem_online_full_width_vs_quant_max_abs:   3.026798367500305e-09
fused_smem_online_full_width_vs_quant_cosine:    1.0000001192092896
fused_smem_online_full_width_vs_exact_mean_abs:  0.0001281225122511387
fused_smem_online_full_width_vs_exact_max_abs:   0.0006941158790141344
```

Benchmark, 128 KV tiles / 32768 tokens, full D=512 output:

```text
bench_fused_smem_online_full_width:
  min_ms:  94.12786865234375
  mean_ms: 94.13963928222657
  max_ms:  94.16028594970703
```

Conclusion:

```text
P-tile reuse across all four V output-column groups is now correct. Compared
with running the one-column scaffold four times, this cuts the scaffold from
roughly 4 * 64.9 ms to 94.1 ms by sharing QK/softmax/P quantization across the
full D=512 output. The kernel is still far from the target because each QK half
is recomputed for tile max, tile sum, and P quantization, and the entire
128-row x 32k x 512 tile is serialized through one CTA. The next optimization is
to remove QK recomputation within a KV tile.
```

## Full-Width QK Recompute Reduction

2026-04-27

The full-width scaffold was changed to reuse the second QK half that remains in
the score buffer after the row-max pass:

```text
before:
  half0 max, half1 max, half0 P, half1 P

after:
  half0 max, half1 max, half1 P, half0 P
```

This removes one QK tile call per 256-token KV tile.

Correctness remained unchanged:

```text
4 KV tiles:
  mean_abs vs online quant ref: 1.2128164028624155e-09
  max_abs  vs online quant ref: 1.1175870895385742e-08
  cosine:                     1.0

128 KV tiles:
  mean_abs vs online quant ref: 3.064804798835752e-10
  max_abs  vs online quant ref: 3.026798367500305e-09
  cosine:                     1.0000001192092896
```

Benchmark impact:

```text
before:
  mean_ms: 94.13963928222657

after:
  mean_ms: 94.07802886962891
```

Conclusion:

```text
Removing one QK tile call per KV tile is correct but not a material speedup.
The dominant cost is now the serialized one-CTA full-KV/full-width scan, not
this local QK recomputation. The next lever is split-KV parallelism: multiple
CTAs compute partial online states and partial outputs over disjoint KV ranges,
then a reduction combines (m, l, O).
```

## Split-KV Full-Width Scaffold

2026-04-27

Added a split-KV version of the full-width online scaffold. Each split CTA
computes partial state over a disjoint KV range:

```text
partial_m: row max for the split
partial_l: row sum for the split
partial_o: unnormalized split output, scaled back by 1 / PROB_GLOBAL_SCALE
```

The reducer combines splits with the standard online merge:

```text
m = max(m_s)
l = sum(l_s * exp(m_s - m))
o = sum(o_s * exp(m_s - m))
out = o / l
```

Correctness at 8 splits over 32768 KV tokens:

```text
fused_smem_online_split_full_width_finite:             True
fused_smem_online_split_full_width_vs_quant_mean_abs:  2.444551783220561e-10
fused_smem_online_split_full_width_vs_quant_max_abs:   1.862645149230957e-09
fused_smem_online_split_full_width_vs_quant_cosine:    0.9999998807907104
fused_smem_online_split_full_width_vs_exact_mean_abs:  0.00012847436300944537
fused_smem_online_split_full_width_vs_exact_max_abs:   0.0007321708835661411
```

Split-count timing over full 32768 KV tokens, one 128-row x 512 output tile:

```text
4 splits:
  mean_ms: 20.470099258422852

8 splits:
  mean_ms: 10.40587501525879

16 splits:
  mean_ms: 5.220211219787598

32 splits:
  mean_ms: 2.653523254394531
```

Conclusion:

```text
Split-KV parallelism is the first material speedup. The scaffold dropped from
~94.1 ms serialized to ~2.65 ms at 32 splits for one 128-row x 512 output tile.
The next step is parallelizing over Q tiles so the kernel covers the real
flattened Shape B M dimension (q_len * group), not only the first 128 rows.
```

## Split-KV Q-Tile Generalization

2026-04-27

The split-KV full-width scaffold now parallelizes over Q tiles as well as KV
splits. The production Shape B flattened M dimension for `q_len=512, group=8`
is 4096 rows, or 32 tiles of 128 rows.

Correctness remained stable when scaling from one Q tile to the production
M dimension:

```text
q_tiles=1, splits=32:
  mean_abs vs online quant ref: 2.444551783220561e-10
  max_abs  vs online quant ref: 1.862645149230957e-09
  cosine:                     0.9999998807907104

q_tiles=4, splits=32:
  mean_abs vs online quant ref: 1.82609e-08
  max_abs  vs online quant ref: 9.64466e-06
  cosine:                     0.9999998807907104

q_tiles=32, splits=32:
  mean_abs vs online quant ref: 2.07399e-08
  max_abs  vs online quant ref: 1.2172e-05
  cosine:                     0.9999998807907104
```

Timing over full 32768 KV tokens:

```text
q_tiles=1:
  splits=32 mean_ms: 2.6535

q_tiles=4:
  splits=32 mean_ms: 2.6693

q_tiles=32:
  splits=4  mean_ms: 20.753
  splits=8  mean_ms: 20.022
  splits=16 mean_ms: 16.468
  splits=32 mean_ms: 16.476
```

Conclusion:

```text
The split-KV scaffold is correct at the production M dimension, but increasing
split count no longer helps after 16 splits. This is not a lack of KV
parallelism anymore. The likely limiter is CTA residency and wave scheduling:
each partial CTA uses the current 128x128x256 CUTLASS TensorStorage plus P/O
scratch, so the kernel probably runs at one heavy CTA per SM. The next
performance lever is changing the CUTLASS atom tile shape. The supported SM120
FP4 shapes are 128x128x128, 128x128x256, and 256x128x128. The next candidates
are:

1. 128x128x128 to reduce per-CTA shared memory and improve residency.
2. 256x128x128 to halve the number of Q tiles if its footprint still fits.

More KV split-count probes are off-target until the CTA footprint is reduced.
```

## SM120 CUTLASS Tile Footprint Metadata

2026-04-27

Added compile-time metadata for all SM120 FP4 CUTLASS tile shapes supported by
FlashInfer's runner:

```text
128x128x128:
  mainloop tensor storage:       73728 bytes
  score scratch:                 65536 bytes
  scaffold min shared storage:   85504 bytes
  opt-in smem margin:            15872 bytes
  score fits in tensor storage:  yes

128x128x256:
  mainloop tensor storage:       73728 bytes
  score scratch:                 65536 bytes
  scaffold min shared storage:   94720 bytes
  opt-in smem margin:             6656 bytes
  score fits in tensor storage:  yes

256x128x128:
  mainloop tensor storage:       82944 bytes
  score scratch:                131072 bytes
  scaffold min shared storage:  154624 bytes
  opt-in smem margin:           -53248 bytes
  score fits in tensor storage:  no
```

Conclusion:

```text
256x128x128 is not viable for the current scaffold because 256 rows require a
131 KiB score tile, exceeding SM120's 99 KiB opt-in per-block shared-memory
limit before P scratch or row state. 128x128x128 fits with better margin than
128x128x256, but it is still too large for two CTAs per SM because the mainloop
TensorStorage alone is 73 KiB. Therefore K128 is not an occupancy fix. Its
value is algorithmic: process one 128-token KV subtile at a time, keep the QK
scores resident, and avoid the K256 path's extra QK recomputation across
half-tiles.
```

## K128 Split-KV Scaffold Result

2026-04-27

Implemented a K128 variant of the split-KV full-width scaffold using the
128x128x128 SM120 CUTLASS block-scaled collective atom. The helper is now
tile-parameterized so K256 and K128 share the same partitioned smem producer and
MMA copy path.

Correctness:

```text
q_tiles=1, splits=32:
  finite:                      true
  mean_abs vs quant ref:       1.892900769462358e-08
  max_abs  vs quant ref:       9.644252713769674e-06
  cosine:                      1.0

q_tiles=32, splits=32:
  finite:                      true
  mean_abs vs quant ref:       2.2898106522006856e-08
  max_abs  vs quant ref:       1.09837856143713e-05
  cosine:                      1.0
```

Timing:

```text
K128 q_tiles=1,  splits=32 mean_ms:  2.7431
K256 q_tiles=1,  splits=32 mean_ms:  2.6535

K128 q_tiles=32, splits=32 mean_ms: 16.5416
K256 q_tiles=32, splits=32 mean_ms: 16.476
```

Conclusion:

```text
K128 is correct but does not improve wall time. The tile-shape hypothesis is
closed for this scaffold: the missing performance is not K256 half-tile
recompute, and the K128 shared-memory margin does not change CTA residency. The
remaining gap is the producer/mainloop structure. The current scaffold uses the
SM120 block-scaled MMA atom, but it still stages operands manually through
synchronous global loads and shared stores. It does not use the fast CUTLASS TMA
producer that makes the two-stage baseline fast.
```

## CUTLASS TMA Collective Bridge Status

2026-04-27

Rechecked the existing QK-only `qk_cutlass_collective_tile_kernel`, which calls
the full CUTLASS collective load and MMA path rather than the manual smem
producer.

Debug-mode isolation:

```text
debug_mode=3: launch/smem only                  -> works
debug_mode=5: pipeline init without prefetch    -> works
debug_mode=4: descriptor prefetch               -> illegal address
debug_mode=6: TMA load with prefetch skipped    -> illegal address
debug_mode=0: full collective path              -> illegal address
```

Conclusion:

```text
The bridge is not failing in MMA or output storage yet. It fails as soon as the
standalone kernel touches CUTLASS TMA descriptors / TMA load state. The runner
path itself works, so the failure is in how this custom kernel constructs or
passes `CutlassGemmKernel::Params`, or in missing launch/initialization
conventions from `GemmUniversalAdapter`. The next fix is to rebuild this bridge
around the exact adapter initialization / kernel launch path or to transplant
the SM120 `sm120_blockscaled_mma_tma.hpp` collective pattern directly, rather
than continuing the manual producer scaffold.
```

### Bridge Fix: `Params` Must Be Grid Constant

2026-04-27

The TMA bridge failure was the kernel-parameter ABI, not invalid descriptors.
CUTLASS launches `device_kernel<GemmKernel>` with:

```c++
CUTLASS_GRID_CONSTANT typename Operator::Params const params
```

The standalone debug kernel had been taking `CutlassGemmKernel::Params` by
ordinary value. With ordinary by-value params:

```text
single descriptor prefetch:
  A    -> works
  B    -> works
  SFA  -> works
  SFB  -> works

two or more descriptor prefetches:
  A+B       -> illegal address
  SFA+SFB   -> illegal address
  A+SFA     -> illegal address
  B+SFB     -> illegal address
  all four  -> illegal address

TMA load path:
  descriptor prefetch skipped -> illegal address
```

After changing the standalone kernel signature to match CUTLASS:

```c++
__global__ void qk_cutlass_collective_tile_kernel(
    CUTLASS_GRID_CONSTANT typename CutlassGemmKernel::Params const params,
    float* out_tile,
    int debug_mode)
```

the same probes pass:

```text
debug_mode=4   aggregate descriptor prefetch -> works
debug_mode=11  A+B descriptor prefetch       -> works
debug_mode=15  all manual descriptor prefetch -> works
debug_mode=6   TMA load only                 -> works
```

Full QK collective smoke now also passes against the C++ CUTLASS runner:

```text
qk_collective_finite:                  true
qk_collective_vs_official_mean_abs:    0.0015516469720751047
qk_collective_vs_official_max_abs:     0.015534400939941406
qk_collective_vs_official_cosine:      0.9999986290931702
```

Timing for one QK tile:

```text
shape: 128 query rows x 128 KV rows x D512
path:  SM120 CUTLASS TMA producer + block-scaled FP4 MMA

min_ms:   0.015200000256299973
mean_ms:  0.015647999988868833
max_ms:   0.016575999557971954
```

Conclusion:

```text
The CUTLASS TMA collective can be called from the standalone prototype kernel
as long as `Params` is passed as grid-constant, matching CUTLASS's own
`device_kernel` ABI. This unblocks using the fast TMA producer and
block-scaled MMA path inside the fused-attention reference kernel.
```

## First Fused TMA-QK + On-Chip PV Tile

2026-04-27

Added a new fixed 128-token prototype:

```text
QK:       CUTLASS SM120 TMA producer + block-scaled FP4 MMA
Softmax:  on-chip row-local softmax over 128 KV tokens
P:        on-chip NVFP4 quantization
PV:       existing K128 smem/copy-atom block-scaled PV path
Output:   FP32 reference output for 128 query rows x D512
```

Correctness against a BF16 runner-derived reference for the same 128-token
tile:

```text
finite:    true
mean_abs:  0.0020574606023728848
max_abs:   0.011530250310897827
cosine:    0.9950427412986755
```

Timing:

```text
fused TMA-QK + softmax/P + PV 128-token tile:
  min_ms:   0.14323200285434723
  mean_ms:  0.14474399983882905
  max_ms:   0.15116800367832184

QK TMA tile alone:
  mean_ms:  ~0.0156

existing manual PV atom tile, 128x128x256:
  mean_ms:  0.051188800297677514
```

Conclusion:

```text
The first fused prototype proves the TMA-QK collective can be composed with
on-chip softmax/P quantization and PV in one 384-thread kernel. Performance is
not yet competitive because the PV side is still the old synchronous smem
producer path. QK is no longer the bottleneck in this prototype; the next
performance lever is replacing the manual V/PV staging with a CUTLASS/TMA-style
producer or equivalent async path.
```

Implementation note:

```text
The existing smem atom helper originally assumed exactly 256 threads, matching
the MMA thread count. The TMA collective uses the CUTLASS GEMM kernel block
shape of 384 threads. The helper was made 384-block safe by allowing all
threads to participate in barriers while only the first 256 threads execute
MMA-fragment copy/gemm work. Existing 256-thread QK and PV checks still pass:

QK smem atom block:
  finite:    true
  mean_abs:  0.0
  max_abs:   0.0
  cosine:    0.9999999403953552

PV smem atom tile:
  finite:    true
  mean_abs:  0.0
  max_abs:   0.0
  cosine:    1.0
```

## Fused Tile Local Optimizations

2026-04-27

Accepted change: stage the softmax probability tile `P` into K128 CUTLASS
shared memory once, then reuse it across the four PV output-column groups.
Before this change, the PV helper restaged `P` and `P` scales for every
128-column output group.

```text
fused TMA-QK + softmax/P + PV 128-token tile:
  before P-smem reuse mean_ms: 0.14474399983882905
  after  P-smem reuse mean_ms: 0.13213759958744048
```

Correctness was unchanged:

```text
finite:    true
mean_abs:  0.0020574606023728848
max_abs:   0.011530250310897827
cosine:    0.9950427412986755
```

Accepted change: reuse the row max computed during softmax setup instead of
recomputing it in the P-quantization loop.

```text
after row-max reuse mean_ms: 0.1297728031873703
restore validation mean_ms: 0.12967840135097503
```

Rejected change: accumulate `row_l` through shared-memory `atomicAdd` in the
P-quantization loop to avoid the separate row-sum pass. It preserved
correctness but regressed wall time.

```text
row_l atomic variant:
  finite:    true
  mean_abs:  0.0020574606023728848
  max_abs:   0.011530257761478424
  cosine:    0.9950427412986755
  mean_ms:   0.13171839863061904
```

Conclusion:

```text
Small softmax/P-local cleanups are real but modest. They moved the fixed
128-token tile from ~0.1447 ms to ~0.1297 ms, not enough to close the structural
gap. The remaining high-leverage target is still PV/V staging: QK already uses
CUTLASS TMA, while PV still manually stages V and SFB through synchronous
partition_D writes before the CUTLASS copy-atom/MMA path.
```

## TMA-V/PV Prototype

2026-04-27

Started a new fixed 128-token prototype that keeps the working QK side
unchanged and replaces the PV-side manual V/SFB producer with the SM120 K128
CUTLASS TMA producer. The intended shape is:

```text
QK:       existing CUTLASS TMA QK tile
Softmax:  existing on-chip row max/sum plus P quantization
P:        manual partition_D producer into K128 CUTLASS A/SFA shared memory
V/SFB:    CUTLASS TMA producer into K128 CUTLASS B/SFB shared memory
PV:       CUTLASS K128 block-scaled MMA consumer
```

First compile result:

```text
failure: local TMA helper used the CUTLASS-source `Tensor` alias without
         importing it in this standalone file.
fix:     change those local declarations to `auto`.
```

This is a harness/type-cleanup issue, not a kernel result yet.

After the `auto` cleanup, the prototype compiles and runs:

```text
fused TMA-QK + softmax/P + TMA-V/PV 128-token tile:
  finite:    true
  mean_abs:  0.011651373468339443
  max_abs:   0.06659617274999619
  cosine:    0.6625540852546692
  min_ms:    0.0682239979505539
  mean_ms:   0.0689312007278204
  max_ms:    0.07257600128650665
```

Conclusion:

```text
TMA V/SFB staging is the right performance direction for the fixed 128-token
tile, cutting wall time from ~0.1297 ms to ~0.0689 ms, but the current
prototype is not correct. The likely bug is in the PV-side TMA tile coordinate,
SFB layout, or the B-side handoff to the K128 consumer; the QK and P paths are
unchanged from the previously correct smem-PV variant.
```

Attempted a separate PV-only TMA-V entry point to isolate the PV helper. The
additional full extension build did not emit compile errors but exceeded the
900s timeout, so that isolated entry point was removed. Next debugging should
stay inside the already-instantiated fused TMA path or use metadata/lightweight
diagnostics that do not force another expensive kernel instantiation.

Added Python-side diagnostics to compare the TMA-PV result against the known
correct smem-PV variant without adding another C++ kernel:

```text
TMA-PV vs BF16 reference:
  finite:    true
  mean_abs:  0.011651694774627686
  max_abs:   0.06659617274999619
  cosine:    0.6625493168830872
  mean_ms:   0.06873759999871254

smem-PV vs BF16 reference:
  mean_abs:  0.0020574606023728848

TMA-PV vs smem-PV:
  mean_abs:  0.012008974328637123
  max_abs:   0.07407797127962112
```

Per-output-group errors are uniform:

```text
group 0: mean_abs 0.011526337824761868, cosine 0.6683505177497864
group 1: mean_abs 0.01141174603253603,  cosine 0.6382735967636108
group 2: mean_abs 0.012301255017518997, cosine 0.6585910320281982
group 3: mean_abs 0.011367443017661572, cosine 0.6868770718574524
```

Conclusion:

```text
The TMA-PV failure is not isolated to one output-column group. This makes an
`out_group` tile-coordinate bug less likely. The next suspects are the TMA
B/SFB smem representation versus the manual A/P smem representation, the K128
SFB scale layout, or an A/P layout convention mismatch that only shows up when
the B operand is TMA-produced.
```

Checked CUTLASS-layout V scales in the TMA-PV path. This requires an explicit
PV global-scale correction because `flashinfer.nvfp4_quantize(...,
SfLayout.layout_128x4)` encodes `V * global_scale`.

After adding `pv_alpha = 1 / v_global_scale` to the TMA-PV normalization:

```text
TMA-PV vs BF16 reference:
  finite:    true
  mean_abs:  0.009542154148221016
  max_abs:   0.06659644097089767
  cosine:    0.6998423337936401
  mean_ms:   0.0840240005403757
```

A proper CUTLASS runner `P x V` check using CUTLASS-layout P and V plus both
global-scale corrections produced:

```text
runner_cutlass_pv_finite: true
mean_abs:                 0.18039363622665405
max_abs:                  1.0002326965332031
cosine:                   0.9902790188789368
```

The runner check is not numerically comparable to the softmax-P tile because it
uses random unnormalized P, but it confirms the CUTLASS-layout V tensor is
coherent at the runner level. The remaining issue is specific to the mixed
manual-A/P plus TMA-B/V embedded kernel path.

Tried rewriting the dynamic P/A producer to use the TMA destination partition
from `pv_params.mainloop.tma_load_a` / `tma_load_sfa` instead of the smem-copy
partition. This did not change the correctness failure and regressed wall time
badly:

```text
TMA-partition P producer:
  finite:    true
  mean_abs:  0.009540891274809837
  max_abs:   0.06659644097089767
  cosine:    0.6998568177223206
  mean_ms:   0.6528399914503098
```

Conclusion:

```text
Reject this implementation. It either did not derive the intended TMA logical
coordinates correctly, or TMA partition_S over an identity tensor is not the
right manual coordinate source for this copy atom. Keeping it would destroy the
performance reference without improving correctness.
```

Checked a second P-producer hypothesis: after the fast partitioned-P staging,
overwrite P directly through canonical `SmemLayoutA/SFA(row,k,stage)` indices.
This tests whether the mixed manual-A/TMA-B failure is caused by the smem-copy
producer putting P in a layout that only works for manual B.

Result:

```text
canonical P overwrite + TMA V:
  finite:             true
  mean_abs:           0.009542791172862053
  max_abs:            0.06659644097089767
  cosine:             0.6998265981674194
  tma_vs_smem_mean:   0.009218800812959671
  mean_ms:            0.08433280028402805
```

Per-output-group errors remained uniform:

```text
group 0: mean_abs 0.009348955005407333, cosine 0.7096272110939026
group 1: mean_abs 0.00874890387058258,  cosine 0.6935325264930725
group 2: mean_abs 0.010260644368827343, cosine 0.6925165057182312
group 3: mean_abs 0.009812665171921253, cosine 0.704400360584259
```

Conclusion:

```text
Reject the raw canonical P overwrite. It does not improve correctness relative
to the 0.084 ms CUTLASS-layout-V baseline. This makes a pure P write-coordinate
bug less likely. The next useful target is the B/SFB TMA handoff into the K128
PV collective, or a schedule/consumer assumption exposed only by mixed
manual-A/TMA-B staging.
```

Ran a no-compile output-structure diagnostic on the restored TMA-PV baseline.
The failure is not a simple output-column permutation:

```text
TMA-PV vs reference:
  mean_abs: 0.009541299194097519
  cosine:   0.6998529434204102

column correlation:
  identity mean/min/max: 0.30165398120880127 / -0.2583620250225067 / 0.7311226725578308
  best mean/min/max:     0.43719106912612915 / 0.0 / 0.7311226725578308
```

The stronger signal is 16-column structure in the output magnitude. TMA-PV
nearly zeros every even 16-column group and matches the smem-PV baseline in
every odd 16-column group:

```text
mean(abs(TMA col group)) / mean(abs(smem col group)), 32 groups of 16 columns:
[0.008, 1.016, 0.007, 1.004, 0.009, 1.019, 0.008, 1.015,
 0.008, 0.992, 0.008, 1.016, 0.008, 1.011, 0.008, 0.980,
 0.008, 1.027, 0.007, 0.993, 0.008, 1.021, 0.007, 1.001,
 0.005, 0.995, 0.008, 0.995, 0.007, 1.013, 0.008, 0.997]
```

Conclusion:

```text
The TMA-PV error has scale-vector granularity: 16 output columns at a time.
That strongly implicates the B/SFB side of the PV collective, especially the
SFB scale-factor TMA path or its N-axis interpretation. It is not random numeric
drift and not a simple output-column permutation.
```

Tried an isolation patch that TMA-loaded only B data while manually staging SFB
from row-major V scales. This would have tested whether the SFB TMA path alone
causes the even-16-column zeros. Both compile attempts timed out while CPU-active:

```text
attempt 1: timeout 900s
attempt 2: timeout 1200s after simplifying B-only transaction bytes to literal 8192
```

Conclusion:

```text
Reject this isolation patch as an iteration vehicle. It changes the heavyweight
CUTLASS helper body/signature enough to produce multi-15-minute rebuilds without
returning a result. Keep the restored baseline and use lighter diagnostics for
SFB layout/coordinate work.
```

Tried a one-line transaction-byte diagnostic to test whether
`tma_transaction_bytes_nk` under-waits the B/SFB TMA transaction:

```text
pipeline_params.transaction_bytes = tma_transaction_bytes_nk + 8192
```

This also timed out at 900s while CPU-active in the extension rebuild.

Conclusion:

```text
No correctness conclusion from this patch. The full
sm120_nvfp4_cutlass_fused_attention.cu translation unit is now too expensive
for microdiagnostics in this edit/test loop. Reverted the one-line change.
Future SFB/TMA diagnostics should use either host-side layout analysis or a
smaller purpose-built translation unit instead of rebuilding the full fused
prototype.
```

Tested a producer-convergence patch that inserted `__syncwarp()` immediately
before `collective.load_tail(pipeline, pipe_write)` in the hand-coded TMA-V/SFB
producer:

```text
fused_tma_qk_tma_pv_128_mean_abs: 0.009540933184325695
fused_tma_qk_tma_pv_128_max_abs:  0.06659644097089767
fused_tma_qk_tma_pv_128_cosine:   0.6998542547225952
tma_vs_smem_mean_abs:             0.009212853386998177
bench mean_ms:                    0.08429280072450637
```

This is identical to the restored baseline.

Conclusion:

```text
Reject the added __syncwarp(). The PV TMA failure is not caused by missing warp
convergence before load_tail in this producer. Reverted the patch. Continue with
SFB/TMA layout or a smaller isolation translation unit.
```

Checked the CUTLASS cooperative launch geometry from the compiled extension:

```text
QK  ThreadCount:              256
QK  kernel block threads:     384
K128 ThreadCount:             256
K128 kernel block threads:    384
```

`ThreadCount` is the MMA consumer thread count. `get_block_shape().x` adds the
extra producer warpgroup. The fused TMA-QK/TMA-PV launcher already uses
`CutlassGemmKernel::get_block_shape()`, so the 16-column alternating failure is
not simply caused by launching only the 256 MMA threads.

Fixed the TMA-PV correctness fault. The bug was not in TMA/SFB data movement; it
was in the output-fragment coordinate mapping. `collective.mma()` receives
`mma_thread_idx = threadIdx.x % ThreadCount`, but the TMA-PV helper wrote output
through a `thread_mma` built from `thread_idx`, where the extra producer
warpgroup caused threads 256-383 to be mapped to thread 0. The second consumer
warpgroup computed valid accumulators but wrote them through the wrong
C-fragment coordinates, producing the alternating 16-column output holes.

Patch:

```text
use tiled_mma.get_thread_slice(mma_thread_idx) for the consumer output mapping
inside cutlass_smem_pv_k128_reuse_p_tma_v_full_width_body()
```

Result:

```text
fused_tma_qk_tma_pv_128_finite:   True
fused_tma_qk_tma_pv_128_mean_abs: 0.002146251266822219
fused_tma_qk_tma_pv_128_max_abs:  0.011889606714248657
fused_tma_qk_tma_pv_128_cosine:   0.9945906400680542

smem_pv_reference_mean_abs:       0.0020574606023728848
tma_vs_smem_mean_abs:             0.0004941504448652267
tma_vs_smem_max_abs:              0.0028809793293476105

bench min_ms:                     0.08649600297212601
bench mean_ms:                    0.08858079984784126
bench max_ms:                     0.09759999811649323
```

Conclusion:

```text
TMA V/SFB staging is now correct for the fixed 128-token Shape-B tile. The
TMA-PV path is ~1.45x faster than the prior correct smem-PV baseline
(~0.0886 ms vs ~0.1297 ms) while preserving the same numerical error envelope.
The next step is scaling this from the one-tile correctness/perf reference into
the split/full Shape-B path, then measuring against FP8 FA2 cells.
```

Added a first split-K scaffold that combines:

```text
QK: existing manual-smem K128 block-scaled QK helper
PV: corrected CUTLASS TMA V/SFB helper
```

This was intended to reuse the now-correct TMA-PV primitive while preserving the
existing online-softmax split/reduction structure.

Results:

```text
8 splits:    kernel pegged GPU and produced no output; killed after several minutes
256 splits:  kernel-only path also pegged GPU and produced no output; killed
```

Conclusion:

```text
Reject the mixed-helper split scaffold in its current form. The manual-smem QK
helper was built around a 256-thread MMA CTA, while the CUTLASS TMA-PV helper
uses the 384-thread cooperative CUTLASS launch shape with an extra producer
warpgroup. Combining them inside one CTA is not a valid production direction
unless the QK side is made explicitly 384-thread cooperative-safe.

The cleaner next implementation is a full cooperative TMA-QK + TMA-PV split
kernel, using the same CUTLASS cooperative launch geometry for both phases.
The fixed 128-token TMA-QK/TMA-PV kernel already proves that geometry works.
```

Implemented the full cooperative split rewrite:

```text
QK: CUTLASS TMA-QK body with CUTLASS-layout Q/K and qk_alpha
PV: corrected CUTLASS TMA-V/SFB body
launch: CUTLASS cooperative 384-thread block shape
```

Kernel-only result for the smallest full-surface split configuration:

```text
num_splits=256, split_kv_tiles=1, q_tiles=1
```

The run still pegged GPU2 and produced no output after the extension compiled;
it was killed rather than letting the 900s timeout expire.

Conclusion:

```text
The previous row-major-input bug and mixed 256/384-thread helper geometry were
real issues, but they were not the only hang. The next fault is specific to
using the cooperative TMA helper bodies across a multi-CTA split grid. The fixed
one-CTA 128-token TMA-QK/TMA-PV kernel remains correct, so the next isolation
point is grid behavior and helper assumptions that differ between one CTA and
256 CTAs.
```

Added a uniform debug-return stage switch to the split TMA-QK/TMA-PV kernel.
Results on `num_splits=256, split_kv_tiles=1, q_tiles=1`:

```text
stage 1, after init:              returns
stage 2, after TMA-QK:            returns
stage 3, after softmax/P quant:   returns
stage 4, after one TMA-PV call:   hangs
```

Then checked the fixed one-tile TMA-QK/TMA-PV wrapper at nonzero KV tile
indices:

```text
kv_tile_128 = 0, 1, 2, 255: all finite and return
```

Conclusion:

```text
The split hang is inside the TMA-PV helper when called from the split kernel
body, but it is not a simple nonzero-KV-tile coordinate bug. The fixed one-tile
wrapper can call the same TMA-PV helper for nonzero tile indices. The next
difference to isolate is the split storage/body state around the PV call:
scratch/output pointer, shared-storage reuse, row-state arrays, or the fact that
the split body is using the helper after additional online-softmax state work.
```

Removed the dead inline online split kernel and replaced the split path with a
stable per-128-token partial kernel derived directly from the known-good fixed
TMA-QK/TMA-PV 128-token body:

```text
grid:    one CTA per (q_tile, kv_tile_128)
QK:      CUTLASS TMA-QK
softmax: local 128-token row max/sum
PV:      corrected CUTLASS TMA-V/SFB
output:  unnormalized partial_out plus partial_m/partial_l
reduce:  existing row-wise split reducer
```

This intentionally gives up multi-KV-tile online accumulation inside one CTA for
now, but keeps the important fused property inside each 128-token tile: QK,
softmax/P quantization, and PV do not round-trip QK or P through HBM.

Results:

```text
q_tiles=1,  num_splits=256: finite, mean_abs vs quant 3.6548e-05, cosine 0.999511, mean 0.1892 ms
q_tiles=4,  num_splits=256: finite, mean_abs vs quant 3.6539e-05, cosine 0.999512, mean 0.6304 ms
q_tiles=32, num_splits=256: finite, mean_abs vs quant 3.6521e-05, cosine 0.999512, mean 4.1239 ms
```

For the Shape-B q_len=512 / kv_len=32768 cell (`q_tiles=32`), this per-tile
TMA-QK/TMA-PV partial baseline is already below the prior FP8 FA2 comparison
number recorded for this cell (~5.01 ms), while preserving correctness against
the quantized reference.

Next optimization target:

```text
The current baseline writes one 128x512 float partial per KV tile and reduces
256 partials. That is correct and already competitive, but it pays a large HBM
partial-output round trip. The next ceiling-seeking step is to group multiple
KV tiles per CTA or per cooperative work unit and do online accumulation before
writing partial_out, without reintroducing the helper deadlock.
```

## Corrected Shape-B Success Bar: Two-Stage CUTLASS, Not FP8 FA2

The current production bar for Shape B `q_len=512, kv_len=32768, D=512,
group=8` is the in-tree two-stage CUTLASS NVFP4 reference, not the older FP8
FA2 comparison point.

```text
two-stage CUTLASS NVFP4 reference: ~0.634 ms
current per-128-token fused partial: 4.1239 ms
gap: ~6.5x too slow
```

Conclusion:

```text
The stable per-128-token TMA-QK/TMA-PV path is a correctness scaffold and a
diagnostic harness. It is not a shippable ceiling path for Shape B unless the
partial-output/reduction architecture is eliminated or collapsed enough to beat
the two-stage CUTLASS reference. If grouped partials do not materially close the
4.1239 ms -> 0.634 ms gap, the next implementation must move to a
CUTLASS-granularity fused FMHA mainloop rather than continuing to tune the
partial-HBM scaffold.
```

Grouped partial attempt:

```text
num_splits=128, q_tiles=1, group_kv_tiles=2
first launch: cudaErrorLaunchOutOfResources
after removing the extra row_alpha shared-memory array: launched, then pegged
GPU2 at 100% with no output for >30s; killed
```

Conclusion:

```text
Reject grouped partials as the next ceiling path. The grouped CTA still calls
the cooperative TMA-PV helper more than once from the same CTA body and returns
to the same dead/pathological behavior seen in earlier inline split attempts.
Even if that were fixed, this architecture still writes global partial_out and
pv_scratch and cannot plausibly close a 6.5x gap to the two-stage CUTLASS
reference.

The active implementation must now move to CUTLASS-granularity fused FMHA:
producer/consumer mainloop, CUTLASS/CuTe atom-level block-scaled MMA, online
softmax between QK and PV, and on-chip P tile reuse. The per-128-token partial
path remains only a correctness harness for QK/P-quant/PV components.
```

## CUTLASS FMHA Reference Re-Check After Grouped Rejection

Fetched `flashinfer-ai/flashinfer#2598` into `origin/pr/2598` for source
inspection. The PR itself only changes backend selection in `prefill.py` and
`utils.py`. Do not use PR #2598's CuTe DSL stack as the implementation
template for this SM120 C++ kernel.

The CuTe DSL files are useful only as a high-level confirmation that modern
Blackwell FMHA is role-based:

```text
flashinfer/cute_dsl/attention/prefill.py
flashinfer/cute_dsl/attention/mainloop_spec.py
flashinfer/cute_dsl/attention/roles/mma.py
```

The useful implementation facts for the C++ SM120 path:

```text
- The working Blackwell FMHA structure is role-based:
  loader, MMA, softmax, correction, epilogue.
- QK and PV are not composed as two GEMM collectives. The MMA role calls two
  atom-level GEMM primitives inside one interleaved mainloop.
- The mainloop owns pipeline state across the full KV loop. Re-initializing
  CUTLASS cooperative pipeline storage inside repeated helper calls is exactly
  the failure mode seen in grouped partials.
```

The CuTe DSL code is still not directly portable to the current target because
it is TCGEN/TMEM-oriented Blackwell code. The SM120 C++ path must use the
`mma.sync.aligned.kind::mxf4nvf4.block_scale` / CUTLASS
`OpClassBlockScaledTensorOp` atom path instead of TCGEN05/TMEM.

Primary implementation references remain:

```text
3rdparty/cutlass/examples/77_blackwell_fmha/collective/
3rdparty/cutlass/examples/88_hopper_fmha/collective/fmha_collective_tma.hpp
3rdparty/cutlass/examples/88_hopper_fmha/collective/fmha_collective_softmax.hpp
include/flashinfer/attention/blackwell/fmha_cutlass_sm100.cuh
```

Next C++ implementation rule:

```text
Do not call the existing TMA-PV helper repeatedly inside one CTA. Build a
single owner mainloop whose pipeline state lives across the whole KV loop, then
call QK MMA, online softmax/P quant, and PV MMA from that mainloop.
```

## Rejected PV Pipeline-State Probe

2026-04-28T00:00:00-05:00

Tried a bounded structural cleanup inside
`cutlass_smem_pv_k128_reuse_p_tma_v_full_width_body`: construct the SM120
`MainloopPipeline` once for the four PV output-column groups and advance
producer/consumer pipe states across the group loop, matching the persistent
state convention in Examples 77/88 more closely than the current helper body.

Results on the Shape-B q_tiles=1 / 256-split TMA-QK/TMA-PV scaffold:

```text
persistent PV pipe, no per-group load_tail:
  finite: true
  mean_abs vs quant: 8.6563e-04
  cosine vs quant:   0.5478
  mean_ms:           0.1916

persistent PV pipe, per-group load_tail restored:
  finite: true
  mean_abs vs quant: 8.6563e-04
  cosine vs quant:   0.5478
  mean_ms:           0.1881

restored helper-local PV pipe baseline:
  finite: true
  mean_abs vs quant: 5.9492e-04
  cosine vs quant:   0.7119
  mean_ms:           0.1733
```

Conclusion:

```text
Do not carry this cleanup forward. Moving only the PV pipeline object/state out
of the output-group loop regresses both correctness and wall time, even when
the per-group producer tail is restored. This confirms that the helper-level
TMA-PV embedding is not the right abstraction boundary for the real fused
kernel.

The earlier high-cosine per-128-token numbers in this worklog refer to the
smem/atom split scaffold, not the embedded TMA-QK/TMA-PV partial scaffold.
The TMA-QK/TMA-PV scaffold remains useful for validating embedded CUTLASS
pieces and launch mechanics, but it is not the correctness or performance
target.
```

## Rejected Naive K128 TMA-QK Bridge

2026-04-28T11:44:17-05:00

Tried a direct K128 TMA-QK bridge using `CutlassCollectiveMainloopK128` so QK
and PV could share the same `128x128x128` tile shape.

Implementation notes:

```text
- The first compile failed because the global K128 debug kernel was inside the
  `__CUDA_ARCH__` guard and the host wrapper could not see the symbol.
- Moving the global launcher outside the device-only guard fixed compilation.
- Runtime compiled and launched, but the kernel pegged GPU2 at 100% for several
  minutes with no output and was killed.
```

Conclusion:

```text
Do not carry the naive K128 QK bridge forward. K128 TMA-QK cannot be obtained by
copying the validated K256 QK bridge and changing only the K tile count to four.

Keep the validated K256 QK bridge and K128 PV helper as scaffold components, but
do not expose a K128 QK benchmark path. The real fused kernel must follow the
Examples 77/88 persistent-mainloop pattern at CUTLASS/CuTe atom level instead
of trying to compose or clone cooperative GEMM collectives per tile.
```

## Persistent-Mainloop Reference Constraints

2026-04-28T11:44:17-05:00

Line-level source review of the required C++ references:

```text
3rdparty/cutlass/examples/88_hopper_fmha/collective/fmha_collective_tma.hpp
3rdparty/cutlass/examples/88_hopper_fmha/collective/fmha_collective_softmax.hpp
3rdparty/cutlass/examples/77_blackwell_fmha/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp
include/flashinfer/attention/blackwell/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp
```

Implementation facts to carry into the SM120 prototype:

```text
- Example 88 owns `pipeline`, `pipeline_q`, read/write pipeline states, Q/K/V
  load states, QK MMA, softmax state, and PV MMA in one `compute()` function.
- The K/V pipeline is prefilled with alternating K and V loads. The mainloop
  performs QK on the current K stage, applies online softmax/rescale, then
  performs PV on the matching V stage without reinitializing pipeline storage.
- `CollectiveSoftmax::step_interleave_begin()` is the reference for row-max
  update plus rescaling the running PV accumulator.
- `CollectiveSoftmax::step_interleave_step()` is the reference for converting
  the current QK accumulator tile into probabilities while accumulating row
  sums before PV.
- The SM100 FlashInfer path uses separate role pipelines for load, MMA, softmax,
  correction, and epilogue. It is not directly portable to SM120 because it uses
  TMEM/TCGEN, but it confirms the same ownership rule: pipeline state lives at
  mainloop scope, not inside repeated helper calls.
```

Next code rule:

```text
The next prototype should introduce one dedicated owner kernel/body for Shape B
that explicitly carries Q/K/V pipeline states, online softmax state, and PV
accumulator state across the KV loop. It may reuse validated CUTLASS atom/layout
pieces, but it must not call the existing QK or PV helper as the unit of
composition inside the KV loop.
```

SM120 shared-memory budget constraint:

```text
SM120 opt-in shared memory:                    101376 bytes
QK 128x128x256 mainloop SharedStorage:          74752 bytes
PV 128x128x128 mainloop SharedStorage:          74752 bytes
Independent QK + PV SharedStorage:             149504 bytes
Independent QK + PV margin vs SM120 opt-in:    -48128 bytes
```

This rules out a literal D512 port of Example 88 with independent Q, K, and V
smem regions using the existing CUTLASS collective storage. The SM120 owner
kernel must either reuse a single storage region phase-by-phase, reduce tile
footprint, or implement a custom atom-level smem layout that stages only the
minimal live Q/K/V/P data needed by the interleaved mainloop.

## Persistent Owner Layout Smoke

2026-04-28T11:44:17-05:00

Added `persistent_mainloop_owner_layout_smoke`, a minimal SM120 launch that
uses the phased single-storage layout intended for the owner mainloop:

```text
dynamic shared storage: CutlassFusedTmaQkPv128Storage
storage bytes:          84992
QK SharedStorage:       74752
PV K128 SharedStorage:  74752
QK/PV storage policy:   alias PV K128 storage over the QK storage region
```

Smoke command:

```bash
timeout 1200s env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/usr/local/cuda-13.2 TORCH_CUDA_ARCH_LIST=12.0f PYTHONPATH=/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv:/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv/benchmarks /home/josh/tdm/infer/current/.venv/bin/python benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py --device 0 --persistent-owner-layout-check-only
```

Result:

```text
marker: [1.0, 84992.0, 74752.0, 74752.0, 0.0, 0.0, 0.0, 0.0]
independent_qk_pv_fits_sm120: false
independent_qk_pv_smem_margin_bytes: -48128
```

This is not a performance path yet. It establishes that the next owner kernel
can launch with one phased storage region and can construct QK and PV pipeline
objects sequentially on that region. The next implementation step is to move the
validated QK TMA load/MMA body into this owner body first, then add the PV phase
without reinitializing helper-level state per KV tile.

## Persistent Owner QK Stage

2026-04-28T11:44:17-05:00

Added `persistent_mainloop_owner_qk_stage`, a real QK TMA load/MMA smoke running
inside the phased owner storage. This is intentionally the first executable
piece of the future owner mainloop:

```text
storage: CutlassFusedTmaQkPv128Storage
QK phase: 128x128x256 SM120 block-scaled CUTLASS mainloop
PV phase: not wired yet
```

Smoke command:

```bash
timeout 1200s env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/usr/local/cuda-13.2 TORCH_CUDA_ARCH_LIST=12.0f PYTHONPATH=/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv:/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv/benchmarks /home/josh/tdm/infer/current/.venv/bin/python benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py --device 0 --persistent-owner-qk-check-only
```

Result versus the official CUTLASS runner tile:

```text
finite:   true
mean_abs: 0.0015516469720751047
max_abs:  0.015534400939941406
cosine:   0.9999986290931702
```

Conclusion:

```text
The validated TMA QK stage can run correctly inside the phased owner storage.
The next step is to add owner-controlled softmax/P quantization against this QK
tile, then add the PV phase using the same owner storage instead of calling the
old PV helper as the unit of composition.
```

Regression smoke after adding the owner hooks:

```text
256-split TMA-QK/TMA-PV fallback remains finite.
min/max output: -0.004217686131596565 / 0.004578343126922846
partial_m_finite: true
partial_l_finite: true
```

## Persistent Owner Softmax/P-Quant Stage

2026-04-28T11:44:17-05:00

Factored the 128-token tile softmax/P quantization into
`softmax_quant_scores_128()` and reused it from the existing TMA-QK/TMA-PV
partial scaffold. Added `persistent_mainloop_owner_softmax_stage`, which runs
the owner-storage QK stage, writes the QK accumulator tile into owner scratch,
then computes row max/sum and quantizes unnormalized P to NVFP4.

Smoke command:

```bash
timeout 1200s env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/usr/local/cuda-13.2 TORCH_CUDA_ARCH_LIST=12.0f PYTHONPATH=/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv:/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv/benchmarks /home/josh/tdm/infer/current/.venv/bin/python benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py --device 0 --persistent-owner-softmax-check-only
```

Result versus the CUTLASS-runner QK reference:

```text
finite:        true
p_mean_abs:    0.08929884433746338
p_max_abs:     0.16704505681991577
p_cosine:      0.9952029585838318
row_m_max_abs: 0.000685274600982666
row_l_max_abs: 0.0724029541015625
```

Conclusion:

```text
The owner path now has QK plus owner-controlled softmax/P quantization running
inside the phased storage policy that fits SM120. The next executable step is a
single PV output-group stage in the same owner kernel. That PV step should
inline the relevant CUTLASS K128 TMA/PV logic for one output group instead of
calling the old full-width PV helper loop as the composition unit.
```

Regression smoke after factoring `softmax_quant_scores_128()`:

```text
256-split TMA-QK/TMA-PV fallback remains finite.
min/max output: -0.004217686131596565 / 0.004578343126922846
partial_m_finite: true
partial_l_finite: true
```

## Persistent Owner PV Group Stage

2026-04-28T12:26:22-05:00

Added `persistent_mainloop_owner_pv_group_stage`, the next executable slice of
the owner kernel. It runs:

```text
QK TMA collective tile
owner softmax/P quantization
manual P staging into the aliased K128 PV shared storage
one TMA-V/PV output group
```

This intentionally validates one 128-column PV group before attempting the full
512-column output inside the owner body.

Important bug fixed while validating this stage:

```text
The old TMA-PV helper used `thread_idx` for C-fragment output mapping.
For Consumer1, that collapsed the thread slice to 0 instead of using
`mma_thread_idx = threadIdx.x % ThreadCount`.
```

Observed effect:

```text
before fix: TMA-PV cosine vs reference ~0.70
after fix:  TMA-PV cosine vs reference ~0.9946
```

The same fix was applied to the new one-group PV primitive.

Also corrected the Python check: the CUTLASS runner is not a valid reference for
owner-produced row-major P scales, and the row-major NVFP4 dequant helper is not
valid for CUTLASS-layout V scales. The owner PV group is now compared against
the existing full-width TMA scaffold and an exact P x V reference using the
original V tensor.

Smoke command:

```bash
timeout 1200s env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/usr/local/cuda-13.2 TORCH_CUDA_ARCH_LIST=12.0f PYTHONPATH=/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv:/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv/benchmarks /home/josh/tdm/infer/current/.venv/bin/python benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py --device 0 --persistent-owner-pv-group-check-only
```

Result:

```text
finite:                         true
owner_vs_full_tma_mean_abs:     1.8463700013349182e-10
owner_vs_full_tma_max_abs:      7.450580596923828e-09
owner_vs_full_tma_cosine:       0.9999998807907104
owner_vs_exact_mean_abs:        0.0016403645277023315
owner_vs_exact_max_abs:         0.007171541452407837
owner_vs_exact_cosine:          0.9952133893966675
```

Regression smokes:

```text
fused_tma_qk_tma_pv_128_check_only:
  finite: true
  mean_abs vs exact: 0.002146251266822219
  cosine vs exact: 0.9945906400680542
  tma_vs_smem_mean_abs: 0.0004941504448652267

256-split TMA-QK/TMA-PV fallback:
  finite: true
  min/max output: -0.004217686131596565 / 0.005271007772535086
  partial_m_finite: true
  partial_l_finite: true
```

Conclusion:

```text
The phased owner path now has QK, P quantization, and one correct TMA-PV output
group running in the same owner kernel. The next implementation step is to
extend the owner body to all four 128-column PV groups, then replace the
per-128-token partial scaffold with an owner-controlled full 128x512 tile.
```

## Persistent Owner Full-Tile Stage

2026-04-28T12:31:12-05:00

Added `persistent_mainloop_owner_full_tile_stage`, which expands the one-group
owner stage to all four 128-column PV groups and normalizes the 128x512 output
inside the kernel:

```text
QK TMA collective tile
owner softmax/P quantization
manual P staging into aliased K128 PV storage once
TMA-V/PV group 0
TMA-V/PV group 1
TMA-V/PV group 2
TMA-V/PV group 3
row_l normalization
```

Smoke command:

```bash
timeout 1200s env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/usr/local/cuda-13.2 TORCH_CUDA_ARCH_LIST=12.0f PYTHONPATH=/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv:/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv/benchmarks /home/josh/tdm/infer/current/.venv/bin/python benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py --device 0 --persistent-owner-full-tile-check-only
```

Result versus the existing full-width TMA scaffold:

```text
finite:                    true
owner_vs_full_tma_mean_abs: 0.0
owner_vs_full_tma_max_abs:  0.0
owner_vs_full_tma_cosine:   1.0
```

Per-tile timing command:

```bash
timeout 1200s env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/usr/local/cuda-13.2 TORCH_CUDA_ARCH_LIST=12.0f PYTHONPATH=/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv:/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv/benchmarks /home/josh/tdm/infer/current/.venv/bin/python benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py --device 0 --persistent-owner-full-tile-check-only --bench --warmup 10 --repeat 50
```

Timing result:

```text
persistent_owner_full_tile min/mean/max ms: 0.075936 / 0.078143 / 0.083296
existing_full_tma_tile     min/mean/max ms: 0.075456 / 0.076749 / 0.078496
```

Conclusion:

```text
The owner body now reproduces the helper-based 128x512 tile exactly. It is not
faster yet because it still reuses the same per-output-group PV pipeline and
does not persist pipeline state across KV tiles. The correctness baseline is now
clean enough to start replacing the 256 independent partial CTAs with an
owner-controlled multi-KV-tile mainloop.
```

## Reference Mainloop Notes For Multi-KV Owner

2026-04-28T12:32:29-05:00

Line-level review of the requested references:

```text
3rdparty/cutlass/examples/88_hopper_fmha/collective/fmha_collective_tma.hpp
```

Key structure:

```text
LoadQ once.
Prime alternating K/V TMA pipeline.
Allocate acc_pv once.
For each KV tile:
  QK MMA into acc_qk.
  online softmax step updates row max/sum and rescales acc_pv.
  convert softmaxed QK tile into PV A operand layout.
  PV MMA accumulates into the persistent acc_pv fragment.
Tail normalizes acc_pv by final row sums.
```

Important implementation point:

```text
The persistent win comes from keeping `acc_pv` live across KV tiles and applying
online-softmax rescale before each PV contribution. It is not achieved by
launching one helper-call-per-KV-tile and reducing global partials later.
```

```text
3rdparty/cutlass/examples/77_blackwell_fmha/collective/sm100_fmha_load_tma_warpspecialized.hpp
```

Key structure:

```text
The producer issues Q1, K1, Q2, V1, then alternates Ki/Vi through one KV
pipeline. This is the producer-side template for the SM120 owner kernel: keep
one persistent KV pipeline and alternate K/V stages instead of constructing a
fresh CUTLASS mainloop pipeline for every tile.
```

```text
3rdparty/cutlass/examples/77_blackwell_fmha/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp
```

Key structure:

```text
The consumer side waits K, computes QK into S, softmax/correction owns S/P, waits
V, then computes PV into O. It keeps O state live and applies correction/rescale
as row max changes.
```

SM120 consequence:

```text
The next real prototype should be a one-output-group persistent owner kernel,
not a full-head four-group kernel. A single group can keep its 128-column
`acc_pv` live across KV tiles. A full 512-column head would require four PV
accumulator groups live at once or HBM round-tripping; neither is the first
target. The price of one-output-group CTAs is recomputing QK per group, which
must be measured against avoiding QK/P HBM traffic.
```

## Owner Online Group Prototype

2026-04-28T12:47:58-05:00

Added `persistent_mainloop_owner_group_online_stage`, a one-output-group online
prototype:

```text
for each KV tile:
  QK TMA collective tile
  softmax/P quantization for the 128-column tile
  online row max/sum update
  PV TMA-V group accumulation into one 128-column output group
normalize the output group by final row sums
```

This is a correctness and state-management bridge, not the final performance
path. It still calls the existing QK and PV helper bodies once per KV tile, so
it reinitializes helper-local pipeline state repeatedly. The shipping direction
remains a CUTLASS Example 77/88-style persistent mainloop where the Q load, K/V
pipeline state, softmax state, and PV accumulator live across the full KV loop.

Launch/register finding:

```text
without targeted launch bounds:
  persistent_mainloop_owner_group_online_stage_kernel: 192 regs/thread
  launch result: cudaErrorLaunchOutOfResources

with __launch_bounds__(384, 1):
  persistent_mainloop_owner_group_online_stage_kernel: 154 regs/thread, 0 stack
  persistent_mainloop_owner_full_tile_stage_kernel:    148 regs/thread, 0 stack
```

Online-group smoke results against exact softmax(QK) @ V for output group 0:

```text
num_kv_tiles=2: mean_abs 0.0018509341, max_abs 0.0096779848, cosine 0.9871958, ~0.068 ms
num_kv_tiles=4: mean_abs 0.0012544346, max_abs 0.0070589883, cosine 0.9887136, ~0.109 ms
num_kv_tiles=8: mean_abs 0.0008975492, max_abs 0.0048882170, cosine 0.9882689, ~0.193 ms
```

Conclusion:

```text
The online row-rescale + PV accumulation math is viable and launches without a
global maxrregcount cap. The current helper-loop implementation is expected to
scale poorly to the full q=512/kv=32768 cell because it rebuilds the QK/PV
collective machinery for every KV tile. The next implementation step is not
more threshold or swizzle tuning; it is replacing the helper-loop body with a
single persistent mainloop based on the CUTLASS Example 77/88 structure.
```

## Register-Accumulating Owner Prototype

2026-04-28T13:18:26-05:00

Attempted a one-output-group register-accumulating variant:

```text
keep PV accumulator fragment live in registers
scale old accumulator rows by online old_scale
stage P into the PV A operand layout
run a copied no-clear SM120 PV MMA body
write output group once at the end
```

The first implementation baked `pv_alpha / PROB_GLOBAL_SCALE` into the P scale
sidecar. That underflowed/over-quantized the e4m3 scale path:

```text
num_kv_tiles=1: mean_abs 0.0169163, max_abs 0.0617323, cosine 0.0
```

Moving the constant scale to final FP32 writeback and leaving tile 0 P scales
unchanged fixed the one-tile result:

```text
num_kv_tiles=1: mean_abs 0.00253876, max_abs 0.0140011, cosine 0.989603, ~0.049 ms
```

However, repeated helper reentry is not safe:

```text
num_kv_tiles=2: deadlocks / runs indefinitely with GPU at 100%
```

Adding an explicit consumer named barrier after the copied no-clear MMA did not
fix the multi-tile deadlock. The register-accumulating helper-loop path is now
guarded to `num_kv_tiles == 1` so it cannot accidentally hang benchmark runs.

Conclusion:

```text
The no-clear PV MMA can reproduce the one-tile PV result, but trying to stitch
multiple helper invocations together still fights the CUTLASS collective
pipeline lifetime. This confirms the prior direction: the real implementation
must be one persistent mainloop with one Q/K/V pipeline lifetime, not repeated
construction and teardown of helper-local collectives.
```

## Register-Accumulating Owner Prototype: Dedicated PV Pipeline Storage

2026-04-28T14:09:00-05:00

The multi-tile deadlock was caused by unsafe reuse of the aliased QK collective
storage for PV helper pipeline state across repeated helper invocations. Adding
a dedicated PV pipeline-storage sidecar to the owner storage fixed the repeated
reentry hang while still fitting SM120 shared memory:

```text
SM120 opt-in shared memory:       101376 bytes
owner phased storage with PV pipe: 88064 bytes
QK SharedStorage:                 74752 bytes
PV K128 SharedStorage:            74752 bytes
independent QK + PV storage:     149504 bytes (does not fit)
```

Register-accumulating helper-loop results against exact softmax(QK) @ V for
output group 0:

```text
num_kv_tiles=2:  mean_abs 0.0018606472, max_abs 0.0094464933, cosine 0.9871230, min 0.067136 ms
num_kv_tiles=4:  mean_abs 0.0012737792, max_abs 0.0071084099, cosine 0.9885631, min 0.108000 ms
num_kv_tiles=8:  mean_abs 0.0009108213, max_abs 0.0049337139, cosine 0.9880465, min 0.185088 ms
num_kv_tiles=16: mean_abs 0.0006633630, max_abs 0.0035498557, cosine 0.9885939, min 0.343168 ms
num_kv_tiles=32: mean_abs 0.0004634701, max_abs 0.0023278608, cosine 0.9893481, min 0.661696 ms
num_kv_tiles=64: mean_abs 0.0003213732, max_abs 0.0015640703, cosine 0.9890051, min 1.296544 ms
```

Interpretation:

```text
The register accumulator is now a correct multi-tile bridge and avoids the
global read/modify/write used by the earlier helper-loop path. It is still
nearly linear in KV tiles because every tile reconstructs helper-local QK and
PV collective state. Extrapolating 64 -> 256 tiles gives roughly 5.2 ms for one
output group at kv=32768, far above the 0.634 ms two-stage CUTLASS reference.

This closes the helper-loop line of work. The next implementation must be a
single CUTLASS Example 77/88-style persistent mainloop: persistent Q/K/V
pipeline state, QK MMA, online softmax + P quantization, and PV MMA in one
kernel body, without helper reconstruction per KV tile.
```

## Persistent PV Pipeline Inside Register Owner

2026-04-28T14:41:00-05:00

Changed the register-owner prototype so the PV TMA pipeline object and
read/write pipeline states live across the KV loop instead of being reconstructed
for every tile. The first attempt staged P only into shared-memory stage 0 while
the persistent pipeline rotated stages, which produced the expected wrong-stage
failure:

```text
num_kv_tiles=2: mean_abs 0.0085940678, max_abs 0.0336251892, cosine 0.6582672
```

Fix: stage P and P scales into `tile % PvPipeline::Stages`, matching the V TMA
stage consumed by the PV MMA. Correctness returned to the prior level.

Timings after persistent PV pipeline:

```text
num_kv_tiles=2:   mean_abs 0.0018606472, max_abs 0.0094464933, cosine 0.9871230, min 0.065888 ms
num_kv_tiles=8:   mean_abs 0.0009108213, max_abs 0.0049337139, cosine 0.9880465, min 0.180960 ms
num_kv_tiles=32:  mean_abs 0.0004634701, max_abs 0.0023278608, cosine 0.9893481, min 0.636640 ms
num_kv_tiles=64:  mean_abs 0.0003213732, max_abs 0.0015640703, cosine 0.9890051, min 1.246272 ms
num_kv_tiles=256: mean_abs 0.0001594797, max_abs 0.0008670320, cosine 0.9926619, min 4.907872 ms
```

Interpretation:

```text
Keeping PV pipeline state alive removes a small fixed/reconstruction cost:
64 tiles improved from 1.296544 ms to 1.246272 ms. This is useful evidence but
not close to the 0.634 ms two-stage reference. The dominant remaining cost is
the QK helper-loop structure: it reloads/rebuilds QK for every KV tile, and the
one-output-group CTA policy would repeat QK once per 128-column output group.

Next target: split the QK collective load path so Q is loaded once per Q tile
and K is streamed across KV tiles with persistent pipeline state, matching the
Example 88 separate-Q and K/V pipeline structure.
```

## Rejected: QK K-Only Reuse With Aliased PV Storage

2026-04-28T15:05:00-05:00

Attempted to make tile 0 load Q/K normally, then reuse resident Q/SFA and load
only K/SFB on later KV tiles. This gave a small timing win:

```text
num_kv_tiles=32: 0.636640 ms -> 0.623264 ms
num_kv_tiles=256: 4.907872 ms -> 4.784288 ms
```

But it is not a valid implementation in the current owner-storage layout.
`owner_storage.qk` is deliberately aliased as the PV shared storage; after each
QK tile, PV staging writes P into `smem_A` and P scales into `smem_SFA`. That
overwrites the Q/SFA stages the K-only QK path tries to reuse. The correctness
drop is visible:

```text
num_kv_tiles=2: cosine 0.9871230 -> 0.9843483
num_kv_tiles=8: cosine 0.9880465 -> 0.9812438
num_kv_tiles=32: cosine 0.9893481 -> 0.9808549
num_kv_tiles=256: cosine 0.9926619 -> 0.9863577
```

A resident Q/SFA sidecar also does not fit SM120 shared memory with the current
tile shape:

```text
owner storage with aliased QK/PV: 88064 bytes
Q/SFA sidecar estimate:          36864 bytes
combined:                       124928 bytes
SM120 opt-in limit:             101376 bytes
```

Conclusion:

```text
Do not retry K-only QK reuse while PV aliases QK tensor storage. Correct Q reuse
requires a different mainloop layout: either a smaller QK/PV tile shape that
leaves room for resident Q/SFA, or an Example 88-style layout with separate Q
pipeline storage designed from the start.
```

## Pivot: Register-Resident Q Is The Active Path

2026-04-28T15:24:00-05:00

The active path is not tile-shape sweeping or split-KV reduction. Those are
sideways moves against already-measured ceilings:

```text
tile-shape sidecar: tries to make smem-resident Q fit, but still avoids the
                    canonical FA mainloop structure.
split-KV reduction: returns to the partial-output HBM round-trip path that
                    measured around 4-5 ms, far above the 0.634 ms target.
```

The canonical path from CUTLASS Examples 77/88 and FlashInfer's SM100 FMHA is
register-resident Q:

```text
load Q once -> partition into MMA A register fragments -> keep those fragments
live across the KV loop -> alias Q smem for P/V staging -> stream K/V tiles.
```

This is the only path that simultaneously avoids Q reload and avoids the SM120
shared-memory budget conflict. The current prototype still calls the full QK
collective once per KV tile, so it cannot reach the two-stage CUTLASS ceiling.
The next implementation step is to replace that full helper call with an
atom-level QK mainloop that owns the Q register fragments across KV tiles.

Resource note after adding the persistent PV pipeline:

```text
persistent_mainloop_owner_group_online_register_stage_kernel:
  REG 168, STACK 232, SHARED 1024 static
persistent_mainloop_owner_group_online_stage_kernel:
  REG 154, STACK 0,   SHARED 1024 static
persistent_mainloop_owner_full_tile_stage_kernel:
  REG 148, STACK 0,   SHARED 1024 static
```

The persistent PV pipeline introduced stack and increased register use, but the
kernel is still below the 255-register architectural ceiling. The stack should
not ship, but it does not change the conclusion: the next useful work is
register-resident Q, not more helper-loop tuning.

## Register-Resident Q First Measurement

2026-04-28T16:15:00-05:00

After the benchmark cleanup checkpoint (`fb9f534`), the active
`persistent_owner_group_online_register_q_stage` path was remeasured:

```text
num_kv_tiles=2:  mean_abs 0.0019029246, max_abs 0.0086733261, cosine 0.9864801, min 0.107008 ms
num_kv_tiles=8:  mean_abs 0.0009381340, max_abs 0.0050215498, cosine 0.9870479, min 0.251072 ms
num_kv_tiles=32: mean_abs 0.0004734976, max_abs 0.0024820382, cosine 0.9886041, min 0.828544 ms
num_kv_tiles=64: mean_abs 0.0003284118, max_abs 0.0016009322, cosine 0.9883098, min 1.600960 ms
```

This is a correctness pass but a performance regression versus the previous
persistent-PV helper-loop path:

```text
previous persistent-PV helper-loop path:
num_kv_tiles=2:  0.065888 ms
num_kv_tiles=8:  0.180960 ms
num_kv_tiles=32: 0.636640 ms
num_kv_tiles=64: 1.246272 ms
```

Interpretation:

```text
Register-resident Q is still the right architectural direction, but the first
prototype is not the right implementation. The likely cost is not the one-time
Q register residency itself; the current path replaces the validated QK
collective helper with a manual K/SFB TMA pipeline and explicit score
writeback, and that path is slower at every measured tile count.

Next step: isolate QK-only cost for the register-Q K-streaming path versus the
validated QK collective helper. Do not tune PV or softmax until the QK delta is
known.

## Reference Diff: Examples 77/88 vs Current Register-Q Prototype

2026-04-28T16:31:00-05:00

Before more tuning, compared the current SM120 prototype against the working
FMHA references:

```text
3rdparty/cutlass/examples/88_hopper_fmha/collective/fmha_collective_load.hpp
3rdparty/cutlass/examples/88_hopper_fmha/collective/fmha_collective_tma.hpp
3rdparty/cutlass/examples/77_blackwell_fmha/collective/sm100_fmha_load_tma_warpspecialized.hpp
3rdparty/cutlass/examples/77_blackwell_fmha/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp
include/flashinfer/attention/blackwell/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp
```

Important differences:

```text
Reference pattern:
- Dedicated load role/warpgroup owns TMA descriptor prefetch and TMA producer
  state.
- Q has its own pipeline (`PipelineQ` / `MainloopPipelineQ`) and is loaded with
  TMA through the MMA-derived `partition_A` / `tma_partition` path.
- K and V are streamed through load pipelines with producer/consumer states
  that persist across the FMHA mainloop.
- The consumer waits on Q once, keeps the Q fragment/view live, and only advances
  the KV pipeline in the loop.

Current SM120 prototype:
- Consumer warpgroups manually read packed Q bytes from global memory and write
  Q/SFA into shared memory.
- Q is then copied from shared memory into register fragments before the KV loop.
- K/SFB streaming is manually reconstructed for QK instead of reusing the same
  load-role/TMA-partition convention as the references.
- PV uses a separate pipeline from QK; the QK/PV phase sequencing is not yet the
  alternating K/V pipeline used by the references.
```

Conclusion:

```text
The first register-Q prototype is slower because it mixes a reference-style
register-Q idea with a non-reference producer path. The next production-shaped
fix should not be more pipeline-depth guessing. It should move Q/SFA loading and
K/SFB streaming into the same loader-owned, partition-derived TMA convention as
Examples 77/88:

1. Add a dedicated Q/SFA TMA load path for the SM120 block-scaled Q operand.
2. Have the load role issue Q once before the KV loop.
3. Have consumers wait on Q once and copy/hold the Q fragments.
4. Keep K/SFB producer state loader-owned and persistent across KV tiles.
5. Only after QK matches the reference ownership model, revisit K/V interleave
   and softmax/PV overlap.
```

## SM100 Mainloop Structural Diff

2026-04-28T14:49:34-05:00

Stopped kernel edits and produced a side-by-side structural diff against the
FlashInfer SM100 FMHA reference:

```text
SM120_NVFP4_SM100_MAINLOOP_DIFF.md
```

Key finding:

```text
The current SM120 benchmark adopted register-resident Q but did not adopt the
SM100 ownership model that makes register-resident Q safe and fast. The failed
TMA-Q attempts confirm that independent PipelineQ storage and lifetime are
load-bearing. Reusing QK pipeline storage to stage Q deadlocks.
```

Next implementation direction from the diff:

```text
Create an SM120 NVFP4 load collective modeled on
Sm100FmhaLoadTmaWarpspecialized:
- dedicated PipelineQ for Q/SFA,
- dedicated PipelineK for K/SFB,
- dedicated PipelineV for V/SFB,
- loader role issues Q1, K1, Q2, V1, K2, V2, ...

Do not continue helper-call-per-tile tuning until this structure exists.
```

## SM120 Q/K Load Collective Port Slice

2026-04-28T15:30:00-05:00

Implemented the first compileable SM100-structured port slice in
`benchmarks/sm120_nvfp4_cutlass_fused_attention.cu`:

```text
Sm120Nvfp4QkLoadCollectiveStorage
- one CUTLASS SM120 NVFP4 tensor tile storage,
- dedicated PipelineQ storage,
- dedicated PipelineK storage.
```

The new smoke kernel is `sm120_nvfp4_qk_load_collective_stage_kernel`. It keeps
the SM100 lifetime pattern for the QK subset:

```text
loader role:
  Q0 -> K0 -> Q1 -> K1

consumer roles:
  wait/copy Q0 into register fragment,
  wait/copy Q1 into register fragment,
  wait K0 and QK MMA with Q0,
  wait K1 and QK MMA with Q1.
```

This intentionally validates only the Q/K load collective before adding V/PV.
It is not another helper-call-per-tile attention path. The specific convention
being validated is independent Q/K pipeline storage plus partition-derived TMA
writes for both packed FP4 payloads and UE4M3 scale sidecars.

Validation:

```text
command:
  CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/usr/local/cuda-13.2
  python benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py
    --device 0 --sm120-qk-load-collective-check-only
    --bench --warmup 1 --repeat 2

result:
  finite: true
  mean_abs vs CUTLASS QK runner: 0.0015516469720751047
  max_abs  vs CUTLASS QK runner: 0.015534400939941406
  cosine   vs CUTLASS QK runner: 0.9999986290931702
  storage: 74752 bytes
  SM120 opt-in shared-memory margin: 26624 bytes
  min wall time: 0.016416 ms
```

Conclusion:

```text
The independent PipelineQ/PipelineK pattern is viable and fits in SM120 shared
memory for the QK subset. The previous TMA-Q deadlock was caused by violating
pipeline lifetime/aliasing rules, not by Q/K tensor storage capacity.
```

## SM120 Q/K/V Load Collective Port Slice

2026-04-28T15:55:00-05:00

Added the next load-collective slice:

```text
Sm120Nvfp4QkvLoadCollectiveStorage
- QK tensor storage for Q/K payloads and scale sidecars,
- dedicated PipelineQ storage,
- dedicated PipelineK storage,
- compact V-only stage-2 storage:
  - PV B smem sidecar,
  - PV SFB smem sidecar,
  - dedicated PipelineV storage.
```

Important storage finding:

```text
Full independent QK + PV shared storage does not fit:
  QK SharedStorage + PV SharedStorage = 149504 bytes
  SM120 opt-in limit                 = 101376 bytes

The compact V-only stage-2 path does fit:
  Q/K/V load collective storage = 95232 bytes
  SM120 margin                  = 6144 bytes
```

This means the SM120 port cannot mirror SM100 by naively allocating full
independent Q/K/V/P tensor storage. The shippable SM120 structure needs compact
V-only storage while reusing the QK tensor region for post-QK P staging.

The storage number includes the row max/sum state added for the first
softmax/PV handoff slice.

Validation:

```text
command:
  CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/usr/local/cuda-13.2
  python benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py
    --device 0 --sm120-qkv-load-collective-check-only
    --bench --warmup 1 --repeat 2

result:
  finite: true
  QK mean_abs vs CUTLASS QK runner: 0.0015516469720751047
  QK max_abs  vs CUTLASS QK runner: 0.015534400939941406
  QK cosine   vs CUTLASS QK runner: 0.9999986290931702
  storage: 95232 bytes
  SM120 opt-in shared-memory margin: 6144 bytes
  min wall time: 0.018688 ms
```

Conclusion:

```text
The loader-owned Q/K/V pipeline topology now fits and runs inside one CTA. The
next step is not more storage probing: use the same compact storage to add the
MMA-to-softmax-to-PV handoff, with the QK tensor region aliased as P staging
after QK has consumed K.
```

## SM120 QK -> Softmax/P -> PV Handoff Slice

2026-04-28T16:20:00-05:00

Added the first fused handoff smoke on top of the compact Q/K/V load collective:

```text
sm120_nvfp4_qkv_handoff_stage_kernel
```

Flow:

```text
1. Loader role issues Q0, K0, Q1, V0, K1 with independent Q/K/V pipelines.
2. Consumers copy Q0/Q1 to register fragments.
3. Consumers copy V0 to register fragments from compact V-only storage.
4. Consumers run QK with register-resident Q and K from PipelineK.
5. QK scores are materialized to an external 128x128 score scratch.
6. Softmax quantizes P directly into aliased QK smem:
   - QK smem_A is reinterpreted as PV A/P storage.
   - QK smem_SFA is reinterpreted as PV SFA/P-scale storage.
7. PV runs with P from aliased smem and V from register fragments.
```

Important limitation:

```text
This is a correctness/structure slice, not the final production mainloop. It
still materializes the 128x128 score tile to an external scratch tensor. The
production path must replace that with role-local score fragments and a
pipeline contract for softmax/P quantization. The important structural result
here is that P staging is already on-chip and aliases the QK tensor region
after K is consumed.
```

Validation:

```text
command:
  CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/usr/local/cuda-13.2
  python benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py
    --device 0 --sm120-qkv-handoff-check-only
    --bench --warmup 1 --repeat 2

result:
  finite: true
  mean_abs vs exact one-tile attention: 0.0025387569330632687
  max_abs  vs exact one-tile attention: 0.014001144096255302
  cosine   vs exact one-tile attention: 0.9896034598350525
  storage: 95232 bytes
  SM120 opt-in shared-memory margin: 6144 bytes
  min wall time: 0.07599999755620956 ms
```

Conclusion:

```text
The compact storage supports the full QK -> softmax/P-quant -> PV dataflow in
one CTA. The next performance-relevant step is removing the external score
scratch by turning QK score fragments into the producer side of a softmax/P
stage, matching the SM100 mainloop's role handoff rather than the current
block-wide score materialization.
```

### On-Chip BF16 Logits Handoff

2026-04-28T16:45:00-05:00

Replaced the external-score compute dependency in the handoff smoke with an
on-chip BF16 logits tile:

```text
QK accumulator fragment
  -> scale by qk_alpha / sqrt(D)
  -> store BF16 logits into aliased QK smem_B
  -> softmax/P quant reads BF16 logits from smem_B
  -> P is still staged into aliased QK smem_A/smem_SFA for PV
```

The `score_scratch` tensor remains in the API only as a debug mirror of raw QK
scores. It is no longer the input to softmax/P quantization.

Important storage fact:

```text
QK smem_B is free after K is consumed and is large enough to hold a 128x128
BF16 logits scratch (32768 bytes). This preserves the compact storage layout:

  total storage: 95232 bytes
  SM120 opt-in shared-memory margin: 6144 bytes
```

Validation:

```text
command:
  CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/usr/local/cuda-13.2
  python benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py
    --device 0 --sm120-qkv-handoff-check-only
    --bench --warmup 1 --repeat 2

result:
  finite: true
  mean_abs vs exact one-tile attention: 0.0025383096653968096
  max_abs  vs exact one-tile attention: 0.014019200578331947
  cosine   vs exact one-tile attention: 0.9895987510681152
  storage: 95232 bytes
  SM120 opt-in shared-memory margin: 6144 bytes
  min wall time: 0.06492800265550613 ms
```

Conclusion:

```text
The first HBM score round-trip is removed from the handoff slice without
changing the shared-memory footprint. The remaining non-production behavior is
that logits are materialized as a whole BF16 tile in smem_B before softmax/P
quantization. The next step is the persistent inner mainloop: keep Q register-
resident across KV tiles, rotate K/V stages through the compact load collective,
and replace this one-tile handoff with online row max/sum + PV rescale over
successive KV tiles.
```

## Compact Q/K/V Online Register-Q Slice

2026-04-28T17:15:00-05:00

Added the first compact-storage multi-KV online-softmax smoke:

```text
sm120_nvfp4_qkv_online_register_q_stage_kernel
```

Dataflow:

```text
1. Q0/Q1 are loaded once through the Q collective and copied into register
   fragments.
2. For each 128-token KV tile:
   - K0/K1 stream through the K collective.
   - V streams through the compact V-only StageCount<2> collective.
   - QK uses register-resident Q.
   - QK B smem is aliased as a BF16 logits tile after K is consumed.
   - row_m/row_l are computed from the BF16 logits tile.
   - global_m/global_l plus old_scale/tile_scale perform online softmax
     rescaling.
   - P is quantized into aliased QK A/SFA smem with tile_scale applied.
   - PV accumulates into register fragments, with old_scale applied to the
     existing accumulator before adding the current tile.
3. The final output divides by global_l and applies the PV alpha scale.
```

Storage:

```text
Added online state to the compact load collective:
  row_m/row_l
  global_m/global_l
  old_scale/tile_scale

total storage: 97280 bytes
SM120 opt-in shared-memory margin: 4096 bytes
```

Validation and scaling:

```text
command:
  CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/usr/local/cuda-13.2
  python benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py
    --device 0 --sm120-qkv-online-check-only
    --persistent-owner-online-kv-tiles {1,2,4,8}
    --bench --warmup 1 --repeat 3

results:
  kv_tiles=1:
    finite: true
    mean_abs vs exact: 0.0025383096653968096
    max_abs  vs exact: 0.014019200578331947
    cosine   vs exact: 0.9895987510681152
    min wall time: 0.0793600007891655 ms

  kv_tiles=2:
    finite: true
    mean_abs vs exact: 0.00184237165376544
    max_abs  vs exact: 0.009134171530604362
    cosine   vs exact: 0.9872949719429016
    min wall time: 0.13152000308036804 ms

  kv_tiles=4:
    finite: true
    mean_abs vs exact: 0.001257083029486239
    max_abs  vs exact: 0.005877412855625153
    cosine   vs exact: 0.9887419939041138
    min wall time: 0.2300799936056137 ms

  kv_tiles=8:
    finite: true
    mean_abs vs exact: 0.0008950772462412715
    max_abs  vs exact: 0.004760183393955231
    cosine   vs exact: 0.9883692860603333
    min wall time: 0.4264639914035797 ms
```

Conclusion:

```text
The compact load collective now supports a real online-softmax register-Q loop
over multiple KV tiles with correctness preserved. Runtime is still close to
serial per-tile cost, so the next perf lever is pipeline lifetime/overlap:
preload the next K/V tile while consumers run QK/softmax/PV for the current
tile, rather than issuing current-tile K/V and immediately waiting on it.
```

### Next-Tile K/V Prefetch Probe

2026-04-28T17:35:00-05:00

Changed the compact online loop so K/V for tile `n+1` are issued after tile
`n` P staging and before tile `n` PV. This is the earliest safe point for K
prefetch because QK smem_B is used as the BF16 logits tile until P staging has
finished. V is already register-resident by then, so its compact smem region is
also free.

Result:

```text
kv_tiles=1: 0.08099199831485748 ms
kv_tiles=2: 0.13065600395202637 ms
kv_tiles=4: 0.23001599311828613 ms
kv_tiles=8: 0.4249599874019623 ms
```

Conclusion:

```text
The schedule is correct but does not produce material overlap. The likely
reason is structural: K/V TMA issue is still serialized behind block-wide
softmax/P staging barriers and the producer has too little independent work
ahead of the consumer. The next step needs profiling on this compact online
kernel, not more blind scheduling tweaks.
```

## Compact Online Kernel NCU And SM100-Diff Gap

2026-04-28T18:05:00-05:00

Profiled the compact online register-Q kernel at `num_kv_tiles=4`:

```text
report:
  reports/sm120_qkv_online_register_q_4tiles_full.ncu-rep

shape:
  D=512, group=8 output slice, q_tile=128 rows, 4 * 128 KV tokens

kernel time in NCU:
  210.24 us

selected counters:
  smsp__inst_executed_op_ldgsts.sum:                    0
  smsp__inst_executed_op_tma_ld.sum:                    28
  smsp__sass_inst_executed_op_tma_ld.sum:               28
  smsp__warps_eligible.avg.per_cycle_active:            0.371951
  sm__warps_active.avg.pct_of_peak_sustained_active:    24.843031
  smsp__issue_active.avg.pct_of_peak_sustained_active:  33.134299
  sm__pipe_tensor_cycles_active.avg.pct_active:         4.308432
  l1tex shared load bank conflicts:                     39114
  l1tex shared store bank conflicts:                    7299
  gpu__dram_throughput.avg.pct_elapsed:                 0.079810
  lts__throughput.avg.pct_elapsed:                      0.173428

dominant issue-stall ratios per issue-active:
  wait:              2.833684
  barrier:           2.922031
  short scoreboard:  0.881129
  long scoreboard:   0.493230
```

Interpretation:

```text
The load path is no longer the old synchronous LDG/STS path: LDGSTS is zero and
TMA loads are present. The kernel is also not DRAM-bandwidth-bound. The
remaining profile matches the structural diff: phase barriers and shared-memory
handoffs dominate because load, MMA, softmax, correction, and epilogue are still
collapsed into one phase-loop owner flow.
```

Current alignment with `SM120_NVFP4_SM100_MAINLOOP_DIFF.md`:

```text
implemented:
  independent Q/K/V pipelines
  independent Q/K/V pipeline storage
  register-resident Q across the KV loop
  NVFP4 data + SFA/SFB sidecar plumbing

not implemented:
  SM100-style role decomposition
  SM100 load issue order Q1,K1,Q2,V1,K2,V2,...
  MMA-to-softmax pipeline
  separate softmax role
  separate correction role
  production kernel/collective stack
```

SM120-specific constraints for the next role port:

```text
1. SM100 hands S/P through TMEM. SM120 has no TMEM, so S/P handoff must use
   shared memory, registers owned by one role, or HBM. Registers cannot hand
   data across roles. HBM reintroduces the rejected score round-trip. The viable
   path is a shared-memory S/P pipeline.

2. The current compact D512 storage is already 97280 bytes with only 4096 bytes
   of SM120 opt-in shared-memory margin. A full 128x128 BF16 logits tile costs
   32768 bytes, so an independent double-buffered S pipeline does not fit unless
   some existing region is aliased by a proved lifetime or the handoff tile is
   reduced.

3. The current compact kernel aliases QK smem_B as the BF16 logits tile after K
   is consumed. That is correct for the phase-loop scaffold, but it blocks true
   SM100-style overlap: the next K tile cannot occupy the same B storage while
   a softmax role is still reading logits from it.

4. P staging into the SM120 block-scaled PV A/SFA layout currently uses the
   CUTLASS StageCount<2> copy atom and expects the 256-thread MMA producer
   mapping. A separate softmax role can own P quantization only if its local
   thread mapping writes through the same `partition_D` convention, or the P
   writer is split into a dedicated 256-thread role.

5. The SM100 16-warp schedule does not map one-to-one. SM100 has a one-warp
   MMA control role because TCGEN/TMEM carries the tensor work. The SM120
   `mma.sync` block-scaled CUTLASS atom consumes a 256-thread MMA group, so the
   SM120 role schedule must budget 8 warps for MMA in addition to load,
   softmax, correction, and epilogue roles.
```

Next implementation direction:

```text
Stop tuning the compact phase-loop kernel. Keep it as a correctness scaffold.
The next kernel slice should define an explicit SM120 role schedule and an
S/P handoff strategy before moving more math:

  Load role:
    owns Q/K/V descriptor prefetch and issues Q1,K1,Q2,V1,K2,V2,...

  MMA role:
    owns the 256-thread SM120 block-scaled MMA group, holds Q fragments in
    registers, consumes K/V pipelines, and produces S/P handoff signals.

  Softmax role:
    consumes S/logits from an explicit pipeline, computes row_m/row_l, and
    writes P/SFA through a partition-derived SM120 PV-A producer mapping.

  Correction role:
    owns global_m/global_l, old_scale/tile_scale, and O rescale.

  Epilogue role:
    owns final normalization/output write and later the production BF16/LSE
    contract.
```

### SM120 Role Schedule Contract

2026-04-28T18:25:00-05:00

Added an explicit SM120 role schedule to the benchmark extension:

```text
role order mirrors SM100:
  warps  0-3:  Softmax0
  warps  4-7:  Softmax1
  warps  8-11: Correction
  warps 12-19: MMA
  warp     20: Load
  warp     21: Epilogue

total:
  22 warps
  704 threads
```

Why this differs from SM100:

```text
SM100 uses one MMA control warp because TCGEN/TMEM owns the tensor work.
SM120 uses `mma.sync.aligned.kind::mxf4nvf4.block_scale...` through the
CUTLASS block-scaled collective, and that collective consumes 256 MMA
participant threads. The SM120 schedule therefore expands only the MMA role
from 1 warp to 8 warps while preserving the SM100 role order around it.
```

Validation:

```text
command:
  CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/usr/local/cuda-13.2
  python benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py
    --device 0 --sm120-role-schedule-check-only

result:
  softmax0 warps: 4
  softmax1 warps: 4
  correction warps: 4
  mma warps: 8
  load warps: 1
  epilogue warps: 1
  empty warps: 0
  mma threads: 256
  total threads: 704
```

Storage feasibility exposed by the same metadata:

```text
current compact Q/K/V online storage:       97280 bytes
SM120 opt-in shared-memory margin:          4096 bytes
128x128 BF16 logits tile:                  32768 bytes
double-buffered independent 128x128 S gap: -61440 bytes
double-buffered independent  64x128 S gap: -28672 bytes
```

Conclusion:

```text
The role split is now a compile-checked contract. A separate independent
S/logits buffer is impossible with the current compact D512 storage. The next
implementation step has to make the S/P pipeline use an explicitly proven
aliasing lifetime, not add side buffers. The viable first target is a
role-decomposed one-tile handoff where:

  Load issues Q1,K1,Q2,V1,K2.
  MMA consumes Q/K/V and writes S into aliased QK smem_B.
  Softmax0/1 consume S and write P into aliased QK smem_A/SFA.
  MMA consumes P/V for PV.

That still does not solve multi-tile overlap, but it moves the code from
block-wide phase ownership to role-specific ownership without changing the
storage footprint.
```

### Dead Kernel Cleanup And Active Gate Sweep

2026-04-28T19:05:00-05:00

Deleted the superseded benchmark kernels and public entry points:

```text
removed:
  persistent_mainloop_owner_layout_smoke
  sm120_nvfp4_qkv_handoff_stage
  persistent_mainloop_owner_qk_stage
  persistent_mainloop_owner_softmax_stage
  persistent_mainloop_owner_pv_group_stage
  persistent_mainloop_owner_full_tile_stage
  persistent_mainloop_owner_group_online_stage
  persistent_mainloop_owner_group_online_register_stage
  persistent_mainloop_owner_group_online_register_q_stage
  old split-reduction kernels
  old non-stage K128 PV helper layer
  old non-role softmax/P staging helpers

kept:
  atom-level QK/PV gates
  SM120 role schedule gate
  QK load collective gate
  QKV load collective gate
  role-decomposed one-tile handoff gate
  compact online register-Q scaffold
  CUTLASS runner baseline hook
```

Compile surface changed from:

```text
CUDA benchmark source: 7282 lines -> 3510 lines
Python harness:        1200 lines -> 858 lines
```

Validation after cleanup:

```text
commands:
  --sm120-role-schedule-check-only
  --sm120-qkv-load-collective-check-only
  --sm120-qkv-role-handoff-check-only
  --sm120-qkv-online-check-only --online-kv-tiles 2

results:
  role schedule: pass
  QKV load collective: finite, mean_abs=0.0015516469720751047,
                       max_abs=0.015534400939941406,
                       cosine=0.9999986290931702
  role handoff: finite, mean_abs=0.0025383096653968096,
                max_abs=0.014019200578331947,
                cosine=0.9895987510681152
  online register-Q: finite, mean_abs=0.00184237165376544,
                     max_abs=0.009134171530604362,
                     cosine=0.9872949719429016
```

Next structural milestone:

```text
Port the multi-tile online register-Q scaffold from the old 384-thread
producer/consumer shape to the explicit 704-thread SM120 role schedule. This
is the first multi-tile kernel where load, MMA, softmax, correction, and
epilogue roles exist in the same block shape as the SM100 reference. The
storage constraint remains unchanged: S/P must alias QK smem_B/A by lifetime.
```

### Online Register-Q Port To Explicit Role Schedule

2026-04-28T16:32:33-05:00

Ported `sm120_nvfp4_qkv_online_register_q_stage_kernel` from the old
384-thread producer/consumer warpgroup shape to the explicit 704-thread SM120
FMHA role schedule:

```text
roles:
  Softmax0/Softmax1: row stats and P quantization
  Correction:        online max/sum correction state
  MMA:               register-resident Q, QK MMA, PV MMA
  Load:              Q/K/V TMA producer
  Epilogue:          reserved in the block shape
```

Implementation changes:

```text
online launch shape:   384 threads -> 704 threads
Q residency:           unchanged, Q fragments stay in registers across tiles
softmax/correction:    moved out of block-wide helpers into role-owned helpers
S/P storage:           still aliases QK smem_B/A by lifetime
old phase barriers:    replaced with role barriers:
  MMA+Softmax+Load:    protects aliased logits from K reload overwrite
  Softmax+Correction:  transfers row_m/row_l to correction
  Softmax+MMA+Load:    protects P staging before PV / next-K issue
```

Important correctness bug found during the port:

```text
The first role-scheduled version let the Load role skip the MMA->Softmax and
Softmax->MMA barriers. Because K staging and logits both alias qk_sB, the Load
role could issue the next K tile after the K pipeline consumer released the
stage but before softmax consumed the current logits. That produced finite but
wildly incorrect output:

  mean_abs = 1.1700510711338762e+18
  max_abs  = 8.43750005914374e+19
  cosine   = 0.0

Fix: include the Load role in the handoff barriers around the aliased qk_sB
lifetime. The Load role can still stream ahead after P is staged, but it cannot
overwrite qk_sB while it contains live logits.
```

Validation after the structural port:

```text
commands:
  git diff --check
  python3 -m py_compile benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py
  --sm120-role-schedule-check-only
  --sm120-qkv-load-collective-check-only
  --sm120-qkv-role-handoff-check-only
  --sm120-qkv-online-check-only --online-kv-tiles 2

results:
  role schedule: pass
  QKV load collective: finite, mean_abs=0.0015516469720751047,
                       max_abs=0.015534400939941406,
                       cosine=0.9999986290931702
  role handoff: finite, mean_abs=0.0025383096653968096,
                max_abs=0.014019200578331947,
                cosine=0.9895987510681152
  online register-Q role schedule: finite,
                                   mean_abs=0.00184237165376544,
                                   max_abs=0.009134171530604362,
                                   cosine=0.9872949719429016
```

Next structural milestone:

```text
Use the explicit role schedule as the active online scaffold, then remove the
remaining helper-call-per-tile behavior from the online path. The next target
is an explicit MMA/softmax/correction/PV state machine that keeps the role
handoff structure but reduces the per-tile phase serialization still present in
the scaffold.
```

### Online Scaffold Timing And V-Consumption Move

2026-04-28T16:37:42-05:00

Measured the role-scheduled online scaffold before the next structural change:

```text
kv_tiles=2:   0.154336 ms min, 0.156764 ms mean
kv_tiles=8:   0.516384 ms min, 0.533705 ms mean
kv_tiles=32:  1.967456 ms min, 2.012096 ms mean
```

Two-stage CUTLASS reference in the same extension:

```text
QK 128x32768:        0.022848 ms min, 0.023870 ms mean
PV 128x512 k32768:   0.131328 ms min, 0.132813 ms mean
QK 4096x32768:       0.222528 ms min, 0.224521 ms mean
PV 4096x512 k32768:  0.159552 ms min, 0.162362 ms mean
```

Conclusion:

```text
The role-scheduled scaffold is correct but still structurally too slow. It
scales roughly linearly with KV tiles because the inner loop is still a
helper-call-per-tile sequence. The next optimization cannot be a small barrier
tweak; the remaining work is replacing that sequence with a real persistent
mainloop state machine.
```

Applied one structural ordering fix from the SM100 pattern:

```text
Before:
  MMA role waited on/copied V into registers before QK.

After:
  MMA role performs QK first, softmax/correction/P staging next, then waits on
  V only at the PV point.
```

Rationale:

```text
V is independent of QK logits. Waiting on V before QK serializes V arrival in
front of the critical QK->softmax path and extends V fragment live ranges. The
SM100 structure lets QK and softmax advance while V is in flight, then consumes
V at PV.
```

Validation after moving V consumption:

```text
role schedule: pass
QKV load collective: finite, mean_abs=0.0015516469720751047,
                     max_abs=0.015534400939941406,
                     cosine=0.9999986290931702
role handoff: finite, mean_abs=0.0025383096653968096,
              max_abs=0.014019200578331947,
              cosine=0.9895987510681152
online register-Q, kv_tiles=2: finite,
                                   mean_abs=0.00184237165376544,
                                   max_abs=0.009134171530604362,
                                   cosine=0.9872949719429016
online register-Q, kv_tiles=32: finite,
                                    mean_abs=0.0004577414656523615,
                                    max_abs=0.002065679058432579,
                                    cosine=0.9896405935287476
                                    min_ms=1.921056
```

Effect:

```text
kv_tiles=32 improved from 1.967456 ms -> 1.921056 ms min. This is only a small
win, but it confirms the corrected V lifetime is safe and removes one
avoidable serialization point before the larger mainloop rewrite.
```

### Full Shape-B Grid Launch Surface

2026-04-28T16:41:24-05:00

Added `sm120_nvfp4_qkv_online_register_q_full_grid`, which launches the
role-scheduled online owner over the full Shape-B output tile grid:

```text
grid:
  q tiles:      4096 / 128 = 32
  output groups: 512 / 128 = 4
  CTAs total:   128
  kv tiles:     32768 / 128 = 256 per CTA
output:
  float32 [4096, 512] lab artifact
```

Correctness gate:

```text
first output tile vs exact reference:
  finite:   true
  mean_abs: 0.00015529035590589046
  max_abs:  0.0009055592236109078
  cosine:   0.9929453134536743
```

Timing:

```text
full Shape-B role owner:
  min_ms:  15.708160
  mean_ms: 15.727123
  max_ms:  15.776544
```

Conclusion:

```text
The full-grid launch surface confirms the scaffold is correct but far from
the shipping target. The gap to the two-stage CUTLASS ceiling (~0.634 ms) is
~25x. This is not a launch-grid problem anymore; it is the inner mainloop.
Each CTA still executes 256 helper-call KV iterations with multiple named
barriers and QK/PV helper boundaries. The next optimization target is the
inner loop itself.
```

### Concurrent Role-Loop Mainloop Rewrite

2026-04-28T16:56:44-05:00

Profile input:

```text
Nsight Compute on the full-grid role owner:
  tensor pipe:      2.46%
  issue slots busy: 19.61%
  barrier stalls:   11.19 cycles / issued instruction
  no eligible:      70.83%
```

Conclusion from the profile:

```text
Barrier stalls dominate. The correct next move is not deleting one handoff at a
time. The online owner needs a role-owned mainloop where each role progresses
through its own loop and synchronizes only at real data-dependency boundaries.
```

Structural change:

```text
Rewrote sm120_nvfp4_qkv_online_register_q_stage_kernel from one serial
role-conditional tile loop into role-specific concurrent loop bodies:

  Load role:
    Q1, K1, Q2, V1, K2 initial issue
    then waits for P-ready before issuing next K/V tile

  MMA role:
    loads Q into register fragments once
    loops QK -> logits handoff -> P-ready wait -> V wait -> PV
    writes the lab float output from PV accumulator

  Softmax role:
    waits for logits
    computes row_m/row_l
    hands stats to correction
    waits for correction scales
    stages P into aliased qk_sA/SFA
    signals P-ready

  Correction role:
    waits for row stats
    updates global_m/global_l and old/tile scales
    signals corrected scales

  Epilogue role:
    still reserved in the block shape. Output remains MMA-owned because the
    current lab kernel has no separate O handoff storage; adding that is a
    later production-epilogue milestone.
```

Remaining synchronization points after the rewrite:

```text
MMA -> Softmax:       logits in aliased qk_sB are valid
Softmax -> Correction: row_m/row_l are valid
Correction -> Softmax: old/tile scales are valid
Softmax -> MMA/Load: P in aliased qk_sA/SFA is valid and logits lifetime ended
```

Validation:

```text
git diff --check: pass
python py_compile bench harness: pass
role schedule: pass
QKV load collective: finite, mean_abs=0.0015516469720751047,
                     max_abs=0.015534400939941406,
                     cosine=0.9999986290931702
role handoff: finite, mean_abs=0.0025383096653968096,
              max_abs=0.014019200578331947,
              cosine=0.9895987510681152
online register-Q, kv_tiles=2: finite,
                                   mean_abs=0.00184237165376544,
                                   max_abs=0.009134171530604362,
                                   cosine=0.9872949719429016
full-grid first tile vs exact: finite,
                               mean_abs=0.00015529035590589046,
                               max_abs=0.0009055592236109078,
                               cosine=0.9929453134536743
```

Performance effect:

```text
full Shape-B role owner before role-loop rewrite:
  min_ms: 15.708160

after concurrent role-loop rewrite:
  min_ms: 12.918880
  mean_ms: 12.950957
  max_ms: 12.976832

speedup: 1.216x
```

Conclusion:

```text
The rewrite is a real structural improvement and validates the concurrent role
ownership direction. It is still ~20x slower than the two-stage CUTLASS
ceiling, so the next milestone must remove the helper-call-per-tile QK/PV
boundaries inside the MMA loop and inline the CUTLASS atom copy/MMA sequence
directly into that role loop.
```

### Helper Tail-Barrier Removal In Online Role Loop

2026-04-28T17:00:23-05:00

The role-loop rewrite still called QK/PV helpers that ended with conservative
`Sm120MainloopBarrier` synchronizations. Those barriers are useful for the
older smoke gates, but they are helper-boundary barriers in the online
role-loop path:

```text
QK helper:
  per-stage consumer_wait/copy/gemm/release still retained
  trailing helper-exit barrier made optional

PV helper:
  P/V copy and GEMM still retained
  trailing helper-exit barrier made optional
```

Implementation:

```text
cutlass_qk_tma_k_mma_register_q_stage<false>(...) in online MMA role
cutlass_pv_stage2_mma_register_v_stage<false>(...) in online MMA role

default remains <true>, so older gates keep the conservative behavior.
```

Validation:

```text
git diff --check: pass
role schedule: pass
QKV load collective: finite, mean_abs=0.0015516469720751047,
                     max_abs=0.015534400939941406,
                     cosine=0.9999986290931702
role handoff: finite, mean_abs=0.0025383096653968096,
              max_abs=0.014019200578331947,
              cosine=0.9895987510681152
online register-Q, kv_tiles=2: finite,
                                   mean_abs=0.00184237165376544,
                                   max_abs=0.009134171530604362,
                                   cosine=0.9872949719429016
full-grid first tile vs exact: finite,
                               mean_abs=0.00015529035590589046,
                               max_abs=0.0009055592236109078,
                               cosine=0.9929453134536743
```

Performance:

```text
role-loop baseline:
  min_ms: 12.918880

helper tail-barriers removed:
  min_ms: 12.896800
  mean_ms: 12.920973
  max_ms: 12.945888
```

Conclusion:

```text
This confirms the helper-exit barriers are not the dominant problem. The next
target is the helper body itself: the online MMA role still constructs/copies
fragment views and runs separate QK0, QK1, V-load, and PV helper bodies per KV
tile instead of one fused CUTLASS atom loop.
```

### Inline QK/PV CUTLASS Atom Bodies In MMA Role

2026-04-28T17:16:12-05:00

The post-tail-barrier profile still showed the same structural bottleneck:

```text
report:
  reports/ncu_sm120_online_fullgrid_role_loop_a0ff781.ncu-rep

tensor pipe active:             4.48% active / 2.98% elapsed
issue slots busy:               22.05%
SM busy:                        22.05%
instructions:                   4.969B
memory throughput:              119.89 GB/s
L2 hit rate:                    99.81%
eligible warps / scheduler:     0.38
no eligible cycles:             66.84%
stall barrier:                  10.07 cycles / issued instruction
stall wait:                     2.22
stall long scoreboard:          1.48
stall short scoreboard:         0.39
warp cycles / issued inst:      15.83
```

Interpretation:

```text
The role-loop rewrite reduced work, but the CTA still spends most cycles with
no eligible warp. Tail barrier removal alone was correctly rejected as too
small. The next structural change is to remove the helper-call-per-tile shape
inside the MMA role and make the CUTLASS atom sequence explicit in the role
loop.
```

Implementation:

```text
The online owner no longer calls these helpers from the MMA role loop:
  cutlass_qk_tma_k_mma_register_q_stage<false>
  cutlass_pv_stage2_tma_v_register_stage
  cutlass_pv_stage2_scale_or_clear_accum
  cutlass_pv_stage2_mma_register_v_stage<false>

The MMA role now hoists the CUTLASS copy/MMA views once, then performs the
sequence inline:
  QK K-stage wait/copy/SFB-copy/fp4-shift
  cute::gemm(Q, K, qk_accum)
  logits write to the aliased BF16 score scratch
  MmaSoftmax -> SoftmaxMma role handoff
  V-stage wait/copy/SFB-copy/fp4-shift
  online PV accumulator rescale/clear
  P copy/SFA-copy/fp4-shift
  cute::gemm(P, V, pv_accum)

The reusable helpers remain for older atom/load/role-handoff gates only. The
online owner is now the first path where the QK/PV atom bodies live directly in
the role-owned mainloop.
```

Validation:

```text
git diff --check: pass
python py_compile bench harness: pass
role schedule: pass
QKV load collective: finite, mean_abs=0.0015516469720751047,
                     max_abs=0.015534400939941406,
                     cosine=0.9999986290931702
role handoff: finite, mean_abs=0.0025383096653968096,
              max_abs=0.014019200578331947,
              cosine=0.9895987510681152
online register-Q, kv_tiles=2: finite,
                                   mean_abs=0.00184237165376544,
                                   max_abs=0.009134171530604362,
                                   cosine=0.9872949719429016
full-grid first tile vs exact: finite,
                               mean_abs=0.00015529035590589046,
                               max_abs=0.0009055592236109078,
                               cosine=0.9929453134536743
```

Performance:

```text
helper-tail-trim baseline:
  min_ms: 12.896800
  mean_ms: 12.920973
  max_ms: 12.945888

inline QK/PV atom bodies:
  min_ms: 12.777376
  mean_ms: 12.878765
  max_ms: 12.951488
```

Conclusion:

```text
Correctness is preserved and the inline atom sequence removes the helper
call-site boundary, but the performance gain is still only ~0.9% on min wall
time. This means the dominant barrier/no-eligible behavior is not simply helper
view setup or helper tail sync. The next structural target must change the
mainloop dataflow itself: remove the smem logits round-trip / block-wide
MMA->softmax handoff by moving the QK accumulator into a register/logit handoff
that feeds softmax/P quantization without materializing the full 128x128 BF16
score tile through shared memory.
```

### Rejected: Direct PipelineAsync Replacement For CTA-Wide Handoffs

2026-04-28T17:24:41-05:00

Tried replacing the four online-owner CTA-wide role handoffs with one-stage
`cutlass::PipelineAsync<1>` barriers:

```text
MMA -> Softmax score-ready
Softmax -> Correction row_m/row_l-ready
Correction -> Softmax old_scale/tile_scale-ready
Softmax -> MMA/Load P-ready and logits-lifetime-ended
```

The implementation kept the same full 128x128 BF16 logits scratch in aliased
QK shared memory and used role-local named barriers only before committing or
releasing a pipeline stage.

Validation:

```text
online register-Q, kv_tiles=2: finite,
                                   mean_abs=0.00184237165376544,
                                   max_abs=0.009134171530604362,
                                   cosine=0.9872949719429016
full-grid first tile vs exact: finite,
                               mean_abs=0.00015529035590589046,
                               max_abs=0.0009055592236109078,
                               cosine=0.9929453134536743
```

Performance:

```text
inline QK/PV atom-body baseline:
  min_ms: 12.777376
  mean_ms: 12.878765
  max_ms: 12.951488

direct PipelineAsync handoffs:
  min_ms: 12.985120
  mean_ms: 13.005005
  max_ms: 13.024000
```

Decision:

```text
Rejected and reverted. This proves the barrier primitive is not the main issue
while the dataflow still materializes a full score tile and blocks K loading on
the same shared-memory region. Mbarriers add overhead but create no overlap
because there is still only one score/logits lifetime and one aliased storage
region.

The next structural target is not another handoff primitive swap. It must
remove or shrink the full-tile BF16 score handoff itself: either direct
QK-accumulator-to-P staging inside the MMA/softmax pipeline, or a smaller
subtile score/P staging scheme that actually permits overlap without requiring
an impossible second 128x128 BF16 logits buffer.
```

### Course Correction: Port Structure, Do Not Re-Derive It

2026-04-28T17:38:37-05:00

The current SM120 prototype has ported the SM120 block-scaled atom mechanics and
some local conventions from the references, but it has not ported the full
Example 77 / SM100 FMHA mainloop structure. That distinction is now explicit and
load-bearing.

Ported so far:

```text
CUTLASS/CuTe SM120 block-scaled NVFP4 MMA atoms
TMA partition_D / partition_S producer convention for Q/K/V
register-resident Q after Q TMA load
separate nominal roles for load, MMA, softmax, correction, epilogue
```

Not yet ported:

```text
SM100-style pipeline storage topology:
  load_q, load_kv, mma_s0, mma_s1, s0_corr, s1_corr,
  mma_corr, corr_epi, order_s01

SM100 method/lifetime topology:
  load(...) -> mma(...) -> softmax(...) -> correction(...) -> epilogue(...)

SM100/Example 77 issue order:
  Q1, K1, Q2, V1, K2, V2, ...

Example 88 accumulator/logit handoff:
  QK accumulator -> online softmax/P staging -> PV
  without a full BF16 score-tile smem round-trip
```

Code change in this checkpoint:

```text
Added the SM120 analogue of the SM100 pipeline-storage contract:
  Sm120Nvfp4PipelineS = cutlass::PipelineAsync<1>
  Sm120Nvfp4PipelineC = cutlass::PipelineAsync<1>
  Sm120Nvfp4PipelineO = cutlass::PipelineAsync<2>
  Sm120Nvfp4PipelineE = cutlass::PipelineAsync<2>
  Sm120Nvfp4OrderBarrierSoftmax = cutlass::OrderedSequenceBarrier<1, 2>

Added Sm120Nvfp4MainloopPipelineStorage with:
  mma_s0, mma_s1, s0_corr, s1_corr, mma_corr, corr_epi, order_s01

Embedded that storage in Sm120Nvfp4QkvLoadCollectiveStorage and exposed its
size in the metadata path.
```

This is not a performance optimization and should not be evaluated as one. It
is the structural anchor for the next port slice: replacing the loose
benchmark-local named-barrier handoffs with the same pipeline-state ownership
model that Example 77 uses.

### Rejected: MMA-Owned Softmax/Correction Collapse

2026-04-28T17:38:37-05:00

Tried collapsing softmax, correction, P staging, and PV ownership into the MMA
role after the inline atom-body rewrite. Softmax/correction roles were idled;
the MMA role computed row stats, online correction, P quantization, and PV
directly.

Validation passed, but full-grid performance regressed badly:

```text
inline QK/PV atom-body baseline:
  min_ms: 12.777376
  mean_ms: 12.878765

MMA-owned softmax/correction/P/PV:
  min_ms: 15.059456
  mean_ms: 15.149126
  max_ms: 15.228544
```

Decision:

```text
Rejected and reverted. Collapsing roles reduces handoff count but destroys the
role separation that Example 77 relies on for overlap and register lifetime.
The correct direction is not "fewer roles"; it is the SM100 role pipeline
topology ported to SM120 mma.sync/register mechanics.
```

### Rejected: Move BF16 Logits Scratch To smem_A

2026-04-28T17:38:37-05:00

Considered moving `smem_logits` from aliased QK `smem_B` to `smem_A` to free the
K destination earlier and remove the load-role wait. This is invalid in the
current storage model:

```text
QK smem_A is also the PV A/P staging region.
Softmax would read logits from smem_A and write P into the same storage.
```

Decision:

```text
Rejected before testing. This is a storage-lifetime conflict, not a scheduling
tweak. The fix must come from the Example 77/88 structure: accumulator/subtile
handoff or a correctly pipelined S/P lifetime, not by moving the full score tile
onto another aliased operand buffer.
```

### Rejected: V Lookahead Without Matching Pipeline Ownership

2026-04-28T17:38:37-05:00

Tried changing only the loader issue order so the prologue staged `V0` and `V1`,
then each loop staged next K and `V+2`. This was intended to move toward Example
77's independent K/V issue order without first porting the full pipeline
ownership model.

Result:

```text
online register-Q correctness gate did not fail at compile time;
the process hung in the run path with no nvcc/ptxas active.
```

Decision:

```text
Rejected and reverted. Moving V issue order alone violates the current
consumer/release lifetime. Example 77's K/V interleave is coupled to its
PipelineKV state machine and S/P/O ownership. The next attempt must port the
pipeline-state ownership together with the issue order, not move V lookahead as
an isolated schedule edit.
```

### Full SM100-Style Role Pipeline Port

2026-04-28T18:26:00-05:00

Ported the active online owner from a partial S/C pipeline hybrid to the full
SM100-style role pipeline graph for the benchmark kernel:

```text
MMA -> Softmax0:      pipeline_mma_s0
MMA -> Softmax1:      pipeline_mma_s1
Softmax0 -> Corr:     pipeline_s0_corr
Softmax1 -> Corr:     pipeline_s1_corr
MMA -> Corr:          pipeline_mma_corr
Corr -> Epilogue:     pipeline_corr_epi
Softmax0/1 ordering:  order_s01
```

The S and C pipelines now use role-specific ownership rather than both softmax
groups consuming every tile:

```text
even tiles:  MMA -> Softmax0 -> Correction
odd tiles:   MMA -> Softmax1 -> Correction
```

`pipeline_s0_corr` and `pipeline_s1_corr` now use producer arrival counts that
match the softmax warpgroups, following the SM100 reference. `pipeline_mma_corr`
is committed by the MMA role after PV, then consumed by the correction role.
`pipeline_corr_epi` is committed by correction on the final tile and consumed by
the epilogue role.

Important benchmark limitation:

```text
The benchmark still writes the float output from the MMA role because the
current shared-memory budget cannot hand a full 128x128 output tile to the
epilogue role. The E pipeline is structural ownership plumbing in this benchmark
path, not the final production epilogue.
```

Validation:

```text
git diff --check: pass
python3 -m py_compile benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py: pass
online kv_tiles=2:  finite, mean_abs=0.00184237, max_abs=0.00913417, cosine=0.987295
online kv_tiles=4:  finite, mean_abs=0.00125708, max_abs=0.00587741, cosine=0.988742
online kv_tiles=16: finite, mean_abs=0.000650447, max_abs=0.00310903, cosine=0.988928
full grid:          finite first tile, mean_abs=0.00015529, max_abs=0.000905559, cosine=0.992945
full-grid min:      13.2259 ms
```

Conclusion:

```text
The full S/C/O/E/order pipeline graph removes the previous long-run hang and
makes role ownership match SM100 more closely, but it does not improve wall
time. The current performance is still far from the two-stage CUTLASS target.

The remaining gap is not missing O/E/order plumbing. The remaining structural
gap is that the SM120 benchmark still lacks the SM100 mainloop's true overlap
model: S/P are aliased through one QK tensor region, output remains
MMA-register-owned, and the epilogue is only a structural placeholder.
```

### Epilogue-Owned Output Handoff

2026-04-28T19:05:00-05:00

Ported the next SM100 lifecycle layer in the active SM120 benchmark kernel:

```text
MMA role:
  keeps PV/O accumulator in registers across the KV loop
  stages final normalized O to aliased shared memory only on the final tile
  commits pipeline_mma_corr only after the final O smem write is complete

Correction role:
  consumes pipeline_mma_corr
  commits pipeline_corr_epi on the final tile

Epilogue role:
  waits pipeline_corr_epi
  owns the global output write
```

The aliased shared-memory region is `qk_tensors.smem_B`:

```text
K tile storage while QK consumes K
BF16 score/logit scratch after QK
BF16 O epilogue tile after final PV
```

This replaces the previous benchmark shortcut where the MMA role wrote global
output directly after the KV loop. The output handoff now follows the SM100
ownership model within SM120's no-TMEM constraint.

Validation:

```text
git diff --check: pass
python3 -m py_compile benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py: pass
online kv_tiles=2:  finite, mean_abs=0.00184219, max_abs=0.00911981, cosine=0.987296
online kv_tiles=16: finite, mean_abs=0.000650525, max_abs=0.00309772, cosine=0.988925
full grid:          finite first tile, mean_abs=0.000155273, max_abs=0.000907625, cosine=0.992944
full-grid min:      14.1334 ms
```

Conclusion:

```text
The handoff is correct, but the benchmark is slower because it now pays an
extra BF16 shared-memory write/read epilogue path. This was expected for the
structural port: it removes the invalid MMA-global-write shortcut but does not
yet create the SM100 overlap model that hides the cost.

The next missing layer from Example 77 is still the S/P alias and softmax/PV
interleave. The current SM120 path aliases K/S/O in `smem_B` and aliases Q/P in
`smem_A`; it does not yet alias score storage and P staging in the same
producer/consumer lifetime. A direct same-base BF16-logit -> FP4-P in-place
rewrite is unsafe because the FP4 P writer can clobber BF16 logits that another
softmax lane has not read. The correct port is to remove the full BF16 score
tile as a durable object and stage P through the same lifetime boundary that
77 uses for S/P, not to add another side buffer.
```

Rejected follow-up:

```text
Tried to port SM100-style setmaxnreg role reconfiguration for the active
SM120 role kernel. The extension built, but the runtime gate hung with no
nvcc/ptxas/ninja active, so the process was in-kernel. The port was removed.

Interpretation: copying SM100's register reconfiguration mechanically is not
valid for this SM120 benchmark shape yet. The SM120 CTA has 22 warps, an
8-warp register-accumulator MMA role, and a partial load/epilogue warpgroup;
the SM100 path uses TMEM for O and a different role/register pressure model.
Register reconfiguration should be revisited only after the S/P/PV lifetime
matches the reference more closely.
```

### Compact A/SFA P Double-Buffer And Final-Only O Handoff

2026-04-28T20:10:00-05:00

Ported the next S/P lifetime correction in the active SM120 benchmark kernel.
The failed first attempt used two aliased S/P slots:

```text
stage 0: qk_tensors.smem_B / smem_SFB
stage 1: qk_tensors.smem_A / smem_SFA
```

That compiled but produced huge incorrect values for `kv_tiles=2`. The reason
is structural: `smem_B` is owned by the K TMA pipeline. It can hold the BF16
score tile only after QK consumes K for the current tile, but it cannot hold a
durable P tile across the next QK tile because later K loads overwrite the same
region before PV consumes the previous tile.

Corrected lifetime model:

```text
qk_tensors.smem_B:
  transient BF16 score tile after current QK
  final BF16 O epilogue tile after final PV

qk_tensors.smem_A:
  compact PV P stage 0
  compact PV P stage 1

qk_tensors.smem_SFA:
  compact PV P scale stage 0
  compact PV P scale stage 1
```

`smem_A` is free after Q is resident in registers. The compact PV P stage is
16 KiB and the scale stage is 2 KiB, so two P stages fit in the existing
32 KiB A region and 4 KiB SFA region without increasing shared-memory usage.

Pipeline corrections:

```text
MMA/QK:
  commit score tile
  wait until softmax has converted score -> compact P
  only then release K pipeline storage

MMA/PV:
  consume compact P from A/SFA stage selected by tile parity
  keep O accumulator in registers
  use pipeline_mma_corr only for the final O handoff

Correction:
  updates double-buffered row/global scale state for every tile
  no longer waits on pipeline_mma_corr for non-final tiles
  waits on final O handoff only before committing pipeline_corr_epi

Softmax:
  uses tile-parity workers with distinct P buffers
  does not use the SM100 order_s01 barrier in this adapted path, because the
  two SM120 softmax workers do not both participate in the same score tile
```

Deadlock diagnoses fixed during the port:

```text
1. Releasing K only after P staging exposed a load-order cycle:
   load K1[0] -> V1 -> K1[1]
   V1 could block behind V0, while MMA needed K1[1] before it could run PV0.
   The next-tile load order is now K1[0] -> K1[1] -> V1.

2. Correction waited on pipeline_mma_corr for tile 0 before processing tile 1.
   MMA waited for tile-1 P before running tile-0 PV, which created:
   correction0 waits PV0, softmax1 waits correction1, MMA waits softmax1.
   Non-final pipeline_mma_corr waits were removed.
```

Validation:

```text
git diff --check: pass
online kv_tiles=1:  finite, mean_abs=0.00253858, max_abs=0.0139102, cosine=0.989597
online kv_tiles=2:  finite, mean_abs=0.00184219, max_abs=0.00911981, cosine=0.987296
online kv_tiles=16: finite, mean_abs=0.000650525, max_abs=0.00309772, cosine=0.988925
full grid:          finite first tile, mean_abs=0.000155273, max_abs=0.000907625, cosine=0.992944
full-grid min:      14.7649 ms
```

Conclusion:

```text
The compact P double-buffer and final-only output handoff are correct, but the
current schedule is slower than the previous 14.1334 ms checkpoint. The
regression is expected from the safety constraint: QK now waits for
softmax/P staging before releasing K storage. That prevents K/logit clobbering,
but it serializes the next QK tile behind softmax.

The next structural port must remove the durable BF16 score tile from the
critical path, matching Example 77 more closely: softmax must consume the QK
result into registers and write compact P without forcing the next QK tile to
wait on a full score-tile lifetime in `smem_B`.
```

Rejected follow-up:

```text
Tried an SM120 substitution where the MMA role owned softmax row stats,
online correction state, compact P staging, and final epilogue commit. This
kept kv_tiles=2 finite but regressed kv_tiles=16 correctness:

MMA-owned softmax, 256 participating MMA threads:
  finite, mean_abs=0.00146469, max_abs=0.0130523, cosine=0.856394

MMA-owned softmax, first 128 MMA threads matching the old softmax group:
  finite, mean_abs=0.00105781, max_abs=0.0128941, cosine=0.925142

Adding an explicit shared-memory visibility fence after P staging did not fix
the regression:

  finite, mean_abs=0.00146297, max_abs=0.0131158, cosine=0.856958

Conclusion: the MMA-owned path is the regression. It collapses the role
decomposition, idles Softmax/Correction warps, and reintroduces active-path
NamedBarrier synchronization around work that Examples 77/88 keep in the
softmax/correction lifecycle. This path was reverted and should not be
debugged further.
```

Reference fact from rereading 77/88:

```text
Softmax0/Softmax1 own softmax state and P production. MMA owns QK/PV tensor
core work and waits for P readiness; it does not compute row stats. Correction
owns output rescale/final normalization before the epilogue store. Any SM120
substitution must preserve those ownership boundaries even though SM120 lacks
TMEM and must use shared memory/register fragments instead.
```

## Correction-Owned Final Output Normalization

Ported the final output ownership boundary back toward the SM100/Example 88
structure:

```text
before:
  MMA:
    PV accum registers
    divide by global_l
    stage normalized BF16 O into epilogue smem
    commit pipeline_mma_corr

  Correction:
    wait final O
    immediately commit pipeline_corr_epi

  Epilogue:
    store normalized BF16 O

after:
  MMA:
    PV accum registers
    stage raw scaled BF16 O into epilogue smem
    commit pipeline_mma_corr

  Correction:
    wait final O
    divide epilogue smem by global_l
    fence shared visibility
    commit pipeline_corr_epi

  Epilogue:
    store normalized BF16 O
```

This mirrors the SM100 `correction_epilogue` ownership boundary: Correction
materializes the final normalized output tile into epilogue storage, and
Epilogue owns the global store. It is not a performance optimization by itself;
it removes the previous MMA-owned normalization responsibility so subsequent
ports do not build on the wrong role boundary.

Validation:

```text
git diff --check: pass
online kv_tiles=16: finite, mean_abs=0.000650689, max_abs=0.00309772, cosine=0.988923
full grid first tile: finite, mean_abs=0.000155332, max_abs=0.000899995, cosine=0.992945
full-grid min: 14.8333 ms
```

Conclusion:

```text
Correctness is preserved. Runtime is neutral/slightly worse versus the compact-P
checkpoint (14.7649 ms -> 14.8333 ms), which is expected because this is an
ownership-boundary correction, not the critical-path fix. The active bottleneck
remains the durable BF16 score tile lifetime: QK cannot release K storage until
Softmax converts the score tile into compact P. The next structural port must
remove that full-score shared-memory lifetime or move to a direct
score-fragment-to-P handoff that preserves Softmax ownership.
```

## Softmax-Owned Online State Port

Ported online max/sum ownership from the per-tile Correction role into the
Softmax roles, matching the 88/SM100 softmax lifecycle more closely.

Previous active structure:

```text
MMA:
  write full BF16 score tile

Softmax0 or Softmax1, alternating by tile parity:
  compute tile row_m / row_l
  signal Correction
  wait Correction
  stage compact P using tile_scale from Correction

Correction, every tile:
  update global_m/global_l
  write old_scale/tile_scale

MMA:
  wait compact P
  rescale PV accumulator by old_scale
  run PV
```

New active structure:

```text
MMA:
  write full BF16 score tile

Softmax0 and Softmax1, both active on every tile:
  Softmax0 owns rows [0, 64)
  Softmax1 owns rows [64, 128)
  keep running_m/running_l in registers for owned rows
  compute old_scale/tile_scale directly
  stage compact P for owned rows
  write final global_l for Correction/Epilogue normalization

Correction:
  removed from the per-tile online-state path
  only waits final O and normalizes epilogue smem by final global_l
```

Implementation notes:

```text
- `pipeline_mma_s0` and `pipeline_mma_s1` now have both softmax roles as
  consumers, so consumer arrival count is 256 threads.
- Both softmax roles must be configured as consumers of both S pipelines. The
  first attempt left Softmax0 consuming only S0 and Softmax1 consuming only S1;
  kv_tiles=1 deadlocked until the role setup was corrected.
- The softmax row-state barrier and the P-scale-ready barrier use distinct
  named-barrier IDs. Reusing one barrier ID for back-to-back 256-thread phases
  is unsafe because one group can enter the next phase while the other is still
  draining the previous phase.
- The now-dead `s0_corr` / `s1_corr` pipelines and row-stat helper code were
  removed from the active kernel.
```

Validation:

```text
git diff --check: pass
online kv_tiles=1:  finite, mean_abs=0.00253812, max_abs=0.0139102, cosine=0.989603
online kv_tiles=16: finite, mean_abs=0.000615264, max_abs=0.00324988, cosine=0.988328
full grid first tile: finite, mean_abs=0.000155821, max_abs=0.000807697, cosine=0.992418
full-grid min: 11.2681 ms
```

Conclusion:

```text
This is the first structural port in this phase that materially moves runtime:
14.8333 ms -> 11.2681 ms, about a 24% reduction. The win comes from removing
the per-tile Correction round trip and making Softmax own the online state.

The kernel is still far from the two-stage CUTLASS ceiling. The remaining
dominant structural problem is unchanged: the full BF16 score tile still lives
in the K/B shared-memory region until compact P is produced, so K release and
next-tile QK remain blocked by score-tile consumption. The next port must target
the S/P lifetime itself, not the online-state math.
```

## Row-Half S Pipeline Ownership And Early K-Half Release

Changed `pipeline_mma_s0` / `pipeline_mma_s1` from tile-parity score ownership
to row-half score ownership, closer to the SM100 meaning of S0/S1.

Previous active structure:

```text
even tile:
  MMA commits pipeline_mma_s0 for the whole 128x128 score tile
  both softmax roles consume S0
  MMA waits P-ready
  MMA releases both K shared-memory stages

odd tile:
  same pattern through pipeline_mma_s1
```

New active structure:

```text
every tile:
  MMA commits pipeline_mma_s0 for rows [0, 64)
  MMA commits pipeline_mma_s1 for rows [64, 128)

  Softmax0 consumes S0 and stages compact P rows [0, 64)
  Softmax1 consumes S1 and stages compact P rows [64, 128)

  MMA reacquires S0 and releases K shared-memory half/stage 0
  MMA reacquires S1 and releases K shared-memory half/stage 1
```

This preserves the softmax-owned online state from the previous checkpoint, but
lets the loader begin the next tile's first K half as soon as Softmax0 has
finished converting the first score half into compact P. It does not eliminate
the BF16 score tile, but it narrows the lifetime fence from one full-tile K
release to two half-tile releases.

Implementation notes:

```text
- `pipeline_mma_s0` is consumed only by Softmax0 and has 128 consumer arrivals.
- `pipeline_mma_s1` is consumed only by Softmax1 and has 128 consumer arrivals.
- Softmax0 and Softmax1 use distinct named-barrier IDs for their internal
  row-state and P-scale phases, so the two row halves can run concurrently
  without sharing a named barrier.
- The P storage remains double-buffered by tile parity in `smem_A`/`smem_SFA`;
  S0/S1 only control readiness and K-half release.
```

Validation:

```text
git diff --check: pass
online kv_tiles=1:  finite, mean_abs=0.00253812, max_abs=0.0139102, cosine=0.989603
online kv_tiles=16: finite, mean_abs=0.000615264, max_abs=0.00324988, cosine=0.988328
full grid first tile: finite, mean_abs=0.000155821, max_abs=0.000807697, cosine=0.992418
full-grid min: 10.4330 ms
```

Conclusion:

```text
Runtime improved again: 11.2681 ms -> 10.4330 ms, about a 7.4% reduction.
The win is from releasing the two K shared-memory halves independently and
allowing the load role to overlap the next K0 load earlier. This confirms that
the remaining S/P lifetime is still on the critical path.

The score tile still occupies the K/B shared-memory region until each row half
is consumed, so the next structural target is reducing or eliminating the BF16
score tile storage itself. Candidate directions are row-strip streaming of S
into compact P, a smaller-M tile that can afford independent S/P storage, or a
direct score-fragment-to-P handoff that preserves Softmax ownership.
```

## Softmax-Owned P Scale Precompute

2026-04-28T21:20:00-05:00

Kept the active SM120 role decomposition intact and explicitly did not revive
the rejected MMA-owned softmax path:

```text
MMA:
  QK/PV tensor-core work only
  writes BF16 score tile
  waits for Softmax-owned compact P readiness

Softmax0/1:
  own online max/sum state
  own P scale-sidecar production
  own compact P staging

Correction/Epilogue:
  own final O normalization and global BF16 store
```

The port moved one more piece of the 88 softmax lifecycle into the softmax role:
P scale-sidecar production now happens during the same row-state pass that
computes `tile_l`, instead of making a separate scale pass over the BF16 score
tile inside `softmax_role_stage_logits_bf16_p_to_pv_smem_stage2`.

The active path now uses:

```text
pass 1: per-row tile_m
pass 2: per-row tile_l + per-16-column P scale bytes
barrier: P scales and row state visible to the softmax group
pass 3: partitioned compact P write
```

Previous active path:

```text
pass 1: per-row tile_m
pass 2: per-row tile_l
pass 3: per-16-column P scale bytes
barrier
pass 4: partitioned compact P write
```

Correctness fix included in the same structural port:

```text
Compact P should be exp(logit - tile_m) * tile_scale.
The helper therefore needs tile-local `tile_m`, not the running max `next_m`.
`global_l` still tracks the online state for final output normalization.
```

Validation:

```text
git diff --check: pass
online kv_tiles=16:
  finite, mean_abs=0.000650689, max_abs=0.00309772, cosine=0.988923

full grid first tile:
  finite, mean_abs=0.000155332, max_abs=0.000899995, cosine=0.992945

full-grid min:
  before BF16-output checkpoint: 10.3441 ms
  after softmax-owned scale precompute: 9.8332 ms
```

Conclusion:

```text
This is a valid forward structural step because it preserves role ownership and
removes one full score-tile read/exp pass plus one softmax-internal barrier.
Runtime improved about 4.9% (10.3441 ms -> 9.8332 ms).

The kernel is still dominated by the durable BF16 score-tile lifetime: K/B
shared memory cannot be reused until each score half has been converted to
compact P. The next structural target remains S/P lifetime reduction:
row-strip streaming, smaller-M independent S/P storage, or a direct
score-fragment-to-P mailbox that preserves Softmax ownership.
```

## BF16 Output Store

Changed the active online/full-grid reference kernel output from float32 to
BF16, matching the production target and removing a lab-only float global store.

Implementation:

```text
- Active `sm120_nvfp4_qkv_online_register_q_stage_kernel` now takes
  `__nv_bfloat16* out_group`.
- The epilogue role copies normalized BF16 values from epilogue smem directly
  to global output instead of converting to float.
- The single-tile and full-grid Python harnesses now allocate BF16 output for
  the active online kernel. Older rejected/debug kernels remain float output.
```

Validation:

```text
git diff --check: pass
online kv_tiles=16: finite, mean_abs=0.000615264, max_abs=0.00324988, cosine=0.988328
full grid first tile: finite, mean_abs=0.000155821, max_abs=0.000807697, cosine=0.992418
full-grid min: 10.3441 ms
```

Conclusion:

```text
Runtime improved modestly: 10.4330 ms -> 10.3441 ms, about 0.85%. The output
store was not the primary bottleneck, but BF16 output is the correct production
contract and should stay.
```

## Rejected: Early S0 Score Publication

2026-04-28T21:35:00-05:00

Tried splitting MMA score publication by row half:

```text
write rows [0, 64)
commit pipeline_mma_s0
write rows [64, 128)
commit pipeline_mma_s1
```

The intent was to let Softmax0 begin P staging while MMA wrote the second row
half, preserving the Softmax-owned role boundary and creating overlap without
collapsing softmax back into MMA.

Validation:

```text
online kv_tiles=16:
  finite, mean_abs=0.000650689, max_abs=0.00309772, cosine=0.988923

full grid first tile:
  finite, mean_abs=0.000155332, max_abs=0.000899995, cosine=0.992945

full-grid min:
  baseline after softmax-owned P scale precompute: 9.8332 ms
  early S0 score publication:                   9.8373 ms
```

Decision:

```text
Rejected and reverted. The extra MMA-side barrier cancels the small overlap
created by publishing S0 earlier. This confirms the remaining gap is not solved
by subdividing the existing full BF16 score-tile store with more barriers.

The next viable S/P lifetime change must remove or shrink the durable BF16
score tile itself, not add finer publication points around the same tile.
```

## Rejected: 4-Row Independent Score Mailbox

2026-04-28T21:50:00-05:00

Tried the first implementation that actually removed the BF16 score tile from
the K/B shared-memory region:

```text
score mailbox:
  2 roles x 4 rows x 128 columns x BF16 = 2048 bytes

QK/MMA:
  compute full QK tile
  release both K shared-memory stages immediately
  stream 4-row score strips to Softmax0/Softmax1 mailboxes

Softmax0/1:
  consume one 4-row strip per pipeline handoff
  keep per-strip running max/sum state in registers
  write compact P into the existing A/SFA double buffer
```

This compiled only by using the entire SM120 opt-in shared-memory budget:

```text
storage_bytes: 101376
storage_margin_bytes: 0
```

Validation:

```text
online kv_tiles=16:
  finite, mean_abs=0.00364356, max_abs=0.0125221, cosine=0.988535

full grid first tile:
  finite, mean_abs=0.00111393, max_abs=0.00456910, cosine=0.992592

full-grid min:
  baseline after softmax-owned P scale precompute: 9.8332 ms
  4-row score mailbox:                         66.1595 ms
```

Decision:

```text
Rejected and reverted. This does remove K/B score-tile lifetime, but it
replaces one tile-level S handoff with 32 score-strip handoffs per tile
(16 strips per softmax role). The handoff and barrier overhead dominate.

Conclusion for the next pass: independent score storage must be coarse enough
to amortize pipeline overhead. A tiny row mailbox is structurally wrong on
SM120. If we revisit independent S/P storage, it should be via a smaller-M CTA
or a larger independent S buffer, not 4-row strip streaming at M=128.
```

## QK / Softmax / PV Interleave Fix

2026-04-28T22:05:00-05:00

Found and fixed a real schedule bug in the active role pipeline.

Previous active order:

```text
for tile n:
  QK(n)
  publish S(n)
  wait Softmax(n) -> P(n)
  release K(n)
  PV(n-1)
```

That serialized PV for the previous tile behind softmax/P staging for the
current tile. It preserved correctness, but it left MMA idle while the softmax
roles converted the current BF16 score tile into compact P.

New active order:

```text
for tile n:
  QK(n)
  publish S(n)
  PV(n-1) while Softmax(n) stages P(n)
  wait Softmax(n) -> P(n)
  release K(n)

tail:
  PV(last)
```

This matches the 88-style QK/softmax/PV interleave more closely while
preserving SM120 role ownership:

```text
MMA owns QK/PV tensor-core work.
Softmax owns online row state and compact P production.
K/B shared memory is still released only after Softmax consumes the score tile.
```

Validation:

```text
git diff --check: pass
online kv_tiles=16:
  finite, mean_abs=0.000650689, max_abs=0.00309772, cosine=0.988923

full grid first tile:
  finite, mean_abs=0.000155332, max_abs=0.000899995, cosine=0.992945

full-grid min:
  before interleave fix: 9.8332 ms
  after interleave fix:  8.2618 ms
```

Conclusion:

```text
This is the largest win since softmax-owned online state. The issue was not the
score tile alone; the schedule also failed to spend the softmax window on useful
MMA work. The remaining gap should be profiled again from this checkpoint
before attempting more S/P storage rewrites.
```

## Post-Interleave NCU Comparison

2026-04-28T22:18:00-05:00

Nsight Compute was rerun on the active full-grid role owner after the QK /
softmax / PV interleave fix.

```text
metric                                      before interleave   after interleave
gpu__time_duration.sum                      10.008 ms           8.428 ms
smsp__inst_executed.sum                     3.237B              3.237B
registers/thread                            80                  80
dynamic shared memory                       99,328 B            99,328 B
tensor pipe active                          5.92%               6.96%
issue active                                28.38%              33.44%
eligible warps/cycle                        0.33                0.40
active warps/cycle                          5.50                5.50
avg warp latency / issued inst              19.38               16.46
sleeping stall / issued inst                12.07               8.96
long scoreboard / issued inst               2.40                2.39
wait stall / issued inst                    2.40                2.42
barrier stall / issued inst                 0.37                0.41
memory throughput                           33.07%              39.22%
DRAM throughput                             5.987 GB/s          7.405 GB/s
```

Conclusion:

```text
The interleave fix improved overlap without reducing instruction count or
resource footprint. The gain came from spending the softmax/P-staging window on
useful PV work, not from less work. Remaining bottleneck is still low issue
eligibility and low tensor-pipe utilization; the next structural win has to
increase overlap or reduce the durable score/P/O handoff cost, not tune scalar
epilogue loops.
```

## Rejected: Naive M64 CTA Variant

2026-04-28T22:20:00-05:00

Tried a direct M64 CTA variant by changing the SM120 block-scaled CUTLASS tile
shape from `128x128x256` to `64x128x256` and adjusting the Python full-grid
check to accept a 64-row output tile.

Failure mode:

```text
static_assert failed:
  SM120 Q/K/V load collective storage must fit SM120 opt-in shared memory

CUTLASS TMA layout assertion failed:
  TMA requires CTA_Tile and SLayout top-level size equivalence.
```

Decision:

```text
Rejected and reverted. The SM120 block-scaled SFA/SFB TMA sidecar layouts are
not shape-generic under a simple `TileShape` swap. If M64 is revisited, it needs
a real SM120 collective/layout port for the scale sidecars, not a global tile
constant edit.
```

## Rejected: Split Correction-to-Epilogue PipelineE

2026-04-28T22:32:00-05:00

Tried porting the SM100 epilogue shape more closely by using both stages of
`pipeline_corr_epi`:

```text
Correction:
  wait final O from MMA
  normalize first 64 rows in epilogue smem
  commit pipeline_corr_epi stage 0
  normalize second 64 rows in epilogue smem
  commit pipeline_corr_epi stage 1

Epilogue:
  wait/store first half
  wait/store second half
```

This preserved the role boundary: Correction owned final normalization, and
Epilogue owned global output stores. It did not collapse softmax/correction work
back into the MMA role and did not add a new active-path `NamedBarrier`.

Validation:

```text
online kv_tiles=16:
  finite, mean_abs=0.000650689, max_abs=0.00309772, cosine=0.988923

full grid first tile:
  finite, mean_abs=0.000155332, max_abs=0.000899995, cosine=0.992945

full-grid min:
  baseline after interleave fix:        8.2618 ms
  split correction->epilogue PipelineE: 8.2809 ms
```

Decision:

```text
Rejected and reverted. With SM120 shared-memory O storage, splitting the final
epilogue handoff adds pipeline transactions but does not expose enough work to
hide. The SM100 two-stage epilogue pattern is useful when backed by TMEM/TMA
store overlap; the SM120 no-TMEM substitution needs a larger structural change
than half-tile epilogue staging.
```

## Rejected: All-MMA-Thread PipelineAsync Score/O Commit

2026-04-28T22:44:00-05:00

Tried replacing two active-path `NamedBarrier` handoffs with PipelineAsync
arrival counts:

```text
before:
  producer_arv_count = 1
  all MMA threads NamedBarrier::sync()
  qk_mma_thread_idx == 0 commits pipeline_mma_s0/s1 or pipeline_mma_corr

experiment:
  producer_arv_count = all MMA threads
  all MMA threads fence_view_async_shared()
  all MMA threads producer_commit(...)
```

This preserved role ownership and removed the score/final-O named barriers from
the active path. It did not change score math, P production, or PV accumulation.

Validation:

```text
online kv_tiles=16:
  finite, mean_abs=0.000650689, max_abs=0.00309772, cosine=0.988923

full grid first tile:
  finite, mean_abs=0.000155332, max_abs=0.000899995, cosine=0.992945

full-grid min:
  baseline after interleave fix:        8.2618 ms
  all-thread PipelineAsync commit path: 8.3570 ms
```

Decision:

```text
Rejected and reverted. The named barrier at score/final-O publication is not
the dominant remaining handoff cost. Replacing it with 256 producer arrivals
increases pipeline transaction overhead and regresses wall time. Future work
should target durable score/P/O storage lifetime or pipeline overlap, not this
barrier substitution.
```

## Cleanup: Remove Superseded Stage Kernels

2026-04-28T23:10:00-05:00

Removed the rejected/intermediate benchmark kernels and entry points from
`benchmarks/sm120_nvfp4_cutlass_fused_attention.cu` and the matching harness
flags from `benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py`.

Deleted from the compiled extension:

```text
sm120_nvfp4_qk_load_collective_stage_kernel
sm120_nvfp4_qkv_load_collective_stage_kernel
sm120_nvfp4_qkv_role_handoff_stage_kernel
old stage-only softmax/P helpers
old stage-only QK/PV register helper bodies
```

Kept:

```text
active online register-Q owner/full-grid kernel
role schedule smoke kernel
QK/PV atom-level correctness gates
FlashInfer CUTLASS FP4 runner hook
metadata hook
```

Reason:

```text
The removed kernels are no longer the active structural path and were
superseded by the role-owned online owner. Keeping them compiled slows rebuilds
and creates editing ambiguity. The atom-level gates remain because they validate
the SM120 block-scaled operand/copy layouts used by the active path.
```

Validation after cleanup:

```text
online kv_tiles=16:
  finite, mean_abs=0.000650689, max_abs=0.00309772, cosine=0.988923

full grid first tile:
  finite, mean_abs=0.000155332, max_abs=0.000899995, cosine=0.992945

full-grid min:
  8.3033 ms
```

## Rejected: SM100 Role Register Reconfiguration

2026-04-28T23:42:00-05:00

Tried porting the SM100 role-level register schedule from
`sm100_fmha_fwd_kernel_tma_warpspecialized.hpp`:

```text
Softmax0/Softmax1: 192 regs
Correction:         64 regs
MMA:               192 regs on SM120, not SM100's 64, because SM120 keeps
                   Q fragments and O accumulators in registers instead of TMEM
Load/Epilogue:      64 regs
```

The first version added two Empty warps to complete the final warpgroup and
called `setmaxnreg` inside each role branch. It compiled and the role schedule
smoke passed, but the online correctness gate timed out. The likely cause was
divergent `setmaxnreg.sync.aligned` use in the mixed final warpgroup.

The second version moved the calls to one warpgroup-uniform site and gave the
mixed Load/Epilogue/Empty warpgroup a single 64-register budget. It also
compiled and the role schedule smoke passed:

```text
role_warp_counts:
  softmax0=4, softmax1=4, correction=4, mma=8, load=1, epilogue=1, empty=2
total_threads=768
```

but the online correctness gate still timed out.

Decision:

```text
Rejected and reverted. The SM100 register-donation mechanism is not a safe
drop-in port for this SM120 no-TMEM benchmark kernel. SM100's MMA/Load/Epilogue
roles all use a low "other" register budget because O lives in TMEM; this SM120
kernel keeps Q and O accumulator state in registers. Completing the final
warpgroup and applying setmaxnreg changes synchronization/runtime behavior
enough to hang before correctness. Do not retry this as a tuning knob unless the
SM120 kernel is first restructured around a warpgroup layout where every
setmaxnreg call is both warpgroup-uniform and matched to the real per-role
register lifetimes.
```

## Port: Softmax Probability Alias + SM120 Epilogue Substitution

2026-04-29T00:28:00-05:00

Ported the next softmax/epilogue structural slice while preserving the
role-owned dataflow:

```text
Softmax role:
  before:
    pass 1: read logits, compute tile max
    pass 2: read logits, compute exp/tile sum and P scales
    barrier
    helper pass: read logits again, compute exp again, write NVFP4 P via CUTLASS partition_D

  after:
    pass 1: read logits, compute tile max
    pass 2: compute scaled probabilities once, overwrite the S/logits SMEM
            region with BF16 probabilities, write P scales
    barrier
    helper pass: read BF16 probabilities, write NVFP4 P via CUTLASS partition_D
```

This keeps the CUTLASS `partition_D` producer convention for the actual NVFP4 P
tile. The rejected direct row-owner variant wrote `p_sA(row, col)` with scalar
nibble stores; it preserved correctness but regressed full-grid time to
~8.36 ms. The kept alias version avoids the second exp pass without abandoning
the vectorized/partitioned producer mapping.

Ported the SM120 epilogue substitution:

```text
SM100:
  Correction role reads O from TMEM, rescales by final row sum, writes epilogue SMEM.

SM120:
  no TMEM exists, so O lives in MMA registers.
  MMA applies final row normalization before writing epilogue SMEM.
  Epilogue remains the only role that writes global output.
```

After this, the old Correction role had no work. Removed it from the active
schedule and deleted the unused MMA->Correction pipeline storage:

```text
old active schedule:
  softmax0=4, softmax1=4, correction=4, mma=8, load=1, epilogue=1
  total_threads=704

new active schedule:
  softmax0=4, softmax1=4, correction=0, mma=8, load=1, epilogue=1
  total_threads=576
```

Validation:

```text
role schedule smoke:
  softmax0=4, softmax1=4, correction=0, mma=8, load=1, epilogue=1
  total_warps=18, total_threads=576

online kv_tiles=16:
  finite, mean_abs=0.000649069, max_abs=0.00317944, cosine=0.988839

full grid first tile:
  finite, mean_abs=0.000155417, max_abs=0.000869478, cosine=0.992919

full-grid min:
  cleanup baseline:                         8.3033 ms
  BF16 probability alias only:              8.3025 ms
  final normalization in MMA regs:          8.2831 ms
  compacted no-correction role schedule:    7.6612 ms
  compacted schedule, repeat=20:             7.6561 ms
```

Conclusion:

```text
The SM120 no-TMEM substitution should not preserve an empty Correction role.
Once final normalization moves to the MMA register drain, compacting the role
schedule is the real win. This is still far from the two-stage CUTLASS ceiling,
so the next work must attack per-tile QK/PV/softmax overlap and P/O storage
lifetime, not split-KV parallelism.
```

## Rejected: Load-Order-Only K/V Interleave

2026-04-29T00:41:00-05:00

Tested a narrow load-order patch in the active Load role:

```text
before: load K chunk 0, load K chunk 1, load V
after:  load K chunk 0, load V,         load K chunk 1
```

Correctness was unchanged:

```text
online kv_tiles=16:
  finite, mean_abs=0.000649069, max_abs=0.00317944, cosine=0.988839
```

Timing did not improve:

```text
compacted baseline repeat=20:  min_ms=7.6561
load-order-only repeat=5:      min_ms=7.6663, mean_ms=7.6714
```

Conclusion:

```text
Rejected and reverted. SM100/FA-3 load ordering is not just a static K/V issue
permutation; V must be temporally anchored to QK/softmax progress so it arrives
before PV consumes it. Reordering the issue sequence inside the same serial
prefetch point does not create overlap.

Do not spend more time on static K/V ordering probes. The next structural port
is register-resident softmax row state from Example 88, followed by tighter
MMA/softmax/PV handoff.
```

## Rejected: Duplicated Register-Resident MMA Row State

2026-04-29T00:55:00-05:00

Attempted to port the Example 88 online softmax row-state lifecycle into the
SM120 MMA role so PV rescale would no longer read per-tile `old_scale` from
SMEM.

Two variants were tested:

```text
full duplicated state:
  MMA role recomputes tile max, running max, running sum, old scale, and final
  normalization in registers.

old-scale-only state:
  MMA role recomputes tile max and running max only, keeps per-tile old scale in
  registers, while final denominator remains produced by the Softmax role.
```

Results:

```text
baseline compacted schedule:
  storage_bytes=96256, min_ms=7.6561

full duplicated state:
  storage_bytes=94208
  online kv_tiles=16 finite, mean_abs=0.00434299, cosine=0.988652
  full-grid min_ms=8.2163

old-scale-only state:
  storage_bytes=94208
  online kv_tiles=16 finite, mean_abs=0.000652661, cosine=0.988712
  full-grid min_ms=7.9107
```

Decision:

```text
Rejected and reverted. Moving row state into MMA registers is not a free port
of Example 88 in this role-split SM120 kernel. Because Softmax remains a
separate role, the MMA role has to duplicate QK row reductions to derive the
same online state. That duplicated max/sum work costs more than the SMEM
old-scale handoff it removes.

The lesson is narrower than "register-resident row state is bad": it is bad in
this hybrid role decomposition where Softmax already owns the row reductions.
To make Example 88's row-state lifecycle pay, the corresponding QK softmax/PV
state must be owned by the same atom-level role, or the row-state handoff must be
made cheaper without recomputing reductions.
```

## Profile: Compacted Schedule After Row-State Rejections

2026-04-29T01:07:00-05:00

Profiled the current fast path (`656e6c2` code shape) with Nsight Compute:

```text
duration:                         7.81 ms under ncu
registers/thread:                 96
dynamic shared memory/block:      96.26 KiB
occupancy limiters:               1 block/SM by registers and shared memory
SM busy:                          22.22%
tensor pipe active:               7.45% active, 5.02% elapsed
eligible warps/scheduler:         0.39
active warps/scheduler:           4.49
issue slots busy:                 22.22%
memory throughput:                54.20 GB/s, 34.17% memory-active
L2 hit rate:                      99.73%
```

Dominant stall ratios:

```text
sleeping:                         6.00 cycles/issue
wait:                             2.59
long scoreboard:                  2.13
barrier:                          0.51
short scoreboard:                 0.43
mio throttle:                     0.26
```

Conclusion:

```text
The current kernel is not bandwidth-bound and is still far from tensor-pipe
saturation. The largest reported stall class is role wait/sleep from the
pipeline/role schedule, followed by wait and long scoreboard. This explains why
removing the empty Correction role helped. Further role compaction is worth
testing only if it does not increase per-thread producer work.
```

## Rejected: 2-Warp Softmax Role Compaction

2026-04-29T01:16:00-05:00

Tested reducing each softmax role from 4 warps to 2 warps:

```text
before:
  softmax0=4, softmax1=4, correction=0, mma=8, load=1, epilogue=1
  total_warps=18, total_threads=576

after:
  softmax0=2, softmax1=2, correction=0, mma=8, load=1, epilogue=1
  total_warps=14, total_threads=448
```

The first version compiled but produced bad output because the CUTLASS
`partition_D` producer for the NVFP4 P tile was only covering virtual producer
threads 0-63. Fixed that by having each physical softmax thread cover the
missing virtual producer slice as well. Correctness then matched the baseline:

```text
online kv_tiles=16:
  finite, mean_abs=0.000649069, max_abs=0.00317944, cosine=0.988839
```

Timing:

```text
baseline compacted schedule:      min_ms=7.6561
2-warp softmax compacted:         min_ms=10.9802
```

Decision:

```text
Rejected and reverted. Although the NCU profile shows many sleeping/waiting
role warps, reducing the softmax producer from 128 physical threads to 64
physical threads doubles per-thread P producer work and dominates any benefit
from fewer parked warps. The 4+4 softmax roles are not the next bottleneck to
compact in this implementation.
```

## Win: Hardware E2M1 Conversion For P Producer

2026-04-29T01:28:00-05:00

Replaced the softmax/P producer's scalar software E2M1 quantization loop:

```text
nearest_e2m1_code(x):
  compare against 16 FP4 grid values
  choose nearest nibble in scalar ALU
```

with the SM120 hardware FP32-to-E2M1 conversion already available in this file:

```text
fp32_to_e2m1_code_hw(x):
  cvt.rn.satfinite.e2m1x2.f32
```

The helper passes `x` twice and takes the low nibble, avoiding pair-order
ambiguity while replacing the 16-entry scalar search with the hardware
conversion instruction.

Validation:

```text
online kv_tiles=16:
  finite, mean_abs=0.000649069, max_abs=0.00317944, cosine=0.988839

full grid first tile:
  finite, mean_abs=0.000155417, max_abs=0.000869478, cosine=0.992919
```

Timing:

```text
baseline compacted schedule:      min_ms=7.6561
hardware E2M1, repeat=5:          min_ms=5.5254, mean_ms=5.5489
hardware E2M1, repeat=20:         min_ms=5.5294, mean_ms=5.5467
```

Conclusion:

```text
This is the largest win since the compacted role schedule. The score/P lifetime
was blocked less by the SMEM old-scale handoff and more by per-element software
FP4 quantization in the P producer. The next profile should be taken from this
new 5.53 ms checkpoint before further structural changes.
```

## Win: Fused Row-Owned P Producer

2026-04-29T01:43:00-05:00

Ported the next P-lifetime slice: the Softmax row owner now computes scaled
probabilities, emits the P scale sidecar, and writes the compact NVFP4 P tile in
the same row pass. The old second phase:

```text
row owner writes BF16 probabilities to score SMEM
group barrier
partition_D helper rereads BF16 probabilities
partition_D helper writes compact NVFP4 P
```

is gone from the active path. With the helper removed, each softmax role can run
with 2 warps instead of 4 because all 64 physical threads own one row.

Implementation details:

```text
softmax0=2, softmax1=2, correction=0, mma=8, load=1, epilogue=1
total_warps=14, total_threads=448

P conversion:
  fp32_to_e2m1_code_hw via cvt.rn.satfinite.e2m1x2.f32

Deleted:
  softmax_role_write_probs_bf16_p_to_pv_smem_stage2
```

Validation:

```text
online kv_tiles=16:
  finite, mean_abs=0.000650521, max_abs=0.00309772, cosine=0.988925

full grid first tile:
  finite, mean_abs=0.000155273, max_abs=0.000907625, cosine=0.992944
```

Timing:

```text
previous hardware-E2M1 helper path:  min_ms=5.5294, mean_ms=5.5467
fused row-owned P, repeat=5:         min_ms=4.5179, mean_ms=4.5368
fused row-owned P, repeat=20:        min_ms=4.4943, mean_ms=4.5196
```

Conclusion:

```text
This fixes the failure mode from the earlier 2-warp softmax rejection. Reducing
softmax warps was only profitable after removing the second partitioned P-helper
phase. The P producer is still on the critical score-buffer lifetime, but it no
longer round-trips probabilities through BF16 SMEM or burns a second group-wide
producer pass.
```

## Win: Avoid P Scale Decode In Row Producer

2026-04-29T01:52:00-05:00

In the fused row-owned P producer, removed the immediate decode of the
just-quantized E4M3 scale when selecting E2M1 P codes:

```text
before:
  scale_byte = fp32_to_e4m3_byte(scale_value)
  output_scale = kProbGlobalScale / e4m3_byte_to_fp32(scale_byte)

after:
  scale_byte = fp32_to_e4m3_byte(scale_value)
  output_scale = kProbGlobalScale / scale_value
```

The MMA still consumes the quantized E4M3 scale sidecar; this change only avoids
the local decode instruction sequence in the P-code selection path.

Validation:

```text
online kv_tiles=16:
  finite, mean_abs=0.000659900, max_abs=0.00350227, cosine=0.988955

full grid first tile:
  finite, mean_abs=0.000158765, max_abs=0.000815836, cosine=0.992936
```

Timing:

```text
row-owned P baseline repeat=20:   min_ms=4.4943, mean_ms=4.5196
no scale decode repeat=20:        min_ms=4.4847, mean_ms=4.5149
```

Conclusion:

```text
Small but positive. Keep it. The accuracy shift is within the existing
reference-kernel tolerance band and the timing is marginally better.
```

## Win: Remove Redundant Softmax Group Barrier

2026-04-29T01:59:00-05:00

After fusing P production into the row-owner softmax pass, the explicit
softmax-group `NamedBarrier` before `consumer_release` became redundant:

```text
before:
  row owners write P/SFA
  NamedBarrier::sync(softmax_group)
  fence_view_async_shared
  pipeline consumer_release

after:
  row owners write P/SFA
  fence_view_async_shared
  pipeline consumer_release
```

The pipeline release is already the cross-role completion signal, and every
physical softmax thread owns one row in the 2-warp role layout.

Validation:

```text
online kv_tiles=16:
  finite, mean_abs=0.000659900, max_abs=0.00350227, cosine=0.988955

full grid first tile:
  finite, mean_abs=0.000158765, max_abs=0.000815836, cosine=0.992936
```

Timing:

```text
with softmax barrier repeat=20:    min_ms=4.4847, mean_ms=4.5149
without barrier repeat=20:         min_ms=4.4768, mean_ms=4.5034
```

Conclusion:

```text
Small but real. Keep it. This is safe only after row-owned P production; the
old helper path needed the group barrier because helper threads consumed
row-owner-produced BF16 probabilities and SFA.
```

## Neutral: Explicit exp2 Softmax Rewrite

2026-04-29T02:05:00-05:00

Tested replacing the row-owned softmax `__expf` calls with explicit
`exp2f(x * log2(e))`, matching the style used by Example 88.

Validation was unchanged:

```text
online kv_tiles=16:
  finite, mean_abs=0.000659900, max_abs=0.00350227, cosine=0.988955

full grid first tile:
  finite, mean_abs=0.000158765, max_abs=0.000815836, cosine=0.992936
```

Timing:

```text
current baseline repeat=20:       min_ms=4.4768, mean_ms=4.5034
explicit exp2 repeat=20:          min_ms=4.4837, mean_ms=4.5023
```

Decision:

```text
Rejected and reverted as noise. The mean changed by ~0.001 ms and the min got
slightly worse. CUDA fast math is already lowering the exponent path well
enough; this is not a meaningful lever.
```

## Win: V-First Steady-State Load Timing

2026-04-28T22:55:21-05:00

Ported the first piece of the Example 77/88 temporal load-order pattern into
the active SM120 owner kernel. The prologue already had the useful shape:

```text
Q0, K0, Q1, V0, K1
```

The steady-state load loop was still issuing the next tile as:

```text
K(n+1, chunk0), K(n+1, chunk1), V(n+1)
```

Because K and V use independent producer states, placing V behind K meant the V
TMA could be delayed by the K pipeline's score-buffer lifetime. The patch
changes the steady-state issue order to:

```text
V(n+1), K(n+1, chunk0), K(n+1, chunk1)
```

This does not by itself implement the full 77/88 temporal anchor, but it stops a
K-side acquire stall from delaying the next V load. That is the key distinction
from the earlier rejected static `K0,V,K1` probe: this variant puts V before any
next-tile K acquire in the steady loop.

Validation:

```text
online kv_tiles=16:
  finite, mean_abs=0.000659900, max_abs=0.00350227, cosine=0.988955

full grid first tile:
  finite, mean_abs=0.000158765, max_abs=0.000815836, cosine=0.992936
```

Timing:

```text
previous best repeat=20:          min_ms=4.4768, mean_ms=4.5034
V-first steady repeat=20:         min_ms=4.3854, mean_ms=4.4270
```

Conclusion:

```text
Keep it. The remaining canonical load-order gap is temporal anchoring: V(n+1)
should be issued while QK(n) is executing and be ready by PV(n+1), not merely
permuted in the producer loop. This change is still useful because it removes
one avoidable producer-side serialization point before the next row-state port.
```

## Win: Packed-Byte P Producer Stores

2026-04-28T23:08:00-05:00

The fresh V-first profile showed the active kernel had moved into a memory and
spill dominated regime:

```text
duration under ncu:                ~4.5 ms
registers/thread:                  128
stack size:                        2512 bytes
local memory spilling requests:    72.59 MB
shared load conflicts:             6.5-way average
shared store conflicts:            5.3-way average
eligible warps/scheduler:          0.13
tensor pipe active:                ~14.6% active
```

The row-owned P producer was still writing compact P one nibble at a time:

```text
p_sA(row, col, 0) = cute::uint4_t(fp32_to_e2m1_code_hw(...))
```

CuTe subbyte assignment performs a read-modify-write of the underlying byte for
each 4-bit element. That doubles the conversion count and creates pathological
shared-store behavior for the P sidecar.

Changed the P producer to write one packed byte per FP4 pair:

```text
auto first_ref = p_sA(row, local_col + 2 * pair, 0)
uint8_t* dst_byte = cute::recast_ptr<uint8_t>(&first_ref)
*dst_byte = fp32_pair_to_e2m1_byte(p0, p1)
```

This preserves the CuTe swizzled destination by taking the address of the
subbyte reference at the even nibble, then recasting that iterator to a byte
pointer. It also uses the SM120 pair conversion instruction directly instead of
calling the one-value wrapper twice.

Validation:

```text
online kv_tiles=16:
  finite, mean_abs=0.000659900, max_abs=0.00350227, cosine=0.988955

full grid first tile:
  finite, mean_abs=0.000158765, max_abs=0.000815836, cosine=0.992936
```

Timing:

```text
V-first baseline repeat=20:        min_ms=4.3854, mean_ms=4.4270
packed P stores repeat=20:         min_ms=3.8795, mean_ms=3.9161
```

Conclusion:

```text
Keep it. The P producer's compact sidecar store was a structural bottleneck.
The direct byte-store path removes subbyte RMW traffic and halves P conversion
instructions. Next profile should verify how much of the local spill/shared
wavefront excess remains before attempting the next row-state handoff change.
```

## Rejected: MMA-Side Row Scale Cache

2026-04-28T23:18:00-05:00

Tried caching the row scale in the MMA thread before applying `old_scale_stage`
and before final output normalization:

```text
before:
  each accumulator element reloads old_scale_stage[row] or global_l[row]

after:
  cache the last row seen by this thread and reuse the scale while row matches
```

Correctness was unchanged:

```text
online kv_tiles=16:
  finite, mean_abs=0.000659900, max_abs=0.00350227, cosine=0.988955

full grid first tile:
  finite, mean_abs=0.000158765, max_abs=0.000815836, cosine=0.992936
```

Timing regressed:

```text
packed P baseline repeat=20:       min_ms=3.8795, mean_ms=3.9161
row-scale cache repeat=20:         min_ms=3.9884, mean_ms=4.0268
```

Decision:

```text
Rejected and reverted. The extra compare/live scalar state costs more than the
saved row-scale SMEM loads in the current fragment order. Do not retry this as a
per-element cache. If row-state traffic is attacked again, it needs a structural
ownership change or a fragment-layout-aware row broadcast, not an in-loop branch.
```

## Profile: Packed P Store Checkpoint

2026-04-28T23:20:00-05:00

Profiled the packed-P-store checkpoint (`e131aa9`) with Nsight Compute:

```text
report:                            reports/sm120_nvfp4_packed_p_current.ncu-rep
duration under ncu:                3.93-3.96 ms
registers/thread:                  128
stack size:                        1888 bytes
local memory spilling requests:    72.66 MB
executed instructions:             ~508.1M
eligible warps/scheduler:          0.13
active warps/scheduler:            3.50
issue slots busy:                  ~7.6%
SM busy:                           ~10.0%
compute throughput:                ~13.7%
memory throughput:                 ~45%
shared load conflicts:             8.3-way average
shared store conflicts:            5.9-way average
```

Compared to the pre-packed profile, packed P stores reduced stack size and
instruction count, but did not reduce spill traffic:

```text
before packed P:                   stack 2512 B, spills 72.59 MB, ~560.0M inst
after packed P:                    stack 1888 B, spills 72.66 MB, ~508.1M inst
```

Conclusion:

```text
The packed-store win came from fewer instructions and less P-store sidecar work,
not from solving the local-memory spill problem. Spill remains >50 MB, so the
next high-probability lever is a separate live-range issue, likely around QK
accumulator/PV fragment lifetime or the nested generic lambdas in the MMA loop.
```

## Rejected: 64-Column PV Tile Probe

2026-04-28T23:22:00-05:00

Tried reducing the PV/output tile from `128x128x128` to `128x64x128` as a
direct attack on the remaining local-memory spill:

```text
motivation:
  NCU after packed P stores still reports ~72.66 MB local spill traffic.
  The full-grid launch has 128 CTAs, below the 188 SMs on the RTX PRO 6000.
  Halving the PV N tile would cut accumulator live state and launch 256 CTAs.

probe:
  CutlassThreadBlockShapeK128 = Shape<128,64,128>
  active output group width = 64
```

The probe failed at compile time inside the SM120 block-scaled CUTLASS builder:

```text
CUTLASS TMA assert:
  TMA requires CTA_Tile and SLayout top-level size equivalence
```

Current metadata and runner configs only expose the practical SM120 FP4 runner
tile shapes:

```text
128x128x128
128x128x256
256x128x128
```

Decision:

```text
Rejected and reverted. Do not retry this as a simple tile-shape edit. A
64-column PV path needs explicit SM120 block-scaled scale-layout/TMA-layout work
or a manual V/scale loader. The next live-range work should stay on the accepted
128-wide CUTLASS collective path unless we intentionally port the scale layout.
```

## Rejected: 64-Row Q Tile Probe

2026-04-28T23:24:00-05:00

Tried the smaller Q-tile alternative before committing to split-KV:

```text
probe:
  QK tile = 64x128x256
  PV tile = 64x128x128
  output shape remains 128 columns

goal:
  double CTA count from 128 to 256 without changing the PV output shape
  reduce PV/QK fragment live state by halving M
```

Compile failed before runtime validation:

```text
storage:
  SM120 Q/K/V load collective storage no longer fit the 99 KiB opt-in limit

CUTLASS TMA:
  TMA requires CTA_Tile and SLayout top-level size equivalence
  failures occurred in the block-scaled scale-side TMA layouts for 64-row tiles
```

Decision:

```text
Rejected and reverted. Like the 64-column probe, this is not a safe simple
shape edit on the current SM120 block-scaled CUTLASS collective. Getting M=64
requires explicit scale-layout/TMA-layout work. The next parallelism path is
split-KV, which preserves the accepted 128x128 MMA shape and adds CTA
parallelism across KV slices.
```

## Win: Split-KV Full-Grid Path

2026-04-28T23:31:00-05:00

Implemented a split-KV benchmark path that preserves the accepted SM120
block-scaled CUTLASS tile shapes:

```text
QK tile: 128x128x256
PV tile: 128x128x128
split axis: KV tiles
per-split output: local normalized BF16 O tile
per-split stats: row max m and row sum l
combine: O = sum_s exp(m_s - m_global) * l_s * O_s / l_global
```

This avoids the rejected `N=64` and `M=64` shape edits and adds CTA
parallelism without changing the MMA layouts.

Split-size sweep:

```text
split_kv_len=1024   splits=32   min_ms=3.7724
split_kv_len=2048   splits=16   min_ms=3.3121
split_kv_len=4096   splits=8    min_ms=3.2352
split_kv_len=8192   splits=4    min_ms=3.1238
split_kv_len=16384  splits=2    min_ms=3.9165
```

Checkpoint rerun with repeat=20:

```text
packed full-grid baseline:
  finite, mean_abs=0.000158765, max_abs=0.000815836, cosine=0.992936
  min_ms=3.8818, mean_ms=3.9103

split-KV full-grid, split_kv_len=8192:
  finite, mean_abs=0.000159051, max_abs=0.000798231, cosine=0.992921
  min_ms=3.1171, mean_ms=3.1351
```

Conclusion:

```text
Split-KV is the first parallelism lever that survives the SM120 block-scaled
layout constraints. The best tested split is 4-way at 8192 tokens, improving
full-grid wall time by ~19.7% versus the packed-store baseline. The result is
still far from the two-stage CUTLASS ceiling, so the next profile should focus
on remaining per-CTA load stalls, tensor-pipe utilization, and whether local
spill traffic drops with shorter KV loops.
```

## Profile And Sweep: Dynamic Split-KV

2026-04-28T23:38:00-05:00

Relaxed split-KV so `split_kv_len` no longer has to divide 32768 exactly. The
last split now handles the remaining KV tiles. This allows testing split counts
that better match the 188-SM wave geometry.

Nsight Compute on the 4-way 8192-token split stage:

```text
report:                         reports/sm120_nvfp4_splitkv_8192_stage.ncu-rep
grid size:                      512 CTAs
waves/SM:                       2.72
registers/thread:               128
stack size:                     1888 bytes
local memory spilling requests: 0
issue slots busy:               ~10.1%
SM busy:                        ~13.1%
memory throughput:              ~142-144 GB/s
mem busy:                       ~60%
eligible warps/scheduler:       0.13
active warps/scheduler:         3.49
warp cycles/issued inst:        ~30.6
top stalls:                     sleep/yield/block ~43%, long scoreboard ~30%
```

Key conclusion:

```text
Split-KV eliminated the local-memory spill traffic. The remaining bottleneck is
not spill; it is low eligible-warp rate plus load/scoreboard stalls and role
pipeline waiting. The grid tail is visible but not the dominant lever by itself.
```

Uneven split sweep:

```text
split_kv_len=11008  splits=3   min_ms=3.8613
split_kv_len=6656   splits=5   min_ms=3.3024
split_kv_len=5504   splits=6   min_ms=3.4023
split_kv_len=4736   splits=7   min_ms=3.0971
split_kv_len=3712   splits=9   min_ms=3.2950
split_kv_len=3328   splits=10  min_ms=3.1482
```

Checkpoint rerun for the best uneven split:

```text
split-KV full-grid, split_kv_len=4736, splits=7, repeat=20:
  finite, mean_abs=0.000158890, max_abs=0.000776978, cosine=0.992912
  min_ms=3.0958, mean_ms=3.1071
```

Decision:

```text
Keep dynamic split-KV support. The best measured split is currently 7-way at
4736 tokens, only modestly ahead of the 4-way split. The next performance work
should target per-CTA stall sources: sleep/wait-heavy role sequencing, long
scoreboard from load staging, and tensor-pipe starvation. Further split-count
tuning alone is not enough to reach the two-stage CUTLASS ceiling.
```

## Win: Unnormalized Split-KV Partials

2026-04-28T23:47:00-05:00

Changed split-KV partial output from locally normalized `O_s` to unnormalized
partial accumulators:

```text
before stage:
  partial_s = acc_s * pv_base_scale / l_s

before combine:
  out = sum_s exp(m_s - M) * l_s * partial_s / L

after stage:
  partial_s = acc_s * pv_base_scale

after combine:
  out = sum_s exp(m_s - M) * partial_s / L
```

This preserves the same online-softmax merge math while removing one per-output
division in the split stage and one multiply by `l_s` on the combine numerator.

Results:

```text
split_kv_len=4736, splits=7:
  before: min_ms=3.0958, mean_ms=3.1071
  after:  min_ms=3.0764, mean_ms=3.0867
  correctness: finite, mean_abs=0.000158847, max_abs=0.000777204,
               cosine=0.992912

split_kv_len=8192, splits=4:
  after: min_ms=3.0995, mean_ms=3.1195
```

Decision:

```text
Keep unnormalized split partials. The win is small but consistent, and the
7-way split remains the best measured point.
```

## Rejected: Two-Thread Softmax Rows

2026-04-28T23:52:00-05:00

Tried increasing softmax roles from 2+2 warps to 4+4 warps and splitting each
softmax row across two threads:

```text
before:
  one thread owns one row and processes all 128 columns

probe:
  two threads own one row
  each thread processes 64 columns / 4 scale groups
  per-row max and l are merged through softmax_tmp plus intra-role NamedBarrier
```

Correctness was unchanged:

```text
split_kv_len=4736:
  finite, mean_abs=0.000158847, max_abs=0.000777204, cosine=0.992912
```

Timing regressed badly:

```text
baseline unnormalized split partials: min_ms=3.0764, mean_ms=3.0867
two-thread softmax rows:             min_ms=4.0369, mean_ms=4.0491
```

Decision:

```text
Rejected and reverted. The extra softmax warps, larger CTA, shared scratch, and
two intra-role barriers per tile cost far more than halving each row's serial
exp/quant work. Do not retry row-splitting with barriers. Any future softmax
parallelism must avoid per-tile barrier expansion, likely by changing the role
pipeline/ownership rather than splitting each row locally.
```

## Win: Span-2 Output-Group Reuse

2026-04-29T00:01:24-05:00

The split-KV kernel was still launching one CTA per 128-column output group,
which meant each output group recomputed the same QK logits, online softmax
state, and P tile:

```text
old per output span:
  grid_y = 4
  each CTA computes QK + softmax + one PV output group

new reuse2 span:
  grid_y = 2
  each CTA computes QK + softmax once, then consumes two V/output groups
```

Implementation details:

```text
- Added templated stage kernel parameter kOutputGroupSpan in {1, 2}.
- Existing public split-KV path instantiates span 1.
- Added sm120_nvfp4_qkv_online_register_q_splitkv_reuse2_full_grid for span 2.
- For span 2, load order is K-before-next-V to avoid deadlock:
    prefill K0.0, Q, K0.1, then V0 groups
    next tile loads K(n+1) before V(n+1) groups
  This is necessary because the V pipeline has two stages; preloading two V
  groups fills it before the next K tile could otherwise be issued.
- Changed the final output pipeline to depth 1 because the epilogue smem tile is
  single-buffered. Span 2 commits two output groups sequentially through the same
  smem tile.
```

Correctness is unchanged:

```text
split_kv_len=4736, reuse2:
  finite, mean_abs=0.000158847, max_abs=0.000777204,
  cosine=0.992912
```

Initial timing:

```text
span 1 split-KV, split_kv_len=4736:
  min_ms=3.0783, mean_ms=3.1000

span 2 reuse, split_kv_len=4736:
  min_ms=2.2694, mean_ms=2.2939
```

Reuse2 changes the optimal split point. Short sweep:

```text
split_kv_len=2048   splits=16  min_ms=2.3129
split_kv_len=3328   splits=10  min_ms=2.2478
split_kv_len=4096   splits=8   min_ms=2.1566
split_kv_len=4736   splits=7   min_ms=2.2903
split_kv_len=5504   splits=6   min_ms=2.5251
split_kv_len=6656   splits=5   min_ms=2.1307
split_kv_len=8192   splits=4   min_ms=2.4877
split_kv_len=11008  splits=3   min_ms=3.0788
```

Repeat-20 checkpoint:

```text
reuse2, split_kv_len=6656:
  finite, mean_abs=0.000158808, max_abs=0.000782972,
  cosine=0.992929
  min_ms=2.1204, mean_ms=2.1422

reuse2, split_kv_len=4096:
  finite, mean_abs=0.000159100, max_abs=0.000798231,
  cosine=0.992921
  min_ms=2.1337, mean_ms=2.1591
```

Profile of the 448-thread reuse2 variant:

```text
report:                         reports/sm120_nvfp4_splitkv_reuse2_6656_stage.ncu-rep
grid size:                      320 CTAs
block size:                     448
duration under NCU:             2.19 ms
registers/thread:               128
local memory spilling requests: 59.57 MB
issue slots busy:               12.27%
SM busy:                        12.56%
L2 throughput:                  71.34%
eligible warps/scheduler:       0.18
active warps/scheduler:         3.48
```

Conclusion:

```text
Span-2 output reuse is a real structural win because it removes duplicated
QK/softmax/P work across output groups. It also reintroduces local spill traffic
because two PV accumulators are live across the KV loop. The next lever is
reducing CTA thread-count/register pressure while preserving the reuse.
```

## Win: 384-Thread Softmax Ownership

2026-04-29T00:01:24-05:00

Changed the softmax roles from two warps per half to one warp per half:

```text
before:
  softmax0 = 2 warps, softmax1 = 2 warps
  CTA threads = 448
  one softmax thread owns one row

after:
  softmax0 = 1 warp, softmax1 = 1 warp
  CTA threads = 384
  each softmax thread owns two rows in its half
```

This keeps the two-role Softmax0/Softmax1 pipeline structure but reduces total
CTA threads, raising the register budget available to the heavy MMA/PV
accumulator path.

Results:

```text
reuse2, split_kv_len=6656:
  finite, mean_abs=0.000158808, max_abs=0.000782972,
  cosine=0.992929
  min_ms=1.8552, mean_ms=1.8694

span 1 split-KV, split_kv_len=4736:
  finite, mean_abs=0.000158847, max_abs=0.000777204,
  cosine=0.992912
  min_ms=2.7461, mean_ms=2.7534
```

Profile of the 384-thread reuse2 variant:

```text
report:                         reports/sm120_nvfp4_splitkv_reuse2_6656_384t_stage.ncu-rep
grid size:                      320 CTAs
block size:                     384
duration under NCU:             1.92 ms
registers/thread:               168
local memory spilling requests: 46.46 MB
issue slots busy:               13.28%
SM busy:                        13.86%
L2 throughput:                  63.05%
eligible warps/scheduler:       0.19
active warps/scheduler:         2.99
```

Conclusion:

```text
Reducing softmax ownership from 4 warps to 2 warps is accepted. It both improves
the span-2 path and improves the span-1 path. The profile confirms the register
budget moved in the intended direction: registers/thread rose from 128 to 168
and local spill dropped from 59.57 MB to 46.46 MB. There is still significant
spill, so further wins must either reduce live PV accumulator state or reduce
CTA thread count again without breaking the CUTLASS MMA role.
```

## Rejected: Reuse Load Warp As Epilogue Consumer

2026-04-29T00:01:24-05:00

Tried removing the dedicated epilogue warp and reusing the load warp as the
output-pipeline consumer after Q/K/V TMA issue completes:

```text
before:
  softmax0 = 1 warp
  softmax1 = 1 warp
  MMA      = 8 warps
  load     = 1 warp
  epilogue = 1 warp
  total    = 384 threads

probe:
  softmax0 = 1 warp
  softmax1 = 1 warp
  MMA      = 8 warps
  load     = 1 warp, then output consumer
  epilogue = 0 warps
  total    = 352 threads
```

This compiled but deadlocked at runtime before producing output. The likely
cause is that the load role can remain blocked in the TMA producer/tail path
while the MMA role reaches the final output pipeline commit. With
`PipelineE<1>`, the second span-2 output commit waits for the output consumer,
but the load warp has not safely transitioned into the consumer role yet.

Decision:

```text
Rejected and reverted. Do not retry the load-as-epilogue role merge without
first redesigning the V-load tail and output pipeline lifetime. The accepted
384-thread variant with a dedicated epilogue warp remains the baseline.
```

## Split Sweep After 384-Thread Change

2026-04-29T00:01:24-05:00

Re-swept reuse2 split length after reducing the softmax roles to one warp per
half:

```text
split_kv_len=2048   splits=16  min_ms=2.0376
split_kv_len=3328   splits=10  min_ms=2.0093
split_kv_len=4096   splits=8   min_ms=1.8675
split_kv_len=4736   splits=7   min_ms=2.0396
split_kv_len=5504   splits=6   min_ms=2.2946
split_kv_len=6656   splits=5   min_ms=1.8758
split_kv_len=8192   splits=4   min_ms=2.2330
```

Repeat-20 checkpoint for the best short-sweep point:

```text
reuse2, split_kv_len=4096:
  finite, mean_abs=0.000159100, max_abs=0.000798231,
  cosine=0.992921
  min_ms=1.8562, mean_ms=1.8691
```

Conclusion:

```text
After the 384-thread change, 4096 and 6656 are effectively tied. Keep 4096 as
the working split because it has slightly better measured min and more CTAs
(512 vs 320), which should make subsequent per-CTA improvements easier to
observe.
```

## Win: Span-4 Output-Group Reuse

2026-04-29T00:01:24-05:00

Extended output-group reuse from two output groups to all four 128-column output
groups:

```text
reuse4:
  grid_y = 1
  each CTA computes QK + online softmax once
  then consumes V groups 0, 1, 2, 3 and emits the full 512-column output tile
```

The V pipeline has only two stages, so the producer cannot simply prefill all
four V groups before loading the next K tile. That deadlocks by filling the V
pipeline before K(n+1) is available. The accepted load sequence is staggered:

```text
prefill:
  Q, K0.0, Q, K0.1, V0.g0, V0.g1

loop tile n:
  K(n+1).0, K(n+1).1     if n+1 exists
  Vn.g2, Vn.g3
  V(n+1).g0, V(n+1).g1   if n+1 exists
```

This preserves the consumer order:

```text
PV consumes Vn.g0, Vn.g1, Vn.g2, Vn.g3
```

Correctness is unchanged:

```text
reuse4, split_kv_len=6656:
  finite, mean_abs=0.000158808, max_abs=0.000782972,
  cosine=0.992929
```

Split sweep:

```text
split_kv_len=1024   splits=32  min_ms=2.1287
split_kv_len=2048   splits=16  min_ms=1.8878
split_kv_len=3328   splits=10  min_ms=1.8465
split_kv_len=4096   splits=8   min_ms=2.1189
split_kv_len=4736   splits=7   min_ms=2.3077
split_kv_len=5120   splits=7   min_ms=2.4157
split_kv_len=5504   splits=6   min_ms=2.4733
split_kv_len=6144   splits=6   min_ms=1.8397
split_kv_len=6656   splits=5   min_ms=1.8042
```

Repeat-20 checkpoints:

```text
reuse4, split_kv_len=6656:
  min_ms=1.7825, mean_ms=1.8485

reuse4, split_kv_len=3328:
  min_ms=1.8406, mean_ms=1.8547
```

Profile:

```text
report:                         reports/sm120_nvfp4_splitkv_reuse4_6656_384t_stage.ncu-rep
grid size:                      160 CTAs
block size:                     384
duration under NCU:             1.80 ms
registers/thread:               168
local memory spilling requests: 45.26 MB
issue slots busy:               11.07%
SM busy:                        11.07%
L2 throughput:                  68.44%
DRAM throughput:                28.49%
eligible warps/scheduler:       0.16
active warps/scheduler:         2.96
```

Conclusion:

```text
Span-4 reuse is accepted as the current fastest Shape-B q512/kv32k reference
path. It underfills the GPU at the 5-way split (160 CTAs for 188 SMs), but the
work removal from eliminating all output-group QK/softmax recompute still wins.
The remaining bottleneck is not duplicated output-group work; it is per-CTA
efficiency: local spills remain ~45 MB, eligible warps/scheduler is only 0.16,
and issue slots busy is ~11%.
```

## Neutral: Narrow Fragment Live Ranges

2026-04-29T00:01:24-05:00

Moved Q fragments, V fragments, and PV accumulator fragments into the MMA role
branch instead of declaring them before the role branch. The goal was to shorten
the lifetime of the heaviest CUTE objects and reduce register pressure.

Result:

```text
reuse4, split_kv_len=6656:
  before: min_ms=1.7825, mean_ms=1.8485
  after:  min_ms=1.7920, mean_ms=1.8481
```

Conclusion:

```text
Keep the narrower scopes because they are cleaner and do not regress mean time,
but this does not materially reduce the remaining gap. The spill source is
inside CUTE copy/fragment movement and PV accumulator pressure, not accidental
top-level object lifetime.
```

## Resource Triage: Remaining 2.8x Levers

2026-04-29T10:24:28-05:00

Re-checked the three plausible sources for the remaining gap:

```text
active dense NVFP4 MMA atom:       m16n8k64_mxf4nvf4_ue4m3
QK CUTLASS mainloop stages:        2
PV K128 CUTLASS mainloop stages:   2
active role launch:                384 threads / 12 warps
QKV shared storage:                96256 bytes
SM120 opt-in shared memory:        101376 bytes
shared-memory margin:              5120 bytes
```

Reuse/occupancy:

```text
reuse2, split_kv_len=4096:
  splits=8,  more CTA-level parallelism than reuse4
  repeat20: mean_ms=1.8387, min_ms=1.8252, cosine=0.992921
  repeat50: mean_ms=1.8363, min_ms=1.8245, cosine=0.992921

reuse4, split_kv_len=6656:
  splits=5,  CTAs=160
  repeat20: mean_ms=1.8589, min_ms=1.7882, cosine=0.992929
  repeat50: mean_ms=1.8534, min_ms=1.7811, cosine=0.992929
```

Interpretation:

```text
Reuse2 raises issue-slot activity in profile (13.28% vs 11.07%) and slightly
wins this serial repeat-20 mean, but it does not reach 2 CTAs/SM. The kernel is
still capped at one resident CTA by both shared memory and registers. Reuse2 is
a valid policy candidate, not a structural 2.8x lever.
```

Pipeline depth:

```text
QK and PV are both already two-stage. A third stage cannot fit as a local bump:
the active QKV storage has only 5120 bytes of shared-memory headroom. The
compact V-only K128 stage is much larger than that, and QK's full CUTLASS
storage also cannot add another operand stage in the current layout.
```

MMA atom:

```text
CUTLASS selects m16n8k64 for dense NVFP4/e4m3 through
rr_blockscaled_op_selector_sm120(). The m16n8k32 atom in this tree is selected
for MXF8/F6/F4-family scale paths, not the dense NVFP4/e4m3 path used here. No
dense NVFP4 m16n8k128 atom is exposed in this CUTLASS tree.
```

Decision:

```text
The next real lever is not atom selection or a one-line stage bump. It is
storage-footprint reduction or tile-shape reduction sufficient to either:
  1. allow deeper K/V load staging, or
  2. allow more resident CTAs/SM, or
  3. reduce PV accumulator/register pressure enough to lift eligible warps.

The already-rejected naive K128 QK bridge remains rejected: it hung for minutes.
Any K128/Q-tile reduction must be a proper atom-level port, not another cloned
cooperative GEMM collective.
```

## Small Win: Row-Owned Split-KV Combine

2026-04-29T10:34:03-05:00

Replaced the element-owned split-KV combine kernel with a row-owned combine
kernel:

```text
old:
  one thread per output element
  recompute global_m/global_l for every row,col element

new:
  one CTA per row
  thread 0 computes normalized split weights once for the row
  CTA threads reuse those weights across output columns
```

Correctness is unchanged:

```text
reuse4, split_kv_len=6656:
  finite=true, mean_abs=0.000158808, max_abs=0.000782972,
  cosine=0.992929
```

Timing:

```text
reuse4, split_kv_len=6656:
  before repeat50: min_ms=1.7811, mean_ms=1.8534
  after  repeat50: min_ms=1.7764, mean_ms=1.8325

reuse2, split_kv_len=4096:
  before repeat50: min_ms=1.8245, mean_ms=1.8363
  after  repeat50: min_ms=1.8316, mean_ms=1.8417
```

Decision:

```text
Keep the row-owned combine because it improves the current best reuse4 policy
and removes obviously redundant row-stat work. This is not the main 2.8x lever:
the gain is ~1.1% on reuse4 and neutral/slightly negative on reuse2. The core
gap remains inside the stage kernel's residency, load pipeline, and PV
accumulator pressure.
```

## Coverage Correction: Gemma4 Shape B Extends To 256K

2026-04-29T10:48:00-05:00

The Gemma4 31B target is model coverage, not one frozen benchmark cell. Shape B
uses the global-attention layers and must support the model-config maximum
context:

```text
max_position_embeddings = 262144

Shape B:
  D=512
  group=8
  q_len in {512, 2048}
  kv_len in {8192, 32768, 131072, 262144}
```

Implementation update:

```text
The split-KV fused wrapper now infers q_rows and kv_len from tensor shapes
instead of enforcing q=512 / kv=32768. The first specialization remains Shape B
D512/group8, but it is no longer tied to one q_len/kv_len cell.

The row-owned combine scratch cap was raised from 131072 to 262144 tokens:
  kShapeBMaxKvLen = 262144
  kShapeBMaxKvTiles = 2048

This uses 8 KiB of shared split-weight scratch per combine CTA, which is still
acceptable for the row-owned combine path.
```

Open coverage gap:

```text
Shape A is still not covered by this fused specialization:
  D=256, group=2, kv_len=1024, q_len in {512,2048}

Gemma4 coverage is not complete until Shape A has either:
  1. a dedicated fused specialization, or
  2. an explicit production dispatch policy proving another backend wins those
     cells.
```

Smoke results after parameterization:

```text
Shape B, q=512, kv=8192:
  output_shape=[4096,512], partial_shape=[2,4096,512]
  finite=true, cosine=0.989759
  repeat1 min_ms=1.4693

Shape B, q=512, kv=262144:
  output_shape=[4096,512], partial_shape=[40,4096,512]
  finite=true, cosine=0.991515
  repeat1 min_ms=13.5523

Shape B, q=2048, kv=8192:
  output_shape=[16384,512], partial_shape=[2,16384,512]
  finite=true, cosine=0.991272
  repeat1 min_ms=2.0016
```

The grid harness was updated to include kv_len=262144 and to pass Shape B
dimensions into the fused benchmark. A smoke run with the harness successfully
recorded fused results for q=512 at kv=8192 and kv=262144.

Shape A existing-path check:

```text
Shape A, q=512, kv=1024, D=256, group=2:
  CUTLASS two-stage NVFP4: 0.048480 ms
  FlashInfer NVFP4 FA2:    0.036480 ms
  FlashInfer FP8 FA2:      0.034784 ms  <-- fastest
  BF16:                    0.173056 ms

Shape A, q=2048, kv=1024, D=256, group=2:
  CUTLASS two-stage NVFP4: 0.050144 ms  <-- fastest
  FlashInfer NVFP4 FA2:    0.064288 ms
  FlashInfer FP8 FA2:      0.057344 ms
  BF16:                    0.176352 ms
```

Decision:

```text
Current Gemma4 coverage policy must be per cell:
  Shape A sliding D256/group2: dispatch per q_len with current measurements:
    q=512  -> FP8 FA2
    q=2048 -> CUTLASS two-stage NVFP4

The fused D512 kernel is now a Shape B-grid specialization, not a full Gemma4
kernel by itself, and not automatically the shipping Shape B policy. Shape B
still has to be selected by the grid table because the CUTLASS two-stage
reference remains faster than the fused prototype on known q=512 cells.
```

## Gemma4 Grid Candidate Policy Smoke

2026-04-29T11:07:00-05:00

Ran the new grid harness with warmup=1/repeat=3 for:

```text
Shape A:
  CUTLASS two-stage, NVFP4 FA2, FP8 FA2, BF16

Shape B:
  SM120 fused split-KV
  CUTLASS two-stage
  FP8 FA2
```

Shape A policy from measured candidates:

```text
q=512,  kv=1024: FP8 FA2                0.034784 ms
q=2048, kv=1024: CUTLASS two-stage      0.050144 ms
```

Shape B candidate table:

```text
q=512, kv=8192:
  fused=1.447648 ms, two-stage=0.154560 ms, fp8_fa2=1.320352 ms
  best: CUTLASS two-stage

q=512, kv=32768:
  fused=1.797984 ms, two-stage=0.633856 ms, fp8_fa2=5.035392 ms
  best: CUTLASS two-stage

q=512, kv=131072:
  fused=7.118656 ms, two-stage=3.715072 ms, fp8_fa2=20.044479 ms
  best: CUTLASS two-stage

q=512, kv=262144:
  fused=13.558464 ms, two-stage=7.751552 ms, fp8_fa2=39.897377 ms
  best: CUTLASS two-stage

q=2048, kv=8192:
  fused=1.944448 ms, two-stage=0.629280 ms, fp8_fa2=5.680544 ms
  best: CUTLASS two-stage

q=2048, kv=32768:
  fused=7.130144 ms, two-stage=2.695968 ms, fp8_fa2=24.322304 ms
  best: CUTLASS two-stage

q=2048, kv=131072:
  fused=27.215712 ms, two-stage=17.324320 ms, fp8_fa2=99.449409 ms
  best: CUTLASS two-stage

q=2048, kv=262144:
  fused=52.566303 ms, two-stage=OOM, fp8_fa2=199.291077 ms
  best among completed candidates: SM120 fused split-KV
```

Correction:

```text
The CUTLASS two-stage benchmark is not a production attention kernel. It is a
raw math-throughput reference for:
  FP4 GEMM -> dense softmax -> FP4 quantize -> FP4 GEMM

It does not implement:
  causal masking
  paged KV
  variable-length sequences
  sliding window
  production GQA head/page mapping
  multi-head attention semantics

Therefore it must not be used as a production dispatch candidate. It remains a
useful ceiling/reference for block-scaled FP4 GEMM throughput only.
```

Revised decision:

```text
For production attention, compare against production-capable attention paths:
  existing FP8/FP4 FA2/XQA/BF16 paths
  SM120 fused NVFP4 paths with attention semantics

Do not compare production dispatch against the two-stage dense GEMM benchmark
unless/until it is upgraded into a real attention kernel with causal masking,
paged KV, var-len handling, and GQA semantics.
```

## Paged Serving Workload Check

2026-04-29T11:39:00-05:00

The single-sequence Gemma4 grid is not the same shape as a vLLM serving call.
Serving uses paged KV and grouped GQA over all in-flight sequences. The relevant
validation is therefore a paged/ragged workload benchmark, not just the
contiguous single-sequence grid.

Added:

```text
benchmarks/bench_gemma4_paged_workload_scenarios.py
```

Scenario 1: few long chats, decode-only

```text
batch=4
q_lens=[1,1,1,1]
kv_lens=[262144,262144,262144,262144]
D=512, group=8, page_size=16

NVFP4 XQA decode:
  min_ms=0.539104
  mean_ms=0.557062

FP8 FA2 tensor-core decode:
  min_ms=1.104256
  mean_ms=1.112941
```

Result:

```text
NVFP4 XQA is ~2.05x faster than FP8 FA2 decode on the few-long-chats
long-context decode scenario.
```

Scenario 2: high-concurrency mixed ragged prefill/decode

```text
batch=96
q_lens=(1 x 92) + (512 x 4)
sum_q=2140
kv_lens=(1024 x 24) + (8192 x 24) + (32768 x 24) +
        (131072 x 16) + (262144 x 8)
sum_kv=5201920
D=512, group=8, page_size=16

NVFP4 FA2 paged prefill:
  min_ms=187.463623
  mean_ms=187.484421

FP8 FA2 paged prefill:
  min_ms=201.327072
  mean_ms=201.370997
```

Result:

```text
NVFP4 FA2 is ~1.07x faster than FP8 FA2 on the high-concurrency mixed
ragged-paged scenario.
```

Important interpretation:

```text
The prior contiguous single-sequence dispatch table is not a final vLLM serving
dispatch policy. It is an offline/contiguous reference table, and its
two-stage rows are math-throughput references rather than shippable attention
kernels.

For actual paged serving shapes measured so far:
  few long decode      -> NVFP4 XQA
  high-concurrency mix -> NVFP4 FA2

CUTLASS two-stage is not a production attention path. The current SM120 fused
reference kernel is structurally closer to production Shape B attention, but it
is still a contiguous/reference implementation until paged-KV integration is
completed.
```

Updated production interpretation:

```text
Shape B should be evaluated against FP8 FA2/XQA and other production attention
paths, not against the two-stage dense GEMM benchmark. On the measured Shape B
single-sequence cells, the SM120 fused NVFP4 reference path is much faster than
FP8 FA2, and on paged serving scenarios NVFP4 also beats FP8:
  few-long decode:      NVFP4 XQA ~2.05x faster than FP8 FA2 decode
  high-concurrency mix: NVFP4 FA2 ~1.07x faster than FP8 FA2 prefill

Shape A remains a separate small-KV/sliding-window policy question; existing
FP8 FA2 is still a valid default unless a dedicated D256 NVFP4 path beats it
with production semantics.
```

## Paged Serving Matrix Expansion

2026-04-29T12:05:00-05:00

Expanded the paged workload benchmark to cover:

```text
global_few_long_decode
global_high_concurrency_decode
global_high_concurrency_mixed
sliding_few_long_decode
sliding_high_concurrency_decode
sliding_high_concurrency_mixed
```

The script now also records per-target errors rather than aborting the whole
scenario when a backend cannot instantiate a D512 FA2 configuration.

Global D512/group8 decode:

```text
few long chats:
  batch=4, q=1 each, kv=262144 each

  NVFP4 XQA:        min_ms=0.540928
  NVFP4 FA2 decode: min_ms=1.376288
  FP8 FA2 decode:   min_ms=1.108256
  BF16 FA2 decode:  invalid FlashInfer configuration

  policy: NVFP4 XQA

high-concurrency decode:
  batch=96, q=1 each
  kv distribution:
    1024 x24, 8192 x24, 32768 x24, 131072 x16, 262144 x8

  NVFP4 XQA:        min_ms=17.414783
  NVFP4 FA2 decode: min_ms=7.755584
  FP8 FA2 decode:   min_ms=5.912576
  BF16 FA2 decode:  invalid FlashInfer configuration

  policy: FP8 FA2 decode
```

Global D512/group8 mixed ragged prefill/decode:

```text
batch=96
q_lens=(1 x92) + (512 x4), sum_q=2140
kv distribution:
  1024 x24, 8192 x24, 32768 x24, 131072 x16, 262144 x8

NVFP4 FA2 prefill: min_ms=187.463623
FP8 FA2 prefill:   min_ms=201.327072
BF16 FA2 prefill:  invalid FlashInfer configuration

policy: NVFP4 FA2
```

Sliding D256/group2 decode:

```text
few long chats, sliding layers:
  batch=4, q=1 each, kv=1024 each

  BF16 FA2 decode:  min_ms=0.022368
  FP8 FA2 decode:   min_ms=0.026112
  NVFP4 FA2 decode: min_ms=0.030752
  NVFP4 XQA:        min_ms=0.032192

  policy: BF16 FA2 decode if capacity permits, otherwise FP8 FA2

high-concurrency sliding decode:
  batch=96, q=1 each, kv=1024 each

  FP8 FA2 decode:   min_ms=0.044000
  BF16 FA2 decode:  min_ms=0.046080
  NVFP4 XQA:        min_ms=0.057472
  NVFP4 FA2 decode: min_ms=0.058176

  policy: FP8 FA2 decode
```

Sliding D256/group2 mixed ragged prefill/decode:

```text
batch=96
q_lens=(1 x92) + (512 x4), sum_q=2140
kv=1024 for all sequences

BF16 FA2 prefill:  min_ms=0.072288
FP8 FA2 prefill:   min_ms=0.096736
NVFP4 FA2 prefill: min_ms=0.108128

policy: BF16 FA2 if capacity permits, otherwise FP8 FA2
```

Layer-weighted estimates:

```text
Gemma4 31B layer mix:
  global Shape B:  10 layers
  sliding Shape A: 50 layers

few-long decode:
  global policy:  NVFP4 XQA        0.540928 ms
  sliding policy: BF16 FA2 decode  0.022368 ms
  weighted: 10*0.540928 + 50*0.022368 = 6.527680 ms

high-concurrency decode:
  global policy:  FP8 FA2 decode   5.912576 ms
  sliding policy: FP8 FA2 decode   0.044000 ms
  weighted: 10*5.912576 + 50*0.044000 = 61.325760 ms

high-concurrency mixed:
  global policy:  NVFP4 FA2 prefill 187.463623 ms
  sliding policy: BF16 FA2 prefill    0.072288 ms
  weighted: 10*187.463623 + 50*0.072288 = 1878.250630 ms
```

Interpretation:

```text
There is no single Gemma4 attention backend policy. The right policy is
workload- and layer-class-dependent:

  global long-context decode with few requests -> NVFP4 XQA
  global high-concurrency decode              -> FP8 FA2 decode
  global high-concurrency mixed/prefill        -> NVFP4 FA2
  sliding decode                               -> FP8/BF16 FA2
  sliding mixed/prefill                        -> BF16/FP8 FA2

The next production decision should be made against the live vLLM workload
mix: fraction of time spent in decode-only vs mixed prefill/decode, and the
batch-size/context distribution at the attention backend boundary.
```

## Dense Shape B NVFP4 FA2 Comparison

Measured the missing dense Shape B comparison against NVFP4 FA2 on the same
single-sequence grid as the SM120 fused kernel.

Command:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
  /home/josh/tdm/infer/current/.venv/bin/python \
  benchmarks/bench_gemma4_attention_grid.py \
  --device 0 --warmup 3 --repeat 10 --timeout-sec 2400 \
  --shapes B --kernels flashinfer_nvfp4_fa2 \
  --output-prefix reports/gemma4_grid_shape_b_nvfp4_fa2_dense
```

Dense Shape B, D512/group8:

```text
q     kv       SM120 fused ms   NVFP4 FA2 ms   winner       gap
512   8192     1.447648         1.239360       NVFP4 FA2    1.17x
512   32768    1.797984         4.702976       SM120 fused  2.62x
512   131072   7.118656         18.605600      SM120 fused  2.61x
512   262144   13.558464        37.257439      SM120 fused  2.75x
2048  8192     1.944448         5.318048       SM120 fused  2.73x
2048  32768    7.130144         22.368095      SM120 fused  3.14x
2048  131072   27.215712        89.971970      SM120 fused  3.31x
2048  262144   52.566303        184.741852     SM120 fused  3.51x
```

Conclusion:

```text
The dense Shape B win is real against NVFP4 FA2, not only against FP8 FA2.
The SM120 fused kernel wins 7/8 dense Shape B cells. The only losing cell is
q=512, kv=8192, where NVFP4 FA2 is 1.17x faster.

This does not change the known production gaps:
  - no Shape A D256/group2 specialization
  - no q<128 decode path
  - no paged KV integration
  - no production causal/ragged integration
```

## Head-Dim Parameterization Bring-Up

Opened the SM120 fused split-KV wrapper and grid harness for head dimensions
128, 256, and 512.

Implementation notes:

```text
- The wrapper now accepts D in {128, 256, 512}.
- The Python benchmark no longer rejects D128/D256.
- The Gemma4 grid harness now allows sm120_fused for D128/D256/D512 cells.
- Output-group reuse dispatch degrades by available width:
    D512 -> reuse4
    D256 -> reuse2
    D128 -> reuse1
- QK scaling is now runtime 1/sqrt(D) instead of fixed 1/sqrt(512).
```

Important finding:

```text
The D128/D256 paths are not true one-K-chunk QK paths. The SM120 QK collective
must keep the existing two-stage QK cadence even when logical head_dim is 128 or
256. When the one-chunk path was tried, the online softmax stats corrupted:
split_l became inf and output scale exploded. Keeping two QK stages restores
correctness.
```

Smoke results:

```text
D256 / group2 / kv1024 / q512:
  mean_abs=0.0009214, max_abs=0.0053578, cosine=0.9898322
  min_ms=0.181056

D128 / group2 / kv1024 / q512:
  mean_abs=0.0009627, max_abs=0.0055012, cosine=0.9916502
  min_ms=0.152352

D128 / group2 / kv1024 / q2048:
  mean_abs=0.0009659, max_abs=0.0056391, cosine=0.9916368
  min_ms=0.156704

D512 / group8 / kv32768 / q512 regression check:
  mean_abs=0.0001588, max_abs=0.0007830, cosine=0.9929289
  min_ms=1.833536
```

Gemma4 Shape A grid through the generic harness:

```text
D256 / group2 / kv1024 / q512:  min_ms=0.174208
D256 / group2 / kv1024 / q2048: min_ms=0.177696
```

Interpretation:

```text
The fused kernel now functionally covers D128 and Gemma4 Shape A D256, but this
is only a correctness/coverage bring-up. It still uses the D512-era 96 KiB
shared-storage footprint and the same role/pipeline shape.

This leaves known D256/D128 specialization headroom:
  - compact shared storage instead of the D512 storage footprint
  - deeper K/V pipelines where smem allows
  - possible 2 CTAs/SM if register/smem occupancy permits
  - D256-native reuse2 and D128-native reuse1 kernels instead of runtime
    dispatch through the D512-shaped scaffold
```

## D128 Full-Context Dense Sweep

The first D128 smoke only covered `kv=1024`, which is not representative for a
full-attention D128 model. Ran the long-KV dense sweep directly through the
SM120 fused benchmark because the Gemma4 grid harness maps Shape B to D512 by
definition.

Command shape:

```bash
benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py \
  --device 0 \
  --q-len {512,2048} \
  --group 2 \
  --head-dim 128 \
  --kv-len {8192,32768,131072,262144} \
  --split-kv-len 6656 \
  --warmup 2 --repeat 5 \
  --sm120-qkv-online-splitkv-reuse4-full-grid-bench
```

D128/group2 dense full-context results:

```text
q     kv       min_ms    mean_ms   cosine    mean_abs       splits
512   8192     0.752576  0.755827  0.991670  0.000358909    2
512   32768    0.744288  0.749530  0.991927  0.000163162    5
512   131072   0.778464  0.780704  0.992852  0.000084713    20
512   262144   1.540544  1.545267  0.989608  0.000060659    40
2048  8192     0.734016  0.737984  0.991690  0.000358841    2
2048  32768    0.781024  0.783661  0.991925  0.000162758    5
2048  131072   2.992704  2.998445  0.992812  0.000084998    20
2048  262144   5.319200  5.329664  0.989474  0.000060928    40
```

Interpretation:

```text
D128 now runs at long KV and preserves the same approximate correctness band as
the D256/D512 bring-up: finite output and cosine around 0.99 against the
first-tile exact reference.

The q=512 times are nearly flat from 8K through 131K because split-KV exposes
enough CTA parallelism to keep the work near one SM wave:
  q_rows = 512 * group2 = 1024
  q_tiles = 1024 / 128 = 8
  splits at 131K = 20
  CTAs = 8 * 20 = 160, still below 188 SMs

At 262K:
  splits = 40
  CTAs = 8 * 40 = 320
  this becomes roughly two waves, and wall time roughly doubles.

For q=2048:
  q_rows = 4096
  q_tiles = 32
  CTAs at 131K = 32 * 20 = 640
  CTAs at 262K = 32 * 40 = 1280
  wall time scales with the additional SM waves.
```

Remaining caveats:

```text
This is still dense, non-paged, q>=128 only, and not production causal/ragged
integration. It also uses group=2 because that matched the D128 smoke shape.
If a target D128 model has a different GQA group, it needs its own sweep.
```

## D128 NVFP4 FA2 Comparison

Measured the matching D128/group2 dense grid against NVFP4 FA2.

Command shape:

```bash
benchmarks/bench_nvfp4_fmha_v2_gqa_grouped_attention.py \
  --device 0 \
  --q-len {512,2048} \
  --kv-len {8192,32768,131072,262144} \
  --head-dim 128 \
  --group-sizes 2 \
  --batch-size 1 \
  --only grouped-fp4 \
  --fp4-backend fa2 \
  --fp4-v-layout nhd \
  --fp4-v-sf-layout linear
```

D128/group2 dense fused vs NVFP4 FA2:

```text
q     kv       SM120 fused ms   NVFP4 FA2 ms   winner     gap
512   8192     0.752576         0.048640       NVFP4 FA2  15.47x
512   32768    0.744288         0.111040       NVFP4 FA2  6.70x
512   131072   0.778464         0.375168       NVFP4 FA2  2.07x
512   262144   1.540544         0.730976       NVFP4 FA2  2.11x
2048  8192     0.734016         0.116224       NVFP4 FA2  6.32x
2048  32768    0.781024         0.386048       NVFP4 FA2  2.02x
2048  131072   2.992704         1.492480       NVFP4 FA2  2.01x
2048  262144   5.319200         2.961472       NVFP4 FA2  1.80x
```

Conclusion:

```text
The fused kernel does not win at D128. NVFP4 FA2 wins all 8 dense D128/group2
long-context cells, by 1.8x to 15.5x.

This means the current fused kernel's structural advantage is not generic to
D128. The D512 win appears tied to the D512/global-attention surface where
output-group reuse and long-KV split-KV amortization overcome the fused kernel's
fixed overhead. At D128, FA2's lower overhead and mature small-head path dominate.

Current dispatch implication:
  - D512 dense prefill long-KV: SM120 fused remains useful.
  - D256/D128: current fused path is correctness coverage only; do not dispatch
    it over NVFP4 FA2 without a real D256/D128 specialization proving a win.
```

## Native D128/D256 Specialization Hypothesis

The current D128/D256 measurements are scaffold measurements, not native
D128/D256 ceilings. They reuse the D512-oriented fused kernel shape:

```text
- 96 KiB shared-storage allocation
- two-stage QK cadence
- D512 role/pipeline sizing
- output-group reuse machinery sized for reuse4
- one CTA/SM resource profile
```

A native head-dim specialization should have different constraints:

```text
choice                 D512 current        D256 native target   D128 native target
smem footprint         ~96 KiB             ~64 KiB              ~48 KiB
CTAs/SM                1                   1-2                  2
K pipeline depth       2                   2-3                  3
V pipeline depth       2                   2-3                  3
K cadence              two-stage natural   single-stage native  single-stage native
output reuse span      4                   2                    1
epilogue smem          32 KiB              16 KiB               8 KiB
register budget        168 capped          lower target         lower target
```

The likely D128 win mechanism is not output-group reuse. D128 has only one
128-column output group, so the win has to come from lower fixed overhead,
compact shared storage, deeper pipelines, higher split-KV parallelism, and
possibly 2 CTAs/SM.

Structural prediction:

```text
Current D128 scaffold at q512/kv32K-131K is ~0.75 ms and loses to NVFP4 FA2.
A native D128 kernel that reaches 2 CTAs/SM and removes the D512 scaffolding
could plausibly land around 0.35-0.40 ms, which would be competitive at long KV.

This is a hypothesis, not a result. It requires a real D128-native kernel:
  - compact D128 shared storage
  - D128-native load cadence
  - D128-native reuse1 path, not reuse4 dispatch degraded at runtime
  - measured resource profile showing 2 CTAs/SM or a concrete reason it cannot
    happen
```

Decision boundary:

```text
The current scaffold result answers only "does the D512-shaped kernel win at
D128?" The answer is no.

The production/ceiling question remains open until a D128-native kernel is
measured against NVFP4 FA2.
```

Resource finding from the active CUTLASS collectives:

```text
SM120 opt-in shared memory:                   101376 bytes
Current QKV fused storage:                     96256 bytes
Current QK-only load collective storage:       74752 bytes
CUTLASS 128x128x128 tensor storage:            73728 bytes
CUTLASS 128x128x128 shared storage:            74752 bytes
Current PV K128 stage-2 tensor storage:        36864 bytes
Current PV K128 stage-2 shared storage:        37888 bytes
```

This blocks one of the hoped-for native-D128 levers:

```text
2 CTAs/SM requires dynamic smem <= 101376 / 2 = 50688 bytes.

The current CUTLASS QK mainloop storage alone is 74752 bytes, before any
softmax, P staging, V staging, row stats, or role-pipeline storage. Therefore a
"native D128" kernel built by only specializing the existing CUTLASS collective
cannot reach 2 CTAs/SM.
```

Implication:

```text
The D128-native path is not just a smaller template instantiation of the current
kernel. To test the actual D128 ceiling, we need one of:
  - a custom compact Q/K/V smem layout around the SM120 block-scaled MMA atom,
    not the full CUTLASS TMA TensorStorage; or
  - a smaller CUTLASS tile shape whose TensorStorage lands under ~50 KiB; or
  - a different D128 strategy that wins without 2 CTAs/SM.

Until then, D128 remains FA2 territory in the dispatch policy.
```

## D128/D256 Kernel File Split

The D128 and D256 specialization work now has separate translation units instead
of routing every head dimension through the D512 kernel file:

```text
benchmarks/sm120_nvfp4_cutlass_fused_attention.cu       D512 baseline
benchmarks/sm120_nvfp4_cutlass_fused_attention_d256.cu  D256 specialization seed
benchmarks/sm120_nvfp4_cutlass_fused_attention_d128.cu  D128 specialization seed
```

The Python benchmark loader now selects the source file and extension name from
`--head-dim`, so the D128/D256 kernels can diverge without invalidating or
destabilizing the D512 path.

Initial smoke results after the split:

```text
D256 q512/group2/kv1024: finite, cosine 0.989832, min 0.17744 ms
D128 q512/group2/kv1024: finite, cosine 0.991650, min 0.153792 ms
```

These files are copied scaffold seeds. They still report the D512-shaped
`storage_bytes=96256` footprint. The next native-specialization work is to shrink
the D256/D128 storage and role machinery inside the dedicated files.

## D256 Shape-A First Specialization Pass

Focused the first D256-only pass on Gemma4 Shape A:

```text
D=256, group=2, kv_len=1024
q_len in {512, 2048}
```

Changes:

```text
1. D256 split-KV wrapper now writes directly to `out` and skips the combine
   kernel when `num_splits == 1`.
2. D256 dispatch is locked to output span 1. Span1 is faster than span2/span4
   for SWA because it exposes more CTAs and avoids over-reuse inside each CTA.
3. Gemma4 grid harness now uses the D256 span1 path and q-dependent split-KV:
   q512  -> split_kv_len=128
   q2048 -> split_kv_len=512
```

Measured D256 span sweep at `kv=1024`:

```text
q512:
  split 1024 -> 0.150240 ms
  split 512  -> 0.098240 ms
  split 256  -> 0.070368 ms
  split 128  -> 0.061888 ms  best

q2048:
  split 1024 -> 0.149856 ms
  split 512  -> 0.106272 ms  best
  split 256  -> 0.132768 ms
  split 128  -> 0.147776 ms
```

Updated Shape A comparison:

```text
q512 / kv1024:
  sm120_fused D256:       0.061792 ms
  CUTLASS two-stage:      0.047712 ms
  FlashInfer NVFP4 FA2:   0.034496 ms
  FlashInfer FP8 FA2:     0.028992 ms  best
  FlashInfer BF16:        0.167264 ms

q2048 / kv1024:
  sm120_fused D256:       0.106528 ms
  CUTLASS two-stage:      0.048576 ms  best benchmark path
  FlashInfer NVFP4 FA2:   0.059584 ms
  FlashInfer FP8 FA2:     0.054016 ms
  FlashInfer BF16:        0.166112 ms
```

Interpretation:

```text
The D256 split-KV/dispatch tuning moved the fused scaffold from ~0.174 ms to:
  q512  -> ~0.062 ms
  q2048 -> ~0.106 ms

This is a real D256 improvement, but not enough to beat the existing Shape A
paths. The remaining gap is still structural: the D256 kernel retains the
D512-shaped 96 KiB shared-memory footprint and 384-thread role structure.
```

NCU spot profile for the best q512 fused cell:

```text
shape: D256/group2/q512/kv1024/split_kv_len=128/span1

stage kernel:
  duration:                    46.592 us
  registers/thread:            168
  dynamic smem/block:          96256 bytes
  issue active:                8.87%
  eligible warps/cycle:        0.10
  active warps/cycle:          2.84
  tensor pipe active:          5.04%
  avg warp latency/inst:       31.97 cycles
  long scoreboard stall:       3.56 cycles/issue-active
  sleeping stall:              16.27 cycles/issue-active

combine kernel:
  duration:                    11.616 us
  issue active:                7.93%
  eligible warps/cycle:        0.10
  avg warp latency/inst:       133.86 cycles
  long scoreboard stall:       49.35 cycles/issue-active
```

Interpretation:

```text
The combine kernel is visible but not the primary 2x gap. The stage kernel is
still the dominant cost, and it is under-utilized: low issue-active, low tensor
activity, and high sleeping stalls from the role-specialized D512-era structure.

For D256 SWA with one or a few KV tiles per split, the FA3-style role pipeline
has less overlap to exploit. The next serious D256-native path should be either:
  - compact storage/aliasing that gets below the one-CTA/SM 96 KiB footprint, or
  - a D256 SWA one-tile/small-tile kernel with less role specialization overhead.
```

## D256 Expanded Hillclimb Baseline

Expanded the D256 hillclimb harness to include the wider GQA group surface:

```text
q_len:  {128, 256, 512, 1024, 2048, 4096}
kv_len: {8192, 32768, 65536, 131072, 262144}
group:  {2, 4, 6, 8, 12, 16}
```

Report files:

```text
reports/d256_hillclimb_baseline_20260429.jsonl
reports/d256_hillclimb_baseline_20260429.csv
reports/d256_hillclimb_baseline_20260429.summary.csv
reports/d256_hillclimb_baseline_20260429.md
```

Matrix size:

```text
180 shape cells
720 raw timing rows
4 kernels per cell:
  sm120_fused
  nvfp4_fa2
  fp8_fa2
  bf16_fa2
```

Current result against the hillclimb target:

```text
cells beating nvfp4_fa2 by >=2x: 0 / 180
cells where sm120_fused beats nvfp4_fa2: 13 / 180
cells where sm120_fused beats fp8_fa2:   6 / 180
cells where sm120_fused beats bf16_fa2:  0 / 180

cosine range:
  min 0.9884289503
  avg 0.9913249615
  max 0.9934706688
```

By group:

```text
group  pass2x  beat_nvfp4  median speedup  best speedup  worst speedup
2      0/30    0/30        0.647           0.963         0.064
4      0/30    0/30        0.693           0.968         0.079
6      0/30    6/30        0.745           1.244         0.101
8      0/30    0/30        0.762           0.972         0.124
12     0/30    7/30        0.803           1.244         0.165
16     0/30    0/30        0.775           0.974         0.206
```

Best current cells are long-context group 6 and group 12:

```text
group q_len kv_len  sm120_fused_ms  nvfp4_fa2_ms  speedup
12    1024  262144  30.645          38.125        1.244
6     2048  262144  30.830          38.355        1.244
12    1024  131072  15.489          18.995        1.226
6     2048  131072  15.580          19.009        1.220
12    1024  65536   7.979           9.467         1.186
6     2048  65536   8.021           9.467         1.180
```

Interpretation:

```text
The current D256 fused path is correct enough for hillclimb comparison but not
yet structurally competitive with the 2x target. The only win pockets are
long-context group 6 and group 12, and even those top out at ~1.24x over
nvfp4_fa2. Group 16 does not create a win pocket in this baseline.

The next optimization should be picked from the worst target gap under this
expanded matrix, not from Gemma4 group2 alone. The baseline says the generic
D256 scaffold still carries too much D512-era overhead across the broader
group surface.
```

## D256 Active Hillclimb Surface Narrowed

The D256 hillclimb surface is now narrowed to the high-q, high-context cells
that match the large-batch operating envelope we care about next:

```text
q_len:  {2048, 4096, 8192, 16384, 32768}
kv_len: {16384, 32768, 65536, 131072, 262144}
group:  {2, 4, 6, 8}
```

Dropped from the active optimization loop:

```text
q_len < 2048
kv_len < 16384
group: {12, 16}
```

Group 2 is retained because it is the Gemma4 D256 sliding-attention shape.
Groups 4, 6, and 8 remain in the active loop because the broader D256 surface
showed better long-context scaling at higher GQA group sizes.

Current harness default:

```text
benchmarks/bench_sm120_d256_hillclimb.py
  DEFAULT_Q_LENS  = (2048, 4096, 8192, 16384, 32768)
  DEFAULT_KV_LENS = (16384, 32768, 65536, 131072, 262144)
  DEFAULT_GROUPS  = (2, 4, 6, 8)
```

## Rejected Baseline: FlashInfer FMHA v2 For NVFP4 Gate

The hillclimb gate remains `nvfp4_fa2`, not FlashInfer FMHA v2.

Reason from the vLLM dispatch path in `worktrees/vllm-nvfp4-kv`:

```text
vllm/platforms/cuda.py
  SM120/SM121 prioritizes the FlashInfer attention backend.

vllm/v1/attention/backends/flashinfer.py
  _get_prefill_wrapper() constructs BatchPrefillWithPagedKVCacheWrapper(...,
  backend="auto") for the native FlashInfer path.

  _can_use_fmha_v2_prefill() rejects quantized KV because it requires
  self.cache_dtype == "auto".

  The FMHA v2 direct prefill branch later asserts not self.is_kvcache_nvfp4.
```

So for NVFP4 KV cache on SM120, the production path is the FlashInfer wrapper
auto path, which resolves to the FA2-style NVFP4 implementation. FMHA v2 is not
the decision gate for this hillclimb.

Smoke timing on D256 group=2 q=2048 kv=16384 also confirms FMHA v2 is not a
hidden fast baseline:

```text
flashinfer_nvfp4_fmha_v2: 5.696 ms
nvfp4_fa2:                0.539 ms
```

The FMHA v2 column was removed from the default D256 hillclimb harness after
this check. Keep `nvfp4_fa2` as the primary gate.

## SM120f Build And Native MMA Verification

The D256 fused attention benchmark extension is compiled with CUDA 13.2 and
SM120 family-conditional codegen:

```text
build.ninja:
  nvcc = /usr/local/cuda-13.2/bin/nvcc
  -gencode=arch=compute_120f,code=sm_120f
```

The torch extension cache directory is still named `py312_cu130` because the
PyTorch wheel is CUDA 13.0, but the actual extension build uses CUDA 13.2 for
headers, nvcc, link flags, and the SM120f gencode.

`cuobjdump --list-elf` labels the emitted cubin as `sm_120`, but the SASS
contains the native scaled FP4 tensor-core path:

```text
OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X
```

This means the active D256 fused kernel is not falling back to a software
dequant + BF16 MMA path. The remaining D256 gap against `nvfp4_fa2` is a kernel
structure / scheduling / occupancy issue, not a missing SM120f codegen issue.

FlashInfer JIT cache check:

```text
0.6.9/120f/.../batch_prefill_dtype_q_bf16_dtype_kv_u8...D256/build.ninja:
  cuda_home = /usr/local/cuda-13.2
  -gencode=arch=compute_120f,code=sm_120f

0.6.9/120a/.../batch_prefill_dtype_q_bf16_dtype_kv_u8...D256/build.ninja:
  cuda_home = /usr/local/cuda-13.0
  -gencode=arch=compute_120a,code=sm_120a
```

Both cache families exist on disk from different phases of the project. The
D256 hillclimb harness now sets `CUDA_HOME=/usr/local/cuda-13.2`,
`TORCH_CUDA_ARCH_LIST=12.0f`, and `FLASHINFER_CUDA_ARCH_LIST=12.0f`, so future
baseline runs target the 0.6.9/120f artifact explicitly. Old 120a reports
should not be compared against current 120f fused runs without labeling the
architecture target.

## D256 Hillclimb Procedure: Structural Wins First

The first broad D256 baseline sweep was intentionally useful but too expensive
for the inner loop. It also exposed a harness bug: results were buffered in
memory and only written at the end, so the final heavy-cell failure destroyed
the final report. The harness now appends every completed row to JSONL/CSV
immediately and rewrites summary Markdown/CSV after each row. Failed cells are
recorded as `status=error` rows instead of losing the run.

Baseline failure to remember:

```text
D=256, group=8, q=32768, kv=262144, split_kv_len=6656, span=1
sm120_fused: illegal memory access during cuda synchronize
```

This is a fused-kernel extreme-shape bug, not a baseline FA2 failure. It should
be debugged when that shape becomes an active target, but it should not block
the D256 structural hillclimb.

Inner-loop policy:

```text
Do not run the full grid after every feature.
Use 1-2 focused cells for each structural change, then run the broader grid only
at checkpoints.
```

Focused cells for the next D256 phase are now group=6 only. This is the active
online/offline D256 operating surface:

```text
q=8192:
  kv={16384, 32768, 65536, 131072, 262144}
  Regime: online vLLM default, mid through max context.

q=16384:
  kv={16384, 32768, 65536, 131072, 262144}
  Regime: offline / 16K setting, mid through max context.

q=32768:
  kv={32768, 65536, 131072, 262144}
  Regime: high-throughput offline, longer through max context.

Optional extreme safety cell:
  group=8, q=32768, kv=262144
  Reason: currently triggers illegal memory access. This is not part of the
  active group=6 hillclimb; run it only when checking indexing/workspace fixes.
```

Structural priority order:

```text
1. D256-native reuse2 kernel.
   D256 is exactly 2 x 128 output groups, so reuse2 is the natural shape. Remove
   D512/reuse4 assumptions rather than only selecting span2 at dispatch.

2. Reduce shared-memory footprint toward <64 KiB.
   Current D256 path still carries D512-era storage. The goal is to unlock
   either 2 CTAs/SM or K/V pipeline depth 3.

3. K/V pipeline depth 3.
   Only after the smem reduction. If D256 cannot fit depth 3, the smem cleanup
   is incomplete.

4. Occupancy target / 2 CTAs per SM.
   Use NCU on the worst focused cell after steps 1-3. If eligible warps and
   issue-slot busy are still low, prefer occupancy even if it sacrifices some
   reuse.

5. Shape-dependent dispatch policy.
   group=2 may prefer a low-overhead path; group=4/6/8 may prefer reuse2 and
   deeper split-KV. Dispatch policy comes after the native variants exist.
```

Per-change loop:

```text
1. Run focused cells only.
2. Record min_ms, cosine, and nvfp4_fa2 speedup.
3. NCU the worst focused cell if the result does not explain itself.
4. Keep the change only if correctness holds and at least one target cell moves
   materially without unacceptable regressions.
5. Write one worklog entry after the structural change lands.
6. Run the broader grid only at checkpoints, not after every edit.
```

## D256 Group=6 Focus Baseline

Baseline reports:

```text
reports/d256_group6_focus_baseline_20260429.{jsonl,csv,summary.csv,md}
reports/d256_group6_focus_q32768_baseline_20260429.{jsonl,csv,summary.csv,md}
```

These runs use CUDA 13.2 / SM120f and stream each completed row to disk. They
cover the active D256 group=6 focus surface:

```text
q=8192,  kv={16384, 32768, 65536, 131072, 262144}
q=16384, kv={16384, 32768, 65536, 131072, 262144}
q=32768, kv={32768, 65536, 131072, 262144}
```

Baseline result:

```text
q      kv      fused_ms  nvfp4_fa2_ms  fp8_fa2_ms  bf16_fa2_ms  nvfp4/fused  cosine
8192   16384   7.844     5.022         4.459       3.044        0.64x        0.9907
8192   32768   15.474    11.639        10.353      7.072        0.75x        0.9935
8192   65536   30.573    24.883        22.051      15.077       0.81x        0.9899
8192   131072  60.693    51.536        45.748      31.585       0.85x        0.9884
8192   262144  125.529   105.270       95.692      67.072       0.84x        0.9910
16384  16384   15.494    6.205         5.493       3.744        0.40x        0.9903
16384  32768   30.721    18.301        16.263      11.065       0.60x        0.9906
16384  65536   60.732    42.675        37.991      25.733       0.70x        0.9903
16384  131072  127.497   93.650        83.637      55.857       0.73x        0.9912
16384  262144  263.427   199.966       180.254     121.975      0.76x        0.9914
32768  32768   60.892    23.579        20.990      14.246       0.39x        0.9908
32768  65536   125.690   70.974        62.952      42.296       0.56x        0.9923
32768  131072  259.593   173.700       156.151     103.531      0.67x        0.9911
32768  262144  525.793   376.761       338.681     230.227      0.72x        0.9915
```

Interpretation:

```text
The D256 fused path is correct but loses to nvfp4_fa2 on every active group=6
cell. The gap is worst at high-q / shorter-kv cells and narrows as kv grows.
This points at D512-era reuse/occupancy structure and launch/CTA scheduling
overheads, not missing native FP4 tensor-core codegen.

Next lever: measure D256 reuse2/span2 on the same focused cells. D=256 maps
naturally to two 128-column output groups, so this is the first structural
check before deeper smem/pipeline edits.
```

## D256 Group=6 Span2 Lever Result

Reports:

```text
reports/d256_group6_focus_span2_fused_20260429.{jsonl,csv,summary.csv,md}
reports/d256_group6_focus_q32768_span2_fused_20260429.{jsonl,csv,summary.csv,md}
```

The span2 path is the existing D256 `requested_output_group_span=2` dispatch.
It is not yet a fully native D256 storage/pipeline rewrite, but it changes the
output-group reuse structure from span1 to the D=256-natural two 128-column
groups.

Result:

```text
q      kv      span1_ms  span2_ms  span2/span1  nvfp4_fa2_ms  nvfp4/span2  cosine
8192   16384   7.844     5.181     0.66x        5.022         0.97x        0.9907
8192   32768   15.474    10.234    0.66x        11.639        1.14x        0.9935
8192   65536   30.573    19.964    0.65x        24.883        1.25x        0.9899
8192   131072  60.693    39.811    0.66x        51.536        1.29x        0.9884
8192   262144  125.529   83.543    0.67x        105.270       1.26x        0.9910
16384  16384   15.494    10.310    0.67x        6.205         0.60x        0.9903
16384  32768   30.721    20.276    0.66x        18.301        0.90x        0.9906
16384  65536   60.732    40.613    0.67x        42.675        1.05x        0.9903
16384  131072  127.497   84.186    0.66x        93.650        1.11x        0.9912
16384  262144  263.427   176.612   0.67x        199.966       1.13x        0.9914
32768  32768   60.892    40.270    0.66x        23.579        0.59x        0.9908
32768  65536   125.690   83.759    0.67x        70.974        0.85x        0.9923
32768  131072  259.593   175.252   0.68x        173.700       0.99x        0.9911
32768  262144  525.793   355.971   0.68x        376.761       1.06x        0.9915
```

Interpretation:

```text
Span2 is a real structural win: 32-35% wall-time reduction on every focused
cell with unchanged correctness. It turns q=8192 into a mostly winning surface
and wins long-context cells at q=16384/q=32768, but short and mid-kv high-q
cells still lose badly.

The remaining worst cell is q=32768, kv=32768, group=6:
  span2 fused: 40.270 ms
  nvfp4_fa2:   23.579 ms
  gap:         nvfp4_fa2/span2 = 0.59x

Next diagnostic target: NCU this span2 worst cell. The likely bottleneck is
still occupancy/scheduler pressure from D512-era storage and role structure,
not output-group reuse. The next implementation lever should be selected from
the profile, with smem-footprint reduction toward <64 KiB and 2 CTAs/SM as the
primary candidate.
```

## D256 Group=6 Span2 NCU: Worst Focus Cell

Profile report:

```text
reports/ncu_d256_g6_q32768_kv32768_span2_20260429.ncu-rep
reports/ncu_d256_g6_q32768_kv32768_span2_20260429.details.txt
```

Profiled cell:

```text
D=256, group=6, q=32768, kv=32768, span2, split_kv_len=6656
```

Key NCU counters:

```text
dynamic shared memory per block:   96256 B
registers/thread:                  168
theoretical occupancy:             25%
achieved occupancy:                24.73%
active warps/scheduler:            2.98
eligible warps/scheduler:          0.19
issued warp/scheduler:             0.16
issue slots busy:                  15.71%
SM busy:                           16.76%
tensor pipe active:                16.76%
mem busy:                          73.48%
max bandwidth:                     69.09%
local memory spilling requests:    0
shared memory spilling requests:   0
top stalls:
  long scoreboard:                 5.54 cycles/issued inst
  sleeping:                        4.25
  wait:                            2.22
  short scoreboard:                1.94
  barrier:                         0.45
```

Interpretation:

```text
The kernel is not spilling. Barrier stalls are no longer the headline for this
cell. The active problem is low eligibility from one CTA/SM plus memory-wait
behavior: 96 KiB smem forces 25% theoretical occupancy, leaving only 0.19
eligible warps/scheduler and 15.7% issue-slot busy.

The previous "<64 KiB" target is insufficient for 2 CTAs/SM on this RTX PRO
6000, because the SM reports ~100 KiB usable shared memory. A true 2-CTA target
requires roughly <=50 KiB dynamic shared memory per CTA after driver overhead.

Current storage breakdown from metadata:
  qk tensor storage:                73728 B
  q/k pipeline overhead included:   qk load collective = 74752 B
  v stage sidecar + pipelines:      total qkv storage = 96256 B
  qk_smem_A:                        32768 B
  qk_smem_B:                        32768 B
  qk_smem_SFA/SFB:                  4096 B each
  PV P stage:                       16384 B
  two compact P stages:             32768 B

Next structural lever is not another span/threshold sweep. It is a native D256
storage rewrite aimed at 2 CTAs/SM:
  - reduce Q tile height from 128 to 64,
  - use the D256-natural K128 QK mainloop chunks,
  - shrink/serialize the P staging so it fits the smaller QK A/SFA storage,
  - keep span2 as the D256 output-group policy.

This may sacrifice some QK/PV overlap, but the profile says the larger lever is
warp eligibility/occupancy. Keep the change only if the focused cells improve
and correctness stays intact.
```

## D256 Data-Movement Pivot: Manual QK Primitive

The naive M64 `CollectiveBuilder` path was rejected because the SM120
block-scaled TMA collective owns scale-side layout constraints that are not
shape-generic under a simple tile-shape swap. The active D256-native direction
is now:

```text
Do not use the CUTLASS GEMM collective as the data-movement layer for the
native D256 path.

Keep:
  SM120 block-scaled FP4 MMA atom
  MMA_Traits fragment layouts
  row-major NVFP4 + UE4M3 scale semantics

Replace:
  CUTLASS CollectiveBuilder TensorStorage / TMA sidecar layout
with:
  our own compact Q/K/V/P data movement around the atom.
```

First landed primitive:

```text
sm120_d256_manual_qk_m64n128
```

Shape:

```text
M = 64 Q rows
N = 128 KV rows
K = 256 head dim
8 warps / 256 threads
one warp computes one or more 16x16 output atoms
direct fragment construction through cute::MMA_Traits<Fp4MmaAtom>
no CUTLASS CollectiveBuilder storage
no TMA collective layout
no dynamic shared memory
```

Correctness:

```text
q=8192, kv=16384, group=6, D=256
tile (qbase=0,    kvbase=0):    mean_abs=0, max_abs=0, cosine=1.0
tile (qbase=64,   kvbase=128):  mean_abs=0, max_abs=0, cosine=1.0
tile (qbase=256,  kvbase=1024): mean_abs=0, max_abs=0, cosine=1.0000001
tile (qbase=4096, kvbase=8192): mean_abs=0, max_abs=0, cosine=0.9999999
```

Benchmark:

```text
bench_manual_qk_m64n128:
  min_ms:  0.06428799778223038
  mean_ms: 0.06570666780074437
  max_ms:  0.06777600198984146
```

Interpretation:

```text
The atom-level D256 QK data movement is correct for nonzero Q and KV offsets.
This is not yet the performance path: it directly constructs fragments from
global memory and therefore is a correctness/contract primitive. The next layer
is to put this same fragment mapping behind a compact producer/cache for Q and
K so the fused mainloop can keep <=50 KiB dynamic shared memory and target
2 CTAs/SM.
```

Second landed primitive:

```text
sm120_d256_manual_pv_m64n128k128
```

Shape:

```text
M = 64 P rows
N = 128 output columns
K = 128 KV tile columns
8 warps / 256 threads
P is the A operand, V^T is the B operand
direct fragment construction through cute::MMA_Traits<Fp4MmaAtom>
no CUTLASS CollectiveBuilder storage
no TMA collective layout
no dynamic shared memory
```

Correctness:

```text
q=8192, kv=16384, group=6, D=256
tile (p_col=0,    v_kv=0,    out_col=0):   mean_abs=0, max_abs=0, cosine=1.0
tile (p_col=128,  v_kv=128,  out_col=128): mean_abs=0, max_abs=0, cosine=1.0000001
tile (p_col=1024, v_kv=1024, out_col=0):   mean_abs=0, max_abs=0, cosine=1.0
tile (p_col=8192, v_kv=8192, out_col=128): mean_abs=0, max_abs=0, cosine=1.0
```

Benchmark:

```text
bench_manual_pv_m64n128k128:
  min_ms:  0.029440000653266907
  mean_ms: 0.03041066663960616
  max_ms:  0.03174399957060814
```

QK regression check after the PV addition:

```text
bench_manual_qk_m64n128:
  mean_abs=0, max_abs=0, cosine=1.0
  min_ms: 0.06230400130152702
```

Interpretation:

```text
Both D256 block-scaled MMA orientations are now validated without the CUTLASS
load collective:

  QK: A=Q[M,D], B=K[N,D]
  PV: A=P[M,KV_TILE], B=V^T[D,KV_TILE]

The next milestone is a compact fused tile that uses these contracts in one
kernel:

  manual QK tile
  row softmax + P quantization
  manual PV tile

That fused tile is still a correctness/structure milestone, not the final
throughput path. The throughput path comes after replacing direct global
fragment construction with a compact producer/cache that targets <=50 KiB smem
and 2 CTAs/SM.
```

First compact manual data-movement primitive:

```text
sm120_d256_manual_qk_smem_m64n128
```

Shape:

```text
M = 64 Q rows
N = 128 KV rows
K = 256 head dim
compact row-major shared-memory staging for Q, K, SFA, and SFB
same direct SM120 block-scaled MMA atom fragments as the direct QK primitive
no CUTLASS CollectiveBuilder storage
```

Correctness:

```text
q=8192, kv=16384, group=6, D=256
mean_abs=0, max_abs=0, cosine=1.0
```

Benchmark:

```text
direct global-fragment QK min_ms: 0.06230400130152702
compact-smem QK min_ms:         0.02147199958562851
speedup:                        2.90x
```

Interpretation:

```text
The direct-fragment primitive was a contract test, not a data movement layer.
Compact manual row-major smem staging immediately gives a ~2.9x QK-tile
speedup while preserving exactness. This validates the pivot away from
CollectiveBuilder/TMA scale-side storage for D256 and toward our own compact
producer/cache around the SM120 FP4 atom.

Next: add the symmetric compact-smem PV primitive, then replace the fused
tile's direct Q/K/V fragment reads with these compact caches.
```

Second compact manual data-movement primitive:

```text
sm120_d256_manual_pv_smem_m64n128k128
```

Shape:

```text
M = 64 P rows
N = 128 output columns
K = 128 KV tile columns
compact row-major shared-memory staging for P, V^T, SFA, and SFB
same direct SM120 block-scaled MMA atom fragments as the direct PV primitive
no CUTLASS CollectiveBuilder storage
```

Correctness:

```text
q=8192, kv=16384, group=6, D=256
mean_abs=0, max_abs=0, cosine=0.9999999
```

Benchmark:

```text
direct global-fragment PV min_ms: 0.029440000653266907
compact-smem PV min_ms:         0.011008000001311302
speedup:                        2.67x
```

Interpretation:

```text
Both QK and PV now have exact compact manual shared-memory staging around the
SM120 NVFP4 atom. The immediate fused-tile integration target is:

  compact Q/K staging
  QK atom
  row softmax + P quantization in shared memory
  compact V staging
  PV atom

This will still be single-tile and single-CTA, but it removes the direct global
fragment construction path from the fused prototype.
```

First compact fused tile:

```text
sm120_d256_manual_fused_tile_smem_m64n128
```

Shape:

```text
M = 64 Q/P rows
N = 128 KV/output columns
D = 256 for QK
K = 128 for PV
dynamic smem = 74240 bytes
Q, K, V, P, scales, and FP32 logits all staged in compact row-major smem
```

Correctness:

```text
vs Python quantized-P reference:
  mean_abs=0.001456401776522398
  max_abs=0.011022660881280899
  cosine=0.9967120289802551

vs exact-P reference:
  mean_abs=0.0021152817644178867
  max_abs=0.009820207953453064
  cosine=0.9949629902839661
```

Benchmark:

```text
direct fused tile min_ms:       0.05648000165820122
compact-smem fused tile min_ms: 0.04265600070357323
speedup:                       1.32x
```

Interpretation:

```text
Replacing direct Q/K/V fragment reads with compact manual smem staging improves
the fused tile, but the first fused-smem version is still too large for 2 CTAs/SM
because it keeps K and V resident simultaneously and stores logits as FP32.

Next structural step:

  alias K and V storage because K is dead after QK
  store logits as BF16 instead of FP32

Expected smem budget:

  q_tile      8192
  kv_tile    16384  (K first, then V)
  q_scales   1024
  kv_scales  2048   (K scales first, then V scales)
  p_tile     4096
  p_scales   512
  logits     16384  (BF16)
  total      48640 bytes

This crosses the <=50 KiB target needed for 2 CTAs/SM.
```

Aliased compact fused tile:

```text
sm120_d256_manual_fused_tile_compact_m64n128
```

Shape:

```text
M = 64 Q/P rows
N = 128 KV/output columns
D = 256 for QK
K = 128 for PV
dynamic smem = 48640 bytes
K and V share the same compact tile storage
K scales and V scales share the same sidecar storage
logits are stored as BF16
__launch_bounds__(256, 2)
```

Correctness:

```text
vs Python quantized-P reference:
  mean_abs=0.0014686554204672575
  max_abs=0.009155270643532276
  cosine=0.9966394901275635

vs BF16-logit exact-P reference:
  mean_abs=0.002107701962813735
  max_abs=0.009784484282135963
  cosine=0.9949572682380676
```

Benchmark:

```text
compact-smem fused tile min_ms: 0.04265600070357323
aliased compact tile min_ms:    0.04451199993491173
single-tile delta:              -4.3%
```

Resource usage:

```text
manual_fused_tile_smem:
  regs/thread: 210
  static shared: 1024
  dynamic shared at launch: 74240

manual_fused_tile_compact:
  regs/thread: 128
  stack: 264
  static shared: 1024
  dynamic shared at launch: 48640
```

Interpretation:

```text
The 48.6 KiB compact variant is slightly slower for a single isolated tile
because it aliases K/V storage and reloads V after softmax, but it is the first
variant with both smem and register usage compatible with 2 CTAs/SM. The
previous 74 KiB variant is a useful correctness/per-tile reference but cannot
unlock the occupancy lever.

The next production integration step is not another isolated tile tweak. It is
to move this compact data movement into the D256 split-KV online-softmax grid
kernel so the full-grid benchmark can test whether the 2-CTA/SM resource shape
beats the current span2 CollectiveBuilder path.
```

Manual compact full-grid bridge:

```text
sm120_d256_manual_compact_splitkv_full_grid
```

Design:

```text
one CTA per (q64 tile, kv128 tile)
the CTA computes both D256 output groups, so QK/softmax/P are reused across
the two output halves
each kv128 tile is emitted as one split partial
the existing split-KV combine kernel merges the split stats/output
dynamic smem = 48640 bytes
```

Smoke cell:

```text
q=512, kv=8192, group=6, D=256
splits=64
partial_shape=[64, 3072, 256]
first tile vs exact:
  finite=true
  mean_abs=0.0022063846699893475
  max_abs=0.009833753108978271
  cosine=0.9919490218162537
min_ms=0.733951985836029
```

Comparison on the same smoke cell:

```text
current CollectiveBuilder span2 path:
  split_kv_len=1024
  splits=8
  smem=96256 bytes
  min_ms=0.3556160032749176

manual compact one-tile split path:
  split_kv_len=128
  splits=64
  smem=48640 bytes
  min_ms=0.733951985836029
```

Interpretation:

```text
The 48.6 KiB path gets the intended 2-CTA/SM resource shape, but one split per
128-token tile creates too much partial-output/combine traffic. It is a valid
manual data-movement bridge but not competitive yet.
```

Manual compact online split-KV bridge:

```text
sm120_d256_manual_compact_online_splitkv_full_grid
```

Design:

```text
one CTA per (q64 tile, output128 group, split)
split_kv_tiles is configurable
shared BF16 O accumulator stores one output group across the local split loop
dynamic smem = 65792 bytes
```

Smoke cell:

```text
q=512, kv=8192, group=6, D=256
split_kv_len=1024
splits=8
partial_shape=[8, 3072, 256]
first tile vs exact:
  finite=true
  mean_abs=0.0022248162422329187
  max_abs=0.009922564029693604
  cosine=0.9918416142463684
min_ms=1.5099200010299683
```

Interpretation:

```text
The online bridge reduces split count by 8x but is slower because it loses the
2-CTA/SM target and keeps a BF16 O accumulator in shared memory while still
recomputing QK/softmax per output group. This is the wrong tradeoff as written.

Do not continue optimizing this BF16-O shared-memory online bridge unless a
profile later shows combine traffic is dominant enough to justify revisiting it.
The next useful manual-data-movement lever is to keep the 48.6 KiB one-tile
resource shape and reduce its per-tile cost/partial overhead without adding
shared O state.
```

## D256 Manual Compact Split-KV: cp.async Staging

Applied 16-byte `cp.async` staging to the active compact split-KV bridge for:

```text
Q packed tile
K packed tile
Q scale tile
K scale tile
V packed tile
```

V scale staging remains scalar because the compact PV sidecar is 8 bytes per row
(`N=128 -> 8 scale bytes`). A 16-byte `cp.async` copy there overwrites adjacent
rows in the compact sidecar layout.

Smoke cell:

```text
q=512, kv=8192, group=6, D=256
splits=64
partial_shape=[64, 3072, 256]
storage_bytes=48640
first tile vs exact:
  finite=true
  mean_abs=0.0022063846699893475
  max_abs=0.009833753108978271
  cosine=0.9919490218162537
min_ms=0.6721919775009155
```

Previous compact split-KV time on the same cell:

```text
sync staging: 0.733951985836029 ms
cp.async staging: 0.6721919775009155 ms
win: ~8.4%
```

Interpretation:

```text
The manual compact data-movement layer is working and async staging helps, but
this bridge is still slower than the current span2 online path on the same cell
(~0.356 ms). The gap is now dominated by split granularity/partial traffic: one
partial per 128-token tile creates 64 splits at kv=8192. The next structural
change should preserve the compact 48.6 KiB resource shape while reducing split
count or carrying online state across multiple 128-token tiles without adding a
large shared BF16 O accumulator.
```

## D256 Manual Register-O Split-KV Bridge

Added a manual compact split-KV bridge with register-resident O across local KV
tiles:

```text
sm120_d256_manual_compact_register_o_splitkv_full_grid
```

Initial design:

```text
one CTA per (q64 tile, split)
compute both D256 output halves in the CTA
keep two PV accumulator sets in registers across split_kv_tiles
emit one partial per split instead of one partial per 128-token KV tile
```

Smoke cell:

```text
q=512, kv=8192, group=6, D=256, split_kv_len=1024
```

Results:

```text
q64 FP32 O, launch_bounds(256,1): 0.8192639946937561 ms, regs=255
q32 FP32 O, launch_bounds(256,1): 0.84825599193573 ms,  regs=255
q32 BF16 O, launch_bounds(256,1): 0.8256319761276245 ms, regs=255
q32 BF16 O, launch_bounds(256,2): 0.7269120216369629 ms, regs=128, stack=64
```

Correctness for the best variant:

```text
finite=true
mean_abs=0.0022248162422329187
max_abs=0.009922564029693604
cosine=0.9918416142463684
```

Comparison on the same cell:

```text
manual compact one-tile split path, cp.async: 0.6721919775009155 ms
current span2 CollectiveBuilder path:        0.35785600543022156 ms
```

Interpretation:

```text
Register-resident O removes the 64-way partial split issue, but the naive manual
mainloop still has too much per-CTA work. Forcing 2 CTAs/SM helps materially
(0.826 -> 0.727 ms), proving occupancy matters, but the path is still slower
than the simpler one-tile bridge and much slower than span2.

The register-O implementation is now resource-shaped correctly enough to profile
(regs=128, smem=49.4 KiB), but it is not yet a winning implementation. The next
question is whether it behaves better at the real large-q focused cells, where
more CTA parallelism and lower split count may matter more than on q=512.
```

## D256 Real-Cell Check: Manual Register-O vs Span2

Focused cell:

```text
q=8192, kv=16384, group=6, D=256, split_kv_len=1024
```

Results:

```text
manual compact register-O split-KV: 18.730911254882812 ms
current span2 CuTe/CUTLASS path:     6.6702399253845215 ms
```

Correctness:

```text
manual register-O cosine: 0.9904744029045105
span2 cosine:             0.9906792044639587
```

Conclusion:

```text
The manual direct-fragment path is not the right base for D256. It has the
compact smem/resource shape, but it cannot match the existing CuTe/CUTLASS atom
mainloop. The next D256 structural path is to keep the span2 CuTe atom mainloop
and replace only the 96 KiB collective data-movement/storage layer with a compact
manual load layout. In other words: do not keep optimizing the direct manual
fragment bridge; port the compact data movement into the winning span2 mainloop.
```

## D256 Span2 Split-KV Length Sweep: Focus Cell

Focused cell:

```text
q=8192, kv=16384, group=6, D=256
```

Span2 sweep:

```text
split_kv_len  splits  min_ms
1024          16      6.673791885375977
2048          8       5.821407794952393
4096          4       5.512832164764404
5120          4       5.32912015914917
6144          3       5.382239818572998
6656          3       5.270304203033447
7168          3       5.177792072296143
7680          3       5.457376003265381
8192          2       5.684703826904297
8704          2       5.62713623046875
9728          2       5.515007972717285
10240         2       5.471776008605957
11264         2       5.355648040771484
12288         2       5.242559909820557
16384         1       6.385087966918945
```

Interpretation:

```text
Split policy is a material dispatch lever. The best tested value is 7168 tokens
for this cell, improving span2 from 6.67 ms to 5.18 ms. That nearly matches the
nvfp4_fa2 baseline from the focus table (~5.02 ms) but does not beat it.

This does not replace the storage rewrite. It means future focused comparisons
must use a tuned split length; otherwise the kernel looks worse than it is. The
remaining gap on short/mid context is still structural: one CTA/SM and memory
wait from the 96 KiB span2 storage footprint.
```

## D256 M64 CollectiveBuilder Feasibility Check

Added metadata-only instantiation for:

```text
Sm120Fp4Tile64x128x256
```

Compile result: passed.

Metadata:

```text
mainloop_tensor_storage_bytes:   80896
mainloop_shared_storage_bytes:   81920
gemm_kernel_shared_storage:      91136
score_scratch_bytes:             32768
p_packed_bytes:                  8192
p_scales_bytes:                  1024
row_state_bytes:                 1280
scaffold_storage_min_bytes:      91392
sm120 margin:                    9984
```

Interpretation:

```text
The M64 CUTLASS collective is feasible but not the storage solution. It still
uses ~91 KiB shared memory, so it cannot unlock 2 CTAs/SM. The native D256 path
cannot be just another CollectiveBuilder tile. The next implementation has to
use CuTe atom/copy primitives with a custom compact storage layout, keeping the
span2 mainloop structure but not the collective TensorStorage allocation.
```

## Rejected: SM120 Block-Scaled StageCount<1>

Tried metadata-only instantiation of fixed `StageCount<1>` variants for:

```text
128x128x256_stage1
64x128x256_stage1
```

Compile result: rejected by CUTLASS.

Failure:

```text
sm120_blockscaled_mma_tma.hpp: static assertion failed:
"Specialization requires Stages set to value 2 or more."
```

Interpretation:

```text
The SM120 block-scaled TMA collective cannot be shrunk to a one-stage
CollectiveBuilder path. Stage-count reduction is not a viable shortcut to the
<=50 KiB 2-CTA/SM target. The custom D256 path must either keep the two-stage
collective storage or bypass the TMA collective with a custom CuTe atom/copy
loader.
```

## Harness Default Update

Updated `benchmarks/bench_sm120_d256_hillclimb.py` so the D256 focused harness
defaults to:

```text
--fused-output-group-span 2
```

Reason:

```text
D=256 has exactly two 128-column output groups. Span2 is the D256-natural fused
path and measured 32-35% faster than span1 across the focused group=6 surface.
Leaving the harness default at span1 makes future reports compare against a
known-stale implementation.
```

The split length default remains `6656` because the existing NCU/focus reports
used that value. Per-cell tuned split lengths should still be passed explicitly
when running short-context focus cells, e.g. `7168` for q=8192/kv=16384/group=6.

## D256 Span2 Split Sweep: Worst Focus Cell

Focused worst cell:

```text
q=32768, kv=32768, group=6, D=256
```

Span2 sweep, repeat=1:

```text
split_kv_len  splits  min_ms
4096          8       41.82281494140625
6144          6       41.322975158691406
6656          5       40.494720458984375
7168          5       40.394142150878906
8192          4       40.06790542602539
12288         3       39.339263916015625
16384         2       39.26464080810547
```

Interpretation:

```text
Unlike the q=8192/kv=16384 short-context cell, split length is not a large lever
on the worst high-q cell. Moving from 6656 to 16384 improves only ~3.0% and the
kernel remains far behind nvfp4_fa2 (~23.6 ms from the focus baseline).

This confirms the NCU conclusion: the high-q/mid-kv gap is not primarily split
combine overhead. It is the one-CTA/SM, 96 KiB-storage, memory-wait regime. The
next implementation work should target compact CuTe storage / 2-CTA resource
shape, not further split tuning.
```

## D256 Compact CuTe Storage Constraint

The D256 span2 path is currently blocked by shared-memory footprint, not by the
M dimension alone:

```text
128x128x256 span2 storage: ~96 KiB
64x128x256 CollectiveBuilder storage: ~91 KiB
2 CTA/SM target on SM120: <= ~49.5 KiB per CTA, practically <= 48 KiB
```

Interpretation:

```text
Halving M only saves the Q-side portion of each staged tile. K, V, and their
scale sidecars dominate the footprint, and CUTLASS's TMA collective multiplies
that footprint by at least two stages. The structural storage lever is therefore
stage count, not M alone.
```

The failed StageCount<1> experiment localizes the constraint to the TMA
collective layer:

```text
sm120_blockscaled_mma_tma.hpp: static assertion failed:
"Specialization requires Stages set to value 2 or more."
```

This does not prove the SM120 NVFP4 MMA atom requires two stages. It proves the
SM120 block-scaled TMA CollectiveBuilder path requires two stages. The next path
is therefore:

```text
Use the SM120 block-scaled CuTe MMA/copy atoms and atom-compatible smem layouts,
but bypass CollectiveBuilder's two-stage TMA storage.

Implementation target:
  - manual cp.async 16B gmem->smem staging
  - one-stage K/V data and scale storage
  - one-stage Q only long enough to copy Q into register fragments
  - reuse the span2 mainloop structure and CuTe gemm atom calls
  - <= 48 KiB dynamic smem so the kernel can reach 2 CTAs/SM
```

The direct manual-fragment compact path proves the footprint is feasible, but it
is the wrong performance base because it bypasses the CuTe atom/copy layout path.
The new path must keep CuTe atom fragment/copy conventions while owning the
storage and load pipeline manually.

## D256 Track 1 CUTLASS Small-K Metadata

Track 1 patched the local CUTLASS tree at `/home/josh/tdm/cutlass` so SM120
NVFP4 cooperative small tiles instantiate correctly:

```text
Sm120Fp4CollectiveTraits<cute::Shape<_64, _N, _K>> works with SFA M padding.
TileK=64 takes the SingleCtaKBlock consumer path in sm120_blockscaled_mma_tma.hpp.
M<128 cooperative preconditions are relaxed for SM120 block-scaled kernels.
Stages>=2 remains a real PipelineTmaAsync correctness constraint.
```

The D256 extension can now target that tree with:

```text
CUTLASS_ROOT=/home/josh/tdm/cutlass
```

Important correction to the previous compact-storage conclusion:

```text
The earlier 64x128x256 metadata result used full K=256 and StageCountAutoCarveout.
It was not evidence that CollectiveBuilder could not reach 2-CTA storage.
The correct probe is fixed StageCount<2> with smaller TileK.
```

Fixed StageCount<2> metadata on SM120:

```text
tile          mainloop  gemm_kernel  scaffold_min  2CTA scaffold margin
128x128x128   37888 B    47104 B       77312 B      -26624 B
128x128x256   74752 B    83968 B       94720 B      -44032 B
128x128x64    19456 B    28672 B       72704 B      -22016 B
256x128x128   56320 B    65536 B      154624 B     -103936 B
64x128x128    29696 B    38912 B       38656 B       12032 B
64x128x256    58368 B    67584 B       67840 B      -17152 B
64x128x64     15360 B    24576 B       36352 B       14336 B
64x256x128    48128 B    57344 B       71424 B      -20736 B
64x64x128     21504 B    30720 B       26368 B       24320 B
```

Interpretation:

```text
CUTLASS small-K is now the active D256 path. No manual compact CuTe storage
rewrite is needed for the next milestone.

2 CTA/SM candidates under the current scaffold model:
  - 64x128x64  (best useful M64/N128 candidate, 36.4 KiB scaffold)
  - 64x128x128 (fewer inner K passes, 38.7 KiB scaffold)
  - 64x64x128  (most headroom, smaller N tile)

128x128x64 has a small CUTLASS mainloop footprint but does not fit the current
fused scaffold because the 128x128 score scratch alone is 64 KiB. It only
becomes a candidate after the score/P staging is fully register/alias based.
```

Next implementation target:

```text
Add D256 policy variants backed by the Track 1 CUTLASS small-K collectives,
starting with 64x128x128 and 64x128x64. Benchmark focused cells:
q=32768, kv in {32768, 65536, 131072, 262144}, group=6.
Compare against the current span2 path and nvfp4_fa2.
```

## D256 Cooperative Tile Pivot Back To Span2 Internals

Timestamp: 2026-04-29 18:17 CDT

The small cooperative CUTLASS tile path is no longer the active optimization
path for D256. It remains useful as metadata and correctness coverage, but the
measured behavior does not justify continuing tile-shape probes before fixing
span2 internals:

```text
64x128x128:
  storage_bytes: 62464 B
  q=32768 kv=32768 group=6 split_kv_len=16384:
    before span2 P-fragment reuse: 38.6588 ms
    cosine: 0.9904504

64x128x64:
  storage_bytes: 73728 B in the current scaffold
  prior focused smoke: ~43.65 ms
  not a valid active candidate yet because D=256 needs four K=64 chunks, while
  the current Q path only carries two Q fragments.

64x64x128:
  metadata storage: 39936 B after output-width decoupling
  QK atom smoke: exact
  fused path: illegal instruction at UTMALDG.4D in the load warp
```

Interpretation:

```text
Even if 64x64x128 reaches 2 CTAs/SM, the cooperative path still runs 4 MMA
warps per CTA, so 2 CTAs/SM gives roughly the same 8 MMA warps/SM as span2's
single CTA. The cooperative CTAs duplicate role state, pipelines, and barriers,
while span2 keeps one coordinated QK -> softmax -> PV pipeline.

The remaining D256 gap is therefore inside span2: role-warp utilization,
role-handoff overhead, softmax/stat staging, and QK/PV handoff structure.
```

Span2-internal P-fragment reuse landed:

```text
Change:
  For kOutputGroupSpan=2 non-final PV tiles, copy the P fragment from smem to
  registers once, then reuse it for both V output groups.

Focused cell:
  q=32768 kv=32768 group=6 split_kv_len=16384 tile=64x128x128

Result:
  before: 38.6588 ms
  after:  36.0104 ms
  delta:  -6.8%
  cosine: 0.9904504
```

This is the first useful result after returning to span2 internals. Continue
there before adding more cooperative tile-shape work.

## D256 Span2 Role-Idling Cuts

Timestamp: 2026-04-29 18:35 CDT

The post-P-fragment-reuse NCU profile for the focused cell:

```text
shape: q=32768 kv=32768 group=6 split_kv_len=16384 tile=64x128x128
wall: 36.05 ms

issue slots busy:          10.30%
SM busy:                   10.46%
tensor pipe:               ~10.6%
mem busy:                  28.54%
L2 hit:                    99.34%
local/shared spills:       0
active warps/scheduler:    3.00
eligible warps/scheduler:  0.12
top stall:                 sleeping, 18.0 cycles/issued instruction
```

Interpretation:

```text
The focused D256 cell is not spilling and is not bandwidth-bound. The dominant
problem is explicit role idling / handoff waiting. The next useful changes are
span2 role-schedule cuts, not load-path or cooperative-tile work.
```

Measured role-idling edits:

```text
Focused cell: q=32768 kv=32768 group=6 split_kv_len=16384 tile=64x128x128

P-fragment reuse baseline:
  36.0104 ms

Direct MMA epilogue:
  small q512/kv8192: 0.6191 -> 0.6157 ms
  focused cell:      36.0104 -> 36.4759 ms
  decision: disabled by default; rejected for the target regime.

Single softmax warp, still consuming both row-group handoffs:
  small q512/kv8192: 0.6157 -> 0.5851 ms
  focused cell:      36.0104 -> 35.6933 ms

Single softmax warp + single row-group handoff:
  small q512/kv8192: 0.5851 -> 0.5857 ms
  focused cell:      35.6933 -> 35.4112 ms
```

The single-softmax direction is correct for D256, but the win is modest. The
remaining gap is likely at the QK -> softmax -> P/PV boundary itself rather than
in the number of pipeline objects.

## D256 MMA-Owned Softmax Breakthrough

Timestamp: 2026-04-29 18:40 CDT

Structural change:

```text
SM120_D256_MMA_OWNS_SOFTMAX=1

For D256, remove the separate softmax role from the active path. The MMA role
now owns:
  - per-row running m_i/l_i in registers
  - old_scale production
  - P quantization/staging to the PV smem layout

One MMA thread owns one softmax row for the 64-row tile. The score pipeline
handoff to Softmax0/Softmax1 is skipped entirely; the QK -> P/PV boundary is now
an in-role barrier instead of a cross-role pipeline.
```

Measured result:

```text
Focused cell: q=32768 kv=32768 group=6 split_kv_len=16384 tile=64x128x128

single-softmax + single-handoff baseline: 35.4112 ms
MMA-owned softmax:                       21.7293 ms
delta:                                   -38.6%
cosine:                                  0.9904504

small q=512 kv=8192 group=6:
single-softmax + single-handoff baseline: 0.5857 ms
MMA-owned softmax:                        0.3728 ms
delta:                                    -36.3%
```

This confirms the prior NCU diagnosis. The dominant D256 cost was the
cross-role QK -> softmax -> PV handoff, not the data movement layer.

The focused cell now beats the known nvfp4_fa2 baseline for the same cell
(~23.58 ms), but it is not yet at the original 2x aspirational gap. Continue
profiling and optimizing from this new 21.7 ms regime.

## D256 Post-MMA-Owned Retune

Timestamp: 2026-04-29 18:46 CDT

Post-MMA-owned-softmax profile for the focused cell:

```text
shape: q=32768 kv=32768 group=6 split_kv_len=16384 tile=64x128x128
wall: 21.78 ms

issue slots busy:          17.68%
SM busy:                   17.68%
mem busy:                  46.08%
L2 hit:                    99.27%
local/shared spills:       0
active warps/scheduler:    2.50
eligible warps/scheduler:  0.21
top stall:                 barrier, 4.5 cycles/issued instruction
```

The structural rewrite did what it was supposed to do: sleeping/role-idle
dominance is gone, issue utilization nearly doubled, and the next bottleneck is
CTA barrier waiting inside the remaining in-role QK/P/PV sequence.

Focused split retune after the rewrite:

```text
q=32768 kv=32768 group=6 tile=64x128x128

split_kv_len  splits  min_ms
4096          8       23.6276
8192          4       22.3550
16384         2       21.7060
32768         1       21.5716
```

The optimal split for this cell moved to no split. The split/combine overhead is
now visible because the per-CTA mainloop is much faster.

Additional direct-epilogue retest in the new regime:

```text
small q=512 kv=8192:       0.3749 -> 0.3716 ms
focused q=32768 kv=32768:  21.5716 -> 21.8377 ms
decision: keep direct epilogue disabled by default; focused cell remains the
          gate.
```

## D256 Remaining Barrier Cuts

Timestamp: 2026-04-29 18:51 CDT

After MMA-owned softmax, the profile moved to barrier stalls. Two remaining
barrier cuts were tested:

```text
Focused cell: q=32768 kv=32768 group=6 tile=64x128x128
Current best split: split_kv_len=32768 (no split)

Delayed QK completion barrier:
  Move the QK-logit readiness barrier from immediately after QK to immediately
  before in-role softmax. This lets PV(tile-1) run before the current tile's
  softmax readiness barrier.

  result: 21.5716 -> 21.7077 ms in one run with direct epilogue disabled at
          split=32768, but 21.7293 -> 21.7077 ms at split=16384. Net effect is
          tiny/noisy; kept for now because it is structurally correct and did
          not affect correctness.

Remove QK K-copy-side barriers in MMA-owned mode:
  The K pipeline stages are not released until after P staging, so the
  per-K-chunk copy barrier before advancing the local read state is redundant
  in the MMA-owned path.

  small q=512 kv=8192:      0.3749 -> 0.3635 ms
  focused q=32768 kv=32768: 21.5716 -> 21.0206 ms

Move V pipeline release after PV GEMM:
  Copy V to registers, run PV GEMM, then release the V pipeline stage. This
  removes the explicit pre-GEMM V-copy barrier and uses the pipeline release as
  the stage lifetime boundary.

  small q=512 kv=8192:      0.3635 -> 0.3657 ms
  focused q=32768 kv=32768: 21.0206 -> 20.6937 ms
```

Current focused best:

```text
q=32768 kv=32768 group=6 split_kv_len=32768 tile=64x128x128
min_ms: 20.6937
cosine: 0.9904627
```

## D256 Four-Thread Row Softmax

Timestamp: 2026-04-29 19:41 CDT

Structural change:

```text
In the MMA-owned softmax path, switch from one MMA thread per score row to four
MMA threads per score row.

Before:
  64 of 256 MMA threads produced P and row stats.
  192 MMA threads waited at the P-ready barrier.

After:
  all 256 MMA threads participate.
  each row uses 4 threads, each thread owns 32 columns / 2 scale groups.
  row max and row sum use 4-lane shfl_xor reductions.
```

Measured result:

```text
small q=512 kv=8192 group=6:
  before: 0.3639 ms
  after:  0.2620 ms

focused q=32768 kv=32768 group=6 split_kv_len=32768:
  before: 20.6937 ms
  after:  14.4057 ms
  delta:  -30.4%
  cosine: 0.9904513
```

This confirms that the remaining barrier wait after MMA-owned softmax was
mostly warp imbalance before the P-ready barrier. All-MMA-thread P production is
now the active D256 structure.

Post-change NCU profile on the focused cell:

```text
q=32768 kv=32768 group=6 split_kv_len=32768 tile=64x128x128
profile wall:                 14.720 ms
issue slots busy:             29.07%
tensor pipe:                  26.54%
mem busy:                     53.46%
max bandwidth:                28.67%
L2 hit:                       99.56%
local/shared spills:          0
active warps/scheduler:       2.50
eligible warps/scheduler:     0.42
top stalls per instruction:
  sleeping:                   1.90
  wait:                       1.35
  MIO throttle:               0.95
  short scoreboard:           0.90
  math pipe throttle:         0.74
  barrier:                    0.28
```

The profile is no longer the old barrier-dominated shape. Issue utilization is
near the 30% target and eligible warps/scheduler is above 0.4. The next obvious
question is whether a second CTA/SM can increase scheduler supply.

## D256 Compact Storage And Occupancy Probe

Timestamp: 2026-04-29 20:28 CDT

Storage layout inspection showed the D256 QKV storage had already aliased all
large P/logits buffers, but two aliased empty P-buffer placeholders still carried
`alignas(1024)` and wasted padding:

```text
before:
  storage bytes: 52224
  p_smem_A0 offset: 49152  (empty, padded to 1024)
  p_smem_A1 offset: 50176  (empty, padded to 1024)

after:
  storage bytes: 50176
  p_smem_A0 offset: 48161
  p_smem_A1 offset: 48177
```

Correctness held:

```text
q=512   kv=8192  group=6: cosine 0.9911097, min_ms 0.2599
q=32768 kv=32768 group=6: cosine 0.9904529, min_ms 14.6326
```

But NCU after compaction still reports one CTA/SM:

```text
active warps/scheduler:       2.50
eligible warps/scheduler:     0.42
issue slots busy:             29.07%
tensor pipe:                  26.54%
```

So shared memory is no longer the occupancy limiter. Resource usage for the
active `kOutputGroupSpan=2` kernel:

```text
launch_bounds min blocks = 1:
  registers/thread:           166
  stack/thread:               0
  focused min_ms:             14.63 ms

launch_bounds min blocks = 2:
  registers/thread:           96
  stack/thread:               344 bytes
  focused min_ms:             23.66 ms
```

Conclusion: forcing occupancy with launch bounds spills heavily and regresses.
The path to 2 CTA/SM requires structural live-range reduction, not a register
cap. Keep min-blocks default at 1.

Additional source-counter profile after compaction:

```text
L1 shared wavefronts:             3.149B
L1 ideal shared wavefronts:       1.653B
excessive shared wavefronts:      1.496B (48%)

largest source lines:
  QK logits BF16 store:           704.6M excessive wavefronts
  P FP4 byte stores:              302.0M
  QK logits BF16 reload, pass 1:  302.0M
  QK logits BF16 reload, pass 2:  151.0M
```

This points back to the BF16 logits round-trip as the next structural target.
The current kernel still materializes QK logits to shared memory, then reloads
them for softmax/P quantization. That was acceptable as a bridge, but it is now
the main on-chip traffic source.

Rejected probe: row-dependent XOR layout for the logits scratch.

```text
change:
  store/read logits through col ^ ((row & 0x0f) << 1)

small q=512 kv=8192:
  0.2598 -> 0.2647 ms

focused q=32768 kv=32768:
  14.65 -> 15.71 ms
```

The swizzle preserved correctness but added address arithmetic / worse access
shape and regressed. Do not retry logits-layout swizzles as a substitute for
removing the logits round-trip.

Rejected probe: direct QK accumulator to P staging.

```text
change:
  skip BF16 logits smem materialization and compute softmax/P directly from
  QK accumulator fragments, using two block-wide barriers to combine the low
  and high column halves for each row.

small q=512 kv=8192 group=6:
  storage: 50176 -> 61440 bytes
  cosine:  0.991102
  min_ms:  0.260 -> 0.587

focused q=32768 kv=32768 group=6:
  cosine:  0.990465
  min_ms:  14.65 -> 35.39
```

Correctness held, but the path is structurally worse. It removes the BF16
logits round trip only by adding extra global row-half reductions, extra
barriers, duplicated exp work, and a second P buffer. This increases storage by
11 KiB and more than doubles wall time. The code path was removed after logging
the result.

Conclusion: removing logits smem traffic must preserve the four-thread row
softmax structure and avoid new block-wide reductions. Do not reintroduce a
direct-QK accumulator path unless it has a different ownership model for the
low/high column halves.

Kept change: coalesce P FP4 shared stores inside the four-thread row softmax.

```text
change:
  for each 16-value P scale group, pack the 8 FP4 bytes into two uint32 values
  and store them with two 32-bit shared stores when the CuTe P layout is
  contiguous/aligned; otherwise fall back to byte stores.

small q=512 kv=8192 group=6:
  cosine: 0.991114
  min_ms: 0.2582 -> 0.2572

focused q=32768 kv=32768 group=6:
  cosine: 0.990407
  min_ms: 14.65 -> 14.29
  mean_ms: 14.29 over repeat=3
```

This is a small but real win (~2.4% on the focused cell). It directly targets
the source-counter line where scalar P byte stores contributed ~302M excessive
shared wavefronts. The improvement is not large enough to change the overall
gap by itself, but it is aligned with the current bottleneck profile and keeps
the stable softmax ownership model.

Kept change: additive row skew for the BF16 logits scratch layout.

```text
reason:
  row-major logits storage maps the same column pairs from different rows to the
  same shared-memory banks because the row stride is 128 bf16 = 256 bytes. The
  source-counter profile after P-store coalescing still showed ~704M excessive
  shared wavefronts in the logits store path.

change:
  store/read logits at:
    row * 128 + ((col + (row & 15) * skew) & 127)
  with skew=4.

small q=512 kv=8192 group=6:
  skew=0: 0.2572 ms
  skew=2: 0.2492 ms
  skew=4: 0.2326 ms
  skew=8: 0.2568 ms

focused q=32768 kv=32768 group=6:
  skew=0: 14.29 ms
  skew=2: 14.52 ms
  skew=4: 13.64 min / 13.66 mean over repeat=3
```

This is a larger layout win (~4.4% over the P-store coalesced focused baseline,
~6.9% over the pre-P-store baseline). Unlike the rejected XOR layout, additive
skew preserves contiguous per-row softmax reads while adding row entropy to the
shared-memory bank index. D=256 default is now `SM120_D256_LOGITS_ROW_SKEW=4`;
the env knob remains for future shape-specific sweeps.

Rejected follow-up: unconditional P uint32 stores.

```text
change:
  remove the runtime contiguous/alignment check from the P-store coalescing path
  and always issue two uint32 shared stores per 16-value P group.

small q=512 kv=8192 group=6:
  checked coalesced store:       0.2324 ms
  unconditional uint32 store:    0.2378 ms
```

Correctness held, but performance regressed. Keep the checked coalesced path.

## D256 Group=6 q=32768 Four-Cell Validation After P-Store + Logits Skew

Report files:

```text
reports/d256_group6_q32768_validation_lskew4_20260429.jsonl
reports/d256_group6_q32768_validation_lskew4_20260429.csv
reports/d256_group6_q32768_validation_lskew4_20260429.summary.csv
reports/d256_group6_q32768_validation_lskew4_20260429.md
```

Configuration:

```text
D=256, group=6, q=32768
tile policy:                 64x128x128
output group span:           2
split_kv_len:                32768
SM120_D256_LOGITS_ROW_SKEW:  4
```

Results:

```text
kv       splits  fused ms   nvfp4_fa2 ms  fp8_fa2 ms  bf16_fa2 ms  speedup vs nvfp4  cosine
32768    1       13.6616    23.7664       21.2194     14.3665      1.740x            0.990448
65536    2       27.2928    71.7212       63.3527     42.5902      2.628x            0.991919
131072   4       54.6482    174.1830      156.9015    104.1255     3.187x            0.990701
262144   8       119.2026   378.3900      345.3158    233.2816     3.174x            0.991230
```

Interpretation:

```text
The long-KV cells now pass the 2x gate against NVFP4 FA2 and also beat FP8 FA2
and BF16 FA2 by a wide margin. The remaining failure is the shortest focused
cell, kv=32768, where fused is 1.74x faster than NVFP4 FA2 but misses the 2x
target by 1.78 ms.

The next optimization target should be the kv=32768 cell specifically. At longer
KV, split-KV gives enough CTA parallelism and the per-tile improvements scale.
At kv=32768, there is only one split and the kernel remains sensitive to per-CTA
latency, register pressure, and on-chip layout.
```

Post-skew NCU for the focused cell:

```text
reports/ncu_d256_g6_q32768_kv32768_lskew4_20260429.ncu-rep
reports/ncu_d256_g6_q32768_kv32768_lskew4_20260429.details.txt
reports/ncu_d256_g6_q32768_kv32768_lskew4_20260429.source.csv

duration under ncu:              14.16 ms
compute throughput:              40.83%
memory throughput:               44.45%
issue slots busy:                37.58%
active warps/scheduler:          2.50
eligible warps/scheduler:        0.51
excessive shared wavefronts:     496M (down from 1.219B after P-store coalesce,
                                  down from 1.496B before both changes)
local memory spilling requests:  31.5M
```

The row skew materially improved scheduler eligibility and shared wavefronts,
but introduced/identified local-memory traffic. The next candidate is to keep
the skewed logits layout while reducing register/local pressure in the softmax
packing loop.

Rejected follow-up: recompute P values during FP4 packing.

```text
change:
  avoid keeping p_vals[16] live across scale selection by recomputing the two
  probabilities used for each packed byte after the scale is known.

small q=512 kv=8192 group=6:
  stable skew4 path:          0.2324 ms
  recompute-pack path:        0.2644 ms
```

Correctness held, but the extra exp/logit reload work is larger than the
register/local-pressure relief. The knob and branch were removed from the
active source; keep the stable p_vals path.

## D256 Group=6 q=32768 Clean Four-Cell Validation

After removing the rejected recompute-pack knob, reran the focused validation
matrix with the cleaned default build.

Report files:

```text
reports/d256_group6_q32768_validation_clean_lskew4_20260429.jsonl
reports/d256_group6_q32768_validation_clean_lskew4_20260429.csv
reports/d256_group6_q32768_validation_clean_lskew4_20260429.summary.csv
reports/d256_group6_q32768_validation_clean_lskew4_20260429.md
```

Configuration:

```text
D=256, group=6, q=32768
tile policy:                 64x128x128
output group span:           2
split_kv_len:                32768
SM120_D256_LOGITS_ROW_SKEW:  4
storage bytes:               50176
```

Results:

```text
kv       splits  fused ms   nvfp4_fa2 ms  fp8_fa2 ms  bf16_fa2 ms  speedup vs nvfp4  cosine
32768    1       13.6291    23.7886       21.1376     14.3606      1.745x            0.990440
65536    2       27.2997    71.5738       63.0309     42.4639      2.622x            0.991921
131072   4       54.7097    174.2117      156.2655    104.0950     3.184x            0.990626
262144   8       119.0001   378.2972      345.4784    232.6751     3.179x            0.991223
```

Conclusion:

```text
The long-KV validation cells pass the 2x gate and beat NVFP4, FP8, and BF16 FA2.
The only remaining miss in this focused D=256 group=6 grid is kv=32768:

  fused:         13.6291 ms
  2x target:     11.8943 ms
  gap:            1.7348 ms
```

Further layout levers are now expected to be incremental. The decision-relevant
state is that the kernel is shippable for the long-context cells in this focused
grid, while kv=32768 still needs either another structural latency reduction or
a dispatch policy that leaves that cell on the incumbent path.

## D256 q=32768 kv=32768 Split Resweep After Skew4

Report:

```text
reports/d256_g6_q32768_kv32768_split_resweep_lskew4_20260429.jsonl
```

Results:

```text
split_kv_len  splits  fused ms   cosine
8192          4       14.7523    0.990417
16384         2       14.0877    0.990369
32768         1       13.6672    0.990451
```

Conclusion:

```text
No split remains best for the shortest focused cell after the P-store and
logits-skew improvements. The remaining 32K miss is per-CTA latency / on-chip
execution overhead, not split-KV scheduling.

## D256 q=32768 kv=32768 Output Span Resweep After Skew4

Report:

```text
reports/d256_g6_q32768_kv32768_span_resweep_lskew4_20260429.jsonl
```

Results:

```text
output group span  fused ms   cosine     status
1                  23.7119    0.990437   ok
2                  13.6798    0.990400   ok
4                  -          -          invalid in current D=256 path
```

Conclusion:

```text
Span2 remains the correct D=256 output policy. Span1 loses the P/V reuse benefit
and is far slower; span4 is not a valid D=256 specialization in the current
implementation. Continue optimizing the span2 kernel.

## D256 Fine Logits-Row-Skew Sweep

Report:

```text
reports/d256_g6_q512_kv8192_logits_skew_fine_20260429.jsonl
```

Small-cell filter:

```text
q=512, kv=8192, group=6

skew  fused ms   cosine
1     0.2841     0.991105
3     0.2313     0.991129
4     0.2150     0.991125
5     0.2260     0.991111
6     0.2297     0.991110
7     0.2271     0.991119
```

Conclusion:

```text
No finer skew beats the current default. Keep SM120_D256_LOGITS_ROW_SKEW=4.
The remaining shared-memory wavefront excess is not removable by this simple
row-skew parameter.

Rejected follow-up: BF16 old-scale sidecar.

```text
change:
  store old_scale_stage as BF16 instead of FP32, converting back to FP32 when
  rescaling the PV accumulator. This reduced D=256 shared storage from 50176 to
  49152 bytes.

small q=512 kv=8192 group=6:
  0.2165 ms, cosine 0.991092

focused q=32768 kv=32768 group=6:
  repeat=1: 13.7058 ms, cosine 0.990434
  repeat=3: 13.6998 min / 13.7113 mean, cosine 0.990489
```

Correctness held, but the focused cell regressed versus the stable FP32 old-scale
path (~13.63-13.67 ms). Reverted. The old-scale sidecar is not the right next
latency lever.
```
```
```

## D256 Small-M CUTLASS Padded-SFA Contract Fix

Problem:

```text
After rebuilding against /home/josh/tdm/cutlass commit 24312f57, the D=256
span2 fused stage kernel launched with CUDA illegal-instruction errors. CUTLASS
standalone tests and the embedded CutlassFp4GemmRunner hook still passed, so the
failure was in the custom fused-attention integration, not in the CUTLASS GEMM
primitive.
```

Root cause:

```text
The small-M CUTLASS patch makes M=64 block-scaled kernels use padded SFA
storage. The cooperative collective handles this by:

  load side:     gSFA_mkl(..., broadcast_m(m_coord), ...)
  consumer side: sSFA_full(make_coord(_, m_coord % SFA_M_Ratio), _, _)

The span2 custom mainloop was still using the pre-small-M convention:

  load side:     gSFA_mkl(..., effective_q_tile, ...)
  consumer side: qk_sSFA(_, _, stage) / p_sSFA(_, _, stage)

That is valid for M=128 but violates the patched M=64 CUTLASS SFA contract.
```

Fix:

```text
benchmarks/sm120_nvfp4_cutlass_fused_attention_d256.cu

  - Q/SFA TMA source now uses the same broadcast_m layout as CUTLASS.
  - Q SFA smem consumers slice by effective_q_tile % SFA_M_Ratio before
    partition_fragment_SFA and SFA smem->register copies.
  - P/PV SFA smem consumers and producers use the same padded-M slice, so the
    softmax-generated P scales are staged where the PV MMA reads them.
```

Isolation:

```text
embedded CutlassFp4GemmRunner in the same extension:
  runner_ok True

span1 fused after fix, q=512 kv=8192 group=6:
  0.200096 ms, cosine 0.991516

span2 fused after fix, q=512 kv=8192 group=6:
  0.219488 ms, cosine 0.991518
```

Focused four-cell sanity after fix:

```text
report:
  reports/d256_group6_q32768_sanity_postopt_20260429.*

q=32768, group=6, split_kv_len=32768, output_group_span=2

kv       splits  fused ms   nvfp4 FA2 ms  fp8 FA2 ms   bf16 FA2 ms  speedup vs nvfp4  cosine
32768    1       13.2364    23.4410       20.8888      14.2584      1.771x            0.990854
65536    2       26.4188    70.3964       62.3308      42.0538      2.665x            0.992301
131072   4       52.6747    172.5565      154.6562     102.8168     3.276x            0.991109
262144   8       114.5021   372.6373      337.2659     228.4874     3.254x            0.991433
```

Conclusion:

```text
The rebuilt D=256 span2 kernel is operational again against the patched CUTLASS
small-M branch. The fix also improves the four focused cells by roughly 3-4%
versus the clean pre-rebuild checkpoint while preserving cosine around 0.991.
Proceed with the postopt sweep only after this SFA-contract fix is retained.
```

## D256 Expanded Postopt Sweep vs Prior Baseline

Report files:

```text
reports/d256_hillclimb_postopt_20260429.jsonl
reports/d256_hillclimb_postopt_20260429.csv
reports/d256_hillclimb_postopt_20260429.summary.csv
reports/d256_hillclimb_postopt_20260429.md
reports/d256_hillclimb_lift_vs_prior_baseline.md
```

Sweep matrix:

```text
D=256
q_len:  {128, 256, 512, 1024, 2048, 4096}
kv_len: {8192, 32768, 65536, 131072, 262144}
group:  {2, 4, 6, 8, 12, 16}
kernels per cell: sm120_fused, nvfp4_fa2, fp8_fa2, bf16_fa2
```

Operating point:

```text
tile=64x128x128
output_group_span=2
SM120_D256_LOGITS_ROW_SKEW=4
CUTLASS_ROOT=/home/josh/tdm/cutlass
split_kv_len=6656
warmup=1
repeat=3
```

Coverage and gate result:

```text
cells compared against prior baseline:       180 / 180
regressions vs prior fused baseline:         0
low-lift cells by 0-5% threshold:            0
postopt cells beating nvfp4_fa2:             163 / 180
postopt cells passing 2x gate vs nvfp4_fa2:  150 / 180
```

Per-group rollup:

```text
group  cells  median fused delta  median speedup vs nvfp4_fa2  beats nvfp4  passes 2x
2      30     -76.5%              2.66x                         23           19
4      30     -76.6%              2.96x                         26           23
6      30     -76.5%              3.09x                         28           26
8      30     -76.5%              3.19x                         28           26
12     30     -76.2%              3.32x                         29           28
16     30     -76.4%              3.25x                         29           28
```

Interpretation:

```text
The role-pipeline/row-skew/SFA-contract postopt state is a broad structural
lift over the prior D=256 fused baseline, not a narrow win on the focused
cells. Every cell improved by roughly 75-78% versus the prior fused baseline.

The remaining misses are concentrated in small-q / short-kv launch-dominated
cells. The larger-q and long-context cells that motivated this D=256 path are
the win zone and usually clear the 2x nvfp4_fa2 gate.

The full per-cell delta table is in:
  reports/d256_hillclimb_lift_vs_prior_baseline.md
```

## D256 Group 16 KV=1024 Q Sweep

Report files:

```text
reports/d256_group16_kv1024_q_sweep_20260430.jsonl
reports/d256_group16_kv1024_q_sweep_20260430.csv
reports/d256_group16_kv1024_q_sweep_20260430.summary.csv
reports/d256_group16_kv1024_q_sweep_20260430.md
```

Sweep:

```text
D=256
group=16
kv_len=1024
q_len={1024,4096,8192,16384,32768}
split_kv_len=6656
output_group_span=2
warmup=1
repeat=3
```

Result:

```text
q      fused ms  nvfp4 FA2 ms  fp8 FA2 ms  bf16 FA2 ms  speedup vs nvfp4  cosine    2x gate
1024   0.0868    0.1490        0.1322      0.0942       1.72x             0.9920    no
4096   0.2197    0.5241        0.4531      0.3129       2.39x             0.9917    yes
8192   0.3895    0.9785        0.8932      0.6034       2.51x             0.9882    yes
16384  0.7621    2.0936        1.8812      1.2108       2.75x             0.9901    yes
32768  1.4905    4.2177        3.8021      2.4235       2.83x             0.9909    yes
```

Interpretation:

```text
At group=16 and kv=1024, the D256 fused kernel beats nvfp4_fa2 for all tested
q values. The 2x gate starts at q=4096. q=1024 wins by 1.72x but remains below
the 2x target, which is consistent with launch/short-KV overhead dominating the
lowest-q cell.
```

## D512 Backport: MMA-Owned Softmax And P-Reuse

Ported the D256 structural wins back into the original D512 fused kernel:

```text
file: benchmarks/sm120_nvfp4_cutlass_fused_attention.cu

default compile knobs:
  SM120_D512_MMA_OWNS_SOFTMAX=1
  SM120_D512_LOGITS_ROW_SKEW=4
  SM120_D512_SOFTMAX_THREADS_PER_ROW=2
  SM120_D512_MIN_BLOCKS_PER_SM=1
```

Changes landed:

```text
- MMA-owned P generation / online softmax replaces the old Softmax0/Softmax1
  role pipeline by default.
- D512 uses 2 MMA threads per score row. A 1-thread row owner was tested and
  regressed.
- V pipeline release is delayed until after PV consumes the V fragment.
- P tile bytes are coalesced into 32-bit stores where the CuTe layout is
  contiguous.
- Logits row skew is supported and defaults to 4, but the D512 focused sweep
  showed skew is not the main lever.
- Nonfinal PV now copies the P fragment once and reuses it across output-group
  span2/span4 accumulators. This is the large D512 win.
```

Focused validation surface requested for D512:

```text
D=512
group=4
q_len=32768
kv_len={32768,65536,131072,262144}
split_kv_len=32768
output_group_span=4
warmup=1
repeat=3
CUDA_HOME=/usr/local/cuda-13.2
CUTLASS_ROOT=/home/josh/tdm/cutlass
```

Old role-softmax comparison on the anchor:

```text
q=32768 kv=32768 group=4
old role-softmax path: 88.6640 ms
MMA-owned softmax only: 77.5065 ms
lift: 12.6%
```

D512 row-skew / softmax-thread checks on the anchor:

```text
skew=0,  threads=2: 77.7464 ms
skew=2,  threads=2: 78.2632 ms
skew=4,  threads=2: 77.5065 ms
skew=8,  threads=2: 78.9255 ms
skew=16, threads=2: 79.0175 ms
skew=4,  threads=1: 79.7990 ms
```

Interpretation:

```text
2-thread MMA-owned row softmax is the right D512 default. Logits row skew is
flat around the focused anchor; keep skew=4 for consistency with the D256 path,
but the main D512 improvement is P-fragment reuse across output groups.
```

Focused D512 result after P-fragment reuse:

```text
q      kv      fused ms  nvfp4 FA2 ms  fp8 FA2 ms  speedup vs nvfp4  cosine
32768  32768   36.7411   69.2557       74.0916     1.88x             0.9928
32768  65536   73.7090   216.0493      211.8449    2.93x             0.9870
32768  131072  154.4947  501.8390      488.4138    3.25x             0.9909
32768  262144  321.1207  1078.7087     1059.6572   3.36x             0.9898
```

Smoke cell after P-fragment reuse:

```text
q=512 kv=8192 group=4 split_kv_len=8192
pre-P-reuse D512 port: 3.5696 ms
post-P-reuse D512 port: 1.2936 ms
cosine: 0.9900
```

Smoke cell baseline comparison:

```text
q=512 kv=8192 group=4
fused D512: 1.2936 ms
nvfp4 FA2:  0.6168 ms
fp8 FA2:    0.6420 ms
```

Interpretation:

```text
The D512 fused kernel is not the short-q dispatch choice. The win zone from
this backport is high-q / long-context global attention. Use the smoke cell for
correctness and build validation, not as a D512 optimization target.
```

BF16 FA2 note:

```text
The BF16 FA2 baseline for this custom D512/group4/q32768 shape hit the local
FlashInfer invalid-configuration path:
NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=1
NUM_WARPS_Q=4 NUM_WARPS_KV=1

Do not use BF16 FA2 as a decision baseline for this focused D512 surface until
that routing/config issue is fixed.
```

Conclusion:

```text
The D512 backport is a real win. It is not just a direct copy of the D256
softmax change: the decisive D512 lever is reusing the staged P fragment across
the span4 PV accumulators. The focused long-context cells now beat NVFP4 FA2 by
1.88x-3.36x and FP8 FA2 by 2.02x-3.30x.
```

## D512 Split-KV Policy Check After P-Reuse

Question:

```text
At q=32768/group=4, the kernel already has 1024 q-tiles, so split-KV may be
unnecessary combine traffic at long KV. Check whether split_kv_len=32768 should
remain the focused default.
```

Result:

```text
q=32768 group=4 D=512 output_group_span=4

kv      split_kv_len  fused ms  cosine
32768   8192          37.2623   0.9928
32768   16384         36.6353   0.9928
32768   32768         36.7411   0.9928

65536   16384         74.5914   0.9870
65536   32768         73.7090   0.9870
65536   65536         74.5500   0.9870

131072  16384         157.4026  0.9909
131072  32768         154.4947  0.9909
131072  65536         156.9540  0.9909
131072  131072        156.8894  0.9909

262144  16384         326.3318  0.9898
262144  32768         321.1207  0.9898
262144  65536         331.6680  0.9898
262144  262144        334.0772  0.9898
```

Decision:

```text
Keep split_kv_len=32768 as the D512 focused default. split_kv_len=16384 is only
noise-level faster on the shortest 32K cell and loses at longer KV. No-split is
not better once KV grows.
```

## D512 Output-Group Span And Post-P-Reuse NCU

Output span check on the weakest D512 focused cell:

```text
D=512 q=32768 kv=32768 group=4 split_kv_len=32768

span1: 62.3594 ms
span2: 43.1676 ms
span4: 37.1393 ms
```

Decision:

```text
Keep span4 for D512/group4. The P-fragment reuse change makes span4 the clear
winner despite the larger accumulator footprint.
```

Post-P-reuse NCU profile:

```text
shape: D=512 q=32768 kv=32768 group=4 split_kv_len=32768 span4
report: /tmp/ncu_d512_anchor_preuse.ncu-rep

duration under ncu:                 37.9 ms
memory throughput:                  73.2%
DRAM throughput:                    37.7%
L2 throughput:                      73.2%
L2 hit rate:                        99.8%
compute throughput:                 28.1%
tensor pipe active:                 17.0%
issue active:                       14.6%
active warps / scheduler:           2.51
eligible warps / scheduler:         0.20
local/shared spilling requests:     0

top warp stalls:
  long scoreboard:                  4.07 inst
  sleeping:                         3.28 inst
  wait:                             1.96 inst
  LG throttle:                      1.34 inst
  short scoreboard:                 1.23 inst
  barrier:                          0.62 inst
```

Interpretation:

```text
The D512 kernel is no longer primarily barrier-bound after P reuse. It is now
limited by memory/L2 pressure and low eligible warp count, with no register
spills. Further D512 wins should target memory traffic/layout and the remaining
QK logits/P staging handoff, not split-KV or output-span policy.
```

## D512 180-Cell Hillclimb Sweep

Run:

```text
date: 2026-04-30
report prefix: reports/d512_hillclimb_180cell_20260430
head_dim: 512
q: 128, 256, 512, 1024, 2048, 4096
kv: 8192, 32768, 65536, 131072, 262144
group: 2, 4, 6, 8, 12, 16
kernels: sm120_fused, nvfp4_fa2, fp8_fa2, bf16_fa2
fused split_kv_len: 32768
fused output_group_span: 4
CUDA_HOME: /usr/local/cuda-13.2
CUTLASS_ROOT: /home/josh/tdm/cutlass
```

Artifacts:

```text
reports/d512_hillclimb_180cell_20260430.jsonl
reports/d512_hillclimb_180cell_20260430.csv
reports/d512_hillclimb_180cell_20260430.summary.csv
reports/d512_hillclimb_180cell_20260430.md
reports/d512_hillclimb_180cell_20260430.run.log
```

Completion:

```text
kernel rows: 720 / 720
summary cells: 180 / 180
ok rows: 540
error rows: 180
```

All error rows are BF16 FA2 invalid-configuration failures at D512:

```text
Invalid configuration:
NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=1
NUM_WARPS_Q=4 NUM_WARPS_KV=1
```

Overall fused-vs-NVFP4-FA2 result:

```text
fused beats nvfp4_fa2: 119 / 180 cells
fused passes 2x gate:   98 / 180 cells
median speedup:         2.37x
min speedup:            0.068x
max speedup:            5.99x
```

Rollup by group:

```text
group  cells  median  min     max     beats  passes_2x
2      30     0.866   0.068   4.843   11     7
4      30     1.768   0.124   4.907   16     13
6      30     2.371   0.181   5.984   21     16
8      30     2.690   0.234   4.929   21     18
12     30     2.946   0.348   5.987   25     21
16     30     3.626   0.467   4.944   25     23
```

Rollup by q:

```text
q     cells  median  beats  passes_2x
128   30     0.449   6      2
256   30     0.871   11     6
512   30     1.757   19     13
1024  30     3.418   25     21
2048  30     3.872   28     27
4096  30     3.840   30     29
```

Gemma4-relevant D512 group=8 slice:

```text
q     kv      fused_ms  nvfp4_fa2_ms  fp8_fa2_ms  speedup  passes_2x
128   8192    1.285     0.340         0.344       0.26x    no
128   32768   4.872     1.138         1.167       0.23x    no
128   65536   4.887     2.203         2.274       0.45x    no
128   131072  4.907     4.336         4.454       0.88x    no
128   262144  4.918     8.687         8.799       1.77x    no
256   8192    1.293     0.624         0.640       0.48x    no
256   32768   4.876     2.267         2.359       0.47x    no
256   65536   4.891     4.420         4.613       0.90x    no
256   131072  4.912     8.803         9.173       1.79x    no
256   262144  5.262     17.645        18.266      3.35x    yes
512   8192    1.295     1.254         1.322       0.97x    no
512   32768   4.889     4.764         5.028       0.97x    no
512   65536   4.904     9.414         9.987       1.92x    no
512   131072  5.211     18.822        19.995      3.61x    yes
512   262144  11.415    37.763        39.814      3.31x    yes
1024  8192    1.312     2.822         3.006       2.15x    yes
1024  32768   4.898     11.439        12.259      2.34x    yes
1024  65536   5.282     22.778        24.623      4.31x    yes
1024  131072  11.464    45.954        49.529      4.01x    yes
1024  262144  18.984    91.989        98.854      4.85x    yes
2048  8192    1.417     5.307         5.672       3.74x    yes
2048  32768   5.315     22.569        24.268      4.25x    yes
2048  65536   11.358    45.577        48.821      4.01x    yes
2048  131072  18.959    91.434        99.068      4.82x    yes
2048  262144  38.015    187.388       199.010     4.93x    yes
4096  8192    3.030     7.302         7.627       2.41x    yes
4096  32768   11.491    34.133        35.718      2.97x    yes
4096  65536   18.762    70.307        73.093      3.75x    yes
4096  131072  37.877    149.667       147.728     3.95x    yes
4096  262144  78.581    308.563       305.534     3.93x    yes
```

Interpretation:

```text
The D512 fused kernel is not a small-q dispatch path. It loses broadly at
q=128/256 and is marginal around q=512 at shorter KV.

The D512 fused kernel is a high-q / long-context path. For the Gemma4 group=8
slice it passes the 2x gate for every q>=1024 cell and for the longer q=512
cells. The strongest production-relevant region is q>=1024, where speedups are
2.15x to 4.93x against NVFP4 FA2 and correctness cosines stay around 0.989-0.993.
```

## Cleanup: Remove Settled Hill-Climb Scaffolding

The D256 and D512 benchmark kernels no longer accept the settled hill-climb
macros from the environment. The production values are fixed in source:

```text
D256 tile:                 64x128x128
D256 direct MMA epilogue:  false
D256 single softmax warp:  true
D256 MMA-owned softmax:    true
D256 min blocks/SM:        1
D256 logits row skew:      4

D512 MMA-owned softmax:    true
D512 softmax threads/row:  2
D512 min blocks/SM:        1
D512 logits row skew:      4
```

The common benchmark harness now builds one stable extension name per head dim
instead of encoding D256/D512 hill-climb permutations in the torch extension
cache key. Removed harness-side compile knobs:

```text
SM120_D256_TILE_POLICY
SM120_D256_MIN_BLOCKS_PER_SM
SM120_D256_LOGITS_ROW_SKEW
SM120_D512_MMA_OWNS_SOFTMAX
SM120_D512_LOGITS_ROW_SKEW
SM120_D512_SOFTMAX_THREADS_PER_ROW
SM120_D512_MIN_BLOCKS_PER_SM
SM120_NVFP4_MAXRREGCOUNT
```

D256 rejected manual experiment paths were removed from the C++ extension and
CLI:

```text
manual QK/PV direct-fragment probes
manual fused-tile direct/smem/compact probes
manual compact split-KV / online split-KV / register-O split-KV variants
```

The active production entrypoints remain:

```text
sm120_nvfp4_qkv_online_register_q_splitkv_full_grid
sm120_nvfp4_qkv_online_register_q_splitkv_reuse2_full_grid
sm120_nvfp4_qkv_online_register_q_splitkv_reuse4_full_grid
```

Smoke validation after cleanup:

```text
python -m py_compile benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py \
  benchmarks/bench_sm120_d256_hillclimb.py

D256 role schedule smoke:
  total_warps=10, mma=8, load=1, epilogue=1, storage=50176 bytes

D512 role schedule smoke:
  total_warps=10, mma=8, load=1, epilogue=1, storage=96256 bytes

D256 active split-KV smoke:
  q=512 kv=8192 group=2 reuse2
  finite=true cosine=0.99149 min_ms=0.2044

D512 active split-KV smoke:
  q=512 kv=8192 group=8 reuse4
  finite=true cosine=0.98977 min_ms=1.3083
```

## D128 Specialization: Focus Set And Starting Point

Active focus cells for D128 specialization:

```text
q=32768 kv=32768  group=8   anchor
q=32768 kv=65536  group=8
q=32768 kv=131072 group=8
q=32768 kv=262144 group=8

q=32768 kv=131072 group=4
q=32768 kv=131072 group=12
q=32768 kv=131072 group=16

q=512   kv=8192   group=8   smoke
```

Current D128 translation unit status before native specialization:

```text
file: benchmarks/sm120_nvfp4_cutlass_fused_attention_d128.cu
role schedule: softmax0=1, softmax1=1, mma=8, load=1, epilogue=1
total warps: 12
storage: 96256 bytes
shared-memory margin: 5120 bytes
```

This confirms the D128 file is still carrying the old large shared-storage
scaffold. It is correct enough to benchmark, but not D128-native: it still uses
the full 96 KiB storage footprint and cannot reach 2 CTAs/SM.

Smoke before D128 specialization:

```text
q=512 kv=8192 group=8 split_kv_len=8192 output_group_span=1
finite=true
cosine=0.9916909
min_ms=0.979712
storage=96256 bytes
```

Next step is a streamed focus baseline against `nvfp4_fa2`, `fp8_fa2`, and
`bf16_fa2`, then D128-native structural work. Primary expected levers are:

```text
compact D128 shared storage
single output-group path only
remove the second softmax warp / ordering machinery if D128 softmax throughput allows
reduce tile/pipeline state enough to target 2 CTAs/SM
retune split-KV length after storage and role reductions land
```

## D128 Slice 1: Backport D256 Role Pattern

Ported the settled D256 role/softmax structure into the D128 specialization:

```text
M partition:              128
K tile:                   128
softmax ownership:        MMA-owned
softmax threads per row:  2
softmax warps:            0
role warps:               mma=8, load=1, epilogue=1
total warps:              10
storage:                  92160 bytes
shared-memory margin:     9216 bytes
```

This slice removes the separate softmax warp pair and the old two-stage
softmax handoff from the active D128 path. It keeps D128 as a single
output-group path (`output_group_span=1`), so there is no D512-style
column-group cycle or P-fragment reuse to optimize.

Correctness stayed stable. Focus-set lift versus the scaffold baseline:

```text
| group | q     | kv     | before fused | after fused | lift  | nvfp4_fa2 | after speedup | cosine   |
|------:|------:|-------:|-------------:|------------:|------:|----------:|--------------:|---------:|
| 4     | 32768 | 131072 | 99.316       | 37.139      | 2.67x | 42.495    | 1.14x         | 0.991741 |
| 8     | 512   | 8192   | 0.979        | 0.373       | 2.62x | 0.125     | 0.34x         | 0.991697 |
| 8     | 32768 | 32768  | 46.264       | 18.643      | 2.48x | 12.127    | 0.65x         | 0.990112 |
| 8     | 32768 | 65536  | 97.969       | 37.030      | 2.65x | 34.653    | 0.94x         | 0.991861 |
| 8     | 32768 | 131072 | 205.156      | 75.374      | 2.72x | 82.431    | 1.09x         | 0.990365 |
| 8     | 32768 | 262144 | 416.870      | 157.049     | 2.65x | 183.515   | 1.17x         | 0.990574 |
| 12    | 32768 | 131072 | 313.984      | 117.470     | 2.67x | 128.393   | 1.09x         | 0.993173 |
| 16    | 32768 | 131072 | 415.420      | 157.943     | 2.63x | 170.720   | 1.08x         | 0.987426 |
```

The port is a real structural win, but it is not the final D128-native kernel:
storage is still 92 KiB, so the path remains 1 CTA/SM, and the shorter focus
cells still trail NVFP4 FA2. Next lever is split-KV policy on the high-q focus
cells: with q=32768 there is already abundant Q-tile parallelism, so excessive
split count and combine traffic may be the next large avoidable overhead.

## D128 Slice 2: M64 Native Partition And Logits Alias Diagnosis

Read the D256 kernel end to end before continuing the D128 port. The important
D256 structural facts are:

```text
D256 production tile:              M64 x N128 x K128
softmax ownership:                 MMA-owned
softmax threads per row:           4
P storage:                         single buffer
QK head chunks:                    2
logits storage:                    aliases QK-B when it fits
P storage/scales:                  alias QK-A / QK-SFA after Q is register resident
```

Porting the same M64 partition to D128 exposed a correctness bug when logits
were allowed to alias QK-B:

```text
D128 M64 + QK-B logits alias:
  storage:                         50176 bytes
  q=512 kv=8192 group=8:           cosine=0.0, huge output error
```

Disabling only the QK-B/logits alias restored correctness:

```text
D128 M64 + explicit BF16 logits:
  storage:                         66560 bytes
  q=512 kv=8192 group=8:           cosine=0.991697, min_ms=0.140032
  q=32768 kv=32768 group=8:        cosine=0.990060, min_ms=11.807712
  q=32768 kv=131072 group=8:       cosine=0.990284, min_ms=47.535263
```

Diagnosis:

```text
M64 D128 BF16 logits size:          64 * 128 * 2 = 16384 bytes
M64 D128 QK-B storage total:        2 K stages * 8192 bytes = 16384 bytes
```

So the old alias uses the entire QK-B region, not one inactive stage. The load
role can prefetch next K into the alternate QK-B stage while current-tile
logits still occupy the full QK-B region. D256 avoided this specific failure
mode with a different K-chunk/lifetime structure (`qk_head_chunks=2`), but
D128 has `qk_head_chunks=1`, so the alias lifetime is invalid.

Current correct D128 state is therefore M64 with explicit logits. It is already
~2.6x faster than the M128 role-port slice and wins long focus cells versus
NVFP4 FA2, but the explicit 16 KiB logits buffer keeps storage at 66.5 KiB and
prevents the 2-CTA/SM target. The next structural lever is removing or shrinking
the logits handoff without reintroducing the QK-B lifetime collision.

K64 was tested as a possible storage escape hatch:

```text
D128 M64 x N128 x K64 + explicit BF16 logits:
  storage:                         63488 bytes
  q=512 kv=8192 group=8:           cosine=0.991697, min_ms=0.168320
  q=32768 kv=32768 group=8:        cosine=0.990060, min_ms=12.973856
```

Rejected for now. K64 is correct, but it still does not reach 2 CTAs/SM and it
regresses both the smoke and anchor cells versus K128 (`0.140 ms` and
`11.81 ms`). The extra K pass costs more than the modest storage reduction buys.

Also tested 2-thread row softmax on the correct M64/K128 path:

```text
D128 M64 x N128 x K128 + explicit logits + 2 softmax threads/row:
  q=512 kv=8192 group=8:           cosine=0.991697, min_ms=0.156832
  q=32768 kv=32768 group=8:        cosine=0.990060, min_ms=13.155488
```

Rejected. The 2-thread row split reduces participating softmax lanes but doubles
per-lane column work and regresses both smoke and anchor. Keep the D256-style
4-thread row split for D128.

Logits row skew was retuned on the explicit-logits M64/K128 path:

```text
skew=0 smoke q=512 kv=8192 group=8:  min_ms=0.163680
skew=8 smoke q=512 kv=8192 group=8:  min_ms=0.162272
skew=4 current reference:            min_ms=0.140032
```

Rejected skew 0 and 8. Keep `SM120_D128_LOGITS_ROW_SKEW=4`.

Direct MMA epilogue was retested for D128:

```text
D128 M64/K128 + explicit logits + direct epilogue:
  q=512 kv=8192 group=8:           cosine=0.991697, min_ms=0.150208
  q=32768 kv=32768 group=8:        cosine=0.990060, min_ms=12.169760
```

Rejected. The dedicated epilogue role remains faster (`0.140 ms` smoke,
`11.81 ms` anchor), matching the D256 conclusion.

Cleanup: removed the D128 direct-epilogue compile-time switch and deleted the
direct-store branch/helper. The dedicated epilogue role is now the only D128
output path. Smoke after cleanup:

```text
q=512 kv=8192 group=8: cosine=0.991697, min_ms=0.140064
```

Follow-up cleanup: removed the inactive Softmax0/Softmax1 PipelineAsync
storage, ordered softmax barrier, old softmax-role branch, and settled
MMA-owned-softmax compile-time switches from the D128 file. The active D128
role stack is now MMA + Load + Epilogue only. Smoke after cleanup:

```text
q=512 kv=8192 group=8: storage=66560 bytes, cosine=0.991697, min_ms=0.141216
q=32768 kv=32768 group=8: storage=66560 bytes, cosine=0.990060, min_ms=11.868736
```

This did not change the shared-memory limiter; the remaining structural lever
is still the explicit 16 KiB BF16 logits handoff.

Follow-up cleanup: made D128 span1-only. D128 has exactly one 128-wide output
group, so the reuse2/reuse4 wrappers, span2/span4 kernel instantiations, and
extra PV accumulator declarations were deleted from the D128 translation unit.

```text
resource usage: active stage kernel REG 168 -> 167
q=512 kv=8192 group=8: storage=66560 bytes, cosine=0.991697, min_ms=0.142208
q=32768 kv=32768 group=8: storage=66560 bytes, cosine=0.990060, min_ms=11.852992
```

This is cleanup and compile-surface reduction, not a material runtime win. It
does not change the conclusion: D128 high-q remains limited by the explicit
BF16 logits handoff and 1-CTA/SM shared-memory footprint.

C-fragment ownership check for a possible direct accumulator-to-P path:

```text
row 0 owners:
  tid 0:   cols 0,1,8,9,32,33,40,41,64,65,72,73,96,97,104,105
  tid 1:   cols 2,3,10,11,34,35,42,43,66,67,74,75,98,99,106,107
  tid 2:   cols 4,5,12,13,36,37,44,45,68,69,76,77,100,101,108,109
  tid 3:   cols 6,7,14,15,38,39,46,47,70,71,78,79,102,103,110,111
  tid 128: cols 16,17,24,25,48,49,56,57,80,81,88,89,112,113,120,121
  tid 129: cols 18,19,26,27,50,51,58,59,82,83,90,91,114,115,122,123
  tid 130: cols 20,21,28,29,52,53,60,61,84,85,92,93,116,117,124,125
  tid 131: cols 22,23,30,31,54,55,62,63,86,87,94,95,118,119,126,127
```

The direct path is structurally possible because each owner has paired columns,
but row reductions cross eight threads and two warpgroups, and scale-group max
crosses four owner threads. It needs a small row/group scratch or atomics, not
just a local register transform.

Important lifetime constraint: with the current single P buffer, the mainloop
must run PV(previous tile) before softmax/P(current tile) overwrites P. A direct
QK-accumulator-to-P path therefore requires either:

```text
1. reorder the loop to PV(previous) -> QK(current) -> direct softmax/P(current), or
2. add a second P buffer so QK(current) -> direct softmax/P(current) can happen
   before PV(previous).
```

Option 2 adds P storage and works against the 2-CTA target. Option 1 is the
next real structural rewrite if we continue chasing removal of the BF16 logits
handoff.

Follow-up cleanup: removed the rejected QK-B logits alias switches and branches.
D128 now always uses explicit BF16 logits storage, so the invalid full-QK-B and
per-stage alias paths cannot be accidentally re-enabled.

```text
q=512 kv=8192 group=8: storage=66560 bytes, cosine=0.991697, min_ms=0.141120
```

Attempted a safe QK-B/logits alias variant to recover the 2-CTA storage target:
duplicate each D128 K tile into both QK-B pipeline stages and release both
stages after MMA-owned softmax consumed logits. This was intended to prevent
next-K prefetch from colliding with logits while keeping storage at ~50 KiB.

Result: rejected. The variant hung at runtime on the smoke cell, indicating the
manual duplicate-stage protocol violated the CUTLASS TMA pipeline state
contract. Reverted to explicit logits. A correct 2-CTA path still needs either
register-resident logits/P staging or a proper pipeline-level rewrite, not a
local duplicate-stage hack.

NCU profile for the current correct M64/K128 explicit-logits anchor:

```text
cell: q=32768 kv=32768 group=8 split_kv_len=32768
report: reports/ncu_d128_m64_k128_anchor_20260430.ncu-rep
wall in profiled run:                 11.93 ms
shared memory per block:              66.56 KiB
occupancy limit:                      shared memory = 1 CTA/SM
issue slots busy:                     45.55%
tensor pipe active:                   24.0%
mem busy:                             49.95%
L1/TEX hit rate:                      99.43%
L2 hit rate:                          99.67%
local/shared spilling:                0
active warps / scheduler:             2.50
eligible warps / scheduler:           0.65
no eligible cycles:                   53.70%
top per-issue stalls:
  wait                                1.17
  sleeping                            0.87
  short scoreboard                    0.48
  math pipe throttle                  0.44
  not selected                        0.40
  MIO throttle                        0.36
  barrier                             0.19
  long scoreboard                     0.16
```

Interpretation: this is no longer the old barrier-dominated profile. The kernel
is primarily occupancy/eligibility limited by the 66.5 KiB shared-memory
footprint; memory hits are excellent, there are no spills, and tensor pipe is
reasonably active for a single-CTA/SM D128 path. The next large lever remains
storage reduction to 2 CTAs/SM, specifically removing the explicit 16 KiB BF16
logits handoff without corrupting K prefetch.

M32 was attempted as an explicit-logits path that could fit 2 CTAs/SM without
QK-B aliasing. It is not currently available through the CUTLASS cooperative
SM120 block-scaled mainloop:

```text
error: Cooperative kernel requires Tile Size to be greater than or equal to 128
along the M-dimension, except SM120 block-scaled kernels also support M=64.
```

Even after adapting the epilogue tile from 64x32 to 32x32, the cooperative
kernel rejects M32. Reverted to M64. If M32 is needed later, it requires a
CUTLASS-side extension analogous to the earlier M64 enablement, not a local
attention-kernel change.

After the CUTLASS M32 extension landed, M32 was retested:

```text
D128 M32 x N128 x K128 + explicit BF16 logits:
  storage:                         53248 bytes
  q=512 kv=8192 group=8:           cosine=0.991697, min_ms=0.106272
  q=32768 kv=32768 group=8:        cosine=0.990060, min_ms=15.270496

D128 M32 x N128 x K128 + per-stage QK-B logits alias:
  storage:                         45056 bytes
  q=512 kv=8192 group=8:           cosine=0.991697, min_ms=0.107392
  q=32768 kv=32768 group=8:        cosine=0.990061, min_ms=15.360256
```

The per-stage QK-B alias is correct for M32 because one BF16 logits tile is 8
KiB and fits inside one QK-B pipeline stage. This is different from M64, where
one logits tile is 16 KiB and spans both K stages.

NCU for M32 per-stage alias:

```text
report:                            reports/ncu_d128_m32_stage_alias_anchor_20260430.ncu-rep
shared memory per block:            45.06 KiB
registers per thread allocated:     168
occupancy limiter:                  registers = 1 block
active warps / scheduler:           2.50
eligible warps / scheduler:         0.43
issue slots busy:                   36.12%
tensor pipe active:                 under-utilized
```

So M32 solves the shared-memory footprint but does not unlock 2 CTAs/SM because
register allocation remains the limiter. A `-maxrregcount=96` probe preserved
correctness but did not improve wall time (`q=32768 kv=32768 group=8` remained
~15.36 ms). Rejected M32 for the high-q D128 path. It may still be useful later
as a small-q dispatch variant, since it beats M64 on the smoke cell, but the
current high-throughput focus path returns to M64/K128 explicit logits.

N64 was tested as a D128-native alternative to reduce QK/logits/P state while
keeping M64:

```text
D128 M64 x N64 x K128:
  q=512 kv=8192 group=8 smoke: illegal instruction at runtime
```

Rejected. The shape compiles after relaxing the local softmax assertion, but it
does not execute correctly with the current CUTLASS SM120 block-scaled
collective/epilogue binding. Reverted to N128.

## D128 Split-KV Policy Retune

Retuned split-KV after the M64/K128 explicit-logits path landed. For high-q
D128 there are already enough Q tiles to saturate the device, so split-KV mostly
adds combine traffic.

```text
q=32768 group=8
kv=65536:   split=65536   23.149 ms
kv=131072:  split=131072  46.270 ms
             split=65536  47.038 ms
             split=32768  47.752 ms
kv=262144:  split=262144 100.200 ms
             split=131072 101.010 ms
             split=65536 101.771 ms
             split=32768 102.756 ms

q=32768 kv=131072
group=4:   split=131072 23.470 ms, split=65536 23.549 ms, split=32768 23.959 ms
group=12:  split=131072 72.455 ms, split=65536 74.022 ms, split=32768 75.001 ms
group=16:  split=131072 101.019 ms, split=65536 102.766 ms, split=32768 103.849 ms
```

Decision: for high-q D128 focus cells, use no split (`split_kv_len = kv_len`).
Split-KV remains a decode/small-q lever, not a high-throughput prefill lever.

## D128 Rejected Probe: FP8 Logits Staging

Tested replacing the intermediate BF16 logits tile with signed E4M3 byte
storage. The motivation was to halve QK->softmax smem traffic and make M64
logits fit inside the two QK-B stages:

```text
D128 M64 x N128 x K128 + E4M3 logits:
  storage:                         50176 bytes
  q=512 kv=8192 group=8:           cosine=0.991750, min_ms=0.154592
  q=32768 kv=32768 group=8:        cosine=0.990118, min_ms=12.876128

Baseline M64 explicit BF16 logits:
  storage:                         66560 bytes
  q=512 kv=8192 group=8:           min_ms ~= 0.139712
  q=32768 kv=32768 group=8:        min_ms ~= 11.845
```

Correctness held, but both smoke and anchor regressed. The conversion cost and
loss of BF16 logit precision outweigh the reduced smem footprint. Reverted.
This means the next meaningful D128 lever is not narrower logit storage; it is
removing the QK-logits smem handoff entirely or reducing role/barrier overhead
around the existing BF16 handoff.

## D128 Rejected Probe: Second P Buffer Scheduling

Tested the bounded form of the second-P-buffer option. The change set:

```text
kSm120D128SinglePBuffer = false
mainloop order:
  QK(current) -> softmax/P(current) -> PV(previous)
```

This materializes a separate P1 data/scale buffer and uses tile parity to avoid
overwriting `P(previous)` before PV consumes it. It does not implement direct
QK-accumulator-to-P packing; it only tests whether the extra P buffer removes a
profitable scheduling dependency in the current BF16-logits path.

Results on 2026-04-30:

```text
D128 M64 x N128 x K128, second P buffer:
  storage:                         77824 bytes
  storage margin:                  23552 bytes
  q=512 kv=8192 group=8:           cosine=0.991697, min_ms=0.177728
  q=32768 kv=32768 group=8:        cosine=0.990060, min_ms=13.690624

Baseline single P buffer:
  storage:                         66560 bytes
  q=512 kv=8192 group=8:           min_ms ~= 0.141120
  q=32768 kv=32768 group=8:        min_ms ~= 11.852992
```

Decision: reject and revert. The extra P buffer costs 11,264 bytes in the real
layout and regresses both smoke and anchor. The ordering freedom alone is not a
win. A future direct accumulator-to-P path must use the single-buffer reordered
loop (`PV(previous) -> QK(current) -> direct softmax/P(current)`) or prove a
separate benefit large enough to justify the extra shared-memory footprint.

## D128 Rejected Probe: Naive Direct Accumulator-To-P

Tested a single-P-buffer direct-P prototype that keeps the storage footprint
unchanged by reusing the old BF16 logits allocation as reduction scratch. The
loop order was changed to:

```text
tile 0:      QK(0) -> direct softmax/P(0)
tile n > 0:  PV(n - 1) -> QK(n) -> direct softmax/P(n)
final:       PV(last)
```

The QK C-fragment values stayed in registers and were used to:

```text
1. write per-row/owner partial maxima into scratch
2. reduce row max from scratch
3. recompute probabilities and write per-row sums / per-group maxima
4. reduce row sum and group maxima from scratch
5. recompute probabilities again and pack P directly to PV smem layout
```

Correctness held, but performance collapsed:

```text
D128 M64 x N128 x K128, naive direct-P:
  storage:                         66560 bytes
  q=512 kv=8192 group=8:           cosine=0.991696, min_ms=0.633152
  q=32768 kv=32768 group=8:        cosine=0.990092, min_ms=51.480801

Baseline BF16-logits handoff:
  storage:                         66560 bytes
  q=512 kv=8192 group=8:           min_ms ~= 0.141120
  q=32768 kv=32768 group=8:        min_ms ~= 11.852992
```

Decision: reject and revert. This validates the owner mapping well enough for
correctness, but the naive implementation adds too many scratch round trips,
barriers, and duplicate `expf` work. Removing the BF16 logits handoff is still a
possible lever, but it needs a different algorithm: row/group reductions must be
warpgroup-local or fused into the existing softmax ownership pattern, not built
as a scratch-heavy accumulator-owner pass.

## D128 Post-Hillclimb Cleanup

After stopping the D128 hill-climb, cleaned the D128 specialization translation
unit for the next phase. Removed code that represented settled or rejected
experiments:

```text
- second-P-buffer storage and parity branches
- D128 experiment-switch names for single-P, row-skew, and softmax-thread count
- zero-skew / two-thread-softmax conditionals from the active path
- stale CUTLASS tile-variant metadata matrix from the M/N/K sweep
- stale `smem_logits1` alias; D128 uses one BF16 logits scratch tile
```

Kept the active D128 kernel path, atom-level debug/correctness entry points,
split-KV wrapper, and active metadata that the bench harness still uses.

Validation after cleanup:

```text
q=512 kv=8192 group=8:
  storage=66560 bytes, cosine=0.991697, min_ms=0.141888

q=32768 kv=32768 group=8:
  storage=66560 bytes, cosine=0.990060, min_ms=11.579968
```

The cleanup does not change shared-memory footprint or correctness. Runtime is
within expected noise and slightly better than the previous anchor measurement.

## D128 180-Cell Post-Cleanup Matrix

Ran the D128 expanded matrix after the D128 post-hillclimb cleanup with the
active D128 fused kernel:

```text
q_len:  128, 256, 512, 1024, 2048, 4096
kv_len: 8192, 32768, 65536, 131072, 262144
group:  2, 4, 6, 8, 12, 16
kernel columns: sm120_fused, nvfp4_fa2, fp8_fa2, bf16_fa2
split_kv_len: 32768
output_group_span: 1
```

Artifacts:

```text
reports/d128_hillclimb_180cell_20260430.jsonl
reports/d128_hillclimb_180cell_20260430.csv
reports/d128_hillclimb_180cell_20260430.summary.csv
reports/d128_hillclimb_180cell_20260430.md
reports/d128_hillclimb_180cell_20260430.run.log
```

Run status:

```text
summary rows:              180 / 180
error rows:                0
min fused cosine:          0.989452
cells beating nvfp4_fa2:   119 / 180
cells passing 2x gate:     37 / 180
```

Per-group rollup versus NVFP4 FA2:

```text
group  cells  wins  pass_2x  median_speedup  max_speedup
2      30     11    0        0.769           1.985
4      30     16    3        1.449           2.456
6      30     21    5        1.176           3.042
8      30     21    8        1.550           2.460
12     30     25    10       1.456           3.050
16     30     25    11       1.862           2.455
```

Best cells remain high-q / long-context. Top observed speedups:

```text
q=2048 kv=262144 group=12: fused=9.147 ms,  nvfp4_fa2=27.901 ms, speedup=3.050x
q=4096 kv=262144 group=6:  fused=9.154 ms,  nvfp4_fa2=27.843 ms, speedup=3.042x
q=2048 kv=131072 group=12: fused=4.808 ms,  nvfp4_fa2=13.907 ms, speedup=2.892x
q=4096 kv=131072 group=6:  fused=4.812 ms,  nvfp4_fa2=13.859 ms, speedup=2.880x
```

Worst cells are small-q / short-to-mid context, where launch/split overhead and
fixed fused-kernel cost dominate. This confirms the D128 kernel is useful in the
same regime as D256/D512: sufficiently large query tiles and long contexts, not
small-q cells.

## D256/D512 Post-Hillclimb Cleanup

Cleaned the D256 and D512 specialization translation units to match the D128
post-hillclimb cleanup pattern. Removed settled or rejected experiment code from
the active source:

```text
D256 removed:
- settled D256 experiment constants/branches:
  direct MMA epilogue, single softmax warp, MMA-owned softmax,
  single-P-buffer switch, min-blocks override, logits-row-skew switch,
  softmax-threads switch
- separate softmax-role score pipeline:
  PipelineS, OrderedSequenceBarrier, pipeline_mma_s0/s1 state,
  score-stage acquire/commit helpers, and the old `is_softmax` branch
- direct epilogue helper/branch
- stale CUTLASS tile-variant metadata matrix from the M/N/K sweep

D512 removed:
- settled D512 experiment constants/branches:
  MMA-owned softmax, min-blocks override, logits-row-skew switch,
  softmax-threads switch
- separate softmax-role score pipeline:
  PipelineS, OrderedSequenceBarrier, pipeline_mma_s0/s1 state,
  score-stage acquire/commit helpers, and the old `is_softmax` branch
- stale CUTLASS tile-variant metadata matrix
```

Kept the active production paths:

```text
D256: M64 x N128 x K128, reuse2-capable split-KV path, MMA-owned softmax,
      single P buffer, dedicated epilogue role
D512: M128 x N128 x K256, reuse4-capable split-KV path, MMA-owned softmax,
      dedicated epilogue role
```

Validation after cleanup:

```text
static checks:
  git diff --check: pass
  python py_compile:
    benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py: pass
    benchmarks/bench_sm120_d256_hillclimb.py: pass

D256 smoke:
  q=512 kv=8192 group=6 span=2 split_kv_len=8192
  storage=50176 bytes, margin=51200 bytes
  finite=true, cosine=0.991516
  mean_abs=0.000338807, max_abs=0.001601808
  min_ms=0.206304

D512 smoke:
  q=512 kv=8192 group=4 span=4 split_kv_len=8192
  storage=96256 bytes, margin=5120 bytes
  finite=true, cosine=0.990005
  mean_abs=0.000335312, max_abs=0.001747830
  min_ms=1.302048
```

## Integration Phase: Score Semantics Milestone

2026-04-30T15:57:16-05:00

The active D128/D256/D512 fused benchmark kernels were still dense score
kernels: contiguous K/V operands, no paged-KV indirection, no ragged scheduling,
and the split-KV harness reference used plain dense softmax. The next production
integration phase starts by making score semantics explicit before changing the
load scheduler.

Implemented across all three specializations:

```text
stage-kernel parameters:
  q_len
  group_size
  kv_len_tokens
  causal
  sliding_window
  logits_soft_cap
```

Score transform in QK:

```text
if score is outside the active causal/sliding window:
  score = -inf
else if logits_soft_cap > 0:
  score = logits_soft_cap * tanh(score / logits_soft_cap)
```

The online softmax path now handles fully masked tile chunks without generating
NaNs by treating their contribution as zero. Split-KV combine also handles
fully masked split rows by assigning zero weight instead of evaluating
`exp(-inf - -inf)`.

Harness changes:

```text
--causal
--sliding-window
--logits-soft-cap
```

The PyTorch reference applies the same score transform before softmax.

Validation:

```text
static checks:
  git diff --check: pass
  python py_compile benchmarks/bench_sm120_nvfp4_cutlass_fused_attention.py: pass

D128 dense smoke:
  q=512 kv=8192 group=8 split_kv_len=8192
  finite=true, cosine=0.991697
  mean_abs=0.000358316, max_abs=0.001803321
  min_ms=0.283584

D256 dense smoke:
  q=512 kv=8192 group=6 span=2 split_kv_len=8192
  finite=true, cosine=0.991516
  mean_abs=0.000338966, max_abs=0.001632325
  min_ms=0.342112

D512 dense smoke:
  q=512 kv=8192 group=4 span=4 split_kv_len=8192
  finite=true, cosine=0.990005
  mean_abs=0.000335312, max_abs=0.001747830
  min_ms=2.084896

D128 causal + sliding + softcap smoke:
  sliding_window=1024, logits_soft_cap=50.0
  finite=true, cosine=0.994457
  mean_abs=0.000988853, max_abs=0.005020801
  min_ms=0.266336

D256 causal + sliding + softcap smoke:
  sliding_window=1024, logits_soft_cap=50.0
  finite=true, cosine=0.990931
  mean_abs=0.000954257, max_abs=0.005854895
  min_ms=0.326912

D512 causal + sliding + softcap smoke:
  sliding_window=1024, logits_soft_cap=50.0
  finite=true, cosine=0.988548
  mean_abs=0.000950641, max_abs=0.005102105
  min_ms=1.994720
```

Remaining production integration work:

```text
1. Paged KV:
   current K/V loads use dense CUTLASS TMA descriptors, where `kv_tile` maps
   directly to a contiguous 128-token tile. vLLM page tables break that
   assumption. The next structural load path must either issue per-page loads
   into the existing smem layout or gather pages into dense/PV-ready staging.

2. Variable-length batches:
   current grid maps blockIdx.x directly to a dense q tile and blockIdx.z to a
   uniform split-KV range. Ragged batches need a tile schedule mapping each CTA
   to `(sequence, q_tile, split, output_group)` plus per-sequence q/kv lengths.

3. Production V layout:
   the benchmark path currently feeds PV with a V operand already arranged for
   the block-scaled PV GEMM. A paged serving path must either store V in that
   PV-ready layout or transform page-major V into the PV operand layout during
   the paged load/gather stage.
```

## Integration Phase: Paged Adapter Milestone

2026-04-30T16:06:28-05:00

Added a shared paged adapter hook used by the D128/D256/D512 benchmark
extensions:

```text
benchmarks/sm120_nvfp4_paged_adapter.cuh
```

Exported binding:

```text
sm120_nvfp4_gather_paged_kv_to_dense_pv(
  k_pages, k_sf_pages,
  v_pages_pv, v_sf_pages_pv,
  block_table,
  k_dense, k_sf_dense,
  v_pv_dense, v_pv_sf_dense,
  kv_head,
  kv_len
)
```

Scope:

```text
- page_size=16
- one sequence
- one KV head
- K source layout:       [num_pages, 16, H_kv, D/2]
- K scale source layout: [num_pages, 16, H_kv, D/16]
- V source layout:       page-major PV-layout NVFP4 bytes
- V scale source layout: page-major PV-layout scale sidecar
- dense output K:        [kv_len, D/2]
- dense output K scales: [kv_len, D/16]
- dense output V/PV:     [D, kv_len/2]
- dense output V scales: [D, kv_len/16]
```

This is not yet the final production load path. It is a correctness bridge from
vLLM-style block-table pages to the existing dense TMA operands. It lets us
validate page-table indirection and the V PV-layout transpose independently
before replacing the in-kernel dense TMA K/V load helpers.

Harness additions:

```text
--paged-adapter-check
--paged-adapter-shuffle-pages
```

The harness now constructs physical pages from the dense CUTLASS operands,
optionally shuffles physical page order, gathers through the adapter, verifies
byte-identical reconstruction, then runs the fused attention kernel using the
gathered operands.

Validation with shuffled physical pages plus causal/sliding/softcap:

```text
D128:
  q=512 kv=8192 group=8 split_kv_len=8192
  paged_adapter_k_equal=true
  paged_adapter_k_sf_equal=true
  paged_adapter_v_equal=true
  paged_adapter_v_sf_equal=true
  finite=true, cosine=0.994457
  min_ms=0.266080

D256:
  q=512 kv=8192 group=6 span=2 split_kv_len=8192
  paged_adapter_k_equal=true
  paged_adapter_k_sf_equal=true
  paged_adapter_v_equal=true
  paged_adapter_v_sf_equal=true
  finite=true, cosine=0.990931
  min_ms=0.321312

D512:
  q=512 kv=8192 group=4 span=4 split_kv_len=8192
  paged_adapter_k_equal=true
  paged_adapter_k_sf_equal=true
  paged_adapter_v_equal=true
  paged_adapter_v_sf_equal=true
  finite=true, cosine=0.988548
  min_ms=1.993952
```

Next structural step:

```text
Move the same page-table mapping from the pre-gather adapter into the load role:
  dense load_k_chunk(kv_tile, k_outer)
    -> paged load over the 8 physical 16-token pages in that 128-token tile

  dense load_v_group_span(kv_tile)
    -> paged V/PV load over the same physical pages and output-column groups

The final version should avoid materializing dense K/V operands in global
memory. The adapter remains useful as an oracle and fallback while the in-kernel
gather path is being built.
```

## Integration Phase: Direct Paged Producer Probe

2026-04-30T17:12:00-05:00

Tried moving the page-table mapping directly into the D256 load role by having
the load warp write paged K/V data into the existing CUTLASS QK/PV shared-memory
operand tensors and then manually completing the `PipelineTmaAsync`
transaction barrier.

Result:

```text
D256 q=512 kv=8192 group=6 span=2
causal=true sliding_window=1024 logits_soft_cap=50.0
adapter reconstruction: byte-identical
direct in-kernel paged producer cosine: 0.934877
```

A second version removed sub-byte write races by writing packed bytes/atomic
nibbles through the CUTLASS smem layout. It still failed correctness:

```text
direct in-kernel paged producer cosine: 0.942942
```

Conclusion:

```text
The page-table data and PV page conversion are correct; the failing piece is the
manual producer convention for CUTLASS SM120 block-scaled shared-memory operands.
The direct paged producer path was removed from callable code rather than left
as a broken experiment.
```

Current shippable correctness path:

```text
paged pages + block_table
  -> sm120_nvfp4_gather_paged_kv_to_dense_pv
  -> existing dense fused D128/D256/D512 kernels
```

Validation after removing the broken direct producer entry point:

```text
D128 q=512 kv=8192 group=8:
  adapter equality: all true
  finite=true cosine=0.994457 min_ms=0.265920

D256 q=512 kv=8192 group=6:
  adapter equality: all true
  finite=true cosine=0.990931 min_ms=0.318272

D512 q=512 kv=8192 group=4:
  adapter equality: all true
  finite=true cosine=0.988548 min_ms=1.998336
```

Next production integration step:

```text
Lift the adapter bridge into a production-facing wrapper with explicit scratch
tensors/workspace, then add ragged-batch scheduling around that wrapper. Keep
the direct in-kernel paged producer as a future optimization only after porting
the exact CUTLASS TMA partition_D producer convention instead of re-deriving it.
```

## Integration Phase: Paged/Varlen Bridge Wrappers

2026-04-30T17:55:00-05:00

Implemented the production-facing bridge layer around the existing dense fused
D128/D256/D512 kernels.

New shared files:

```text
benchmarks/sm120_nvfp4_paged_adapter.cuh
benchmarks/sm120_nvfp4_paged_attention_bridge.cuh
```

New extension entry points in all three specialization translation units:

```text
sm120_nvfp4_paged_qkv_online_register_q_splitkv_full_grid
sm120_nvfp4_varlen_paged_qkv_online_register_q_splitkv_full_grid
```

Current production-correct path:

```text
paged NVFP4 K/V pages + block table
  -> gather one KV-head group to dense K and PV-ready V scratch
  -> dense SM120 NVFP4 fused split-KV attention kernel
```

The bridge is explicit about scratch ownership:

```text
single-sequence wrapper:
  caller provides K scratch, K-scale scratch, PV-layout V scratch,
  PV-layout V-scale scratch, partial output, split_m, split_l, out, workspace

varlen wrapper:
  caller provides q/q_scales, paged K/V tensors, block_tables,
  cu_seqlens_q, kv_lens, out, workspace
  wrapper allocates per-sequence dense K/V scratch and split-KV scratch
```

This is semantically correct but not the final fastest load path. It avoids
materializing impossible production assumptions in the kernel while leaving the
direct in-kernel paged producer for a later `partition_D`-based port.

Important contract changes:

```text
physical q rows may be padded to the kernel tile-M multiple
logical q_len * group_size <= physical q rows

physical KV length may be padded to the kernel tile-N multiple
logical kv_len_tokens <= physical KV length

the kernel masks padded Q rows and padded KV tokens before softmax
```

This contract is required for ragged production batches because vLLM requests
are not guaranteed to land on 64/128-row/tile boundaries.

Validation:

```text
D128 q=512 kv=8192 group=8 causal sliding_window=1024 softcap=50:
  paged adapter equality: all true
  paged bridge vs dense: mean_abs=0, max_abs=0, cosine=1.0
  varlen paged bridge vs dense: mean_abs=0, max_abs=0, cosine=1.0
  exact-ref cosine=0.994457

D256 q=512 kv=8192 group=6 causal sliding_window=1024 softcap=50:
  paged adapter equality: all true
  paged bridge vs dense: mean_abs=0, max_abs=0, cosine=1.0000001
  varlen paged bridge vs dense: mean_abs=0, max_abs=0, cosine=1.0000001
  exact-ref cosine=0.990931

D512 q=512 kv=8192 group=4 causal sliding_window=1024 softcap=50:
  paged adapter equality: all true
  paged bridge vs dense: mean_abs=0, max_abs=0, cosine=1.0000001
  varlen paged bridge vs dense: mean_abs=0, max_abs=0, cosine=1.0000001
  exact-ref cosine=0.988548
```

Ragged edge-case validation:

```text
D256 q_len=513 group=6:
  logical q rows: 3078
  physical q rows: 3136

logical KV: 8208
physical KV: 8320

varlen paged bridge vs manually padded dense kernel:
  finite=true
  mean_abs=0
  max_abs=0
  cosine=0.99999994
```

Current limitations:

```text
- The varlen bridge is single-KV-head-group scoped; integration must call it per
  KV head group or add a higher-level head loop.
- The bridge allocates per-sequence scratch internally. That is acceptable for
  semantic integration but should be replaced with caller-managed workspace
  before performance tuning.
- Decode-sized q rows still use the padded prefill kernel path. The existing
  XQA/decode policy remains the likely fast path for q=1.
- Direct in-kernel paged loads remain blocked on porting the CUTLASS
  partition_D producer convention exactly.
```

## Integration Phase: Dynamic BF16 Q Quantization

2026-04-30T18:10:00-05:00

The production bridge cannot rely on Python-only `flashinfer.nvfp4_quantize` for
Q packing because vLLM hands attention BF16 Q. The in-extension
`quantize_q_rowmajor` entry point is now shape-generic for the active
specializations:

```text
accepted q shapes:
  [rows, D]
  [q_len, group, D]

supported D:
  128, 256, 512 via the matching specialization extension
```

The quantizer writes row-major NVFP4 Q and e4m3 group scales for the local
fused kernel. It uses direct per-block scales, so the attention wrapper's
`qk_alpha` for this path should compensate the K global scale but not a Q
global scale. The benchmark path that uses `flashinfer.nvfp4_quantize` for Q
still uses `qk_alpha = 1 / (q_global * k_global)`.

Validation:

```text
D256 dynamic quantizer:
  q_len=513 group=6 D=256
  q_rows=3078
  finite=true
  mean_abs=0.017862
  cosine=0.995469 vs BF16 Q

D256 paged/varlen bridge after quantizer change:
  paged bridge vs dense: mean_abs=0, max_abs=0, cosine=1.0000001
  varlen paged bridge vs dense: mean_abs=0, max_abs=0, cosine=1.0000001
  exact-ref cosine=0.990931
```

## Integration Phase: Specialization File Naming

2026-04-30T18:18:00-05:00

Matched the D512 benchmark/prototype filename to the D128/D256 specialization
pattern:

```text
benchmarks/sm120_nvfp4_cutlass_fused_attention_d128.cu
benchmarks/sm120_nvfp4_cutlass_fused_attention_d256.cu
benchmarks/sm120_nvfp4_cutlass_fused_attention_d512.cu
```

The unsuffixed D512 name was a prototype artifact from when D512 was the only
specialization. Production naming still needs to move out of `benchmarks/` and
follow FlashInfer's operation/backend/arch convention.

## Integration Phase: Raw Launch Seam

2026-04-30T18:45:00-05:00

The benchmark extension wrappers are no longer the only callable boundary for
the fused kernels. D128, D256, and D512 now have a lower-level raw CUDA launch
function under the Torch wrapper:

```text
sm120_nvfp4_qkv_online_register_q_splitkv_full_grid_raw(...)
```

The raw launcher accepts device pointers, scalar launch parameters, workspace
bytes, and a CUDA stream. It owns:

```text
- CUTLASS QK params initialization
- CUTLASS PV params initialization
- dynamic shared-memory attribute setup
- split-KV stage kernel launch
- split-KV combine launch when needed
```

The existing pybind/Torch wrapper now performs tensor validation and converts
tensors to raw pointers before calling the raw launcher. This is the first
production extraction step: the same raw launch interface can be moved under
`include/flashinfer/attention/blackwell` and called by a TVM-FFI
`csrc/fmha_nvfp4_sm120*.cu` wrapper without carrying `torch::Tensor` into the
production kernel implementation.

Validation after the refactor:

```text
D128 q=512 kv=8192 group=8 causal sliding_window=1024 softcap=50:
  paged bridge vs dense cosine=1.0
  varlen paged bridge vs dense cosine=1.0
  exact-ref cosine=0.994457

D256 q=512 kv=8192 group=6 causal sliding_window=1024 softcap=50:
  paged bridge vs dense cosine=1.0000001
  varlen paged bridge vs dense cosine=1.0000001
  exact-ref cosine=0.990931

D512 q=512 kv=8192 group=4 causal sliding_window=1024 softcap=50:
  paged bridge vs dense cosine=1.0000001
  varlen paged bridge vs dense cosine=1.0000001
  exact-ref cosine=0.988548
```

Remaining production migration:

```text
1. Move raw launchers and shared CUDA implementation out of benchmarks.
2. Replace Torch-only paged bridge scratch allocation with caller-managed
   workspace.
3. Add TVM-FFI run bindings and a FlashInfer JIT module name.
4. Wire backend dispatch only after the source-tree module compiles and passes
   the same D128/D256/D512 correctness gates.
```

The paged gather adapter also now has a raw CUDA boundary in the FlashInfer
source include tree:

```text
include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_paged_adapter.cuh
gather_paged_kv_to_dense_pv_raw(...)
```

It accepts raw K/V page pointers, block-table pointer, dense scratch pointers,
shape scalars, and stream. The Torch adapter remains as validation/plumbing but
now delegates the actual page gather to that source-tree launcher. D256 paged
and varlen bridge validation was rerun after this extraction and remained
bit-identical to the dense path.

## Integration Phase: Source-Tree Utility JIT Module

2026-04-30T20:05:00-05:00

The Q quantizer and paged K/V gather now have FlashInfer source-tree entry
points that do not depend on the benchmark pybind extension:

```text
include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_quantization.cuh
  quantize_q_rowmajor_raw(...)

include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_paged_adapter.cuh
  gather_paged_kv_to_dense_pv_raw(...)

csrc/fmha_nvfp4_sm120_utils.cu
  TVM-FFI exports:
    quantize_q(...)
    gather_paged_kv_to_dense_pv(...)

flashinfer.jit.gen_fmha_nvfp4_sm120_utils_module()
```

The benchmark D128/D256/D512 extensions now call the source-tree Q quantizer
instead of their local launch body. The local benchmark helpers remain only as
validation scaffolding until the fused attention kernels themselves move out of
`benchmarks/`.

Validation:

```text
JIT module build:
  gen_fmha_nvfp4_sm120_utils_module().build_and_load()
  exports quantize_q=true, gather_paged_kv_to_dense_pv=true

SM120 functional check on RTX PRO 6000 Blackwell:
  quantize_q: launched on BF16 q=[4,2,128], produced nonzero packed/scales
  paged gather: deterministic K/K-scale/V-PV/V-scale references all matched
                exactly with shuffled physical pages and kv_head=1

D256 bridge smoke after source-tree Q quantizer:
  paged bridge vs dense cosine=1.0000001
  varlen paged bridge vs dense cosine=1.0000001
  exact-ref cosine=0.990931
```

Remaining production migration:

```text
1. Move the fused attention raw launchers and kernel implementation out of
   benchmark .cu files.
2. Replace Torch-only paged/varlen bridge allocation with caller-managed
   workspace layout.
3. Add a TVM-FFI fused attention run binding for D128/D256/D512.
4. Wire backend dispatch only after the source-tree run binding compiles and
   passes the same D128/D256/D512 paged/varlen correctness gates.
```

Layering correction:

```text
Rejected: dual-purpose benchmark files controlled by a SOURCE_ONLY-style macro.

Expected FlashInfer production structure:
  include/flashinfer/attention/blackwell/...
    reusable CUDA implementation and raw launch helpers

  csrc/...
    thin TVM-FFI production bindings that validate TensorView and call the
    launch helpers

  benchmarks/...
    Torch/pybind benchmark wrappers that call the same launch helpers

Benchmarks should measure the production kernel implementation, but the
benchmark .cu file should not become the production translation unit. Extract
the shared implementation into headers first, then wire csrc and benchmarks to
that shared implementation.
```

## Integration Phase: Production Header And Run Binding

2026-04-30T18:45:55-05:00

The fused D128/D256/D512 kernels have been extracted from benchmark-owned
translation units into reusable FlashInfer include headers:

```text
include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d128.cuh
include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d256.cuh
include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d512.cuh
```

The benchmark `.cu` files are now thin Torch/pybind wrappers around the same
source-tree implementation. They no longer own the production kernels. The
shared implementation headers are free of Torch and pybind symbols.

The Gemma grid's `sm120_fused` column now benchmarks the csrc/JIT production
module through:

```text
benchmarks/bench_fmha_nvfp4_sm120.py
```

The old pybind benchmark extension remains available for low-level diagnostics,
but it is no longer the grid path for the fused-kernel column.

A production TVM-FFI fused attention module now exists:

```text
csrc/fmha_nvfp4_sm120.cu
  TVM-FFI exports:
    run_dense(...)
    run_paged_single(...)
    run_paged_batch(...)

flashinfer.jit.gen_fmha_nvfp4_sm120_module()
```

`run_dense` dispatches D128/D256/D512 by `head_dim` and selects the
specialized output-group span. `run_paged_single` gathers one sequence's paged
K/V cache into caller-managed dense scratch and then launches the same dense
fused kernel.

`run_paged_batch` adds the source-tree varlen production boundary. It accepts
block tables, `qo_indptr`, `kv_lens`, and caller-managed scratch tensors. The
csrc layer does host-side sequence scheduling, gathers each sequence's paged
K/V into scratch, pads Q rows to the head-dim specialization's tile-M
requirement, launches the fused dense kernel, and copies only real rows back to
the output tensor. This removes the Torch-only benchmark bridge from the
validation path. It is still a gather-bridge implementation, not the final
native in-mainloop block-table load path.

Validation:

```text
JIT module build:
  gen_fmha_nvfp4_sm120_module().build_and_load()
  exports run_dense=true, run_paged_single=true

D256 production FFI dense vs benchmark wrapper:
  q=512 kv=8192 group=6 head_dim=256 causal sliding_window=1024 softcap=50
  ffi_vs_ext_equal=true
  ffi_vs_ext_max_abs=0.0

D256 production FFI paged_single vs dense:
  paged_vs_dense_equal=true
  paged_vs_dense_max_abs=0.0
  scratch K/K-scale/V-PV/V-scale all matched the paged adapter reference

D256 production FFI paged_batch, two sequences:
  q_lens=[256,512], kv_lens=[8192,8192], group=6
  paged_batch_vs_single_equal=true
  paged_batch_vs_single_max_abs=0.0

D256 production FFI paged_batch with Q padding:
  q_lens=[257,511], kv_lens=[8192,8192], group=6
  paged_batch_padding_vs_manual_dense_equal=true
  paged_batch_padding_vs_manual_dense_max_abs=0.0

D128/D512 production FFI paged_batch with Q padding:
  D128 q_lens=[129,257], kv_lens=[8192,8192], group=8
    paged_batch_padding_equal=true, max_abs=0.0
  D512 q_lens=[129,257], kv_lens=[8192,8192], group=4
    paged_batch_padding_equal=true, max_abs=0.0

D128 benchmark smoke after extraction:
  q=512 kv=8192 group=8 causal sliding_window=1024 softcap=50
  paged bridge vs dense cosine=1.0
  varlen paged bridge vs dense cosine=1.0
  exact-ref cosine=0.994457

D512 benchmark smoke after extraction:
  q=512 kv=8192 group=4 causal sliding_window=1024 softcap=50
  paged bridge vs dense cosine=1.0000001
  varlen paged bridge vs dense cosine=1.0000001
  exact-ref cosine=0.988548

csrc/JIT benchmark smoke:
  D256 q=512 kv=8192 group=6 output_group_span=2
    min_ms=0.491840, output_finite=true
  D256 q=512 kv=8192 group=6 output_group_span=2 without CUTLASS_ROOT
    min_ms=0.490816, output_finite=true
  Gemma grid sm120_fused smoke:
    Shape B q=512 kv=8192 D512 group=8
    min_ms=1.943552, status=ok
```

Important dependency note:

```text
The fused JIT module originally needed CUTLASS_ROOT=/home/josh/tdm/cutlass so
the build could see the SM120 cooperative/block-scaled CUTLASS fixes. The repo
submodule has now been moved to the patched local CUTLASS commit:

  e095a676 Support M32 SM120 blockscaled NVFP4 GEMM

`gen_fmha_nvfp4_sm120_module()` now prefers `3rdparty/cutlass` when running
from a source checkout, and falls back to packaged `flashinfer.data.cutlass` in
installed builds. Clean JIT validation without CUTLASS_ROOT:

  fmha_module_loaded_no_cutlass_root=true

For PR hygiene, the CUTLASS submodule commit must be reachable by whichever
remote the FlashInfer fork uses before this branch is shared.
```

Remaining production migration:

```text
1. Replace the paged gather bridge with native block-table loads inside the
   fused mainloop if the bridge is not fast enough for vLLM serving.
2. Wire backend dispatch only after the source-tree run binding is wrapped by a
   FlashInfer Python API and benchmarked against the real vLLM call shape.
3. Remove any benchmark-only debug kernels or exports that are no longer part
   of the production validation surface.
```

## Production PV KV Quantization API

The SM120 NVFP4 fused attention path needs V stored in the PV operand layout:

```text
V data pages:  [num_pages, 16, num_kv_heads, head_dim / 2]
V scale pages: [num_pages, 16, num_kv_heads, head_dim / 16]
```

The existing `nvfp4_quantize_paged_kv_cache` API now has:

```text
v_data_layout="normal"  # historical behavior
v_data_layout="pv"      # SM120 PV-reblocked V data
v_scale_layout="pv"     # matching SM120 PV scale sidecar
```

Constraints for the PV layout:

```text
kv_layout must be "NHD"
page_size must be 16
v_data_layout="pv" requires v_scale_layout="pv"
```

Validation:

```text
PV layout compatibility against the old benchmark-private helper:
  pv_data_equal=true
  pv_sf_equal=true
  K shape=(4, 16, 2, 128), K scale shape=(4, 16, 2, 16)
  V shape=(4, 16, 2, 128), V scale shape=(4, 16, 2, 16)

Normal layout backward-compat smoke:
  normal_shapes=(4, 16, 2, 128), (4, 16, 2, 128),
                (4, 16, 2, 16), (4, 16, 2, 16)
  normal_dtypes=uint8, uint8, float8_e4m3fn, float8_e4m3fn

Production csrc/JIT run_paged_batch using the public quantizer output:
  D128 paged_batch vs per-seq paged_single equal=true, max_abs=0.0, finite=true
  D256 paged_batch vs per-seq paged_single equal=true, max_abs=0.0, finite=true
  D512 paged_batch vs per-seq paged_single equal=true, max_abs=0.0, finite=true
```

The grouped-attention benchmark no longer has a separate PV reblock helper.
It now calls the same public quantizer path, so benchmark validation and the
production csrc binding share one V/PV layout implementation.

## SM120 NVFP4 Paged Prefill Python Wrapper

Added a FlashInfer Python wrapper:

```text
flashinfer.BatchPrefillWithPagedKVCacheSM120Nvfp4Wrapper
```

The wrapper is the production call boundary above the csrc/JIT module. It owns:

```text
- plan-time scratch sizing for D128/D256/D512
- BF16 Q -> NVFP4 Q quantization through csrc/fmha_nvfp4_sm120_utils.cu
- per-KV-head grouped dispatch for GQA/MQA
- run_paged_batch invocation with PV-layout NVFP4 K/V pages
- output scatter back to [total_q, num_qo_heads, head_dim]
```

Current constraints:

```text
- SM120/SM121 target
- BF16 Q and BF16 output
- NHD paged KV
- page_size=16
- NVFP4 K pages in normal page layout
- NVFP4 V pages in PV-reblocked layout from
  nvfp4_quantize_paged_kv_cache(..., v_data_layout="pv", v_scale_layout="pv")
- csrc implementation still gathers paged K/V into dense scratch before the
  fused dense SM120 kernel; native in-mainloop page-table loads remain the next
  performance migration if the gather bridge is not enough.
```

Validation:

```text
Ad hoc wrapper smoke:
  D128 two-KV-head wrapper equals independent single-KV-head wrappers, max_abs=0.0
  D256 two-KV-head wrapper equals independent single-KV-head wrappers, max_abs=0.0
  D512 two-KV-head wrapper equals independent single-KV-head wrappers, max_abs=0.0

Pytest:
  tests/attention/test_nvfp4_kv_head_dim_512.py::test_sm120_nvfp4_wrapper_multi_kv_matches_single_kv_sm12x
  3 passed
```

## Production Wrapper Benchmark Path

The Gemma grid no longer benchmarks `sm120_fused` through the dense/prepacked
microbenchmark path. It now calls:

```text
benchmarks/bench_fmha_nvfp4_sm120.py --mode paged-wrapper
```

This measures the production wrapper boundary:

```text
BF16 Q -> csrc Q quantization
PV-layout NVFP4 paged K/V input
run_paged_batch paged gather bridge
fused SM120 attention
output scatter to [q, heads, d]
```

The grid also now passes production Gemma semantics explicitly:

```text
causal=true
logits_soft_cap=50.0
Shape A window_left=1024
Shape B window_left=-1
```

FMHAv2 on SM120 does not support sliding-window prefill, so
`flashinfer_nvfp4_fmha_v2` is marked unsupported for Shape A instead of being
timed under the wrong mask. BF16 Shape A uses FA2 in the baseline grid.

Validation:

```text
bench_fmha_nvfp4_sm120.py --mode paged-wrapper
  D256 q=128 kv=1024 group=6 causal sliding_window=1024 softcap=50
  min_ms=0.068192, output_finite=true

bench_gemma4_attention_grid.py sm120_fused smoke
  Shape A q=512 kv=1024 D256 group=2
  min_ms=0.068064, status=ok

bench_gemma4_attention_grid.py Shape A smoke with sm120_fused + BF16 baseline
  sm120_fused min_ms=0.068352, status=ok
  flashinfer_bf16 min_ms=0.069024, status=ok
```
