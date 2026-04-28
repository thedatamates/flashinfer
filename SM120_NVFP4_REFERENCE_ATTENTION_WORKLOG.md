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
