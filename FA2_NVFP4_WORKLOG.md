# FA2 NVFP4 Worklog

Primary target:

```text
Production target is Gemma4 31B dense on Blackwell sm120.

Model: google/gemma-4-31B-it
Validation weights: nvidia/Gemma-4-31B-IT-NVFP4
Hardware: RTX PRO 6000 Blackwell, sm120

Make the FP4/NVFP4 KV path beat the FP8 KV FA2 path on wall time for Gemma4 31B
production attention shapes, with correctness preserved. Parity with FP8 is not
a win.
```

Success bar:

```text
For every cell in the production grid:

  FP4 path wall_ms < FP8 FA2 wall_ms
  PyTorch reference correctness passes

The FP4 path can be FA2, FMHA-v2, or a per-shape hybrid policy. The comparison
that matters from here is FP4 path versus FP8 FA2 on the same cell.
```

Production shapes:

```text
Shape A, sliding_attention layers, 50/60 layers:
  D=256, group=2, kv_len=1024, q_len in {1, 512, 2048}

Shape B, full_attention layers, 10/60 layers:
  D=512, group=8, kv_len in {8k, 32k, 131k}, q_len in {1, 512, 2048}

D=128 is not a Gemma4 31B head dimension. It is no longer a target shape.
```

Rejected optimizations from the FMHA-v2 phase:

```text
Do not retry K=V buffer sharing. attention_k_eq_v=true is a projection-level
weight-tying optimization only. After projection, K goes through k_norm with
learnable weight and RoPE; V goes through v_norm without weight and skips RoPE.
The KV cache stores distinct K and V tensors.

Do not retry cross-layer KV sharing. Gemma4 31B has num_kv_shared_layers: 0.

Do not retry scale-load warp shuffles without new evidence. In the FMHA-v2
phase, shuffle-based scale dedup increased instruction/register pressure and
regressed D128/D256 by roughly 30%. At small head dim with tight register
budgets, redundant lane work can be faster than extra warp coordination.
```

## 2026-04-27 - Why FA2 Needs Its Own Investigation

Current FMHA-v2 result summary:

```text
D512/global, group=8, q=512, kv=32768:
  FP4 FMHA-v2 min: 4.8608 ms
  FP8 FA2 min:     4.9436 ms
  FP4 vs FP8:      1.02x

D256/sliding, group=2, q=512, kv=1024:
  FP4 FMHA-v2 min: 0.1930 ms
  FP8 FA2 min:     0.0538 ms
  FP4 vs FP8:      0.28x

D256/sliding, group=2, q=2048, kv=1024:
  FP4 FMHA-v2 min: 0.2630 ms
  FP8 FA2 min:     0.0849 ms
  FP4 vs FP8:      0.32x
```

Conclusion:

```text
FMHA-v2 is not delivering a compelling FP4 win. D512 parity/slight win is not
enough, and D256 sliding is a large regression versus FP8 FA2. D256/sliding
should be treated as a likely FA2-port or FA2-native optimization problem unless
new evidence shows FMHA-v2 can close the gap.
```

Code facts verified:

```text
flashinfer/prefill.py documents NVFP4 KV support for:
  - fa2
  - fmha_v2
  - trtllm-gen

The standard NVFP4 KV scale layout is supported by FA2.

The FMHA-v2-specific option:
  nvfp4_v_cache_uses_pv_layout

is explicitly rejected unless backend == "fmha_v2".
```

Implication:

```text
FA2 is not missing NVFP4 KV entirely. What appears missing is the specific
FMHA-v2 PV-layout/native-FP4-MMA path. The investigation should determine
whether to:

1. Use FA2's existing NVFP4 support directly where it wins.
2. Port the PV-layout/native FP4 pieces into FA2.
3. Keep a hybrid policy: FA2 for D256 sliding, FMHA-v2 or XQA for shapes where
   they actually win.
```

First FA2 questions:

```text
1. What does FA2 actually do for NVFP4 KV today?
   - Does it dequantize FP4 to BF16/FP16 before matmul?
   - Does it use native Blackwell FP4 MMA anywhere?
   - Does it use the same FA2 split-KV planner/merge path as FP8?

2. Why is FP8 FA2 so fast on D256 sliding?
   - Kernel selection and split policy
   - CTA shape
   - Scale handling
   - Whether it avoids FMHA-v2's PV/output overhead

3. What would it take for FA2 to consume the faster NVFP4 layout?
   - Standard row-major V layout only?
   - Can PV-layout V be represented in FA2 without invasive changes?
   - Would adding native FP4 MMA to FA2 be smaller or larger than continuing
     FMHA-v2 D256 work?
```

Immediate next actions:

```text
1. Trace FA2 NVFP4 paged-prefill source path from wrapper.run() into generated
   kernels before benchmarking. Confirm whether FA2 uses native FP4 MMA,
   dequantizes before matmul, or only exposes a documented-but-broken surface.
2. Run one-shape correctness first: FA2 FP4 KV on Shape A
   (D=256, group=2, kv=1024) versus PyTorch reference.
3. If correctness passes, benchmark Shape A:
   - FP8 FA2
   - NVFP4 FA2 standard layout
   - FP4 FMHA-v2 PV layout
4. Inspect generated FA2 CUDA/SASS for NVFP4:
   - native FP4 MMA vs dequant path
   - split-KV use
   - shared/global access pattern
5. Verify the suspicious D512 FP8 FA2 baseline by checking kv scaling at
   kv={8k,32k,131k} and ncu if needed.
6. Decide whether D256 should route to FA2 by policy or whether FA2 needs a new
   native-FP4/PV-layout implementation.
```

## 2026-04-27 - FA2 NVFP4 Source Trace

Scope:

```text
This trace is for the current flashinfer-nvfp4-kv worktree, not upstream main.
The worktree already contains local NVFP4 FA2/FMHA-v2 edits.
```

Gemma4 config confirmation:

```text
google/gemma-4-31B-it text_config:
  num_hidden_layers: 60
  enable_moe_block: false
  layer_types: 50 sliding_attention, 10 full_attention
  head_dim: 256
  global_head_dim: 512
  num_attention_heads: 32
  num_key_value_heads: 16
  num_global_key_value_heads: 4
  sliding_window: 1024
  max_position_embeddings: 262144
  attention_k_eq_v: true
  num_kv_shared_layers: 0
```

Dispatch/plumbing facts:

```text
flashinfer/jit/utils.py maps torch.uint8 KV to __nv_fp4x2_e2m1 through dtype_map_kv.
So a FA2 wrapper planned with kv_data_type=torch.uint8 generates DTypeKV=__nv_fp4x2_e2m1,
not plain uint8_t math.

FA2 batch-prefill accepts extra tensors:
  maybe_k_cache_sf: uint8_t*
  maybe_v_cache_sf: uint8_t*

and extra scalar strides:
  k_cache_sf_stride_page/h/n
  v_cache_sf_stride_page/h/n
  v_cache_sf_logical_cols
  v_cache_sf_col_offset

The Python wrapper requires kv_cache_sf whenever K/V cache dtype is uint8.
nvfp4_v_cache_uses_pv_layout is explicitly rejected unless backend == "fmha_v2".
```

Scale layout:

```text
nvfp4_quantize_paged_kv_cache returns:
  K data: packed FP4, same page/head/token layout as input, last dim head_dim/2
  V data: packed FP4, same page/head/token layout as input, last dim head_dim/2
  K scales: linear per-token/per-head layout, one UE4M3 byte per 16 values
  V scales: TRT-LLM-style 4-token interleaved/swizzled layout, one UE4M3 byte per 16 values

FA2 page_produce_kv_sf matches that: K scales use independent strides and cp.async 32-bit loads;
V scales use the 4-token interleaved address transform. In the current wrapper, the public run path
passes v_cache_sf_logical_cols=0 and v_cache_sf_col_offset=0, so it consumes the default full-head
swizzled V-scale tensor.
```

Kernel data movement:

```text
Packed K/V data staging uses cp.async into shared memory.
For FP4, the K/V path issues 64-bit cp.async loads from packed GMEM into the lower half of a
128-bit shared-memory slot, then keeps the existing FA2 shared-memory addressing.

K scale staging uses cp.async.ca.shared.global with a 4-byte transfer.
Paged V scale staging is not cp.async in the current source: it manually gathers four scale bytes
through scalar global loads, packs a uint32_t, then stores to shared memory. This is a likely
remaining FA2 FP4 bottleneck.
```

Math path:

```text
QK is not native FP4 MMA. FA2 loads packed FP4 K fragments from shared memory, converts them to
DTypeQ with vec_cast, applies the per-16-value UE4M3 scale in BF16/FP16 registers, and then uses
standard f16/bf16 MMA:

  mma_sync_m16n16k16_row_col_f16f16f32

PV has two paths:

1. Native FP4 PV helper:
     compute_sfm_v_native_fp4
     mma_sync_m16n16k64_row_col_f4f4f32

   But it is only enabled when:
     DTypeKV is FP4
     DTypeQKAccum is float
     NUM_MMA_KV == 4
     NUM_WARPS_KV == 1
     swizzle == k128B
     HEAD_DIM_QK == 512
     HEAD_DIM_VO == 256

   That is not the Gemma4 31B full-attention production shape, which is D=512/VO=512.

2. Generic PV fallback:
     loads packed FP4 V from shared memory
     converts V to DTypeQ
     applies V scale in BF16/FP16 registers
     uses standard f16/bf16 MMA

Therefore FA2 does not currently provide a native FP4-MMA path for either Gemma4 production shape:
  Shape A: D=256/VO=256 -> generic dequant + f16/bf16 MMA
  Shape B: D=512/VO=512 -> generic dequant + f16/bf16 MMA
```

Implication:

```text
FA2 "supports NVFP4 KV" in the current worktree, but for Gemma4 production shapes that support is
mostly a packed-storage/dequant path, not an end-to-end native FP4 attention path.

The immediate value of FA2 is its mature scheduler/tile/short-KV behavior, especially Shape A.
The immediate risk is that FP4 cannot beat FP8 FA2 if both QK and PV fall back to dequant + f16/bf16
MMA. The first correctness and timing check is still useful, but if it does not beat FP8, the next
engineering target is native FP4 PV for D512/VO512 and possibly D256/VO256, plus async/coalesced V
scale staging.
```

Shape A correctness smoke:

```text
Ran FA2 FP4 paged prefill against dequantized PyTorch reference:
  shape: Shape A sliding
  D: 256
  group: 2
  q_len: 512
  kv_len: 1024
  page_size: 16
  causal: true
  window_left: 1024
  q dtype/out dtype: bf16

Result:
  max_abs: 0.000244140625
  mean_abs: 2.2703283320879564e-05
  out finite: true
  ref finite: true
  assert_close(rtol=2e-1, atol=2e-1): pass
```

Shape A FA2 timing, q_len=512:

```text
Command:
  benchmarks/bench_nvfp4_fmha_v2_gqa_grouped_attention.py
    --gemma4-shape sliding
    --q-len 512
    --kv-len 1024
    --group-sizes 2
    --fp4-backend fa2
    --fp4-v-layout nhd
    --warmup 5
    --repeat 20
    --workspace-mib 512

Result:
  FP4 FA2 min: 0.034240 ms
  FP8 FA2 min: 0.028320 ms
  BF16 FA2 min: 0.163744 ms
  FP4 TFLOP/s: 31.36
  FP8 TFLOP/s: 37.91
  BF16 TFLOP/s: 6.56
  FP4 vs FP8: 0.83x
  FP4 vs BF16: 4.78x

Conclusion:
  Current FA2 FP4 does not meet the success bar for this Shape A cell. It is much faster than BF16,
  but slower than mature FP8 FA2.
```

Shape A FA2 timing, q_len=1:

```text
Command:
  benchmarks/bench_nvfp4_fmha_v2_gqa_grouped_attention.py
    --gemma4-shape sliding
    --q-len 1
    --kv-len 1024
    --group-sizes 2
    --fp4-backend fa2
    --fp4-v-layout nhd
    --warmup 5
    --repeat 20
    --workspace-mib 512

Result:
  FP4 FA2 min: 0.026464 ms
  FP8 FA2 min: 0.020896 ms
  BF16 FA2 min: 0.156032 ms
  FP4 TFLOP/s: 0.079
  FP8 TFLOP/s: 0.100
  BF16 TFLOP/s: 0.013
  FP4 vs FP8: 0.79x
  FP4 vs BF16: 5.90x

Conclusion:
  Current FA2 FP4 also misses the Shape A decode-cell bar. At q_len=1, fixed overhead and
  dequant/scale handling dominate; packed FP4 storage is not enough to beat FP8 FA2.
```

Shape A FA2 timing, q_len=2048:

```text
Command:
  benchmarks/bench_nvfp4_fmha_v2_gqa_grouped_attention.py
    --gemma4-shape sliding
    --q-len 2048
    --kv-len 1024
    --group-sizes 2
    --fp4-backend fa2
    --fp4-v-layout nhd
    --warmup 5
    --repeat 20
    --workspace-mib 512

Result:
  FP4 FA2 min: 0.065920 ms
  FP8 FA2 min: 0.051776 ms
  BF16 FA2 min: 0.161696 ms
  FP4 TFLOP/s: 65.15
  FP8 TFLOP/s: 82.95
  BF16 TFLOP/s: 26.56
  FP4 vs FP8: 0.79x
  FP4 vs BF16: 2.45x

Conclusion:
  Shape A q_len=2048 matches q_len=1 and q_len=512 directionally: FA2 FP4 is correctly
  using packed NVFP4 storage and beats BF16, but loses to FP8 FA2.
```

Shape A FA2 summary:

```text
All measured Shape A FA2 FP4 cells fail the success bar versus FP8 FA2:
  q_len=1:    FP4/FP8 = 0.79x
  q_len=512:  FP4/FP8 = 0.83x
  q_len=2048: FP4/FP8 = 0.79x

This is consistent with the source trace: for Gemma4 D=256/VO=256, FA2 FP4 is a
packed-storage/dequant path, not a native FP4-MMA attention path. The next useful benchmark is
Shape B D=512/group=8 because it decides whether FA2 has any immediate value for global layers and
also validates the suspicious FP8 FA2 D512 baseline.
```

Shape B FA2 timing, q_len=512, kv_len=32768:

```text
Command:
  benchmarks/bench_nvfp4_fmha_v2_gqa_grouped_attention.py
    --gemma4-shape global
    --q-len 512
    --kv-len 32768
    --group-sizes 8
    --fp4-backend fa2
    --fp4-v-layout nhd
    --warmup 5
    --repeat 20
    --workspace-mib 1024

Result:
  FP4 FA2 grouped min: 5.172608 ms
  FP8 FA2 grouped min: 5.002848 ms
  BF16 FA2 min:        4.656256 ms
  FP4 TFLOP/s:         53.14
  FP8 TFLOP/s:         54.94
  BF16 TFLOP/s:        59.03
  FP4 vs FP8:          0.97x
  FP4 vs BF16:         0.90x

Diagnostic:
  The same harness also reports FP4 FA2 "separate" min 4.771904 ms. That mode runs one
  num_qo_heads=1 wrapper per grouped Q head sequentially, so it is a diagnostic for GQA grouping
  overhead / tile policy, not yet a production dispatch path. It needs same-input correctness and
  launch-policy validation before it can count toward the success bar.

Conclusion:
  Current grouped FA2 FP4 misses the Shape B q512/kv32k bar by ~3.4% versus FP8 FA2, and both FP4
  and FP8 are slower than BF16 FA2 on this cell. This confirms that the D512 FA2 baseline is
  suspicious: the mature FP8 path is not clearly optimized here. Continue the D512 kv-scaling smoke
  before claiming any D512 win or loss.
```

Shape B FA2 kv-scaling smoke, q_len=512:

```text
Commands:
  Same as Shape B q512/kv32k run, with kv_len in {8192, 131072}.

Results:
  kv_len=8192:
    FP4 FA2 grouped min: 1.335616 ms
    FP8 FA2 grouped min: 1.300864 ms
    BF16 FA2 min:        1.209376 ms
    FP4 TFLOP/s:         51.45
    FP8 TFLOP/s:         52.83
    BF16 TFLOP/s:        56.82
    FP4 vs FP8:          0.97x
    FP4 vs BF16:         0.91x

  kv_len=32768:
    FP4 FA2 grouped min: 5.172608 ms
    FP8 FA2 grouped min: 5.002848 ms
    BF16 FA2 min:        4.656256 ms
    FP4 TFLOP/s:         53.14
    FP8 TFLOP/s:         54.94
    BF16 TFLOP/s:        59.03
    FP4 vs FP8:          0.97x
    FP4 vs BF16:         0.90x

  kv_len=131072:
    FP4 FA2 grouped min: 20.577856 ms
    FP8 FA2 grouped min: 19.870975 ms
    BF16 FA2 min:        23.421087 ms
    FP4 TFLOP/s:         53.43
    FP8 TFLOP/s:         55.33
    BF16 TFLOP/s:        46.95
    FP4 vs FP8:          0.97x
    FP4 vs BF16:         1.14x

Conclusion:
  D512 grouped FA2 scales roughly linearly with kv_len for both FP4 and FP8. The FP8 D512 baseline
  is not an obvious non-linear fallback, but it is also not clearly faster than BF16 at shorter
  kv. Grouped FP4 consistently trails grouped FP8 by ~3-4%, so D512 needs a targeted FA2 FP4 kernel
  improvement to clear the bar.

  The "separate" FP4 diagnostic is mixed:
    kv=8192:   separate 1.378848 ms, slower than grouped 1.335616 ms
    kv=32768:  separate 4.771904 ms, faster than grouped 5.172608 ms
    kv=131072: separate 18.253792 ms, faster than grouped 20.577856 ms

  That suggests grouped-GQA tile policy/occupancy overhead grows with long kv, but separate mode is
  still only a diagnostic because it means multiple num_qo_heads=1 launches. It cannot count as the
  production result until it is converted into an intentional dispatch policy and validated on
  same-input correctness and q_len=1/2048 cells.
```

Next FA2 experiment:

```text
The native FP4 PV helper already loops over NUM_MMA_D_VO, but the predicate only allowed
HEAD_DIM_QK=512/HEAD_DIM_VO=256. Shape B is HEAD_DIM_QK=512/HEAD_DIM_VO=512, so the next test is to
allow the existing helper for VO=512 and rerun correctness + D512 timing.

Patch under test:
  use_native_fp4_pv: allow HEAD_DIM_VO in {256, 512}
  prefer_native_fp4_pv_tile: allow HEAD_DIM_VO in {256, 512}
  FA2 plan override: route NVFP4 D512/VO512 through CTA_TILE_Q=32 so NUM_MMA_KV=4 fits and the
  native FP4 PV predicate can actually become true

This is not a completed fix yet. It must compile, pass PyTorch-reference correctness, and improve
Shape B FP4 grouped wall time below FP8 FA2 before it counts.

Intermediate result before the plan override:
  D512 correctness passed, but q512/kv32k timing did not move:
    pre-change FP4 grouped min: 5.172608 ms
    predicate-only FP4 grouped min: 5.174368 ms
  Reason: the planner still selected CTA_TILE_Q=64. For D512/VO512 this leaves enough shared memory
  for only NUM_MMA_KV=2, while native FP4 PV requires NUM_MMA_KV=4. The widened predicate was
  compiled but not selected for the benchmark tile.

Result with the plan override selecting CTA_TILE_Q=32:
  Correctness:
    tests/attention/test_nvfp4_kv_head_dim_512.py::test_nvfp4_paged_prefill_d512_random_quantized_matches_torch_sm12x
    passed.
  Performance, Shape B q512/kv32k:
    pre-change FP4 grouped min:     5.172608 ms
    predicate-only FP4 grouped min: 5.174368 ms
    forced-native FP4 grouped min:  7.246464 ms
    FP8 FA2 grouped min:            5.003264 ms

Conclusion:
  Reject this direct native-PV enablement for D512/VO512. It is correct but substantially slower.
  The existing helper was written for the D512/VO256 tile and does not become a production D512/VO512
  solution just by widening the predicate. Keep the finding; revert the code path before further
  work.
```

Next FA2 scale-staging experiment:

```text
Source finding:
  FA2 K scales are row-major and staged with 32-bit cp.async.
  FA2 V scales currently use TRT-LLM's 4-token interleaved layout and are gathered through scalar
  global loads, packed into a uint32_t, then stored to shared memory.

Patch under test:
  Add an explicit FA2 flag for row-major V scale factors:
    nvfp4_v_cache_sf_uses_linear_layout
  When true, page_produce_kv_sf<produce_v=true> stages V scales with the same 32-bit cp.async path
  used by K scales.

Benchmark support:
  benchmarks/bench_nvfp4_fmha_v2_gqa_grouped_attention.py now accepts:
    --fp4-v-sf-layout {swizzled,linear}
  The linear option unswizzles the V scale tensor once before timing and passes the new FA2 flag.

Success condition for this patch:
  Correctness must pass against the PyTorch reference and wall time must improve versus the swizzled
  scalar-gather V-scale path. If it improves but still does not beat FP8 FA2, it is a partial
  structural fix, not done.

Implementation note:
  Adding a new scalar to the generated FA2 FFI signature was rejected by the Python wrapper arity.
  For the experiment, the Python API flag is encoded through the existing V-scale layout fields:
    v_cache_sf_logical_cols == HEAD_DIM_VO / 16 and v_cache_sf_col_offset == 0
  means row-major V scales. This keeps the generated function arity stable.

Runtime finding:
  cp.async 32-bit V-scale staging faulted on this path:
    cuda error: operation not supported on global/shared address space
  A synchronous four-byte contiguous gather is correct and runs. This means the linear-layout result
  below isolates layout/coalescing, not a completed async scale-staging implementation.

Correctness, Shape B q512/kv1024:
  max_abs: 0.01348876953125
  mean_abs: 0.0025437804870307446
  finite: true
  assert_close(rtol=2e-1, atol=2e-1): pass

Timing, Shape B q512/kv32k:
  swizzled V scales FP4 grouped min: 5.172608 ms
  linear V scales FP4 grouped min:   5.164928 ms
  FP8 FA2 grouped min:               5.012032 ms

Conclusion:
  Row-major V scales are correct but only a ~0.15% D512 improvement with synchronous byte-gather
  staging. This does not materially close the FP8 gap. The cp.async form still needs a real fix if
  scale staging is to become a meaningful lever.

Timing, Shape A q512/kv1024:
  swizzled V scales FP4 grouped min: 0.034240 ms
  linear V scales FP4 grouped min:   0.035008 ms
  FP8 FA2 grouped min:               0.028256 ms

Conclusion update:
  Shape A regresses with row-major V scales, and Shape B improves by only ~0.15%. This experiment
  should not be promoted as production behavior. Keep it as evidence that V-scale layout alone is
  not the next big lever; if scale staging is revisited, it needs a real async/coalesced sidecar
  load design rather than just changing the scale tensor layout.
```

Grouped-vs-separate D512 q_len sweep:

```text
Setup:
  Shape B, D512/group8/kv32k, FA2 FP4 NHD layout, FP8 FA2 baseline.
  "separate" launches one group-size=1 wrapper per query head. This is a diagnostic only: it uses
  multiple launches and different synthetic Q tensors, so it is not directly production behavior.

Results:
  q_len=1:
    FP4 grouped min:   0.137760 ms
    FP4 separate min:  1.015712 ms
    FP8 grouped min:   0.133792 ms
    grouped vs FP8:    0.97x
    separate/grouped:  7.37x slower

  q_len=512:
    FP4 grouped min:   5.172608 ms
    FP4 separate min:  4.771904 ms
    FP8 grouped min:   5.002848 ms
    grouped vs FP8:    0.97x
    separate/grouped:  0.92x

  q_len=2048:
    FP4 grouped min:   25.027296 ms
    FP4 separate min:  19.241888 ms
    FP8 grouped min:   24.145056 ms
    BF16 grouped min:  11.807840 ms
    grouped vs FP8:    0.96x
    separate/grouped:  0.77x

Conclusion:
  Splitting the GQA group into group-size=1 work is a real long-q lever, but a naive multi-launch
  policy is invalid for decode and q_len=1 because launch overhead dominates. If this path is
  pursued, it needs either an in-kernel split/group policy or a q_len/kv_len-dependent dispatch that
  never uses multi-launch splitting for q_len=1. This is large enough to clear FP8 for q_len>=512,
  but not a complete production answer yet.
```

D512 FA2 tile-depth/tile-Q tuning:

```text
Experiment:
  Raise the FP4 D512/VO512 max NUM_MMA_KV cap from 1 to 2. The prior cap limited the KV tile depth
  for the production global-attention shape and was a plausible source of loop/staging overhead.

Results, Shape B D512/group8/kv32k:
  q_len=512:
    before FP4 grouped min: 5.172608 ms
    after  FP4 grouped min: 4.854208 ms
    FP8 grouped min:        4.995520 ms
    result:                 FP4 is 1.03x faster than FP8

  q_len=2048:
    before FP4 grouped min: 25.027296 ms
    after  FP4 grouped min: 23.234016 ms
    FP8 grouped min:        24.178783 ms
    result:                 FP4 is 1.04x faster than FP8

  q_len=1 with CTA_TILE_Q=64:
    before FP4 grouped min: 0.137760 ms
    after  FP4 grouped min: 0.139584 ms
    FP8 grouped min:        0.133568 ms
    result:                 still slower than FP8

Conclusion:
  NUM_MMA_KV=2 is a keeper for D512 long-q cells. It clears FP8 at q_len=512 and q_len=2048.
  It does not solve q_len=1.

Short-Q tile experiment:
  Changing D512 head_dim>=512 tile selection from CTA_TILE_Q=64 to CTA_TILE_Q=16 for
  avg_packed_qo_len<=16 failed at runtime with cudaErrorInvalidValue. This matches the existing
  warning that the 1-Q-warp/4-KV-warp layout is invalid for some 512-wide shapes.

  CTA_TILE_Q=32 for avg_packed_qo_len<=16 is valid and improves q_len=1 substantially:
    CTA64 FP4 grouped min: 0.137760 ms
    CTA32 FP4 grouped min: 0.091680 ms
    CTA32 FP8 grouped min: 0.091616 ms

  CTA32 gets q_len=1 to parity but not a clean win over FP8. A CTA32 + NUM_MMA_KV=1 probe was
  slightly worse:
    CTA32/MMA1 FP4 grouped min: 0.091712 ms
    CTA32/MMA1 FP8 grouped min: 0.090912 ms

Current keeper candidate:
  - D512/VO512 FP4 max NUM_MMA_KV=2
  - D512 short packed-Q tile policy CTA_TILE_Q=32 for avg_packed_qo_len<=16

Remaining D512 gap:
  q_len=1 remains effectively tied but still fractionally behind FP8 by min-time measurement.
```

D512 production grid after keeper candidates:

```text
Current code under test:
  - D512/VO512 FP4 max NUM_MMA_KV=2
  - D512 head_dim>=512 uses CTA_TILE_Q=32 when avg_packed_qo_len<=16; otherwise CTA_TILE_Q=64

Shape B D512/group8:
  q_len  kv_len   FP4 FA2 ms   FP8 FA2 ms   BF16 ms      FP8/FP4
  1      8192     0.078400     0.072928     1.169536     0.9302
  1      32768    0.092672     0.091776     4.443264     0.9903
  1      131072   0.204448     0.219104     23.597664    1.0717
  512    8192     1.293152     1.300608     1.211264     1.0058
  512    32768    4.845696     4.992224     4.645792     1.0302
  512    131072   19.143007    19.902945    23.416128    1.0397
  2048   8192     5.504608     5.622976     2.824608     1.0215
  2048   32768    23.164448    24.136801    11.946656    1.0420
  2048   131072   93.250175    98.588036    52.994495    1.0572

Conclusion:
  The D512 global prefill cells q_len>=512 now all beat FP8 FA2. The remaining D512 failures are
  q_len=1 at kv=8k and kv=32k. That is the decode/short-prefill region, not the long-prefill region.
```

D512 q_len=1 decode check:

```text
Tensor-core BatchDecode FA2, batch=1, D512/group8:
  kv_len   FP4 FA2 decode ms   FP8 FA2 decode ms   FP8/FP4
  8192     0.082304            0.080736            0.9809
  32768    0.188896            0.202720            1.0732
  131072   0.733248            0.739200            1.0081

XQA NVFP4 decode, batch=1, D512/group8:
  kv_len   FP4 XQA decode ms
  8192     0.078528
  32768    0.127392
  131072   0.378848

Finding:
  D512 q_len=1 should route to XQA for NVFP4. XQA FP8 rejects D512 in this build
  (head_dim range ends at 256), so the practical FP8 comparison is tensor-core FA2 decode.
  NVFP4 XQA beats that FP8 decode baseline for all three D512 kv lengths.

Plumbing fix:
  BatchDecodeWithPagedKVCacheWrapper's tensor-core FA2 plan path was missing the new
  cta_tile_q_override argument added to BatchPrefillWithKVCachePlan. Added the trailing zero
  override so tensor-core decode planning reaches the allocator/kernel path.
```

Shape A D256 sliding status:

```text
FA2 FP4 vs FP8, D256/group2/kv1024:
  q_len=1:
    FP4 FA2 min: 0.026240 ms
    FP8 FA2 min: 0.020352 ms
    FP8/FP4:     0.7756
  q_len=512:
    FP4 FA2 min: 0.034464 ms
    FP8 FA2 min: 0.028448 ms
    FP8/FP4:     0.8254
  q_len=2048:
    FP4 FA2 min: 0.065312 ms
    FP8 FA2 min: 0.051680 ms
    FP8/FP4:     0.7913

FMHA-v2 FP4 with PV-layout V cache is not a Shape A escape hatch:
  q_len=1:    0.252320 ms
  q_len=512:  0.263104 ms
  q_len=2048: 0.261376 ms

XQA decode, D256/group2/kv1024/batch=1:
  FP4 XQA min: 0.031264 ms
  FP8 XQA min: 0.023168 ms
  FP8/FP4:     0.7410

Rejected native-PV probe:
  Expanding the existing native FP4 PV helper from D512/VO256 to D256/VO256 and forcing
  CTA_TILE_Q=32 compiled, but q_len=512 regressed to 7.603072 ms versus the FA2 FP4 baseline
  of ~0.034 ms. Reverted. The D512/VO256 native helper does not transfer to Shape A by predicate
  widening.

Rejected CTA_TILE_Q=32-only probe:
  Forcing D256/VO256 NVFP4 FA2 to CTA_TILE_Q=32 without native PV also regressed q_len=512:
    baseline CTA64 FP4 FA2 min: 0.034464 ms
    CTA32 FP4 FA2 min:          0.063136 ms
    FP8 FA2 min:                0.028448 ms
  Reverted. Shape A prefill should keep the existing CTA64 policy for q_len>=512.

Rejected deeper-KV probe:
  Forcing D256/VO256 NVFP4 FA2 to one CTA/SM with max NUM_MMA_KV=4 also regressed q_len=512:
    baseline FP4 FA2 min: 0.034464 ms
    deep-KV FP4 min:      0.038912 ms
    FP8 FA2 min:          0.028320 ms
  Reverted. The existing two-CTA/SM, NUM_MMA_KV=2 policy is better for the sliding shape.

Conclusion:
  Shape A remains unsolved. The gap is small in absolute time but consistent across q_len. The next
  work needs to target the FA2 generic FP4 D256 path directly, not FMHA-v2, XQA, or the existing
  native-PV helper.
```

Shape A D256 NCU comparison:

```text
Profiled Shape A q512/kv1024/group2, same FA2 kernel grid for FP4 and FP8:
  grid: (128, 1, 1) blocks x (32, 4, 1) threads
  traits FP4: KernelTraits<..., CTA_TILE_Q=64, NUM_MMA_Q=1, NUM_MMA_KV=2,
                           D_QK=16, D_VO=16, NUM_WARPS_Q=4, NUM_WARPS_KV=1,
                           DTypeKV=__nv_fp4x2_e2m1>
  traits FP8: same, DTypeKV=__nv_fp8_e4m3

NCU SpeedOfLight:
  FP4 duration: 22.85 us
  FP8 duration: 15.94 us
  FP4 memory throughput: 39.71 GB/s
  FP8 memory throughput: 69.86 GB/s
  FP4 SM throughput: 18.63%
  FP8 SM throughput: 20.37%
  FP4 eligible warps/scheduler: 0.32
  FP8 eligible warps/scheduler: 0.34
  Local/shared spills: 0 for both

Finding:
  Shape A is not bandwidth-bound. FP4 moves less useful KV data but is slower because the generic
  FP4 path pays scale/dequant/instruction overhead before feeding the same BF16-style MMA path.
  The grid is also too small to fill the GPU (0.34 waves), which explains why absolute time is tiny
  and why launch/tile overheads matter. The next useful work is reducing FP4-specific instruction
  overhead or adding a real native FP4 QK/PV path for the D256 shape; memory-layout-only changes are
  unlikely to close a 17-22% gap.

Reports:
  reports/fa2_d256_shapeA_fp4_q512_kv1024.ncu-rep
  reports/fa2_d256_shapeA_fp8_q512_kv1024.ncu-rep
```

Rejected Shape A BF16-dequant fallback probe:

```text
Tried replacing CUDA 13.0's vec_cast<nv_bfloat16, __nv_fp4x2_e2m1> fallback
(fp4 -> fp16x2 cvt -> fp32 -> bf16x2) with direct integer BF16 bit expansion.
Rationale: E2M1 values are exactly representable in BF16.

Result on Shape A q512/kv1024/group2 FA2:
  baseline FP4 FA2 min: ~0.034464 ms
  integer BF16 expansion FP4 min: 0.078848 ms
  FP8 FA2 min in same run: 0.028576 ms

Rejected and reverted. The integer nibble expansion adds enough per-fragment integer work to
lose badly versus the hardware fp4->fp16 conversion plus float/bf16 conversion chain. Do not retry
this form of direct BF16 bit expansion unless it is implemented with fewer instructions or a native
compiler conversion becomes available.
```

Shape A instruction-delta and planning probes:

```text
Existing q512/kv1024/group2 FA2 NCU reports show same grid but FP4 executes many more
instructions than FP8:
  FP4 executed instructions: 6.54M
  FP8 executed instructions: 4.44M
  FP4 tensor pipe active:    15.9%
  FP8 tensor pipe active:    25.0%
  spills:                   0 for both

Static SASS count from cached kernels:
  FP4 static instructions: 951,944
    F2FP 151,904; HADD2 126,720; LOP3 77,555; SHF 50,112; SHFL 46,464; LDS 46,219
  FP8 static instructions: 654,984
    LOP3 105,741; PRMT 82,784; IMAD 63,611; SHFL 57,984; HMUL2 46,080; F2FP 7,904

Tried native BF16 E2M1 conversion probe with CUDA 13.0 ptxas:
  cvt.rn.bf16x2.e2m1x2 is rejected: "Unexpected instruction types specified for 'cvt'".
So the direct native BF16 conversion path cannot be enabled with the current CUDA 13.0 toolchain.

Added benchmark --dtype to measure FP16 Q/O lower bound. Shape A q512/kv1024/group2 with fp16:
  FP4 FA2 min: 0.032448 ms
  FP8 FA2 min: 0.026720 ms
  FP8/FP4:     0.823x
This improves FP4 slightly versus bf16 (~0.0345 ms), but FP4 still loses. The gap is broader than
BF16 conversion alone.

Rejected disable-split-kv planning probe on Shape A:
  q512:  FP4 0.108960 ms, FP8 0.075744 ms
  q2048: FP4 0.118432 ms, FP8 0.084512 ms
This is far worse than default planning. Keep default split-KV behavior for sliding prefill.
```

Shape A additional rejected probes:

```text
Batch parallelism does not flip the Shape A FA2 result. D256/group2/kv1024/q512:
  batch=1:  FP4 0.035456 ms, FP8 0.028288 ms, FP8/FP4 0.798
  batch=4:  FP4 0.065184 ms, FP8 0.051712 ms, FP8/FP4 0.793
  batch=16: FP4 0.199104 ms, FP8 0.145600 ms, FP8/FP4 0.731
  batch=64: FP4 0.518080 ms, FP8 0.385120 ms, FP8/FP4 0.743
This rules out "too little parallelism in the single-request microbench" as the explanation for
the persistent FP4 gap.

CUDA 13.2 JIT-only native BF16 E2M1 conversion probe:
  A tiny standalone test compiles and runs cvt.rn.bf16x2.e2m1x2 with CUDA 13.2 when targeting
  sm_120a. Shape A q512 then improves slightly:
    FP4 0.032320 ms, FP8 0.028160 ms, FP8/FP4 0.871
  This is useful headroom but still not enough to beat FP8, and vLLM's current runtime stack is
  built against CUDA 13.0. Do not treat CUDA 13.2's conversion instruction as the whole fix.

Fixed split-size probes on Shape A q512/kv1024/group2 all regressed versus default planning:
  split=16 pages:  FP4 0.047008 ms, FP8 0.035616 ms, FP8/FP4 0.758
  split=32 pages:  FP4 0.072032 ms, FP8 0.052800 ms, FP8/FP4 0.733
  split=64 pages:  FP4 0.118560 ms, FP8 0.084512 ms, FP8/FP4 0.713
  split=128 pages: FP4 0.118528 ms, FP8 0.083552 ms, FP8/FP4 0.705
Keep default split-KV planning for the sliding shape.

Rejected typed-scale sidecar probe:
  Implemented a D256-only sidecar that expanded UE4M3 scale bytes to DTypeQ in shared memory once
  per tile, then used those typed values in QK/PV instead of converting scale bytes inside each
  fragment path.
  Results:
    q1:    FP4 0.028160 ms, FP8 0.020224 ms, FP8/FP4 0.718
    q512:  FP4 0.034272 ms, FP8 0.028224 ms, FP8/FP4 0.824
    q2048: FP4 0.073984 ms, FP8 0.051136 ms, FP8/FP4 0.691
  q512 was only noise-level better than baseline, and q2048 regressed materially from the prior
  ~0.065 ms class. Removed. Moving scale conversion from compute to producer increases producer
  work and shared-memory footprint enough that it is not a useful production path for Shape A.

Rejected D256 short-Q CTA32 policy:
  Mirrored the D512 short-Q policy by forcing head_dim=256 and avg_packed_qo_len<=16 to CTA_TILE_Q=32.
  Shape A q1/kv1024/group2 result:
    FP4 0.039456 ms, FP8 0.026016 ms, FP8/FP4 0.659
  This is worse than the prior CTA16 FP4 class (~0.026-0.028 ms). Removed. Shape A should keep the
  default FA2 tile policy: CTA16 for q_len=1 and CTA64 for q_len>=512.

SM120 arch-family check:
  All prior FA2 benchmarks in this work used explicit FLASHINFER_CUDA_ARCH_LIST=12.0a and cached
  under ~/.cache/flashinfer/0.6.9/120a. FlashInfer also supports sm120f flags:
    sm120a: -gencode=arch=compute_120a,code=sm_120a
    sm120f: -gencode=arch=compute_120f,code=sm_120f

  Controlled Shape A q512/kv1024/group2, same source and CUDA 13.0, separate fresh JIT caches:
    120a: FP4 0.033504 ms, FP8 0.029856 ms, FP8/FP4 0.891
    120f: FP4 0.034656 ms, FP8 0.028544 ms, FP8/FP4 0.824
  sm120f compiled and ran, but did not improve FA2 FP4. The known sm120f win for
  cvt.rn.satfinite.e2m1x2.f32 appears relevant to FP32->FP4 quantization/GEMM paths, not this FA2
  attention path, which consumes already-packed FP4 KV and expands it toward BF16-style MMA.

Rejected trtllm-gen backend for SM120:
  Added trtllm-gen as a local benchmark option and tried Shape A q512/kv1024/group2 with NVFP4 KV.
  Runtime failed before benchmarking:
    RuntimeError: TllmGenFmhaRunner ... fmhaRunner.cuh:30: Unsupported architecture
  Source confirms trtllm-gen FMHA currently checks only kSM_100 or kSM_103:
    FLASHINFER_CHECK(mSM == kSM_100 || mSM == kSM_103, "Unsupported architecture");
  So trtllm-gen is not an SM120 production path in this tree. It also warns that NHD NVFP4 KV would
  be converted to HND with extra transpose/copy overhead. Do not rank it against FA2 for Gemma4 on
  RTX PRO 6000 until upstream adds SM120 kernels.
```

FA2 SM120 arch-flag clarification:
  Launcher scripts do not set FLASHINFER_CUDA_ARCH_LIST. In that path, FlashInfer's CompilationContext
  normalizes SM120 to 12.0f when CUDA >= 12.9, so service-time FA2 JIT defaults to:
    -gencode=arch=compute_120f,code=sm_120f

  Earlier benchmark/profiling runs explicitly set FLASHINFER_CUDA_ARCH_LIST=12.0a, so those FA2
  cubins were:
    -gencode=arch=compute_120a,code=sm_120a

  Controlled fresh-cache arch sweep, same source, CUDA 13.0, backend=fa2, fp4_v_layout=nhd:
    Shape A q512/kv1024/group2:
      12.0a: FP4 0.035456 ms, FP8 0.028192 ms, FP8/FP4 0.795
      12.0f: FP4 0.034720 ms, FP8 0.029248 ms, FP8/FP4 0.842
    Shape A q2048/kv1024/group2:
      12.0a: FP4 0.065184 ms, FP8 0.051872 ms, FP8/FP4 0.796
      12.0f: FP4 0.066176 ms, FP8 0.052608 ms, FP8/FP4 0.795
    Shape B q512/kv32768/group8:
      12.0a: FP4 4.835008 ms, FP8 4.997600 ms, FP8/FP4 1.034
      12.0f: FP4 4.858592 ms, FP8 4.997888 ms, FP8/FP4 1.029

  Conclusion: sm120f is the correct default for service JIT and should remain the default unless we
  find a per-kernel regression. However, for FA2 attention on these Gemma4 cells, sm120f does not
  materially improve FP4. The Shape A blocker is still the FP4 FA2 path's extra dequant/scale
  instruction overhead versus FP8 FA2, not the a-vs-f arch flag.

Shape A instruction-class breakdown, sm120f source counters:
  Re-collected matching NCU SourceCounters for D256/group2/kv1024/q512 using the service-default
  sm120f arch target. The FP4 path still executes materially more instructions than FP8 with the
  same grid and same tensor-core work:
    FP4 total source instructions: 6,535,936
    FP8 total source instructions: 4,442,880
    Delta: +2,093,056 instructions (+47.1%)

  FP4 top executed opcodes:
    F2FP   1,299,712  (19.89%)
    HADD2  1,103,872  (16.89%)
    LOP3     606,976
    SHFL     407,680
    HMUL2    401,408
    LDS      309,248

  FP8 top executed opcodes:
    LOP3     865,312
    PRMT     702,464
    SHFL     508,032
    IMAD     481,792
    HMUL2    401,408
    F2FP      45,312

  Largest FP4-FP8 deltas:
    F2FP  +1,254,400
    HADD2 +1,103,872
    LDS     +301,056
    SHF     +122,688
    LEA     +112,000
    IADD3    +94,560

  Interpretation: the Shape A gap is not grid shape, memory bandwidth, or MMA count. The dominant
  extra work is FP4 E2M1/UE4M3 conversion and scale/dequant plumbing around the FA2 path. The
  practical target is reducing conversion/scale instructions per FP4 byte without moving the cost
  into shared-memory pressure or producer overhead.

Rejected half-internal MMA probe:
  Tried using FP16 as the internal FA2 MMA operand type for FP4 KV when Q/O remain BF16, so the FP4
  KV expansion would target half instead of BF16. Shape A q512/kv1024/group2:
    FP4 0.035072 ms
    FP8 0.029184 ms
    FP8/FP4 0.832x
  This is slightly worse than the prior FP4 class. The BF16 Q-to-FP16 conversion and internal type
  changes eat the intended KV conversion savings. Reverted. Do not retry this form unless the
  whole Q/O path is FP16.

Rejected vectorized UE4M3 scale-dequant probe:
  Replaced per-scale `static_cast<DTypeQ>(__nv_fp8_e4m3)` with one packed
  `fast_dequant_f8f16x4<__nv_fp8_e4m3, DTypeQ>` call for K/V scale bytes. This targeted the
  `F2FP.F16.E4M3.UNPACK_B` scale-conversion cluster without changing dispatch.
  Shape A q512/kv1024/group2, sm120f:
    FP4 0.037472 ms
    FP8 0.029472 ms
    FP8/FP4 0.787x
  This regressed versus the prior FP4 class. The packed helper adds enough byte packing/register
  pressure that removing the scalar E4M3 casts is not a win. Reverted.

CUTLASS SM120 block-scaled NVFP4 reference:
  The local CUTLASS tree contains the relevant SM120 reference:
    3rdparty/cutlass/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu

  Important facts from the source:
    ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>
    ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>
    ElementD = cutlass::bfloat16_t
    ArchTag = cutlass::arch::Sm120
    OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp
    ThreadBlockShape = Shape<_128,_128,_128>
    ClusterShape = Shape<_1,_1,_1>
    LayoutSFA/SFB come from CollectiveMainloop::LayoutSFA/LayoutSFB and are generated via
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB(...)

  This is the structural reference for eliminating the FA2 FP4 software path
  {FP4->BF16 conversion + UE4M3 scale conversion + BF16 scale multiply + BF16 MMA}. The caveat is
  that 79a is FP4 x FP4 -> BF16/FP32-accumulator GEMM. In attention QK, Q is BF16 and K is FP4;
  in PV, softmax probabilities are not pre-existing FP4. So using this instruction in FA2 requires
  quantizing the non-KV operand into FP4 with a valid scale layout, then feeding both FP4 operands
  and UE4M3 scales to block-scaled MMA. It is not a drop-in replacement for BF16 x FP4, but it is
  the only path that removes both the F2FP and scale-apply instruction classes.

SM120 block-scaled MMA atom contract:
  Source:
    3rdparty/cutlass/include/cute/arch/mma_sm120.hpp
    3rdparty/cutlass/include/cute/atom/mma_traits_sm120.hpp

  Relevant specialization:
    SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue4m3_t, VS>

  Instruction:
    mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3

  Register contract:
    D/C: float[4]
    A: uint32[4]  for m16 x k64 E2M1
    B: uint32[2]  for k64 x n8  E2M1
    SFA: one UE4M3 scale register
    SFB: one UE4M3 scale register

  Trait layouts:
    Shape_MNK = 16 x 8 x 64
    ALayout maps (T32,V32) -> (M16,K64)
    BLayout maps (T32,V16) -> (M16,K64) in the atom trait's register view
    SFALayout maps (T32,V64) -> (M16,K64)
    SFBLayout maps (T32,V64) -> (N8,K64)

  Consequence for FA2:
    QK needs BF16-Q -> FP4-Q quantization before the atom can be used.
    PV needs softmax-probability -> FP4-P quantization before the atom can be used.
    The current native-PV helpers and layout probes are relevant infrastructure, but Shape A needs
    a native QK path too if we want to remove the dominant F2FP/HADD2/HMUL classes.

Unfused CUTLASS block-scaled attention probe, Shape A:
  Command:
    benchmarks/bench_nvfp4_native_attention_gemm.py
      --m 512 --n 1024 --d 256 --warmup 2 --repeat 5 --device 0

  Result:
    QK FP4 CUTLASS min:                    0.017152 ms
    QK torch BF16 min:                     0.011872 ms
    softmax min:                           0.007584 ms
    P quantize min:                        0.018624 ms
    fused softmax+P-quantize min:          0.018336 ms
    PV FP4 CUTLASS min:                    0.021152 ms
    two-stage FP4 fused-softmax-quant min: 0.047552 ms
    two-stage FP4 unfused min:             0.052448 ms
    torch BF16 unfused attention min:      0.032864 ms
    attention cosine vs BF16 reference:    0.980469
    attention mean abs error:              0.00766
    attention max abs error:               0.06299

  Interpretation:
    This confirms the CUTLASS block-scaled GEMM machinery runs for the attention dtype pattern, but
    the unfused path is not a Shape A performance solution. On this small sliding shape, native FP4
    QK GEMM is slower than torch BF16 GEMM, and the separate softmax/P-quant/PV launches are slower
    than both FA2 FP4 and FA2 FP8. The useful output is not a dispatch candidate; it is evidence that
    Q/P quantization must be fused inside FA2 to have any chance of beating FP8.

Native SM120 FP4 MMA atom smoke:
  Command:
    nvcc -std=c++17 -O3 --expt-relaxed-constexpr --expt-extended-lambda
      -gencode=arch=compute_120f,code=sm_120f
      -Iinclude -I3rdparty/cutlass/include
      benchmarks/bench_sm120_nvfp4_mma_dynamic_probe.cu

  Results:
    pattern=4, blocks=4096, iters=4096: 0.384704 ms, 1429.04 atom TFLOP/s
    pattern=2, blocks=4096, iters=4096: 0.354976 ms, 1548.71 atom TFLOP/s

  Interpretation:
    The raw block-scaled FP4 MMA atom is healthy and fast on sm120f. The production gap is not the
    tensor-core instruction itself; it is the fused-attention integration cost: quantizing the
    non-KV operand, moving/owning scale registers, preserving online-softmax normalization, and
    avoiding extra conversion/scale instructions.

Native QK fragment probe:
  Added:
    benchmarks/bench_sm120_nvfp4_qk_fragment_probe.cu

  Purpose:
    Isolate the proposed FA2 QK structural change: quantize Q to E2M1 + UE4M3 scale registers,
    consume existing FP4 K through the SM120 block-scaled MMA atom, and compare one D=256
    m16n16 tile against references.

  Layout finding:
    The first attempt used the older hand-derived register mapping and was wrong. The correct
    CUTE atom contract for this wrapper is:
      A value layout:   linear = k * 16 + row
      B value layout:   linear = k * 8 + col
      SFA scale layout: linear = kg * 16 + row
      SFB scale layout: linear = kg * 8 + col
    with kg = k / 16. After switching the probe to those layouts, native QK output matches its
    own dequantized FP4 reference exactly:
      qref_max_abs: 0
      qref_mean_abs: 0
      qref_cosine: 1.0

  Correctness versus original FP32 synthetic tile:
    Six-candidate scale search:
      ref_max_abs: 4.67651
      ref_mean_abs: 1.28181
      ref_cosine: 0.949199
    Fast max_abs/6 scale:
      ref_max_abs: 3.59039 on the earlier modulo-valued pattern
      ref_mean_abs: 1.29734
      ref_cosine: 0.957923

    With a cheaper bitwise synthetic pattern used for timing, the reference cosine is lower
    (~0.808). That pattern is intentionally not a model-distribution proxy; it is only for avoiding
    modulo dominating the timing loop.

  Timing probes with fast max_abs/6 scale and cheap synthetic values:
    bench_blocks=4096, bench_iters=16, prepacked K:
      0.774752 ms, 11.09 TFLOP/s
    bench_blocks=4096, bench_iters=64, prepacked K:
      3.044864 ms, 11.28 TFLOP/s
    bench_blocks=4096, bench_iters=16, recompute K quantization too:
      1.568672 ms, 5.48 TFLOP/s

  Interpretation:
    The native QK register math is now correct, but the naive runtime Q quantization path is far
    too expensive to drop directly into FA2. Even with prepacked K, the probe is far below the
    current fused FA2 Shape A effective rate (~31 TFLOP/s) and far below the raw atom rate. This
    does not reject block-scaled QK globally; it rejects a naive per-fragment Q scale/pack loop as
    the production implementation. A viable path would need a much more optimized Q-quant producer
    or a larger CUTLASS-style tiled mainloop that amortizes scale computation, not scalar-ish
    per-fragment helper code.

CUTLASS block-scaled GEMM path by production shape:
  Shape A-like QK, m=2048, n=1024, d=256:
    FP4 CUTLASS QK min:  0.016896 ms, 63.55 TFLOP/s
    torch BF16 QK min:   0.015520 ms, 69.18 TFLOP/s
    cosine vs BF16 ref:  0.992188
    Interpretation: the CUTLASS block-scaled GEMM path does not beat BF16 on the small sliding
    Shape A QK surface. This matches the FA2 result: Shape A is dominated by overhead and small-N
    scheduling, not raw FP4 MMA throughput.

  Shape B-like QK, m=512, n=32768, d=512:
    FP4 CUTLASS QK min:  0.036320 ms, 473.01 TFLOP/s
    torch BF16 QK min:   0.072992 ms, 235.37 TFLOP/s
    cosine vs BF16 ref:  0.988281
    Interpretation: on the long global-attention shape, the CUTLASS block-scaled QK path is a real
    2x QK win. This supports the current Shape B direction and explains why D512/global is the
    surface where FP4 can beat FP8/BF16 first.

  Shape B-like full two-stage unfused attention, m=512, n=32768, d=512:
    QK FP4 min:                         0.037216 ms
    softmax min:                        0.031264 ms
    fused softmax+P-quantize min:       0.042208 ms
    PV FP4 after fused P min:           0.067392 ms
    two-stage FP4 fused-softmax min:    0.125312 ms
    torch BF16 unfused attention min:   0.201312 ms
    attention cosine vs BF16 reference: 0.980469

  Interpretation:
    The CUTLASS 79a-style block-scaled path is strong for large Shape B matrices, but not for Shape
    A's small sliding KV. For Shape A, trying to force native block-scaled QK through a scalar
    fragment helper is going in the wrong direction. For Shape B, the existing FA2/FMHA-v2 wins are
    plausible because the problem size amortizes Q/P quantization and FP4 MMA setup.

CUDA 13.2 target status after driver update:
  Host driver after reboot:
    NVIDIA-SMI: 595.58.03
    reported CUDA capability: 13.2
    target GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, sm120
    CUDA toolkit used for JIT: /usr/local/cuda-13.2, nvcc 13.2.51

  PyTorch remains a cu130 wheel:
    torch: 2.11.0+cu130
    torch.version.cuda: 13.0

  Interpretation:
    The previous CUDA 13.2 PTX launch blocker was driver-side. With the updated driver, CUDA 13.2 is
    now usable as the FlashInfer JIT/compiler target even though the Python environment still uses a
    cu130 PyTorch wheel.

  Fresh-cache Shape A smoke with CUDA 13.2:
    Command shape:
      --gemma4-shape sliding --q-len 512 --kv-len 1024 --fp4-backend fa2
      --fp4-v-layout nhd --warmup 3 --repeat 10

    Results:
      FP4 FA2 min:  0.031072 ms, 34.56 TFLOP/s
      FP8 FA2 min:  0.027232 ms, 39.43 TFLOP/s
      BF16 min:     0.164064 ms, 6.54 TFLOP/s
      FP4 vs FP8:   0.88x
      FP4 vs BF16:  5.28x

  Conclusion:
    CUDA 13.2 is now the right default target for this work. The direct scale-conversion path is
    valid and slightly helpful, but it does not solve Shape A. The remaining Shape A gap is still in
    FA2's FP4-specific instruction overhead and scheduling around the small sliding KV surface, not
    in driver/toolkit availability.

  Full Shape A q-length sweep, CUDA 13.2:
    Backend: FA2, FP4 V layout: NHD, kv_len=1024, group=2

    q_len=1:
      FP4 FA2 min:  0.025216 ms
      FP8 FA2 min:  0.021536 ms
      BF16 min:     0.159648 ms
      FP4 vs FP8:   0.85x

    q_len=512:
      FP4 FA2 min:  0.031072 ms
      FP8 FA2 min:  0.027232 ms
      BF16 min:     0.164064 ms
      FP4 vs FP8:   0.88x

    q_len=2048:
      FP4 FA2 min:  0.059616 ms
      FP8 FA2 min:  0.051712 ms
      BF16 min:     0.161120 ms
      FP4 vs FP8:   0.87x

  Interpretation:
    The Shape A miss is stable across decode-like and prefill-like q lengths. CUDA 13.2 moves the
    path in the right direction, but the remaining target is still roughly a 14-18% wall-time
    reduction against FP8 FA2 on the small sliding KV shape.

  SASS confirmation, Shape A q512 cached FA2 paged kernel:
    The CUDA 13.2 JIT output contains the direct E2M1-to-BF16 instruction:
      F2FP.BF16.E2M1.UNPACK_B

    Static SASS line counts for FP4 vs FP8 paged mask_0 object:
      FP4:
        F2FP.BF16.E2M1: 5760
        F2FP.BF16.E4M3: 2880
        HMUL2.BF16:     5760
        HMMA.BF16:      2925
        LDG lines:      2746
        LDS lines:      6678

      FP8:
        F2FP.BF16.E2M1: 0
        F2FP.BF16.E4M3: 0
        HMUL2.BF16:     5760
        HMMA.BF16:      2925
        LDG lines:      1686
        LDS lines:      2364

  Interpretation:
    CUDA 13.2 removes the old multi-instruction E2M1->FP16->BF16 conversion fallback, but FP4 still
    carries substantially more conversion and load-side instruction pressure than FP8 with the same
    BF16 MMA count. The next Shape A work should target the FP4 sidecar/layout/load path, not the
    tensor-core MMA count.

Rejected Shape A V-scale col-major smem sidecar:
  Hypothesis:
    Shape A PV consumes V scale bytes as row pairs at the same scale column. The existing row-major
    V-scale smem layout makes those reads strided by SF_COLS. A D256-only col-major V-scale smem
    layout would make the consumer's r/r+1 scale pair contiguous and could reduce LDS pressure.

  Implementation tested:
    Only D256 FP4 V scales were remapped to col-major-per-KV-warp in smem. D512/global and K scales
    kept the previous layout. The compute_sfm_v consumer used stride-1 row-pair reads for V scales.

  Shape A q512/kv1024 result, CUDA 13.2:
    Baseline prior: FP4 0.031072 ms, FP8 0.027232 ms, FP4/FP8 0.88x
    Col-major V SF: FP4 0.032288 ms, FP8 0.028096 ms, FP4/FP8 0.87x

  Conclusion:
    Rejected and reverted. The consumer-side locality improvement was outweighed by extra scalar
    staging/store work. Keep the existing packed 32-bit V-scale staging layout.

Rejected Shape A paged V-scale row-quartet producer:
  Hypothesis:
    The paged V-scale global layout is contiguous for four adjacent KV rows at the same scale
    column. The current V producer loads one row and four scale columns, which maps to strided
    global bytes. A D256-only producer that loads row quartets as one 32-bit global word and
    scatters back to the existing row-major smem layout might reduce the FP4 LDG delta without
    changing the compute_sfm_v consumer.

  Shape A q512/kv1024 result, CUDA 13.2:
    Baseline prior:     FP4 0.031072 ms, FP8 0.027232 ms, FP4/FP8 0.88x
    Row-quartet V SF:   FP4 0.032608 ms, FP8 0.028640 ms, FP4/FP8 0.88x

  Conclusion:
    Rejected and reverted. The reduced global-load pattern was outweighed by scatter/addressing
    overhead. For Shape A, small extra producer-side instruction count is enough to erase any
    bandwidth-side win from scale-sidecar rearrangement.

NCU Shape A q512 CUDA 13.2, FP4 vs FP8 FA2:
  Kernel: BatchPrefillWithPagedKVCacheKernel
  Shape: D=256, group=2, q_len=512, kv_len=1024

  Summary:
    FP4 duration:                  18.14 us
    FP8 duration:                  15.17 us
    FP4 executed instructions:     5,027,232
    FP8 executed instructions:     4,459,776
    FP4 registers/thread:          251
    FP8 registers/thread:          252
    FP4 dynamic smem/block:        50.18 KiB
    FP8 dynamic smem/block:        49.18 KiB
    Grid size:                     128 blocks for both
    Waves per SM:                  0.34 for both
    Achieved occupancy:            FP4 8.34%, FP8 8.40%
    FP4 excessive global sectors:  20,544
    FP8 excessive global sectors:  128
    FP4 excessive shared waves:    57,728
    FP8 excessive shared waves:    448

  Interpretation:
    The remaining Shape A gap is not occupancy, register count, or tensor-core count; those are
    effectively the same as FP8. FP4 has ~13% more instructions and much worse coalescing. However,
    the two direct sidecar-layout/staging probes both regressed, so the current FA2 row-major
    staging is near a local optimum for this structure.

Rejected Shape A CTA_TILE_Q=32 retest under CUDA 13.2:
  Hypothesis:
    NCU reports only 128 blocks for 188 SMs at q512/CTA_TILE_Q=64. Reducing CTA_TILE_Q to 32 would
    double the block count and might offset FP4's extra instruction pressure.

  Result:
    Baseline prior: FP4 0.031072 ms, FP8 0.027232 ms, FP4/FP8 0.88x
    CTA_TILE_Q=32:  FP4 0.052480 ms, FP8 0.029152 ms, FP4/FP8 0.56x

  Conclusion:
    Rejected and reverted. The grid-underfill warning is real, but smaller Q tiles add far too much
    FP4 overhead. Keep CTA_TILE_Q=64 for Shape A q_len > 16.

Shape B prefill grid, CUDA 13.2, FA2 FP4 vs FP8:
  Shape:
    Gemma4 global attention, D=512, group=8, q_len in {512, 2048}, kv_len in {8192, 32768, 131072}.

  Results:
    q512 kv8192:
      FP4 1.2808 ms, FP8 1.3071 ms, BF16 1.2196 ms
      FP4/FP8 1.02x, FP4/BF16 0.95x

    q512 kv32768:
      FP4 4.8984 ms, FP8 5.0289 ms, BF16 4.6385 ms
      FP4/FP8 1.03x, FP4/BF16 0.95x

    q512 kv131072:
      FP4 19.3506 ms, FP8 19.9987 ms, BF16 23.4109 ms
      FP4/FP8 1.03x, FP4/BF16 1.21x

    q2048 kv8192:
      FP4 5.4773 ms, FP8 5.6827 ms, BF16 2.8508 ms
      FP4/FP8 1.04x, FP4/BF16 0.52x

    q2048 kv32768:
      FP4 23.3743 ms, FP8 24.3018 ms, BF16 12.0397 ms
      FP4/FP8 1.04x, FP4/BF16 0.52x

    q2048 kv131072:
      FP4 94.3920 ms, FP8 99.2323 ms, BF16 53.3185 ms
      FP4/FP8 1.05x, FP4/BF16 0.56x

  Interpretation:
    With CUDA 13.2 and the D512 NUM_MMA_KV=2 policy, Shape B FA2 FP4 beats FP8 FA2 for the
    measured prefill cells. The q2048 cells still trail BF16 badly, so the FP4-vs-FP8 success
    condition is met there, but the fused FA2 quantized-KV path is not capturing the standalone
    CUTLASS block-scaled FP4 GEMM headroom for large-Q prefill.

Rejected Shape A FP4 data cp.async 16-byte destination probe:
  Hypothesis:
    Current FA2 FP4 data staging emits LDGSTS.E.64 for packed FP4 K/V data, and NCU attributes all
    Shape A shared-memory excess to those 64-bit async copies. Changing the helper to issue a
    16-byte destination copy with src-size=8 would zero-fill the high half explicitly and might move
    the kernel onto the same low-conflict LDGSTS.128-style path used by FP8.

  Correctness:
    Passed targeted D256 FA2 prefill correctness:
      test_nvfp4_paged_prefill_sm12x[256-1024-q_lens1-last_page_lens1]

  Shape A q512/kv1024 result, CUDA 13.2:
    Baseline prior:        FP4 0.031072 ms, FP8 0.027232 ms, FP4/FP8 0.88x
    16-byte destination:   FP4 0.032736 ms, FP8 0.027840 ms, FP4/FP8 0.85x

  Conclusion:
    Rejected and reverted. The LDGSTS.E.64 excess is visible in NCU, but doubling the shared
    destination footprint for packed FP4 data is worse on this shape. The 64-bit data staging is
    likely a necessary local compromise unless the whole FP4 smem layout/consumer path is changed.

Shape A explicit linear V-scale sidecar layout, CUDA 13.2:
  Hypothesis:
    The remaining FP4-vs-FP8 delta includes scalar global byte loads for the V scale sidecar in the
    TRT-LLM-interleaved scale layout. Quantizing V scales into a simple linear
    [page, head, kv_entry, scale_col] layout lets the FA2 producer load each row's four scale bytes
    as one 32-bit cp.async instead of deinterleaving scalar bytes from the TRT-LLM layout.

  Correctness:
    Passed targeted D256 FA2 smoke against a PyTorch reference:
      mean 6.3889e-05, p99 4.8828e-04, max 4.8828e-04

  Shape A results, D=256/group=2/kv1024:
    q1:
      FP4 0.024480 ms, FP8 0.021952 ms, BF16 0.159872 ms
      FP4/FP8 0.90x

    q512:
      FP4 0.030560 ms, FP8 0.028192 ms, BF16 0.164672 ms
      FP4/FP8 0.92x

    q2048:
      FP4 0.059008 ms, FP8 0.051840 ms, BF16 0.162656 ms
      FP4/FP8 0.88x

  Interpretation:
    This is a real but small improvement over the default interleaved V-scale layout; it does not
    close Shape A. Keep the implementation as an explicit FA2-only layout option while profiling
    whether the scalar V-scale global-load excess is actually removed or only shifted into the
    32-bit async sidecar path.

  NCU q512:
    Duration:                 16.93 us
    Executed instructions:    4,688,864
    L2 global excess:         0
    L1 shared excess:         57,728

    The linear layout removes the global coalescing excess from the scalar V-scale sidecar and
    removes ~338K executed instructions versus the default interleaved FP4 path. The remaining gap
    is now dominated by unchanged shared-memory excess from packed FP4 data staging, primarily
    LDGSTS.E.64.

Rejected Shape A FP4 64-bit cp.async L2-prefetch hint:
  Hypothesis:
    The packed FP4 data path uses 64-bit cp.async and ignores the helper's prefetch mode, while
    the 128-bit FP8 path uses an L2::128B hint. Adding the same hint to 64-bit FP4 loads might
    improve the LDGSTS data path without changing layout.

  Shape A q512/kv1024 result with linear V-scale layout:
    Baseline linear V SF:     FP4 0.030560 ms
    64-bit L2 prefetch hint:  FP4 0.031872 ms

  Conclusion:
    Rejected and reverted. The hint compiles, but the 64-bit FP4 data path gets slower. The
    remaining LDGSTS.E.64 issue is not fixed by cache-prefetch policy; it needs a structural smem
    layout/consumer change or a different MMA path.

Rejected Shape A FP4 D256 k64B KV-smem swizzle:
  Hypothesis:
    Shape A uses 64-bit packed FP4 data copies into the k128B KV-smem swizzle. The existing k64B
    swizzle path might reduce the LDGSTS.E.64 shared-memory store conflict if enabled for D256.

  Correctness:
    After widening the k64B stride compile guard, the targeted D256 paged-prefill test passed:
      test_nvfp4_paged_prefill_sm12x[256-1024-q_lens1-last_page_lens1]

  Shape A q512/kv1024 result with linear V-scale layout:
    Baseline linear V SF:  FP4 0.030560 ms
    D256 k64B swizzle:     FP4 0.032736 ms

  Conclusion:
    Rejected and reverted. The k64B swizzle is correct for D256 after guard widening, but slower.
    The default k128B layout is still the best measured KV-smem layout for Shape A.
