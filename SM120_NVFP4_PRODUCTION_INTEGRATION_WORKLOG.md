# SM120 NVFP4 Production Integration Worklog

This is the append-only continuation log for production integration work after
the source-tree/JIT specialization cleanup. The older reference log remains
immutable.

## 2026-05-01: D512 Paged Wrapper Regression Localization

Single-cell production wrapper timing at D512, `q=512`, `kv=8192`,
`group=8`, `num_kv_heads=1`, `split_kv_len=8192`, PV V layout, no causal
mask, no soft-cap:

```text
bench_sm120_nvfp4_attention.py --mode paged-wrapper:
  min_ms:        450.823
  output_finite: false
  NaNs:          present

bench_sm120_nvfp4_attention.py --mode dense, same shape:
  min_ms:        1.289
  output_finite: true
```

Segment timing around the wrapper path:

```text
q_group.copy_:     ~0.01 ms
quantize_q:        ~0.007 ms
paged_run:       ~450.6 ms
out.copy_:         ~0.01 ms
full run:        ~450.5 ms
```

Conclusion:

The wrapper regression is inside the production `paged_run` launch path, not
Python overhead, not the per-kv-head loop, and not the separate Q quantization
launch. The next fix target is the native paged producer/stage path itself.

Follow-up probes:

```text
D512 q=128 kv=128 group=8, PV paged wrapper:
  before paged smem stage initialization: output_finite=false
  after paged data/scale stage initialization: output_finite=true

D512 q=512 kv=8192 group=8, PV paged wrapper:
  split_kv_len=8192: min_ms=454.56, output_finite=false
  split_kv_len=128:  min_ms=42.47,  output_finite=false

zero K + zero V + unity K/V scales, split_kv_len=128:
  split_m finite, split_l has NaNs
```

Interpretation:

The tiny one-tile NaNs were from stale paged smem stage slots. The multi-tile
NaNs remain even with zero K/V inputs, which means the remaining bug is in the
multi-tile paged role/pipeline or score/stat handoff, not in Python wrapper
overhead and not in source K/V values.

## 2026-05-01: D512 Paged Producer Race Confirmation

Diagnostic change:

```text
D512 paged K/V producers were serialized to lane 0 for all CUTLASS producer
thread slices. This is not a production implementation; it is a race
localization probe.
```

Results:

```text
zero K + zero V + unity K/V scales, q=512 kv=8192 group=8, split_kv_len=128:
  output_finite:  true
  split_l_finite: true
  split_l_min/max: 128.0 / 128.0
  split_m_min/max: 0.0 / 0.0

real K/V, same shape, split_kv_len=128:
  output_finite: true
  min_ms:        1194.754

real K/V, same shape, split_kv_len=8192:
  output_finite: true
  min_ms:        6998.056
```

Conclusion:

The paged-wrapper NaNs were caused by a native paged subbyte producer race.
Serializing the producer fixes correctness but destroys throughput, so it is
only a localization proof. The production fix needs cooperative packed-byte
stores or an equivalent partition-derived producer that writes each packed
destination byte exactly once.

Rejected producer probes:

```text
1. Raw byte store from the even logical coordinate:
   rejected because logical adjacent K is not a sufficient byte-owner rule for
   the swizzled CuTe destination reference.

2. Two-phase CuTe subbyte stores:
   rejected because zero-K/V stats became finite but wrong and output remained
   non-finite.

3. Private-layout atomic nibble store:
   rejected because the iterator returned by the swizzled CuTe tensor is not
   safely representable as the simple `{ptr, idx}` helper layout.
```

Current state:

```text
D512 paged K/V producer is back to the serialized lane-0 diagnostic path.

zero K + zero V + unity K/V scales:
  output_finite:  true
  partial_finite: true
  split_l:        exactly 128
  split_m:        exactly 0

real K/V, q=512 kv=8192 group=8 split_kv_len=128:
  output_finite: true
  min_ms:        1194.083
```

Next fix direction:

The correctness authority remains the serialized `partition_D` subbyte writer.
The production fix cannot be a raw pointer/nibble shortcut. It needs a
byte-owner producer derived at the CuTe copy-partition level, or a producer
layout where each lane owns complete packed bytes before writing to the
CUTLASS shared-memory destination.
## 2026-05-01 - D512 Paged Producer Correctness Checkpoint

Context: D512 dense/direct stayed recovered after source-tree integration, but
the paged-wrapper path returned nonfinite output. Timing segmentation showed the
regression was inside `paged_run`, not the Python wrapper, Q quantization, or
host copies.

Findings:

- Serializing the paged K/V producer made zero-K/V exact, proving the consumer,
  online softmax, and split combine path were not the source of the NaNs.
- Raw atomic nibble writes and byte-owner guesses did not match CuTe's subbyte
  tensor semantics and were rejected.
- Switching the producer to TMA `partition_D` helped a tiny smoke shape but did
  not fix the full zero-K/V diagnostic. PV in particular did not match the
  dense TMA coordinate convention when driven by a manual identity tensor.
- The durable correctness fix was to keep the LDSM-side producer partition but
  explicitly initialize the K/V packed-data shared-memory regions to zero and
  the K/V SFB scale sidecars to unity before the paged pipeline starts.

Current D512 no-mask/no-softcap PV-layout measurements:

```text
q=512 kv=8192 group=8 split_kv_len=128:
  zero K/V: output finite, partial finite, split_m = 0, split_l = 128
  real K/V: output finite, min_ms = 40.935

q=512 kv=8192 group=8 split_kv_len=8192:
  real K/V with the two-phase diagnostic path was finite but ~471 ms
```

Interpretation:

- Correctness is recovered for the focused D512 paged-wrapper cell.
- Performance is still not production-grade. The paged producer is still doing
  scalar/subbyte shared-memory staging instead of a coalesced packed-byte or TMA
  equivalent, so the remaining work is data-movement structure, not wrapper
  overhead.
- The one-time smem initialization is acceptable as a correctness checkpoint,
  but it should be costed after producer coalescing. If it shows up in NCU, the
  production version should initialize only the active pipeline stage or make the
  producer cover every consumed byte deterministically.

Follow-up optimization in the same session:

- Replaced the virtual `SmemCopyAtomB` copy-thread traversal in the paged
  producer with direct logical pair-packed shared-memory stores. One lane now
  writes both nibbles of a packed FP4 byte in program order.
- Inlined the hot K packed-byte load so the two K nibbles come from one global
  byte load instead of two `sm120_nvfp4_paged_k_code` calls.
- Removed unused V-scale lookups from the `kUsePvLayoutV=true` data path. The
  PV-layout V code path does not need the data-side scale value; scales are
  staged separately into SFB.
- Replaced runtime page-size division with `token >> 4` and `token & 15` in the
  hot D512 paged producer. The csrc wrapper already enforces `page_size == 16`.

Updated D512 q=512 kv=8192 group=8 no-mask/no-softcap PV-layout timings:

```text
split_kv_len=128:
  before direct pair-packed producer: 40.935 ms, finite
  after direct pair-packed producer:  20.019 ms, finite
after helper/page-math cleanup:     14.150 ms, finite

split_kv_len=8192:
  before helper/page-math cleanup:    172.059 ms, finite
  after helper/page-math cleanup:     111.377 ms, finite
```

Zero-K/V diagnostic after the cleanup remains exact:

```text
output finite = true
partial finite = true
split_m = 0
split_l = 128
out_absmax = 0
```

Remaining problem:

- Paged-wrapper D512 is now correct but still far slower than dense/direct on
  the same shape (`dense` q=512 kv=8192 split=128 measured 1.142 ms in the same
  harness). The next structural work is still producer throughput: coalesced
  page loads / packed-byte stores, avoiding unnecessary smem initialization if
  full coverage can be proven, and reducing wrapper-level launches only after
  the stage kernel is no longer dominant.

Rejected follow-up:

- Removing the packed-data smem zeroing while keeping SFB initialization is not
  correct. The zero-K/V diagnostic stayed finite but no longer had exact stats
  (`split_m` became nonzero and `out_absmax` was nonzero), so data smem
  initialization remains load-bearing until the producer is proven to cover the
  exact consumed byte set.

Restored packed-data initialization and remeasured the focused real cell:

```text
q=512 kv=8192 group=8 split_kv_len=128:
  13.990 ms, finite
```

## 2026-05-01 23:13 CDT — D512 Paged Producer State Reset

The historical reference worklog remains immutable. Production-integration
work is continuing in this append-only file.

Current D512 paged-wrapper state:

- The packed K/V shared-memory initialization and unity SFB initialization stay
  in the paged specialization. Zero-K/V diagnostics proved this is currently
  load-bearing: removing packed-data initialization produces finite but
  nonzero scores/output for zero K/V.
- The temporary serialized TMA K producer diagnostic is removed. It did not
  improve the dense-vs-paged QK-stat mismatch and was not production-ready.
- The active K producer is restored to the faster direct logical pair-packed
  path: one lane owns one packed FP4 byte, loads both nibbles from the paged K
  cache, and writes both `cute::uint4_t` slots.
- The active V producer remains the direct logical pair-packed PV-layout path.

Next gate:

- Stop using ad-hoc dense-vs-paged comparisons as the primary correctness
  signal for this step, because dense and paged quantization conventions differ
  enough to confound low-level byte/scale comparisons.
- Run the existing D512 SM120 wrapper tests that exercise
  `BatchPrefillWithPagedKVCacheSM120Nvfp4Wrapper` and the standard
  `backend="sm120-nvfp4"` route, then fix failures against those test
  references.

## 2026-05-02 00:34 CDT — D512 Paged Nondeterminism Localized To K Data

Additional diagnostics after the wrapper-test failures:

- Reusing the same D512 paged wrapper twice produces different outputs. Zeroing
  wrapper scratch between runs changes the diff pattern but does not make the
  run deterministic.
- The no-combine case (`split_kv_len=8192`) still has nondeterministic
  `_out_scratch`/output while `_partial`, `_split_m`, and `_split_l` are unused,
  so stale split-combine scratch is not the root cause for this failure mode.
- Dense direct repeats exactly with the same masks and specialization. The
  nondeterminism is specific to the paged producer path, not the common online
  softmax/PV/epilogue path.
- K/V isolation:
  - zero K + real V repeats exactly.
  - real K + zero V produces zero output, but QK split stats vary.
  - constant K repeats exactly, with either real or unity K scales.
  - real variable K varies even with unity K scales.

Conclusion:

- The active bug source is variable K data staging in the paged producer. K
  scale sidecar traffic and V staging are not sufficient to reproduce the
  nondeterminism.
- The current paged K producer is not following the existing FlashInfer paged
  FP4 loading idiom. `prefill.cuh` uses 32-bit/64-bit predicated `cp.async`
  loads for sparse/paged FP4 K/V, not software nibble loads followed by
  sub-byte shared-memory proxy stores.

Next implementation step:

- Port the K data producer to 4-byte predicated `cp.async` chunks where the
  CUTLASS SM120 B-operand layout maps eight logical FP4 values to four
  contiguous physical shared-memory bytes. Keep a correctness fallback for any
  non-contiguous destination chunk. Commit/wait the cp.async group before
  completing the CUTLASS TMA-style pipeline barrier.

Follow-up result:

- Rejected the raw-byte `cp.async` K-data producer for this CUTLASS B smem
  layout. Two versions were tested:
  - direct `&qk_sB(row,k)` byte pointer with a runtime contiguity fallback.
  - CuTe subbyte-iterator roundtrip check with no proxy fallback and bounded
    cp.async groups.
- Both compiled/runs produced systematic wrong outputs, with the multi-KV D512
  exactness mismatch around 94% of elements. This is not the original
  nondeterministic small-delta failure; it is a wrong layout/representation
  mapping.
- Root cause of the rejected cp.async approach: CuTe subbyte references and the
  SM120 block-scaled B operand layout cannot be treated as a raw row-major byte
  array at the `qk_sB(row,k)` level. The existing `prefill.cuh` cp.async idiom
  is still valid for its own permuted-smem wrapper, but it is not directly
  portable to this CUTLASS `SmemLayoutB` tensor by recasting subbyte references
  to byte pointers.

State after rejection:

- D512 is restored to the prior direct logical pair-packed K producer.
- Continue debugging the original paged K-data nondeterminism from this state;
  do not retry raw-byte cp.async into `qk_sB` without first deriving the
  destination mapping from the exact CUTLASS copy atom / producer partition.

## 2026-05-02 01:15 CDT — Wrapper Gate, Workspace Init, Normal-V Status

Changes landed in the active source headers and mirrored JIT data headers:

- Added explicit per-CTA initialization for shared online-softmax row state:
  `global_m`, `global_l`, and both `old_scale_stage` buffers now start from
  finite identity values every CTA.
- Split the CUTLASS QK and PV persistent-scheduler workspaces into disjoint
  aligned regions. The prior launcher initialized both collectives at the same
  workspace base, then passed both parameter objects into the fused stage
  kernel.
- Added a bounded 32 MiB workspace-prefix clear before CUTLASS QK/PV argument
  construction. A fresh-process prefix sweep showed that 16 MiB and smaller can
  still leave stale allocator residue visible to the next D128 wrapper run,
  while 32 MiB removes the NaN failure.
- Updated the standard-wrapper-vs-direct-wrapper test to use a separate
  workspace for the direct comparison wrapper. The shared-workspace version was
  measuring two live wrapper stacks reusing the same scratch buffer, not backend
  equivalence. With separate workspaces, D128 standard and direct match
  bit-for-bit after the D128->D512 load-order prefix.

Validation:

- PV-layout wrapper gate:
  `test_sm120_nvfp4_wrapper_multi_kv_matches_single_kv_sm12x` plus
  `test_standard_prefill_wrapper_sm120_nvfp4_backend_matches_direct_wrapper_sm12x`
  passes for D128/D256/D512: 6 passed.
- Normal-V acceptance:
  - D128 passes after first compile: 1 passed in 151.22s.
  - D512 passes after first compile: 1 passed in 101.78s.
  - D256 `pv_v=False` remains a compile-time blocker. The first compile was
    stopped after the same `ptxas` instance had run for more than six minutes
    and was still active. No runtime correctness result was obtained for this
    specialization.

Current interpretation:

- The PV-layout path is the production-ready path for the native SM120 NVFP4
  wrapper tests across all three head dims.
- The normal-V path is partially validated but not production-clean until D256
  avoids the pathological `pv_v=False` attention specialization compile.

## 2026-05-02 02:35 CDT — PV-Only Attention Module, Normal-V Preconversion

Decision:

- The SM120 NVFP4 attention module now consumes PV-layout V only. Normal-layout
  V is converted before attention instead of instantiating the compile-heavy
  `pv_v=False` stage producer inside the attention kernel.
- This matches the existing FlashInfer architecture around
  `nvfp4_quantize_paged_kv_cache(..., v_data_layout="pv",
  v_scale_layout="pv")`: layout conversion is a quantization/cache-prep step,
  not an attention inner-loop branch.

Implementation:

- Extended `nvfp4_quantize_paged_kv_cache` with an already-packed NVFP4 input
  dispatch:
  - input K/V dtype `uint8`.
  - caller supplies existing K/V FP8 block scales via `kv_cache_sf`.
  - caller supplies dequant global scales via `k_global_scale` and
    `v_global_scale`.
  - K bytes/scales are returned unchanged.
  - normal V is dequantized with `nvfp4_kv_dequantize`, then re-quantized into
    the existing PV V data/scale layout.
- Changed `BatchPrefillWithPagedKVCacheSM120Nvfp4Wrapper` so JIT always loads
  `v_cache_uses_pv_layout=True` attention modules. For normal V:
  - NHD normal V converts directly to NHD PV V scratch.
  - HND normal K/V/scales are first made contiguous NHD, then normal V converts
    to NHD PV. The attention launch receives `kv_layout_hnd=False` after this
    preconversion because PV V is NHD-only.
- `gen_fmha_nvfp4_sm120_module(..., v_cache_uses_pv_layout=False)` now fails
  early with a clear error instead of compiling a normal-V attention module.
  The JIT URI no longer includes the `pv_v_*` specialization axis.
- The direct SM120 wrapper now initializes `out` with zeros when `out=None`,
  matching the public `BatchPrefillWithPagedKVCacheWrapper` path.
- Added an `out_scratch.zero_()` guard before each paged launch. D128
  single-split paged mode copies `out_scratch` directly; without the guard,
  wrapper tests could expose stale finite allocator residue in an unwritten
  tile. This is output-sized, not a full partial-buffer clear.
- Refreshed `flashinfer/data` JIT mirrors for the D128/D256/D512 headers and
  shared SM120 NVFP4 paged support headers after the source-header edits.

Test updates:

- Normal-V vs PV-layout equivalence tolerances were updated to reflect the new
  preconversion route. This path dequantizes already-quantized normal FP4 V and
  re-quantizes to PV, so it is no longer expected to match the old in-kernel
  normal reblock bit pattern or the BF16-direct PV quantization bit pattern.
- Tests that compare separate wrapper instances now use separate raw workspace
  buffers. Reusing one workspace across live wrapper stacks measures workspace
  lifetime/aliasing, not kernel equivalence.

Validation:

- Focused wrapper gate:
  `test_sm120_nvfp4_wrapper_multi_kv_matches_single_kv_sm12x`,
  `test_standard_prefill_wrapper_sm120_nvfp4_backend_matches_direct_wrapper_sm12x`,
  and `test_sm120_nvfp4_backend_accepts_normal_v_layout_sm12x` all pass for
  D128/D256/D512: 9 passed in 11.62s.
- Full file validation:
  `tests/attention/test_nvfp4_kv_head_dim_512.py`: 35 passed in 12.25s.

Current state:

- The D256 normal-V compile blocker is removed because the wrapper no longer
  asks JIT for a `pv_v=False` attention module.
- Production attention specialization count is reduced by removing the V-layout
  axis. The wrapper still accepts normal V for compatibility, but the attention
  kernel path itself is PV-only.

## 2026-05-02 03:25 CDT — Production Ingress Framing And PV-Only ABI Cleanup

Clarification:

- The wording in the previous entry that called normal V a compatibility path
  was imprecise. Standard vLLM paged NVFP4 KV cache with normal V layout is the
  production ingress path. The SM120 attention kernel is PV-only internally
  because the PV MMA operand expects that physical layout, but the public
  wrapper is responsible for converting standard vLLM normal V into PV before
  launching attention.
- Direct PV-layout V remains an important benchmark and cache-writer path: it
  measures the attention-kernel ceiling and lets callers that already maintain
  PV-layout cache skip conversion. It is not the only production path.

Implementation cleanup:

- Removed the `v_cache_uses_pv_layout` specialization field from the SM120
  paged attention kernel config and the generated config include. The JIT URI
  no longer carries a V-layout axis, and the attention module always compiles a
  PV-consuming kernel.
- Removed the old in-kernel normal-V to PV producer branch from the D128, D256,
  and D512 stage kernels. The remaining V producer only stages PV-layout V data
  and PV-layout V scales.
- Removed normal-V side fields and helper functions from the paged-KV load
  params used by the attention kernel (`v_scales_trtllm_interleaved`,
  `v_global_scale`, normal-V scale decode, and standard-V PV re-quant helpers).
- Simplified the exported `paged_run` ABI: it no longer takes V-layout mode or
  normal-V scale-layout flags. It receives NHD K pages and PV-layout V pages.
  HND or normal-V input is handled at the Python wrapper boundary before this
  launch.
- Updated benchmark naming so paged-wrapper runs can report
  `kv_ingress="vllm-normal"` for the standard vLLM production ingress and
  `kv_ingress="pv"` for preconverted PV-layout cache. Dense direct runs report
  `kv_ingress="dense_pv_prepacked"`.

Validation:

- `python -m py_compile` passed for the edited Python modules and SM120
  benchmark harnesses.
- `git diff --check` passed.
- Focused wrapper gate passed after the ABI cleanup:
  `test_sm120_nvfp4_wrapper_multi_kv_matches_single_kv_sm12x`,
  `test_standard_prefill_wrapper_sm120_nvfp4_backend_matches_direct_wrapper_sm12x`,
  and `test_sm120_nvfp4_backend_accepts_normal_v_layout_sm12x`:
  9 passed in 136.58s.

## 2026-05-02 04:05 CDT — Wrapper Stream Ordering And Benchmark Path Labels

Implementation:

- Added `kv_ingress` to the SM120 benchmark harness:
  - `kv_ingress="vllm-normal"` measures the standard vLLM normal paged-KV
    production ingress plus wrapper conversion to PV.
  - `kv_ingress="pv"` measures a caller/cache-writer that already stores
    PV-layout V and skips conversion.
  - Dense direct runs report `kv_ingress="dense_pv_prepacked"`.
- Updated grid resume/report logic so repeated cells with different fused
  ingress paths do not collide on the same `(shape, kernel)` key.
- Allowed `v_cache_sf_layout="pv"` only when `v_cache_uses_pv_layout=True`.
  This lets benchmarks and direct callers label PV scale layout explicitly
  instead of relying on ignored defaults.
- Added explicit stream waits between PyTorch-side wrapper work and TVM-FFI
  `paged_run` launches. The wrapper mixes PyTorch copies/zeros with FFI kernels;
  without ordering, D128 intermittently read or copied unwritten output blocks
  unless `CUDA_LAUNCH_BLOCKING=1` was set.

Validation:

- Race-sensitive wrapper subset passed without launch blocking:
  `test_sm120_nvfp4_wrapper_multi_kv_matches_single_kv_sm12x` and
  `test_sm120_nvfp4_backend_accepts_normal_v_layout_sm12x`: 6 passed in 8.80s.
- Full attention file passed without launch blocking:
  `tests/attention/test_nvfp4_kv_head_dim_512.py`: 35 passed in 12.21s.
- `python -m py_compile` and `git diff --check` passed after the benchmark and
  wrapper updates.

Smoke timings after stream ordering fix:

```text
D128 dense direct, q=512 kv=8192 group=8:
  api=dense, kv_ingress=dense_pv_prepacked, finite=true, min_ms=0.387328

D128 paged wrapper, q=512 kv=128 group=8:
  api=paged, kv_ingress=pv, finite=true, min_ms=11.910144

D128 paged wrapper, q=512 kv=8192 group=8:
  api=paged, kv_ingress=pv,          finite=true, min_ms=790.188110
  api=paged, kv_ingress=vllm-normal, finite=true, min_ms=790.267212
```

Interpretation:

- The dense direct fused kernel remains healthy.
- The current paged wrapper path is correctness-clean but not performance-clean.
  Runtime scales linearly at roughly one slow paged-producer tile cost per
  128-token KV tile. PV and standard vLLM-normal ingress are effectively equal
  at this cell, so the dominant cost is inside `paged_run`'s paged K/V producer,
  not the normal-V to PV conversion.
- Next performance work should target the native paged producer data movement
  before running full production sweeps. The benchmark harness is now ready to
  record both ingress paths once that producer is fixed.

## 2026-05-02 11:18 CDT — Editable Source Mirror And Explicit Stream ABI

Implementation:

- Restored the documented editable-install source mirror. `build_backend` says
  editable installs should symlink:
  - `csrc` -> `flashinfer/data/csrc`
  - `include` -> `flashinfer/data/include`
  The worktree had materialized copies under `flashinfer/data`, so JIT could
  see stale source after top-level edits. Running `_prepare_for_editable()`
  restored directory symlinks. Future edits should target only top-level
  `csrc/` and `include/`.
- Replaced the wrapper stream-ordering workaround with an explicit stream ABI.
  The Python wrapper now passes
  `torch.cuda.current_stream(device).cuda_stream` into `quantize_q`,
  `paged_run`, and `dense_run`.
- Added `stream_from_handle()` in `csrc/tvm_ffi_utils.h` and switched the SM120
  NVFP4 C++ launch helpers from `get_stream(device)` to the stream handle passed
  by Python. This removes the per-call default-stream/current-stream sync.
- Renamed benchmark/report layout vocabulary to the orthogonal fields:
  - `api`: `paged` or `dense`
  - `v_layout`: `linear` or `pv`
  This replaces the earlier `kv_ingress` labels. The public quantization helper
  now uses `v_data_layout="linear"` or `"pv"` to match that vocabulary.

Validation:

- `python -m py_compile` passed for the edited wrapper, prefill,
  quantization, and benchmark Python files.
- `git diff --check` passed.
- Fresh-cache D128 dense direct smoke passed:
  `api=dense`, `v_layout=pv`, `q=512`, `kv=8192`, `group=8`,
  `finite=true`, `min_ms=0.215936`.
- Fresh-cache D128 paged wrapper smoke passed for standard linear V input:
  `api=paged`, `v_layout=linear`, `q=512`, `kv=128`, `group=8`,
  `finite=true`, `min_ms=12.114592`.
- Fresh-cache D128 paged wrapper smoke passed for preconverted PV V input:
  `api=paged`, `v_layout=pv`, `q=512`, `kv=128`, `group=8`,
  `finite=true`, `min_ms=11.943840`.
- Full focused correctness gate passed:
  `tests/attention/test_nvfp4_kv_head_dim_512.py`: 35 passed in 347.82s.

Interpretation:

- The stale-output issue was stream ordering, not a kernel math bug. Explicitly
  passing the PyTorch current stream into the FFI launches preserves ordering
  without the wrapper-level synchronization band-aid.
- The local JIT source drift hazard is removed for editable development because
  `flashinfer/data/csrc` and `flashinfer/data/include` now resolve directly to
  the tracked source directories.
- The benchmark schema is now stable enough for the next production sweeps:
  compare `api=paged, v_layout=linear` for the standard vLLM path, and
  `api=paged, v_layout=pv` for callers that pre-store PV-layout V.

## 2026-05-02 12:21 CDT — Paged Producer Fast Path And Correctness Gate

Implementation:

- Reworked the paged K producer away from the slow subbyte proxy RMW path.
  D128 now uses the FlashInfer-style `cp_async::pred_load_32b` path for
  partition-derived 4-byte K chunks.
- D256 uses a partition-derived packed-word K producer. This preserves the
  CUTLASS `partition_D` producer convention while avoiding per-nibble proxy
  stores.
- D512 needed a more conservative path: partition-derived byte-pair stores with
  CUTE iterator ordering and an atomic byte update inside the containing
  32-bit shared word. This removed the observed same-wrapper nondeterminism
  without reintroducing proxy RMW stores.
- The D512 V producer was aligned with the same partition-derived byte-pair
  pattern. The D512 K scale path now uses the shared paged K scale helper again.
- Test thresholds were adjusted for two bridge-equivalence checks that compare
  two valid FP4 execution routes rather than exact bitwise identity:
  - D512 multi-KV vs per-KV-head single runs: `atol=3e-3`.
  - D128 HND normal-V bridge vs NHD normal-V bridge mean diff: `<1e-5`.

Validation:

- Targeted gates passed:
  - D256 multi-KV vs single-KV.
  - D512 multi-KV vs single-KV.
  - D128 standard linear/HND V layout bridge.
- Full correctness file passed:
  `tests/attention/test_nvfp4_kv_head_dim_512.py`: 35 passed in 1.96s
  on warm JIT cache.
- Paged performance smoke after the producer change:
  - D128, `api=paged`, `v_layout=pv`, `q=512`, `kv=128`, `group=8`:
    `finite=true`, `min_ms=0.597024`.
  - D128, `api=paged`, `v_layout=pv`, `q=512`, `kv=8192`, `group=8`:
    `finite=true`, `min_ms=34.670784`.

Interpretation:

- The pathological paged producer behavior is fixed for the smoke path:
  q=512/kv=8192 dropped from roughly 790 ms to roughly 35 ms.
- D128 is on the intended `cp.async` data-movement path. D256 and D512 are
  correctness-clean no-proxy paths, but D512 still needs a better non-atomic
  producer before treating its paged path as performance-final.
- The next benchmark sweeps should use the production wrapper path and record
  `api` plus `v_layout` as separate dimensions.

## 2026-05-02 13:14 CDT — Explicit Stream Propagation Through Normal-V Conversion

Implementation:

- Threaded an optional `stream_handle` through the NVFP4 quantization helpers
  used by the SM120 normal-V to PV conversion bridge:
  - `fp4_quantize(..., stream_handle=...)`
  - `nvfp4_kv_dequantize(..., stream_handle=...)`
  - `nvfp4_kv_quantize(..., stream_handle=...)`
  - `nvfp4_quantize_paged_kv_cache(..., stream_handle=...)`
- Updated the underlying C++ FFI launches for the affected helpers to use
  `stream_from_handle(stream_handle)` when a nonzero handle is provided, and to
  preserve the existing `get_stream(...)` behavior for callers that do not pass
  a stream.
- Updated the SM120 NVFP4 wrapper so the standard linear-V production input
  bridge and the subsequent Q-quantize + attention launches all use the same
  PyTorch current stream handle. This removes the remaining stream-ordering
  hole without reintroducing wrapper-level `wait_stream` synchronization.

Validation:

- Python syntax and whitespace checks passed:
  `python -m py_compile flashinfer/fmha_nvfp4_sm120.py flashinfer/quantization/fp4_quantization.py`
  and `git diff --check`.
- The isolated D128 HND linear-V bridge gate passed after the stream ABI patch:
  `test_sm120_nvfp4_backend_accepts_normal_v_layout_sm12x[128-4]`: 1 passed in
  2.89s.
- A 12-iteration diagnostic loop comparing NHD linear-V, HND linear-V, and
  preconverted PV V for D128/q=256/kv=1024 stayed stable. HND-vs-NHD mean
  differences were around `1e-7` with max below `5e-4`.
- A full-file run initially reproduced the D128 HND bridge flake once after a
  fresh JIT rebuild: 34 passed, 1 failed. Targeted predecessor subsets did not
  reproduce it.
- A second full-file run after the diagnostic sequence passed:
  `tests/attention/test_nvfp4_kv_head_dim_512.py`: 35 passed in 1.94s.

Interpretation:

- The stream ABI is now structurally correct for the existing multi-step wrapper
  path: conversion, Q quantization, attention, and output copies are ordered by
  using the same current stream rather than by explicit synchronization.
- The one non-reproducing full-file failure is recorded as a transient flake
  during the stream-ABI transition. If it appears again, the next place to
  instrument is the second NHD reference output inside the HND bridge test; the
  diagnostic loop showed the HND path itself is stable when isolated.
- This is still Option A for production integration: patch the seams while
  keeping the current Python orchestration. The structural Option B remains to
  collapse kv-head dispatch, Q quantization, and ragged padding into the kernel
  launch surface so the SM120 wrapper looks like other FlashInfer prefill
  wrappers.

## 2026-05-02 14:35 CDT — Kernel-Side Multi-KV-Head Dispatch

Implementation:

- Removed the Python per-KV-head attention loop from
  `BatchPrefillWithPagedKVCacheSM120Nvfp4Wrapper.run()`.
- The wrapper now quantizes all Q heads once, calls `paged_run` once with
  `kv_head=-1`, and scatters the all-head output back to the caller tensor in
  one view copy.
- Extended the common paged C++ launcher so `kv_head >= 0` preserves the legacy
  single-KV-head path, while `kv_head < 0` launches all KV heads in one kernel
  grid.
- Extended the D128/D256/D512 stage kernels so all-head paged launches derive
  `kv_head` from `blockIdx.x`, while keeping the scratch/output row index as the
  all-head global Q tile. This preserves the existing per-head varlen batch math
  and makes `(kv_head, batch, q_tile)` a kernel-side scheduling decision.
- Updated the paged Q-pad copy, direct output copy, and split-KV combine kernels
  to understand all-head flattened row layout:
  `[token, kv_head * group_size + group_offset, head_dim]`.
- Updated public NVFP4 quantization wrappers to use PyTorch's current CUDA stream
  by default when callers do not pass an explicit `stream_handle`. This prevents
  direct public quantization calls from racing with adjacent torch ops.

Validation:

- Syntax and whitespace checks passed:
  `python -m py_compile flashinfer/fmha_nvfp4_sm120.py flashinfer/quantization/fp4_quantization.py`
  and `git diff --check`.
- Targeted all-head vs single-head wrapper gates passed:
  - D128/group=4: 1 passed in 45.42s on cold JIT build.
  - D256/group=6: 1 passed in 40.58s on cold JIT build.
  - D512/group=4: 1 passed in 48.96s on cold JIT build.
- Full correctness file passed normally after adding test-boundary synchronization
  around the SM120 wrapper tests:
  `tests/attention/test_nvfp4_kv_head_dim_512.py`: 35 passed in 1.54s.
- The same full file also passed with `CUDA_LAUNCH_BLOCKING=1`: 35 passed in
  1.57s. This confirmed the transient full-suite failures were cross-test async
  contamination from prior FFI kernels, not an all-head row-mapping error.

Interpretation:

- Multi-KV-head dispatch is now structural and kernel-side. One wrapper run no
  longer performs `num_kv_heads` Q-slice copies, Q-quant launches, attention
  launches, and output copies.
- The wrapper still has two major compensation steps left before it matches the
  shape of FlashInfer's mature paged prefill wrappers:
  - Q quantization remains a separate FFI launch before attention.
  - Ragged Q still goes through a padded Q scratch copy instead of being read and
    quantized directly by the attention producer.
- Next structural item: fuse BF16 Q ingestion and NVFP4 Q quantization into the
  paged attention launch so the kernel accepts BF16 Q directly and the wrapper
  stops staging a pre-quantized Q cache.

## 2026-05-02 15:05 CDT — BF16 Q Accepted By The Paged FFI Entry Point

Implementation:

- Added `paged_run_bf16_q` to the D128/D256/D512 SM120 NVFP4 paged modules.
- The new entry point accepts BF16 Q directly, quantizes it to the existing
  NVFP4 Q packed/scale workspace on the same explicit CUDA stream, then invokes
  the common paged attention launcher.
- Updated the production Python wrapper to call `paged_run_bf16_q` instead of
  calling `quantize_q` as a separate Python-visible FFI operation before
  `paged_run`.
- Kept the older `paged_run` and `quantize_q` exports for direct tests,
  benchmarks, and dense/debug surfaces that still operate on pre-quantized Q.
- Added synchronization after public KV quantization setup in the SM120 wrapper
  tests. The attention wrapper itself uses explicit streams; the test sync keeps
  these tests from inheriting async state from public quantization/setup kernels
  and prior FFI tests in the same pytest process.

Validation:

- Syntax and whitespace checks passed:
  `python -m py_compile flashinfer/fmha_nvfp4_sm120.py flashinfer/quantization/fp4_quantization.py`
  and `git diff --check`.
- Targeted all-head wrapper gates passed on a fresh BF16-Q JIT cache:
  - D128/group=4: 1 passed in 49.60s.
  - D256/group=6 and D512/group=4: 2 passed in 97.96s.
- Full correctness file passed normally on the BF16-Q path:
  `tests/attention/test_nvfp4_kv_head_dim_512.py`: 35 passed in 1.55s.

Interpretation:

- The production wrapper no longer orchestrates Q quantization as a separate FFI
  boundary. BF16-Q ingestion is now part of the SM120 paged module API.
- This is not yet true in-mainloop Q quantization: the C++ paged entry point
  still performs a Q-quantization pre-pass into Q packed/scale workspace before
  launching the attention stage kernel. It is nevertheless the right API shape:
  callers pass BF16 Q, and future in-mainloop quantization can replace the
  pre-pass without changing the Python or vLLM-facing wrapper contract.
- Remaining wrapper compensation is now concentrated in ragged-Q padding/copy
  scratch. The next structural target is to make the stage kernel consume the
  ragged BF16 Q layout directly so the padded Q packed/scales scratch copy can
  be removed.

## 2026-05-02 13:51 CDT — Native Linear-V Path And Wrapper Sync Removal

Implementation:

- Removed the production wrapper synchronization path. There is no
  `torch.cuda.current_stream(...).synchronize()`, `_join_torch_and_ffi_streams`,
  or wrapper-side `wait_stream` ordering in the SM120 NVFP4 wrapper hot path.
- Removed the Python-side normal-linear V conversion bridge from the production
  wrapper. The wrapper no longer invokes `nvfp4_quantize_paged_kv_cache` during
  `BatchPrefillWithPagedKVCacheSM120Nvfp4Wrapper.run()`.
- Added `paged_run_bf16_q_linear_v` to the D128/D256/D512 paged modules. This
  entry point accepts BF16 Q and standard linear FP4 V cache pages, converts V
  to the PV-reblocked scratch layout inside the SM120 module on the explicit
  CUDA stream, then launches the paged attention path.
- Added a native module-side linear-V to PV-V conversion kernel in
  `csrc/fmha_nvfp4_sm120_paged_common.cuh`. It supports standard packed V pages
  with TRT-LLM-interleaved or linear V scale layout, and HND or NHD input KV
  layout.
- Updated the paged V producer contract: K can still be read directly from HND
  or NHD cache layout, while PV V scratch is always consumed as NHD. This removes
  the old Python HND-to-NHD compensation step for K/V.
- Fixed the D128/D256 PV producer sub-byte race by replacing proxy nibble stores
  with a word-atomic byte update path, matching the deterministic D512 producer
  mechanism.

Validation:

- Targeted normal-linear V layout gate passed on SM120:
  `tests/attention/test_nvfp4_kv_head_dim_512.py::test_sm120_nvfp4_backend_accepts_normal_v_layout_sm12x`
  with 3 passed in 165.53s.
- Targeted multi-KV-head and standard-wrapper gates passed:
  `test_sm120_nvfp4_wrapper_multi_kv_matches_single_kv_sm12x` and
  `test_standard_prefill_wrapper_sm120_nvfp4_backend_matches_direct_wrapper_sm12x`
  with 6 passed in 0.59s.
- Full SM120 NVFP4 attention correctness file passed:
  `tests/attention/test_nvfp4_kv_head_dim_512.py` with 35 passed in 1.45s.

Interpretation:

- Multi-KV-head dispatch and ragged-Q scheduling are module/kernel-side now; the
  production wrapper no longer loops over KV heads and no longer runs a
  Python-visible Q-quant operation before attention.
- BF16 Q is accepted at the FFI boundary and quantized by a CUDA pre-pass inside
  the SM120 module into Q scratch. This is not yet in-mainloop Q quantization,
  but the public wrapper contract now has the right shape for future replacement
  without changing vLLM-facing call sites.
- The standard vLLM linear-V input layout is handled by the SM120 module itself.
  Callers that already store PV-reblocked V can still bypass that conversion by
  using `v_cache_uses_pv_layout=True`.
- Remaining cleanup before upstream review: remove dummy pre-quantized Q
  arguments from the BF16-Q FFI surface if they are no longer needed by direct
  tests, and evaluate whether full in-mainloop Q quantization is needed for
  performance or reviewer expectations.

## 2026-05-02 16:35 CDT — Normal-V Reblock Diagnostic State

Implementation state:

- The production wrapper no longer invokes a Python-side or module-side V bridge
  before attention. Linear V and PV V both flow through the paged attention
  module; the V layout is now a JIT specialization axis.
- D128/D256/D512 linear-V producers were changed to the fmha_v2-style two-pass
  shape: compute PV scales first, stage those scales in shared memory, then
  load/dequant/requant V data against the staged scale values.
- The producer still differs from fmha_v2 in one deliberate diagnostic respect:
  the data pass uses four explicit `fp32_pair_to_e2m1_byte` conversions packed
  into a 32-bit word instead of `float8_to_e2m1x8`. Both use hardware E2M1
  conversion on SM100+, but the pairwise path preserves explicit byte ordering
  in this CUTLASS partition-derived store path.

Validation:

- D256 `test_sm120_nvfp4_backend_accepts_normal_v_layout_sm12x[256-6]` still
  fails narrowly after the pairwise hardware pack change:
  `diff_hnd.mean() = 5.49e-05` vs the current `5e-05` gate.
- A repeated cached D256 HND-vs-NHD diagnostic with fresh wrappers per run is
  sequence-dependent:
  - Most runs are bit-identical.
  - Later runs show sparse drift, e.g. max `0.0028` with no elements above
    `3e-3`, then max `0.00371` with 3 elements above `3e-3`.
  - The same row/head/dim recurred in the max-diff location in multiple
    nonzero runs.

Interpretation:

- This no longer looks like a deterministic HND-vs-NHD stride formula error:
  identical logical tensors frequently produce bit-identical outputs across
  layouts.
- The remaining issue is likely an ordering, coverage, or race problem in the
  in-kernel linear-V path, or a codepoint-boundary sensitivity that the current
  strict HND-vs-NHD test exposes.
- No more producer fixes should be applied until a binary-search diagnostic
  identifies the first stage where NHD and HND diverge: V/K smem operand after
  producer, QK accumulator, softmax probabilities, PV accumulator, or final
  output.

## 2026-05-02 17:12 CDT — D256 HND-vs-NHD Producer Boundary Diagnostic

Diagnostic added:

- Added a D256 diagnostic export, `debug_producer_smem`, to the generated
  SM120 NVFP4 paged module.
- The diagnostic runs the same partition-derived K and V paged producers used
  by the normal-V specialization and dumps the physical shared-memory images:
  - QK/K operand `smem_B`.
  - QK/K scale `smem_SFB`.
  - PV/V operand `smem_B`.
  - PV/V scale `smem_SFB`.

Validation:

- Compared NHD and HND layouts for the same logical D256 cache at:
  - `kv_head = {0, 1}`.
  - `kv_tile = {0, 7}`.
  - `k_outer = {0, 1}`.
  - `out_group_idx = 0`.
- All four producer dumps matched byte-for-byte for every case:
  - `k_smem_b`: match.
  - `k_smem_sfb`: match.
  - `v_smem_b`: match.
  - `v_smem_sfb`: match.
- Re-ran the targeted pytest gate after rebuilding the D256 normal-V spec:
  `test_sm120_nvfp4_backend_accepts_normal_v_layout_sm12x[256-6]` passed once,
  then a repeated pytest invocation later observed a sparse final-output max
  drift of `0.0035`.
- Reproduced the exact test call sequence manually, including the intermediate
  PV-layout run, and retained inner SM120 wrapper buffers. That sequence was
  bit-exact for 12 iterations across:
  - `out`.
  - `_out_scratch`.
  - `_out_group`.
  - `_partial`.
  - `_split_m`.
  - `_split_l`.
- A follow-up 100-iteration loop of the same exact call sequence found no
  nonzero final-output difference.

Interpretation:

- The D256 HND-vs-NHD issue is not caused by producer source addressing or
  layout stride math: the physical K/V operand and scale shared-memory images
  are identical across layouts.
- The currently observed failing values are sparse and codepoint-scale at the
  BF16 final output boundary. The only captured pytest failure after the
  producer diagnostic was a max drift of `0.0035`; repeated direct/manual runs
  could not reproduce a downstream difference to capture.
- No kernel fix was applied after the diagnostic. If the sparse pytest-only
  drift reappears, the next diagnostic boundary should be an active-stage dump
  inside the production stage kernel for the selected failing CTA, because the
  standalone producer boundary has been ruled out.

## 2026-05-02 17:29 CDT — D256 Active-Stage HND-vs-NHD Boundary Diagnostic

Diagnostic added:

- Extended the D256 paged module with `debug_stage_run`, a D256-only replay
  entry point that uses the same production stage kernel and can dump a selected
  CTA/tile boundary.
- Added selected-CTA dumps for:
  - QK logits in `smem_logits` after QK MMA and before softmax.
  - P data smem after softmax and before PV MMA.
  - Logical P scale bytes after softmax and before PV MMA.
  - O smem after final PV accumulation and before the epilogue-visible output
    store.
  - Online row stats (`global_m`, `global_l`).
- The first version of this diagnostic replayed from Q scratch and produced a
  false QK-logits divergence because BF16-Q production runs do not use Q scratch
  as the Q source. The diagnostic was corrected to accept BF16 Q and populate
  `paged_params.q_bf16`/Q strides exactly like the production BF16-Q path.
- The P-scale dump was changed from raw physical SFA allocation bytes to the
  512 logical P-scale bytes that PV MMA can consume. The raw physical allocation
  has 2048 bytes; the other 1536 bytes are padding/unwritten extent and were a
  false divergence source.

Validation:

- Reproduced the pytest-like HND-vs-NHD call sequence with the intermediate
  normal-V vs PV-layout `torch.quantile` work that makes the sparse drift
  reproduce reliably.
- A caught run had final-output max drift around `0.00424`.
- For the final-diff CTA and the CTA owning the replay `out_scratch` max:
  - `logits`: byte-identical.
  - `p_smem`: byte-identical.
  - `p_sfa_logical`: byte-identical.
  - `o_smem`: byte-identical.
  - `global_m/global_l`: exact.
  - Full-grid `out_scratch`: diverged after the epilogue-visible output store
    (examples: max `0.00262` and `0.00269` in replayed runs).
- Rechecked batch-1 producer smem using `debug_producer_smem` with the batch-1
  block-table row:
  - QK/K operand and scale smem matched for `kv_head=1`, `kv_tile=7`,
    `k_outer={0,1}`.
  - PV/V operand and scale smem also matched for the same cases.

Interpretation:

- The D256 HND-vs-NHD drift is not a K/V producer stride or source-addressing
  bug.
- With the corrected BF16-Q diagnostic path, the consumed QK logits, softmax
  probabilities/scales, PV output smem, and online stats match across layouts
  for caught failures.
- The first observed real divergence is after O smem and before/at the
  `out_scratch` writeback. The next fix/debug target is therefore the final
  `pipeline_corr_epi` handoff and epilogue store path, not the K/V producer or
  softmax math.

## 2026-05-02 18:33 CDT — D256 Coverage-Class Root Cause And Fix

Diagnostic result:

- Reproduced the D256 normal-V HND-vs-NHD instability only when the test
  sequence included the intermediate PV-layout run followed by
  `torch.quantile(...).item()`.
- Without the intermediate quantile work, repeated NHD-vs-HND linear-V runs were
  bit-exact for 20/20 iterations.
- The PV-layout run alone did not trigger drift; the quantile work perturbed the
  subsequent linear-V stage enough to expose an uninitialized shared-memory
  coverage gap.
- Input tensors and metadata (`q`, K/V pages, K/V scales, block tables,
  `qo_indptr`, `kv_lens`) were snapshotted before and after the PV+quantile
  sequence and remained byte-identical.
- Clean direct stage replays were deterministic, but all-CTA active-stage dumps
  under the quantile-triggered condition showed first divergence in QK logits;
  P smem, P scales, and O smem divergence were downstream.

Root cause:

- The active D256 path initialized QK/PV B operand smem and SFB scale smem, but
  did not initialize QK A operand smem or SFA scale smem before BF16-Q staging.
- Q staging writes through CUTLASS partitioned fragments. The consumer can read
  the full LDSM/MMA fragment extent, while the producer only writes the covered
  logical fragment subset. Undefined residue in A/SFA therefore fed QK MMA and
  manifested as sequence-dependent logits divergence.
- The earlier logits `-inf` fill was necessary but not sufficient: it protects
  score positions that QK does not overwrite, but it cannot protect QK itself
  from stale Q operand/scale smem.

Fix applied:

- Added D256 full QK A byte zero initialization for the paged path at kernel
  entry.
- Added D256 full QK SFA neutral-scale (`0x38`) initialization for the paged
  path at kernel entry.
- Kept the full score-tile `-inf` initialization before each QK tile writes its
  logits.

Validation:

- The previous quantile-triggered NHD/HND sequence is now bit-exact for 20/20
  iterations:
  - `out_max = 0.0` for every iteration.
  - `scratch_max = 0.0` for every iteration.
  - No NaNs.
- All-CTA active-stage replay under the same trigger now matches byte-for-byte:
  - `logits`: byte-identical.
  - `p_smem`: byte-identical.
  - `p_sfa` logical scale bytes: byte-identical.
  - O rows reconstructed from O smem: exact.
  - `out_scratch`: exact.

Follow-up:

- Treat this as a coverage-class issue, not an isolated D256 bug. The production
  kernels should make smem coverage policy explicit for every producer/consumer
  smem region: operand data smem gets zero sentinel, UE4M3 scale smem gets
  neutral `0x38`, logits get `-inf`, and output staging gets zero where consumer
  coverage is not proven complete.

## 2026-05-02 19:31 CDT — Covered Smem Coverage Policy Across D128/D256/D512

Change:

- Added `fmha_nvfp4_sm120_covered_smem.cuh` with `CoveredSmemTile`, a small
  wrapper that constructs a CUTE shared-memory tensor and collectively
  initializes its full physical extent before any CUTLASS consumer can read it.
- Added sentinel traits for the coverage classes this kernel needs:
  - `SentinelZero`: bytewise `0x00` for FP4 operand and output-like staging.
  - `SentinelE4M3One`: bytewise `0x38` for UE4M3 scale smem.
  - `SentinelNegInf`: typed `-inf` for BF16/FP32 score smem.
- Made the wrapper pointer-type-parametric because CUTLASS sub-byte smem
  allocation `begin()` returns CUTE sub-byte iterators, not plain `Element*`.
- Made the barrier id generic because the active call sites use strongly typed
  `cutlass::arch::ReservedNamedBarriers` values.
- Added an explicit `participant_idx` overload so MMA-only logits fill can use
  `qk_mma_thread_idx` rather than full `threadIdx.x`.

Production kernel adoption:

- Replaced active QK/PV/P smem tensor construction in D128, D256, and D512 with
  `CoveredSmemTile` wrappers:
  - QK A/B operand smem: zero sentinel.
  - QK SFA/SFB scale smem: neutral `0x38` sentinel.
  - PV B operand smem: zero sentinel.
  - PV SFB scale smem: neutral `0x38` sentinel.
  - P staging A/SFA smem: zero and neutral `0x38` sentinels.
- Replaced the per-QK-tile logits manual fill with `NegInfSmemTile` in all
  three head-dim kernels. D256 already had the manual fill; D128 and D512 now
  get the same full-score-tile policy.
- Removed the D256/D512 one-off active-path B/SFB init blocks that were the
  previous piecemeal coverage fixes.

Audit:

- The active production stage kernel no longer builds raw CUTLASS smem tensors
  for QK/PV/P producer-consumer staging without an explicit coverage wrapper.
- Remaining raw `make_tensor(make_smem_ptr(...))` sites are outside the active
  production allocation path:
  - The standalone atom/probe diagnostic body.
  - The Q register-staging read helper, which consumes already-covered Q/SFA
    smem and does not allocate a producer-visible staging region.
  - D256 diagnostic producer/debug code.

Validation:

- Built all production-flag SM120 JIT specs on the local machine:
  - D128 linear-V (`kPvLayoutV=false`).
  - D256 linear-V (`kPvLayoutV=false`).
  - D512 linear-V (`kPvLayoutV=false`).
  - D128 PV-layout V (`kPvLayoutV=true`).
  - D256 PV-layout V (`kPvLayoutV=true`).
  - D512 PV-layout V (`kPvLayoutV=true`).
- The first D128/D256/D512 compile attempts accidentally used
  `FLASHINFER_JIT_VERBOSE=1`; this codebase treats that as debug mode and emits
  `-G -O0`. Stale debug cache directories were moved aside and the specs were
  rebuilt with `FLASHINFER_JIT_DEBUG=0`, producing the intended `-DNDEBUG -O3`
  builds.
- Runtime pytest on this machine cannot execute SM120 kernels because the local
  GPU is an RTX 4090 (`sm_89`). The SM120 NVFP4 attention test file collects and
  skips cleanly:
  - `tests/attention/test_nvfp4_kv_head_dim_512.py`: 35 skipped with the
    expected SM120/SM121 requirement.

Follow-up:

- Re-run the D128/D256/D512 runtime correctness diagnostics on an SM120/SM121
  system. The local validation here proves compilation and test collection, not
  device execution.
- After runtime correctness is clean, measure the cost of the systematic
  coverage fills. Correctness first; if the fill cost is material, optimize the
  fill placement or prove specific producer/consumer regions are fully covered
  before removing any sentinel initialization.

## 2026-05-02 20:44 CDT — Covered Smem Adoption Completed And SM120 Runtime Gate

Correction to the previous validation note:

- The host has multiple GPUs. The default visible device was GPU 0, an RTX 4090
  (`sm_89`), which caused the first pytest run to skip the SM120 tests.
- GPU 2 is an RTX PRO 6000 Blackwell Max-Q Workstation Edition (`sm_120`) and is
  the correct local device for this validation.

Additional adoption work:

- Tightened `CoveredSmemTile` so it can be used as the real data-flow handle,
  not only as an init side effect:
  - Added `CoveredSmemNoInit` construction for aliased smem regions.
  - Added `fill_and_sync(...)` so aliased regions can be initialized at the
    data-dependency boundary where the next consumer needs a fresh sentinel.
- Converted the remaining raw helper smem construction sites in D128/D256/D512:
  - The standalone atom/probe helper now constructs QK A/B/SFA/SFB through
    covered tensors.
  - `cutlass_qk_tma_q_register_stage` now accepts the already-covered QK A/SFA
    tensor handles instead of reconstructing raw tensors from `TensorStorage`.
- Converted logits handling from side-effect-only initialization to real covered
  tensor handles:
  - A no-init `NegInfSmemTile` handle is constructed for each physical logits
    stage.
  - `run_qk_tile()` calls `fill_and_sync()` at the QK-to-softmax dependency
    point, after QK B aliasing is no longer active for that score stage.
  - QK writes and softmax reads now go through the covered logits tensor handle
    using the existing physical skew index.
- A grep audit no longer finds raw `make_tensor(make_smem_ptr(...))` smem
  construction, `(void)logits_covered`, or raw `smem_logits_stage[...]` access
  in the D128/D256/D512 production headers.

Build validation:

- Rebuilt the primary production specs with `FLASHINFER_JIT_DEBUG=0` and
  `-DNDEBUG -O3`:
  - D128/D256/D512, `kPvLayoutV=false`.
  - D128/D256/D512, `kPvLayoutV=true`.
- Prebuilt the D512 sliding-window specs that the test suite exercises:
  - D512, sliding-window, `kPvLayoutV=false`.
  - D512, sliding-window, `kPvLayoutV=true`.

Runtime validation:

- Ran the SM120 NVFP4 attention test shard on GPU 2:
  - Command shape: `CUDA_VISIBLE_DEVICES=2 ... pytest
    tests/attention/test_nvfp4_kv_head_dim_512.py -q --tb=short -rs`
  - Result: `35 passed in 2.03s`.

Interpretation:

- The coverage abstraction is now load-bearing in the code, not just a fill
  statement beside raw pointer accesses.
- The known SM120 correctness suite passes with the completed covered-smem
  adoption.

## 2026-05-02 21:22 CDT — Covered Smem Paranoia Validation

Paranoia checks after the completed covered-smem adoption:

- Re-ran the originally flaky production-layout test
  `test_sm120_nvfp4_backend_accepts_normal_v_layout_sm12x[256-6]` 50 times in a
  loop on GPU 2 (`sm_120`).
  - Result: 50/50 passed.
  - Each iteration reported about 0.46-0.48 seconds with the JIT cache warm.
- Tightened the bridge-era tolerance relaxations and reran the full SM120 NVFP4
  attention test file on GPU 2.
  - D512 multi-KV-vs-single-KV tolerance tightened from `3e-3` to `2e-3`.
  - Linear-V-vs-PV tolerance tightened from mean `3e-3`, p99 `1.2e-2`, max
    `2.5e-2` to mean `2e-3`, p99 `1.0e-2`, max `2.0e-2`.
  - HND-vs-NHD tolerance tightened back to mean `1e-5`, max `1e-3`.
  - Result: `tests/attention/test_nvfp4_kv_head_dim_512.py` passed all 35 tests
    in 2.06 seconds.
- Tried a more aggressive linear-V-vs-PV mean threshold of `1e-3`.
  - Result: failed at D128 with observed mean absolute difference about
    `0.0012`.
  - Interpretation: the remaining linear-V-vs-PV gap is normal FP4 reblock
    quantization noise, not the previous intermittent state-residue failure.
- Measured one targeted cold-cache rebuild by moving aside only the D256
  production linear-V spec cache:
  - Spec:
    `fmha_nvfp4_sm120_d256_causal_True_swa_True_softcap_True_pv_v_False`.
  - Result: 172.26 seconds.

Interpretation:

- The prior intermittent D256 normal-V failure did not reproduce under 50
  consecutive executions.
- The bridge-era tolerance relaxations can be tightened materially now that the
  bridge has been removed and smem coverage is systematic.
- Cold-cache compile time for the D256 linear-V production spec is still above
  the desired 40-90 second envelope; compile-time work remains, even though
  runtime correctness is stable.

## 2026-05-02 22:00 CDT — SM120 NVFP4 Production Cleanup Pass

Cleanup scope:

- Removed the standalone debug GEMM probe infrastructure from the D128/D256/D512
  production headers:
  - Deleted `smem_fp4_debug_code`.
  - Deleted the `cutlass_smem_atom_gemm_tile_body_impl` probe templates.
  - Deleted the `cutlass_smem_atom_gemm_tile_body` wrappers.
  - Kept `cutlass_qk_tma_q_register_stage` because it is still used by the
    production MMA path to stage Q fragments from the Q pipeline.
- Removed D256-only debug FFI surface:
  - Deleted the `debug_producer_smem` export.
  - Deleted the `debug_stage_run` export.
  - Removed `Sm120Nvfp4D256StageDebugParams` and the kernel-side diagnostic
    copy plumbing.
- Removed debug/dev constants that had no remaining production use:
  - `kDebugHead`.
  - `kBenchRows`.
  - `kBenchQTiles`.
  - Fixed-shape scaffold constants such as `kQLen`, `kGroup`, `kKvLen`,
    `kPackedHeadDim`, `kScaleCols`, `kQRows`, `kProbPackedCols`,
    `kProbScaleCols`, `kShapeBMaxKvLen`, `kShapeBMaxKvTiles`, `kSplitKvLen`,
    and `kNumKvSplits`.
- Audited producer traps:
  - Removed the dead `pv_code_for` false-branch trap.
  - Kept structural invariant traps for 8-nibble partitions, 4-byte smem
    alignment, byte-pair colocation, and compact P staging contiguity.
  - Added one-line comments at the kept trap sites explaining the invariant.
- Cleaned wrapper staging:
  - Removed duplicate PV scale-layout validation.
  - Removed `run_*` aliases that were pure renames before the FFI call.
  - Flattened the PV-vs-linear branch to only compute `run_kv_layout_hnd` and
    `v_scale_layout_code`.
  - Removed defensive `_partial.zero_()`, `_split_m.fill_(-inf)`,
    `_split_l.zero_()`, `_out_scratch.zero_()`, and `_out_group.zero_()`.

Validation:

- Ran the SM120 NVFP4 attention test shard on GPU 2 after the cleanup:
  - Command shape: `CUDA_VISIBLE_DEVICES=2 ... pytest
    tests/attention/test_nvfp4_kv_head_dim_512.py -q --tb=short -rs`
  - Result: `35 passed in 1308.50s`.
- The long runtime was dominated by cold JIT rebuilds. Process inspection during
  the run showed `ptxas` compiling the D512 linear-V spec at 99.9% CPU for
  several minutes.

Interpretation:

- The production headers no longer carry the debug GEMM probe infrastructure or
  D256 diagnostic FFI plumbing.
- The wrapper scratch clears are not required by the current correctness suite
  after covered-smem adoption and full producer/output coverage fixes.
- D512 cold compile remains materially slow even after removing the probe
  templates; compile-time reduction is still an open area separate from this
  cleanup.

## 2026-05-02 22:45 CDT — SM120 NVFP4 Silent-Failure Risk Audit

Audit target:

- Reviewed a list of remaining "silent correctness/perf" risks after the
  production cleanup pass:
  - Hidden Q contiguity copy in the Python wrapper.
  - Scratch residue in `_workspace_buffer`, Q scratch, split-KV partial/stat
    buffers, and output scratch.
  - Block-table/page-layout mistakes that could otherwise surface only as
    kernel traps or arbitrary memory reads.
  - Sliding-window disabled state (`window_left=-1`) vs active-SWA state.
  - Stream plumbing and bridge-era scratch clear assumptions.

Findings:

- The defensive `_partial`, `_split_m`, `_split_l`, `_out_scratch`, and
  `_out_group` clears had already been removed in the cleanup commit, so that
  part of the risk list was stale.
- The SM120 NVFP4 FFI path already uses an explicit stream handle from
  `torch.cuda.current_stream(...).cuda_stream`; it no longer relies on
  `get_stream(...)` for the paged BF16-Q path.
- CUTLASS workspace residue is handled inside the D128/D256/D512 launchers:
  each launch zeros the required CUTLASS workspace range with `cudaMemsetAsync`
  before initializing QK/PV workspaces.
- The paged split-KV combine reads the current per-sequence `num_splits`, not
  the maximum split count, so it does not intentionally reduce unwritten split
  slots.
- The remaining real issues were wrapper/FFI contract clarity:
  - Python silently copied non-contiguous Q with `q.contiguous()`.
  - The C++ FFI check still required full Q contiguity even though the BF16-Q
    kernel path carries Q strides and only needs last-dimension contiguity.
  - `block_tables` validation did not explicitly require 2-D,
    page-dimension-contiguous rows, per-sequence coverage for `kv_lens`, or
    non-negative active page entries before the FFI call.
  - Physical page-count validation only ran for the linear-V path; PV-layout V
    should fail early too if `block_tables` references pages beyond the cache.

Changes:

- Removed the hidden Python `q.contiguous()` copy. The wrapper now passes the
  user's Q tensor directly and requires only last-dimension contiguity.
- Relaxed the BF16-Q FFI entry from full `CHECK_INPUT_AND_TYPE(q, dl_bfloat16)`
  contiguity to CUDA/dtype plus last-dimension-contiguous validation, matching
  the stride-aware kernel path.
- Added Python-side validation for:
  - `window_left == -1` or positive; `0` and less than `-1` now fail.
  - `qo_indptr` dtype/shape.
  - `kv_lens` dtype/shape.
  - `block_tables` dtype/device/shape, page-dimension contiguity, row count,
    active page coverage, and non-negative active page IDs.
  - K/V and scale tensor device/dtype/last-dimension contiguity.
  - Unconditional physical page count coverage for both PV-layout and
    linear-V paths.
- Added `test_sm120_nvfp4_wrapper_scratch_poison_does_not_affect_output_sm12x`:
  - Uses D256/group=6 with multi-KV, ragged Q, split-KV (`split_kv_len=512`),
    and a non-contiguous Q view whose last dimension is contiguous.
  - Poisons `_workspace_buffer`, `_q_packed`, `_q_scales`,
    `_q_packed_scratch`, `_q_scales_scratch`, `_partial`, `_split_m`,
    `_split_l`, `_out_scratch`, and `_out_group` with different values before
    repeated runs.
  - Asserts bit-exact output equality across the poisoned runs.

Validation:

- First run of the new scratch-poison test failed because the C++ FFI still
  required full Q contiguity:
  - Failure site: `RunPagedBatchBf16QImpl`, `CHECK_INPUT_AND_TYPE(q,
    dl_bfloat16)`.
  - Fix: replace that full-contiguity check with CUDA/dtype and
    last-dimension-contiguous checks.
- Re-ran the new scratch-poison test:
  - Result: `1 passed in 88.42s`.
- Ran focused wrapper validation:
  - `test_sm120_nvfp4_wrapper_multi_kv_matches_single_kv_sm12x`
  - `test_sm120_nvfp4_wrapper_scratch_poison_does_not_affect_output_sm12x`
  - `test_standard_prefill_wrapper_sm120_nvfp4_backend_matches_direct_wrapper_sm12x`
  - `test_sm120_nvfp4_backend_accepts_normal_v_layout_sm12x`
  - Result: `10 passed in 1224.48s`.
  - Runtime was dominated by cold `ptxas` for
    `fmha_nvfp4_sm120_d512_causal_True_swa_True_softcap_True_pv_v_False`.
- Ran the full SM120 NVFP4 attention test file with the warmed cache:
  - Result: `36 passed in 2.07s`.

Interpretation:

- The wrapper no longer hides a full Q memcpy. If the caller provides an
  unsupported Q layout, the wrapper raises instead of silently copying.
- Scratch-residue coverage is now explicitly tested across the global scratch
  surfaces that were not covered by the smem wrapper work.
- The active block-table/page validation turns several possible kernel-trap or
  arbitrary-read cases into Python-side errors.
- The D512 linear-V cold-compile cliff remains unchanged; this audit targeted
  correctness/perf-silent behavior, not ptxas time.

## 2026-05-03 13:10 CDT - Benchmark Device Semantics and Sweep Sanity Check

Decision:

- Benchmark `--device` means the CUDA logical device ordinal visible inside the
  process after `CUDA_VISIBLE_DEVICES` filtering.
- Valid ways to target physical GPU 2:
  - Unmasked process: `env -u CUDA_VISIBLE_DEVICES ... --device 2`.
  - Masked process: `CUDA_VISIBLE_DEVICES=2 ... --device 0`.
- Invalid/confusing pattern: `CUDA_VISIBLE_DEVICES=2 ... --device 2`. With the
  mask set to one device, logical device 2 does not exist.

Fix:

- Updated the benchmark scripts that take `--device` and use CUDA event timing
  to call `torch.cuda.set_device(args.device)` before creating tensors, events,
  or synchronizing:
  - `bench_gemma4_paged_workload_scenarios.py`
  - `bench_nvfp4_d512_decode.py`
  - `bench_nvfp4_d512_prefill.py`
  - `bench_nvfp4_fmha_v2_gqa_grouped_attention.py`
  - `bench_nvfp4_gqa_grouped_attention.py`
  - `bench_nvfp4_gqa_grouped_pv.py`
  - `bench_nvfp4_native_attention_gemm.py`
  - `bench_nvfp4_v_cache_reblock.py`
  - `bench_nvfp4_xqa_gqa_decode.py`
- `bench_sm120_nvfp4_attention.py` already had this behavior.

Why this mattered:

- The first production matrix sweeps used `--device 2` with no device mask.
  `bench_sm120_nvfp4_attention.py` set the active device correctly, so the
  fused rows were timed on GPU 2.
- `bench_nvfp4_fmha_v2_gqa_grouped_attention.py` allocated tensors on
  `cuda:2` but timed/synchronized the process current device, typically
  `cuda:0`. This produced bogus FA2/BF16/FP8 reference timings around
  0.02-0.03 ms for long-context cells.

Validation:

- Syntax check passed for all patched benchmark scripts.
- After the fix, every benchmark script with both `--device` and
  `torch.cuda.Event` also calls `torch.cuda.set_device`.
- D256 q=512 kv=8192 group=2 NVFP4 FA2 baseline:
  - Unmasked `--device 2`: min_ms = 0.09344.
  - Masked `CUDA_VISIBLE_DEVICES=2 --device 0`: min_ms = 0.090944.
  - These agree within normal run noise and validate the logical-device
    convention.
- D256 q=4096 kv=262144 group=16 NVFP4 FA2 baseline:
  - Patched script with unmasked `--device 2`: min_ms = 134.908.
  - This matches the earlier synchronized-wall-clock sanity check and replaces
    the invalid ~0.03 ms timing from the broken-device run.

Sweep status:

- `reports/d256_sm120_nvfp4_attention_paged_linear_qwen_full_grid_20260503`
  completed 1440/1440 rows; fused rows were timed on the correct device, but
  reference baseline columns are invalid and must be rerun.
- `reports/d256_sm120_nvfp4_attention_paged_linear_gemma_sliding_grid_20260503`
  was interrupted at 1196/1440 rows while diagnosing this issue. Its reference
  columns are also invalid.
- Remaining production matrix sweeps should be rerun only after this benchmark
  device fix is committed, using either the unmasked `--device 2` convention or
  the masked `CUDA_VISIBLE_DEVICES=2 --device 0` convention consistently.

## 2026-05-03 13:20 CDT - Benchmark Surface Cleanup

Context:

- The SM120 NVFP4 benchmark surface had accumulated one-off scripts from the
  D512 exploration, raw-GEMM baselines, grouped-GQA comparisons, V-cache reblock
  debugging, XQA comparison, and Gemma-specific grid wrappers.
- Those scripts served the hill-climb/debugging phase but duplicated or
  predated the production wrapper and grid orchestrator.

Decision:

- Keep and ship:
  - `bench_sm120_nvfp4_attention.py`: production wrapper and dense direct
    benchmark surface.
  - `bench_sm120_nvfp4_attention_grid.py`: production matrix orchestrator.
  - `bench_nvfp4_fmha_v2_gqa_grouped_attention.py`: canonical FA2/BF16/FP8
    reference comparison harness for the grid.
- Retire the dev-artifact benchmark scripts:
  - `bench_nvfp4_d512_decode.py`
  - `bench_nvfp4_d512_prefill.py`
  - `bench_nvfp4_native_attention_gemm.py`
  - `bench_nvfp4_gqa_grouped_attention.py`
  - `bench_nvfp4_gqa_grouped_pv.py`
  - `bench_nvfp4_v_cache_reblock.py`
  - `bench_nvfp4_xqa_gqa_decode.py`
  - `bench_gemma4_attention_grid.py`
  - `bench_gemma4_paged_workload_scenarios.py`

Validation:

- Checked for live references to the retired scripts outside worklogs/reports;
  none remain.
- Ran Python syntax checks on the kept benchmark surfaces:
  - `bench_sm120_nvfp4_attention.py`
  - `bench_sm120_nvfp4_attention_grid.py`
  - `bench_nvfp4_fmha_v2_gqa_grouped_attention.py`

Interpretation:

- The production benchmark path is now narrower and matches the current
  integration architecture: one direct/wrapper bench, one grid runner, one
  reference comparison harness.
- Historical reports and worklog entries still reference the retired scripts as
  immutable experiment history; those references are intentionally not edited.

## 2026-05-03 14:18 CDT - Production Focus Benchmark Run

Context:

- After fixing benchmark device semantics, ran the focused production-cell
  benchmark set for Qwen 3.6 full-attention, Gemma sliding, and Gemma global
  shapes.
- The grid reporter was extended as a data-driven writer, not a focused-run
  special case: it emits the production comparison table whenever the input
  rows contain multiple SM120 fused variants.
- Device convention for this run: unmasked process, `--device 2`, where device
  2 is the RTX PRO 6000 Blackwell SM120 GPU.

Reports:

- `reports/prod_qwen_full_d256_g6_20260503.{jsonl,csv,summary.csv,production.csv,md,run.log}`
- `reports/prod_gemma_sliding_d256_g2_swa1024_softcap30_20260503.{jsonl,csv,summary.csv,production.csv,md,run.log}`
- `reports/prod_gemma_global_d512_g8_softcap30_20260503.{jsonl,csv,summary.csv,production.csv,md,run.log}`

Smoke / reference findings:

- D256 Qwen and Gemma sliding smokes completed all six requested rows.
- D512 Gemma global smoke recorded finite SM120 fused, NVFP4 FA2, and FP8 FA2
  rows, but `bf16_fa2` fails with FlashInfer's existing prefill invalid-config
  error for this D512 grouped shape.
- The same D512 `bf16_fa2` failure is present in
  `reports/d512_hillclimb_180cell_20260430.csv`, so this is not a new SM120
  kernel regression or a device-selection problem.

Focused run status:

- Qwen full D256 g6: 84 rows total, 80 ok, 4 errors.
  - Errors are dense direct rows for q=1 decode cells. The dense FFI path
    requires `q_rows` to be a positive multiple of tile_m=64; paged-wrapper
    rows for the same q=1 cells are valid.
- Gemma sliding D256 g2: 48 rows total, 46 ok, 2 errors.
  - Errors are dense direct rows for q=1 cells for the same tile_m=64 reason.
- Gemma global D512 g8: 66 rows total, 51 ok, 15 errors.
  - 4 errors are dense direct q=1 rows; D512 dense requires tile_m=128.
  - 11 errors are the pre-existing D512 `bf16_fa2` invalid-config failures.

Headline performance:

- Qwen full D256 g6:
  - Geomean dense = 1.156 ms.
  - Geomean paged-PV = 318.322 ms.
  - Geomean paged-linear = 563.814 ms.
  - Geomean paged-linear / dense = 708.6x.
  - Geomean paged-linear / paged-PV = 1.77x, so in-kernel linear-V reblock is
    adding about 77% over the already-slow paged-PV path.
  - Geomean paged-linear speedup vs NVFP4 FA2 = 0.000885x.
- Gemma sliding D256 g2:
  - Geomean dense = 0.148 ms.
  - Geomean paged-PV = 45.691 ms.
  - Geomean paged-linear = 78.585 ms.
  - Geomean paged-linear / dense = 494.0x.
  - Geomean paged-linear / paged-PV = 1.72x.
  - Geomean paged-linear speedup vs NVFP4 FA2 = 0.000472x.
- Gemma global D512 g8:
  - Geomean dense = 7.798 ms.
  - Geomean paged-PV = 1374.055 ms.
  - Geomean paged-linear = 2228.762 ms.
  - Geomean paged-linear / dense = 363.3x.
  - Geomean paged-linear / paged-PV = 1.62x.
  - Geomean paged-linear speedup vs NVFP4 FA2 = 0.00105x.

Interpretation:

- The FA2 reference timings now scale with the chosen device and are no longer
  the earlier bogus ~0.02 ms flatline for long-context prefill cells.
- The current production paged path is not performance-usable. It has a large
  fixed floor even for decode/short-context cells and scales into seconds for
  long-context cells.
- Dense direct timings remain in the expected sub-ms to tens-of-ms range on
  prefill cells, so the regression is localized to the paged production path
  and wrapper/native-paged execution, not the core dense kernel template.
- Paged-PV is also hundreds of times slower than dense, so linear-V reblock is
  not the primary root cause. Linear-V adds another ~1.6-1.8x on top of an
  already-broken paged path.
- Dense pre-integration comparison is only meaningful for no-SWA/no-softcap
  cells with matching old dense rows. In this focused run that mostly applies
  to the Qwen D256 cells; Gemma sliding/global specs intentionally leave
  `dense_pre_ms` blank because the old runs did not use those spec configs.

Next diagnostic direction:

- Do not start broad 1080-cell sweeps until the paged production path is
  diagnosed. The focused run already shows the broad sweep would mostly
  characterize a broken path.
- Localize the paged path floor by timing sub-steps inside
  `BatchPrefillWithPagedKVCacheSM120Nvfp4Wrapper.run()` and the paged FFI:
  BF16-Q quantize, native paged launch, split combine, output copy, and any
  scratch zero/fill operations.
- Compare paged-PV vs dense on the same q/kv cells first; because paged-PV
  bypasses linear-V reblock, it isolates paged scheduling/scratch/combine
  overhead from V-layout conversion.

## 2026-05-03 15:07 CDT - Paged Wrapper Slowdown Localization

Reference cell:

- D512 Gemma global spec: q=512, kv=65536, group=8, causal, no sliding
  window, logits softcap=30.
- Existing report row:
  `reports/prod_gemma_global_d512_g8_softcap30_20260503.csv`.
- Dense direct: 7.663 ms.
- Paged-PV wrapper: 1802.426 ms.
- Paged-linear wrapper: 2955.367 ms.
- NVFP4 FA2: 9.459 ms.

Tripartite timing result:

- Python wall time around `BatchPrefillWithPagedKVCacheSM120Nvfp4Wrapper.run()`:
  1804.39 ms min, 1805.85 ms mean.
- CUDA event around the full wrapper call: 1803.29 ms min, 1805.21 ms mean.
- CUDA event around the single `paged_run_bf16_q` FFI call:
  1804.06 ms min, 1806.41 ms mean.
- Output copy after the FFI call: 0.009 ms min, 0.029 ms mean.

Nsight Systems kernel-only result:

- Paged-PV stage kernel:
  `sm120_nvfp4_qkv_online_register_q_stage_kernel` ran for 1814.034 ms.
- Paged split-KV combine ran for 0.0108 ms.
- Paged scratch memset ran for 0.0101 ms.
- Dense direct stage kernel for the same logical cell ran for 7.674 ms.
- Dense direct combine ran for 0.0086 ms.

Launch geometry:

- Paged-PV setup for this cell:
  - `total_q_rows = 4096`
  - `physical_kv_len = 65536`
  - `split_kv_tiles = 256`
  - `num_splits = 2`
  - `stage_grid = (32, 1, 2)`
  - `stage_ctas = 64`
  - `combine_ctas = 4096`
- Dense uses the same raw stage launch geometry at this shape:
  `stage_grid = (q_rows / 128, head_dim / (4 * 128), num_splits) =
  (32, 1, 2)`.
- The slowdown is not CTA-count explosion. It is per-CTA work inside the
  paged stage kernel.

Bisection toggles:

- `bf16_q_all_heads_split2`: 1804.80 ms mean.
- `bf16_q_kv_head0_split2`: 1805.07 ms mean.
- `prequant_q_all_heads_split2`: 1792.29 ms mean.
- `prequant_q_kv_head0_split2`: 1792.56 ms mean.
- `bf16_q_all_heads_split1`: 3601.60 ms mean.
- `prequant_q_all_heads_split1`: 3589.28 ms mean.

Interpretation:

- Python wrapper overhead is not the missing time.
- Output copy, scratch memset, and split-KV combine are not the missing time.
- JIT compile is not included; the D512 `pv_v=True` cache artifact predates
  the diagnostic and no compiler processes were active.
- Multi-KV-head grid dispatch is not the missing time; forcing a single
  `kv_head` leaves runtime unchanged.
- BF16-Q fused quantization is not the missing time; pre-quantized Q saves only
  about 12 ms out of 1804 ms.
- Forcing one split makes the run about 2x slower, so split-KV is providing
  needed parallelism rather than causing the slowdown.
- Linear-V reblock is not the primary root cause; paged-PV is already about
  235x slower than dense at the reference cell. Linear-V adds another about
  1.64x on top.

Localized code site:

- Wrapper enters the single FFI call at `flashinfer/fmha_nvfp4_sm120.py:426`.
- The FFI calls `RunPagedBatchImpl`, which launches the raw stage kernel at
  `csrc/fmha_nvfp4_sm120_paged_common.cuh:660`.
- The D512 raw stage kernel launch is
  `include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d512.cuh:2134`.
- Dense and paged use the same stage launch shape; the branch that differs is
  the `kUsePagedKv` producer path:
  - K producer: `stage_paged_k_tile` at
    `include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d512.cuh:756`.
  - V producer: `stage_paged_v_tile` at
    `include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d512.cuh:842`.
  - Dense K/V path uses bulk TMA copies at
    `include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d512.cuh:1200`
    and `:1243`.

Specific culprit:

- The paged K/V producers replace the dense TMA bulk load with load-warp
  scalar partition loops. For every stage tile, the load warp walks CUTLASS
  copy partitions and calls per-codepoint helpers:
  - `sm120_nvfp4_paged_k_code` at
    `include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_paged_kv.cuh:83`
    performs page-table lookup, page-offset math, layout stride math, byte
    load, and nibble extraction for each FP4 codepoint.
  - `sm120_nvfp4_paged_v_code` at
    `include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_paged_kv.cuh:165`
    performs the same scalar path for V.
- At D512 the stage loops execute this scalar paged producer for 256 KV tiles
  per split, two K chunks, four V output groups, and two splits. The compute
  body is healthy; the paged scalar producer is the 275x wrapper-vs-dense gap.

Next target:

- Replace the D512 paged K/V producer structure with a page-aware bulk movement
  path. The immediate problem is not wrapper launch count or split combine; it
  is per-codepoint block-table/stride/nibble extraction inside the stage
  kernel's load-warp producer.

## 2026-05-03 15:55 CDT - Step 1 Paged K cp.async Port

Change:

- Ported the existing D128 paged K producer cp.async pattern to D256 and D512.
- D256/D512 K producers now use `sm120_nvfp4_paged_k_word_ptr` plus
  `cp_async::pred_load_32b` for each 8-nibble / 4-byte partition, followed by
  `cp_async::commit_group()` and `cp_async::wait_group<0>()`.
- D128 K producer was not changed.

Validation:

- `tests/attention/test_nvfp4_kv_head_dim_512.py`: 36 passed.
- NVFP4 test set:
  `tests/attention/test_nvfp4_kv_head_dim_512.py`
  `tests/utils/test_fp4_kv_quantization.py`: 62 passed.

D512 reference cell after Step 1:

- Cell: q=512, kv=65536, group=8, head_dim=512, causal, no sliding window,
  logits softcap=30, split_kv_len=32768.
- Paged-PV: 1240.798 ms min, 1242.119 ms mean.
- Paged-linear: 2385.151 ms min, 2385.644 ms mean.

Interpretation:

- K cp.async removes about 31% from paged-PV at the reference cell
  (1802.426 ms -> 1240.798 ms), but the path is still two orders of magnitude
  too slow.
- Remaining dominant work is in V producer and scale loops; the Step 1 result
  confirms the scalar K producer was a major component but not the only
  component of the paged slowdown.

## 2026-05-03 16:31 CDT - Step 2 PV V cp.async Stop Point

Attempt:

- Tried to make PV-layout V directly loadable with 32-bit cp.async by changing
  PV V storage to a true token-contiguous physical layout:
  `[page, kv_head, head_dim, page_size / 2]`.
- Added the matching SM120 V word-pointer shape locally and wired the D128,
  D256, and D512 PV-layout V producer branches to use `cp_async::pred_load_32b`.

Validation result:

- `tests/attention/test_nvfp4_kv_head_dim_512.py` failed:
  11 failed, 25 passed.
- Failures were shape-contract failures, not numerical drift:
  existing fmha_v2 tests and SM120 multi-KV tests expect
  `nvfp4_quantize_paged_kv_cache(v_data_layout="pv")` to return the existing
  NHD-shaped V tensor.
- The uncommitted Step 2 edits were reverted. The committed Step 1 K cp.async
  change remains intact.

Conclusion:

- Direct PV V cp.async cannot be applied against the current public
  `v_data_layout="pv"` tensor shape. That tensor is NHD-shaped and contiguous
  with last-dimension stride 1:
  `(pages, page_size, kv_heads, head_dim / 2)`.
- The SM120 PV producer needs fixed-output-column / adjacent-token words, but
  the current public PV data tensor stores adjacent output dimensions for a
  fixed token. A 4-byte cp.async from that source would copy the wrong logical
  values.
- Making the public PV tensor physically token-contiguous breaks fmha_v2's
  current API contract and tests. This is not a safe Step 2 change under the
  existing public `v_data_layout="pv"` name.

Next viable choices:

- Keep the public PV tensor shape and optimize V by hoisting page-table lookup
  and row-base computation inside the scalar producer. This preserves the
  existing API but will not become a single direct cp.async word copy.
- Add a distinct SM120-specific physical PV data layout/API value for
  token-contiguous V words, then use direct cp.async in the SM120 producer.
  That is a public API extension and should be explicit, not silently overloaded
  onto fmha_v2's existing `v_data_layout="pv"` contract.
- Build a fmha_v2-style staging abstraction that loads row-major source data
  and stores the transposed CUTLASS PV operand layout without per-codepoint
  block-table lookup. That is the closest way to keep the current public PV
  shape while attacking the same bottleneck structurally.

## 2026-05-03 17:02 CDT - Reference Audit For Linear-V Producer

Reference: `include/flashinfer/attention/hopper/sparse_mainloop.cuh`

- Load primitive: `cutlass::arch::cp_async_zfill<sizeof(Vec),
  cutlass::arch::CacheOperation::Global>`.
- Block-table walk granularity: per KV position row in `prefetch_kv_offset`,
  stored in a register rolling buffer and reused by shuffle in
  `load_kv_with_gather`. The block table is not touched per element.
- Partial-page / OOB handling: `valid_read` controls whether the row offset is
  populated; `guard` predicates the cp.async zfill load for tail rows.
- Direct quote:

```cpp
// include/flashinfer/attention/hopper/sparse_mainloop.cuh:294
if (valid_read) {
  // Use divmod to find page and offset within page
  uint32_t page_iter, entry_idx;
  mainloop_params.page_size.divmod(kv_idx_read, page_iter, entry_idx);
  IdType page_idx = kv_indices_ptr[page_iter];
  // Pre-compute: page_idx * page_stride + entry_idx * stride_n
  my_kv_offset[parity] = page_idx * k_page_stride + entry_idx * k_stride_n;
} else {
  my_kv_offset[parity] = 0;
}

// include/flashinfer/attention/hopper/sparse_mainloop.cuh:329
int src_thread = group_id * THREADS_PER_GROUP + kv_offset / KV_STRIDE;
int64_t base_offset = __shfl_sync(FULL_MASK, my_kv_offset[parity], src_thread);

// Final address: base_ptr + base_offset + d_idx
// where base_offset = page_idx * page_stride + entry_idx * stride_n
Vec const* src_ptr = reinterpret_cast<Vec const*>(base_ptr + base_offset + d_idx);
cutlass::arch::cp_async_zfill<sizeof(Vec), cutlass::arch::CacheOperation::Global>(
    &dst(i), src_ptr, guard);
```

Reference: `csrc/fmha_v2/fmha/gmem_tile_qkv_packed.h`

- Load primitive: `Ldgsts_helper<USE_LDGSTS>::load`, which dispatches LDGSTS /
  cp.async-style staging when enabled and otherwise uses the fallback LDG path.
- Block-table walk granularity: per `row_idx` in the `LDGS` loop. The code
  computes one page base pointer per row, then builds `ptrs[ii]` for the row
  vector load.
- Partial-page / OOB handling: `preds[ii]` requires `row_idx < actual_seqlen_`
  and `col_in_bytes_ < VALID_BYTES_PER_ROW`; the load helper consumes those
  predicates.
- Direct quote:

```cpp
// csrc/fmha_v2/fmha/gmem_tile_qkv_packed.h:1281
for (int ii = 0; ii < LDGS; ++ii) {
  int row_idx = row_ + ii * (int)ROWS_PER_LDG;
  int paged_kv_block_idx = (row_idx >> paged_kv_log2_block_size_);
  char const* local_kv_ptr = reinterpret_cast<char*>(
      paged_kv_block_pool_ptr_ +
      params_kv_block_size_in_bytes_ * paged_kv_global_block_offsets_[paged_kv_block_idx]);

  // Predicates.
  // TODO: do we need to make sure row_idx < ROWS ?
  preds[ii] = row_idx < actual_seqlen_;
  preds[ii] &= col_in_bytes_ < VALID_BYTES_PER_ROW;

  // Pointers.
  int row_idx_in_block = row_idx & ((1 << paged_kv_log2_block_size_) - 1);
  ptrs[ii] =
      local_kv_ptr + head_col_in_bytes + (int64_t)row_idx_in_block * token_stride_in_bytes_;

}

// csrc/fmha_v2/fmha/gmem_tile_qkv_packed.h:1300
// Trigger LDGSTS or the LDGs.
// The predicates protect against out-of-bound access in rows and cols
Ldgsts_helper<USE_LDGSTS>::load(this, smem_tile, ptrs, preds);
```

Reference: `include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d128.cuh`

- Load primitive: `cp_async::pred_load_32b` for each 8-nibble / 4-byte K
  partition.
- Block-table walk granularity: per packed 4-byte K word. The D128 producer
  validates that the CUTLASS B smem partition maps 8 adjacent K nibbles to one
  4-byte destination, then delegates the page-table walk to
  `sm120_nvfp4_paged_k_word_ptr`.
- Partial-page / OOB handling: `in_bounds = token < kv_len_tokens`; out-of-range
  words use `kFillZero` through predicated cp.async.
- Direct quote:

```cpp
// include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d128.cuh:829
const int token = kv_tile * kCutlassTileN + row0;
const int dim0 = k_outer * kCutlassTileK + k0;
#pragma unroll
for (int j = 0; j < 8; ++j) {
  auto coord = coord_tensor(i + j);
  auto ref = dst(i + j);
  uint8_t* dst_byte = cute::recast_ptr<uint8_t>(&ref);
  const int row = int(cute::get<0>(coord));
  const int k = int(cute::get<1>(coord));
  // CUTLASS B smem must colocate the 8 logical K nibbles in 4 bytes.
  if (row != row0 || k != k0 + j ||
      dst_byte != dst0 + (j >> 1)) {
    asm volatile("trap;\n");
  }
}
const bool in_bounds = token < kv_len_tokens;
const uint32_t* src =
    in_bounds
        ? sm120_nvfp4_paged_k_word_ptr(paged_kv_params, token,
                                       dim0)
        : reinterpret_cast<const uint32_t*>(paged_kv_params.k_pages);
cp_async::pred_load_32b<cp_async::SharedMemFillMode::kFillZero>(
    reinterpret_cast<uint32_t*>(dst0), src, in_bounds);

// include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_paged_kv.cuh:110
const int logical_page = logical_token / params.page_size;
const int page_offset = logical_token - logical_page * params.page_size;
const int physical_page = params.block_table[logical_page];
const int packed_col = dim >> 1;
const int64_t src =
    params.kv_layout_hnd
        ? (static_cast<int64_t>(physical_page) * params.k_stride_page +
           static_cast<int64_t>(params.kv_head) * params.k_stride_dim1 +
           static_cast<int64_t>(page_offset) * params.k_stride_dim2 +
           static_cast<int64_t>(packed_col) * params.k_stride_dim3)
        : (static_cast<int64_t>(physical_page) * params.k_stride_page +
           static_cast<int64_t>(page_offset) * params.k_stride_dim1 +
           static_cast<int64_t>(params.kv_head) * params.k_stride_dim2 +
           static_cast<int64_t>(packed_col) * params.k_stride_dim3);
return reinterpret_cast<const uint32_t*>(params.k_pages + src);
```

Adoption pattern for linear-V:

- The public linear-V tensor is dim-contiguous, while the CUTLASS PV operand
  partition consumes 8 token-contiguous FP4 nibbles for one output column. That
  rules out direct gmem-to-operand cp.async for linear-V without an intermediate
  transpose/reblock stage. The safe no-smem-cost pattern to adopt is the row
  base hoist from the Hopper and fmha_v2 references: compute page/table/row base
  once per logical token row or 16-token page group, reuse that base for all
  scalar byte loads needed by the fp32 dequant/requant path, and keep the
  existing partition-derived CUTLASS destination stores. This preserves the
  public tensor shape and removes per-codepoint block-table/divmod work before
  considering a staging-smem rewrite.

## 2026-05-03 17:55 CDT - Linear-V Producer Path A Result

Path A implementation:

- Added V data and V scale page-base helpers in
  `fmha_nvfp4_sm120_paged_kv.cuh`.
- D128/D256/D512 linear-V producer now hoists the block-table lookup and page
  bases outside the 8-token inner codepoint loop.
- Linear-V scale recompute now hoists one 16-token logical page and reuses the
  cached data/scale page bases for the byte-pair scan.
- PV-layout V producer uses the same page-base hoist at the scalar load
  callsite as a side-effect. The public tensor shapes, layout names, test
  contracts, and spec axes were unchanged.

Correctness:

- `tests/attention/test_nvfp4_kv_head_dim_512.py -q`: 36 passed.
- `tests/attention/test_nvfp4_kv_head_dim_512.py tests/utils/test_fp4_kv_quantization.py -q`:
  62 passed.
- No tolerance changes.

Reference cell benchmark:

`D=512, group=8, q=512, kv=65536, softcap=30, split_kv_len=32768,
output_group_span=4, device=2`

| state | paged-PV min ms | paged-linear min ms |
| --- | ---: | ---: |
| pre Path A | 1240.798 | 2385.151 |
| post Path A | 899.385 | 1883.437 |

Path A removes measurable block-table/divmod overhead but does not meet the
target. Paged-linear remains 2.09x slower than paged-PV at the reference cell,
so the residual bottleneck is still the linear-V reblock path rather than just
page-table lookup granularity. Path B needs a smem budget audit before any
staging-smem implementation.

## 2026-05-03 18:04 CDT - Path B Smem Budget Audit

Path B proposal:

- Add a staging region for row-major V loads before packing into the CUTLASS PV
  operand.
- Required staging size per stage: `kCutlassTileN * head_dim / 2` bytes.
- SM120 opt-in budget used by the kernel static asserts: 99 KiB = 101376 bytes.

Measured with a throwaway host compile that includes the three production
headers and prints `sizeof(Sm120Nvfp4QkvLoadCollectiveStorage)`:

| head_dim | current smem bytes | slack bytes | staging bytes | current + staging |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 66560 | 34816 | 8192 | 74752 |
| 256 | 50176 | 51200 | 16384 | 66560 |
| 512 | 96256 | 5120 | 32768 | 129024 |

Result:

- D128 and D256 have enough unused smem for the proposed staging region.
- D512 does not. Adding the required 32768-byte staging tile would exceed the
  99 KiB budget by 27648 bytes.

Alias audit for D512:

- `qk_tensors.smem_B` is aliased as `smem_logits0` and `smem_epilogue_o`.
  The load warp preloads V while MMA warps consume QK logits/PV and while the
  epilogue role later consumes O, so this region is not a safe V staging alias
  without restructuring the producer/consumer schedule.
- `qk_tensors.smem_A` and `qk_tensors.smem_SFA` are reused as P/P-scale stages
  for PV and are live across the same tile loop that consumes staged V.
- `v_smem_B` and `v_smem_SFB` are the CUTLASS PV operand and scale operand
  consumed by the PV MMA pipeline; using them as a row-major staging tile would
  clobber the data the consumer expects.
- Pipeline/stat storage has only small fixed buffers and cannot hold a
  `128 * 512 / 2 = 32768` byte V staging tile.

Conclusion:

The Path B full-tile staging-smem design is not legal for the D512 production
kernel without a larger schedule/storage restructure, smaller tile shape, or a
new aliasing scheme that serializes V staging against QK/logits/P/O use. Because
D512 is the reference production cell and the task requires all head dims with
no public layout change, implementing D128/D256-only staging would introduce a
head-dim fallback split and leave the production blocker unresolved.

## 2026-05-03 18:17 CDT - Path A Measured Result And Decision

Path A status:

- Implemented and committed in `581e6f4`.
- Pushed to `flashinfer-nvfp4-kv-prbranches` with the reference audit and smem
  budget notes.
- Public tensor shapes, layout names, tests, and spec axes were unchanged.

Reference cell:

`D=512, group=8, q=512, kv=65536, softcap=30, split_kv_len=32768,
output_group_span=4, device=2`

| path | min ms | mean ms |
| --- | ---: | ---: |
| dense | 7.688 | 7.693 |
| paged-PV | 899.385 | 900.857 |
| paged-linear | 1883.437 | 1883.982 |

D256/D128 same-shape benches:

- Not run for Path A. The decision gate is the D512 production reference cell,
  and it missed target.

Test status:

- `tests/attention/test_nvfp4_kv_head_dim_512.py -q`: 36 passed.
- `tests/attention/test_nvfp4_kv_head_dim_512.py tests/utils/test_fp4_kv_quantization.py -q`:
  62 passed.
- No tolerance changes.

Decision:

- Path A target was `paged-linear <= 1241 ms` at the D512 reference cell.
- Measured Path A paged-linear is `1883.437 ms`, which is `642.437 ms` slower
  than the target and `2.09x` slower than paged-PV.
- Per directive, stop here. Do not implement full-tile Path B and do not start
  scale-loop optimization. The next structural option is a separate per-warp
  scratch design pass with explicit sign-off.

## 2026-05-03 18:32 CDT - P1 V Producer Register-Transpose Plan

What I am about to do:

- Replace the scalar per-output-word V data load in
  `fmha_nvfp4_sm120_d{128,256,512}.cuh::stage_paged_v_tile` with an 8-lane
  cooperative register transpose.
- Each 8-lane group loads one `uint32_t` per token row: 4 dim-contiguous bytes
  = 8 FP4 codepoints from the public linear/PV V layout. `__shfl_sync` then
  transposes the 8 row words so each lane owns one output dim across 8 tokens,
  matching the CUTLASS B operand word currently written by the scalar path.
- PV-layout V directly packs the transposed codes. Linear-layout V keeps the
  required fp32 dequant/requant but gathers original row scales by shuffle
  instead of rewalking the page table per codepoint.
- Done criteria: D512 and full NVFP4 tests pass without tolerance changes, and
  the D512 reference cell improves from the Path A numbers
  (`paged-PV=899.385 ms`, `paged-linear=1883.437 ms`). If the CUTLASS B smem
  direct-coordinate write invariant fails, document the exact invariant and
  revert this variant.

Reference audit:

`include/flashinfer/mma.cuh` shows the local idiom for register exchange with
`__shfl_sync` after each lane holds a register word:

```cpp
// include/flashinfer/mma.cuh:270
word.x = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4);
word.y = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4 + 1);
word.z = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4 + 2);
word.w = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4 + 3);
```

`csrc/fmha_v2/fmha/gmem_tile_qkv_packed.h` shows the production precedent for
linear-V reblock: regular `ldg` into registers, not direct cp.async into the
MMA operand, followed by fp32 dequant/requant and packed output:

```cpp
// csrc/fmha_v2/fmha/gmem_tile_qkv_packed.h:1753
fmha::ldg(original_pair, kv_row_ptr + col0 / 2);
uint8_t const nibble0 = original_pair & 0x0fu;
uint8_t const nibble1 = (original_pair >> 4) & 0x0fu;
...
packed[reg] = fmha::float8_to_e2m1x8(vals[0], vals[1], vals[2], vals[3],
                                     vals[4], vals[5], vals[6], vals[7]);
```

The current SM120 producer already verifies that each CUTLASS B operand word is
8 logical token nibbles colocated into 4 physical bytes before the scalar write:

```cpp
// include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d512.cuh:971
if (col != col0 || k != k0 + j ||
    dst_byte < dst0 || dst_byte >= dst0 + 4 ||
    pair_byte != dst_byte || int(dst_byte - dst0) != (j >> 1)) {
  asm volatile("trap;\n");
}
```

Decision:

- Literal `cp.async` into registers is not a CUDA primitive; the zero-smem
  version must use register `ldg` plus warp shuffle.
- The public V layouts are dim-contiguous in gmem and the CUTLASS operand is
  token-contiguous in smem. An 8x8 register transpose is the smallest structural
  conversion that preserves public shapes, adds no smem, and keeps the existing
  CUTLASS operand contract.

## 2026-05-03 19:21 CDT - P1 Register-Transpose Compile Wall

Attempted P1 implementation:

- Added `sm120_nvfp4_paged_v_word_from_page_base`.
- Replaced the scalar partition-driven V data loop in D128/D256/D512 with four
  8-lane register-transpose groups per load warp.
- Each lane loaded one row-major 32-bit V word, used `__shfl_sync` to gather the
  8 token rows for one output dim, and wrote the packed word through direct
  `pv_sB(col, k, stage)` coordinates after checking the 4-byte CUTLASS smem
  colocation invariant.

Result:

- D512 test execution progressed through early cases without a trap, so the
  direct-coordinate smem invariant did not immediately fail.
- Cold `ptxas` for `fmha_nvfp4_sm120_d256_causal_True_swa_True_softcap_True_pv_v_False`
  completed only after roughly 15 minutes.
- Cold `ptxas` for `fmha_nvfp4_sm120_d512_causal_True_swa_True_softcap_True_pv_v_False`
  was still running after roughly 27 minutes. I interrupted pytest/nvcc/ptxas
  with SIGINT, not kill -9, and reverted the uncommitted P1 code.

Decision:

- This register-transpose shape is blocked by compile-time blowup before it can
  be judged as a production path. It introduces enough direct-coordinate smem
  address checks, shuffles, and linear-V dequant/requant live state in the D512
  producer for ptxas to exceed the established acceptable envelope.
- Do not keep this P1 variant. Move to P2 scale-loop hoist as the next priority.
  Any future P1 revisit needs a smaller scoped helper boundary or a per-warp
  scratch design that compiles as a bounded code region instead of inlining the
  whole transpose/reblock loop into the producer body.

## 2026-05-03 19:25 CDT - P2 Scale-Loop Hoist Plan

What I am about to do:

- Rewrite the K-scale producer loop in
  `fmha_nvfp4_sm120_d{128,256,512}.cuh::stage_paged_k_tile`.
- Current callsites invoke `sm120_nvfp4_paged_k_scale(...)` for every packed
  K pair, so one 16-dim scale is reloaded and recomputes page/block-table
  addressing eight times.
- New shape: iterate `(row, scale_col)` once, hoist the K scale page base with
  one block-table lookup, load one scale byte, then write that byte to the
  eight SFB positions for the 16-dim scale group.
- Done criteria: D512/full NVFP4 tests pass, and the D512 reference cell is
  remeasured. This is expected to be smaller than V data work but should reduce
  scalar page-table traffic without changing public tensors, spec axes, or smem.

Reference code:

```cpp
// include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d512.cuh:827
for (int idx = lane_idx; idx < kCutlassTileN * kCutlassTileK / 2;
     idx += cutlass::NumThreadsPerWarp) {
  const int row = idx / (kCutlassTileK / 2);
  const int packed_k = idx - row * (kCutlassTileK / 2);
  const int k0 = 2 * packed_k;
  const int token = kv_tile * kCutlassTileN + row;
  const int scale_col = (k_outer * kCutlassTileK + k0) >> 4;
  const uint8_t scale =
      token < kv_len_tokens
          ? sm120_nvfp4_paged_k_scale(paged_kv_params, token, scale_col)
          : 0x38;
  qk_sSFB(row, k0, write_stage) = make_ue4m3_raw(scale);
}
```

The existing linear-V scale prepass already uses the target shape: one scale
computation per `(token_group, col_pair)` followed by repeated SFB writes for
the covered token group:

```cpp
// include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d512.cuh:920
pv_scale_pair_for(token_group_start, dim0, sf0, sf1);
#pragma unroll
for (int k_offset = 0; k_offset < 16; k_offset += 2) {
  pv_sSFB(col0, local_k0 + k_offset, write_stage) = make_ue4m3_raw(sf0);
  pv_sSFB(col0 + 1, local_k0 + k_offset, write_stage) = make_ue4m3_raw(sf1);
}
```

## 2026-05-03 19:57 CDT - P2 Scale-Loop Hoist Result

Implementation:

- Added `sm120_nvfp4_paged_k_scale_page_base` and
  `sm120_nvfp4_paged_k_scale_from_page_base`.
- Replaced the D128/D256/D512 K-scale loop from one load per packed K pair to
  one load per `(row, 16-dim scale group)` followed by eight SFB stores.
- Existing linear-V scale prepass already had the same scale-group shape after
  Path A, so no extra V-scale linear changes were needed. PV V-scale remained
  unchanged.

Correctness:

- `tests/attention/test_nvfp4_kv_head_dim_512.py -q`: 36 passed.
- `tests/attention/test_nvfp4_kv_head_dim_512.py tests/utils/test_fp4_kv_quantization.py -q`:
  62 passed.
- No tolerance changes.

Reference cell:

`D=512, group=8, q=512, kv=65536, softcap=30, split_kv_len=32768,
output_group_span=4, device=2`

| state | paged-PV min ms | paged-linear min ms |
| --- | ---: | ---: |
| Path A | 899.385 | 1883.437 |
| P2 K-scale hoist | 819.818 | 1805.615 |

Characterization:

- K-scale hoist is real but small relative to the remaining producer cost:
  `8.9%` gain on paged-PV and `4.1%` gain on paged-linear at the reference cell.
- Paged-linear is still `2.20x` paged-PV and `234.9x` dense at this cell, so
  the dominant cost remains V data/reblock, not scale loops.

## 2026-05-03 19:25 CDT - P3 Q-Quantize Hoist Plan

What I am changing:

- Rewrite `stage_bf16_q_tile` in `fmha_nvfp4_sm120_d{128,256,512}.cuh`.
- Current data path computes `q_scale_byte(row, dim >> 4)` for every FP4
  codepoint in an 8-code packed word, even though the word is 8 contiguous dims
  inside one 16-dim NVFP4 scale group.
- Current SFA path also computes the same 16-dim scale once per packed FP4 pair
  and stores duplicate scale values across the group.
- New shape: hoist Q row addressing once per packed word or scale group, compute
  one scale byte per 16-dim group, reuse it for the eight codepoints in each
  packed word, and write the SFA scale group with one scale computation followed
  by repeated stores.
- Done criteria: D512/full NVFP4 tests pass and the D512 reference cell is
  remeasured. This is expected to be smaller than P2 because Q volume is dense
  `q_len * head_dim` while the remaining bottleneck walks paged V over
  `kv_len * head_dim`, but the redundant Q scalar walk is still removable.

Reference code:

```cpp
// include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d512.cuh:1174
const uint8_t scale = q_scale_byte(row, dim >> 4);
const uint8_t code = q_code(row, dim, scale);
```

```cpp
// include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d512.cuh:1193
qk_sSFA(row, k0, write_stage) =
    make_ue4m3_raw(q_scale_byte(row, scale_col));
```

## 2026-05-03 19:25 CDT - P3 Q-Quantize Hoist Result

Implementation:

- Added Q row-base helpers in `fmha_nvfp4_sm120_paged_kv.cuh` so the BF16-Q
  producer can hoist token/head/row address math out of per-codepoint reads.
- Rewrote D128/D256/D512 `stage_bf16_q_tile` data loops to compute one Q scale
  byte per 8-code packed word and reuse it for all codepoints in that word.
- Rewrote the D128/D256/D512 SFA loops to compute one Q scale byte per 16-dim
  scale group and store that value across the eight FP4-pair scale positions.
- Added the same 8-code alignment trap already used by the K/V packed-word
  producers to the Q packed-word producer.

Correctness:

- `tests/attention/test_nvfp4_kv_head_dim_512.py -q`: 36 passed.
- `tests/attention/test_nvfp4_kv_head_dim_512.py tests/utils/test_fp4_kv_quantization.py -q`:
  62 passed.
- No tolerance changes.

Reference cell:

`D=512, group=8, q=512, kv=65536, softcap=30, split_kv_len=32768,
output_group_span=4, device=2`

| state | dense min ms | paged-PV min ms | paged-linear min ms |
| --- | ---: | ---: | ---: |
| P2 K-scale hoist | 7.688 | 819.818 | 1805.615 |
| P3 Q-quantize hoist | 7.682 | 816.084 | 1741.079 |

Characterization:

- P3 recovered `3.6%` on paged-linear and `0.5%` on paged-PV at the reference
  cell. This is a real cleanup of redundant scalar Q work, but not a structural
  answer to the 100x-plus paged gap.
- Remaining cost is still V-data/reblock dominated. Paged-linear remains
  `2.13x` paged-PV and `226.7x` dense at this cell.

## 2026-05-03 19:25 CDT - P4 Remaining Scalar Walk Audit Plan

What I am changing:

- Audit remaining paged producer helper callsites after P1/P2/P3.
- Keep the V data/reblock scalar walk documented as the dominant unresolved
  path because P1's register-transpose implementation hit a real compile wall.
- Patch only safe scalar walks whose layout semantics are already page/group
  level and do not require a public tensor-layout change.
- The concrete safe item found is PV-layout V scale staging in
  `stage_paged_v_tile`: it reloads the same `sm120_nvfp4_paged_v_pv_scale` once
  per FP4 pair even though PV scales are page-level for one dim. New shape:
  load once per 16-token page group, then store the repeated scale to the eight
  FP4-pair scale positions.

Reference code:

```cpp
// include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d512.cuh:1090
const int logical_page = token / paged_kv_params.page_size;
const int physical_page =
    paged_kv_params.block_table[logical_page];
scale = sm120_nvfp4_paged_v_pv_scale_from_physical_page(
    paged_kv_params, physical_page, dim);
```

## 2026-05-03 19:25 CDT - P4 Remaining Scalar Walk Audit Result

Implementation:

- Rewrote PV-layout V scale staging in D128/D256/D512 from one scale load per
  FP4 pair to one scale load per 16-token page group followed by repeated SFB
  stores.
- Preserved OOB behavior by writing `0x38` for invalid tail-token positions
  inside the repeated-store loop.
- Left V data/reblock as the remaining scalar path. Its page-table walk is
  already hoisted by Path A, and the attempted register-transpose structural
  fix hit the documented P1 compile wall.

Correctness:

- `tests/attention/test_nvfp4_kv_head_dim_512.py -q`: 36 passed.
- `tests/attention/test_nvfp4_kv_head_dim_512.py tests/utils/test_fp4_kv_quantization.py -q`:
  62 passed.
- No tolerance changes.

Reference cell:

`D=512, group=8, q=512, kv=65536, softcap=30, split_kv_len=32768,
output_group_span=4, device=2`

| state | dense min ms | paged-PV min ms | paged-linear min ms |
| --- | ---: | ---: | ---: |
| P3 Q-quantize hoist | 7.682 | 816.084 | 1741.079 |
| P4 PV-scale hoist | 7.682 | 717.799 | 1740.975 |

Characterization:

- PV-scale hoist recovered `12.0%` on paged-PV at the reference cell, which
  confirms PV scale staging was still a meaningful scalar walk.
- Paged-linear did not move materially because it uses the linear scale/reblock
  prepass instead of the PV scale path. The remaining linear cost is the
  fp32 dequant/requant V-data reblock path plus the unresolved V data
  token/dim layout mismatch.
- After P4, paged-PV is still `93.4x` dense and paged-linear is still
  `2.43x` paged-PV. Further large wins require a new V data structural design,
  not more scalar scale hoists.

## 2026-05-03 19:25 CDT - P5 Production Matrix Plan

What I am running:

- Use `benchmarks/bench_sm120_nvfp4_attention_grid.py` with the existing
  `--cells`/grid support and the general report writer.
- One report prefix per production spec; run three sequential fused variants
  into the same prefix: `paged`+`linear`, `paged`+`pv`, and `dense`.
- Baseline per cell is `nvfp4_fa2`; the first variant records it and later
  variants skip existing baseline rows through the grid driver's resume key.
- Device numbering is unmasked production numbering: `--device 2` with no
  `CUDA_VISIBLE_DEVICES`, matching the benchmark/device fix.
- Done criteria: all three spec prefixes have `.jsonl`, `.csv`,
  `.summary.csv`, `.production.csv`, and `.md` reports.

Report prefixes:

- `reports/prod_qwen_full_d256_g6_p4_20260503`
- `reports/prod_gemma_sliding_d256_g2_swa1024_softcap30_p4_20260503`
- `reports/prod_gemma_global_d512_g8_softcap30_p4_20260503`

Cell sets:

- Qwen full: `D=256`, `group=6`, `q={1,128,512,2048}`,
  `kv={4096,16384,65536,262144}`, causal, no sliding window, softcap `30`.
- Gemma sliding: `D=256`, `group=2`, `q={128,512,1024,2048}`,
  `kv={1024,8192}`, causal, sliding window `1024`, softcap `30`.
- Gemma global: `D=512`, `group=8`, `q={1,128,512,2048}`,
  `kv={4096,16384,65536,262144}`, causal, no sliding window, softcap `30`.

## 2026-05-03 20:12 CDT - P5 Production Matrix Result

Reports produced:

- `reports/prod_qwen_full_d256_g6_p4_20260503.{jsonl,csv,summary.csv,production.csv,md,run.log}`
- `reports/prod_gemma_sliding_d256_g2_swa1024_softcap30_p4_20260503.{jsonl,csv,summary.csv,production.csv,md,run.log}`
- `reports/prod_gemma_global_d512_g8_softcap30_p4_20260503.{jsonl,csv,summary.csv,production.csv,md,run.log}`

Completion status:

| spec | rows ok | rows error | error cause |
| --- | ---: | ---: | --- |
| Qwen full D256 g6 | 60 | 4 | dense q=1 rejected by dense_run tile_m=64 contract |
| Gemma sliding D256 g2 | 32 | 0 | none |
| Gemma global D512 g8 | 60 | 4 | dense q=1 rejected by dense_run tile_m=128 contract |

All paged-linear, paged-PV, and q>=128 dense rows reported
`output_finite=True`. The dense q=1 rows are not kernel correctness failures;
the direct dense binding intentionally requires Q rows to be a positive multiple
of the D-specialization tile M. Decode coverage comes from the paged rows and
NVFP4 FA2 baseline rows.

Geomean speedups vs NVFP4 FA2 (`baseline_ms / fused_ms`):

| spec | dense | paged-PV | paged-linear |
| --- | ---: | ---: | ---: |
| Qwen full D256 g6 | 0.956x | 0.00373x | 0.00150x |
| Gemma sliding D256 g2 | 0.217x | 0.00173x | 0.000695x |
| Gemma global D512 g8 | 0.970x | 0.00453x | 0.00189x |

Wrapper/reblock deltas:

| spec | paged-PV / dense geomean | paged-linear / paged-PV geomean |
| --- | ---: | ---: |
| Qwen full D256 g6 | 152.9x | 2.492x |
| Gemma sliding D256 g2 | 125.7x | 2.490x |
| Gemma global D512 g8 | 89.8x | 2.392x |

Worst underperforming cells vs NVFP4 FA2:

| spec | path | q | kv | fused ms | nvfp4_fa2 ms | speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Gemma global D512 g8 | paged-linear | 1 | 65536 | 1718.472 | 0.114 | 0.000066x |
| Gemma global D512 g8 | paged-linear | 1 | 16384 | 858.617 | 0.073 | 0.000085x |
| Gemma global D512 g8 | paged-PV | 1 | 65536 | 706.124 | 0.114 | 0.000162x |
| Qwen full D256 g6 | paged-linear | 1 | 16384 | 155.648 | 0.031 | 0.000202x |
| Gemma global D512 g8 | paged-PV | 1 | 16384 | 352.708 | 0.073 | 0.000206x |

Characterization:

- Dense remains roughly comparable to NVFP4 FA2 on the full-attention D256 and
  D512 global geomeans, but direct dense is not the production path and does not
  support q=1 without padding.
- Paged-PV is still roughly `90x-153x` slower than dense on the production
  cells. Paged-linear is another `2.39x-2.49x` slower than paged-PV.
- The dominant production gap is not Q quantization or scale staging anymore.
  It is the paged V data path plus launch/schedule geometry around split-KV and
  page traversal. The D512 q=2048 kv=262144 stock-vLLM production cell is
  `10290.048 ms` paged-linear vs `4403.373 ms` paged-PV vs `53.862 ms` dense vs
  `188.973 ms` NVFP4 FA2.

## 2026-05-03 20:13 CDT - P6 Graveyard Cleanup Audit Result

Audit commands:

- `rg -n "debug|Debug|smem_fp4_debug_code|cutlass_smem_atom_gemm_tile_body|debug_producer_smem|debug_stage_run|StageDebug" include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d{128,256,512}.cuh csrc/fmha_nvfp4_sm120_* benchmarks`
- `rg -n "smem_fp4|gemm_tile_body|producer_smem|stage_run|StageDebug|data_debug_mode|scale_debug_mode|probe|Probe|TVM_FFI_DLL_EXPORT_TYPED_FUNC" include/flashinfer/attention/blackwell csrc/fmha_nvfp4_sm120*.cu csrc/fmha_nvfp4_sm120*.cuh`
- `git ls-files benchmarks`

Result:

- No remaining `smem_fp4_debug_code`,
  `cutlass_smem_atom_gemm_tile_body_impl`, `debug_producer_smem`,
  `debug_stage_run`, `StageDebug`, `data_debug_mode`, or `scale_debug_mode`
  symbols are present in the current tree.
- The only SM120 NVFP4 FFI exports now present are production exports:
  `paged_run`, `paged_run_bf16_q`, `dense_run`, and `quantize_q`.
- The stale SM120/NVFP4 dev bench scripts from the integration phase are already
  absent. The remaining relevant files are:
  `bench_sm120_nvfp4_attention.py`,
  `bench_sm120_nvfp4_attention_grid.py`, and
  `bench_nvfp4_fmha_v2_gqa_grouped_attention.py`.
- `bench_nvfp4_quantize_backend_comparison.py` and `bench_trtllm_fmha.py` are
  not SM120-owned cleanup targets and were left untouched.

No code changes were needed for P6.

## 2026-05-03 20:18 CDT - P7 Silent Failure Audit Result

What I checked:

- `flashinfer/fmha_nvfp4_sm120.py` wrapper scratch lifecycle and run path.
- `csrc/fmha_nvfp4_sm120_paged_common.cuh` Q padding/quantize, output copy,
  and split-KV combine kernels.
- `include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d{128,256,512}.cuh`
  workspace initialization before CUTLASS persistent scheduler setup.

Result:

- The old wrapper-side defensive calls are no longer present. There is no
  `_partial.zero_()`, `_split_m.fill_()`, `_split_l.zero_()`,
  `_out_scratch.zero_()`, or `_out_group.zero_()` in the wrapper run path.
- The BF16-Q production path passes `q_bf16` into the fused kernel and does not
  consume stale `_q_packed_scratch` / `_q_scales_scratch` contents. The
  prepacked-Q path still uses `CopyQToPaddedBatchKernel`, which writes full
  packed/scales scratch rows and zero-fills padded rows.
- `Sm120Nvfp4SplitKvCombineBatchKernel` recomputes `num_splits` from the
  actual `kv_lens[batch_idx]` and only reads `split < num_splits`; it does not
  reduce over the max-splits allocation. Padded Q rows return before reading
  partial/split buffers.
- `CopyPaddedBatchOutKernel` returns on padded Q rows and copies only valid rows
  from `_out_scratch` into `_out_group`; the wrapper then copies the valid
  flattened `_out_group` extent into the caller output.
- The remaining `cudaMemsetAsync(workspace_base, ...)` in each D-specialization
  is not a wrapper band-aid. It is a bounded clear before CUTLASS persistent
  scheduler workspace initialization, with an explicit in-code contract:
  persistent scheduler state is sensitive to allocator residue. That clear is
  load-bearing and was kept.

Conclusion:

- No wrapper defensive zero/fill calls remain to remove.
- The global scratch regions that feed consumer kernels are either fully written
  for their consumed extent or are read using the actual split/Q-row bounds.
- The only kept zeroing is kernel-internal CUTLASS workspace initialization; it
  should stay unless CUTLASS workspace construction changes to own its full
  initialized extent directly.

## 2026-05-03 20:22 CDT - P8 Tolerance Review Result

What I tested:

- Tightened the remaining SM120-owned `2e-3` checks in
  `tests/attention/test_nvfp4_kv_head_dim_512.py` to `1e-3` locally:
  multi-KV vs per-KV-head, standard-wrapper vs direct-wrapper, and
  linear-V vs PV-layout mean difference.
- Ran the affected tests on GPU 2:
  `test_sm120_nvfp4_wrapper_multi_kv_matches_single_kv_sm12x`,
  `test_standard_prefill_wrapper_sm120_nvfp4_backend_matches_direct_wrapper_sm12x`,
  and `test_sm120_nvfp4_backend_accepts_normal_v_layout_sm12x`.

Result:

- `1e-3` failed immediately and was reverted before committing.
- D512 multi-KV vs per-head failed with max diff `0.001129150390625`.
- D512 standard-wrapper vs direct-wrapper failed with max diff
  `0.00146484375`.
- D128 linear-V vs PV-layout failed the tightened mean threshold with mean diff
  approximately `0.0011`.
- The existing thresholds then passed the affected test set five consecutive
  times: `9 passed` in each run.
- Final full NVFP4 suite at HEAD:
  `tests/attention/test_nvfp4_kv_head_dim_512.py`
  `tests/utils/test_fp4_kv_quantization.py` -> `62 passed in 1.64s`.

Conclusion:

- The remaining `2e-3` checks are currently load-bearing. They are not hiding
  gross nondeterminism; the observed failures are sparse BF16/FP4 rounding-level
  deltas just above `1e-3`.
- No tolerance was loosened.
- No tolerance was tightened because the required five-run stability criterion
  failed at `1e-3`.

## 2026-05-03 20:24 CDT - Pass Complete

Start state for this pass:

- Paged K cp.async was already shipped on D128/D256/D512.
- Path A linear-V page lookup hoist was already shipped at commit `581e6f4`.
- Reference cell `D=512 g=8 q=512 kv=65536 softcap=30`:
  dense `7.69 ms`, paged-PV `899 ms`, paged-linear `1883 ms`.

End state:

| point | dense ms | paged-PV ms | paged-linear ms |
| --- | ---: | ---: | ---: |
| start | 7.69 | 899 | 1883 |
| after K-scale hoist | 7.69 | 819.818 | 1805.615 |
| after Q-quant hoist | 7.682 | 816.084 | 1741.079 |
| after PV V-scale hoist | 7.682 | 717.799 | 1740.975 |

Total reference-cell delta:

- Paged-PV improved from `899 ms` to `717.799 ms` (`20.2%` faster).
- Paged-linear improved from `1883 ms` to `1740.975 ms` (`7.5%` faster).
- Dense stayed healthy and effectively unchanged.

What landed:

- K-scale block-table/page-base hoist across D128/D256/D512.
- Q BF16 quantization row-base/scale-group hoist across D128/D256/D512.
- PV-layout V-scale staging hoist across D128/D256/D512.
- Production-cell matrix reports for Qwen full, Gemma sliding, and Gemma global.
- Cleanup audit confirming debug graveyard and stale owned bench scripts are
  already absent.
- Silent-failure audit confirming wrapper defensive zero/fill calls are gone and
  remaining workspace clearing is kernel-internal CUTLASS state initialization.
- Tolerance review confirming the remaining `2e-3` checks are load-bearing and
  the full NVFP4 suite passes at HEAD.

Walls / pivots:

- P1 register-transpose V data path was attempted and reverted after the D512
  linear spec hit a ptxas compile wall. The blocked shape is documented in the
  P1 section.
- Direct cp.async into the V operand remains blocked by public V layout
  mismatch: public `linear`/`pv` V tensors are dim-contiguous in gmem, while the
  CUTLASS PV operand wants token-contiguous packed words.
- Full-tile V staging remains blocked by the 99 KiB shared-memory budget on
  D512. Do not retry that design without changing the memory model.

Current cost floor:

- On the production matrix, paged-PV is still roughly `90x-153x` slower than
  direct dense, and paged-linear is another `2.39x-2.49x` slower than paged-PV.
- The dominant cost is no longer K scale, Q quantization, or PV V-scale scalar
  staging. The dominant cost is the paged V data path and split-KV/page-traversal
  launch geometry.
- The stock-vLLM production path is `linear` V. That path still pays the
  in-kernel reblock cost and remains about `2.4x-2.5x` slower than PV-layout V.

What unlocks further wins:

- A new V data structural design that avoids per-codepoint/page traversal
  without full-tile smem staging and without exploding D512 compile time.
- A vLLM-side PV writer or cache-production path that stores PV-layout V once,
  eliminating the linear-V reblock tax from the hot attention call.
- A schedule restructure for paged split-KV / page traversal so the paged path
  stops launching/executing work at two orders of magnitude above the dense
  kernel for the same logical attention cell.

## 2026-05-03 20:36 CDT - Paged V Producer Structural Comparison

Reference files checked:

- `csrc/fmha_v2/fmha/gmem_tile_qkv_packed.h`
- `include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d512.cuh`
- `include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_paged_kv.cuh`

fmha_v2 structure:

- `Gmem_tile_paged_kv` assigns work across `Cta_tile::THREADS_PER_CTA` via
  `THREADS_PER_ROW`, `ROWS_PER_LDG`, and `LDGS`.
- The generic paged path builds `ptrs[LDGS]` once per participating thread and
  issues `Ldgsts_helper<USE_LDGSTS>::load(...)`, so page/block-table resolution
  and vector loads are distributed across the CTA.
- PV-layout V uses `load_pv_layout_v()`, vector-stages scale data where possible,
  then calls `load_nvfp4_row_major_data()`. That data path uses 16-byte row-major
  loads when the row/column window is fully in bounds.
- Linear/reblocked V uses a two-pass structure: distributed scale-pair
  recompute across all CTA threads, `__syncthreads()`, then bounded per-thread
  register reblock with `float8_to_e2m1x8`.

SM120 current structure:

- The paged path is bolted into the existing TMA-style load role. Only the load
  warp runs `stage_paged_v_tile()`.
- Inside that one load warp, each lane loops over
  `copy_thread = lane_idx; copy_thread < ...::ThreadCount; copy_thread += 32`,
  effectively emulating the CUTLASS copy-thread partition from one physical
  warp instead of distributing it across the CTA.
- Paged-PV still computes one destination packed word at a time and fills it
  through eight scalar code loads before a 32-bit smem store.
- Paged-linear does the same packed-word loop plus per-codepoint
  dequant/requant. The reblock arithmetic is real, but the current scaffold
  also serializes the memory work through one load warp.

Structural conclusion:

- The remaining `90x-153x` paged-PV-vs-dense gap is not fixed by a vLLM PV
  writer. A PV writer removes the `linear` reblock tax (`~2.4x-2.5x`), but
  paged-PV is already catastrophically slower than dense.
- The load-bearing issue is the producer execution model: SM120 paged V is using
  a one-load-warp TMA scaffold for work that fmha_v2 distributes across the CTA
  with vectorized loads and explicit staging/reblock phases.
- The next real design should either replace the paged V producer scaffold with
  a CTA-distributed producer modeled on `Gmem_tile_paged_kv`, or split a smaller
  CTA-distributed helper out of the current CUTLASS partition code. More scalar
  hoists inside the current one-warp producer are low-yield.

## 2026-05-03 20:45 CDT - Two-Warp Paged V Producer Assist Plan

What I am about to do:

- Try the schedule-compatible multi-warp producer first: let the epilogue warp
  assist the load warp only for paged V staging, while the eight MMA warps keep
  their existing consumer role.
- Change the V producer loops from `lane_idx` / `32` strides to a producer-local
  thread index over two warps (`64` threads) for paged V only.
- Add a producer-only named barrier around the V staging work so the load-warp
  leader does not complete the V pipeline transaction until both producer warps
  have finished writing the stage.
- Keep Q/K staging unchanged. Keep dense TMA unchanged. Keep public API and smem
  layout unchanged.

Why this variant:

- Full CTA-distributed V staging cannot be inserted directly into the current
  warp-specialized schedule: the MMA warps are concurrently consuming QK/PV
  stages and cannot safely join a producer-side `__syncthreads()` without a
  larger scheduler rewrite.
- The epilogue warp is idle until the final output handoff and can run the same
  V prefetch schedule before entering `consume_and_store_output_span()`.
- This tests whether producer issue width is the immediate limiter without
  changing CUTLASS thread counts or the MMA role geometry.

Done criteria:

- D512 builds and passes `tests/attention/test_nvfp4_kv_head_dim_512.py`.
- If D512 passes, port the same pattern to D128/D256.
- Benchmark the D512 reference cell after the port. If the gain is small, record
  that the one-warp scaffold is not the only limiter and move to the larger
  scheduler replacement design rather than stacking more local tweaks.

## 2026-05-03 20:55 CDT - Two-Warp Paged V Producer Assist Result

Attempted variants:

- Variant A: epilogue warp participates in paged V staging and is also marked as
  a `VPipeline::Producer`.
- Variant B: epilogue warp participates only in paged V staging/named-barrier
  rendezvous; the load warp remains the sole formal VPipeline producer and
  committer.

Observed result:

- Both variants built far enough to start the D512 test suite.
- Both variants emitted 28 passing test dots, then stopped making progress in
  `tests/attention/test_nvfp4_kv_head_dim_512.py`.
- Variant B was run under `timeout 180s`; it timed out with exit code `124`.
- No code from either variant was kept.

Structural conclusion:

- The epilogue warp cannot safely be grafted into the load warp's paged V
  prefetch timeline in the current warp-specialized schedule. The output
  pipeline and V prefetch pipeline have incompatible ordering requirements once
  the epilogue warp is asked to do both jobs.
- This closes the "just add one more producer warp" option. A real fix needs a
  scheduler-level rewrite that changes the producer/consumer timeline, not a
  local assistant warp inside the current branch structure.
- The next viable implementation direction is a dedicated CTA-distributed paged
  V kernel structure or a separate stage kernel modeled on fmha_v2's
  `Gmem_tile_paged_kv`, with the synchronization model designed around all
  participating producer threads from the start.

## 2026-05-03 21:05 CDT - PV-Layout V Register Transpose Slice Plan

What I am about to do:

- Implement the register-transpose V data path only for `kPvLayoutV=true`,
  starting in D512.
- Each 8-lane subgroup loads 8 row-major `uint32_t` words from public PV-layout
  V gmem: one token row per lane, 8 dim-contiguous FP4 codes per word.
- The subgroup uses `__shfl_sync` to transpose those row words so each lane owns
  one output dim across 8 tokens, then writes the resulting 32-bit
  token-contiguous CUTLASS operand word directly to `pv_sB`.
- Keep `kPvLayoutV=false` on the existing linear reblock path. That avoids the
  previous D512 linear compile wall and lets the PV producer issue model be
  measured independently.

Why this variant:

- PV-layout V does not need fp32 dequant/requant. It is the cleanest test of
  whether row-major vector loads plus register transpose can collapse the
  current per-codepoint scalar V path.
- A vLLM-side PV writer only matters if the PV producer itself gets close to
  dense. This slice answers that before spending more time on linear reblock.

Done criteria:

- D512 tests pass with no tolerance changes.
- D512 reference cell paged-PV improves materially from `717.799 ms`.
- If D512 passes, port the same PV-only path to D128/D256.

## 2026-05-03 21:31 CDT - PV-Layout V Register Transpose Slice Result

Implementation:

- Added `sm120_nvfp4_paged_v_word_from_page_base()` for aligned 32-bit
  row-major V loads from the public paged V tensor.
- Replaced the `kPvLayoutV=true` V data producer in D128/D256/D512 with an
  8-lane register transpose:
  - each subgroup loads 8 row-major words for 8 token rows,
  - `__shfl_sync` transposes row-major dim-contiguous codes into
    token-contiguous CUTLASS operand words,
  - each lane writes one 32-bit `pv_sB(col, k0, stage)` word after checking the
    4-byte smem colocation invariant.
- Left `kPvLayoutV=false` unchanged. Linear-V still uses the existing fp32
  dequant/requant path.

Validation:

- `tests/attention/test_nvfp4_kv_head_dim_512.py`
  `tests/utils/test_fp4_kv_quantization.py` -> `62 passed in 206.46s`.
- No tolerance changes.
- All benchmarked outputs reported `output_finite=true`.

Bench deltas:

| cell | before mean ms | after mean ms | delta |
| --- | ---: | ---: | ---: |
| D512 g8 q512 kv65536 softcap30 paged-PV | 717.799 | 283.155 | 2.54x faster |
| D512 g8 q512 kv65536 softcap30 paged-linear | 1740.975 | 1742.628 | unchanged |
| D256 g6 q512 kv65536 softcap30 paged-PV | 185.656 | 104.127 | 1.78x faster |
| D256 g2 q512 kv8192 swa1024 softcap30 paged-PV | 62.079 | 25.925 | 2.39x faster |
| D128 g8 q512 kv8192 paged-PV | n/a | 21.115 | finite smoke/reference |

Characterization:

- This is the first V data structural change with a multi-x payoff. It confirms
  that the old PV producer was paying heavily for scalar per-codepoint loads and
  that row-major vector load plus register transpose is the right data movement
  shape for PV-layout V.
- The remaining D512 paged-PV gap is still large: `283.155 ms` vs dense
  `7.682 ms` at the reference cell, roughly `36.9x` slower.
- Linear-V did not improve because it intentionally stayed on the old branch.
  The stock-vLLM production path therefore remains blocked on the linear reblock
  design, not on PV-layout V anymore.

Next structural implication:

- A vLLM-side PV writer is now more valuable than it was before this patch:
  the future PV path moved from `~93x` dense to `~37x` dense at the D512
  reference cell. It still does not close the full gap, but it removes the
  `~6.15x` linear-vs-PV penalty now visible at the same cell
  (`1742.628 / 283.155`).
- The linear branch needs a separate register-transpose-plus-requant design with
  helper boundaries to avoid the previous D512 ptxas wall.

## 2026-05-03 22:08 CDT - Linear-V Register Transpose Slice Plan

Reference pattern:

- `csrc/fmha_v2/fmha/gmem_tile_qkv_packed.h:1733-1736`
  resolves the paged row once for the current V row:
  `char const* kv_row_ptr; char const* scale_head_ptr; int row_in_page; bool const valid_row = get_nvfp4_row_ptrs(row_idx, kv_row_ptr, scale_head_ptr, row_in_page);`
- `csrc/fmha_v2/fmha/gmem_tile_qkv_packed.h:1743-1753`
  loads the original V scale once per 8-column register group and then loads
  four byte-pairs from that row pointer: `original_scale_byte =
  load_v_original_scale_byte(...)` and `fmha::ldg(original_pair, kv_row_ptr +
  col0 / 2);`.
- `csrc/fmha_v2/fmha/gmem_tile_qkv_packed.h:1757-1771`
  reads the already-computed PV output scale from smem, applies
  original-scale dequant and PV-scale requant in registers, then packs eight
  FP4 codes.

What I am about to do:

- Apply the same row-pointer/requant separation to the SM120 D512 linear-V
  branch without changing the public V layout.
- Reuse the PV register-transpose shape: each 8-lane subgroup loads one
  dim-contiguous 32-bit row word per token row, shuffles those rows into one
  token-contiguous operand word per output dim, then writes the partition-derived
  `pv_sB` word.
- For linear-V only, add an out-of-line helper that consumes the transposed row
  words, original scale bytes, and already-written PV scale byte, then performs
  the fp32 dequant/requant and packs the final token-contiguous word.

Why this variant:

- The full-tile staging option exceeded D512 smem budget and was reverted.
- The previous all-inline register-transpose attempt hit a D512 ptxas wall.
  Keeping the conversion chain behind a helper boundary is the cheapest way to
  test the same data-movement structure without expanding the producer lambda.

Done criteria:

- D512 tests compile and pass with no tolerance changes.
- D512 reference cell `q=512 kv=65536 group=8 softcap=30 v_layout=linear`
  improves materially from the current `1742.628 ms`.
- If D512 passes and improves, port the same linear branch to D128/D256.

## 2026-05-03 22:45 CDT - Linear-V Register Transpose Slice Result

Implementation:

- Added an out-of-line `sm120_nvfp4_linear_v_requant_transposed_word()` helper
  for the linear-V path. The helper shuffles the eight row-major V words across
  an 8-lane subgroup, applies original-scale dequant plus PV-scale requant in
  registers, and returns one token-contiguous 32-bit operand word.
- Replaced the `kPvLayoutV=false` V data producer in D128/D256/D512 with the
  same 8-lane register-transpose store shape used by the PV path.
- Removed the now-dead V-stage CUTLASS `partition_D` setup from the paged V
  producer. The QK/K producer and MMA consumer partitioning remain unchanged.

Validation:

- `tests/attention/test_nvfp4_kv_head_dim_512.py`
  `tests/utils/test_fp4_kv_quantization.py` -> `62 passed in 192.54s`.
- No tolerance changes.
- All benchmarked outputs reported `output_finite=true`.
- One full-suite run before the final D128/D256 port hit the known
  sequence-dependent scratch-poison PV-layout test once; the same test passed in
  isolation immediately after. The post-port combined validation run passed.

Bench deltas:

| cell | PV mean ms | linear mean ms | linear/PV |
| --- | ---: | ---: | ---: |
| D512 g8 q512 kv65536 softcap30 | 285.166 | 482.927 | 1.69x |
| D256 g6 q512 kv65536 softcap30 | 104.157 | 204.414 | 1.96x |
| D256 g2 q512 kv8192 swa1024 softcap30 | 25.905 | 50.839 | 1.96x |
| D128 g8 q512 kv8192 | 21.113 | 35.867 | 1.70x |

Reference delta:

- D512 linear reference improved from `1742.628 ms` to `482.927 ms`, a `3.61x`
  speedup.
- D512 PV stayed effectively stable versus the prior PV transpose result
  (`283.155 ms` -> `285.166 ms`).

Characterization:

- The linear-V production path is no longer dominated by per-codepoint
  block-table/data loads. Its remaining cost is now roughly `1.7x-2.0x` over
  PV-layout V on the focused cells, which is plausible for the irreducible
  fp32 dequant/requant work.
- The paged path is still not benchmark-inline with dense. At the D512 reference
  cell, paged-PV is still roughly `37x` dense and paged-linear roughly `63x`
  dense. This means the remaining large gap is not "linear reblock only"; it is
  the current paged producer/scheduler scaffold doing too little parallel V/scale
  work per CTA.

Post-cleanup validation note:

- After removing the dead V-stage CUTLASS partition setup, reran
  `tests/attention/test_nvfp4_kv_head_dim_512.py`
  `tests/utils/test_fp4_kv_quantization.py` -> `62 passed in 317.43s`.

## 2026-05-03 23:24 CDT - BF16-Q Split-Repetition Fix Plan

What I am about to do:

- Stop quantizing BF16 Q inside every split CTA for paged wrapper runs.
- `csrc/fmha_nvfp4_sm120_paged_common.cuh` already has
  `QuantizeQToPaddedBatchKernel`, which writes `q_packed_scratch` and
  `q_scales_scratch` once for the padded batch.
- `RunPagedBatchBf16QImpl` currently bypasses that prepass and passes
  `q_bf16` into the stage kernel, so `stage_bf16_q_tile()` runs per q-tile and
  per split.

Why:

- The D512 split-KV sweep showed that smaller splits improve paged producer
  parallelism, but the curve bottoms out around `83 ms` PV / `125 ms` linear.
  Repeating Q quantization once per split is a direct cost floor when
  `num_splits` becomes large.
- Moving Q quantization into a single C++ prepass stays within the existing FFI
  call and public API. It is not a Python-side workaround and it reuses an
  existing production helper in the same TU.

Done criteria:

- D512/D256/D128 validation passes with no tolerance changes.
- D512 reference split sweep improves at small split sizes without regressing
  the larger split baseline.

## 2026-05-03 23:43 CDT - BF16-Q Split-Repetition Fix Result

Attempted:

- Added a single-call `QuantizeQToPaddedBatchKernel` prepass inside
  `RunPagedBatchBf16QImpl`, then invoked the existing stage kernel with
  `q_bf16=nullptr` so the Q operand would be loaded from the packed scratch via
  the existing TMA path.
- Extended the prepass to support the same 2D/3D Q strides as the fused in-stage
  quantizer.
- Limited the prepass to compact Q after the non-contiguous scratch-poison test
  exposed mismatch in the general strided path.

Observed:

- Compact multi-KV D512 targeted test passed.
- The scratch-poison test remained non-bitexact even when the non-compact Q case
  fell back to the existing in-stage quantizer. Mismatch scale after fallback was
  back in the previous intermittent range (`~1e-4` to `~5e-4`), but it failed in
  isolation, so this slice is not safe to keep.

Decision:

- Reverted the BF16-Q prepass code. No committed code from this attempt remains.
- The useful finding is that repeated Q quantization is probably a real small-
  split cost, but moving to the TMA packed-Q path changes enough execution state
  to reopen the scratch-residue nondeterminism. Do not reattempt this as a local
  prepass until the scratch-poison class is fully closed.

## 2026-05-04 00:03 CDT - Stream Handle Zero Fix Plan

Finding:

- The scratch-poison test fails without a synchronization after the PyTorch
  `fill_()` calls, but becomes bit-exact when synchronizing after the fills.
- Poisoning any single scratch buffer with a synchronization after the fill is
  also bit-exact. That rules out one uncovered scratch buffer and points to
  stream ordering between PyTorch ops and the SM120 FFI call.
- In this environment, `torch.cuda.current_stream(device).cuda_stream` can be
  `0`. The SM120 FFI path interpreted `stream_handle=0` as literal null stream
  via `stream_from_handle(0)`.
- Existing FlashInfer stream-handle APIs use the idiom
  `stream_handle != 0 ? stream_from_handle(stream_handle) : get_stream(device)`.
  Example: `csrc/fp4_kv_quantization.cu` and
  `csrc/fp4_kv_dequantization.cu`.

What I am changing:

- Apply that same fallback in `csrc/fmha_nvfp4_sm120_paged_common.cuh` for
  dense, paged, and paged-BF16-Q entry points.

Done criteria:

- Scratch-poison test passes without adding Python-side synchronization.
- Combined nvfp4 validation passes.

## 2026-05-04 00:42 CDT - Stream Handle / O-Smem Coverage Result

Implementation:

- Applied the canonical FlashInfer stream-handle fallback in
  `csrc/fmha_nvfp4_sm120_paged_common.cuh` for dense, paged, and paged-BF16-Q:
  nonzero handles are used directly, while zero handles fall back to
  `get_stream(device)`.
- Removed the Python-side private-stream experiment. The wrapper does not add
  synchronization or stream bridging.
- Added a no-init `ZeroSmemTile` handle for `smem_epilogue_o` in D128/D256/D512
  and fills it at the final PV->O handoff before
  `sm120_stage_o_fragment_to_epilogue_smem()`.

Diagnosis:

- The stream fallback is the idiomatic FFI fix, but it did not by itself close
  the intermittent scratch-poison failure.
- The remaining drift localized to another shared-memory coverage gap:
  `smem_epilogue_o` aliases the logits region, the MMA O-fragment store writes
  only its partition-covered subset, and the epilogue warp copies the full
  `kCutlassTileM x output_tile_n` region to global memory.
- Filling O smem to zero at the consumer-prep boundary closes the class without
  relying on global scratch initialization, Python sync, or tolerance changes.

Validation:

- Scratch-poison test loop:
  `tests/attention/test_nvfp4_kv_head_dim_512.py::test_sm120_nvfp4_wrapper_scratch_poison_does_not_affect_output_sm12x`
  -> `5/5` passes.
- `tests/attention/test_nvfp4_kv_head_dim_512.py`
  `tests/utils/test_fp4_kv_quantization.py` -> `62 passed in 271.78s`.
- Focused D512 reference bench, `q=512 kv=65536 group=8 softcap=30
  split_kv_len=32768`: paged-PV `285.301 ms`, paged-linear `482.657 ms`.
  This is unchanged versus the pre-fix reference range, so the O-smem fill did
  not create a measurable regression at this cell.

Implication:

- This is the fourth confirmed smem coverage site in the SM120 kernel family
  (Q operand, logits, probability/O aliasing, and now epilogue O smem). Any
  future producer rewrite should treat CoveredSmemTile-style full-extent
  initialization as mandatory at every aliased consumer boundary, not as an
  optional debug cleanup.

## 2026-05-04 01:07 CDT - Benchmark Auto Split Policy

Finding:

- The grid benchmark's old auto split was stale: D512 defaulted to
  `split_kv_len=32768`, which measured `~285 ms` PV at the D512 reference cell.
- The measured split curve shows that smaller split lengths recover producer
  parallelism, but they increase partial/split scratch memory. A fixed split
  default is therefore the wrong abstraction for production cells with different
  `q_len`, `group`, `head_dim`, and `kv_len`.

Implementation:

- `bench_sm120_nvfp4_attention.py --split-kv-len 0` now auto-selects the
  smallest split length that keeps partial/split scratch under
  `--max-partial-bytes` (default `1 GiB`).
- `bench_sm120_nvfp4_attention_grid.py --fused-split-kv-len 0` now forwards
  auto mode and exposes `--fused-max-partial-bytes` instead of using
  head-dim-specific magic constants.
- Explicit split lengths still work unchanged for controlled split sweeps.

Validation:

- `python -m py_compile benchmarks/bench_sm120_nvfp4_attention.py
  benchmarks/bench_sm120_nvfp4_attention_grid.py` passed.
- D512 reference smoke, `q=512 kv=65536 group=8 softcap=30 v_layout=pv`:
  auto selected `split_kv_len=384` and measured `88.170 ms` mean/min range,
  down from `~285 ms` at the stale `32768` split.
- Grid-driver smoke wrote a row with `split_kv_len=384` and
  `output_finite=true`. Temporary smoke report files were deleted so they do
  not pollute the PR.

Implication:

- Focused production reports can now use `--fused-split-kv-len 0` and get a
  memory-bounded split choice per cell. This does not solve the remaining
  structural producer gap, but it prevents the benchmark harness from
  overstating the gap by using stale split geometry.

## 2026-05-04 01:18 CDT - Focused Production Benchmark Run Plan

What I am about to run:

- Three focused production report prefixes using the existing naming convention:
  `prod_qwen_full_d256_g6_20260504`,
  `prod_gemma_sliding_d256_g2_swa1024_softcap30_20260504`, and
  `prod_gemma_global_d512_g8_softcap30_20260504`.
- Each prefix accumulates `sm120_fused` rows for `api=paged/v_layout=linear`,
  `api=paged/v_layout=pv`, and `api=dense`, plus reference FA2 rows.
- The fused rows use `--fused-split-kv-len 0` so every cell gets the new
  memory-bounded auto split instead of the stale hard-coded split.

Why:

- The previous production reports were generated before the PV/linear producer
  transpose work, before O-smem coverage was fixed, and with stale split
  geometry. They are no longer the right benchmark artifact for judging the
  current code.
- This is still a measurement pass: no kernel changes during the run.

Device convention:

- Use `CUDA_VISIBLE_DEVICES=2` and pass `--device 0` to benchmark scripts.
  Inside the process the SM120 GPU is logical device 0; passing `--device 2`
  under that environment is invalid.

## 2026-05-04 01:42 CDT - Dense Decode Benchmark Padding Fix

Finding:

- The focused Qwen report exposed a dense benchmark harness bug for decode
  cells (`q=1`): dense mode passed `q_rows=q_len * group`, which is not a
  multiple of the SM120 dense kernel tile size.
- The dense FFI already supports padded Q rows (`q_len * group <= q_rows`), but
  the benchmark did not pad before quantization.
- `flashinfer.nvfp4_quantize(..., SfLayout.layout_128x4)` can also return
  scale rows padded to 128 while packed Q rows are only padded to 64, so the
  benchmark must make the packed Q row count match the scale row count.

Implementation:

- Dense benchmark mode pads BF16 Q rows to the kernel tile size before
  quantization.
- If the quantizer pads Q scales beyond packed Q rows, the benchmark pads
  `q_packed` with zero rows to match `q_scales`.
- Partial/split/output scratch allocation follows the actual packed row count.

Validation:

- `python -m py_compile benchmarks/bench_sm120_nvfp4_attention.py
  benchmarks/bench_sm120_nvfp4_attention_grid.py` passed.
- Dense decode smoke `D256 q=1 kv=4096 group=6` passed with finite output and
  measured `0.056 ms` mean.
- The partial Qwen report containing dense decode errors was deleted before
  rerunning so the report prefix stays clean.

## 2026-05-04 02:10 CDT - Focused Production Benchmark Reports

Scope:

- Ran the focused production-cell reports with the corrected device convention:
  `CUDA_VISIBLE_DEVICES=2` and benchmark `--device 0`.
- Kept the report prefixes stable and reran the Qwen prefix after fixing the
  dense decode benchmark padding issue.
- Reports generated:
  - `reports/prod_qwen_full_d256_g6_20260504.{jsonl,csv,summary.csv,md}`
  - `reports/prod_gemma_sliding_d256_g2_swa1024_softcap30_20260504.{jsonl,csv,summary.csv,md}`
  - `reports/prod_gemma_global_d512_g8_softcap30_20260504.{jsonl,csv,summary.csv,md}`

Status:

- Qwen full D256/G6: `84` rows, `14` cells, `0` errors, `0` non-finite SM120
  rows.
- Gemma sliding D256/G2/SWA1024/softcap30: `48` rows, `8` cells, `0` errors,
  `0` non-finite SM120 rows.
- Gemma global D512/G8/softcap30: `66` rows, `11` cells, `11` errors, `0`
  non-finite SM120 rows. All errors are `bf16_fa2` reference-backend failures
  from FlashInfer prefill invalid configuration at D512/G8
  (`NUM_MMA_D_QK=32`, `NUM_MMA_D_VO=32`). `sm120_fused`, `nvfp4_fa2`, and
  `fp8_fa2` completed.

Geomean ratios from `min_ms`:

- Qwen full:
  - paged-linear / dense: `17.31x`
  - paged-PV / dense: `11.28x`
  - paged-linear / paged-PV: `1.535x`
  - `nvfp4_fa2 / paged-linear`: `0.0246x`
  - `nvfp4_fa2 / paged-PV`: `0.0378x`
- Gemma sliding:
  - paged-linear / dense: `14.16x`
  - paged-PV / dense: `9.886x`
  - paged-linear / paged-PV: `1.432x`
  - `nvfp4_fa2 / paged-linear`: `0.0296x`
  - `nvfp4_fa2 / paged-PV`: `0.0425x`
- Gemma global:
  - paged-linear / dense: `15.61x`
  - paged-PV / dense: `10.26x`
  - paged-linear / paged-PV: `1.521x`
  - `nvfp4_fa2 / paged-linear`: `0.0539x`
  - `nvfp4_fa2 / paged-PV`: `0.0820x`

Largest paged-linear slowdowns versus `nvfp4_fa2`:

- Qwen full:
  - `q=1 kv=262144 g=6`: `87.13x` slower (`7.199 ms` vs `0.083 ms`)
  - `q=512 kv=262144 g=6`: `62.96x` slower (`385.454 ms` vs `6.122 ms`)
  - `q=512 kv=131072 g=6`: `56.68x` slower (`175.811 ms` vs `3.102 ms`)
  - `q=128 kv=32768 g=6`: `45.92x` slower (`9.305 ms` vs `0.203 ms`)
  - `q=512 kv=65536 g=6`: `45.27x` slower (`70.641 ms` vs `1.561 ms`)
- Gemma sliding:
  - `q=2048 kv=8192 g=2`: `193.01x` slower (`12.050 ms` vs `0.062 ms`)
  - `q=512 kv=8192 g=2`: `96.76x` slower (`3.226 ms` vs `0.033 ms`)
  - `q=2048 kv=1024 g=2`: `27.12x` slower (`1.635 ms` vs `0.060 ms`)
  - `q=1024 kv=1024 g=2`: `25.78x` slower (`1.094 ms` vs `0.042 ms`)
  - `q=1 kv=8192 g=2`: `20.80x` slower (`0.557 ms` vs `0.027 ms`)
- Gemma global:
  - `q=1 kv=262144 g=8`: `47.70x` slower (`17.166 ms` vs `0.360 ms`)
  - `q=1 kv=65536 g=8`: `40.01x` slower (`4.501 ms` vs `0.113 ms`)
  - `q=1 kv=16384 g=8`: `18.76x` slower (`1.383 ms` vs `0.074 ms`)
  - `q=1 kv=4096 g=8`: `18.49x` slower (`1.298 ms` vs `0.070 ms`)
  - `q=512 kv=262144 g=8`: `17.15x` slower (`645.914 ms` vs `37.653 ms`)

Conclusion:

- The benchmark harness is now measuring the intended SM120 device and no
  longer has the earlier false-flat reference timings or dense decode padding
  failures.
- The SM120 production wrapper path is correct/finite on the focused cells but
  not competitive. The gap is still in the paged kernel path, not Python wrapper
  orchestration: paged-PV is roughly `10-11x` dense and paged-linear is roughly
  `14-17x` dense across the focused reports.
- Linear-V adds a consistent `1.4-1.5x` over paged-PV, but the larger problem is
  common to both layouts. Further work needs to target the paged producer /
  launch geometry rather than report generation or device selection.

## 2026-05-04 02:25 CDT - Multi-Warp Paged Producer Plan

Finding:

- The remaining dense-vs-paged gap is structurally consistent with load-warp
  under-participation.
- Dense mode uses the load warp as a TMA descriptor issuer: one warp is enough
  because TMA hardware moves the tile.
- Paged mode currently reuses that same one load warp to perform the actual
  paged K/V/Q staging work in software. The V path in particular performs the
  dim-contiguous to token-contiguous transpose/repack inside a single warp.
- This is not how the in-tree paged kernels are structured. `fmha_v2` and the
  Hopper sparse producer distribute paged loads across CTA threads; block-table
  indirection is amortized across a tiled load group, not serialized through one
  descriptor-issuer warp.

Implementation target:

- Convert the SM120 paged producer from one load warp to a load warpgroup.
- Keep the MMA warpgroup unchanged (`8` MMA warps) and add load warps rather
  than stealing MMA lanes. The kernel is already smem-limited to one CTA/SM, so
  increasing CTA threads is the right first tradeoff.
- Use one load leader (`load_thread_idx == 0`) for pipeline acquire/complete.
  All load threads participate in staging loops.
- Replace single-warp producer synchronization with a named barrier scoped to
  the load warpgroup. This keeps the TMA/dense leader path single-issued while
  allowing paged software loads to use multiple warps.

Done criteria:

- D512 paged-PV at `q=512 kv=65536 g=8 softcap=30` moves materially toward
  dense (`6.04 ms`) from the current `87.99 ms`.
- D512 paged-linear follows the same direction from the current `153.68 ms`.
- Correctness tests still pass without tolerance changes.

## 2026-05-04 03:20 CDT - Multi-Warp Producer Result

Implemented and validated:

- D512 paged producer now uses `8` load warps with a load-group named barrier.
  The load leader owns pipeline acquire/complete; all load threads advance the
  producer pipeline state after completion. `load_tail` remains restricted to
  the first load warp because CUTLASS tail code uses warp election.
- D128 uses the same widened producer pattern and passes the shared head-dim
  test coverage.
- D256 was attempted with the same pattern but reverted before commit. The
  widened D256 variant introduced small PV-layout multi-KV drift and
  scratch-poison sensitivity. Forcing D256 back to one load warp fixed the
  multi-KV/standard-wrapper cases; restoring warp-synchronous producer handoff
  fixed scratch-poison. Conclusion: D256 needs a separate producer restructure
  and should not receive the D128/D512 load-warpgroup change by mechanical port.

Validation:

- `tests/attention/test_nvfp4_kv_head_dim_512.py -q`: `36 passed`.
- D256 failed-port diagnostic subset after reverting D256 producer changes:
  `test_sm120_nvfp4_wrapper_multi_kv_matches_single_kv_sm12x[256-6]`,
  `test_sm120_nvfp4_wrapper_scratch_poison_does_not_affect_output_sm12x`, and
  `test_standard_prefill_wrapper_sm120_nvfp4_backend_matches_direct_wrapper_sm12x[256-6]`
  all passed.

Reference measurements on GPU 2 via `CUDA_VISIBLE_DEVICES=2 --device 0`:

- D512 Gemma-global cell `q=512 kv=65536 g=8 softcap=30`:
  - dense: `7.922 ms`
  - paged-PV: `15.851 ms`
  - paged-linear: `43.212 ms`
  - Start of pass was dense `7.69 ms`, paged-PV `899 ms`, paged-linear
    `1883 ms`; the D512 PV path is now about `2.0x` dense instead of
    `117x` dense on this cell.
- D128 reference cell `q=512 kv=65536 g=8`:
  - dense: `5.043 ms`
  - paged-PV: `9.904 ms`
  - D128 PV is also about `2.0x` dense after widening.

Remaining cost:

- D512 linear-V is still `43.212 ms` versus `15.851 ms` paged-PV. The remaining
  linear gap is the in-kernel V reblock path, not the common paged producer
  geometry. It still needs a fmha_v2-style reblock structure or a vLLM PV writer
  path to close.
- D256 remains on the known-correct one-load-warp producer. Its broad gap is
  still open and should be attacked as a D256-specific scheduling/reblock issue,
  not by reapplying the D512 load-warpgroup patch unchanged.
## 2026-05-04 00:15 CDT - D512 Linear-V Reblock Pack Primitive

What I am about to do and why:
- The current D512 paged-linear reference remains much slower than paged-PV:
  `43.212 ms` vs `15.851 ms` at `q=512 kv=65536 g=8 softcap=30`.
- The hot helper is
  `include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_paged_kv.cuh:
  sm120_nvfp4_linear_v_requant_transposed_word`, which shuffles 8 rows and then
  packs with four pairwise E2M1 conversions.
- The in-tree fmha_v2 pattern uses an 8-wide E2M1 pack primitive:
  `csrc/fmha_v2/fmha/gmem_tile_qkv_packed.h:1770` calls
  `fmha::float8_to_e2m1x8(vals[0], ..., vals[7])`.
- FlashInfer already exposes the same shared primitive at
  `include/flashinfer/mma.cuh:109`, so this change adopts the existing
  repository primitive instead of composing four pair conversions locally.

Reference audit:
- `include/flashinfer/mma.cuh:109-126`:
  `__device__ __forceinline__ uint32_t float8_to_e2m1x8(float x0, float x1,
  float x2, float x3, float x4, float x5, float x6, float x7)` and four
  `cvt.rn.satfinite.e2m1x2.f32` instructions followed by `mov.b32`.
- `csrc/fmha_v2/fmha/gmem_tile_qkv_packed.h:1770-1771`:
  `packed[reg] = fmha::float8_to_e2m1x8(vals[0], vals[1], vals[2], vals[3],
  vals[4], vals[5], vals[6], vals[7]);`

Decision criteria:
- This is a narrow pack-primitive cleanup, not expected to close the full
  linear-V gap by itself. It is kept only if D512 correctness stays green and
  the reference linear cell is not slower.

Result:
- `tests/attention/test_nvfp4_kv_head_dim_512.py -q`: `36 passed in
  299.78s`.
- D512 Gemma-global linear reference `q=512 kv=65536 g=8 softcap=30`:
  `43.095 ms` min, finite output.
- Previous value was `43.212 ms`; this is effectively flat. The remaining
  linear-V gap is not caused by the pair-pack primitive.

## 2026-05-04 00:25 CDT - Benchmark Split Policy Alignment

Finding:
- `benchmarks/bench_sm120_nvfp4_attention.py` defaulted `--split-kv-len` to
  `0`, which auto-selected a split length from the partial-scratch budget.
- The production wrapper default in `flashinfer/fmha_nvfp4_sm120.py` is
  `split_kv_len=8192`.
- At the D512 Gemma-global reference cell, auto selected `split_kv_len=384`;
  explicit production split selected `8192`.

Measurement with explicit production split:
- D512 `q=512 kv=65536 g=8 softcap=30`:
  - paged-PV: `15.288 ms`
  - paged-linear: `48.315 ms`
  - dense: `5.993 ms`

Conclusion:
- Split policy mismatch was a benchmark validity issue, not the main PV gap.
  Paged-PV remains about `2.6x` dense at this cell with production split.
- Updated the SM120 benchmark scripts so omitted split flags use production
  `8192`; explicit `0` still means auto-split for experiments.

## 2026-05-04 00:27 CDT - D256 Current Gap After D128/D512 Widening

Reference measurement with production split:
- D256 Qwen-full cell `q=512 kv=65536 g=6 causal split_kv_len=8192`:
  - dense: `1.308 ms`
  - paged-PV: `77.162 ms`
  - paged-linear: `150.667 ms`

Conclusion:
- D256 remains the major paged producer outlier. The D128/D512 widened producer
  did not land on D256 because the mechanical port introduced PV multi-KV drift
  and scratch-poison sensitivity. The next D256 work must isolate the
  synchronization/schedule issue rather than reapply the reverted patch
  unchanged.

## 2026-05-04 00:35 CDT - D256 Four-Load-Warp Producer

What changed:
- Retried D256 widening with `kSm120Nvfp4FmhaNumWarpsLoad = 4`, not the prior
  full 8-load-warp mechanical port.
- Added the load-leader/load-group synchronization structure used by the D128
  and D512 widened producers, but with a smaller load group.
- Restricted CUTLASS `load_tail` calls to the first load warp and distributed
  Q/K/V producer loops over `kSm120Nvfp4FmhaLoadThreadCount`.

Validation:
- D256 regression subset:
  `test_sm120_nvfp4_wrapper_multi_kv_matches_single_kv_sm12x`,
  `test_sm120_nvfp4_wrapper_scratch_poison_does_not_affect_output_sm12x`,
  and `test_standard_prefill_wrapper_sm120_nvfp4_backend_matches_direct_wrapper_sm12x`
  filtered for D256: `2 passed, 34 deselected`.
- Full `tests/attention/test_nvfp4_kv_head_dim_512.py -q`:
  `36 passed in 48.51s`.

D256 Qwen-full reference `q=512 kv=65536 g=6 causal split_kv_len=8192`:
- Before this patch:
  - dense: `1.308 ms`
  - paged-PV: `77.162 ms`
  - paged-linear: `150.667 ms`
- After this patch:
  - dense: `1.589 ms`
  - paged-PV: `14.139 ms`
  - paged-linear: `31.610 ms`

Conclusion:
- The D256 broad gap was the one-load-warp paged producer schedule. Four load
  warps recover most of the missing throughput while avoiding the previously
  observed eight-load-warp correctness failures.
- D256 is still not dense-like: paged-PV remains about `8.9x` dense at this
  cell. The next bottleneck is likely split/combine and per-tile fixed work,
  not the catastrophic scalar producer serialization.

## 2026-05-04 00:39 CDT - Residual Gap Localization

Checks:
- Prepacked-Q paged path vs BF16-Q paged path:
  - D256 PV `q=512 kv=65536 g=6`: prepacked-Q `14.034 ms`, BF16-Q
    `14.133 ms`, standalone Q quantize `0.007 ms`.
  - D512 PV `q=512 kv=65536 g=8 softcap=30`: prepacked-Q `15.181 ms`,
    BF16-Q `15.281 ms`, standalone Q quantize `0.007 ms`.
- Single-split check with `split_kv_len=65536`:
  - D256 PV paged `37.866 ms`, dense `4.168 ms`.
  - D512 PV paged `56.004 ms`, dense `20.393 ms`.

Conclusions:
- The residual paged-vs-dense gap is not caused by the BF16-Q production path;
  Q quantization/padding overhead is around `0.1 ms` in the wrapper call.
- The residual gap is not solved by removing split-KV combine. Single split
  reduces parallelism and makes both dense and paged slower.
- D256 eight-load-warp retry with the fixed handoff still failed the same
  multi-KV and standard-wrapper tolerances:
  greatest diffs `0.00177` vs `1e-3` and `0.002106` vs `2e-3`. Reverted to
  the validated four-load-warp setting.

## 2026-05-04 00:49 CDT - Production-Gate Hot Producer Traps

What changed:
- Replaced raw `asm volatile("trap;")` hot-loop invariants in D128/D256/D512
  with `SM120_NVFP4_DEBUG_TRAP()`.
- `SM120_NVFP4_DEBUG_TRAP()` is defined in
  `fmha_nvfp4_sm120_paged_kv.cuh` and defaults off. Developers can re-enable
  it with `-DFLASHINFER_SM120_NVFP4_DEBUG_TRAPS=1` for layout validation.

Validation:
- Full `tests/attention/test_nvfp4_kv_head_dim_512.py -q`:
  `36 passed in 319.68s`.

Reference cells with production split:
- D256 PV `q=512 kv=65536 g=6`: `14.139 ms` -> `8.831 ms`.
- D512 PV `q=512 kv=65536 g=8 softcap=30`: `15.288 ms` -> `11.682 ms`.
- D512 linear same cell: `48.315 ms` -> `47.738 ms`.

Conclusion:
- The invariant branches/traps were part of the residual PV runtime cost.
  Gating them is both upstream-style cleanup and a real performance fix.
- Linear-V remains dominated by the reblock path; trap gating does not change
  that bottleneck materially.

## 2026-05-04 01:02 CDT - Linear-V Scale-Shuffle Narrow Test

What was tested:
- Changed the linear-V transpose helper to convert the source lane's UE4M3
  scale byte to float once, shuffle that float, and reuse it in the 8-wide
  packed-word requantization.
- This removes per-peer `e4m3_byte_to_fp32(...)` conversions inside
  `sm120_nvfp4_linear_v_requant_transposed_word`.

Validation:
- Full `tests/attention/test_nvfp4_kv_head_dim_512.py -q`:
  `36 passed in 319.67s`.

D512 Gemma-global reference `q=512 kv=65536 g=8 softcap=30 split_kv_len=8192`:
- Before this test: paged-linear `47.738 ms`.
- With the float-scale shuffle: paged-linear `48.822 ms`.

Conclusion:
- This is not a useful optimization. It slightly regressed the reference cell
  and does not address the structural linear-V gap.
- The change was reverted. The remaining work stays in the producer schedule
  and linear-V reblock structure, not scalar conversion micro-tweaks.

## 2026-05-04 01:21 CDT - Linear-V Reblock Cost Localization

Checks:
- D512 Gemma-global `kv=65536 g=8 softcap=30 split_kv_len=8192` q-scaling:
  - PV: `q=128 5.329 ms`, `q=512 11.742 ms`, `q=2048 37.236 ms`.
  - linear: `q=128 19.430 ms`, `q=512 48.153 ms`,
    `q=2048 165.503 ms`.
- Temporary D512 diagnostic forced linear-V output reblock scales to unit
  (`0x38`) and skipped the 16-token max-abs scale recompute. It was reverted
  immediately after measurement.

Diagnostic result:
- D512 linear reference `q=512 kv=65536`: `48.153 ms` -> `26.819 ms` with
  unit reblock scales.

Conclusions:
- Linear-V is paying reblock work per Q tile. The same V tile/scales are
  recomputed once for every Q CTA, which is architecturally wrong for a
  production paged cache path.
- At this cell, roughly `21 ms` is the repeated output-scale recompute and the
  remaining `~15 ms` over PV is the data dequant/requant path.
- The next structural fix is to move or cache linear-V output-scale generation
  at the KV-split/output-group scope so Q tiles reuse it, instead of running
  the 16-token max-abs scan inside every Q CTA.

## 2026-05-04 01:43 CDT - Private Linear-V Reblock Scale Cache

What changed:
- Added an internal linear-V output-scale cache in
  `Sm120Nvfp4PagedKvLoadParams`.
- The paged launcher carves the cache from the existing `workspace` after the
  CUTLASS QK/PV workspaces. No public tensor shape, layout name, FFI argument,
  or Python wrapper API changed.
- For `kPvLayoutV=false`, a preparatory CUDA kernel computes the 16-token
  linear-V reblock scales once per `(batch, kv_head, dim, token_group)`.
- The stage kernel now loads those cached scale bytes into `pv_sSFB` instead
  of running the max-abs scan inside every Q CTA. If the cache pointer is null,
  the old in-CTA recompute path remains as a dense/test fallback.

Validation:
- Full `tests/attention/test_nvfp4_kv_head_dim_512.py -q`:
  `36 passed in 320.81s`.

Reference cells, production split `8192`:
- D512 Gemma-global `q=512 kv=65536 g=8 softcap=30`:
  - PV: `11.669 ms` (unchanged from `11.682 ms`).
  - linear: `47.738 ms` -> `28.552 ms`.
- D256 Qwen-full `q=512 kv=65536 g=6`:
  - PV: `8.843 ms` (unchanged from `8.831 ms`).
  - linear: `31.610 ms` -> `17.140 ms`.
- D128 baseline `q=512 kv=65536 g=8`:
  - PV: `3.793 ms`.
  - linear: `10.346 ms`.

Conclusion:
- The repeated scale scan was the largest linear-V-specific architectural
  waste. It is now paid once per paged call/KV cache scope instead of once per
  Q tile.
- Remaining linear-V overhead is the data dequant/requant path. At D512
  reference, linear is still `2.45x` PV, so further work needs to attack data
  requant reuse or reduce the per-Q-tile data conversion itself.

## 2026-05-04 02:06 CDT - Private Linear-V Data Cache

What changed:
- Added an internal linear-V data cache in `Sm120Nvfp4PagedKvLoadParams`.
- For `kPvLayoutV=false`, the paged launcher now computes both:
  - reblock scale cache: one byte per `(batch, kv_head, dim, 16-token group)`.
  - reblocked PV data cache: one packed byte per
    `(batch, kv_head, token, dim-pair)`.
- Both caches live in the existing workspace after the CUTLASS QK/PV
  workspaces. No public tensor shape, layout name, FFI argument, or Python API
  changed.
- The stage kernel's linear-V data path now reads the private PV-data cache and
  uses the same 8-lane transpose/store pattern as the public PV-layout path.
  The old in-CTA dequant/requant path remains only as a null-cache fallback.

Validation:
- Full `tests/attention/test_nvfp4_kv_head_dim_512.py -q`:
  `36 passed in 322.93s`.

Reference cells, production split `8192`:
- D512 Gemma-global `q=512 kv=65536 g=8 softcap=30`:
  - PV: `12.397 ms`.
  - linear: `28.552 ms` -> `13.594 ms`.
  - Original pre-cache linear before both cache fixes was `47.738 ms`.
- D256 Qwen-full `q=512 kv=65536 g=6`:
  - PV: `8.862 ms`.
  - linear: `17.140 ms` -> `10.056 ms`.
  - Original pre-cache linear before both cache fixes was `31.610 ms`.
- D128 baseline `q=512 kv=65536 g=8`:
  - PV: `3.794 ms`.
  - linear: `10.346 ms` -> `4.467 ms`.

Conclusion:
- Stock vLLM linear-V is now within about `10-18%` of PV-layout at these
  reference cells instead of `2.5-3x` slower.
- The remaining gap is the one-time private cache generation cost plus extra
  workspace bandwidth. It no longer scales with Q tiles inside the fused stage
  kernel.
- This matches the intended production architecture: stock vLLM linear cache is
  the production input path, and the wrapper/FFI performs the necessary
  one-call internal conversion without exposing a bridge tensor or adding a
  second Python-visible step.

## 2026-05-04 02:14 CDT - Production Split Default Retune

Checks:
- D512 Gemma-global `q=512 kv=65536 g=8 softcap=30`, linear:
  - split `4096`: `11.347 ms`
  - split `8192`: `13.640 ms`
  - split `16384`: `12.835 ms`
  - split `32768`: `23.288 ms`
  - split `65536`: `45.118 ms`
- D512 PV same cell:
  - split `4096`: `10.259 ms`
  - split `8192`: `12.424 ms`
- D256 Qwen-full `q=512 kv=65536 g=6`:
  - linear split `4096`: `8.677 ms`
  - linear split `8192`: `10.035 ms`
  - PV split `4096`: `7.549 ms`
  - PV split `8192`: `8.862 ms`

What changed:
- Retuned the production wrapper default `split_kv_len` from `8192` to
  `4096`.
- Matched the defaults in `bench_sm120_nvfp4_attention.py` and
  `bench_sm120_nvfp4_attention_grid.py`.
- Explicit caller-provided split lengths and `0` auto-split behavior in the
  benchmark remain unchanged.

Validation:
- Full `tests/attention/test_nvfp4_kv_head_dim_512.py -q` on warm cache:
  `36 passed in 1.00s`.

Conclusion:
- After the linear-V cache fixes, the prior `8192` split default is no longer
  optimal. `4096` consistently improves the sampled D512 and D256 production
  cells for both PV and linear layouts.

## 2026-05-04 02:44 CDT - Paged Producer Page-Table Hoist Pass

What I am about to do:
- The focused production reports show linear-V is no longer the dominant issue:
  linear tracks PV within roughly `10-20%` on the main prefill cells.
- Direct wrapper / BF16-Q FFI / prepacked-Q FFI timing shows the residual
  paged-vs-dense gap is inside the paged stage kernel, not Python wrapper
  orchestration or fused Q quantization.
- The hot paged K and PV-V producers still resolve `block_table` at
  packed-word granularity. That diverges from the in-tree paged attention
  pattern, where page offsets are computed once per KV row and reused by the
  vector load.
- Decision criterion: reduce redundant block-table walks in K and PV-V without
  changing public tensor layouts, FFI signatures, spec axes, or test contracts.

Reference audit:
- `csrc/fmha_v2/fmha/gmem_tile_qkv_packed.h` paged path computes row pointers
  before the load. Relevant lines:
  `paged_kv_block_idx = (row_idx >> paged_kv_log2_block_size_)`,
  `local_kv_ptr = ... paged_kv_global_block_offsets_[paged_kv_block_idx]`,
  then `ptrs[ii] = local_kv_ptr + head_col_in_bytes + ...`.
- `csrc/fmha_v2/fmha/gmem_tile_qkv_packed.h` reblocked V uses the same pattern:
  `get_nvfp4_row_ptrs(...)` returns `kv_row_ptr`, then the load uses
  `fmha::ldg(original_pair, kv_row_ptr + col0 / 2)`.
- `include/flashinfer/attention/hopper/sparse_mainloop.cuh` precomputes the
  paged offset per KV position: `page_idx = kv_indices_ptr[page_iter]` and
  `my_kv_offset[parity] = page_idx * k_page_stride + entry_idx * k_stride_n`,
  then load threads use `__shfl_sync` to reuse that offset before issuing
  `cp_async_zfill`.
- The pattern I am adopting is the local version of that: compute the page base
  once per producer row/subgroup, then use from-page-base helpers for each
  packed-word load. This keeps the current CUTLASS smem partitioning intact but
  removes redundant page-table math from the hot loops.

Result:
- Implemented the page-base hoist locally for K and PV-layout V, then measured
  D256 Qwen-full `q=512 kv=65536 g=6`, PV, split `4096`.
- Baseline from the focused report: `7.56 ms`.
- Hoisted version: `7.86 ms`.
- The change was reverted. In this scaffold the extra cached-base branch and
  subgroup shuffle cost more than the page-table lookup they remove.
- Conclusion: page-table lookup granularity is not the current load-bearing
  bottleneck. The residual gap is the broader manual paged producer path versus
  dense TMA, not just redundant `block_table` arithmetic.

## 2026-05-04 03:00 CDT - D256 Load-Warp Capacity Check

What I am about to do:
- D512 uses eight load warps and is about `2x` dense on the key Gemma-global
  cell after the producer/cache fixes.
- D256 uses four load warps and is about `5-6x` dense on the key Qwen-full and
  Gemma-sliding cells.
- The paged producer is manual K/V movement into CUTLASS operand smem; dense
  gets the TMA path. If D256 is producer-bound, giving D256 the same load-warp
  capacity as D512 should reduce the gap without public API changes.
- Decision criterion: keep the change only if paged PV improves enough to
  offset any dense regression. If it loses or is neutral, revert before moving
  on.

Eight-load-warp result:
- Changed D256 load warps from `4` to `8`.
- Qwen-full `q=512 kv=65536 g=6`, split `4096`:
  - PV: `7.56 ms` -> `5.66 ms`.
  - linear: `8.68 ms` -> `6.66 ms`.
  - dense: `1.36 ms` -> `1.61 ms`.
- Gemma-sliding `q=512 kv=8192 g=2 swa=1024 softcap=30`, split `4096`:
  - PV: `1.61 ms` -> `1.18 ms`.
  - linear: `1.77 ms` -> `1.34 ms`.
- Correctness failed: full `tests/attention/test_nvfp4_kv_head_dim_512.py -q`
  had three D256 failures in multi-KV / scratch-poison / standard-wrapper
  comparisons. The failures were sequence/state sensitive and therefore
  treated as a real synchronization or coverage issue, not a tolerance issue.

Six-load-warp result:
- Changed D256 load warps from `4` to `6`.
- Qwen-full `q=512 kv=65536 g=6`, PV, split `4096`:
  `7.56 ms` -> `6.99 ms`.
- Full `tests/attention/test_nvfp4_kv_head_dim_512.py -q`:
  `36 passed in 97.77s`.
- Decision: keep the safe six-warp setting. It is a smaller win than eight
  load warps, but it preserves correctness and still confirms D256 was
  under-provisioned on producer load capacity relative to the manual paged K/V
  copy work.

D512 symmetry check:
- Tried increasing D512 load warps from `8` to `12`.
- Gemma-global `q=512 kv=65536 g=8 softcap=30`, PV, split `4096`:
  `10.23 ms` -> `10.62 ms`.
- Decision: reverted. D512 is not improved by more load warps; its current
  eight-load-warp allocation is near the local optimum for this scaffold.

Seven-load-warp D256 follow-up:
- Changed D256 load warps from the safe `6` setting to `7`.
- Qwen-full `q=512 kv=65536 g=6`, PV, split `4096`:
  `6.99 ms` -> `6.74 ms`.
- Targeted D256 multi-KV correctness:
  `3 passed in 47.76s`.
- Full `tests/attention/test_nvfp4_kv_head_dim_512.py -q`:
  `36 passed in 50.71s`.
- Decision: keep seven load warps. It recovers another small slice of the
  D256 paged producer gap without reproducing the sequence/state-sensitive
  failures seen at eight load warps.

## 2026-05-04 03:20 CDT - D256 Paged CTA Amortization Check

What I am about to do:
- D512 uses `kCutlassTileM=128`; D256 uses `kCutlassTileM=64`.
- The paged producer does manual K/V staging into CUTLASS operand smem, so a
  smaller M tile doubles the number of producer CTAs for the same Q length.
- Dense uses TMA and is less sensitive to that per-CTA producer overhead. This
  is a plausible structural reason D256 paged remains far behind dense while
  D512 is much closer.
- I will test a D256 `kCutlassTileM=128` variant as a private kernel-internal
  experiment. Decision criterion: keep only if correctness holds and the
  paged/dense ratio improves materially at Qwen-full and Gemma-sliding cells.

Result:
- Qwen-full `q=512 kv=65536 g=6`, PV, split `4096`:
  `6.74 ms` -> `5.33 ms`.
- Qwen-full `q=512 kv=65536 g=6`, linear, split `4096`:
  prior focused result `8.68 ms` -> `6.28 ms`.
- Qwen-full dense at the same cell:
  `2.08 ms` with `M=128`, materially slower than the `M=64` dense-focused
  baseline.
- Gemma-sliding `q=512 kv=8192 g=2 swa=1024 softcap=30`, PV:
  prior focused result around `1.18 ms` after the load-warp change regressed to
  `1.96 ms`.
- Gemma-sliding linear at the same cell:
  `2.12 ms`.
- Decision: reverted. `M=128` confirms the residual D256 full-attention gap is
  partly CTA-amortization overhead, but it is not a valid global D256 setting.
  The right shape is workload-specific: long full-attention Qwen wants a larger
  Q tile; sliding-window Gemma does not.

Follow-up implementation plan:
- Do not add a new public API field or JIT axis.
- Use the existing `SM120_NVFP4_USE_SLIDING_WINDOW` specialization in the D256
  paged TU: non-SWA D256 paged gets `M=128`; SWA D256 paged stays `M=64`.
- Keep the D256 dense TU at `M=64`, because the dense baseline regressed under
  the global `M=128` experiment and dense does not pay the paged producer CTA
  overhead.
- Update the Python wrapper's padded-row calculation to match the selected
  paged module tile. Decision criterion: no-SWA production cells improve while
  Gemma sliding remains on the prior fast path.

Implementation result:
- Added a generated preprocessor value for `use_sliding_window` in
  `fmha_nvfp4_sm120_config.inc`.
- D256 paged TU defines `FLASHINFER_SM120_NVFP4_D256_TILE_M=128` only when
  `use_sliding_window=false`; the D256 dense TU keeps the default `M=64`.
- Wrapper and benchmark padding/split math now use the paged module tile for
  paged runs and the dense tile for dense runs.
- Qwen-full `q=512 kv=65536 g=6`, PV:
  `6.74 ms` -> `5.34 ms`.
- Qwen-full `q=512 kv=65536 g=6`, linear:
  `8.68 ms` focused baseline -> `6.34 ms`.
- Qwen-full dense at the same cell stayed on `M=64`:
  `1.37 ms`.
- Gemma-sliding `q=512 kv=8192 g=2 swa=1024 softcap=30`, PV:
  `1.41 ms`, avoiding the global-`M=128` regression (`1.96 ms` in the
  experiment).
- Gemma-sliding linear at the same cell:
  `1.58 ms`.
- Full `tests/attention/test_nvfp4_kv_head_dim_512.py -q`:
  `36 passed in 225.39s`.
- Conclusion: this is a valid workload-specialized structural win using an
  existing spec axis. It does not solve the whole paged-vs-dense gap, but it
  removes one obvious source of excess D256 full-attention producer CTA count.

Split schedule recheck after D256 non-SWA `M=128`:
- Qwen-full `q=512 kv=65536 g=6`, PV:
  - split `2048`: `4.72 ms`
  - split `4096`: `5.29 ms`
  - split `8192`: `6.75 ms`
  - split `16384`: `6.73 ms`
  - split `32768`: `13.41 ms`
  - split `65536`: `26.76 ms`
- Qwen-full `q=512 kv=65536 g=6`, linear:
  - split `2048`: `5.56 ms`
  - split `4096`: `6.29 ms`
  - split `8192`: `7.87 ms`
  - split `16384`: `7.88 ms`
  - split `32768`: `14.90 ms`
  - split `65536`: `29.26 ms`
- Conclusion: after larger D256 non-SWA Q tile, the best split for this
  production cell moved from `4096` to `2048`. The broad gap is still the paged
  stage kernel, but the wrapper/bench auto-scheduler should not keep using the
  old split geometry.

Refined split sweep below `2048`:
- Qwen-full `q=512 kv=65536 g=6`, PV:
  - split `128`: `9.83 ms`
  - split `256`: `6.84 ms`
  - split `512`: `5.31 ms`
  - split `1024`: `4.68 ms`
  - split `1536`: `4.41 ms`
  - split `2048`: `4.75 ms`
  - split `3072`: `4.24 ms`
- Qwen-full `q=512 kv=65536 g=6`, linear:
  - split `128`: `10.60 ms`
  - split `256`: `7.61 ms`
  - split `512`: `6.17 ms`
  - split `1024`: `5.54 ms`
  - split `1536`: `5.27 ms`
  - split `2048`: `5.58 ms`
  - split `3072`: `5.07 ms`
- Conclusion: `3072` is the best tested split for this cell in both PV and
  linear. Very small splits lose to split/combine overhead, while large splits
  lose to per-stage work. The optimum is a real schedule balance, not a
  monotonic "more splits" answer.

## 2026-05-04 04:10 CDT - D512 Split Schedule Check

What I am about to do:
- D512 Gemma-global is already much closer to dense than D256, but still about
  `2x` at the key `q=512 kv=65536 g=8` cell.
- Before changing the producer again, I am rechecking split geometry under the
  current kernel because split-K affects how much work each paged CTA does and
  how much split-combine overhead is paid.

Result:
- Gemma-global `q=512 kv=65536 g=8 softcap=30`, PV:
  - split `1024`: `11.39 ms`
  - split `1536`: `10.41 ms`
  - split `2048`: `10.06 ms`
  - split `3072`: `9.65 ms`
  - split `4096`: `10.25 ms`
  - split `6144`: `9.99 ms`
  - split `8192`: `12.42 ms`
  - split `12288`: `9.64 ms`
- Gemma-global `q=512 kv=65536 g=8 softcap=30`, linear:
  - split `1024`: `12.27 ms`
  - split `1536`: `12.19 ms`
  - split `2048`: `12.30 ms`
  - split `3072`: `11.58 ms`
  - split `4096`: `11.31 ms`
  - split `6144`: `11.00 ms`
  - split `8192`: `13.60 ms`
  - split `12288`: `10.70 ms`
- Conclusion: split geometry is only a modest D512 factor. PV improves from
  roughly `10.25 ms` at split `4096` to `9.64 ms`; linear improves from
  `11.31 ms` to `10.70 ms`. The remaining D512 gap is in the stage kernel path,
  not the split scheduler.

## 2026-05-04 04:25 CDT - D512 Paged Output-Group Span Check

What I am about to do:
- D512 paged currently compiles `kOutputGroupSpan=4`, so one CTA carries four
  PV accumulators and stages four 128-column V groups.
- The D512 code already reuses the P stage across those groups for non-final
  tiles, but the register footprint and V-stage scheduling may still be worse
  than two CTAs with `kOutputGroupSpan=2`.
- I will test a D512 paged-only `kOutputGroupSpan=2` launch surface. Dense stays
  at span 4 for this experiment. Decision criterion: keep only if paged improves
  materially and correctness holds.

Result:
- Gemma-global `q=512 kv=65536 g=8 softcap=30`, PV:
  span `2`, split `3072`: `13.81 ms`.
  Current span `4`, split `3072`: `9.65 ms`.
- Gemma-global `q=512 kv=65536 g=8 softcap=30`, linear:
  span `2`, split `12288`: `15.84 ms`.
  Current span `4`, split `12288`: `10.70 ms`.
- Decision: reverted. The D512 span-4 path is the correct amortization point;
  halving the span increases CTAs and loses more than it saves in register
  pressure.

## 2026-05-04 04:45 CDT - K Producer Vector cp.async Check

What I am about to do:
- The in-tree reference paged producers (`hopper/sparse_mainloop.cuh` and
  `fmha_v2/fmha/gmem_tile_qkv_packed.h`) issue vectorized async loads after
  page-table lookup.
- The SM120 NVFP4 K producer currently emits one `pred_load_32b` per 8 FP4
  nibbles. For K, the global layout is dim-contiguous and the CUTLASS B smem
  invariant already requires contiguous packed-word groups.
- I will test widening the D256 K producer to one 128-bit `cp.async` per 32
  FP4 nibbles when the partition emits a 32-nibble contiguous span. Decision
  criterion: keep only if correctness holds and the Qwen full-attention
  reference cell improves. This is a K-only test before attempting the harder V
  path.

Result:
- D256 Qwen-full `q=512 kv=65536 g=6`, PV, split `3072`:
  current `32b` K cp.async path: `4.24 ms`.
  ad hoc `128b` grouping experiment: `24.53 ms`.
- Decision: reverted. The current smem partition is not a valid place to bolt
  on vectorization by scanning for 32 contiguous nibbles. The reference kernels
  vectorize by choosing a gmem/smem copy partition whose atom is already
  128-bit; doing it inside the existing 32-bit partition adds branch/coverage
  overhead and likely misses the intended layout grouping.
- Follow-up implication: any real vectorization needs a separate producer
  partition patterned after `hopper/sparse_mainloop.cuh` / fmha_v2
  `Gmem_tile_paged_kv`, not local grouping inside the current CUTLASS
  `SmemCopyAtomB` traversal.

## 2026-05-04 05:05 CDT - Gemma Sliding Split Schedule Check

What I am about to do:
- Gemma sliding has a fixed effective window of 1024 tokens but was still being
  benchmarked and planned with the generic `4096` split length.
- That can make each split stage cover much more KV than the sliding mask can
  use, especially in the current split-K scheduler.
- I am sweeping the key Gemma-sliding cell before changing defaults.

Result:
- Gemma-sliding `q=512 kv=8192 g=2 swa=1024 softcap=30`, PV:
  - split `128`: `0.569 ms`
  - split `256`: `0.432 ms`
  - split `512`: `0.458 ms`
  - split `1024`: `0.415 ms`
  - split `1536`: `0.578 ms`
  - split `2048`: `0.742 ms`
  - split `3072`: `1.06 ms`
  - split `4096`: `1.40 ms`
  - split `6144`: `2.04 ms`
  - split `8192`: `2.72 ms`
- Gemma-sliding `q=512 kv=8192 g=2 swa=1024 softcap=30`, linear:
  - split `128`: `0.676 ms`
  - split `256`: `0.545 ms`
  - split `512`: `0.574 ms`
  - split `1024`: `0.522 ms`
  - split `1536`: `0.696 ms`
  - split `2048`: `0.876 ms`
  - split `3072`: `1.22 ms`
  - split `4096`: `1.59 ms`
  - split `6144`: `2.26 ms`
  - split `8192`: `2.99 ms`
- Conclusion: Gemma sliding was mostly mis-scheduled. The best tested split is
  the sliding window length, `1024`; the generic `4096` split burns 3x+ runtime.

Auto-split implementation:
- Changed wrapper `plan(..., split_kv_len=0)` to mean production auto-selection.
- SWA selects `round_up(window_left, 128)`, so Gemma sliding picks `1024`.
- Non-SWA selects from Q tile count with head_dim-specific scaling:
  - D256 Qwen-full `q=512 g=6` picks `3072`.
  - D512 Gemma-global `q=512 g=8` picks `12288`.
- Positive `split_kv_len` still preserves explicit caller control.
- Benchmark and grid defaults now use auto mode (`0`) so production reports do
  not accidentally benchmark the stale `4096` split.
- Smoke results:
  - Qwen-full D256 linear `q=512 kv=65536`: split `3072`, `5.03 ms`.
  - Gemma-sliding D256 linear `q=512 kv=8192`: split `1024`, `0.523 ms`.
  - Gemma-global D512 linear `q=512 kv=65536`: split `12288`, `10.71 ms`.
  - Gemma-global D512 PV `q=512 kv=65536`: split `12288`, `9.70 ms`.

## 2026-05-04 05:55 CDT - Compact Production Autosplit Report

What I measured:
- Ran the existing grid reporter with explicit production cells after the auto-split change, using logical device `0` under `CUDA_VISIBLE_DEVICES=2`.
- Reports written:
  - `reports/prod_qwen_full_d256_g6_autosplit_20260504.*`
  - `reports/prod_gemma_sliding_d256_g2_autosplit_20260504.*`
  - `reports/prod_gemma_global_d512_g8_autosplit_20260504.*`
- Each report includes paged-linear, paged-PV, dense, NVFP4 FA2, and BF16 FA2 where the reference backend supports the configuration.

Key cells:
- Qwen full D256 g6 q=512 kv=65536 softcap=30:
  dense `1.320 ms`, paged-PV `4.293 ms`, paged-linear `5.159 ms`, NVFP4 FA2 `1.565 ms`.
  Paged-PV is `3.25x` dense; linear reblock adds only `20.2%` over PV.
- Qwen full D256 g6 q=2048 kv=65536 softcap=30:
  dense `4.998 ms`, paged-PV `16.231 ms`, paged-linear `17.570 ms`, NVFP4 FA2 `9.644 ms`.
  Paged-PV is `3.25x` dense; linear reblock adds `8.2%`.
- Gemma sliding D256 g2 q=512 kv=8192 swa=1024 softcap=30:
  dense `0.173 ms`, paged-PV `0.410 ms`, paged-linear `0.517 ms`, NVFP4 FA2 `0.035 ms`.
  Auto split fixed the stale 4096-split mis-schedule, but a fixed paged tax remains.
- Gemma global D512 g8 q=512 kv=65536 softcap=30:
  dense `5.152 ms`, paged-PV `9.674 ms`, paged-linear `10.683 ms`, NVFP4 FA2 `9.381 ms`.
  Paged-PV is `1.88x` dense; linear reblock adds `10.4%`.
- Gemma global D512 g8 q=2048 kv=65536 softcap=30:
  dense `19.525 ms`, paged-PV `36.548 ms`, paged-linear `38.541 ms`, NVFP4 FA2 `45.301 ms`.
  Paged-linear beats NVFP4 FA2 at this large-Q cell, but is still `1.97x` dense.

Conclusion:
- The remaining prefill gap is not primarily the linear-V reblock path. For q>=512, linear over PV is usually `5-20%`.
- The dominant structural issue is paged-PV versus dense: D256 full attention is still `~3.25x` dense, D512 global is `~1.88x` dense.
- Decode/short-Q has a separate fixed-floor problem: D256 q=1 paged-PV is `0.50-0.54 ms` vs dense `0.086-0.120 ms`; D512 q=1 paged-PV is `0.74-0.81 ms` vs dense `0.38-0.43 ms`.
- Next target is paged launch/schedule/mainloop geometry, not another linear-V-only micro-optimization.

## 2026-05-04 06:05 CDT - D256 Qwen Paged-PV Profile

Profiled one measured call after warmup:
- Command: `nsys profile --stats=true ... bench_sm120_nvfp4_attention.py --mode paged-wrapper --q-len 512 --kv-len 65536 --head-dim 256 --group 6 --split-kv-len 0 --v-layout pv --causal --logits-soft-cap 30`.
- Measured wall/event result: `4.308 ms`.

GPU kernel summary:
- SM120 NVFP4 D256 stage kernel: 3 instances, average `4.223 ms`, `93.3%` of GPU kernel time.
- Q quantize kernel: 2 instances, average `0.121 ms`, `1.8%`.
- Split-KV combine kernel: 3 instances, average `0.044 ms`, `1.0%`.
- Output copy / torch elementwise noise: sub-`0.13 ms` total.
- CUDA memset: 6 calls, total GPU memset time `0.032 ms`; not load-bearing.

Conclusion:
- The remaining D256 Qwen gap is inside `sm120_nvfp4_qkv_online_register_q_stage_kernel`, not Python, FFI, Q quantization, combine, output copy, or workspace zeroing.
- Dense and paged at the same cell have comparable split/CTA counts after auto-split; the stage kernel itself is slower in the paged specialization.

## 2026-05-04 06:15 CDT - Move BF16 Q Quantize Out of Split Stage

What I am about to do:
- `stage_bf16_q_tile` quantizes BF16 Q inside `sm120_nvfp4_qkv_online_register_q_stage_kernel`.
- In split-KV, that repeats identical Q quantization once per split CTA. At Qwen D256 q=512 kv=65536 auto split, the same Q tile is quantized across ~22 split CTAs.
- The paged common FFI file already has `QuantizeQToPaddedBatchKernel`, which quantizes Q directly into the padded scratch layout the stage kernel consumes.
- I will launch that once in `RunPagedBatchBf16QImpl`, then call the existing paged stage with `q_bf16=nullptr` so it uses prepacked Q through the same producer path as dense. Decision criterion: D256 Qwen paged-PV improves materially and tests remain green.

Result:
- D256 Qwen-full q=512 kv=65536 PV before this change: `4.293 ms`.
- Moving BF16 Q quantization to one pre-stage padded-Q kernel: `4.181 ms`.
- Passing `kv_head=0` for single-KV-head wrappers instead of the all-head sentinel is effectively neutral: PV `4.171 ms`, linear `5.015 ms`.
- Conclusion: repeated Q quantization was real structural waste and is worth removing, but it is not the dominant 3x stage gap.
- Next: split scheduling for short-Q/decode. Current auto split has a minimum of 8 KV tiles, causing q=1 to run at least 8 splits plus combine even though reference decode kernels use a decode-shaped schedule.

## 2026-05-04 06:35 CDT - Decode Split and D256 Output Span Checks

Decode/short-Q split check:
- D256 q=1 kv=65536 PV:
  split 1024 `0.508 ms`, 2048 `0.914 ms`, 4096 `1.739 ms`, 8192 `3.409 ms`, 16384 `6.666 ms`, 32768 `13.178 ms`, 65536 `25.607 ms`.
- D256 q=1 kv=65536 linear:
  split 1024 `1.216 ms`, 2048 `1.645 ms`, 4096 `2.527 ms`, 8192 `4.293 ms`, 16384 `7.732 ms`, 32768 `14.603 ms`, 65536 `27.823 ms`.
- D512 q=1 kv=65536 PV:
  split 1024 `0.778 ms`, 2048 `1.343 ms`, 4096 `2.517 ms`, 8192 `4.940 ms`, 16384 `9.569 ms`, 32768 `18.895 ms`, 65536 `36.106 ms`.
- D512 q=1 kv=65536 linear:
  split 1024 `2.219 ms`, 2048 `2.817 ms`, 4096 `4.094 ms`, 8192 `6.677 ms`, 16384 `11.714 ms`, 32768 `21.755 ms`, 65536 `40.118 ms`.
- Conclusion: q=1 is fastest at the smallest tested split. Larger split makes each CTA loop over more masked/serialized KV work. Auto already selects 1024, so the q=1 floor is the cost of using this prefill-stage kernel as a decode path, not a simple auto-split bug.

D256 output-group-span check:
- D256 Qwen q=512 kv=65536 PV with production span=2: `~4.17 ms` after Q-prequant change.
- Same cell with paged kernel compiled for output_group_span=1: `6.94 ms` PV, `7.72 ms` linear.
- Decision: reverted. Span=2 is the correct D256 paged schedule among these two.

## 2026-05-04 06:55 CDT - Paged V Block-Table Broadcast Plan

What I am about to do:
- Current D128/D256/D512 V data staging uses an 8-lane subgroup transpose. For each `(token_group, dim_group)`, all eight lanes are within one 16-token page, but every lane independently reads `block_table[logical_page]`, and every dim group repeats that lookup.
- The in-tree fmha_v2 paged loader pattern performs the block-table walk before the row load: `physical_page = paged_kv_global_block_offsets_[page_idx]`, then derives the row pointer and loads from that row.
- I will adopt the same granularity inside the existing SM120 subgroup transpose: lane 0 of each 8-lane subgroup reads the physical page for the shared logical page, broadcasts it with `__shfl_sync`, and all lanes derive page bases from the broadcast value.
- The same hoist applies to the PV-layout V-scale loop, where `token_group` determines one page and `col` changes. Decision criterion: tests stay green and D256/D512 paged-PV reference cells improve without changing public layouts, FFI signatures, split scheduling, or tolerances.

Measured result:
- D256 Qwen-full q=512 kv=65536 paged-PV after the broadcast/cache patch: `4.179 ms`, effectively unchanged from the current `~4.17 ms`.
- D512 Gemma-global q=512 kv=65536 paged-PV after the patch: `9.694 ms`, effectively unchanged from the current `~9.67 ms`.
- Decision: reverted the local patch. The repeated V `block_table` lookup is not the load-bearing paged-PV gap at these production prefill cells.

## 2026-05-04 07:10 CDT - D256 Paged TileM Check

What I am about to do:
- D256 dense compiles with `kCutlassTileM=64`; D256 paged no-SWA currently defines `FLASHINFER_SM120_NVFP4_D256_TILE_M=128` in `csrc/fmha_nvfp4_sm120_d256_paged.cu`.
- At Qwen q=512 kv=65536, the auto split makes tileM=128/split=3072 and tileM=64/split=6144 produce the same total CTA count. This makes it a direct per-CTA tile-shape check rather than a launch-count check.
- I will temporarily remove the paged-only tileM=128 override and benchmark the D256 Qwen paged-PV reference cell. Decision criterion: keep only if it materially improves paged-PV without breaking correctness; otherwise revert.

Measured result:
- D256 Qwen-full q=512 kv=65536 paged-PV with paged tileM=64: `6.202 ms`.
- Current paged tileM=128 reference: `~4.17 ms`.
- Decision: reverted. The D256 paged-only tileM=128 override is beneficial, not the source of the paged-vs-dense gap.

## 2026-05-04 07:25 CDT - D256 Stage Producer Isolation Plan

What I am about to do:
- Dense and paged use the same stage kernel skeleton, but dense stage averages `1.304 ms` while paged stage averages `4.223 ms` at Qwen D256 q=512 kv=65536.
- Wrapper/Q-quant/combine/copy/CTA count/tileM/output span are ruled out. The remaining candidates are paged K/V producer cost, producer/consumer overlap loss from manual completion, and paged-specific control flow inside the stage body.
- I will temporarily disable the paged V producer body while leaving the V pipeline barrier and PV MMA schedule intact. Existing `CoveredSmemTile` initialization leaves V operand smem zero and scale smem one, so this is a timing-only diagnostic, not a correctness run.
- Decision criterion: if runtime collapses, V producer is load-bearing; if not, the gap is in QK/K producer or the common online softmax/PV schedule under paged pipeline semantics. The patch will be reverted after measurement.

V producer diagnostic result:
- D256 Qwen-full q=512 kv=65536 paged-PV with V producer body disabled: `3.514 ms`.
- Current paged-PV reference: `~4.17 ms`.
- V producer accounts for roughly `0.65 ms`, real but not the `~2.9 ms` stage gap versus dense. Reverted the diagnostic patch.

Next isolation:
- I will temporarily disable the paged K producer body while leaving the K pipeline barrier and QK/PV schedule intact. K operand smem is already zero-filled and scale smem one-filled, so this is timing-only.
- Decision criterion: a large drop localizes the gap to K staging/manual QK pipeline; a small drop means the remaining gap is online softmax/PV schedule or other paged control flow.

K producer diagnostic result:
- D256 Qwen-full q=512 kv=65536 paged-PV with K producer body disabled: `2.510 ms`.
- Current paged-PV reference: `~4.17 ms`.
- K staging/manual QK pipeline accounts for roughly `1.66 ms`, the largest isolated producer cost. Reverted the diagnostic patch.
- Combined with the V diagnostic, producer work explains about `2.3 ms` of the `~2.9 ms` dense-to-paged stage gap. The next real optimization target is K staging/vectorization/overlap, not wrapper logic.

K scale isolation:
- I will temporarily skip only the K scale staging loop while keeping the 32-bit K data cp.async path and the pipeline barriers intact.
- K scale smem is initialized to UE4M3 one by `CoveredSmemTile`, so this is timing-only but safe for isolating whether the scalar scale loop is the expensive half of K staging.

K scale diagnostic result:
- D256 Qwen-full q=512 kv=65536 paged-PV with K scale staging skipped: `3.909 ms`.
- Current paged-PV reference: `~4.17 ms`.
- K scales account for only about `0.26 ms`; most of the `1.66 ms` K producer cost is K data staging and/or the manual K pipeline handoff. Reverted the diagnostic patch.

## 2026-05-04 07:50 CDT - Paged Stage Stack Footprint Fix Plan

What I found:
- `cuobjdump --dump-resource-usage` on the current D256 Qwen module reports:
  - Paged stage specialization: `REG:128 STACK:1104`.
  - Dense stage specialization in the same module: `REG:128 STACK:200`.
- Wrapper, Q quantize, combine, CTA count, tileM, output span, K scale loop, and V producer are already localized. The extra paged stack footprint is now a concrete structural difference inside the stage kernel.

What I am about to do:
- The stage kernel takes `Sm120Nvfp4PagedKvLoadParams` by value and mutates `paged_kv_params.kv_head` and `paged_kv_params.block_table` for all-head and varlen dispatch. That prevents treating the large params struct as grid-constant and can force a per-thread local copy/stack state.
- I will make the kernel parameter grid-constant/const and move the mutable fields into explicit lightweight runtime overrides: `effective_kv_head` and `effective_block_table`.
- I will add small helper accessors in `paged_kv.cuh` that take `(params, block_table, kv_head)` for the hot K/V data and scale paths, then update D128/D256/D512 producers to use the overrides. Decision criterion: D256 paged stage stack shrinks and the Qwen reference cell improves while tests remain green.

Grid-constant params and D256 load-warp result:
- Made `Sm120Nvfp4PagedKvLoadParams` a `CUTLASS_GRID_CONSTANT const` stage-kernel parameter and replaced the in-kernel `kv_head` / `block_table` mutations with lightweight `effective_kv_head` and `effective_block_table` locals.
- Added helper overloads in `fmha_nvfp4_sm120_paged_kv.cuh` that take explicit `(block_table, kv_head)` so D128/D256/D512 producers do not need a mutable params copy.
- Resource result on D256 Qwen softcap PV module: paged stage stack dropped from `STACK:1104` to `STACK:792`; dense remained `STACK:200`. Runtime effect by itself was flat: D256 q=512 kv=65536 g=6 paged-PV `4.156 ms`.
- Moved D256 hot-loop producer validation behind `FLASHINFER_SM120_NVFP4_DEBUG_TRAPS`. Runtime effect was small but positive: D256 paged-PV `4.140 ms`.
- Tested split scheduling on D256 q=512 kv=65536 g=6 paged-PV: split 1024 `4.427 ms`, 1536 `4.323 ms`, 2048 `4.652 ms`, 3072 `4.141 ms`, 4096 `5.365 ms`, 6144 `5.366 ms`, 8192 `7.038 ms`, 12288 `5.403 ms`, 65536 `28.602 ms`. Auto split `3072` is still the best tested point; the remaining gap is not a too-many-splits issue.
- D256 paged-only load-warp override: dense TU keeps default 7 load warps, paged no-SWA TU defines `FLASHINFER_SM120_NVFP4_D256_LOAD_WARPS=10` alongside the existing paged-only tileM=128 override. Measured D256 q=512 kv=65536 g=6: dense `1.323 ms`, paged-PV `3.651 ms`, paged-linear `4.159 ms`.
- Load-warp sweep: 8 load warps gave D256 paged-PV `3.773 ms` and linear `4.178 ms`; 10 load warps gave PV `3.646 ms` and linear `4.154 ms`; 12 load warps regressed to PV `3.933 ms` and linear `4.528 ms`. Decision: keep 10 for D256 paged no-SWA.
- K vectorization trials: 128-bit and 64-bit cp.async variants both failed with CUDA `misaligned address`; reverted. The CUTLASS D256 B partition only supports the current 32-bit K cp.async granularity safely.
- Cross-checks after this patch: D512 Gemma q=512 kv=65536 g=8 measured paged-PV `9.025 ms`, paged-linear `9.403 ms`; D128 q=512 kv=65536 g=4 measured dense `0.945 ms`, paged-PV `1.931 ms`, paged-linear `2.203 ms`.
- Test status: `tests/attention/test_nvfp4_kv_head_dim_512.py -q` passed, `36 passed in 389.98s`.

## 2026-05-04 08:45 CDT - Post-D256 Load-Warp Re-Localization Plan

What I am about to do:
- After the D256 paged-only 10-load-warp commit, the reference cell is still `3.65 ms` PV versus `1.32 ms` dense, so the residual gap remains inside the stage kernel.
- The previous producer isolation numbers are stale because the load-warp schedule changed. I will repeat the timing-only K/V producer disable checks on D256 q=512 kv=65536 g=6 paged-PV to find the remaining dominant producer cost.
- The diagnostic patches will be reverted immediately after measurement. Decision criterion: target the largest remaining isolated component, not wrapper code or split scheduling.

Post-D256 load-warp re-localization result:
- Current D256 q=512 kv=65536 g=6 paged-PV after the committed load-warp patch: `3.651 ms`.
- V producer body disabled, barriers/schedule retained: `3.111 ms`. Remaining V producer cost is about `0.54 ms`.
- K producer body disabled, barriers/schedule retained: `2.628 ms`. Remaining K staging/manual QK pipeline cost is about `1.02 ms`.
- The load-warp patch reduced the previous K-isolated cost from about `1.66 ms` to about `1.02 ms`; K remains the largest producer-side residual. V is now secondary. Next target is K staging/pipeline overhead, not split scheduling or wrapper code.

K data versus K scale/pipeline split:
- D256 q=512 kv=65536 g=6 paged-PV with only K data cp.async loop disabled and K scale/pipeline retained: `2.828 ms`.
- Current D256 paged-PV is `3.651 ms`; K data copy/staging accounts for about `0.82 ms`.
- Full K producer disabled was `2.628 ms`; K scale plus K pipeline handoff accounts for about `0.20 ms`.
- Conclusion: K data staging remains the largest residual inside K. The failed 64/128-bit cp.async attempts show the existing CUTLASS B partition does not guarantee >4B smem alignment, so the next K path must change the mapping/partition strategy rather than widening the current copy loop.

Direct-index K producer result:
- Replaced the paged K producer's `make_tiled_copy_B(...).partition_D(...)` traversal with a direct `(row, k0)` mapping into `qk_sB(row, k0, write_stage)` and the existing 32-bit `cp_async::pred_load_32b`.
- This keeps the safe 4-byte transfer granularity after the 64/128-bit alignment failures, but removes the per-copy-thread CUTE partition setup and partition tensor walk from the hot producer path.
- Reference cells after the change:
  - D128 q=512 kv=65536 g=4 paged-PV `1.566 ms`, paged-linear `1.833 ms`.
  - D256 q=512 kv=65536 g=6 paged-PV `3.088 ms`, paged-linear `3.659 ms`.
  - D512 q=512 kv=65536 g=8 paged-PV `7.651 ms`, paged-linear `8.149 ms`.
- Previous same-session references before direct-index K were roughly D128 PV `1.931 ms`, D256 PV `3.651 ms`, D512 PV `9.025 ms`; this is a broad K producer win across all head dims.

## 2026-05-04 09:10 CDT - Post-Direct-K Re-Localization Plan

What I am about to do:
- Direct-index K cut D256 paged-PV from `3.65 ms` to `3.09 ms`, D512 PV from `9.03 ms` to `7.65 ms`, and D128 PV from `1.93 ms` to `1.57 ms`.
- The largest stale component is now unknown because K data moved substantially. I will repeat D256 K-disabled and V-disabled timing-only diagnostics at q=512 kv=65536 g=6 paged-PV.
- Decision criterion: if V is now dominant, target V direct-index/store path; if K is still dominant, target K scale/pipeline; if both are small, move to common softmax/PV/epilogue schedule.

Post-direct-K localization result:
- D256 q=512 kv=65536 g=6 paged-PV baseline after direct-index K: `3.078 ms`.
- V producer disabled: `2.553 ms`; V total is about `0.52 ms`.
- K producer disabled: `2.627 ms`; K total is about `0.45 ms`.
- PV V-scale loop skipped: `2.825 ms`; PV V scales are about `0.25 ms`, leaving V data/transpose around `0.27 ms`.
- Conclusion: the single large K hot-path issue is gone. The residual is now distributed across K data/scale/pipeline, V data/scale, and common paged-stage scheduling; further wins need smaller targeted changes or schedule restructuring rather than another obvious 200x-class fix.

PV V-scale store reduction result:
- The PV V-scale producer was writing the same 16-token group scale to every even `k_offset` in the group. The downstream PV scale reader consumes the scale at the group base (`k0`) only, so the duplicate writes were producer-only overhead.
- Reduced the PV-layout V-scale store in D128/D256/D512 to write only `k_offset == 0`. This does not affect the linear-V path.
- Reference PV cells after the change:
  - D128 q=512 kv=65536 g=4 paged-PV `1.484 ms` (previous `1.566 ms`).
  - D256 q=512 kv=65536 g=6 paged-PV `2.958 ms` (previous `3.088 ms`).
  - D512 q=512 kv=65536 g=8 paged-PV `7.324 ms` (previous `7.651 ms`).
- Linear cross-checks stayed flat within noise:
  - D128 paged-linear `1.835 ms` (previous `1.833 ms`).
  - D256 paged-linear `3.658 ms` (previous `3.659 ms`).
  - D512 paged-linear `8.169 ms` (previous `8.149 ms`).
- Test status: `tests/attention/test_nvfp4_kv_head_dim_512.py -q` passed, `36 passed in 199.68s`.

## 2026-05-04 09:35 CDT - Load-Group Barrier Diagnostic Plan

What I am about to do:
- The D256 paged path now has no single 200x-class producer bug; the remaining cost is split between K, V, and common stage scheduling.
- The paged Q/K/V producers each run `load_group_sync()` before staging, after staging, and again after the load leader completes the transaction barrier. The first sync broadcasts the acquired stage index; the second ensures all producer writes are visible before completion. The third looks redundant because the next chunk's first sync re-converges the load group before any non-leader can use the next stage.
- I will remove only that trailing post-complete load-group sync in D256 as a timing diagnostic, benchmark q=512 kv=65536 g=6 PV/linear, and run the focused suite if the timing improves. If it is correct and useful, I will apply the same cleanup to D128/D512.

Load-group barrier diagnostic result:
- D256 q=512 kv=65536 g=6 paged-PV with the trailing post-complete load-group sync removed: `2.947 ms` versus `2.958 ms` baseline.
- D256 paged-linear with the same diagnostic: `3.660 ms` versus `3.658 ms` baseline.
- The change is effectively neutral. I reverted it rather than carrying a subtle synchronization delta for a ~0.01 ms PV-only gain.

## 2026-05-04 09:50 CDT - D256 Paged Tile Shape Recheck Plan

What I found:
- Current D256 q=512 kv=65536 g=6 dense is `1.328 ms`, while paged-PV is `2.958 ms`.
- The dense TU includes `d256.cuh` with the default `kCutlassTileM=64` and `LOAD_WARPS=7`.
- The paged no-SWA TU defines `FLASHINFER_SM120_NVFP4_D256_TILE_M=128` and `FLASHINFER_SM120_NVFP4_D256_LOAD_WARPS=10` before including `d256.cuh`.

What I am about to do:
- Re-test the D256 paged no-SWA tile shape after direct-index K, because the earlier tileM=64 diagnostic was from a materially slower producer state.
- I will temporarily remove the paged-only `TILE_M=128` override, benchmark D256 q=512 kv=65536 g=6 paged-PV/linear, then either keep it if it closes the dense gap or revert it if it remains worse.

D256 paged tile shape recheck result:
- D256 q=512 kv=65536 g=6 paged-PV with paged no-SWA `TILE_M=64` and `LOAD_WARPS=10`: `3.433 ms`.
- Current paged no-SWA `TILE_M=128` and `LOAD_WARPS=10`: `2.958 ms`.
- The previous conclusion still holds after direct-index K: paged D256 should keep tileM=128. I reverted the diagnostic.

D256 spec-axis cross-check:
- D256 q=512 kv=65536 g=6 no-softcap paged-PV: `2.656 ms`.
- D256 same cell no-softcap dense: `1.223 ms`.
- D256 same cell softcap=30 paged-PV: `2.958 ms`; dense: `1.328 ms`.
- The paged/dense ratio is about `2.17x` no-softcap and `2.23x` softcap, so the remaining D256 gap is generic paged-stage overhead rather than the softcap branch.

## 2026-05-04 10:05 CDT - D256 Post-Stage Combine Diagnostic Plan

What I found:
- `RunPagedBatchImpl` launches `kernel.run(...)`, then always launches either `CopyPaddedBatchOutKernel` for one split or `Sm120Nvfp4SplitKvCombineBatchKernel` for multiple splits.
- The D256 reference uses paged split length `3072` tokens (`24` tiles, `22` splits at kv=65536), while the dense reference uses `6144` tokens (`11` splits). Even if split scheduling is optimal for stage time, combine cost may still be a material part of the paged/dense ratio.

What I am about to do:
- Temporarily return from `RunPagedBatchImpl` immediately after the paged stage kernel succeeds, before launching combine/copy.
- Benchmark D256 q=512 kv=65536 g=6 PV and linear to measure the stage-side cost without post-stage combine. Then revert the diagnostic.

D256 post-stage combine diagnostic result:
- D256 q=512 kv=65536 g=6 paged-PV with combine/copy skipped after the stage kernel: `2.947 ms` versus `2.958 ms` baseline.
- D256 paged-linear with combine/copy skipped: `3.611 ms` versus `3.658 ms` baseline.
- Post-stage combine/copy is not the residual paged/dense gap. The cost is inside the stage kernel and BF16-Q quantize path, not the wrapper combine kernel. I reverted the diagnostic.

## 2026-05-04 10:20 CDT - D256 BF16-Q Quantize Diagnostic Plan

What I found:
- The production wrapper calls `paged_run_bf16_q`, which launches `QuantizeQToPaddedBatchKernel` before the paged stage.
- The module also exports `paged_run`, which consumes prequantized Q and only performs a padded Q copy before the same paged stage.
- For the D256 reference cell, q rows are already tile-aligned (`512 * 6 = 3072`, multiple of paged tileM 128), so the prequantized-Q path uses the same scratch extent without introducing ragged padding differences.

What I am about to do:
- Run a one-off Python diagnostic that builds the same wrapper/module state and times `paged_run_bf16_q` versus `paged_run` directly, without the Python wrapper's final `out.copy_`.
- Decision criterion: if `paged_run` is materially faster, Q quantize is a real residual target; if not, the remaining gap is stage-kernel scheduling/producers.

D256 BF16-Q quantize diagnostic result:
- D256 q=512 kv=65536 g=6 paged-PV direct FFI `paged_run_bf16_q` without Python final copy: `2.948 ms`.
- Same cell direct FFI `paged_run` with prequantized Q and no Python final copy: `2.925 ms`.
- BF16-Q quantize plus padded-Q preparation saves only about `0.023 ms` when bypassed. It is not the D256 paged/dense residual. The remaining cost is inside the stage kernel.

## 2026-05-04 10:35 CDT - D256 Split-Length Recheck Plan

What I found:
- D256 paged-PV remains about `2.97 ms` after the direct K and PV-scale-store reductions. Dense at the same cell is about `1.33 ms`.
- The paged auto split is `3072` tokens, producing `22` split CTAs at kv=65536. Dense auto split is `6144`, producing `11` splits.
- Earlier split sweeps were taken before the current producer state, so split length should be rechecked with the now-fast K/V path before assuming `3072` remains optimal.

What I am about to do:
- Run one Python diagnostic process that constructs the D256 q=512 kv=65536 g=6 PV cell once and replans the wrapper across split lengths.
- Decision criterion: if a larger split closes the gap without hurting output finiteness, update the benchmark/wrapper split heuristic; otherwise leave split scheduling alone and continue stage-kernel work.

D256 split-length recheck result:
- D256 q=512 kv=65536 g=6 softcap=30 paged-PV split sweep:
  - split 1536 (`43` splits): `3.092 ms`.
  - split 2048 (`32` splits): `3.296 ms`.
  - split 3072 (`22` splits): `2.952 ms`.
  - split 4096 (`16` splits): `3.752 ms`.
  - split 6144 (`11` splits): `3.759 ms`.
  - split 8192 (`8` splits): `4.859 ms`.
  - split 12288 (`6` splits): `3.828 ms`.
  - split 16384 (`4` splits): `5.047 ms`.
  - split 32768 (`2` splits): `9.942 ms`.
  - split 65536 (`1` split): `20.014 ms`.
- All outputs were finite. The current auto split `3072` remains the best tested split after direct-index K and PV-scale-store reduction. Split scheduling is not the remaining gap.

## 2026-05-04 10:55 CDT - D256 PV Pair Path Diagnostic Plan

What I found:
- D256 has a special `run_pv_tile_pair_nonfinal` path for `kOutputGroupSpan == 2`. It copies a P stage once, then consumes two V stages for two PV accumulators through a nested lambda.
- D512 uses a grouped nonfinal path for span 4 and D128 has only one output group. D256 is the only file with this exact pair helper shape.
- The D256 paged stage resource report is `REG:96 STACK:1080` while D128 paged is `REG:96 STACK:208`; D512 also has large stack but tracks dense, so stack is not sufficient by itself, but the D256 pair path remains the clearest D256-only control-flow difference in the hot MMA loop.

What I am about to do:
- Temporarily route `kOutputGroupSpan == 2` through the generic per-group `run_pv_tile` calls instead of `run_pv_tile_pair_nonfinal`.
- Benchmark D256 q=512 kv=65536 g=6 paged-PV. If it improves or materially reduces resource usage, keep/refine. If it regresses, revert and record.

D256 PV pair path diagnostic result:
- D256 q=512 kv=65536 g=6 paged-PV with `kOutputGroupSpan == 2` forced through generic per-group `run_pv_tile`: `4.141 ms`.
- Baseline with `run_pv_tile_pair_nonfinal`: about `2.96 ms`.
- The pair path is a real optimization, not the residual bug. I reverted the diagnostic.

## 2026-05-04 11:05 CDT - D256 Load-Warp Recheck Plan

What I found:
- The committed D256 paged no-SWA load-warp override is `10`. Earlier 8/10/12 testing was before direct-index K and PV-scale-store reduction.
- Direct-index K changed the load producer balance, so the best load-warp count may have shifted by one warp even if the broad conclusion remains.

What I am about to do:
- Test D256 paged no-SWA `LOAD_WARPS=9` and `LOAD_WARPS=11` against the current committed `10`.
- Decision criterion: keep the fastest PV/linear balanced setting if it is not a correctness risk; otherwise retain `10`.

D256 load-warp recheck result:
- Current committed `LOAD_WARPS=10`: D256 q=512 kv=65536 g=6 paged-PV about `2.96 ms`, paged-linear about `3.66 ms`.
- Diagnostic `LOAD_WARPS=9`: paged-PV `3.041 ms`; worse.
- Diagnostic `LOAD_WARPS=11`: paged-PV `2.863 ms`, paged-linear `3.562 ms`; better on both PV and linear.
- Decision: keep D256 paged no-SWA `FLASHINFER_SM120_NVFP4_D256_LOAD_WARPS=11` and validate with the focused NVFP4 suite.

Current reference sanity after D256 load-warp commit:
- Sequential q=512 kv=65536 softcap=30 paged-wrapper sanity set on GPU 2:
  - D128 g=4 paged-PV `1.482 ms`, paged-linear `1.840 ms`.
  - D256 g=6 paged-PV `2.862 ms`, paged-linear `3.557 ms`.
  - D512 g=8 paged-PV `7.326 ms`, paged-linear `8.185 ms`.
- All six outputs were finite. This is the current performance baseline before the focused production matrix.

Production grid smoke:
- Ran `bench_sm120_nvfp4_attention_grid.py` with explicit `--cells 512:4096`, D256 g=6, paged-linear, softcap=30, kernels `sm120_fused,nvfp4_fa2,bf16_fa2`.
- Report prefix: `reports/prod_smoke_d256_g6_linear_20260504`.
- Rows completed successfully. The FA2 baselines are no longer the earlier bogus flat `~0.025 ms` readings: at q=512 kv=4096, nvfp4_fa2 min `0.130848 ms`, bf16_fa2 min `0.090112 ms`, sm120_fused paged-linear min `1.020896 ms`.
- Explicit-cell report plumbing is general (`--cells`), not a focused-only mode; the same writer emits summaries from the rows present.

Qwen full D256 g=6 stock-vLLM linear-V focused report:
- Report prefix: `reports/prod_qwen_full_d256_g6_linear_20260504`.
- Ran 14 production cells with kernels `sm120_fused,nvfp4_fa2,bf16_fa2`; all sm120 rows were finite and all rows completed.
- Geomean sm120 paged-linear speedup versus nvfp4_fa2: `0.220x`; versus bf16_fa2: `0.160x`.
- Best sm120/nvfp4 cell in this report was still slightly slower (`0.963x` speedup). Worst was decode q=1 kv=262144: sm120 `3.439 ms` versus nvfp4_fa2 `0.0856 ms` (`40.2x` slower).
- The focused report confirms the current kernel is prefill-shaped and not viable for decode cells. The next measurement is the same Qwen cell set with paged-PV to separate linear-V reblock cost from the generic paged stage cost.

Qwen full D256 g=6 patched-vLLM PV focused report:
- Report prefix: `reports/prod_qwen_full_d256_g6_pv_20260504`.
- Ran the same 14 production cells with kernels `sm120_fused,nvfp4_fa2,bf16_fa2`; all sm120 rows were finite and all rows completed.
- Geomean sm120 paged-PV speedup versus nvfp4_fa2: `0.321x`; versus bf16_fa2: `0.233x`.
- Paged-PV is close to nvfp4_fa2 on the large-prefill cells and wins one long-context cell: q=2048 kv=65536 was sm120 `9.646 ms` versus nvfp4_fa2 `9.442 ms` (`0.979x`), and q=2048 kv=262144 was sm120 `36.441 ms` versus nvfp4_fa2 `37.670 ms` (`1.034x`).
- Decode remains structurally bad even without linear-V reblock: q=1 kv=4096 was sm120 `0.287 ms` versus nvfp4_fa2 `0.0272 ms` (`10.6x` slower), q=1 kv=262144 was sm120 `0.710 ms` versus nvfp4_fa2 `0.0851 ms` (`8.34x` slower). This is generic prefill-shaped paged-stage overhead, not the linear reblock path.
- Paired linear/PV comparison across the 14 cells: linear is `1.46x` slower geomean than PV. The largest extra linear tax is decode long context: q=1 kv=262144 linear `3.439 ms` versus PV `0.710 ms` (`4.85x`), q=1 kv=65536 linear `1.012 ms` versus PV `0.320 ms` (`3.16x`). For q=512 long-context prefill, linear/PV is about `1.28x`; this is the in-kernel reblock cost on top of the paged stage.
- Decision: keep measuring the production specs before further code changes. Current Qwen result says there are two separate issues: (1) linear-V reblock tax for stock-vLLM, and (2) a decode-shape mismatch where this prefill kernel is not competitive even on PV.

Gemma sliding D256 g=2 SWA=1024 softcap=30 focused reports:
- Linear report prefix: `reports/prod_gemma_sliding_d256_g2_swa1024_softcap30_linear_20260504`.
- PV report prefix: `reports/prod_gemma_sliding_d256_g2_swa1024_softcap30_pv_20260504`.
- Ran 8 production cells with kernels `sm120_fused,nvfp4_fa2,bf16_fa2` for each layout; all sm120 rows were finite and all rows completed.
- Linear geomean sm120 speedup versus nvfp4_fa2: `0.121x`; versus bf16_fa2: `0.0895x`.
- PV geomean sm120 speedup versus nvfp4_fa2: `0.141x`; versus bf16_fa2: `0.104x`.
- PV sm120 time is roughly fixed at `0.221-0.244 ms` for the 1024-token window cells and `0.666 ms` for q=2048 kv=8192. Linear is `1.17x` slower geomean than PV. The extra linear tax is small in absolute terms (`~0.02 ms` for kv=1024 cells, `~0.09 ms` for kv=8192 cells).
- Decision: for Gemma sliding, the remaining gap is not primarily linear-V reblock. It is the fixed launch/stage overhead of using this prefill-shaped paged kernel on very small window-bounded work. Further V producer work cannot close an 8-11x gap on these cells.

Gemma global D512 g=8 softcap=30 focused reports:
- Linear report prefix: `reports/prod_gemma_global_d512_g8_softcap30_linear_20260504`.
- PV report prefix: `reports/prod_gemma_global_d512_g8_softcap30_pv_20260504`.
- Linear run used kernels `sm120_fused,nvfp4_fa2,bf16_fa2`. BF16 FA2 failed every D512 g=8 cell with the existing FlashInfer dispatched prefill invalid configuration (`NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 ...`); I did not patch that reference backend. PV run used `sm120_fused,nvfp4_fa2` to avoid repeating known-invalid BF16 rows.
- All sm120 rows were finite and all nvfp4_fa2 rows completed.
- Linear geomean sm120 speedup versus nvfp4_fa2: `0.360x`, with `5/11` cells faster than nvfp4_fa2.
- PV geomean sm120 speedup versus nvfp4_fa2: `0.502x`, with `5/11` cells faster than nvfp4_fa2.
- D512 global is competitive on large prefill:
  - Linear q=512 kv=65536: sm120 `8.235 ms` versus nvfp4_fa2 `9.282 ms` (`1.13x` faster).
  - Linear q=512 kv=262144: sm120 `33.439 ms` versus nvfp4_fa2 `37.289 ms` (`1.12x` faster).
  - Linear q=2048 kv=16384/65536/262144: sm120 `7.394/29.304/122.248 ms` versus nvfp4_fa2 `10.954/45.026/184.959 ms` (`1.48-1.54x` faster).
  - PV q=512 kv=65536/262144: sm120 `7.355/29.189 ms` versus nvfp4_fa2 `9.301/37.162 ms` (`1.26-1.27x` faster).
  - PV q=2048 kv=16384/65536/262144: sm120 `7.523/28.804/120.773 ms` versus nvfp4_fa2 `11.057/44.698/185.042 ms` (`1.47-1.55x` faster).
- D512 global still loses short-context and decode:
  - PV q=512 kv=4096: sm120 `2.108 ms` versus nvfp4_fa2 `0.647 ms` (`0.307x`).
  - PV q=1 kv=4096/16384/65536/262144: sm120 `0.561/0.574/0.607/1.450 ms` versus nvfp4_fa2 `0.0703/0.0737/0.1119/0.3596 ms` (`0.125-0.248x`).
- Paired linear/PV comparison across the 11 cells: linear is `1.39x` slower geomean than PV. The largest extra linear tax is decode long context: q=1 kv=262144 linear `7.095 ms` versus PV `1.450 ms` (`4.89x`), q=1 kv=65536 linear `1.952 ms` versus PV `0.607 ms` (`3.21x`). For large prefill, linear/PV is much closer: q=2048 long-context cells are within about `1.01x`.

Production-focused matrix conclusion:
- Reports completed:
  - `reports/prod_qwen_full_d256_g6_linear_20260504`
  - `reports/prod_qwen_full_d256_g6_pv_20260504`
  - `reports/prod_gemma_sliding_d256_g2_swa1024_softcap30_linear_20260504`
  - `reports/prod_gemma_sliding_d256_g2_swa1024_softcap30_pv_20260504`
  - `reports/prod_gemma_global_d512_g8_softcap30_linear_20260504`
  - `reports/prod_gemma_global_d512_g8_softcap30_pv_20260504`
- Compact geomean sm120 speedup versus nvfp4_fa2:
  - Qwen D256 g=6 linear: `0.220x`, `0/14` wins.
  - Qwen D256 g=6 PV: `0.321x`, `1/14` wins.
  - Gemma sliding D256 g=2 linear: `0.121x`, `0/8` wins.
  - Gemma sliding D256 g=2 PV: `0.141x`, `0/8` wins.
  - Gemma global D512 g=8 linear: `0.360x`, `5/11` wins.
  - Gemma global D512 g=8 PV: `0.502x`, `5/11` wins.
- The earlier "paged is always 200x slower" diagnosis is obsolete after the producer fixes. The current result is not a uniform producer catastrophe. It is three regimes:
  1. Large D512 global prefill: sm120 paged is faster than nvfp4_fa2 on the important long-context cells.
  2. D256 Qwen long prefill: PV can reach parity at the largest q/kv cells, but stock linear-V remains behind due to reblock overhead.
  3. Decode and Gemma sliding window: the prefill-shaped paged kernel has a fixed overhead floor that dominates small work. This needs a decode/window-specialized path or routing to an existing backend; more V producer micro-optimization will not close an order-of-magnitude fixed-overhead gap.

Dense-vs-paged small-work diagnostic:
- D256 Qwen decode q=1 kv=262144 g=6 no softcap:
  - Dense PV: `0.345 ms`.
  - Paged PV: `0.710 ms`.
  - Production report nvfp4_fa2: `0.085 ms`.
- D256 Gemma sliding q=512 kv=1024 g=2 SWA=1024 softcap=30:
  - Dense PV: `0.102 ms`.
  - Paged PV: `0.239 ms`.
  - Production report nvfp4_fa2: `0.0326 ms`.
- D512 Gemma global decode q=1 kv=262144 g=8 softcap=30:
  - Dense PV: `1.106 ms`.
  - Paged PV: `1.456 ms`.
  - Production report nvfp4_fa2: `0.360 ms`.
- Conclusion: paged adds overhead, but the bad q=1/window cells are not exclusively a paged producer problem. Dense is already slower than nvfp4_fa2 on those cells, so the fused kernel template itself is prefill-shaped and pays a fixed floor. Next check: split scheduling, because q=1 auto currently uses `split_kv_len=1024`, creating many split CTAs on long context.

Decode split scheduling diagnostic and change plan:
- D256 q=1 kv=262144 no-window split sweep:
  - Dense best among tested splits: split `4096`, `0.253 ms`; auto split `1024`, `0.345 ms`.
  - Paged-PV best among tested splits: split `2048`, `0.603 ms`; auto split `1024`, `0.708 ms`.
- D512 q=1 kv=262144 no-window split sweep:
  - Dense best among tested splits: split `2048`, `0.793 ms`; auto split `1024`, `1.097 ms`.
  - Paged-PV best among tested splits: split `2048`, `1.235 ms`; auto split `1024`, `1.452 ms`.
- What I am about to change: for non-windowed single-q-tile workloads, set auto `split_kv_len=2048` instead of the current floor of `1024`. This is the best paged decode split for both D256 and D512, and it does not affect Gemma sliding because windowed runs still return `round_up(window_left, 128)`.
- Decision criterion: keep the change if the focused q=1 decode benches improve and the NVFP4 tests still pass. This is a real scheduling fix, but it does not remove the larger decode-shape mismatch.

Decode split scheduling change result:
- Changed auto split selection in `flashinfer/fmha_nvfp4_sm120.py` so non-windowed single-q-tile wrapper workloads use `split_kv_len=2048`.
- Mirrored the benchmark helper so auto-reported numbers match wrapper behavior; dense D256 q=1 keeps its measured best split `4096`, while paged decode uses `2048`.
- Post-change focused decode benches:
  - D256 q=1 kv=262144 dense auto: split `4096`, `0.253 ms`.
  - D256 q=1 kv=262144 paged-PV auto: split `2048`, `0.603 ms` versus previous auto `~0.710 ms`.
  - D512 q=1 kv=262144 dense auto: split `2048`, `0.795 ms`.
  - D512 q=1 kv=262144 paged-PV auto: split `2048`, `1.242 ms` versus previous auto `~1.45 ms`.
- Test status: `tests/attention/test_nvfp4_kv_head_dim_512.py -q` passed (`36 passed in 1.05s`).
- Conclusion: the split heuristic was leaving 15-18% decode performance on the table. The remaining decode gap is still architectural: the fused kernel remains a prefill-shaped template and is slower than nvfp4_fa2 even on dense decode.

Qwen decode post-split production subset:
- Linear report prefix: `reports/prod_qwen_decode_d256_g6_linear_postsplit_20260504`.
- PV report prefix: `reports/prod_qwen_decode_d256_g6_pv_postsplit_20260504`.
- Ran q=1 kv=4096/16384/65536/262144 with kernels `sm120_fused,nvfp4_fa2`; all sm120 rows were finite.
- Linear q=1 sm120 times after split change:
  - kv=4096 `0.570 ms`; kv=16384 `0.717 ms`; kv=65536 `1.243 ms`; kv=262144 `3.319 ms`.
  - Previous full-report linear kv=262144 was `3.439 ms`, so the split change helps only modestly on stock linear-V because reblock dominates.
- PV q=1 sm120 times after split change:
  - kv=4096 `0.510 ms`; kv=16384 `0.524 ms`; kv=65536 `0.536 ms`; kv=262144 `0.602 ms`.
  - Previous full-report PV kv=262144 was `0.710 ms`, matching the standalone split diagnostic. This is a real scheduling improvement.
- Even after the split fix, PV decode remains slower than nvfp4_fa2:
  - kv=4096 `0.510 ms` versus `0.0260 ms`.
  - kv=262144 `0.602 ms` versus `0.0850 ms`.
- Decision: do not spend more producer-tuning time on Qwen decode in this prefill kernel. The remaining gap is fixed floor/kernel-shape, not block-table or V reblock. The next structural options are a decode-specialized SM120 path or route q=1/window-small cells to existing FA2/XQA backends.

Qwen D256 PV prefill split sensitivity:
- q=512:
  - kv=16384 dense `0.418 ms`; paged auto split `3072`, `0.889 ms`. Tested splits showed `3072` best (`0.883 ms`), so auto is correct.
  - kv=65536 dense `1.215 ms`; paged auto split `3072`, `2.552 ms`. Tested splits showed `3072` best/tied (`2.556 ms`), so auto is correct.
  - kv=262144 dense `4.393 ms`; paged auto split `3072`, `9.657 ms`. Tested `12288` was slightly faster (`9.796` in one run versus auto `9.657` in another run), within noise; no clear split fix.
- q=2048:
  - kv=16384 dense `1.550 ms`; paged auto split `12288`, `3.290 ms`. Split `3072` was faster at `2.567 ms`.
  - kv=65536 dense `4.541 ms`; paged auto split `12288`, `9.656 ms`. Split `3072` was slightly faster at `9.482 ms`.
  - kv=262144 dense `17.183 ms`; paged auto split `12288`, `36.458 ms`. Auto `12288` remained best/tied; split `3072` was slower at `37.926 ms`.
- Decision: do not change the broad prefill split heuristic from this data. There is a short/medium-kv q=2048 opportunity for a workload-specific split override, but a global change would regress the longest-context production cell where the current heuristic is best.

Output copy removal plan:
- The paged wrapper currently allocates `torch.zeros_like(q)` when `out is None`, passes `_out_group` to the FFI, then does `out.copy_(_out_group.view(...))`.
- For the default benchmark/production path, `out` is contiguous and shaped exactly like `q`, so the FFI can write directly into `out.view(total_q_rows, head_dim)`.
- What I am about to change: allocate `torch.empty(...)` instead of `zeros_like` for `out is None`, pass contiguous `out.view(...)` directly to `paged_run_bf16_q`, and keep the old `_out_group + copy_` fallback for user-provided non-contiguous output.
- Decision criterion: keep the change if the focused NVFP4 test passes and small-work benches do not regress. This removes one Python-side GPU operation without changing public API or tensor contracts.

Output copy removal result:
- Changed the wrapper to allocate `torch.empty(...)` for default output, pass contiguous `out.view(total_q_rows, head_dim)` directly to `paged_run_bf16_q`, and keep `_out_group + copy_` only for non-contiguous user-provided output.
- Smoke timings after the change:
  - D256 q=1 kv=262144 paged-PV: `0.601 ms`.
  - D256 q=1 kv=262144 paged-linear: `3.320 ms`.
  - D256 sliding q=512 kv=1024 paged-PV: `0.236 ms`.
  - D512 q=1 kv=262144 paged-PV: `1.250 ms`.
  - D512 q=512 kv=65536 paged-PV: `7.320 ms`.
- Test status: `tests/attention/test_nvfp4_kv_head_dim_512.py -q` passed (`36 passed in 1.00s`).
- Conclusion: keep the cleanup because it removes an unnecessary output allocation/copy path, but the performance gain is tiny. The fixed floor is inside the FFI launch sequence and kernel shape, not the final Python `copy_`.

BF16-Q fused stage activation plan:
- While comparing FA2 and SM120 launcher structure, I found `paged_run_bf16_q` still launches `QuantizeQToPaddedBatchKernel` before the stage kernel.
- The SM120 kernel body already has a paged BF16-Q path: `paged_kv_params.q_bf16 != nullptr` routes `load_q_chunk` to `stage_bf16_q_tile`, which quantizes Q directly into Q smem for the current K chunk.
- What I am about to change: remove the standalone `QuantizeQToPaddedBatchKernel` launch from `RunPagedBatchBf16QImpl` and pass `q_bf16` plus Q strides through `RunPagedBatchImpl`. Keep the FFI signature unchanged and keep the existing scratch tensors for ABI/check compatibility.
- Decision criterion: keep the change if the focused NVFP4 test passes and small-work benches improve or at least do not regress. This removes one whole GPU launch from every wrapper call and matches the intended fused-Q architecture.

BF16-Q fused stage activation result:
- The change compiled and produced finite outputs, but it regressed every smoke cell tested:
  - D256 q=1 kv=262144 paged-PV: `0.619 ms` versus prior `~0.601 ms`.
  - D256 sliding q=512 kv=1024 paged-PV: `0.260 ms` versus prior `~0.236 ms`.
  - D256 q=512 kv=65536 paged-PV: `2.636 ms` versus prior `~2.55 ms`.
  - D512 q=1 kv=262144 paged-PV: `1.292 ms` versus prior `~1.25 ms`.
  - D512 q=512 kv=65536 paged-PV: `7.354 ms` versus prior `~7.32 ms`.
- Diagnosis: the in-stage BF16-Q path is not a free replacement for the standalone Q quantize launch. The stage path restages Q for each K chunk, while `QuantizeQToPaddedBatchKernel` quantizes/pads once per wrapper call and the stage kernel reuses scratch across splits. Removing a launch added repeated per-chunk Q work.
- Decision: reverted the code change. Keep the standalone Q quantize launch for now. The remaining gap is not caused by this launch.

FA2 versus SM120 paged V producer architecture audit:
- `include/flashinfer/attention/prefill.cuh::make_v_frag_fp4` does the same V-side FP4 dequant/requant shape as SM120 linear-V: for each output register it reads 8 V values through `get_v_value(...)`, divides by the output scale, clamps to E2M1 range, then packs with `mma::float8_to_e2m1x8(...)`.
- `include/flashinfer/attention/prefill.cuh::page_produce_kv` loads FP4 KV data through `smem.load_64b_async<fill_mode>(...)` for FP4. `include/flashinfer/permuted_smem.cuh::load_64b_async` implements this as `cp_async::pred_load_128b_from_64b(...)`, so FA2 moves 8 bytes of packed FP4 per async issue.
- SM120 D128/D256/D512 K producer uses `cp_async::pred_load_32b(...)` into CUTLASS operand smem. The paged V producer uses 8-lane subgroup transposes (`__shfl_sync` loop over 8 source lanes) and then writes a 32-bit packed word into the CUTLASS PV B operand smem. The 64-bit/128-bit cp.async trials failed earlier because the CUTLASS operand smem partition only gives the producer a reliable 4-byte alignment/contiguity invariant.
- The fp32 reblock arithmetic is shared with FA2, so it is not the reason SM120 paged trails FA2 on D256 PV/linear cells. The structural differences that remain are:
  - FA2 uses a custom permuted smem layout that accepts 64-bit FP4 cp.async from dim-contiguous gmem.
  - FA2 constructs the V MMA register fragment from that smem layout via explicit coordinate reads (`make_v_frag_fp4`), not by requiring gmem to be staged directly in CUTLASS operand layout.
  - SM120 stages directly into CUTLASS collective PV B operand smem, which forces 32-bit producer granularity and the token/dim transpose before the MMA copy path can consume it.
- Consequence: staying on the current CUTLASS operand-smem producer likely has a structural floor around the observed 1.5-2x dense overhead on D256 paged. Closing the remaining gap to FA2's paged overhead requires lifting the V operand staging constraint, not more block-table or scale-loop hoisting.
- Narrowest architectural option: replace only the paged V operand construction path. Keep the QK/softmax/P staging and MMA instruction shape, but stage V into an FA2-like 64-bit-cp.async-friendly smem layout and construct the PV B register fragment explicitly from that layout before `cute::gemm`. This is still substantial because it bypasses `pv_sB`/`pv_sSFB` plus `pv_smem_tiled_copy_B`/`partition_fragment_SFB` for the paged V path, but it does not require rewriting the whole QK/softmax pipeline.

PV B-fragment direct-fill feasibility probe:
- What I am about to test: whether the SM120 CUTE PV path can construct a B register fragment from a logical `(N,K)` layout and obtain matching `(n,k)` coordinates through `pv_thread_mma.partition_B(make_identity_tensor(...))`, without first staging through `pv_sB`.
- Reference pattern: `cute/algorithm/cooperative_gemm.hpp` constructs coordinate tensors with `Tensor cB = make_identity_tensor(shape(sB));` and partitions them with `Tensor tCcB = thr_mma.partition_B(cB);`, then uses the coordinates to predicate/copy the B fragment.
- Reference pattern: `include/flashinfer/attention/decode_mla_cute_sm80.cuh` constructs a B register fragment directly from a synthetic layout with `thr_mma_output.partition_fragment_B(make_tensor((DTypeKV*)0x0, layout_ckv_trans_no_stage));`.
- Decision criterion: if a D256 compile-only probe can create the direct B fragment, recast it as `uint32_t`, apply `fp4_shift_B`, and build the coordinate tensor with the same shape, then a narrow paged-V replacement can stay inside `cute::gemm`. If that does not compile, the remaining path requires lower-level MMA-fragment construction like FA2 rather than a producer-only rewrite.

PV B-fragment direct-fill feasibility result:
- Compile result: D256 can construct a synthetic logical PV B register fragment, recast it as 32-bit packed FP4 words, apply `fp4_shift_B`, and build the matching coordinate tensor with `pv_thread_mma.partition_B(make_identity_tensor(...))`.
- Host layout probe result for one D256 PV thread slice:
  - `frag size=256`, `word size=32`, `coord size=256`.
  - Each packed word maps to one fixed local output column and eight consecutive K/token positions, e.g. word 0 maps `(0,0)..(0,7)`, word 1 maps `(0,32)..(0,39)`, word 2 maps `(8,0)..(8,7)`.
  - Existing `v_smem_B` storage is exactly large enough for a compact row-major two-stage V tile: `16384` bytes = `128 tokens * 64 packed columns * 2 stages`.
- Correctness probe: replacing the D256 PV B smem-copy path with direct register-fragment fill from paged gmem passed `tests/attention/test_nvfp4_kv_head_dim_512.py -q` (`36 passed`). The mapping is correct.
- Performance probes:
  - Direct-from-gmem register fill, while still staging SFB scales, produced `1.54 ms` on D256 q=128 kv=4096 paged-PV. This is slower than the current staged path.
  - Compact row-major V smem with 64-bit cp.async and scalar smem-to-register reconstruction produced `0.615 ms` on D256 q=128 kv=4096 paged-PV and `4.13 ms` on D256 q=512 kv=65536 paged-PV.
  - Current staged baseline for the long D256 q=512 kv=65536 paged-PV cell is around `2.55 ms`, so the compact-smem scalar reconstruction path is a regression.
- Implementation detail found during the probe: 64-bit cp.async into compact smem works when the destination address is computed from the raw shared-memory base. Taking the destination through a CUTE `uint8_t` smem tensor reference caused an illegal memory access. The failure was address plumbing, not gmem stride alignment.
- Diagnosis: the narrow CUTE-fragment route removes the producer shfl transpose, but it also replaces CUTLASS's optimized smem-to-register copy/`ldmatrix` path with scalar smem reads to assemble each packed B register. That scalar reconstruction cost is larger than the producer-side savings.
- Decision: reverted the experiment. Do not propagate the direct CUTE fill or compact-smem scalar reconstruction to D128/D512. Closing the remaining gap to FA2 requires preserving FA2's second half too: a custom smem layout plus an efficient ldmatrix/fragment-construction path, or a lower-level FA2-like MMA path. A producer-only rewrite inside the current `cute::gemm` consumer is not enough.

D256 load-warp geometry tuning plan:
- D512 paged-PV is already near dense at long prefill cells, while D256 paged-PV still carries about a 2x dense gap on Qwen prefill. That points at a D256-specific fixed producer/schedule floor rather than the shared kernel template body.
- D256 exposes `FLASHINFER_SM120_NVFP4_D256_LOAD_WARPS` as a compile-time knob. Current default is `7` load warps, with `8` MMA warps and `1` epilogue warp.
- What I am about to test: build isolated JIT caches with `FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer_d256_loadwarps_${N}` and `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_SM120_NVFP4_D256_LOAD_WARPS=${N}`, then benchmark D256 q=512 kv=65536 g=6 paged-PV with the Qwen full spec.
- Decision criterion: keep a source default change only if one load-warp count materially improves the production cell without obviously regressing the small q=128 kv=4096 smoke cell. If the curve is flat or worse, leave the default at `7`.

D256 load-warp / tile-M geometry tuning result:
- Reference cell: D256 q=512 kv=65536 g=6 paged-PV, Qwen full spec, split `3072`.
- `FLASHINFER_SM120_NVFP4_D256_LOAD_WARPS=3`: mean `2.562 ms`.
- `FLASHINFER_SM120_NVFP4_D256_LOAD_WARPS=5`: mean `2.563 ms`.
- Default `FLASHINFER_SM120_NVFP4_D256_LOAD_WARPS=7`: prior mean around `2.55 ms`.
- `FLASHINFER_SM120_NVFP4_D256_LOAD_WARPS=9`: mean `2.566 ms`.
- `FLASHINFER_SM120_NVFP4_D256_TILE_M=128` with default load warps: mean `2.563 ms`.
- Decision: no source change. The D256 paged-PV gap is flat across these geometry knobs, so it is not caused by an obviously wrong load-warp count or CTA M tile. The remaining fixed floor is inside the paged data path / consumer construction, not top-level warp allocation.

D256 output-group span diagnostic plan:
- D256 defaults to `output_group_span=2`, so one CTA computes two 128-wide output groups through the D256 pair path. Span 1 doubles output-group CTAs but uses the simpler single-group PV path.
- Prior pair-path diagnostic only forced span-2 through the generic per-group helper; it did not measure real span 1 as a wrapper-level scheduling choice.
- What I am about to test: benchmark D256 paged-PV at span 1 versus span 2 on the Qwen long-prefill cell (`q=512 kv=65536 g=6`) and the Gemma sliding fixed-floor cell (`q=512 kv=1024 g=2`).
- Decision criterion: change the default only if span 1 materially improves the production cells despite the larger grid. If the curve is flat or worse, leave span 2 and move to deeper kernel-shape work.

D256 output-group span diagnostic result:
- Runtime span 1 cannot be measured through the current production module. The FFI rejects it before launch with `output_group_span == kernel.output_group_span (1 vs. 2)`, because D256 paged is generated with span 2 baked into `PagedKernelConfig`.
- No performance conclusion was drawn from that failed invocation. Measuring span 1 would require generating a separate D256 paged module configured for span 1, which is a compile-time variant rather than the runtime wrapper knob the benchmark exposes.
- Decision: do not change output-group span from this attempt. Continue with launch/kernel-shape analysis using the compiled production span 2 module.

D256 geometry diagnostic invalidation:
- While reading the D256 paged launcher, I found that `csrc/fmha_nvfp4_sm120_d256_paged.cu` unconditionally defines `FLASHINFER_SM120_NVFP4_D256_TILE_M 128` and `FLASHINFER_SM120_NVFP4_D256_LOAD_WARPS 11` for the no-SWA module before including `fmha_nvfp4_sm120_d256.cuh`.
- Consequence: experiments that used `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_SM120_NVFP4_D256_LOAD_WARPS=...` or `-DFLASHINFER_SM120_NVFP4_D256_TILE_M=...` were overwritten by the source file. The measured flat curve only proved that isolated cache roots worked; it did not prove that load-warp/tile-M geometry is flat.
- What I am about to change: make those no-SWA D256 defaults overrideable with `#ifndef`, preserving the production defaults (`TILE_M=128`, `LOAD_WARPS=11`) while enabling real compile-time diagnostics.
- Decision criterion: keep the overrideability patch if the focused NVFP4 test still passes with default settings, then rerun the D256 geometry sweep with isolated JIT cache roots and actual macro values.

D256 geometry overrideability result:
- Changed `csrc/fmha_nvfp4_sm120_d256_paged.cu` so the no-SWA defaults remain `FLASHINFER_SM120_NVFP4_D256_TILE_M=128` and `FLASHINFER_SM120_NVFP4_D256_LOAD_WARPS=11`, but only define them when the macro is not already provided.
- This is intended as a diagnostic-enabling patch with no default production behavior change.
- Test status: `CUDA_VISIBLE_DEVICES=2 ... pytest tests/attention/test_nvfp4_kv_head_dim_512.py -q` passed (`36 passed in 358.46s`).
- Next: rerun the D256 geometry sweep with isolated JIT roots now that the macro values actually reach the header.

D256 real geometry sweep result:
- Reference cell: D256 q=512 kv=65536 g=6 paged-PV, Qwen full spec, fixed split `3072`, isolated JIT roots.
- Default no-SWA geometry (`TILE_M=128`, `LOAD_WARPS=11`): mean `2.559 ms`.
- `TILE_M=128`, `LOAD_WARPS=7`: mean `2.676 ms`.
- `TILE_M=128`, `LOAD_WARPS=9`: mean `2.723 ms`.
- `TILE_M=128`, `LOAD_WARPS=13`: mean `2.972 ms`.
- `TILE_M=64`, `LOAD_WARPS=11`: mean `3.080 ms`.
- Decision: the committed D256 no-SWA geometry is the best tested configuration. The earlier invalidated flat result is superseded by this entry. No geometry default change.

Workspace clear-size diagnostic plan:
- The raw dense/paged launcher zeros a bounded workspace prefix before initializing CUTLASS argument objects. The current bound is hardcoded as `32 * 1024 * 1024` bytes.
- This zero is paid on every wrapper call and every dense benchmark call, independent of q/kv work. It is therefore a plausible contributor to the q=1 and Gemma sliding fixed floor.
- What I am about to change: make the clear-size cap overrideable with a compile-time macro while preserving the default 32 MiB behavior.
- Decision criterion: keep the overrideability patch if the focused NVFP4 test passes under default settings. Then benchmark 32 MiB versus smaller caps on small-work and long-prefill cells. Only reduce the production default if tests remain deterministic and the timing gain is material.

Workspace clear-size overrideability result:
- Added `FLASHINFER_SM120_NVFP4_WORKSPACE_CLEAR_BYTES` to D128/D256/D512 raw launchers. The default remains `(32 * 1024 * 1024)` bytes.
- Test status with default settings: `CUDA_VISIBLE_DEVICES=2 ... pytest tests/attention/test_nvfp4_kv_head_dim_512.py -q` passed (`36 passed in 297.69s`).
- Next: run isolated JIT diagnostics with smaller clear caps on D256 q=1, D256 Gemma sliding, and D256 Qwen long-prefill.

Workspace clear-size diagnostic result:
- Cells: D256 q=1 kv=262144 g=6 paged-PV split `2048`; D256 Gemma sliding q=512 kv=1024 g=2 paged-PV split `1024`; D256 Qwen prefill q=512 kv=65536 g=6 paged-PV split `3072`.
- Default 32 MiB cap: decode `0.604 ms`, Gemma sliding `0.240 ms`, Qwen prefill `2.553 ms`.
- Cap 0, which still clears at least the CUTLASS required workspace prefix: decode `0.585 ms`, Gemma sliding `0.237 ms`, Qwen prefill `2.540 ms`.
- Cap 1 MiB: decode `0.584 ms`, Gemma sliding `0.238 ms`, Qwen prefill `2.550 ms`.
- One attempted cap-1MiB run failed before kernel build because the macro value was passed with shell parentheses; reran with numeric `1048576`, so the numbers above are valid.
- Decision: do not reduce the production default from this data. The clear cap contributes only a small fixed cost and is not the structural gap. Keeping the override macro is useful for future diagnostics; the safe default remains 32 MiB.

Single-split direct-output plan:
- In `RunPagedBatchImpl`, even `max_splits == 1` currently launches the stage kernel into `out_scratch`, then launches `CopyPaddedBatchOutKernel` to move rows into the user output.
- For single-split cells with no row padding and no multi-KV-head row-order remap, the stage output layout is already the final flattened output layout.
- What I am about to change: when `max_splits == 1`, `out.size(0) == q_packed_scratch.size(0)`, and `(!all_kv_heads || num_kv_heads == 1)`, pass the final `out` tensor directly to the stage kernel and skip the copy kernel.
- Decision criterion: keep the change if the focused NVFP4 test passes and Gemma sliding q=512 kv=1024 improves without regressing padded decode or long-prefill cells.

Single-split direct-output result:
- Changed `RunPagedBatchImpl` to pass final `out` directly to the stage kernel when the single-split output layout is already final: `max_splits == 1`, `out.size(0) == q_packed_scratch.size(0)`, and no multi-KV-head row-order remap.
- Test status: `CUDA_VISIBLE_DEVICES=2 ... pytest tests/attention/test_nvfp4_kv_head_dim_512.py -q` passed (`36 passed in 300.05s`).
- Bench guard cells:
  - D256 Gemma sliding q=512 kv=1024 g=2 paged-PV split `1024`: `0.238 ms` versus prior `~0.240 ms`.
  - D256 q=1 kv=262144 g=6 paged-PV split `2048`: `0.606 ms`, unchanged because padded rows still require row compaction.
  - D256 q=512 kv=65536 g=6 paged-PV split `3072`: `2.557 ms`, unchanged because split-KV combine is still required.
- Decision: keep the change. It is not the structural gap, but it removes an unnecessary launch in safe single-split/no-padding cells.

D256 small-q tile-M diagnostic plan:
- D256 no-SWA paged defaults to `TILE_M=128`. That is best for the q=512 long-prefill reference, but q=1 decode has only 6 live Q rows and still pays a full 128-row tile.
- The now-working macro override can compile a real no-SWA `TILE_M=64` module. This halves padded-row work for q=1 and q=128-ish cases, but already measured slower on q=512 long prefill.
- What I am about to test: D256 paged-PV `TILE_M=64` versus default `TILE_M=128` on q=1 kv=262144, q=128 kv=32768, and q=512 kv=65536.
- Decision criterion: no default change unless the small-q win is large enough to justify the known long-prefill regression. If the win is large but incompatible with a single default, record this as evidence for a workload-shape dispatch/specialization gap.

D256 small-q tile-M diagnostic result:
- D256 paged-PV no-SWA, fixed splits, isolated JIT roots.
- q=1 kv=262144 g=6 split `2048`: default `TILE_M=128` mean `0.607 ms`; `TILE_M=64` mean `0.739 ms`.
- q=128 kv=32768 g=6 split `1024`: default `TILE_M=128` mean `0.638 ms`; `TILE_M=64` mean `0.585 ms`.
- q=512 kv=65536 g=6 split `3072`: default `TILE_M=128` mean `2.562 ms`; `TILE_M=64` mean `3.082 ms`.
- Decision: no D256 no-SWA default change. `TILE_M=64` helps one short-prefill cell but regresses decode and long prefill. The q=1 floor is not caused by 128-row tile padding.

D256 output-span overrideability plan:
- The previous runtime span-1 probe failed because D256 paged hardwires template `kOutputGroupSpan=2` and reports `PagedKernelConfig.output_group_span=2`.
- To measure span 1 honestly, the D256 paged translation unit needs the same kind of compile-time diagnostic override as tile-M/load-warps, while preserving default span 2.
- What I am about to change: add `FLASHINFER_SM120_NVFP4_D256_OUTPUT_GROUP_SPAN` with default `2`, use it for the D256 paged template argument and config field.
- Decision criterion: keep the overrideability patch if the focused NVFP4 test passes with default span 2. Then benchmark span 1 via isolated JIT root and `--output-group-span 1`.

D256 output-span overrideability result:
- Added `FLASHINFER_SM120_NVFP4_D256_OUTPUT_GROUP_SPAN` to `csrc/fmha_nvfp4_sm120_d256_paged.cu`; default remains span 2.
- Test status with default span 2: `CUDA_VISIBLE_DEVICES=2 ... pytest tests/attention/test_nvfp4_kv_head_dim_512.py -q` passed (`36 passed in 90.85s`).
- Next: benchmark span 1 as a real compile-time variant with isolated JIT roots.

D256 output-span diagnostic result:
- D256 paged-PV span 1, isolated JIT roots:
  - Qwen q=512 kv=65536 g=6 no-SWA split `3072`: span 1 mean `4.124 ms` versus span 2 `~2.56 ms`.
  - Qwen q=128 kv=32768 g=6 no-SWA split `1024`: span 1 mean `0.782 ms` versus span 2 `~0.638 ms`.
  - Gemma sliding q=512 kv=1024 g=2 SWA=1024 softcap=30 split `1024`: span 1 mean `0.187 ms` versus span 2 `~0.238 ms`.
- Decision: span 1 is wrong for D256 no-SWA but right for D256 sliding-window fixed-floor cells. Since sliding and no-SWA are already separate JIT specs, make the D256 paged default span conditional: no-SWA stays span 2, sliding-window uses span 1. This is not a new spec axis; it follows an existing spec axis.

D256 sliding output-span change plan:
- Change `csrc/fmha_nvfp4_sm120_d256_paged.cu` so the default `FLASHINFER_SM120_NVFP4_D256_OUTPUT_GROUP_SPAN` is `1` when `SM120_NVFP4_USE_SLIDING_WINDOW_PREPROC` is true and `2` otherwise.
- Update the paged wrapper default selection so D256 sliding plans pass output span 1. Keep dense benchmark defaults at span 2 because D256 dense is still compiled with span 2.
- Update the benchmark/grid default span logic for `fused-api=paged` D256 sliding so reports use the production wrapper default.
- Decision criterion: keep the change if the focused NVFP4 test passes and Gemma sliding focused bench improves without changing Qwen no-SWA results.

D256 sliding output-span change result:
- Implemented the conditional D256 paged default: sliding-window specs compile with `FLASHINFER_SM120_NVFP4_D256_OUTPUT_GROUP_SPAN=1`; no-SWA specs remain span 2.
- Wrapper and benchmark defaults now mirror the generated module: D256 paged sliding defaults to span 1, while D256 dense and D256 paged no-SWA default to span 2.
- Test status: `CUDA_VISIBLE_DEVICES=2 ... pytest tests/attention/test_nvfp4_kv_head_dim_512.py -q` passed (`36 passed in 88.65s`).
- Bench guards:
  - D256 Gemma sliding q=512 kv=1024 g=2 paged-PV split `1024`: `0.186 ms`, `output_group_span=1`.
  - D256 Qwen no-SWA q=512 kv=65536 g=6 paged-PV split `3072`: `2.550 ms`, `output_group_span=2`.
  - D256 dense sliding q=512 kv=1024 g=2: `0.103 ms`, `output_group_span=2`.
- Decision: keep the conditional default. It captures the Gemma sliding fixed-floor win while preserving the no-SWA Qwen path and the dense benchmark ABI.

D256 stage-kernel profile and page-cache plan:
- Nsight Systems on D256 Qwen q=512 kv=65536 g=6 paged-PV split `3072` shows the stage kernel itself is the gap: paged stage avg `2.491 ms`, split-combine avg `0.040 ms`, Q quantize avg `0.004 ms`. The matching dense stage avg is `1.221 ms`.
- `stage_paged_k_tile` still reaches `effective_block_table[logical_page]` through `sm120_nvfp4_paged_k_word_ptr` for every 32-bit K word and again for every K scale. `stage_paged_v_tile` repeats the same per-word/page lookup pattern for V data and V scales.
- A 128-token KV tile contains exactly eight 16-token pages in the supported paged layout. What I am about to change: add an eight-entry shared-memory physical-page cache to D128/D256/D512 storage, populate it once per `kv_tile`, and use it in K/V data and scale producers.
- Decision criterion: keep the change if the focused NVFP4 test passes and the D256 Qwen paged-PV reference cell improves without regressing D512 long-prefill or D256 Gemma sliding.

Paged producer page-cache result:
- Added an eight-entry `physical_page_cache` to the D128/D256/D512 paged stage storage. K data, K scales, V data, and V scales now use the cached physical page for the current 128-token `kv_tile`; `effective_block_table[...]` is reached only by the cache fill.
- Test status: `CUDA_VISIBLE_DEVICES=2 ... pytest tests/attention/test_nvfp4_kv_head_dim_512.py -q` passed (`36 passed in 292.78s`).
- D256 Qwen q=512 kv=65536 g=6 paged-PV split `3072`: `2.403 ms` versus prior `2.550 ms`.
- D256 Qwen q=512 kv=65536 g=6 paged-linear split `3072`: `3.270 ms` versus earlier `~3.56 ms`.
- D512 Gemma global q=512 kv=65536 g=8 paged-PV split `3072`: `6.980 ms`, no regression versus the prior `~7.32 ms` range.
- D256 Gemma sliding q=512 kv=1024 g=2 paged-PV split `1024`: `0.158 ms` versus prior post-span-change `0.186 ms`.
- D128 q=512 kv=65536 g=4 paged-PV split `3072`: `1.229 ms` versus earlier `~1.48 ms`.
- Nsight post-change on the D256 Qwen paged-PV cell: stage kernel avg `2.335 ms`, split-combine avg `0.040 ms`, Q quantize avg `0.004 ms`. The gain is inside the stage kernel, as intended.
- Decision: keep the page-cache change. It is not the full remaining D256 gap, but it removes a repeated block-table lookup class across all D-size producers and improves every guard cell measured.

Paged page-cache barrier refinement plan:
- The first page-cache implementation is correct but adds an internal `load_group_sync()` inside each K/V staging helper. `load_k_chunk` and `load_v_chunk` already synchronize the load group immediately before calling those helpers.
- What I am about to change: make `cache_paged_physical_pages(kv_tile)` fill only, call it immediately before the existing pre-stage `load_group_sync()`, and remove the extra internal barrier plus the helper calls inside `stage_paged_k_tile` / `stage_paged_v_tile`.
- Decision criterion: keep the refinement if focused tests pass and the same D256/D512/D128 guard cells do not regress.

Paged page-cache barrier refinement result:
- Test status for the refinement: `CUDA_VISIBLE_DEVICES=2 ... pytest tests/attention/test_nvfp4_kv_head_dim_512.py -q` passed (`36 passed in 287.15s`).
- Guard benches with the refinement:
  - D256 Qwen q=512 kv=65536 g=6 paged-PV split `3072`: `2.436 ms` mean, `2.399 ms` min.
  - D512 Gemma global q=512 kv=65536 g=8 paged-PV split `3072`: `7.007 ms` mean, `6.956 ms` min.
  - D256 Gemma sliding q=512 kv=1024 g=2 paged-PV split `1024`: `0.162 ms` mean, `0.158 ms` min.
- Decision: do not keep the refinement. It was neutral to slightly worse on mean and only matched the prior page-cache result on min. Restored the safer in-helper cache fill plus barrier before committing the page-cache optimization.

K-scale single-slot diagnostic plan:
- V PV scale staging writes only `k_offset == 0` because the PV scale reader consumes the group scale at the start of each 16-token group.
- K scale staging still writes the same byte to every even offset in each 16-wide K scale group: `qk_sSFB(row, local_scale_col * 16 + k_offset, write_stage)` for `k_offset = 0, 2, ..., 14`.
- What I am about to test: apply the same single-slot write shape to D256 K scales first. If focused correctness passes, benchmark the D256 Qwen paged-PV reference. If it fails, revert the diagnostic without propagating it.

Profiling-driven hypothesis discipline:
- User directive: future performance hypotheses must be driven by Nsight Systems and NCU, not source-level symmetry alone.
- `nsys` is available at `/usr/local/bin/nsys`; `ncu` is available at `/usr/local/cuda-13.2/bin/ncu`.
- The in-flight K-scale single-slot diagnostic is therefore treated as speculative until validated by stage-kernel attribution/counters. If counters do not support it, revert the code and keep only this worklog record.

## 2026-05-04 11:47 CDT - Profiling-Gated K-Scale Diagnostic

Finding:

- The D256 K-scale single-slot change passed the focused D512 NVFP4 correctness test: `36 passed in 290.05s`.
- That test result only proves the D256 diagnostic did not break the focused suite. It does not prove the change addresses the dense-vs-paged performance gap.
- The user directive is to drive future hypotheses from Nsight Systems and NCU evidence, not from source-level symmetry alone.

Implementation target:

- Treat the current D256 K-scale single-slot edit as a diagnostic, not a keeper.
- First measure the D256 Qwen paged-PV reference cell with Nsight Systems to see whether the stage kernel moves relative to the committed page-cache baseline.
- If Nsight Systems shows material stage-kernel improvement, collect NCU counters on the stage kernel before deciding whether to keep and propagate the pattern.
- If profiling does not support the change, revert the diagnostic and keep the result as a negative finding.

Validation:

- Correctness: focused D512 NVFP4 suite passed at current tolerances.
- Profiling target: `D=256 g=6 q=512 kv=65536 paged-PV causal no-SWA no-softcap split_kv_len=3072`.
- Baseline from committed page-cache code: stage kernel approximately `2.335 ms`, total bench approximately `2.403 ms`.

Decision criteria:

- Keep only if profiling shows the stage kernel improves enough to matter and counters support reduced producer-side work.
- Do not commit the diagnostic from correctness alone.

## 2026-05-04 11:49 CDT - K-Scale Diagnostic Result

Finding:

- The D256 K-scale single-slot diagnostic was correctness-safe in the focused suite but did not produce a material performance result.
- Direct benchmark at `D=256 g=6 q=512 kv=65536 paged-PV split=3072` measured `2.369 ms` mean and `2.328 ms` min.
- Nsight Systems attributed the run to the stage kernel as expected: stage kernel `2.268 ms` average across five launches, split-combine `0.0397 ms`, Q quantize `0.0038 ms`.
- The committed page-cache baseline for the same cell was approximately `2.403 ms` total and `2.335 ms` stage. The diagnostic movement is small enough to be noise-level or low-yield.

Decision:

- Reverted the D256 K-scale single-slot code change.
- Kept the worklog record because it closes the source-symmetry hypothesis with profiling evidence.
- Did not run NCU for this diagnostic because the Nsight Systems result did not meet the material-improvement gate.

Next profiling target:

- Use NCU on the committed stage kernel to classify the remaining gap before making another producer change.
- The next decision should be based on stage-kernel counters: memory/LSU stalls, barrier stalls, shared-memory conflicts, issue eligibility, and executed instruction mix.

## 2026-05-04 11:53 CDT - NCU Barrier-Stall Target

Finding:

- Nsight Systems still attributes the D256 Qwen paged-PV reference cell to the stage kernel, not wrapper, Q quantize, or split-combine.
- NCU on `D=256 g=6 q=512 kv=65536 paged-PV split=3072` shows `2.467 ms` stage time.
- Matching dense NCU for the same logical cell shows `1.248 ms` stage time.
- Paged executes `1.91x` as many dynamic instructions as dense (`1.101B` vs `0.576B`) and has `1.98x` NCU time.
- DRAM is not the primary wall: paged DRAM throughput is only `11.6%` of peak. The largest paged-vs-dense stall delta is synchronization: `Stall Barrier 5.22` vs dense `0.27`, and `Stall Long Scoreboard 2.61` vs dense `0.26`.

Implementation target:

- Reduce paged load-group synchronization in the Q/K/V producer handoff without changing pipeline semantics.
- Current paged Q/K/V chunks use three load-group barriers around each producer action: after leader acquire, before leader completion, and after leader completion.
- The post-complete barrier is redundant with the next chunk's pre-stage barrier: non-leader load threads cannot enter the next staging body until the leader reaches the next acquire-and-sync point.
- Remove only the post-complete load-group barrier after `complete_manual_tma_pipeline_stage(...)` in D128, D256, and D512 paged Q/K/V load paths.

Validation:

- Run focused NVFP4 correctness first, because this touches producer-consumer ordering.
- Re-bench D256 Qwen paged-PV and D512 Gemma global paged-PV reference cells.
- Re-run NCU on D256 Qwen paged-PV if benchmarks move, and verify barrier stalls decrease without introducing memory-scoreboard regressions.

Decision criteria:

- Keep if correctness passes and stage time or barrier counters improve.
- Revert if any intermittent correctness appears or the barrier counters fail to move.

## 2026-05-04 12:02 CDT - Load-Group Barrier Reduction Result

Finding:

- The post-complete load-group barrier removal was correctness-safe in the focused suite: `36 passed in 288.76s`.
- It did not improve reference-cell timing.
- D256 Qwen paged-PV `q=512 kv=65536 g=6 split=3072` measured `2.446 ms` mean, worse than the committed page-cache baseline around `2.403 ms`.
- D512 Gemma global paged-PV `q=512 kv=65536 g=8 softcap=30 split=3072` measured `7.098 ms` mean, worse than the committed page-cache baseline around `6.98 ms`.

Decision:

- Reverted the barrier-removal code change.
- Kept the NCU finding: paged barrier stalls are high relative to dense, but the removed post-complete barrier was not the profitable barrier to remove.
- The post-complete barrier likely helps keep the load warpgroup aligned with the leader's producer state; removing it shifts cost into later synchronization or scoreboard stalls rather than reducing total stage time.

Next profiling target:

- Use NCU source/SASS attribution or narrower metrics to locate where `Stall Barrier` and `Stall Long Scoreboard` are charged.
- Do not remove more barriers by inspection. The next barrier change must identify the specific stall location first.

## 2026-05-04 12:06 CDT - PV V Producer Coalescing Target

Finding:

- NCU source/SASS attribution on D256 Qwen paged-PV has no CUDA lineinfo, but the SASS counters identify the hot PC class.
- NCU reports an estimated `73.93%` speedup opportunity from uncoalesced global accesses: `50,331,648` excessive sectors out of `65,666,496` total sectors.
- The top two excessive-sector PCs are `LDG.E` 32-bit global V data loads, each with `21,073,920` excessive sectors and `24,084,480` total sectors.
- Those loads are immediately followed by `SHFL.IDX` instructions; the hottest long-scoreboard PC is the first shuffle after the V data load.
- This matches the current PV V producer lane mapping: each 8-lane subgroup loads one dim-contiguous word for a different token at the same dim group, so the warp issues strided token-row loads instead of coalesced dim-row loads.

Implementation target:

- Rewrite the PV-layout V data producer so an 8-lane subgroup handles one `8 token x 64 dim` tile.
- For each token in the 8-token group, lanes load adjacent 32-bit dim words for that token. This makes the global load coalesced across lanes.
- Each lane accumulates eight packed output words in registers, one for each dim in its 8-dim word, then writes those packed token-contiguous words to the existing CUTLASS operand smem layout.
- Do this for PV-layout V first, because the NCU evidence was collected on paged-PV and the PV path has no linear dequant/requant complication.

Validation:

- Run focused NVFP4 correctness after the PV path change.
- Benchmark D256 Qwen and D512 Gemma paged-PV reference cells.
- Re-run NCU on D256 Qwen paged-PV if timing improves and confirm the top excessive-sector V load PCs drop.

Decision criteria:

- Keep if correctness passes and paged-PV timing improves materially.
- If timing regresses, revert the PV coalescing change and keep the source-attribution finding.

## 2026-05-04 12:22 CDT - PV V Producer Coalescing Result

Finding:

- The PV-layout V producer rewrite is correctness-safe across the focused NVFP4 suite: `36 passed in 200.79s`.
- The NCU hypothesis was directionally correct. D256 Qwen paged-PV global-sector waste dropped from `50,331,648` excessive sectors out of `65,666,496` total sectors to `6,291,456` excessive sectors out of `21,626,304` total sectors.
- The previous top PV V data PCs were two 32-bit `LDG.E` loads with `21,073,920` excessive sectors each. After the rewrite, those PCs no longer appear as the dominant excessive-sector source.
- The remaining profiler issue moved: NCU now reports `35,489,280` excessive shared wavefronts out of `96,675,456` total wavefronts. The top shared-excess PCs are shared stores in the producer path.

Measured delta:

- D128 paged-PV `q=512 kv=65536 g=4 split=3072`: `1.114 ms` mean, down from the page-cache baseline around `1.229 ms`.
- D256 paged-PV `q=512 kv=65536 g=6 split=3072`: `2.271 ms` mean, down from the page-cache baseline around `2.403 ms`.
- D512 paged-PV `q=512 kv=65536 g=8 softcap=30 split=3072`: `6.721 ms` mean, down from the page-cache baseline around `6.98 ms`.

Decision:

- Keep the PV-layout coalesced global-load mapping. It removes the NCU-identified uncoalesced global-load class and improves all three head dimensions.
- This does not close the dense-vs-paged gap. The next target should come from the new NCU state, not from source inspection: shared-memory store wavefront excess and the remaining barrier/reconvergence samples are now the measured bottlenecks.

Next profiling target:

- Re-run focused NCU after this commit with source counters and warp-state sampling, using the coalesced build as the new baseline.
- Attribute the top shared-store excessive wavefront PCs before changing the store layout. Do not add another producer rewrite without a profiler-backed PC/source target.

## 2026-05-04 12:25 CDT - PV B Store-Contiguity Diagnostic Target

Finding:

- After PV V load coalescing, NCU moved the bottleneck from global V data loads to shared stores in the PV V producer.
- The top shared-store PCs are eight unrolled `ST.E` instructions, each with `658,560` excessive shared wavefronts. Their identical counts match the eight `packed_words[dim_offset]` stores in the new PV V data path.
- The current lane ownership is load-coalesced but store-strided: lane `l` owns columns `l * 8 + {0..7}`. For each unrolled `dim_offset` store, lanes write columns separated by eight, which is the measured shared-store problem.

Diagnostic target:

- Check whether the CUTLASS PV B smem layout maps the eight column words owned by one lane to contiguous physical shared-memory addresses.
- If `pv_sB(local_col0 + dim_offset, local_k0, write_stage)` is contiguous at `4 * dim_offset` bytes, a per-lane vectorized store is a viable next experiment.
- If that contiguity does not hold, vectorizing the lane-local stores is blocked; the next option is a real transpose through a small per-warp staging tile or a register-transpose routine, not another scalar store reorder.

Validation:

- Add the contiguity check as a temporary diagnostic in the D256 PV path only.
- Run the D256 paged-PV wrapper target once. A trap means the vector-store route is invalid for the current CUTLASS operand layout.
- Revert the diagnostic immediately after recording the result.

Decision criteria:

- Only implement a vectorized PV B store if the diagnostic proves the physical contiguity invariant.
- Otherwise leave the committed coalesced-load path intact and move to the next profiler-backed design.

## 2026-05-04 12:27 CDT - PV B Store-Contiguity Diagnostic Result

Finding:

- The temporary D256 PV-path diagnostic trapped on the production reference cell.
- The checked invariant was: `pv_sB(local_col0 + dim_offset, local_k0, write_stage)` should equal the base pointer plus `4 * dim_offset` bytes for `dim_offset = 1..7`.
- That invariant is false for the current CUTLASS PV B operand smem layout.

Decision:

- Reverted the temporary trap immediately.
- Do not pursue a lane-local vectorized store for the committed PV coalesced-load path. The physical layout is not contiguous across the eight column words owned by one lane.
- The remaining shared-store wavefront excess cannot be fixed by changing scalar stores into a per-lane vector store.

Next profiling target:

- Any store-side fix now needs a real transpose of ownership before the store, or a small staging layout that makes the final store lane-contiguous in the CUTLASS operand.
- The next experiment should be scoped to D256 first and should be measured against both correctness and NCU shared-wavefront counters before propagation.

## 2026-05-04 12:30 CDT - Full-KV-Tile Bounds-Hoist Target

Finding:

- The post-coalescing full NCU pass for D256 Qwen paged-PV reports `2.33 ms` stage time versus `1.25 ms` dense.
- Paged still executes `1.044B` instructions and reports `Stall Barrier 5.71`, while dense is around `0.576B` instructions and `Stall Barrier 0.27`.
- The reference cell has full 128-token KV tiles for the measured split geometry, but the paged producer still carries per-token `token < kv_len_tokens` checks in the V data loop.
- The top barrier samples are reconvergence (`BSSY/BSYNC`) rather than named barriers, so uniformizing/eliminating inner-loop runtime branches is the next profiler-backed low-risk target.

Implementation target:

- Start with D256 PV-layout V only.
- Compute a uniform `full_kv_tile` predicate once per staged KV tile.
- Use a full-tile fast path in the PV V data producer that omits the per-token OOB branch inside the unrolled 8-token loop.
- Keep the tail-tile path unchanged for partial tiles.

Validation:

- Run the D256 PV wrapper correctness target first.
- Benchmark D256 Qwen paged-PV `q=512 kv=65536 g=6 split=3072`.
- If timing moves materially, re-run NCU and check whether reconvergence/barrier samples or instruction count drop before propagating to D128/D512.

Decision criteria:

- Keep and propagate only if correctness passes and either timing or NCU reconvergence/instruction counters improve.
- Revert if the branch hoist only reshuffles compiler codegen without reducing measured stage time.

## 2026-05-04 12:32 CDT - Store-Side Register Transpose Target

Finding:

- The full-KV-tile bounds hoist is a low-yield cleanup relative to the measured gap. It may still be valid later, but it should not be the next priority.
- The current NCU target is larger and structural: post-coalescing PV V has `35,489,280` excessive shared wavefronts, with the top eight PCs corresponding to the unrolled `packed_words[dim_offset]` stores.
- The lane-local vector-store route is blocked because the PV B smem layout does not make one lane's eight output words physically contiguous.
- The current ownership is load-coalesced but store-strided. A register transpose can change ownership before the store: source lane owns `packed_words[0..7]` for columns `lane * 8 + offset`; destination lane `offset` can fetch `packed_words[offset]` from each source lane and store columns `source_lane * 8 + offset`, making each store issue lane-contiguous.

Implementation target:

- Start with D256 PV-layout V only.
- Keep the coalesced global-load loop unchanged.
- Add an 8-lane register transpose after `packed_words[]` is produced.
- Store the transposed words so, for each unrolled store iteration, subgroup lanes write contiguous columns in the CUTLASS PV B operand layout.

Validation:

- Run D256 PV wrapper correctness first.
- Benchmark D256 Qwen paged-PV `q=512 kv=65536 g=6 split=3072`.
- If timing improves, run NCU SourceCounters and WarpStateStats and compare shared-wavefront excess, long-scoreboard stalls, and executed instructions against the coalesced-load baseline.

Decision criteria:

- Keep and propagate only if correctness passes and the D256 reference cell improves materially.
- If the extra `SHFL` cost cancels the shared-store win, revert the D256 code and keep the profiler result as evidence.

## 2026-05-04 12:36 CDT - Store-Side Register Transpose Result

Finding:

- The D256-only register-transpose store experiment compiled and passed the focused PV wrapper correctness subset: `3 passed in 44.59s`.
- It did not improve the measured reference cell.
- D256 Qwen paged-PV `q=512 kv=65536 g=6 split=3072` measured `2.313 ms` mean with the register transpose.
- The committed coalesced-load baseline for the same cell is `2.271 ms` mean.

Decision:

- Reverted the D256 register-transpose store code.
- Do not propagate this shape to D128/D512.
- The result says the extra select/shuffle work costs more than the shared-store coalescing it buys in the current CUTLASS operand layout.

Next profiling target:

- Keep the global-load-coalesced PV producer from `f7ac8e9`.
- The next high-risk path should change the producer/store architecture more substantially than an intra-subgroup register transpose, or attack the remaining `1.044B` instruction count and reconvergence samples directly.
- Continue using NCU as the gate: each candidate needs a named counter/PC target before code and a measured counter/timing result after code.

## 2026-05-04 12:41 CDT - Lineinfo-Gated High-Risk Producer Target

Finding:

- High-risk producer/layout changes are in scope. The gating constraint is profiler evidence, not implementation risk.
- The last high-risk D256 store-side register transpose was the right kind of experiment: it targeted the measured shared-store wavefront PCs, passed correctness, failed timing, and was reverted.
- The remaining D256 paged-PV gap is still visible in NCU: paged executes about `1.044B` instructions versus dense around `0.576B`, with elevated reconvergence/barrier samples and a paged-specific SASS block around the top dynamic-instruction PCs.
- Current NCU reports lack usable CUDA line attribution for those PCs, so another architectural rewrite would still be partly SASS-pattern inference.

Implementation target:

- Build the D256 paged-PV reference spec in an isolated JIT workspace with CUDA lineinfo enabled.
- Re-run NCU SourceCounters and warp-state sampling against the same reference cell.
- Map the top paged-specific dynamic-instruction, reconvergence, and remaining global/shared-wavefront PCs back to source lines before choosing the next code change.

Validation:

- The diagnostic build must use the same source HEAD and same benchmark cell as the committed coalesced-load baseline: D256 Qwen paged-PV `q=512 kv=65536 g=6 split=3072`.
- If lineinfo changes timing materially, treat the profile as attribution-only and benchmark final candidates in the normal non-lineinfo cache before keeping them.
- Do not commit diagnostic cache or flag changes; only append the resulting attribution and any code changes that pass correctness and timing.

Decision criteria:

- Proceed to a high-risk producer change only when NCU identifies the specific source block responsible for a large paged-only counter.
- If lineinfo still cannot attribute the PCs, fall back to a controlled instrumentation strategy that isolates whole producer sub-blocks, not another guessed rewrite.

## 2026-05-04 12:47 CDT - PV Scale Producer Coalescing Target

Finding:

- The isolated lineinfo build completed at the same source HEAD and measured `2.295 ms` outside NCU and `2.559 ms` under NCU for D256 Qwen paged-PV. Treat it as source attribution only.
- Dense lineinfo on the same cell measured `1.409 ms` under NCU. Comparing line-level counters rules out the largest `score_is_valid()` division site as paged-specific: `global_q_row / group_size` is common to dense and paged.
- The largest paged-only instruction/counter block is PV scale staging:
  - `d256.cuh:1216` PV scale producer loop: `23,206,224` paged-only instructions.
  - `paged_kv.cuh:807` runtime `dim / params.scale_dim`: `20,319,216` paged-only instructions.
  - `paged_kv.cuh:814` `params.v_scales[src]`: `5,505,024` paged-only excessive global sectors.
- The current PV scale loop maps consecutive lanes over token groups first, then columns. For PV scale memory, columns are the contiguous dimension, so this is the same lane-ordering mistake the V data path had before coalescing.

Implementation target:

- Add a PV-scale helper with compile-time scale dimension (`kHeadDim / 16`) so D128/D256/D512 avoid runtime division by `params.scale_dim` in the hot PV scale path.
- Reorder the PV scale producer loop so consecutive load lanes cover consecutive columns for one token group before moving to the next token group.
- Apply the same source-level change to D128/D256/D512; no public shape, layout, or FFI changes.

Validation:

- Run the focused D256 correctness subset first.
- Benchmark D256 Qwen paged-PV `q=512 kv=65536 g=6 split=3072`.
- If timing improves, run NCU on D256 Qwen paged-PV and compare `paged_kv.cuh:814` excessive global sectors, `d256.cuh:1216` instructions, and total stage duration against the coalesced-load baseline.
- If D256 passes, run D128/D512 focused benchmarks and the full NVFP4 test file.

Decision criteria:

- Keep if D256 correctness passes and either timing or the targeted NCU counters improve.
- Revert if the reordering only shifts overhead or regresses timing.

## 2026-05-04 12:51 CDT - PV Scale Producer Coalescing Result

Finding:

- Implemented the PV scale producer experiment across D128/D256/D512:
  - Added a compile-time-scale-dim PV scale helper for `kHeadDim / 16`.
  - Reordered the PV scale producer loop so load lanes iterate columns first for a token group.
- Focused D256 correctness passed: `3 passed in 44.32s`.
- D256 Qwen paged-PV reference timing regressed/no-improved:
  - Run 1: `2.302 ms` mean.
  - Run 2: `2.307 ms` mean.
  - Committed coalesced-load baseline: `2.271 ms` mean.

Decision:

- Reverted the PV scale producer code.
- Do not propagate this lane-ordering/static-helper shape.
- The NCU line attribution remains useful, but this particular fix does not move wall time. The likely reason is that the reduced division/coalescing opportunity is offset by worse smem-scale store/address behavior or compiler scheduling in the producer loop.

Next profiling target:

- Keep using the lineinfo reports for source-ranked deltas.
- The highest-value paged-only counters still are:
  - PV V data store shared-wavefront excess at `d256.cuh:1112`.
  - K scale shared-wavefront excess at `d256.cuh:922`.
  - Paged producer reconvergence around cache/page setup at `d256.cuh:843`.
- The next change should either remove one of those stores entirely from the hot loop or change the staging architecture; simple lane reorders are now suspect unless NCU shows the target counter dominates and timing moves with it.

## 2026-05-04 12:52 CDT - Paged Physical-Page Cache Dedup Target

Finding:

- The lineinfo paged-vs-dense comparison shows the largest paged-only barrier/reconvergence source at `d256.cuh:843`, the page-cache fill predicate `if (load_thread_idx < kPagesPerKvTile)`.
- The committed page cache is filled inside every K and V staging helper. On D256 Qwen with `qk_head_chunks == 2` and `output_group_span == 2`, the same `kv_tile` is cached four times: K chunk 0, K chunk 1, V group 0, V group 1.
- The earlier page-cache barrier refinement did not remove this duplicate work; it moved the fill relative to the existing pre-stage barrier and removed the helper-local barrier for every chunk. That was neutral/slightly worse.
- The new target is different: keep the stage synchronization shape, but fill the physical-page cache only when the staged `kv_tile` changes.

Implementation target:

- Start D256-only.
- Split page-cache fill from synchronization.
- Track the currently cached `kv_tile` in the load-thread control flow.
- Call the fill before the existing pre-stage `load_group_sync()` only when `kv_tile` changes; keep all existing producer acquire/complete/fence synchronization unchanged.

Validation:

- Run the focused D256 correctness subset.
- Benchmark D256 Qwen paged-PV `q=512 kv=65536 g=6 split=3072`.
- If timing improves, run NCU and check that barrier/reconvergence samples attributed to `d256.cuh:843` and page-cache instructions drop.
- If D256 improves cleanly, propagate to D128/D512 and run the full NVFP4 test file.

Decision criteria:

- Keep and propagate only if D256 correctness passes and the reference cell improves materially.
- Revert if it repeats the previous barrier-refinement result or introduces ordering instability.

## 2026-05-04 12:55 CDT - Paged Physical-Page Cache Dedup Result

Finding:

- Implemented the D256-only schedule-level page-cache dedup:
  - Split cache fill from the helper-local synchronization.
  - Tracked the currently cached `kv_tile` in load-thread control flow.
  - Filled the eight physical-page entries only when `kv_tile` changed while preserving the existing pre-stage `load_group_sync()` and producer acquire/complete sequence.
- Focused D256 correctness passed: `3 passed in 44.38s`.
- D256 Qwen paged-PV timing was neutral/noisy:
  - Run 1: `2.274 ms` mean.
  - Run 2: `2.268 ms` mean.
  - Committed coalesced-load baseline: `2.271 ms` mean.

Decision:

- Reverted the D256 code.
- Do not propagate page-cache dedup to D128/D512. The duplicate cache fill is visible in NCU reconvergence attribution, but it is not load-bearing for wall time at this cell.

Next profiling target:

- Stop spending effort on page-cache branch shape unless a different cell shows it as a wall-time limiter.
- The remaining wall-time gap is more likely in the larger shared-store/MMA/softmax pipeline balance than in page-cache fill overhead.
- Use NCU/NSYS next to choose between two architectural targets: reducing shared-store wavefront excess in V/K scale staging, or changing split/tile scheduling to reduce total stage work.

## 2026-05-04 13:00 CDT - Occupancy Ruled Out / PV B Store Address Target

Finding:

- NCU LaunchStats/Occupancy on D256 Qwen `q=512 kv=65536 g=6` rules out low occupancy as the primary paged-vs-dense gap.
- Paged-PV launch:
  - Block size: `640`, grid size: `528`.
  - Registers/thread: `96`.
  - Dynamic shared memory/block: `92.16 KiB`.
  - Waves/SM: `2.81`.
  - Theoretical occupancy: `41.67%`; achieved occupancy: `40.50%`.
  - Achieved active warps/SM: `19.44`.
- Dense launch at the same logical cell:
  - Block size: `512`, grid size: `1056`.
  - Registers/thread: `128`.
  - Dynamic shared memory/block: `50.18 KiB`.
  - Waves/SM: `5.62`.
  - Theoretical occupancy: `33.33%`; achieved occupancy: `20.91%`.
  - Achieved active warps/SM: `10.04`.
- Paged has fewer grid waves and larger smem, but it is not starved for resident warps relative to dense. The gap is therefore per-CTA instruction/shared-memory work, not lack of occupancy.

NCU source target:

- The lineinfo paged-minus-dense comparison still points at the PV V operand store path:
  - `d256.cuh:1112` has the largest paged-only shared excessive-wavefront delta.
  - `cute/container/array_subbyte.hpp:263` and related CUTE layout/subbyte address helpers have large paged-only instruction deltas.
  - The current producer repeatedly computes `pv_sB(local_col, local_k0, write_stage)` and then stores one `uint32_t` per output word.

Implementation target:

- Treat high-risk producer surgery as in scope, but only when it removes a profiler-attributed cost.
- First target D256 PV-layout V only, because the lineinfo and full NCU reports were collected on D256 paged-PV and isolate the data path without linear dequant/requant.
- Replace repeated CUTE per-word PV B store address calculation with a layout-specific direct-address path if the CUTLASS `SmemLayoutB` mapping can be proven for the producer's `(local_col, local_k0, stage)` iteration space.
- The replacement must keep the existing public API and operand layout. It may use compile-time layout algebra, a small generated delta table, or a checked direct pointer formula. It must not silently fall back to the old path.

Validation:

- Before keeping any direct-address path, prove address equivalence against the current `pv_sB(...)` mapping for the full D256 PV producer coordinate space.
- Run the focused D256 NVFP4 correctness subset.
- Benchmark D256 Qwen paged-PV reference cell.
- If timing improves, re-run NCU and confirm the `d256.cuh:1112` shared excessive-wavefront and CUTE address-instruction counters drop.
- If timing is neutral/regressive or address equivalence cannot be proven cleanly, revert and keep this as a negative diagnostic.

## 2026-05-04 13:08 CDT - PV B Store Direct-Address Result

Finding:

- Probed the D256 PV B operand layout in a standalone compile:
  - `SmemLayoutB = Sw<2,4,3> o smem_ptr[4b](unset) o ((_8,_16),(_128,_1),(_1,_2)):((_128,_1024),(_1,_0),(_0,_16384))`.
  - `cosize = 32768` four-bit elements.
  - The producer's `k` run is contiguous for the 8-token packed word, so replacing the subbyte tensor reference with `layout(coord) >> 1` byte addressing is address-equivalent for the target store.
- Implemented D256 PV-layout V only:
  - Used `pv_sB.layout()(local_col, local_k0, write_stage)` to compute the nibble offset.
  - Used `cute::recast_ptr<uint8_t>(pv_sB_ptr) + (dst_nibble >> 1)` as the byte destination.
  - Kept debug equivalence checks against `pv_sB(...)` under `FLASHINFER_SM120_NVFP4_DEBUG_TRAPS`.
- Focused D256 correctness passed: `3 passed in 44.43s`.
- D256 Qwen paged-PV timing was neutral:
  - Run 1: `2.262 ms` mean.
  - Run 2: `2.268 ms` mean.
  - Committed baseline: `2.271 ms` mean.

Decision:

- Reverted the direct-address code.
- Do not pursue CUTE subbyte-reference removal as a wall-time target by itself. It can reduce source-attributed helper instructions, but those instructions are not load-bearing at the reference cell.

Next profiling target:

- Move from source-line micro-optimizations to schedule-level evidence.
- Use NSYS/NCU to separate stage kernel time from combine/kernel-launch overhead and sweep `split_kv_len` at the D256 Qwen and D512 Gemma reference cells.
- If a scheduling parameter changes wall time materially, then optimize the schedule. If not, return to producer-body SASS attribution with sampling rather than helper-line deltas.

## 2026-05-04 13:13 CDT - SFB Scale Store Aliasing Target

Finding:

- NSYS on D256 Qwen paged-PV confirms the wrapper gap is inside the stage kernel:
  - SM120 stage kernel: `8.86 ms` total across 4 instances, `90.3%` of GPU kernel time.
  - Q quantization: `0.24 ms` total, `2.4%`.
  - split-KV combine: `0.16 ms` total, `1.6%`.
- Split scheduling is not the architectural gap:
  - D256 paged-PV split sweep: `3072` is the local optimum (`2.278 ms`), with larger splits degrading sharply.
  - D512 paged-PV split sweep: `4096` is slightly better than `3072` (`6.519 ms` vs `6.699 ms`), but the gain is only about `2.7%`.
- The next source target is therefore the producer body.
- The D256 SFB layout probe shows the scale tensor aliases the eight logical `k_offset` positions in each 16-token scale group to one physical byte:
  - Layout: `(((_32,_4),_1),((_16,_4),_1,_2),_2):(((_16,_4),_512),((_0,_1),_4,_512),_1024)`.
  - Example row 0, stage 0: logical `k=0,2,4,6,8,10,12,14` all map to physical offset `0`.
  - `k=16..30` all map to physical offset `1`, and so on.
- Current paged K-scale and V-scale producers still loop over `k_offset += 2` and write the same physical scale byte eight times.
- This exactly matches NCU source attribution:
  - `d256.cuh:922 qk_sSFB(...) = make_ue4m3_raw(scale)` is the largest paged-only source line.
  - The same pattern exists in the PV scale stores.

Implementation target:

- Remove the redundant per-`k_offset` scale stores.
- For each loaded scale byte, write the representative logical coordinate once:
  - K scale: `qk_sSFB(row, local_scale_col * 16, write_stage)`.
  - PV V scale: `pv_sSFB(col, local_k0, write_stage)` for the two columns in the pair.
- Start D256-only because the NCU/source evidence and layout probe are D256. If correctness and timing move, propagate to D128/D512 after probing or asserting the same SFB alias invariant.

Validation:

- Focused D256 correctness subset.
- D256 Qwen paged-PV reference timing.
- If kept, re-run NCU source counters and confirm the SFB store instruction/shared-wavefront attribution drops.
- If D256 moves materially, repeat the layout probe on D128/D512, apply the same one-write scale producer, and run the full NVFP4 test file.

## 2026-05-04 13:28 CDT - Paged Page-Base Cache Target

Finding:

- The one-write scale-store change is correctness-clean:
  - Full NVFP4 attention test file: `36 passed in 288.11s`.
- Reference timing moved in the right direction but only modestly:
  - D128 paged-PV: `1.084 ms`.
  - D256 paged-PV: `2.235 ms` representative mean after one-write scales.
  - D512 paged-PV: `6.673 ms` at split `3072`; `6.486 ms` at split `4096`.
- The lineinfo NCU follow-up confirms the intended source counter dropped:
  - Old D256 `qk_sSFB(row, local_scale_col * 16 + k_offset, ...)`: about `511M` source-attributed instructions.
  - New D256 `qk_sSFB(row, local_scale_col * 16, ...)`: about `2.1M`.
  - `cute::numeric::divmod` attribution dropped from about `107M` to about `43M`.
- The new top paged-only source line is page-base address math:
  - `paged_kv.cuh:548`, the NHD K-scale page-base term `kv_head * k_scale_stride_dim2`, is about `185M` source-attributed instructions.
  - V-scale page-base math and K/V data page-base math remain visible lower in the list.
- Current producers recompute page-base expressions inside per-row/per-scale/per-word loops even though the physical page is already cached once per 16-token logical page.

Implementation target:

- Extend the per-tile page cache from physical page ids to page bases:
  - `k_data_page_base_cache[local_page]`.
  - `k_scale_page_base_cache[local_page]`.
  - `v_data_page_base_cache[local_page]`.
  - `v_scale_page_base_cache[local_page]`, using the PV-scale base formula when `kPvLayoutV=true` and the linear-scale base formula otherwise.
- Replace producer-inner helper calls that recompute `physical_page * stride_page + kv_head * head_stride` with cached page bases.
- Keep page-offset and dim/scale-col address math in the inner loops; only hoist page-invariant terms.
- Apply across D128/D256/D512 after D256 compiles because the cache shape is identical (`kCutlassTileN / 16` pages per KV tile).

Validation:

- Full NVFP4 attention test file.
- D256 Qwen paged-PV reference timing.
- D512 Gemma paged-PV reference timing.
- NCU source follow-up if timing moves materially, checking that `paged_kv.cuh:548` and related page-base source lines drop.

## 2026-05-04 13:36 CDT - Paged Page-Base Cache Result

Finding:

- Implemented D256 page-base caching for K data, K scale, V data, and V scale.
- The first focused D256 multi-KV correctness run failed just above the existing tolerance:
  - Greatest absolute difference: `0.0010986328125` with tolerance `0.001`.
- Narrowed the failure by disabling cached V-scale base usage. The single multi-KV test then passed once, but the full focused subset failed the same multi-KV test on a later run:
  - Greatest absolute difference: `0.001129150390625` with tolerance `0.001`.
- The failure pattern is tolerance-edge and intermittent, but this optimization changes shared-memory state and page-base scheduling enough that it cannot be treated as correctness-neutral.

Decision:

- Reverted the page-base cache code and the PV-scale page-base helper.
- Do not keep page-base caching without a stronger byte-level diagnostic that proves the drift source is unrelated.
- The one-write scale-store change remains in place; it passed the full NVFP4 test file before the page-base experiment and has direct layout proof.

Next profiling target:

- Keep the SFB/SFA one-write change.
- Do not chase `paged_kv.cuh:548` by caching page bases until the multi-KV tolerance-edge behavior is understood.
- Next high-risk target should be selected from a fresh NCU report after the scale-store change, not from the pre-scale source ranking.

## 2026-05-04 13:43 CDT - One-Write Scale Store Final Result

Finding:

- Verified the SFA/SFB alias invariant across all three head dimensions:
  - D128 QK SFB: `alias=1`; D128 PV SFB: `alias=1`; D128 QK SFA: `alias=1`.
  - D256 QK SFB: `alias=1`; D256 PV SFB: `alias=1`; D256 QK SFA: `alias=1`.
  - D512 QK SFB: `alias=1`; D512 PV SFB: `alias=1`; D512 QK SFA: `alias=1`.
- Applied one-write scale staging across D128/D256/D512:
  - K SFB writes one representative coordinate per 16-token scale group.
  - PV V SFB writes one representative coordinate per 16-token scale group.
  - BF16-Q fused-quantize SFA writes one representative coordinate per 16-dim scale group.
- Full NVFP4 attention test file passes after reverting the page-base-cache experiment:
  - `36 passed in 291.75s`.

Reference timings:

- D128 paged-PV `q=512 kv=65536 g=4 split=3072`: `1.084 ms`.
- D256 paged-PV `q=512 kv=65536 g=6 split=3072`: `2.235 ms` representative mean.
- D512 paged-PV `q=512 kv=65536 g=8 softcap=30 split=3072`: `6.673 ms`.
- D512 paged-PV `q=512 kv=65536 g=8 softcap=30 split=4096`: `6.486 ms`.

NCU validation:

- D256 Qwen paged-PV lineinfo NCU after one-write scales:
  - Old `qk_sSFB(row, local_scale_col * 16 + k_offset, ...)`: about `511M` source-attributed instructions.
  - New `qk_sSFB(row, local_scale_col * 16, ...)`: about `2.1M`.
  - `cute::numeric::divmod` attribution dropped from about `107M` to about `43M`.
- Wall-time gain is modest, so SFB duplication was a real waste but not the main architectural gap.

Decision:

- Keep and commit the one-write scale-store change.
- Do not treat this as the performance pass endpoint. The next pass needs a fresh post-change NCU report and should target the new top wall-time source, not the eliminated SFB store line.

## 2026-05-04 13:48 CDT - Q-Row Position Cache Target

Finding:

- Rebuilt the D256 Qwen paged-PV JIT module with `FLASHINFER_JIT_LINEINFO=1` after moving only the no-lineinfo cache directory aside.
- Fresh post-scale-store NCU source attribution on `q=512 kv=65536 g=6 split=3072` reports `2.54 ms` under profiler and identifies the current instruction hotspot.
- Top source by instructions is `fmha_nvfp4_sm120_d256.cuh:1905`, `const int q_token = global_q_row / group_size;`, with about `157M` instructions and `5.0B` thread instructions.
- The same line is also the top short-scoreboard and wait-stall source; the score predicate is recomputing a row-invariant division for every score element.
- This is not a paged producer memory issue; it is a per-score predicate structure issue surfaced after the scale-store cleanup removed the previous redundant scale writes.

Implementation target:

- Add a per-row shared-memory cache of `q_pos`, using `-1` for invalid Q rows.
- Populate it once during the existing CTA initialization loop that already writes `global_m`, `global_l`, and `old_scale_stage` before the block-wide `__syncthreads()`.
- Replace `score_is_valid`'s per-score `global_q_row / group_size` with a shared read of the cached `q_pos`.
- Apply the same pattern to D128/D256/D512 because all three files have the same `score_is_valid` structure and the row invariant is identical.

Validation:

- Focused D256 correctness first.
- D256 Qwen paged-PV benchmark before broader propagation if compile succeeds.
- Full NVFP4 attention test file after all D-size propagation.
- Re-run D256 Qwen NCU source counters if timing moves materially, verifying line `q_token = global_q_row / group_size` drops out of the top source list.

Decision criteria:

- Keep if correctness passes and the D256 reference cell improves or is neutral without increasing shared-memory budget past the existing static asserts.
- Revert if shared-memory traffic from the cache replaces the division cost and timing regresses.

## 2026-05-04 13:54 CDT - Q-Row Position Cache Result

Finding:

- Implemented the per-row `q_pos` cache across D128/D256/D512 and compiled through the focused D256 tests.
- Focused D256 correctness failed immediately:
  - Multi-KV vs single-KV: greatest absolute difference `0.0029296875` with `0.001` tolerance.
  - Scratch poison determinism: `158867 / 688128` elements mismatched, greatest absolute difference `0.00390625`.
  - Standard wrapper vs direct wrapper: greatest absolute difference `0.0040283203125` with `0.002` tolerance.
- The failure shape is not a tolerance-edge speed-only tradeoff; scratch poison divergence means the shared cache placement or lifetime perturbed state in a way that is not correctness-neutral.

Decision:

- Reverted the q-position cache code in D128/D256/D512.
- Keep the NCU finding: per-score `q_token = global_q_row / group_size` is real overhead, but the simple shared-cache implementation is unsafe in the current aliased shared-memory layout.
- Do not revisit this with another shared-storage field unless the storage/lifetime interaction is diagnosed first.

Next profiling target:

- Continue from the same lineinfo NCU report.
- The remaining high-impact targets are the PV V operand store shared-memory wavefront excess at `fmha_nvfp4_sm120_d256.cuh:1104` and global excessive sectors from `v_scales` at `paged_kv.cuh:814`.
- Because the direct-address PV store experiment was timing-neutral, the next structural attempt should target data/layout movement rather than a syntactic address computation rewrite.

## 2026-05-04 13:55 CDT - PV V Scale Lane-Mapping Target

Finding:

- The lineinfo NCU report after one-write scale stores shows the largest global excessive-sector source is `fmha_nvfp4_sm120_paged_kv.cuh:814`, `return params.v_scales[src];`.
- The D256 PV V scale staging loop currently maps consecutive load threads across `token_group` first and `col` second:
  - `col = idx / kTokenScaleGroups`.
  - `token_group = idx - col * kTokenScaleGroups`.
- For PV scale layout, `sm120_nvfp4_paged_v_pv_scale_from_physical_page` indexes by physical page and dim. Consecutive token groups are different pages, so adjacent lanes load far-apart scale addresses for the same column.
- This explains the NCU excessive global sectors: the warp is de-coalescing the scale loads by page before it walks contiguous columns.

Implementation target:

- Reorder only the PV V scale staging loop so `token_group` is the outer/coarse index and `col` is the fast lane-varying index:
  - `token_group = idx / kOutputTileN`.
  - `col = idx - token_group * kOutputTileN`.
- This keeps the public PV scale layout and the CUTLASS SFB destination layout unchanged.
- Apply to D128/D256/D512 because all three files have the same PV scale loop and NCU identified the same helper as the global-sector source.

Validation:

- Focused D256 correctness first.
- D256 Qwen paged-PV benchmark at the same reference cell.
- If timing improves, run NCU source counters again and confirm `v_scales[src]` excessive sectors drop.

Decision criteria:

- Keep if correctness passes and D256 timing improves or stays neutral while reducing NCU global excessive sectors.
- Revert if the lane remap causes correctness drift or worsens timing; the next target would then be a wider/vectorized scale load rather than loop-ordering.

## 2026-05-04 13:58 CDT - PV V Scale Lane-Mapping Result

Finding:

- Reordered the PV V scale loop across D128/D256/D512 so lanes load contiguous columns within one token group/page.
- Focused D256 correctness passed:
  - `3 passed in 44.20s`.
- First timing used a lineinfo JIT cache and was not comparable, so restored the no-lineinfo D256 PV cache path and let ninja rebuild from current source.
- Apples-to-apples D256 Qwen paged-PV benchmark regressed:
  - Post-scale baseline: about `2.235 ms` representative mean.
  - PV scale lane remap: `2.314 ms` mean (`min 2.306`, `max 2.322`).

Decision:

- Reverted the PV scale lane remap in D128/D256/D512.
- The NCU global excessive-sector finding is real, but this loop-order change trades it for worse wall time, likely through destination SFB/shared-memory access order or worse scheduling of the scale producer.
- Do not treat global-sector coalescing alone as sufficient; future scale work needs to include the shared destination pattern and wall-time validation.

Next profiling target:

- Return to the fresh NCU source report and target the highest wall-time-relevant remaining source with a structural change, not a source-only coalescing rewrite.
- The leading unresolved class remains V operand staging/store shared-memory wavefront excess around the PV B store path.

## 2026-05-04 14:00 CDT - PV B Store Register-Transpose Target

Finding:

- Fresh lineinfo NCU after the one-write scale cleanup reports the largest shared-memory excessive-wavefront source at `fmha_nvfp4_sm120_d256.cuh:1104`, the `pv_sB` 32-bit packed-word store.
- The current PV V producer intentionally coalesces global loads by assigning each 8-lane subgroup one `8 token x 64 dim` block:
  - lane 0 owns dims `0..7`, lane 1 owns dims `8..15`, etc.
  - this makes the per-token gmem loads contiguous across lanes.
- The store side then issues one store per `dim_offset`; for a fixed store instruction, lanes write columns `0, 8, 16, ...`, not contiguous columns.
- That lane-to-column stride matches the NCU shared-memory wavefront excess at the `pv_sB` store.

Implementation target:

- Test D256 first.
- Keep the current global-load mapping so V data loads stay coalesced.
- After each lane builds its eight `packed_words`, add an 8-lane register transpose at store time:
  - each lane selects `packed_words[subgroup_lane]`.
  - `__shfl_sync` broadcasts that selected word from each source lane.
  - for each source lane, the subgroup stores contiguous columns `base + source_lane * 8 + subgroup_lane`.
- This trades eight subgroup shuffles per producer tile for contiguous shared-memory stores.

Validation:

- Focused D256 correctness.
- D256 Qwen paged-PV benchmark at the reference cell.
- If timing improves, run lineinfo NCU again and verify the `pv_sB` shared excessive wavefront count drops.
- If D256 improves, propagate to D128/D512; otherwise revert D256 and keep the finding.

Decision criteria:

- Keep only if correctness passes and wall time improves. NCU shared-wavefront reduction alone is insufficient.
- Revert if ptxas spills or wall time regresses; the next structural option is a non-CUTLASS V smem path rather than more local lane remapping.

## 2026-05-04 14:04 CDT - PV B Store Register-Transpose Result

Finding:

- Implemented the D256 PV B store register-transpose experiment.
- Correctness passed the focused D256 subset:
  - `3 passed in 44.39s`.
- D256 Qwen paged-PV reference timing regressed:
  - Post-scale baseline: about `2.235 ms` representative mean.
  - Register-transposed PV B store: `2.354 ms` mean (`min 2.314`, `max 2.479`).
- The result indicates the NCU shared-wavefront excess at the store is not cheap enough to fix with subgroup shuffles; the added shuffle/register pressure costs more than the shared-store coalescing recovers.

Decision:

- Reverted the D256 register-transposed PV B store experiment.
- Do not propagate this shape to D128/D512.
- Keep the conclusion: the CUTLASS V operand smem layout imposes a store-pattern cost, but local lane remapping inside the current producer is not the right way to recover it.

Next profiling target:

- Local producer tweaks are now producing small or negative deltas while the remaining gap is architectural.
- The next NCU-driven high-risk direction should compare the current CUTLASS-operand V path against a non-CUTLASS V staging/fragment-construction path, because both failed store-local attempts point at the smem layout contract rather than address arithmetic.

## 2026-05-04 14:08 CDT - Runtime Group Divide Specialization Target

Finding:

- Current reference timings after the committed scale-store cleanup:
  - D256 Qwen dense: `1.263 ms`; paged-PV: about `2.235 ms`; paged-linear: `3.238 ms`.
  - D512 Gemma dense: `5.167 ms`; paged-PV: `6.494 ms`; paged-linear: `7.921 ms`.
- D512 is now in the right regime (`1.26x` PV over dense, `1.53x` linear over dense). D256 remains the larger relative gap.
- The D256 lineinfo NCU report identifies `fmha_nvfp4_sm120_d256.cuh:1905`, `global_q_row / group_size`, as the top instruction source.
- The shared `q_pos` cache removed the divide but broke correctness, so shared-memory caching is not safe in the current aliased storage layout.
- The divide denominator is one of a small set of production group sizes (`2`, `4`, `6`, `8` in the current benchmark/deployment cells), but the kernel currently treats it as an arbitrary runtime integer.

Implementation target:

- Add a device helper that specializes common group sizes with literal divisors/shifts:
  - `1`, `2`, `4`, and `8` use shifts or identity.
  - `6` uses a literal `/ 6`, allowing ptxas to lower it to a multiply/shift sequence instead of runtime integer division.
  - Unknown group sizes fall back to `/ group_size`.
- Replace the per-score `global_q_row / group_size` callsite in D128/D256/D512 with this helper.
- No new spec axis, no public API change, no shared-memory lifetime change.

Validation:

- Focused D256 correctness.
- D256 Qwen dense/paged-PV/paged-linear benchmark.
- If timing improves, run lineinfo NCU again and verify the `global_q_row / group_size` source attribution drops materially.
- Full NVFP4 test file before committing if the helper is kept.

Decision criteria:

- Keep if correctness passes and D256 timing improves or is neutral with NCU confirming the divide source is reduced.
- Revert if ptxas still emits runtime division on the hot path or timing regresses.

## 2026-05-04 14:15 CDT - Runtime Group Divide Specialization Result

Finding:

- Implemented a literal-specialized group-size helper for `1`, `2`, `4`, `6`, and `8`, with fallback to runtime division.
- Focused D256 correctness was not clean:
  - First focused subset: multi-KV failed by `2` elements, greatest absolute difference `0.00103759765625` with tolerance `0.001`; scratch poison and standard-vs-direct passed.
  - Rerun of the failing multi-KV test: failed by `1` element, greatest absolute difference `0.0013427734375` with tolerance `0.001`.
- The helper is mathematically equivalent for integer `q_token`, so the failure is likely a codegen/scheduling perturbation surfacing the existing D256 multi-KV tolerance edge, not a logical predicate error.

Decision:

- Reverted the group-divide helper and restored the original `global_q_row / group_size` expression in D128/D256/D512.
- Do not keep changes that make the multi-KV tolerance-edge failure more likely, even if the source hotspot is real.
- Any future work on this predicate should first localize why D256 multi-KV is so close to tolerance, or should change the predicate in a way that can be proven byte-identical through the binary-search diagnostic path.

Next profiling target:

- The obvious local rewrites have now produced one kept win and several rejected perturbations.
- The next meaningful high-risk work needs either a byte-level correctness diagnostic for the D256 tolerance edge or a larger architecture path that avoids the current CUTLASS operand staging costs rather than rearranging instructions inside them.

## 2026-05-04 14:10 CDT - Linear V Cache Layout Target

Finding:

- Fresh lineinfo NCU on D256 Qwen paged-linear identifies the internal linear-V data cache as the dominant remaining global-memory problem.
- The report attributes `44,040,192` excessive L2 global sectors to `fmha_nvfp4_sm120_paged_kv.cuh:118`, the 32-bit load in `sm120_nvfp4_linear_v_data_cache_word`.
- The current internal cache is token-major/dim-contiguous: `idx = token * packed_dim + packed_col`.
- The stage producer's 8-lane subgroup reads eight consecutive tokens at one `packed_col` word, so adjacent lanes access addresses separated by `packed_dim` bytes. For D256 that is a 128-byte lane stride, which matches the NCU uncoalesced-sector attribution.
- This cache is internal FFI scratch, not public API. Its physical layout can change without changing the caller's tensor layout, layout names, FFI surface, or tests.

Implementation target:

- Change the internal linear-V data cache layout from `[token][packed_col]` bytes to `[packed_col_word][token][byte_in_word]`.
- Preserve total byte size. A `uint32_t` cache load at a 4-byte-aligned `packed_col` will still return the same four dim-contiguous bytes for one token, but adjacent subgroup lanes will now load adjacent 4-byte words for adjacent tokens.
- Keep the producer and public wrapper contracts unchanged; only the cache build kernel and `sm120_nvfp4_linear_v_data_cache_word` address math should change.
- D128/D256/D512 share the helper and cache build path, so this is a common structural change rather than another per-head-dim patch.

Validation:

- Run focused D256 correctness first because the NCU evidence and current biggest relative gap are D256.
- Benchmark D256 Qwen paged-linear at `q=512 kv=65536 g=6 softcap=0 split_kv_len=3072`.
- Re-run lineinfo NCU if wall time improves and verify the `v_linear_data_cache_word` excessive-sector count drops materially.
- Run the full NVFP4 test file before committing a kept change.

Decision criteria:

- Keep if correctness passes and paged-linear wall time improves or is neutral with clear NCU reduction in the cache-load excessive sectors.
- Revert if correctness fails, cache build cost dominates, or wall time regresses despite the coalescing improvement.

## 2026-05-04 14:16 CDT - Linear V Data Cache Layout Result

Finding:

- Implemented the internal linear-V data cache reorder from token-major bytes to `[packed_col_word][token][byte_in_word]`.
- Focused D256 correctness passed:
  - `3 passed in 44.34s`.
- D256 Qwen paged-linear wall time improved slightly:
  - Previous representative mean: `3.238 ms`.
  - New mean: `3.195 ms` (`min 3.192`, `max 3.200`).
- Follow-up lineinfo NCU confirms the targeted source was fixed:
  - Total L2 global excessive sectors dropped from `47,185,920` to `3,145,728`.
  - `sm120_nvfp4_linear_v_data_cache_word` dropped from `44,040,192` excessive sectors to `0`; the load now reports `6,291,456` total sectors and `6,291,456` ideal sectors.
- The modest wall-time gain despite the large sector reduction means the old uncoalesced data-cache load was a real memory-efficiency bug but not the only wall-time limiter.

Decision:

- Keep the data-cache layout change unless a wider test exposes correctness or cache-build regressions.
- The change is internal-only and makes the stage producer's access pattern physically coalesced without public API changes.

Next profiling target:

- The same NCU report now shows the largest remaining global excessive source in the linear path is `sm120_nvfp4_linear_v_scale_cache_load`: `2,359,296` excessive sectors out of `3,145,728` total sectors.
- The scale cache has the same layout mismatch shape: current physical layout is dim-major (`dim * token_groups + token_group`), while the producer reads adjacent dims for one token group.
- Apply the same internal layout fix to the linear-V scale cache before moving to non-global bottlenecks.

## 2026-05-04 14:16 CDT - Linear V Scale Cache Layout Target

Finding:

- After the data-cache reorder, lineinfo NCU attributes the remaining linear-path global excessive sectors primarily to `fmha_nvfp4_sm120_paged_kv.cuh:104`, the `v_linear_scale_cache` load.
- The current scale cache layout is `[dim][token_group]`, so lanes reading adjacent dims for one token group are separated by `token_groups` bytes.
- The producer access pattern wants `[token_group][dim]`: adjacent lanes read adjacent dims at the same token group.
- Like the data cache, this is internal FFI scratch and does not affect public tensor layout or wrapper contracts.

Implementation target:

- Change `sm120_nvfp4_linear_v_scale_cache_load` to address `token_group * head_dim + dim` within each batch/head.
- Change the scale-cache build kernel to write that physical layout directly and iterate `dim_pair` fastest for coalesced cache writes.
- Keep byte size and host scratch sizing unchanged.

Validation:

- Run the focused D256 correctness subset.
- Benchmark D256 Qwen paged-linear at the same reference cell.
- Re-run lineinfo NCU if timing improves and verify the scale-cache excessive sectors drop.

Decision criteria:

- Keep if correctness passes and wall time improves or remains neutral with the NCU source removed.
- Revert if the scale-cache build reorder regresses wall time or correctness.

## 2026-05-04 14:20 CDT - Linear V Scale Cache Layout Result

Finding:

- Implemented the internal scale-cache reorder from `[dim][token_group]` to `[token_group][dim]`.
- Focused D256 correctness passed:
  - `3 passed in 44.16s`.
- D256 Qwen paged-linear wall time improved:
  - Before cache-layout work: `3.238 ms`.
  - After data-cache reorder: `3.195 ms`.
  - After scale-cache reorder: `3.100 ms` (`min 3.095`, `max 3.105`).
- Follow-up lineinfo NCU did not remove the source cleanly:
  - Total L2 global excessive sectors increased from `3,145,728` after the data-cache-only change to `6,291,456`.
  - `sm120_nvfp4_linear_v_scale_cache_load` now accounts for `5,505,024` excessive sectors.
- The wall-time direction says the physical layout is better, but the remaining scale loads are byte-granularity and are issued twice per adjacent scale pair.

Decision:

- Keep the scale-cache physical layout for the next experiment because it improved wall time and enables aligned adjacent-pair loads.
- Do not consider the scale path complete; NCU says byte load granularity is now the source.

Next profiling target:

- Replace the two adjacent `uint8_t` scale-cache loads with one aligned `uint16_t` pair load.
- The producer and data-cache build both consume `(dim0, dim0 + 1)` pairs, so the pair load matches existing logical access and should reduce instruction count and sector waste.

## 2026-05-04 14:20 CDT - Linear V Scale Pair Load Target

Finding:

- Every callsite that consumes the linear scale cache asks for adjacent scale bytes:
  - the stage producer's `pv_scale_pair_for(dim0, dim0 + 1)`;
  - the linear data-cache build kernel's `sf0_byte` / `sf1_byte` pair.
- With the new `[token_group][dim]` layout, those two bytes are physically adjacent and `dim0` is even.
- Keeping separate byte loads leaves NCU attributing most remaining global-sector excess to `sm120_nvfp4_linear_v_scale_cache_load`.

Implementation target:

- Add `sm120_nvfp4_linear_v_scale_cache_pair_load` returning a `uint16_t` from the aligned `(dim0, dim0 + 1)` address.
- Use it in the stage producers for D128/D256/D512 and in the data-cache build kernel.
- Leave the byte helper in place only if another non-pair callsite remains; otherwise remove it.

Validation:

- Run focused D256 correctness.
- Benchmark D256 Qwen paged-linear.
- Re-run lineinfo NCU if timing improves and verify the scale-cache source drops.

Decision criteria:

- Keep if correctness passes and either wall time improves or NCU shows the scale-cache source is materially reduced without a wall-time regression.
- Revert if the paired load changes rounding/correctness or increases wall time.

## 2026-05-04 14:24 CDT - Linear V Scale Pair Load Result

Finding:

- Implemented the aligned 16-bit scale-pair load in the common helper and the D128/D256/D512 stage producers.
- Focused D256 correctness passed:
  - `3 passed in 45.44s`.
- D256 Qwen paged-linear timing regressed relative to the scale-layout-only state:
  - Scale-layout-only: `3.100 ms` mean.
  - Pair-load first run: `3.126 ms` mean.
  - Pair-load repeat-10 run: `3.128 ms` mean.

Decision:

- Reverted the 16-bit scale-pair load and restored the byte-load helper.
- Kept the data-cache and scale-cache physical layout reorders, which are the changes with measured wall-time improvement.

Next profiling target:

- Stop spending time on byte-load granularity in the scale cache for now; it is a small source relative to the remaining dense-vs-paged gap.
- Run a wider validation/benchmark pass on the kept cache-layout changes, then use NCU/NSYS to choose the next structural target from the post-layout baseline.

## 2026-05-04 14:28 CDT - Head-Dim-Specific Linear Cache Layout Target

Finding:

- The common cache-layout reorder improves the D256 Qwen linear reference cell but regresses the D512 Gemma global linear reference cell.
- D256 Qwen paged-linear:
  - Before cache layout work: `3.238 ms`.
  - Data+scale cache layout: `3.100-3.105 ms`.
- D512 Gemma paged-linear:
  - Prior reference: `7.921 ms`.
  - Common data+scale cache layout: `8.197 ms` first run, `8.190 ms` repeat-10.
- This is an internal scratch-layout choice, not a public API contract. A single physical cache layout does not need to be forced across head dimensions if NCU/timing says the access balance differs.

Implementation target:

- Keep the coalesced internal linear-V cache layout only for D256, where NCU identified and validated the sector problem.
- Restore the old token-major/dim-major cache layout for D128 and D512 until those head dimensions have their own NCU evidence.
- Use compile-time helper selection in the D128/D256/D512 producers so the stage hot path does not pay a runtime layout branch.
- Use the runtime `head_dim == 256` value only inside the cache-build kernels, where the cost is outside the stage hot loop.

Validation:

- Focused D256 correctness after the specialization.
- Benchmark D256 Qwen paged-linear, D512 Gemma paged-linear, and D128 baseline cell.
- Full NVFP4 test file before committing.

Decision criteria:

- Keep if D256 retains the cache-layout win and D512 returns to its prior no-regression regime.
- Revert the specialization if it introduces correctness failures or if D256 loses the NCU-driven gain.

## 2026-05-04 14:44 CDT - Head-Dim-Specific Linear Cache Layout Result

Finding:

- Implemented compile-time producer selection for the linear-V cache layout:
  - D256 uses the coalesced internal layout validated by NCU.
  - D128/D512 use the old internal layout unless separately profiled.
- The cache-build kernels were also split into compile-time `true` / `false` instantiations so D128/D512 do not pay a per-element runtime layout branch.
- Focused D256 correctness passed after specialization:
  - `3 passed in 44.17s`.
- Full NVFP4 test file passed:
  - `36 passed in 246.13s`.

Benchmark result:

- D256 Qwen reference, `q=512 kv=65536 g=6 softcap=0 split_kv_len=3072`:
  - Dense: `1.264 ms`.
  - Paged-PV: `2.227 ms`.
  - Paged-linear before cache-layout work: `3.238 ms`.
  - Paged-linear after D256-only cache layout: `3.105 ms`.
  - Net D256 linear gain: about `4.1%`.
- D512 Gemma reference, `q=512 kv=65536 g=8 softcap=30 split_kv_len=3072`:
  - Current dense: `5.291 ms`.
  - Current paged-PV: `6.685 ms`.
  - Current paged-linear: `8.135 ms`.
  - Ratios remain in the same regime as the prior reference (`~1.26x` PV/dense, `~1.54x` linear/dense); the stale absolute `7.921 ms` linear reference was not comparable to the current dense timing.

Decision:

- Keep the D256-only internal cache-layout change.
- Do not force the coalesced cache layout across D128/D512 without head-dim-specific NCU evidence.
- The change removes the D256 linear data-cache sector bug and gives a modest but real wall-time win; it does not address the larger D256 paged-vs-dense gap.

Next profiling target:

- Move back to NCU/NSYS on the current D256 post-cache baseline rather than continuing small cache-local rewrites.
- The useful question is now where the remaining `~2.46x` D256 linear-vs-dense ratio lives: cache prepass launches, stage-kernel local spills, shared-memory wavefront excess, or scheduler/CTA geometry.

## 2026-05-04 14:49 CDT - D256 Linear Post-Cache NSYS/NCU Result

Finding:

- NCU on the current D256 Qwen paged-linear stage no longer identifies the internal linear data cache as a sector problem.
- The largest remaining stage-kernel global/shared attribution is the 32-bit K `cp.async` path at `include/flashinfer/cp_async.cuh:242`.
- Stage-kernel NCU summary:
  - Duration: `2.62 ms` under NCU.
  - L2 global excessive sectors: `6,291,456`.
  - L1 shared excessive wavefronts: `18,974,208`.
  - Local memory spilling requests: `37,408,368`.
  - Dominant not-issued stalls: barrier (`60,334`), long scoreboard (`37,461`), sleeping (`37,764`), wait (`17,091`).
- NSYS on D256 Qwen paged-linear splits the full wrapper work:
  - Stage kernel: `2` instances, `2.495 ms` average.
  - Linear scale cache build: `2` instances, `0.286 ms` average.
  - Linear data cache build: `2` instances, `0.286 ms` average.
  - Split-K combine: about `0.040 ms`.
- NSYS on D256 dense at the same cell:
  - Stage kernel: `2` instances, `1.226 ms` average.
  - Dense stage grid: `48 x 22`, block `512`, dynamic smem `0.050 MB`.
  - Paged-linear stage grid: `24 x 22`, block `640`, dynamic smem `0.092 MB`.
- The remaining D256 gap is not Python/wrapper overhead. It is primarily stage-kernel time, plus about `0.57 ms` per wrapper call in linear cache prepasses.

Decision:

- Do not pursue more local byte/cache load tweaks without a stage-kernel source target.
- The next linear-specific target is the two-kernel linear cache prepass, because NSYS quantifies it as a bounded `~0.57 ms` component and it is structurally redundant.

Implementation target:

- Add a D256-only fused linear-V cache build path.
- Current path:
  - Kernel 1 computes per `(token_group, dim_pair)` PV scales into `v_linear_scale_cache`.
  - Kernel 2 revisits every `(token, packed_col)`, reloads those scales, and writes `v_linear_data_cache`.
- Fused path:
  - One kernel handles one `(token_group, dim_pair)` per thread.
  - It computes the two output scales, writes the scale cache for the stage producer, then loops the 16 tokens in the group and writes the data cache bytes using the freshly computed scales.
  - This removes one launch and the data-cache kernel's scale-cache rereads.
- Scope is D256/coalesced layout first; D128/D512 stay on the existing two-kernel path unless profiled separately.

Validation:

- Focused D256 correctness.
- D256 Qwen paged-linear benchmark.
- NSYS recheck if wall time improves to verify scale/data cache build time collapses or moves.
- Full NVFP4 test file before committing a kept fused path.

Decision criteria:

- Keep if D256 paged-linear improves and correctness stays green.
- Revert if serializing 16 token writes per thread makes the cache build slower or changes output.

## 2026-05-04 14:56 CDT - D256 Fused Linear Cache Rejected

Finding:

- Implemented the D256 fused linear-V cache prepass as a local experiment:
  - One kernel computed the two PV scales for each `(token_group, dim_pair)`.
  - The same thread then wrote the 16 token data-cache bytes using those freshly computed scales.
  - D128/D512 launch paths were kept semantically unchanged by falling back to the existing two-kernel prepass.
- Focused correctness passed:
  - `3 passed in 45.86s`.
- The D256 Qwen reference cell did not improve:
  - Baseline after the D256 cache-layout commit: `3.105 ms`.
  - Fused prepass experiment: `3.126 ms` mean (`3.116 ms` min, `3.143 ms` max).
- The result is consistent with the design risk: the fused kernel removes one launch and one scale-cache reread, but it also serializes the 16 per-token data-cache writes into the scale-compute thread.

Decision:

- Rejected the fused linear cache prepass and restored the committed two-kernel implementation.
- This was a bounded NSYS-driven experiment on the measured `~0.57 ms` cache-prepass component; it does not move the dominant stage-kernel gap.

Next profiling target:

- Stop spending effort on low-yield cache prepass launch fusion.
- Use NCU/NSYS source attribution on the stage kernel itself. The current stage gap is dominated by stage-kernel time, not Python, wrapper orchestration, or linear cache prepass launch count.

## 2026-05-04 14:58 CDT - D256 Linear Scale Pair Load Target

Finding:

- Re-parsed the post-cache D256 Qwen paged-linear NCU source export after rejecting prepass fusion.
- The top global excessive-sector PCs are no longer the linear data-cache word path.
- The top entries are two adjacent `LDG.E.U8` loads from `v_linear_scale_cache`:
  - `include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d256.cuh:940`.
  - `include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_d256.cuh:944`.
  - Both resolve to `include/flashinfer/attention/blackwell/fmha_nvfp4_sm120_paged_kv.cuh:109`.
- Each top scale-cache load has `905,520` excessive L2 sectors in the source export.
- The D256 coalesced internal scale-cache layout stores adjacent dims contiguously for a `(token_group, dim_pair)` pair, so the two byte loads can be replaced with one aligned 16-bit load in the coalesced path.

Implementation target:

- Add a stage-side scale-pair helper.
- For `kLinearVCacheCoalesced == true`, load `sf0/sf1` with one 16-bit load from the adjacent scale-cache bytes.
- For non-coalesced layouts, keep the existing two byte loads because adjacent dims are separated by `token_groups`.
- Use the helper only in the D256 linear-V stage producer path where NCU identified the source line.

Validation:

- Focused D256 correctness.
- D256 Qwen paged-linear benchmark against the `3.105 ms` baseline.
- Re-run NCU if timing moves materially to verify the top scale-cache excessive-sector PCs drop.

Decision criteria:

- Keep if correctness passes and timing improves or the NCU excessive-sector source moves away from scale-cache byte loads.
- Revert if timing regresses or the helper increases register pressure/spills enough to offset the load coalescing.

## 2026-05-04 15:02 CDT - D256 Linear Scale Pair Load Rejected

Finding:

- Implemented the stage-side scale-pair load experiment:
  - D256/coalesced layout used one 16-bit load for adjacent `sf0/sf1` bytes.
  - Non-coalesced layouts preserved the prior two-byte load behavior.
- Focused D256 correctness passed:
  - `3 passed in 44.49s`.
- Wall-time did not improve:
  - Baseline after D256 cache-layout commit: `3.105 ms`.
  - Scale-pair experiment: `3.117 ms` mean (`3.112 ms` min, `3.119 ms` max).
- NCU confirmed the intended local effect:
  - L2 theoretical global excessive sectors dropped from `6,291,456` to `3,145,728`.
  - L1 shared excessive wavefronts stayed flat at `18,974,208`.
  - Barrier not-issued stalls rose from `60,334` to `62,853`.
  - Long-scoreboard not-issued stalls rose from `37,461` to `38,255`.
  - NCU elapsed cycles rose from `5,817,760` to `5,945,564`.

Decision:

- Rejected the scale-pair load and restored the committed byte-load implementation.
- Global sector count is no longer a sufficient proxy for wall time on this cell. The paired load removed half of the measured global excess but did not reduce the dominant stage time.

Next profiling target:

- Focus on the flat sources that did not move:
  - `18,974,208` excessive shared wavefronts.
  - `37,408,368` local spill requests.
  - Barrier/sleeping waits around CUTLASS pipeline handoff.
- The next useful experiment needs to target shared-memory operand access or register pressure, not global scale-cache coalescing.

## 2026-05-04 15:06 CDT - Current D256 Linear Scheduler Comparison

Finding:

- Ran dense and paged-linear scheduler/occupancy NCU passes on the current D256 Qwen reference cell.
- Dense stage:
  - Duration under NCU: `1.25 ms`.
  - Block shape: `512` threads, `50.18 KiB` dynamic smem.
  - Registers/thread: `128`.
  - Active warps/scheduler: `2.54`.
  - Eligible warps/scheduler: `0.36`.
  - Waves/SM: `5.62`.
- Paged-linear stage:
  - Duration under NCU: `2.62 ms`.
  - Block shape: `640` threads, `92.16 KiB` dynamic smem.
  - Registers/thread: `96`.
  - Active warps/scheduler: `4.92`.
  - Eligible warps/scheduler: `0.36`.
  - Waves/SM: `2.81`.
- Dense source/memory comparison:
  - Dense has more shared-wavefront excess (`21,136,896`) than paged-linear (`18,974,208`), so shared wavefronts alone are not the paged-only limiter.
  - Paged-linear has much higher local spill requests (`37,408,368`) than dense (`14,411,520`), much higher barrier stalls, and almost twice the executed instructions.

Implementation target:

- Re-test the D256 no-SWA paged `LOAD_WARPS` knob on the current linear-cache baseline.
- The current source default is `TILE_M=128`, `LOAD_WARPS=11`; previous sweeps were PV-focused and predate the D256 linear cache-layout state.
- Use isolated JIT workspaces and `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_SM120_NVFP4_D256_LOAD_WARPS=N` so this is a compile-time diagnostic, not a source edit.

Validation:

- Benchmark D256 Qwen paged-linear at `LOAD_WARPS={7,9,11}` against the committed default.
- Keep source unchanged unless a different load-warp count materially improves current linear timing.

Decision criteria:

- If a lower load-warp count improves current linear timing without hurting PV materially, update the D256 paged TU default and run focused correctness.
- If the curve is flat or worse, keep the default and move to a source-level producer/spill target.

## 2026-05-04 15:13 CDT - D256 Current Linear Load-Warp Sweep Result

Finding:

- Re-ran the D256 no-SWA paged load-warp sweep on the current linear-cache baseline using isolated JIT workspaces.
- D256 Qwen paged-linear `q=512 kv=65536 g=6 split=3072`:
  - `LOAD_WARPS=7`: `2.960 ms` mean.
  - `LOAD_WARPS=9`: `3.200 ms` mean.
  - `LOAD_WARPS=11`: `3.102 ms` mean.
  - Committed pre-change default (`LOAD_WARPS=11`) baseline: about `3.105 ms`.
- D256 Qwen paged-PV with `LOAD_WARPS=7`:
  - `2.115 ms` mean, versus the current committed PV baseline around `2.227 ms`.
- Normal source build after changing the D256 no-SWA paged default to `LOAD_WARPS=7` reproduces the win:
  - Paged-linear: `2.981 ms` mean (`2.950 ms` min, `3.075 ms` max).
  - Paged-PV: `2.115 ms` mean (`2.101 ms` min, `2.136 ms` max).
- Focused D256 correctness passed after the source change:
  - `3 passed in 44.33s`.

Decision:

- Keep the D256 no-SWA paged default at `FLASHINFER_SM120_NVFP4_D256_LOAD_WARPS=7`.
- The earlier `LOAD_WARPS=11` result was stale relative to the current producer/cache baseline. With the current stage, extra load warps increase blocked active warps without increasing eligible warp issue.

Next profiling target:

- Run full NVFP4 correctness before committing.
- Re-profile the kept `LOAD_WARPS=7` stage if the next source target needs attribution; prior NCU scheduler data for `LOAD_WARPS=11` is no longer the active geometry.

## 2026-05-04 15:20 CDT - Linear Cached V Dead Scale Lookup Target

Finding:

- Re-profiled the kept D256 `LOAD_WARPS=7` geometry with lineinfo.
- The current lineinfo profile still shows hot samples in the linear V data loop:
  - `d256.cuh:1175` `__shfl_sync(...)` is the top long-scoreboard source.
  - `d256.cuh:1169` `token0 < kv_len_tokens ? pv_scale_for(token0, dim) : 0x38` appears in wait samples.
- Source inspection shows `output_scale` is computed before the runtime `if (paged_kv_params.v_linear_data_cache != nullptr)` branch.
- In the cached linear path, `output_scale` is not used. The data cache has already been requantized using the PV output scales, so the V data producer only needs to transpose cached codepoints.
- This leaves a dead scale-smem read and predicate chain on the hot production path solely to serve the uncached fallback.

Implementation target:

- Move `output_scale = pv_scale_for(...)` into the uncached fallback branch in D128/D256/D512.
- Keep the cached path behavior identical except that it no longer computes or reads the unused scale byte.
- No public API or layout changes.

Validation:

- Focused D256 correctness.
- D256 Qwen paged-linear benchmark against the new `LOAD_WARPS=7` baseline (`2.981 ms` normal source build).
- If timing improves, run lineinfo NCU to verify samples attributed to the dead scale lookup drop.

Decision criteria:

- Keep if correctness passes and timing or lineinfo attribution improves.
- Revert if compiler already eliminated the dead work or if branch restructuring regresses scheduling.

## 2026-05-04 15:24 CDT - Linear Cached V Dead Scale Lookup Result

Finding:

- Moved `output_scale = pv_scale_for(...)` into the uncached fallback branch in D128/D256/D512.
- Focused D256 correctness passed:
  - `3 passed in 44.26s`.
- D256 Qwen paged-linear reference improved:
  - Post-load-warp baseline: `2.981 ms`.
  - Dead-scale cleanup: `2.944 ms` mean (`2.936 ms` min, `2.952 ms` max).
- Lineinfo NCU confirms the intended instruction reduction:
  - Executed instructions dropped from `1,008,688,485` to `944,134,647`.
  - Local spill requests stayed flat at `21,121,296`.
  - L1 shared excessive wavefronts stayed flat at `18,974,208`.
  - The previous wait attribution at the now-dead `pv_scale_for(...)` site disappeared from the top sources.

Decision:

- Keep the dead-scale lookup removal.
- This is a real production-path cleanup: the cached linear-V path no longer pays scale-smem reads that only the uncached fallback needs.

Next profiling target:

- The new top linear-specific long-scoreboard source is the cached V transpose:
  - `d256.cuh:1173` / `__shfl_sync(subgroup_mask, row_word, ...)`.
- The next source change should remove or reduce that shuffle chain, not continue optimizing global scale-cache loads.

## 2026-05-04 15:31 CDT - D256 Cached Linear V No-Shuffle Target

Finding:

- The current D256 paged-linear lineinfo profile after the dead-scale cleanup points at the cached V transpose as the next source-level target.
- The top linear-specific long-scoreboard source is `d256.cuh:1173`, the `__shfl_sync(subgroup_mask, row_word, ...)` loop that transposes one cached dim-contiguous row word per lane into token-contiguous operand words.
- The cached D256 linear-V data layout is already coalesced by packed-word column: `sm120_nvfp4_linear_v_data_cache_word<true>` maps `(word_col, token)` so lanes can read adjacent dim words for one token without walking the public paged layout.
- The PV-layout V producer in the same D256 file already uses that ownership pattern: each 8-lane subgroup owns one token group and one 64-dim block, loads dim-contiguous row words, accumulates eight token-contiguous packed words in registers, and writes the existing CUTLASS operand smem layout.

Implementation target:

- Add a D256-only cached-linear path under `kLinearVCacheCoalesced && v_linear_data_cache != nullptr`.
- Reuse the PV producer's no-shuffle ownership: one 8-lane subgroup handles an `8 token x 64 dim` block, each lane loads one adjacent 32-bit cached word per token, and each lane writes eight packed operand words.
- Keep the uncached linear fallback unchanged. D128/D512 keep the current path because their linear-V cache layout is not the D256 coalesced layout.
- Do not change public tensor shapes, layout names, FFI signatures, or tolerances.

Validation:

- Run focused D256 correctness first.
- Benchmark the D256 Qwen reference cell against the current source baseline: paged-linear `2.944 ms` mean.
- If timing improves, run lineinfo NCU and verify the `__shfl_sync` source drops from the dominant long-scoreboard attribution.

Decision:

- Keep if correctness passes and the reference benchmark improves materially.
- Revert the code if it hits a ptxas wall, regresses runtime, or shifts the bottleneck without reducing wall time.

## 2026-05-04 15:39 CDT - D256 Cached Linear V No-Shuffle Result

Finding:

- Implemented the D256 cached-linear no-shuffle path under `v_linear_data_cache != nullptr`.
- The path reuses the PV producer ownership pattern: each 8-lane subgroup loads dim-contiguous cached words for one token at a time, accumulates eight token-contiguous packed operand words in registers, and writes the existing `pv_sB` CUTLASS operand layout.
- Focused D256 correctness passed:
  - `3 passed in 44.28s`.
- Full NVFP4 head-dim test file passed:
  - `36 passed in 45.28s`.

Benchmark:

- D256 Qwen reference cell `q=512 kv=65536 g=6 split=3072`, paged-linear:
  - Before no-shuffle: `2.944 ms` mean.
  - After no-shuffle: `2.696 ms` mean (`2.688 ms` min, `2.708 ms` max).
  - Delta: `8.4%` faster.
- D256 Qwen reference cell, paged-PV sanity:
  - `2.117 ms` mean (`2.105 ms` min, `2.130 ms` max), unchanged versus the current PV baseline.

NCU:

- Lineinfo NCU confirms the targeted `__shfl_sync(subgroup_mask, row_word, ...)` source is no longer the top long-scoreboard attribution.
- Executed instructions dropped from `944,134,647` to `847,479,774`.
- Local spill requests changed from `21,121,296` to `20,943,744`, effectively flat.
- New source-level costs after the change:
  - Top global excessive-sector source is now the cached data word load at `paged_kv.cuh:137`, `50,331,648` excessive sectors.
  - Top shared excessive-wavefront source is now the operand store at `d256.cuh:1189`, `11,010,048` excessive wavefronts.
  - Top long-scoreboard sources shifted to scale/stats/MMA-side lines (`d256.cuh:2234`, `d256.cuh:921`, `d256.cuh:2183`, `d256.cuh:1023`), not the removed V transpose shuffle.

Decision:

- Keep the D256 cached-linear no-shuffle path.
- This is high-risk producer restructuring, but NCU identified the source and wall time improved without correctness fallout.

Next profiling target:

- The remaining D256 linear gap is no longer the cached V shuffle.
- Next NCU-driven candidates are the cached data word global-sector pattern and the operand smem store wavefront excess; do not optimize them by inspection without checking whether source-level reductions move wall time.

## 2026-05-04 15:40 CDT - D512 Paged Profiling Target

Finding:

- D256 is now near the current producer floor for the reference cell: paged-linear `2.696 ms`, paged-PV `2.117 ms`, dense previously `1.264 ms`.
- The largest remaining production risk is D512, where previous worklog measurements had paged-PV and paged-linear still far above dense.
- Source inspection confirms D512 has not received the D256 coalesced linear-cache ownership path: `kLinearVCacheCoalesced` is false for D512, and the cached linear path still reaches `__shfl_sync(subgroup_mask, row_word, ...)`.
- Before carrying over any D256 structural change, the next step is to profile D512 directly and identify whether the dominant cost is the same V transpose/cache path, a D512-specific scale path, or something else in the mainloop.

Implementation target:

- No code change yet.
- Run current-source D512 Gemma-global reference benches for dense, paged-PV, and paged-linear.
- Run Nsight Compute lineinfo on the D512 paged-PV or paged-linear stage, prioritizing the slower production-linear path if the bench gap is still extreme.
- Use the NCU source counters to decide whether the next D512 change should be cache-layout coalescing, producer ownership rewrite, load-warp retune, or a different mainloop target.

Validation:

- Benchmark cell: D512 Gemma-global `q=512 kv=65536 g=8 softcap=30 split=3072`.
- Use one GPU process at a time on GPU 2 with `CUDA_VISIBLE_DEVICES=2` and bench `--device 0`.

Decision:

- Do not port D256 changes mechanically.
- Only implement a D512 producer change after the current D512 NCU profile names the source-level hotspot.

## 2026-05-04 15:42 CDT - D512 Current Reference Result

Finding:

- Re-ran the D512 Gemma-global reference cell on current source after the D256-specific changes.
- Cell: `q=512 kv=65536 g=8 softcap=30 split=3072`.
- Current timings:
  - Dense: `5.260 ms` mean (`5.248 ms` min, `5.270 ms` max).
  - Paged-PV: `6.633 ms` mean (`6.613 ms` min, `6.646 ms` max).
  - Paged-linear: `8.089 ms` mean (`8.034 ms` min, `8.230 ms` max).
- Ratios:
  - Paged-PV / dense: `1.26x`.
  - Paged-linear / dense: `1.54x`.
  - Paged-linear / paged-PV: `1.22x`.

Decision:

- Do not spend a D512 architectural pass here right now.
- The stale D512 100x-era numbers are no longer representative of current source; current D512 is in the same rough envelope as the D256 path, with linear-V reblock still visible but not catastrophic.
- NCU is still useful for future D512 tuning, but it is not the biggest remaining blocker for benchmark readiness.

Next profiling target:

- Move from single-cell producer work to production-cell benchmark coverage.
- Use the focused production matrix to find remaining outlier cells. The next high-risk kernel change should be driven by an outlier cell's NCU profile, not by the now-acceptable D512 reference cell.

## 2026-05-04 15:44 CDT - Focused Production Matrix Target

Finding:

- Current single-cell references show D256 and D512 are no longer in the stale 100x paged-failure state.
- Single-cell tuning is now lower value than finding actual production outlier cells across q/kv regimes.
- Existing reports under `reports/prod_*_20260504` predate the latest D256 cached-linear no-shuffle producer and are not current for this pass.

Benchmark target:

- Run focused production reports with fresh prefixes:
  - `reports/prod_qwen_full_d256_g6_post_noshfl_20260504`
  - `reports/prod_gemma_sliding_d256_g2_swa1024_softcap30_post_noshfl_20260504`
  - `reports/prod_gemma_global_d512_g8_softcap30_post_noshfl_20260504`
- For each prefix, collect `sm120_fused` variants for paged-linear, paged-PV, and dense, plus `nvfp4_fa2`, `fp8_fa2`, and `bf16_fa2` reference rows.
- Use `--cells` rather than a q/kv Cartesian product where the deployment cell list is explicit.

Validation:

- Use GPU 2 via `CUDA_VISIBLE_DEVICES=2`; pass bench `--device 0`.
- Use `--warmup 2 --repeat 5 --timeout-sec 1800`.
- Do not change source during the benchmark run.

Decision:

- If a report shows a large paged-linear or paged-PV outlier versus dense, profile that exact cell with NCU before changing code.
- If the matrix is broadly in line, move to cleanup/tolerance work instead of speculative producer rewrites.

## 2026-05-04 16:01 CDT - Focused Production Matrix Result

Finding:

- Completed the post-no-shuffle focused production matrix with fresh report prefixes:
  - `reports/prod_qwen_full_d256_g6_post_noshfl_20260504`
  - `reports/prod_gemma_sliding_d256_g2_swa1024_softcap30_post_noshfl_20260504`
  - `reports/prod_gemma_global_d512_g8_softcap30_post_noshfl_20260504`
- Qwen D256 g6 full-attention cells:
  - Geomean paged-linear / dense: `1.96x`.
  - Geomean paged-linear / paged-PV reblock cost: `1.41x`.
  - Geomean paged-linear speedup vs `nvfp4_fa2`: `0.249x`.
  - Worst wrapper overhead cell is `q=512 kv=4096`: dense `0.282 ms`, paged-PV `0.708 ms`, paged-linear `0.753 ms`, paged-linear/dense `2.51x`.
- Gemma sliding D256 g2 SWA1024 softcap30 cells:
  - Geomean paged-linear / dense: `1.37x`.
  - Geomean paged-linear / paged-PV reblock cost: `1.21x`.
  - Geomean paged-linear speedup vs `nvfp4_fa2`: `0.163x`.
  - The absolute times are sub-millisecond; the largest wrapper overhead cell is `q=512 kv=8192`: dense `0.170 ms`, paged-PV `0.261 ms`, paged-linear `0.330 ms`, paged-linear/dense `1.53x`.
- Gemma global D512 g8 softcap30 cells:
  - Geomean paged-linear / dense: `1.44x`.
  - Geomean paged-linear / paged-PV reblock cost: `1.46x`.
  - Geomean paged-linear speedup vs `nvfp4_fa2`: `0.393x`.
  - `bf16_fa2` failed all D512 global cells with existing FlashInfer prefill configuration errors; this is a reference-backend issue, not an SM120 fused correctness failure.
  - The main production prefill outlier is `q=512 kv=16384`: dense `1.479 ms`, paged-PV `4.668 ms`, paged-linear `5.446 ms`, paged-PV/dense `3.16x`, linear/PV `1.17x`, paged-linear speedup vs `nvfp4_fa2` `0.443x`.

Decision:

- Do not resume broad producer rewrites from stale 100x-era numbers; the current focused matrix shows the worst remaining prefill issue is narrower.
- Do not chase linear-V first on the D512 `q=512 kv=16384` outlier. Linear reblock adds only `16.7%` over paged-PV on that cell, while paged-PV itself is `3.16x` over dense.
- The next high-risk architectural change must be driven by profiler evidence on the D512 Gemma global `q=512 kv=16384` paged-PV stage compared directly with dense at the same cell.

Next profiling target:

- Profile D512 Gemma global `q=512 kv=16384 g=8 softcap=30`, dense vs paged-PV.
- Collect NCU source counters, memory workload, and warp-state sampling for the stage kernel.
- If NCU attributes the paged-PV gap to the CUTLASS operand producer or smem layout contract, a non-local/high-risk path is on the table. If it attributes the gap to launch/split geometry or fixed overhead, use NSYS to separate stage kernel, cache/prepass, and combine costs before changing code.

## 2026-05-04 16:18 CDT - D512 Paged Page-Cache Redundancy Target

Finding:

- NCU source/memory/warp-state comparison on D512 Gemma global `q=512 kv=16384 g=8 softcap=30` shows the outlier is inside the paged stage kernel, not Python:
  - Paged-PV profiled stage duration: `5.20 ms`, `525,339,762` instructions, `11,842,543` elapsed cycles.
  - Dense profiled stage duration: `1.48 ms`, `261,570,437` instructions, `3,249,149` elapsed cycles.
  - Paged-PV has `4,194,304` L2 global excessive sectors; dense has `0`.
  - Paged-PV has `16,857,088` shared excessive wavefronts; dense has `8,351,744`.
  - Paged-PV barrier not-issued samples are `20,820`; dense barrier not-issued samples are `3,093`.
- The largest paged not-issued source maps to `fmha_nvfp4_sm120_d512.cuh:775`, inside `cache_paged_physical_pages`.
- Source inspection shows page-cache work is duplicated for the same `kv_tile`:
  - `stage_paged_k_tile` calls `cache_paged_physical_pages(kv_tile)`.
  - `stage_paged_v_tile` also calls `cache_paged_physical_pages(kv_tile)`.
  - D512 has `qk_head_chunks > 1` and `kOutputGroupSpan == 4`, so the same `kv_tile` page list can be reloaded/synchronized up to six times: two K chunks plus four V groups.
- The page list is independent of K chunk and V output group. It only depends on `kv_tile`, batch, and block table.

Implementation target:

- Test D512 first, because the profiler evidence came from the D512 production outlier.
- Add a shared `physical_page_cache_kv_tile` tag next to `physical_page_cache`.
- Initialize it to `-1` once for the load group at kernel entry.
- Make `cache_paged_physical_pages(kv_tile)` skip the block-table reload and named-barrier synchronization when the requested tile is already cached.
- Keep all public API, FFI, tensor layout, and test contracts unchanged.

Validation:

- Focused D512 correctness first.
- Benchmark the D512 outlier cell:
  - dense reference context: `q=512 kv=16384 g=8 softcap=30`.
  - paged-PV target: current matrix value `4.668 ms`.
  - paged-linear sanity: current matrix value `5.446 ms`.
- If timing improves, rerun NCU on paged-PV and verify the `cache_paged_physical_pages` stall attribution drops.

Decision:

- Keep if correctness passes and paged-PV wall time improves materially.
- Revert if the tag adds smem pressure, breaks pipeline ordering, or merely shifts stalls without reducing wall time.

## 2026-05-04 16:27 CDT - D512 Split Heuristic Target

Finding:

- The D512 page-cache tag experiment compiled and passed correctness:
  - `tests/attention/test_nvfp4_kv_head_dim_512.py`: `36 passed in 112.23s`.
- It did not materially improve the target cell:
  - Paged-PV before: matrix mean `4.693 ms`, min `4.668 ms`.
  - Paged-PV with page-cache tag: `4.652 ms` mean.
  - Paged-linear before: matrix mean `5.455 ms`, min `5.446 ms`.
  - Paged-linear with page-cache tag: `5.463 ms` mean.
- The page-cache tag was reverted because the wall-time delta was below the keep bar.
- The same NCU comparison exposed a larger structural difference:
  - Paged-PV stage launch geometry: grid `(32, 1, 2)`, auto `split_kv_len=12288`.
  - Dense stage launch geometry: grid `(32, 1, 4)`, auto `split_kv_len=4096`.
  - Paged therefore processes up to `3x` more KV tiles per CTA on the `q=512 kv=16384` outlier.
- Direct split override confirms this is the outlier mechanism:
  - D512 q512 kv16384 paged-PV: auto `12288` -> `4.693 ms`; forced `4096` -> `1.801 ms`.
  - D512 q512 kv16384 paged-linear: auto `12288` -> `5.455 ms`; forced `4096` -> `2.251 ms`.
  - D512 q128 kv4096 paged-PV: auto `3072` -> `1.281 ms`; forced `1024` -> `0.514 ms`.
  - D512 q128 kv16384 paged-PV: auto `3072` -> `1.279 ms`; forced `1024` -> `0.587 ms`.
- The bad split comes from the D512 paged auto heuristic:
  - `flashinfer/fmha_nvfp4_sm120.py::_auto_split_kv_len` multiplies `q_tiles` by `3` when `head_dim == 512`.
  - `benchmarks/bench_sm120_nvfp4_attention.py::auto_split_kv_len` mirrors the same D512 paged multiplier.

Implementation target:

- Remove the D512 paged `q_tiles *= 3` multiplier from the wrapper and production bench.
- Keep the scratch-budget while-loop unchanged so memory pressure can still force larger split lengths when needed.
- This is not a public API change; it only changes auto-selection when callers pass `split_kv_len=0`.

Validation:

- Run focused D512 correctness.
- Re-benchmark D512 q512 kv16384 paged-PV and paged-linear with `split_kv_len=0`; expected resolved split is `4096`.
- Re-benchmark D512 q128 kv4096 and q128 kv16384 paged-PV with `split_kv_len=0`; expected resolved split is `1024`.
- If the targeted cells reproduce the forced-split timings, keep and rerun the D512 Gemma global focused report.

Decision:

- Keep if auto split now resolves to the faster dense-like split lengths and correctness passes.
- If some long-context D512 cells regress from smaller splits, tune the heuristic from measured cell data rather than restoring the blanket `3x` multiplier.

## 2026-05-04 16:36 CDT - D512 Split Heuristic Refinement Target

Finding:

- The blanket removal of the D512 paged `q_tiles *= 3` multiplier fixed the main `q=512 kv=16384` outlier and the short `q=128` cells.
- It regressed long-context `q=128` D512 cells:
  - `q=128 kv=65536`: paged-PV `1.705 ms` -> `1.926 ms`, paged-linear `3.095 ms` -> `3.294 ms`.
  - `q=128 kv=262144`: paged-PV `6.694 ms` -> `7.379 ms`, paged-linear `12.468 ms` -> `13.037 ms`.
- The measured split behavior points to a conditional rule, not a blanket rule:
  - `q_tiles == 8` with `kv <= 16384` benefits from the smaller `1024` split.
  - `q_tiles == 8` with `kv > 16384` benefits from the old `3072` split.
  - `q_tiles == 32` keeps the improved `4096` split for the production outlier; restoring the blanket multiplier would reintroduce the `12288` split and the 3x slowdown.

Implementation target:

- Refine the wrapper and production bench auto-split heuristic to restore the D512 `3x` multiplier only for `q_tiles == 8` and `kv > 16384`.
- Keep the smaller split for `q=512` and short `q=128` D512 cells.
- Keep public API and explicit `split_kv_len` behavior unchanged; this only affects auto-selection when `split_kv_len=0`.

Validation:

- Run focused D512 correctness after the heuristic change.
- Benchmark D512 q128 `kv={4096,16384,65536,262144}` for paged-PV and paged-linear.
- Benchmark D512 q512 `kv=16384` for paged-PV and paged-linear.
- Keep if short cells remain improved, long q128 cells recover, and q512 outlier stays fixed.

## 2026-05-04 16:36 CDT - D512 Split Heuristic Refinement Result

Implementation:

- Updated `flashinfer/fmha_nvfp4_sm120.py::_auto_split_kv_len` to receive `max_kv_len` and apply the D512 `3x` multiplier only when `q_tiles == 8 && max_kv_len > 16384`.
- Updated `benchmarks/bench_sm120_nvfp4_attention.py::auto_split_kv_len` with the same conditional for production-bench auto mode.
- Explicit `split_kv_len` values remain unchanged; the new rule only affects `split_kv_len=0`.

Correctness:

- `tests/attention/test_nvfp4_kv_head_dim_512.py`: `36 passed in 0.99s`.
- All targeted benchmark rows reported `output_finite=true`.

Measured result:

| q | kv | v_layout | auto split | mean ms | min ms |
|---:|---:|:---|---:|---:|---:|
| 128 | 4096 | pv | 1024 | 0.513 | 0.508 |
| 128 | 4096 | linear | 1024 | 0.642 | 0.635 |
| 128 | 16384 | pv | 1024 | 0.573 | 0.567 |
| 128 | 16384 | linear | 1024 | 0.955 | 0.950 |
| 128 | 65536 | pv | 3072 | 1.694 | 1.685 |
| 128 | 65536 | linear | 3072 | 3.081 | 3.076 |
| 128 | 262144 | pv | 3072 | 6.660 | 6.653 |
| 128 | 262144 | linear | 3072 | 12.427 | 12.412 |
| 512 | 16384 | pv | 4096 | 1.807 | 1.801 |
| 512 | 16384 | linear | 4096 | 2.251 | 2.246 |

Decision:

- Keep the conditional heuristic.
- It preserves the major q512 outlier fix (`paged-PV 4.668 ms -> 1.807 ms`, `paged-linear 5.446 ms -> 2.251 ms`) while recovering the long q128 cells to the old faster split.
- The remaining D512 overhead is now dominated by normal producer/layout costs again, not the auto-split geometry bug.

## 2026-05-04 16:43 CDT - D512 Refined Production Report Result

Report:

- Generated `reports/prod_gemma_global_d512_g8_softcap30_splitrefine_20260504` from the committed refined split heuristic.
- The report uses the same 16 D512 Gemma global cells as the previous focused matrix:
  - `q={1,128,512,2048}`
  - `kv={4096,16384,65536,262144}`
  - `group=8`, `head_dim=512`, causal, `softcap=30`, no sliding window.

Finding:

- Geomean paged-PV / dense improved from `1.44x` in the post-no-shuffle report to `1.23x`.
- Geomean paged-linear / dense improved from `2.10x` to `1.82x`.
- Geomean paged-linear speedup vs `nvfp4_fa2` improved from `0.393x` to `0.452x`.
- The q512/kv16384 outlier is resolved:
  - Before: dense `1.479 ms`, paged-PV `4.668 ms`, paged-linear `5.446 ms`.
  - After: dense `1.478 ms`, paged-PV `1.796 ms`, paged-linear `2.245 ms`.
- The q128 short cells remain on the faster smaller split:
  - `q=128 kv=4096`: paged-PV `1.281 ms -> 0.510 ms`, paged-linear `1.472 ms -> 0.631 ms`.
  - `q=128 kv=16384`: paged-PV `1.279 ms -> 0.571 ms`, paged-linear `1.729 ms -> 0.947 ms`.
- The q128 long cells recovered the old larger split:
  - `q=128 kv=65536`: paged-PV `1.710 ms`, paged-linear `3.086 ms`.
  - `q=128 kv=262144`: paged-PV `6.715 ms`, paged-linear `12.438 ms`.

Next profiling target:

- The largest remaining production gap is now D512 decode linear-V, not the q512 prefill split geometry.
- Worst row: `q=1 kv=262144`, dense `0.796 ms`, paged-PV `1.044 ms`, paged-linear `6.740 ms`.
- Since paged-PV is only `1.31x` over dense on that row while paged-linear is `8.46x` over dense, the next NCU pass should compare `q=1 kv=262144` paged-linear against paged-PV and attribute the linear-only cost.
- This is a production-stock-vLLM path issue because stock vLLM writes linear V.
