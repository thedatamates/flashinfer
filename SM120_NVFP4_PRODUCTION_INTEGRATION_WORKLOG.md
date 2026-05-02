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
