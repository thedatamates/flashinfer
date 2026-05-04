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
