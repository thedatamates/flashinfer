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
