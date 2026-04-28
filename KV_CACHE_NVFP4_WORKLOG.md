# KV Cache NVFP4 Worklog

## Primary Target

Gemma4 31B dense, 60 layers:

```text
Shape A - sliding attention, 50/60 layers:
  head_dim = 256
  group = 32 query heads / 16 KV heads = 2
  kv_len = 1024 sliding-window cap
  q_len in {1, 512, 2048}

Shape B - global attention, 10/60 layers:
  head_dim = 512
  group = 32 query heads / 4 KV heads = 8
  kv_len in {8192, 32768, 131072}
  q_len in {1, 512, 2048}
```

Shape B is prioritized for long-context wall time. D128 is not a Gemma4 31B shape and should only be used as a sanity/stress diagnostic.

## Goal

Build a production-quality fused dense GQA NVFP4 attention path for paged KV cache on Blackwell, including prefill and decode. The target is to beat the existing fused BF16 production path on the Gemma4 production shapes above.

## Measurement Policy

Use wall time as the primary shipping metric and effective attention TFLOP/s as a diagnostic metric. The scoreboard for each benchmark shape is:

```text
shape, group, wall_ms, effective_TFLOP/s, speedup_vs_BF16
```

Wall time decides whether the FP4 path is useful in production. TFLOP/s identifies utilization/headroom and helps compare across D128/D256/D512.

## 2026-04-26

### Correctness Baseline

`tests/attention/test_nvfp4_kv_head_dim_512.py` passes after the D512 prefill/decode correctness work:

```text
25 passed
```

### Scale-Load Shuffle Experiment

Tried reducing duplicate K/V scale shared-memory loads with warp shuffles. Correctness passed, but performance regressed:

```text
D128 group8: ~1.47 ms -> 1.92 ms
D256 group8: ~2.76 ms -> 3.17 ms
D512 group8: ~2.53 ms -> 2.52 ms
```

Conclusion: the shuffle increases instruction/register pressure enough to hurt D128/D256 and does not materially improve D512. Reverted the experiment.

### Current Grouped FP4 vs BF16 Baseline

After reverting the scale-shuffle experiment and keeping the byte-load lane-pair optimization:

```text
D128 group8: FP4 1.4696 ms, 11.69 TFLOP/s vs BF16 0.3140 ms, 54.71 TFLOP/s, 0.21x
D256 group8: FP4 2.7642 ms, 12.43 TFLOP/s vs BF16 0.7942 ms, 43.26 TFLOP/s, 0.29x
D512 group8: FP4 2.4912 ms, 27.58 TFLOP/s vs BF16 1.2069 ms, 56.94 TFLOP/s, 0.48x
```

Conclusion: FP4 is still underutilizing the card. D512 is closest, but still slower than BF16. The next bottleneck investigation is the staged split across K staging, V global staging, V shared-fragment load, BMM1, BMM2, softmax/pack, and output.

### Synthetic D128 Group8 Split Profile

This is not a Gemma4 31B production shape. Keep it as a structural diagnostic only.

Shape: `q_len=512`, `kv_len=8192`, `head_dim=128`, `group=8`.

```text
base:                 1.4856 ms
skip K staging:       0.8787 ms
skip V gmem staging:  1.1079 ms
skip V smem load:     0.9477 ms
skip V staging:       0.5628 ms
skip BMM1 GEMM:       1.3937 ms
skip BMM2 GEMM:       0.9388 ms
skip softmax pack:    1.3835 ms
skip output:          0.9108 ms
```

Critical-path sensitivity:

```text
remove K staging/load:        -0.61 ms
remove V gmem staging:        -0.38 ms
remove V smem fragment load:  -0.54 ms
remove V staging combined:    -0.92 ms
remove BMM1 GEMM:             -0.09 ms
remove BMM2 GEMM:             -0.55 ms
remove softmax pack:          -0.10 ms
remove output:                -0.57 ms
```

These numbers are not additive costs. The kernel is pipelined, so removing one stage exposes the next critical path. Interpret each number as "how much shorter the wall time gets when this work is removed," not "how much isolated time this stage consumes."

Conclusion: D128 is not QK/BMM1 math-bound. The large FP4-vs-BF16 gap is structural: FP4 is spending wall time on data movement, dequant/scale handling, output movement/reduction, or poor overlap. Skip profiling alone cannot distinguish long-scoreboard, MIO throttle, shared-memory bank conflicts, or tensor-pipe underutilization; Nsight Compute is required before committing to the next optimization.

### Scale-Load Shuffle Lesson

At small head dimensions this kernel is register/coordination sensitive. A warp-shuffle dedup of scale loads reduced nominal shared-memory loads but regressed D128/D256 by about 30%. Avoid retrying "dedup via `__shfl_sync`" unless Nsight shows the tradeoff has changed; redundant lane work can be faster than added warp coordination and register pressure.

### Synthetic D256 Group8 Split Profile

This is not a Gemma4 31B production shape. Gemma4 D256 appears in sliding layers with `group=2` and `kv_len<=1024`, not `group=8`, `kv_len=8192`.

Shape: `q_len=512`, `kv_len=8192`, `head_dim=256`, `group=8`.

```text
base:                 2.7744 ms
skip K staging:       1.2343 ms
skip V gmem staging:  2.3111 ms
skip V smem load:     1.3641 ms
skip V staging:       0.9194 ms
skip BMM1 GEMM:       1.9830 ms
skip BMM2 GEMM:       1.3199 ms
skip softmax pack:    2.6031 ms
skip output:          1.2799 ms
```

Critical-path sensitivity:

```text
remove K staging/load:        -1.54 ms
remove V gmem staging:        -0.46 ms
remove V smem fragment load:  -1.41 ms
remove V staging combined:    -1.85 ms
remove BMM1 GEMM:             -0.79 ms
remove BMM2 GEMM:             -1.45 ms
remove softmax pack:          -0.17 ms
remove output:                -1.49 ms
```

Conclusion: D256 is highly sensitive to K staging/load, V shared-fragment load, BMM2, and output. Optimizations that only target D128 V staging are not enough.

### Synthetic D512 Group8 Split Profile

This is closer to Gemma4 global attention by `head_dim=512, group=8`, but `kv_len=8192` is only one short global-context point. Gemma4 long-context priority includes `kv_len=32768` and `131072`.

Shape: `q_len=512`, `kv_len=8192`, `head_dim=512`, `group=8`.

```text
base:                 2.4981 ms
skip K staging:       0.9981 ms
skip V gmem staging:  2.2713 ms
skip V smem load:     2.0546 ms
skip V staging:       1.8492 ms
skip BMM1 GEMM:       1.5244 ms
skip BMM2 GEMM:       2.0036 ms
skip softmax pack:    2.3611 ms
skip output:          1.8466 ms
```

Critical-path sensitivity:

```text
remove K staging/load:        -1.50 ms
remove V gmem staging:        -0.23 ms
remove V smem fragment load:  -0.44 ms
remove V staging combined:    -0.65 ms
remove BMM1 GEMM:             -0.97 ms
remove BMM2 GEMM:             -0.49 ms
remove softmax pack:          -0.14 ms
remove output:                -0.65 ms
```

Conclusion: K staging/load is the common cross-shape sensitivity for D256/D512 and still matters for D128. Before changing code again, run Nsight Compute on D128 grouped FP4 to classify the structural gap: long scoreboard versus MIO throttle versus shared-memory conflicts versus tensor-pipe underutilization.

### D128 Group8 Nsight Compute

Profile:

```text
reports/ncu/nvfp4_d128_group8_fmha.ncu-rep
kernel: fmha_v2_flash_attention_bf16_32_64_S_q_paged_kv_128_causal_output_bf16_kv_e2m1_gqa_m_sm120_kernel_nl_tiled
shape: q_len=512, kv_len=8192, head_dim=128, group=8
```

Key metrics:

```text
duration:                         1.5698 ms
SM throughput:                    14.82%
DRAM throughput:                   0.09%
L2 throughput:                     9.33%
tensor pipe active:                0.91%
ALU-heavy pipe active:            10.62%
LSU pipe active:                  14.82%
MIO inst issued:                   5.17%
active warps / SMSP:               1.00
eligible warps / SMSP:             0.17
registers/thread:                   255
shared memory/block:             19,200 B
occupancy limit by registers:         2 blocks/SM
occupancy limit by shared memory:      5 blocks/SM
ldgsts instructions:                  0
TMA load instructions:                0
shared loads:                11,223,040 instructions
shared stores:                  870,272 instructions
shared bank conflicts:       26,821,632
shared wavefronts:           39,847,296
ideal shared wavefronts:     13,025,664
shared wavefront / ideal:          3.06x
```

Warp-sampling stall mix:

```text
long_scoreboard: 24.0%
selected:        17.4%
wait:            16.2%
no_instructions: 16.1%
barrier:         13.2%
short_scoreboard:10.2%
mio_throttle:     0.5%
```

Conclusion: this is not HBM-bandwidth bound and not tensor-core saturated. The kernel is dominated by on-chip/shared-memory behavior, register-limited occupancy, and poor issue eligibility. The lack of `ldgsts`/TMA means the paged FP4 path is not using the same kind of global-to-shared async copy machinery expected from the fastest Blackwell path. The immediate investigation should focus on K/V shared-memory layout and global-to-shared staging, not more math tuning.

### Profiling Priority

D512 is the production lead target, followed by D256. D128 is useful as a sanity/stress shape but should not drive prioritization.

Current grouped FP4/BF16 ratios:

```text
D512 group8: 0.48x
D256 group8: 0.29x
D128 group8: 0.21x
```

Next optimization decisions should be based on what is common to D512 and D256. D128 may be structurally hardest because there is less math per staged byte to amortize the pipeline overhead.

### Synthetic D512 FP4 vs BF16 Nsight Compare

Shape: `q_len=512`, `kv_len=8192`, `head_dim=512`, `group=8`.

This is a synthetic short-global shape, not the primary Gemma4 long-context point, but it confirms the structural delta against BF16:

```text
FP4 duration:                  2.4291 ms
BF16 duration:                 1.4196 ms

FP4 ldgsts instructions:             0
BF16 ldgsts instructions:    2,039,808

FP4 TMA load instructions:            0
BF16 TMA load instructions:           0

FP4 tensor pipe active:           3.95%
BF16 tensor pipe active:         16.16%

FP4 shared bank conflicts:  65,901,846
BF16 shared bank conflicts:     11,288

FP4 shared wavefront/ideal:       2.43x
BF16 shared wavefront/ideal:      1.00x

FP4 shared loads:          38,544,384 instructions
BF16 shared loads:             8,192 instructions

FP4 registers/thread:             255
BF16 registers/thread:            254
```

Conclusion: the production BF16 path uses `ldgsts` async global-to-shared staging and has near-ideal shared-memory access. The FP4 path uses no async copy, issues tens of millions of shared loads, and has large shared-memory bank conflicts. This is the clearest structural gap so far and supports prioritizing async K/V staging plus shared-memory layout fixes.

### Rejected Optimization: K/V Buffer Sharing

Do not retry folding K and V loads based on `attention_k_eq_v=true`.

`attention_k_eq_v=true` is a weight-tying optimization at the QKV projection level only. At the attention kernel input, K and V are distinct tensors: K passes through a different per-head norm with learnable weight and gets RoPE applied, while V passes through a weight-less norm and skips RoPE. Folding K/V loads would be incorrect.

### Structural Fix Priority

Rank against Gemma4 production shapes, especially Shape B first:

```text
1. Convert K/V global-to-shared staging to cp.async with >=2 stages.
2. Use TMA cp.async.bulk.tensor for K/V if paged-table indirection allows.
3. Fix shared-memory swizzle to reduce bank conflicts toward 1.0x ideal.
4. Reduce register pressure enough to reach 3 blocks/SM if possible.
```

### Gemma4 Shape B Baseline

Shape B global attention:

```text
head_dim=512
group=8
q_len=512
kv_len=32768
page_size=16
```

Baseline:

```text
fused FP4 grouped: 9.8661 ms, 27.86 TFLOP/s
BF16 production:   4.6230 ms, 59.46 TFLOP/s
FP4/BF16:          0.47x
```

Conclusion: the production Gemma4 long-global shape has the same high-level problem as the synthetic D512/kv8192 shape. FP4 grouped is about 2.1x slower than BF16 production despite smaller KV payload.

### Gemma4 Shape B Nsight Compute

Profile:

```text
reports/ncu/gemma4_shapeB_d512_g8_kv32768_fp4_fmha.ncu-rep
kernel: fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_512_causal_output_bf16_kv_e2m1_gqa_m_w2x2_o128_sm120_kernel_nl_tiled
shape: D=512, group=8, q_len=512, kv_len=32768
```

Key metrics:

```text
duration:                         9.6321 ms
SM throughput:                    19.77%
DRAM throughput:                   0.14%
L2 throughput:                    22.72%
tensor pipe active:                4.06%
ALU-heavy pipe active:            19.16%
LSU pipe active:                  19.61%
MIO inst issued:                   7.16%
active warps / SMSP:               1.36
eligible warps / SMSP:             0.22
registers/thread:                   255
shared memory/block:             47,104 B
occupancy limit by registers:         2 blocks/SM
occupancy limit by shared memory:      2 blocks/SM
ldgsts instructions:                  0
TMA load instructions:                0
shared loads:               156,902,400 instructions
shared stores:               11,776,512 instructions
shared bank conflicts:      267,664,196
shared wavefronts:          449,982,976
ideal shared wavefronts:    185,008,640
shared wavefront / ideal:          2.43x
```

Warp-sampling stall mix:

```text
wait:            23.6%
long_scoreboard: 23.4%
no_instructions: 19.6%
selected:        14.6%
short_scoreboard: 9.1%
barrier:          2.2%
mio_throttle:     1.3%
```

Conclusion: the real Gemma4 global-attention shape is not HBM-bandwidth bound and is not tensor-pipe saturated. It has the same core structural gap as the synthetic profile: no async global-to-shared staging, huge shared-memory load count, large shared bank conflicts, low eligible warp rate, and register/shared-memory-limited occupancy. D512 optimization starts with async K/V staging and shared-memory layout.

### FP8 KV Baseline Column

Added FP8 KV as a third benchmark column in:

```text
benchmarks/bench_nvfp4_fmha_v2_gqa_grouped_attention.py
```

Important backend detail:

```text
FP4 KV:  fmha_v2, BF16 Q/output, NVFP4 paged KV
BF16:    fmha_v2, BF16 Q/output/KV
FP8 KV:  fa2,     BF16 Q/output, FP8 E4M3 paged KV
```

The FMHAv2 BF16-Q/FP8-KV call is not valid even though it accepts the input: a small PyTorch reference check showed finite but wrong output (`maxdiff ~= 1.0`). The correct BF16-Q/FP8-KV prefill path is `fa2`/`auto`.

Initially, `fa2` rejected Gemma4 Shape B (`D=512`) with:

```text
Invalid configuration : NUM_MMA_Q=1 NUM_MMA_D_QK=32 NUM_MMA_D_VO=32 NUM_MMA_KV=2 NUM_WARPS_Q=4 NUM_WARPS_KV=1
```

The guard came from the generic 8-bit KV register-budget heuristic:

```text
NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV)
= 1 * (8 * 32 + 2 * 4 * 2)
= 272
```

That exceeded the old generic budget of `256`. The existing D512 NVFP4 exception already raised this budget to `320` for `is_fp4_type_v<DTypeKV>`. I generalized that exception to D512 8-bit KV (`sizeof(DTypeKV_) == 1`) in both:

```text
flashinfer/data/include/flashinfer/attention/prefill.cuh
include/flashinfer/attention/prefill.cuh
```

Validation after the change:

```text
shape: BF16 Q/output, FP8 KV, D=512, group=2, q_len=4, kv_len=16
backend: fa2
maxdiff vs PyTorch reference: 0.0009765625
mean diff: 9.53e-05
```

### Gemma4 FP4/FP8/BF16 Grid

All timings are min CUDA-event milliseconds from warmup=1, repeat=3. FLOP/s uses the same attention FLOP estimate for all columns:

```text
4 * batch_size * group * q_len * kv_len * head_dim
```

Shape B, Gemma4 global attention (`D=512`, `group=8`, `q_len=512`):

| kv_len | FP4 KV ms | FP8 KV ms | BF16 ms | FP4 TFLOP/s | FP8 TFLOP/s | BF16 TFLOP/s | FP4 vs FP8 | FP4 vs BF16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 2.5240 | 1.3171 | 1.2090 | 27.23 | 52.18 | 56.84 | 0.52x | 0.48x |
| 32768 | 9.7804 | 5.0130 | 4.6111 | 28.11 | 54.83 | 59.61 | 0.51x | 0.47x |
| 131072 | 38.5375 | 19.9758 | 22.3814 | 28.53 | 55.04 | 49.13 | 0.52x | 0.58x |

Shape A, Gemma4 sliding attention (`D=256`, `group=2`, `q_len=512`, `kv_len=1024`):

| kv_len | FP4 KV ms | FP8 KV ms | BF16 ms | FP4 TFLOP/s | FP8 TFLOP/s | BF16 TFLOP/s | FP4 vs FP8 | FP4 vs BF16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.4413 | 0.0324 | 0.1657 | 2.43 | 33.12 | 6.48 | 0.07x | 0.38x |

Conclusion: FP8 KV is now a valid reference. On the priority D512 Shape B, current FP4 is consistently about half the speed of FP8 KV. At the longest sampled KV length, FP8 also beats BF16, while current FP4 remains much slower.

### FP8 Shape B Nsight Compute

Profile:

```text
reports/ncu/gemma4_shapeB_d512_g8_kv32768_fp8_fa2_prefill.ncu-rep
kernel: BatchPrefillWithPagedKVCacheKernel
shape: D=512, group=8, q_len=512, kv_len=32768
backend: fa2
```

Side-by-side with the prior FP4 Shape B profile:

| metric | FP4 KV fmha_v2 | FP8 KV fa2 |
|---|---:|---:|
| duration | 9.632 ms | 5.060 ms |
| SM throughput | 19.77% | 22.55% |
| DRAM throughput | 0.14% | 0.44% |
| L2 throughput | 22.72% | 20.34% |
| tensor pipe active | 4.06% | 12.42% |
| active warps / SMSP | 1.36 | 1.00 |
| eligible warps / SMSP | 0.22 | 0.18 |
| registers/thread | 255 | 255 |
| shared memory/block | 47,104 B | 99,360 B |
| occupancy limit by registers | 2 blocks/SM | 2 blocks/SM |
| occupancy limit by shared memory | 2 blocks/SM | 1 block/SM |
| ldgsts instructions | 0 | 4,225,024 |
| TMA load instructions | 0 | 0 |
| shared bank conflicts | 267,664,196 | 5,852,373 |
| shared wavefronts | 608,647,610 | 276,458,629 |
| ideal shared wavefronts | 185,008,640 | 117,120,512 |
| shared wavefront / ideal | 3.29x | 2.36x |

FP8 stall mix:

```text
short_scoreboard: 39.1%
wait:             28.0%
selected:         17.8%
long_scoreboard:  11.4%
barrier:           0.9%
dispatch_stall:    0.8%
lg_throttle:       0.8%
no_instructions:   0.8%
```

Comparison to FP4:

```text
FP4 had no ldgsts and large long-scoreboard/no-instruction stalls.
FP8 uses ldgsts for global-to-shared staging and cuts shared bank conflicts by ~46x.
FP8 still has no TMA, still sits at 255 regs/thread, and is shared-memory-limited to 1 block/SM.
```

Current implication: the cp.async/ldgsts hypothesis is directly confirmed on the same Gemma4 Shape B problem. Porting async global-to-shared staging into the FP4 path remains the first structural fix. The FP8 path also shows that async staging alone is not the entire story: shared-memory layout and register pressure still matter.

### FP4 LDGSTS Staging, First Patch

Changed the Blackwell NVFP4 paged K/V row-major data path to use the existing smem-tile `store(ptrs, preds_)` LDGSTS path when the load is a full 16-byte vector. The byte-tail fallback remains the old `ldg` into registers followed by `sts`.

Target cell:

```text
shape=global, D=512, group=8, q_len=512, kv_len=32768
```

Result:

| version | FP4 KV min ms | FP8 KV min ms | BF16 min ms | FP4 TFLOP/s | FP4 vs FP8 |
|---|---:|---:|---:|---:|---:|
| before LDGSTS data staging | 9.7804 | 5.0130 | 4.6111 | 28.11 | 0.51x |
| after LDGSTS data staging | 9.0762 | 5.0457 | 4.6339 | 30.29 | 0.56x |

Learning: async K/V data staging helps but is not sufficient. The target cell improved by about 7.2%, so the zero-LDGSTS issue was real, but the remaining gap is still ~4.03 ms. Next step is NCU on the patched FP4 kernel to confirm LDGSTS count and identify whether the remaining wall time is dominated by scale loads, shared-memory conflicts, synchronization, or register pressure.

Patched FP4 NCU profile:

```text
reports/ncu/gemma4_shapeB_d512_g8_kv32768_fp4_fmha_after_ldgsts_data.ncu-rep
kernel: fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_512_causal_output_bf16_kv_e2m1_gqa_m_w2x2_o128_sm120_kernel_nl_tiled
duration: 9.207 ms
ldgsts instructions: 5,212,160
TMA load instructions: 0
tensor pipe active: 4.23% elapsed, 4.84% active
active warps / SMSP: 1.39
eligible warps / SMSP: 0.26
registers/thread: 255
shared memory/block: 46.08 KiB
occupancy limits: 2 blocks/SM by registers, 2 blocks/SM by shared memory
shared bank conflicts: 267,958,533
shared wavefronts: 594,877,502
shared load conflicts: 251,569,716
shared store conflicts: 16,212,285
local memory spilling requests: 5,203,968
```

Learning: LDGSTS is now present and bank conflicts did not materially improve. The next wall-time reduction must come from the FP4 shared-memory layout/read path and register pressure. More global-load work is secondary until shared load conflicts and spilling are reduced.

Register-cap probe:

```text
FLASHINFER_EXTRA_CUDAFLAGS='--maxrregcount=192'
shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 11.8922
```

Learning: blunt register caps make this already-spilling kernel worse. This does not disprove register-pressure reduction; it only rejects `--maxrregcount` as a fix. Any register work must structurally reduce live fragment/scale/pipeline state instead of forcing extra local-memory spills.

Additional staging notes:

```text
FP4 data cp.async size: 16 bytes
implementation: fmha::ldgsts128 -> cp.async.cg.shared.global [...], [...], 16
```

Learning: the FP4 K/V data path now matches FP8 in transfer width for data. The remaining scale path is different: K scales and PV-layout V scales are loaded synchronously with LDG and stored with scalar STS/STU8-style writes. The source counters show large scalar LDG global excessive sectors and many STS.U8 conflicts, matching the scale path. Next target is vectorizing/asyncing the scale path or changing scale storage to reduce scalar shared stores and shared load wavefronts.

Rejected page-table broadcast probe:

```text
change: use __match_any_sync/__shfl_sync in get_nvfp4_row_ptrs() so lanes reading the same page share one block-table LDG
shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 9.5361
baseline after LDGSTS data staging: 9.0762
```

Learning: for this kernel, the extra match/shuffle overhead is larger than the saved page-table reads. Leave the simple direct block-offset load in place unless a lower-overhead page cache is added outside the per-row address calculation.

V smem swizzle probe:

```text
change: Smem_tile_v_blackwell_nvf4_mma COLS_PER_XOR_PATTERN 1 -> 2
shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 8.9332
baseline after LDGSTS data staging: 9.0762
```

Learning: V shared-memory layout is on-goal. A simple column XOR change gives a small but real reduction on the target cell. Continue sweeping this axis before larger layout changes.

Rejected V swizzle variant:

```text
change: Smem_tile_v_blackwell_nvf4_mma COLS_PER_XOR_PATTERN 4
shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 9.0611
```

Learning: column XOR pattern `4` gives back almost all of the gain. Keep column pattern `2` for now.

Rejected V swizzle variant:

```text
change: Smem_tile_v_blackwell_nvf4_mma ROWS_PER_XOR_PATTERN hardcoded to 8, COLS_PER_XOR_PATTERN 2
shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 9.0479
```

Learning: increasing the V row XOR period also gives back the column-pattern gain. Keep the default row pattern with column pattern `2`.

Scale/staging note:

```text
FP4 data path: 16-byte cp.async.cg.shared.global via fmha::ldgsts128
K scale path: synchronous LDG of 32-bit scale words, then STS into scale sidecar
PV-layout V scale path: synchronous LDG of uint4 scale vectors, then scalar byte STS into scale sidecar
```

Learning: data staging now uses the right 16-byte cp.async width, but scale traffic is still a separate synchronous path. The next useful layout probes must preserve the same store and load swizzle; store-only or load-only fixes can just move bank conflicts between cp.async stores, scalar scale stores, and MMA shared loads.

K smem swizzle probe:

```text
change: Smem_tile_b<Blackwell_mma_nvf4_fp32_traits> ROWS_PER_XOR_PATTERN default -> 4
shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 8.8092
previous best: 8.9332
baseline after LDGSTS data staging: 9.0762
```

Learning: K shared-memory layout is on-goal. A longer K row XOR period improves the production D512 cell by another ~1.4% over the V-swizzle-only result. Keep K row pattern `4` unless the neighboring point beats it.

K smem swizzle probe:

```text
change: Smem_tile_b<Blackwell_mma_nvf4_fp32_traits> ROWS_PER_XOR_PATTERN 4 -> 8
shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 8.7716
previous best: 8.8092
```

Learning: K row XOR `8` is the current best production D512 point. The win is small but repeatably on-goal, so keep it unless row pattern `16` beats it.

Rejected K smem swizzle variant:

```text
change: Smem_tile_b<Blackwell_mma_nvf4_fp32_traits> ROWS_PER_XOR_PATTERN 16
result: compile-time static_assert in Smem_tile_ampere_col_b
allowed row patterns in this path: 8, 4, 2, 1
```

Learning: `8` is the largest supported K row XOR period for the current Ampere-style column-B shared-memory tile. Larger row periods are not a simple tuning knob without changing that tile's constructor mapping.

Rejected V scale sidecar word-store probe:

```text
change: group four PV-layout V scale vec16 loads and store each logical column as one 32-bit scale word
shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 9.2249
previous best: 8.7716
```

Learning: replacing scalar V scale stores with grouped word stores is not enough and regresses. The grouping increases live state and register pressure, and it does not make the source load asynchronous. A valid scale fix needs to stage scales with cp.async and a sidecar layout chosen for both producer stores and consumer 32-bit loads, not just reduce store instruction count.

Pipeline-depth note:

```text
fused_multihead_flash_attention_kernel_noloop_tiled.h
Q/K main loop: cp.async.commit_group after staging, then cp.async.wait_group 1 before BMM1
V tail path: cp.async.commit_group after V staging, then cp.async.wait_group 1 before BMM1 tail
final drain: cp.async.wait_group 0
```

Learning: the current tiled FP4 path is not a deep async pipeline. It allows one outstanding group at the compute boundary and then synchronizes. The 7% LDGSTS gain is consistent with limited overlap, so the next high-leverage work is deeper K/V pipeline staging and/or the FP8-style scheduler, not more narrow swizzle sweeps.

Current-best FP4 NCU profile:

```text
reports/ncu/gemma4_shapeB_d512_g8_kv32768_fp4_fmha_ldgsts_vcol2_krow8.ncu-rep
shape=global, D=512, group=8, q_len=512, kv_len=32768
duration: 8.825 ms
ldgsts instructions: 5,212,160
TMA load instructions: 0
tensor pipe active: 4.42% elapsed, 4.85% active
active warps / SMSP: 1.37
eligible warps / SMSP: 0.26
registers/thread: 255
shared memory/block: 46.08 KiB
occupancy limits: 2 blocks/SM by registers, 2 blocks/SM by shared memory
shared bank conflicts: 266,936,845
shared wavefronts: 592,571,819
shared load conflicts: 251,527,533
shared store conflicts: 15,283,878
stall samples/ratios: wait dominant, then no-instructions, long_scoreboard, short_scoreboard, barrier
```

Learning: even after LDGSTS plus the K/V swizzle wins, FP4 is still tensor-starved. The remaining gap is not bandwidth-bound and not mainly store-side bank conflicts; the consumer is waiting on shared/global staging and issuing too few WGMMA operations. Prioritize pipeline depth, WGMMA cadence, and scheduler/tile shape over additional narrow swizzle probes.

Rejected Q/K tile-width probe:

```text
compile flag: -DFLASHINFER_FMHA_V2_NVFP4_CTA_P_TILE_K=128
shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 13.2782
current best with CTA_P_TILE_K=64: 8.7716
```

Learning: simply widening the Q/K D tile from 64 to 128 makes the production D512 cell much worse. The larger tile likely increases live state/shared-memory pressure and reduces the limited overlap the tiled loop currently has. Keep `CTA_P_TILE_K=64` unless the loop is structurally redesigned.

Rejected output tile-width probe:

```text
compile flag: -DFLASHINFER_FMHA_V2_NVFP4_SPLIT_O_TILE_N=64
shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 14.0802
current best with split O tile N=128: 8.7716
```

Learning: reducing the split output tile from 128 to 64 is a large regression. The D512 target needs the current `o128` split; smaller output tiles increase CTA work/overhead more than they relieve register/shared pressure.

Rejected trivial stage-3 pipeline probe:

```text
change: K/V BUFFERS_PER_TILE_SMEM 2 -> 3 for NVFP4 granular tiling
change: four tiled-kernel waits depbar_<USE_LDGSTS, 1>() -> depbar<USE_LDGSTS, 3>()
shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 15.0893
current best with two K/V buffers: 8.7716
```

Learning: raising buffer capacity without issuing additional future K/V groups is worse. The current loop still prefetches only one future stage, so the extra shared-memory footprint just increases pressure. A real pipeline-depth fix must change the schedule to keep multiple future groups in flight; buffer-count-only changes are rejected.

Rejected K-only stage-3 pipeline probe:

```text
change: K BUFFERS_PER_TILE_SMEM 2 -> 3 for NVFP4 granular tiling
change: BMM1 main loop primes a second future K tile, waits with depbar<USE_LDGSTS, 4>() / wait_group 2, and skips the final redundant K prefetch
shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 15.3756
current best with the retained two-buffer K/V schedule: 8.7716
```

Learning: simply keeping an additional K tile in flight is still worse in the current loop. The extra schedule state and shared-memory pressure overwhelm any additional copy overlap, and the BMM2/V side remains inline. The next pipeline attempt needs phase-level restructuring or reduced synchronization, not a one-sided deeper K prefetch.

Correctness fix for retained V swizzle:

```text
problem: global V COLS_PER_XOR_PATTERN=2 made the D256 grouped-M sharp-token test return zeros for target token 15
isolation: setting V COLS_PER_XOR_PATTERN back to 1 makes tests/attention/test_nvfp4_kv_head_dim_512.py::test_fmha_v2_grouped_m_nvfp4_paged_prefill_sharp_tokens_sm12x[256] pass
fix: use V COLS_PER_XOR_PATTERN=2 only when Cta_tile::N == 128, and keep the original pattern 1 otherwise
validation: tests/attention/test_nvfp4_kv_head_dim_512.py -- 25 passed in 72.23s
target rebench: D512 group=8 q_len=512 kv_len=32768 grouped FP4 min_ms 8.7615
```

Learning: the V swizzle win is shape-specific. The D512 split-output tile uses `Cta_tile::N == 128` and benefits from col XOR 2, but the D256 path does not preserve the same value layout under that swizzle. Keep V layout tuning keyed to the actual MMA tile shape, not as a global Blackwell-NVFP4 default.

Rejected batched WARPS_K output-reduction probe:

```text
change: batch the grouped-M output WARPS_K reduction across all VALID_MMAS_N slices to reduce per-ni __syncthreads()
shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 8.7001
current retained best after reverting probe and fixing V swizzle: 8.7615
correctness: unsafe; D256 sharp-token test failed before reverting the probe while the retained baseline was also found to need the V swizzle gate
```

Learning: the output-reduction batching is not retained. It showed a small D512 performance win, but correctness was not clean during the probe and the gain is too small relative to the remaining FP8 gap. Revisit only after the structural data pipeline work is solved, and test D256/D512 before keeping any epilogue rewrite.

Retained K scale-sidecar async staging:

```text
first attempt: 4-byte cp.async for K scale words
result: rejected at compile time; ptxas on sm120 reports cp.async size 4 as invalid and expects 16

second attempt: pad the K scale sidecar row in shared memory to 16 bytes, then use 16-byte cp.async only when the source scale group is 16-byte aligned and has 16 contiguous scale bytes available
edge handling: fall back to the existing 4-byte synchronous scale-word store for unaligned/tail K tiles
shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 8.6821
previous retained best: 8.7615
validation: tests/attention/test_nvfp4_kv_head_dim_512.py -- 25 passed in 72.48s
```

Learning: this SM120 toolchain only accepts 16-byte `cp.async` for this path. Scale-sidecar async staging therefore needs a padded/shared layout and alignment-aware fallback; a naive 4-byte scale copy is not viable. The win is real but small because only K tiles whose scale group offset is 16-byte aligned can use the async path with the current source layout.

Rejected global stage-3 pipeline probe after K-scale async:

```text
change: K/V BUFFERS_PER_TILE_SMEM 2 -> 3 for NVFP4 granular tiling
change: four tiled-kernel waits depbar_<USE_LDGSTS, 1>() -> depbar<USE_LDGSTS, 3>()
shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 15.4004
current retained best with two K/V buffers and K-scale async: 8.6821
```

Learning: the exact global stage-depth bump still regresses after K-scale async. The issue is not lack of buffer capacity by itself; the loop still waits at the same phase boundaries and pays extra shared-memory/register pressure. Future pipeline work needs a schedule rewrite that overlaps phases without growing live state blindly.

Retained post-depbar barrier removal:

```text
change: remove the four __syncthreads() calls immediately after depbar_<USE_LDGSTS, 1>() in the tiled QK main, QK tail, PV main, and PV-tail prefetch paths
unchanged: keep the read-complete __syncthreads() calls after BMM1/BMM2 consumption and the final depbar<0> tail barrier
shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 8.1075
previous retained best: 8.6821
validation: tests/attention/test_nvfp4_kv_head_dim_512.py -- 25 passed in 73.25s
```

Learning: the post-depbar barriers were stronger than required for this tiled LDGSTS path. `depbar_<..., 1>` is sufficient to make the committed async writes visible before the consumer fragment loads, while the later read-complete barriers still protect buffer reuse. This is a real phase-synchronization win, but it does not solve the remaining FP8 gap by itself.

Gemma4 production grid after barrier removal:

```text
report: reports/gemma4_nvfp4_grid_after_barrier.jsonl

Shape B/global D512 group=8:
q=1    kv=8192    FP4 2.1385 ms   FP8 0.0749 ms   BF16 1.1557 ms   FP4/FP8 28.56x slower
q=1    kv=32768   FP4 8.2635 ms   FP8 0.1401 ms   BF16 4.3947 ms   FP4/FP8 59.00x slower
q=1    kv=131072  FP4 32.7727 ms  FP8 0.3316 ms   BF16 22.4169 ms  FP4/FP8 98.84x slower
q=512  kv=8192    FP4 2.1628 ms   FP8 1.3128 ms   BF16 1.2129 ms   FP4/FP8 1.65x slower
q=512  kv=32768   FP4 8.0612 ms   FP8 5.0058 ms   BF16 4.5744 ms   FP4/FP8 1.61x slower
q=512  kv=131072  FP4 31.9756 ms  FP8 19.9007 ms  BF16 22.3018 ms  FP4/FP8 1.61x slower
q=2048 kv=8192    FP4 7.0614 ms   FP8 5.6945 ms   BF16 2.8631 ms   FP4/FP8 1.24x slower
q=2048 kv=32768   FP4 31.4172 ms  FP8 24.1105 ms  BF16 12.1487 ms  FP4/FP8 1.30x slower
q=2048 kv=131072  FP4 129.0841 ms FP8 98.9206 ms  BF16 51.4700 ms  FP4/FP8 1.30x slower

Shape A/sliding D256 group=2 kv=1024:
q=1    FP4 0.3854 ms   FP8 0.0276 ms   BF16 0.1586 ms   FP4/FP8 13.97x slower
q=512  FP4 0.4020 ms   FP8 0.0356 ms   BF16 0.1662 ms   FP4/FP8 11.28x slower
q=2048 FP4 0.3785 ms   FP8 0.0571 ms   BF16 0.1631 ms   FP4/FP8 6.63x slower
```

Learning: D512 long-prefill now has a consistent ~1.3-1.6x gap to FP8, but q=1 and D256/sliding are dominated by fixed overhead/dispatch shape mismatch. Do not judge the whole project from only D512 q512. The next work needs separate treatment for decode-like q=1 and the short-KV D256 sliding shape.

NCU after retained barrier removal:

```text
reports:
  reports/ncu/gemma4_shapeB_d512_g8_kv32768_fp4_after_barrier_sections.ncu-rep
  reports/ncu/gemma4_shapeB_d512_g8_kv32768_fp4_after_barrier_raw.ncu-rep
shape=global, D=512, group=8, q_len=512, kv_len=32768

SM Busy: 23.28%
Issue Slots Busy: 23.28%
Active Warps Per Scheduler: 1.36
Eligible Warps Per Scheduler: 0.27
No Eligible: 76.19%
Tensor pipe active: 4.88%
Shared LD instructions: 156,902,400
Shared ST instructions: 6,043,136
Shared LD wavefronts: 410,425,415
Shared ST wavefronts: 26,375,405
Shared LD bank conflicts: 252,952,647
Shared ST bank conflicts: 20,209,389
L2 hit rate: 99.73%
Shared/local spilling requests: 0
```

Learning: barrier removal improved wall time by reducing synchronization, but the post-barrier kernel is still not bandwidth-bound and still WGMMA-starved. The dominant remaining signal is shared-load inefficiency: ~2.6 shared-load wavefronts per shared-load instruction plus ~253M load-side bank conflicts. The next D512 prefill work should target the K/V shared-memory consumer layout and PV load cadence, not global-memory bandwidth.

Skip probes after retained barrier removal:

```text
report: reports/gemma4_d512_q512_kv32768_after_barrier_skip_probes.jsonl
shape=global, D=512, group=8, q_len=512, kv_len=32768

baseline:          8.1789 ms
skip_k_smem:       5.4061 ms   saves 2.7729 ms
skip_v_smem:       6.4872 ms   saves 1.6917 ms
skip_bmm1_gemm:    5.0243 ms   saves 3.1546 ms
skip_bmm2_gemm:    6.1463 ms   saves 2.0326 ms
skip_softmax_pack: 7.6715 ms   saves 0.5075 ms
skip_output:       5.7206 ms   saves 2.4583 ms
```

Learning: K/BMM1 is still the largest critical-path bucket, but output is large enough to optimize now. The direct grouped-M output path reduces across `WARPS_K` once per output-N slice with two CTA barriers per slice; batching that reduction across all N slices should be revisited now that the retained V swizzle is correctness-clean.

Rejected batched output reduction after barrier removal:

```text
change: batch direct grouped-M output WARPS_K reduction across all VALID_MMAS_N slices for each M slice
shape=global, D=512, group=8, q_len=512, kv_len=32768
D512 correctness: test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed
D512 FP4 min_ms: 8.0971
retained baseline after barrier removal: 8.1075-8.1789 ms depending on repeat set
additional cells:
  global q=2048 kv=32768 FP4 31.3819 ms
  global q=512 kv=8192 FP4 2.0964 ms
  sliding q=512 kv=1024: illegal memory access
```

Learning: output batching is not retained. It is at best noise-level on D512 and breaks the D256/sliding production shape, likely because the temporary reduction scratch layout assumed enough reusable shared memory for all valid N slices across tile shapes. The output bucket is real, but this batching strategy is not production-safe.

Rejected K shared-load lane-pair vectorization:

```text
change: load K data as two 16-byte shared-memory chunks per lane pair, then distribute words with __shfl_sync instead of four per-lane 32-bit LDS loads
shape=global, D=512, group=8, q_len=512, kv_len=32768
correctness: test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed
FP4 min_ms: 8.4442
retained baseline after barrier removal: ~8.11 ms
```

Learning: reducing K shared-load instruction count with lane-pair shuffles regresses. This matches the earlier scale-shuffle lesson: the kernel is register/issue tight enough that extra shuffles and live `uint4` state are more expensive than redundant simple LDS instructions. K/BMM1 remains the largest bucket, but it needs a layout/cadence fix, not warp-cooperative shuffling.

Rejected D256 dispatcher default-variant probe:

```text
change: force D128/D256 NVFP4 GQA causal dispatch away from the generated w2x2 32-step variant and into the default w4x1 64-step variant
shape=sliding, D=256, group=2, q_len=512, kv_len=1024
FP4 min_ms with default variant: 0.4805
retained w2x2 variant: ~0.4020
FP8 fa2 target: ~0.0312-0.0356
```

Learning: the current D256 `w2x2` variant is still better than the default variant. The D256/sliding gap is not solved by picking the existing default kernel; it needs a smaller/faster specialized short-KV path or a different backend, not dispatching to w4x1.

Decode-path q=1 check with XQA:

```text
reports:
  reports/gemma4_d512_q1_xqa_decode_nvfp4.jsonl

Shape B/global D512 group=8 batch=1:
kv=8192    NVFP4 XQA 0.0789 ms   q=1 FP8 FA2 prefill table 0.0749 ms
kv=32768   NVFP4 XQA 0.0899 ms   q=1 FP8 FA2 prefill table 0.1401 ms
kv=131072  NVFP4 XQA 0.1596 ms   q=1 FP8 FA2 prefill table 0.3316 ms

Shape A/sliding D256 group=2 batch=1 kv=1024:
NVFP4 XQA 0.0303 ms
BF16 XQA 0.0241 ms
q=1 FP8 FA2 prefill table 0.0276 ms
```

Learning: the q=1 FMHA rows are the wrong path to optimize first. Existing NVFP4 XQA decode is already near FP8 at short KV and faster at long D512 KV. Treat q=1 as decode-path work; the remaining q=1 gaps are small-kernel overhead/tuning, not the 8-32 ms FMHA prefill problem.

Rejected three-stage K/V smem pipeline probe:

```text
change:
  - set NVFP4 granular K and V smem buffers from 2 to 3
  - replace the four tiled-kernel `depbar_<USE_LDGSTS, 1>()` waits with `depbar<USE_LDGSTS, 3>()`

shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 15.0737
retained baseline after barrier removal: ~8.1 ms
```

Learning: three K/V smem buffers fit and compile, but they are much slower on the D512 production tracking cell. This is not the missing overlap win. Also, `depbar<USE_LDGSTS, 3>()` emits the same `cp.async.wait_group 1` as the previous explicit wait; the material change is the larger K/V smem ring and its changed footprint/register pressure. Do not try four stages until there is a real producer/consumer schedule that can exploit the extra buffers.

Rejected single-stage V early-preload phase-overlap probe:

```text
selected D512 split-O production variant:
  S=128, D=512, STEP=64, WARPS_M=2, WARPS_N=2
  BMM2_MAIN_MMAS_K_BOUND == 0

change:
  - preload the single V stage in the initial Q/K cp.async group
  - skip the BMM1-tail V load for single-stage BMM2
  - prefetch the next KV tile's V together with next Q/K in the BMM2 tail

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 8.9621
retained baseline after barrier removal: 8.0745
```

Learning: the active D512 split-O kernel has no BMM2 main loop, so hoisting line-646 style V loads is dead-code work for this production cell. Preloading the only V stage earlier is correctness-safe but slower, likely because the early V cp.async group interferes with the Q/K wait-group cadence. Keep the current BMM1-tail V preload schedule.

Rejected D512 split-O fragment-limiting probes:

```text
shape=global, D=512, group=8, q_len=512, kv_len=32768
retained baseline after barrier removal: 8.0745 ms

limit_qk_fragments=True, limit_v_fragments=True:
  correctness: test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed
  FP4 min_ms: 8.1670

limit_qk_fragments=True, limit_v_fragments=False:
  FP4 min_ms: 8.1757

limit_qk_fragments=False, limit_v_fragments=True:
  FP4 min_ms: 8.1517
```

Learning: fragment limiting does not recover enough register/occupancy benefit to offset the extra reload/cadence cost on the selected D512 split-O kernel. Do not retain QK, V, or combined fragment limiting for this shape.

Rejected D512 split-O `kv_loop_step=256` probe:

```text
change:
  selected D512 split-O w2x2 variant from kv_loop_step=128 to kv_loop_step=256

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 16.7022
retained baseline after barrier removal: 8.0745
```

Learning: doubling the split-O KV step is not a route to lower loop overhead. It activates a BMM2-main path but roughly doubles wall time, so the 128-token split-O step is the correct retained D512 production variant among these two choices.

Rejected K smem column-XOR probe:

```text
change:
  Blackwell NVFP4 K smem tile from row_xor=8, col_xor=1 to row_xor=8, col_xor=2

shape=global, D=512, group=8, q_len=512, kv_len=32768
FP4 min_ms: 8.2492
retained baseline after barrier removal: 8.0745
```

Learning: adding a K-side column XOR factor regresses. The retained K row period 8 with column factor 1 is better for the selected D512 split-O kernel.

Retained D256 `kv_loop_step=128` for the w2x2 grouped-M variant:

```text
change:
  D128/D256 NVFP4 GQA w2x2 variant generated with kv_loop_step=128 instead of 64

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[256] passed

Shape A/sliding, D=256, group=2, kv_len=1024:
q=1    FP4 min_ms: 0.3535   previous retained grid: 0.3854
q=512  FP4 min_ms: 0.3707   previous retained grid: 0.4020
q=2048 FP4 min_ms: 0.3385   previous retained grid: 0.3785
```

Learning: for the short-KV D256 sliding shape, a larger 128-token KV step reduces fixed loop overhead and improves every tested q length. This is a retained Shape A improvement, though it is still far from the FP8 FA2 target.

Updated retained D256 w2x2 `kv_loop_step` to 256:

```text
change:
  D128/D256 NVFP4 GQA w2x2 variant generated with kv_loop_step=256 instead of 128

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[256] passed

Shape A/sliding, D=256, group=2, kv_len=1024:
q=1    FP4 min_ms: 0.3221   previous 128-step: 0.3535   original retained grid: 0.3854
q=512  FP4 min_ms: 0.3385   previous 128-step: 0.3707   original retained grid: 0.4020
q=2048 FP4 min_ms: 0.3177   previous 128-step: 0.3385   original retained grid: 0.3785
```

Learning: 256 is better than 128 for the D256 sliding production shape. This cuts q512 wall time by about 16% from the previous retained grid, and about 8.7% over the 128-step probe.

Rejected D256 w2x2 `kv_loop_step=512`:

```text
correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[256] passed

Shape A/sliding, D=256, group=2, kv_len=1024, q_len=512:
FP4 min_ms: 0.3899
retained 256-step: 0.3385
```

Learning: 512 over-extends the D256 tile and gives back most of the loop-step win. Keep 256.

Rejected D256 w2x2 Q loop-step 64:

```text
change:
  D256 w2x2 loop_step/noloop_step from 32 to 64, with retained kv_loop_step=256

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[256] passed

Shape A/sliding, D=256, group=2, kv_len=1024, q_len=512:
FP4 min_ms: 0.4845
retained loop_step=32 / kv_loop_step=256: 0.3385
```

Learning: the D256 w2x2 Q tile should stay at 32 rows. The larger Q loop step reduces scheduling granularity but hurts kernel cadence enough to lose the KV-step win.

Retained D512 split-O threshold expansion for q_len=2048:

```text
change:
  NVFP4_D512_SPLIT_O_MAX_GROUPED_ROWS from 4096 to 16384

Shape B/global, D=512, group=8, q_len=2048:
kv=8192    FP4 min_ms: 6.2372    previous retained grid: 7.0614    FP8 FA2 target: 5.6945
kv=32768   FP4 min_ms: 26.6961   previous retained grid: 31.4172   FP8 FA2 target: 24.1105
kv=131072  FP4 min_ms: 109.3801  previous retained grid: 129.0841  FP8 FA2 target: 98.9206
```

Learning: the split-O D512 variant should stay selected through at least grouped_rows=16384. It improves every q2048 long-context production cell tested by roughly 12-15%, but it does not close the FP8 FA2 gap. The next work still needs to reduce shared-load / tensor-pipe starvation inside the selected split-O kernel, not just tune dispatch thresholds.

Correctness check after retaining D256 kv_loop_step=256 and D512 split-O threshold=16384:

```text
command:
  CUDA_HOME=/usr/local/cuda-13.0 CUDA_VISIBLE_DEVICES=2 FLASHINFER_CUDA_ARCH_LIST=12.0a \
  PYTHONPATH=/home/josh/tdm/infer/worktrees/flashinfer-nvfp4-kv \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  /home/josh/tdm/infer/current/.venv/bin/python -m pytest -q \
  tests/attention/test_nvfp4_kv_head_dim_512.py

result:
  25 passed in 99.39s
```

Current retained FMHA production grid:

```text
report:
  reports/gemma4_production_grid_retained_20260427_014547.jsonl

shape    q_len  kv_len  D    group  FP4 ms    FP8 FA2 ms  BF16 ms   FP4/FP8
global   1      8192    512  8      2.1250    0.0737      1.1526    28.83x
global   1      32768   512  8      8.2534    0.1392      4.3870    59.28x
global   1      131072  512  8      32.7293   0.3285      22.3824   99.63x
global   512    8192    512  8      2.1074    1.3095      1.2039    1.61x
global   512    32768   512  8      8.1110    4.9945      4.5736    1.62x
global   512    131072  512  8      31.9947   19.9161     22.2413   1.61x
global   2048   8192    512  8      6.1491    5.6340      2.8398    1.09x
global   2048   32768   512  8      26.6667   24.0814     12.0653   1.11x
global   2048   131072  512  8      108.8467  98.7692     51.4273   1.10x
sliding  1      1024    256  2      0.3279    0.0238      0.1577    13.77x
sliding  512    1024    256  2      0.3408    0.0308      0.1618    11.05x
sliding  2048   1024    256  2      0.3197    0.0542      0.1642    5.90x
```

Learning: for FMHA prefill, D512/q2048 is the closest remaining gap after the split-O threshold change, but D512/q512 still has a structural ~1.6x gap. D256 sliding FMHA is dominated by fixed overhead relative to FP8 FA2 and remains a separate problem. The q=1 FMHA rows are not the serving decode target because the XQA path is already the right route for q=1.

Rejected D512 non-split dispatch for q512:

```text
change:
  force NVFP4_D512_SPLIT_O_MAX_GROUPED_ROWS=0 so q512/group8 selects non-split D512

Shape B/global, D=512, group=8, q_len=512, kv_len=32768:
FP4 min_ms: 14.7768
retained split-O threshold=16384: 8.1110
```

Learning: the D512 split-O path is required for production q512/q2048 cells. The remaining gap is inside the split-O kernel, not caused by the dispatch threshold choosing split-O too broadly.

Rejected original NHD-reblocked V layout for D512 split-O:

```text
change:
  benchmark switch added: --fp4-v-layout {pv,nhd}
  run D512 split-O with nvfp4_v_cache_uses_pv_layout=False and original reblocked V cache

Shape B/global, D=512, group=8, q_len=512, kv_len=32768:
NHD-reblocked V FP4 min_ms: 35.6445
retained PV-layout V FP4 min_ms: 8.1110
```

Learning: the PV-layout V cache is mandatory for the D512 split-O path. The compatibility split variant is not a viable production route for Gemma4 D512.

Rejected D512 split-O `w4x1` warp shape with PV-layout V:

```text
change:
  dispatch the compat split-O spec first and allow it to run with nvfp4_v_cache_uses_pv_layout=True

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

Shape B/global, D=512, group=8, q_len=512, kv_len=32768:
w4x1 split-O + PV V FP4 min_ms: 14.0892
retained w2x2 split-O + PV V FP4 min_ms: 8.1110
```

Learning: the retained `w2x2` split-O warp shape is the correct existing D512 variant. The `w4x1` split path is slower even with the production PV V layout, so the next work should not be dispatching among existing D512 variants.

Rejected D512 split-O `w1x4` warp shape:

```text
change:
  add a D512 split-O spec with warps_m=1, warps_n=4 and route it to PV-layout V

result:
  compile failure in fmha/kernel_traits.h:
    static_assert(S == 0 || S % CTA_O_TILE_K == 0, "")
  instantiated with:
    S=128, VALID_D=512, STEP=64, WARPS_M=1, WARPS_N=4
```

Learning: `w1x4` is not a legal D512 split-O shape under the current CTA-O tile constraints. The simple D512 split-O warp-shape search is exhausted: retained `w2x2` wins, `w4x1` is slow, and `w1x4` does not compile.

Rejected D512 split-O 128-row Q tile:

```text
change:
  add a D512 w2x2 split-O spec with loop_step=128, noloop_step=128, kv_loop_step=128

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

Shape B/global, D=512, group=8:
q=512,  kv=32768: FP4 min_ms 12.4909   retained 64-row tile: 8.1110
q=2048, kv=32768: FP4 min_ms 36.9172   retained 64-row tile: 26.6667
```

Learning: larger D512 grouped-M tiles reduce CTA count but hurt kernel cadence/register pressure enough to lose badly. Keep the D512 split-O Q tile at 64 rows.

Rejected D512 split-O 32-row Q tile:

```text
change:
  add a D512 w2x2 split-O spec with loop_step=32, noloop_step=32, kv_loop_step=128

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

Shape B/global, D=512, group=8, q_len=512, kv_len=32768:
FP4 min_ms: 15.9180
retained 64-row tile: 8.1110
```

Learning: smaller D512 grouped-M tiles also lose badly. The current 64-row split-O tile is bracketed by 32 and 128 and remains the right Q tile height among these options.

Current FP4-vs-FP8 NCU side-by-side for the D512 q512 tracking cell:

```text
shape:
  D=512, group=8, q_len=512, kv_len=32768

reports:
  FP8: reports/ncu/gemma4_shapeB_d512_g8_kv32768_fp8_fa2_prefill.ncu-rep
  FP4: reports/ncu/gemma4_shapeB_d512_g8_kv32768_fp4_fmha_ldgsts_vcol2_krow8.ncu-rep

metric                                             FP8 FA2          FP4 FMHA
duration_ms                                        5.0605           8.8248
tensor_pipe_active_pct                             12.4185          4.4163
eligible_warps_per_scheduler                       0.1784           0.2629
active_warps_per_scheduler                         1.0002           1.3728
issue_active_per_cycle                             0.18             0.24
ldgsts_inst                                        4,225,024        5,212,160
tma_ld_inst                                        0                0
shared_bank_conflicts_total                        5,852,373        266,936,845
shared_bank_conflicts_ld                           5,867,052        251,527,533
shared_bank_conflicts_st                           43               15,283,878
shared_wavefronts_total                            276,458,629      592,571,819
shared_wavefronts_ld                               105,956,147      409,018,722
shared_wavefronts_st                               163,896          21,966,508
derived_shared_conflict_nway                       888              2,071
registers_per_thread                               255              255
dynamic_smem_kib                                   98.336           46.080
occupancy_limit_registers_blocks_per_sm            2                2
occupancy_limit_shared_mem_blocks_per_sm           1                2
grid_size_ctas                                     320              256
block_size_threads                                 128              128
```

Learning: FP4 is not slower because it lacks cp.async or uses more global bandwidth. Both paths use cp.async and neither uses TMA here. FP4 is slower because the selected split-O kernel drives far worse shared-memory load/store behavior and much lower tensor-pipe activity. The next real target is reducing FP4 shared-memory wavefronts/bank conflicts and improving WGMMA issue cadence; global-memory staging and simple tile dispatch knobs are no longer the primary lever.

V-scale sidecar skip probes:

```text
report:
  reports/gemma4_d512_q512_kv32768_v_scale_skip_probes_20260427_022840.jsonl

shape:
  D=512, group=8, q_len=512, kv_len=32768

retained baseline from current grid:
  8.1110 ms

compile flags:
  -DFLASHINFER_FMHA_V2_PROFILE_SKIP_V_SCALE_STAGING
    FP4 min_ms: 7.9498

  -DFLASHINFER_FMHA_V2_PROFILE_SKIP_V_SCALE_LOAD
    FP4 min_ms: 7.9765

  -DFLASHINFER_FMHA_V2_PROFILE_SKIP_V_SCALE_STAGING -DFLASHINFER_FMHA_V2_PROFILE_SKIP_V_SCALE_LOAD
    FP4 min_ms: 8.0170
```

Learning: V-scale sidecar staging/loads are measurable but not the main gap. Removing them buys only about 1-2%, so the 1.6x D512/q512 gap is dominated by FP4 data shared-load layout/WGMMA cadence/output, not V-scale sidecar traffic. Keep V-scale work secondary unless it falls out naturally from a broader shared-memory layout rewrite.

K-scale sidecar skip probes:

```text
report:
  reports/gemma4_d512_q512_kv32768_k_scale_skip_probes_20260427_023404.txt

shape:
  D=512, group=8, q_len=512, kv_len=32768

retained baseline from current grid:
  8.1110 ms

compile flags:
  -DFLASHINFER_FMHA_V2_PROFILE_SKIP_K_SCALE_STAGING
    FP4 min_ms: 7.3367

  -DFLASHINFER_FMHA_V2_PROFILE_SKIP_K_SCALE_LOAD
    FP4 min_ms: 8.1173

  -DFLASHINFER_FMHA_V2_PROFILE_SKIP_K_SCALE_STAGING -DFLASHINFER_FMHA_V2_PROFILE_SKIP_K_SCALE_LOAD
    FP4 min_ms: 6.9979
```

Learning: K-scale staging is a real ~10% contributor on the D512 tracking cell, but the consumer-side scale load by itself is not the main gap. This points at the producer/staging side and its interaction with the K data pipeline rather than at `make_ue4m3_scale_reg` or the scale-register consumer alone.

Rejected aligned K-scale sidecar cp.async probe:

```text
change:
  - for D512 unaligned K scale-group offsets, copy the aligned 16-byte source scale row with cp.async
  - track a sidecar shared-memory column offset so the current 4-scale tile can read from the copied 16-byte row

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

shape:
  D=512, group=8, q_len=512, kv_len=32768

FP4 min_ms: 8.4833
retained baseline: 8.1110
```

Learning: widening more K-scale tiles into 16-byte cp.async is correctness-safe but slower with the current sidecar layout. The extra offset state and larger copied sidecar footprint cost more than the synchronous 4-byte fallback saves. The rejected sidecar-offset change was reverted; keep the simpler alignment-aware K-scale async path.

Rejected single-stage BMM2 next-K/Q early-prefetch probe:

```text
change:
  - for the selected D512 split-O path where BMM2_MAIN_MMAS_K_BOUND == 0, issue the next-K/Q tile prefetch before softmax.pack instead of inside the BMM2 tail block
  - keep the existing depbar_<USE_LDGSTS, 1>() before current PV tail so current V is visible while next K/Q may remain in flight

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

shape:
  D=512, group=8, q_len=512, kv_len=32768

FP4 min_ms: 8.1133
retained baseline: 8.0948
```

Learning: moving next-K/Q prefetch earlier into the softmax-pack/PV boundary is correctness-safe but not a measurable win. The current schedule already overlaps current V with QK tail and softmax enough that this specific hoist only adds schedule/live-state noise. Deeper phase overlap needs a larger producer/consumer rewrite, not just moving the existing tail prefetch upward.

Rejected grouped-M output power-of-two row/head mapping:

```text
change:
  - in the direct Blackwell NVFP4 grouped-M output path, replace runtime row / params.num_grouped_heads and row % params.num_grouped_heads with a power-of-two shift/mask fast path
  - keep a fallback for non-power-of-two groups

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

shape:
  D=512, group=8, q_len=512, kv_len=32768

FP4 min_ms: 8.2427
retained baseline: 8.0948
```

Learning: grouped-M row/head division is not the output bottleneck in practice, or the extra branch/live state regresses the already register-tight direct output path. Do not replace the simple runtime division/modulo mapping unless a source-level profile proves it dominates.

Source-counter NCU after retained code:

```text
reports:
  reports/ncu/gemma4_shapeB_d512_g8_kv32768_fp4_retained_source_20260427.ncu-rep
  reports/ncu/gemma4_shapeB_d512_g8_kv32768_fp4_skip_v_gmem_source_20260427.ncu-rep
  reports/ncu/gemma4_shapeB_d512_g8_kv32768_fp4_skip_k_gmem_source_20260427.ncu-rep

shape:
  D=512, group=8, q_len=512, kv_len=32768

shared-memory excessive wavefront totals:
  retained:    338,478,080
  skip V gmem: 313,484,288
  skip K gmem: 269,922,816

worst retained source rows:
  LDGSTS.E.BYPASS.128 inst=1,824,256 wavefronts=29,188,096 ideal=7,297,024 excess=21,891,072 conflicts=16
  LDGSTS.E.BYPASS.128 inst=1,824,256 wavefronts=29,188,096 ideal=7,297,024 excess=21,891,072 conflicts=16
  LDGSTS.E.BYPASS.128 inst=260,608 wavefronts=8,339,456 ideal=1,042,432 excess=7,297,024 conflicts=32
  LDGSTS.E.BYPASS.128 inst=259,584 wavefronts=8,306,688 ideal=1,038,336 excess=7,268,352 conflicts=32
  STS [..+0x2000] inst=1,563,648 wavefronts=6,254,592 ideal=1,563,648 excess=4,690,944 conflicts=4
```

Learning: the remaining dominant source-level problem is cp.async/LDGSTS destination layout and the downstream shared-memory consumer layout. Skipping V staging removes only about 25M excessive wavefronts; skipping K staging removes about 68.5M, but still leaves about 270M excessive wavefronts. K staging is the bigger store-side culprit, but a K-only fix cannot close the table by itself because V and consumer LDS remain large.

Rejected direct-global K-scale consumer probe:

```text
change:
  - under FLASHINFER_FMHA_V2_NVFP4_DIRECT_K_SCALE_GLOBAL, skip K-scale sidecar staging
  - load the four K scale bytes/word directly from the global scale tensor in Smem_tile_b::load_nvfp4_scale_reg()

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

shape:
  D=512, group=8, q_len=512, kv_len=32768

FP4 min_ms: 10.7103
retained baseline: 8.1110
```

Learning: direct global scale loads are much slower than staging the K-scale sidecar, even though staging is a measured cost. The scale sidecar is the right abstraction for this kernel; the next work should fix its destination/layout cost or reduce shared-memory wavefronts, not bypass shared memory with consumer-side global loads.

Rejected K data row-XOR adjacent probes:

```text
change:
  - temporarily make the Blackwell NVFP4 K Smem_tile_b row-XOR period compile-time tunable
  - test row-XOR period 16 and 4 against the retained row-XOR period 8

row=16:
  result: compile-time rejection
  reason: Smem_tile_ampere_col_b only implements read mappings for row periods 8, 4, and 2

row=4:
  correctness: test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed
  shape: D=512, group=8, q_len=512, kv_len=32768
  FP4 min_ms: 8.1916
  retained row=8 baseline: 8.1110
```

Learning: the retained K row-XOR period 8 remains the best supported point for the D512 tracking cell. The temporary tuning hook was removed to avoid carrying a non-winning review burden.

Retained K consumer vectorized LDS:

```text
change:
  - add Smem_tile_b<Blackwell_mma_nvf4_fp32_traits>::load_k_data_word8()
  - load adjacent FP4 K 8-value words with one ld.shared.v2.b32 / uint2 instead of two separate 32-bit LDS instructions
  - use two 64-bit shared loads for the four K fragment registers in each ni/ki tile

diagnostic motivation:
  broad skip probes at D=512, group=8, q_len=512, kv_len=32768:
    retained baseline:         8.1110 ms
    skip K shared load:        5.4002 ms
    skip V shared load:        6.4713 ms
    skip K+V shared load:      3.8963 ms

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

shape:
  D=512, group=8, q_len=512, kv_len=32768

FP4 min_ms: 7.2836
previous retained baseline: 8.1110
```

Learning: K consumer LDS was one of the largest remaining buckets. Adjacent reg0/reg1 and reg2/reg3 K words are physically contiguous in the retained row-8 layout, so vectorizing them is correctness-preserving and saves about 10.2% on the D512 tracking cell. This confirms the source-counter diagnosis that shared-memory consumer behavior, not global bandwidth, is a primary limiter.

Retained V consumer word-broadcast LDS:

```text
change:
  - add Smem_tile_v_blackwell_nvf4_mma::load_v_data_col_word4()
  - for VALID_N == 128, load one 32-bit packed-column word for each same-row 8-column lane group
  - broadcast that word from lanes 0..3 to lanes with the same lane&3, replacing per-pair byte loads plus adjacent-column shuffle

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

shape:
  D=512, group=8, q_len=512, kv_len=32768

FP4 min_ms after K+V consumer changes: 6.8496
previous K-only min_ms: 7.2836
previous retained baseline before K+V consumer changes: 8.1110
```

Learning: the split-D512 V fragment mapping lets lanes with the same lane&3 consume the same K row across an 8-column group. A single 32-bit shared load plus warp broadcast replaces four byte loads for that group while preserving the packed-byte/nibble semantics. This is a real but smaller win than K vectorization, and it keeps the same shared-memory layout.

Post K+V consumer bucket probes:

```text
shape:
  D=512, group=8, q_len=512, kv_len=32768

baseline after K+V consumer changes: 6.8496 ms

skip K_SMEM_LOAD: 4.9332 ms
skip V_SMEM_LOAD: 5.2683 ms
skip BMM1_GEMM:   4.5761 ms
skip BMM2_GEMM:   5.0407 ms
skip SOFTMAX:     4.5612 ms
skip OUTPUT:      4.7909 ms
```

Learning: after vectorizing K LDS and replacing V byte-load groups with a word+broadcast path, the dominant removable time is no longer isolated to shared-memory consumers. BMM1 and softmax are now the largest broad buckets, with output and BMM2 still large. More K/V LDS work can still help, but the next major win likely has to shorten the BMM1-to-softmax-to-PV dependency chain.

Softmax sub-bucket probes after K+V consumer changes:

```text
shape:
  D=512, group=8, q_len=512, kv_len=32768

baseline after K+V consumer changes: 6.8496 ms

skip SOFTMAX_UNPACK:      4.5221 ms
skip SOFTMAX_MASK:        6.6153 ms
skip SOFTMAX_REDUCE_MAX:  6.6447 ms
skip SOFTMAX_EXP:         6.7457 ms
skip SOFTMAX_REDUCE_SUM:  6.7685 ms
skip SOFTMAX_PACK:        6.1887 ms
```

Learning: `SOFTMAX_UNPACK` removing almost the same wall time as all-softmax is not evidence that the assignment loop alone costs 2.3 ms. It breaks the dependency from BMM1 accumulators into the softmax state, so the measurement points at the BMM1-to-softmax handoff, register pressure, and probability pack path. `SOFTMAX_PACK` is a real but smaller standalone cost.

Production-shape grid after K+V consumer changes:

```text
report:
  reports/gemma4_production_grid_after_kv_consumer_20260427_035644.jsonl

shape    q_len  kv_len  D    group  FP4 ms    FP8 FA2 ms  BF16 ms   FP4/FP8
global   1      8192    512  8      1.5034    0.0728      1.1572    0.05x
global   1      32768   512  8      5.7215    0.1356      4.3821    0.02x
global   1      131072  512  8      22.7108   0.3295      22.3969   0.01x
global   512    8192    512  8      1.7767    1.3072      1.1914    0.74x
global   512    32768   512  8      6.8548    4.9928      4.5780    0.73x
global   512    131072  512  8      27.1571   19.8516     22.2565   0.73x
global   2048   8192    512  8      4.6126    5.6538      2.8609    1.23x
global   2048   32768   512  8      19.7831   24.1054     12.0851   1.22x
global   2048   131072  512  8      83.7439   98.7197     51.5625   1.18x
sliding  1      1024    256  2      0.2852    0.0230      0.1556    0.08x
sliding  512    1024    256  2      0.3067    0.0319      0.1617    0.10x
sliding  2048   1024    256  2      0.2916    0.0554      0.1620    0.19x
```

Learning: D512 global with large q_len is now ahead of FP8 FA2, but the q_len=512 D512 cells still need about 27% wall-time reduction and D256 sliding remains the largest miss. q_len=1 in this FMHA table is not the serving decode target because XQA is the intended single-token path; keep it in the table for completeness, but optimize decode through the XQA path.

NCU after K+V consumer changes:

```text
report:
  reports/ncu/gemma4_shapeB_d512_g8_q512_kv32768_fp4_after_kv_consumer_20260427_035812.ncu-rep

shape:
  D=512, group=8, q_len=512, kv_len=32768

wall in NCU run: 7.01 ms kernel duration, 7.7716 ms event timing
registers/thread: 255
shared memory/block: 50.18 KiB
occupancy limits: 2 blocks by registers, 2 blocks by shared memory
local spilling: 8,327,168 spill instructions, 4.16 MB read + 4.17 MB write
tensor pipe active: 6.78% active, 5.56% elapsed
TMA active: 0%
active warps/scheduler: 1.41
eligible warps/scheduler: 0.32
no eligible: 71.56%
long scoreboard: 0.92 cycles/issue
short scoreboard: 0.56 cycles/issue
wait: 1.52 cycles/issue
shared excessive wavefronts: 305,120,256 of 491,696,640 total
global excessive sectors: 52,776,960 of 160,939,264 total

top shared-excessive source rows:
  LDGSTS 16-way: 21,891,072 excess each, 2 rows
  LDGSTS 32-way: 7,297,024 and 7,268,352 excess
  STS sidecar:   4,690,944 excess
  LDS 2-way:     1,824,256 excess each, 8 adjacent rows
```

Learning: K/V consumer vectorization reduced wall time substantially but did not solve the structural issue. The kernel is still register-capped and spilling, with low eligible warp count and low tensor-pipe utilization. Further stage-depth changes that increase live state are unlikely to help until the BMM1-softmax-PV live range is shortened. The next optimization should reduce live fragments/register pressure or shorten the softmax handoff, while keeping the retained K/V consumer paths.

Rejected fragment live-range narrowing refactor:

```text
change:
  - move Q/K fragment arrays from function scope into each BMM1 ki loop
  - move V fragment arrays from function scope into each BMM2 ki loop

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

shape:
  D=512, group=8, q_len=512, kv_len=32768

FP4 min_ms: 6.8530
retained baseline after K+V consumer changes: 6.8496
```

Learning: simple source-level fragment scoping does not reduce the effective register allocation or wall time. The compiler already handles these fragment live ranges, or the dominant spills come from longer-lived softmax/accumulator state. Reverted to avoid a no-op review diff.

Rejected D256 sliding alternate dispatch:

```text
change:
  - temporarily force D256/group2 to skip the w2x2 causal variant
  - dispatch falls through to the default w4x1 variant

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[256] passed

shape:
  D=256, group=2, kv_len=1024

retained w2x2:
  q_len=512:  0.3067 ms
  q_len=2048: 0.2916 ms

forced w4x1:
  q_len=512:  0.3904 ms
  q_len=2048: 0.3794 ms
```

Learning: the retained D256 w2x2 route is the right dispatch choice for Gemma4 sliding attention. The D256 gap is inside the w2x2 kernel implementation, not from choosing the wrong generated variant.

Rejected V word-broadcast generalization to 256-wide V tiles:

```text
change:
  - change Smem_tile_v_blackwell_nvf4_mma::load_v_data_reg() fast path from VALID_N == 128 to VALID_N >= 128
  - intended to reuse the D512 split-O word+broadcast path for D256 sliding's 256-wide V tile

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[256] passed
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

results:
  D256/group2/q512/kv1024:   0.3443 ms vs retained 0.3067 ms
  D256/group2/q2048/kv1024:  0.3279 ms vs retained 0.2916 ms
  D512/group8/q512/kv32768:  6.9207 ms vs retained 6.8496 ms
```

Learning: the 32-bit V word+broadcast mapping is a win for the D512 split-O 128-wide tile but regresses the 256-wide tile, likely because the wider tile's ni progression/smem-read-offset toggles make the extra shuffle path worse than the scalar byte loads. Keep the fast path restricted to `VALID_N == 128` until a separate 256-wide mapping is designed.

D256 sliding q512 skip buckets:

```text
report:
  reports/d256_sliding_q512_skip_buckets_20260427_041125.jsonl

shape:
  D=256, group=2, q_len=512, kv_len=1024

baseline:       0.3053 ms
skip K_SMEM:    0.2605 ms
skip V_SMEM:    0.1841 ms
skip BMM1:      0.2646 ms
skip SOFTMAX:   0.2657 ms
skip BMM2:      0.1626 ms
skip OUTPUT:    0.1640 ms
```

Learning: D256 sliding is dominated by the PV/output side. V shared loads, BMM2, and output each remove far more wall time than K shared loads, BMM1, or softmax. This is a different profile than D512/q512. The next D256 work should target the 256-wide V consumer / PV layout / output path, not QK math.

Rejected full-tile grouped-M output reduction staging:

```text
change:
  - pre-stage all direct-output accumulator fragments for every mi/ni into shared memory
  - replace per-ni WARPS_K output reduction barriers with one upfront barrier plus smem reads during output

result:
  - compile failed for both D256 and D512 generated variants
  - static_assert hit because Mma_tile_o::MMAS_M * VALID_MMAS_N * THREADS_PER_CTA * 8 floats exceeds Kernel_traits::BYTES_PER_SMEM

recovery:
  - restored the retained per-ni reduction using one small shared-memory slice
  - correctness after restore:
      test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[256] passed
      test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed
```

Learning: batching all output-reduction fragments is not viable under the current shared-memory budget. Any output-path reduction cleanup needs a streaming or warp-local design, not full-tile accumulator staging.

Retained one-`mi` output reduction staging when it fits:

```text
change:
  - add a compile-time output reduction mode that stages one mi worth of direct-output fragments at a time
  - use this mode only when VALID_MMAS_N * THREADS_PER_CTA * 8 floats fits in the existing shared-memory budget
  - otherwise fall back to the retained per-ni WARPS_K reduction path

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[256] passed
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

results:
  D512/group8/q512/kv32768:  6.7955 ms vs retained baseline 6.8496 ms
  D256/group2/q512/kv1024:   0.3030 ms vs retained baseline 0.3067 ms
  D256/group2/q2048/kv1024:  0.2860 ms vs retained baseline 0.2916 ms
```

Learning: reducing direct-output WARPS_K barriers one mi at a time is viable and reviewable. The gain is small but consistent enough to retain. Full-tile output staging is too large; per-mi staging is the safe boundary.

D512 global q512/kv32768 current skip buckets:

```text
report:
  reports/d512_global_q512_kv32768_skip_buckets_20260427_043745.jsonl

baseline:       6.8280 ms
skip K_SMEM:    4.9247 ms
skip V_SMEM:    5.2687 ms
skip BMM1:      4.5984 ms
skip BMM2:      5.0731 ms
skip SOFTMAX:   4.5636 ms
skip OUTPUT:    4.7790 ms

V scale probes:
  skip V-scale staging: 6.7540 ms
  skip V-scale load:    6.6788 ms
```

Learning: D512 q512 is no longer dominated by one obvious sidecar-scale cost. Scale staging/load are small compared with K/V shared loads, QK/PV MMA, softmax/P-pack, and output. The next productive edits should target shared-load layout/cadence or softmax/output structure, not V scale sidecar staging.

Rejected D512 V swizzle `COLS_PER_XOR_PATTERN` 2 -> 4:

```text
change:
  - for 128-wide Blackwell NVFP4 V tiles, set COLS_PER_XOR_PATTERN_ from 2 to 4

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

result:
  D512/group8/q512/kv32768: 6.8229 ms vs retained 6.7955 ms
```

Learning: the retained D512 V swizzle column period 2 remains better than 4. Larger V column XOR period likely adds address/smem wavefront cost faster than it reduces conflicts for the current 128-wide split-O tile.

Rejected D512 V swizzle `COLS_PER_XOR_PATTERN` 2 -> 1:

```text
change:
  - for 128-wide Blackwell NVFP4 V tiles, set COLS_PER_XOR_PATTERN_ from 2 to 1

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

result:
  D512/group8/q512/kv32768: 6.8165 ms vs retained 6.7955 ms
```

Learning: adjacent V swizzle sweep is complete for the D512 128-wide split-O tile. The retained column period 2 is better than both 1 and 4 on the current kernel.

Rejected D512 split-O implicit `kv_loop_step=64`:

```text
change:
  - add the default D512 split-O spec without overriding kv_loop_step
  - this lets S=64 reach the 2x2 split-O path

result:
  compile failed:
    static_assert(S == 0 || S % CTA_O_TILE_K == 0, "")

reason:
  - for D512 split-O with WARPS_N=2, CTA_O_TILE_K is 128
  - S=64 is not divisible by 128, so the output reduction tile is not representable

decision:
  - keep D512 split-O variants at kv_loop_step=128 unless the output tile mapping is redesigned
```

Learning: D512 split-O loop-step reductions are constrained by the output-reduction tile, not just by K/V staging granularity. Do not reintroduce S=64 for this path without a real output tile redesign.

Current D512 tracking baseline after restoring retained variants:

```text
shape:
  Gemma4 global D512/group8/q512/kv32768, PV V layout

result:
  min 6.8012 ms
```

Current NCU snapshot:

```text
report:
  reports/ncu/gemma4_shapeB_d512_g8_q512_kv32768_fp4_current_20260427_050150.ncu-rep

duration: 6.86 ms
registers/thread: 255
shared/block: 50.18 KiB
grid: 256 CTAs, 0.68 waves/SM
memory throughput: 35.72%
compute throughput: 26.59%
DRAM throughput: 0.20%
SM busy: 23.82%
eligible warps/scheduler: 0.32
no eligible: 71.60%
local spill requests: 8,327,168
shared excessive wavefronts: 305,120,256 / 491,696,640
global excessive sectors: 52,776,960 / 160,939,264

top source counters:
  shared excessive:
    21,891,072  LDGSTS.E.BYPASS.128
    21,891,072  LDGSTS.E.BYPASS.128
     7,297,024  LDGSTS.E.BYPASS.128
     7,268,352  LDGSTS.E.BYPASS.128
     4,690,944  STS [scale sidecar]
  global excessive:
    43,782,144  LDG.E R2 [scalar scale sidecar path]
  stall samples:
    22,882      STS [scale sidecar]
```

Learning: after the retained K/V consumer and one-mi output reduction, the kernel is still starved by a mix of low wave count, 255-register spills, LDGSTS/store wavefront excess, and scalar scale-sidecar traffic. It is not DRAM-bandwidth-bound.

Rejected D512 split-O output tile `N=64`:

```text
change:
  - compile with FLASHINFER_FMHA_V2_NVFP4_SPLIT_O_TILE_N=64
  - intended to halve output fragments per thread and reduce spill pressure

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

result:
  D512/group8/q512/kv32768: 9.4217 ms vs retained 6.8012 ms
```

Learning: reducing D512 split-O width to 64 increases CTA count and output coordination enough to dominate any register-footprint benefit. Retain split-O `N=128`.

Rejected distributed PV V-scale staging:

```text
change:
  - keep the existing contiguous 16-byte global scale load
  - distribute the 16 sidecar shared stores across 16 lanes using shuffles
  - intended to replace one thread issuing 16 scalar STS instructions with 16 lanes issuing one STS each

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

result:
  D512/group8/q512/kv32768: 7.5560 ms vs retained 6.8012 ms
```

Learning: the added shuffles/lane coordination are worse than the scalar store unroll for PV V-scale staging. The scale sidecar is a visible NCU hotspot, but this lane-distribution strategy is not the fix.

Rejected output-reduction store skip for `warp_k == 0`:

```text
change:
  - in the WARPS_K output reduction, avoid writing warp_k 0's accumulator to shared memory
  - warp_k 0 already keeps its accumulator in registers and only reads other K partitions

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

result:
  D512/group8/q512/kv32768: 6.8997 ms vs retained 6.8012 ms
```

Learning: the extra predicate/control-flow cost is larger than the saved shared stores. Keep the unconditional reduction stores unless the reduction is redesigned more deeply.

FP8 FA2 A/B profile on the same D512 production cell:

```text
shape:
  Gemma4 global D512/group8/q512/kv32768

report:
  reports/ncu/gemma4_shapeB_d512_g8_q512_kv32768_fp8_fa2_batchprefill_20260427_051455.ncu-rep

kernel:
  BatchPrefillWithPagedKVCacheKernel

duration: 5.07 ms
memory throughput: 22.50%
compute throughput: 22.50%
DRAM throughput: 0.44%
SM busy: 14.95%
eligible warps/scheduler: 0.18
no eligible: 82.18%
local spill requests: 43,362,800
source table:
  no shared excessive rows reported
  no global excessive rows reported
```

Learning: FP8 FA2 is not faster because it has better occupancy or fewer spills; it is faster while having worse spill count. The concrete structural delta is that FP8 avoids FP4's uncoalesced shared/global access patterns. Prioritize FP4 LDGSTS destination/source layout and shared-load wavefronts over generic register-cap probes.

Rejected D512 V row-XOR period 4 -> 8:

```text
change:
  - for Blackwell NVFP4 V tiles with N=128, set ROWS_PER_XOR_PATTERN from 4 to 8
  - intended to match the retained K row period and reduce LDGSTS/LDS wavefront excess

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

result:
  D512/group8/q512/kv32768: 6.8724 ms vs retained 6.8012 ms
```

Learning: the retained D512 V row-XOR period 4 is better than 8. For the current V tile, increasing the row period hurts more than it helps, likely from worse load-side mapping despite any store-side improvement.

Rejected D512 V row-XOR period 4 -> 2:

```text
change:
  - for Blackwell NVFP4 V tiles with N=128, set ROWS_PER_XOR_PATTERN from 4 to 2

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

result:
  D512/group8/q512/kv32768: 6.8373 ms vs retained 6.8012 ms
```

Learning: D512 V row-XOR adjacent sweep is complete. The retained V layout is row period 4, column period 2; both adjacent directions regress.

Rejected K column-XOR period 1 -> 2:

```text
change:
  - temporarily parameterize Smem_tile_ampere_col_b column XOR period
  - set Blackwell NVFP4 K tile row period 8, column period 2

correctness:
  test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512] passed

short run:
  D512/group8/q512/kv32768: 6.8031 ms, effectively tied with retained 6.8012 ms

longer run:
  D512/group8/q512/kv32768: min 6.8285 ms, mean 6.8989 ms
```

Learning: K column period 2 is noise/regression. Retain K row period 8, column period 1.

Split-KV scheduler A/B on D512 production tracking cell:

```text
shape:
  Gemma4 global D512/group8/q512/kv32768

FP8 FA2 default:
  ~5.0 ms

FP8 FA2 with split-KV disabled:
  min 12.5345 ms
  mean 12.5416 ms

FP4 FMHA v2 with fixed_split_size=128 forwarded through the benchmark wrapper:
  min 6.8243 ms
  mean 6.8878 ms
```

Learning: FP8 FA2 is faster than FP4 FMHA on q512/long-KV primarily because FA2 is using its split-KV planner/merge path. When split-KV is disabled, FP8 FA2 is much slower than FP4 FMHA. Forwarding `fixed_split_size` does not materially affect FP4 FMHA because the FMHA v2 path bypasses FlashInfer's generic batch-prefill planning path and calls `trtllm_fmha_v2_prefill` directly. The next highest-leverage milestone is a real split-KV + merge path for NVFP4 FMHA v2, not another local swizzle or buffer-depth probe.

Rejected repeated pipeline-depth probe:

```text
already tested:
  - K/V BUFFERS_PER_TILE_SMEM 2 -> 3
  - four depbar_<USE_LDGSTS, 1>() waits -> depbar<USE_LDGSTS, 3>()

result:
  D512/group8/q512/kv32768 regressed to ~15.07 ms
```

Learning: the naive buffer-depth bump fits but is not a real producer/consumer pipeline. It increases K/V ring footprint and pressure without creating useful overlap. Do not retry depth 4 until the scheduler/load order is redesigned to keep independent stages in flight.

## 2026-04-27 06:31 CDT - FMHA v2 split-KV state and current blockers

Implemented and verified split-KV plumbing for the Blackwell NVFP4 FMHA v2 paged-KV path:

```text
layout:
  partial output: [q_tokens, kv_splits, qo_heads, head_dim]
  partial stats:  [q_tokens, kv_splits, qo_heads, 2]

merge:
  partial_lse = partial_stats[..., 0] + log(clamp_min(partial_stats[..., 1], 1e-30))
  merge_states(partial_out, partial_lse)

validation:
  tests/attention/test_nvfp4_kv_head_dim_512.py::test_fmha_v2_nvfp4_split_kv_partials_merge_sm12x
  tests/attention/test_nvfp4_kv_head_dim_512.py::test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512]
  result: 2 passed in 79.53s
```

Correctness trap found and fixed in the split-KV test:

```text
wrong:
  passing page indptr as cum_seq_lens_kv made the FMHA kernel see actual_kv=16 for a 256-token test.

right:
  cum_seq_lens_kv must be token cumulative lengths, e.g. [0, kv_len].
```

D512/global split-KV result on Gemma4 production shape:

```text
shape:
  D=512, group=8, q_len=512

kv=8192:
  FP4 split 48 pages: kernel-level nsys avg ~1.322 ms
  FP8 FA2:            kernel-level nsys avg ~1.288 ms
  merge overhead:     ~9-13 us class
  conclusion: FP4 is only a few percent slower at the kernel level; the remaining gap is in FMHA itself, not merge.

kv=32768:
  FP4 split 240 pages: min 4.9245 ms, median 4.9987 ms, mean 5.0236 ms
  FP8 FA2:             min 4.9989 ms, median 5.0107 ms, mean 5.0102 ms
  conclusion: FP4 now beats FP8 on min/median and is effectively tied on mean with tail noise.

kv=131072:
  FP4 split 240 pages: min 19.1778 ms, median 19.1878 ms, mean 19.6224 ms
  FP8 FA2:             min 19.8678 ms, median 19.8795 ms, mean 19.8903 ms
  conclusion: FP4 beats FP8 on long-context D512.
```

D512/q2048 global result:

```text
kv=8192:
  FP4 min 4.5317 ms, median 4.6003 ms, mean 4.5919 ms
  FP8 min 5.6042 ms, median 5.6428 ms, mean 5.6403 ms

kv=32768:
  FP4 min 18.5004 ms, median 18.6615 ms, mean 18.9740 ms
  FP8 min 24.1243 ms, median 24.1415 ms, mean 24.1777 ms

kv=131072:
  FP4 min 80.3238 ms, median 82.4351 ms, mean 82.0159 ms
  FP8 min 98.7747 ms, median 98.8609 ms, mean 98.8464 ms

conclusion:
  D512 q2048 is already a clear FP4 win across tested long-context cells.
```

D256/sliding is now the dominant production blocker:

```text
shape:
  D=256, group=2, q_len=512, kv_len=1024

best FP4 split-KV so far:
  split=32 pages: min 0.2108 ms, median 0.2136 ms, mean 0.2177 ms

FP8 FA2:
  min 0.0295 ms, median 0.0308 ms, mean 0.0326 ms

nsys:
  FP4 FMHA kernel avg ~136 us over two split CTAs
  merge_states avg ~1.3 us

conclusion:
  D256 is not a split/merge overhead issue. It is a kernel-path mismatch for the short sliding-window shape.
```

Rejected repeated pipeline-depth probe remains rejected:

```text
probe:
  K/V BUFFERS_PER_TILE_SMEM 2 -> 3
  four depbar_<USE_LDGSTS, 1>() waits -> depbar<USE_LDGSTS, 3>()

result:
  D512/group8/q512/kv32768 regressed to ~15.07 ms

reason:
  This is not a real phase-overlapped producer/consumer schedule. It adds buffer footprint and pressure without hiding the loads under independent math.

next valid pipeline work:
  redesign load ordering / phase overlap first, then increase in-flight stages.
```

## 2026-04-27 06:46 CDT - D256 sliding NCU baseline and dispatch finding

Gemma4 Shape A remains the main blocker:

```text
shape:
  D=256, group=2, q_len=512, kv_len=1024

benchmark:
  FP4 grouped split=32 pages: min 0.21168 ms, mean 0.21585 ms
  FP8 FA2:                   min 0.05414 ms, mean 0.05633 ms
  BF16:                      min 0.16429 ms, mean 0.16874 ms

conclusion:
  FP4 is slower than BF16 and about 0.26x FP8 on the production sliding shape.
```

NCU side-by-side:

```text
FP4 kernel:
  fmha_v2_flash_attention_bf16_32_256_S_q_paged_kv_256_causal_output_bf16_kv_e2m1_gqa_m_sm120_kernel_nl_tiled
  duration: 159.87 us
  grid/block: 32x1x2 CTAs, 128 threads
  regs/thread: 255
  dynamic smem/block: 51.71 KiB
  local spill requests: 212,608
  one-or-more eligible: 15.26%
  issued warp/scheduler: 0.15
  L1 shared excessive wavefronts: 1,563,264
  L2 theoretical global excessive sectors: 131,072
  instructions executed: 12,865,240

FP8 FA2 kernel:
  BatchPrefillWithPagedKVCacheKernel<..., head_dim=256, ...>
  duration: 16.06 us
  grid/block: 128x1x1 CTAs, 128 threads
  regs/thread: 251
  dynamic smem/block: 49.18 KiB
  local spill requests: 0
  one-or-more eligible: 33.82%
  issued warp/scheduler: 0.34
  L1 shared excessive wavefronts: 0
  L2 theoretical global excessive sectors: 0
  instructions executed: 4,442,880

conclusion:
  The D256 FP4 path is not bandwidth-bound. It has heavy shared/global layout excess, spills, and about 3x the instruction count of FP8.
```

Dispatch finding:

```text
generated FMHA dispatch:
  D256 NVFP4 causal -> bf16_32_256_S_q_paged_kv_256_output_bf16_kv_e2m1_gqa_m
  D256 NVFP4 non-causal fallback -> bf16_64_64_S_q_paged_kv_256_output_bf16_kv_e2m1_gqa_m

reason:
  fmha_library.py emits an extra warps_m=2, warps_n=2, loop_step=32,
  kv_loop_step=256 variant for D128/D256 and gates it on causal masks.

next probe:
  force D256 causal to use the existing 64_64 variant. If this closes the gap,
  the fix belongs in generator selection/dispatch, not in generated cache files.
```

Probe result:

```text
change:
  disabled the D256 warps_n=2 causal dispatch gate so the existing 64_64
  D256 NVFP4 kernel handled the same causal sliding shape.

result:
  FP4 grouped min: 0.25306 ms
  FP8 FA2 min:     0.05290 ms
  BF16 min:        0.16499 ms

baseline before probe:
  FP4 grouped min: ~0.2117 ms

conclusion:
  The existing 64_64 D256 path is slower than the current 32_256 causal
  specialization. Restore the dispatch. The D256 blocker is not simply the
  causal-specialization choice.
```

Split policy probe:

```text
shape:
  D=256, group=2, q_len=512, kv_len=1024

FP4 grouped:
  disable split-KV: min 0.30758 ms, median 0.31184 ms, mean 0.31454 ms
  split=32 pages:  min 0.20698 ms, median 0.21262 ms, mean 0.24256 ms
  split=64 pages:  min 0.29990 ms, median 0.30360 ms, mean 0.30516 ms
  split=128 pages: min 0.30262 ms, median 0.30622 ms, mean 0.30766 ms

conclusion:
  The short sliding-window shape still benefits from two split-KV partitions.
  The remaining D256 gap is inside the 32_256 FP4 FMHA kernel, not split/merge
  policy.
```

D256 tile-shape probe:

```text
candidate:
  16_256_w1x4
  loop_step=16, noloop_step=16, kv_loop_step=256
  warps_m=1, warps_n=4

result:
  FP4 grouped min: 0.19507 ms
  FP4 grouped mean: 0.19972 ms
  FP8 FA2 min: 0.05459 ms
  BF16 min: 0.16800 ms

baseline:
  32_256_w2x2 FP4 grouped min: ~0.207-0.212 ms

conclusion:
  Smaller grouped-M tile plus more N-parallel warps is a real D256 improvement
  and doubles CTA count to match FP8's launch shape. This is the current best
  D256 FP4 route, but it still leaves a large gap to FP8.
```

Follow-up D256 tile-shape probes:

```text
32_256_w1x4:
  result: min 0.25398 ms
  conclusion: worse than baseline. Warp layout alone does not help.

16_256_w2x2:
  result: compile failure
  error: Gmem_tile_o_16bit static_assert(ROWS % ROWS_PER_LOOP == 0)
  conclusion: M=16 requires the w1x4 output tile mapping in this path.

16_128_w1x4:
  result: min 0.21898 ms
  conclusion: smaller KV tile increases loop overhead and regresses.

16_512_w1x4:
  result: min 0.19056 ms, median 0.19414 ms, mean 0.19589 ms
  conclusion: current best. One KV tile per 512-token split partition wins.

8_512_w1x4:
  result: compile failure
  error: Gmem_tile_o_16bit static_assert(ROWS % ROWS_PER_LOOP == 0)
  conclusion: output store path does not support M=8.
```

D256 current-best correctness check:

```text
candidate:
  16_512_w1x4 first candidate for D=256 NVFP4 GQA

result:
  causal random torch comparison: passed
  non-causal sharp-token test: failed, all selected outputs stayed at token 0

cause:
  Dispatch selected the 16_512_w1x4 candidate for non-causal masks. The D128/D256 high-parallelism candidates are causal-specialized; generic non-causal should fall back to the existing 64_64 route.

fix:
  Make the SM120 NVFP4 GQA head_size in {128,256}, warps_n>1 candidates require CAUSAL, including the new D256 w1x4 candidate.
```

Correctness after causal-only dispatch fix:

```text
command:
  pytest -q \
    tests/attention/test_nvfp4_kv_head_dim_512.py::test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[256] \
    tests/attention/test_nvfp4_kv_head_dim_512.py::test_fmha_v2_grouped_m_nvfp4_paged_prefill_sharp_tokens_sm12x[256]

result:
  2 passed in 87.84s
```

D512 focused correctness after D256 dispatch fix:

```text
command:
  pytest -q \
    tests/attention/test_nvfp4_kv_head_dim_512.py::test_fmha_v2_nvfp4_split_kv_partials_merge_sm12x \
    tests/attention/test_nvfp4_kv_head_dim_512.py::test_fmha_v2_grouped_m_nvfp4_paged_prefill_matches_torch_sm12x[512]

result:
  2 passed in 0.72s
```

D256 Shape A benchmark after causal-only dispatch fix:

```text
shape:
  D=256, group=2, q_len=512, kv_len=1024, split=32 pages

result:
  FP4 grouped min: 0.1915 ms
  FP8 FA2 min:     0.0531 ms
  BF16 min:        0.1638 ms
  FP4 vs FP8:      0.28x
  FP4 vs BF16:     0.86x

conclusion:
  The corrected dispatch keeps the current-best causal production path. The
  blocker remains the D256 FP4 kernel body, not dispatch or split policy.
```

D256 split parallelism hypothesis:

```text
observation:
  Current D256 Shape A q512/kv1024 uses split=32 pages => 2 KV splits => 128 CTAs,
  below the 188 SMs on RTX PRO 6000 Blackwell.

issue:
  split < q_len/page_size was blocked by a Python guard. The mask math uses
  absolute KV offsets, so smaller splits are semantically valid, but some split
  partitions can have rows with no valid causal keys.

probe:
  Enable the existing all-masked-row softmax guard for causal kernels and allow
  split_tokens < max_q_len, then test split=16 pages.
```

D256 sub-query split probe:

```text
change under test:
  Allow causal split_tokens < q_len and enable CHECK_NEG_INF for causal kernels
  so split partitions with all-masked rows can be merged safely.

shape:
  D=256, group=2, q_len=512, kv_len=1024

result:
  split=16 pages: FP4 grouped min 0.2841 ms, mean 0.2934 ms
  split=32 pages baseline: FP4 grouped min ~0.1907 ms

conclusion:
  More split partitions increase CTA count but regress wall time. The overhead of
  extra split work/all-masked-row handling is larger than the launch-underfill
  benefit for Shape A. Keep split=32 as the D256 policy and do not relax the
  split_size >= q_len guard for now.
```

D256 sub-query split follow-up:

```text
additional result:
  split=24 pages: FP4 grouped min 0.2810 ms

final decision:
  Reverted the split-relaxation/causal-CHECK_NEG_INF probe. Keep the guard that
  requires split_tokens >= q_len for causal split-KV, because smaller splits are
  slower on the production Shape A cell.
```

D256 M=8 direct-output probe:

```text
problem:
  The 8_512_w1x4 candidate previously failed in the generic Gmem_tile_o_16bit
  static_assert, even though Blackwell NVFP4 grouped-M uses the direct output
  epilogue and does not need the generic smem-O output path.

change under test:
  Avoid instantiating/checking the generic smem-O output tile when USE_SMEM_O is
  false, move generic O tile construction below the direct-output return, and
  add an 8_512_w1x4 D256 candidate before the 16_512_w1x4 candidate.
```

D256 M=8 direct-output probe result:

```text
shape:
  D=256, group=2, q_len=512, kv_len=1024, split=32 pages

result:
  8_512_w1x4 compiled after the direct-output epilogue refactor.
  FP4 grouped min:  0.2862 ms
  FP4 grouped mean: 0.2912 ms

baseline:
  16_512_w1x4 min: ~0.191 ms

conclusion:
  M=8 doubles q-block CTA count, but the smaller M tile loses enough per-CTA
  efficiency that it regresses Shape A by roughly 50%. Reject 8_512_w1x4 and
  keep 16_512_w1x4 as the D256 candidate.
```

D256 candidate restore check:

```text
shape:
  D=256, group=2, q_len=512, kv_len=1024, split=32 pages

result after removing 8_512_w1x4:
  FP4 grouped min:  0.1929 ms
  FP4 grouped mean: 0.1972 ms

conclusion:
  Dispatch is back on the expected 16_512_w1x4 path. The direct-output epilogue
  refactor does not change the current-best D256 performance materially.
```

D256 32_512_w1x4 tile-shape probe:

```text
shape:
  D=256, group=2, q_len=512, kv_len=1024, split=32 pages

candidate:
  loop_step=32, noloop_step=32, kv_loop_step=512, warps_m=1, warps_n=4

result:
  FP4 grouped min:  0.2468 ms
  FP4 grouped mean: 0.2744 ms

baseline:
  16_512_w1x4 min: ~0.193 ms

conclusion:
  Larger M lowers CTA count and increases tail/latency enough to regress the
  short sliding shape. Reject 32_512_w1x4. The local optimum among tested D256
  tile shapes remains 16_512_w1x4.
```

Retained direct-output full-tile epilogue fast path:

```text
change:
  In the Blackwell NVFP4 direct-output epilogue, compute a per-CTA
  full_output_tile predicate and bypass the four per-lane row/column bounds
  checks when the whole output tile is valid. Tail tiles still use the checked
  path.

results:
  D256/group2/q512/kv1024 split32:
    before: FP4 grouped min 0.1929 ms, mean 0.1972 ms
    after:  FP4 grouped min 0.1915 ms, mean 0.1959 ms

  D512/group8/q512/kv32768 split240:
    before: FP4 grouped min ~4.92 ms
    after:  FP4 grouped min 4.849 ms, mean 4.944 ms

correctness:
  Focused D256/D512 tests: 4 passed.

conclusion:
  Retain. This is a small but safe epilogue improvement on both production
  shapes and does not change tail-tile semantics.
```

Current post-epilogue comparison:

```text
D256 Shape A/sliding, group=2, q=512, kv=1024, split=32:
  FP4 grouped min: 0.1930 ms
  FP8 FA2 min:     0.0538 ms
  BF16 min:        0.1671 ms
  FP4 vs FP8:      0.28x

D512 Shape B/global, group=8, q=512, kv=32768, split=240:
  FP4 grouped min: 4.8608 ms
  FP8 FA2 min:     4.9436 ms
  BF16 min:        4.6527 ms
  FP4 vs FP8:      1.02x

conclusion:
  D512 q512/kv32k now narrowly beats FP8. The dominant failing production cell
  remains D256 sliding; it needs a different short-KV strategy, not more D512
  long-context tuning.
```

D256 q2048 Shape A check:

```text
shape:
  D=256, group=2, q_len=2048, kv_len=1024, split=128 pages

result:
  FP4 grouped min: 0.2630 ms
  FP8 FA2 min:     0.0849 ms
  BF16 min:        0.1602 ms
  FP4 vs FP8:      0.32x
  FP4 vs BF16:     0.61x

conclusion:
  The D256/sliding miss is not just q512 underfill. It persists and worsens
  against BF16 at q2048. The current FP4 D256 short-KV path is structurally
  inefficient relative to FA2, likely in the PV/output path already identified
  by skip buckets.
```

Rejected D256 16_512_w4x1 warp-decomposition probe:

```text
hypothesis:
  The 16_512_w1x4 path pays a shared-memory output reduction across four
  K-reduction warps. Try the same 16x512 tile with warps_m=4, warps_n=1 to
  remove that reduction and see whether output cost is the primary blocker.

shape:
  D=256, group=2, q_len=512, kv_len=1024, split=32 pages

result:
  16_512_w4x1 FP4 grouped min:  0.2812 ms
  16_512_w4x1 FP4 grouped mean: 0.3078 ms

baseline:
  16_512_w1x4 FP4 grouped min: ~0.193 ms

conclusion:
  K/PV parallelism from warps_n=4 is more valuable than avoiding the output
  reduction. Reject w4x1 and keep w1x4.

note:
  The JIT object name does not encode warp decomposition, so w4x1 cannot be
  generated alongside w1x4 for the same M/N tile without a naming change. The
  probe was run as a replacement, then reverted.
```

Rejected D256 16_512_w1x2 warp-decomposition probe:

```text
hypothesis:
  Try the middle point between w1x4 and w4x1: two K-reduction warps should
  reduce output-reduction pressure while preserving more K/PV parallelism than
  w4x1.

shape:
  D=256, group=2, q_len=512, kv_len=1024, split=32 pages

result:
  16_512_w1x2 FP4 grouped min:  0.2084 ms
  16_512_w1x2 FP4 grouped mean: 0.2146 ms

baseline:
  16_512_w1x4 FP4 grouped min: ~0.193 ms

conclusion:
  w1x2 is better than w4x1 but still slower than w1x4. Keep w1x4; the D256
  gap is not solved by lowering the K-reduction warp count.
```

Rejected D256 16_512_w2x4 warp-decomposition probe:

```text
hypothesis:
  Keep four K/PV warps like w1x4, but add M parallelism to see whether extra
  row-side work improves the short-KV D256 cell.

implementation note:
  The first compile failed because Warp_masks had no specialization for
  (WARPS_M=2, WARPS_N=1, WARPS_K=4). Adding the canonical mask let it compile,
  but that specialization is not retained because the candidate is rejected.

shape:
  D=256, group=2, q_len=512, kv_len=1024, split=32 pages

result:
  16_512_w2x4 FP4 grouped min:  0.2132 ms
  16_512_w2x4 FP4 grouped mean: 0.2171 ms

baseline:
  16_512_w1x4 FP4 grouped min: ~0.193 ms

conclusion:
  Extra M parallelism increases CTA cost/register/layout pressure more than it
  helps. Keep 16_512_w1x4.
```
