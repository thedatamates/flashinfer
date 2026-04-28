# SM120 NVFP4 Prototype vs SM100 FlashInfer FMHA Mainloop

Date: 2026-04-28

Goal: force the next SM120 NVFP4 fused-attention implementation step to be
driven by the working SM100 FlashInfer/CUTLASS mainloop structure, not by
empirical tuning of the current benchmark kernel.

Primary references:

- `include/flashinfer/attention/blackwell/fmha_cutlass_sm100.cuh`
- `include/flashinfer/attention/blackwell/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp`
- `include/flashinfer/attention/blackwell/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`
- `include/flashinfer/attention/blackwell/collective/sm100_fmha_load_tma_warpspecialized.hpp`
- `include/flashinfer/attention/blackwell/collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp`
- `benchmarks/sm120_nvfp4_cutlass_fused_attention.cu`

## Executive Diff

The current SM120 benchmark kernel has adopted one SM100 idea, register-resident
Q, but it has not adopted the SM100 ownership model that makes that idea safe
and fast. SM100 uses a loader role, independent Q/K/V pipelines, MMA-to-softmax
pipelines, softmax-to-correction pipelines, correction-to-epilogue pipelines,
and a persistent tile scheduler. The current SM120 path is a monolithic helper
sequence inside one benchmark kernel.

The failed TMA-Q attempt on 2026-04-28 confirms the mismatch: reusing the QK
pipeline storage to stage Q separately deadlocked. In SM100, `PipelineQ` has its
own shared storage and lifetime. That is not optional.

## Side-By-Side Structural Diff

| Area | SM100 Reference | Current SM120 Prototype | Structural Difference | Porting Consequence |
| --- | --- | --- | --- | --- |
| Entry point | `fmha_cutlass_sm100.cuh` builds a `Mainloop`, `Epilogue`, and `FMHA<Sm100FmhaFwdKernelTmaWarpspecialized<...>>` operation. See `fmha_cutlass_sm100.cuh:60-68`. | A benchmark-only monolithic CUDA kernel `persistent_mainloop_owner_group_online_register_q_stage_kernel`. See `benchmarks/sm120_nvfp4_cutlass_fused_attention.cu:2593-2605`. | SM100 is a production device/kernel/collective stack. SM120 is still a benchmark extension kernel. | Performance work can continue in the benchmark, but the shape must mirror production kernel/collective boundaries if it is expected to become shippable. |
| Scheduler | All active roles iterate `tile_scheduler.is_valid()` and call `tile_scheduler.get_block_coord()`. See `sm100_fmha_fwd_kernel_tma_warpspecialized.hpp:184`, `369-395`, `400-423`, `440-460`, `467-489`, `495-515`. | One CTA handles explicit `q_tile`, `kv_tile_start`, `num_kv_tiles`, and `out_group_idx`. See `sm120_nvfp4_cutlass_fused_attention.cu:2602-2605`, `2843-2850`. | SM100 is persistent over scheduled tiles; SM120 is a single manually parameterized tile owner. | Do not tune split policies as a substitute for a scheduler. A shippable SM120 path needs a scheduler loop or an explicit decision that fixed-shape launch is only a perf-ceiling lab. |
| Warp roles | SM100 maps 16 warps into `Softmax0`, `Softmax1`, `Correction`, `MMA`, `Load`, `Epilogue`, and `Empty`. See `sm100_fmha_fwd_kernel_tma_warpspecialized.hpp:48-74`. | SM120 uses one producer warpgroup and two consumer warpgroups. See `sm120_nvfp4_cutlass_fused_attention.cu:2626-2639`. | SM120 collapses load, MMA, softmax, correction, and epilogue responsibilities into too few roles. | The current role model cannot reproduce SM100 overlap. Next port should define roles first, then assign pipelines. |
| Register allocation by role | SM100 explicitly sets or deallocates registers per role: softmax gets 192 regs, correction gets fewer, other roles get fewer. See `sm100_fmha_fwd_kernel_tma_warpspecialized.hpp:67-71`, `369-370`, `397-398`, `433-434`, `462-463`, `491-492`, `524-525`. | SM120 has one `__launch_bounds__(384, 1)` kernel with mixed responsibilities and no role-specific register donation. See `sm120_nvfp4_cutlass_fused_attention.cu:2593`. | SM100 budgets registers according to role; SM120 makes every active thread carry a mixed live range. | Register pressure fixes should be structural, not `maxrregcount`; split roles reduce live ranges. |
| Shared storage | SM100 has a union of mainloop/epilogue tensor storage plus separate pipeline storage for `load_q`, `load_k`, `load_v`, `mma_s0`, `mma_s1`, `s0_corr`, `s1_corr`, `mma_corr`, `corr_epi`, and softmax ordering. See `sm100_fmha_fwd_kernel_tma_warpspecialized.hpp:104-125`. | SM120 storage has one QK `SharedStorage`, one PV pipeline storage, P/scales/stats arrays, and aliases the QK storage as PV shared storage. See `sm120_nvfp4_cutlass_fused_attention.cu:198-216`, `2839-2841`. | SM120 reuses storage that SM100 keeps independent at the pipeline level. | The Q TMA deadlock came from violating this. Add independent pipeline storages before trying Q TMA again. |
| TMA descriptor ownership | SM100 loader role prefetches mainloop TMA descriptors; epilogue role prefetches store descriptors. See `sm100_fmha_fwd_kernel_tma_warpspecialized.hpp:190-196`. | SM120 producer warp prefetches B/SFB and PV descriptors after manual Q staging. See `sm120_nvfp4_cutlass_fused_attention.cu:2770-2777`. | SM120 descriptor prefetch is late and partial; Q/SFA descriptors are absent in the active path. | Loader role should own Q/K/V and scale descriptor prefetch. |
| Load collective | SM100 factors loading into `Sm100FmhaLoadTmaWarpspecialized`, which owns TMA atoms and layout construction. See `sm100_fmha_load_tma_warpspecialized.hpp:46-93`, `95-124`. | SM120 manually wires loads inside the benchmark kernel and helper lambdas. See `sm120_nvfp4_cutlass_fused_attention.cu:2697-2768`, `2851-2876`. | SM120 does not have a load collective. | Create an SM120 NVFP4 load collective before adding more mainloop features. |
| Q load mapping | SM100 computes Q TMA through `mma_qk.partition_A(gQ)` and `tma_partition(... group_modes<0,3>(sQ), group_modes<0,3>(tSgQ_qdl))`. See `sm100_fmha_load_tma_warpspecialized.hpp:171-184`. | SM120 writes Q by scalar unpacking packed bytes into `qk_tAsA_prod` from consumer threads. See `sm120_nvfp4_cutlass_fused_attention.cu:2697-2736`. | SM120 manually reconstructs producer layout instead of deriving it from the MMA/TMA partition. | The next Q port must use partition-derived TMA writes, not scalar layout indexing. |
| Q pipeline | SM100 uses dedicated `PipelineQ` with `StageCountQ = 2`, independent shared storage, and `TransactionBytesLoadQ`. See `sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:66`, `123-125`, `151-153`; kernel construction at `sm100_fmha_fwd_kernel_tma_warpspecialized.hpp:200-211`. | SM120 has no `PipelineQ`; Q is staged before QK/PV pipelines are constructed. See `sm120_nvfp4_cutlass_fused_attention.cu:2715-2768`, `2779-2824`. | SM120 treats Q as pre-loop setup instead of a pipeline-owned operand. | Independent `PipelineQ` is load-bearing. Do not reuse QK pipeline storage for Q staging. |
| Q residency | SM100 MMA waits for Q1/Q2, creates `tSrQ0` and `tSrQ1`, uses them throughout the KV loop, and releases Q only after the loop. See `sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:294-327`, `447-455`. | SM120 copies two Q fragments into `q_frag0` and `q_frag1` before the loop, then passes them into a helper. See `sm120_nvfp4_cutlass_fused_attention.cu:2657-2666`, `2766-2768`, `2878-2888`. | The high-level residency idea matches, but the producer and lifetime management do not. | Keep register-resident Q, but change its producer and pipeline lifetime to match SM100. |
| K/V streaming order | SM100 loader issues `Q1, K1, Q2, V1, K2, V2, ...`. See `sm100_fmha_load_tma_warpspecialized.hpp:157-159`, `197-260`. | SM120 loops per KV tile: load K chunk 0, QK, load K chunk 1, QK, write scores, softmax/P quant, then load V in the PV helper. See `sm120_nvfp4_cutlass_fused_attention.cu:2843-2949`. | SM120 separates QK and PV phases; SM100 interleaves K and V around softmax/P readiness. | Port should introduce separate K and V pipelines before optimizing tile sizes. |
| K/V pipelines | SM100 has independent `PipelineK` and `PipelineV`, with independent transaction bytes and states. See `sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:127-132`, `154-158`; kernel setup at `sm100_fmha_fwd_kernel_tma_warpspecialized.hpp:213-232`. | SM120 has one QK pipeline for K/SFB and one PV pipeline for V/SFB, constructed separately and sequenced by helper calls. See `sm120_nvfp4_cutlass_fused_attention.cu:2779-2824`. | SM120 has two pipelines, but not one shared loader role coordinating K/V issue order. | Move K/V issue into one loader-owned load collective. |
| QK output handoff | SM100 QK MMA writes S into TMEM and signals either `pipeline_mma_s0` or `pipeline_mma_s1`. See `sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:306-312`, `335-341`, `380-420`, `477-481`. | SM120 writes QK accumulators into a shared-memory `scores` buffer, then block-synchronizes. See `sm120_nvfp4_cutlass_fused_attention.cu:2890-2902`. | SM100 has an async handoff from MMA to softmax; SM120 materializes scores and synchronizes. | On SM120 there is no TMEM, but the port still needs an explicit MMA-to-softmax pipeline rather than a block-wide phase barrier. |
| Softmax role | SM100 uses two dedicated softmax warpgroups and `OrderBarrierSoftmax`. See kernel roles at `sm100_fmha_fwd_kernel_tma_warpspecialized.hpp:48-59`, softmax call at `389-395`, implementation at `sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:488-736`. | SM120 calls `softmax_quant_scores_128` in-line after QK score materialization. See `sm120_nvfp4_cutlass_fused_attention.cu:2904-2906`. | SM120 serializes softmax into the main owner flow; it does not overlap with MMA or correction. | Implement a softmax role/pipeline even if the storage is SMEM/registers instead of TMEM. |
| P representation | SM100 softmax converts S to P-like storage in TMEM using a store path that overlaps with S/O allocations. See `sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:508-516`, `587-633`. | SM120 quantizes P to `owner_storage.p_packed` and `owner_storage.p_scales` in shared memory. See `sm120_nvfp4_cutlass_fused_attention.cu:2904-2906`, `2932-2941`. | SM120 needs NVFP4 P plus scales, which SM100 BF16/FP16 does not. | Keep SM120-specific P quantization, but drive its readiness through a pipeline, not a full phase barrier. |
| PV issue | SM100 alternates PV MMA against P0/P1 and V tiles inside the MMA role, committing correction signals between them. See `sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:347-370`, `391-444`, `457-481`. | SM120 calls `cutlass_smem_pv_k128_tma_v_group_accum_persistent_step` after softmax/P staging. See `sm120_nvfp4_cutlass_fused_attention.cu:2942-2949`. | SM120 PV is helper-call-per-tile, not mainloop-interleaved. | Replace helper-call sequencing with a mainloop state machine once load/softmax roles exist. |
| Correction/rescale | SM100 correction role consumes softmax stats and O pipeline, rescales prior O tiles when row max changes, then forwards to epilogue. See `sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:907-1098`. | SM120 updates `global_m/global_l/old_scale/tile_scale` in shared memory, scales/clears `pv_accum`, and applies final normalization at output. See `sm120_nvfp4_cutlass_fused_attention.cu:2908-2930`, `2942-2945`, `2959-2973`. | SM120 performs correction in the same owner flow instead of a separate pipelined role. | The math can stay, but the ownership should move into a correction stage if the goal is SM100-style overlap. |
| Epilogue | SM100 epilogue waits on `PipelineE` and stores output with TMA store, including LSE handling. See `sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp:113-177`. | SM120 writes float `out_group` directly from consumer threads. See `sm120_nvfp4_cutlass_fused_attention.cu:2959-2973`. | SM120 benchmark output is not a production epilogue. | End-to-end shipping will need a BF16/FP16 output epilogue and optional LSE contract; benchmark float output is only a lab artifact. |
| SM100-only machinery | SM100 uses TMEM, UMMA/TCGEN, and SM100-specific schedules. See `sm100_fmha_fwd_kernel_tma_warpspecialized.hpp:102`, `315-318`, and TMEM usage throughout `sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:264-288`, `488-683`. | SM120 uses `mma.sync.aligned.kind::mxf4nvf4.block_scale...` through CUTLASS block-scaled SM120 GEMM atoms and has no TMEM. | Direct code copy is impossible. | Port the structure, not TMEM instructions: roles, pipeline topology, partition-derived TMA loads, and interleaving order. |
| NVFP4 scale sidecars | SM100 BF16/FP16 path has no NVFP4 SFA/SFB sidecar. | SM120 QK and PV operands require packed FP4 data plus UE4M3 scale tensors. See SM120 collective setup in `sm120_nvfp4_cutlass_fused_attention.cu:2641-2695`, K/SFB copy at `2851-2876`. | SM120 has additional producer/consumer payloads and transaction bytes. | SM120 load collective must treat data and scale sidecars as one transaction per operand pipeline. |

## What The Failed TMA-Q Attempt Proved

On 2026-04-28, two variants were tried and reverted from the active path:

1. A Q/SFA-only TMA staging path using `tma_transaction_bytes_mk`.
2. A safer staging path using `qk_collective.load(...)` to load A/B/SFA/SFB
   through the existing CUTLASS collective transaction convention.

Both compiled. Both deadlocked at runtime on a two-tile smoke.

Interpretation:

- The problem is not only transaction byte accounting.
- Reusing `owner_storage.qk.pipeline_storage` for a temporary Q staging
  pipeline before reconstructing the K pipeline violates the reference lifetime
  model.
- SM100's independent `PipelineQ::SharedStorage` and independent producer /
  consumer states are necessary, not optional cleanup.

The active C++ path has been restored to the last known-good scalar-Q
register-resident prototype.

## Porting Order From This Diff

The next code change should not tune the current monolithic kernel. The next
code change should introduce the SM120 analogue of the SM100 load/mainloop
structure in the smallest compileable slice:

1. Define an SM120 NVFP4 load collective modeled on
   `Sm100FmhaLoadTmaWarpspecialized`, but with block-scaled operands and scale
   sidecars:
   - Q data + Q SFA through a dedicated `PipelineQ`.
   - K data + K SFB through a dedicated `PipelineK`.
   - V data + V SFB through a dedicated `PipelineV`.
2. Give Q/K/V independent pipeline storage in shared memory. Do not alias Q
   staging through the existing QK pipeline.
3. Port only the load role and MMA role first:
   - Load role issues `Q1, K1, Q2, V1, K2, V2, ...`.
   - MMA role waits `PipelineQ`, holds Q fragments/views across the KV loop,
     waits `PipelineK`/`PipelineV`, and produces score/PV handoff signals.
4. After the Q/K/V lifetime matches SM100, add the SM120 softmax/P-quant role.
5. After softmax/P-quant has a pipeline contract, add correction/rescale and
   epilogue.

Do not add another helper-call-per-tile path unless it directly implements one
of these structural steps.
