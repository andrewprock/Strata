# PP speed-up record (gfx1030 / V620)

Recorded 2026-11-23 on branch `v620-gfx1030-tuned` (generation 6: 0.2.20-dev + F-2 + GDN env).
The prompt-processing campaign is mature - records set, the working stack adopted, the rest measured
and closed - so this is the consolidated state-of-PP document, not an open plan like
`tg-speedup-plan.md` (its decode-side companion). Full details of every item live in the migration
doc's sections 4-6 and 10; this file is the map.

## Hardware constraints

**RDNA2 has no matrix cores.** Every WMMA kernel in the tree (Aurora experts, the GDN chunked scan,
the upmix WMMA path) is `__gfx11xx__`/`__gfx12xx__` guarded - gfx1030 has one accelerant, the byte
dot `v_dot4_i32_i8` (dp4a), plus packed-FP16 VOP3P ALU ops. INT8 compute runs at quarter rate (= FP32),
so packed FP16 is the only 2x path. Upstream already absorbed the dp4a lesson (`i8_expert.cuh:27`:
three-operand `v_dot4_i32_i8` with src2 = 0 on gfx1030-1032) - the idea from our llamacpp-mmq patches
survives, the patches do not (stranded pre-0.2.16).

## Where the PP time goes (phase shares, 11-19, STRATA_PREFILL_TIMING=1)

Stable across 16k/32k/64k/128k, shares of the GPU timeline:

| Phase family | Share | Note |
|---|---|---|
| Expert GEMM (i8_expert tiles) | 26-28% | biggest single phase; a 10% kernel win = only ~2.5-3% overall PP |
| GDN family (recurrence + out proj + conv/gates) | 28-29% | as large as the experts; gfx1030 runs the non-WMMA scalar path - now the most interesting research target |
| QSA attention family (attn + proj + select) | ~26% | pure SIMT |
| HC family (write/read/up/mix/inject) | ~12% | F-2 lives here |
| Everything else | <= ~4% each | router+shared, combine, host grouping, dequant, ple, embed |
| Host staging | +2-3% on top | 4,102 ms on 133,410 ms wall at 128k |

## The five loops that matter

1. **MoE expert tile GEMM** (`engine/src/kernels/cuda/i8_expert.cuh`): 64 rows x 16 tokens per block,
   K in 128-value stages, one shared-memory decode per tile, 8 dp4a + integer scale algebra per 32
   values, exact integer arithmetic. Tuned on a 6700 XT, not a 72-CU V620 - the main retune target.
2. **Dense projection GEMMs** (`engine/src/prefill/gemm.cu` -> rocBLAS): FP16-in/FP32-accumulate
   (default-on gfx103x). This is F-2's loop. Mostly harvested; opt-in own SGEMM
   (`STRATA_RDNA2_SGEMM=1`, not bit-identical, never tested here) remains.
3. **Expert GEMV + PLE, decode** (`native_mmvq.cu`, `ple.cu`): bandwidth-bound; cache hit rate,
   PCIe and PLE batching decide it - see the TG plan doc.
4. **GDN recurrence** (`native_gdn.cu`): gfx1030 runs the direct scalar form (chunked WMMA scan is
   gfx11+). Latency-shaped, fixed per-layer cost; the quad path env halved it (adopted, gen 6).
5. **Attention + HC GEMVs** (`qsa*.cu`, `gr.cu`, `kv_q8.cu`): pure SIMT; warp-per-row lesson
   documented in gr.cu's header (first version cost 262 ms/token).

## Adopted (what the gen-6 stack is made of)

┌────────────────────────────────┬──────────┬────────────────────────────────────────────────────────┬──────────────────────┐
│ What                           │ When     │ Result                                                 │ Status               │
├────────────────────────────────┼──────────┼────────────────────────────────────────────────────────┼──────────────────────┤
│ ROCm tuning table (upstream PR │ 10-14    │ Reading +1.2-2.7% every tier, writing unchanged        │ in gen-6 stack       │
├────────────────────────────────┼──────────┼────────────────────────────────────────────────────────┼──────────────────────┤
│ SIMT1 scorer (upstream PR #115 │ 10-14    │ 64k +2.7%, 128k +8.6% reading speed                    │ in gen-6 stack       │
├────────────────────────────────┼──────────┼────────────────────────────────────────────────────────┼──────────────────────┤
│ DPP wave-reduce (a small kerne │ 10-14    │ Reading +0.3-1.0%, bit-identical; now upstream         │ in gen-6 stack       │
├────────────────────────────────┼──────────┼────────────────────────────────────────────────────────┼──────────────────────┤
│ Batched expert reading (PLE)   │ 10-19    │ Reading +11.6-14.0% every tier                         │ in gen-6 stack       │
├────────────────────────────────┼──────────┼────────────────────────────────────────────────────────┼──────────────────────┤
│ 5-token drafting + window      │ 10-15    │ Writing +6.2-8.5%, reading flat                        │ in gen-6 stack       │
├────────────────────────────────┼──────────┼────────────────────────────────────────────────────────┼──────────────────────┤
│ F-2 (skip-widen, STRATA_HC_MIX │ 10-24    │ -3.2% GPU timeline at 33k, bit-identical; +1.0-2.2% on │ in gen-6 stack       │
├────────────────────────────────┼──────────┼────────────────────────────────────────────────────────┼──────────────────────┤
│ CPU offload + persistent hc    │ 11-03    │ TTFT -12% short prompts; writing +2.3-4.8%             │ in gen-6 stack       │
├────────────────────────────────┼──────────┼────────────────────────────────────────────────────────┼──────────────────────┤
│ STRATA_HIP_PROMPT_F16 (FP16 pr │ 0.2.16   │ Default-on for gfx103x (11-14% faster upstream)        │ default in gen-6     │
├────────────────────────────────┼──────────┼────────────────────────────────────────────────────────┼──────────────────────┤
│ GDN quad path (STRATA_GDN_HEAD │ 11-19/23 │ +4.1-5.2% PP every tier on 0.2.20; bit-exact           │ in gen-6 stack       │
├────────────────────────────────┼──────────┼────────────────────────────────────────────────────────┼──────────────────────┤
│ Patched expert kernels (MMQ/VO │ 11-04    │ Reading +0.9-1.4% every tier on pre-0.2.16 bases       │ retired 11-23 (no up │
├────────────────────────────────┼──────────┼────────────────────────────────────────────────────────┼──────────────────────┤
│ Strata base bumps (0.2.10/0.2. │ 11-05..2 │ +0.4% then +3.2-3.3%; 0.2.20 parity with F-2/GDN re-te │ gen 6 = 0.2.20-dev   │
└────────────────────────────────┴──────────┴────────────────────────────────────────────────────────┴──────────────────────┘
## Rejected (do not re-test without new evidence)

┌──────────────────────────────────┬────────────────────────────────────────────────────────┐
│ What (PP-side)                   │ Why, in one line                                       │
├──────────────────────────────────┼────────────────────────────────────────────────────────┤
│ SIMT2 scorer                     │ loses reading speed on every tier, -9.9% at 128k       │
├──────────────────────────────────┼────────────────────────────────────────────────────────┤
│ rocBLAS self-tune on this card   │ 56/112 shapes pick differently but tier deltas +/-0.2% │
├──────────────────────────────────┼────────────────────────────────────────────────────────┤
│ DOT2 attention score path        │ 1.9x faster micro-bench, 1.48x SLOWER in-engine (fixed │
├──────────────────────────────────┼────────────────────────────────────────────────────────┤
│ reduce12 warp_sum port           │ 1.71x slower on this chip                              │
├──────────────────────────────────┼────────────────────────────────────────────────────────┤
│ fused_gr/gr kernel extension     │ only 2 of 19 call sites convert bit-identically        │
├──────────────────────────────────┼────────────────────────────────────────────────────────┤
│ gr_upmix GEMM port               │ the real waste was the widen step, not the GEMM - beca │
├──────────────────────────────────┼────────────────────────────────────────────────────────┤
│ k-quant experts (Q4_K_XL/Q5/Q6/Q │ no GPU kernels exist for k-quant experts; engine hard- │
├──────────────────────────────────┼────────────────────────────────────────────────────────┤
│ INT8 compute conversions         │ INT8 runs at quarter rate = same as FP32; packed FP16  │
├──────────────────────────────────┼────────────────────────────────────────────────────────┤
│ GDN out-norm decouple (F-3)      │ kernel timings moved, end-to-end tok/s did not; hash c │
└──────────────────────────────────┴────────────────────────────────────────────────────────┘
Full one-line verdicts (including the TG-side and serving-side rejections): migration doc section 4.2.

## Open PP threads

- **I8 expert tile retune for the V620** (72 CU, 32 GB) - the tile shape and VGPR budget were tuned
  on a 6700 XT; upstream measured 1.10-1.35x over MMQ on a 6700 XT, our card was never the target.
- **GDN family on gfx1030** - now the largest non-expert phase; the quad path is harvested, the
  out-norm decouple is rejected, but the scalar recurrence itself is un-researched territory.
- **STRATA_RDNA2_SGEMM=1** - opt-in own dense SGEMM, author-side +56% on an RX 6800, not
  bit-identical, never measured on this box.
- **Hugepages/memlock** - listed in the TG plan (idea 7) but it helps PP's expert reads too.

## Records

Current generation-6 tier medians (PP/TG): 1061.6/103.6, 1054.8/99.6, 1048.4/94.7, 1037.8/93.7 -
first configuration over 1000 PP at every tier. Full history: migration doc section 5.
