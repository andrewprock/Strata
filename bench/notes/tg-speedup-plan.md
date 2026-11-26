# TG speed-up plan (gfx1030 / V620)

Recorded 2026-11-23 on branch `v620-gfx1030-tuned` (generation 6: 0.2.20-dev + F-2 + GDN env).
This is the working document for the decode-side speed-up campaign; the migration doc's section 6.39
holds the one-line record and points here. Each idea gets a Status update and its verdict appended
below as it is measured, same discipline as the migration doc's experiments.

## The per-token budget

Read from `engine/src/split/src/{engine.cu, moe.cuh, dense.cuh, qdot.cuh, iqdot_hip.cuh, model.cpp}`,
the boot log, and the pack layout. Per generated token the GPU moves roughly:

| Stream | Bytes/token | Source |
|---|---|---|
| Routed experts (48 L x 10 x 2.05 MB) | ~0.98 GB | 23.4 GiB VRAM cache, 89.7% hit; misses read ~2 MB blobs from the 46 GiB page-locked arena over PCIe |
| Forced-Q8_0 dense (qkvz, ssm_out, wo, hc down/up, shared experts) | ~0.85 GB | `dense.bin`, 1.43 GiB total, VRAM, every token |
| Output head (2560 x 248320, Q8_0) | 0.68 GB per window | amortized over the ~4 accepted rows of a spec-5 window |
| KV (int8), GDN state, draft model (Q2_0, 802 MiB) | small | VRAM |

Total ~2 GB/token; at TG ~100 tok/s that is ~200 GB/s effective against the V620's ~672 GB/s peak:
30%. The gap is expert-cache miss stalls, the ~400-kernel-per-token launch chain (explicit streams,
no CUDA graphs), and latency-bound small kernels. TG is not noise here; there is headroom.

## The ten ideas, best first

┌────┬──────────────────────────────────────────────────────────────┬──────────────┬────────────┬──────────────┐
│ #  │ Idea                                                         │ Expect       │ Cost       │ Status       │
├────┼──────────────────────────────────────────────────────────────┼──────────────┼────────────┼──────────────┤
│ 1  │ Expert-cache misses: rebuild expert-profile.bin on the real  │ +3-7%        │ cfg        │ not started  │
├────┼──────────────────────────────────────────────────────────────┼──────────────┼────────────┼──────────────┤
│ 2  │ Re-pack forced-Q8_0 dense below Q8_0 (qkvz, ssm_out, wo, hc, │ +5-10%       │ med        │ not started  │
├────┼──────────────────────────────────────────────────────────────┼──────────────┼────────────┼──────────────┤
│ 3  │ Output head Q8_0 -> Q6_K or IQ4_XS (0.68 GB per window, 44%  │ +2-4%        │ lo-med     │ not started  │
├────┼──────────────────────────────────────────────────────────────┼──────────────┼────────────┼──────────────┤
│ 4  │ Router logits F32 -> BF16 (ffn_gate_inp forced F32, 252 MB/t │ +1-2%        │ low        │ not started  │
├────┼──────────────────────────────────────────────────────────────┼──────────────┼────────────┼──────────────┤
│ 5  │ Sweep --spec 5 -> 6/7/8 and --spec-min-p 0.70 (chosen before │ 0-5%         │ triv       │ REJECT 11-24 │
├────┼──────────────────────────────────────────────────────────────┼──────────────┼────────────┼──────────────┤
│ 6  │ Fuse the GDN layer's 6-kernel chain: .ab into the Q8Segs mul │ +2-4%        │ med        │ CLOSED 11-25 │
├────┼──────────────────────────────────────────────────────────────┼──────────────┼────────────┼──────────────┤
│ 7  │ Hugepages + memlock raise for the 46 GiB expert arena (vm.nr │ +1-3%        │ sudo       │ CLOSED 11-25 │
├────┼──────────────────────────────────────────────────────────────┼──────────────┼────────────┼──────────────┤
│ 8  │ Sweep the CPU expert share, round 2: 0.05/0.10 + auto/0.55   │ 0-3%         │ triv       │ CLOSED 11-24 │
├────┼──────────────────────────────────────────────────────────────┼──────────────┼────────────┼──────────────┤
│ 9  │ Fuse attention-layer tail (qsa_select -> attn_part -> attn_f │ +1-2%        │ med        │ CLOSED 11-25 │
├────┼──────────────────────────────────────────────────────────────┼──────────────┼────────────┼──────────────┤
│ 10 │ Second V620 (NCARD=2 layer split): cache capacity -> ~99% hi │ +20-40%      │ $$         │ not started  │
└────┴──────────────────────────────────────────────────────────────┴──────────────┴────────────┴──────────────┘

## Order of attack

1. Config sweeps first (5, 8, then 1's reserve/telemetry levers) — an afternoon of tier benches.
2. Hugepages + memlock boot (7) — CLOSED 11-25: already delivered by the THP fallback (see
   Verdicts); no sudo needed.
3. The dense re-pack trio (2, 3, 4) — one `tools/iq_pack.py` rebuild can test all three; verify with
   the parity harness before adopting.
4. Kernel fusions (6, 9) — CLOSED 11-25 before any kernel work: the decode-timing probe bounds
   everything they could touch at ~0.4 ms of a 27.8 ms window (see Verdicts).

## Measurement discipline

TG drift on this box is +/-3-4% same-binary (proven repeatedly; see migration doc section 5). Any
idea claiming less than ~3% needs med-of-5 or interleaved arms, not the usual med-of-3. PP is the
reliable +/-0.3-0.5% signal and several ideas move PP too (7, 10) — watch both.

## Verdicts (append as measured)

**Idea 5 (--spec sweep) — REJECTED (11-24).** Screening at 32k/128k, soak + med-of-3 per arm,
TIERS=32k,128k, spec-5 baseline from the same day. spec 6: TG -6.8% at 32k, -2.2% at 128k.
spec 8: -5.1% / -0.2%. Deeper windows lose: acceptance decays past depth ~4-5 (per-window logs
showed 69-90% while spec 5 ran), and every rejected drafted row costs compute without payback.
spec 5 stays. spec-min-p untuned (a --spec-min-p sweep below 0.70 with deeper drafts is the only
remaining shape here, not scheduled).

**Idea 8 (--pcie-frac sweep) — NEUTRAL, no adoption (11-24).** Same screening protocol.
pcie-frac 0.05: TG -0.3% at 32k, +1.1% at 128k. pcie-frac 0.10: -0.5% / +0.9%. The 128k lean is
inside the +/-3-4% same-binary TG drift band; PP drifted -1.3 to -1.7% at 32k across all four arms
(warm-box order effect), which does not change the TG read. CPU-computed misses are as good as PCIe
reads at a 10.3% miss rate — revisit only if idea 1 raises the miss rate or the second card lands.

Screening numbers (PP/TG, med-of-3):

| Arm | 32k | 128k |
|---|---|---|
| baseline (spec 5, pcie 0.00, same day) | 1054.8 / 99.6 | 1037.8 / 93.7 |
| spec 6 | 1040.3 / 92.5 | 1038.1 / 91.6 |
| spec 8 | 1036.6 / 94.2 | 1039.4 / 93.5 |
| pcie-frac 0.05 | 1036.6 / 99.0 | 1035.2 / 94.7 |
| pcie-frac 0.10 | 1040.7 / 98.8 | 1038.2 / 94.5 |

**Idea 8, round 2 (auto / 0.55) — CLOSED, 0.00 stays (11-24).** The first round only probed the
bottom of the range; the engine's design-intent operating points are higher: with `--pcie-frac`
omitted the native-pack default is 0.55 DMA, and `STRATA_PCIE_FRAC_DEFAULT=auto` probes the link
in the first decode windows. Both arms ran the same screening protocol. The probe itself measured
**27.9 GB/s host->device** vs the CPU pool's effective **78.5 GB/s** on missed experts and revised
0.55 down to 0.15; the bench says even 0.15 loses. Numbers (PP/TG, med-of-3): pfauto (0.15)
32k 1041.8/95.0, 128k 1039.7/91.9; pfd55 (0.55) 32k 1037.2/97.1, 128k 1036.4/93.3 — vs baseline
32k 1054.8/99.1-99.6, 128k 1037.8/93.7. Reading: DMA misses contend with other PCIe traffic in a
way the probe's bandwidth model does not capture, and the CPU computes a missed expert ~3x faster
than the link can fetch it on this box. The calibration-era 0.00 (6.20) is quantified right, not
folklore. Provenance: pcie-frac 0.35 lost on the 0.1.47 stack too (same sweep that adopted 0.00).
Provisional pass files: strata_tier_pass_20261124_{164401,164650,164939,165545,165835,170125}.

**Idea 9 (attention-tail fusion) — CLOSED 11-25, measured before any kernel work.** Two false starts
first: the decode-fusion edits went into src/split (the opt-in multi-card engine, built OFF) and then
into src/qwen35/q36_engine.cu (which only runs qwen35moe files - this model is qwen4exp and decodes
in src/core/layer.cpp via gemv_quantized). Both were dead code for the daily driver; the q36
experiment was reverted before commit, the split-tree edits were left uncommitted, then reverted on 11-25 after review (never built, unused).

The real-engine read also changed the fusion's shape: the 12 attention (QSA) layers carry separate
attn_q/k/v in MIXED types (Q4_K / IQ3_XS / Q3_K_L per layer, equal on 5 of 12), so a row-concat fuse
is type-gated, and the chain it would trim (q/k/v MMVQ + prep + quantize + wo) turned out to sit in a
slice the engine's own probe measures directly. STRATA_DECODE_TIMING=1, 384-token decode, 208 windows,
27.82 ms/window: GPU-reach wait 20.54 ms, of which VRAM-hit expert MMVs are 20.38 ms - i.e. the
entire non-MoE GPU work (attention, GDN, norms, every projection) is ~0.2 ms per window, and host
submit is ~0.18 ms (plan 0.08 + actq 0.09 + jobs 0.01). The fusion classes (wq/wk/wv combine,
quantize fold into the epilogue, per-row prep merge) and the GDN chain fusion (idea 6) can only attack
that ~0.4 ms: an upper bound of ~1.4% of the window if they deleted it outright, below the +/-3-4% TG
noise. Both close. The same snapshot sizes the remaining levers: experts 20.4 ms (ideas 2/3/4
requant, model changes needing sign-off), draft 2.43 ms (spec shape, tuned), CPU-expert misses 1.18
ms at a 93.6% hit rate (idea 1), commit/emit 0.57, adapt wait 0.50, host 1.77.

**Idea 7 (hugepages + memlock) — CLOSED 11-25, already delivered by the THP fallback.** The engine
tries MAP_HUGETLB first, and on this box it fails by design (vm.nr_hugepages=0; the boot log asks for
23983 2 MiB pages) - then falls back to a 2 MiB-aligned anonymous mapping plus MADV_HUGEPAGE. THP on
this host is madvise mode, so the request lands: the live engine's 47,965 MiB arena region reads
AnonHugePages 47,964 MiB - 99.998% huge-page coverage today, the same 2 MiB granularity a 24,000-page
hugetlbfs reservation would provide. The memlock half was moot all along: the process page-locks the
46.8 GiB arena with an 8 MiB RLIMIT_MEMLOCK (ROCm's cudaHostRegister path does not enforce it here).
What explicit hugepages would still buy: immovable pages only (THP can theoretically split/move under
pressure; pinned.cu cites a gfx11 bug where an in-arena huge-page move reset the GPU mid-prompt -
never observed on gfx1030, and the arena is allocated once at boot into free RAM). If that insurance
is ever wanted: stop the server, `sudo sysctl -w vm.nr_hugepages=24000`, start the server - the boot
log will say hugetlb 2 MiB pages instead of the MADV_HUGEPAGE line - at the cost of ~47 GB of
ordinary RAM. Note for any future A/B: STRATA_NO_LARGEPAGES=1 suppresses both the hugetlb try and the
MADV_HUGEPAGE request, so under THP=madvise it gives an approximate (khugepaged-permeable) 4 KB arm,
not a guaranteed one.

## Idea 11 (2026-11-25): EXPDEC on gfx1030 - screened and closed, banner result only

The daily's routed-expert kernels (`native_expert_grouped`, iq_kernels.cu) run the default path on this
card: `STRATA_EXPDEC=1` is only in the gfx12 defaults table (arch_defaults.cpp), and the native pack's
experts are IQ2_S gate/up (type 22) + IQ4_NL down (type 20), layers 0/1 IQ3_XXS(18)+Q2_0(42) and
IQ2_S(22)+Q2_0(42) (packs/iq3_s/native_experts.txt). `expdec_bench` on the V620, real shape, the daily
pair 22/20 single T=4: def 268.1 us / 294 GB/s vs dec 237.1 us / 333 GB/s (-11.6%); 18/20 flat
(258.5 vs 257.9, b2 -7%). EXPDEC `r2` loses on the V620 (gu rows per lane 2 measured 20% slower than
R 1 at real shape). The kernels run at ~45% of the ~672 GB/s peak with per-expert cost a ~6.1 us vs a
~3.1 us pure-DRAM floor, so the micro-bench still leaves headroom, but a real-engine A/B did not show
it: `ab_engine.py` ref vs EXPDEC:STRATA_EXPDEC=1,STRATA_EXPDEC_D20=1, 3 interleaved rounds, the daily
config (logs/ab-expdec.log, ab-expdec.jsonl): decode tok/s within noise, no arm ahead per round and no
per-request arm ahead on the long prompt. Closed: not worth pursuing on gfx1030. What the screen DID
establish: the 20.38 ms expert slice in the idea-9 timing snapshot is ~2/3 real kernel time
(~13-16 ms by micro-bench arithmetic: ~45 hit groups x ~6 us + entries, x48 layers) and ~1/3
bubbles/copies - a kernel-level fix needs a different shape than EXPDEC's, and the remaining ceiling on
one card is the ~2x between a=6.1 us and the DRAM floor.

## Idea 12 (2026-11-25): the AMD grouped-expert layouts (STRATA_EXP_MODE 1/2) on gfx1030 - screened and closed

The gfx906-era wave64 layouts in iq_kernels.cu build for every HIP arch but default to mode 0
("the CUDA layout") everywhere but gfx906; the V620 is also wave64, so the W64 mode (1) was the
one untested free knob on the biggest slice. expdec_bench, real shapes, the daily's dominant pair
22/20 (46 of 48 layers), server stopped, 20 reps x 5 rounds, output parity "identical" in all runs
(no env, and STRATA_EXP_MODE=1 and =2):
  no env  def: T=1 82.3 us (239.5 GB/s), T=2 147.2 us (267.9); dec 132.8 (296.9)
  MODE=1: T=1 90.3 us (218.3), T=2 169.2 us (233.1) - ~10-15% SLOWER than mode 0 at every shape
  MODE=2: 16/20 T=1 68.0 vs 66.2, T=2 124.0 vs 114.6 (+8%) - also slower
With the env set every arm in the bench collapses to one timing (the env overrides the per-choice
mode), so the comparison is mode-X vs mode-0 def, and mode 0 wins both times. The gfx906-tuned
shapes do not transfer to the V620's memory system; the default layout is the best-known config
for the card's expert kernels. Closed: STRATA_EXP_MODE stays unset. Together with idea 11 this
says the remaining ~2x to the DRAM floor is not reachable by any existing alternative layout -
only a new kernel shape (a wave64 Q8_0 DPP GEMV is still unexplored for the dense/head slice,
and row_dot64's 64-lane __shfl_xor butterfly compiles under HIP, so a hand-tuned variant is
possible, just not via env).

## Idea 3 sizing (2026-11-26): the daily's Q8_0 dense/head slice on the V620 - measured, build target confirmed

`q8_gemv_bench` (registered, built), server stopped, V620 (sm_103, L2 4096 KB), 500-rep medians, cold weights:
  head 2560->248320 (675 MB): 430.6 GB/s (w=1) 421.5 (2) 418.0 (3) 410.9 (4) 361.1 (6) - 1.57-1.87 ms per window
  gdn qkv 2560->10240 (27.9 MB): 374.9-388.3 GB/s at w=1-2, 338.5 at w=6
  gdn gate/out (16.7 MB): 306-370 GB/s; qsa q 2560->12288: 384-405; shared expert + kv (1.4-3.5 MB): 108-305 (latency-bound)
  EVERY shape loses 10-25% GB/s from w=1 to w=6 - the verify width is exactly where the loss lands.
Both existing multi-row knobs are inapplicable to the daily: STRATA_B6_MMVQ_ROWS is the K-quant family
(old IQ3_XXS-era pack; b6_mmvq_bench A-vs-B on the V620: B ~= A, mixed +-5%), and STRATA_Q80_ROWS1 is
native_mmvq's sm_120 Q8_0. The daily's Q8_0 routes quantize_q8_0 + s2_gemv_q8, which has NO rows mode.
Accounting: the idea-9 20.38 ms "VRAM-hit expert MMVs" bucket includes the dense/head GEMVs; split
~12-14 ms routed-expert kernels + ~6-8 ms dense/head. Dense+head runs at 46-64% of the ~672 GB/s peak
vs the 83-84% the gfx12 DPP kernels reach, so a wave64-tuned s2_gemv_q8 (multi-col reuse of the
activation fragment, head included) is worth ~2-3 ms/window ~= +7-11% TG - the biggest measured
single-card lever so far. Next: build it (Idea 3 proper).

## Idea 3, stage 1 (2026-11-26): the dense Q8_0 one-row layout (STRATA_NO_MMVQ_ROWS2=1) - measured, adopted-candidate

The sizing pass found the daily's dense Q8_0 projections (attention, SSM, shared expert, head) route through
native_mmvq, whose layouts are per-card: gfx1201 and sm_86 measured the ONE-row-per-block layout faster (73-79% ->
83-84% of peak on gfx1201) and default to it; other cards, including gfx1030, keep two rows "until measured".
STRATA_NO_MMVQ_ROWS2=1 forces one row everywhere and is bitwise exact (each row keeps its own partials and
reduction). Measured today, server stopped (build-hip/q8_gemv_bench --bw 672, 50 reps, cold weights):
one-row wins at nearly every shape and column count: GDN qkv/gate/out -2..-6%, QSA q -2..-4%, QSA k/v -8..-15%,
HC down 10240->320 -24..-32%, head -2..-3% at 2-4 columns (+4% at 6), shared gate/up +10% at 2 columns only
(5 us scale). Real-engine A/B (ab_engine.py, 3 interleaved rounds, logs/ab-rows1.{log,jsonl}): decode medians
+1.3/+1.6/+0.4/+0.3% (chat0/1/2/long), and both arms produced IDENTICAL token counts on every prompt - the
bitwise-exact claim holds at the engine level. Adopt candidate: set STRATA_NO_MMVQ_ROWS2=1 in the daily config.
Still open after this: a B6-style Q8_0 multi-column kernel (activation fragment reused across rows; B6X covers
only the K-quants) plus wave64 tuning - the remaining gap is the 48-60% of peak the best layout still leaves
against gfx12's 83-84%, which sizes the full idea-3 build at ~+5-10% TG.

Full-tier A/B of the same change (llm-benchmark/bench_strata_tiers.py, TIERS=16k..128k, 4 counterbalanced
boots alternating arm order, 2 counted passes each, med-of-4; logs /tmp/ft-*, ~/engines/logs/ftier-sweep.log):
TG ref -> rows1: 16k 101.2 -> 102.6 (+1.3%), 32k 98.8 -> 97.5 (-1.4%), 64k 95.2 -> 99.9 (+4.9%),
128k 93.5 -> 96.0 (+2.6%). PP dead even at every tier (+-0.1% or better; the reduced strata_bench run's
-1% PP was harness artifact). Absolute levels match the canonical tier-sweep record (PP ~985-1043,
TG ~93-101). Verdict: TG +1.3..+4.9% at three tiers, wash at 32k, PP-neutral, bit-exact outputs -
the layout stays adopted. Supersedes the reduced strata_bench 4K/16K/32K probe run in between.

## Idea 3, stage 2 (planned 2026-11-26): B6X-style Q8_0 multi-column kernel + wave64 tuning - ENGINE-ONLY build

Scope: kernel source + bench + env flag. No pack rebuild, no re-quantization, no model data, no quality
change. All work in engine/src/kernels/cuda/s2_gemv.cu (+ dispatch in layer.cpp gemv_quantized), opt-in
behind an env flag so the default path stays byte-identical until measured (the STRATA_NO_MMVQ_ROWS2
pattern). Bitwise parity (memcmp, exit 1 on diff, mmvf_rows_parity/native_dense_parity templates) is the
acceptance test at every step.

The build itself: the daily's dense/head Q8_0 dot (s2_gemv_q8) re-reads the activation blocks once per
output row; at verify width w=5-7 that is w-fold redundant activation traffic. Two levers:
1. a B6X-style multi-column Q8_0 kernel - activation block loaded once, reused across w rows
   (B6X exists only for the K-quant family today; the Q8_0 path has no rows mode at all);
2. wave64 tuning - DPP row reductions and v_dot4 int8 intrinsics (all present on RDNA2;
   row_dot64's 64-lane __shfl_xor butterfly already compiles under HIP).

Phases (bench-first):
1. Screen cheap variants in q8_gemv_bench (dp4a dot vs per-quad dequant, column-paired loads);
   cold weights, GB/s per shape/width, memcmp exact. Decides the direction before kernel writing.
2. Write the multi-column kernel (w up to 8), bitwise parity vs s2_gemv_q8.
3. Real-engine A/B on the full tier harness (bench_strata_tiers.py, counterbalanced boots), env-flag arms.
4. Adopt only on a measured win: gfx1030 entry in arch_defaults, documented with numbers.

Target: dense+head ~6-8 ms/window of the ~14 ms TG window at 46-64% of peak vs gfx12's demonstrated
83-84% - closing even half is ~+5-10% TG, stacking on stage 1's +1-5%. The ceiling is extrapolated from
another card's kernel (gfx12), not measured here - hence phase 1 is a screen, not a commitment.
Cost: days-scale. Status: NOT STARTED; phase 1 next.
