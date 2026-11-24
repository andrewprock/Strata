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
│ 5  │ Sweep --spec 5 -> 6/7/8 and --spec-min-p 0.70 (chosen before │ 0-5%         │ triv       │ REJECTED 11-24 │
├────┼──────────────────────────────────────────────────────────────┼──────────────┼────────────┼──────────────┤
│ 6  │ Fuse the GDN layer's 6-kernel chain: .ab into the Q8Segs mul │ +2-4%        │ med        │ not started  │
├────┼──────────────────────────────────────────────────────────────┼──────────────┼────────────┼──────────────┤
│ 7  │ Hugepages + memlock raise for the 46 GiB expert arena (vm.nr │ +1-3%        │ sudo       │ not started  │
├────┼──────────────────────────────────────────────────────────────┼──────────────┼────────────┼──────────────┤
│ 8  │ Sweep the CPU expert share, round 2: 0.05/0.10 + auto/0.55  │ 0-3%         │ triv       │ CLOSED 11-24   │
├────┼──────────────────────────────────────────────────────────────┼──────────────┼────────────┼──────────────┤
│ 9  │ Fuse attention-layer tail (qsa_select -> attn_part -> attn_f │ +1-2%        │ med        │ not started  │
├────┼──────────────────────────────────────────────────────────────┼──────────────┼────────────┼──────────────┤
│ 10 │ Second V620 (NCARD=2 layer split): cache capacity -> ~99% hi │ +20-40%      │ $$         │ not started  │
└────┴──────────────────────────────────────────────────────────────┴──────────────┴────────────┴──────────────┘

## Order of attack

1. Config sweeps first (5, 8, then 1's reserve/telemetry levers) — an afternoon of tier benches.
2. Hugepages + memlock boot (7) — one interactive sudo, zero code.
3. The dense re-pack trio (2, 3, 4) — one `tools/iq_pack.py` rebuild can test all three; verify with
   the parity harness before adopting.
4. Kernel fusions (6, 9) — real engine work, bit-exactness care on summation order.

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
