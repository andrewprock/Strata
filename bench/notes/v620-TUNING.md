# v620-TUNING.md — gfx1030 research tasks on 0.2.16

Started 2026-11-19, the day after the 0.2.16 pull (see `strata-migration.md` §5.2, §6.35, §10).
This file is the working plan for kernel- and pipeline-level tuning on the Radeon Pro V620
(gfx1030, RDNA2, 72 CUs, 32 GB). One section per research task: what runs there today, why it
matters, what to try, and how to measure it.

## Baseline (what we measure against)

Arms and rows live in `strata-migration.md`. The short version:

- Daily arm: `~/git/wt-0216-f2` (0.2.16 tip + F-2, commit `f786675dc`), config
  `~/strata-v620-iq3_s-T9-0216-f2.json`, IQ3_S pack, med-of-3.
- 11-19 med-of-3 reading (PP tok/s): 998.2 / 1004.2 / 1010.0 / 990.9 at 16k/32k/64k/128k.
  Writing 95.0 / 93.1 / 92.2 / 91.2.
- 0.2.16 vanilla same-day: 980.0 / 985.3 / 991.9 / 973.5 — F-2 is +1.8–1.9%.
- PP noise floor on this rig: ±0.3%. Anything smaller is a wash; report it as one.

## Measured prefill phase shares (the ranking that decides priority)

One tier pass with `STRATA_PREFILL_TIMING=1` on the F-2 build, 11-19 (§10 of the migration
notes). Shares were stable across the 16k–128k tiers, so the ranking does not depend on
context length:

| Family | Share of GPU timeline | Files |
|---|---|---|
| GDN family (recurrence, out proj, conv+gates, base) | ~29% | `engine/src/kernels/cuda/native_gdn.cu`, `fused_gdn.cu`, `gdn.cu` |
| Expert GEMM ("gemm gate/up") | 26–28% | `engine/src/kernels/cuda/i8_expert.cuh` |
| QSA attention family (attn, proj, select) | ~26% | `engine/src/kernels/cuda/qsa.cu`, `qsa_prompt_attn.cu`, `qsa_decode_attn.cu`, `kv_q8.cu` |
| HC family (write + read parts) | ~12% | `engine/src/kernels/cuda/gr.cu`, `fused_gr.cu` |
| Everything else + host staging | ~3–5% | `ple.cu`, router, combine |

Absolute 128k numbers: GPU timeline 133,301 ms, host staging 4,102 ms (3%), wall 133,410 ms
at 1,001.6 tok/s — the timer added no measurable overhead.

Reading of the table: the three big families are nearly equal. No single inner loop is more
than ~29% of prefill, so no single-loop win can buy more than a fraction of that. The GDN
family is both the largest AND the one running a known-lesser algorithm on gfx1030 (below),
which is why it leads the list.

Ground rule from AGENTS.md, repeated here: every claim below that has no measurement behind
it is marked **[unmeasured]**. The point of the tasks is to replace those marks with numbers.

## How to measure anything in this file

- Server: `cd ~/git/Strata && setsid nohup .venv/bin/python serve/server.py --engine strata
  --config <cfg>.json --port 8080 > /tmp/boot.log 2>&1 &`. **Gotcha learned 11-19: the
  `ready:` banner goes to server.py stdout, not the engine log.** Poll `/tmp/boot.log` or
  curl `/v1/status` (401 = up, 200 with key = ready). Boot takes ~5 min (experts from page
  cache read fast, the rest is fixed).
- Tier harness: `STRATA_API_KEY=<key> STRATA_MODEL=<model_name> python3
  ~/git/llm-benchmark/bench_strata_tiers.py` — one invocation = one pass over all four tiers.
- Prefill phase shares: set `STRATA_PREFILL_TIMING=1` in the config's env map; the breakdown
  lands in the engine log as `strata prefill timing: ...` lines.
- Parity: the byte-identical smoke test is retired (§8.8 — greedy decode is not reproducible
  within one boot, on any build). Use the in-server checks instead
  (`STRATA_HC_UPMIX_CHECK=2` reports rel RMS / max |diff| per layer) or a dedicated parity
  bench from `engine/tests/hip/` and `engine/src/kernels/*_parity.cpp`.
- Repo microbenches that already exist: `upmix_gemm_bench`, `expdec_bench`,
  `b6_mmvq_bench`, `batch_rows_parity`, `bf16_gemv_parity`, `decode_cluster_parity`,
  `iq_parity`, `pf_wmma_parity`. Prefer extending these over writing new harnesses.
- Builds: `-S engine -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1030
  -DCMAKE_BUILD_TYPE=Release`, ROCm 7.15 (`/opt/rocm/bin/hipcc`). Worktrees `~/git/wt-0216`
  (vanilla) and `~/git/wt-0216-f2` (F-2). New work goes on a branch off `v620-0216-f2`.

---

## R1 — GDN on gfx1030 (~29% of prefill) — top priority

### R1a feasibility note (11-19, code read) — the answer is better than expected: no port needed

The original question was "can the gfx11 WMMA chunked scan map to RDNA2?" The code read says that
was the wrong question. The Aurora block in `engine/src/prefill/kernels.cu` (from line 1409,
`#if defined(__HIPCC__)` — **AMD builds only, no gfx11 arch guard on these kernels**) contains
`gdn_rec_quad_kernel`, `gdn_rec_quad_pp_kernel` and `gdn_rec_quad2c_kernel`: the recurrence with
**four lanes per column**, each lane owning 32 state rows in registers, q/k/v/gate/beta staged in
LDS four tokens at a time (double-buffered, next chunk prefetched into registers while the current
one computes), the four per-column partials meeting through wave shuffles and summing in the same
order as the old kernels (`red[0]+red[1]+red[2]+red[3]`) — **the same bits** (parity test:
`tests/hip/gdn_rec_head.cpp`). **It uses no WMMA at all.** Plain FP32 SIMT: LDS, shuffles, FMA —
every instruction exists on gfx1030.

Dispatch (`kernels.cu:3216` `gdn_recurrence`): the env `STRATA_GDN_HEAD=1` selects the quad path
(variant 1). It defaults on only for gfx1151; every other card keeps the older kernels. The quad
kernels are compiled into our HIP build already — so on gfx1030 **`STRATA_GDN_HEAD=1` is a
zero-port, env-only experiment**. Upstream's own numbers from other cards: the old kernels took
~4 barriers per token and ~3.5 µs/token/layer on the Radeon 8060S (gfx1151 iGPU); the quad path
removes the per-token barriers. The `STRATA_GDN_PP=1` variant (`gdn_rec_quad_pp_kernel`) additionally
fixes VGPR pressure — the base quad kernel keeps 64 state-address VGPRs live across the whole walk
(196 VGPRs = 7 waves per SIMD, some blocks not resident); the PP variant recomputes addresses
through an asm (133 VGPRs, all blocks resident) and alternates two register sets so the update
emits no 32 register copies. `STRATA_GDN_PP=2` is a two-column variant. VGPR pressure is exactly
RDNA2's occupancy constraint too (256 VGPR/thread ceiling), so the PP variant is the one to expect
most from here. **[upstream-measured on gfx1151; unmeasured on the V620]**

The quad path also pairs with `gdn_out_norm_loop_kernel` (grid-stride, caps at 4096 blocks)
instead of the old `gdn_out_norm_kernel` launching T×HV blocks — at 128k that is ~3.2M blocks
per layer avoided. **[unmeasured]**

What gfx1030 runs today at PP: `gdn_rec_cols_pipe_kernel` (the HIP fall-through in
`gdn_recurrence_impl`; `STRATA_GDN_PIPELINE=1` default) — itself already the software-pipelined
improvement, same bits as `gdn_rec_cols_kernel`. (`native_gdn.cu`'s per-token `step` kernel is the
TG/decode path, not the prompt path — the earlier draft of this section conflated the two.)

**R1a steps (all measurement, no code):**
1. Add `STRATA_GDN_HEAD=1` to the timing config, reboot, tier pass; read the `gdn` /
   `gdn recurrence` phase deltas and end-to-end PP. Then `STRATA_GDN_HEAD=1 STRATA_GDN_PP=1`, then
   `PP=2`. Success bar: ≥1% end-to-end PP at 64k/128k, outside the ±0.3% noise floor.
2. Parity: the quad path is bit-exact by construction and has a test; additionally hash the GDN
   states (`STRATA_STATE_HASH_GDN=1`) across an arm swap to confirm on this build.
3. If it wins: keep the env in the daily config and propose upstream extending the gfx1151
   `STRATA_GDN_HEAD` default to gfx103x (upstream will want V620 numbers — we will have them).

### R1b — the chunked WY-form scan without WMMA (later, opt-in, not bit-exact)

The genuinely matrix-core-bound parts are `gdn_rec_wmma` (HIP chunked, FP16 operands) and the
CUDA-only `gdn_rec_chunked` / `gdn_rec_tc` WY-form paths. A scalar HIP port of the WY-form scan is
possible (the algorithm is ISA-independent; RDNA2 even has packed-FP16 VOP3P for the chunk products),
but the chunk-level reordering is FP32-level, **not bit-exact** against the recurrence
(`gdn_chunk.hpp:12` documents this explicitly — the precedent is `qsa_prompt_attn.hpp`). Upstream's
CUDA numbers: the chunked scan only pays above ~128-token chunks (RTX 5090: 0.58× at 32 tokens,
1.20× at 128; the TC scan 3.75× on a CMP 170HX) — on the V620 the crossover and the win are
**[unmeasured]**. Park this until R1a's numbers are in: if the quad path already closes most of the
gap to the gfx11 algorithm, R1b may never be worth its parity cost.

**Payoff ceiling.** Recurrence phases (`gdn recurrence` 9.1% + parts of `gdn` 13.9%) are the
target. If the quad path halves them: ~5% PP. R1b's ceiling is similar. **[unmeasured]**
### R1a results — first pass, 11-19 (single runs, same day, same server; med-of-3 still owed)

All three arms are env-only on the F-2 build (`f786675dc`), `STRATA_PREFILL_TIMING=1` on,
parking off, tier harness, 128k tier phase table from the timing log:

| 128k PP tok/s (GPU timeline) | baseline cols_pipe | `GDN_HEAD=1` | `+GDN_PP=1` | `+GDN_PP=2` |
|---|---|---|---|---|
| end-to-end PP | 1001.6 (133,301 ms) | 1020.5 (130,843) | 1020.9 (130,790) | **1027.1 (130,002)** |
| `gdn recurrence` ms | 12,135 | 8,477 (−30%) | 7,244 (−40%) | **5,962 (−51%)** |
| `gdn out proj` ms | 5,903 | 6,527 (+10.6%) | 6,803 (+15.2%) | 6,946 (+17.7%) |

Per-tier PP vs the same-day baseline: HEAD=1 is +0.6/0.7/2.1/1.9% (16k→128k);
PP=2 is −0.1/+0.1/+1.8/**+2.55%** — the win concentrates at 64k/128k, wash at 16k/32k.

Findings:

- The quad kernels are real on gfx1030 and the mechanism is exactly as the code read
  predicted: recurrence −51% at PP=2. All non-GDN phases flat (±0.3%) — clean causality.
- The win does not fully reach the wall: phase-table sums improve by ~1.9–2.5 s more than
  the GPU timeline does (~1 ms unattributed per ~2.4 ms saved; suspect mark folding around
  the quad kernels' host syncs). The end-to-end tok/s is the number that counts and it moves.
- New top cost inside the GDN family: the quad path's paired grid-stride out-norm kernel
  (`gdn out proj` +18% at PP=2, +419 ms). The old per-token launch `<<<T, HV>>>` schedules
  better on 72 CUs. A patch that decouples out-norm kernel choice from the recurrence
  variant would reclaim ~0.3% PP. **[candidate for the upstream proposal]**
- PP=1 buys recurrence only to hand most of it back; PP=2 (two-column) is the keeper.

Decision: **PP=2 promoted — the promotion checklist passed on 11-19.**

- **Med-of-3 (128k): PP=2 1024.0 vs baseline 987.0 = +3.75%**, and 64k: 1036.0 vs 1005.4 = +3.0%.
  Both far outside the ±0.3% noise floor. The baseline ran last and warmest (reversed round),
  so the bias goes against PP=2 — the true gap is if anything larger.
- **`STRATA_STATE_HASH_GDN=1` cross-arm check: identical hashes.** Same fixed prompt through
  the `gdn_rec_cols_pipe_kernel` path and the quad PP=2 path produces the same FNV-1a state
  hash on the V620 — the quad recurrence is bit-exact here, as the kernel header claims.
- Daily configs updated with `STRATA_GDN_HEAD=1 STRATA_GDN_PP=2` (§6.36 of the migration notes
  carries the same verdict).

## R2 — Expert i8 tile GEMM shape on the V620 (26–28% of prefill)

**What runs today.** `i8_kernel<T, GU>` in `i8_expert.cuh`: 64 weight rows × 16 token-entries
per block, K in 128-value stages, per-thread decode of one 32-value sub-block into shared
memory, then 8 dp4a + scale algebra + one FMA per 32 values per (row, entry). The dp4a
semantics lesson is already upstream (3-operand `v_dot4_i32_i8`, src2 = 0, on gfx1030-1032).
`I8_BLK_UNROLL=1` because full K-unroll spills 55–200 VGPRs on gfx1031.

**Why it matters.** Second-largest family; it is also the best-understood loop, so a shape
experiment is low-risk. The tile constants were tuned on RX 6700 XT / 6800 shapes, not a
72-CU V620 with 23.4 GiB expert cache. **[unmeasured]:** how much of the gap between this
card and the tuning target is tile shape.

**Concrete steps.**
1. Establish the per-phase baseline per tier (have it: `gemm gate/up` 26.0% at 128k).
2. Sweep the cheap knobs first, each as a separate build: `I8_KS` (K-stage 128 → 64/256),
   `I8_LD` (LDS pad 36), tile rows (64 → 32/128), `I8_BLK_UNROLL` 1→2 with a VGPR report
   (`--save-temps` or compiler VGPR spill warnings) to see where gfx1030 actually spills.
3. Extend `batch_rows_parity` / an i8 bench rather than measuring through the server for the
   sweep; confirm the top 2 shapes end-to-end with tier passes.
4. Also measure the few-rows boundary on this card: groups averaging <~7 rows route to
   `native_expert_grouped` (threshold in `i8_expert_rule.hpp`). At our prompt sizes the tile
   kernel always wins the dispatch; the interesting question is whether the 6700-XT-tuned
   threshold is right for 72 CUs. **[unmeasured]**

**Payoff ceiling.** 10% kernel win ≈ 2.5–3% PP. Worth doing after R1 only because the
experiments are cheap and mechanical. **[unmeasured]**

## R3 — QSA attention family (~26% of prefill)

**What runs today.** `qsa.cu` / `qsa_prompt_attn.cu` / `qsa_decode_attn.cu`: Q8/Q4 KV,
scores in shared memory (8 KB per head, one block per head), top-k by binary-lifting
threshold search (no sort, reproduces ggml tie rule), `f16_from_f32` instead of
`__float2half`, pooling in doubles with order-fixed contraction. Projections
(`qsa proj` 7.4%) go through the dense GEMM path (R4).

**Why it matters.** Third family, roughly tied with experts. It is already carefully written
for coalescing; the open question is block/grid shape vs 72 CUs and whether the top-k search
or the score dot dominates inside the 15.4%. **[unmeasured]** — the timing lump groups
`qsa attn` as one phase.

**Concrete steps.**
1. Sub-divide the phase: either extend `PfPhase` marks around the top-k and score stages
   (small engine patch, keep it out of the default path) or write a standalone parity/bench
   from `decode_cluster_parity` parts. Get the internal split before touching code.
2. Only then consider experiments: head-per-block vs multi-head-per-block on 72 CUs, score
   dot vector width, KV layout (Q8 vs Q4) for the prompt path.

**Payoff ceiling.** Similar to R2, ~2–3% PP for a strong win; likely less. Do after R1/R2.
**[unmeasured]**

## R4 — Dense projection GEMMs (inside the 26% attention family + hc/up)

**What runs today.** rocBLAS via `gemm.cu` with the FP16-in/FP32-accumulate route
(`prompt_f16()`, default-on for gfx103x, measured upstream at 11–14% over the FP32 route).
F-2's skip-widen already removed the widen pass on the consumer side (+1.8–1.9% measured,
ours). Tuning tables: `STRATA_ROCBLAS_TUNING`, `tools/hip/gfx1030-rocblas-*.txt`.

**Why it matters (a little).** This is a black box we can only tune via tables, plus one
opt-in lever: `STRATA_RDNA2_SGEMM=1`, upstream's own SGEMM path, author-reported 343→534
tok/s on an RX 6800 in one shape, **not bit-identical**, not default. It has never been
measured on the V620.

**Concrete steps.**
1. Cheap first test: flip `STRATA_RDNA2_SGEMM=1` in a config copy, tier pass, compare.
   Bit-exactness: expect none — check output drift with the in-server RMS check where
   applicable, and treat any speed win as suspicious until parity is understood.
2. rocBLAS table: only if a specific GEMM shape shows up badly in the timing log. Low
   priority.

**Payoff ceiling.** Low single-digit % at best. Last in line.

## R5 — Decode (TG): measure before touching anything

**What runs today.** TG 90–95 tok/s, memory-bound: `iq_mmvq` (expert GEMV, decode-once-
apply-to-all-columns) + PLE block (`ple.cu`) + expert cache streaming (12,333 experts /
23.4 GiB VRAM, rest from 46.8 GiB locked RAM). The dp4a chain inside is not the bottleneck;
bandwidth and hit rate are. **[unmeasured]** — no decode phase timer exists yet.

**Concrete steps.**
1. First task is the still-open §8 item 7: wire `STRATA_DECODE_TIMING` (decode-side PfTimer
   equivalent). Without it, every TG claim in this file stays structural.
2. Then measure: expert cache hit rate per tier, time split resident-vs-streamed experts,
   PCIe share, PLE batch behavior.
3. Only after the split: decide whether anything here is a kernel problem at all. Likely
   candidates are cache policy and stream batching, not inner loops.

**Payoff ceiling.** Unknown until measured — could be the largest single lever on the TG
column, or nothing. That is exactly why it needs the timer first.

---

## Order of attack

1. **R1 feasibility note** (read-only, days) — decide if the RDNA2 chunked GDN exists.
2. **R5 decode timer** (small engine patch) — unblocks the other half of the table.
3. **R1 implementation** if the note says yes; **R2 shape sweep** while builds of R1 cook.
4. **R3 internal split**, then only targeted work.
5. **R4** whenever a spare build slot exists; it is one env flag and one tier pass.

Each task lands its result as a dated row in `strata-migration.md` §5.2 (end-to-end) plus a
paragraph here (mechanism). Nothing merges into the F-2 branch without a med-of-3 win outside
±0.3% and a parity statement.
