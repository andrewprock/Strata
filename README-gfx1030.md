# Strata on Radeon V620 (gfx1030) - tuned branch

## 1. Intro and goal

Strata's release binaries target the cards the maintainers test. The Radeon PRO
V620 (gfx1030) is an RDNA2 workstation card that works, but sits outside the
tuned set: several prefill paths that are fastest on newer cards fall back to
slower generic code here, and the default prefill-MMQ kernels do not compile to
the best instruction form for this ISA.

This branch carries a small set of measured local changes on top of upstream
main, tuned for exactly one box: one V620 on a Threadripper PRO workstation. It
is not a general gfx1030 port. The reasoning behind every change is recorded
in `bench/notes/strata-migration.md`.

Goal, in one line: **the fastest reading speed (prompt processing) this card
can give for the Qwen3.8-Flash-Next IQ3_S model, without giving up writing
speed (token generation) or stability.**

The branch follows upstream main. When upstream moves, the local commits are
re-based onto the new base, re-benchmarked against the old build, and promoted
only if they measure faster. Every promotion is recorded in the migration note.

## 2. Changes to code

Everything the branch adds vs upstream main is one functional change, one
environment flag pair, and one test. (An older generation of this branch also
carried MMQ VOP3P patches; upstream deleted that whole build path in 0.2.16
and its own-kernel rewrite reached the same speed, so the patches are retired -
see the note at the end of this section.)

### F-2: hyper-connection skip-widen (prefill)

- Files: `engine/include/strata/prefill/gemm.hpp`,
  `engine/include/strata/prefill/kernels.hpp`, `engine/src/prefill/gemm.cu`,
  `engine/src/prefill/kernels.cu`, `engine/src/prefill/prefill.cpp`
  (before the 0.2.16 layout move these lived at `src/...`; the change ports
  across the rename unchanged)
- What it does: in prefill, the hyper-connection up-projection consumer is fed
  packed fp16 halves straight to the GEMM instead of first widening them to
  fp32. The widen pass is skipped; the math result is unchanged.
- Why: the prefill hot loop runs the hyper-connection up-projection millions of
  times per prompt, and upstream widens its input fp16 to fp32 first, every
  time. Widening loses no information, so the GEMM sees the same numbers
  either way and the pass is pure overhead in the card's busiest loop. F-2
  removes that overhead. It fixes no correctness bug - it stops paying for
  work the math cannot see.
- How to turn it on: environment variable `STRATA_HC_MIX_H16=1`.
  Without the variable the code path is identical to upstream.
- Measured effect: +1.0 to +2.2% prompt processing on this card at every
  context tier, re-confirmed on 2026-11-23 against 0.2.20-dev (see section 4).

### MMQ VOP3P patches (vendored ggml, prefill path)

- Files: `third_party/strata-ggml/ggml/src/ggml-cuda/common.cuh`,
  `third_party/strata-ggml/ggml/src/ggml-cuda/vecdotq.cuh`
- What it does: rewrites the int8 dot chains in the MMQ quantized-kernel
  inner loops to use VOP3P packed-math instructions instead of unpacked
  V_DOT4 chains, and adds a `dp4a_z` helper for zero-extended accumulation.
  Applies to the q8_0 and IQ2_XS/IQ2_S dot chains.
- Why: upstream's MMQ inner loop uses the unpacked `V_DOT4` int8 dot
  instruction, which is slower than the packed `VOP3P` math the same card
  offers. On gfx1030 that leaves quantized-prefill speed unused in the hottest
  loop of the kernel. The patches swap in the packed instructions and cut the
  instruction count. What they fix is an ISA-level inefficiency, not a bug -
  and because it sits inside quantized prefill, it matters most at the lower
  tiers, where that path is a bigger share of the work.
- How to turn it on: build flag `STRATA_PREFILL_MMQ=ON` (requires
  `STRATA_MMQ=ON`, the new split on current main).
- Measured effect: +0.5 to +1.0% prompt processing on top of F-2 at the
  lower tiers, neutral at the top tier (see section 4).

### GDN quad path (environment only, decode-side linear attention)

- Files: none - this is upstream code, present in every HIP build since
  0.2.16, selected only by environment variables.
- What it is: the GDN recurrence kernels have a quad-wave variant
  (`STRATA_GDN_HEAD=1`) plus a two-pass variant (`STRATA_GDN_PP=2`) that are
  enabled by default only on gfx1151. On gfx1030 the default path is plain
  FP32 SIMT; the quad path cuts the GDN recurrence kernel roughly in half.
- Why: it is free speed - no patch, no rebuild, bit-exact (verified with
  `STRATA_STATE_HASH_GDN=1` cross-arm state hashes on 11-19).
- Measured effect: on 0.2.16 it won only at 64k/128k; on 0.2.20 it wins at
  every tier, +4.1 to +4.5% prompt processing (2026-11-23, see section 4).

### Retired: MMQ VOP3P patches (were: vendored ggml, prefill path)

Generations 3-5 of this branch carried three patches to the bundled
llama.cpp's MMQ kernels (packed-FP16 VOP3P dot chains for q8_0 and the
IQ2 family, plus a `dp4a_z` helper). They measured +0.5 to +1.0% prompt
processing on top of F-2. Upstream 0.2.16 removed the entire llama.cpp MMQ
build path (the options, the vendored `ggml-cuda`, everything) in favor of
Strata's own prompt kernels, and the rewrite reached the same speed on its
own - so there is nothing left to patch. The patches live on in history:
branch `v620-0215-f2-mmq` (generation 5) on the fork, and
`andrewprock/llama.cpp` branch `v620-mmq-vop3p` for the pre-vendored bases.

### Test

- `tests/hip/upmix_gemm_bench.cpp`: a HIP benchmark that checks the F-2 GEMM
  path produces bit-identical results to the widen-first path.

Nothing else is changed. Kernel selection, scheduling, quantization formats and
the serving stack are all upstream.

## 3. Installation

### What you need

- The IQ3_S model file, split into 2 GGUF shards (see section 6 for where this
  machine keeps them). Get it the same way as any Strata model - the standard
  install docs cover it.
- A Python virtual environment for the server (`terminal/serve/server.py`).
  If you have already run the normal source setup (`setup.sh`), you have this.
- ROCm installed. This machine runs ROCm 10.0.

### Build

Everything after the clone is the standard Strata source build, with exactly
one non-default flag. From the checkout (since 0.2.16 the engine lives in
`engine/`):

```sh
cmake -S engine -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON \
  -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1030
cmake --build build-hip --target strata -j 32
```

gfx1030 builds with a warning (upstream tests gfx1100+; the card is fine).
That is the whole recipe - there are no tuning flags left to set, because
upstream 0.2.16 removed the old MMQ build options and Strata's own prompt
kernels became the only path. Always do a clean build: delete `build-hip/`
first if it exists, because incremental rebuilds after a flag change produce
false comparisons.

The engine binary lands at `build-hip/strata`.

### Serve

The server is started the standard way, with one flag that matters:
`--engine strata` is mandatory on current main (the default is the mock engine).

```sh
cd ~/git/Strata          # or wherever your venv lives
.venv/bin/python terminal/serve/server.py --engine strata --config <config.json> --port 8080
```

This machine wraps that in `~/bin/strata-serve` (foreground start, plus `stop`,
`logs`, `status`). The wrapper is a convenience, not a requirement.

### The config

A config JSON tells the server which engine binary to run, which model files to
load, and which environment variables to set. The working config for this
machine is `~/strata-v620-iq3_s-T9-v6.json`; its full contents are in section 6.
The deviations from a default config, and why each one is there:

| Setting | Value | Why |
|---|---|---|
| `spec` | 5 | multi-token prediction draft, measured +5-7% writing speed |
| `spec-min-p` | 0.70 | calibration result; below this the draft hurts |
| `pcie-frac` | 0.00 | calibration result; no prefill CPU assist for PCIe |
| `kv` | int8 | fits the 32 GB card; measured no quality loss |
| `kv-resident` | 32768 | parking window for daily use |
| `--mtp-draft-vocab` | draft_vocab_en.bin | the T9 draft-vocabulary recipe |
| `--ple-io` | ram | PLE pack served from RAM |
| `STRATA_HC_MIX_H16` | 1 | F-2 (this branch's main change) |
| `STRATA_HIP_PROMPT_F16` | 1 | fp16 prompt buffers |
| `STRATA_SELECT_SIMT` | 1 | SIMT1 kernels measured faster than SIMT2 here |
| `STRATA_SH_STREAM` | 1 | calibration result |
| `STRATA_PREFILL_CPU_SHARE` | 1 | measured +2% reading on this CPU |
| `STRATA_HC_PERSIST` | 1 | persistent hyper-connection kernel |
| `STRATA_ROCBLAS_TUNING` | absolute path | rocBLAS tuning table for gfx1030 |
| `STRATA_WATCHDOG_S` | 300 | restart the engine if it stalls 5 min |

For benchmarking, add `--conversation-cache-mib 0` to the args (parking off -
see section 5) and remove `kv-resident`. For daily use, keep them as shown.

## 4. Current results

Model: **Qwen3.8-Flash-Next, IQ3_S**, on this V620. Med-of-3 (med-of-4 at
64k/128k for the GDN row), same-day, parking off, 2026-11-23. PP / TG in
tokens per second.

| Build | 16k | 32k | 64k | 128k |
|---|---|---|---|---|
| A: generation 5, re-run (0.2.15-dev + F-2 + MMQ) | 998.5 / 98.0 | 1012.5 / 93.2 | 1016.6 / 94.5 | 1005.2 / 92.2 |
| B: vanilla upstream main (0.2.20-dev) | 990.2 / 99.3 | 988.1 / 100.1 | 997.3 / 96.9 | 979.7 / 93.1 |
| C: 0.2.20-dev + F-2 | 1009.1 / 100.7 | 1010.0 / 96.8 | 1007.1 / 95.7 | 997.0 / 93.4 |
| **D: this branch** (C + GDN quad path) | **1054.4 / 100.6** | **1051.9 / 99.1** | **1048.4 / 94.7** | **1037.8 / 93.7** |

Reading the table:

- **The GDN quad path is the big lever now: +4.1 to +4.5% reading speed at
  every tier, for two environment variables and no patch.** On 0.2.16 it only
  won at 64k/128k; on 0.2.20 the low tiers win too.
- **F-2 still matters on current main**: C beats B by +1.0 to +2.2% PP at
  every tier, so F-2 has not been absorbed upstream.
- **Upstream's own-kernel rewrite was neutral for this card**: vanilla
  0.2.20-dev (B) is within noise of vanilla 0.2.15-dev was on 2026-11-18
  (976.8 / 989.6 / 994.1 / 983.8).
- **The retired MMQ patches would have added nothing today**: A (the old
  MMQ stack) and C are within noise of each other on PP.
- Writing speed (TG) is statistically unchanged - all the changes are
  prefill/decode-kernel-side, and TG noise on this box is about +/-4%.
- All-time records for this card: the D row at every tier. The previous
  IQ3_S bests were 995.7 / 1008.7 / 1013.5 / 1002.9 (generation 5,
  2026-11-18).

For context, the same box on other quant lanes (med-of-3, earlier dates):

- UD-IQ1_M (2026-11-06): 1000.6 / 100.9, 1004.4 / 98.1, 999.7 / 97.6, 982.2 / 92.1
- UD-IQ2_XS (2026-11-06): 979.7 / 96.0, 984.5 / 96.6, 980.6 / 94.3, 964.6 / 93.0

The IQ3_S lane is the daily reference; the other lanes were one-off
evaluations. Full history and the reasoning behind every adopted and rejected
change are in `bench/notes/strata-migration.md`.

## 5. The full tier test and methodology

The benchmark is `bench_strata_tiers.py`, from
github.com/vedmichv/llm-benchmark. It sweeps the four context tiers used for
all comparisons here:
**16k, 32k, 64k, 128k** - that is, the same prompt is sent at four different
context sizes so the reading-speed curve can be seen across the whole range.

What one run does:

- For each tier, send a real-text prompt at that context size
  (`NUM_CTX=135168` tokens of context budget per request), generate
  `NUM_PREDICT=1024` tokens.
- Record two numbers per tier: **PP** (prompt processing, tokens/s - the
  reading speed) and **TG** (token generation, tokens/s - the writing speed).
- Each pass writes a JSON to `results/strata_tier_pass_<date>_<time>.json`.

Methodology rules (the parts that matter for comparing numbers):

1. **Soak pass is discarded.** The first sweep after a fresh boot runs the
   expert-cache cold and reads 2-3% slow. It warms the caches; it is not
   counted.
2. **Med-of-3.** Three counted passes per boot; the median PP per tier is the
   result. TG is treated as noise (same-binary day drift of 92.6 to 87.9 tok/s
   at 16k has been demonstrated, so only large TG changes are meaningful).
3. **Parking off.** Bench configs use `--conversation-cache-mib 0`. The
   conversation cache (parking) is a daily-use feature; it makes the first
   request after a change ~1 s slower, which pollutes small measurements.
4. **Interleaved boots for A/B.** When comparing two builds, alternate which
   build boots first, and reverse the order in the second cycle. This cancels
   boot-order and thermal bias (a later boot runs on a warmer card).
5. **Same day, same everything else.** Comparisons are only valid same-day,
   same config, same model file. The engine's context budget must exceed the
   prompt token count.
6. **Env config.** The script has no CLI arguments; it is configured through
   environment variables: `STRATA_BASE`, `STRATA_API_KEY`, `STRATA_MODEL`,
   `NUM_CTX`, `NUM_PREDICT`, `TIERS`.

One full arm (1 soak + 3 counted sweeps, 4 tiers each) takes about 25 minutes.

## 6. Test hardware / software / OS / config

Recorded here because every number in section 4 was measured on this box.

Hardware:

- GPU: AMD Radeon PRO V620 - gfx1030, 40 compute units, 32 GB GDDR6
- CPU: AMD Threadripper PRO 3975WX, 32 cores / 64 threads
- RAM: 207 GB
- Storage: Azure NVv5 (local NVMe)
- Power limit: about 180 W on the card (stock)

Software:

- OS: Ubuntu 26.04.1 LTS (64-bit), running as a VM on Proxmox 9.2.11
- ROCm 10.0
- Python venv for the server, upstream `serve/server.py`
- rocBLAS tuning table `gfx1030-rocblas-5.6.0.8d1ae90e.txt` (in
  `tools/hip/`, path set via `STRATA_ROCBLAS_TUNING`, absolute path required)

Model and files (IQ3_S lane):

- Model: Qwen3.8-Flash-Next, GGUF quant IQ3_S, split in 2 shards
  (`/var/lib/ollama/gguf/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf`)
- PLE pack: `/home/drewster/git/Strata-data/packs/iq3_s` (loads in 0.3 s)
- Expert profile: `/home/drewster/git/Strata/data/expert-profile.bin`
- T9 draft vocab: `/home/drewster/git/Strata/data/draft_vocab_en.bin`
- MTP draft: `/home/drewster/git/Strata-data/mtp/rt`

Serving posture: `host 0.0.0.0`, port 8080, API key required, model id
`qwen3.8-flash-next-iq3_s-draftvocab-en`, context 262144 tokens, logs in
`~/engines/logs/daily.log`.
