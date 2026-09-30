# M1 Max / Mojo 1.1 inference campaign

Objective: improve greedy CPU inference throughput without changing generated
text. Baseline: `b86ad44`. Machine: Apple M1 Max, 8 performance + 2 efficiency
cores, 32 GB RAM, macOS 26.3; Mojo 1.1.0 (8189361e). Build: default `mojo build`
(O3, host CPU, default floating-point contraction), held constant throughout.
No CPU affinity is available on macOS. Other desktop applications remain open;
canary measurements qualify all performance conclusions.

Fixed workload: stories15M, stories42M, stories110M, seed 1, greedy temperature
0, prompt `Once upon a time`, requested maximum 256 tokens (normal EOS applies).
Model SHA-256 hashes and toolchain details: outer repository
`bench/autotune-m1/environment.json`. Sweep: every thread count 1–8.
Correctness: all five test suites plus byte-identical generated text versus
baseline for every model/thread cell, excluding banners and timing lines.
Baseline passed all tests and all 24 inference cells.

An unchanged baseline executable is retained as `bench/autotune-m1/baseline`,
and its source at the `autotune/m1-canary` worktree/branch. Every uploaded suite
contains all 24 candidate and all 24 canary cells, with one warmup and three
measured runs per cell. The server records mean tok/s. Acceptance instead uses
three A/B/canary rounds, one warmup plus seven invocations per group, best tok/s
(minimum inference time) on both sides. Initial focus: 15M/8 and 110M/8 threads.
Every round and invocation is retained in JSON. No speed claim is based on
comparing separate uploaded reports. Failed attempts remain on named branches.

Implementation pacing: investigate one profile-supported change at a time;
stop after consecutive non-wins without a new profile signal or a measured
hardware ceiling. No input, output, or correctness requirement may be weakened.

## Baseline

Initial smoke measurements flatten near four threads; eight threads are slower
than four on all three models. This is a hypothesis signal, not a speed claim.
The full baseline report, profiler samples, and noise measurements accompany
this log in the outer repository under `bench/autotune-m1/`.

## m1-h1-rope-barrier: omit the local-head RoPE barrier

Hypothesis written before timing: ordinary MHA rotates and consumes each Q/K
head on the same worker. Removing its cross-worker barrier should improve the
8-thread focus cells by approximately 5–15%, with single-threaded work unchanged.
GQA retains its barrier because KV ownership can cross query-head workers.
`sample` on baseline 110M/8 collected 2,701 thread samples, of which 337 (12.5%)
were in the scheduler yield under forward-worker barriers; the main thread
waited in the runtime semaphore for 294/301 (97.7%) of its samples. The latter
includes useful worker execution and is not a claim of pure dispatch overhead.

Correctness: all 5 test suites and all 24 inference cells passed.

| Round | Model / threads | Parent best tok/s | Attempt best tok/s | Canary best tok/s | B/A |
|---|---|---:|---:|---:|---:|
| 1 | stories15M / 8 | 983.87 | 973.40 | 994.57 | 0.9894 |
| 1 | stories110M / 8 | 182.34 | 183.60 | 184.88 | 1.0069 |
| 2 | stories15M / 8 | 871.43 | 928.93 | 943.30 | 1.0660 |
| 2 | stories110M / 8 | 175.30 | 173.01 | 169.69 | 0.9869 |
| 3 | stories15M / 8 | 871.43 | 928.93 | 897.06 | 1.0660 |
| 3 | stories110M / 8 | 176.08 | 177.06 | 173.01 | 1.0056 |

Verdict: **neutral, dropped**. Worst ratios 0.9894 (15M) and 0.9869 (110M)
are below the 1.10 acceptance bar; directions disagree. No speed claim.
Re-profiled before dropping: 731/4,689 thread samples (15.6%) still yield in
barriers. Removing the redundant barrier did not remove the dominant waiting.
The canary drifted 10–11% over the A/B sitting; this further precludes accepting
small differences. Attempt is retained at `attempt/m1-h1-rope-barrier`, commit
`5de9501`. Uploaded with `kept=no, verdict=neutral`:
https://swtune.engiware.com/reports/25797d4b-40d2-4101-adc7-08e70c5a39e8

Noise: the skill's seven-invocation script found single-invocation spreads of
24.3% on 15M/8 and 7.4% on 110M/8. Three repeated best-of-seven groups reduced
that to 3.35% and 6.32%; a conservative 10% bar applies to each paired round.
Aggregate 10-thread copy bandwidth (1.25 GiB source+destination allocation,
20 copies per sample, 1 warmup + 7 samples): 172.93–184.74 GB/s read+write.
This is a measured copy reference, not proof of a read-only inference ceiling.
No exact instruction/FLOP ceiling is claimed: sustained clock is unavailable.

## m1-h2-row-tiles: reuse input loads across four output rows

Hypothesis written before timing: one-matrix GEMV loads the same input vector
for each output row. A four-row tile should cut these input loads by 75%,
expose more independent FMA chains, and improve 15M by roughly 10–25%.
Baseline profile puts 2,256/2,701 thread samples (83.5%, inclusive of waits)
in forward_worker; sampled offsets 9396 and 7208 resolve to GEMV FMA/load
loops in disassembly. There is no instruction-level percentage claim because
`sample` aggregates multiple PCs per frame. Batched QKV/FFN retain the original
kernel to avoid register pressure. Lane accumulation order remains unchanged.
Add scalar-reference tests for row subranges, row tails, column tails and empty
ranges. Focus expands to 15M/1 as well as 15M/8 and 110M/8 for this kernel trial.

Correctness: all five test suites, including the new scalar-reference edge cases,
and all 24 inference cells passed.

| Round | Model / threads | Parent best tok/s | Attempt best tok/s | Canary best tok/s | B/A |
|---|---|---:|---:|---:|---:|
| 1 | stories15M / 1 | 778.72 | 788.79 | 768.91 | 1.0129 |
| 1 | stories15M / 8 | 915.00 | 884.06 | 963.16 | 0.9662 |
| 1 | stories110M / 8 | 182.13 | 210.04 | 220.22 | 1.1532 |
| 2 | stories15M / 1 | 768.91 | 802.63 | 772.15 | 1.0439 |
| 2 | stories15M / 8 | 1000.00 | 973.40 | 1051.72 | 0.9734 |
| 2 | stories110M / 8 | 222.07 | 202.81 | 221.76 | 0.9133 |
| 3 | stories15M / 1 | 768.91 | 795.65 | 768.91 | 1.0348 |
| 3 | stories15M / 8 | 1095.81 | 1057.80 | 1089.29 | 0.9653 |
| 3 | stories110M / 8 | 216.33 | 205.43 | 215.74 | 0.9496 |

Verdict: **neutral, dropped**. 15M/1 gained 1.3–4.4% (worst 1.0129,
0.921 times the 1.10 acceptance ratio), while 15M/8 lost 2.7–3.5% and stable
later 110M/8 rounds lost 5.0–8.7%. These losses do not clear the conservative
10% bar, so no formal regression claim; they still argue against keeping it.
The first 110M round is invalid for a verdict: baseline A=182.13 and the
identical canary C=220.22 moved 20.9% in the same round. Re-profile: 703/5,203
thread samples (13.5%) in scheduler yield. Disassembly confirms four-row input
reuse and no stack spills inside the hot tiled FMA loop; the intended mechanism
is present but does not pay. Retained at `attempt/m1-h2-row-tiles`, commit
`b49116e`. The 15M/1 best-of-seven estimator floor was 1.72% (three groups).
Uploaded with `kept=no, verdict=neutral`:
https://swtune.engiware.com/reports/25a9a022-f043-4033-b4b1-244be65bad35

## m1-h3-generation-pool: dispatch workers once per generation

Hypothesis written before timing: Mojo 1.1 currently creates and joins a MAX
parallel region per token. Baseline main-thread samples include allocation/free
and runtime dispatch, and spend 97.7% waiting for the per-token parallel region.
That percentage includes useful computation and does not measure dispatch cost.
Move the entire decode loop into one supported `parallelize` call. Worker zero
owns sampling and output; two barriers publish complete logits and the next
token/EOS. Expected gain: approximately 10–25% on 15M/8, less on 110M/8.
The standalone Transformer API remains available. No private async API or
platform-specific thread creation is introduced. This was initially the final planned experiment; the user subsequently requested
continuing the campaign. Measurement counts stay fixed.

Initial implementation failed the correctness gate (single-thread hang): copying
only the raw pool pointer into the closure let Mojo destroy the Transformer
owner after its last use. Capturing the owner in the closure fixed the lifetime.
This failed implementation was never timed. The corrected implementation passed
all 5 unit suites, all 24 model/thread text comparisons, and 32 extra control-flow
checks against baseline (short generation, EOS, empty/long prompts, seeded
sampling). A reusable CLI regression test is retained on the attempt branch.

| Round | Model / threads | Parent best tok/s | Attempt best tok/s | Canary best tok/s | B/A |
|---|---|---:|---:|---:|---:|
| 1 | stories15M / 1 | 775.42 | 775.42 | 772.15 | 1.0000 |
| 1 | stories15M / 8 | 1102.41 | 1022.35 | 1070.18 | 0.9274 |
| 1 | stories110M / 8 | 205.16 | 199.75 | 190.65 | 0.9736 |
| 2 | stories15M / 1 | 762.50 | 759.34 | 772.15 | 0.9959 |
| 2 | stories15M / 8 | 1051.72 | 1345.59 | 1057.80 | 1.2794 |
| 2 | stories110M / 8 | 207.84 | 202.03 | 209.49 | 0.9720 |
| 3 | stories15M / 1 | 762.50 | 772.15 | 778.72 | 1.0127 |
| 3 | stories15M / 8 | 1095.81 | 1279.72 | 1089.29 | 1.1678 |
| 3 | stories110M / 8 | 202.55 | 213.71 | 198.75 | 1.0551 |

Verdict: **neutral, dropped**. The 15M/8 ratios were 0.9274, 1.2794, 1.1678;
the worst is below the 1.10 bar despite promising later rounds. 110M/8 was
0.9720–1.0551 and changed direction. Single-thread performance was unchanged.
Re-profile confirmed that generation lives inside one parallel region, while
479/3681 (13.0%) thread samples still yielded in barriers.
No repeatable win is claimed. Retained at `attempt/m1-h3-generation-pool`,
commit `d2d83f0`. Uploaded with `kept=no, verdict=neutral`:
https://swtune.engiware.com/reports/b05208c6-0121-49c5-85dd-7e98af37f202

## m1-h4-barrier-padding: separate barrier counter and generation cache lines

Hypothesis written before timing: the arrival counter and polled generation
flag currently sit eight bytes apart. Each arrival's atomic read-modify-write
invalidates the cache line all waiting workers read. Separate the two atomics
by at least 128 bytes, covering M1 cache lines, without changing synchronization
semantics. Expected gain: 10–20% on 15M/8 and little effect on single-thread.
This follows the persistent barrier/yield signal in all three profiles. It is
independent of the three rejected changes and starts from baseline `b86ad44`.

All five unit suites and all 24 model/thread output checks passed.

| Round | Model / threads | Parent best tok/s | Attempt best tok/s | Canary best tok/s | B/A |
|---|---|---:|---:|---:|---:|
| 1 | stories15M / 8 | 1057.80 | 1082.84 | 1082.84 | 1.0237 |
| 1 | stories110M / 8 | 211.72 | 209.49 | 217.21 | 0.9895 |
| 2 | stories15M / 8 | 1063.95 | 1102.41 | 1122.70 | 1.0361 |
| 2 | stories110M / 8 | 216.62 | 215.45 | 216.03 | 0.9946 |
| 3 | stories15M / 8 | 1051.72 | 1076.47 | 1045.71 | 1.0235 |
| 3 | stories110M / 8 | 210.04 | 214.86 | 216.33 | 1.0230 |

Verdict: **neutral, dropped**. 15M/8 gained 2.35–3.61%, but the worst 1.0235
ratio is only 0.930 times the 1.10 bar. 110M/8 ranged 0.9895–1.0230.
Disassembly confirms counter offset 8 and generation offset 256, so the
intended separation happened. Re-profile: 277/3,759 (7.4%) thread samples
yielded; this is lower sampled waiting but not a measured throughput win.
Retained at `attempt/m1-h4-barrier-padding`, commit `93cc675`.
https://swtune.engiware.com/reports/b4b8da53-73d2-4138-9a1c-f78df92b71c6

## Follow-up to m1-h3-generation-pool: test the four-thread sweep signal

The uploaded H3 sweep's 15M/4 mean was 1,407.7 tok/s versus its canary's
1,105.9 tok/s. That 1.273x is only a signal: the earlier baseline report's
four-thread figure was higher, and across-report comparisons do not judge wins.
Before new timing, designate 15M/4 the primary claim cell; 15M/5 and 110M/4
are guardrails, alongside the completed 1/8-thread pairs and full 24-cell sweep.
Measure a fresh best-of-seven estimator floor at 15M/4, then three A/B/canary
rounds with the unchanged baseline and H3 binaries. Retain the conservative
1.10 bar for a speed claim. No code changes or additional techniques are bundled.

15M/4 best-of-seven noise floor: 5.19% over three groups.

| Round | Model / threads | Parent best tok/s | H3 best tok/s | Canary best tok/s | B/A |
|---|---|---:|---:|---:|---:|
| 1 | stories15M / 4 | 1335.77 | 1396.95 | 1297.87 | 1.0458 |
| 1 | stories15M / 5 | 1158.23 | 1270.83 | 1158.23 | 1.0972 |
| 1 | stories110M / 4 | 225.21 | 223.63 | 224.58 | 0.9930 |
| 2 | stories15M / 4 | 1262.07 | 1375.94 | 1270.83 | 1.0902 |
| 2 | stories15M / 5 | 1150.94 | 1236.49 | 1143.75 | 1.0743 |
| 2 | stories110M / 4 | 225.53 | 226.82 | 230.10 | 1.0057 |
| 3 | stories15M / 4 | 1288.73 | 1429.69 | 1316.55 | 1.1094 |
| 3 | stories15M / 5 | 1165.61 | 1288.73 | 1203.95 | 1.1056 |
| 3 | stories110M / 4 | 231.78 | 219.92 | 220.53 | 0.9488 |

Follow-up verdict: **neutral, still dropped**. 15M/4 gained 4.58–10.94%,
but worst 1.0458 is below 1.10 (0.951 times the acceptance ratio). 15M/5 gained
7.43–10.56%; 110M/4 ranged 0.9488–1.0057. This rules out treating the 27%
mean gap in the earlier uploaded sweep as an established improvement.
The unchanged H3 code received another full 48-cell run, uploaded here:
https://swtune.engiware.com/reports/5cb1dd42-790d-41cc-9ac7-f42e5569e2d5

## m1-h5-worker-arrivals: replace the contended arrival counter

Hypothesis written before timing: cache-line separation alone did not remove
contention between workers' atomic read-modify-writes to the same arrival
counter. Give each worker an independently written generation flag, separated
by 128 bytes. Worker zero observes each arrival and releases the shared
completion generation. Keep sequentially consistent atomics and the same
barrier positions. Expected improvement: 10–20% on the primary 15M/8 cell;
110M/8 is the paired guardrail and the full 1–8-thread sweep checks other cells.
A dedicated test checks shared-memory publication and generation reuse across
2,000 barriers at each of 1–8 threads, in addition to normal unit and inference
checks. This starts from baseline and does not include any rejected change.

All five unit suites, 24 output comparisons, and the dedicated barrier test
(2,000 barrier cycles at each worker count 1–8) passed.

| Round | Model / threads | Parent best tok/s | Attempt best tok/s | Canary best tok/s | B/A |
|---|---|---:|---:|---:|---:|
| 1 | stories15M / 8 | 1051.72 | 1051.72 | 968.25 | 1.0000 |
| 1 | stories110M / 8 | 205.69 | 209.49 | 202.29 | 1.0184 |
| 2 | stories15M / 8 | 1063.95 | 1122.70 | 1051.72 | 1.0552 |
| 2 | stories110M / 8 | 208.39 | 207.03 | 207.57 | 0.9935 |
| 3 | stories15M / 8 | 1000.00 | 968.25 | 958.12 | 0.9683 |
| 3 | stories110M / 8 | 202.81 | 202.55 | 186.62 | 0.9987 |

Verdict: **neutral, dropped**. 15M/8 ranged 0.9683–1.0552 and changed
direction; worst/bar = 0.880. 110M/8 ranged 0.9935–1.0184. Disassembly shows
seven arrival-counter RMW instructions in baseline forward_worker and zero in
the attempt. Re-profile still finds 567/4,494 (12.6%) samples yielding: the
counter was not the dominant source of waiting. Retained at
`attempt/m1-h5-worker-arrivals`, commit `7527b44`.
https://swtune.engiware.com/reports/cb365c97-59e9-40c2-a007-08d5560045b8

## m1-h6-worker-qos: equalize inference worker scheduling priority on macOS

A final platform probe found a concrete new signal: MAX worker 0 reports QoS
33 (user-interactive), while workers 1–7 report QoS 21 (default). All calls to
pthread_get_qos_class_np and pthread_set_qos_class_self_np succeed. This can
amplify spin-barrier priority inversion on heterogeneous cores; Apple documents
that risk, but this campaign has not directly measured which core each worker
runs on. Source: https://developer.apple.com/documentation/apple-silicon/tuning-your-code-s-performance-for-apple-silicon

Hypothesis before timing: temporarily give all multi-threaded generation
workers the user-initiated QoS class (25), restoring each original class and
relative priority on return. Expect 10–25% higher/stabler 15M/8 throughput.
Parent is H3 (`d2d83f0`), since its generation-long worker lifetime provides one
well-defined scheduling scope. Single-threaded execution and non-macOS builds
do not change priority. A/B compares H3 against H6 with the original baseline
as canary; primary cell is 15M/8 and 110M/8 is a paired guardrail. No affinity or
performance-core guarantee is claimed. The original acceptance bar remains 1.10.

Correctness: all five unit suites, 24 model/thread output comparisons, and
32 extra generation checks (EOS, short/empty/long prompts, seeded sampling)
passed before timing. One warmup plus seven measured invocations per group,
three rounds; round-to-round and within-group spreads are tabulated below.

| Round | Model / threads | Parent best tok/s | Attempt best tok/s | Canary best tok/s | B/A |
|---|---|---:|---:|---:|---:|
| 1 | stories15M / 8 | 1262.07 | 1279.72 | 1005.49 | 1.0140 |
| 1 | stories110M / 8 | 210.88 | 207.30 | 208.39 | 0.9831 |
| 2 | stories15M / 8 | 1270.83 | 1270.83 | 1082.84 | 1.0000 |
| 2 | stories110M / 8 | 212.28 | 216.33 | 207.84 | 1.0190 |
| 3 | stories15M / 8 | 1307.14 | 1316.55 | 989.19 | 1.0072 |
| 3 | stories110M / 8 | 211.44 | 211.72 | 207.84 | 1.0013 |

Verdict: **neutral, dropped**. 15M/8 ratios were 1.0000–1.0140; worst/bar
= 0.909. 110M/8 ranged 0.9831–1.0190. This implementation did not establish
a scheduling improvement over H3. Canary 15M/8 best values moved 9.47%
across rounds, further limiting inference about small effects. The binary
contains the intended QoS get/set and restoration calls. Re-profile collected
5,202 thread samples, with 530 (10.2%) in cthread_yield under generation
barriers. No new actionable bottleneck was established. Retained at
`attempt/m1-h6-worker-qos`, commit `56c783f`.
https://swtune.engiware.com/reports/e91183c3-8da7-4eb5-b4d2-608d5143fb71

## Paired measurement spreads

Each A/B/canary group has one warmup and seven measured invocations. Each
cell has three rounds. Round spread is `(max(best) / min(best) - 1) × 100%`;
group spread reports the largest corresponding seven-invocation spread.
A = parent, B = attempt, C = unchanged baseline. These ranges expose noise;
they do not override the per-round verdicts or qualify void rounds as wins.

| Stage / cell | A round spread | B round spread | C round spread | Max group spread A / B / C |
|---|---:|---:|---:|---|
| m1-h1-rope-barrier / stories110M / 8 | 4.01% | 6.12% | 8.95% | 16.17% / 11.32% / 17.09% |
| m1-h1-rope-barrier / stories15M / 8 | 12.90% | 4.79% | 10.87% | 34.41% / 34.01% / 36.96% |
| m1-h2-row-tiles / stories110M / 8 | 21.93% | 3.57% | 2.79% | 22.11% / 20.79% / 11.91% |
| m1-h2-row-tiles / stories15M / 1 | 1.28% | 1.75% | 0.42% | 5.53% / 3.45% / 4.20% |
| m1-h2-row-tiles / stories15M / 8 | 19.76% | 19.65% | 13.10% | 24.59% / 30.64% / 23.21% |
| m1-h3-generation-pool / stories110M / 8 | 2.61% | 6.99% | 9.88% | 21.70% / 29.10% / 17.75% |
| m1-h3-generation-pool / stories15M / 1 | 1.69% | 2.12% | 0.85% | 10.42% / 6.64% / 4.64% |
| m1-h3-generation-pool / stories15M / 8 | 4.82% | 31.62% | 2.98% | 56.02% / 58.09% / 21.43% |
| m1-h4-barrier-padding / stories110M / 8 | 3.13% | 2.85% | 0.55% | 24.17% / 12.03% / 20.95% |
| m1-h4-barrier-padding / stories15M / 8 | 1.16% | 2.41% | 7.36% | 21.26% / 17.65% / 30.29% |
| m1-h3-generation-pool-4t / stories110M / 4 | 2.92% | 3.14% | 4.34% | 8.60% / 12.41% / 15.81% |
| m1-h3-generation-pool-4t / stories15M / 4 | 5.84% | 3.91% | 3.60% | 27.59% / 7.81% / 7.19% |
| m1-h3-generation-pool-4t / stories15M / 5 | 1.27% | 4.23% | 5.26% | 9.55% / 6.34% / 8.23% |
| m1-h5-worker-arrivals / stories110M / 8 | 2.75% | 3.43% | 11.23% | 12.58% / 17.39% / 12.01% |
| m1-h5-worker-arrivals / stories15M / 8 | 6.40% | 15.95% | 9.77% | 34.48% / 30.67% / 18.97% |
| m1-h6-worker-qos / stories110M / 8 | 0.67% | 4.35% | 0.26% | 7.16% / 10.88% / 20.71% |
| m1-h6-worker-qos / stories15M / 8 | 3.57% | 3.60% | 9.47% | 9.29% / 13.67% / 23.08% |

## Uploaded results (server read-back)

These are uploaded **mean tok/s**, not the best-of-seven values used for
paired verdicts. Canary variation makes cross-report differences unsuitable
for accepting these small changes. Every row contains 24 attempt cells and
24 unchanged-canary cells; the table shows the original eight-thread focus.

| Stage | 15M / 8 | Canary | 110M / 8 | Canary |
|---|---:|---:|---:|---:|
| [baseline](https://swtune.engiware.com/reports/7154defd-d573-4d7e-ab42-8aa74c4f06a5) | 1002.58 | 1034.19 | 198.62 | 196.02 |
| [m1-h1-rope-barrier](https://swtune.engiware.com/reports/25797d4b-40d2-4101-adc7-08e70c5a39e8) | 895.66 | 823.57 | 166.60 | 165.16 |
| [m1-h2-row-tiles](https://swtune.engiware.com/reports/25a9a022-f043-4033-b4b1-244be65bad35) | 857.31 | 1034.15 | 175.60 | 193.29 |
| [m1-h3-generation-pool](https://swtune.engiware.com/reports/b05208c6-0121-49c5-85dd-7e98af37f202) | 1193.97 | 1072.32 | 222.52 | 213.43 |
| [m1-h4-barrier-padding](https://swtune.engiware.com/reports/b4b8da53-73d2-4138-9a1c-f78df92b71c6) | 906.65 | 979.83 | 209.62 | 210.61 |
| [m1-h3-generation-pool-4t](https://swtune.engiware.com/reports/5cb1dd42-790d-41cc-9ac7-f42e5569e2d5) | 1359.23 | 1013.25 | 236.96 | 235.69 |
| [m1-h5-worker-arrivals](https://swtune.engiware.com/reports/cb365c97-59e9-40c2-a007-08d5560045b8) | 880.33 | 855.08 | 190.26 | 188.23 |
| [m1-h6-worker-qos](https://swtune.engiware.com/reports/e91183c3-8da7-4eb5-b4d2-608d5143fb71) | 1216.12 | 924.28 | 214.92 | 207.32 |

Canary spread across these uploaded means:

| Cell | Minimum | Median | Maximum | Max/min |
|---|---:|---:|---:|---:|
| stories15M / 8 | 823.57 | 996.54 | 1072.32 | 1.302x |
| stories42M / 8 | 374.88 | 479.11 | 542.56 | 1.447x |
| stories110M / 8 | 165.16 | 201.67 | 235.69 | 1.427x |

## m1-reviewed-baseline: retain the measured baseline

Six code hypotheses and one unchanged-code follow-up were measured. None
established a repeatable improvement above the 1.10 acceptance ratio in every
paired round. The generation-wide pool remains a promising unproven candidate:
its four-thread 15M follow-up gained 4.58–10.94%, but its worst round did not
clear the bar and its 110M/4 guardrail ranged from -5.12% to +0.57%.

Retain baseline inference source and binary. All six attempts remain reachable
on `attempt/m1-h*` branches; the reviewed branch adds this record only. Thus
there is no retained implementation step with a regression to adjudicate.
Rejected variants' cell results remain visible in their complete uploaded
sweeps. The final documentation commit receives its own full 48-cell report,
`bench/autotune-m1/m1-reviewed-baseline-report.json`, with `kept=yes` and
`verdict=neutral`, to attest to the exact retained commit.

Stop reason: repeated non-wins, including the fresh QoS environment signal,
with no further actionable evidence from the final profile. No hardware ceiling
was proven and no inference speedup is claimed. Desktop interference and
canary drift limit small-effect conclusions; these are inconclusive speed
experiments, not proof that the ideas can never work. A useful next campaign
would measure the generation-wide pool on a dedicated quiet machine before
introducing further code changes.

Uploads use the user-updated default credential from `~/.swtune.yaml` and target
owner `tairov`, suite `llama2-mojo-optimizations`:
https://swtune.engiware.com/suites/7020a31f-3ead-4251-9531-017dc6f3de1a
The manifest, server read-back data, exact per-invocation A/B values, saved
binaries, profiles and correctness logs live in the outer repository under
`bench/autotune-m1/`. Earlier reports sent with the previous credential remain
in the previous suite; every campaign stage was reuploaded to the requested
suite and verified there. No credentials are stored in these artifacts.
