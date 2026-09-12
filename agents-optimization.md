# Optimizing the Mojo 1.0 port — agent experiments

After `llama2.mojo` was migrated to Mojo 1.0 ([PR #101](https://github.com/tairov/llama2.mojo/pull/101), Claude Fable 5.1), the
code compiled and produced correct output but multi-threaded inference got
slower than single-threaded: on a 4-core box stories15M ran at ~100 tok/s with
`-j 1` and ~85 tok/s with `-j 4`, while llama2.c with OpenMP did ~300.

Two coding agents were asked, independently, to fix that. Both were given the
same migrated code, the same `run-tests.sh`, and the stories 15M/42M/110M models.

## Diagnosis (shared by both attempts)

- Single-threaded the forward pass is memory-bandwidth bound, same as
  llama2.c: the SIMD kernels were fine.
- In Mojo 1.0 `parallelize` moved into MAX (`max.algorithm`). Its pool
  threads sleep between dispatches and wake slowly, so every call cost
  ~200 µs on the VPS, and the forward pass issued ~37 calls per token
  (6 per layer + classifier). That is ~7 ms of overhead on a ~10 ms token.

## Attempt 1 — [PR #103](https://github.com/tairov/llama2.mojo/pull/103), Codex (Astr-6, extra-high reasoning)

*Avoid the dispatches.* Adds `parallel_worker_count()`: a kernel only gets a
thread-pool dispatch when each worker would have ≥ 262,144 floats of work;
otherwise it runs inline on the calling thread. RoPE is always serial. On
stories15M that leaves exactly one dispatch per token (the 32000×288
classifier). ~25 lines, no new machinery, plus a scalar-reference matmul test.

## Attempt 2 — [PR #102](https://github.com/tairov/llama2.mojo/pull/102), Claude (Fable 5.1)

*Make synchronization cheap instead.* `Transformer` starts `workers - 1`
persistent threads once (via `std.runtime.asyncrt.TaskGroup`); the calling
thread takes part as worker 0. Each worker owns a fixed slice of rows for
every matmul and of heads for RoPE/attention; stages are separated by a
~4 µs spin barrier on `Atomic[DType.int]` (31 barriers per token instead of
37 dispatches). Both rmsnorms are recomputed per worker into a private buffer
and residual adds / SiLU run on the rows the worker just produced, removing two
barriers per layer. Also sets `nelts = max(16, simd_width)`: the previous
`4 × simd_width` (64 on AVX-512) left a 32-element tail on every 288-wide row
and cost ~30% single-threaded.

## Results

Greedy decoding (`-t 0`), same prompt and seed, tok/s, best of 3. Output is
byte-identical between both PRs and llama2.c.

**Ubuntu 26.04 VPS, 4 vCPU Intel Xeon Skylake (AVX-512), Mojo 1.0.0**

| Model | master (#101) `-j 4` | llama2.c OpenMP 4 thr | [PR #103](https://github.com/tairov/llama2.mojo/pull/103) `-j 4` | [PR #102](https://github.com/tairov/llama2.mojo/pull/102) `-j 4` | #102 vs #103 |
|---|---:|---:|---:|---:|---:|
| 15M  | 87 | 292 | 138 | **327–400** | 2.4× |
| 42M  | 58 | 117 | 63  | **150** | 2.4× |
| 110M | 30 | 53  | 31  | **66**  | 2.1× |

Single-threaded (`-j 1`): #103 96 / 51 / 21, #102 154 / 54 / 22 — the 15M gap
is the `nelts` change. With 4 threads both llama2.c and #102 sit at the VM's
~20 GB/s memory-bandwidth ceiling, so ±10% run-to-run noise is normal.

**Apple M1 Max, Mojo 1.0.0, 8 workers**

| Model | [PR #102](https://github.com/tairov/llama2.mojo/pull/102) | [PR #103](https://github.com/tairov/llama2.mojo/pull/103) | Gain |
|---|---:|---:|---:|
| 15M  | 1,283 tok/s | 947 tok/s | 36% |
| 42M  | 506 tok/s   | 330 tok/s | 53% |
| 110M | 218 tok/s   | 118 tok/s | 84% |

## Takeaways

- #103 removes the overhead but caps throughput near single-core speed plus a
  parallel classifier; the cap is high on an M1 Max and low on a VPS.
- #102 scales with cores on both machines and beats llama2.c/OpenMP; the gain
  grows with model size because more of the token time is in matmuls that
  #103 runs serially.
- The dispatch cost, not the kernels, was the regression. Mojo 1.0's own
  `std.runtime.asyncrt` (the substrate `parallelize` is built on) is public in
  1.0.0 but already private on Modular's `main`, so the thread-start line in
  #102 will need another primitive when that ships; barriers and partitioning
  do not depend on it.
- Worth keeping from #103 regardless: the scalar-reference matmul test and the
  README notes on `-i`/`-j`.
