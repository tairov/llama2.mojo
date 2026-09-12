# Performance: persistent worker pool (after the Mojo 1.0 upgrade)

The Mojo 1.0 port initially scaled badly across threads: on a 4-core box
stories15M ran at ~100 tok/s single-threaded but only ~85 tok/s with 4
workers (llama2.c with OpenMP: ~300-345 tok/s).

## Diagnosis (per-stage timers, stories15M)

- Single-threaded the forward pass is memory-bandwidth bound (~6 GB/s of
  weights per core), same as llama2.c.
- With 4 workers every `parallelize()` call cost ~200 us: MAX's pool threads
  sleep between dispatches and wake slowly. The forward pass issued ~37
  dispatches per token (RoPE alone went from 8 us to 1270 us per token), i.e.
  ~7 ms of pure dispatch overhead on a ~10 ms token.
- `nelts = 4 * simd_width` (64 floats on AVX-512) left a 32-element remainder
  on every 288-wide row and burned 12 zmm accumulators; ~30% slower than 16.

## Changes

- `Transformer` now owns a persistent worker pool: `workers - 1` threads are
  started once (on the first forward pass) via `std.runtime.asyncrt.TaskGroup`
  and spin until the next token; the calling thread takes part as worker 0.
- One forward pass = one "epoch". Every worker owns a fixed slice of rows for
  every matmul and a fixed slice of heads for RoPE/attention; stages that need
  the complete output of the previous stage are separated by a sense-reversing
  spin barrier (`Atomic[DType.int]`, ~4 us). 31 barriers per token on stories15M
  instead of 37 thread-pool dispatches.
- The attention-rmsnorm and FFN-rmsnorm outputs are computed redundantly by
  each worker into a private buffer (`dim` floats), which removes two barriers
  per layer.
- Residual adds and SiLU run on the rows the same worker just produced, so they
  need no extra synchronization.
- `nelts = max(16, simd_width_of[Float32]())`.
- `batch_matmul()` / `matmul()` keep their signatures (still MAX `parallelize`
  based, one dispatch per call) for the tests and external callers; the
  forward pass uses the serial `matmul_rows()` inside the pool.
- Workers are clamped to `parallelism_level()`; spin loops call `sched_yield`
  every 4096 iterations so an oversubscribed machine still makes progress.
- `Transformer.transformer()` is now `mut self` (starts the pool lazily) and
  raises if it is called with a different model dimension than the first call.

## Results (4 vCPU Intel Xeon Skylake VM, AVX-512, greedy decoding, 256 tokens)

| Model | llama2.c 1 thread | llama2.c 4 threads (OpenMP) | llama2.mojo `-j 1` | llama2.mojo `-j 4` before | llama2.mojo `-j 4` after |
|---|---|---|---|---|---|
| stories15M  | 105 tok/s | 292 tok/s | 143 tok/s | 87 tok/s | **400 tok/s** |
| stories42M  | 40 tok/s  | 117 tok/s | 51 tok/s  | 58 tok/s | **159 tok/s** |
| stories110M | 16 tok/s  | 53 tok/s  | 21 tok/s  | 30 tok/s | **58 tok/s**  |

Greedy output is byte-identical to llama2.c. At 4 threads both implementations
sit at the VM's ~20 GB/s memory-bandwidth ceiling, so run-to-run noise of
+-10% is normal.

---

# Upgrade Log: Mojo 0.26 -> 1.0

Toolchain: Mojo 1.0.0 (`modular` 26.5.0 from PyPI, `pip install modular`).
`parallelize` now ships in the MAX package (`max.algorithm`), so MAX is a
hard dependency of `llama2.mojo`.

## Scope

- Source files updated: 6 (`llama2.mojo`, all 5 files in `tests/`)
- Also touched: `run-tests.sh`, `README.md`
- All 5 test suites pass; `mojo llama2.mojo stories15M.bin ...` produces the
  same kind of output as before.

## Language / stdlib changes applied

- `fn` was removed from the language: every `fn` is now `def`.
- The standard library moved under the `std` namespace:
  `from algorithm import ...` -> `from std.algorithm import ...`,
  `import math` -> `from std import math`, etc.
- `parallelize` moved out of the stdlib: `from max.algorithm import parallelize`,
  and it is called with the closure as a runtime argument:
  `parallelize[f](n, workers)` -> `parallelize(f, n, workers)`.
- `sys.param_env` no longer exists; the unused `env_get_int` / `exit`
  imports were dropped.
- Pointers: `UnsafePointer` is unified into `Pointer`, and every unsafe
  operation is prefixed:
  - `UnsafePointer[Float32, MutExternalOrigin]` -> `Pointer[Float32, MutUntrackedOrigin]`
  - `ptr[i]` -> `ptr[unsafe_offset=i]`
  - `ptr + i` -> `ptr.unsafe_offset(i)`
  - `.load` / `.store` -> `.unsafe_load` / `.unsafe_store`
  - `.bitcast` -> `.unsafe_bitcast`
  - `memcpy` / `memset_zero` -> `unsafe_memcpy` / `unsafe_memset_zero`
  - `alloc[Float32](n)` -> `unsafe_alloc[Float32](n)` (from `std.memory.alloc`)
  - `List.steal_data()` -> `List.unsafe_take_allocation().unsafe_leak()`
- Closures: the `unified {mut}` effect and `@parameter` closures are gone.
  Nested `def`s now declare an explicit capture list, e.g.
  `def k[w: Int](i: Int) {imm}:` for kernels that only write through
  pointers, `{imm, mut acc}` for kernels that accumulate into a local, and
  `raises {mut} -> Int` for `argparse`.
- `@parameter for` -> `comptime for`.
- Strings are UTF-8 aware: `len(s)` -> `s.byte_length()` and `s[a:b]` ->
  `s[byte=a:b]`.
- Implicit `Int` <-> `Float32` conversions are gone (`ss / Float32(size)`,
  `Float32(atol(...))`, `UInt(...)` for `time_in_ms`), and `Float32(Bool)`
  is rejected (fixed in `test-matmul.mojo`).

## Refactor notes

### `llama2.mojo`

- `Matrix`: the owning variadic `__init__(out self, *dims: Int)` is
  miscompiled by Mojo 1.0 (garbage dims when many fields are built in one
  `__init__`, e.g. `RunState`), so it was replaced with explicit 1-, 2- and
  3-dim constructors. The unused variadic view constructor was removed; the
  `Matrix(ptr, List[Int])` view constructor is unchanged.
- The attention inner loops used closures nested inside the `parallelize`
  closure, which Mojo 1.0 cannot capture from loop-scoped locals. They were
  replaced with two module-level SIMD helpers, `dot()` and `axpy()`, used
  by `loop_over_heads`.
- `rmsnorm` / `softmax` no longer take their pointer argument as `mut`
  (the pointee is mutable through `MutUntrackedOrigin`), which lets them be
  called from `{imm}` closures.
- `math.sqrt[dtype=..., width=1](x)` -> `math.sqrt(x)`.
- Local `char` / `str` were renamed to `ch` / `pair`.

### `tests/`

- Same `def` / `std.` / `byte_length()` changes as above.
- `test-matmul.mojo`: `Float32(row == col)` -> `Float32(1.0) if row == col else Float32(0.0)`.

### `run-tests.sh`

- `((PASSED++))` returns a non-zero status when the counter is 0, which
  aborted the script under `set -e` after the first passing test. Counters
  now use `PASSED=$((PASSED + 1))`.

### Host setup used for this migration

```bash
uv venv --python 3.13 ~/.modular-venv
uv pip install --python ~/.modular-venv/bin/python modular   # Mojo 1.0.0 + MAX 26.5.0
export PATH="$HOME/.modular-venv/bin:$PATH"
sudo apt-get install -y gcc      # needed to link Mojo executables
```

---

# Upgrade Log: Mojo 0.25 -> 0.26

## Scope

- Source files updated: 5
- Diff size: 156 insertions, 97 deletions
- Files:
  - `llama2.mojo`
  - `tests/test-calculations.mojo`
  - `tests/test-llama2.mojo`
  - `tests/test-tokenizer.mojo`
  - `tests/test-transformer.mojo`

## Refactor and Upgrade Notes

### `llama2.mojo`

- Migrated compile-time declarations from `alias` to `comptime` where required by Mojo 0.26 (`NUM_CONFIG_INT`, `nelts`, pointer alias, and `wrap` constants).
- Updated pointer origin type to Mojo 0.26 style:
  - `UnsafePointer[Float32, MutOrigin.external]` -> `UnsafePointer[Float32, MutExternalOrigin]`.
- Replaced pointer offset usage with pointer arithmetic:
  - `.offset(...)` -> `ptr + ...` across matrix slicing and vectorized kernels.
- Updated vectorized kernel definitions and calls for 0.26:
  - Kernel closures now use `fn ... unified {mut}`.
  - `vectorize` call form updated to `vectorize[nelts](size, kernel_fn)`.
- Kept behavior but adapted string handling to slice-based indexing:
  - Single-character extraction now uses ranges like `String(text[pos:pos+1])`.
  - Token checks in `get_token_str` and CLI parsing now compare `String(...)` slices.
- Added helper string functions:
  - `str_concat(a, b)`
  - `string_compare(a, b)`
- Stabilized token-per-second output:
  - Avoids integer-division truncation and divide-by-zero by checking elapsed time and computing with `Float32`.

### `tests/test-calculations.mojo`

- Updated list initialization for sampling counters:
  - `List[Int](N)` -> `List[Int]()` + explicit `append(0)` loop.
- Minor formatting cleanup.

### `tests/test-llama2.mojo`

- Added model file discovery helpers:
  - `file_exists(path)`
  - `resolve_model_path()`
- Tests now skip gracefully when `stories15M.bin` is not available locally.
- Replaced hardcoded model path usage with resolved path in `Config`, `TransformerWeights`, and `RunState` tests.

### `tests/test-tokenizer.mojo`

- Removed unused import (`str_concat`) to keep test module clean with 0.26 checks.
- Minor formatting cleanup.

### `tests/test-transformer.mojo`

- Added `file_exists` and `resolve_model_path` helpers (same approach as `test-llama2`).
- `test_rope_rotation` now skips cleanly when `stories15M.bin` is missing.
- Added `os` import for `HOME`-based fallback path resolution.
