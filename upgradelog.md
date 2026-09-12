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
