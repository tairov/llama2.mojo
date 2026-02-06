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
