from std.algorithm import vectorize
from max.algorithm import parallelize
from std.runtime.asyncrt import TaskGroup, parallelism_level
from std.atomic import Atomic
from std.ffi import external_call
from std.math import ceildiv
from std.collections import List, Dict
from std.memory import unsafe_memset_zero, unsafe_memcpy, stack_allocation
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.utils import StaticTuple
from std.sys import argv
from std.sys.info import simd_width_of, size_of, num_performance_cores
from std import math
from std import os
from std import random
from std import time

comptime NUM_CONFIG_INT = 7

# Vector width for the kernels: at least 16 floats so narrow ISAs (NEON) still
# get some ILP, but no wider than the native SIMD width. Wider (e.g. 4x on
# AVX-512 = 64) leaves large remainders on 288-wide rows and was ~30% slower.
comptime nelts = max(16, simd_width_of[Float32]())
comptime BufferPtrFloat32 = Pointer[Float32, MutUntrackedOrigin]

struct Matrix(Movable):
    var data: BufferPtrFloat32
    var allocated: Int
    var dims: List[Int]

    # Owning constructors. Mojo 1.0 miscompiles a variadic `*dims: Int`
    # initializer here (garbage dims when many fields are built in one
    # __init__), so the 1/2/3-dim forms are spelled out explicitly.
    def __init__(out self, d0: Int):
        self.dims = List[Int]()
        self.dims.append(d0)
        self.data = unsafe_alloc[Float32](d0)
        self.allocated = 1

    def __init__(out self, d0: Int, d1: Int):
        self.dims = List[Int]()
        self.dims.append(d0)
        self.dims.append(d1)
        self.data = unsafe_alloc[Float32](d0 * d1)
        self.allocated = 1

    def __init__(out self, d0: Int, d1: Int, d2: Int):
        self.dims = List[Int]()
        self.dims.append(d0)
        self.dims.append(d1)
        self.dims.append(d2)
        self.data = unsafe_alloc[Float32](d0 * d1 * d2)
        self.allocated = 1

    # Constructor for creating views/slices without allocation
    def __init__(out self, ptr: BufferPtrFloat32, var dims: List[Int]):
        self.data = ptr
        self.allocated = 0
        self.dims = dims^

    @always_inline
    def alloc(mut self, fill: Int = 0):
        self.data = unsafe_alloc[Float32](self.size())
        self.allocated = 1
        if fill == 1:
            self.zero()

    @always_inline
    def size(self) -> Int:
        var s = 1
        for i in range(len(self.dims)):
            s *= self.dims[i]
        return s

    @always_inline
    def alloc_zero(mut self):
        self.alloc(1)

    @always_inline
    def zero(mut self):
        unsafe_memset_zero(self.data, self.size())

    @always_inline
    def set_buf_ptr(mut self, ptr: BufferPtrFloat32):
        self.data = ptr

    @always_inline
    def __getitem__(self, x: Int) -> Float32:
        return self.data[unsafe_offset=x]

    @always_inline
    def __getitem__(self, y: Int, x: Int) -> Float32:
        # 2D access: y * cols + x
        return self.data[unsafe_offset=y * self.dims[len(self.dims)-1] + x]

    @always_inline
    def __getitem__(self, z: Int, y: Int, x: Int) -> Float32:
        # 3D access: z * (rows * cols) + y * cols + x
        var cols = self.dims[len(self.dims)-1]
        var rows = self.dims[len(self.dims)-2]
        return self.data[unsafe_offset=z * (rows * cols) + y * cols + x]

    @always_inline
    def __setitem__(self, x: Int, val: Float32):
        self.data[unsafe_offset=x] = val

    @always_inline
    def __setitem__(self, y: Int, x: Int, val: Float32):
        self.data[unsafe_offset=y * self.dims[len(self.dims)-1] + x] = val

    @always_inline
    def rank(self) -> Int:
        return len(self.dims)

    # Slice method: return UnsafePointer for a specific layer/row depending on rank
    @always_inline
    def slice(self, idx: Int) -> BufferPtrFloat32:
        # If 3D (rank 3), slice the first dim (layer) -> offset = idx * rows * cols
        # If 2D (rank 2), slice the first dim (row) -> offset = idx * cols
        # If 1D (rank 1), slice the element -> offset = idx
        if len(self.dims) > 2:
             var stride = self.dims[1] * self.dims[2]
             return self.data.unsafe_offset(idx * stride)
        elif len(self.dims) > 1:
             return self.data.unsafe_offset(idx * self.dims[1])
        else:
             return self.data.unsafe_offset(idx)

    # Slice method: return UnsafePointer for a specific layer and row
    # This assumes standard 3D mapping or 2D mapping where layer=0
    @always_inline
    def slice(self, idx1: Int, idx2: Int) -> BufferPtrFloat32:
        var cols = self.dims[len(self.dims)-1]
        var rows = self.dims[len(self.dims)-2]
        var offset = idx1 * rows * cols + idx2 * cols
        return self.data.unsafe_offset(offset)

    @always_inline
    def dim(self, idx: Int) -> Int:
        if idx < len(self.dims):
            return self.dims[idx]
        return 0

    @always_inline
    def num_elements(self) -> Int:
        return self.size()

def wrap(token: String) -> String:
    comptime a = String("\\n")
    comptime b = String("\\t")
    comptime c = String("'")
    comptime d = String('"')
    if token == a:
        return String("\n")
    if token == b:
        return String("\t")
    if token == c:
        return String("'")
    if token == d:
        return String('"')

    return token

def str_concat(a: String, b: String) -> String:
    return a + b

def string_compare(a: String, b: String) -> Int:
    if a < b:
        return -1
    if a > b:
        return 1
    return 0

def string_from_bytes(var bytes: List[UInt8]) -> String:
    var result = String("")
    for i in range(len(bytes)):
        result += chr(Int(bytes[i]))
    return result

struct Tokenizer:
    var vocab: List[String]
    var vocab_scores: List[Float32]
    var max_token_length: Int
    var vocab_size: Int
    var map_vocab_to_index: Dict[String, Int]

    def __init__(out self, vocab_size: Int, filename: String) raises:
        with open(filename, "r") as f:
            def read_bytes_as[dtype: DType](size: Int) raises {imm} -> SIMD[dtype, 1]:
                var bytes = f.read_bytes(size)
                var result = bytes.unsafe_ptr().unsafe_bitcast[SIMD[dtype, 1]]()[unsafe_offset=0]
                _ = bytes
                return result

            self.vocab_size = vocab_size
            self.vocab_scores = List[Float32]()
            self.vocab = List[String]()

            var max_token_bytes = f.read_bytes(4)
            var max_token_ptr = max_token_bytes.unsafe_ptr().unsafe_bitcast[Int32]()
            self.max_token_length = Int(max_token_ptr[unsafe_offset=0])

            self.map_vocab_to_index = Dict[String, Int]()

            for i in range(self.vocab_size):
                var score = read_bytes_as[DType.float32](4)
                var slen = read_bytes_as[DType.int32](4)
                var token = string_from_bytes(f.read_bytes(Int(slen)))
                self.vocab.append(token)
                self.vocab_scores.append(score)
                self.map_vocab_to_index[self.vocab[i]] = i

    def find(self, token_o: String) -> Int:
        var token = wrap(token_o)

        # Handle TinyLlama specific newline mapping
        if token == "\n":
            var idx = self.map_vocab_to_index.find("<0x0A>")
            if idx: return idx.value()

        var index = self.map_vocab_to_index.find(token)
        return index.or_else(-1)

    def print_tokens(self, n: Int):
        var count = min(n, self.vocab_size)
        print("First", count, "tokens:")
        for i in range(count):
            print(i, ":", self.vocab[i])

struct Config:
    var dim: Int
    var kv_dim: Int
    var hidden_dim: Int
    var n_layers: Int
    var n_heads: Int
    var n_kv_heads: Int
    var kv_mul: Int
    var vocab_size: Int
    var seq_len: Int
    var head_size: Int
    var shared_weights: Bool

    def __init__(out self, filename: String, print_config: Bool) raises:
        var f = open(filename, "r")
        var bytes_of_config_params = NUM_CONFIG_INT * size_of[DType.int32]()
        var config_data_raw = f.read_bytes(bytes_of_config_params)
        f.close()
        var int32_ptr = config_data_raw.unsafe_take_allocation().unsafe_leak().unsafe_bitcast[Int32]()
        self.dim = Int(int32_ptr[unsafe_offset=0])
        self.hidden_dim = Int(int32_ptr[unsafe_offset=1])
        self.n_layers = Int(int32_ptr[unsafe_offset=2])
        self.n_heads = Int(int32_ptr[unsafe_offset=3])
        self.n_kv_heads = Int(int32_ptr[unsafe_offset=4])
        self.vocab_size = Int(int32_ptr[unsafe_offset=5])
        self.seq_len = Int(int32_ptr[unsafe_offset=6])
        self.head_size = self.dim // self.n_heads
        self.kv_dim = (self.n_kv_heads * self.dim) // self.n_heads
        self.kv_mul = self.n_heads // self.n_kv_heads
        self.shared_weights = self.vocab_size > 0
        if not self.shared_weights:
            self.vocab_size = -self.vocab_size

        if print_config:
            print("config: dim, hidden_dim", self.dim, self.hidden_dim)
            print("config: n_layers, n_heads", self.n_layers, self.n_heads)
            print("config: vocab_size, seq_len", self.vocab_size, self.seq_len)
            print("config: head_size", self.head_size)
            print("config: kv_dim, kv_mul", self.kv_dim, self.kv_mul)

struct RunState:
    var x: Matrix  # activation at current time stamp (dim,)
    var xb: Matrix  # same, but inside a residual branch (dim,)
    var xb2: Matrix  # an additional buffer just for convenience (dim,)
    var hb: Matrix  # buffer for hidden dimension in the ffn (hidden_dim,)
    var hb2: Matrix  # buffer for hidden dimension in the ffn (hidden_dim,)
    var q: Matrix  # query (dim,)
    var att: Matrix  # buffer for scores/attention values (n_heads, seq_len)
    var logits: Matrix  # output logits
    var key_cache: Matrix  # (layer, seq_len, dim)
    var value_cache: Matrix  # (layer, seq_len, dim)

    def __init__(out self, config: Config) raises:
        self.x = Matrix(config.dim)
        self.xb = Matrix(config.dim)
        self.xb2 = Matrix(config.dim)
        self.hb = Matrix(config.hidden_dim)
        self.hb2 = Matrix(config.hidden_dim)
        self.q = Matrix(config.dim)

        self.logits = Matrix(config.vocab_size)
        self.att = Matrix(config.n_heads, config.seq_len)

        self.key_cache = Matrix(config.n_layers, config.seq_len, config.kv_dim)
        self.value_cache = Matrix(config.n_layers, config.seq_len, config.kv_dim)

struct TransformerWeights:
    var token_embedding_table: Matrix
    var freq_cis_real: Matrix
    var freq_cis_imag: Matrix
    var rms_att_weight: Matrix
    var wq: Matrix
    var wk: Matrix
    var wv: Matrix
    var wo: Matrix
    var rms_ffn_weight: Matrix
    var w1: Matrix
    var w3: Matrix
    var w2: Matrix
    var rms_final_weight: Matrix
    var wcls: Matrix

    def __init__(out self, file_name: String, config: Config) raises:
        var bytes_read = 0
        var f = open(file_name, "r")

        _ = f.read_bytes(NUM_CONFIG_INT * size_of[DType.int32]())
        bytes_read += NUM_CONFIG_INT * size_of[DType.int32]()
        def read_weights(*dims: Int) raises {mut} -> Matrix:
            var dim_list = List[Int]()
            var num_elements = 1
            for i in range(len(dims)):
                dim_list.append(dims[i])
                num_elements *= dims[i]

            var tmp = f.read_bytes(
                num_elements * size_of[Float32]()
            )
            bytes_read += num_elements * size_of[Float32]()
            var data = tmp.unsafe_take_allocation().unsafe_leak().unsafe_bitcast[Float32]()
            return Matrix(data, dim_list^)

        self.token_embedding_table = read_weights(config.vocab_size, config.dim)
        self.rms_att_weight = read_weights(config.n_layers, config.dim)
        self.wq = read_weights(config.n_layers, config.dim, config.dim)
        self.wk = read_weights(config.n_layers, config.kv_dim, config.dim)
        self.wv = read_weights(config.n_layers, config.kv_dim, config.dim)
        self.wo = read_weights(config.n_layers, config.dim, config.dim)
        self.rms_ffn_weight = read_weights(config.n_layers, config.dim)
        self.w1 = read_weights(config.n_layers, config.hidden_dim, config.dim)
        self.w2 = read_weights(config.n_layers, config.dim, config.hidden_dim)
        self.w3 = read_weights(config.n_layers, config.hidden_dim, config.dim)
        self.rms_final_weight = read_weights(config.dim)
        self.freq_cis_real = read_weights(config.seq_len, config.head_size // 2)
        self.freq_cis_imag = read_weights(config.seq_len, config.head_size // 2)


        if config.shared_weights:
            # Copy dims properly for view
            var dims = self.token_embedding_table.dims.copy()
            self.wcls = Matrix(self.token_embedding_table.data, dims^)
            # Not own data
            self.wcls.allocated = 0
        else:
            self.wcls = read_weights(config.vocab_size, config.dim)

        f.close()

        print(
            "Total bytes read:",
            bytes_read,
            "Estimated checkpoint size: ",
            bytes_read // 1024 // 1024,
            "MB",
        )

@always_inline
def rmsnorm(
    o: BufferPtrFloat32,
    x: BufferPtrFloat32,
    weight: BufferPtrFloat32,
    size: Int
):
    # Calculate sum of squares
    var tmp_ptr = stack_allocation[nelts, Float32]()
    tmp_ptr.unsafe_store[width=nelts](0, SIMD[DType.float32, nelts](0))

    def _sum2[_nelts: Int](j: Int) {imm}:
        var val = x.unsafe_load[width=_nelts](j) ** 2
        var curr = tmp_ptr.unsafe_load[width=_nelts](0)
        tmp_ptr.unsafe_store[width=_nelts](0, curr + val)

    vectorize[nelts](size, _sum2)

    var ss: Float32 = tmp_ptr.unsafe_load[width=nelts](0).reduce_add()
    ss = ss / Float32(size) + 1e-5
    ss = 1.0 / math.sqrt(ss)

    # Normalize and scale
    def _norm[_nelts: Int](j: Int) {imm}:
        var val = weight.unsafe_load[width=_nelts](j) * ss * x.unsafe_load[width=_nelts](j)
        o.unsafe_store[width=_nelts](j, val)

    vectorize[nelts](size, _norm)

@always_inline
def softmax(x: BufferPtrFloat32, size: Int):
    softmax(x, 0, size)

@always_inline
def softmax(x: BufferPtrFloat32, start: Int, end: Int):
    var max_val: Float32 = -1e9

    def _max[_nelts: Int](ii: Int) {imm, mut max_val}:
        var val = x.unsafe_load[width=_nelts](start + ii).reduce_max()
        if val > max_val:
            max_val = val

    vectorize[nelts](end - start, _max)

    var acc_ptr = stack_allocation[nelts, Float32]()
    acc_ptr.unsafe_store[width=nelts](0, SIMD[DType.float32, nelts](0))

    def _exp[_nelts: Int](ii: Int) {imm}:
        var val = math.exp(x.unsafe_load[width=_nelts](start + ii) - max_val)
        x.unsafe_store[width=_nelts](start + ii, val)
        var curr = acc_ptr.unsafe_load[width=_nelts](0)
        acc_ptr.unsafe_store[width=_nelts](0, curr + val)

    vectorize[nelts](end - start, _exp)

    var ssum = acc_ptr.unsafe_load[width=nelts](0).reduce_add()

    def _norm[_nelts: Int](ii: Int) {imm}:
        x.unsafe_store[width=_nelts](
            start + ii, x.unsafe_load[width=_nelts](start + ii) / ssum
        )

    vectorize[nelts](end - start, _norm)

@always_inline
def matmul_rows[
    n: Int
](
    C: StaticTuple[BufferPtrFloat32, n],
    A: BufferPtrFloat32,
    B: StaticTuple[BufferPtrFloat32, n],
    row_start: Int,
    row_end: Int,
    cols: Int,
):
    # C[k][i] = B[k][i, :] . A  for i in [row_start, row_end), k < n.
    # Serial; callers decide how rows are split across threads.
    def compute_row(i: Int) {imm}:
        var tmp_ptr = stack_allocation[n * nelts, Float32]()

        comptime for k in range(n):
            tmp_ptr.unsafe_store[width=nelts](k * nelts, SIMD[DType.float32, nelts](0))

        var row_offset = i * cols

        def dot[_nelts: Int](j: Int) {imm}:
            var a = A.unsafe_load[width=_nelts](j)

            comptime for k in range(n):
                var val = a * B[k].unsafe_load[width=_nelts](row_offset + j)
                var curr = tmp_ptr.unsafe_load[width=_nelts](k * nelts)
                tmp_ptr.unsafe_store[width=_nelts](k * nelts, curr + val)

        vectorize[nelts](cols, dot)

        comptime for k in range(n):
            C[k].unsafe_store(i, tmp_ptr.unsafe_load[width=nelts](k * nelts).reduce_add())

    for i in range(row_start, row_end):
        compute_row(i)

@always_inline
def batch_matmul[
    n: Int
](
    C: StaticTuple[BufferPtrFloat32, n],
    A: BufferPtrFloat32,
    B: StaticTuple[BufferPtrFloat32, n],
    rows: Int,
    cols: Int,
    workers: Int,
):
    # Standalone parallel matmul (one thread-pool dispatch per call). The
    # forward pass uses matmul_rows() inside the persistent worker pool instead.
    if workers <= 1:
        matmul_rows[n](C, A, B, 0, rows, cols)
        return

    var chunk = ceildiv(rows, workers)

    def compute_chunk(w: Int) {imm}:
        matmul_rows[n](C, A, B, w * chunk, min((w + 1) * chunk, rows), cols)

    parallelize(compute_chunk, workers, workers)

@always_inline
def matmul(C: BufferPtrFloat32, A: BufferPtrFloat32, B: BufferPtrFloat32, rows: Int, cols: Int, workers: Int) raises:
    batch_matmul[1](
        StaticTuple[BufferPtrFloat32, 1](C),
        A,
        StaticTuple[BufferPtrFloat32, 1](B),
        rows,
        cols,
        workers,
    )

@always_inline
def add(dest: BufferPtrFloat32, src: BufferPtrFloat32, size: Int):
    def add_kernel[_nelts: Int](i: Int) {imm}:
        var a = dest.unsafe_load[width=_nelts](i)
        var b = src.unsafe_load[width=_nelts](i)
        dest.unsafe_store[width=_nelts](i, a + b)
    vectorize[nelts](size, add_kernel)

@always_inline
def dot(a: BufferPtrFloat32, b: BufferPtrFloat32, size: Int) -> Float32:
    # Dot product of two vectors of length `size`
    var acc: Float32 = 0.0

    def dot_kernel[_nelts: Int](i: Int) {imm, mut acc}:
        acc += (
            a.unsafe_load[width=_nelts](i) * b.unsafe_load[width=_nelts](i)
        ).reduce_add()

    vectorize[nelts](size, dot_kernel)
    return acc

@always_inline
def axpy(dest: BufferPtrFloat32, src: BufferPtrFloat32, scale: Float32, size: Int):
    # dest += scale * src, over `size` elements
    def axpy_kernel[_nelts: Int](i: Int) {imm}:
        var val = dest.unsafe_load[width=_nelts](i) + scale * src.unsafe_load[width=_nelts](i)
        dest.unsafe_store[width=_nelts](i, val)

    vectorize[nelts](size, axpy_kernel)

@always_inline
def silu_mul(hb: BufferPtrFloat32, hb2: BufferPtrFloat32, size: Int):
    # hb = silu(hb) * hb2, over `size` elements
    def silu_kernel[_nelts: Int](i: Int) {imm}:
        var v = hb.unsafe_load[width=_nelts](i)
        var sv = v * (1.0 / (1.0 + math.exp(-v)))
        hb.unsafe_store[width=_nelts](i, sv * hb2.unsafe_load[width=_nelts](i))

    vectorize[nelts](size, silu_kernel)

@always_inline
def rope_head(
    q_ptr: BufferPtrFloat32,
    k_ptr: BufferPtrFloat32,
    freq_cis_real_row: BufferPtrFloat32,
    freq_cis_imag_row: BufferPtrFloat32,
    h: Int,
    head_size: Int,
    n_kv_heads: Int,
):
    # RoPE rotation of query head h (and key head h, if it exists)
    for j in range(0, head_size, 2):
        var fcr = freq_cis_real_row[unsafe_offset=j // 2]
        var fci = freq_cis_imag_row[unsafe_offset=j // 2]

        var q_idx = h * head_size + j
        var q0 = q_ptr[unsafe_offset=q_idx]
        var q1 = q_ptr[unsafe_offset=q_idx + 1]
        q_ptr[unsafe_offset=q_idx] = q0 * fcr - q1 * fci
        q_ptr[unsafe_offset=q_idx + 1] = q0 * fci + q1 * fcr

        if h < n_kv_heads:
            var k0 = k_ptr[unsafe_offset=q_idx]
            var k1 = k_ptr[unsafe_offset=q_idx + 1]
            k_ptr[unsafe_offset=q_idx] = k0 * fcr - k1 * fci
            k_ptr[unsafe_offset=q_idx + 1] = k0 * fci + k1 * fcr

# Everything one forward pass needs, as plain pointers and ints, so it can be
# handed to the worker threads by value.
struct ForwardArgs(ImplicitlyCopyable):
    var token: Int
    var pos: Int
    var dim: Int
    var hidden_dim: Int
    var head_size: Int
    var kv_dim: Int
    var kv_mul: Int
    var n_layers: Int
    var n_heads: Int
    var n_kv_heads: Int
    var seq_len: Int
    var vocab_size: Int
    var x: BufferPtrFloat32
    var xb: BufferPtrFloat32
    var xb2: BufferPtrFloat32
    var hb: BufferPtrFloat32
    var hb2: BufferPtrFloat32
    var q: BufferPtrFloat32
    var att: BufferPtrFloat32
    var logits: BufferPtrFloat32
    var key_cache: BufferPtrFloat32
    var value_cache: BufferPtrFloat32
    var token_embedding_table: BufferPtrFloat32
    var freq_cis_real: BufferPtrFloat32
    var freq_cis_imag: BufferPtrFloat32
    var rms_att_weight: BufferPtrFloat32
    var wq: BufferPtrFloat32
    var wk: BufferPtrFloat32
    var wv: BufferPtrFloat32
    var wo: BufferPtrFloat32
    var rms_ffn_weight: BufferPtrFloat32
    var w1: BufferPtrFloat32
    var w2: BufferPtrFloat32
    var w3: BufferPtrFloat32
    var rms_final_weight: BufferPtrFloat32
    var wcls: BufferPtrFloat32

    def __init__(out self, token: Int, pos: Int, config: Config, state: RunState, weights: TransformerWeights):
        self.token = token
        self.pos = pos
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.head_size = config.head_size
        self.kv_dim = config.kv_dim
        self.kv_mul = config.kv_mul
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.seq_len = config.seq_len
        self.vocab_size = config.vocab_size
        self.x = state.x.data
        self.xb = state.xb.data
        self.xb2 = state.xb2.data
        self.hb = state.hb.data
        self.hb2 = state.hb2.data
        self.q = state.q.data
        self.att = state.att.data
        self.logits = state.logits.data
        self.key_cache = state.key_cache.data
        self.value_cache = state.value_cache.data
        self.token_embedding_table = weights.token_embedding_table.data
        self.freq_cis_real = weights.freq_cis_real.data
        self.freq_cis_imag = weights.freq_cis_imag.data
        self.rms_att_weight = weights.rms_att_weight.data
        self.wq = weights.wq.data
        self.wk = weights.wk.data
        self.wv = weights.wv.data
        self.wo = weights.wo.data
        self.rms_ffn_weight = weights.rms_ffn_weight.data
        self.w1 = weights.w1.data
        self.w2 = weights.w2.data
        self.w3 = weights.w3.data
        self.rms_final_weight = weights.rms_final_weight.data
        self.wcls = weights.wcls.data

@always_inline
def chunk_start(total: Int, wid: Int, workers: Int) -> Int:
    return min(wid * ceildiv(total, workers), total)

@always_inline
def chunk_end(total: Int, wid: Int, workers: Int) -> Int:
    return min((wid + 1) * ceildiv(total, workers), total)

@always_inline
def spin_wait(spins: Int):
    # Yield the core after a while so an oversubscribed machine still makes progress
    if spins % 4096 == 4095:
        _ = external_call["sched_yield", Int32]()

# State shared by the main thread and the persistent worker threads.
# Heap-allocated and never moved: the worker coroutines hold a pointer to it.
struct PoolShared:
    var workers: Int
    var epoch: Atomic[DType.int]      # bumped once per forward pass
    var done: Atomic[DType.int]       # workers finished with the current pass
    var bar_count: Atomic[DType.int]  # spin barrier: arrivals
    var bar_gen: Atomic[DType.int]    # spin barrier: generation
    var stop: Atomic[DType.int]
    var args: ForwardArgs
    var xbp: BufferPtrFloat32         # per-worker private rmsnorm buffers (workers * dim)
    var xbp_dim: Int
    var ids: List[Int]

    def __init__(out self, workers: Int, args: ForwardArgs):
        self.workers = workers
        self.epoch = Atomic[DType.int](0)
        self.done = Atomic[DType.int](0)
        self.bar_count = Atomic[DType.int](0)
        self.bar_gen = Atomic[DType.int](0)
        self.stop = Atomic[DType.int](0)
        self.args = args
        self.xbp = unsafe_alloc[Float32](workers * args.dim)
        self.xbp_dim = args.dim
        self.ids = List[Int]()
        for i in range(workers):
            self.ids.append(i)

comptime PoolSharedPtr = Pointer[PoolShared, MutUntrackedOrigin]

@always_inline
def barrier(sh: PoolSharedPtr):
    # Sense-reversing spin barrier across all workers (no-op for 1 worker)
    if sh[].workers == 1:
        return
    var gen = sh[].bar_gen.load()
    if sh[].bar_count.fetch_add(1) + 1 == sh[].workers:
        sh[].bar_count.store(0)
        sh[].bar_gen.store(gen + 1)
    else:
        var spins = 0
        while sh[].bar_gen.load() == gen:
            spin_wait(spins)
            spins += 1

@always_inline
def attention_head(a: ForwardArgs, l: Int, h: Int):
    # Single-head attention for the current position; writes xb[h*head_size:]
    var loff = l * a.seq_len * a.kv_dim
    var q_offset = h * a.head_size
    var att_offset = h * a.seq_len
    var kv_head_offset = (h // a.kv_mul) * a.head_size
    var sqrt_head_size = math.sqrt(Float32(a.head_size))

    for t in range(a.pos + 1):
        var k_offset = loff + t * a.kv_dim + kv_head_offset
        var score = dot(
            a.q.unsafe_offset(q_offset),
            a.key_cache.unsafe_offset(k_offset),
            a.head_size,
        )
        a.att[unsafe_offset=att_offset + t] = score / sqrt_head_size

    softmax(a.att, att_offset, att_offset + a.pos + 1)

    var xb_head = a.xb.unsafe_offset(q_offset)
    unsafe_memset_zero(xb_head, a.head_size)
    for t in range(a.pos + 1):
        var v_offset = loff + t * a.kv_dim + kv_head_offset
        axpy(
            xb_head,
            a.value_cache.unsafe_offset(v_offset),
            a.att[unsafe_offset=att_offset + t],
            a.head_size,
        )

def forward_worker(a: ForwardArgs, xbp: BufferPtrFloat32, wid: Int, sh: PoolSharedPtr):
    # One worker's share of a full forward pass. Every worker owns a fixed
    # slice of rows (matmuls) and heads (RoPE/attention); stages that need the
    # complete output of the previous stage are separated by a barrier.
    var workers = sh[].workers
    var dim = a.dim
    var hidden_dim = a.hidden_dim
    var kv_dim = a.kv_dim
    var head_size = a.head_size

    var d0 = chunk_start(dim, wid, workers)
    var d1 = chunk_end(dim, wid, workers)
    var kv0 = chunk_start(kv_dim, wid, workers)
    var kv1 = chunk_end(kv_dim, wid, workers)
    var f0 = chunk_start(hidden_dim, wid, workers)
    var f1 = chunk_end(hidden_dim, wid, workers)
    var h0 = chunk_start(a.n_heads, wid, workers)
    var h1 = chunk_end(a.n_heads, wid, workers)
    var v0 = chunk_start(a.vocab_size, wid, workers)
    var v1 = chunk_end(a.vocab_size, wid, workers)

    # Copy the token embedding into x
    if wid == 0:
        unsafe_memcpy(dest=a.x, src=a.token_embedding_table.unsafe_offset(a.token * dim), count=dim)
    barrier(sh)

    # Pluck out the "pos" row of freq_cis_real and freq_cis_imag
    var freq_cis_real_row = a.freq_cis_real.unsafe_offset(a.pos * (head_size // 2))
    var freq_cis_imag_row = a.freq_cis_imag.unsafe_offset(a.pos * (head_size // 2))

    for l in range(a.n_layers):
        # Attention rmsnorm: each worker computes the full (tiny) vector into
        # its private buffer, so no barrier is needed before the QKV matmul
        rmsnorm(xbp, a.x, a.rms_att_weight.unsafe_offset(l * dim), dim)

        var k_ptr = a.key_cache.unsafe_offset(l * a.seq_len * kv_dim + a.pos * kv_dim)
        var v_ptr = a.value_cache.unsafe_offset(l * a.seq_len * kv_dim + a.pos * kv_dim)
        var wq = a.wq.unsafe_offset(l * dim * dim)
        var wk = a.wk.unsafe_offset(l * kv_dim * dim)
        var wv = a.wv.unsafe_offset(l * kv_dim * dim)

        # QKV matmuls over this worker's rows
        if kv_dim == dim:
            matmul_rows[3](
                StaticTuple[BufferPtrFloat32, 3](a.q, k_ptr, v_ptr),
                xbp,
                StaticTuple[BufferPtrFloat32, 3](wq, wk, wv),
                d0, d1, dim,
            )
        else:
            matmul_rows[1](
                StaticTuple[BufferPtrFloat32, 1](a.q),
                xbp,
                StaticTuple[BufferPtrFloat32, 1](wq),
                d0, d1, dim,
            )
            matmul_rows[2](
                StaticTuple[BufferPtrFloat32, 2](k_ptr, v_ptr),
                xbp,
                StaticTuple[BufferPtrFloat32, 2](wk, wv),
                kv0, kv1, dim,
            )
        barrier(sh)

        # RoPE over this worker's heads
        for h in range(h0, h1):
            rope_head(a.q, k_ptr, freq_cis_real_row, freq_cis_imag_row, h, head_size, a.n_kv_heads)
        barrier(sh)

        # Multihead attention over this worker's heads
        for h in range(h0, h1):
            attention_head(a, l, h)
        barrier(sh)

        # Output projection + residual, over this worker's rows
        matmul_rows[1](
            StaticTuple[BufferPtrFloat32, 1](a.xb2),
            a.xb,
            StaticTuple[BufferPtrFloat32, 1](a.wo.unsafe_offset(l * dim * dim)),
            d0, d1, dim,
        )
        add(a.x.unsafe_offset(d0), a.xb2.unsafe_offset(d0), d1 - d0)
        barrier(sh)

        # FFN rmsnorm (private buffer again), then w1/w3 + SiLU over this worker's rows
        rmsnorm(xbp, a.x, a.rms_ffn_weight.unsafe_offset(l * dim), dim)
        matmul_rows[2](
            StaticTuple[BufferPtrFloat32, 2](a.hb, a.hb2),
            xbp,
            StaticTuple[BufferPtrFloat32, 2](
                a.w1.unsafe_offset(l * hidden_dim * dim),
                a.w3.unsafe_offset(l * hidden_dim * dim),
            ),
            f0, f1, dim,
        )
        silu_mul(a.hb.unsafe_offset(f0), a.hb2.unsafe_offset(f0), f1 - f0)
        barrier(sh)

        # w2 + residual, over this worker's rows
        matmul_rows[1](
            StaticTuple[BufferPtrFloat32, 1](a.xb),
            a.hb,
            StaticTuple[BufferPtrFloat32, 1](a.w2.unsafe_offset(l * dim * hidden_dim)),
            d0, d1, hidden_dim,
        )
        add(a.x.unsafe_offset(d0), a.xb.unsafe_offset(d0), d1 - d0)
        barrier(sh)

    # Final rmsnorm + classifier over this worker's slice of the vocabulary
    rmsnorm(xbp, a.x, a.rms_final_weight, dim)
    matmul_rows[1](
        StaticTuple[BufferPtrFloat32, 1](a.logits),
        xbp,
        StaticTuple[BufferPtrFloat32, 1](a.wcls),
        v0, v1, dim,
    )

# Body of a persistent worker thread: wait for the next forward pass, run
# its share, signal completion, repeat until stopped.
struct PoolWorker(def(Int) -> None, ImplicitlyCopyable):
    var sh: PoolSharedPtr

    def __init__(out self, sh: PoolSharedPtr):
        self.sh = sh

    def __call__(self, wid: Int):
        var seen = 0
        while True:
            var spins = 0
            while self.sh[].epoch.load() == seen:
                if self.sh[].stop.load() != 0:
                    return
                spin_wait(spins)
                spins += 1
            seen += 1
            var a = self.sh[].args
            forward_worker(a, self.sh[].xbp.unsafe_offset(wid * a.dim), wid, self.sh)
            _ = self.sh[].done.fetch_add(1)

async def _run_pool_worker(w: PoolWorker, wid: Int):
    w(wid)

struct Transformer:
    var workers: Int
    var sh: PoolSharedPtr
    var worker: Pointer[PoolWorker, MutUntrackedOrigin]
    var tg: Pointer[TaskGroup, MutUntrackedOrigin]
    var started: Bool

    def __init__(out self, workers: Int):
        # Worker threads come from the runtime's pool; the calling thread is
        # worker 0, so never ask for more threads than the pool can supply.
        self.workers = max(1, min(workers, parallelism_level()))
        self.sh = unsafe_alloc[PoolShared](1)
        self.worker = unsafe_alloc[PoolWorker](1)
        self.tg = unsafe_alloc[TaskGroup](1)
        self.started = False

    def __deinit__(deinit self):
        if self.started:
            self.sh[].stop.store(1)
            self.tg[].wait()
            self.tg.unsafe_deinit_pointee()
            self.worker.unsafe_deinit_pointee()
            self.sh[].xbp.unsafe_free()
            self.sh.unsafe_deinit_pointee()
        self.tg.unsafe_free()
        self.worker.unsafe_free()
        self.sh.unsafe_free()

    def _start(mut self, args: ForwardArgs):
        self.sh.unsafe_write(PoolShared(self.workers, args))
        self.worker.unsafe_write(PoolWorker(self.sh))
        self.tg.unsafe_write(TaskGroup())
        for i in range(1, self.workers):
            self.tg[].create_task(_run_pool_worker(self.worker[], self.sh[].ids[i]))
        self.started = True

    @always_inline
    def rope_rotation_llama(
        self,
        q_ptr: BufferPtrFloat32,
        k_ptr: BufferPtrFloat32,
        freq_cis_real_row: BufferPtrFloat32,
        freq_cis_imag_row: BufferPtrFloat32,
        config: Config,
        head_size: Int
    ):
        for h in range(config.n_heads):
            rope_head(q_ptr, k_ptr, freq_cis_real_row, freq_cis_imag_row, h, head_size, config.n_kv_heads)

    def transformer(
        mut self,
        token: Int,
        pos: Int,
        config: Config,
        mut state: RunState,
        weights: TransformerWeights,
    ) raises:
        var args = ForwardArgs(token, pos, config, state, weights)
        if not self.started:
            self._start(args)
        elif self.sh[].xbp_dim != args.dim:
            raise Error("Transformer was started with a different model dimension")

        if self.workers == 1:
            self.sh[].args = args
            forward_worker(args, self.sh[].xbp, 0, self.sh)
            return

        # Publish the pass, take part as worker 0, then wait for the others
        self.sh[].args = args
        self.sh[].done.store(0)
        _ = self.sh[].epoch.fetch_add(1)
        forward_worker(args, self.sh[].xbp, 0, self.sh)
        var spins = 0
        while self.sh[].done.load() < self.workers - 1:
            spin_wait(spins)
            spins += 1

def argmax(v: BufferPtrFloat32, size: Int) -> Int:
    var max_i: Int = 0
    var max_p: Float32 = v[unsafe_offset=0]
    for i in range(size):
        if v[unsafe_offset=i] > max_p:
            max_i = i
            max_p = v[unsafe_offset=i]
    return max_i

def sample(probabilities: BufferPtrFloat32, size: Int) -> Int:
    var r = random.random_float64().cast[DType.float32]()
    var cdf: Float32 = 0.0
    for i in range(size):
        cdf += probabilities[unsafe_offset=i]
        if r < cdf:
            return i
    return size - 1

def bpe_encode(mut tokens: List[Int], text: String, tok: Tokenizer):
    for pos in range(text.byte_length()):
        var ch = String(text[byte=pos:pos+1])
        var id = tok.find(ch)
        if id == -1:
            print("Not a good prompt token at pos ", pos)
            return
        tokens.append(id)

    while True:
        var best_score = Float32(-1e10)
        var best_id = -1
        var best_idx = -1

        for i in range(len(tokens) - 1):
            var pair = tok.vocab[tokens[i]] + tok.vocab[tokens[i + 1]]
            var id = tok.find(pair)
            if id != -1 and tok.vocab_scores[id] > best_score:
                best_score = tok.vocab_scores[id]
                best_id = id
                best_idx = i

        if best_idx == -1:
            break

        tokens[best_idx] = best_id
        var _tokens = List[Int]()
        for i in range(0, best_idx + 1):
            _tokens.append(tokens[i])
        for i in range(best_idx + 2, len(tokens)):
            _tokens.append(tokens[i])
        tokens = _tokens^

def get_token_str(var token: Int, var token_str: String) -> String:
    var is_byte_token = False

    # add special token retrieval for TinyLlama
    if token_str.byte_length() == 6 and String(token_str[byte=0:1]) == "<" and String(token_str[byte=1:2]) == "0" and String(token_str[byte=2:3]) == "x":
        if token_str == "<0x0A>":
            token_str = "\n"
        elif token_str == "<0x09>":
            token_str = "\t"
        else:
            is_byte_token = True

    if not is_byte_token:
        if token == 1 and token_str.byte_length() > 0 and String(token_str[byte=0:1]) == " ":
            var tail = String(token_str[byte=1:])
            token_str = tail^
    else:
        token_str = ""

    return token_str

def time_in_ms() -> UInt:
    return UInt(time.perf_counter_ns() // 1_000_000)

def print_usage():
    print("Usage: mojo llama2.mojo <checkpoint> [options]")
    print(
        'Example: mojo llama2.mojo stories15M.bin -j 6 -s 99 -n 256 -t 0.5 -i "Once upon a time"'
    )
    print("Options:")
    print("  -s <int>    random seed, default time.now()")
    print("  -t <float>  temperature in [0,1.0], default 0.9")
    print(
        "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len"
    )
    print("  -i <string> input prompt")
    print("  -z <string> tokenizer path")
    print("  -j <int>    number of parallel workers (default: number of performance cores)")
    print("  -pc <int>   print config (0 or 1)")

def main() raises:

    var tokenizer = "tokenizer.bin"
    var checkpoint = "stories15M.bin"
    var temperature = Float32(0.9)
    var steps = 256
    var prompt = String("")
    var rng_seed: Int = Int(time.perf_counter_ns() // 1_000_000)
    var print_config = 0
    var workers: Int = num_performance_cores()
    def argparse() raises {mut} -> Int:
        var args = argv()
        if len(args) < 2:
            return 0
        checkpoint = args[1]
        for i in range(2, len(args), 2):
            if args[i] == "-p":
                print("Option not supported: ", args[i])
            if args[i] == "-n":
                steps = atol(args[i + 1])
            if args[i] == "-z":
                tokenizer = args[i + 1]
            if args[i] == "-s":
                rng_seed = atol(args[i + 1])
            if args[i] == "-i":
                prompt = args[i + 1]
            if args[i] == "-j":
                workers = atol(args[i + 1])
            if args[i] == "-pc":
                print_config = atol(args[i + 1])
            if args[i] == "-t":
                var val = args[i + 1]
                temperature = 0.0
                for c in range(0, val.byte_length()):
                    if String(val[byte=c:c+1]) == ".":
                        temperature += Float32(atol(String(val[byte=c+1:c+2]))) * Float32(0.1)
                        break
                    else:
                        temperature = Float32(atol(String(val[byte=c:c+1])))
        return 1

    var res = argparse()
    if res == 0:
        print_usage()
        return

    var transformer = Transformer(workers)
    print("num parallel workers:", transformer.workers, " SIMD width:", nelts)
    random.seed(rng_seed)
    var config = Config(checkpoint, print_config == 1)
    var weights = TransformerWeights(checkpoint, config)

    if steps <= 0 or steps > config.seq_len:
        steps = config.seq_len

    var tok = Tokenizer(config.vocab_size, tokenizer)

    print("n layers:", config.n_layers, "| vocab size:", tok.vocab_size)

    var state = RunState(config)
    var prompt_tokens = List[Int]()

    if prompt:
        bpe_encode(prompt_tokens, prompt, tok)

    var start = UInt(0)
    var next_token: Int
    var token = 1
    var pos = 0
    while pos < steps:
        transformer.transformer(token, pos, config, state, weights)

        if pos < len(prompt_tokens):
            next_token = prompt_tokens[pos]
        else:
            if temperature == 0.0:
                next_token = argmax(state.logits.data, config.vocab_size)
            else:
                for q in range(config.vocab_size):
                    state.logits.data[unsafe_offset=q] = state.logits.data[unsafe_offset=q] / temperature

                softmax(state.logits.data, config.vocab_size)
                next_token = sample(state.logits.data, config.vocab_size)

            if next_token == 1 or next_token == 2:
                break

        var token_str = get_token_str(token, tok.vocab[next_token])

        print(token_str, end="")

        token = next_token
        pos += 1

        if start == 0:
            start = time_in_ms()

    var end = time_in_ms()
    var elapsed_ms = end - start
    var achieved_tok_s = Float32(0.0)
    if elapsed_ms > 0 and pos > 1:
        achieved_tok_s = Float32(pos - 1) * Float32(1000.0) / Float32(elapsed_ms)

    print("\nachieved tok/s: ", achieved_tok_s)
