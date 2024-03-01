from algorithm import sum
from algorithm import vectorize, parallelize, unroll
from builtin import string
from math import round
from memory import memset_zero, memcpy, stack_allocation
from memory.buffer import Buffer
from memory.unsafe import DTypePointer
from random import rand
from sys.info import num_performance_cores
from sys import argv
from tensor import Tensor, TensorShape, TensorSpec

# The SIMD vector width.
from sys.info import simdwidthof
import math
import os
import random
import time

alias NUM_CONFIG_INT = 7
var workers = 0

alias nelts = (4 * simdwidthof[DType.float32]())

alias PointerString = Pointer[UInt8]
alias BufferPtrType = DTypePointer[DType.uint8]
alias BufferPtrFloat32 = DTypePointer[DType.float32]
alias PointerStrings = Pointer[PointerString]
alias TensorF32 = Tensor[DType.float32]


@register_passable
struct Accumulator[T: DType, width: Int]:
    # ideally this could be SIMD[T, width] but the width
    # in accumulate() method is compared by identity
    var data: DTypePointer[T]

    @always_inline
    fn __init__() -> Self:
        # allocate a DTypePointer on stack that doesn't need to be freed.
        var data = stack_allocation[width, T]()
        memset_zero(data, width)
        return Self {data: data}

    @always_inline
    fn accumulate[_width: Int](inout self, val: SIMD[T, _width]) -> None:
        # This is a hack to make sure both SIMD have _width length.
        # SIMD[T, width] += SIMD[T, _width] is always an error.
        var newVal = self.data.simd_load[_width]() + val
        self.data.simd_store[_width](newVal)

    @always_inline
    fn total(self) -> SIMD[T, 1]:
        return self.data.simd_load[width]().reduce_add()


struct TensorSlice:
    # Provides a view into a tensor representing a 1D slice on its first or first 2 dimensions.
    # Same function signatures as Tensor but without owning the data.
    var _data: BufferPtrFloat32
    var _shape: TensorShape

    fn __init__(inout self, t: TensorF32, layer: Int) raises:
        var elements_per_layer = t.num_elements() // t.dim(0)
        self._data = t.data().offset(layer * elements_per_layer)
        if t.rank() == 2:
            self._shape = TensorShape(t.dim(1))
        elif t.rank() == 3:
            self._shape = TensorShape(t.dim(1), t.dim(2))
        else:
            # Compiler complains if _shape not defined
            self._shape = TensorShape(1)
            raise Error("TensorSlice: rank greater than 3 not implemented.")

    fn __init__(inout self, t: TensorF32, layer: Int, row: Int) raises:
        var elements_per_layer = t.num_elements() // t.dim(0)
        var elements_per_row = elements_per_layer // t.dim(1)
        self._data = t.data().offset(
            layer * elements_per_layer + row * elements_per_row
        )
        if t.rank() == 3:
            self._shape = TensorShape(t.dim(2))
        elif t.rank() == 1:
            # Compiler complains if _shape not defined
            self._shape = TensorShape(1)
            raise Error(
                "Trying to slice a 1D Tensor by layer and row.  This requires a 3D"
                " Tensor."
            )
        else:
            # Compiler complains if _shape not defined
            self._shape = TensorShape(1)
            raise Error("TensorSlice: rank greater than 3 not implemented.")

    fn data(self) -> BufferPtrFloat32:
        return self._data

    fn shape(self) -> TensorShape:
        return self._shape

    fn num_elements(self) -> Int:
        return self._shape.num_elements()

    fn dim(self, idx: Int) -> Int:
        return self._shape[idx]

    fn rank(self) -> Int:
        return self._shape.rank()

    fn simd_load[nelts: Int](self, idx: Int) -> SIMD[DType.float32, nelts]:
        return self._data.simd_load[nelts](idx)

    fn simd_load[nelts: Int](self, *indices: Int) -> SIMD[DType.float32, nelts]:
        if len(VariadicList(indices)) > 2:
            print(
                "Warning: TensorSlice only supports 1D and 2D indexing.  Results are"
                " unlikely to be correct."
            )
        return self.simd_load[nelts](indices[0] * self._shape[1] + indices[1])

    fn simd_load[
        nelts: Int
    ](self, indices: StaticIntTuple[2]) -> SIMD[DType.float32, nelts]:
        return self._data.simd_load[nelts](indices[0] * self._shape[1] + indices[1])

    fn __getitem__(self, idx: Int) -> SIMD[DType.float32, 1]:
        return self._data.simd_load[1](idx)

    fn simd_store[nelts: Int](self, idx: Int, val: SIMD[DType.float32, nelts]):
        return self._data.simd_store[nelts](idx, val)

    fn __setitem__(self, idx: Int, val: SIMD[DType.float32, 1]):
        return self.simd_store[1](idx, val)


fn read_val_int(inout buf: FileBuf) raises -> Int:
    # DTypePointer[DType.ui8](buf.data).bitcast[DType.ui8]()
    var data = buf.data.offset(buf.get_offset()).bitcast[DType.int32]()
    var result = data.load(0)
    buf.move_offset(4)
    return result.to_int()


fn read_val_float32(inout buf: FileBuf) raises -> Float32:
    # DTypePointer[DType.ui8](buf.data).bitcast[DType.ui8]()
    var val = buf.data.offset(buf.get_offset()).bitcast[DType.float32]().load(0)
    buf.move_offset(4)
    return val


fn read_val_str(inout buf: FileBuf, slen: Int) raises -> PointerString:
    var str = PointerString.alloc(slen + 1)
    for i in range(slen):
        str.store(i, buf.data.load(buf.get_offset()))
        buf.move_offset(1)
    str.store(slen, 0)

    return str


fn str_len(s: PointerString) -> Int:
    var len = 0
    while s[len] != 0:
        len += 1
    return len


# not optimal concat
fn str_concat(s1: PointerString, s2: PointerString) -> PointerString:
    var l1 = str_len(s1)
    var l2 = str_len(s2)
    var str = PointerString.alloc(l1 + l2 + 1)
    memcpy[UInt8](str, s1, l1)
    memcpy[UInt8](str.offset(l1), s2, l2)
    str.store(l1 + l2, 0)
    return str


fn str_to_ptr(s: String) -> PointerString:
    var ret = PointerString.alloc(len(s) + 1)
    for i in range(len(s)):
        ret.store(i, ord(s[i]))
    ret.store(len(s), 0)
    return ret


fn string_compare(a: PointerString, b: PointerString) -> Int:
    var index = 0
    while a[index] != 0 and b[index] != 0:
        if a[index] < b[index]:
            return -1
        if a[index] > b[index]:
            return 1

        index += 1

    if a[index] != 0 and b[index] == 0:
        return 1

    if a[index] == 0 and b[index] != 0:
        return -1

    return 0


# Quicksort helper function to find the partition position
fn partition(
    inout array: PointerStrings, inout indices: DynamicVector[Int], low: Int, high: Int
) -> Int:
    var pivot = array[high]
    var ii = low - 1
    for jj in range(low, high):
        if string_compare(pivot, array[jj]) == 1:
            # If element smaller than pivot, swap
            ii = ii + 1

            var tmp = array[ii]
            var tmp_idx = indices[ii]
            array.store(ii, array[jj])
            indices[ii] = indices[jj]
            array.store(jj, tmp)
            indices[jj] = tmp_idx

    # Swap the pivot element
    var tmp = array[ii + 1]
    var tmp_idx = indices[ii + 1]
    array.store(ii + 1, array[high])
    indices[ii + 1] = indices[high]
    array.store(high, tmp)
    indices[high] = tmp_idx

    return ii + 1


fn quicksort(
    inout array: PointerStrings, inout indices: DynamicVector[Int], low: Int, high: Int
):
    if low < high:
        var pi = partition(array, indices, low, high)
        quicksort(array, indices, low, pi - 1)
        quicksort(array, indices, pi + 1, high)


struct FileBuf:
    var data: BufferPtrType
    var offset: Int
    var size: Int

    fn __init__(inout self):
        self.data = BufferPtrType()
        self.offset = 0
        self.size = 0

    fn __del__(owned self):
        self.data.free()

    fn move_offset(inout self, size: Int) raises:
        var new_offset = self.offset + size
        if new_offset > self.size:
            raise Error("Resulting offset will be past the end of the FileBuf")
        if new_offset < 0:
            raise Error("Resulting offset will be before the beginning of the FileBuf")
        self.offset = new_offset

    fn bitcast_offset_f32(inout self, size: Int) raises -> BufferPtrFloat32:
        var ret = self.data.offset(self.offset).bitcast[DType.float32]()
        self.move_offset(size * sizeof[DType.float32]())
        return ret

    fn get_offset(self) raises -> Int:
        if self.offset > self.size:
            raise Error("Offset is past the end of the FileBuf")
        if self.offset < 0:
            raise Error("Offset is before the beginning of the FileBuf")
        return self.offset


fn wrap(token: PointerString) -> PointerString:
    if string_compare(token, str_to_ptr("\\n")) == 0:
        return str_to_ptr("<0x0A>")
    if string_compare(token, str_to_ptr("\\t")) == 0:
        return str_to_ptr("<0x09>")
    if string_compare(token, str_to_ptr("'")) == 0:
        return str_to_ptr("<0x27>")
    elif string_compare(token, str_to_ptr('"')) == 0:
        return str_to_ptr("<0x22>")
    return token


struct Tokenizer:
    var vocab: PointerStrings
    var vocab_scores: BufferPtrFloat32
    var max_token_length: Int
    var vocab_size: Int
    var sorted_vocab: PointerStrings
    var sorted_indices: DynamicVector[Int]

    fn __init__(inout self, vocab_size: Int, inout buf: FileBuf) raises -> None:
        self.vocab_size = vocab_size
        self.max_token_length = read_val_int(buf)
        self.vocab_scores = BufferPtrFloat32.alloc(self.vocab_size)
        self.vocab = PointerStrings.alloc(self.vocab_size)
        # lazy load sorted vocab
        self.sorted_vocab = PointerStrings.alloc(0)
        self.sorted_indices = DynamicVector[Int]()

        # read vocab_scores & vocab values (tokens)
        for i in range(0, self.vocab_size):
            var score = read_val_float32(buf)
            var slen = read_val_int(buf)
            var token = read_val_str(buf, slen)
            self.store_token(i, token, score)
        return None

    fn __del__(owned self):
        for i in range(0, self.vocab_size):
            self.vocab[i].free()
        self.vocab.free()
        self.vocab_scores.free()
        self.sorted_vocab.free()

    fn store_token(
        inout self, index: Int, owned token: PointerString, score: Float32
    ) -> None:
        self.vocab_scores.store(index, score)
        self.vocab.store(index, token)

    # sort vocab by string_compare
    fn sort(inout self) -> None:
        if len(self.sorted_indices) < self.vocab_size:
            self.sorted_indices = DynamicVector[Int](capacity=self.vocab_size)
            self.sorted_vocab = PointerStrings.alloc(self.vocab_size)
            for ii in range(self.vocab_size):
                self.sorted_vocab.store(ii, self.vocab[ii])
                self.sorted_indices.push_back(ii)

        var n = self.vocab_size
        quicksort(self.sorted_vocab, self.sorted_indices, 0, n - 1)
        return None

    # Binary search that returns -1 if string is not found
    fn find(inout self, token_o: PointerString) -> Int:
        var token = wrap(token_o)
        var n = self.vocab_size
        if len(self.sorted_indices) < n:
            self.sort()
        var left = 0
        var right = n - 1
        while left <= right:
            var mid = left + (right - left) // 2
            var comparison = string_compare(self.sorted_vocab[mid], token)
            if comparison == 0:
                return self.sorted_indices[mid]
            if comparison < 0:
                left = mid + 1
            else:
                right = mid - 1
        return -1


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

    fn __init__(inout self, fileName: String, print_config: Bool) raises:
        var f = open(fileName, "r")
        # reading 7 vars of type DType.int32 from the file
        var bytes_of_config_params = NUM_CONFIG_INT * sizeof[DType.int32]()
        # config_data_raw id Tensor[DType.int8] with bytes_of_config_params elements
        var config_data_raw = f.read_bytes(bytes_of_config_params)
        f.close()
        # correct Tensor type and shape for easy reading, without copying data
        var int32_ptr = config_data_raw._steal_ptr().bitcast[DType.int32]()
        var config_data = Tensor[DType.int32](int32_ptr, NUM_CONFIG_INT)
        self.dim = config_data[0].to_int()
        self.hidden_dim = config_data[1].to_int()
        self.n_layers = config_data[2].to_int()
        self.n_heads = config_data[3].to_int()
        self.n_kv_heads = config_data[4].to_int()
        self.vocab_size = config_data[5].to_int()
        self.seq_len = config_data[6].to_int()
        self.head_size = self.dim // self.n_heads
        self.kv_dim = (self.n_kv_heads * self.dim) // self.n_heads
        self.kv_mul = self.n_heads // self.n_kv_heads
        # negative vocab size is hacky way of signaling unshared weights. bit yikes.
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
    var x: TensorF32  # activation at current time stamp (dim,)
    var xb: TensorF32  # same, but inside a residual branch (dim,)
    var xb2: TensorF32  # an additional buffer just for convenience (dim,)
    var hb: TensorF32  # buffer for hidden dimension in the ffn (hidden_dim,)
    var hb2: TensorF32  # buffer for hidden dimension in the ffn (hidden_dim,)
    var q: TensorF32  # query (dim,)
    var k: TensorSlice  # key (kv_dim,)
    var v: TensorSlice  # value (kv_dim,)
    var att: TensorF32  # buffer for scores/attention values (n_heads, seq_len)
    var logits: TensorF32  # output logits
    var key_cache: TensorF32  # (layer, seq_len, dim)
    var value_cache: TensorF32  # (layer, seq_len, dim)

    fn __init__(inout self, config: Config) raises:
        self.x = TensorF32(config.dim)
        self.xb = TensorF32(config.dim)
        self.xb2 = TensorF32(config.dim)
        self.hb = TensorF32(config.hidden_dim)
        self.hb2 = TensorF32(config.hidden_dim)
        self.q = TensorF32(config.dim)
        self.att = TensorF32(config.n_heads, config.seq_len)
        self.logits = TensorF32(config.vocab_size)
        self.key_cache = TensorF32(config.n_layers, config.seq_len, config.kv_dim)
        self.value_cache = TensorF32(config.n_layers, config.seq_len, config.kv_dim)
        # So their updates flow to the caches, k and v are slices with shared memory.
        # Initialize with placeholders. The real tensors reference layer and position during forward pass.
        self.k = TensorSlice(TensorF32(TensorShape(1, config.kv_dim)), 1)
        self.v = TensorSlice(TensorF32(TensorShape(1, config.kv_dim)), 1)


struct TransformerWeights:
    var token_embedding_table: TensorF32
    var freq_cis_real: TensorF32
    var freq_cis_imag: TensorF32
    var rms_att_weight: TensorF32
    var wq: TensorF32
    var wk: TensorF32
    var wv: TensorF32
    var wo: TensorF32
    var rms_ffn_weight: TensorF32
    var w1: TensorF32
    var w3: TensorF32
    var w2: TensorF32
    var rms_final_weight: TensorF32
    var wcls: TensorF32

    fn __init__(inout self, file_name: String, config: Config) raises:
        var bytes_read = 0
        var f = open(file_name, "r")

        # throw away config data
        _ = f.read_bytes(NUM_CONFIG_INT * sizeof[DType.int32]())
        bytes_read += NUM_CONFIG_INT * sizeof[DType.int32]()

        @parameter
        fn read_weights(*dims: Int) raises -> TensorF32:
            var shape = TensorShape(dims)
            # The created tensor takes a 1D shape equal to bytes read
            # So we can't reshape to target shape because dims don't match
            var tmp = f.read_bytes(shape.num_elements() * sizeof[DType.float32]())
            bytes_read += shape.num_elements() * sizeof[DType.float32]()
            var data = tmp._steal_ptr().bitcast[DType.float32]()
            return TensorF32(data, shape)

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
        # maybe need modifying for different model
        # config.head_size // 2 for stories and tinyllama-1.1
        self.freq_cis_real = read_weights(config.seq_len, config.head_size // 2)
        self.freq_cis_imag = read_weights(config.seq_len, config.head_size // 2)
        if config.shared_weights:
            self.wcls = self.token_embedding_table
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


fn read_file(file_name: String, inout buf: FileBuf) raises:
    var fd = open(file_name, "r")
    var data = fd.read()
    fd.close()
    buf.size = data._buffer.size
    buf.data = data._steal_ptr().bitcast[DType.uint8]()
    buf.offset = 0
    return None


@always_inline
fn accum(inout a: TensorF32, b: TensorF32) -> None:
    var size = a.dim(0)

    @parameter
    fn _acc[_nelts: Int](j: Int):
        a.simd_store[_nelts](j, a.simd_load[_nelts](j) + b.simd_load[_nelts](j))

    vectorize[_acc, nelts](size)


@always_inline
fn rmsnorm(
    inout o: BufferPtrFloat32, x: BufferPtrFloat32, weight: BufferPtrFloat32, size: Int
) -> None:
    # Calculate sum of squares
    var tmp = Accumulator[DType.float32, nelts]()

    @parameter
    fn _sum2[_nelts: Int](j: Int):
        tmp.accumulate(x.offset(j).simd_load[_nelts](0) ** 2)

    vectorize[_sum2, nelts](size)

    var ss: Float32 = tmp.total()
    ss = ss / size + 1e-5
    ss = 1.0 / math.sqrt(ss)

    # Normalize and scale
    @parameter
    fn _norm[_nelts: Int](j: Int):
        var val = weight.simd_load[_nelts](j) * ss * x.simd_load[_nelts](j)
        o.offset(j).simd_store[_nelts](0, val)

    vectorize[_norm, nelts](size)


@always_inline
fn rmsnorm(inout o: TensorF32, x: TensorF32, weight: TensorF32):
    rmsnorm(o._ptr, x.data(), weight.data(), weight.dim(weight.rank() - 1))


@always_inline
fn rmsnorm(inout o: TensorF32, x: TensorF32, weight: TensorSlice):
    rmsnorm(o._ptr, x.data(), weight.data(), weight.dim(weight.rank() - 1))


@always_inline
fn softmax(inout x: TensorF32) -> None:
    softmax(x, 0, x.dim(0))


@always_inline
fn softmax(inout x: TensorF32, start: Int, end: Int):
    var max_val: Float32 = -1e9

    @parameter
    fn _max[_nelts: Int](ii: Int):
        var val = x.simd_load[_nelts](start + ii).reduce_max()
        if val > max_val:
            max_val = val

    vectorize[_max, nelts](end - start)

    var acc = Accumulator[DType.float32, nelts]()

    @parameter
    fn _exp[_nelts: Int](ii: Int):
        var val = math.exp(x.simd_load[_nelts](start + ii) - max_val)
        x.simd_store[_nelts](start + ii, val)
        acc.accumulate(val)

    vectorize[_exp, nelts](end - start)

    var ssum = acc.total()

    @parameter
    fn _norm[_nelts: Int](ii: Int):
        x.simd_store[_nelts](start + ii, x.simd_load[_nelts](start + ii) / ssum)

    vectorize[_norm, nelts](end - start)


@always_inline
fn batch_matmul[
    n: Int
](
    C: StaticTuple[n, BufferPtrFloat32],
    A: BufferPtrFloat32,
    B: StaticTuple[n, BufferPtrFloat32],
    rows: Int,
    cols: Int,
):
    @parameter
    fn compute_row(i: Int):
        var tmp = StaticTuple[n, Accumulator[DType.float32, nelts]]()

        @unroll
        for k in range(n):
            tmp[k] = Accumulator[DType.float32, nelts]()

        var row_offset = i * cols

        @parameter
        fn dot[_nelts: Int](j: Int):
            var a = A.simd_load[_nelts](j)

            @unroll
            for k in range(n):
                tmp[k].accumulate(a * B[k].simd_load[_nelts](row_offset + j))

        vectorize[dot, nelts](cols)

        @unroll
        for k in range(n):
            C[k].store(i, tmp[k].total())

    parallelize[compute_row](rows, workers)


@always_inline
fn matmul(C: TensorF32, A: TensorF32, B: TensorF32) raises:
    # B (d,n) @ A (n,) -> C (d,)
    matmul_dimension_checks(A.shape(), B.shape())
    batch_matmul[1](
        StaticTuple[1, BufferPtrFloat32](C.data()),
        A.data(),
        StaticTuple[1, BufferPtrFloat32](B.data()),
        B.dim(0),
        B.dim(1),
    )


@always_inline
fn matmul(C: TensorF32, A: TensorF32, B: TensorSlice) raises:
    # B (d,n) @ A (n,) -> C (d,)
    matmul_dimension_checks(A.shape(), B.shape())
    batch_matmul[1](
        StaticTuple[1, BufferPtrFloat32](C.data()),
        A.data(),
        StaticTuple[1, BufferPtrFloat32](B.data()),
        B.dim(0),
        B.dim(1),
    )


@always_inline
fn matmul(C: TensorSlice, A: TensorF32, B: TensorSlice) raises:
    # B (d,n) @ A (n,) -> C (d,)
    matmul_dimension_checks(A.shape(), B.shape())
    batch_matmul[1](
        StaticTuple[1, BufferPtrFloat32](
            C.data(),
        ),
        A.data(),
        StaticTuple[1, BufferPtrFloat32](B.data()),
        B.dim(0),
        B.dim(1),
    )


fn matmul_dimension_checks(a: TensorShape, b: TensorShape) raises:
    if a[0] != b[1]:
        raise Error(
            "matmul dimension mismatch. A rows (dim 0) not equal to B columns (dim 1)"
        )
    if b.rank() != 2:
        raise Error("matmul expects B to be a 2D matrix")


# Apply RoPE rotation to the q and k vectors for each head
# rotate odd and even dim
@always_inline
fn rope_rotation_llama(
    inout state: RunState,
    freq_cis_real_row: TensorSlice,
    freq_cis_imag_row: TensorSlice,
    config: Config,
) -> None:
    # stories model, llama2
    var head_size = config.head_size

    @parameter
    fn head_loop(i: Int):
        # Simple vectorization with (head_size // 2) steps gave junk transformer output.
        # Maybe because the nelt ranges end up overlapping between the steps.
        for j in range(0, config.head_size, 2):
            var fcr = freq_cis_real_row[j // 2]
            var fci = freq_cis_imag_row[j // 2]
            var q0 = state.q[i * head_size + j]
            var q1 = state.q[i * head_size + j + 1]
            state.q[i * head_size + j] = q0 * fcr - q1 * fci
            state.q[i * head_size + j + 1] = q0 * fci + q1 * fcr
            if i < config.n_kv_heads:
                var k0 = state.k[i * head_size + j]
                var k1 = state.k[i * head_size + j + 1]
                state.k[i * head_size + j] = k0 * fcr - k1 * fci
                state.k[i * head_size + j + 1] = k0 * fci + k1 * fcr

    parallelize[head_loop](config.n_heads, workers)


@always_inline
fn transformer(
    token: Int,
    pos: Int,
    config: Config,
    inout state: RunState,
    weights: TransformerWeights,
) raises -> None:
    # A few convenience variables
    var dim = config.dim
    var hidden_dim = config.hidden_dim
    var head_size = config.head_size
    var kv_dim = config.kv_dim
    var kv_mul = config.kv_mul

    # Copy the token embedding into x
    var content_row = weights.token_embedding_table.data().offset(token * dim)
    memcpy[DType.float32](state.x.data(), content_row, dim)

    # Pluck out the "pos" row of freq_cis_real and freq_cis_imag
    var freq_cis_real_row = TensorSlice(weights.freq_cis_real, pos)
    var freq_cis_imag_row = TensorSlice(weights.freq_cis_imag, pos)

    # Forward all the layers
    for l in range(config.n_layers):
        # Attention rmsnorm
        rmsnorm(state.xb, state.x, TensorSlice(weights.rms_att_weight, l))
        # QKV matmuls for this position
        var loff = l * config.seq_len * config.kv_dim
        state.k = TensorSlice(state.key_cache, l, pos)
        state.v = TensorSlice(state.value_cache, l, pos)
        if kv_dim == dim:
            batch_matmul[3](
                StaticTuple[3, BufferPtrFloat32](
                    state.q.data(), state.k.data(), state.v.data()
                ),
                state.xb.data(),
                StaticTuple[3, BufferPtrFloat32](
                    TensorSlice(weights.wq, l).data(),
                    TensorSlice(weights.wk, l).data(),
                    TensorSlice(weights.wv, l).data(),
                ),
                dim,
                dim,
            )
        else:
            matmul(state.q, state.xb, TensorSlice(weights.wq, l))
            batch_matmul[2](
                StaticTuple[2, BufferPtrFloat32](state.k.data(), state.v.data()),
                state.xb.data(),
                StaticTuple[2, BufferPtrFloat32](
                    TensorSlice(weights.wk, l).data(), TensorSlice(weights.wv, l).data()
                ),
                kv_dim,
                dim,
            )

        # Apply RoPE rotation to the q and k vectors for each head
        rope_rotation_llama(state, freq_cis_real_row, freq_cis_imag_row, config)

        memset_zero(state.xb.data(), state.xb.num_elements())

        # Multihead attention. Iterate over all heads in parallel.
        @parameter
        fn loop_over_heads(h: Int):
            # Get the query vector for this head
            var q_offset = h * head_size

            # Index of attention scores for this head
            var att_offset = h * config.seq_len

            # Iterate over all timesteps, including the current one
            for t in range(pos + 1):
                # Starting index of the key vector for this head and at this timestep
                var k_offset = loff + t * kv_dim + (h // kv_mul) * head_size
                # Calculate the attention score as the dot product of q and k
                var score: Float32 = 0.0

                @parameter
                fn score_fn[_nelts: Int](i: Int):
                    score += (
                        state.q.simd_load[_nelts](q_offset + i)
                        * state.key_cache.simd_load[_nelts](k_offset + i)
                    ).reduce_add()

                vectorize[score_fn, nelts](head_size)
                score /= math.sqrt[DType.float32, 1](head_size)

                # Save the score to the attention buffer
                state.att[att_offset + t] = score

            # Softmax the scores to get attention weights, from 0..pos inclusively
            softmax(state.att, att_offset, att_offset + pos + 1)
            # Weighted sum of the values, store back into xb
            var xb_offset = h * head_size
            for t in range(pos + 1):
                # Starting index of the value vector for this head and at this timestep
                var v_offset = loff + t * kv_dim + (h // kv_mul) * head_size

                # Get the attention weight for this timestep
                var a = state.att[att_offset + t]
                # Accumulate the weighted value into xb

                @parameter
                fn xb_accumulate[_nelts: Int](i: Int):
                    var xbi = state.xb.simd_load[_nelts](
                        xb_offset + i
                    ) + a * state.value_cache.simd_load[_nelts](v_offset + i)
                    state.xb.simd_store[_nelts](xb_offset + i, xbi)

                vectorize[xb_accumulate, nelts](head_size)

        parallelize[loop_over_heads](config.n_heads, workers)
        # Final matrix multiplication to get the output of the attention
        matmul(state.xb2, state.xb, TensorSlice(weights.wo, l))
        # Residual connection back into x
        accum(state.x, state.xb2)
        # FFN rmsnorm
        rmsnorm(state.xb, state.x, TensorSlice(weights.rms_ffn_weight, l))

        # Calculate self.w1(x) and self.w3(x) for FFN
        batch_matmul[2](
            StaticTuple[2, BufferPtrFloat32](state.hb.data(), state.hb2.data()),
            state.xb.data(),
            StaticTuple[2, BufferPtrFloat32](
                TensorSlice(weights.w1, l).data(), TensorSlice(weights.w3, l).data()
            ),
            hidden_dim,
            dim,
        )

        @parameter
        fn silu[_nelts: Int](i: Int):
            var initial_hb = state.hb.simd_load[_nelts](i)
            # Apply SiLU activation function (silu(x) = x * sigmoid(x))
            var hbi = initial_hb * (1.0 / (1.0 + math.exp(-initial_hb)))
            # Elementwise multiply with w3(x)
            state.hb.simd_store[_nelts](i, hbi * state.hb2.simd_load[_nelts](i))

        vectorize[silu, nelts](hidden_dim)
        # Final matrix multiplication to get the output of the FFN
        matmul(state.xb, state.hb, TensorSlice(weights.w2, l))

        # Residual connection
        accum(state.x, state.xb)

    # Final rmsnorm
    rmsnorm(state.x, state.x, weights.rms_final_weight)

    # Classifier into logits
    matmul(state.logits, state.x, weights.wcls)


fn argmax(v: TensorF32) -> Int:
    # return argmax of v
    var max_i: Int = 0
    var max_p: Float32 = v[0]
    for i in range(v.dim(0)):
        if v[i] > max_p:
            max_i = i
            max_p = v[i]
    return max_i


fn sample(probabilities: TensorF32) -> Int:
    var n = probabilities.dim(0)
    # Sample index from probabilities, they must sum to 1
    # get random value within (min, max) float32 range
    var r = rand[DType.float32](1)
    var cdf: Float32 = 0.0
    for i in range(n):
        cdf += probabilities[i]
        if r[0] < cdf:
            return i
    return n - 1  # In case of rounding errors


fn bpe_encode(inout tokens: DynamicVector[Int], text: String, inout tok: Tokenizer):
    for pos in range(len(text)):
        var char = str_to_ptr(text[pos])
        var id = tok.find(char)
        if id == -1:
            print("Not a good prompt token at pos ", pos)
            return
        tokens.push_back(id)

    while True:
        var best_score = Float32(-1e10)
        var best_id = -1
        var best_idx = -1

        for i in range(len(tokens) - 1):
            # Check if we can merge the pair (tokens[i], tokens[i+1])
            var str = str_concat(tok.vocab[tokens[i]], tok.vocab[tokens[i + 1]])
            var id = tok.find(str)
            if id != -1 and tok.vocab_scores.load(id) > best_score:
                best_score = tok.vocab_scores.load(id)
                best_id = id
                best_idx = i

        if best_idx == -1:
            # We couldn't find any more pairs to merge, so we're done
            break

        # Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id
        # Delete token at position best_idx+1, shift the entire sequence back 1
        var _tokens = DynamicVector[Int]()
        for i in range(0, best_idx + 1):
            _tokens.push_back(tokens[i])
        for i in range(best_idx + 2, len(tokens)):
            _tokens.push_back(tokens[i])
        tokens = _tokens


fn str2num(d: Int) -> Int:
    # covert Hex to decimal
    if d >= ord("A"):
        return d - ord("A") + 10
    return d - ord("0")


fn print_str(s: PointerString):
    # print raw byte like <0x0A>
    if (s[1].to_int() == ord("0")) and (s[2].to_int() == ord("x")):
        var d1: Int = s[3].to_int()
        var d2: Int = s[4].to_int()
        print_no_newline(chr(str2num(d1) * 16 + str2num(d2)))
        return
    # print all chars till null character
    var p: Int = 0
    while s[p].to_int() != 0:
        print_no_newline(chr(s[p].to_int()))
        p += 1


fn time_in_ms() -> Int:
    # Returns time in milliseconds for benchmarking the model speed
    return time.now() // 1_000_000


fn print_usage():
    print("Usage: mojo llama2.mojo <checkpoint> [options]")
    print(
        'Example: mojo llama2.mojo stories15M.bin -s 99 -n 256 -t 0.5 -i "Llama is an'
        ' animal"'
    )
    print("Options:")
    print("  -s <int>    random seed, default time.now()")
    print("  -t <float>  temperature in [0,1.0], default 1.0")
    print("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len")
    print("  -i <string> input prompt")
    print("  -z          tokenizer path")
    print("  -j          number of workers to use, default num_cores()")


fn main() raises:
    workers = num_performance_cores()
    var tokenizer = StringRef("tokenizer.bin")
    var checkpoint = StringRef("stories15M.bin")
    var temperature = 0.9
    var steps = 256
    var prompt = String("")
    var rng_seed: Int = time.now()
    var print_config = 0

    @parameter
    fn argparse() raises -> Int:
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
                # hacky parse float, keep only 1 digit
                for c in range(0, len(val)):
                    if val[c] == ".":
                        temperature += atol(val[c + 1]) * 0.1
                        break
                    else:
                        temperature = atol(val[c])
                if temperature < -1e9 or temperature > (1 + 1e9):
                    print("Wrong temperature value", temperature)
                    return 0
        return 1

    var res = argparse()
    if res == 0:
        print_usage()
        return

    print("num parallel workers:", workers, " SIMD width:", nelts)
    random.seed(rng_seed)
    var config = Config(checkpoint, print_config == 1)
    var weights = TransformerWeights(checkpoint, config)

    if steps <= 0 or steps > config.seq_len:
        steps = config.seq_len

    # Read in the tokenizer.bin file
    var tbuf = FileBuf()
    read_file(tokenizer, tbuf)
    var tok = Tokenizer(config.vocab_size, tbuf)

    print(
        "n layers:",
        config.n_layers,
        "| vocab size:",
        tok.vocab_size,
    )

    # Create and initialize the application RunState
    var state = RunState(config)

    # Process the prompt, if any
    var prompt_tokens = DynamicVector[Int]()

    if prompt:
        bpe_encode(prompt_tokens, prompt, tok)

    # Start the main loop
    var start = 0  # Used to time our code, only initialized after the first iteration
    var next_token = 0  # Will store the next token in the sequence
    # Initialize with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
    var token = 1

    # Position in the sequence
    var pos = 0
    while pos < steps:
        # Forward the transformer to get logits for the next token
        transformer(token, pos, config, state, weights)

        if pos < len(prompt_tokens):
            next_token = prompt_tokens[pos]
        else:
            # Sample the next token
            if temperature == 0.0:
                # Greedy argmax sampling: take the token with the highest probability
                next_token = argmax(state.logits)
            else:
                # Apply the temperature to the logits
                for q in range(config.vocab_size):
                    state.logits[q] = state.logits[q] / temperature

                # Apply softmax to the logits to get the probabilities for the next token
                softmax(state.logits)
                # Sample from this distribution to get the next token
                next_token = sample(state.logits)

            # Finish generating when EOS, BOS appear
            if next_token == 1 or next_token == 2:
                break
        var token_str: PointerString = tok.vocab[next_token]
        if token == 1 and token_str[0] == ord(" "):
            token_str = token_str.offset(1)

        print_str(token_str)

        # Advance forward
        token = next_token
        pos += 1

        if start == 0:
            start = time_in_ms()

    var end = time_in_ms()
    print("\nachieved tok/s: ", (pos - 1) / (end - start) * 1000)
