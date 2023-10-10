from algorithm import sum
from algorithm import vectorize, parallelize
from builtin import string
from math import round
from memory import memset_zero, memcpy
from memory.buffer import Buffer
from memory.unsafe import DTypePointer
from python import Python
from random import rand
from read import BufReader, File
from runtime.llcl import num_cores, Runtime
from sys import argv
from tensor import Tensor, TensorShape, TensorSpec

# The SIMD vector width.
from sys.info import simdwidthof
import math
import os
import random
import time

var workers = 0

alias nelts = (2*simdwidthof[DType.float32]())

alias PointerString = Pointer[UInt8]
alias BufferPtrType = DTypePointer[DType.uint8]
alias BufferPtrFloat32 = DTypePointer[DType.float32]
alias PointerStrings = Pointer[PointerString]
alias TensorF32 = Tensor[DType.float32]


struct TensorSlice:
    # Provides a view into a tensor representing a 1D slice on its first or first 2 dimensions.
    # Same function signatures as Tensor but without owning the data.
    var _data: BufferPtrFloat32
    var _shape: TensorShape

    fn __init__(inout self, t: TensorF32, layer: Int) raises:
        let elements_per_layer = t.num_elements() // t.dim(0)
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
        let elements_per_layer = t.num_elements() // t.dim(0)
        let elements_per_row = elements_per_layer // t.dim(1)
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
    let data = buf.data.offset(buf.get_offset()).bitcast[DType.int32]()
    let result = data.load(0)
    buf.move_offset(4)
    return result.to_int()


fn read_val_float32(inout buf: FileBuf) raises -> Float32:
    # DTypePointer[DType.ui8](buf.data).bitcast[DType.ui8]()
    let val = buf.data.offset(buf.get_offset()).bitcast[DType.float32]().load(0)
    buf.move_offset(4)
    return val


fn read_val_str(inout buf: FileBuf, slen: Int) raises -> PointerString:
    let str = PointerString.alloc(slen + 1)
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
    let l1 = str_len(s1)
    let l2 = str_len(s2)
    let str = PointerString.alloc(l1 + l2 + 1)
    memcpy[UInt8](str, s1, l1)
    memcpy[UInt8](str.offset(l1), s2, l2)
    str.store(l1 + l2, 0)
    return str


fn str_to_ptr(s: String) -> PointerString:
    let ret = PointerString.alloc(len(s) + 1)
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
    let pivot = array[high]
    var ii = low - 1
    for jj in range(low, high):
        if string_compare(pivot, array[jj]) == 1:
            # If element smaller than pivot, swap
            ii = ii + 1

            let tmp = array[ii]
            let tmp_idx = indices[ii]
            array.store(ii, array[jj])
            indices[ii] = indices[jj]
            array.store(jj, tmp)
            indices[jj] = tmp_idx

    # Swap the pivot element
    let tmp = array[ii + 1]
    let tmp_idx = indices[ii + 1]
    array.store(ii + 1, array[high])
    indices[ii + 1] = indices[high]
    array.store(high, tmp)
    indices[high] = tmp_idx

    return ii + 1


fn quicksort(
    inout array: PointerStrings, inout indices: DynamicVector[Int], low: Int, high: Int
):
    if low < high:
        let pi = partition(array, indices, low, high)
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
        let new_offset = self.offset + size
        if new_offset > self.size:
            raise Error("Resulting offset will be past the end of the FileBuf")
        if new_offset < 0:
            raise Error("Resulting offset will be before the beginning of the FileBuf")
        self.offset = new_offset

    fn bitcast_offset_f32(inout self, size: Int) raises -> BufferPtrFloat32:
        let ret = self.data.offset(self.offset).bitcast[DType.float32]()
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
        self.sorted_indices = DynamicVector[Int](0)

        # read vocab_scores & vocab values (tokens)
        for i in range(0, self.vocab_size):
            self.vocab_scores.store(i, read_val_float32(buf))
            let slen = read_val_int(buf)
            self.vocab.store(i, read_val_str(buf, slen))

        return None

    # sort vocab by string_compare
    fn sort(inout self) -> None:
        if len(self.sorted_indices) < self.vocab_size:
            self.sorted_indices = DynamicVector[Int](self.vocab_size)
            self.sorted_vocab = PointerStrings.alloc(self.vocab_size)
            for ii in range(self.vocab_size):
                self.sorted_vocab.store(ii, self.vocab[ii])
                self.sorted_indices.push_back(ii)

        let n = self.vocab_size
        quicksort(self.sorted_vocab, self.sorted_indices, 0, n - 1)
        return None

    # Binary search that returns -1 if string is not found
    fn find(inout self, token_o: PointerString) -> Int:
        let token = wrap(token_o)
        let n = self.vocab_size
        if len(self.sorted_indices) < n:
            self.sort()
        var left = 0
        var right = n - 1
        while left <= right:
            let mid = left + (right - left) // 2
            let comparison = string_compare(self.sorted_vocab[mid], token)
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

    fn __init__(inout self):
        self.dim = 0
        self.hidden_dim = 0
        self.n_layers = 0
        self.n_heads = 0
        self.n_kv_heads = 0
        self.vocab_size = 0
        self.seq_len = 0
        self.kv_dim = 0
        self.kv_mul = 0
        self.head_size = 0


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

    fn __init__(
        inout self, config: Config, shared_weights: Int, inout buf: FileBuf
    ) raises:
        fn load_weights(inout buf: FileBuf, *dims: Int) raises -> TensorF32:
            # Ensure returned Tensor doesn't share a pointer with FileBuf
            let shape = TensorShape(dims)
            let result_data = BufferPtrFloat32.alloc(shape.num_elements())
            memcpy(
                result_data,
                buf.bitcast_offset_f32(shape.num_elements()),
                shape.num_elements(),
            )
            return TensorF32(result_data, shape)

        self.token_embedding_table = load_weights(buf, config.vocab_size, config.dim)
        self.rms_att_weight = load_weights(buf, config.n_layers, config.dim)
        self.wq = load_weights(buf, config.n_layers, config.dim, config.dim)
        self.wk = load_weights(buf, config.n_layers, config.kv_dim, config.dim)
        self.wv = load_weights(buf, config.n_layers, config.kv_dim, config.dim)
        self.wo = load_weights(buf, config.n_layers, config.dim, config.dim)
        self.rms_ffn_weight = load_weights(buf, config.n_layers, config.dim)
        self.w1 = load_weights(buf, config.n_layers, config.hidden_dim, config.dim)
        self.w2 = load_weights(buf, config.n_layers, config.dim, config.hidden_dim)
        self.w3 = load_weights(buf, config.n_layers, config.hidden_dim, config.dim)
        self.rms_final_weight = load_weights(buf, config.dim)
        # maybe need modifying for different model
        # config.head_size // 2 for stories and tinyllama-1.1
        self.freq_cis_real = load_weights(buf, config.seq_len, config.head_size // 2)
        self.freq_cis_imag = load_weights(buf, config.seq_len, config.head_size // 2)
        if shared_weights:
            self.wcls = self.token_embedding_table
        else:
            self.wcls = load_weights(buf, config.vocab_size, config.dim)


fn read_file(file_name: String, inout buf: FileBuf) raises:
    let _os = Python.import_module("os")
    let ff_size = _os.path.getsize(file_name)
    let cp_size = string.atol(ff_size.to_string())
    let cp_buf: BufferPtrType = BufferPtrType.alloc(cp_size)
    # set window buffer to read binary data from file
    let f = File(file_name)
    var reader = BufReader[4096](f ^)
    var bytes_read = 1
    var offset = 0

    while bytes_read > 0:
        let buf = Buffer[4096, DType.uint8](cp_buf.offset(offset))
        bytes_read = reader.read(buf)
        offset += bytes_read
    reader.do_nothing()  # keeps lifetimes working
    buf.data = cp_buf
    buf.size = cp_size
    buf.offset = 0
    return None


fn config_init(inout config: Config, inout buf: FileBuf) raises:
    config.dim = read_val_int(buf)
    config.hidden_dim = read_val_int(buf)
    config.n_layers = read_val_int(buf)
    config.n_heads = read_val_int(buf)
    config.n_kv_heads = read_val_int(buf)
    config.vocab_size = read_val_int(buf)
    config.seq_len = read_val_int(buf)
    config.head_size = config.dim // config.n_heads
    config.kv_dim = (config.n_kv_heads * config.dim) // config.n_heads
    config.kv_mul = config.n_heads // config.n_kv_heads
    return None


@always_inline
fn accum(inout a: TensorF32, b: TensorF32) -> None:
    let size = a.dim(0)

    @parameter
    fn _acc[_nelts: Int](j: Int):
        a.simd_store[_nelts](j, a.simd_load[_nelts](j) + b.simd_load[_nelts](j))

    vectorize[nelts, _acc](size)


@always_inline
fn rmsnorm(
    inout o: BufferPtrFloat32, x: BufferPtrFloat32, weight: BufferPtrFloat32, size: Int
) -> None:
    # Calculate sum of squares
    var tmp = SIMD[DType.float32, nelts](0)

    @parameter
    fn _sum2[_nelts: Int](j: Int):
        if _nelts < nelts:
            tmp[0] += (x.offset(j).simd_load[_nelts](0) ** 2).reduce_add()
        else:
            tmp += x.offset(j).simd_load[nelts](0) ** 2

    vectorize[nelts, _sum2](size)

    var ss: Float32 = tmp.reduce_add()
    ss = ss / size + 1e-5
    ss = 1.0 / math.sqrt(ss)

    # Normalize and scale
    @parameter
    fn _norm[_nelts: Int](j: Int):
        let val = weight.simd_load[_nelts](j) * ss * x.simd_load[_nelts](j)
        o.offset(j).simd_store[_nelts](0, val)

    vectorize[nelts, _norm](size)


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
        let val = x.simd_load[_nelts](start + ii).reduce_max()
        if val > max_val:
            max_val = val

    vectorize[nelts, _max](end - start)

    var ssum: Float32 = 0.0

    @parameter
    fn _exp[_nelts: Int](ii: Int):
        x.simd_store[_nelts](
            start + ii, math.exp(x.simd_load[_nelts](start + ii) - max_val)
        )
        ssum += x.simd_load[_nelts](start + ii).reduce_add()

    vectorize[nelts, _exp](end - start)

    @parameter
    fn _norm[_nelts: Int](ii: Int):
        x.simd_store[_nelts](start + ii, x.simd_load[_nelts](start + ii) / ssum)

    vectorize[nelts, _norm](end - start)


@always_inline
fn matmul_parallelized(C: BufferPtrFloat32,A: BufferPtrFloat32,B: BufferPtrFloat32,rows: Int,cols: Int,):
    @parameter
    fn compute_row(i: Int):
        var tmp = SIMD[DType.float32, nelts](0)

        @parameter
        fn dot[_nelts: Int](j: Int):
            if _nelts < nelts:  # take care of tail array elements with length <  nelts
                tmp[0] += (
                A.simd_load[_nelts](j) * B.simd_load[_nelts](i * cols + j)
                ).reduce_add()
            else:
                tmp += A.simd_load[nelts](j) * B.simd_load[nelts](i * cols + j)

        vectorize[nelts, dot](cols)
        C.store(i, tmp.reduce_add())
    

    parallelize[compute_row](rows, workers)

    

    
    
@always_inline
fn matmul(C: TensorF32, A: TensorF32, B: TensorF32) raises:
    # B (d,n) @ A (n,) -> C (d,)
    matmul_dimension_checks(A.shape(), B.shape())
    matmul_parallelized(C.data(), A.data(), B.data(), B.dim(0), B.dim(1))


@always_inline
fn matmul(C: TensorF32, A: TensorF32, B: TensorSlice) raises:
    # B (d,n) @ A (n,) -> C (d,)
    matmul_dimension_checks(A.shape(), B.shape())
    matmul_parallelized(C.data(), A.data(), B.data(), B.dim(0), B.dim(1))


@always_inline
fn matmul(C: TensorSlice, A: TensorF32, B: TensorSlice) raises:
    # B (d,n) @ A (n,) -> C (d,)
    matmul_dimension_checks(A.shape(), B.shape())
    matmul_parallelized(C.data(), A.data(), B.data(), B.dim(0), B.dim(1))


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
    let head_size = config.head_size
    @parameter
    fn head_loop(i:Int):
        # Simple vectorization with (head_size // 2) steps gave junk transformer output.
        # Maybe because the nelt ranges end up overlapping between the steps.
        for j in range(0, config.head_size, 2):
            let fcr = freq_cis_real_row[j // 2]
            let fci = freq_cis_imag_row[j // 2]
            let q0 = state.q[i * head_size + j]
            let q1 = state.q[i * head_size + j + 1]
            state.q[i * head_size + j] = q0 * fcr - q1 * fci
            state.q[i * head_size + j + 1] = q0 * fci + q1 * fcr
            if i < config.n_kv_heads:
                let k0 = state.k[i * head_size + j]
                let k1 = state.k[i * head_size + j + 1]
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
    let dim = config.dim
    let hidden_dim = config.hidden_dim
    let head_size = config.head_size
    let kv_dim = config.kv_dim
    let kv_mul = config.kv_mul

    # Copy the token embedding into x
    let content_row = weights.token_embedding_table.data().offset(token * dim)
    memcpy[DType.float32](state.x.data(), content_row, dim)

    # Pluck out the "pos" row of freq_cis_real and freq_cis_imag
    let freq_cis_real_row = TensorSlice(weights.freq_cis_real, pos)
    let freq_cis_imag_row = TensorSlice(weights.freq_cis_imag, pos)

    # Forward all the layers
    for l in range(config.n_layers):
        # Attention rmsnorm
        rmsnorm(state.xb, state.x, TensorSlice(weights.rms_att_weight, l))
        # QKV matmuls for this position
        matmul(state.q, state.xb, TensorSlice(weights.wq, l))

        let loff = l * config.seq_len * config.kv_dim
        state.k = TensorSlice(state.key_cache, l, pos)
        matmul(state.k, state.xb, TensorSlice(weights.wk, l))

        state.v = TensorSlice(state.value_cache, l, pos)
        matmul(state.v, state.xb, TensorSlice(weights.wv, l))

        # Apply RoPE rotation to the q and k vectors for each head
        rope_rotation_llama(state, freq_cis_real_row, freq_cis_imag_row, config)

        memset_zero(state.xb.data(), state.xb.num_elements())

        # Multihead attention. Iterate over all heads in parallel.
        @parameter
        fn loop_over_heads(h:Int):
            # Get the query vector for this head
            let q_offset = h * head_size

            # Index of attention scores for this head
            let att_offset = h * config.seq_len

            # Iterate over all timesteps, including the current one
            for t in range(pos + 1):
                # Starting index of the key vector for this head and at this timestep
                let k_offset = loff + t * kv_dim + (h // kv_mul) * head_size
                # Calculate the attention score as the dot product of q and k
                var score: Float32 = 0.0

                @parameter
                fn score_fn[_nelts: Int](i: Int):
                    score += (
                        state.q.simd_load[_nelts](q_offset + i)
                        * state.key_cache.simd_load[_nelts](k_offset + i)
                    ).reduce_add()

                vectorize[nelts, score_fn](head_size)
                score /= math.sqrt[DType.float32, 1](head_size)

                # Save the score to the attention buffer
                state.att[att_offset + t] = score

            # Softmax the scores to get attention weights, from 0..pos inclusively
            softmax(state.att, att_offset, att_offset + pos + 1)
            # Weighted sum of the values, store back into xb
            let xb_offset = h * head_size
            for t in range(pos + 1):
                # Starting index of the value vector for this head and at this timestep
                let v_offset = loff + t * kv_dim + (h // kv_mul) * head_size

                # Get the attention weight for this timestep
                let a = state.att[att_offset + t]
                # Accumulate the weighted value into xb

                @parameter
                fn xb_accumulate[_nelts: Int](i: Int):
                    let xbi = state.xb.simd_load[_nelts](
                        xb_offset + i
                    ) + a * state.value_cache.simd_load[_nelts](v_offset + i)
                    state.xb.simd_store[_nelts](xb_offset + i, xbi)

                vectorize[nelts, xb_accumulate](head_size)

        parallelize[loop_over_heads](config.n_heads, workers)
        # Final matrix multiplication to get the output of the attention
        matmul(state.xb2, state.xb, TensorSlice(weights.wo, l))
        # Residual connection back into x
        accum(state.x, state.xb2)
        # FFN rmsnorm
        rmsnorm(state.xb, state.x, TensorSlice(weights.rms_ffn_weight, l))

        # Calculate self.w1(x) and self.w3(x) for FFN
        matmul(state.hb, state.xb, TensorSlice(weights.w1, l))

        matmul(state.hb2, state.xb, TensorSlice(weights.w3, l))

        @parameter
        fn silu[_nelts: Int](i: Int):
            let initial_hb = state.hb.simd_load[_nelts](i)
            # Apply SiLU activation function (silu(x) = x * sigmoid(x))
            let hbi = initial_hb * (1.0 / (1.0 + math.exp(-initial_hb)))
            # Elementwise multiply with w3(x)
            state.hb.simd_store[_nelts](i, hbi * state.hb2.simd_load[_nelts](i))

        vectorize[nelts, silu](hidden_dim)
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
    let n = probabilities.dim(0)
    # Sample index from probabilities, they must sum to 1
    # get random value within (min, max) float32 range
    let r = DTypePointer[DType.float32].alloc(1)
    rand[DType.float32](r, 1)
    var cdf: Float32 = 0.0
    for i in range(n):
        cdf += probabilities[i]
        if r.load(0) < cdf:
            return i
    return n - 1  # In case of rounding errors


fn bpe_encode(inout tokens: DynamicVector[Int], text: String, inout tok: Tokenizer):
    for pos in range(len(text)):
        let char = str_to_ptr(text[pos])
        let id = tok.find(char)

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
            let str = str_concat(tok.vocab[tokens[i]], tok.vocab[tokens[i + 1]])
            let id = tok.find(str)
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
        let d1: Int = s[3].to_int()
        let d2: Int = s[4].to_int()
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
    workers = num_cores()
    var tokenizer = StringRef("tokenizer.bin")
    var checkpoint = StringRef("stories15M.bin")
    var temperature = 0.9
    var steps = 256
    var prompt = String("")
    var rng_seed: Int = time.now()

    @parameter
    fn argparse() raises -> Int:
        let args = argv()
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
            if args[i] == "-t":
                let val = args[i + 1]
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

    let res = argparse()
    if res == 0:
        print_usage()
        return

    print("num parallel workers:", workers, " SIMD width:", nelts)
    random.seed(rng_seed)
    var fbuf: FileBuf = FileBuf()
    var tbuf: FileBuf = FileBuf()
    var config: Config = Config()

    read_file(checkpoint, fbuf)
    config_init(config, fbuf)

    # negative vocab size is hacky way of signaling unshared weights. bit yikes.
    let shared_weights = 1 if config.vocab_size > 0 else 0
    config.vocab_size = (
        -config.vocab_size if config.vocab_size < 0 else config.vocab_size
    )

    let weights: TransformerWeights = TransformerWeights(config, shared_weights, fbuf)

    if steps <= 0 or steps > config.seq_len:
        steps = config.seq_len

    # Read in the tokenizer.bin file
    read_file(tokenizer, tbuf)
    var tok = Tokenizer(config.vocab_size, tbuf)

    # print the layers number and vocab size
    print("checkpoint size: ", fbuf.size, "[", fbuf.size // 1024 // 1024, "MB ]", 
        "| n layers:", config.n_layers, "| vocab size:", tok.vocab_size)

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

    let end = time_in_ms()
    print("\nachieved tok/s: ", (pos - 1) / (end - start) * 1000)
