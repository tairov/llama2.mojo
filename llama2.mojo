from algorithm import sum
from algorithm import vectorize, parallelize
from memory import memset_zero, memcpy, stack_allocation
from memory import UnsafePointer
from random import rand
from sys.info import num_physical_cores
from sys import argv
import math
import os
import random
import time

alias NUM_CONFIG_INT = 7

# SIMD width calculation - use a fixed value for now
alias nelts = 16  # 4 * typical SIMD width of 4 for float32

alias BufferPtrType = UnsafePointer[UInt8]
alias BufferPtrFloat32 = UnsafePointer[Float32]
alias PointerStrings = UnsafePointer[String]

# Simple Tensor replacement using List
struct TensorF32(Movable):
    var _data: BufferPtrFloat32
    var shape: List[Int]
    var size: Int
    
    fn __init__(out self, *dims: Int):
        var total = 1
        self.shape = List[Int]()
        for i in range(len(dims)):
            self.shape.append(dims[i])
            total *= dims[i]
        self.size = total
        self._data = BufferPtrFloat32.alloc(total)
        memset_zero(self._data, total)
    
    fn __init__(out self, var shape: List[Int], data: BufferPtrFloat32):
        self.shape = shape^
        self._data = data
        var total = 1
        for i in range(len(self.shape)):
            total *= self.shape[i]
        self.size = total
    
    fn __del__(deinit self):
        if self._data:
            self._data.free()
    
    fn data(self) -> BufferPtrFloat32:
        return self._data
    
    fn dim(self, idx: Int) -> Int:
        return self.shape[idx]
    
    fn rank(self) -> Int:
        return len(self.shape)
    
    fn num_elements(self) -> Int:
        return self.size
    
    fn __getitem__(self, idx: Int) -> Float32:
        return self._data[idx]
    
    fn __setitem__(mut self, idx: Int, val: Float32):
        self._data[idx] = val
    
    fn load[width: Int](self, idx: Int) -> SIMD[DType.float32, width]:
        return self._data.load[width=width](idx)
    
    fn store[width: Int](mut self, idx: Int, val: SIMD[DType.float32, width]):
        self._data.store[width=width](idx, val)
    
    fn argmax(self) -> Tuple[Int, Float32]:
        var max_idx = 0
        var max_val = self[0]
        for i in range(1, self.size):
            if self[i] > max_val:
                max_val = self[i]
                max_idx = i
        return Tuple(max_idx, max_val)
    
    fn __add__(self, other: Self) raises -> Self:
        var new_shape = List[Int]()
        for i in range(len(self.shape)):
            new_shape.append(self.shape[i])
        var new_data = BufferPtrFloat32.alloc(self.size)
        for i in range(self.size):
            new_data[i] = self._data[i] + other._data[i]
        return TensorF32(new_shape^, new_data)
    
    @always_inline
    fn _ptr(self) -> BufferPtrFloat32:
        return self._data


struct Accumulator[T: DType, width: Int]:
    # ideally this could be SIMD[T, width] but the width
    # in accumulate() method is compared by identity
    var data: UnsafePointer[Scalar[T]]

    @always_inline
    fn __init__(out self):
        # allocate a UnsafePointer on stack that doesn't need to be freed.
        self.data = stack_allocation[width, Scalar[T]]()
        memset_zero(self.data, width)

    @always_inline
    fn accumulate[_width: Int](mut self, val: SIMD[T, _width]) -> None:
        # This is a hack to make sure both SIMD have _width length.
        # SIMD[T, width] += SIMD[T, _width] is always an error.
        var newVal = self.data.load[width=_width]() + val
        self.data.store[width=_width](newVal)

    @always_inline
    fn total(self) -> SIMD[T, 1]:
        return self.data.load[width=width]().reduce_add()


struct TensorSlice(Movable, Copyable):
    # Provides a view into a tensor representing a 1D slice on its first or first 2 dimensions.
    # Same function signatures as Tensor but without owning the data.
    var _data: BufferPtrFloat32
    var _shape: List[Int]

    fn __init__(out self, t: TensorF32, layer: Int) raises:
        var elements_per_layer = t.num_elements() // t.dim(0)
        self._data = t.data() + layer * elements_per_layer
        self._shape = List[Int]()
        if t.rank() == 2:
            self._shape.append(t.dim(1))
        elif t.rank() == 3:
            self._shape.append(t.dim(1))
            self._shape.append(t.dim(2))
        else:
            raise Error("TensorSlice: rank greater than 3 not implemented.")

    fn __init__(out self, t: TensorF32, layer: Int, row: Int) raises:
        var elements_per_layer = t.num_elements() // t.dim(0)
        var elements_per_row = elements_per_layer // t.dim(1)
        self._data = t.data() + layer * elements_per_layer + row * elements_per_row
        self._shape = List[Int]()
        if t.rank() == 3:
            self._shape.append(t.dim(2))
        elif t.rank() == 1:
            raise Error(
                "Trying to slice a 1D Tensor by layer and row.  This requires a"
                " 3D Tensor."
            )
        else:
            raise Error("TensorSlice: rank greater than 3 not implemented.")

    fn data(self) -> BufferPtrFloat32:
        return self._data

    fn num_elements(self) -> Int:
        var total = 1
        for i in range(len(self._shape)):
            total *= self._shape[i]
        return total

    fn dim(self, idx: Int) -> Int:
        return self._shape[idx]

    fn rank(self) -> Int:
        return len(self._shape)

    fn load[width: Int](self, idx: Int) -> SIMD[DType.float32, width]:
        return self._data.load[width=width](idx)

    fn load[width: Int](self, *indices: Int) -> SIMD[DType.float32, width]:
        if len(indices) > 2:
            print(
                "Warning: TensorSlice only supports 1D and 2D indexing. "
                " Results are unlikely to be correct."
            )
        return self.load[width=width](indices[0] * self._shape[1] + indices[1])

    fn __getitem__(self, idx: Int) -> Float32:
        return self._data[idx]

    fn store[width: Int](self, idx: Int, val: SIMD[DType.float32, width]):
        return self._data.store[width=width](idx, val)

    fn __setitem__(self, idx: Int, val: Float32):
        self._data[idx] = val


# not optimal concat
fn str_concat(s1: String, s2: String) -> String:
    return s1 + s2


fn string_compare(a: String, b: String) -> Int:
    if a < b:
        return -1
    elif a > b:
        return 1
    return 0


fn wrap(token: String) -> String:
    alias a = String("\\n")
    alias b = String("\\t")
    alias c = String("'")
    alias d = String('"')
    if token == a:
        return String("\n")
    if token == b:
        return String("\t")
    if token == c:
        return String("'")
    if token == d:
        return String('"')

    return token


fn string_from_bytes(var bytes: List[UInt8]) -> String:
    # Create a String from raw bytes
    var ptr = UnsafePointer[UInt8].alloc(len(bytes) + 1)
    for i in range(len(bytes)):
        ptr[i] = bytes[i]
    ptr[len(bytes)] = 0
    return String(ptr, len(bytes))


struct Tokenizer:
    var vocab: List[String]
    var vocab_scores: List[Float32]
    var max_token_length: Int
    var vocab_size: Int

    fn __init__(out self, vocab_size: Int, filename: String) raises:
        var f = open(filename, "r")
        
        self.vocab_size = vocab_size
        self.vocab_scores = List[Float32]()
        self.vocab = List[String]()
        
        # Read max_token_length (4 bytes as int32)
        var max_token_bytes = f.read_bytes(4)
        var max_token_ptr = max_token_bytes.unsafe_ptr().bitcast[Int32]()
        self.max_token_length = Int(max_token_ptr[0])

        # read vocab_scores & vocab values (tokens)
        for i in range(self.vocab_size):
            var score_bytes = f.read_bytes(4)
            var score_ptr = score_bytes.unsafe_ptr().bitcast[Float32]()
            var score = score_ptr[0]
            
            var slen_bytes = f.read_bytes(4)
            var slen_ptr = slen_bytes.unsafe_ptr().bitcast[Int32]()
            var slen = Int(slen_ptr[0])
            
            var token_bytes = f.read_bytes(slen)
            var token = string_from_bytes(token_bytes^)
            
            self.vocab.append(token)
            self.vocab_scores.append(score)
        
        f.close()

    fn find(self, token_o: String) -> Int:
        var token = wrap(token_o)
        for i in range(len(self.vocab)):
            if self.vocab[i] == token:
                return i
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

    fn __init__(out self, fileName: String, print_config: Bool) raises:
        var f = open(fileName, "r")
        # reading 7 vars of type int32 from the file
        var bytes_of_config_params = NUM_CONFIG_INT * 4  # sizeof Int32 = 4
        var config_data_raw = f.read_bytes(bytes_of_config_params)
        f.close()
        
        # Parse config data
        var int32_ptr = config_data_raw.unsafe_ptr().bitcast[Int32]()
        self.dim = Int(int32_ptr[0])
        self.hidden_dim = Int(int32_ptr[1])
        self.n_layers = Int(int32_ptr[2])
        self.n_heads = Int(int32_ptr[3])
        self.n_kv_heads = Int(int32_ptr[4])
        self.vocab_size = Int(int32_ptr[5])
        self.seq_len = Int(int32_ptr[6])
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

    fn __init__(out self, config: Config) raises:
        self.x = TensorF32(config.dim)
        self.xb = TensorF32(config.dim)
        self.xb2 = TensorF32(config.dim)
        self.hb = TensorF32(config.hidden_dim)
        self.hb2 = TensorF32(config.hidden_dim)
        self.q = TensorF32(config.dim)
        self.att = TensorF32(config.n_heads, config.seq_len)
        self.logits = TensorF32(config.vocab_size)
        self.key_cache = TensorF32(
            config.n_layers, config.seq_len, config.kv_dim
        )
        self.value_cache = TensorF32(
            config.n_layers, config.seq_len, config.kv_dim
        )
        # So their updates flow to the caches, k and v are slices with shared memory.
        # Initialize with placeholders. The real tensors reference layer and position during forward pass.
        var placeholder = TensorF32(1, config.kv_dim)
        self.k = TensorSlice(placeholder, 0)
        self.v = TensorSlice(placeholder, 0)


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

    fn __init__(out self, file_name: String, config: Config) raises:
        var bytes_read = 0
        var f = open(file_name, "r")

        # throw away config data
        _ = f.read_bytes(NUM_CONFIG_INT * 4)  # sizeof Int32 = 4
        bytes_read += NUM_CONFIG_INT * 4

        # Read all tensors directly
        alias sizeof_float32 = 4
        
        # token_embedding_table
        var num_elements = config.vocab_size * config.dim
        var tmp = f.read_bytes(num_elements * sizeof_float32)
        bytes_read += num_elements * sizeof_float32
        var data = tmp.unsafe_ptr().bitcast[Float32]()
        var new_data = BufferPtrFloat32.alloc(num_elements)
        memcpy(dest=new_data, src=data, count=num_elements)
        var shape1 = List[Int]()
        shape1.append(config.vocab_size)
        shape1.append(config.dim)
        self.token_embedding_table = TensorF32(shape1^, new_data)

        # Continue with other weights...
        # rms_att_weight
        num_elements = config.n_layers * config.dim
        tmp = f.read_bytes(num_elements * sizeof_float32)
        bytes_read += num_elements * sizeof_float32
        data = tmp.unsafe_ptr().bitcast[Float32]()
        new_data = BufferPtrFloat32.alloc(num_elements)
        memcpy(dest=new_data, src=data, count=num_elements)
        var shape2 = List[Int]()
        shape2.append(config.n_layers)
        shape2.append(config.dim)
        self.rms_att_weight = TensorF32(shape2^, new_data)

        # wq
        num_elements = config.n_layers * config.dim * config.dim
        tmp = f.read_bytes(num_elements * sizeof_float32)
        bytes_read += num_elements * sizeof_float32
        data = tmp.unsafe_ptr().bitcast[Float32]()
        new_data = BufferPtrFloat32.alloc(num_elements)
        memcpy(dest=new_data, src=data, count=num_elements)
        var shape3 = List[Int]()
        shape3.append(config.n_layers)
        shape3.append(config.dim)
        shape3.append(config.dim)
        self.wq = TensorF32(shape3^, new_data)

        # wk
        num_elements = config.n_layers * config.kv_dim * config.dim
        tmp = f.read_bytes(num_elements * sizeof_float32)
        bytes_read += num_elements * sizeof_float32
        data = tmp.unsafe_ptr().bitcast[Float32]()
        new_data = BufferPtrFloat32.alloc(num_elements)
        memcpy(dest=new_data, src=data, count=num_elements)
        var shape4 = List[Int]()
        shape4.append(config.n_layers)
        shape4.append(config.kv_dim)
        shape4.append(config.dim)
        self.wv = TensorF32(shape4^, new_data)

        # wv
        num_elements = config.n_layers * config.kv_dim * config.dim
        tmp = f.read_bytes(num_elements * sizeof_float32)
        bytes_read += num_elements * sizeof_float32
        data = tmp.unsafe_ptr().bitcast[Float32]()
        new_data = BufferPtrFloat32.alloc(num_elements)
        memcpy(dest=new_data, src=data, count=num_elements)
        var shape5 = List[Int]()
        shape5.append(config.n_layers)
        shape5.append(config.kv_dim)
        shape5.append(config.dim)
        self.wk = TensorF32(shape5^, new_data)

        # wo
        num_elements = config.n_layers * config.dim * config.dim
        tmp = f.read_bytes(num_elements * sizeof_float32)
        bytes_read += num_elements * sizeof_float32
        data = tmp.unsafe_ptr().bitcast[Float32]()
        new_data = BufferPtrFloat32.alloc(num_elements)
        memcpy(dest=new_data, src=data, count=num_elements)
        var shape6 = List[Int]()
        shape6.append(config.n_layers)
        shape6.append(config.dim)
        shape6.append(config.dim)
        self.wo = TensorF32(shape6^, new_data)

        # rms_ffn_weight
        num_elements = config.n_layers * config.dim
        tmp = f.read_bytes(num_elements * sizeof_float32)
        bytes_read += num_elements * sizeof_float32
        data = tmp.unsafe_ptr().bitcast[Float32]()
        new_data = BufferPtrFloat32.alloc(num_elements)
        memcpy(dest=new_data, src=data, count=num_elements)
        var shape7 = List[Int]()
        shape7.append(config.n_layers)
        shape7.append(config.dim)
        self.rms_ffn_weight = TensorF32(shape7^, new_data)

        # w1
        num_elements = config.n_layers * config.hidden_dim * config.dim
        tmp = f.read_bytes(num_elements * sizeof_float32)
        bytes_read += num_elements * sizeof_float32
        data = tmp.unsafe_ptr().bitcast[Float32]()
        new_data = BufferPtrFloat32.alloc(num_elements)
        memcpy(dest=new_data, src=data, count=num_elements)
        var shape8 = List[Int]()
        shape8.append(config.n_layers)
        shape8.append(config.hidden_dim)
        shape8.append(config.dim)
        self.w1 = TensorF32(shape8^, new_data)

        # w2
        num_elements = config.n_layers * config.dim * config.hidden_dim
        tmp = f.read_bytes(num_elements * sizeof_float32)
        bytes_read += num_elements * sizeof_float32
        data = tmp.unsafe_ptr().bitcast[Float32]()
        new_data = BufferPtrFloat32.alloc(num_elements)
        memcpy(dest=new_data, src=data, count=num_elements)
        var shape9 = List[Int]()
        shape9.append(config.n_layers)
        shape9.append(config.dim)
        shape9.append(config.hidden_dim)
        self.w2 = TensorF32(shape9^, new_data)

        # w3
        num_elements = config.n_layers * config.hidden_dim * config.dim
        tmp = f.read_bytes(num_elements * sizeof_float32)
        bytes_read += num_elements * sizeof_float32
        data = tmp.unsafe_ptr().bitcast[Float32]()
        new_data = BufferPtrFloat32.alloc(num_elements)
        memcpy(dest=new_data, src=data, count=num_elements)
        var shape10 = List[Int]()
        shape10.append(config.n_layers)
        shape10.append(config.hidden_dim)
        shape10.append(config.dim)
        self.w3 = TensorF32(shape10^, new_data)

        # rms_final_weight
        num_elements = config.dim
        tmp = f.read_bytes(num_elements * sizeof_float32)
        bytes_read += num_elements * sizeof_float32
        data = tmp.unsafe_ptr().bitcast[Float32]()
        new_data = BufferPtrFloat32.alloc(num_elements)
        memcpy(dest=new_data, src=data, count=num_elements)
        var shape11 = List[Int]()
        shape11.append(config.dim)
        self.rms_final_weight = TensorF32(shape11^, new_data)

        # freq_cis_real
        num_elements = config.seq_len * (config.head_size // 2)
        tmp = f.read_bytes(num_elements * sizeof_float32)
        bytes_read += num_elements * sizeof_float32
        data = tmp.unsafe_ptr().bitcast[Float32]()
        new_data = BufferPtrFloat32.alloc(num_elements)
        memcpy(dest=new_data, src=data, count=num_elements)
        var shape12 = List[Int]()
        shape12.append(config.seq_len)
        shape12.append(config.head_size // 2)
        self.freq_cis_real = TensorF32(shape12^, new_data)

        # freq_cis_imag
        num_elements = config.seq_len * (config.head_size // 2)
        tmp = f.read_bytes(num_elements * sizeof_float32)
        bytes_read += num_elements * sizeof_float32
        data = tmp.unsafe_ptr().bitcast[Float32]()
        new_data = BufferPtrFloat32.alloc(num_elements)
        memcpy(dest=new_data, src=data, count=num_elements)
        var shape13 = List[Int]()
        shape13.append(config.seq_len)
        shape13.append(config.head_size // 2)
        self.freq_cis_imag = TensorF32(shape13^, new_data)

        if config.shared_weights:
            # Copy shape and data pointer to simulate aliasing
            var shape_copy = List[Int]()
            for i in range(len(self.token_embedding_table.shape)):
                shape_copy.append(self.token_embedding_table.shape[i])
            self.wcls = TensorF32(shape_copy^, self.token_embedding_table.data())
        else:
            # wcls
            num_elements = config.vocab_size * config.dim
            tmp = f.read_bytes(num_elements * sizeof_float32)
            bytes_read += num_elements * sizeof_float32
            data = tmp.unsafe_ptr().bitcast[Float32]()
            new_data = BufferPtrFloat32.alloc(num_elements)
            memcpy(dest=new_data, src=data, count=num_elements)
            var shape14 = List[Int]()
            shape14.append(config.vocab_size)
            shape14.append(config.dim)
            self.wcls = TensorF32(shape14^, new_data)
        f.close()
        print(
            "Total bytes read:",
            bytes_read,
            "Estimated checkpoint size: ",
            bytes_read // 1024 // 1024,
            "MB",
        )


@always_inline
fn rmsnorm(
    mut o: BufferPtrFloat32,
    x: BufferPtrFloat32,
    weight: BufferPtrFloat32,
    size: Int,
) -> None:
    # Calculate sum of squares
    var tmp = Accumulator[DType.float32, nelts]()

    @parameter
    fn _sum2[_nelts: Int](j: Int):
        tmp.accumulate((x + j).load[width=_nelts]() ** 2)

    vectorize[_sum2, nelts](size)

    var ss: Float32 = tmp.total()
    ss = ss / size + 1e-5
    ss = 1.0 / math.sqrt(ss)

    # Normalize and scale
    @parameter
    fn _norm[_nelts: Int](j: Int):
        var val = weight.load[width=_nelts](j) * ss * x.load[width=_nelts](j)
        (o + j).store[width=_nelts](val)

    vectorize[_norm, nelts](size)


@always_inline
fn rmsnorm(mut o: TensorF32, x: TensorF32, weight: TensorF32):
    var size = weight.dim(weight.rank() - 1)
    var tmp = Accumulator[DType.float32, nelts]()

    @parameter
    fn _sum2[_nelts: Int](j: Int):
        tmp.accumulate((x.data() + j).load[width=_nelts]() ** 2)

    vectorize[_sum2, nelts](size)

    var ss: Float32 = tmp.total()
    ss = ss / size + 1e-5
    ss = 1.0 / math.sqrt(ss)

    @parameter
    fn _norm[_nelts: Int](j: Int):
        var val = weight.data().load[width=_nelts](j) * ss * x.data().load[width=_nelts](j)
        (o.data() + j).store[width=_nelts](val)

    vectorize[_norm, nelts](size)


@always_inline
fn rmsnorm(mut o: TensorF32, x: TensorF32, weight: TensorSlice):
    var size = weight.dim(weight.rank() - 1)
    var tmp = Accumulator[DType.float32, nelts]()

    @parameter
    fn _sum2[_nelts: Int](j: Int):
        tmp.accumulate((x.data() + j).load[width=_nelts]() ** 2)

    vectorize[_sum2, nelts](size)

    var ss: Float32 = tmp.total()
    ss = ss / size + 1e-5
    ss = 1.0 / math.sqrt(ss)

    @parameter
    fn _norm[_nelts: Int](j: Int):
        var val = weight.data().load[width=_nelts](j) * ss * x.data().load[width=_nelts](j)
        (o.data() + j).store[width=_nelts](val)

    vectorize[_norm, nelts](size)


@always_inline
fn softmax(mut x: TensorF32) -> None:
    softmax(x, 0, x.dim(0))


@always_inline
fn softmax(mut x: TensorF32, start: Int, end: Int):
    var max_val: Float32 = -1e9

    @parameter
    fn _max[_nelts: Int](ii: Int):
        var val = x.load[width=_nelts](start + ii).reduce_max()
        if val > max_val:
            max_val = val

    vectorize[_max, nelts](end - start)

    var acc = Accumulator[DType.float32, nelts]()

    @parameter
    fn _exp[_nelts: Int](ii: Int):
        var val = math.exp(x.load[width=_nelts](start + ii) - max_val)
        x.store[width=_nelts](start + ii, val)
        acc.accumulate(val)

    vectorize[_exp, nelts](end - start)

    var ssum = acc.total()

    @parameter
    fn _norm[_nelts: Int](ii: Int):
        x.store[width=_nelts](
            start + ii, x.load[width=_nelts](start + ii) / ssum
        )

    vectorize[_norm, nelts](end - start)


# Helper function for single matrix multiplication
@always_inline
fn _single_matmul(
    C: BufferPtrFloat32,
    A: BufferPtrFloat32,
    B: BufferPtrFloat32,
    rows: Int,
    cols: Int,
    workers: Int,
):
    @parameter
    fn compute_row(i: Int):
        var tmp = Accumulator[DType.float32, nelts]()
        var row_offset = i * cols

        @parameter
        fn dot[_nelts: Int](j: Int):
            var a = A.load[width=_nelts](j)
            tmp.accumulate(a * B.load[width=_nelts](row_offset + j))

        vectorize[dot, nelts](cols)
        C[i] = tmp.total()

    parallelize[compute_row](rows, workers)


# Specialized versions for different batch sizes
@always_inline
fn batch_matmul_2(
    C0: BufferPtrFloat32,
    C1: BufferPtrFloat32,
    A: BufferPtrFloat32,
    B0: BufferPtrFloat32,
    B1: BufferPtrFloat32,
    rows: Int,
    cols: Int,
    workers: Int,
):
    @parameter
    fn compute_row(i: Int):
        var tmp0 = Accumulator[DType.float32, nelts]()
        var tmp1 = Accumulator[DType.float32, nelts]()
        var row_offset = i * cols

        @parameter
        fn dot[_nelts: Int](j: Int):
            var a = A.load[width=_nelts](j)
            tmp0.accumulate(a * B0.load[width=_nelts](row_offset + j))
            tmp1.accumulate(a * B1.load[width=_nelts](row_offset + j))

        vectorize[dot, nelts](cols)
        C0[i] = tmp0.total()
        C1[i] = tmp1.total()

    parallelize[compute_row](rows, workers)


@always_inline
fn batch_matmul_3(
    C0: BufferPtrFloat32,
    C1: BufferPtrFloat32,
    C2: BufferPtrFloat32,
    A: BufferPtrFloat32,
    B0: BufferPtrFloat32,
    B1: BufferPtrFloat32,
    B2: BufferPtrFloat32,
    rows: Int,
    cols: Int,
    workers: Int,
):
    @parameter
    fn compute_row(i: Int):
        var tmp0 = Accumulator[DType.float32, nelts]()
        var tmp1 = Accumulator[DType.float32, nelts]()
        var tmp2 = Accumulator[DType.float32, nelts]()
        var row_offset = i * cols

        @parameter
        fn dot[_nelts: Int](j: Int):
            var a = A.load[width=_nelts](j)
            tmp0.accumulate(a * B0.load[width=_nelts](row_offset + j))
            tmp1.accumulate(a * B1.load[width=_nelts](row_offset + j))
            tmp2.accumulate(a * B2.load[width=_nelts](row_offset + j))

        vectorize[dot, nelts](cols)
        C0[i] = tmp0.total()
        C1[i] = tmp1.total()
        C2[i] = tmp2.total()

    parallelize[compute_row](rows, workers)


@always_inline
fn matmul(C: TensorF32, A: TensorF32, B: TensorF32, workers: Int) raises:
    # B (d,n) @ A (n,) -> C (d,)
    if A.dim(0) != B.dim(1):
        raise Error("matmul dimension mismatch. A rows (dim 0) not equal to B columns (dim 1)")
    if B.rank() != 2:
        raise Error("matmul expects B to be a 2D matrix")
    _single_matmul(C.data(), A.data(), B.data(), B.dim(0), B.dim(1), workers)


@always_inline
fn matmul(C: TensorF32, A: TensorF32, B: TensorSlice, workers: Int) raises:
    # B (d,n) @ A (n,) -> C (d,)
    if A.dim(0) != B.dim(1):
        raise Error("matmul dimension mismatch. A rows (dim 0) not equal to B columns (dim 1)")
    if B.rank() != 2:
        raise Error("matmul expects B to be a 2D matrix")
    _single_matmul(C.data(), A.data(), B.data(), B.dim(0), B.dim(1), workers)


@always_inline
fn matmul(C: TensorSlice, A: TensorF32, B: TensorSlice, workers: Int) raises:
    # B (d,n) @ A (n,) -> C (d,)
    if A.dim(0) != B.dim(1):
        raise Error("matmul dimension mismatch. A rows (dim 0) not equal to B columns (dim 1)")
    if B.rank() != 2:
        raise Error("matmul expects B to be a 2D matrix")
    _single_matmul(C.data(), A.data(), B.data(), B.dim(0), B.dim(1), workers)


# Apply RoPE rotation to the q and k vectors for each head
# rotate odd and even dim
@always_inline
fn rope_rotation_llama(
    mut state: RunState,
    freq_cis_real_row: TensorSlice,
    freq_cis_imag_row: TensorSlice,
    config: Config,
    workers: Int,
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
    mut state: RunState,
    weights: TransformerWeights,
    workers: Int,
) raises -> None:
    # A few convenience variables
    var dim = config.dim
    var hidden_dim = config.hidden_dim
    var head_size = config.head_size
    var kv_dim = config.kv_dim
    var kv_mul = config.kv_mul

    # Copy the token embedding into x
    var content_row = weights.token_embedding_table.data() + token * dim
    memcpy(dest=state.x.data(), src=content_row, count=dim)

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
            batch_matmul_3(
                state.q.data(), state.k.data(), state.v.data(),
                state.xb.data(),
                TensorSlice(weights.wq, l).data(),
                TensorSlice(weights.wk, l).data(),
                TensorSlice(weights.wv, l).data(),
                dim,
                dim,
                workers,
            )
        else:
            matmul(state.q, state.xb, TensorSlice(weights.wq, l), workers)
            batch_matmul_2(
                state.k.data(), state.v.data(),
                state.xb.data(),
                TensorSlice(weights.wk, l).data(),
                TensorSlice(weights.wv, l).data(),
                kv_dim,
                dim,
                workers,
            )

        # Apply RoPE rotation to the q and k vectors for each head
        rope_rotation_llama(state, freq_cis_real_row, freq_cis_imag_row, config, workers)

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
                        state.q.load[width=_nelts](q_offset + i)
                        * state.key_cache.load[width=_nelts](k_offset + i)
                    ).reduce_add()

                vectorize[score_fn, nelts](head_size)
                score /= math.sqrt(Float32(head_size))

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
                    var xbi = state.xb.load[width=_nelts](
                        xb_offset + i
                    ) + a * state.value_cache.load[width=_nelts](v_offset + i)
                    state.xb.store[width=_nelts](xb_offset + i, xbi)

                vectorize[xb_accumulate, nelts](head_size)

        parallelize[loop_over_heads](config.n_heads, workers)
        # Final matrix multiplication to get the output of the attention
        matmul(state.xb2, state.xb, TensorSlice(weights.wo, l), workers)
        # Residual connection back into x
        var temp_x = state.x + state.xb2
        state.x = temp_x^
        # FFN rmsnorm
        rmsnorm(state.xb, state.x, TensorSlice(weights.rms_ffn_weight, l))

        # Calculate self.w1(x) and self.w3(x) for FFN
        batch_matmul_2(
            state.hb.data(), state.hb2.data(),
            state.xb.data(),
            TensorSlice(weights.w1, l).data(),
            TensorSlice(weights.w3, l).data(),
            hidden_dim,
            dim,
            workers,
        )

        @parameter
        fn silu[_nelts: Int](i: Int):
            var initial_hb = state.hb.load[width=_nelts](i)
            # Apply SiLU activation function (silu(x) = x * sigmoid(x))
            var hbi = initial_hb * (1.0 / (1.0 + math.exp(-initial_hb)))
            # Elementwise multiply with w3(x)
            state.hb.store[width=_nelts](
                i, hbi * state.hb2.load[width=_nelts](i)
            )

        vectorize[silu, nelts](hidden_dim)
        # Final matrix multiplication to get the output of the FFN
        matmul(state.xb, state.hb, TensorSlice(weights.w2, l), workers)

        # Residual connection
        var temp_x2 = state.x + state.xb
        state.x = temp_x2^

    # Final rmsnorm - create a temp tensor to avoid aliasing
    var final_x = TensorF32(dim)
    rmsnorm(final_x, state.x, weights.rms_final_weight)
    state.x = final_x^

    # Classifier into logits
    matmul(state.logits, state.x, weights.wcls, workers)


fn sample(probabilities: TensorF32) -> Int:
    var n = probabilities.dim(0)
    # Sample index from probabilities, they must sum to 1
    # get random value within (min, max) float32 range
    var r = Float32(random.random_float64())
    var cdf: Float32 = 0.0
    for i in range(n):
        cdf += probabilities[i]
        if r < cdf:
            return i
    return n - 1  # In case of rounding errors


fn bpe_encode(mut tokens: List[Int], text: String, tok: Tokenizer):
    for pos in range(len(text)):
        var char = String(text[pos])
        var id = tok.find(char)
        if id == -1:
            print("Not a good prompt token at pos ", pos)
            return
        tokens.append(id)

    while True:
        var best_score = Float32(-1e10)
        var best_id = -1
        var best_idx = -1

        for i in range(len(tokens) - 1):
            # Check if we can merge the pair (tokens[i], tokens[i+1])
            var str = str_concat(tok.vocab[tokens[i]], tok.vocab[tokens[i + 1]])
            var id = tok.find(str)
            if id != -1 and tok.vocab_scores[id] > best_score:
                best_score = tok.vocab_scores[id]
                best_id = id
                best_idx = i

        if best_idx == -1:
            # We couldn't find any more pairs to merge, so we're done
            break

        # Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id
        # Delete token at position best_idx+1, shift the entire sequence back 1
        var _tokens = List[Int]()
        for i in range(0, best_idx + 1):
            _tokens.append(tokens[i])
        for i in range(best_idx + 2, len(tokens)):
            _tokens.append(tokens[i])
        tokens = _tokens^


fn time_in_ms() -> Int:
    # Returns time in milliseconds for benchmarking the model speed
    return Int(time.perf_counter_ns() // 1_000_000)


fn print_usage():
    print("Usage: mojo llama2.mojo <checkpoint> [options]")
    print(
        'Example: mojo llama2.mojo stories15M.bin -s 99 -n 256 -t 0.5 -i "Llama'
        ' is an animal"'
    )
    print("Options:")
    print("  -s <int>    random seed, default time.now()")
    print("  -t <float>  temperature in [0,1.0], default 1.0")
    print(
        "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len"
    )
    print("  -i <string> input prompt")
    print("  -z          tokenizer path")
    print("  -j          number of workers to use, default num_cores()")


fn main() raises:
    var workers = num_physical_cores()
    var tokenizer = String("tokenizer.bin")
    var checkpoint = String("stories15M.bin")
    var temperature = 0.9
    var steps = 256
    var prompt = String("")
    var rng_seed: Int = Int(time.perf_counter_ns())
    var print_config = 0

    var args = argv()
    if len(args) >= 2:
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
                    print_usage()
                    return
    else:
        print_usage()
        return

    print("num parallel workers:", workers, " SIMD width:", nelts)
    random.seed(rng_seed)
    var config = Config(checkpoint, print_config == 1)
    var weights = TransformerWeights(checkpoint, config)

    if steps <= 0 or steps > config.seq_len:
        steps = config.seq_len

    var tok = Tokenizer(config.vocab_size, tokenizer)

    print(
        "n layers:",
        config.n_layers,
        "| vocab size:",
        tok.vocab_size,
    )

    # Create and initialize the application RunState
    var state = RunState(config)

    # Process the prompt, if any
    var prompt_tokens = List[Int]()

    if len(prompt) > 0:
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
        transformer(token, pos, config, state, weights, workers)

        if pos < len(prompt_tokens):
            next_token = prompt_tokens[pos]
        else:
            # Sample the next token
            if temperature == 0.0:
                # Greedy argmax sampling: take the token with the highest probability
                var argmax_result = state.logits.argmax()
                next_token = argmax_result[0]
            else:
                # Apply the temperature to the logits
                var temp_f32 = Float32(temperature)
                for q in range(config.vocab_size):
                    state.logits[q] = state.logits[q] / temp_f32

                # Apply softmax to the logits to get the probabilities for the next token
                softmax(state.logits)
                # Sample from this distribution to get the next token
                next_token = sample(state.logits)

            # Finish generating when EOS, BOS appear
            if next_token == 1 or next_token == 2:
                break
        var token_str: String = tok.vocab[next_token]
        if token == 1 and len(token_str) > 0:
            if token_str[0] == " ":
                token_str = token_str[1:]

        print(token_str, end="")

        # Advance forward
        token = next_token
        pos += 1

        if start == 0:
            start = time_in_ms()

    var end = time_in_ms()
    print("\nachieved tok/s: ", (pos - 1) / (end - start) * 1000)
