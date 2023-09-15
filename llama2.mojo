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

# The SIMD vector width.
from sys.info import simdwidthof
import math
import os
import random
import time

alias nelts = (2 * simdwidthof[DType.float32]())

alias PointerString = Pointer[UInt8]
alias BufferPtrType = DTypePointer[DType.uint8]
alias BufferPtrFloat32 = DTypePointer[DType.float32]
alias PointerStrings = Pointer[PointerString]


struct Matrix:
    var data: BufferPtrFloat32
    var rows: Int
    var cols: Int
    var layers: Int
    var allocated: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.data = BufferPtrFloat32.alloc(0)
        self.rows = rows
        self.cols = cols
        self.layers = 1
        self.allocated = 0

    fn __init__(inout self, cols: Int):
        self.data = BufferPtrFloat32.alloc(0)
        self.rows = 1
        self.layers = 1
        self.cols = cols
        self.allocated = 0

    fn __init__(inout self, layers: Int, rows: Int, cols: Int):
        self.__init__(rows, cols)
        self.layers = layers

    fn __del__(owned self):
        if self.allocated == 1:
            self.data.free()

    @always_inline
    fn alloc(inout self, fill: Int = 0):
        self.data = BufferPtrFloat32.alloc(self.size())
        self.allocated = 1
        if fill == 1:
            self.zero()

    @always_inline
    fn alloc_zero(inout self):
        self.alloc(1)

    @always_inline
    fn zero(inout self):
        memset_zero(self.data, self.size())

    @always_inline
    fn set_buf_ptr(inout self, ptr: BufferPtrFloat32):
        self.data = ptr

    # set buf ptr with redefined rows, colss
    fn set_buf_ptr(inout self, ptr: BufferPtrFloat32, rows: Int, cols: Int):
        self.data = ptr
        self.rows = rows
        self.cols = cols

    @always_inline
    fn size(inout self) -> Int:
        return self.cols * self.rows * self.layers

    @always_inline
    fn __getitem__(self, y: Int, x: Int) -> Float32:
        return self.load[1](y, x)

    @always_inline
    fn __getitem__(self, x: Int) -> Float32:
        return self.load[1](0, x)

    @always_inline
    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    @always_inline
    fn __setitem__(self, y: Int, x: Int, val: Float32):
        return self.store[1](y, x, val)

    @always_inline
    fn __setitem__(self, x: Int, val: Float32):
        return self.store[1](0, x, val)

    @always_inline
    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
        self.data.simd_store[nelts](y * self.cols + x, val)

    @always_inline
    fn load[nelts: Int](self, x: Int) -> SIMD[DType.float32, nelts]:
        return self.data.simd_load[nelts](x)

    @always_inline
    fn store[nelts: Int](self, x: Int, val: SIMD[DType.float32, nelts]):
        self.data.simd_store[nelts](x, val)

    @always_inline
    fn __getitem__(self, z: Int, y: Int, x: Int) -> Float32:
        return self.load[1](z, y, x)

    @always_inline
    fn load[nelts: Int](self, z: Int, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        return self.data.simd_load[nelts](z * self.layers + y * self.cols + x)

    @always_inline
    fn __setitem__(self, z: Int, y: Int, x: Int, val: Float32):
        return self.store[1](z, y, x, val)

    @always_inline
    fn store[nelts: Int](self, z: Int, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
        self.data.simd_store[nelts](z * self.layers + y * self.cols + x, val)


fn read_val_int(inout buf: FileBuf) -> Int:
    # DTypePointer[DType.ui8](buf.data).bitcast[DType.ui8]()
    let data = buf.data.offset(buf.offset).bitcast[DType.uint32]()
    let result = data.load(0)
    buf.offset += 4
    return result.to_int()


fn read_val_float32(inout buf: FileBuf) -> Float32:
    # DTypePointer[DType.ui8](buf.data).bitcast[DType.ui8]()
    let val = buf.data.offset(buf.offset).bitcast[DType.float32]().load(0)
    buf.offset += 4
    return val


fn read_val_str(inout buf: FileBuf, slen: Int) -> PointerString:
    let str = PointerString.alloc(slen + 1)
    for i in range(slen):
        str.store(i, buf.data.load(buf.offset))
        buf.offset += 1
    str.store(slen, 0)

    return str


# not optimal concat
fn str_concat(s1: PointerString, s2: PointerString) -> PointerString:
    var l1 = 0
    var l2 = 0

    while s1[l1] != 0:
        l1 += 1
    while s2[l2] != 0:
        l2 += 1

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


struct FileBuf:
    var data: BufferPtrType
    var offset: Int
    var size: Int

    fn __init__(inout self):
        self.data = BufferPtrType()
        self.offset = 0
        self.size = 0

    fn move_offset(inout self, size: Int):
        self.offset += size

    fn bitcast_offset_float32(inout self, size: Int) -> BufferPtrFloat32:
        let ret = self.data.offset(self.offset).bitcast[DType.float32]()
        self.offset += size * sizeof[DType.float32]()
        return ret


struct Tokenizer:
    var vocab: PointerStrings
    var vocab_scores: BufferPtrFloat32
    var max_token_length: Int
    var vocab_size: Int

    fn __init__(inout self, vocab_size: Int):
        self.vocab_size = vocab_size
        self.vocab = PointerStrings.alloc(vocab_size)
        self.vocab_scores = BufferPtrFloat32.alloc(vocab_size)
        self.max_token_length = 0


struct Config:
    var dim: Int
    var hidden_dim: Int
    var n_layers: Int
    var n_heads: Int
    var n_kv_heads: Int
    var vocab_size: Int
    var seq_len: Int

    fn __init__(inout self):
        self.dim = 0
        self.hidden_dim = 0
        self.n_layers = 0
        self.n_heads = 0
        self.n_kv_heads = 0
        self.vocab_size = 0
        self.seq_len = 0


struct RunState:
    var x: Matrix  # activation at current time stamp (dim,)
    var xb: Matrix  # same, but inside a residual branch (dim,)
    var xb2: Matrix  # an additional buffer just for convenience (dim,)
    var hb: Matrix  # buffer for hidden dimension in the ffn (hidden_dim,)
    var hb2: Matrix  # buffer for hidden dimension in the ffn (hidden_dim,)
    var q: Matrix  # query (dim,)
    var k: Matrix  # key (dim,)
    var v: Matrix  # value (dim,)
    var att: Matrix  # buffer for scores/attention values (n_heads, seq_len)
    var logits: Matrix  # output logits
    var key_cache: Matrix  # (layer, seq_len, dim)
    var value_cache: Matrix  # (layer, seq_len, dim)
    var rt: Runtime

    fn __init__(inout self, config: Config):
        self.x = Matrix(config.dim)
        self.x.alloc_zero()
        self.xb = Matrix(config.dim)
        self.xb.alloc_zero()
        self.xb2 = Matrix(config.dim)
        self.xb2.alloc_zero()
        self.hb = Matrix(config.hidden_dim)
        self.hb.alloc_zero()
        self.hb2 = Matrix(config.hidden_dim)
        self.hb2.alloc_zero()
        self.q = Matrix(config.dim)
        self.q.alloc_zero()
        self.k = Matrix(config.dim)
        self.k.alloc_zero()
        self.v = Matrix(config.dim)
        self.v.alloc_zero()
        self.att = Matrix(config.n_heads, config.seq_len)
        self.att.alloc_zero()
        self.logits = Matrix(config.vocab_size)
        self.logits.alloc_zero()
        self.key_cache = Matrix(config.n_layers, config.seq_len, config.dim)
        self.key_cache.alloc_zero()
        self.value_cache = Matrix(config.n_layers, config.seq_len, config.dim)
        self.value_cache.alloc_zero()
        self.rt = Runtime(num_cores() // 2)


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

    fn __init__(inout self, config: Config, shared_weights: Int, inout buf: FileBuf):
        self.token_embedding_table = Matrix(config.vocab_size, config.dim)
        # set buf ptr to buf data from file
        self.token_embedding_table.set_buf_ptr(
            buf.bitcast_offset_float32(self.token_embedding_table.size())
        )
        self.rms_att_weight = Matrix(config.n_layers, config.dim)
        self.rms_att_weight.set_buf_ptr(
            buf.bitcast_offset_float32(self.rms_att_weight.size())
        )
        self.wq = Matrix(config.n_layers, config.dim, config.dim)
        self.wq.set_buf_ptr(buf.bitcast_offset_float32(self.wq.size()))
        self.wk = Matrix(config.n_layers, config.dim, config.dim)
        self.wk.set_buf_ptr(buf.bitcast_offset_float32(self.wk.size()))
        self.wv = Matrix(config.n_layers, config.dim, config.dim)
        self.wv.set_buf_ptr(buf.bitcast_offset_float32(self.wv.size()))
        self.wo = Matrix(config.n_layers, config.dim, config.dim)
        self.wo.set_buf_ptr(buf.bitcast_offset_float32(self.wo.size()))
        self.rms_ffn_weight = Matrix(config.n_layers, config.dim)
        self.rms_ffn_weight.set_buf_ptr(
            buf.bitcast_offset_float32(self.rms_ffn_weight.size())
        )
        self.w1 = Matrix(config.n_layers, config.dim, config.hidden_dim)
        self.w1.set_buf_ptr(buf.bitcast_offset_float32(self.w1.size()))
        self.w2 = Matrix(config.n_layers, config.dim, config.hidden_dim)
        self.w2.set_buf_ptr(buf.bitcast_offset_float32(self.w2.size()))
        self.w3 = Matrix(config.n_layers, config.dim, config.hidden_dim)
        self.w3.set_buf_ptr(buf.bitcast_offset_float32(self.w3.size()))
        self.rms_final_weight = Matrix(config.dim)
        self.rms_final_weight.set_buf_ptr(
            buf.bitcast_offset_float32(self.rms_final_weight.size())
        )
        self.freq_cis_real = Matrix(config.seq_len, (config.dim // config.n_heads) // 2)
        self.freq_cis_real.set_buf_ptr(
            buf.bitcast_offset_float32(self.freq_cis_real.size())
        )
        self.freq_cis_imag = Matrix(config.seq_len, (config.dim // config.n_heads) // 2)
        self.freq_cis_imag.set_buf_ptr(
            buf.bitcast_offset_float32(self.freq_cis_imag.size())
        )
        self.wcls = Matrix(
            config.vocab_size, config.dim
        )  # if shared_weights else rest_floats
        self.wcls.set_buf_ptr(self.token_embedding_table.data)


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
    return None


fn tokenizer_init(inout tok: Tokenizer, inout buf: FileBuf) -> None:
    tok.max_token_length = read_val_int(buf)
    tok.vocab_scores = BufferPtrFloat32.alloc(tok.vocab_size)
    tok.vocab = PointerStrings.alloc(tok.vocab_size)

    # read vocab_scores & vocab values (tokens)
    for i in range(0, tok.vocab_size):
        tok.vocab_scores.store(i, read_val_float32(buf))
        let slen = read_val_int(buf)
        tok.vocab.store(i, read_val_str(buf, slen))

    tok.vocab_scores = buf.data.offset(buf.offset).bitcast[DType.float32]()
    buf.offset += tok.vocab_size * 4
    return None


fn accum(inout a: BufferPtrFloat32, b: BufferPtrFloat32, size: Int) -> None:
    @parameter
    fn _acc[_nelts: Int](j: Int):
        a.offset(j).simd_store[_nelts](
            0, a.offset(j).simd_load[_nelts](0) + b.offset(j).simd_load[_nelts](0)
        )

    vectorize[nelts, _acc](size)


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


fn softmax(inout x: BufferPtrFloat32, size: Int) -> None:
    # Find max value (for numerical stability)
    var max_val: Float32 = -1e9

    @parameter
    fn _max[_nelts: Int](j: Int):
        let val = x.simd_load[_nelts](j).reduce_max()
        if val > max_val:
            max_val = val

    vectorize[nelts, _max](size)

    # Exp and sum
    var ssum: Float32 = 0.0

    @parameter
    fn _sum_exp[_nelts: Int](j: Int):
        x.simd_store[_nelts](j, math.exp(x.simd_load[_nelts](j) - max_val))
        ssum += x.simd_load[_nelts](j).reduce_add()

    vectorize[nelts, _sum_exp](size)

    @parameter
    fn _norm[_nelts: Int](j: Int):
        x.simd_store[_nelts](j, x.simd_load[_nelts](j) / ssum)

    vectorize[nelts, _norm](size)


fn matmul_parallelized(C: Matrix, A: Matrix, B: Matrix, rt: Runtime):
    @parameter
    fn compute_row(i: Int):
        var tmp = SIMD[DType.float32, nelts](0)

        @parameter
        fn dot[_nelts: Int](j: Int):
            if _nelts < nelts:  # take care of tail array elements with length <  nelts
                tmp[0] += (A.load[_nelts](j) * B.load[_nelts](i, j)).reduce_add()
            else:
                tmp += A.load[nelts](j) * B.load[nelts](i, j)

        vectorize[nelts, dot](B.cols)
        C[i] = tmp.reduce_add()

    parallelize[compute_row](rt, B.rows, rt.parallelism_level())


fn matmul(inout C: Matrix, A: Matrix, B: Matrix, rt: Runtime) -> None:
    # B (d,n) @ A (n,) -> C (d,)
    matmul_parallelized(C, A, B, rt)


fn transformer(
    token: Int,
    pos: Int,
    config: Config,
    inout state: RunState,
    weights: TransformerWeights,
) -> None:
    # A few convenience variables
    var x = state.x.data
    let dim = config.dim
    let hidden_dim = config.hidden_dim
    let head_size = dim // config.n_heads

    # tmp matrix for matmul operations
    var tmpw = Matrix(0, 0)

    # Copy the token embedding into x
    let content_row = weights.token_embedding_table.data.offset(token * dim)
    memcpy[DType.float32](x, content_row, config.dim)

    # Pluck out the "pos" row of freq_cis_real and freq_cis_imag
    let freq_cis_real_row = weights.freq_cis_real.data.offset(pos * head_size // 2)
    let freq_cis_imag_row = weights.freq_cis_imag.data.offset(pos * head_size // 2)

    # Forward all the layers
    for l in range(config.n_layers):
        # Attention rmsnorm
        rmsnorm(state.xb.data, x, weights.rms_att_weight.data.offset(l * dim), dim)

        # QKV matmuls for this position
        tmpw.set_buf_ptr(weights.wq.data.offset(l * dim * dim), dim, dim)
        matmul(state.q, state.xb, tmpw, state.rt)

        tmpw.set_buf_ptr(weights.wk.data.offset(l * dim * dim), dim, dim)
        matmul(state.k, state.xb, tmpw, state.rt)

        tmpw.set_buf_ptr(weights.wv.data.offset(l * dim * dim), dim, dim)
        matmul(state.v, state.xb, tmpw, state.rt)

        # Apply RoPE rotation to the q and k vectors for each head
        for h in range(config.n_heads):
            # Get the q and k vectors for this head
            let q = state.q.data.offset(h * head_size)
            let k = state.k.data.offset(h * head_size)

            # Rotate q and k by the freq_cis_real and freq_cis_imag
            for i in range(0, head_size, 2):
                let q0 = q.offset(i).load(0)
                let q1 = q.offset(i + 1).load(0)
                let k0 = k.offset(i).load(0)
                let k1 = k.offset(i + 1).load(0)
                let fcr = freq_cis_real_row.offset(i // 2).load(0)
                let fci = freq_cis_imag_row.offset(i // 2).load(0)
                q.offset(i).store(0, q0 * fcr - q1 * fci)
                q.offset(i + 1).store(0, q0 * fci + q1 * fcr)
                k.offset(i).store(0, k0 * fcr - k1 * fci)
                k.offset(i + 1).store(0, k0 * fci + k1 * fcr)

        # Save key,value at this time step (pos) to our kv cache
        let loff = l * config.seq_len * dim  # kv cache layer offset for convenience
        let key_cache_row = state.key_cache.data.offset(loff + pos * dim)
        let value_cache_row = state.value_cache.data.offset(loff + pos * dim)
        memcpy[DType.float32](key_cache_row, state.k.data, config.dim)
        memcpy[DType.float32](value_cache_row, state.v.data, config.dim)

        # Multihead attention. Iterate over all heads
        for h in range(config.n_heads):
            # Get the query vector for this head
            let q = state.q.data.offset(h * head_size)

            # Attention scores for this head
            var att = state.att.data.offset(h * config.seq_len)

            # Iterate over all timesteps, including the current one
            for t in range(pos + 1):
                # Get the key vector for this head and at this timestep
                let k = state.key_cache.data.offset(loff + t * dim + h * head_size)
                # Calculate the attention score as the dot product of q and k
                var score: Float32 = 0.0
                for i in range(head_size):
                    score += q.offset(i).load(0) * k.offset(i).load(0)
                score /= math.sqrt[DType.float32, 1](head_size)

                # Save the score to the attention buffer
                att.offset(t).store(0, score)

            # Softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1)

            # Weighted sum of the values, store back into xb
            let xb = state.xb.data.offset(h * head_size)
            memset_zero(xb, head_size)
            for t in range(pos + 1):
                # Get the value vector for this head and at this timestep
                let v = state.value_cache.data.offset(loff + t * dim + h * head_size)
                # Get the attention weight for this timestep
                let a = att.offset(t).load(0)
                # Accumulate the weighted value into xb
                for i in range(head_size):
                    let xbi = xb.offset(i).load(0) + a * v.offset(i).load(0)
                    xb.offset(i).store(0, xbi)
        # Final matrix multiplication to get the output of the attention
        tmpw.set_buf_ptr(weights.wo.data.offset(l * dim * dim), dim, dim)
        matmul(state.xb2, state.xb, tmpw, state.rt)

        # Residual connection back into x
        accum(x, state.xb2.data, dim)

        # FFN rmsnorm
        rmsnorm(state.xb.data, x, weights.rms_ffn_weight.data.offset(l * dim), dim)

        # Calculate self.w1(x) and self.w3(x) for FFN
        tmpw.set_buf_ptr(weights.w1.data.offset(l * dim * hidden_dim), hidden_dim, dim)
        matmul(state.hb, state.xb, tmpw, state.rt)

        tmpw.set_buf_ptr(weights.w3.data.offset(l * dim * hidden_dim), hidden_dim, dim)
        matmul(state.hb2, state.xb, tmpw, state.rt)

        # Apply SiLU activation function (silu(x) = x * sigmoid(x))
        for i in range(hidden_dim):
            let hbi = state.hb[i]
            state.hb[i] = hbi * (1.0 / (1.0 + math.exp(-hbi)))

        # Elementwise multiply with w3(x)
        for i in range(hidden_dim):
            state.hb[i] = state.hb[i] * state.hb2[i]

        # Final matrix multiplication to get the output of the FFN
        tmpw.set_buf_ptr(weights.w2.data.offset(l * dim * hidden_dim), dim, hidden_dim)
        matmul(state.xb, state.hb, tmpw, state.rt)

        # Residual connection
        accum(x, state.xb.data, dim)

    # Final rmsnorm
    rmsnorm(x, x, weights.rms_final_weight.data, dim)

    # Classifier into logits
    tmpw.set_buf_ptr(weights.wcls.data, config.vocab_size, dim)
    matmul(state.logits, state.x, tmpw, state.rt)


fn argmax(v: Matrix) -> Int:
    # return argmax of v
    var max_i: Int = 0
    var max_p: Float32 = v[0]
    for i in range(v.cols):
        if v[i] > max_p:
            max_i = i
            max_p = v[i]
    return max_i


fn sample(probabilities: Matrix) -> Int:
    let n = probabilities.cols
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


fn str_lookup(str: PointerString, tok: Tokenizer) -> Int:
    for pos in range(tok.vocab_size):
        let s1 = tok.vocab[pos]
        var p1 = 0
        while s1[p1] != 0 and str[p1] != 0:
            if s1[p1] != str[p1]:
                break
            p1 += 1
        if s1[p1] != 0 or str[p1] != 0:
            continue
        return pos
    return -1


fn bpe_encode(inout tokens: DynamicVector[Int], text: String, tok: Tokenizer):
    for pos in range(len(text)):
        let char = str_to_ptr(text[pos])
        let id = str_lookup(char, tok)

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
            let id = str_lookup(str, tok)
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


fn print_str(s: PointerString):
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


fn main() raises:
    print("num hardware threads: ", num_cores())
    print("SIMD vector width: ", nelts)
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
            if args[i] == "-s":
                rng_seed = atol(args[i + 1])
            if args[i] == "-i":
                prompt = args[i + 1]
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

    random.seed(rng_seed)
    var fbuf: FileBuf = FileBuf()
    var tbuf: FileBuf = FileBuf()
    var config: Config = Config()

    read_file(checkpoint, fbuf)
    print("checkpoint size: ", fbuf.size)
    config_init(config, fbuf)

    # negative vocab size is hacky way of signaling unshared weights. bit yikes.
    let shared_weights = 1 if config.vocab_size > 0 else 0
    config.vocab_size = (
        -config.vocab_size if config.vocab_size < 0 else config.vocab_size
    )

    let weights: TransformerWeights = TransformerWeights(config, shared_weights, fbuf)

    var tok: Tokenizer = Tokenizer(config.vocab_size)

    if steps <= 0 or steps > config.seq_len:
        steps = config.seq_len

    # Read in the tokenizer.bin file
    read_file(tokenizer, tbuf)
    tokenizer_init(tok, tbuf)

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
                softmax(state.logits.data, config.vocab_size)
                # Sample from this distribution to get the next token
                next_token = sample(state.logits)

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
    print("\nachieved tok/s: ", (steps - 1) / (end - start) * 1000)
