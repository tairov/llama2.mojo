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
from tensor import Tensor, TensorSpec, TensorShape
from utils.index import Index

alias nelts = (2 * simdwidthof[DType.float32]())

alias PointerString = Pointer[UInt8]
alias BufferPtrType = DTypePointer[DType.uint8]
alias BufferPtrFloat32 = DTypePointer[DType.float32]
alias PointerStrings = Pointer[PointerString]

alias TensorF32 = Tensor[DType.float32]


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

    fn bitcast_offset_f32(inout self, size: Int) -> BufferPtrFloat32:
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
    var x: TensorF32  # activation at current time stamp (dim,)
    var xb: TensorF32  # same, but inside a residual branch (dim,)
    var xb2: TensorF32  # an additional buffer just for convenience (dim,)
    var hb: TensorF32  # buffer for hidden dimension in the ffn (hidden_dim,)
    var hb2: TensorF32  # buffer for hidden dimension in the ffn (hidden_dim,)
    var q: TensorF32  # query (dim,)
    var k: TensorF32  # key (dim,)
    var v: TensorF32  # value (dim,)
    var att: TensorF32  # buffer for scores/attention values (n_heads, seq_len)
    var logits: TensorF32  # output logits
    var key_cache: TensorF32  # (layer, seq_len, dim)
    var value_cache: TensorF32  # (layer, seq_len, dim)
    var rt: Runtime

    fn __init__(inout self, config: Config):
        self.x = TensorF32(config.dim)
        self.xb = TensorF32(config.dim)
        self.xb2 = TensorF32(config.dim)
        self.hb = TensorF32(config.hidden_dim)
        self.hb2 = TensorF32(config.hidden_dim)
        self.q = TensorF32(config.dim)
        self.k = TensorF32(config.dim)
        self.v = TensorF32(config.dim)
        self.att = TensorF32(config.n_heads, config.seq_len)
        self.logits = TensorF32(config.vocab_size)
        self.key_cache = TensorF32(config.n_layers, config.seq_len, config.dim)
        self.value_cache = TensorF32(config.n_layers, config.seq_len, config.dim)
        self.rt = Runtime(num_cores() // 2)


fn get_tspec_f32(*dims: Int) -> TensorSpec:
    let spec = TensorSpec(DType.float32, dims)
    return spec


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

    fn __init__(inout self, config: Config, shared_weights: Int, inout buf: FileBuf):
        var tspec = get_tspec_f32(config.vocab_size, config.dim)
        # __init__(owned ptr: DTypePointer[dtype], owned spec: TensorSpec)
        self.token_embedding_table = TensorF32(
            buf.bitcast_offset_f32(tspec.bytecount()), tspec
        )
        # set buf ptr to buf data from file
        tspec = get_tspec_f32(config.n_layers, config.dim)
        self.rms_att_weight = TensorF32(
            buf.bitcast_offset_f32(tspec.bytecount()), tspec
        )
        tspec = get_tspec_f32(config.n_layers, config.dim, config.dim)
        self.wq = TensorF32(buf.bitcast_offset_f32(tspec.bytecount()), tspec)
        tspec = get_tspec_f32(config.n_layers, config.dim, config.dim)
        self.wk = TensorF32(buf.bitcast_offset_f32(tspec.bytecount()), tspec)
        tspec = get_tspec_f32(config.n_layers, config.dim, config.dim)
        self.wv = TensorF32(buf.bitcast_offset_f32(tspec.bytecount()), tspec)
        tspec = get_tspec_f32(config.n_layers, config.dim, config.dim)
        self.wo = TensorF32(buf.bitcast_offset_f32(tspec.bytecount()), tspec)
        tspec = get_tspec_f32(config.n_layers, config.dim)
        self.rms_ffn_weight = TensorF32(
            buf.bitcast_offset_f32(tspec.bytecount()), tspec
        )
        tspec = get_tspec_f32(config.n_layers, config.dim, config.hidden_dim)
        self.w1 = TensorF32(buf.bitcast_offset_f32(tspec.bytecount()), tspec)
        tspec = get_tspec_f32(config.n_layers, config.dim, config.hidden_dim)
        self.w2 = TensorF32(buf.bitcast_offset_f32(tspec.bytecount()), tspec)
        tspec = get_tspec_f32(config.n_layers, config.dim, config.hidden_dim)
        self.w3 = TensorF32(buf.bitcast_offset_f32(tspec.bytecount()), tspec)
        tspec = get_tspec_f32(config.dim)
        self.rms_final_weight = TensorF32(
            buf.bitcast_offset_f32(tspec.bytecount()), tspec
        )
        tspec = get_tspec_f32(config.seq_len, (config.dim // config.n_heads) // 2)
        self.freq_cis_real = TensorF32(buf.bitcast_offset_f32(tspec.bytecount()), tspec)
        tspec = get_tspec_f32(config.seq_len, (config.dim // config.n_heads) // 2)
        self.freq_cis_imag = TensorF32(buf.bitcast_offset_f32(tspec.bytecount()), tspec)
        tspec = get_tspec_f32(config.vocab_size, config.dim)
        self.wcls = TensorF32(buf.bitcast_offset_f32(tspec.bytecount()), tspec)


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


fn accum(inout a: TensorF32, b: TensorF32) -> None:
    let size = a.dim(0)

    @parameter
    fn _acc[_nelts: Int](j: Int):
        a.simd_store[_nelts](j, a.simd_load[_nelts](j) + b.simd_load[_nelts](j))

    vectorize[nelts, _acc](size)


fn rmsnorm(inout o: TensorF32, x: TensorF32, weight: TensorF32) -> None:
    # Calculate sum of squares
    var tmp = SIMD[DType.float32, nelts](0)
    let size = x.dim(0)

    @parameter
    fn _sum2[_nelts: Int](j: Int):
        if _nelts < nelts:
            tmp[0] += (x.simd_load[_nelts](j) ** 2).reduce_add()
        else:
            tmp += x.simd_load[nelts](j) ** 2

    vectorize[nelts, _sum2](size)

    var ss: Float32 = tmp.reduce_add()
    ss = ss / size + 1e-5
    ss = 1.0 / math.sqrt(ss)

    # Normalize and scale
    @parameter
    fn _norm[_nelts: Int](j: Int):
        let val = weight.simd_load[_nelts](j) * ss * x.simd_load[_nelts](j)
        o.simd_store[_nelts](j, val)

    vectorize[nelts, _norm](size)


fn softmax(inout x: TensorF32) -> None:
    # Find max value (for numerical stability)
    var max_val: Float32 = -1e9
    let size = x.dim(0)

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


fn matmul_parallelized(inout C: TensorF32, A: TensorF32, B: TensorF32, rt: Runtime):
    @parameter
    fn compute_row(i: Int):
        var tmp = SIMD[DType.float32, nelts](0)

        @parameter
        fn dot[_nelts: Int](j: Int):
            if _nelts < nelts:  # take care of tail array elements with length <  nelts
                tmp[0] += (
                    A.simd_load[_nelts](j) * B.simd_load[_nelts](i, j)
                ).reduce_add()
            else:
                tmp += A.simd_load[nelts](j) * B.simd_load[nelts](i, j)

        vectorize[nelts, dot](B.dim(0))
        C[Index(i)] = tmp.reduce_add()

    parallelize[compute_row](rt, B.dim(1), rt.parallelism_level())


fn matmul(inout C: TensorF32, A: TensorF32, B: TensorF32, rt: Runtime) -> None:
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
    var x = state.x
    let dim = config.dim
    let hidden_dim = config.hidden_dim
    let head_size = dim // config.n_heads

    # tmp matrix for matmul operations
    var tmpw = TensorF32()

    # Copy the token embedding into x
    let content_row = weights.token_embedding_table.data().offset(token * dim)
    memcpy[DType.float32](x.data(), content_row, config.dim)

    # Pluck out the "pos" row of freq_cis_real and freq_cis_imag
    let freq_cis_real_row = weights.freq_cis_real.data().offset(pos * head_size // 2)
    let freq_cis_imag_row = weights.freq_cis_imag.data().offset(pos * head_size // 2)

    # Forward all the layers
    for l in range(config.n_layers):
        # Attention rmsnorm
        tmpw = TensorF32(
            weights.rms_att_weight.data().offset(l * dim), get_tspec_f32(dim)
        )
        rmsnorm(state.xb, x, tmpw)

        # QKV matmuls for this position
        tmpw = TensorF32(
            weights.wq.data().offset(l * dim * dim), get_tspec_f32(dim, dim)
        )
        matmul(state.q, state.xb, tmpw, state.rt)

        tmpw = TensorF32(
            weights.wk.data().offset(l * dim * dim), get_tspec_f32(dim, dim)
        )
        matmul(state.k, state.xb, tmpw, state.rt)

        tmpw = TensorF32(
            weights.wv.data().offset(l * dim * dim), get_tspec_f32(dim, dim)
        )
        matmul(state.v, state.xb, tmpw, state.rt)

        # Apply RoPE rotation to the q and k vectors for each head
        for h in range(config.n_heads):
            # Get the q and k vectors for this head
            let q = state.q.data().offset(h * head_size)
            let k = state.k.data().offset(h * head_size)

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
        let key_cache_row = state.key_cache.data().offset(loff + pos * dim)
        let value_cache_row = state.value_cache.data().offset(loff + pos * dim)
        memcpy[DType.float32](key_cache_row, state.k.data(), config.dim)
        memcpy[DType.float32](value_cache_row, state.v.data(), config.dim)

        # Multihead attention. Iterate over all heads
        for h in range(config.n_heads):
            # Get the query vector for this head
            let q = state.q.data().offset(h * head_size)

            # Attention scores for this head
            var att = TensorF32(
                state.att.data().offset(h * config.seq_len),
                get_tspec_f32(config.seq_len),
            )

            # Iterate over all timesteps, including the current one
            for t in range(pos + 1):
                # Get the key vector for this head and at this timestep
                let k = state.key_cache.data().offset(loff + t * dim + h * head_size)
                # Calculate the attention score as the dot product of q and k
                var score: Float32 = 0.0
                for i in range(head_size):
                    score += q.offset(i).load(0) * k.offset(i).load(0)
                score /= math.sqrt[DType.float32, 1](head_size)

                # Save the score to the attention buffer
                att[t] = score

            # Softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att)

            # Weighted sum of the values, store back into xb
            let xb = state.xb.data().offset(h * head_size)
            memset_zero(xb, head_size)
            for t in range(pos + 1):
                # Get the value vector for this head and at this timestep
                let v = state.value_cache.data().offset(loff + t * dim + h * head_size)
                # Get the attention weight for this timestep
                let a = att[t]
                # Accumulate the weighted value into xb
                for i in range(head_size):
                    let xbi = xb.offset(i).load(0) + a * v.offset(i).load(0)
                    xb.offset(i).store(0, xbi)
        # Final matrix multiplication to get the output of the attention
        tmpw = TensorF32(
            weights.wo.data().offset(l * dim * dim), get_tspec_f32(dim, dim)
        )
        matmul(state.xb2, state.xb, tmpw, state.rt)

        # Residual connection back into x
        accum(x, state.xb2)

        # FFN rmsnorm
        tmpw = TensorF32(
            weights.rms_ffn_weight.data().offset(l * dim), get_tspec_f32(dim)
        )
        rmsnorm(state.xb, x, tmpw)

        # Calculate self.w1(x) and self.w3(x) for FFN
        tmpw = TensorF32(
            weights.w1.data().offset(l * dim * hidden_dim),
            get_tspec_f32(dim, hidden_dim),
        )
        matmul(state.hb, state.xb, tmpw, state.rt)

        tmpw = TensorF32(
            weights.w3.data().offset(l * dim * hidden_dim),
            get_tspec_f32(dim, hidden_dim),
        )
        matmul(state.hb2, state.xb, tmpw, state.rt)

        # Apply SiLU activation function (silu(x) = x * sigmoid(x))
        for i in range(hidden_dim):
            let hbi = state.hb[i]
            state.hb[i] = hbi * (1.0 / (1.0 + math.exp(-hbi)))

        # Elementwise multiply with w3(x)
        for i in range(hidden_dim):
            state.hb[i] = state.hb[i] * state.hb2[i]

        # Final matrix multiplication to get the output of the FFN
        tmpw = TensorF32(
            weights.w2.data().offset(l * dim * hidden_dim),
            get_tspec_f32(dim, hidden_dim),
        )
        matmul(state.xb, state.hb, tmpw, state.rt)

        # Residual connection
        accum(x, state.xb)

    # Final rmsnorm
    rmsnorm(x, x, weights.rms_final_weight)

    # Classifier into logits
    tmpw = TensorF32(weights.wcls.data(), get_tspec_f32(config.vocab_size, dim))
    matmul(state.logits, state.x, tmpw, state.rt)


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
    _ = fbuf.data
    _ = tbuf.data

    read_file(checkpoint, fbuf)
    print("checkpoint size: ", fbuf.size)
    config_init(config, fbuf)

    # negative vocab size is hacky way of signaling unshared weights. bit yikes.
    let shared_weights = 1 if config.vocab_size > 0 else 0
    config.vocab_size = (
        -config.vocab_size if config.vocab_size < 0 else config.vocab_size
    )

    
    print("good to go!")

    let weights: TransformerWeights = TransformerWeights(config, shared_weights, fbuf)
    
    print("crashing")

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
                softmax(state.logits)
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
