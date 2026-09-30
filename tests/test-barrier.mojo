from std.collections import List
from std.sys import argv
from std.testing import assert_equal
from max.algorithm import parallelize
from llama2 import Config, RunState, TransformerWeights, Transformer, ForwardArgs, Matrix, barrier

# Run with a model path, e.g. mojo -I . tests/test-barrier.mojo stories15M.bin
def main() raises:
    var args = argv()
    var config = Config(args[1], False)
    var weights = TransformerWeights(args[1], config)
    var state = RunState(config)
    var counts: List[Int] = [1, 2, 3, 4, 5, 6, 7, 8]
    for count in counts:
        var transformer = Transformer(count)
        transformer._start(ForwardArgs(1, 0, config, state, weights))
        var workers = transformer.workers
        var values = Matrix(workers)
        var failures = Matrix(1)
        failures[0] = 0.0
        var value_ptr = values.data
        var error_ptr = failures.data

        def exercise(wid: Int) {imm}:
            # Keep the owning Transformer alive throughout the parallel region.
            var sh = transformer.sh
            for iteration in range(1000):
                value_ptr[unsafe_offset=wid] = Float32(iteration * workers + wid)
                barrier(sh, wid)
                if wid == 0:
                    for other in range(workers):
                        if value_ptr[unsafe_offset=other] != Float32(iteration * workers + other):
                            error_ptr[unsafe_offset=0] += 1.0
                # A fast writer must not overwrite the next generation before
                # the leader has checked all publications in this generation.
                barrier(sh, wid)

        parallelize(exercise, workers, workers)
        assert_equal(failures[0], Float32(0.0))
        print("PASS:", workers, "workers, 2000 barriers with shared-memory checks")
