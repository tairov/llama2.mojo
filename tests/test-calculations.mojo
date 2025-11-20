from testing import assert_true, assert_almost_equal, assert_equal
from llama2 import Matrix, rmsnorm, softmax, argmax, sample
import math

fn test_rmsnorm_basic() raises:
    """Test basic RMS normalization."""
    print("\nTesting basic rmsnorm:")
    
    # Create input, output, and weight matrices
    var x = Matrix(4)
    var o = Matrix(4)
    var weight = Matrix(4)
    
    # Fill x with [1, 2, 3, 4]
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    x[3] = 4.0
    
    # Fill weight with all ones (no scaling)
    for i in range(4):
        weight[i] = 1.0
    
    print("  Input x:", x[0], x[1], x[2], x[3])
    print("  Weight: all ones")
    
    rmsnorm(o.data, x.data, weight.data, 4)
    
    print("  Output o:", o[0], o[1], o[2], o[3])
    
    # RMS = sqrt((1^2 + 2^2 + 3^2 + 4^2) / 4) = sqrt(7.5) ≈ 2.739
    # With epsilon: rms = sqrt(7.5 + 1e-5) ≈ 2.739
    # Normalized values: x[i] / rms
    var rms = math.sqrt((Float32(1.0)**2 + Float32(2.0)**2 + Float32(3.0)**2 + Float32(4.0)**2) / Float32(4.0) + Float32(1e-5))
    
    assert_almost_equal(o[0], Float32(1.0) / rms, atol=0.001)
    assert_almost_equal(o[1], Float32(2.0) / rms, atol=0.001)
    assert_almost_equal(o[2], Float32(3.0) / rms, atol=0.001)
    assert_almost_equal(o[3], Float32(4.0) / rms, atol=0.001)
    
    print("✓ Basic rmsnorm test passed")

fn test_rmsnorm_with_weights() raises:
    """Test RMS normalization with non-uniform weights."""
    print("\nTesting rmsnorm with custom weights:")
    
    var x = Matrix(3)
    var o = Matrix(3)
    var weight = Matrix(3)
    
    # Fill x with [2, 2, 2]
    for i in range(3):
        x[i] = 2.0
    
    # Fill weight with [1, 2, 3]
    weight[0] = 1.0
    weight[1] = 2.0
    weight[2] = 3.0
    
    print("  Input x: all 2.0")
    print("  Weight:", weight[0], weight[1], weight[2])
    
    rmsnorm(o.data, x.data, weight.data, 3)
    
    print("  Output o:", o[0], o[1], o[2])
    
    # RMS = sqrt((2^2 + 2^2 + 2^2) / 3 + 1e-5) = sqrt(4 + 1e-5) ≈ 2.0
    # ss = 1.0 / rms = 0.5
    # o[i] = weight[i] * ss * x[i] = weight[i] * 0.5 * 2.0 = weight[i]
    var rms = math.sqrt((Float32(2.0)**2 + Float32(2.0)**2 + Float32(2.0)**2) / Float32(3.0) + Float32(1e-5))
    var ss = Float32(1.0) / rms
    
    assert_almost_equal(o[0], weight[0] * ss * Float32(2.0), atol=0.001)
    assert_almost_equal(o[1], weight[1] * ss * Float32(2.0), atol=0.001)
    assert_almost_equal(o[2], weight[2] * ss * Float32(2.0), atol=0.001)
    
    print("✓ Rmsnorm with weights test passed")

fn test_rmsnorm_zeros() raises:
    """Test RMS normalization with zero input (tests epsilon handling)."""
    print("\nTesting rmsnorm with zero input:")
    
    var x = Matrix(4)
    var o = Matrix(4)
    var weight = Matrix(4)
    
    # Fill with zeros
    x.zero()
    for i in range(4):
        weight[i] = 1.0
    
    rmsnorm(o.data, x.data, weight.data, 4)
    
    print("  Input x: all zeros")
    print("  Output o:", o[0], o[1], o[2], o[3])
    
    # With epsilon: ss = 1.0 / sqrt(1e-5) ≈ 316.23
    # o[i] = 1.0 * 316.23 * 0.0 = 0.0
    for i in range(4):
        assert_almost_equal(o[i], 0.0, atol=0.001)
    
    print("✓ Rmsnorm with zeros test passed")

fn test_softmax_basic() raises:
    """Test basic softmax function."""
    print("\nTesting basic softmax:")
    
    var x = Matrix(3)
    
    # Fill with [1, 2, 3]
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    
    print("  Input x:", x[0], x[1], x[2])
    
    softmax(x.data, 3)
    
    print("  After softmax:", x[0], x[1], x[2])
    
    # Expected: exp(1-3), exp(2-3), exp(3-3) = exp(-2), exp(-1), exp(0)
    # Then normalize by sum
    var e0 = math.exp(Float32(-2.0))
    var e1 = math.exp(Float32(-1.0))
    var e2 = math.exp(Float32(0.0))
    var sum = e0 + e1 + e2
    
    assert_almost_equal(x[0], e0 / sum, atol=0.001)
    assert_almost_equal(x[1], e1 / sum, atol=0.001)
    assert_almost_equal(x[2], e2 / sum, atol=0.001)
    
    # Sum should be 1.0
    var total = x[0] + x[1] + x[2]
    assert_almost_equal(total, 1.0, atol=0.001)
    
    print("  Sum of probabilities:", total)
    print("✓ Basic softmax test passed")

fn test_softmax_uniform() raises:
    """Test softmax with uniform input."""
    print("\nTesting softmax with uniform input:")
    
    var x = Matrix(4)
    
    # Fill with all 5.0
    for i in range(4):
        x[i] = 5.0
    
    print("  Input x: all 5.0")
    
    softmax(x.data, 4)
    
    print("  After softmax:", x[0], x[1], x[2], x[3])
    
    # All values same -> all probabilities should be 1/n
    for i in range(4):
        assert_almost_equal(x[i], 0.25, atol=0.001)
    
    print("✓ Softmax uniform test passed")

fn test_softmax_large_values() raises:
    """Test softmax with large values (numerical stability)."""
    print("\nTesting softmax with large values:")
    
    var x = Matrix(3)
    
    # Fill with large values [100, 101, 102]
    x[0] = 100.0
    x[1] = 101.0
    x[2] = 102.0
    
    print("  Input x:", x[0], x[1], x[2])
    
    softmax(x.data, 3)
    
    print("  After softmax:", x[0], x[1], x[2])
    
    # Should still produce valid probabilities
    assert_true(x[0] > 0.0 and x[0] < 1.0, "x[0] should be a valid probability")
    assert_true(x[1] > 0.0 and x[1] < 1.0, "x[1] should be a valid probability")
    assert_true(x[2] > 0.0 and x[2] < 1.0, "x[2] should be a valid probability")
    
    # Sum should be 1.0
    var total = x[0] + x[1] + x[2]
    assert_almost_equal(total, 1.0, atol=0.001)
    
    # Largest input should have largest probability
    assert_true(x[2] > x[1] and x[1] > x[0], "Probabilities should be ordered")
    
    print("  Sum of probabilities:", total)
    print("✓ Softmax large values test passed")

fn test_softmax_negative_values() raises:
    """Test softmax with negative values."""
    print("\nTesting softmax with negative values:")
    
    var x = Matrix(4)
    
    # Fill with [-2, -1, 0, 1]
    x[0] = -2.0
    x[1] = -1.0
    x[2] = 0.0
    x[3] = 1.0
    
    print("  Input x:", x[0], x[1], x[2], x[3])
    
    softmax(x.data, 4)
    
    print("  After softmax:", x[0], x[1], x[2], x[3])
    
    # All should be positive probabilities
    for i in range(4):
        assert_true(x[i] > 0.0, "All probabilities should be positive")
    
    # Sum should be 1.0
    var total = x[0] + x[1] + x[2] + x[3]
    assert_almost_equal(total, 1.0, atol=0.001)
    
    # Should be in increasing order
    assert_true(x[0] < x[1] and x[1] < x[2] and x[2] < x[3], "Should be ordered")
    
    print("  Sum of probabilities:", total)
    print("✓ Softmax negative values test passed")

fn test_argmax_basic() raises:
    """Test basic argmax function."""
    print("\nTesting basic argmax:")
    
    var v = Matrix(5)
    
    # Fill with [1, 5, 3, 2, 4]
    v[0] = 1.0
    v[1] = 5.0
    v[2] = 3.0
    v[3] = 2.0
    v[4] = 4.0
    
    print("  Input:", v[0], v[1], v[2], v[3], v[4])
    
    var idx = argmax(v.data, 5)
    
    print("  Argmax index:", idx)
    print("  Value at max index:", v[idx])
    
    assert_equal(idx, 1)
    assert_almost_equal(v[idx], 5.0, atol=0.001)
    
    print("✓ Basic argmax test passed")

fn test_argmax_first_element() raises:
    """Test argmax when first element is maximum."""
    print("\nTesting argmax with first element as max:")
    
    var v = Matrix(4)
    
    v[0] = 10.0
    v[1] = 5.0
    v[2] = 3.0
    v[3] = 1.0
    
    print("  Input:", v[0], v[1], v[2], v[3])
    
    var idx = argmax(v.data, 4)
    
    print("  Argmax index:", idx)
    
    assert_equal(idx, 0)
    
    print("✓ Argmax first element test passed")

fn test_argmax_last_element() raises:
    """Test argmax when last element is maximum."""
    print("\nTesting argmax with last element as max:")
    
    var v = Matrix(4)
    
    v[0] = 1.0
    v[1] = 2.0
    v[2] = 3.0
    v[3] = 10.0
    
    print("  Input:", v[0], v[1], v[2], v[3])
    
    var idx = argmax(v.data, 4)
    
    print("  Argmax index:", idx)
    
    assert_equal(idx, 3)
    
    print("✓ Argmax last element test passed")

fn test_argmax_negative_values() raises:
    """Test argmax with negative values."""
    print("\nTesting argmax with negative values:")
    
    var v = Matrix(5)
    
    # Fill with [-5, -2, -8, -1, -10]
    v[0] = -5.0
    v[1] = -2.0
    v[2] = -8.0
    v[3] = -1.0
    v[4] = -10.0
    
    print("  Input:", v[0], v[1], v[2], v[3], v[4])
    
    var idx = argmax(v.data, 5)
    
    print("  Argmax index:", idx)
    print("  Value at max index:", v[idx])
    
    # -1 is the largest (least negative)
    assert_equal(idx, 3)
    assert_almost_equal(v[idx], -1.0, atol=0.001)
    
    print("✓ Argmax negative values test passed")

fn test_argmax_equal_values() raises:
    """Test argmax with equal values (should return first occurrence)."""
    print("\nTesting argmax with equal max values:")
    
    var v = Matrix(5)
    
    # Fill with [3, 5, 5, 2, 4]
    v[0] = 3.0
    v[1] = 5.0
    v[2] = 5.0
    v[3] = 2.0
    v[4] = 4.0
    
    print("  Input:", v[0], v[1], v[2], v[3], v[4])
    
    var idx = argmax(v.data, 5)
    
    print("  Argmax index:", idx)
    
    # Should return first occurrence (index 1)
    assert_equal(idx, 1)
    
    print("✓ Argmax equal values test passed")

fn test_sample_basic() raises:
    """Test basic sampling from probability distribution."""
    print("\nTesting basic sample:")
    
    var probs = Matrix(3)
    
    # Create a simple probability distribution [0.2, 0.5, 0.3]
    probs[0] = 0.2
    probs[1] = 0.5
    probs[2] = 0.3
    
    print("  Probabilities:", probs[0], probs[1], probs[2])
    
    # Sample multiple times and check distribution
    var counts = List[Int](3)
    for _ in range(3):
        counts.append(0)
    
    var num_samples = 100
    for _ in range(num_samples):
        var idx = sample(probs.data, 3)
        assert_true(idx >= 0 and idx < 3, "Sample index should be in valid range")
        counts[idx] += 1
    
    print("  Samples (out of", num_samples, "):")
    print("    Index 0:", counts[0])
    print("    Index 1:", counts[1])
    print("    Index 2:", counts[2])
    
    # All indices should be sampled at least once (with high probability)
    # Note: This is probabilistic, but with 100 samples it should almost always pass
    assert_true(counts[0] > 0, "Index 0 should be sampled")
    assert_true(counts[1] > 0, "Index 1 should be sampled")
    assert_true(counts[2] > 0, "Index 2 should be sampled")
    
    # Index 1 (prob=0.5) should generally have more samples than others
    print("✓ Basic sample test passed")

fn test_sample_deterministic() raises:
    """Test sampling with deterministic probabilities."""
    print("\nTesting sample with deterministic distribution:")
    
    var probs = Matrix(4)
    
    # Create deterministic distribution [0, 0, 1, 0]
    probs[0] = 0.0
    probs[1] = 0.0
    probs[2] = 1.0
    probs[3] = 0.0
    
    print("  Probabilities:", probs[0], probs[1], probs[2], probs[3])
    
    # Should always return index 2
    for _ in range(10):
        var idx = sample(probs.data, 4)
        assert_equal(idx, 2)
    
    print("  All samples returned index 2 (as expected)")
    print("✓ Sample deterministic test passed")

fn test_sample_uniform() raises:
    """Test sampling from uniform distribution."""
    print("\nTesting sample with uniform distribution:")
    
    var probs = Matrix(4)
    
    # Create uniform distribution [0.25, 0.25, 0.25, 0.25]
    for i in range(4):
        probs[i] = 0.25
    
    print("  Probabilities: all 0.25")
    
    # Sample multiple times
    var counts = List[Int](4)
    for _ in range(4):
        counts.append(0)
    
    var num_samples = 100
    for _ in range(num_samples):
        var idx = sample(probs.data, 4)
        assert_true(idx >= 0 and idx < 4, "Sample index should be in valid range")
        counts[idx] += 1
    
    print("  Samples (out of", num_samples, "):")
    for i in range(4):
        print("    Index", i, ":", counts[i])
    
    # All indices should be sampled at least once
    for i in range(4):
        assert_true(counts[i] > 0, "Each index should be sampled at least once")
    
    print("✓ Sample uniform test passed")

fn test_sample_edge_cases() raises:
    """Test sampling edge cases."""
    print("\nTesting sample edge cases:")
    
    # Test with single element
    var probs1 = Matrix(1)
    probs1[0] = 1.0
    
    var idx1 = sample(probs1.data, 1)
    assert_equal(idx1, 0)
    print("  Single element: returns index 0 ✓")
    
    # Test with first element = 1.0
    var probs2 = Matrix(3)
    probs2[0] = 1.0
    probs2[1] = 0.0
    probs2[2] = 0.0
    
    var idx2 = sample(probs2.data, 3)
    assert_equal(idx2, 0)
    print("  First element = 1.0: returns index 0 ✓")
    
    # Test with last element = 1.0
    var probs3 = Matrix(3)
    probs3[0] = 0.0
    probs3[1] = 0.0
    probs3[2] = 1.0
    
    var idx3 = sample(probs3.data, 3)
    assert_equal(idx3, 2)
    print("  Last element = 1.0: returns index 2 ✓")
    
    print("✓ Sample edge cases test passed")

fn test_rmsnorm_softmax_integration() raises:
    """Test integration of rmsnorm and softmax."""
    print("\nTesting rmsnorm + softmax integration:")
    
    var x = Matrix(4)
    var o = Matrix(4)
    var weight = Matrix(4)
    
    # Fill x with some values
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    x[3] = 4.0
    
    # Weight all ones
    for i in range(4):
        weight[i] = 1.0
    
    print("  Input:", x[0], x[1], x[2], x[3])
    
    # Apply rmsnorm
    rmsnorm(o.data, x.data, weight.data, 4)
    print("  After rmsnorm:", o[0], o[1], o[2], o[3])
    
    # Apply softmax
    softmax(o.data, 4)
    print("  After softmax:", o[0], o[1], o[2], o[3])
    
    # Should be valid probability distribution
    var total = o[0] + o[1] + o[2] + o[3]
    assert_almost_equal(total, 1.0, atol=0.001)
    
    # All values should be positive
    for i in range(4):
        assert_true(o[i] > 0.0, "All probabilities should be positive")
    
    print("  Sum of probabilities:", total)
    print("✓ Integration test passed")

fn main() raises:
    print("=" * 60)
    print("Testing calculation functions")
    print("=" * 60)
    
    # RMS normalization tests
    test_rmsnorm_basic()
    test_rmsnorm_with_weights()
    test_rmsnorm_zeros()
    
    # Softmax tests
    test_softmax_basic()
    test_softmax_uniform()
    test_softmax_large_values()
    test_softmax_negative_values()
    
    # Argmax tests
    test_argmax_basic()
    test_argmax_first_element()
    test_argmax_last_element()
    test_argmax_negative_values()
    test_argmax_equal_values()
    
    # Sample tests
    test_sample_basic()
    test_sample_deterministic()
    test_sample_uniform()
    test_sample_edge_cases()
    
    # Integration test
    test_rmsnorm_softmax_integration()
    
    print("\n" + "=" * 60)
    print("All calculation tests passed! ✓")
    print("=" * 60)

