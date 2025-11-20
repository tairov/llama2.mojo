from testing import assert_true, assert_almost_equal, assert_equal
from llama2 import Matrix, matmul, batch_matmul, BufferPtrFloat32
from utils import StaticTuple

fn test_matmul_basic() raises:
    """Test basic matrix-vector multiplication."""
    print("\nTesting basic matmul:")
    
    # Test: B (3,4) @ A (4,) -> C (3,)
    var A = Matrix(4)  # Vector of size 4
    var B = Matrix(3, 4)  # Matrix 3x4
    var C = Matrix(3)  # Result vector of size 3
    
    # Fill A with [1, 2, 3, 4]
    A[0] = 1.0
    A[1] = 2.0
    A[2] = 3.0
    A[3] = 4.0
    
    # Fill B with identity-like pattern for easy verification
    # Row 0: [1, 0, 0, 0] -> dot with A = 1
    # Row 1: [0, 1, 0, 0] -> dot with A = 2
    # Row 2: [0, 0, 1, 0] -> dot with A = 3
    B[0, 0] = 1.0
    B[0, 1] = 0.0
    B[0, 2] = 0.0
    B[0, 3] = 0.0
    
    B[1, 0] = 0.0
    B[1, 1] = 1.0
    B[1, 2] = 0.0
    B[1, 3] = 0.0
    
    B[2, 0] = 0.0
    B[2, 1] = 0.0
    B[2, 2] = 1.0
    B[2, 3] = 0.0
    
    print("  A (vector):", A[0], A[1], A[2], A[3])
    print("  B (matrix 3x4): identity-like pattern")
    
    matmul(C.data, A.data, B.data, 3, 4)
    
    print("  C (result):", C[0], C[1], C[2])
    
    # Verify results
    assert_almost_equal(C[0], 1.0, atol=0.001)
    assert_almost_equal(C[1], 2.0, atol=0.001)
    assert_almost_equal(C[2], 3.0, atol=0.001)
    
    print("✓ Basic matmul test passed")

fn test_matmul_all_ones() raises:
    """Test matmul with all ones."""
    print("\nTesting matmul with all ones:")
    
    # Test: B (2,3) @ A (3,) -> C (2,)
    var A = Matrix(3)
    var B = Matrix(2, 3)
    var C = Matrix(2)
    
    # Fill with all ones
    for i in range(3):
        A[i] = 1.0
    
    for row in range(2):
        for col in range(3):
            B[row, col] = 1.0
    
    matmul(C.data, A.data, B.data, 2, 3)
    
    print("  A: all ones (size 3)")
    print("  B: all ones (2x3)")
    print("  C (result):", C[0], C[1])
    
    # Each row of B dot A should be 3.0 (1+1+1)
    assert_almost_equal(C[0], 3.0, atol=0.001)
    assert_almost_equal(C[1], 3.0, atol=0.001)
    
    print("✓ Matmul with all ones test passed")

fn test_matmul_larger() raises:
    """Test matmul with larger matrices."""
    print("\nTesting matmul with larger matrices:")
    
    # Test: B (5,10) @ A (10,) -> C (5,)
    var A = Matrix(10)
    var B = Matrix(5, 10)
    var C = Matrix(5)
    
    # Fill A with sequential values
    for i in range(10):
        A[i] = Float32(i + 1)  # 1, 2, 3, ..., 10
    
    # Fill B with pattern: row i has all values equal to i
    for row in range(5):
        for col in range(10):
            B[row, col] = Float32(row)
    
    matmul(C.data, A.data, B.data, 5, 10)
    
    print("  A: [1, 2, 3, ..., 10]")
    print("  B: row i has all values = i")
    print("  C (result):", end=" ")
    for i in range(5):
        print(C[i], end=" ")
    print()
    
    # Verify: row i dot A = i * (1+2+...+10) = i * 55
    for i in range(5):
        var expected = Float32(i) * 55.0
        assert_almost_equal(C[i], expected, atol=0.1)
    
    print("✓ Larger matmul test passed")

fn test_matmul_zero() raises:
    """Test matmul with zero vector."""
    print("\nTesting matmul with zero vector:")
    
    var A = Matrix(4)
    var B = Matrix(3, 4)
    var C = Matrix(3)
    
    # A is all zeros
    A.zero()
    
    # B has some values
    for row in range(3):
        for col in range(4):
            B[row, col] = Float32(row + col)
    
    matmul(C.data, A.data, B.data, 3, 4)
    
    print("  A: all zeros")
    print("  B: non-zero values")
    print("  C (result):", C[0], C[1], C[2])
    
    # Result should be all zeros
    assert_almost_equal(C[0], 0.0, atol=0.001)
    assert_almost_equal(C[1], 0.0, atol=0.001)
    assert_almost_equal(C[2], 0.0, atol=0.001)
    
    print("✓ Matmul with zero vector test passed")

fn test_batch_matmul_single() raises:
    """Test batch_matmul with n=1 (equivalent to regular matmul)."""
    print("\nTesting batch_matmul with single matrix (n=1):")
    
    var A = Matrix(4)
    var B = Matrix(3, 4)
    var C = Matrix(3)
    
    # Fill A and B
    for i in range(4):
        A[i] = Float32(i + 1)
    
    for row in range(3):
        for col in range(4):
            B[row, col] = Float32(row == col)  # Identity-like
    
    var C_tuple = StaticTuple[BufferPtrFloat32, 1](C.data)
    var B_tuple = StaticTuple[BufferPtrFloat32, 1](B.data)
    
    batch_matmul[1](C_tuple, A.data, B_tuple, 3, 4)
    
    print("  A: [1, 2, 3, 4]")
    print("  B: identity-like (3x4)")
    print("  C (result):", C[0], C[1], C[2])
    
    # Verify
    assert_almost_equal(C[0], 1.0, atol=0.001)
    assert_almost_equal(C[1], 2.0, atol=0.001)
    assert_almost_equal(C[2], 3.0, atol=0.001)
    
    print("✓ Batch matmul with n=1 test passed")

fn test_batch_matmul_two() raises:
    """Test batch_matmul with n=2."""
    print("\nTesting batch_matmul with two matrices (n=2):")
    
    var A = Matrix(3)
    var B1 = Matrix(2, 3)
    var B2 = Matrix(2, 3)
    var C1 = Matrix(2)
    var C2 = Matrix(2)
    
    # Fill A with [1, 2, 3]
    A[0] = 1.0
    A[1] = 2.0
    A[2] = 3.0
    
    # B1: all ones
    for row in range(2):
        for col in range(3):
            B1[row, col] = 1.0
    
    # B2: all twos
    for row in range(2):
        for col in range(3):
            B2[row, col] = 2.0
    
    var C_tuple = StaticTuple[BufferPtrFloat32, 2](C1.data, C2.data)
    var B_tuple = StaticTuple[BufferPtrFloat32, 2](B1.data, B2.data)
    
    batch_matmul[2](C_tuple, A.data, B_tuple, 2, 3)
    
    print("  A: [1, 2, 3]")
    print("  B1: all ones (2x3)")
    print("  B2: all twos (2x3)")
    print("  C1 (result):", C1[0], C1[1])
    print("  C2 (result):", C2[0], C2[1])
    
    # C1 should be [6, 6] (1+2+3 for each row)
    assert_almost_equal(C1[0], 6.0, atol=0.001)
    assert_almost_equal(C1[1], 6.0, atol=0.001)
    
    # C2 should be [12, 12] (2*(1+2+3) for each row)
    assert_almost_equal(C2[0], 12.0, atol=0.001)
    assert_almost_equal(C2[1], 12.0, atol=0.001)
    
    print("✓ Batch matmul with n=2 test passed")

fn test_batch_matmul_three() raises:
    """Test batch_matmul with n=3."""
    print("\nTesting batch_matmul with three matrices (n=3):")
    
    var A = Matrix(4)
    var B1 = Matrix(2, 4)
    var B2 = Matrix(2, 4)
    var B3 = Matrix(2, 4)
    var C1 = Matrix(2)
    var C2 = Matrix(2)
    var C3 = Matrix(2)
    
    # Fill A with [1, 1, 1, 1]
    for i in range(4):
        A[i] = 1.0
    
    # B1: row i has all values = i
    for row in range(2):
        for col in range(4):
            B1[row, col] = Float32(row)
    
    # B2: row i has all values = i+1
    for row in range(2):
        for col in range(4):
            B2[row, col] = Float32(row + 1)
    
    # B3: row i has all values = i+2
    for row in range(2):
        for col in range(4):
            B3[row, col] = Float32(row + 2)
    
    var C_tuple = StaticTuple[BufferPtrFloat32, 3](C1.data, C2.data, C3.data)
    var B_tuple = StaticTuple[BufferPtrFloat32, 3](B1.data, B2.data, B3.data)
    
    batch_matmul[3](C_tuple, A.data, B_tuple, 2, 4)
    
    print("  A: [1, 1, 1, 1]")
    print("  B1, B2, B3: different row patterns")
    print("  C1 (result):", C1[0], C1[1])
    print("  C2 (result):", C2[0], C2[1])
    print("  C3 (result):", C3[0], C3[1])
    
    # C1: row 0 = 0*4=0, row 1 = 1*4=4
    assert_almost_equal(C1[0], 0.0, atol=0.001)
    assert_almost_equal(C1[1], 4.0, atol=0.001)
    
    # C2: row 0 = 1*4=4, row 1 = 2*4=8
    assert_almost_equal(C2[0], 4.0, atol=0.001)
    assert_almost_equal(C2[1], 8.0, atol=0.001)
    
    # C3: row 0 = 2*4=8, row 1 = 3*4=12
    assert_almost_equal(C3[0], 8.0, atol=0.001)
    assert_almost_equal(C3[1], 12.0, atol=0.001)
    
    print("✓ Batch matmul with n=3 test passed")

fn test_matmul_dimension_validation() raises:
    """Test that matmul works with correct dimensions."""
    print("\nTesting matmul dimension handling:")
    
    # Note: matmul doesn't validate dimensions at runtime - it just computes
    # with whatever dimensions are provided. The caller is responsible for
    # ensuring dimensions match. This test verifies matmul works correctly
    # when dimensions are properly aligned.
    
    var A = Matrix(4)  # Vector of size 4
    var B = Matrix(2, 4)  # Matrix 2x4 (B.cols=4 matches A.size=4)
    var C = Matrix(2)
    
    # Fill with test values
    for i in range(4):
        A[i] = Float32(i + 1)
    
    for row in range(2):
        for col in range(4):
            B[row, col] = Float32(row == col)  # Identity-like
    
    matmul(C.data, A.data, B.data, 2, 4)
    
    print("  A: [1, 2, 3, 4]")
    print("  B: identity-like (2x4)")
    print("  C (result):", C[0], C[1])
    
    # Verify results
    assert_almost_equal(C[0], 1.0, atol=0.001)
    assert_almost_equal(C[1], 2.0, atol=0.001)
    
    print("✓ Dimension handling test passed")

fn test_batch_matmul_consistency() raises:
    """Test that batch_matmul with n=1 produces same result as matmul."""
    print("\nTesting batch_matmul consistency with matmul:")
    
    var A = Matrix(5)
    var B = Matrix(3, 5)
    var C_regular = Matrix(3)
    var C_batch = Matrix(3)
    
    # Fill with some values
    for i in range(5):
        A[i] = Float32(i * 2 + 1)
    
    for row in range(3):
        for col in range(5):
            B[row, col] = Float32(row * col + 1)
    
    # Regular matmul
    matmul(C_regular.data, A.data, B.data, 3, 5)
    
    # Batch matmul with n=1
    var C_tuple = StaticTuple[BufferPtrFloat32, 1](C_batch.data)
    var B_tuple = StaticTuple[BufferPtrFloat32, 1](B.data)
    batch_matmul[1](C_tuple, A.data, B_tuple, 3, 5)
    
    print("  Regular matmul result:", C_regular[0], C_regular[1], C_regular[2])
    print("  Batch matmul result:  ", C_batch[0], C_batch[1], C_batch[2])
    
    # Results should be identical
    for i in range(3):
        assert_almost_equal(C_regular[i], C_batch[i], atol=0.001)
    
    print("✓ Consistency test passed")

fn main() raises:
    print("=" * 60)
    print("Testing matmul and batch_matmul functions")
    print("=" * 60)
    
    # Regular matmul tests
    test_matmul_basic()
    test_matmul_all_ones()
    test_matmul_larger()
    test_matmul_zero()
    
    # Batch matmul tests
    test_batch_matmul_single()
    test_batch_matmul_two()
    test_batch_matmul_three()
    
    # Validation and consistency tests
    test_matmul_dimension_validation()
    test_batch_matmul_consistency()
    
    print("\n" + "=" * 60)
    print("All matmul tests passed! ✓")
    print("=" * 60)

