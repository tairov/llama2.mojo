#!/bin/bash

# Script to run all tests in the tests directory

set -e  # Exit on error

echo "============================================================"
echo "Running all tests"
echo "============================================================"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# List of test files
TESTS=(
    "tests/test-calculations.mojo"
    "tests/test-matmul.mojo"
    "tests/test-tokenizer.mojo"
    "tests/test-transformer.mojo"
    "tests/test-llama2.mojo"
)

PASSED=0
FAILED=0
FAILED_TESTS=()

# Run each test
for test in "${TESTS[@]}"; do
    echo "Running $test..."
    echo "------------------------------------------------------------"
    
    if mojo -I . "$test"; then
        echo "✓ $test PASSED"
        ((PASSED++))
    else
        echo "✗ $test FAILED"
        ((FAILED++))
        FAILED_TESTS+=("$test")
    fi
    
    echo ""
done

# Print summary
echo "============================================================"
echo "Test Summary"
echo "============================================================"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "All tests passed! ✓"
    exit 0
else
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    exit 1
fi

