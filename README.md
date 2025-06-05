# SIMD-Optimization-of-Matrix-Transposition-and-Multiplication

This project demonstrates the performance benefits of SIMD (Single Instruction, Multiple Data) by optimizing matrix transposition and element-wise multiplication using AVX intrinsics on x86 architectures.

The implementation includes:
- Scalar version (2D and 1D array)
- SIMD-optimized version using AVX

---

## Objective

The goal is to speed up the following matrix operation:

1. Transpose matrix `A` → `A_T`
2. Perform element-wise multiplication with matrix `B`:  
   `C[i][j] = A_T[i][j] * B[i][j]`

---



## Implementation Details

### Scalar Implementations
1. **2D Array**:
   - Uses nested loops on `float A[N][N]`, `B[N][N]`, `C[N][N]`.
2. **1D Array**:
   - Uses `float* A`, simulating 2D arrays via linear indexing.

Both versions ensure correctness and can run on any matrix size `N`.

### SIMD (AVX) Optimization
- Uses `__m256` data types from `<immintrin.h>`
- Vectorized the transpose and multiplication operations
- Works best when `N` is a multiple of 8 (for 256-bit registers)

---

## How to Build and Run

### Compile with AVX:
---
bash
g++ -O2 -mavx -o simd_exec Source.cpp

./simd_exec 512

---
###  Results

| Matrix Size (N) | Scalar 2D Time (s) | Scalar 1D Time (s) | SIMD Time (s) | Speedup (2D vs SIMD) | Speedup (1D vs SIMD) |
|-----------------|--------------------|---------------------|----------------|------------------------|------------------------|
| 256             | 0.000479           | 0.000057            | 0.000027       | 17.74×                 | 2.11×                  |
| 512             | 0.001493           | 0.000352            | 0.000115       | 12.98×                 | 3.06×                  |
| 1024            | 0.006936           | 0.004746            | 0.000539       | 12.86×                 | 8.80×                  |

> *Performance measured in seconds. Results may vary based on hardware and compiler optimizations.*

---

## Challenges Faced

- **Memory Alignment**: Correct alignment using `aligned_alloc` was necessary for AVX loads/stores to avoid crashes and incorrect results.
- **Vectorizing Transpose**: Efficiently transposing large matrices with AVX while maintaining cache locality was a key challenge.
- **Verification**: Ensuring identical results between scalar and SIMD versions, especially across various matrix sizes, required careful debugging.

