//Rehan Tariq
//22i-0965
//Cs-6A


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h> // AVX intrinsics (for x86)
// #include <arm_neon.h> // NEON intrinsics (for Apple Silicon)

#define MAX_N 1024 // Define an upper limit for matrix size

// Function for 2D Scalar Implementation
void scalar_2Dimplementation(float A[MAX_N][MAX_N], float B[MAX_N][MAX_N], float C[MAX_N][MAX_N], int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = A[j][i] * B[i][j];  // Transpose A and multiply with B
        }
    }
}

// Function for 1D Scalar Implementation (Flattened Array)
void scalar_1Dimplementation(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = A[j * N + i] * B[i * N + j];  // Transpose A and multiply with B
        }
    }
}

// Function for SIMD-Optimized Implementation using AVX
void simd_implementation(float A[MAX_N][MAX_N], float B[MAX_N][MAX_N], float C[MAX_N][MAX_N], int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 8) {  // Process 8 elements at a time
            __m256 rowA = _mm256_loadu_ps(&A[j][i]);  // Load transposed A
            __m256 rowB = _mm256_loadu_ps(&B[i][j]);  // Load row from B
            __m256 result = _mm256_mul_ps(rowA, rowB); // Multiply element-wise
            _mm256_storeu_ps(&C[i][j], result); // Store result
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <Matrix Size (N)>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0 || N > MAX_N || N % 8 != 0) {
        printf("Error: N must be a positive multiple of 8 and ≤ %d\n", MAX_N);
        return 1;
    }

    // Allocate matrices dynamically
    float (*A)[MAX_N] = malloc(sizeof(float[MAX_N][MAX_N]));
    float (*B)[MAX_N] = malloc(sizeof(float[MAX_N][MAX_N]));
    float (*C)[MAX_N] = malloc(sizeof(float[MAX_N][MAX_N]));

    // Initialize matrices A and B with random values
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)rand() / RAND_MAX;
            B[i][j] = (float)rand() / RAND_MAX;
        }
    }

    // Measure performance of Scalar 2D implementation
    clock_t start = clock();
    scalar_2Dimplementation(A, B, C, N);
    clock_t end = clock();
    printf("Scalar 2D time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Measure performance of Scalar 1D implementation
    float* A1D = (float*)A;
    float* B1D = (float*)B;
    float* C1D = (float*)C;
    
    start = clock();
    scalar_1Dimplementation(A1D, B1D, C1D, N);
    end = clock();
    printf("Scalar 1D time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Measure performance of SIMD implementation
    start = clock();
    simd_implementation(A, B, C, N);
    end = clock();
    printf("SIMD time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}


