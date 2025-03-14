#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void sgemm_naive_1a(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[y * K + i] * B[i * N + x];
    }
    // C = α*(A@B)+β*C
    C[y * N + x] = alpha * tmp + beta * C[y * N + x];
  }
}