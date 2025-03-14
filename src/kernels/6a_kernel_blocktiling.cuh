#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

namespace sbt {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                            int innerRowB, int innerColB, const int SMEM_PADD) {
  for (int offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }

  for (int offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * (BM + SMEM_PADD) + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * (BM + SMEM_PADD) + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * (BM + SMEM_PADD) + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * (BM + SMEM_PADD) + innerRowA + offset] = tmp.w;
  }
}

template <const int BM, const int BN, const int BK, const int SBM, const int SBN,
          const int NUM_SBM, const int NUM_SBN, const int TM, const int TN, const int SMEM_PADD>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const int threadRowInSubTile, const int threadColInSubTile) {
  for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
    for (int iSBN = 0; iSBN < NUM_SBN; iSBN++){
      for (int i = 0; i < TN; ++i) {
        regN[iSBN * TN + i] = Bs[(dotIdx * BN) + iSBN*SBN + threadColInSubTile*TN + i];
      }
    }

    for (int iSBM = 0; iSBM < NUM_SBM; iSBM++){
      for (int i = 0; i < TM; ++i) {
        regM[iSBM * TM + i] = As[(dotIdx * (BM + SMEM_PADD)) + iSBM*SBM + threadRowInSubTile*TM + i];
      }
    }

    for (int iSBM = 0; iSBM < NUM_SBM; iSBM++){
      for (int iSBN = 0; iSBN < NUM_SBN; iSBN++){
        for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(iSBM * TM + resIdxM)*(NUM_SBN*TN) + (iSBN * TN + resIdxN)] += regM[iSBM * TM + resIdxM] * regN[iSBN * TN + resIdxN];
          }
        }
      }
    }
  }
}
}

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam SBM M dim of continuous subtile computed all threads
 * @tparam SBN N dim of continuous subtile computed all threads
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template <const int BM, const int BN, const int BK, const int SBM, const int SBN,
          const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmSubBlockTiling(int M, int N, int K, float alpha, const float *__restrict__ A, const float *__restrict__ B,
                    float beta, float *__restrict__ C) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  constexpr int NUM_SBN = BN/SBN;
  constexpr int NUM_SBM = BM/SBM;

  const int threadColInSubTile = threadIdx.x % (SBN/TN);
  const int threadRowInSubTile = threadIdx.x / (SBN/TN);

  // allocate space for the current blocktile in SMEM
  constexpr int SMEM_PADD = 0;
  __shared__ float As[(BM+SMEM_PADD) * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += (cRow * BM) * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const int innerRowA = threadIdx.x / (BK / 4);
  const int innerColA = threadIdx.x % (BK / 4);
  constexpr int rowStrideA = (NUM_THREADS * 4) / BK;
  const int innerRowB = threadIdx.x / (BN / 4);
  const int innerColB = threadIdx.x % (BN / 4);
  constexpr int rowStrideB = NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[NUM_SBM * TM * NUM_SBN * TN] = {0.0};
  // we cache into registers on the warptile level
  float regM[NUM_SBM * TM] = {0.0};
  float regN[NUM_SBN * TN] = {0.0};

  // outer-most loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    sbt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB, SMEM_PADD);
    __syncthreads();
    sbt::processFromSmem<BM, BN, BK, SBM, SBN, NUM_SBM, NUM_SBN, TM, TN, SMEM_PADD>(regM, regN, threadResults, As, Bs, threadRowInSubTile, threadColInSubTile);
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }

  for (int iSBM = 0; iSBM < NUM_SBM; iSBM++){
    for (int iSBN = 0; iSBN < NUM_SBN; iSBN++){
      for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          const int iC = (iSBM * SBM + threadRowInSubTile * TM + resIdxM) * N +  iSBN * SBN + threadColInSubTile * TN + resIdxN;
          float4 tmp = reinterpret_cast<float4*>(&C[iC])[0];
          // perform GEMM update in reg
          const int i = (iSBM * TM + resIdxM)*(NUM_SBN*TN) + (iSBN * TN + resIdxN);
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(&C[iC])[0] = tmp;
        }
      }
    }
  }
}
