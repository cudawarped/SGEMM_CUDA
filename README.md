# Fast CUDA SGEMM from Scratch

Step-by-step optimization of matrix multiplication, implemented in CUDA.
For an explanation of each kernel, see [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM).

## Overview

Running the kernels on a NVIDIA A6000 (Ampere):

![](benchmark_results.png)

GFLOPs at matrix size 4096x4096:
<!-- benchmark_results -->
| Kernel                              |   GFLOPs/s | Performance relative to cuBLAS   |
|:------------------------------------|-----------:|:---------------------------------|
| 1: Naive                            |      306.8 | 1.3%                             |
| 2: GMEM Coalescing                  |     2075.3 | 8.8%                             |
| 12: Naive GMEM Coalescing 32x32     |     2164   | 9.1%                             |
| 13: Naive GMEM Coalescing 32x16     |     2269.8 | 9.6%                             |
| 3: SMEM Caching                     |     2977.3 | 12.6%                            |
| 4: 1D Blocktiling                   |     8614.2 | 36.4%                            |
| 5: 2D Blocktiling                   |    15739.4 | 66.5%                            |
| 7: Avoid Bank Conflicts (Linearize) |    15999.3 | 67.6%                            |
| 8: Avoid Bank Conflicts (Offset)    |    16348.5 | 69.1%                            |
| 11: Double Buffering                |    17970.5 | 76.0%                            |
| 6: Vectorized Mem Access            |    18169.4 | 76.8%                            |
| 9: Autotuning                       |    19706.4 | 83.3%                            |
| 10: Warptiling                      |    21347.5 | 90.2%                            |
| 14: 2D Sublocktiling                 |    21558   | 91.1%                            |
| 0: cuBLAS                           |    23658.5 | 100.0%                           |
<!-- benchmark_results -->

## Setup

1. Install dependencies: CUDA toolkit 12, Python (+ Seaborn), CMake, Ninja. See [environment.yml](environment.yml).
1. Configure NVCC compilation parameters. Look up your GPUs compute
   capability [here](https://developer.nvidia.com/cuda-gpus). Then configure the `CMakeLists.txt` and change:
    ```cmake
    set(CUDA_COMPUTE_CAPABILITY 80)
    ```
1. Build: `mkdir build && cd build && cmake .. && cmake --build .`
1. Run one of the kernels: `DEVICE=<device_id> ./sgemm <kernel number>`
1. Profiling via [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (ncu): `make profile KERNEL=<kernel number>`

Credit goes to [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) for the benchmarking setup.
