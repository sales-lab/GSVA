#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define GSVA_THREAD_NUM 128

static_assert(GSVA_THREAD_NUM % 2 == 0, "GSVA_THREAD_NUM must be divisible by 2");

__global__ void gsea_walk_kernel(
    const int* __restrict__ gsetofft,
    const int* __restrict__ gsetidxs,
    const int* __restrict__ decordstat_block,
    const double* __restrict__ symrnkstat_block,
    double* __restrict__ out_es,
    int S,
    int G,
    int C,
    double tau,
    int score_type
);

#endif
