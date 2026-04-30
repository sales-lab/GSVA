#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define GSVA_THREAD_NUM 128
#define GSVA_R2S_THREADS 512

static_assert(GSVA_THREAD_NUM % 2 == 0, "GSVA_THREAD_NUM must be divisible by 2");

/* Fetch type constants (must match RANKSTYPE_* in ranks.h) */
#define R2S_MATRIX_INT   1
#define R2S_SPARSE       2  /* DGC or SVT_INT */

__global__ void ranks2stats_gpu(
    int fetch_type,
    const int* __restrict__ dense_ranks,
    const int* __restrict__ sparse_offt,
    const int* __restrict__ sparse_vals,
    const int* __restrict__ cell_offt,
    const int* __restrict__ nnzpercell,
    int* __restrict__ d_r,
    int* __restrict__ decordstat,
    double* __restrict__ symrnkstat,
    int G,
    int block_size,
    int sparse_mode
);

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
