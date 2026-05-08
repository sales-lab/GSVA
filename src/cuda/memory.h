#ifndef CUDA_MEMORY_H
#define CUDA_MEMORY_H

#include <R.h>
#include <Rdefines.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int G, S;
    int *gsetofft, *gsetidxs;

    cudaStream_t stream[GSVA_CUDA_STREAMS];

    int *r_scratch[GSVA_CUDA_STREAMS];   // [G*B_C] scratch for rank shift

    // Buffers for sparse ranks
    int *sparse_offt[GSVA_CUDA_STREAMS]; // [G*B_C] gene offsets
    int *sparse_vals[GSVA_CUDA_STREAMS]; // [G*B_C] rank values
    int *cell_offt[GSVA_CUDA_STREAMS];   // [B_C+1] prefix sum of nnz
    int *nnzpercell[GSVA_CUDA_STREAMS];  // [B_C] nnz per cell

    // Buffer for dense ranks
    int *dense_ranks[GSVA_CUDA_STREAMS]; // [G*B_C] dense ranks

    // Buffers for random walk
    int          *decordstat[GSVA_CUDA_STREAMS];
    gsva_float_t *symrnkstat[GSVA_CUDA_STREAMS];
    gsva_float_t *es[GSVA_CUDA_STREAMS];
} gsva_device_t;

gsva_device_t* gsva_device_create(SEXP genesetsidxR, int G, Rboolean sparse);
void gsva_device_destroy(gsva_device_t* device);

typedef struct {
    int *sparse_offt[GSVA_CUDA_STREAMS]; // [G*B_C] gene offsets
    int *sparse_vals[GSVA_CUDA_STREAMS]; // [G*B_C] rank values
    int *cell_offt[GSVA_CUDA_STREAMS];   // [B_C+1] prefix sum
    int *nnzpercell[GSVA_CUDA_STREAMS];  // [B_C] nnz per cell

    gsva_float_t *es[GSVA_CUDA_STREAMS];
} gsva_host_t;

gsva_host_t* gsva_host_create(int G, int S, Rboolean sparse);
void gsva_host_destroy(gsva_host_t* h);

#ifdef __cplusplus
}
#endif

#endif
