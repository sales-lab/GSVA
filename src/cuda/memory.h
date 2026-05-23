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
    int         *decordstat[GSVA_CUDA_STREAMS];
    double      *symrnkstat[GSVA_CUDA_STREAMS];
    double      *es[GSVA_CUDA_STREAMS];
} gsva_device_t;

gsva_device_t* gsva_device_create(SEXP genesetsidxR, int G);
void gsva_device_destroy(gsva_device_t* device);

typedef struct {
    int    *decordstat[GSVA_CUDA_STREAMS];
    double *symrnkstat[GSVA_CUDA_STREAMS];
    double *es[GSVA_CUDA_STREAMS];
} gsva_host_t;

gsva_host_t* gsva_host_create(int G, int S);
void gsva_host_destroy(gsva_host_t* h);

#ifdef __cplusplus
}
#endif

#endif
