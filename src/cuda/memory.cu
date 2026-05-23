#include <stddef.h>
#include <stdlib.h>
#include <R.h>
#include <Rdefines.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda/types.h"
#include "cuda/memory.h"

gsva_device_t* gsva_device_create(SEXP genesetsidxR, int G) {
    int S = length(genesetsidxR);

    int total_gset_entries = 0;
    for (int s = 0; s < S; s++) {
        SEXP gsetidxR = VECTOR_ELT(genesetsidxR, s);
        total_gset_entries += length(gsetidxR);
    }

    int *gsetofft = (int*)R_alloc(S+1, sizeof(int));
    int *gsetidxs = (int*)R_alloc(total_gset_entries, sizeof(int));

    total_gset_entries = 0;
    for (int s = 0; s < S; s++) {
        SEXP gsetidxR = VECTOR_ELT(genesetsidxR, s);
        int* gsetidx  = INTEGER(gsetidxR);
        int k = length(gsetidxR);
        gsetofft[s] = total_gset_entries;
        for (int i = 0; i < k; i++) {
            gsetidxs[total_gset_entries++] = gsetidx[i];
        }
    }
    gsetofft[S] = total_gset_entries;

    size_t size_offt = (size_t)(S+1) * sizeof(int);
    size_t size_idxs = (size_t)gsetofft[S] * sizeof(int);
    size_t size_block_int    = (size_t)GSVA_BLOCK_C * G * sizeof(int);
    size_t size_block_double = (size_t)GSVA_BLOCK_C * G * sizeof(double);
    size_t size_es_block     = (size_t)S * GSVA_BLOCK_C * sizeof(double);

    gsva_device_t *ptr = (gsva_device_t*)R_alloc(1, sizeof(gsva_device_t));

    ptr->G = G;
    ptr->S = S;

    GSVA_CUDA_CALL(cudaMalloc(&ptr->gsetofft, size_offt));
    GSVA_CUDA_CALL(cudaMalloc(&ptr->gsetidxs, size_idxs));

    for (int i = 0; i < GSVA_CUDA_STREAMS; i++) {
        GSVA_CUDA_CALL(cudaStreamCreate(&ptr->stream[i]));
        GSVA_CUDA_CALL(cudaMalloc(&ptr->decordstat[i], size_block_int));
        GSVA_CUDA_CALL(cudaMalloc(&ptr->symrnkstat[i], size_block_double));
        GSVA_CUDA_CALL(cudaMalloc(&ptr->es[i], size_es_block));
    }

    GSVA_CUDA_CALL(cudaMemcpy(ptr->gsetofft, gsetofft, size_offt, cudaMemcpyHostToDevice));
    GSVA_CUDA_CALL(cudaMemcpy(ptr->gsetidxs, gsetidxs, size_idxs, cudaMemcpyHostToDevice));

    return ptr;
}

void gsva_device_destroy(gsva_device_t* device) {
    for (int i = 0; i < GSVA_CUDA_STREAMS; i++) {
        cudaFree(device->decordstat[i]);
        cudaFree(device->symrnkstat[i]);
        cudaFree(device->es[i]);
        cudaStreamDestroy(device->stream[i]);
    }
    cudaFree(device->gsetofft);
    cudaFree(device->gsetidxs);
}

gsva_host_t* gsva_host_create(int G, int S) {
    gsva_host_t *h = (gsva_host_t*)R_alloc(1, sizeof(gsva_host_t));

    size_t size_dec  = (size_t)GSVA_BLOCK_C * G * sizeof(int);
    size_t size_sym  = (size_t)GSVA_BLOCK_C * G * sizeof(double);
    size_t size_es   = (size_t)S * GSVA_BLOCK_C * sizeof(double);

    for (int i = 0; i < GSVA_CUDA_STREAMS; i++) {
        GSVA_CUDA_CALL(cudaMallocHost(&h->decordstat[i], size_dec, 0));
        GSVA_CUDA_CALL(cudaMallocHost(&h->symrnkstat[i], size_sym, 0));
        GSVA_CUDA_CALL(cudaMallocHost(&h->es[i], size_es, 0));
    }

    return h;
}

void gsva_host_destroy(gsva_host_t* h) {
    for (int i = 0; i < GSVA_CUDA_STREAMS; i++) {
        cudaFreeHost(h->decordstat[i]);
        cudaFreeHost(h->symrnkstat[i]);
        cudaFreeHost(h->es[i]);
    }
}
