#include <stdlib.h>
#include "cuda/memory.h"
#include "cuda/types.h"

struct gsva_device_t* gsva_device_create(
    int G, int S, int* gsetofft, int* gsetidxs
) {
    size_t size_offt = (size_t)(S+1) * sizeof(int);
    size_t size_idxs = (size_t)gsetofft[S] * sizeof(int);
    size_t size_block_int = (size_t)GSVA_BLOCK_C * G * sizeof(int);
    size_t size_block_double = (size_t)GSVA_BLOCK_C * G * sizeof(double);
    size_t size_es = S * GSVA_BLOCK_C * sizeof(double);

    struct gsva_device_t *ptr = (struct gsva_device_t*)malloc(sizeof(struct gsva_device_t));
    if (!ptr) return NULL;

    ptr->G = G;
    ptr->S = S;

    GSVA_CUDA_CALL(cudaMalloc(&ptr->gsetofft, size_offt));
    GSVA_CUDA_CALL(cudaMalloc(&ptr->gsetidxs, size_idxs));
    GSVA_CUDA_CALL(cudaMalloc(&ptr->decordstat_block, size_block_int));
    GSVA_CUDA_CALL(cudaMalloc(&ptr->symrnkstat_block, size_block_double));
    GSVA_CUDA_CALL(cudaMalloc(&ptr->es, size_es));

    GSVA_CUDA_CALL(cudaMemcpy(ptr->gsetofft, gsetofft, size_offt, cudaMemcpyHostToDevice));
    GSVA_CUDA_CALL(cudaMemcpy(ptr->gsetidxs, gsetidxs, size_idxs, cudaMemcpyHostToDevice));

    return ptr;
}

void gsva_device_destroy(struct gsva_device_t* device) {
    cudaFree(device->gsetofft);
    cudaFree(device->gsetidxs);
    cudaFree(device->decordstat_block);
    cudaFree(device->symrnkstat_block);
    cudaFree(device->es);
    free(device);
}
