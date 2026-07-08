#include "cuda/memory.h"
#include "cuda/types.h"
#include <R.h>
#include <Rdefines.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdlib.h>

gsva_device_t *gsva_device_create(SEXP genesetsidxR, int G, Rboolean sparse) {
  int S = length(genesetsidxR);

  int total_gset_entries = 0;
  for (int s = 0; s < S; s++) {
    SEXP gsetidxR = VECTOR_ELT(genesetsidxR, s);
    total_gset_entries += length(gsetidxR);
  }

  int *gsetofft = (int *)R_alloc(S + 1, sizeof(int));
  int *gsetidxs = (int *)R_alloc(total_gset_entries, sizeof(int));

  total_gset_entries = 0;
  for (int s = 0; s < S; s++) {
    SEXP gsetidxR = VECTOR_ELT(genesetsidxR, s);
    int *gsetidx = INTEGER(gsetidxR);
    int k = length(gsetidxR);
    gsetofft[s] = total_gset_entries;
    for (int i = 0; i < k; i++) {
      gsetidxs[total_gset_entries++] = gsetidx[i];
    }
  }
  gsetofft[S] = total_gset_entries;

  size_t size_offt = (size_t)(S + 1) * sizeof(int);
  size_t size_idxs = (size_t)gsetofft[S] * sizeof(int);
  size_t size_block_int = (size_t)GSVA_BLOCK_C * G * sizeof(int);
  size_t size_block_float = (size_t)GSVA_BLOCK_C * G * sizeof(gsva_float_t);
  size_t size_es_block = (size_t)S * GSVA_BLOCK_C * sizeof(gsva_float_t);
  size_t size_sparse_arr = (size_t)G * GSVA_BLOCK_C * sizeof(int);
  size_t size_cell_offt = (size_t)(GSVA_BLOCK_C + 1) * sizeof(int);
  size_t size_nnzpercell = (size_t)GSVA_BLOCK_C * sizeof(int);

  gsva_device_t *ptr = (gsva_device_t *)R_alloc(1, sizeof(gsva_device_t));

  ptr->G = G;
  ptr->S = S;
  ptr->max_k = 0;
  for (int s = 0; s < S; s++) {
    SEXP gsetidxR = VECTOR_ELT(genesetsidxR, s);
    int k = length(gsetidxR);
    if (k > ptr->max_k)
      ptr->max_k = k;
  }

  int cat_count[GSVA_NUM_CATS] = {0};
  for (int s = 0; s < S; s++) {
    SEXP gsetidxR = VECTOR_ELT(genesetsidxR, s);
    int k = length(gsetidxR);
    if (k <= 160) cat_count[0]++;
    else if (k <= 256) cat_count[1]++;
    else if (k <= 644) cat_count[2]++;
    else if (k <= 2048) cat_count[3]++;
    else cat_count[4]++;
  }

  int *cat_gset[GSVA_NUM_CATS];
  for (int cat = 0; cat < GSVA_NUM_CATS; cat++) {
    if (cat_count[cat] > 0) {
      cat_gset[cat] = (int *)R_alloc(cat_count[cat], sizeof(int));
    } else {
      cat_gset[cat] = NULL;
    }
  }
  
  int cat_pos[GSVA_NUM_CATS] = {0};
  for (int s = 0; s < S; s++) {
    SEXP gsetidxR = VECTOR_ELT(genesetsidxR, s);
    int k = length(gsetidxR);
    if (k > GSVA_MAX_GPU_GSET_SIZE) {
      error("Gene set %d has %d genes, exceeding GPU limit of %d.",
            s + 1, k, GSVA_MAX_GPU_GSET_SIZE);
    }
    int cat;
    if (k <= 160) cat = 0;
    else if (k <= 256) cat = 1;
    else if (k <= 644) cat = 2;
    else if (k <= 2048) cat = 3;
    else cat = 4;
    cat_gset[cat][cat_pos[cat]++] = s;
  }

  GSVA_CUDA_CALL(cudaMalloc(&ptr->gsetofft, size_offt));
  GSVA_CUDA_CALL(cudaMalloc(&ptr->gsetidxs, size_idxs));

  for (int i = 0; i < GSVA_CUDA_STREAMS; i++) {
    GSVA_CUDA_CALL(cudaStreamCreate(&ptr->stream[i]));
    GSVA_CUDA_CALL(cudaMalloc(&ptr->r_scratch[i], size_sparse_arr));
    if (sparse) {
      GSVA_CUDA_CALL(cudaMalloc(&ptr->sparse_offt[i], size_sparse_arr));
      GSVA_CUDA_CALL(cudaMalloc(&ptr->sparse_vals[i], size_sparse_arr));
      GSVA_CUDA_CALL(cudaMalloc(&ptr->cell_offt[i], size_cell_offt));
      GSVA_CUDA_CALL(cudaMalloc(&ptr->nnzpercell[i], size_nnzpercell));
    } else {
      ptr->sparse_offt[i] = NULL;
      ptr->sparse_vals[i] = NULL;
      ptr->cell_offt[i] = NULL;
      ptr->nnzpercell[i] = NULL;
    }
    if (!sparse) {
      GSVA_CUDA_CALL(cudaMalloc(&ptr->dense_ranks[i], size_sparse_arr));
    } else {
      ptr->dense_ranks[i] = NULL;
    }
    GSVA_CUDA_CALL(cudaMalloc(&ptr->decordstat[i], size_block_int));
    GSVA_CUDA_CALL(cudaMalloc(&ptr->symrnkstat[i], size_block_float));
    GSVA_CUDA_CALL(cudaMalloc(&ptr->es[i], size_es_block));
  }
  GSVA_CUDA_CALL(
      cudaMemcpy(ptr->gsetofft, gsetofft, size_offt, cudaMemcpyHostToDevice));
  GSVA_CUDA_CALL(
      cudaMemcpy(ptr->gsetidxs, gsetidxs, size_idxs, cudaMemcpyHostToDevice));

  for (int cat = 0; cat < GSVA_NUM_CATS; cat++) {
    ptr->cat_count[cat] = cat_count[cat];
    if (cat_count[cat] > 0) {
      GSVA_CUDA_CALL(cudaMalloc(&ptr->cat_gset[cat],
                                 (size_t)cat_count[cat] * sizeof(int)));
      GSVA_CUDA_CALL(cudaMemcpy(ptr->cat_gset[cat], cat_gset[cat],
                                 (size_t)cat_count[cat] * sizeof(int),
                                 cudaMemcpyHostToDevice));
    } else {
      ptr->cat_gset[cat] = NULL;
    }
  }

  return ptr;
}

void gsva_device_destroy(gsva_device_t *device) {
  for (int i = 0; i < GSVA_CUDA_STREAMS; i++) {
    cudaFree(device->decordstat[i]);
    cudaFree(device->symrnkstat[i]);
    cudaFree(device->es[i]);
    if (device->sparse_offt[i])
      cudaFree(device->sparse_offt[i]);
    if (device->sparse_vals[i])
      cudaFree(device->sparse_vals[i]);
    if (device->cell_offt[i])
      cudaFree(device->cell_offt[i]);
    if (device->nnzpercell[i])
      cudaFree(device->nnzpercell[i]);
    if (device->dense_ranks[i])
      cudaFree(device->dense_ranks[i]);
    cudaFree(device->r_scratch[i]);
    cudaStreamDestroy(device->stream[i]);
  }
  cudaFree(device->gsetofft);
  cudaFree(device->gsetidxs);
  for (int cat = 0; cat < GSVA_NUM_CATS; cat++) {
    if (device->cat_gset[cat])
      cudaFree(device->cat_gset[cat]);
  }
}

gsva_host_t *gsva_host_create(int G, int S, Rboolean sparse) {
  gsva_host_t *h = (gsva_host_t *)R_alloc(1, sizeof(gsva_host_t));

  size_t size_es = (size_t)S * GSVA_BLOCK_C * sizeof(gsva_float_t);

  size_t size_cast = (size_t)G * GSVA_BLOCK_C * sizeof(int);
  for (int i = 0; i < GSVA_CUDA_STREAMS; i++) {
    GSVA_CUDA_CALL(cudaMallocHost(&h->dense_cast[i], size_cast, 0));
    GSVA_CUDA_CALL(cudaMallocHost(&h->es[i], size_es, 0));
    if (sparse) {
      size_t size_rank_sparse = (size_t)G * GSVA_BLOCK_C * sizeof(int);
      size_t size_cell_offt_h = (size_t)(GSVA_BLOCK_C + 1) * sizeof(int);
      size_t size_nnz_h = (size_t)GSVA_BLOCK_C * sizeof(int);
      GSVA_CUDA_CALL(cudaMallocHost(&h->sparse_offt[i], size_rank_sparse, 0));
      GSVA_CUDA_CALL(cudaMallocHost(&h->sparse_vals[i], size_rank_sparse, 0));
      GSVA_CUDA_CALL(cudaMallocHost(&h->cell_offt[i], size_cell_offt_h, 0));
      GSVA_CUDA_CALL(cudaMallocHost(&h->nnzpercell[i], size_nnz_h, 0));
    } else {
      h->sparse_offt[i] = NULL;
      h->sparse_vals[i] = NULL;
      h->cell_offt[i] = NULL;
      h->nnzpercell[i] = NULL;
    }
  }

  return h;
}

void gsva_host_destroy(gsva_host_t *h) {
  for (int i = 0; i < GSVA_CUDA_STREAMS; i++) {
    cudaFreeHost(h->dense_cast[i]);
    cudaFreeHost(h->es[i]);
    if (h->sparse_offt[i])
      cudaFreeHost(h->sparse_offt[i]);
    if (h->sparse_vals[i])
      cudaFreeHost(h->sparse_vals[i]);
    if (h->cell_offt[i])
      cudaFreeHost(h->cell_offt[i]);
    if (h->nnzpercell[i])
      cudaFreeHost(h->nnzpercell[i]);
  }
}
