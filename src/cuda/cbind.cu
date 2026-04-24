#include "cuda/cbind.h"
#include "cuda/internal.h"
#include "cuda/kernels.cuh"
#include "ranks.h"
#include "rnd_walk.h"

static void gsva_rnd_walk_cuda(
    const int* h_decordstat_block,
    const double* h_symrnkstat_block,
    double* h_es,
    int S, int G, int C,
    double tau, int score_type,
    struct gsva_device_t* device
)
{
    size_t size_block_int = (size_t)C * G * sizeof(int);
    size_t size_block_double = (size_t)C * G * sizeof(double);
    size_t size_es = S * C * sizeof(double);

    GSVA_CUDA_CALL(cudaMemcpy(device->decordstat_block, h_decordstat_block, size_block_int, cudaMemcpyHostToDevice));
    GSVA_CUDA_CALL(cudaMemcpy(device->symrnkstat_block, h_symrnkstat_block, size_block_double, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(GSVA_THREAD_NUM);
    dim3 numBlocks(C, S);

    gsea_walk_kernel<<<numBlocks, threadsPerBlock>>>(
        device->gsetofft, device->gsetidxs, device->decordstat_block,
        device->symrnkstat_block, device->es,
        S, G, C, tau, score_type
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        error("CUDA kernel error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err));
    }

    GSVA_CUDA_CALL(cudaMemcpy(h_es, device->es, size_es, cudaMemcpyDeviceToHost));
}

SEXP
gsva_score_genesets_gpu_R(SEXP ranksR, SEXP genesetsidxR, SEXP intrnksR,
                          SEXP sparseR, SEXP maxdiffR, SEXP absrnkR, SEXP tauR,
                          SEXP minsizeR, SEXP verboseR) {
  Rboolean intrnks = (Rboolean)asLogical(intrnksR);
  Rboolean sparse = (Rboolean)asLogical(sparseR);
  double tau = REAL(tauR)[0];

  Rboolean maxdiff = (Rboolean)asLogical(maxdiffR);
  Rboolean absrnk = (Rboolean)asLogical(absrnkR);
  int score_type;
  if (maxdiff) {
    score_type = absrnk ? 1 : 0;
  } else {
    score_type = 2;
  }

  ranks_ctx_t *ctx = ranks_ctx_create(ranksR, intrnks, sparse);
  int G = ctx->p;
  int C = ctx->n;
  int S = length(genesetsidxR);

  int total_gset_entries = 0;
  for (int s = 0; s < S; s++) {
    SEXP gsetidxR = VECTOR_ELT(genesetsidxR, s);
    int k = length(gsetidxR);
    total_gset_entries += k;
  }

  int *gsetofft = (int*)R_alloc(S+1, sizeof(int));
  int *gsetidxs = (int*)R_alloc(total_gset_entries, sizeof(int));

  total_gset_entries = 0;
  for (int s = 0; s < S; s++) {
    SEXP gsetidxR = VECTOR_ELT(genesetsidxR, s);
    int* gsetidx = INTEGER(gsetidxR);
    int k = length(gsetidxR);
    
    gsetofft[s] = total_gset_entries;
    for (int i = 0; i < k; i++) {
      gsetidxs[total_gset_entries++] = gsetidx[i];
    }
  }
  gsetofft[S] = total_gset_entries;

  struct gsva_device_t *device = gsva_device_create(G, S, gsetofft, gsetidxs);

  int* decordstat_block = (int*)R_alloc(GSVA_BLOCK_C * G, sizeof(int));
  double* symrnkstat_block = (double*)R_alloc(GSVA_BLOCK_C * G, sizeof(double));
  double* tmp = (double*)R_alloc(S * GSVA_BLOCK_C, sizeof(double));

  SEXP esR;
  PROTECT(esR = allocMatrix(REALSXP, S, C));
  double *es = REAL(esR);

  for (int block_c = 0; block_c < C; block_c += GSVA_BLOCK_C) {
    int block_c_end = block_c + GSVA_BLOCK_C;
    if (block_c_end > C) block_c_end = C;
    int block_c_size = block_c_end - block_c;
    
    for (int c = 0; c < block_c_size; c++) {
      ranks2stats(ctx, block_c + c,
                  &decordstat_block[c * G], &symrnkstat_block[c * G]);
    }

    gsva_rnd_walk_cuda(
      decordstat_block, symrnkstat_block, tmp,
      S, G, block_c_size, tau, score_type, device
    );

    for (int s = 0; s < S; s++) {
      for (int c = 0; c < block_c_size; c++) {
        es[(block_c + c) * S + s] = tmp[s * block_c_size + c];
      }
    }
  }

  cudaDeviceSynchronize();
  gsva_device_destroy(device);

  UNPROTECT(1);
  return (esR);
}
