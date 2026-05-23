#include <R.h>
#include <Rdefines.h>
#include <cuda_runtime.h>
#include "cuda/types.h"
#include "cuda/memory.h"
#include "cuda/cbind.h"
#include "cuda/kernels.cuh"
#include "ranks.h"
#include "rnd_walk.h"

static inline int min2(int a, int b) { return a < b ? a : b; }
static inline int max2(int a, int b) { return a > b ? a : b; }

static void gsva_rnd_walk_gpu(
    const int*    h_decordstat,
    const double* h_symrnkstat,
    double*       h_es,
    int S, int G, int C,
    double tau, int score_type,
    gsva_device_t* device,
    int stream_idx
) {
    cudaStream_t stream = device->stream[stream_idx];
    size_t size_block_int    = (size_t)C * G * sizeof(int);
    size_t size_block_double = (size_t)C * G * sizeof(double);
    size_t size_es           = (size_t)S * C * sizeof(double);

    GSVA_CUDA_CALL(cudaMemcpyAsync(device->decordstat[stream_idx],
        h_decordstat, size_block_int, cudaMemcpyHostToDevice, stream));
    GSVA_CUDA_CALL(cudaMemcpyAsync(device->symrnkstat[stream_idx],
        h_symrnkstat, size_block_double, cudaMemcpyHostToDevice, stream));

    dim3 threadsPerBlock(GSVA_THREAD_NUM);
    dim3 numBlocks(C, S);
    gsea_walk_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        device->gsetofft, device->gsetidxs,
        device->decordstat[stream_idx],
        device->symrnkstat[stream_idx],
        device->es[stream_idx],
        S, G, C, tau, score_type
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        error("CUDA kernel launch error at %s:%d: %s",
              __FILE__, __LINE__, cudaGetErrorString(err));
    }

    GSVA_CUDA_CALL(cudaMemcpyAsync(h_es, device->es[stream_idx],
        size_es, cudaMemcpyDeviceToHost, stream));
}

SEXP
gsva_score_genesets_gpu_R(SEXP ranksR, SEXP genesetsidxR, SEXP intrnksR,
                          SEXP sparseR, SEXP maxdiffR, SEXP absrnkR, SEXP tauR,
                          SEXP minsizeR, SEXP verboseR) {
  Rboolean intrnks = (Rboolean)asLogical(intrnksR);
  Rboolean sparse  = (Rboolean)asLogical(sparseR);
  double tau = REAL(tauR)[0];

  Rboolean maxdiff = (Rboolean)asLogical(maxdiffR);
  Rboolean absrnk  = (Rboolean)asLogical(absrnkR);
  int score_type;
  if (maxdiff) {
    score_type = absrnk ? 1 : 0;
  } else {
    score_type = 2;
  }

  ranks_ctx_t *ctx = ranks_ctx_create(ranksR, intrnks, sparse);
  int G = ctx->p;
  int C = ctx->n;

  gsva_device_t *device = gsva_device_create(genesetsidxR, G);
  int            S      = device->S;
  gsva_host_t   *h      = gsva_host_create(G, S);

  SEXP esR;
  PROTECT(esR = allocMatrix(REALSXP, S, C));
  double *es = REAL(esR);

  int total_blocks = (C + GSVA_BLOCK_C - 1) / GSVA_BLOCK_C;

  for (int block_id = 0; block_id < total_blocks; block_id++) {
      int stream_idx  = block_id % GSVA_CUDA_STREAMS;
      int block_c     = block_id * GSVA_BLOCK_C;
      int block_c_end = min2(block_c + GSVA_BLOCK_C, C);
      int block_size  = block_c_end - block_c;

      GSVA_CUDA_CALL(cudaStreamSynchronize(device->stream[stream_idx]));

      if (block_id >= GSVA_CUDA_STREAMS) {
          int prev_block = block_id - GSVA_CUDA_STREAMS;
          int prev_c     = prev_block * GSVA_BLOCK_C;
          int prev_end   = min2(prev_c + GSVA_BLOCK_C, C);
          int prev_size  = prev_end - prev_c;

          double* prev_es = h->es[stream_idx];
          for (int s = 0; s < S; s++) {
              for (int c = 0; c < prev_size; c++) {
                  es[(prev_c + c) * S + s] = prev_es[s * prev_size + c];
              }
          }
      }

      int*    h_dec = h->decordstat[stream_idx];
      double* h_sym = h->symrnkstat[stream_idx];
      for (int c = 0; c < block_size; c++) {
          ranks2stats(ctx, block_c + c,
                      &h_dec[c * G], &h_sym[c * G]);
      }

      gsva_rnd_walk_gpu(
          h_dec, h_sym, h->es[stream_idx],
          S, G, block_size, tau, score_type,
          device, stream_idx
      );
  }

  cudaDeviceSynchronize();
  for (int block_id = max2(0, total_blocks - GSVA_CUDA_STREAMS);
       block_id < total_blocks;
       block_id++) {
      int stream_idx  = block_id % GSVA_CUDA_STREAMS;
      int block_c     = block_id * GSVA_BLOCK_C;
      int block_c_end = min2(block_c + GSVA_BLOCK_C, C);
      int block_size  = block_c_end - block_c;

      double* h_es = h->es[stream_idx];
      for (int s = 0; s < S; s++) {
          for (int c = 0; c < block_size; c++) {
              es[(block_c + c) * S + s] = h_es[s * block_size + c];
          }
      }
  }

  gsva_host_destroy(h);
  gsva_device_destroy(device);

  UNPROTECT(1);
  return (esR);
}
