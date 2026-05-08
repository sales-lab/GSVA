#include <R.h>
#include <Rdefines.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cuda/types.h"
#include "cuda/memory.h"
#include "cuda/cbind.h"
#include "cuda/kernels.cuh"
#include "ranks.h"
#include "rnd_walk.h"

static inline int min2(int a, int b) { return a < b ? a : b; }
static inline int max2(int a, int b) { return a > b ? a : b; }

SEXP gsva_cuda_thread_num_R(void) {
    SEXP res = PROTECT(allocVector(INTSXP, 1));
    INTEGER(res)[0] = GSVA_THREAD_NUM;
    UNPROTECT(1);
    return res;
}

/* Pack sparse DGCMatrix columns into COO-style (gene_offset, value) arrays.
 * Returns total_nnz across the block. */
static int
pack_ranks_dgc(const int* dgc_i, const int* dgc_p, const double* dgc_x,
               int G, int block_c, int block_size,
               int* out_offt, int* out_vals,
               int* out_cell_offt, int* out_nnzpercell)
{
    int offset = 0;
    for (int c = 0; c < block_size; c++) {
        int col = block_c + c;
        out_cell_offt[c] = offset;              /* save start offset BEFORE packing */
        out_nnzpercell[c] = dgc_p[col + 1] - dgc_p[col];
        for (int idx = dgc_p[col]; idx < dgc_p[col + 1]; idx++) {
            out_offt[offset]   = dgc_i[idx];    /* gene offset (0-based) */
            out_vals[offset]   = (int)dgc_x[idx]; /* rank value */
            offset++;
        }
    }
    out_cell_offt[block_size] = offset;
    return offset;
}

/* Pack sparse SVT_INT columns into COO-style (gene_offset, value) arrays.
 * Returns total_nnz across the block. */
static int
pack_ranks_svt(SEXP svt, int G, int block_c, int block_size,
               int* out_offt, int* out_vals,
               int* out_cell_offt, int* out_nnzpercell)
{
    int offset = 0;
    for (int c = 0; c < block_size; c++) {
        int col     = block_c + c;
        SEXP leaf   = VECTOR_ELT(svt, col);
        out_cell_offt[c] = offset;              /* save start offset BEFORE packing */
        if (leaf == R_NilValue) {
            out_nnzpercell[c] = 0;
        } else {
            SEXP   valsR    = VECTOR_ELT(leaf, 0);
            SEXP   offsetsR = VECTOR_ELT(leaf, 1);
            int    nvals    = length(valsR);
            int    noffsets = length(offsetsR);
            int*   offsets  = INTEGER(offsetsR);
            if (nvals > 0) {
                int* vals = INTEGER(valsR);
                for (int k = 0; k < nvals; k++) {
                    out_offt[offset] = offsets[k];
                    out_vals[offset] = vals[k];
                    offset++;
                }
            } else {
                /* lacunar: all values = 1 */
                for (int k = 0; k < noffsets; k++) {
                    out_offt[offset] = offsets[k];
                    out_vals[offset] = 1;
                    offset++;
                }
            }
            out_nnzpercell[c] = noffsets;
        }
    }
    out_cell_offt[block_size] = offset;
    return offset;
}

/* Transfer packed ranks to device and launch ranks2stats_gpu.
 * After this call, device->decordstat[stream_idx] and
 * device->symrnkstat[stream_idx] hold the computed stats on the GPU. */
static void
ranks2stats_gpu(ranks_ctx_t* ctx, int G, int block_c, int block_size,
                       int sparse_mode,
                       gsva_device_t* device, gsva_host_t* host,
                       int stream_idx, SEXP ranksR)
{
    cudaStream_t stream = device->stream[stream_idx];
    int fetch_type;

    int* h_offt  = host->sparse_offt[stream_idx];
    int* h_vals  = host->sparse_vals[stream_idx];
    int* h_cofft = host->cell_offt[stream_idx];
    int* h_nnzc  = host->nnzpercell[stream_idx];

    size_t sz_cofft = (size_t)(block_size + 1) * sizeof(int);
    size_t sz_nnzc  = (size_t)block_size * sizeof(int);

    if (ctx->type == RANKSTYPE_MATRIX_INT) {
        /* Dense integer matrix: transfer from column-major R to row-major device */
        fetch_type = R2S_MATRIX_INT;
        const int* ranks_data = INTEGER(ranksR);
        for (int c = 0; c < block_size; c++) {
            GSVA_CUDA_CALL(cudaMemcpyAsync(
                device->dense_ranks[stream_idx] + c * G,
                ranks_data + (block_c + c) * G,
                (size_t)G * sizeof(int),
                cudaMemcpyHostToDevice, stream));
        }
    } else if (ctx->type == RANKSTYPE_DGC) {
        fetch_type = R2S_SPARSE;
        int total_nnz = pack_ranks_dgc(
            ctx->u.dgc.i, ctx->u.dgc.p, ctx->u.dgc.x,
            G, block_c, block_size,
            h_offt, h_vals, h_cofft, h_nnzc);
        GSVA_CUDA_CALL(cudaMemcpyAsync(device->sparse_offt[stream_idx],
            h_offt, (size_t)total_nnz * sizeof(int), cudaMemcpyHostToDevice, stream));
        GSVA_CUDA_CALL(cudaMemcpyAsync(device->sparse_vals[stream_idx],
            h_vals, (size_t)total_nnz * sizeof(int), cudaMemcpyHostToDevice, stream));
        GSVA_CUDA_CALL(cudaMemcpyAsync(device->cell_offt[stream_idx],
            h_cofft, sz_cofft, cudaMemcpyHostToDevice, stream));
        GSVA_CUDA_CALL(cudaMemcpyAsync(device->nnzpercell[stream_idx],
            h_nnzc, sz_nnzc, cudaMemcpyHostToDevice, stream));
    } else {
        /* RANKSTYPE_SVT_INT */
        fetch_type = R2S_SPARSE;
        int total_nnz = pack_ranks_svt(
            ctx->u.svt, G, block_c, block_size,
            h_offt, h_vals, h_cofft, h_nnzc);
        GSVA_CUDA_CALL(cudaMemcpyAsync(device->sparse_offt[stream_idx],
            h_offt, (size_t)total_nnz * sizeof(int), cudaMemcpyHostToDevice, stream));
        GSVA_CUDA_CALL(cudaMemcpyAsync(device->sparse_vals[stream_idx],
            h_vals, (size_t)total_nnz * sizeof(int), cudaMemcpyHostToDevice, stream));
        GSVA_CUDA_CALL(cudaMemcpyAsync(device->cell_offt[stream_idx],
            h_cofft, sz_cofft, cudaMemcpyHostToDevice, stream));
        GSVA_CUDA_CALL(cudaMemcpyAsync(device->nnzpercell[stream_idx],
            h_nnzc, sz_nnzc, cudaMemcpyHostToDevice, stream));
    }

    int threads = min2(G, GSVA_R2S_THREADS);
    ranks2stats_gpu<<<block_size, threads, 0, stream>>>(
        fetch_type,
        device->dense_ranks[stream_idx],
        device->sparse_offt[stream_idx],
        device->sparse_vals[stream_idx],
        device->cell_offt[stream_idx],
        device->nnzpercell[stream_idx],
        device->r_scratch[stream_idx],
        device->decordstat[stream_idx],
        device->symrnkstat[stream_idx],
        G,
        block_size,
        (int)(sparse_mode != 0)
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        error("ranks2stats_gpu launch error at %s:%d: %s",
              __FILE__, __LINE__, cudaGetErrorString(err));
    }
}

static void gsva_rnd_walk_gpu(
    gsva_float_t*       h_es,
    int S, int G, int C,
    gsva_float_t tau, int score_type,
    gsva_device_t* device,
    int stream_idx
) {
    cudaStream_t stream = device->stream[stream_idx];
    size_t size_es      = (size_t)S * C * sizeof(gsva_float_t);

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
  Rboolean sparse = (Rboolean)asLogical(sparseR);
  gsva_float_t tau = (gsva_float_t)REAL(tauR)[0];

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

  gsva_device_t *device = gsva_device_create(genesetsidxR, G, (Rboolean)sparse);
  int S = device->S;
  gsva_host_t *host = gsva_host_create(G, S, (Rboolean)sparse);

  SEXP esR;
  PROTECT(esR = allocMatrix(REALSXP, S, C));
  double *es = REAL(esR);

  int total_blocks = (C + GSVA_BLOCK_C - 1) / GSVA_BLOCK_C;
  for (int block_id = 0; block_id < total_blocks; block_id++) {
    int stream_idx = block_id % GSVA_CUDA_STREAMS;
    int block_c = block_id * GSVA_BLOCK_C;
    int block_c_end = min2(block_c + GSVA_BLOCK_C, C);
    int block_size = block_c_end - block_c;

    GSVA_CUDA_CALL(cudaStreamSynchronize(device->stream[stream_idx]));

    if (block_id >= GSVA_CUDA_STREAMS) {
      int prev_block = block_id - GSVA_CUDA_STREAMS;
      int prev_c = prev_block * GSVA_BLOCK_C;
      int prev_end = min2(prev_c + GSVA_BLOCK_C, C);
      int prev_size = prev_end - prev_c;

      gsva_float_t* prev_es = host->es[stream_idx];
      for (int s = 0; s < S; s++) {
          for (int c = 0; c < prev_size; c++) {
              es[(prev_c + c) * S + s] = (double)prev_es[s * prev_size + c];
          }
      }
    }

    ranks2stats_gpu(ctx, G, block_c, block_size, (int)sparse,
                    device, host, stream_idx, ranksR);
    gsva_rnd_walk_gpu(
      host->es[stream_idx],
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

      gsva_float_t* h_es = host->es[stream_idx];
      for (int s = 0; s < S; s++) {
          for (int c = 0; c < block_size; c++) {
              es[(block_c + c) * S + s] = (double)h_es[s * block_size + c];
          }
      }
  }

  gsva_host_destroy(host);
  gsva_device_destroy(device);

  UNPROTECT(1);
  return (esR);
}
