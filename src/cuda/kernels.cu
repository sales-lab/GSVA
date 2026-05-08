#include <math.h>
#include <limits.h>
#include <cub/cub.cuh>
#include "kernels.cuh"
#include "cuda/types.h"

__global__ void ranks2stats_gpu(
    int fetch_type,
    const int* __restrict__ dense_ranks,
    const int* __restrict__ sparse_offt,
    const int* __restrict__ sparse_vals,
    const int* __restrict__ cell_offt,
    const int* __restrict__ nnzpercell,
    int* __restrict__ d_r,
    int* __restrict__ decordstat,
    gsva_float_t* __restrict__ symrnkstat,
    int G,
    int block_size,
    int sparse_mode
)
{
    int cell = blockIdx.x;
    int tid  = threadIdx.x;
    int threads = blockDim.x;
    if (cell >= block_size) return;

    int is_dense = (fetch_type == R2S_MATRIX_INT);
    int nnz, nzs;

    /* --- Phase 1: Init and scatter directly into global memory d_r[] --- */
    if (is_dense) {
        nnz = G;
        nzs = 0;
        /* Dense: copy directly into global scratch buffer */
        const int* dr = dense_ranks + cell * G;
        for (int g = tid; g < G; g += threads) {
            d_r[cell * G + g] = dr[g];
        }
    } else {
        /* Sparse: zero-init global region, then scatter stored entries */
        int base = cell_offt[cell];
        nnz = nnzpercell[cell];
        nzs = G - nnz;

        // Grid-stride zero-initialization of the cell's global rank buffer
        for (int g = tid; g < G; g += threads) {
            d_r[cell * G + g] = 0;
        }
        __syncthreads(); // Barrier: all threads must finish zero-init before scattering

        // Scatter non-zero stored ranks into their correct gene positions
        for (int i = tid; i < nnz; i += threads) {
            int pos    = base + i;
            int gene   = sparse_offt[pos];
            int val    = sparse_vals[pos];
            d_r[cell * G + gene] = val;  // Direct global memory write
        }
    }
    __syncthreads(); // Barrier: all scatters complete before Phase 2 begins

    /* --- Phase 2: Parallel prefix scan over d_r[] using global memory tiling --- */
    if (!is_dense) {
        struct ZeroCountPrefixOp {
            int running_total;
            __device__ ZeroCountPrefixOp(int r) : running_total(r) {}
            __device__ int operator()(int chunk_zeros) {
                int old = running_total;
                running_total += chunk_zeros;
                return old;
            }
        };

        ZeroCountPrefixOp prefix_op(0);
        int num_chunks = (G + threads - 1) / threads;

        using BlockScan = cub::BlockScan<int, GSVA_R2S_THREADS>;
        __shared__ typename BlockScan::TempStorage scan_storage;

        for (int chunk = 0; chunk < num_chunks; chunk++) {
            int idx = chunk * threads + tid;

            // Read directly from global memory
            int cur_val = (idx < G) ? d_r[cell * G + idx] : 0;
            int is_zero = (cur_val == 0) ? 1 : 0;

            // Exclusive scan across this chunk, auto-carry via callback
            int scan_result = 0;
            BlockScan(scan_storage).ExclusiveSum(is_zero, scan_result, prefix_op);

            // Assign final rank directly back to global memory
            if (idx < G) {
                if (is_zero) {
                    d_r[cell * G + idx] = scan_result + 1;
                } else {
                    d_r[cell * G + idx] = cur_val + nzs;
                }
            }
            __syncthreads(); // Protect scan_storage reuse across chunks
        }
    }

    /* --- Phase 3: Compute stats (parallel over genes, read from global d_r[]) --- */
    for (int g = tid; g < G; g += threads) {
        int rshifted = d_r[cell * G + g];  /* now holds SHIFTED rank in all cases */

        /* decordstat = G - rshifted + 1 */
        decordstat[cell * G + g] = G - rshifted + 1;

        /* symrnkstat — choose formula */
        if (sparse_mode && nzs > 0 && !is_dense) {
            /* Sparse formula: uses RAW rank (before shift) */
            /* rshifted <= nzs means it was an implicit zero */
            if (rshifted > nzs) {
                int raw_r = rshifted - nzs;
                symrnkstat[cell * G + g] = GSVA_FABS((gsva_float_t)(nnz + 1) / 2.0f - (gsva_float_t)(raw_r + 1));
            } else {
                /* was zero */
                symrnkstat[cell * G + g] = GSVA_FABS((gsva_float_t)(nnz + 1) / 2.0f - 1.0f);
            }
        } else {
            /* Dense formula: uses SHIFTED rank */
            symrnkstat[cell * G + g] = GSVA_FABS((gsva_float_t)G / 2.0f - (gsva_float_t)rshifted);
        }
    }
}

__global__ void gsea_walk_kernel(
    const int* __restrict__ gsetofft,
    const int* __restrict__ gsetidxs,
    const int* __restrict__ decordstat_block,
    const gsva_float_t* __restrict__ symrnkstat_block,
    gsva_float_t* __restrict__ out_es,
    int S,
    int G,
    int C,
    gsva_float_t tau,
    int score_type
) {
    int c = blockIdx.x;
    int gset = blockIdx.y;
    int tid = threadIdx.x;

    if (c >= C || gset >= S) return;

    int offset = gsetofft[gset];
    int k = gsetofft[gset + 1] - offset;
    assert(k <= GSVA_THREAD_NUM && "gene set exceeds GPU shared-memory capacity");

    typedef cub::BlockRadixSort<int, GSVA_THREAD_NUM, 1, gsva_float_t> BlockRadixSort;
    union shared_mem_t {
        struct {
            int ranks[GSVA_THREAD_NUM];
            gsva_float_t scores[GSVA_THREAD_NUM];
        } phase1;
        typename BlockRadixSort::TempStorage sort_storage;
        struct {
            gsva_float_t max_peak[GSVA_THREAD_NUM];
            gsva_float_t min_valley[GSVA_THREAD_NUM];
        } phase2;
    };
    __shared__ shared_mem_t s;

    if (tid < k) {
        int gene_idx = gsetidxs[offset + tid]; 
        int block_idx = c * G + (gene_idx - 1);
        
        s.phase1.ranks[tid] = decordstat_block[block_idx] - 1;
        s.phase1.scores[tid] = GSVA_POW(GSVA_FABS(symrnkstat_block[block_idx]), tau);
    } else {
        s.phase1.ranks[tid] = INT_MAX;
        s.phase1.scores[tid] = 0.0f;
    }
    __syncthreads();

    if (k <= 32) {
        int my_rank = s.phase1.ranks[tid];
        gsva_float_t my_score = s.phase1.scores[tid];

        for (int k_step = 2; k_step <= 32; k_step <<= 1) {
            for (int j = k_step >> 1; j > 0; j >>= 1) {
                int partner_rank = __shfl_xor_sync(0xffffffff, my_rank, j);
                gsva_float_t partner_score = __shfl_xor_sync(0xffffffff, my_score, j);
                
                bool ascending = ((tid & k_step) == 0);
                int partner_idx = tid ^ j;
                
                if (ascending) {
                    if (my_rank > partner_rank && partner_idx > tid) {
                        my_rank = partner_rank;
                        my_score = partner_score;
                    } else if (my_rank < partner_rank && partner_idx < tid) {
                        my_rank = partner_rank;
                        my_score = partner_score;
                    }
                } else {
                    if (my_rank < partner_rank && partner_idx > tid) {
                        my_rank = partner_rank;
                        my_score = partner_score;
                    } else if (my_rank > partner_rank && partner_idx < tid) {
                        my_rank = partner_rank;
                        my_score = partner_score;
                    }
                }
            }
        }

        if (tid < k) {
            s.phase1.ranks[tid] = my_rank;
            s.phase1.scores[tid] = my_score;
        }
    } else {
        int thread_ranks[1];
        gsva_float_t thread_scores[1];

        thread_ranks[0] = s.phase1.ranks[tid];
        thread_scores[0] = s.phase1.scores[tid];
        __syncthreads();
        
        BlockRadixSort(s.sort_storage).Sort(thread_ranks, thread_scores);
        __syncthreads();

        if (tid < k) {
            s.phase1.ranks[tid] = thread_ranks[0];
            s.phase1.scores[tid] = thread_scores[0];
        }
    }
    __syncthreads();

    for (int step = 1; step < GSVA_THREAD_NUM; step *= 2) {
        gsva_float_t temp = 0.0f;
        if (tid >= step) temp = s.phase1.scores[tid - step];
        __syncthreads();
        if (tid >= step) s.phase1.scores[tid] += temp;
        __syncthreads();
    }

    gsva_float_t total_pos_sum = s.phase1.scores[GSVA_THREAD_NUM - 1]; 
    if (total_pos_sum < 1e-6f) {
        if (tid == 0) {
            out_es[gset * C + c] = 0.0f;
        }
        return; 
    }

    gsva_float_t max_peak_local = 0.0f;
    gsva_float_t min_valley_local = 0.0f;

    if (tid < k) {
        gsva_float_t running_pos_sum_prev = (tid == 0) ? 0.0f : s.phase1.scores[tid - 1];
        gsva_float_t running_pos_curr = s.phase1.scores[tid];
        
        gsva_float_t neg_step = 1.0f / (gsva_float_t)(G - k);
        gsva_float_t neg_penalty = (s.phase1.ranks[tid] - tid) * neg_step;
        
        gsva_float_t current_valley = (running_pos_sum_prev / total_pos_sum) - neg_penalty;
        gsva_float_t current_peak = (running_pos_curr / total_pos_sum) - neg_penalty;
        
        max_peak_local = current_peak;
        min_valley_local = current_valley;
    }
    __syncthreads();

    s.phase2.max_peak[tid] = max_peak_local;
    s.phase2.min_valley[tid] = min_valley_local;
    __syncthreads();

    for (int stride = GSVA_THREAD_NUM / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s.phase2.max_peak[tid] = fmax(s.phase2.max_peak[tid], s.phase2.max_peak[tid + stride]);
            s.phase2.min_valley[tid] = fmin(s.phase2.min_valley[tid], s.phase2.min_valley[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        gsva_float_t max_peak = s.phase2.max_peak[0];
        gsva_float_t min_valley = s.phase2.min_valley[0];
        gsva_float_t result = 0.0f;
        
        switch (score_type) {
            case 0: result = max_peak + min_valley; break;
            case 1: result = max_peak - min_valley; break;
            case 2: result = (max_peak > GSVA_FABS(min_valley)) ? max_peak : min_valley; break;
            default: result = -1.0f;
        }
        
        out_es[gset * C + c] = result;
    }
}
