#include <math.h>
#include <limits.h>
#include <cub/cub.cuh>
#include "kernels.cuh"

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
) {
    int c = blockIdx.x;
    int gset = blockIdx.y;
    int tid = threadIdx.x;

    if (c >= C || gset >= S) return;

    int offset = gsetofft[gset];
    int k = gsetofft[gset + 1] - offset;
    assert(k <= GSVA_THREAD_NUM && "gene set exceeds GPU shared-memory capacity");

    typedef cub::BlockRadixSort<int, GSVA_THREAD_NUM, 1, double> BlockRadixSort;
    union shared_mem_t {
        struct {
            int ranks[GSVA_THREAD_NUM];
            double scores[GSVA_THREAD_NUM];
        } phase1;
        typename BlockRadixSort::TempStorage sort_storage;
        struct {
            double max_peak[GSVA_THREAD_NUM];
            double min_valley[GSVA_THREAD_NUM];
        } phase2;
    };
    __shared__ shared_mem_t s;

    if (tid < k) {
        int gene_idx = gsetidxs[offset + tid]; 
        int block_idx = c * G + (gene_idx - 1);
        
        s.phase1.ranks[tid] = decordstat_block[block_idx] - 1;
        s.phase1.scores[tid] = pow(fabs(symrnkstat_block[block_idx]), tau);
    } else {
        s.phase1.ranks[tid] = INT_MAX;
        s.phase1.scores[tid] = 0.0;
    }
    __syncthreads();

    if (k <= 32) {
        int my_rank = s.phase1.ranks[tid];
        double my_score = s.phase1.scores[tid];

        for (int k_step = 2; k_step <= 32; k_step <<= 1) {
            for (int j = k_step >> 1; j > 0; j >>= 1) {
                int partner_rank = __shfl_xor_sync(0xffffffff, my_rank, j);
                double partner_score = __shfl_xor_sync(0xffffffff, my_score, j);
                
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
        double thread_scores[1];

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
        double temp = 0.0;
        if (tid >= step) temp = s.phase1.scores[tid - step];
        __syncthreads();
        if (tid >= step) s.phase1.scores[tid] += temp;
        __syncthreads();
    }

    double total_pos_sum = s.phase1.scores[GSVA_THREAD_NUM - 1]; 
    if (total_pos_sum < 1e-6) {
        if (tid == 0) {
            out_es[gset * C + c] = 0.0;
        }
        return; 
    }

    double max_peak_local = 0.0;
    double min_valley_local = 0.0;

    if (tid < k) {
        double running_pos_sum_prev = (tid == 0) ? 0.0 : s.phase1.scores[tid - 1];
        double running_pos_curr = s.phase1.scores[tid];
        
        double neg_step = 1.0 / (double)(G - k);
        double neg_penalty = (s.phase1.ranks[tid] - tid) * neg_step;
        
        double current_valley = (running_pos_sum_prev / total_pos_sum) - neg_penalty;
        double current_peak = (running_pos_curr / total_pos_sum) - neg_penalty;
        
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
        double max_peak = s.phase2.max_peak[0];
        double min_valley = s.phase2.min_valley[0];
        double result = 0.0;
        
        switch (score_type) {
            case 0: result = max_peak + min_valley; break;
            case 1: result = max_peak - min_valley; break;
            case 2: result = (max_peak > fabs(min_valley)) ? max_peak : min_valley; break;
            default: result = -1.0;
        }
        
        out_es[gset * C + c] = result;
    }
}
