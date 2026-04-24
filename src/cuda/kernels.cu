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
)
{
    int c = blockIdx.x;
    int gset = blockIdx.y;
    int tid = threadIdx.x;

    if (c >= C || gset >= S) return;

    int offset = gsetofft[gset];
    int k = gsetofft[gset + 1] - offset;

    __shared__ int s_ranks[GSVA_THREAD_NUM];
    __shared__ double s_scores[GSVA_THREAD_NUM];

    if (tid < k) {
        int gene_idx = gsetidxs[offset + tid]; 
        int block_idx = c * G + (gene_idx - 1);
        
        s_ranks[tid] = decordstat_block[block_idx] - 1;
        s_scores[tid] = pow(fabs(symrnkstat_block[block_idx]), tau);
    } else {
        s_ranks[tid] = INT_MAX;
        s_scores[tid] = 0.0;
    }
    __syncthreads();

    if (k <= 32) {
        int my_rank = (tid < k) ? s_ranks[tid] : INT_MAX;
        double my_score = (tid < k) ? s_scores[tid] : 0.0;

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
            s_ranks[tid] = my_rank;
            s_scores[tid] = my_score;
        }
    } else {
        typedef cub::BlockRadixSort<int, GSVA_THREAD_NUM, 1, double> BlockRadixSort;
        __shared__ typename BlockRadixSort::TempStorage temp_storage;

        int thread_ranks[1];
        double thread_scores[1];

        thread_ranks[0] = (tid < k) ? s_ranks[tid] : INT_MAX;
        thread_scores[0] = (tid < k) ? s_scores[tid] : 0.0;

        BlockRadixSort(temp_storage).Sort(thread_ranks, thread_scores);

        if (tid < k) {
            s_ranks[tid] = thread_ranks[0];
            s_scores[tid] = thread_scores[0];
        }
        __syncthreads();
    }

    for (int step = 1; step < GSVA_THREAD_NUM; step *= 2) {
        double temp = 0.0;
        if (tid >= step) temp = s_scores[tid - step];
        __syncthreads();
        if (tid >= step) s_scores[tid] += temp;
        __syncthreads();
    }

    double total_pos_sum = s_scores[GSVA_THREAD_NUM - 1]; 
    if (total_pos_sum < 1e-6) {
        if (tid == 0) {
            out_es[gset * C + c] = 0.0;
        }
        return; 
    }

    double max_peak_local = 0.0;
    double min_valley_local = 0.0;

    if (tid < k) {
        double running_pos_sum_prev = (tid == 0) ? 0.0 : s_scores[tid - 1];
        double running_pos_curr = s_scores[tid];
        
        double neg_step = 1.0 / (double)(G - k);
        double neg_penalty = (s_ranks[tid] - tid) * neg_step;
        
        double current_valley = (running_pos_sum_prev / total_pos_sum) - neg_penalty;
        double current_peak = (running_pos_curr / total_pos_sum) - neg_penalty;
        
        max_peak_local = current_peak;
        min_valley_local = current_valley;
    }

    __shared__ double s_max_peak[GSVA_THREAD_NUM];
    __shared__ double s_min_valley[GSVA_THREAD_NUM];
    
    s_max_peak[tid] = max_peak_local;
    s_min_valley[tid] = min_valley_local;
    __syncthreads();

    for (int stride = GSVA_THREAD_NUM / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max_peak[tid] = fmax(s_max_peak[tid], s_max_peak[tid + stride]);
            s_min_valley[tid] = fmin(s_min_valley[tid], s_min_valley[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        double max_peak = s_max_peak[0];
        double min_valley = s_min_valley[0];
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
