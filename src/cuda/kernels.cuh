#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <math.h>
#include <limits.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda/types.h"

#define GSVA_THREAD_NUM 128
#define GSVA_R2S_THREADS 512

static_assert(GSVA_THREAD_NUM % 2 == 0, "GSVA_THREAD_NUM must be divisible by 2");

typedef enum {
    R2S_DENSE  = 1,  // Dense rank matrix (int or double)
    R2S_SPARSE = 2   // Sparse layout: DGC or SVT_INT/DGC
} r2s_fetch_type_t;

__global__ void ranks2stats_gpu(
    r2s_fetch_type_t fetch_type,
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
) {
    int cell = blockIdx.x;
    int tid  = threadIdx.x;
    int threads = blockDim.x;
    if (cell >= block_size) return;

    int nnz, nzs;
    if (fetch_type == R2S_DENSE) {
        nnz = G;
        nzs = 0;
        const int* dr = dense_ranks + cell * G;
        for (int g = tid; g < G; g += threads) {
            d_r[cell * G + g] = dr[g];
        }
    } else {
        int base = cell_offt[cell];
        nnz = nnzpercell[cell];
        nzs = G - nnz;

        for (int g = tid; g < G; g += threads) {
            d_r[cell * G + g] = 0;
        }
        __syncthreads();

        for (int i = tid; i < nnz; i += threads) {
            int pos    = base + i;
            int gene   = sparse_offt[pos];
            int val    = sparse_vals[pos];
            d_r[cell * G + gene] = val;
        }
    }
    __syncthreads();

    if (fetch_type == R2S_SPARSE) {
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

            int cur_val = (idx < G) ? d_r[cell * G + idx] : 0;
            int is_zero = (cur_val == 0) ? 1 : 0;

            int scan_result = 0;
            BlockScan(scan_storage).ExclusiveSum(is_zero, scan_result, prefix_op);

            if (idx < G) {
                if (is_zero) {
                    d_r[cell * G + idx] = scan_result + 1;
                } else {
                    d_r[cell * G + idx] = cur_val + nzs;
                }
            }
            __syncthreads();
        }
    }

    for (int g = tid; g < G; g += threads) {
        int rshifted = d_r[cell * G + g];
        decordstat[cell * G + g] = G - rshifted + 1;
        
        if (sparse_mode && nzs > 0 && fetch_type == R2S_SPARSE) {
            if (rshifted > nzs) {
                int raw_r = rshifted - nzs;
                symrnkstat[cell * G + g] = GSVA_FABS((gsva_float_t)(nnz + 1) / 2.0f - (gsva_float_t)(raw_r + 1));
            } else {
                /* Was zero */
                symrnkstat[cell * G + g] = GSVA_FABS((gsva_float_t)(nnz + 1) / 2.0f - 1.0f);
            }
        } else {
            symrnkstat[cell * G + g] = GSVA_FABS((gsva_float_t)G / 2.0f - (gsva_float_t)rshifted);
        }
    }
}

template<int GSVA_WALK_ITEMS, int GSVA_WALK_THREADS>
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
    typedef cub::BlockRadixSort<int, GSVA_WALK_THREADS, GSVA_WALK_ITEMS, gsva_float_t> BlockRadixSort;
    typedef cub::BlockScan<gsva_float_t, GSVA_WALK_THREADS> BlockScanFloat;

    struct PeakValley {
        gsva_float_t peak;
        gsva_float_t valley;
    };
    struct PVReduceOp {
        __device__ PeakValley operator()(PeakValley a, PeakValley b) const {
            return {
                a.peak > b.peak ? a.peak : b.peak,
                a.valley < b.valley ? a.valley : b.valley
            };
        }
    };
    typedef cub::BlockReduce<PeakValley, GSVA_WALK_THREADS> BlockReducePV;

    union TempStorage {
        typename BlockRadixSort::TempStorage sort_storage;
        typename BlockScanFloat::TempStorage scan_storage;
        typename BlockReducePV::TempStorage reduce_storage;
    };

    __shared__ TempStorage tmp;
    __shared__ gsva_float_t thread_total;

    int c = blockIdx.x;
    int gset = blockIdx.y;
    int tid = threadIdx.x;

    if (c >= C || gset >= S) return;

    int offset = gsetofft[gset];
    int k = gsetofft[gset + 1] - offset;

    int keys[GSVA_WALK_ITEMS];
    gsva_float_t vals[GSVA_WALK_ITEMS];

    int thread_start = tid * GSVA_WALK_ITEMS;
    for (int i = 0; i < GSVA_WALK_ITEMS; i++) {
        int idx = thread_start + i;
        if (idx < k) {
            int gene_idx = gsetidxs[offset + idx];
            int block_idx = c * G + (gene_idx - 1);
            keys[i] = decordstat_block[block_idx] - 1;
            vals[i] = GSVA_POW(GSVA_FABS(symrnkstat_block[block_idx]), tau);
        } else {
            keys[i] = INT_MAX;
            vals[i] = 0.0f;
        }
    }

    BlockRadixSort(tmp.sort_storage).Sort(keys, vals);
    __syncthreads();

    gsva_float_t stat_sums[GSVA_WALK_ITEMS];
    for (int i = 0; i < GSVA_WALK_ITEMS; i++) {
        if (keys[i] != INT_MAX) {
            stat_sums[i] = vals[i];
        } else {
            stat_sums[i] = 0.0f;
        }
    }

    BlockScanFloat(tmp.scan_storage).InclusiveSum(stat_sums, stat_sums);
    if (tid == GSVA_WALK_THREADS - 1) thread_total = stat_sums[GSVA_WALK_ITEMS - 1];
    __syncthreads();

    gsva_float_t stat_total = thread_total;
    if (stat_total < 1e-6f || k == G) {
        if (tid == 0) out_es[c * S + gset] = 0.0f;
        return;
    }

    gsva_float_t thread_max_peak = -1e30f;
    gsva_float_t thread_min_valley = 1e30f;
    gsva_float_t neg_step = 1.0f / (gsva_float_t)(G - k);

    for (int i = 0; i < GSVA_WALK_ITEMS; i++) {
        if (keys[i] == INT_MAX) continue;

        gsva_float_t stat_running = stat_sums[i] - vals[i];
        gsva_float_t stat_norm = stat_running / stat_total;
        gsva_float_t neg_penalty = (keys[i] - (thread_start + i)) * neg_step;
        gsva_float_t current_valley = stat_norm - neg_penalty;
        if (current_valley < thread_min_valley) thread_min_valley = current_valley;

        stat_running += vals[i];
        stat_norm = stat_running / stat_total;
        gsva_float_t current_peak = stat_norm - neg_penalty;
        if (current_peak > thread_max_peak) thread_max_peak = current_peak;
    }

    PeakValley pv = {thread_max_peak, thread_min_valley};
    PeakValley result = BlockReducePV(tmp.reduce_storage).Reduce(pv, PVReduceOp());

    if (tid == 0) {
        gsva_float_t final_max_peak = result.peak;
        gsva_float_t final_min_valley = result.valley;
        gsva_float_t res = 0.0f;
        switch (score_type) {
            case 0: res = final_max_peak + final_min_valley; break;
            case 1: res = final_max_peak - final_min_valley; break;
            case 2: res = (final_max_peak > GSVA_FABS(final_min_valley)) ? final_max_peak : final_min_valley; break;
            default: res = -1.0f; break;
        }
        out_es[c * S + gset] = res;
    }
}

#endif
