#ifndef CUDA_MEMORY_H
#define CUDA_MEMORY_H

#ifdef __cplusplus
extern "C" {
#endif

struct gsva_device_t {
    int G, S;
    int *gsetofft, *gsetidxs;
    int *decordstat_block;
    double *symrnkstat_block;
    double *es;
};

struct gsva_device_t* gsva_device_create(
    int G, int S, int* gsetofft, int* gsetidxs
);

void gsva_device_destroy(struct gsva_device_t* device);

#ifdef __cplusplus
}
#endif

#endif
