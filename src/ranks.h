#ifndef RANKS_H
#define RANKS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <R.h>
#include <Rdefines.h>

#define RANKSTYPE_MATRIX_INT 1
#define RANKSTYPE_MATRIX_DBL 2
#define RANKSTYPE_DGC 3
#define RANKSTYPE_SVT_INT 4
#define RANKSTYPE_SVT_DBL 5

typedef struct ranks_ctx_s {
  int type; // RANKSTYPE_*
  union {
    const void *data;
    struct {
      const int *i;
      const int *p;
      const double *x;
    } dgc;
    SEXP svt;
  } u;
  int p;
  int n;
  Rboolean sparse;
  int (*fetch_col)(struct ranks_ctx_s *ctx, int j);
  int *r;
} ranks_ctx_t;

ranks_ctx_t *ranks_ctx_create(SEXP XR, Rboolean intrnks, Rboolean sparse);

void ranks2stats(ranks_ctx_t *ctx, int j, int *decordstat, double *symrnkstat);
void ranks2stats_nas(ranks_ctx_t *ctx, int j, int *decordstat,
                     double *symrnkstat);

#ifdef __cplusplus
}
#endif

#endif // RANKS_H
