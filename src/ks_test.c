#include <math.h>
#include <string.h>
#include <assert.h>
#include <R.h>
#include <Rdefines.h>
#include <R_ext/Rdynload.h>
#include <cli/progress.h>
#include "ranks.h"
#include "rnd_walk.h"

/* global variables */
extern SEXP GSVA_attrNAsSym;

SEXP
gsva_score_genesets_cpu_R(SEXP ranksR, SEXP genesetsidxR, SEXP intrnksR,
                          SEXP sparseR, SEXP maxdiffR, SEXP absrnkR, SEXP tauR,
                          SEXP anynaR, SEXP nauseR, SEXP minsizeR, SEXP verboseR) {
  int      p, n;
  int      m = length(genesetsidxR);
  Rboolean intrnks=asLogical(intrnksR);
  Rboolean sparse=asLogical(sparseR);
  Rboolean maxdiff=asLogical(maxdiffR);
  Rboolean absrnk=asLogical(absrnkR);
  double   tau=REAL(tauR)[0];
  Rboolean anyna=asLogical(anynaR);
  int      nause=INTEGER(nauseR)[0]; /* everything=1, all.obs=2, na.rm=3 */
  int      minsize=INTEGER(minsizeR)[0];
  SEXP     esR;
  double*  es;
  int      wna = 0;
  Rboolean abort = FALSE;
  Rboolean verbose = asLogical(verboseR);
  SEXP     pb = R_NilValue;
  int      nunprotect = 0;
  int*     decordstat_col;
  double*  symrnkstat_col;
  ranks_ctx_t* ctx;

  ctx = ranks_ctx_create(ranksR, intrnks, sparse);
  p = ctx->p; /* number of rows/genes/features */
  n = ctx->n; /* number of columns/samples/cells/spots */

  decordstat_col = (int*)R_alloc(p, sizeof(int));
  symrnkstat_col = (double*)R_alloc(p, sizeof(double));

  PROTECT(esR = allocMatrix(REALSXP, m, n)); nunprotect++;
  es = REAL(esR);

  if (verbose) {
    pb = PROTECT(cli_progress_bar(p, NULL)); nunprotect++;
    cli_progress_set_name(pb, "Calculating GSVA scores");
  }

  for (int i=0; i < n; i++) {
    if (verbose) { /* show progress */
      if (i % 100 == 0 && CLI_SHOULD_TICK)
        cli_progress_set(pb, i);
    }

    if (anyna)
      ranks2stats_nas(ctx, i, decordstat_col, symrnkstat_col);
    else
      ranks2stats(ctx, i, decordstat_col, symrnkstat_col);

    for (int j=0; j < m; j++) {
      SEXP     gsetidxR = VECTOR_ELT(genesetsidxR, j);
      int*     gsetidx;
      int      k = length(gsetidxR);
#ifdef LONG_VECTOR_SUPPORT
      R_xlen_t idx = (R_xlen_t) m * i + j;
#else
      int      idx = (size_t) m * i + j;
#endif
      double  walkstatpos, walkstatneg;

      walkstatpos = walkstatneg = NA_REAL;
      gsetidx = INTEGER(gsetidxR);
      if (anyna)
        gsva_rnd_walk_nas(gsetidx, k, decordstat_col, symrnkstat_col, p, tau,
                          nause, minsize, &walkstatpos, &walkstatneg, &wna);
      else
        gsva_rnd_walk(gsetidx, k, decordstat_col, symrnkstat_col, p, tau,
                      &walkstatpos, &walkstatneg);

      es[idx] = NA_REAL;
      if (!anyna || (!ISNA(walkstatpos) && !ISNA(walkstatneg))) {
        if (maxdiff) {
          es[idx] = walkstatpos + walkstatneg;
          if (absrnk)
            es[idx] = walkstatpos - walkstatneg;
        } else {
          es[idx] = (walkstatpos > fabs(walkstatneg)) ? walkstatpos : walkstatneg;
        }
      } else {
        if (anyna && (ISNA(walkstatpos) || ISNA(walkstatneg)) && nause == 2) { /* all.obs */
          abort=TRUE;
          break;
        }
      }
    }
  }

  if (anyna) {
    SEXP attr;

    if (nause == 2 && abort) {
      PROTECT(attr = allocVector(STRSXP, 1));
      SET_STRING_ELT(attr, 0, mkChar("abort"));
      Rf_setAttrib(esR, GSVA_attrNAsSym, attr);
      UNPROTECT(1); /* attr */
    } else if (nause == 3 && wna == 1) {
      PROTECT(attr = allocVector(STRSXP, 1));
      SET_STRING_ELT(attr, 0, mkChar("wna"));
      Rf_setAttrib(esR, GSVA_attrNAsSym, attr);
      UNPROTECT(1); /* attr */
    }
  }

  if (verbose)
    cli_progress_done(pb);

  UNPROTECT(nunprotect); /* esR pb */

  return(esR);
}
