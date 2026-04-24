# verify_golden.R — Correctness verification against golden reference
#
# Compares CPU and GPU enrichment scores against a pre-saved golden reference.
#
# Exit code 0 on all checks passing, 1 on any failure.

suppressPackageStartupMessages({
  library(GSEABase)
  library(GSVA)
  library(scuttle)
  library(SingleCellExperiment)
  library(TENxPBMCData)
})

# ── Load golden reference ────────────────────────────────────────────────────
gold_path <- "benchmarks/golden/reference.rds"
if (!file.exists(gold_path)) {
  stop("Golden reference not found at ", gold_path,
       "\nRun `Rscript benchmarks/save_golden.R` first.")
}

es_gold <- readRDS(gold_path)
cat("Loaded golden reference: ", dim(assay(es_gold, "es")), "\n")

# ── Data preprocessing ────────────────────────────────────────────────────────
sce <- TENxPBMCData(dataset = "pbmc68k")
sce <- sce[, 1:8000]
is_mito <- grepl("^MT-", rowData(sce)$Symbol_TENx)
sce <- quickPerCellQC(sce, subsets = list(Mito = is_mito),
                      sub.fields = "subsets_Mito_percent")
cntxgene <- rowSums(assays(sce)$counts) + 1
sce <- sce[cntxgene >= 100, ]
sce <- computeLibraryFactors(sce)
sce <- logNormCounts(sce)

fname <- file.path(system.file("extdata", package = "GSVAdata"),
                   "pbmc_cell_type_gene_set_signatures.gmt.gz")
gsets <- readGMT(fname)

gsvaAnnotation(sce) <- ENSEMBLIdentifier("org.Hs.eg.db")
gsvapar <- gsvaParam(sce, gsets, verbose = FALSE)
gsvarownorm <- gsvaRowNorm(gsvapar, verbose = FALSE)
gsvaranks <- gsvaColRanks(gsvarownorm, verbose = FALSE)

# ── Compute scores ───────────────────────────────────────────────────────────
cat("Computing CPU scores... ")
es_cpu <- gsvaColScores(gsvaranks, verbose = FALSE, device = "cpu")
cat("done\n")

gpu_available <- GSVA:::.gsva_cuda_available()
if (!gpu_available) {
  stop("GPU not available (CUDA support not detected).")
}

cat("Computing GPU scores... ")
es_gpu <- gsvaColScores(gsvaranks, verbose = FALSE, device = "gpu")
cat("done\n")

# ── Pairwise comparisons ─────────────────────────────────────────────────────
tolerance <- 1e-6
all_pass <- TRUE

cmp_gold_cpu <- all.equal(assay(es_gold, "es"), assay(es_cpu, "es"), tolerance = tolerance)
cmp_gold_gpu <- all.equal(assay(es_gold, "es"), assay(es_gpu, "es"), tolerance = tolerance)
cmp_cpu_gpu  <- all.equal(assay(es_cpu, "es"),  assay(es_gpu, "es"), tolerance = tolerance)

checks <- list(
  `Golden vs CPU` = cmp_gold_cpu,
  `Golden vs GPU` = cmp_gold_gpu,
  `CPU vs GPU`    = cmp_cpu_gpu
)

cat("\n")
cat("================================================================\n")
cat("  verify_golden.R — Correctness Checks (tolerance = ", tolerance, ")\n", sep = "")
cat("================================================================\n")

for (nm in names(checks)) {
  res <- checks[[nm]]
  ok <- isTRUE(res)
  all_pass <- all_pass && ok
  status <- if (ok) "OK" else as.character(res)
  cat(sprintf("  %-16s  %s\n", nm, status))
}

cat("================================================================\n")

if (!all_pass) {
  cat("\nFAIL: One or more comparisons did not match.\n")
  quit(status = 1, save = "no")
}

cat("\nAll checks passed.\n")
