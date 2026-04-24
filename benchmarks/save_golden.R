suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(TENxPBMCData)
  library(scuttle)
  library(GSVA)
  library(GSEABase)
})

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

es <- gsvaColScores(gsvaranks, verbose = FALSE, device = "cpu")
golden_dir <- "benchmarks/golden"
dir.create(golden_dir, showWarnings = FALSE, recursive = TRUE)
saveRDS(es, file = file.path(golden_dir, "reference.rds"))
cat("Reference saved to benchmarks/golden/reference.rds\n")
