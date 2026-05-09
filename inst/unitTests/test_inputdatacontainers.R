source(system.file("unitTests", "test_helpers.R", package = "GSVA"))

test_inputdatacontainers <- function() {
    message("Running unit tests for input data containers")

    p <- 10 ## number of genes
    n <- 30 ## number of samples
    nGrp1 <- 15 ## number of samples in group 1
    nGrp2 <- n - nGrp1 ## number of samples in group 2

    ## consider three disjoint gene sets
    gsets <- list(set1=paste("g", 1:3, sep=""),
                  set2=paste("g", 4:6, sep=""),
                  set3=paste("g", 7:10, sep=""))

    ## sample data from a normal distribution with mean 0 and st.dev. 1
    ## seeding the random number generator for the purpose of this test
    set.seed(123)
    y <- matrix(rnorm(n*p), nrow=p, ncol=n,
                dimnames=list(paste("g", 1:p, sep="") , paste("s", 1:n, sep="")))

    ## genes in set1 are expressed at higher levels in the last 'nGrp1+1' to 'n' samples
    y[gsets$set1, (nGrp1+1):n] <- y[gsets$set1, (nGrp1+1):n] + 2

    ## estimate GSVA enrichment scores with input as a matrix
    es.mat <- gsva(gsvaParam(y, gsets, verbose = FALSE), verbose = FALSE, device = "cpu")
    gsets.mat <- geneSets(es.mat)

    .run_with_gpu({
        es.mat_gpu <- gsva(gsvaParam(y, gsets, verbose = FALSE), verbose = FALSE, device = "gpu")
        .check_gpu_equiv(es.mat, es.mat_gpu)
    })

    library(cli)
    out <- cli_fmt(gsvaParam(y, gsets, assay="dummy"))
    checkTrue(endsWith(out, "argument assay='dummy' ignored since input argument 'exprData' has no assay names."))

    ## estimate GSVA enrichment scores with input as an ExpressionSet object
    suppressPackageStartupMessages(library(Biobase))

    y2 <- y
    rownames(y2) <- NULL
    eset <- ExpressionSet(assayData=y2,
                          phenoData=as(data.frame(dummy=1:ncol(y),
                                                  row.names=colnames(y)),
                                       "AnnotatedDataFrame"),
                          featureData=as(data.frame(dummy=1:nrow(y),
                                                    row.names=rownames(y)),
                                         "AnnotatedDataFrame"))
    es.eset <- gsva(gsvaParam(eset, gsets, verbose = FALSE), verbose = FALSE, device = "cpu")
    gsets.eSet <- geneSets(es.eset)

    .run_with_gpu({
        es.eset_gpu <- gsva(gsvaParam(eset, gsets, verbose = FALSE), verbose = FALSE, device = "gpu")
        .check_gpu_equiv(es.eset, es.eset_gpu)
    })
    es.mat2 <- es.mat
    checkEqualsNumeric(es.mat2, exprs(es.eset))
    checkTrue(identical(gsets.mat, gsets.eSet))

    ## estimate GSVA enrichment scores with input as a SummarizedExperiment object
    suppressPackageStartupMessages({
        library(S4Vectors)
        library(SummarizedExperiment)
    })

    se <- SummarizedExperiment(assay=list(counts=y2),
                               rowData=DataFrame(data.frame(dummy=1:nrow(y),
                                                            row.names=rownames(y))),
                               colData=DataFrame(data.frame(dummy=1:ncol(y),
                                                            row.names=colnames(y))))
    gsvapar <- gsvaParam(se, gsets, verbose=FALSE)
    es.se <- gsva(gsvapar, verbose=FALSE, device="cpu")
    gsets.se <- geneSets(es.se)

    .run_with_gpu({
        es.se_gpu <- gsva(gsvapar, verbose=FALSE, device="gpu")
        .check_gpu_equiv(es.se, es.se_gpu)
    })

    checkEqualsNumeric(es.mat2, assay(es.se))
    checkTrue(identical(gsets.mat, gsets.se))

    out <- cli_fmt(gsvaParam(se, gsets))
    checkTrue(endsWith(out, "No assay name provided; using default assay 'counts'"))
    checkException(gsvaParam(se, gsets, assay="dummy"))

    gsvarownr <- gsvaRowNorm(gsvapar, dropExistingAssays=TRUE, verbose=FALSE)
    gsvaranks <- gsvaColRanks(gsvarownr, dropExistingAssays=TRUE, verbose=FALSE)
    es.se2 <- gsvaColScores(gsvaranks, verbose=FALSE)
    checkEqualsNumeric(assay(es.se), assay(es.se2), tolerance = 1e-6)

    ## estimate GSVA enrichment scores with input as a dgCMatrix object
    suppressPackageStartupMessages(library(Matrix))

    yMat <- Matrix(y, sparse=TRUE)

    ## check show() method for a gsvaParam object
    param <- gsvaParam(yMat, gsets, sparse=FALSE, checkNA="auto", verbose=FALSE)
    out <- capture.output(show(param))
    checkTrue(length(out) > 0 && sum(nchar(out)) > 0,
	      "gsvaParam object show method output is empty")
    out <- capture.output(details(param))
    checkTrue(length(out) > 0 && sum(nchar(out)) > 0,
	      "gsvaParam object details method output is empty")

    es.dgCMat <- gsva(param, verbose=FALSE, device="cpu")
    gsets.dgCMat <- geneSets(es.dgCMat)

    .run_with_gpu({
        es.dgCMat_gpu <- gsva(param, verbose=FALSE, device="gpu")
        .check_gpu_equiv(es.dgCMat, es.dgCMat_gpu)
    })

    checkEqualsNumeric(es.mat2, es.dgCMat)
    checkTrue(identical(gsets.mat, gsets.dgCMat))

    ## testing geneIdsToGeneSetCollection()
    suppressPackageStartupMessages(library(GSEABase))

    suppressWarnings(gsc <- geneIdsToGeneSetCollection(gsets.dgCMat,
						       geneIdType="whatever"))
    checkTrue(is(gsc, "GeneSetCollection"))

    sp <- 0.5 * prod(dim(y))
    ysp <- as.vector(y)
    ysp[sample(length(ysp), sp)] <- 0
    yMatSp <- Matrix(ysp, nrow=nrow(yMat), ncol=ncol(yMat),
		     dimnames=dimnames(yMat), sparse=TRUE)
    paramSp <- gsvaParam(yMatSp, gsets, verbose=FALSE)
    es.dgCMatSp <- gsva(paramSp, verbose=FALSE, device="cpu")
    
    .run_with_gpu({
        es.dgCMatSp_gpu <- gsva(paramSp, verbose=FALSE, device="gpu")
        .check_gpu_equiv(es.dgCMatSp, es.dgCMatSp_gpu)
    })
    
    ## estimate GSVA enrichment scores with input as a SingleCellExperiment object
    suppressPackageStartupMessages(library(SingleCellExperiment))

    sce <- SingleCellExperiment(assays=list(logcounts=yMatSp),
				rowData=DataFrame(data.frame(dummy=1:nrow(y),
							     row.names=rownames(y))),
				colData=DataFrame(data.frame(dummy=1:ncol(y),
							     row.names=colnames(y))))
    gsvaAnnotation(sce) <- SymbolIdentifier("org.Hs.eg.db")
    out <- gsvaAnnotation(sce)
    param <- gsvaParam(sce, gsets, verbose=FALSE)
    show(param)
    es.sce <- gsva(param, verbose=FALSE, device="cpu")
    gsets.sce <- geneSets(es.sce)
    
    .run_with_gpu({
        es.sce_gpu <- gsva(param, verbose=FALSE, device="gpu")
        .check_gpu_equiv(es.sce, es.sce_gpu)
    })

    checkEqualsNumeric(es.dgCMatSp, assay(es.sce))
    checkTrue(identical(gsets.mat, gsets.sce))

    gsets.ov.list <- computeGeneSetsOverlap(gsets.sce, rownames(sce))
    gsets.ov.gsc <- computeGeneSetsOverlap(gsc, rownames(sce))
    checkTrue(identical(gsets.ov.list, gsets.ov.gsc))
}
