.gsva_cuda_available <- function() {
    GSVA:::.gsva_cuda_available()
}

.run_with_gpu <- function(expr) {
    if (!.gsva_cuda_available()) return(invisible(NULL))
    expr
}

.check_gpu_equiv <- function(cpu, gpu, tol = 1e-6) {
    if (inherits(cpu, "ExpressionSet")) {
        cpu <- Biobase::exprs(cpu)
    } else if (inherits(cpu, "SummarizedExperiment")) {
        cpu <- SummarizedExperiment::assay(cpu)
    }
    if (inherits(gpu, "ExpressionSet")) {
        gpu <- Biobase::exprs(gpu)
    } else if (inherits(gpu, "SummarizedExperiment")) {
        gpu <- SummarizedExperiment::assay(gpu)
    }
    checkEqualsNumeric(cpu, gpu, tolerance = tol)
}
