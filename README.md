# GSVA: gene set variation analysis for microarray and RNA-seq data

This is a fork of the [rcastelo/GSVA](https://github.com/rcastelo/GSVA) repository. It provides an **experimental**
version of the R package accelerated through the use of the [CUDA toolkit](https://docs.nvidia.com/cuda/).

🧪 **Warning**: This code is under active development and is not yet recommended for production use.

Most modifications are located within the `src/cuda` directory.

## Installation

To build and run this package, you will need:

* A Linux system with an NVIDIA GPU;
* A recent version of the `nvcc` compiler (version 13 was used for development).

You can install the package from source using [remotes](https://github.com/r-lib/remotes):

```r
if (!require("remotes")) install.packages("remotes")
remotes::install_github("sales-lab/GSVA")
```

## Questions, bug reports and issues

Please submit feature requests, bug reports, or general issues via the
[GitHub issues tab](https://github.com/sales-lab/GSVA/issues) at the top of this page.

## Funding

Supported by the [Chan Zuckerberg Initiative](https://chanzuckerberg.com) through the
[Essential Open Source Software for Science (EOSS)](https://czi.co/EOSS)
program (project "GPU-accelerated computing in Bioconductor" ,`EOSS6-0000000644`).
