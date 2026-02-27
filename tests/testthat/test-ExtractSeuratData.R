## Tests for ExtractSeuratData()
##
## These tests use a minimal mock Seurat-like object so the package can be
## tested without a full Seurat installation.  The mock exposes only the
## slots / methods that ExtractSeuratData() accesses.

library(testthat)
library(Matrix)

# ---------------------------------------------------------------------------
# Helpers – build a tiny mock Seurat object
# ---------------------------------------------------------------------------

# A bare-bones stand-in for a Seurat object.
# We only need:
#   - is(obj, "Seurat") to return TRUE
#   - DefaultAssay(obj) to return an assay name
#   - obj@assays to list assay names
#   - obj[[assay]] to return an assay sub-object
#   - rownames(assay_obj) for gene names
#   - GetAssayData(assay_obj, layer) to return a genes × cells sparse matrix
#   - colnames(obj) for cell barcodes
#   - obj@meta.data for metadata

.make_mock_seurat <- function(n_cells = 20L, n_genes = 30L,
                               n_conds = 2L) {
  set.seed(42L)
  genenames  <- paste0("Gene",  seq_len(n_genes))
  cellnames  <- paste0("Cell",  seq_len(n_cells))

  # sparse genes × cells matrix
  expr <- Matrix::rsparsematrix(n_genes, n_cells, density = 0.4)
  rownames(expr) <- genenames
  colnames(expr) <- cellnames

  # metadata: one continuous, one categorical conditional
  meta <- data.frame(
    row.names   = cellnames,
    continuous1 = stats::rnorm(n_cells),
    category1   = sample(c("A", "B", "C"), n_cells, replace = TRUE),
    stringsAsFactors = FALSE
  )

  # Minimal mock using R5/reference classes would be complex; use a list +
  # S3 dispatch trick instead.  We register an S3 "is" method by abusing
  # the methods package's is() generic via a class attribute.
  assayObj <- list(
    data      = expr,
    row.names = genenames
  )
  class(assayObj) <- "MockAssay"

  obj <- list(
    assays    = list(RNA = assayObj),
    meta.data = meta,
    cell_names = cellnames,
    default_assay = "RNA"
  )
  class(obj) <- c("Seurat", "MockSeurat")
  obj
}

# Register stub generics so ExtractSeuratData() resolves correctly
# when Seurat is NOT installed.
if (!requireNamespace("Seurat", quietly = TRUE)) {
  # methods::is()  works on the class attribute – no stub needed.
  # Seurat::DefaultAssay / GetAssayData are only called inside functions
  # guarded by skip_if_not_installed("Seurat") below.
}

# ---------------------------------------------------------------------------
# We test .filterCells / .filterGenes directly (they are in the package env)
# ---------------------------------------------------------------------------

test_that(".filterCells applies whitelist correctly", {
  cells <- paste0("Cell", 1:10)
  expect_equal(
    cVAEPyro:::.filterCells(cells, paste0("Cell", c(1, 3, 5)), NULL),
    paste0("Cell", c(1, 3, 5))
  )
})

test_that(".filterCells applies blacklist correctly", {
  cells <- paste0("Cell", 1:5)
  expect_equal(
    cVAEPyro:::.filterCells(cells, NULL, paste0("Cell", c(2, 4))),
    paste0("Cell", c(1, 3, 5))
  )
})

test_that(".filterCells applies both whitelist and blacklist", {
  cells <- paste0("Cell", 1:10)
  result <- cVAEPyro:::.filterCells(
    cells,
    paste0("Cell", 1:5),   # whitelist
    paste0("Cell", c(2, 4)) # blacklist
  )
  expect_equal(result, paste0("Cell", c(1, 3, 5)))
})

test_that(".filterCells returns all cells when both filters are NULL", {
  cells <- paste0("Cell", 1:5)
  expect_equal(cVAEPyro:::.filterCells(cells, NULL, NULL), cells)
})

test_that(".filterGenes applies whitelist correctly", {
  genes <- paste0("Gene", 1:10)
  expect_equal(
    cVAEPyro:::.filterGenes(genes, paste0("Gene", c(1, 5, 10)), NULL),
    paste0("Gene", c(1, 5, 10))
  )
})

test_that(".filterGenes applies blacklist correctly", {
  genes <- paste0("Gene", 1:5)
  expect_equal(
    cVAEPyro:::.filterGenes(genes, NULL, paste0("Gene", c(1, 3))),
    paste0("Gene", c(2, 4, 5))
  )
})

# ---------------------------------------------------------------------------
# ExtractSeuratData() – integration-style tests using mock
# ---------------------------------------------------------------------------

# Only run these if Seurat is installed (the function calls Seurat::*)
skip_if_not_installed <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    testthat::skip(paste("Package", pkg, "is not available"))
  }
}

test_that("ExtractSeuratData errors on non-Seurat input", {
  expect_error(
    ExtractSeuratData(list()),
    "Seurat object"
  )
})

test_that("ExtractSeuratData errors on missing conditional variable", {
  skip_if_not_installed("Seurat")

  seurat_obj <- .make_mock_seurat()
  expect_error(
    ExtractSeuratData(seurat_obj,
                       conditionalVars = c("continuous1", "MISSING_VAR")),
    "not found in metadata"
  )
})

test_that("ExtractSeuratData returns correct structure with Seurat", {
  skip_if_not_installed("Seurat")

  seurat_obj <- .make_mock_seurat()
  result <- ExtractSeuratData(
    seurat_obj,
    assay           = "RNA",
    layer           = "data",
    conditionalVars = c("continuous1", "category1")
  )

  expect_named(result, c("geneExpression", "conditionals"))
  expect_s4_class(result$geneExpression, "dgCMatrix")
  expect_equal(ncol(result$geneExpression), 30L)   # genes
  expect_equal(nrow(result$geneExpression), 20L)   # cells
  expect_is(result$conditionals, "data.frame")
  expect_equal(colnames(result$conditionals), c("continuous1", "category1"))
})

test_that("ExtractSeuratData respects gene whitelist", {
  skip_if_not_installed("Seurat")

  seurat_obj  <- .make_mock_seurat()
  geneSubset  <- paste0("Gene", 1:5)
  result <- ExtractSeuratData(seurat_obj, geneWhitelist = geneSubset)
  expect_equal(ncol(result$geneExpression), 5L)
  expect_equal(sort(colnames(result$geneExpression)), sort(geneSubset))
})

test_that("ExtractSeuratData respects cell blacklist", {
  skip_if_not_installed("Seurat")

  seurat_obj <- .make_mock_seurat()
  remove     <- paste0("Cell", 1:5)
  result     <- ExtractSeuratData(seurat_obj, cellBlacklist = remove)
  expect_equal(nrow(result$geneExpression), 15L)
  expect_false(any(rownames(result$geneExpression) %in% remove))
})

test_that("ExtractSeuratData returns NULL conditionals when not requested", {
  skip_if_not_installed("Seurat")

  seurat_obj <- .make_mock_seurat()
  result     <- ExtractSeuratData(seurat_obj)
  expect_null(result$conditionals)
})
