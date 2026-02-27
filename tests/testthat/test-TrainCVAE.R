## Tests for TrainCVAE() and the R→Python→R handoff
##
## Full end-to-end tests require Python 3 with pyro-ppl and torch.  Tests are
## guarded with skip_if() checks so they degrade gracefully when those
## dependencies are absent (e.g. in lightweight R-only CI jobs).

library(testthat)
library(Matrix)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

.python_has_pyro <- function(python = "python3") {
  tryCatch({
    res <- system2(python,
                   args   = c("-c", shQuote("import pyro; import torch")),
                   stdout = TRUE, stderr = TRUE)
    status <- attr(res, "status")
    is.null(status) || status == 0L
  }, error = function(e) FALSE)
}

.make_tiny_train_data <- function(n_cells = 30L, n_genes = 20L,
                                   tmpdir = tempdir()) {
  set.seed(1L)
  expr <- matrix(stats::rpois(n_cells * n_genes, lambda = 2),
                 nrow = n_cells, ncol = n_genes)
  rownames(expr) <- paste0("Cell",  seq_len(n_cells))
  colnames(expr) <- paste0("Gene",  seq_len(n_genes))

  cond <- data.frame(
    row.names   = rownames(expr),
    treatment   = sample(c("ctrl", "treated"), n_cells, replace = TRUE),
    batch_score = stats::rnorm(n_cells),
    stringsAsFactors = FALSE
  )

  # Write data files the same way TrainCVAE does internally
  exprPath <- file.path(tmpdir, "gene_expression.csv")
  condPath <- file.path(tmpdir, "conditionals.csv")
  utils::write.csv(expr, exprPath, row.names = TRUE)
  utils::write.csv(cond, condPath, row.names = TRUE)

  list(exprPath = exprPath, condPath = condPath,
       n_cells = n_cells, n_genes = n_genes)
}

# ---------------------------------------------------------------------------
# .invokePythonScript – unit test (no Pyro needed)
# ---------------------------------------------------------------------------

test_that(".invokePythonScript runs a trivial Python script correctly", {
  skip_on_cran()
  # Write a minimal standalone Python script that defines a function and
  # check that invocation appends the call correctly.
  tmpDir  <- file.path(tempdir(), "invoke_test")
  dir.create(tmpDir, showWarnings = FALSE)

  # Create a stub script that writes a sentinel file when called
  sentinel <- file.path(tmpDir, "sentinel.txt")
  stubScript <- file.path(tmpDir, "stub.py")
  writeLines(
    c(
      "import os",
      "def stub_func(output_dir='.'):",
      "    with open(os.path.join(output_dir, 'sentinel.txt'), 'w') as f:",
      "        f.write('ok')"
    ),
    stubScript
  )

  args <- list(output_dir = tmpDir)
  cVAEPyro:::.invokePythonScript("python3", stubScript, "stub_func", args)

  expect_true(file.exists(sentinel))
  expect_equal(trimws(readLines(sentinel)), "ok")
})

# ---------------------------------------------------------------------------
# TrainCVAE() – argument validation (no Python needed)
# ---------------------------------------------------------------------------

test_that("TrainCVAE errors on invalid latentDim", {
  skip_if_not_installed("Seurat")
  seurat_obj <- .make_mock_seurat_for_train()
  expect_error(
    TrainCVAE(seurat_obj, latentDim = 0L),
    regexp = "latentDim"
  )
})

test_that("TrainCVAE errors on non-positive nEpochs", {
  skip_if_not_installed("Seurat")
  seurat_obj <- .make_mock_seurat_for_train()
  expect_error(
    TrainCVAE(seurat_obj, nEpochs = 0L),
    regexp = "nEpochs"
  )
})

test_that("TrainCVAE errors on non-positive batchSize", {
  skip_if_not_installed("Seurat")
  seurat_obj <- .make_mock_seurat_for_train()
  expect_error(
    TrainCVAE(seurat_obj, batchSize = 0L),
    regexp = "batchSize"
  )
})

test_that("TrainCVAE errors on non-positive learningRate", {
  skip_if_not_installed("Seurat")
  seurat_obj <- .make_mock_seurat_for_train()
  expect_error(
    TrainCVAE(seurat_obj, learningRate = -0.01),
    regexp = "learningRate"
  )
})

# ---------------------------------------------------------------------------
# End-to-end training test (requires Python + Pyro)
# ---------------------------------------------------------------------------

test_that("TrainCVAE produces expected output files (end-to-end)", {
  skip_on_cran()
  skip_if_not(.python_has_pyro(), "Python + pyro-ppl not available")
  skip_if_not_installed("Seurat")

  tmpDir     <- file.path(tempdir(), "cvae_train_e2e")
  seurat_obj <- .make_mock_seurat_for_train()

  TrainCVAE(
    seuratObject    = seurat_obj,
    conditionalVars = c("treatment", "batch_score"),
    latentDim       = 4L,
    hiddenDims      = c(16L, 8L),
    nEpochs         = 5L,
    batchSize       = 16L,
    outputDir       = tmpDir
  )

  expected <- c(
    "gene_expression.csv",
    "conditionals.csv",
    "data_metadata.json",
    "cvae_model.pt",
    "pyro_params.pt",
    "model_metadata.json",
    "training_diagnostics.csv"
  )
  for (f in expected) {
    expect_true(file.exists(file.path(tmpDir, f)),
                label = paste("Expected output file:", f))
  }

  # Check diagnostics CSV structure
  diag <- utils::read.csv(file.path(tmpDir, "training_diagnostics.csv"))
  expect_true("epoch" %in% colnames(diag))
  expect_true("loss"  %in% colnames(diag))
  expect_equal(nrow(diag), 5L)
})

# ---------------------------------------------------------------------------
# Stub helper (used in validation tests above)
# ---------------------------------------------------------------------------

.make_mock_seurat_for_train <- function(n_cells = 30L, n_genes = 20L) {
  set.seed(7L)
  genenames <- paste0("Gene",  seq_len(n_genes))
  cellnames <- paste0("Cell",  seq_len(n_cells))
  expr <- Matrix::rsparsematrix(n_genes, n_cells, density = 0.5)
  rownames(expr) <- genenames
  colnames(expr) <- cellnames
  meta <- data.frame(
    row.names   = cellnames,
    treatment   = sample(c("ctrl", "treated"), n_cells, replace = TRUE),
    batch_score = stats::rnorm(n_cells),
    stringsAsFactors = FALSE
  )
  assayObj <- list(data = expr, row.names = genenames)
  class(assayObj) <- "MockAssay"
  obj <- list(
    assays = list(RNA = assayObj),
    meta.data = meta,
    cell_names = cellnames,
    default_assay = "RNA"
  )
  class(obj) <- c("Seurat", "MockSeurat")
  obj
}

skip_if_not_installed <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    testthat::skip(paste("Package", pkg, "is not available"))
  }
}
