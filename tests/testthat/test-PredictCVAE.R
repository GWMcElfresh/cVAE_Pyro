## Tests for PredictCVAE()
##
## Tests at three levels:
##   1. .validateNewdata() – pure R, no external deps
##   2. PredictCVAE() argument checks – no Python needed
##   3. Full R→Python→R round-trip (requires Python + Pyro)

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

skip_if_not_installed <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    testthat::skip(paste("Package", pkg, "is not available"))
  }
}

# Build a minimal fake model directory so .validateNewdata() has metadata
.make_fake_model_dir <- function(gene_names   = paste0("Gene", 1:10),
                                  cond_vars    = c("treatment"),
                                  tmpdir       = tempdir()) {
  modelDir <- file.path(tmpdir, "fake_model")
  dir.create(modelDir, showWarnings = FALSE)

  # Minimal model metadata JSON
  meta <- list(
    input_dim     = length(gene_names),
    cond_dim      = 2L,
    hidden_dims   = list(16L, 8L),
    latent_dim    = 4L,
    gene_names    = gene_names,
    expr_metadata = list(
      mean       = as.list(rep(0, length(gene_names))),
      std        = as.list(rep(1, length(gene_names))),
      gene_names = gene_names,
      cell_names = list()
    ),
    cond_metadata = stats::setNames(
      lapply(cond_vars, function(v)
        list(type = "categorical", categories = list("ctrl", "treated"))),
      cond_vars
    )
  )
  jsonlite::write_json(meta, file.path(modelDir, "model_metadata.json"),
                       auto_unbox = TRUE, pretty = TRUE)

  # Create empty placeholder files for the other required files
  file.create(file.path(modelDir, "cvae_model.pt"))
  file.create(file.path(modelDir, "pyro_params.pt"))

  modelDir
}

# ---------------------------------------------------------------------------
# .validateNewdata() – pure R unit tests
# ---------------------------------------------------------------------------

test_that(".validateNewdata errors when newdata is not a list", {
  expect_error(
    cVAEPyro:::.validateNewdata("not a list", tempdir()),
    "named list"
  )
})

test_that(".validateNewdata errors when both components are NULL", {
  expect_error(
    cVAEPyro:::.validateNewdata(
      list(geneExpression = NULL, conditionals = NULL),
      tempdir()
    ),
    "at least one"
  )
})

test_that(".validateNewdata warns on unknown list keys", {
  geneNames <- paste0("Gene", 1:10)
  modelDir  <- .make_fake_model_dir(gene_names = geneNames)
  expr      <- matrix(1, nrow = 3, ncol = 10,
                       dimnames = list(NULL, geneNames))
  expect_warning(
    cVAEPyro:::.validateNewdata(
      list(geneExpression = expr, unknownKey = 42),
      modelDir
    ),
    "ignored"
  )
})

test_that(".validateNewdata errors when required genes are missing", {
  geneNames <- paste0("Gene", 1:10)
  modelDir  <- .make_fake_model_dir(gene_names = geneNames)
  # Only provide 5 of 10 genes
  expr      <- matrix(1, nrow = 3, ncol = 5,
                       dimnames = list(NULL, paste0("Gene", 1:5)))
  expect_error(
    cVAEPyro:::.validateNewdata(
      list(geneExpression = expr),
      modelDir
    ),
    "missing"
  )
})

test_that(".validateNewdata errors when conditional columns are missing", {
  geneNames <- paste0("Gene", 1:10)
  modelDir  <- .make_fake_model_dir(gene_names = geneNames,
                                    cond_vars  = "treatment")
  expr <- matrix(1, nrow = 3, ncol = 10,
                  dimnames = list(NULL, geneNames))
  # Pass conditionals without the required 'treatment' column
  cond <- data.frame(wrong_col = c("A", "B", "C"))
  expect_error(
    cVAEPyro:::.validateNewdata(
      list(geneExpression = expr, conditionals = cond),
      modelDir
    ),
    "missing"
  )
})

test_that(".validateNewdata passes with conformant newdata", {
  geneNames <- paste0("Gene", 1:10)
  modelDir  <- .make_fake_model_dir(gene_names = geneNames,
                                    cond_vars  = "treatment")
  expr <- matrix(1, nrow = 3, ncol = 10,
                  dimnames = list(NULL, geneNames))
  cond <- data.frame(treatment = c("ctrl", "treated", "ctrl"),
                     stringsAsFactors = FALSE)
  expect_silent(
    cVAEPyro:::.validateNewdata(
      list(geneExpression = expr, conditionals = cond),
      modelDir
    )
  )
})

# ---------------------------------------------------------------------------
# PredictCVAE() – argument validation (no Python needed)
# ---------------------------------------------------------------------------

test_that("PredictCVAE errors when modelDir does not exist", {
  expect_error(
    PredictCVAE(
      modelDir = "/nonexistent/path",
      newdata  = list(geneExpression = matrix(1, 1, 1))
    ),
    "does not exist"
  )
})

test_that("PredictCVAE errors when required model files are missing", {
  tmpDir <- file.path(tempdir(), "empty_model")
  dir.create(tmpDir, showWarnings = FALSE)
  expect_error(
    PredictCVAE(
      modelDir = tmpDir,
      newdata  = list(geneExpression = matrix(1, 1, 1))
    ),
    "Required model files"
  )
})

test_that("PredictCVAE errors on nSamples < 1", {
  geneNames <- paste0("Gene", 1:10)
  modelDir  <- .make_fake_model_dir(gene_names = geneNames)
  expr      <- matrix(1, nrow = 3, ncol = 10,
                       dimnames = list(NULL, geneNames))
  expect_error(
    PredictCVAE(
      modelDir  = modelDir,
      newdata   = list(geneExpression = expr),
      nSamples  = 0L
    ),
    regexp = "nSamples"
  )
})

# ---------------------------------------------------------------------------
# Full R→Python→R round-trip (requires Python + Pyro)
# ---------------------------------------------------------------------------

test_that("PredictCVAE round-trip: train then predict (end-to-end)", {
  skip_on_cran()
  skip_if_not(.python_has_pyro(), "Python + pyro-ppl not available")
  skip_if_not_installed("Seurat")

  set.seed(99L)
  n_cells <- 40L
  n_genes <- 15L
  tmpDir  <- file.path(tempdir(), "predict_e2e")

  # Build mock Seurat object
  gn  <- paste0("Gene", seq_len(n_genes))
  cn  <- paste0("Cell", seq_len(n_cells))
  expr <- Matrix::rsparsematrix(n_genes, n_cells, density = 0.5)
  rownames(expr) <- gn
  colnames(expr) <- cn
  meta <- data.frame(
    row.names = cn,
    batch     = sample(c("A", "B"), n_cells, replace = TRUE),
    stringsAsFactors = FALSE
  )
  assayObj <- list(data = expr, row.names = gn)
  class(assayObj) <- "MockAssay"
  sobj <- list(assays = list(RNA = assayObj), meta.data = meta,
               cell_names = cn, default_assay = "RNA")
  class(sobj) <- c("Seurat", "MockSeurat")

  # Train
  trainDir <- file.path(tmpDir, "model")
  TrainCVAE(
    seuratObject    = sobj,
    conditionalVars = "batch",
    latentDim       = 3L,
    hiddenDims      = c(8L),
    nEpochs         = 3L,
    batchSize       = 20L,
    outputDir       = trainDir
  )

  # Build newdata with permuted conditional
  newExpr <- Matrix::t(expr)[, , drop = FALSE]  # cells x genes
  newCond <- data.frame(
    row.names = cn,
    batch     = rep("A", n_cells),   # permute all cells to batch A
    stringsAsFactors = FALSE
  )

  predDir <- file.path(tmpDir, "preds")
  preds   <- PredictCVAE(
    modelDir  = trainDir,
    newdata   = list(geneExpression = newExpr, conditionals = newCond),
    nSamples  = 2L,
    outputDir = predDir
  )

  # Shape checks
  expect_is(preds, "data.frame")
  expect_equal(nrow(preds), n_cells)
  expect_equal(ncol(preds), n_genes)

  # Output files
  expect_true(file.exists(file.path(predDir, "predictions.csv")))
  expect_true(file.exists(file.path(predDir, "latent_mean.csv")))
  expect_true(file.exists(file.path(predDir, "latent_std.csv")))
})

test_that("PredictCVAE prior-sampling (no expression input) works", {
  skip_on_cran()
  skip_if_not(.python_has_pyro(), "Python + pyro-ppl not available")
  skip_if_not_installed("Seurat")

  # Re-use the model trained in the test above if present; otherwise skip
  trainDir <- file.path(tempdir(), "predict_e2e", "model")
  skip_if(!dir.exists(trainDir), "Prior-sampling test requires a pre-trained model")

  meta_path <- file.path(trainDir, "model_metadata.json")
  meta      <- jsonlite::read_json(meta_path, simplifyVector = TRUE)
  n_cells   <- 5L
  cn        <- paste0("NewCell", seq_len(n_cells))
  newCond   <- data.frame(
    row.names = cn,
    batch     = rep("B", n_cells),
    stringsAsFactors = FALSE
  )

  predDir <- file.path(tempdir(), "predict_prior")
  preds   <- PredictCVAE(
    modelDir  = trainDir,
    newdata   = list(conditionals = newCond),
    nSamples  = 3L,
    outputDir = predDir
  )

  expect_equal(nrow(preds), n_cells)
  expect_equal(ncol(preds), length(meta$gene_names))
})
