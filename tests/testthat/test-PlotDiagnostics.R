## Tests for LoadDiagnostics() and PlotTrainingDiagnostics()

library(testthat)

# ---------------------------------------------------------------------------
# LoadDiagnostics()
# ---------------------------------------------------------------------------

test_that("LoadDiagnostics errors when file does not exist", {
  expect_error(
    LoadDiagnostics("/nonexistent/path/diag.csv"),
    "not found"
  )
})

test_that("LoadDiagnostics returns a data.frame", {
  tmp <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp), add = TRUE)
  utils::write.csv(
    data.frame(epoch = 1:5, loss = seq(1, 0.5, length.out = 5)),
    tmp, row.names = FALSE
  )
  result <- LoadDiagnostics(tmp)
  expect_is(result, "data.frame")
  expect_equal(nrow(result), 5L)
  expect_true("epoch" %in% colnames(result))
  expect_true("loss"  %in% colnames(result))
})

# ---------------------------------------------------------------------------
# PlotTrainingDiagnostics()
# ---------------------------------------------------------------------------

test_that("PlotTrainingDiagnostics accepts a file path and returns ggplot", {
  skip_if_not_installed("ggplot2")
  tmp <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp), add = TRUE)
  utils::write.csv(
    data.frame(epoch = 1:10, loss = 10:1),
    tmp, row.names = FALSE
  )
  p <- PlotTrainingDiagnostics(tmp)
  expect_s3_class(p, "ggplot")
})

test_that("PlotTrainingDiagnostics accepts a data.frame and returns ggplot", {
  skip_if_not_installed("ggplot2")
  diagData <- data.frame(epoch = 1:10, loss = 10:1)
  p <- PlotTrainingDiagnostics(diagData)
  expect_s3_class(p, "ggplot")
})

test_that("PlotTrainingDiagnostics handles multiple loss columns", {
  skip_if_not_installed("ggplot2")
  diagData <- data.frame(
    epoch   = 1:10,
    loss    = 10:1,
    kl_loss = seq(5, 0.5, length.out = 10)
  )
  p <- PlotTrainingDiagnostics(diagData)
  expect_s3_class(p, "ggplot")
  # The reshaped data should have 20 rows (10 epochs × 2 loss types)
  built <- ggplot2::ggplot_build(p)
  expect_equal(nrow(built$data[[1]]), 20L)
})

test_that("PlotTrainingDiagnostics errors on missing 'epoch' column", {
  diagData <- data.frame(step = 1:5, loss = 5:1)
  expect_error(PlotTrainingDiagnostics(diagData), "epoch")
})

test_that("PlotTrainingDiagnostics errors on data.frame with only epoch", {
  diagData <- data.frame(epoch = 1:5)
  expect_error(PlotTrainingDiagnostics(diagData), "No loss columns")
})

test_that("PlotTrainingDiagnostics errors on invalid input type", {
  expect_error(PlotTrainingDiagnostics(42L), "file path")
})

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

skip_if_not_installed <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    testthat::skip(paste("Package", pkg, "is not available"))
  }
}
