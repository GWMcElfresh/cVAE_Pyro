#' Perform a forward pass through a trained cVAE (counterfactual prediction)
#'
#' Runs a generative forward pass through a previously trained cVAE model.
#' Accepts permuted gene expression and/or conditional variables to produce
#' counterfactual predictions.
#'
#' @param modelDir Character scalar. Path to the directory produced by
#'   \code{\link{TrainCVAE}} (must contain \code{cvae_model.pt},
#'   \code{pyro_params.pt}, and \code{model_metadata.json}).
#' @param newdata A named list with one or both of:
#'   \describe{
#'     \item{\code{geneExpression}}{A sparse or dense numeric matrix with
#'       dimensions cells × genes. Column names must match the gene names used
#'       during training. If \code{NULL}, the model samples from the prior.}
#'     \item{\code{conditionals}}{A \code{data.frame} of cells ×
#'       conditional-variable columns. Column names and factor levels must
#'       match those seen during training. If \code{NULL} and the model was
#'       trained without conditionals this is acceptable; otherwise an error
#'       is raised.}
#'   }
#' @param nSamples Positive integer. Number of latent samples to draw per
#'   cell when computing predictions (default: \code{1L}).
#' @param outputDir Character scalar. Directory in which to save
#'   \code{predictions.csv}. Created if absent. Default: \code{tempdir()}.
#' @param pythonPath Character scalar. Path to the Python 3 executable
#'   (default: \code{"python3"}).
#'
#' @return A \code{data.frame} of predicted (reconstructed) gene expression
#'   values, with rows corresponding to cells and columns to genes.
#'
#' @details
#' The function follows the same R-invokes-Python pattern as
#' \code{\link{TrainCVAE}}: it reads \code{inst/python/predict_cvae.py},
#' appends a concrete function call, writes to a temporary file, and calls it
#' via \code{system2}.
#'
#' Conformance checking ensures that:
#' \itemize{
#'   \item \code{newdata} is a list.
#'   \item At least one of \code{geneExpression} or \code{conditionals} is
#'     non-\code{NULL}.
#'   \item Gene names in \code{newdata$geneExpression} match the training gene
#'     set (read from \code{model_metadata.json}).
#'   \item Conditional column names match those used during training.
#' }
#'
#' @examples
#' \dontrun{
#' # Permute the cell-type conditional while keeping gene expression fixed
#' newdata <- list(
#'   geneExpression = my_sparse_matrix,
#'   conditionals   = data.frame(
#'     cell_type = rep("T cell", ncol(my_sparse_matrix)),
#'     batch     = original_batch_labels
#'   )
#' )
#' preds <- PredictCVAE(
#'   modelDir  = "results/model1",
#'   newdata   = newdata,
#'   nSamples  = 10L,
#'   outputDir = "results/predictions"
#' )
#' }
#'
#' @export
PredictCVAE <- function(modelDir,
                         newdata,
                         nSamples   = 1L,
                         outputDir  = tempdir(),
                         pythonPath = "python3") {

  # ---- Validate modelDir -----------------------------------------------------
  if (!dir.exists(modelDir)) {
    stop("'modelDir' does not exist: ", modelDir)
  }
  requiredFiles <- c("cvae_model.pt", "pyro_params.pt", "model_metadata.json")
  missing <- requiredFiles[!file.exists(file.path(modelDir, requiredFiles))]
  if (length(missing) > 0L) {
    stop("Required model files not found in '", modelDir, "': ",
         paste(missing, collapse = ", "))
  }

  # ---- Validate newdata ------------------------------------------------------
  .validateNewdata(newdata, modelDir)

  nSamples <- as.integer(nSamples)
  stopifnot(nSamples >= 1L)

  dir.create(outputDir, showWarnings = FALSE, recursive = TRUE)

  # ---- Persist newdata to disk -----------------------------------------------
  newExprPath <- ""
  if (!is.null(newdata$geneExpression)) {
    newExprPath <- file.path(outputDir, "new_gene_expression.csv")
    .writeSparseMatrixCSV(newdata$geneExpression, newExprPath)
  }

  newCondPath <- ""
  if (!is.null(newdata$conditionals)) {
    newCondPath <- file.path(outputDir, "new_conditionals.csv")
    utils::write.csv(newdata$conditionals, newCondPath, row.names = TRUE)
  }

  # ---- Invoke Python script --------------------------------------------------
  predictScript <- system.file("python", "predict_cvae.py", package = "cVAEPyro")
  if (nchar(predictScript) == 0L) {
    stop("Could not locate 'predict_cvae.py'. Is the package installed correctly?")
  }

  args <- list(
    model_dir  = modelDir,
    expr_path  = newExprPath,
    cond_path  = newCondPath,
    output_dir = outputDir,
    n_samples  = nSamples
  )

  .invokePythonScript(pythonPath, predictScript, "predict_cvae", args)

  # ---- Read predictions ------------------------------------------------------
  predPath <- file.path(outputDir, "predictions.csv")
  if (!file.exists(predPath)) {
    stop("Prediction output not found: ", predPath)
  }

  utils::read.csv(predPath, row.names = 1L, check.names = FALSE)
}
