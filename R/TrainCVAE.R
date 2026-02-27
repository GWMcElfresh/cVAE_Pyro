#' Train a conditional variational autoencoder on Seurat gene expression data
#'
#' Extracts gene expression and metadata from a Seurat object, writes them to
#' disk, and calls the Pyro cVAE training script via \code{system2}. The
#' trained model, Pyro parameter store, and a loss-curve CSV are all saved to
#' \code{outputDir}.
#'
#' @param seuratObject A Seurat object.
#' @param assay Character scalar. Assay to use. Defaults to
#'   \code{DefaultAssay(seuratObject)}.
#' @param layer Character scalar. Layer to extract (default: \code{"data"}).
#' @param geneWhitelist Character vector of genes to include (\code{NULL} =
#'   all).
#' @param geneBlacklist Character vector of genes to exclude (\code{NULL} =
#'   none).
#' @param cellWhitelist Character vector of cells to include (\code{NULL} =
#'   all).
#' @param cellBlacklist Character vector of cells to exclude (\code{NULL} =
#'   none).
#' @param conditionalVars Character vector of metadata column names to use as
#'   conditioning variables. May be continuous or categorical; the Python script
#'   handles encoding automatically.
#' @param latentDim Positive integer. Dimensionality of the latent space
#'   (default: \code{10L}).
#' @param hiddenDims Integer vector of hidden layer widths (default:
#'   \code{c(128L, 64L)}).
#' @param nEpochs Positive integer. Number of training epochs (default:
#'   \code{100L}).
#' @param learningRate Positive numeric. Learning rate for the Adam optimizer
#'   (default: \code{1e-3}).
#' @param batchSize Positive integer. Mini-batch size (default: \code{128L}).
#' @param outputDir Character scalar. Directory in which to save the model,
#'   Pyro param store, diagnostics CSV, and data metadata. Created if it does
#'   not exist. Default: \code{tempdir()}.
#' @param pythonPath Character scalar. Path to the Python 3 executable
#'   (default: \code{"python3"}).
#'
#' @return Invisibly returns \code{outputDir}.
#'
#' @details
#' The function follows the R-invokes-Python pattern: it reads the template
#' training script bundled in \code{inst/python/train_cvae.py}, appends a
#' concrete function call with the resolved arguments, writes the result to a
#' temporary \code{.py} file, and executes it via \code{system2}.
#'
#' On successful completion \code{outputDir} will contain:
#' \describe{
#'   \item{\code{gene_expression.csv}}{Cells × genes expression matrix.}
#'   \item{\code{conditionals.csv}}{Cells × conditional variables (if any).}
#'   \item{\code{data_metadata.json}}{Gene/cell names and conditional variable
#'     info.}
#'   \item{\code{cvae_model.pt}}{Saved PyTorch model weights.}
#'   \item{\code{pyro_params.pt}}{Saved Pyro parameter store.}
#'   \item{\code{model_metadata.json}}{Architecture and normalisation metadata
#'     needed for prediction.}
#'   \item{\code{training_diagnostics.csv}}{Per-epoch loss values suitable for
#'     plotting with \code{\link{PlotTrainingDiagnostics}}.}
#' }
#'
#' @examples
#' \dontrun{
#' TrainCVAE(
#'   seuratObject    = seurat_obj,
#'   conditionalVars = c("cell_type", "batch"),
#'   latentDim       = 20L,
#'   nEpochs         = 200L,
#'   outputDir       = "results/model1"
#' )
#' }
#'
#' @export
TrainCVAE <- function(seuratObject,
                       assay           = NULL,
                       layer           = "data",
                       geneWhitelist   = NULL,
                       geneBlacklist   = NULL,
                       cellWhitelist   = NULL,
                       cellBlacklist   = NULL,
                       conditionalVars = NULL,
                       latentDim       = 10L,
                       hiddenDims      = c(128L, 64L),
                       nEpochs         = 100L,
                       learningRate    = 1e-3,
                       batchSize       = 128L,
                       outputDir       = tempdir(),
                       pythonPath      = "python3") {

  # ---- Input validation ------------------------------------------------------
  latentDim  <- as.integer(latentDim)
  hiddenDims <- as.integer(hiddenDims)
  nEpochs    <- as.integer(nEpochs)
  batchSize  <- as.integer(batchSize)

  stopifnot(latentDim  >= 1L)
  stopifnot(all(hiddenDims >= 1L))
  stopifnot(nEpochs    >= 1L)
  stopifnot(batchSize  >= 1L)
  stopifnot(is.numeric(learningRate), learningRate > 0)

  # ---- Extract data ----------------------------------------------------------
  data <- ExtractSeuratData(
    seuratObject    = seuratObject,
    assay           = assay,
    layer           = layer,
    geneWhitelist   = geneWhitelist,
    geneBlacklist   = geneBlacklist,
    cellWhitelist   = cellWhitelist,
    cellBlacklist   = cellBlacklist,
    conditionalVars = conditionalVars
  )

  dir.create(outputDir, showWarnings = FALSE, recursive = TRUE)

  # ---- Persist data to disk --------------------------------------------------
  exprPath <- file.path(outputDir, "gene_expression.csv")
  .writeSparseMatrixCSV(data$geneExpression, exprPath)

  condPath <- ""
  if (!is.null(data$conditionals)) {
    condPath <- file.path(outputDir, "conditionals.csv")
    utils::write.csv(data$conditionals, condPath, row.names = TRUE)
  }

  metaPath <- file.path(outputDir, "data_metadata.json")
  .writeDataMetadata(data, conditionalVars, metaPath)

  # ---- Build and invoke Python script ----------------------------------------
  trainScript <- system.file("python", "train_cvae.py", package = "cVAEPyro")
  if (nchar(trainScript) == 0L) {
    stop("Could not locate 'train_cvae.py'. Is the package installed correctly?")
  }

  args <- list(
    expr_path    = exprPath,
    cond_path    = condPath,
    output_dir   = outputDir,
    latent_dim   = latentDim,
    hidden_dims  = paste(hiddenDims, collapse = ","),
    n_epochs     = nEpochs,
    learning_rate = learningRate,
    batch_size   = batchSize
  )

  .invokePythonScript(pythonPath, trainScript, "train_cvae", args)

  invisible(outputDir)
}
