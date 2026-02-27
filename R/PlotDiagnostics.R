#' Load training diagnostics from a CSV file
#'
#' Reads the \code{training_diagnostics.csv} written by
#' \code{\link{TrainCVAE}} into a \code{data.frame}.
#'
#' @param diagnosticsPath Character scalar. Path to the diagnostics CSV. Must
#'   contain at least an \code{epoch} column.
#'
#' @return A \code{data.frame} with one row per epoch and one column per
#'   tracked loss component.
#'
#' @examples
#' \dontrun{
#' diag <- LoadDiagnostics("results/model1/training_diagnostics.csv")
#' head(diag)
#' }
#'
#' @export
LoadDiagnostics <- function(diagnosticsPath) {
  if (!file.exists(diagnosticsPath)) {
    stop("Diagnostics file not found: ", diagnosticsPath)
  }
  utils::read.csv(diagnosticsPath, stringsAsFactors = FALSE)
}

#' Plot training diagnostics with ggplot2
#'
#' Produces a line plot of all loss components across training epochs from the
#' \code{training_diagnostics.csv} written by \code{\link{TrainCVAE}}.
#'
#' @param diagnosticsPath Either a character scalar path to the diagnostics
#'   CSV (passed to \code{\link{LoadDiagnostics}}), or a \code{data.frame}
#'   already loaded by \code{LoadDiagnostics}.
#'
#' @return A \code{ggplot2} object that can be further modified or printed.
#'
#' @details
#' The diagnostics \code{data.frame} is reshaped to long format so that all
#' loss components (all columns except \code{epoch}) are plotted on the same
#' panel. Use \code{ggplot2::facet_wrap} on the returned object if you prefer
#' separate panels.
#'
#' @examples
#' \dontrun{
#' p <- PlotTrainingDiagnostics("results/model1/training_diagnostics.csv")
#' print(p)
#'
#' # or load first, then plot
#' diag <- LoadDiagnostics("results/model1/training_diagnostics.csv")
#' PlotTrainingDiagnostics(diag)
#' }
#'
#' @importFrom ggplot2 ggplot aes geom_line labs theme_minimal
#' @export
PlotTrainingDiagnostics <- function(diagnosticsPath) {

  if (is.character(diagnosticsPath)) {
    diagData <- LoadDiagnostics(diagnosticsPath)
  } else if (is.data.frame(diagnosticsPath)) {
    diagData <- diagnosticsPath
  } else {
    stop("'diagnosticsPath' must be a file path (character) or a data.frame.")
  }

  if (!"epoch" %in% colnames(diagData)) {
    stop("Diagnostics data must contain an 'epoch' column.")
  }

  lossVars <- colnames(diagData)[colnames(diagData) != "epoch"]
  if (length(lossVars) == 0L) {
    stop("No loss columns found in diagnostics data (columns other than 'epoch').")
  }

  # Reshape to long format without importing reshape2 / tidyr
  longData <- do.call(rbind, lapply(lossVars, function(v) {
    data.frame(
      epoch      = diagData[["epoch"]],
      loss_value = diagData[[v]],
      loss_type  = v,
      stringsAsFactors = FALSE
    )
  }))

  ggplot2::ggplot(
    longData,
    ggplot2::aes(x = .data[["epoch"]], y = .data[["loss_value"]],
                 color = .data[["loss_type"]])
  ) +
    ggplot2::geom_line() +
    ggplot2::labs(
      title = "cVAE Training Diagnostics",
      x     = "Epoch",
      y     = "Loss",
      color = "Loss Component"
    ) +
    ggplot2::theme_minimal()
}
