# ---- Internal helpers -------------------------------------------------------
# These functions are not exported.  They follow the snake_case naming
# convention for private/internal symbols described in the package spec.


# Cell filtering --------------------------------------------------------------

#' @keywords internal
.filterCells <- function(allCells, cellWhitelist, cellBlacklist) {
  cells <- allCells
  if (!is.null(cellWhitelist)) {
    cells <- intersect(cells, cellWhitelist)
  }
  if (!is.null(cellBlacklist)) {
    cells <- setdiff(cells, cellBlacklist)
  }
  cells
}


# Gene filtering --------------------------------------------------------------

#' @keywords internal
.filterGenes <- function(allGenes, geneWhitelist, geneBlacklist) {
  genes <- allGenes
  if (!is.null(geneWhitelist)) {
    genes <- intersect(genes, geneWhitelist)
  }
  if (!is.null(geneBlacklist)) {
    genes <- setdiff(genes, geneBlacklist)
  }
  genes
}


# CSV writer for sparse matrices ----------------------------------------------

#' Write a sparse (or dense) matrix to CSV with row and column names.
#' @keywords internal
.writeSparseMatrixCSV <- function(mat, path) {
  # Convert to dense for writing; acceptable for reasonably-sized subsets.
  # For very large matrices users should supply smaller gene/cell sets.
  denseMat <- as.matrix(mat)
  utils::write.csv(denseMat, file = path, row.names = TRUE)
}


# JSON data-metadata writer ---------------------------------------------------

#' Write gene names, cell names, and conditional variable info to JSON.
#' @keywords internal
.writeDataMetadata <- function(data, conditionalVars, path) {
  meta <- list(
    gene_names      = colnames(data$geneExpression),
    cell_names      = rownames(data$geneExpression),
    conditional_vars = if (is.null(conditionalVars)) list() else as.list(conditionalVars)
  )
  jsonlite::write_json(meta, path, auto_unbox = TRUE, pretty = TRUE)
}


# Python script invocation ----------------------------------------------------

#' Read a bundled Python script, append a concrete function call, and execute
#' it via system2.
#'
#' The R-invokes-Python pattern:
#' 1. Read the script template from `inst/python/`.
#' 2. Append a function call with the resolved arguments.
#' 3. Write to a temporary `.py` file.
#' 4. Call `system2(pythonPath, tmpFile)`.
#'
#' Argument values are serialised via `jsonlite::toJSON` so that paths with
#' spaces, backslashes, or embedded quotes are correctly escaped in the
#' generated Python call.
#'
#' @param pythonPath Path to the Python 3 executable.
#' @param scriptPath Path to the bundled `.py` template script.
#' @param funcName Name of the top-level function to call within the script.
#' @param args Named list of arguments to pass to the function.
#' @keywords internal
.invokePythonScript <- function(pythonPath, scriptPath, funcName, args) {
  scriptContent <- paste(readLines(scriptPath, warn = FALSE), collapse = "\n")

  # Serialise each argument value to a Python literal.
  # jsonlite::toJSON produces valid JSON which is also valid Python for the
  # primitive types we use (strings, numbers, booleans).
  argsStr <- paste(
    mapply(
      function(k, v) {
        if (is.character(v)) {
          # Use JSON string encoding so special characters are properly escaped
          pyVal <- jsonlite::toJSON(v, auto_unbox = TRUE)
        } else if (is.logical(v)) {
          pyVal <- if (v) "True" else "False"
        } else {
          pyVal <- as.character(v)
        }
        sprintf("%s=%s", k, pyVal)
      },
      names(args),
      args,
      SIMPLIFY = TRUE
    ),
    collapse = ", "
  )

  callStr  <- sprintf('\n\n%s(%s)\n', funcName, argsStr)
  fullCode <- paste0(scriptContent, callStr)

  tmpScript <- tempfile(fileext = ".py")
  writeLines(fullCode, tmpScript)
  on.exit(unlink(tmpScript), add = TRUE)

  result <- system2(
    command = pythonPath,
    args    = shQuote(tmpScript),
    stdout  = TRUE,
    stderr  = TRUE
  )

  status <- attr(result, "status")
  if (!is.null(status) && status != 0L) {
    stop("Python script '", funcName, "' failed (exit status ", status, "):\n",
         paste(result, collapse = "\n"))
  }

  invisible(result)
}


# newdata validator -----------------------------------------------------------

#' Validate that a newdata list conforms to the trained model specification.
#' @keywords internal
.validateNewdata <- function(newdata, modelDir) {
  if (!is.list(newdata)) {
    stop("'newdata' must be a named list.")
  }

  validKeys <- c("geneExpression", "conditionals")
  unknownKeys <- setdiff(names(newdata), validKeys)
  if (length(unknownKeys) > 0L) {
    warning("Unknown elements in 'newdata' will be ignored: ",
            paste(unknownKeys, collapse = ", "))
  }

  if (is.null(newdata$geneExpression) && is.null(newdata$conditionals)) {
    stop("'newdata' must contain at least one of 'geneExpression' or ",
         "'conditionals'.")
  }

  # Load model metadata for conformance checks
  metaPath <- file.path(modelDir, "model_metadata.json")
  if (!file.exists(metaPath)) {
    return(invisible(NULL))   # metadata not available; skip further checks
  }

  meta <- jsonlite::read_json(metaPath, simplifyVector = TRUE)

  # Check gene conformance
  if (!is.null(newdata$geneExpression)) {
    trainGenes  <- meta$gene_names
    newGenes    <- colnames(newdata$geneExpression)
    if (is.null(newGenes)) {
      stop("'newdata$geneExpression' must have column names (gene names).")
    }
    missingGenes <- setdiff(trainGenes, newGenes)
    if (length(missingGenes) > 0L) {
      stop(length(missingGenes), " gene(s) present at training time are ",
           "missing from 'newdata$geneExpression': ",
           paste(utils::head(missingGenes, 5L), collapse = ", "),
           if (length(missingGenes) > 5L) " ..." else "")
    }
    extraGenes <- setdiff(newGenes, trainGenes)
    if (length(extraGenes) > 0L) {
      warning(length(extraGenes), " gene(s) in 'newdata$geneExpression' were ",
              "not seen during training and will be ignored.")
    }
  }

  # Check conditional conformance
  if (!is.null(newdata$conditionals) && length(meta$cond_metadata) > 0L) {
    trainConds <- names(meta$cond_metadata)
    newConds   <- colnames(newdata$conditionals)
    if (is.null(newConds)) {
      stop("'newdata$conditionals' must have column names.")
    }
    missingConds <- setdiff(trainConds, newConds)
    if (length(missingConds) > 0L) {
      stop("Conditional variable(s) used at training time are missing from ",
           "'newdata$conditionals': ",
           paste(missingConds, collapse = ", "))
    }
  }

  invisible(NULL)
}
