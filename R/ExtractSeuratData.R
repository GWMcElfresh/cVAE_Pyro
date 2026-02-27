#' Extract gene expression and metadata from a Seurat object
#'
#' Harvests gene expression (as a sparse cells-x-genes matrix) and optional
#' metadata columns (conditionals) from a Seurat object, applying gene and
#' cell whitelist/blacklist filters before returning.
#'
#' @param seuratObject A Seurat object.
#' @param assay Character scalar. Assay to harvest from. Defaults to
#'   `DefaultAssay(seuratObject)`.
#' @param layer Character scalar. Layer within the assay to use (default:
#'   `"data"`).
#' @param geneWhitelist Character vector of gene names to **include**. If
#'   `NULL` (default), all genes are eligible.
#' @param geneBlacklist Character vector of gene names to **exclude**. Applied
#'   after `geneWhitelist`. If `NULL` (default), no genes are removed.
#' @param cellWhitelist Character vector of cell barcodes to **include**. If
#'   `NULL` (default), all cells are eligible.
#' @param cellBlacklist Character vector of cell barcodes to **exclude**.
#'   Applied after `cellWhitelist`. If `NULL` (default), no cells are removed.
#' @param conditionalVars Character vector of metadata column names to use as
#'   conditional variables. If `NULL` (default), no conditionals are returned.
#'
#' @return A named list with two elements:
#'   \describe{
#'     \item{`geneExpression`}{A sparse \code{dgCMatrix} of dimensions
#'       cells × genes.}
#'     \item{`conditionals`}{A \code{data.frame} of dimensions cells ×
#'       `conditionalVars`, or `NULL` when `conditionalVars` is `NULL`.}
#'   }
#'
#' @examples
#' \dontrun{
#' data <- ExtractSeuratData(
#'   seuratObject  = seurat_obj,
#'   assay         = "RNA",
#'   layer         = "data",
#'   geneWhitelist = c("CD3E", "CD8A"),
#'   conditionalVars = c("cell_type", "sample_id")
#' )
#' dim(data$geneExpression) # cells x genes
#' head(data$conditionals)
#' }
#'
#' @importFrom methods is
#' @importFrom Matrix t
#' @export
ExtractSeuratData <- function(seuratObject,
                               assay           = NULL,
                               layer           = "data",
                               geneWhitelist   = NULL,
                               geneBlacklist   = NULL,
                               cellWhitelist   = NULL,
                               cellBlacklist   = NULL,
                               conditionalVars = NULL) {

  if (!methods::is(seuratObject, "Seurat")) {
    stop("'seuratObject' must be a Seurat object.")
  }

  if (is.null(assay)) {
    assay <- Seurat::DefaultAssay(seuratObject)
  }

  if (!assay %in% names(seuratObject@assays)) {
    stop("Assay '", assay, "' not found in the Seurat object.")
  }

  # ---- Cell filtering --------------------------------------------------------
  allCells      <- colnames(seuratObject)
  selectedCells <- .filterCells(allCells, cellWhitelist, cellBlacklist)

  if (length(selectedCells) == 0L) {
    stop("No cells remain after applying whitelist/blacklist filters.")
  }

  # ---- Gene filtering --------------------------------------------------------
  allGenes      <- rownames(seuratObject[[assay]])
  selectedGenes <- .filterGenes(allGenes, geneWhitelist, geneBlacklist)

  if (length(selectedGenes) == 0L) {
    stop("No genes remain after applying whitelist/blacklist filters.")
  }

  # ---- Extract expression matrix (genes x cells → cells x genes) -------------
  exprMatrix <- Seurat::GetAssayData(seuratObject[[assay]], layer = layer)
  exprMatrix <- exprMatrix[selectedGenes, selectedCells, drop = FALSE]
  exprMatrix <- Matrix::t(exprMatrix)   # cells x genes

  # ---- Extract conditionals --------------------------------------------------
  condData <- NULL
  if (!is.null(conditionalVars)) {
    missingVars <- setdiff(conditionalVars, colnames(seuratObject@meta.data))
    if (length(missingVars) > 0L) {
      stop("Conditional variables not found in metadata: ",
           paste(missingVars, collapse = ", "))
    }
    condData <- seuratObject@meta.data[selectedCells, conditionalVars,
                                       drop = FALSE]
  }

  list(
    geneExpression = exprMatrix,
    conditionals   = condData
  )
}
