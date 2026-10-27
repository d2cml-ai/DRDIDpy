# Rscript tests/r_ref/export.R /clean/DRDID /new/output.json [fastglm-wrapper]
# Source identity and reference acceptance are reviewed separately from export.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) %in% c(2L, 3L))
source <- normalizePath(args[[1L]], mustWork = TRUE)
if (file.exists(args[[2L]])) stop("Output already exists; use a new file")
git_output <- function(...) system2("git", c("-C", shQuote(source), ...), stdout = TRUE)
revision <- git_output("rev-parse", "HEAD")
stopifnot(length(revision) == 1L, grepl("^[0-9a-f]{40}$", revision))
if (length(git_output("status", "--porcelain", "--untracked-files=all"))) {
  stop("R checkout must be clean, including untracked files")
}
# Bind the complete tracked tree: native code, namespace declarations and build
# hooks can affect behavior too. Documentation-only changes also require review.
source_files <- git_output("ls-files")
source_hashes <- as.list(setNames(vapply(file.path(source, source_files), function(path) {
  digest::digest(file = path, algo = "sha256")
}, character(1L)), source_files))
suppressPackageStartupMessages(pkgload::load_all(source, quiet = TRUE, recompile = TRUE))
if (length(args) == 3L) {
  stopifnot(args[[3L]] == "fastglm-wrapper")
  namespace <- asNamespace("DRDID")
  unlockBinding("fastglm_fit", namespace)
  assign("fastglm_fit", function(x, y, family, weights = NULL, method = 0L) {
    fastglm::fastglm(x = x, y = y, family = family, weights = weights,
                    method = method, tol = 1e-8, maxit = 100L)
  }, envir = namespace)
  lockBinding("fastglm_fit", namespace)
}
file_arg <- grep("^--file=", commandArgs(), value = TRUE)
root <- dirname(normalizePath(sub("^--file=", "", file_arg[[1L]])))
cases <- jsonlite::read_json(file.path(root, "cases.json"), simplifyVector = FALSE)
out <- list()
for (case in cases) {
  data <- read.csv(file.path(root, paste0(case$dataset, ".csv")))
  n <- nrow(data)
  weights <- switch(case$weights, none = NULL, ones = rep(1, n), weights = data$w, zero = data$w)
  if (case$weights == "zero") weights[seq(1L, n, by = 11L)] <- 0
  design <- if (case$covariates) cbind(1, data$x, data$z) else NULL
  params <- list(D = data$D, covariates = design, i.weights = weights,
                 boot = FALSE, inffunc = TRUE)
  if (endsWith(case$method, "_rc")) {
    params$y <- data$y * case$scale
    params$post <- data$post
  } else {
    params$y1 <- data$y1 * case$scale
    params$y0 <- data$y0 * case$scale
  }
  if (!startsWith(case$method, "reg_")) params$trim.level <- case$trim_level
  result <- do.call(get(case$method, envir = asNamespace("DRDID")), params)
  out[[case$id]] <- list(att = result$ATT, influence = as.numeric(result$att.inf.func),
                         se = result$se, lci = result$lci, uci = result$uci)
}
inputs <- c("ordinary.csv", "trim.csv", "cases.json", "export.R", "generate_inputs.py")
input_hashes <- as.list(setNames(vapply(file.path(root, inputs), function(path) {
  digest::digest(file = path, algo = "sha256")
}, character(1L)), inputs))
payload <- list(schema_version = 1L,
                source = list(revision = revision, version = as.character(packageVersion("DRDID")),
                              hashes = source_hashes),
                inputs = input_hashes, configurations = length(out), results = out)
jsonlite::write_json(payload, args[[2L]], auto_unbox = TRUE, digits = 17, pretty = FALSE)
cat(length(out), "R configurations exported\n")
