# Multi-stage Dockerfile for cVAEPyro
#
# Stage 1 "deps"   – system libraries + R packages + Python packages
# Stage 2 "runtime" – copies the installed package on top of deps
#
# Build args (matched by the docker-cache.yml workflow):
#   R_VERSION       – R version (e.g. 4.4.0); uses CRAN binary if empty
#   BIOC_VERSION    – Bioconductor version (e.g. 3.20); overrides .bioc_version
#   PYTHON_VERSION  – Python version (e.g. 3.11)
#   BASE_IMAGE      – base image to build FROM (injected by workflow)
#   SKIP_BASE_DEPS  – if "true", skip reinstalling base deps (injected by workflow)

ARG BASE_IMAGE=ubuntu:22.04
ARG SKIP_BASE_DEPS=false
ARG R_VERSION=""
ARG BIOC_VERSION=""
ARG PYTHON_VERSION="3.11"

# ============================================================================
# Stage: deps
# Installs all OS, R, and Python dependencies.  This layer is cached by the
# docker-cache.yml workflow keyed on a hash of the dependency files.
# ============================================================================
FROM ${BASE_IMAGE} AS deps

ARG R_VERSION
ARG BIOC_VERSION
ARG PYTHON_VERSION
ARG SKIP_BASE_DEPS

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    R_LIBS_USER=/usr/local/lib/R/site-library

# ---- System libraries -------------------------------------------------------
RUN if [ "${SKIP_BASE_DEPS}" != "true" ]; then \
      apt-get update -qq && \
      apt-get install -y --no-install-recommends \
        software-properties-common \
        dirmngr \
        gnupg2 \
        wget \
        curl \
        ca-certificates \
        libssl-dev \
        libcurl4-openssl-dev \
        libxml2-dev \
        libfontconfig1-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        libfreetype6-dev \
        libpng-dev \
        libtiff5-dev \
        libjpeg-dev \
        libhdf5-dev \
        libgit2-dev \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev \
        libglpk-dev \
        libgmp3-dev \
        pandoc \
        git \
        python3 \
        python3-pip \
        python3-venv && \
      apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# ---- Install requested Python version (if specified) -----------------------
RUN if [ -n "${PYTHON_VERSION}" ] && [ "${SKIP_BASE_DEPS}" != "true" ]; then \
      add-apt-repository ppa:deadsnakes/ppa -y && \
      apt-get update -qq && \
      apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv && \
      update-alternatives --install /usr/bin/python3 python3 \
        /usr/bin/python${PYTHON_VERSION} 1 && \
      python3 -m ensurepip --upgrade; \
    fi

# ---- Install R (via r2u / CRAN binary) -------------------------------------
RUN if [ "${SKIP_BASE_DEPS}" != "true" ]; then \
      wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
        | tee /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc && \
      add-apt-repository \
        "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" && \
      apt-get update -qq && \
      if [ -n "${R_VERSION}" ]; then \
        apt-get install -y --no-install-recommends r-base=${R_VERSION}*; \
      else \
        apt-get install -y --no-install-recommends r-base; \
      fi && \
      apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# ---- Python ML dependencies -------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ---- R package dependencies -------------------------------------------------
COPY renv.lock* /tmp/
COPY .bioc_version /tmp/.bioc_version

RUN Rscript -e "\
  options(repos = c(CRAN = 'https://cloud.r-project.org')); \
  bioc_ver <- trimws(readLines('/tmp/.bioc_version')); \
  if (!requireNamespace('BiocManager', quietly=TRUE)) install.packages('BiocManager'); \
  BiocManager::install(version = bioc_ver, ask = FALSE, update = FALSE); \
  pkgs <- c( \
    'Matrix', 'ggplot2', 'methods', 'jsonlite', \
    'testthat', 'devtools', 'roxygen2' \
  ); \
  install.packages(pkgs, dependencies = TRUE); \
  if (!requireNamespace('Seurat', quietly=TRUE)) \
    BiocManager::install('Seurat', ask = FALSE); \
  "

# ============================================================================
# Stage: runtime
# Adds the package source and installs it.
# ============================================================================
FROM deps AS runtime

ARG DEPS_IMAGE

COPY . /workspace/cVAEPyro
WORKDIR /workspace/cVAEPyro

RUN Rscript -e "devtools::install('.', dependencies = FALSE, upgrade = 'never')"

CMD ["bash"]
