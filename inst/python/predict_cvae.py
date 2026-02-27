"""
Conditional Variational Autoencoder (cVAE) prediction / forward-pass script.

This script defines the ``predict_cvae()`` function.  It is used by the R
package ``cVAEPyro``, which:
  1. Reads this file as a template.
  2. Appends a concrete ``predict_cvae(...)`` call with resolved arguments.
  3. Writes the combined code to a temporary ``.py`` file.
  4. Executes it via ``system2``.

Do **not** add top-level executable code to this file.  Only define the
``predict_cvae()`` function (and any helpers it requires).

The model architecture classes (``Encoder``, ``Decoder``, ``CVAE``) and the
preprocessing helpers (``encode_conditionals``, ``normalize_expression``)
used here are intentionally duplicated from ``train_cvae.py`` so that each
script is self-contained after the R append-and-run step.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyro.distributions as dist


# ---------------------------------------------------------------------------
# Neural network components  (mirrored from train_cvae.py)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dims, latent_dim):
        super().__init__()
        dims = [input_dim + cond_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x, c):
        h = self.net(torch.cat([x, c], dim=-1))
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dims, output_dim):
        super().__init__()
        dims = [latent_dim + cond_dim] + list(reversed(hidden_dims)) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=-1))


class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dims, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, cond_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, cond_dim, hidden_dims, input_dim)
        self.latent_dim = latent_dim

    def model(self, x, c):
        # Not used during inference; included for API symmetry with train_cvae.
        pass

    def guide(self, x, c):
        # Not used during inference; included for API symmetry with train_cvae.
        pass

    def encode(self, x, c):
        z_mu, z_log_var = self.encoder(x, c)
        z_scale = torch.exp(0.5 * z_log_var).clamp(min=1e-6)
        return z_mu, z_scale

    def decode(self, z, c):
        return self.decoder(z, c)

    def forward(self, x, c):
        z_mu, z_scale = self.encode(x, c)
        z = dist.Normal(z_mu, z_scale).rsample()
        return self.decode(z, c)


# ---------------------------------------------------------------------------
# Preprocessing helpers  (mirrored from train_cvae.py)
# ---------------------------------------------------------------------------

def encode_conditionals(cond_df, cond_metadata=None):
    encoded_parts = []
    metadata = {} if cond_metadata is None else dict(cond_metadata)

    for col in cond_df.columns:
        vals     = cond_df[col]
        col_type = metadata.get(col, {}).get("type", None)

        if col_type is None:
            col_type = "continuous" if pd.api.types.is_numeric_dtype(vals) else "categorical"

        if col_type == "categorical":
            if col not in metadata:
                categories = sorted(str(v) for v in vals.unique())
                metadata[col] = {"type": "categorical", "categories": categories}
            else:
                categories = metadata[col]["categories"]

            ohe = pd.get_dummies(vals.astype(str), prefix=col)
            for cat in categories:
                cat_col = f"{col}_{cat}"
                if cat_col not in ohe.columns:
                    ohe[cat_col] = 0
            ohe = ohe[[f"{col}_{cat}" for cat in categories]]
            encoded_parts.append(ohe.values.astype(np.float32))

        else:
            arr = vals.values.astype(np.float32)
            if col not in metadata:
                mean_val = float(arr.mean())
                std_val  = float(arr.std()) + 1e-8
                metadata[col] = {"type": "continuous", "mean": mean_val, "std": std_val}
            else:
                mean_val = metadata[col]["mean"]
                std_val  = metadata[col]["std"]
            arr = (arr - mean_val) / std_val
            encoded_parts.append(arr.reshape(-1, 1))

    if not encoded_parts:
        return np.zeros((len(cond_df), 0), dtype=np.float32), metadata

    return np.concatenate(encoded_parts, axis=1), metadata


def normalize_expression(expr, expr_metadata=None):
    expr = np.log1p(expr.astype(np.float64))
    if expr_metadata is None:
        mean_val = expr.mean(axis=0)
        std_val  = expr.std(axis=0) + 1e-8
        expr_metadata = {"mean": mean_val.tolist(), "std": std_val.tolist()}
    else:
        mean_val = np.array(expr_metadata["mean"])
        std_val  = np.array(expr_metadata["std"])

    expr = (expr - mean_val) / std_val
    return expr.astype(np.float32), expr_metadata


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------

def predict_cvae(
    model_dir,
    expr_path="",
    cond_path="",
    output_dir=".",
    n_samples=1,
):
    """
    Run a forward pass through a trained cVAE to obtain (counterfactual)
    predictions.

    Parameters
    ----------
    model_dir : str
        Directory produced by ``train_cvae`` containing ``cvae_model.pt``,
        ``pyro_params.pt``, and ``model_metadata.json``.
    expr_path : str
        Path to new gene-expression CSV (cells × genes).  When ``""`` the
        model samples ``z`` from the prior N(0, I).
    cond_path : str
        Path to new conditionals CSV (cells × variables).  When ``""`` and
        the model was trained without conditionals this is valid; otherwise
        an error is raised.
    output_dir : str
        Directory in which to write ``predictions.csv``,
        ``latent_mean.csv``, and ``latent_std.csv``.
    n_samples : int
        Number of latent samples to average over when computing the
        expected reconstruction.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_samples = int(n_samples)

    # -- Load model metadata ---------------------------------------------------
    meta_path = os.path.join(model_dir, "model_metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Model metadata not found: {meta_path}")
    with open(meta_path) as fh:
        meta = json.load(fh)

    input_dim     = int(meta["input_dim"])
    cond_dim      = int(meta["cond_dim"])
    hidden_dims   = [int(x) for x in meta["hidden_dims"]]
    latent_dim    = int(meta["latent_dim"])
    gene_names    = meta["gene_names"]
    expr_metadata = meta["expr_metadata"]
    cond_metadata = meta.get("cond_metadata", {})

    # -- Rebuild model and load weights ----------------------------------------
    # We only need the PyTorch state dict for inference; the Pyro param store
    # (pyro_params.pt) contains training-time optimiser state and is not
    # required for forward passes.
    cvae      = CVAE(input_dim, cond_dim, hidden_dims, latent_dim)
    ckpt_path = os.path.join(model_dir, "cvae_model.pt")
    # weights_only=True (available since PyTorch 2.0, required ≥ 2.6) prevents
    # arbitrary code execution when loading checkpoints.  The checkpoint
    # contains only primitive Python types that are safe to deserialise.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cvae.load_state_dict(checkpoint["model_state_dict"])
    cvae.eval()
    print("Model loaded from:", model_dir)

    # -- Determine cell count --------------------------------------------------
    # Derive from expr_path or cond_path; one must be provided.
    if expr_path and os.path.isfile(expr_path):
        ref_df = pd.read_csv(expr_path, index_col=0)
        cell_names = list(ref_df.index)
        n_cells    = len(cell_names)
    elif cond_path and os.path.isfile(cond_path):
        ref_df     = pd.read_csv(cond_path, index_col=0)
        cell_names = list(ref_df.index)
        n_cells    = len(cell_names)
    else:
        raise ValueError(
            "At least one of 'expr_path' or 'cond_path' must point to an "
            "existing file so the number of cells can be determined."
        )

    # -- Prepare expression tensor ---------------------------------------------
    if expr_path and os.path.isfile(expr_path):
        expr_df  = pd.read_csv(expr_path, index_col=0)
        # Reorder columns to match training gene order
        missing_genes = [g for g in gene_names if g not in expr_df.columns]
        if missing_genes:
            raise ValueError(
                f"{len(missing_genes)} gene(s) required by the model are "
                f"absent from the expression matrix: "
                f"{missing_genes[:5]} ..."
            )
        expr_df  = expr_df[gene_names]
        expr_np, _ = normalize_expression(expr_df.values, expr_metadata)
        x_tensor   = torch.tensor(expr_np)
        use_prior  = False
    else:
        x_tensor  = None
        use_prior = True

    # -- Prepare conditional tensor --------------------------------------------
    if cond_path and os.path.isfile(cond_path):
        cond_df = pd.read_csv(cond_path, index_col=0)
        cond_df = cond_df.loc[cell_names]
        cond_np, _ = encode_conditionals(cond_df, cond_metadata)
        c_tensor   = torch.tensor(cond_np)
    else:
        if cond_dim > 0:
            raise ValueError(
                "Model was trained with conditional variables but 'cond_path' "
                "was not provided or does not exist."
            )
        c_tensor = torch.zeros(n_cells, 0)

    # -- Forward pass ----------------------------------------------------------
    print(f"Running forward pass  n_cells={n_cells}  n_samples={n_samples}  "
          f"use_prior={use_prior}")

    with torch.no_grad():
        if use_prior:
            # Sample z from prior, decode with supplied conditionals
            accumulated = torch.zeros(n_cells, input_dim)
            for _ in range(n_samples):
                z = torch.randn(n_cells, latent_dim)
                accumulated += cvae.decode(z, c_tensor)
            preds = (accumulated / n_samples).numpy()
            z_mu  = np.zeros((n_cells, latent_dim))
            z_std = np.ones((n_cells, latent_dim))
        else:
            z_mu_t, z_std_t = cvae.encode(x_tensor, c_tensor)
            accumulated = torch.zeros(n_cells, input_dim)
            for _ in range(n_samples):
                z = dist.Normal(z_mu_t, z_std_t).rsample()
                accumulated += cvae.decode(z, c_tensor)
            preds = (accumulated / n_samples).numpy()
            z_mu  = z_mu_t.numpy()
            z_std = z_std_t.numpy()

    # -- Save outputs ----------------------------------------------------------
    pred_df = pd.DataFrame(preds, index=cell_names, columns=gene_names)
    pred_path = os.path.join(output_dir, "predictions.csv")
    pred_df.to_csv(pred_path)
    print(f"Saved predictions      → {pred_path}")

    latent_cols = [f"z{i + 1}" for i in range(latent_dim)]
    pd.DataFrame(z_mu,  index=cell_names, columns=latent_cols).to_csv(
        os.path.join(output_dir, "latent_mean.csv")
    )
    pd.DataFrame(z_std, index=cell_names, columns=latent_cols).to_csv(
        os.path.join(output_dir, "latent_std.csv")
    )
    print("Saved latent embeddings → latent_mean.csv, latent_std.csv")
    print("Prediction complete.")
