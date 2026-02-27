"""
Conditional Variational Autoencoder (cVAE) training script using Pyro.

This script defines the ``train_cvae()`` function.  It is used by the R
package ``cVAEPyro``, which:
  1. Reads this file as a template.
  2. Appends a concrete ``train_cvae(...)`` call with resolved arguments.
  3. Writes the combined code to a temporary ``.py`` file.
  4. Executes it via ``system2``.

Do **not** add top-level executable code to this file.  Only define
classes and the ``train_cvae()`` function.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam


# ---------------------------------------------------------------------------
# Neural network components
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Encodes (gene expression, conditionals) → (mu_z, log_var_z)."""

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
    """Decodes (latent z, conditionals) → reconstructed gene expression."""

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
    """Conditional Variational Autoencoder."""

    def __init__(self, input_dim, cond_dim, hidden_dims, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, cond_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, cond_dim, hidden_dims, input_dim)
        self.latent_dim = latent_dim

    # -- Pyro model (generative process) -------------------------------------

    def model(self, x, c):
        pyro.module("cvae", self)
        with pyro.plate("data", x.shape[0]):
            z_loc   = x.new_zeros(x.shape[0], self.latent_dim)
            z_scale = x.new_ones(x.shape[0], self.latent_dim)
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
            x_loc = self.decoder(z, c)
            # Fixed unit observation variance: expression data is log1p-normalised
            # and z-score standardised, so a variance of ~1 is a reasonable
            # default.  Learned variance would require an extra decoder head.
            pyro.sample(
                "obs",
                dist.Normal(x_loc, x.new_ones(x_loc.shape)).to_event(1),
                obs=x,
            )

    # -- Pyro guide (approximate posterior) ----------------------------------

    def guide(self, x, c):
        pyro.module("cvae", self)
        with pyro.plate("data", x.shape[0]):
            z_mu, z_log_var = self.encoder(x, c)
            z_scale = torch.exp(0.5 * z_log_var).clamp(min=1e-6)
            pyro.sample("z", dist.Normal(z_mu, z_scale).to_event(1))

    # -- Convenience methods -------------------------------------------------

    def encode(self, x, c):
        """Return (mu_z, sigma_z) for the approximate posterior."""
        z_mu, z_log_var = self.encoder(x, c)
        z_scale = torch.exp(0.5 * z_log_var).clamp(min=1e-6)
        return z_mu, z_scale

    def decode(self, z, c):
        """Decode latent z to gene-expression space."""
        return self.decoder(z, c)

    def forward(self, x, c):
        z_mu, z_scale = self.encode(x, c)
        z = dist.Normal(z_mu, z_scale).rsample()
        return self.decode(z, c)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def encode_conditionals(cond_df, cond_metadata=None):
    """
    Encode conditional variables for the model.

    Categorical variables are one-hot encoded; continuous variables are
    z-score standardised.  When ``cond_metadata`` is ``None`` the encoding
    parameters are inferred from the data and returned alongside the encoded
    array so they can be saved and reused at prediction time.

    Parameters
    ----------
    cond_df : pd.DataFrame
        Cells × variables data frame.
    cond_metadata : dict or None
        Previously computed encoding metadata (used at prediction time).

    Returns
    -------
    encoded : np.ndarray  shape (n_cells, cond_dim)
    metadata : dict
    """
    encoded_parts = []
    metadata = {} if cond_metadata is None else dict(cond_metadata)

    for col in cond_df.columns:
        vals = cond_df[col]
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

        else:  # continuous
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
    """
    Log1p-normalise then z-score standardise gene expression.

    Parameters
    ----------
    expr : np.ndarray  shape (n_cells, n_genes)
    expr_metadata : dict or None

    Returns
    -------
    normalised : np.ndarray  (float32)
    metadata : dict  with keys ``mean``, ``std``, ``gene_names``, ``cell_names``
    """
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
# Main training function
# ---------------------------------------------------------------------------

def train_cvae(
    expr_path,
    cond_path="",
    output_dir=".",
    latent_dim=10,
    hidden_dims="128,64",
    n_epochs=100,
    learning_rate=1e-3,
    batch_size=128,
):
    """
    Train a conditional VAE on gene expression data.

    Parameters
    ----------
    expr_path : str
        Path to gene-expression CSV (cells × genes, row/column names
        included; written by ``TrainCVAE`` in R).
    cond_path : str
        Path to conditionals CSV (cells × variables), or ``""`` when no
        conditioning variables are used.
    output_dir : str
        Directory in which to save the model checkpoint, Pyro parameter
        store, model metadata JSON, and training diagnostics CSV.
    latent_dim : int
        Dimensionality of the latent space.
    hidden_dims : str or list
        Comma-separated hidden layer widths (e.g. ``"128,64"``), or a
        Python list of ints.
    n_epochs : int
        Number of full passes through the data.
    learning_rate : float
        Adam learning rate.
    batch_size : int
        Mini-batch size.
    """
    pyro.clear_param_store()
    os.makedirs(output_dir, exist_ok=True)

    # -- Parse arguments -------------------------------------------------------
    if isinstance(hidden_dims, str):
        hidden_dims = [int(x) for x in hidden_dims.split(",")]
    latent_dim    = int(latent_dim)
    n_epochs      = int(n_epochs)
    batch_size    = int(batch_size)
    learning_rate = float(learning_rate)

    # -- Load expression -------------------------------------------------------
    print(f"Loading gene expression from: {expr_path}")
    expr_df    = pd.read_csv(expr_path, index_col=0)
    gene_names = list(expr_df.columns)
    cell_names = list(expr_df.index)
    expr_np    = expr_df.values.astype(np.float32)

    expr_np, expr_metadata = normalize_expression(expr_np)
    expr_metadata["gene_names"] = gene_names
    expr_metadata["cell_names"] = cell_names

    # -- Load conditionals -----------------------------------------------------
    cond_np       = None
    cond_metadata = {}
    if cond_path and os.path.isfile(cond_path):
        print(f"Loading conditionals from: {cond_path}")
        cond_df = pd.read_csv(cond_path, index_col=0)
        cond_df = cond_df.loc[cell_names]          # align cell order
        cond_np, cond_metadata = encode_conditionals(cond_df)
        print(f"  Encoded {len(cond_metadata)} conditional variable(s), "
              f"dimension = {cond_np.shape[1]}")

    input_dim = expr_np.shape[1]
    cond_dim  = cond_np.shape[1] if cond_np is not None else 0
    print(f"Architecture  input={input_dim}  cond={cond_dim}  "
          f"hidden={hidden_dims}  latent={latent_dim}")

    # -- Build model & optimiser -----------------------------------------------
    cvae      = CVAE(input_dim, cond_dim, hidden_dims, latent_dim)
    optimizer = ClippedAdam({"lr": learning_rate, "clip_norm": 10.0})
    svi       = SVI(cvae.model, cvae.guide, optimizer, loss=Trace_ELBO())

    x_tensor = torch.tensor(expr_np)
    c_tensor = (torch.tensor(cond_np)
                if cond_np is not None
                else torch.zeros(len(cell_names), 0))
    n_cells = x_tensor.shape[0]

    # -- Training loop ---------------------------------------------------------
    losses = []
    print(f"Training for {n_epochs} epochs …")
    for epoch in range(n_epochs):
        perm       = torch.randperm(n_cells)
        x_shuffled = x_tensor[perm]
        c_shuffled = c_tensor[perm]

        epoch_loss = 0.0
        n_batches  = 0
        for i in range(0, n_cells, batch_size):
            x_batch     = x_shuffled[i : i + batch_size]
            c_batch     = c_shuffled[i : i + batch_size]
            epoch_loss += svi.step(x_batch, c_batch)
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append({"epoch": epoch + 1, "loss": avg_loss})
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:>5}/{n_epochs}  loss = {avg_loss:.4f}")

    # -- Save diagnostics CSV --------------------------------------------------
    loss_path = os.path.join(output_dir, "training_diagnostics.csv")
    pd.DataFrame(losses).to_csv(loss_path, index=False)
    print(f"Saved training diagnostics → {loss_path}")

    # -- Save model checkpoint -------------------------------------------------
    model_path = os.path.join(output_dir, "cvae_model.pt")
    torch.save(
        {
            "model_state_dict": cvae.state_dict(),
            "input_dim":        input_dim,
            "cond_dim":         cond_dim,
            "hidden_dims":      hidden_dims,
            "latent_dim":       latent_dim,
        },
        model_path,
    )
    print(f"Saved model checkpoint → {model_path}")

    # -- Save Pyro parameter store ---------------------------------------------
    pyro_path = os.path.join(output_dir, "pyro_params.pt")
    pyro.get_param_store().save(pyro_path)
    print(f"Saved Pyro param store  → {pyro_path}")

    # -- Save model metadata (needed for prediction) ---------------------------
    model_metadata = {
        "input_dim":     input_dim,
        "cond_dim":      cond_dim,
        "hidden_dims":   hidden_dims,
        "latent_dim":    latent_dim,
        "gene_names":    gene_names,
        "expr_metadata": expr_metadata,
        "cond_metadata": cond_metadata,
    }
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w") as fh:
        json.dump(model_metadata, fh, indent=2)
    print(f"Saved model metadata    → {meta_path}")
    print("Training complete.")
