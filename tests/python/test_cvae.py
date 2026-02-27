"""
Python tests for the cVAE training and prediction scripts.

Run with:
    python -m pytest tests/python/ -v

Requirements: pytest, pyro-ppl, torch, pandas, numpy, scipy
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make the inst/python scripts importable
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "inst", "python"
)
sys.path.insert(0, os.path.abspath(SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_expr(tmp_path):
    """Write a small gene-expression CSV and return its path."""
    np.random.seed(0)
    n_cells, n_genes = 40, 15
    data = np.random.poisson(lam=2, size=(n_cells, n_genes)).astype(float)
    df = pd.DataFrame(
        data,
        index   =[f"Cell{i}"  for i in range(n_cells)],
        columns =[f"Gene{j}"  for j in range(n_genes)],
    )
    path = str(tmp_path / "expr.csv")
    df.to_csv(path)
    return path, n_cells, n_genes


@pytest.fixture
def small_cond(tmp_path, small_expr):
    """Write a small conditionals CSV aligned with small_expr."""
    _, n_cells, _ = small_expr
    np.random.seed(1)
    df = pd.DataFrame(
        {
            "treatment":   np.random.choice(["ctrl", "treated"], n_cells),
            "batch_score": np.random.randn(n_cells),
        },
        index=[f"Cell{i}" for i in range(n_cells)],
    )
    path = str(tmp_path / "cond.csv")
    df.to_csv(path)
    return path


# ---------------------------------------------------------------------------
# Tests for preprocessing helpers
# ---------------------------------------------------------------------------

class TestEncodeConditionals:
    def test_categorical_encoding(self):
        from train_cvae import encode_conditionals
        df = pd.DataFrame({"grp": ["A", "B", "A", "C"]})
        enc, meta = encode_conditionals(df)
        assert enc.shape == (4, 3)         # 3 categories
        assert meta["grp"]["type"] == "categorical"
        assert sorted(meta["grp"]["categories"]) == ["A", "B", "C"]

    def test_continuous_encoding(self):
        from train_cvae import encode_conditionals
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        df   = pd.DataFrame({"score": vals})
        enc, meta = encode_conditionals(df)
        assert enc.shape == (4, 1)
        assert meta["score"]["type"] == "continuous"
        # After standardisation the mean should be ~0
        assert abs(enc[:, 0].mean()) < 1e-5

    def test_mixed_encoding(self):
        from train_cvae import encode_conditionals
        df = pd.DataFrame({
            "grp":   ["A", "B", "A"],
            "score": [1.0, 2.0, 3.0],
        })
        enc, meta = encode_conditionals(df)
        # 2 one-hot cols (A, B) + 1 continuous col = 3
        assert enc.shape == (3, 3)

    def test_reuse_metadata_at_prediction(self):
        from train_cvae import encode_conditionals
        df_train = pd.DataFrame({"grp": ["A", "B", "C"]})
        _, meta  = encode_conditionals(df_train)

        # At prediction time, only category "A" is present
        df_pred = pd.DataFrame({"grp": ["A", "A"]})
        enc, _  = encode_conditionals(df_pred, cond_metadata=meta)
        # Should still produce 3 columns (with zeros for B and C)
        assert enc.shape == (2, 3)
        assert enc[:, 1].sum() == 0   # column for B is all zeros
        assert enc[:, 2].sum() == 0   # column for C is all zeros

    def test_empty_dataframe(self):
        from train_cvae import encode_conditionals
        df = pd.DataFrame()
        enc, meta = encode_conditionals(df)
        assert enc.shape[1] == 0


class TestNormalizeExpression:
    def test_output_shape_preserved(self):
        from train_cvae import normalize_expression
        expr = np.random.rand(10, 20).astype(np.float32)
        norm, meta = normalize_expression(expr)
        assert norm.shape == expr.shape

    def test_metadata_keys(self):
        from train_cvae import normalize_expression
        expr = np.random.rand(10, 20).astype(np.float32)
        _, meta = normalize_expression(expr)
        assert "mean" in meta
        assert "std"  in meta

    def test_reuse_metadata(self):
        from train_cvae import normalize_expression
        expr1 = np.ones((5, 4), dtype=np.float32)
        _, meta = normalize_expression(expr1)
        # Normalising a different matrix with the saved metadata should not error
        expr2 = np.ones((3, 4), dtype=np.float32) * 2
        norm2, _ = normalize_expression(expr2, expr_metadata=meta)
        assert norm2.shape == (3, 4)


# ---------------------------------------------------------------------------
# Tests for the full CVAE class
# ---------------------------------------------------------------------------

class TestCVAE:
    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            import torch
            import pyro
            from train_cvae import CVAE
            self.torch = torch
            self.pyro  = pyro
            self.CVAE  = CVAE
        except ImportError:
            pytest.skip("torch / pyro not available")

    def _make_model(self, input_dim=20, cond_dim=3, hidden_dims=None,
                    latent_dim=5):
        hidden_dims = hidden_dims or [16, 8]
        return self.CVAE(input_dim, cond_dim, hidden_dims, latent_dim)

    def test_forward_shape(self):
        import torch
        model = self._make_model()
        x = torch.randn(8, 20)
        c = torch.randn(8, 3)
        out = model(x, c)
        assert out.shape == (8, 20)

    def test_encode_shape(self):
        import torch
        model = self._make_model()
        x = torch.randn(8, 20)
        c = torch.randn(8, 3)
        mu, sigma = model.encode(x, c)
        assert mu.shape    == (8, 5)
        assert sigma.shape == (8, 5)
        # sigma must be positive
        assert (sigma > 0).all()

    def test_decode_shape(self):
        import torch
        model = self._make_model()
        z = torch.randn(8, 5)
        c = torch.randn(8, 3)
        out = model.decode(z, c)
        assert out.shape == (8, 20)

    def test_zero_cond_dim(self):
        """Model should work with cond_dim = 0 (unconditional VAE)."""
        import torch
        model = self._make_model(cond_dim=0)
        x = torch.randn(4, 20)
        c = torch.zeros(4, 0)
        out = model(x, c)
        assert out.shape == (4, 20)


# ---------------------------------------------------------------------------
# Tests for train_cvae()
# ---------------------------------------------------------------------------

class TestTrainCVAE:
    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            import torch   # noqa: F401
            import pyro    # noqa: F401
            from train_cvae import train_cvae
            self.train_cvae = train_cvae
        except ImportError:
            pytest.skip("torch / pyro not available")

    def test_output_files_created(self, tmp_path, small_expr, small_cond):
        expr_path, _, _ = small_expr
        self.train_cvae(
            expr_path   = expr_path,
            cond_path   = small_cond,
            output_dir  = str(tmp_path),
            latent_dim  = 4,
            hidden_dims = "8,4",
            n_epochs    = 3,
            batch_size  = 16,
        )
        expected = [
            "training_diagnostics.csv",
            "cvae_model.pt",
            "pyro_params.pt",
            "model_metadata.json",
        ]
        for fname in expected:
            assert (tmp_path / fname).exists(), f"Missing: {fname}"

    def test_diagnostics_csv_structure(self, tmp_path, small_expr, small_cond):
        expr_path, _, _ = small_expr
        self.train_cvae(
            expr_path   = expr_path,
            cond_path   = small_cond,
            output_dir  = str(tmp_path),
            latent_dim  = 4,
            hidden_dims = "8,4",
            n_epochs    = 5,
            batch_size  = 16,
        )
        diag = pd.read_csv(tmp_path / "training_diagnostics.csv")
        assert "epoch" in diag.columns
        assert "loss"  in diag.columns
        assert len(diag) == 5

    def test_loss_decreases_overall(self, tmp_path, small_expr, small_cond):
        """The ELBO loss should generally decrease over training."""
        expr_path, _, _ = small_expr
        self.train_cvae(
            expr_path   = expr_path,
            cond_path   = small_cond,
            output_dir  = str(tmp_path),
            latent_dim  = 4,
            hidden_dims = "16,8",
            n_epochs    = 20,
            batch_size  = 16,
        )
        diag = pd.read_csv(tmp_path / "training_diagnostics.csv")
        first_half = diag["loss"].iloc[:10].mean()
        second_half = diag["loss"].iloc[10:].mean()
        # Allow up to 10 % increase to tolerate stochastic mini-batch noise;
        # the overall trend should be downward over 20 epochs.
        assert second_half <= first_half * 1.1, (
            f"Loss did not decrease: first_half={first_half:.4f}, "
            f"second_half={second_half:.4f}"
        )

    def test_no_conditionals(self, tmp_path, small_expr):
        """Training should work without any conditional variables."""
        expr_path, _, _ = small_expr
        self.train_cvae(
            expr_path   = expr_path,
            cond_path   = "",
            output_dir  = str(tmp_path),
            latent_dim  = 3,
            hidden_dims = "8",
            n_epochs    = 2,
            batch_size  = 16,
        )
        assert (tmp_path / "cvae_model.pt").exists()

    def test_model_metadata_contents(self, tmp_path, small_expr, small_cond):
        expr_path, n_cells, n_genes = small_expr
        self.train_cvae(
            expr_path   = expr_path,
            cond_path   = small_cond,
            output_dir  = str(tmp_path),
            latent_dim  = 4,
            hidden_dims = "8,4",
            n_epochs    = 2,
            batch_size  = 16,
        )
        with open(tmp_path / "model_metadata.json") as fh:
            meta = json.load(fh)

        assert meta["input_dim"]   == n_genes
        assert meta["latent_dim"]  == 4
        assert len(meta["gene_names"]) == n_genes
        assert "treatment"    in meta["cond_metadata"]
        assert "batch_score"  in meta["cond_metadata"]


# ---------------------------------------------------------------------------
# Tests for predict_cvae()
# ---------------------------------------------------------------------------

class TestPredictCVAE:
    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            import torch   # noqa: F401
            import pyro    # noqa: F401
            from train_cvae  import train_cvae
            from predict_cvae import predict_cvae
            self.train_cvae  = train_cvae
            self.predict_cvae = predict_cvae
        except ImportError:
            pytest.skip("torch / pyro not available")

    @pytest.fixture
    def trained_model(self, tmp_path, small_expr, small_cond):
        """Train a tiny model and return model_dir + expr metadata."""
        expr_path, n_cells, n_genes = small_expr
        model_dir = str(tmp_path / "model")
        self.train_cvae(
            expr_path   = expr_path,
            cond_path   = small_cond,
            output_dir  = model_dir,
            latent_dim  = 4,
            hidden_dims = "8,4",
            n_epochs    = 3,
            batch_size  = 16,
        )
        return model_dir, expr_path, small_cond, n_cells, n_genes

    def test_predictions_shape(self, tmp_path, trained_model):
        model_dir, expr_path, cond_path, n_cells, n_genes = trained_model
        out_dir = str(tmp_path / "preds")
        self.predict_cvae(
            model_dir  = model_dir,
            expr_path  = expr_path,
            cond_path  = cond_path,
            output_dir = out_dir,
            n_samples  = 1,
        )
        preds = pd.read_csv(os.path.join(out_dir, "predictions.csv"),
                            index_col=0)
        assert preds.shape == (n_cells, n_genes)

    def test_predictions_are_finite(self, tmp_path, trained_model):
        model_dir, expr_path, cond_path, n_cells, n_genes = trained_model
        out_dir = str(tmp_path / "preds_finite")
        self.predict_cvae(
            model_dir  = model_dir,
            expr_path  = expr_path,
            cond_path  = cond_path,
            output_dir = out_dir,
        )
        preds = pd.read_csv(os.path.join(out_dir, "predictions.csv"),
                            index_col=0)
        assert np.all(np.isfinite(preds.values))

    def test_latent_output_files(self, tmp_path, trained_model):
        model_dir, expr_path, cond_path, n_cells, n_genes = trained_model
        out_dir = str(tmp_path / "preds_latent")
        self.predict_cvae(
            model_dir  = model_dir,
            expr_path  = expr_path,
            cond_path  = cond_path,
            output_dir = out_dir,
        )
        assert os.path.exists(os.path.join(out_dir, "latent_mean.csv"))
        assert os.path.exists(os.path.join(out_dir, "latent_std.csv"))
        lat = pd.read_csv(os.path.join(out_dir, "latent_mean.csv"),
                          index_col=0)
        assert lat.shape == (n_cells, 4)   # latent_dim=4

    def test_counterfactual_conditionals(self, tmp_path, trained_model):
        """Permuting conditionals should change predictions."""
        model_dir, expr_path, cond_path, n_cells, n_genes = trained_model

        # Original predictions
        out1 = str(tmp_path / "pred_orig")
        self.predict_cvae(
            model_dir  = model_dir,
            expr_path  = expr_path,
            cond_path  = cond_path,
            output_dir = out1,
        )

        # Build counterfactual: flip all cells to "treated"
        new_cond = pd.read_csv(cond_path, index_col=0)
        new_cond["treatment"] = "treated"
        new_cond_path = str(tmp_path / "new_cond.csv")
        new_cond.to_csv(new_cond_path)

        out2 = str(tmp_path / "pred_cf")
        self.predict_cvae(
            model_dir  = model_dir,
            expr_path  = expr_path,
            cond_path  = new_cond_path,
            output_dir = out2,
        )

        preds1 = pd.read_csv(os.path.join(out1, "predictions.csv"),
                             index_col=0).values
        preds2 = pd.read_csv(os.path.join(out2, "predictions.csv"),
                             index_col=0).values

        # The predictions should differ (at least for some cells)
        assert not np.allclose(preds1, preds2, atol=1e-4)

    def test_prior_sampling(self, tmp_path, trained_model):
        """When no expression is given, model should sample from the prior."""
        model_dir, expr_path, cond_path, n_cells, n_genes = trained_model
        out_dir = str(tmp_path / "pred_prior")
        self.predict_cvae(
            model_dir  = model_dir,
            expr_path  = "",       # no expression → sample from prior
            cond_path  = cond_path,
            output_dir = out_dir,
            n_samples  = 5,
        )
        preds = pd.read_csv(os.path.join(out_dir, "predictions.csv"),
                            index_col=0)
        assert preds.shape == (n_cells, n_genes)

    def test_multiple_samples(self, tmp_path, trained_model):
        """n_samples > 1 should produce valid averaged predictions."""
        model_dir, expr_path, cond_path, n_cells, n_genes = trained_model
        out_dir = str(tmp_path / "pred_multi")
        self.predict_cvae(
            model_dir  = model_dir,
            expr_path  = expr_path,
            cond_path  = cond_path,
            output_dir = out_dir,
            n_samples  = 10,
        )
        preds = pd.read_csv(os.path.join(out_dir, "predictions.csv"),
                            index_col=0)
        assert preds.shape == (n_cells, n_genes)
        assert np.all(np.isfinite(preds.values))
