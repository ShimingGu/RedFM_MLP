from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def load_module():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "iotfm_mlp_gnlll.py"
    spec = importlib.util.spec_from_file_location("iotfm_mlp_gnlll", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gnlll_name_is_expanded_and_model_variance_is_positive():
    module = load_module()
    assert module.GNLLL_EXPANSION == "Gaussian negative log-likelihood loss"
    model = module.HeteroscedasticPhotoZRegressor(
        66, 55, mean_hidden_dim=16, variance_hidden_dim=8, dropout=0.0,
    )
    mean, variance = model(torch.randn(4, 66), torch.randn(4, 55))
    assert mean.shape == (4,)
    assert variance.shape == (4,)
    assert torch.all(variance > 0)


def test_select_gnlll_columns_and_gaussian_pdf():
    module = load_module()
    bands = [f"band{i}" for i in range(11)]
    flux = [f"{prefix}{band}" for prefix in module.FLUX_PREFIXES for band in bands]
    errors = [f"{prefix}{band}" for prefix in module.FLUX_ERROR_PREFIXES for band in bands]
    radii = [f"RADIUS_KRON_{band}" for band in bands]
    selected = module.select_gnlll_columns(["ID", *flux, *errors, *radii, "ZPHOT"])
    assert tuple(map(len, selected)) == (55, 11, 55)

    raw = {
        "mean": torch.tensor([0.5, 1.0]),
        "variance": torch.tensor([0.04, 0.09]),
        "target": torch.tensor([0.6, 0.8]),
        "metrics": {},
    }
    edges = torch.linspace(0, 2, 21)
    evaluation = module.gaussian_evaluation(raw, edges, (edges[:-1] + edges[1:]) / 2)
    assert evaluation["pz"].shape == (2, 20)
    assert torch.allclose(evaluation["pz"].sum(dim=1), torch.ones(2))
