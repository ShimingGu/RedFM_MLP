from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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

def test_physical_robust_scaling_handles_sentinel_and_signed_flux():
    module = load_module()
    table = {
        "FLUX_CMODEL_band": np.array([-99.0, -0.2, 0.0, 0.3, 3.0, 3000.0]),
        "FLUXERR_CMODEL_band": np.array([-99.0, 0.1, 0.2, 0.3, 0.4, 10.0]),
        "RADIUS_KRON_band": np.array([-99.0, 0.0, 0.5, 1.0, 2.0, 100.0]),
    }
    rows = np.arange(6)
    splits = np.array(["train", "train", "train", "train", "val", "test"])
    features, metadata = module.build_gnlll_features(
        table, rows, list(table), ["flux", "error", "radius"], splits,
        mode="physical_robust",
    )
    assert features.shape == (6, 3)
    assert torch.isfinite(features).all()
    assert features.abs().max() <= module.ROBUST_CLIP
    assert torch.all(features[0] == 0)  # exact -99 sentinel is median-space imputation
    assert metadata[0]["transform"] == "asinh"
    assert metadata[1]["transform"] == "log"
    assert metadata[2]["transform"] == "log1p"
    assert metadata[0]["n_missing"] == 1
    assert features[1, 0] < features[2, 0]  # valid negative flux was retained


def test_target_standardization_is_train_fitted_and_reversible():
    module = load_module()
    target = torch.tensor([0.2, 0.6, 1.0, 1.4, 3.0])
    splits = np.array(["train", "train", "train", "train", "test"])
    scaled, metadata = module.scale_target_from_train(
        target, splits, mode="physical_robust",
    )
    train = torch.from_numpy(splits == "train")
    assert torch.allclose(scaled[train].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(scaled[train].std(unbiased=False), torch.tensor(1.0), atol=1e-6)
    recovered = metadata["offset"] + metadata["scale"] * scaled
    assert torch.allclose(recovered, target, atol=1e-6)

def test_two_stage_training_uses_target_transform_on_cpu(tmp_path):
    module = load_module()
    generator = torch.Generator().manual_seed(9)
    n_rows = 24
    mean_x = torch.randn(n_rows, 6, generator=generator)
    error_x = torch.rand(n_rows, 4, generator=generator)
    physical_target = 0.8 + 0.3 * mean_x[:, 0] - 0.1 * mean_x[:, 1]
    splits = np.array(["train"] * 16 + ["val"] * 4 + ["test"] * 4)
    model_target, target_transform = module.scale_target_from_train(
        physical_target, splits, mode="physical_robust",
    )
    args = SimpleNamespace(
        seed=9, device="cpu", mean_hidden_dim=16, variance_hidden_dim=8,
        dropout=0.0, variance_floor=1e-6, train_batch_size=8,
        eval_batch_size=8, learning_rate=1e-2, weight_decay=0.0,
        mean_warmup_epochs=1, gnlll_epochs=1,
    )
    model, result = module.train_branch(
        "test_gnlll", mean_x, error_x, model_target, splits, args, tmp_path,
        target_transform,
        {},
    )
    assert isinstance(model, module.HeteroscedasticPhotoZRegressor)
    assert [row["stage"] for row in result["history"]] == [
        "mean_warmup", "joint_gnlll",
    ]
    assert Path(result["checkpoint_path"]).is_file()
