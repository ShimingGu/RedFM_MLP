"""GPU reservations must prevent overlap and release on rejected launches."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


spec = importlib.util.spec_from_file_location(
    "survey_gpu_guard",
    Path(__file__).resolve().parents[1] / "scripts/qwen_survey_gpu_guard.py",
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_overlapping_reservations_rejected_but_disjoint_allowed(tmp_path):
    memory = {"total": 12.0, "used": 0.0, "free": 12.0}
    with patch.object(guard, "memory_info", return_value=memory):
        with guard.reserve_devices(["a", "b"], tmp_path):
            with pytest.raises(RuntimeError, match="already reserved"):
                with guard.reserve_devices(["b", "c"], tmp_path):
                    pytest.fail("Overlapping slice reservation succeeded")
            with guard.reserve_devices(["c", "d"], tmp_path) as available:
                assert set(available) == {"c", "d"}
        with guard.reserve_devices(["a", "b"], tmp_path):
            pass


def test_busy_slice_rejected_and_partial_reservation_released(tmp_path):
    def memory(device):
        return {"total": 12.0, "used": 7.0, "free": 5.0 if device == "b" else 12.0}

    with patch.object(guard, "memory_info", side_effect=memory):
        with pytest.raises(RuntimeError, match="at least 8 GiB free"):
            with guard.reserve_devices(["a", "b"], tmp_path):
                pytest.fail("Busy slice reservation succeeded")
        with guard.reserve_devices(["a", "c"], tmp_path):
            pass


@pytest.mark.parametrize("resolved", ['["a", "a"]', '["a"]', "[]"])
def test_device_resolution_rejects_aliases_and_inaccessible_devices(resolved):
    with patch.object(guard.subprocess, "check_output", return_value=resolved):
        with pytest.raises(RuntimeError, match="two distinct accessible"):
            guard.resolve_devices("0,1")


def test_visibility_is_set_before_driver_initialization():
    with patch.object(
        guard.subprocess, "check_output", return_value='["a", "b"]'
    ) as resolve:
        assert guard.resolve_devices("MIG-a,MIG-b") == ["a", "b"]
        assert resolve.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "MIG-a,MIG-b"
