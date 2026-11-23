"""Checks for the card-approved named checkpoints in src/repa/artifacts.py.

Runs without torch: artifacts falls back to pickle, which is enough to check file layout.
"""
import importlib.util
import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("repa_artifacts", ROOT / "src" / "repa" / "artifacts.py")
artifacts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(artifacts)


def _load(path):
    if "torch" in sys.modules:
        return artifacts.load_checkpoint(path.parent.parent, which=path.stem)
    with open(path, "rb") as handle:
        return pickle.load(handle)


def test_full_name_keeps_step_state_after_next_save(tmp_path):
    first = {"model": 1, "ema": {"w": 1}, "opt": 1, "steps": 100_000}
    artifacts.save_checkpoint(tmp_path, first, ema_name="ema_100k", full_name="full_100k")
    folder = tmp_path / "checkpoints"
    assert _load(folder / "full_100k.pt") == first
    assert _load(folder / "ema_100k.pt") == {"w": 1}

    second = {"model": 2, "ema": {"w": 2}, "opt": 2, "steps": 150_000}
    artifacts.save_checkpoint(tmp_path, second, ema_name="ema_150k")
    assert _load(folder / "last.pt") == second
    assert _load(folder / "full_100k.pt") == first
    assert sorted(p.name for p in folder.iterdir()) == ["ema_100k.pt", "ema_150k.pt", "full_100k.pt", "last.pt"]


@pytest.mark.parametrize("bad", ["full_100", "last", "full-100k", "ema_100k"])
def test_full_name_rejects_other_names(tmp_path, bad):
    with pytest.raises(ValueError):
        artifacts.save_checkpoint(tmp_path, {"ema": {}}, full_name=bad)
