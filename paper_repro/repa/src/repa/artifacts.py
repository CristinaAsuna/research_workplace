"""The only way this project writes checkpoints, caches and smoke outputs (AGENTS.md §5).

- save_checkpoint: a run keeps last.pt and best.pt, nothing else. No step_*, no smoke.pt,
  no extra best variants unless the card lists them and you pass name=.
- cache_dir: one directory per cache key, so the same cache is never stored twice.
- run_dir: formal runs go to runs/<id>/, smoke runs to scratch/smoke/<id>/.

Set PROJECT_RUNS_ROOT to put runs/ on another disk (for example a cluster runs_root).
"""
import hashlib
import json
import os
import pickle
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")


def runs_root():
    return Path(os.environ.get("PROJECT_RUNS_ROOT") or ROOT / "runs")


def run_dir(exp_id, smoke=False):
    """Directory for one experiment's run. Smoke runs never land in runs/."""
    if not _NAME.fullmatch(exp_id):
        raise ValueError("exp_id must match [a-z0-9][a-z0-9_-]*")
    path = ROOT / "scratch" / "smoke" / exp_id if smoke else runs_root() / exp_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dump(state, path):
    try:
        import torch
    except ImportError:
        with open(path, "wb") as handle:
            pickle.dump(state, handle)
    else:
        torch.save(state, path)


def save_checkpoint(run, state, metric=None, *, lower_is_better=True, name=None, ema_name=None, full_name=None):
    """Overwrite last.pt; also overwrite best.pt (or best_<name>.pt) when metric improves.

    An optional card-approved ema_name saves the bare EMA weights alongside last.
    An optional card-approved full_name keeps this step's full state (same bytes as last.pt)
    as <full_name>.pt, so later runs can fork from it.
    Returns True when the best checkpoint was updated.
    """
    if name is not None and not _NAME.fullmatch(name):
        raise ValueError("name must match [a-z0-9][a-z0-9_-]*")
    if ema_name is not None and not re.fullmatch(r"ema_[0-9]{3,}k", ema_name):
        raise ValueError("ema_name must be an EMA milestone such as ema_050k")
    if full_name is not None and not re.fullmatch(r"full_[0-9]{3,}k", full_name):
        raise ValueError("full_name must be a full-state milestone such as full_200k")
    folder = Path(run) / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    best_stem = "best" if name is None else "best_" + name
    record_path = folder / (best_stem + ".json")
    record = json.loads(record_path.read_text()) if record_path.exists() else None
    improved = metric is not None and (
        record is None or (metric < record["metric"] if lower_is_better else metric > record["metric"]))
    tmp = folder / ".last.pt.tmp"
    _dump(state, tmp)
    if improved:
        best_tmp = folder / ".best.pt.tmp"
        if best_tmp.exists():
            best_tmp.unlink()
        try:
            os.link(tmp, best_tmp)  # same bytes, no second copy on disk
        except OSError:
            _dump(state, best_tmp)
        os.replace(best_tmp, folder / (best_stem + ".pt"))
        record_tmp = folder / ".best.json.tmp"
        record_tmp.write_text(json.dumps({"metric": metric, "lower_is_better": lower_is_better}))
        os.replace(record_tmp, record_path)
    if ema_name is not None:
        ema_tmp = folder / f".{ema_name}.pt.tmp"
        _dump(state["ema"], ema_tmp)
        os.replace(ema_tmp, folder / f"{ema_name}.pt")
    if full_name is not None:
        full_tmp = folder / f".{full_name}.pt.tmp"
        if full_tmp.exists():
            full_tmp.unlink()
        try:
            os.link(tmp, full_tmp)  # same bytes as last.pt; survives the next last.pt replace
        except OSError:
            _dump(state, full_tmp)
        os.replace(full_tmp, folder / f"{full_name}.pt")
    os.replace(tmp, folder / "last.pt")
    return improved


def load_checkpoint(run, which="best"):
    path = Path(run) / "checkpoints" / (which + ".pt")
    try:
        import torch
    except ImportError:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    return torch.load(path, map_location="cpu", weights_only=False)


def cache_dir(kind, **key):
    """Directory for a cache identified by kind and key, e.g. cache_dir("latents", data="val", vae="mse", size=256).

    The same key always maps to the same directory; check is_complete() before rebuilding it.
    """
    if not _NAME.fullmatch(kind):
        raise ValueError("kind must match [a-z0-9][a-z0-9_-]*")
    if not key:
        raise ValueError("cache_dir needs at least one key, e.g. data=, model=, size=")
    text = json.dumps(key, sort_keys=True, default=str)
    path = runs_root() / "_cache" / kind / hashlib.sha256(text.encode()).hexdigest()[:12]
    path.mkdir(parents=True, exist_ok=True)
    key_file = path / "key.json"
    if key_file.exists():
        if json.loads(key_file.read_text()) != json.loads(text):
            raise RuntimeError("cache key collision in {}".format(path))
    else:
        key_file.write_text(text)
    return path


def is_complete(path):
    return (Path(path) / ".complete").exists()


def mark_complete(path):
    (Path(path) / ".complete").touch()
