"""Load configs/trunk.yaml and overlay an experiment's configs/exp/<id>.yaml (deep merge)."""
import copy
import subprocess
from pathlib import Path

import yaml

from .artifacts import ROOT

TRUNK = ROOT / "configs" / "trunk.yaml"


def _merge(base, over):
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(trunk=TRUNK, exp=None):
    cfg = yaml.safe_load(Path(trunk).read_text()) or {}
    if exp:
        cfg = _merge(cfg, yaml.safe_load(Path(exp).read_text()) or {})
    return cfg


def git_commit():
    """HEAD commit of the repo, with a '-dirty' suffix when src/configs/tests have local changes."""
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src", "configs", "tests"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("-dirty" if dirty else "")
