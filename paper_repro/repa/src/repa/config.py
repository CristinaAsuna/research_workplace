"""Load stage defaults from the trunk and optional experiment overrides."""
import argparse
from pathlib import Path

import yaml

from .artifacts import ROOT


def parse_config(parser, stage, argv=None):
    parser.add_argument("--config", default=str(ROOT / "configs" / "trunk.yaml"))
    parser.add_argument("--exp", help="YAML overrides relative to the trunk configuration")
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=str(ROOT / "configs" / "trunk.yaml"))
    bootstrap.add_argument("--exp")
    paths, _ = bootstrap.parse_known_args(argv)

    allowed = {action.dest for action in parser._actions if action.dest != "help"}
    for path in (paths.config, paths.exp):
        if path is None:
            continue
        with Path(path).open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict) or not isinstance(config.get(stage, {}), dict):
            parser.error(f"{path}: expected a mapping for {stage}")
        defaults = config.get(stage, {})
        unknown = defaults.keys() - (allowed - {"config", "exp"})
        if unknown:
            parser.error(f"{path}: unknown {stage} settings: {', '.join(sorted(unknown))}")
        for action in parser._actions:
            if action.dest in defaults and action.choices is not None:
                if defaults[action.dest] not in action.choices:
                    parser.error(f"{path}: invalid {stage}.{action.dest}: {defaults[action.dest]!r}")
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)
