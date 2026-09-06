"""Config loader. All tunables live in v5/configs/default.yaml."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "v5" / "configs" / "default.yaml"


def load_config(path: Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(cfg: dict) -> dict:
    """Return a copy with absolute paths under cfg['paths']."""
    out = copy.deepcopy(cfg)
    for key, value in out["paths"].items():
        p = Path(value)
        out["paths"][key] = str(p if p.is_absolute() else ROOT / p)
    return out
