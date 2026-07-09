"""
configs/dataset_config.py
==========================
Minimal dataset configuration loader for GAN-PH multi-dataset support.

Usage
-----
    from configs.dataset_config import get_config, default

    cfg = default()                    # anode_niysz (no behavior change)
    cfg = get_config("cathode_s1_supercrop")

    NI_VAL       = cfg.phase_value("Ni")    # 255 for anode
    PHASES       = cfg.phases_dict()        # {"Ni": 255, "YSZ": 127, "Pore": 0}
    PHASE_PIXEL  = cfg.phase_pixel_array()  # np.array([255, 127, 0]) channel order
    voxel_um     = cfg.voxel_size_um        # 0.1 for anode; 0.04034 for S1

Design
------
- Channel order in YAML phases list is the softmax channel order [0,1,2].
- Validates: exactly channels 0,1,2; all three roles present; no duplicate bmp_values.
- Absolute paths (source_file) live in configs/local_paths.yaml (gitignored).
  local_paths.yaml maps yaml_name -> absolute path on this machine.
- Scripts pass --dataset-config <name>; omitting the flag gives anode default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import yaml


_CONFIGS_DIR = Path(__file__).parent
_CONFIGS_CACHE: dict[str, "DatasetConfig"] = {}


class DatasetConfig:
    def __init__(self, data: dict, name: str):
        self._d = data
        self._name = name
        self._validate()

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate(self):
        phases = self._d.get("phases", [])
        channels = {p["channel"] for p in phases}
        if channels != {0, 1, 2}:
            raise ValueError(
                f"Config {self._name!r}: phases must have exactly channels "
                f"[0,1,2], got {sorted(channels)}"
            )
        roles = {p.get("role") for p in phases}
        required_roles = {"electron_conductor", "ion_conductor", "gas"}
        if not required_roles.issubset(roles):
            raise ValueError(
                f"Config {self._name!r}: missing roles "
                f"{required_roles - roles} (got {roles})"
            )

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_label_map(self) -> dict:
        return self._d.get("source_label_map", {})

    @property
    def exclude_values(self) -> list[int]:
        return [int(v) for v in self._d.get("exclude_values", [])]

    @property
    def metric_policy(self) -> dict:
        return self._d.get("metric_policy", {})

    # ── Phase accessors ───────────────────────────────────────────────────────

    def phase_value(self, phase_name: str) -> int:
        """Return bmp_value for a phase by name (e.g. 'Ni', 'YSZ', 'Pore')."""
        for p in self._d["phases"]:
            if p["name"] == phase_name:
                return int(p["bmp_value"])
        raise KeyError(f"Phase {phase_name!r} not in config {self._name!r}")

    def phases_dict(self) -> dict[str, int]:
        """Return {phase_name: bmp_value} in channel order."""
        return {
            p["name"]: int(p["bmp_value"])
            for p in sorted(self._d["phases"], key=lambda p: p["channel"])
        }

    def phase_pixel_array(self) -> np.ndarray:
        """Channel-ordered pixel values: index=channel, value=bmp_value."""
        phases = sorted(self._d["phases"], key=lambda p: p["channel"])
        return np.array([p["bmp_value"] for p in phases], dtype=np.uint8)

    def channel_names(self) -> list[str]:
        """Phase names in channel order [ch0, ch1, ch2]."""
        phases = sorted(self._d["phases"], key=lambda p: p["channel"])
        return [p["name"] for p in phases]

    # ── Voxel size ───────────────────────────────────────────────────────────

    @property
    def voxel_size_um(self) -> Optional[float]:
        """Mean voxel size in µm. None if not set."""
        vs = self._d.get("voxel_size_um", {})
        vals = [v for v in [vs.get("x"), vs.get("y"), vs.get("z")] if v is not None]
        return float(sum(vals) / len(vals)) if vals else None

    @property
    def voxel_size_x_um(self) -> Optional[float]:
        return self._d.get("voxel_size_um", {}).get("x")

    @property
    def voxel_size_y_um(self) -> Optional[float]:
        return self._d.get("voxel_size_um", {}).get("y")

    @property
    def voxel_size_z_um(self) -> Optional[float]:
        return self._d.get("voxel_size_um", {}).get("z")

    # ── Source file ───────────────────────────────────────────────────────────

    @property
    def source_file(self) -> Optional[Path]:
        """
        Absolute path to the source TIF, resolved via configs/local_paths.yaml.
        Returns None if local_paths.yaml is absent or has no entry for this config.
        """
        key = self._d.get("source_file_key")
        if key is None:
            return None
        local_paths_file = _CONFIGS_DIR / "local_paths.yaml"
        if not local_paths_file.exists():
            return None
        with open(local_paths_file) as f:
            lp = yaml.safe_load(f) or {}
        path_str = lp.get(key)
        return Path(path_str) if path_str else None

    def __repr__(self):
        vox = self.voxel_size_um
        return (
            f"DatasetConfig({self._name!r}, "
            f"phases={self.channel_names()}, "
            f"voxel_size_um={vox})"
        )


# ── Loader ───────────────────────────────────────────────────────────────────

def get_config(name: str = "anode_niysz") -> DatasetConfig:
    """
    Load a named config from configs/<name>.yaml.

    Parameters
    ----------
    name : str
        Config name, e.g. "anode_niysz", "cathode_s1_supercrop", "cathode_s2".
        Default is "anode_niysz" (the existing Ni-YSZ-Pore anode pipeline).

    Returns
    -------
    DatasetConfig
    """
    if name in _CONFIGS_CACHE:
        return _CONFIGS_CACHE[name]

    yaml_path = _CONFIGS_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        available = [p.stem for p in _CONFIGS_DIR.glob("*.yaml")
                     if p.stem != "local_paths"]
        raise FileNotFoundError(
            f"Config {name!r} not found at {yaml_path}.\n"
            f"Available configs: {available}"
        )

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    cfg = DatasetConfig(data, name)
    _CONFIGS_CACHE[name] = cfg
    return cfg


def default() -> DatasetConfig:
    """Return the default anode config (backward-compatible baseline)."""
    return get_config("anode_niysz")
