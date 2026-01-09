"""Configuration loader and basic validation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import yaml


@dataclass
class ModelConfig:
    geometry: Dict[str, Any] = field(default_factory=dict)
    species: Dict[str, Any] = field(default_factory=dict)
    kinetics: Dict[str, Any] = field(default_factory=dict)
    transfer: Dict[str, Any] = field(default_factory=dict)
    solver: Dict[str, Any] = field(default_factory=dict)
    inlet: Dict[str, Any] = field(default_factory=dict)
    boundaries: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)


DEFAULT_CONFIG: Dict[str, Any] = {
    "geometry": {
        "length": 0.1,
        "area": 0.006,
        "void_fraction": 0.6836,
        "surface_area": 1500.0,
        "catalyst_area": 26895.0,
        "hydraulic_diameter": 1.2124e-3,
        "rho_g": 2500.0,
        "rho_s": 2500.0,
        "cp_s_coeffs": {"a": 1071.0, "b": 0.156, "c": -3.435e7},
        "lambda_s": 1.675,
    },
    "species": {
        "names": [
            "CO",
            "O2",
            "CO2",
            "C3H6",
            "CH4",
            "H2",
            "H2O",
            "NO",
            "N2",
        ],
        "hc_weights": {"C3H6": 1.0, "CH4": 1.0},
    },
    "kinetics": {
        "k0": [1.0e5] * 10,
        "ea": [7.5e4] * 10,
        "ka": [1.0e-3] * 5,
        "dha": [5.0e4] * 5,
        "dh_reactions": [-2.8e5] * 10,
        "kp_params": {"a": 1.0, "b": 0.0},
    },
    "transfer": {
        "nu1": 3.0,
        "sh1": 3.0,
        "combiner": "max",
        "power": 4.0,
        "x_min": 1.0e-6,
    },
    "solver": {
        "nx": 51,
        "dt": 0.05,
        "t_final": 1.0,
        "outer_max_iter": 25,
        "outer_tol": 1.0e-3,
        "axial_relax": 0.6,
        "newton_max_iter": 25,
        "newton_tol": 1.0e-8,
        "newton_damping": 0.5,
        "marcher": "upwind",
    },
    "inlet": {
        "temperature": 500.0,
        "pressure": 101300.0,
        "mass_flow": 0.04,
        "composition": {
            "CO": 0.01,
            "O2": 0.1,
            "CO2": 0.02,
            "C3H6": 0.0,
            "CH4": 0.0,
            "H2": 0.0,
            "H2O": 0.1,
            "NO": 0.001,
            "N2": 0.767,
        },
    },
    "boundaries": {
        "solid": {
            "type": "adiabatic",
        }
    },
    "output": {
        "directory": "output",
    },
}


REQUIRED_SECTIONS = {
    "geometry",
    "species",
    "kinetics",
    "transfer",
    "solver",
    "inlet",
    "boundaries",
    "output",
}

REQUIRED_SPECIES = {"CO", "O2", "CO2", "C3H6", "CH4", "H2", "H2O", "NO", "N2"}


class ConfigError(ValueError):
    """Raised when configuration validation fails."""


def _merge_dicts(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _merge_dicts(base[key], value)
        else:
            base[key] = value


def _validate_required(config: Mapping[str, Any], required: Iterable[str]) -> None:
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"Missing required config sections: {missing}")


def _validate_species(config: Mapping[str, Any]) -> None:
    names = config["species"]["names"]
    if len(set(names)) != len(names):
        raise ConfigError("Species names must be unique")
    missing_required = sorted(REQUIRED_SPECIES.difference(names))
    if missing_required:
        raise ConfigError(f"Species list missing required entries: {missing_required}")
    inlet_comp = config["inlet"]["composition"]
    missing = [name for name in names if name not in inlet_comp]
    if missing:
        raise ConfigError(f"Inlet composition missing species: {missing}")


def load_config(path: str | Path | None = None) -> ModelConfig:
    config: Dict[str, Any] = json.loads(json.dumps(DEFAULT_CONFIG))
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(path.read_text())
        elif path.suffix.lower() == ".json":
            data = json.loads(path.read_text())
        else:
            raise ConfigError("Config file must be .yaml, .yml, or .json")
        if data:
            _merge_dicts(config, data)
    _validate_required(config, REQUIRED_SECTIONS)
    _validate_species(config)
    return ModelConfig(**config)
