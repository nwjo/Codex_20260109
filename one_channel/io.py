"""I/O utilities for one-channel solver."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .solver import SimulationResult


def read_inlet_csv(path: str | Path, species: Iterable[str]) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"time", "temperature", "mass_flow", "pressure"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing inlet columns: {sorted(missing)}")
    for name in species:
        if name not in data.columns:
            raise ValueError(f"Missing species column: {name}")
    return data


def write_results(result: SimulationResult, outdir: str | Path) -> Dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    time = result.time
    x = result.x
    saved = {}
    np.save(outdir / "time.npy", time)
    np.save(outdir / "x.npy", x)
    np.save(outdir / "t_s.npy", result.t_s)
    np.save(outdir / "t_g.npy", result.t_g)
    np.save(outdir / "c_g.npy", result.c_g)
    np.save(outdir / "c_s.npy", result.c_s)
    np.save(outdir / "rates.npy", result.rates)
    if result.pressure_drop is not None:
        np.save(outdir / "pressure_drop.npy", result.pressure_drop)
    saved["arrays"] = outdir
    return saved
