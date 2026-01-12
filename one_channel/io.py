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


def _save_png(array: np.ndarray, path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed; skipping PNG export.")
        return
    fig, ax = plt.subplots()
    if array.ndim == 1:
        ax.plot(array)
        ax.set_xlabel("Index")
        ax.set_ylabel(title)
    else:
        im = ax.imshow(array, aspect="auto", origin="lower")
        fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    png_dir = outdir / "png"
    png_dir.mkdir(exist_ok=True)
    _save_png(time, png_dir / "time.png", "time")
    _save_png(x, png_dir / "x.png", "x")
    _save_png(result.t_s, png_dir / "t_s.png", "solid_temperature")
    _save_png(result.t_g, png_dir / "t_g.png", "gas_temperature")
    _save_png(result.c_g.reshape(result.c_g.shape[0], -1), png_dir / "c_g.png", "gas_species")
    _save_png(result.c_s.reshape(result.c_s.shape[0], -1), png_dir / "c_s.png", "surface_species")
    _save_png(result.rates.reshape(result.rates.shape[0], -1), png_dir / "rates.png", "reaction_rates")
    if result.pressure_drop is not None:
        _save_png(result.pressure_drop, png_dir / "pressure_drop.png", "pressure_drop")
    saved["arrays"] = outdir
    return saved
