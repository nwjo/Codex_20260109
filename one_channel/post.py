"""Post-processing utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np


def conversion(c_in: np.ndarray, c_out: np.ndarray) -> np.ndarray:
    return 1.0 - (c_out / (c_in + 1.0e-12))


def cumulative_emissions(time: np.ndarray, flow: np.ndarray, c_out: np.ndarray) -> np.ndarray:
    return np.trapz(flow[:, None] * c_out, time, axis=0)


def summary_metrics(time: np.ndarray, flow: np.ndarray, c_in: np.ndarray, c_out: np.ndarray, species: Iterable[str]) -> Dict[str, float]:
    conv = conversion(c_in, c_out)
    emissions = cumulative_emissions(time, flow, c_out)
    metrics = {}
    for idx, name in enumerate(species):
        metrics[f"conversion_{name}"] = conv[idx]
        metrics[f"emissions_{name}"] = emissions[idx]
    return metrics


def plot_wall_temperature_profiles(
    time: np.ndarray,
    x: np.ndarray,
    t_s: np.ndarray,
    times: Sequence[float],
    outpath: str | Path,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed; skipping wall-temperature plot.")
        return None
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    norm_x = x / x[-1]
    fig, ax = plt.subplots()
    for target in times:
        idx = int(np.argmin(np.abs(time - target)))
        ax.plot(norm_x, t_s[idx], label=f"{time[idx]:.1f} s")
    ax.set_xlabel("Normalized Axial Distance")
    ax.set_ylabel("Wall Temperature (K)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath
