"""Post-processing utilities."""
from __future__ import annotations

from typing import Dict, Iterable

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
