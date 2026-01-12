"""Transport correlations and helper functions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import numpy as np


R_GAS = 8.314


@dataclass
class GasProperties:
    cp: Callable[[np.ndarray], np.ndarray]
    lambda_g: Callable[[np.ndarray], np.ndarray]
    mu: Callable[[np.ndarray], np.ndarray]
    diff: Dict[str, Callable[[np.ndarray], np.ndarray]]


def reynolds(rho: np.ndarray, velocity: np.ndarray, dh: float, mu: np.ndarray) -> np.ndarray:
    return rho * velocity * dh / mu


def prandtl(cp: np.ndarray, mu: np.ndarray, lambda_g: np.ndarray) -> np.ndarray:
    return cp * mu / lambda_g


def schmidt(mu: np.ndarray, rho: np.ndarray, diff: np.ndarray) -> np.ndarray:
    return mu / (rho * diff)


def combine_numbers(values: Sequence[np.ndarray], mode: str = "max", power: float = 4.0) -> np.ndarray:
    if mode == "max":
        return np.maximum.reduce(values)
    if mode == "power":
        return np.power(np.sum(np.power(values, power), axis=0), 1.0 / power)
    raise ValueError(f"Unknown combiner: {mode}")


def nusselt_numbers(re: np.ndarray, pr: np.ndarray, dh: float, x: np.ndarray, nu1: float, combiner: str, power: float) -> np.ndarray:
    re_pr = re * pr
    nu2 = 1.615 * (re_pr * dh / x) ** (1.0 / 3.0)
    nu3 = 0.5 * (2.0 / (1.0 + 22.0 * pr)) ** (1.0 / 6.0) * (re_pr * dh / x) ** 0.5
    return combine_numbers([np.full_like(nu2, nu1), nu2, nu3], combiner, power)


def sherwood_numbers(
    re: np.ndarray,
    sc: np.ndarray,
    dh: float,
    x: np.ndarray,
    sh1: float,
    combiner: str,
    power: float,
) -> np.ndarray:
    re_sc = re * sc
    sh2 = 1.615 * (re_sc * dh / x) ** (1.0 / 3.0)
    sh3 = 0.5 * (2.0 / (1.0 + 22.0 * sc)) ** (1.0 / 6.0) * (re_sc * dh / x) ** 0.5
    return combine_numbers([np.full_like(sh2, sh1), sh2, sh3], combiner, power)


def heat_transfer_coeff(nu: np.ndarray, lambda_g: np.ndarray, dh: float) -> np.ndarray:
    return nu * lambda_g / dh


def mass_transfer_coeff(sh: np.ndarray, diff: np.ndarray, dh: float) -> np.ndarray:
    return sh * diff / dh


def ideal_gas_density(p_tot: float, t_g: np.ndarray) -> np.ndarray:
    return p_tot / (R_GAS * t_g)
