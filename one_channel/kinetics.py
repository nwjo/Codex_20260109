"""Kinetics module for surface reactions and inhibition terms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping

import numpy as np

from .transport import R_GAS


@dataclass
class KineticsConfig:
    species: List[str]
    k0: np.ndarray
    ea: np.ndarray
    ka: np.ndarray
    dha: np.ndarray
    dh_reactions: np.ndarray
    kp_params: Dict[str, float]
    hc_weights: Dict[str, float]


class KineticsModel:
    """Implements 10-reaction global kinetics with inhibition."""

    def __init__(self, config: KineticsConfig) -> None:
        self.config = config
        self.species_index = {name: idx for idx, name in enumerate(config.species)}
        self.stoich = self._build_stoich()

    def _build_stoich(self) -> np.ndarray:
        n_species = len(self.config.species)
        stoich = np.zeros((n_species, 10))
        def s(name: str, coeff: float, rxn: int) -> None:
            stoich[self.species_index[name], rxn] = coeff
        s("CO", -1.0, 0)
        s("O2", -0.5, 0)
        s("CO2", 1.0, 0)
        s("C3H6", -1.0, 1)
        s("O2", -4.5, 1)
        s("CO2", 3.0, 1)
        s("H2O", 3.0, 1)
        s("CH4", -1.0, 2)
        s("O2", -2.0, 2)
        s("CO2", 1.0, 2)
        s("H2O", 2.0, 2)
        s("H2", -1.0, 3)
        s("O2", -0.5, 3)
        s("H2O", 1.0, 3)
        s("CO", -1.0, 4)
        s("H2O", -1.0, 4)
        s("CO2", 1.0, 4)
        s("H2", 1.0, 4)
        s("C3H6", -1.0, 5)
        s("H2O", -3.0, 5)
        s("CO", 3.0, 5)
        s("H2", 6.0, 5)
        s("CH4", -1.0, 6)
        s("H2O", -1.0, 6)
        s("CO", 1.0, 6)
        s("H2", 3.0, 6)
        s("CO", -1.0, 7)
        s("NO", -1.0, 7)
        s("CO2", 1.0, 7)
        s("N2", 0.5, 7)
        s("C3H6", -1.0, 8)
        s("NO", -9.0, 8)
        s("H2O", 3.0, 8)
        s("CO2", 3.0, 8)
        s("N2", 4.5, 8)
        s("H2", -1.0, 9)
        s("NO", -1.0, 9)
        s("H2O", 1.0, 9)
        s("N2", 0.5, 9)
        return stoich

    def _adsorption_constants(self, t_s: float) -> np.ndarray:
        return self.config.ka * np.exp(-self.config.dha / (R_GAS * t_s))

    def _kp(self, t_s: float) -> float:
        return np.exp(self.config.kp_params["a"] + self.config.kp_params["b"] / t_s)

    def _hc_proxy(self, c_s: np.ndarray) -> float:
        hc = 0.0
        for name, weight in self.config.hc_weights.items():
            hc += weight * c_s[self.species_index[name]]
        return hc

    def inhibition_terms(self, c_s: np.ndarray, t_s: float) -> tuple[float, float]:
        k = self._adsorption_constants(t_s)
        c_co = c_s[self.species_index["CO"]]
        c_no = c_s[self.species_index["NO"]]
        hc = self._hc_proxy(c_s)
        g1 = (
            t_s
            * (1.0 + k[0] * c_co + k[1] * hc) ** 2
            * (1.0 + k[2] * c_co**2 * hc**2)
            * (1.0 + k[3] * c_no**0.7)
        )
        g2 = t_s ** (-0.17) * (t_s + k[4] * c_co) ** 2
        return g1, g2

    def equilibrium_modifiers(self, c_s: np.ndarray, t_s: float) -> tuple[float, float, float]:
        kp = self._kp(t_s)
        c = {name: c_s[idx] for name, idx in self.species_index.items()}
        eq5 = 1.0 - (c["CO2"] * c["H2"]) / (c["CO"] * c["H2O"] * kp + 1.0e-12)
        eq6 = 1.0 - (c["CO"] ** 3 * c["H2"] ** 6) / (
            c["C3H6"] * c["H2O"] ** 3 * kp + 1.0e-12
        )
        eq7 = 1.0 - (c["CO"] * c["H2"] ** 3) / (c["CH4"] * c["H2O"] * kp + 1.0e-12)
        return eq5, eq6, eq7

    def rates(self, c_s: np.ndarray, t_s: float, t_g: float) -> np.ndarray:
        k = self.config.k0 * np.exp(-self.config.ea / (R_GAS * t_g))
        g1, g2 = self.inhibition_terms(c_s, t_s)
        eq5, eq6, eq7 = self.equilibrium_modifiers(c_s, t_s)
        c = {name: c_s[idx] for name, idx in self.species_index.items()}
        m1 = 3500.0
        m = -0.19 * (1.0 - 6.26 * np.exp(-m1 * c["CO"]))
        rates = np.zeros(10)
        rates[0] = k[0] * c["CO"] * c["O2"] / g1
        rates[1] = k[1] * c["C3H6"] * c["O2"] / g1
        rates[2] = k[2] * c["CH4"] * c["O2"] / g1
        rates[3] = k[3] * c["H2"] * c["O2"] / g1
        rates[4] = k[4] * c["CO"] * c["H2O"] * eq5 / g1
        rates[5] = k[5] * c["C3H6"] * c["H2O"] * eq6 / g1
        rates[6] = k[6] * c["CH4"] * c["H2O"] * eq7 / g1
        rates[7] = k[7] * c["CO"] * np.sqrt(c["NO"]) / g2
        rates[8] = k[8] * c["C3H6"] * c["NO"] / g1
        rates[9] = k[9] * c["H2"] * c["NO"] / g1
        rates *= 10.0 ** m
        return rates

    def species_rates(self, c_s: np.ndarray, t_s: float, t_g: float) -> np.ndarray:
        r = self.rates(c_s, t_s, t_g)
        return self.stoich @ r
