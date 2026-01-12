"""Partitioned coupling solver for 1D one-channel monolith model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from .config import ModelConfig
from .kinetics import KineticsConfig, KineticsModel
from .transport import (
    R_GAS,
    GasProperties,
    heat_transfer_coeff,
    ideal_gas_density,
    mass_transfer_coeff,
    nusselt_numbers,
    prandtl,
    reynolds,
    schmidt,
    sherwood_numbers,
)


@dataclass
class SimulationResult:
    time: np.ndarray
    x: np.ndarray
    t_s: np.ndarray
    t_g: np.ndarray
    c_g: np.ndarray
    c_s: np.ndarray
    rates: np.ndarray
    pressure_drop: np.ndarray | None
    diagnostics: Dict[str, List[int]]


class NewtonError(RuntimeError):
    pass


def default_gas_properties(species: Iterable[str]) -> GasProperties:
    def cp(t: np.ndarray) -> np.ndarray:
        return np.full_like(t, 1089.0)

    def lambda_g(t: np.ndarray) -> np.ndarray:
        return np.full_like(t, 0.05)

    def mu(t: np.ndarray) -> np.ndarray:
        return np.full_like(t, 2.0e-5)

    diff = {name: (lambda t: np.full_like(t, 1.0e-5)) for name in species}
    return GasProperties(cp=cp, lambda_g=lambda_g, mu=mu, diff=diff)


def thomas_solver(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = len(d)
    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
    dp[-1] = (d[-1] - a[-1] * dp[-2]) / (b[-1] - a[-1] * cp[-2])
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def damped_newton(
    func: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float,
    max_iter: int,
    damping: float,
    min_damping: float,
    max_backtrack: int,
    bounds: Tuple[float, float],
) -> Tuple[np.ndarray, int]:
    x = x0.copy()
    lower, upper = bounds
    for iteration in range(max_iter):
        fval = func(x)
        norm = np.linalg.norm(fval, ord=np.inf)
        if norm < tol:
            return x, iteration + 1
        jac = np.zeros((len(x), len(x)))
        eps = 1.0e-7
        for j in range(len(x)):
            xp = x.copy()
            xp[j] = min(max(xp[j] + eps, lower), upper)
            jac[:, j] = (func(xp) - fval) / eps
        try:
            dx = np.linalg.solve(jac, -fval)
        except np.linalg.LinAlgError as exc:
            raise NewtonError("Newton Jacobian solve failed") from exc
        step = damping
        accepted = False
        for _ in range(max_backtrack):
            x_trial = np.clip(x + step * dx, lower, upper)
            trial_norm = np.linalg.norm(func(x_trial), ord=np.inf)
            if trial_norm <= norm:
                x = x_trial
                accepted = True
                break
            step *= 0.5
            if step < min_damping:
                break
        if not accepted:
            x = np.clip(x + min_damping * dx, lower, upper)
    raise NewtonError("Newton solver did not converge")


class OneChannelSolver:
    def __init__(self, config: ModelConfig, gas_properties: GasProperties | None = None) -> None:
        self.config = config
        self.geometry = config.geometry
        self.solver = config.solver
        self.transfer = config.transfer
        self.species = config.species["names"]
        self.inlet = config.inlet
        self.boundaries = config.boundaries
        self.n_species = len(self.species)
        self.gas_properties = gas_properties or default_gas_properties(self.species)
        kin_conf = KineticsConfig(
            species=self.species,
            k0=np.asarray(config.kinetics["k0"], dtype=float),
            ea=np.asarray(config.kinetics["ea"], dtype=float),
            ka=np.asarray(config.kinetics["ka"], dtype=float),
            dha=np.asarray(config.kinetics["dha"], dtype=float),
            dh_reactions=np.asarray(config.kinetics["dh_reactions"], dtype=float),
            kp_params=config.kinetics["kp_params"],
            hc_weights=config.species.get("hc_weights", {}),
        )
        if kin_conf.k0[7] < 1.0:
            print("Warning: k0,8 units may require scaling; using as-is.")
        self.kinetics = KineticsModel(kin_conf)

    def _inlet_state(self, t: float) -> Tuple[float, np.ndarray, float]:
        temp = self.inlet["temperature"]
        flow = self.inlet["mass_flow"]
        comp = self.inlet["composition"]
        c_in = np.array([comp[name] for name in self.species], dtype=float)
        return temp, c_in, flow

    def _pressure_drop(self, mu: np.ndarray, velocity: np.ndarray) -> float:
        model = self.geometry.get("pressure_drop")
        if not model:
            return 0.0
        d1 = model.get("D1", 1.0)
        k_r = model.get("k_r", 1.0)
        length = self.geometry["length"]
        dh = self.geometry["hydraulic_diameter"]
        return -d1 * k_r * mu.mean() * velocity.mean() * length / (dh**2)

    def _solve_surface(
        self, c_g: np.ndarray, t_s: float, t_g: float, km: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        a = self.geometry["catalyst_area"]
        s = self.geometry["surface_area"]
        p_tot = self.inlet["pressure"]
        factor = p_tot / (R_GAS * t_g)
        bounds = (0.0, 1.0)

        def residual(c_s: np.ndarray) -> np.ndarray:
            r_i = self.kinetics.species_rates(c_s, t_s, t_g)
            return a * r_i - factor * km * s * (c_g - c_s)

        try:
            c_s, iters = damped_newton(
                residual,
                np.clip(c_g.copy(), *bounds),
                self.solver["newton_tol"],
                self.solver["newton_max_iter"],
                self.solver["newton_damping"],
                self.solver["newton_damping_min"],
                self.solver["newton_backtrack"],
                bounds,
            )
        except NewtonError:
            policy = self.solver.get("newton_fail_policy", "fallback")
            if policy == "raise":
                raise
            c_s = np.clip(c_g.copy(), *bounds)
            iters = self.solver["newton_max_iter"]
        rates = self.kinetics.rates(c_s, t_s, t_g)
        return c_s, rates, iters

    def _march_gas(
        self,
        t_s: np.ndarray,
        inlet_t: float,
        inlet_c: np.ndarray,
        mass_flow: float,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, List[int]]]:
        nx = len(x)
        c_g = np.zeros((nx, self.n_species))
        c_s = np.zeros_like(c_g)
        t_g = np.zeros(nx)
        r_store = np.zeros((nx, 10))
        c_g[0] = inlet_c
        t_g[0] = inlet_t
        diagnostics = {"newton_iters": []}
        alpha = self.solver["axial_relax"]
        cp_g = self.gas_properties.cp
        lambda_g = self.gas_properties.lambda_g
        mu = self.gas_properties.mu
        dh = self.geometry["hydraulic_diameter"]
        area = self.geometry["area"]
        s_area = self.geometry["surface_area"]
        combiner = self.transfer["combiner"]
        power = self.transfer["power"]
        x_min = self.transfer["x_min"]
        for i in range(nx - 1):
            x_eff = max(x[i], x_min)
            tgi = np.array([t_g[i]])
            rho_override = self.geometry.get("rho_g")
            rho = rho_override if rho_override is not None else ideal_gas_density(self.inlet["pressure"], tgi)[0]
            mu_g = mu(tgi)[0]
            cp = cp_g(tgi)[0]
            lam = lambda_g(tgi)[0]
            vel = mass_flow / (rho * area)
            re = reynolds(np.array([rho]), np.array([vel]), dh, np.array([mu_g]))
            pr = prandtl(np.array([cp]), np.array([mu_g]), np.array([lam]))
            nu = nusselt_numbers(re, pr, dh, np.array([x_eff]), self.transfer["nu1"], combiner, power)
            h = heat_transfer_coeff(nu, np.array([lam]), dh)[0]
            sc = np.array(
                [
                    schmidt(np.array([mu_g]), np.array([rho]), self.gas_properties.diff[name](tgi))
                    for name in self.species
                ]
            ).flatten()
            sh = sherwood_numbers(re, sc, dh, np.array([x_eff]), self.transfer["sh1"], combiner, power)
            diff = np.array([self.gas_properties.diff[name](tgi) for name in self.species]).flatten()
            km = mass_transfer_coeff(sh, diff, dh)
            c_s_i, rates, iters = self._solve_surface(c_g[i], t_s[i], t_g[i], km)
            c_s[i] = c_s_i
            r_store[i] = rates
            diagnostics["newton_iters"].append(iters)
            dx = x[i + 1] - x[i]
            rhs_c = -(km * area * s_area) * (c_g[i] - c_s[i])
            c_next = c_g[i] + dx * rhs_c / (mass_flow / rho)
            c_g[i + 1] = (1 - alpha) * c_g[i] + alpha * c_next
            rhs_t = -(h * area * s_area) * (t_g[i] - t_s[i])
            t_next = t_g[i] + dx * rhs_t / (mass_flow * cp)
            t_g[i + 1] = (1 - alpha) * t_g[i] + alpha * t_next
        c_s[-1] = c_s[-2]
        r_store[-1] = r_store[-2]
        return t_g, c_g, c_s, r_store, diagnostics

    def _solve_solid(
        self,
        t_s_prev: np.ndarray,
        t_s_guess: np.ndarray,
        t_g: np.ndarray,
        r_store: np.ndarray,
        dt: float,
        dx: float,
    ) -> np.ndarray:
        nx = len(t_s_prev)
        rho_s = self.geometry.get("rho_s", 2500.0)
        cp_s = self.geometry.get("cp_s")
        cp_coeffs = self.geometry.get("cp_s_coeffs")
        lambda_s = self.geometry.get("lambda_s", 2.5)
        epsilon = self.geometry["void_fraction"]
        s_area = self.geometry["surface_area"]
        a = self.geometry["catalyst_area"]
        h = 15.0
        dh_reac = self.kinetics.config.dh_reactions
        source = a * np.sum((-dh_reac) * r_store, axis=1)
        main = np.zeros(nx)
        lower = np.zeros(nx - 1)
        upper = np.zeros(nx - 1)
        coeff = lambda_s * (1 - epsilon) / dx**2
        temp_clip = self.solver.get("temperature_clip", {"min": 250.0, "max": 2000.0})
        t_s_guess = np.asarray(t_s_guess, dtype=float)
        t_s_guess = np.clip(t_s_guess, temp_clip["min"], temp_clip["max"])
        if callable(cp_s):
            cp_vals = cp_s(t_s_guess)
        elif cp_coeffs is not None:
            a = float(cp_coeffs["a"])
            b = float(cp_coeffs["b"])
            c = float(cp_coeffs["c"])
            cp_vals = a + b * t_s_guess + c / (t_s_guess**2)
        else:
            cp_vals = np.full_like(t_s_guess, 900.0 if cp_s is None else cp_s)
        storage = (1 - epsilon) * rho_s * cp_vals / dt
        for i in range(nx):
            main[i] = storage[i] + 2 * coeff + h * s_area
        for i in range(nx - 1):
            lower[i] = -coeff
            upper[i] = -coeff
        rhs = storage * t_s_prev + h * s_area * t_g + source
        bc = self.boundaries.get("solid", {}).get("type", "adiabatic")
        if bc == "adiabatic":
            main[0] = storage[0] + coeff + h * s_area
            main[-1] = storage[-1] + coeff + h * s_area
        elif bc == "dirichlet":
            bc_vals = self.boundaries.get("solid", {}).get("values", {})
            t0 = bc_vals.get("left", t_s_guess[0])
            tL = bc_vals.get("right", t_s_guess[-1])
            main[0] = 1.0
            rhs[0] = t0
            upper[0] = 0.0
            main[-1] = 1.0
            rhs[-1] = tL
            lower[-1] = 0.0
        else:
            raise ValueError("Unknown solid boundary condition")
        return thomas_solver(lower, main, upper, rhs)

    def run(self) -> SimulationResult:
        nx = self.solver["nx"]
        dt = self.solver["dt"]
        t_final = self.solver["t_final"]
        x = np.linspace(0.0, self.geometry["length"], nx)
        dx = x[1] - x[0]
        times = np.arange(0.0, t_final + dt, dt)
        t_s_init = self.solver.get("initial_solid_temp")
        if t_s_init is None:
            t_s_init = self.inlet["temperature"]
        t_s = np.full((len(times), nx), t_s_init)
        t_g = np.zeros_like(t_s)
        c_g = np.zeros((len(times), nx, self.n_species))
        c_s = np.zeros_like(c_g)
        r_store = np.zeros((len(times), nx, 10))
        pressure = np.zeros(len(times))
        diagnostics = {"outer_iters": [], "newton_iters": []}
        for n in range(1, len(times)):
            inlet_t, inlet_c, mass_flow = self._inlet_state(times[n])
            t_s_guess = t_s[n - 1].copy()
            t_g_n = t_g[n - 1].copy()
            c_g_n = c_g[n - 1].copy()
            c_s_n = c_s[n - 1].copy()
            rates_n = r_store[n - 1].copy()
            for outer in range(self.solver["outer_max_iter"]):
                t_g_n, c_g_n, c_s_n, rates_n, diag = self._march_gas(
                    t_s_guess, inlet_t, inlet_c, mass_flow, x
                )
                t_s_new = self._solve_solid(t_s[n - 1], t_s_guess, t_g_n, rates_n, dt, dx)
                if np.max(np.abs(t_s_new - t_s_guess)) < self.solver["outer_tol"]:
                    t_s[n] = t_s_new
                    t_g[n] = t_g_n
                    c_g[n] = c_g_n
                    c_s[n] = c_s_n
                    r_store[n] = rates_n
                    diagnostics["outer_iters"].append(outer + 1)
                    diagnostics["newton_iters"].extend(diag["newton_iters"])
                    break
                relax = self.solver.get("outer_relax", 1.0)
                t_s_guess = relax * t_s_new + (1.0 - relax) * t_s_guess
            else:
                policy = self.solver.get("outer_fail_policy", "raise")
                if policy == "raise":
                    raise RuntimeError("Outer iteration failed to converge")
                print("Warning: outer iteration did not converge; accepting last iterate.")
                t_s[n] = t_s_new
                t_g[n] = t_g_n
                c_g[n] = c_g_n
                c_s[n] = c_s_n
                r_store[n] = rates_n
            mu = self.gas_properties.mu(t_g[n])
            rho = ideal_gas_density(self.inlet["pressure"], t_g[n])
            vel = mass_flow / (rho * self.geometry["area"])
            pressure[n] = self._pressure_drop(mu, vel)
        return SimulationResult(
            time=times,
            x=x,
            t_s=t_s,
            t_g=t_g,
            c_g=c_g,
            c_s=c_s,
            rates=r_store,
            pressure_drop=pressure,
            diagnostics=diagnostics,
        )
