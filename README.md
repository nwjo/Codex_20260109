# One-Channel Monolith Catalytic Converter Solver

This package provides a production-ready 1D transient solver for a single monolith channel using a
partitioned coupling strategy between gas-phase axial marching, surface algebraic balances, and a
solid-phase transient energy equation with axial conduction.

## Model overview

**Surface mass-transfer–reaction balance** (species `i`):

```
  a(x) R_i(c_s, T_s) = (P_tot / (R_g T_g)) k_{m,i} S (c_{g,i} - c_{s,i})
```

**Solid-phase energy** (implicit in time, 2nd order in space):

```
  (1-ε) ρ_s d(C_{p,s} T_s)/dt = λ_s (1-ε) d²T_s/dx² + h S (T_g - T_s)
    + a(x) Σ_j (-ΔH_j) r_j(c_s, T_s, T_g)
```

**Gas-phase species marching**:

```
  (w_g / ρ_g) dc_{g,i}/dx + k_{m,i} A S (c_{g,i} - c_{s,i}) = 0
```

**Gas-phase energy marching**:

```
  w_g C_{p,g} dT_g/dx + h A S (T_g - T_s) = 0
```

Heat/mass transfer coefficients use combined Nusselt/Sherwood correlations with configurable
combiner (`max` or power mean).

## Package layout

```
one_channel/
  config.py    # YAML/JSON config loader + validation
  transport.py # properties, Re/Pr/Sc, Nu/Sh, transfer coefficients
  kinetics.py  # 10-reaction global kinetics + inhibition terms
  solver.py    # partitioned coupling solver + tridiagonal solve
  io.py        # inlet CSV reader + output writers
  post.py      # conversion/emissions metrics
  examples/
```

## Installation

```
pip install -e .
```

## Run the example

```
python -m one_channel.examples.run_transient
```

Results are saved as NumPy arrays in the configured output directory.

To set the initial solid temperature separately from the inlet gas temperature, specify
`solver.initial_solid_temp` in the configuration (defaults to `inlet.temperature`).

The model represents a single channel. If your inlet mass flow is for the full monolith,
set `geometry.channels` to the total channel count (e.g., 2790 for 300 CPSI with 60 cm²
frontal area) so the solver automatically uses `mass_flow / channels` for the
single-channel calculation.

Solid heat capacity can be specified as a constant (`geometry.cp_s`) or as a coefficient set
`geometry.cp_s_coeffs` using `a + b T_s + c / T_s^2` (SI units) to match common monolith fits.

## Example config

An example YAML configuration is provided at:

```
one_channel/examples/example_config.yaml
```

Note: YAML treats unquoted `NO` as a boolean in some parsers; the example quotes `"NO"` to ensure it
is parsed as a species name.

## Notes

- Newton iteration diagnostics and outer-iteration counts are tracked in the solver result.
- A pressure-drop model is supported through the `geometry.pressure_drop` configuration block.
