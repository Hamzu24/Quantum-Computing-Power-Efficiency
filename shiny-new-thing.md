# Plan: Per-Qubit Parameter Derivation in DefaultBuilder

## Context

Currently, DefaultBuilder uses **global** hardware parameters (E_c, E_J, R_n, y0, gamma_psi_base, T_env) from `hardware_constants.json` and fits some of them via numerical optimization (`optimise_parameters`). It then applies multiplicative adjustment factors (adj_T1, adj_T2) per qubit to compensate for the global approximation.

The physics dictates that these parameters vary per qubit (different junction parameters, frequencies, etc.). By deriving them from IBM's per-qubit calibration data (frequency, anharmonicity, T1, T2, readout errors), every parameter gets a **closed-form per-qubit expression** — eliminating the need for numerical optimization and adjustment factors entirely.

**Design decisions:**
- **Median T_env**: Compute T_env per qubit from readout data, then take the median. Use that single median T_env for all qubits when computing γ₁⁰ and γ_φ_base.
- **Fallback for missing anharmonicity**: If a backend lacks anharmonicity data (3 legacy backends: almaden, rochester, tokyo), fall back to global E_c from hardware_constants.json. E_J and downstream params are still computed per-qubit using the qubit's frequency.

## Files to Modify

1. **`_helpers/builders/default_builder.py`** — Major rewrite (primary file)
2. **`_helpers/builders/base.py`** — Remove `optimise_parameters` from Builder Protocol; update ConfigTracker
3. **`_helpers/builders/builder_wrapper.py`** — Replace `optimise_parameters()` call with `initialize_per_qubit_params()`
4. **`_helpers/registry.py`** — Remove `OptimiserRegistry` and `optimiser_registry`
5. **`qiskit_backend_configs/hardware_constants.json`** — Remove params that are now always derived; keep E_c/E_J/R_n as optional fallbacks

## Implementation Steps

### Step 1: Update `hardware_constants.json`

Remove parameters that are always derived per-qubit (`y0`, `T_env`, `gamma_psi_base`). Keep `E_c`, `E_J`, `R_n` as **optional fallbacks** for backends without anharmonicity data:

```json
{
    "modern": {
        "delta": 2.88e-23,
        "ymxc_y0": 0.9,
        "subgap_transparency": 0.00005,
        "builder_class": "DefaultBuilder",
        "E_c": 1.524e-25,
        "E_J": 9.944e-24,
        "R_n": 5000
    }
}
```

### Step 2: Rewrite `default_builder.py`

#### 2a: Update `__init__` and `is_valid`

**`is_valid`** — new required params (global only):
```python
required_params = ["delta", "ymxc_y0", "subgap_transparency"]
```

Remove from required: `y0`, `T_env`, `gamma_psi_base`. `E_c`, `E_J`, `R_n` are optional (only needed as fallbacks).

#### 2b: Add two-phase initialization

Replace `optimise_parameters()` with `initialize_per_qubit_params()`:

**Phase 1 — Compute median T_env** (same logic as current `calculate_T_env` but returns it rather than storing globally):
```python
def _compute_median_T_env(self):
    qubit_paths = self.json_manager.get_qubit_paths()
    T_env_estimates = []
    gamma_env_ratio = 1 - self.config.get("ymxc_y0")

    for qb_path in qubit_paths:
        p_e = self.json_manager.find_value_with_units("prob_meas1_prep0", qb_path)
        frequency = self.json_manager.find_value_with_units("frequency", qb_path)
        w_ge = 2 * pi * frequency

        if p_e is None or frequency is None or p_e >= 0.5 or p_e <= 0:
            continue

        n_eff = p_e / (1 - 2 * p_e)
        arg = 1 + gamma_env_ratio / n_eff
        if arg <= 1:
            continue

        T_env_i = hbar * w_ge / (k * log(arg))
        T_env_estimates.append(T_env_i)

    if not T_env_estimates:
        logging.warning("Could not estimate T_env from any qubit. Using init_T as fallback.")
        return self.init_T

    return median(T_env_estimates)
```

**Phase 2 — Derive per-qubit params** for every qubit, using the median T_env:
```python
def initialize_per_qubit_params(self):
    self.median_T_env = self._compute_median_T_env()
    logging.debug(f"Median T_env = {self.median_T_env}")

    self._qubit_params_cache = {}
    for qb_path in self.json_manager.get_qubit_paths():
        self._qubit_params_cache[qb_path] = self._derive_qubit_params(qb_path)
```

#### 2c: Add `_derive_qubit_params(qb_path)` method

Follows Steps 1–9 from `parameters.md`:

| Step | Parameter | Formula | Data Source |
|------|-----------|---------|-------------|
| 1 | E_c | `h * abs(anharmonicity_Hz)` — fallback to `self.config["E_c"]` if anharmonicity missing | anharmonicity (per-qubit) or global fallback |
| 2 | E_J | `(h*freq_Hz + E_c)² / (8*E_c)` — fallback to `self.config["E_J"]` if E_c is from fallback and no frequency | frequency (per-qubit), E_c |
| 3 | w_p | `sqrt(8*E_J*E_c) / hbar` | E_J, E_c |
| 4 | R_n | `pi*hbar*delta / (4*e²*E_J)` | E_J, delta (global) |
| 5 | N_e | `(1/zeta) * 1/(2*R_n*g_K)` where `g_K = e²/h` | R_n, zeta (global) |
| 6 | p_e | `prob_meas1_prep0` (direct use) | per-qubit |
| 7 | T_env | `self.median_T_env` (precomputed) | — |
| 8 | y0 (γ₁⁰) | `1 / (T1_meas * (2*n_eff + 1))` where `n_eff = ymxc_y0 * n_BE(w_ge, init_T) + (1-ymxc_y0) * n_BE(w_ge, T_env)` | T1 (per-qubit), T_env |
| 9 | gamma_psi_base | `max(0, 1/T2_meas - 1/(2*T1_meas))` | T1, T2 (per-qubit) |

Returns a dict: `{"E_c", "E_J", "w_p", "R_n", "N_e", "y0", "gamma_psi_base", "T_env", "w_ge"}`.

Uses `self.json_manager.find_value_with_units()` which already handles GHz→Hz conversion via `get_unit_multiplier` from `_helpers/helpers.py`.

#### 2d: Add `get_qubit_params(qb_path)` accessor

```python
def get_qubit_params(self, qb_path):
    return self._qubit_params_cache[qb_path]
```

#### 2e: Update intermediate physics functions

Change signatures from `(T, frequency)` to `(T, qubit_params)`:

- **`n_eff(T, qp)`**: Uses `qp["w_ge"]` and `qp["T_env"]`; `self.config["ymxc_y0"]`
- **`gamma_qp(T, qp)`**: Uses `qp["w_ge"]`, `qp["w_p"]`; `self.config["delta"]`
- **`gamma_psi_qp(T, qp)`**: Uses `qp["w_p"]`, `qp["w_ge"]`, `qp["N_e"]`; `self.config["delta"]`
- **`T1(T, qp)`**: Calls `gamma_qp`, `n_eff`; uses `qp["y0"]`
- **`T_psi(T, qp)`**: Calls `gamma_psi_qp`; uses `qp["gamma_psi_base"]`
- **`T2(T, qp)`**: Calls `T1`, `T_psi`

`x_qp(T)` and `bose_einstein(w, T)` remain unchanged (only use global `delta` / pure math).

#### 2f: Remove `w_p` method

`w_p` is now precomputed in `_derive_qubit_params` and stored in the qubit_params dict. The `w_p(self, T)` method is removed.

#### 2g: Update `F_N` for per-qubit gate fidelity

New signature: `F_N(T, N, gate_length, qubit_params_list)`:

```python
def F_N(self, T, N, gate_length, qubit_params_list):
    d = 2 ** N
    rate_sum = 0
    for qp in qubit_params_list:
        T1_i = self.T1(T, qp)
        T_psi_i = self.T_psi(T, qp)
        rate_sum += 1/T1_i + 1/T_psi_i
    return 1 - (N * gate_length) / (2 * (d + 1)) * rate_sum
```

This replaces the old average-frequency approximation.

#### 2h: Simplify `calculate_qb_config`

```python
def calculate_qb_config(self, control_parameters, qb_path):
    T = get_config_value(control_parameters, "temperature")
    qp = self.get_qubit_params(qb_path)
    T1 = self.T1(T, qp)
    T2 = self.T2(T, qp)
    qb_config = {"T1": T1, "T2": T2}
    self.config_tracker.add_config({"config": qb_config}, "qb")
    return qb_config
```

No more adj_T1/adj_T2 multiplicative factors.

#### 2i: Update `calculate_gate_config`

```python
def calculate_gate_config(self, control_parameters, gate_path):
    gate_param_path = gate_path + "parameters."
    relevant_qubits = self.json_manager.resolve(gate_path + "qubits")
    N = len(relevant_qubits)
    T = get_config_value(control_parameters, "temperature")
    gate_length = self.json_manager.find_value_with_units("gate_length", gate_param_path)

    qubit_params_list = [self.get_qubit_params(f"qubits.[{qb}].") for qb in relevant_qubits]

    gate_error = max(0, min(1, 1 - self.F_N(T, N, gate_length, qubit_params_list)))
    self.config_tracker.add_config({"config": {"gate_error": gate_error}}, "gate")
    return {"gate_error": gate_error}
```

No more average-frequency approximation or gate adjustment factors.

#### 2j: Remove these methods entirely

- `optimise_parameters()` — no longer needed (closed-form derivation replaces optimization)
- `calculate_parameter_error()` — was for computing adj_T1/adj_T2 and optimization
- `total_init_parameter_error()` — was for optimization objective function
- `calculate_T_env()` — replaced by `_compute_median_T_env()`

#### 2k: Update imports

Remove: `scipy.optimize.minimize_scalar`, `_helpers.registry.optimiser_registry`
Keep: `statistics.median` (still needed for median T_env)

### Step 3: Update `base.py`

- Remove `optimise_parameters` from the `Builder` Protocol class
- Simplify `ConfigTracker.log_info()`: remove adj_T1/adj_T2 averaging. Log average computed T1/T2 values instead.

### Step 4: Update `builder_wrapper.py`

Replace:
```python
self.builder.optimise_parameters()
logging.info(f"Optimised parameters. config: {self.builder.config}")
```
With:
```python
self.builder.initialize_per_qubit_params()
logging.info(f"Initialized per-qubit parameters for {self.name}")
```

Remove unused imports: `minimize_scalar`, `deepcopy` (check if still needed elsewhere).

### Step 5: Update `registry.py`

Remove `OptimiserRegistry` class and `optimiser_registry` instance entirely. Keep `CircuitSubmitterRegistry` and `ControlParameterRegistry` unchanged.

## Verification

1. **Sanity check derived values**: Run with sherbrooke backend, log E_J/E_c ratio per qubit (should be ~30-80 for IBM transmons), R_n (should be ~5-10 kOhm)
2. **Run metric**: `python metric_executor.py tutorials/circuit_execution_quality_metrics/quantum_volume/quantum_volume.py --log DEBUG` — verify T1/T2 computations complete without errors
3. **Run optimiser** (temperature sweep): `python optimiser.py tutorials/circuit_execution_quality_metrics/quantum_volume/quantum_volume.py` — verify sweep still works
4. **Test fallback**: Run with a legacy backend (tokyo) to verify fallback to global E_c works
