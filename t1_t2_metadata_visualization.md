# Plan: T1/T2 Metadata Extraction and Visualization

## Context

The codebase models temperature-dependent noise on superconducting qubit backends using a per-qubit parameter architecture. `DefaultBuilder` derives per-qubit physics parameters (E_c, E_J, w_p, R_n, N_e, y0, gamma_psi_base, T_env, w_ge) from calibration data during `initialize_per_qubit_params()`, caching them in `_qubit_params_cache`. At each temperature iteration, `calculate_qb_config()` computes T1(T, qp) and T2(T, qp) per qubit using these cached parameters and stores the results in `ConfigTracker.qb_config_infos`.

However, this per-qubit T1/T2 data is discarded after the backend properties JSON is written — the `BuilderWrapper` is a local variable in `nm_helper.build_backend()` (line 451) and goes out of scope. We need to propagate it upstream so the optimiser can visualize how T1/T2 distributions evolve across the temperature sweep, providing physical insight alongside the existing performance-vs-temperature plots.

Only non-Pauli fake-backend noise models are in scope. Custom registry-based noise models and Pauli mode return empty metadata.

### Per-qubit architecture notes

- Each qubit has its own derived parameters (`_qubit_params_cache[qb_path]`), so T1/T2 values genuinely vary across qubits at each temperature — this is what makes the box-and-whisker visualization meaningful.
- `ConfigTracker` is instantiated fresh with each `DefaultBuilder` (which is created per `BuilderWrapper`), so `qb_config_infos` contains exactly one entry per qubit per call to `build_backend()`. No clearing needed.
- The config entries have the simplified format `{"config": {"T1": ..., "T2": ...}}` (no adjustment/actual/calculated fields from the old global-parameter architecture).

---

## Step 1: Add `get_T1_T2_values()` to ConfigTracker

**File**: `_helpers/builders/base.py`

- Add `get_T1_T2_values(self) -> dict` to the `ConfigTracker` class. Extracts T1/T2 from the already-stored `qb_config_infos` list. Each entry has `info["config"]["T1"]` and `info["config"]["T2"]` (set at `default_builder.py:254`).
  ```python
  def get_T1_T2_values(self) -> dict:
      if not self.qb_config_infos:
          return {}
      return {
          "T1_values": [info["config"]["T1"] for info in self.qb_config_infos],
          "T2_values": [info["config"]["T2"] for info in self.qb_config_infos],
      }
  ```

**Justification**: All builders already use `ConfigTracker` to store per-qubit data. Adding the extraction here keeps it simple without modifying the `Builder` Protocol. `BuilderWrapper` accesses it via `self.builder.config_tracker`.

---

## Step 2: Add `get_noise_model_metadata()` to BuilderWrapper; modify `build_backend()` return

**File**: `_helpers/builders/builder_wrapper.py`

- Add method:
  ```python
  def get_noise_model_metadata(self) -> dict:
      if hasattr(self.builder, 'config_tracker'):
          return self.builder.config_tracker.get_T1_T2_values()
      return {}
  ```
- Modify `build_backend()` to return the metadata dict (currently returns `None` implicitly). No existing caller captures the return value, so this is backward-compatible.
  ```python
  def build_backend(self, control_parameters) -> dict:
      # ... existing code ...
      return self.get_noise_model_metadata()
  ```
- **Bug fix (line 60)**: Change `if (self.builder, "config_tracker"):` to `if hasattr(self.builder, "config_tracker"):` — the current code creates a tuple which is always truthy.

---

## Step 3: Thread metadata through `nm_helper.py` (4 functions)

**File**: `_helpers/nm_helper.py`

All changes are in return signatures. The 3-tuple is `(noise_model, backend, metadata)`.

### 3a. `build_backend()` (module-level, line 449)
Capture and return metadata from `BuilderWrapper.build_backend()`:
```python
def build_backend(config, backend_name) -> dict:
    init_control_parameters = config.get("init_control_parameters")
    builder = BuilderWrapper(backend_name, init_control_parameters)
    control_parameters = get_control_parameters(config)
    metadata = builder.build_backend(control_parameters)
    return metadata
```

### 3b. `_prepare_fake_backend()` (line 66)
Return `(backend, backend_name, metadata)` instead of `(backend, backend_name)`:
```python
def _prepare_fake_backend(config):
    backend_name = config.get("name")
    exists = config_exists(backend_name)
    fetch_config_files(backend_name, exit_if_unavailable=exists)
    matching_class = get_backend_class(config, backend_name)
    create_backend_symlinks(config, matching_class)
    metadata = build_backend(config, backend_name)
    backend = matching_class()
    return backend, backend_name, metadata
```

### 3c. `nm_from_fake_backend()` (line 80)
Unpack 3-tuple, pass metadata through:
```python
def nm_from_fake_backend(config):
    backend, backend_name, metadata = _prepare_fake_backend(config)
    # ... existing noise model creation ...
    return noise_model, backend, metadata
```

### 3d. `nm_from_fake_backend_no_twirl()` (line 241)
Same pattern: unpack 3-tuple at line 258, return 3-tuple at line 388.
```python
def nm_from_fake_backend_no_twirl(config):
    backend, backend_name, metadata = _prepare_fake_backend(config)
    # ... existing coherent error noise model creation ...
    return noise_model, backend, metadata
```

### 3e. `pauli_nm_from_fake_backend()` (line 96)
Skip metadata per requirements (Pauli mode excluded):
```python
def pauli_nm_from_fake_backend(config):
    backend, backend_name, _metadata = _prepare_fake_backend(config)
    # ... existing code ...
    return noise_model, backend, {}
```

### 3f. `craft_noise_model()` (line 40)
Fake backend paths already return 3-tuples from above. For registry-based NMs, wrap the existing 2-tuple:
```python
if config_type in noise_model_registry:
    wrapper = NoiseModelWrapper(config)
    if pauli_mode:
        nm, backend = wrapper.build_pauli()
        return nm, backend, {}
    nm, backend = wrapper.build()
    return nm, backend, {}
```

**Justification**: `NoiseModelWrapper.build()` and `build_pauli()` continue to return 2-tuples internally. Only `craft_noise_model` wraps with `{}`. This avoids touching the `NoiseModelFactory` Protocol and all registered implementations.

---

## Step 4: Store and expose metadata in EnhancedCircuitSubmitter

**File**: `_helpers/enhanced_circuit_submitter.py`

- In `_setup_noise_model()` (line 75): unpack the new 3-tuple from `craft_noise_model`:
  ```python
  noise_model_instance, nm_backend, nm_metadata = craft_noise_model(noise_model_specs, self.pauli_mode)
  self.nm_backend = nm_backend
  self.nm_metadata = nm_metadata
  ```
- Initialize `self.nm_metadata = {}` at the top of `_setup_noise_model()` (so it's always defined even if the method returns early for non-noisy devices).
- Add public method:
  ```python
  def get_noise_model_metadata(self) -> dict:
      """Return noise model metadata (T1/T2 per-qubit values) for upstream storage."""
      return getattr(self, 'nm_metadata', {})
  ```

**Justification**: The submitter already stores `nm_backend`. Storing `nm_metadata` alongside it is the natural place. The method returns opaque data — callers don't need to know it contains T1/T2.

---

## Step 5: Pass metadata through `metric_executor.run_metric()`

**File**: `metric_executor.py`

- After retrieving the submitter from the registry (line 21), call `get_noise_model_metadata()`:
  ```python
  submitter = submitter_registry.get_submitter("noisy_sim")
  nm_metadata = {}
  if submitter is not None and hasattr(submitter, 'get_noise_model_metadata'):
      nm_metadata = submitter.get_noise_model_metadata()
  ```
- Include `"nm_metadata": nm_metadata` in both return paths (with and without consumption).

**Justification**: `metric_executor` doesn't interpret the metadata (no T1/T2 knowledge). It just passes through "additional data to store".

---

## Step 6: Collect metadata and create plots in `optimiser.py`

**File**: `optimiser.py`

### 6a. Modify `optimise()` signature
Add `display_metadata: bool = True` parameter.

### 6b. Collect metadata per iteration
Add `metadata_per_iteration = []` list. In the loop, append `output.get("nm_metadata", {})`.

### 6c. Create metadata box-and-whisker plot
New function `create_metadata_plot(temps, metadata_per_iteration, metric_name)`:
- Uses `matplotlib.pyplot.boxplot()` with `patch_artist=True`, `showmeans=True`
- Two side-by-side subplots: T1 (left, blue boxes) and T2 (right, yellow boxes)
- X-axis: temperature in mK; Y-axis: values in microseconds
- Clean box-and-whisker only (no overlaid data points): box = Q1-Q3, line = median, diamond = mean, whiskers = min/max
- Works well even with 127 qubits (Sherbrooke) since the box summarizes the distribution
- Returns `None` if no T1/T2 data present (e.g. registry-based NMs returning `{}`)

### 6d. Save metadata plot
Use existing `save_plot()` with `_T1T2` suffix appended to filename:
- Saves to `images/` with filename `{num_qb}qb_{metric_name}_{nm_name}_T1T2.png`

### 6e. Update return value
Add `"nm_metadata": metadata_per_iteration` and `"metadata_figure": metadata_fig` to the return dict.

---

## Step 7: Store metadata in `experiment_runner.py`

**File**: `experiment_runner.py`

### 7a. In `run_trial()`
- Capture `nm_metadata = output.get("nm_metadata", [])` and `metadata_fig = output.get("metadata_figure")` from optimise output.
- For single runs: default to `nm_metadata = []`, `metadata_fig = None`.
- Save metadata figure: `self.save_plot(metadata_fig, f"trial{N}_T1T2_{label}.png", output_dir)`.
- Add to result dict: `"nm_metadata": nm_metadata`, `"metadata_plot_path": str(path)`.

### 7b. In experiment_data.json
Each trial now includes:
```json
{
  "nm_metadata": [
    {"T1_values": [5.2e-05, 4.8e-05, ...], "T2_values": [7.1e-05, 6.9e-05, ...]},
    {"T1_values": [4.9e-05, 4.5e-05, ...], "T2_values": [6.8e-05, 6.5e-05, ...]},
    ...
  ],
  "metadata_plot_path": "experiment_results/experiment_0/trials/trial0_T1T2_4qb, fake_backend (sherbrooke).png"
}
```
Each list entry corresponds to one temperature iteration. The T1/T2 arrays within each entry have one value per qubit (reflecting the per-qubit parameter architecture). For non-fake-backend NMs, entries are `{}`.

---

## Implementation Order (by dependency)

1. `base.py` — ConfigTracker.get_T1_T2_values() method
2. `builder_wrapper.py` — Add extraction method, modify build_backend return, fix hasattr bug
3. `nm_helper.py` — Thread 3-tuple through build_backend, _prepare_fake_backend, nm_from_*, craft_noise_model
4. `enhanced_circuit_submitter.py` — Unpack 3-tuple, store metadata, expose getter
5. `metric_executor.py` — Pass metadata through
6. `optimiser.py` — Collect metadata, create box-and-whisker plots, save images
7. `experiment_runner.py` — Store in JSON, save figures
8. Tests — Update mocks for 3-tuple returns

---

## Verification

1. **Unit**: Run existing tests to confirm 3-tuple change doesn't break (mocks need updating)
2. **Integration**: Run `python optimiser.py quantum_volume -s` with a fake_backend config. Verify:
   - `images/` directory contains both `*_T1T2.png` and regular performance plot
   - T1T2 plot shows box-and-whisker with correct number of temperature points
   - Each box represents the distribution across all qubits at that temperature (per-qubit variation)
3. **Experiment runner**: Run a single experiment. Verify `experiment_data.json` contains `nm_metadata` with T1/T2 arrays and `metadata_plot_path`
4. **Non-fake backend**: Run with a registry noise model (e.g., `simple_nm`). Verify empty metadata flows through without errors and no T1T2 plot is created
