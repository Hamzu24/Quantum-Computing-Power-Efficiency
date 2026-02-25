# Mirror Benchmarking

Randomized mirror benchmarking measures circuit execution quality by building random Clifford circuits of variable depth, appending the inverse, and measuring the probability of returning to |00...0⟩.

## Protocol

Each mirror circuit at depth `d`:
1. Build forward circuit U with `d` layers, where each layer applies:
   - One random single-qubit Clifford (from the 24-element group) per qubit
   - A random maximal matching of 2-qubit gates from the backend connectivity graph
2. Append U† (the inverse of U)
3. Measure all qubits

The survival probability P(|00...0⟩) is computed for each circuit. An exponential decay `a * alpha^d + b` is fitted across depths to extract the decay parameter `alpha`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `depths` | `[1, 2, 4, 8, 16, 32]` | Number of layers per circuit |
| `num_trials` | `10` | Random circuits per depth |
| `num_shots` | `100` | Measurement shots per circuit |

## Usage

```bash
python metric_executor.py tutorials/circuit_execution_quality_metrics/mirror_benchmarking/mirror_benchmarking.py
```

With visual plots:
```bash
python metric_executor.py tutorials/circuit_execution_quality_metrics/mirror_benchmarking/mirror_benchmarking.py -v
```

## Output

- `PERF_VALUE`: Mean survival probability across all depths and trials
- `mirror_results.pkl`: Full results dictionary
- `mirror_benchmarking.png`: Survival probability vs depth plot with exponential fit
