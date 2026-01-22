# Grover's Search Algorithm Benchmark

## Overview

This benchmark measures the performance of Grover's quantum search algorithm under noisy conditions. Grover's algorithm provides a quadratic speedup for unstructured search problems, finding a marked item in an unsorted database of N items with O(sqrt(N)) queries instead of O(N).

## Algorithm Description

Grover's search consists of:

1. **Initialization**: Prepare uniform superposition over all N = 2^n states
2. **Oracle**: Phase-flip the marked (target) state: |m> -> -|m>
3. **Diffusion**: Reflect amplitudes about the mean (amplitude amplification)
4. **Repeat**: Apply oracle + diffusion for optimal number of iterations
5. **Measure**: Probability of measuring marked state approaches 1

### Optimal Parameters

For a search space of N items with 1 marked item:
- **Optimal iterations**: k = floor(pi/4 * sqrt(N))
- **Theoretical success probability**: sin^2((2k+1) * arcsin(1/sqrt(N)))

| Qubits | Search Space | Optimal Iterations | Theoretical Success |
|--------|--------------|-------------------|---------------------|
| 2      | 4            | 1                 | 1.000               |
| 3      | 8            | 2                 | 0.945               |
| 4      | 16           | 3                 | 0.961               |
| 5      | 32           | 4                 | 0.962               |
| 6      | 64           | 6                 | 0.966               |

## Performance Metric

**PERF_VALUE**: Mean success probability of measuring the marked state across all trials.

Higher values indicate better performance. The metric compares:
- Ideal (noiseless) success probability
- Noisy simulation success probability

## Usage

### Run with metric_executor

```bash
python metric_executor.py tutorials/circuit_execution_quality_metrics/grovers_search/grovers_search.py
```

### Run with visualization

```bash
python metric_executor.py tutorials/circuit_execution_quality_metrics/grovers_search/grovers_search.py -v
```

### Run with optimizer

```bash
python optimiser.py tutorials/circuit_execution_quality_metrics/grovers_search/grovers_search.py
```

## Configuration

The benchmark uses settings from `configs.json`:

- `num_qubits`: Number of qubits for the search space
- `circuit_optimisation_level`: Transpilation optimization (0-3)

Internal parameters (in script):
- `num_trials`: Number of random target states tested (default: 100)
- `num_shots`: Measurement shots per circuit (default: 1000)

## Output

The benchmark produces:
1. **Console output**: Mean success probabilities for ideal vs noisy
2. **Plots** (saved to benchmark_path):
   - Success probability histograms per qubit count
   - Success probability vs qubits comparison
3. **Results file**: `grovers_results.pkl` with full trial data

## Circuit Structure

For n qubits and k iterations, the circuit depth is approximately:
- Initialization: n Hadamard gates
- Per iteration: O(n) gates for oracle + O(n) gates for diffusion
- Total: O(k * n) ~ O(n * sqrt(2^n))

The oracle uses multi-controlled X (MCX) gates, which decompose into O(n) basic gates.

## Noise Sensitivity

Grover's algorithm is particularly sensitive to:
- **Coherent errors**: Accumulate over iterations, causing over/under-rotation
- **Decoherence**: Amplitude damping destroys superposition
- **Gate errors**: MCX gates require many 2-qubit gates in decomposition

This makes it a good benchmark for testing gate fidelity and circuit depth tolerance.

## References

- Grover, L.K. (1996). "A fast quantum mechanical algorithm for database search"
- Nielsen & Chuang, Chapter 6: "Quantum Search Algorithms"
