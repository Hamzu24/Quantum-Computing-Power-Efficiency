# Clifford Quantum Volume Metric

A Quantum Volume benchmark adapted for Clifford-only circuits, compatible with Stim stabilizer simulation.

## Overview

Standard Quantum Volume (QV) uses random SU(4) two-qubit gates, which include non-Clifford operations. Since Stim only simulates stabilizer (Clifford) circuits, this metric adapts QV to use random 2-qubit Clifford gates instead.

## Protocol

1. **Circuit Generation**: Create random Clifford circuits of depth d on d qubits (square circuits)
   - Each layer applies random 2-qubit Cliffords to random qubit pairings
   - 2-qubit Cliffords decomposed as: single-qubit Clifford + entangler + single-qubit Clifford

2. **Ideal Simulation**: Compute heavy outputs using Stim's efficient stabilizer sampling
   - Heavy outputs = bitstrings with probability > median

3. **Noisy Simulation**: Run circuit with noise model and measure heavy output probability (HOP)

4. **QV Criterion**: QV = 2^d achieved if HOP > 2/3 with statistical confidence

## Usage

```bash
python metric_executor.py tutorials/qec_metrics/clifford_quantum_volume/clifford_quantum_volume.py
```

## Configuration

Uses `configs.json`:
- `num_qubits`: Circuit width and depth (limited to 12 for performance)
- `selected_noise_model`: Noise model to apply

## Output

- `PERF_VALUE`: Mean heavy output probability
- `quantum_volume`: Achieved QV (2^n if passed, 0 otherwise)
- `pass_rate`: Fraction of trials with HOP > 2/3

## Notes

- Clifford QV is less demanding than standard QV since Clifford gates form a smaller group
- Useful for benchmarking QEC-relevant Clifford fidelity under noise
- At high qubit counts, expect low HOP due to error accumulation
