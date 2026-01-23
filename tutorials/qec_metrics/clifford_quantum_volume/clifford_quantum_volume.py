"""
Clifford Quantum Volume Metric for Stim

A QEC-compatible Quantum Volume benchmark using only Clifford gates,
suitable for stabilizer simulation with Stim.

Standard QV uses random SU(4) gates, but Stim only supports Clifford operations.
This metric uses random 2-qubit Clifford gates instead, providing a meaningful
benchmark for Clifford circuit fidelity under noise.

The protocol:
1. Generate random Clifford circuits of depth d on d qubits
2. Compute ideal heavy outputs using Stim's tableau simulator
3. Run noisy simulation and measure heavy output probability
4. QV = 2^d where heavy output probability > 2/3

Compatible with metric_executor.py and reads configuration from configs.json.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Set
import numpy as np
import stim

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from _helpers.circuit_submitter import CircuitSubmitter
from _helpers.helpers import read_config, get_num_qubits


# Single-qubit Clifford gates available in Stim
SINGLE_QUBIT_CLIFFORDS = [
    # Identity (do nothing)
    [],
    # Paulis
    ['X'], ['Y'], ['Z'],
    # Hadamard variants
    ['H'],
    ['H', 'S'], ['H', 'S', 'S'],  ['H', 'S', 'S', 'S'],
    # S variants
    ['S'], ['S', 'S'], ['S', 'S', 'S'],
    # H-S combinations (generates all 24 single-qubit Cliffords)
    ['S', 'H'], ['S', 'S', 'H'], ['S', 'S', 'S', 'H'],
    ['H', 'S', 'H'], ['H', 'S', 'S', 'H'], ['H', 'S', 'S', 'S', 'H'],
    ['S', 'H', 'S'], ['S', 'H', 'S', 'S'], ['S', 'H', 'S', 'S', 'S'],
    ['S', 'S', 'H', 'S'], ['S', 'S', 'H', 'S', 'S'],
    ['S', 'S', 'S', 'H', 'S'],
]

# Two-qubit entangling gates available in Stim
TWO_QUBIT_ENTANGLERS = ['CNOT', 'CZ', 'ISWAP', 'SWAP']


def apply_random_single_qubit_clifford(circuit: stim.Circuit, qubit: int, rng: np.random.Generator) -> None:
    """Apply a random single-qubit Clifford gate to a qubit."""
    gates = rng.choice(SINGLE_QUBIT_CLIFFORDS)
    for gate in gates:
        circuit.append(gate, [qubit])


def apply_random_two_qubit_clifford(circuit: stim.Circuit, q0: int, q1: int, rng: np.random.Generator) -> None:
    """
    Apply a random 2-qubit Clifford gate.

    Decomposition: single-qubit Cliffords + entangler + single-qubit Cliffords
    This generates a large subset of the 11,520 two-qubit Clifford gates.
    """
    # Pre-entangler single-qubit Cliffords
    apply_random_single_qubit_clifford(circuit, q0, rng)
    apply_random_single_qubit_clifford(circuit, q1, rng)

    # Random entangling gate (or identity with some probability)
    if rng.random() > 0.2:  # 80% chance of entangling gate
        entangler = rng.choice(TWO_QUBIT_ENTANGLERS)
        circuit.append(entangler, [q0, q1])

    # Post-entangler single-qubit Cliffords
    apply_random_single_qubit_clifford(circuit, q0, rng)
    apply_random_single_qubit_clifford(circuit, q1, rng)


def generate_random_pairing(n_qubits: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Generate a random pairing of qubits for one layer."""
    qubits = list(range(n_qubits))
    rng.shuffle(qubits)
    pairs = []
    for i in range(0, n_qubits - 1, 2):
        pairs.append((qubits[i], qubits[i + 1]))
    return pairs


def create_clifford_qv_circuit(n_qubits: int, depth: int, seed: int) -> stim.Circuit:
    """
    Create a random Clifford circuit for Quantum Volume.

    Args:
        n_qubits: Number of qubits (also determines width)
        depth: Circuit depth (number of layers)
        seed: Random seed for reproducibility

    Returns:
        stim.Circuit with random Clifford gates and measurements
    """
    rng = np.random.default_rng(seed)
    circuit = stim.Circuit()

    # Apply depth layers of random 2-qubit Cliffords
    for _ in range(depth):
        pairs = generate_random_pairing(n_qubits, rng)
        for q0, q1 in pairs:
            apply_random_two_qubit_clifford(circuit, q0, q1, rng)

    # Measure all qubits
    circuit.append('M', range(n_qubits))

    return circuit


def compute_ideal_probabilities(circuit: stim.Circuit, n_qubits: int) -> dict:
    """
    Compute ideal output probabilities using Stim's tableau simulator.

    For Clifford circuits, the output is a stabilizer state which can be
    efficiently sampled. We sample many times to estimate probabilities.
    """
    # Remove measurement from circuit for state analysis
    circuit_no_measure = stim.Circuit()
    for instruction in circuit:
        if instruction.name != 'M':
            circuit_no_measure.append(instruction)

    # Sample from the ideal circuit to get probability distribution
    # For stabilizer states, we need to sample since the state may be
    # a superposition of computational basis states
    sampler = circuit.compile_sampler()
    n_samples = 10000  # Enough samples for good probability estimates
    samples = sampler.sample(n_samples)

    # Convert samples to bitstrings and count
    counts = {}
    for sample in samples:
        bitstring = ''.join('1' if b else '0' for b in sample)
        counts[bitstring] = counts.get(bitstring, 0) + 1

    # Convert to probabilities
    probabilities = {k: v / n_samples for k, v in counts.items()}
    return probabilities


def compute_heavy_outputs(probabilities: dict) -> Set[str]:
    """
    Compute the set of heavy outputs.

    Heavy outputs are bitstrings with probability greater than the median.
    For an ideal QV circuit, heavy outputs should occur with probability > 2/3.
    """
    if not probabilities:
        return set()

    probs = list(probabilities.values())
    median_prob = np.median(probs)

    heavy = {bitstring for bitstring, prob in probabilities.items() if prob > median_prob}
    return heavy


def calculate_heavy_output_probability(counts: dict, heavy_outputs: Set[str]) -> float:
    """Calculate the probability of measuring a heavy output."""
    total = sum(counts.values())
    if total == 0:
        return 0.0

    heavy_count = sum(counts.get(h, 0) for h in heavy_outputs)
    return heavy_count / total


def run_single_qv_trial(
    submitter: CircuitSubmitter,
    n_qubits: int,
    depth: int,
    shots: int,
    seed: int
) -> dict:
    """
    Run a single QV trial.

    Returns:
        Dict with trial results including heavy output probability
    """
    # Create random circuit
    circuit = create_clifford_qv_circuit(n_qubits, depth, seed)

    # Compute ideal heavy outputs
    ideal_probs = compute_ideal_probabilities(circuit, n_qubits)
    heavy_outputs = compute_heavy_outputs(ideal_probs)

    # Run noisy simulation
    result = submitter.submit_circuits(shots, stim_circuits=[circuit], print_summary=False)
    noisy_counts = result.tasks[0].result().measurement_counts

    # Calculate heavy output probability
    hop = calculate_heavy_output_probability(noisy_counts, heavy_outputs)

    return {
        'seed': seed,
        'n_heavy_outputs': len(heavy_outputs),
        'n_unique_outputs': len(ideal_probs),
        'heavy_output_probability': hop,
        'passed': hop > 2/3
    }


def run_clifford_qv_benchmark(
    submitter: CircuitSubmitter,
    n_qubits: int,
    num_trials: int = 100,
    shots_per_trial: int = 1000,
    base_seed: int = 42
) -> dict:
    """
    Run the full Clifford Quantum Volume benchmark.

    Args:
        submitter: Circuit submitter for execution
        n_qubits: Number of qubits (depth = n_qubits for square circuits)
        num_trials: Number of random circuits to test
        shots_per_trial: Shots per circuit
        base_seed: Base random seed

    Returns:
        Dict with benchmark results
    """
    depth = n_qubits  # Square circuits for QV

    trial_results = []
    hops = []

    for trial in range(num_trials):
        seed = base_seed + trial
        result = run_single_qv_trial(submitter, n_qubits, depth, shots_per_trial, seed)
        trial_results.append(result)
        hops.append(result['heavy_output_probability'])

    mean_hop = np.mean(hops)
    std_hop = np.std(hops)
    num_passed = sum(1 for r in trial_results if r['passed'])
    pass_rate = num_passed / num_trials

    # QV is achieved if mean HOP > 2/3 with high confidence
    # Using 2-sigma confidence interval
    hop_lower_bound = mean_hop - 2 * std_hop / np.sqrt(num_trials)
    qv_achieved = hop_lower_bound > 2/3

    return {
        'n_qubits': n_qubits,
        'depth': depth,
        'num_trials': num_trials,
        'shots_per_trial': shots_per_trial,
        'mean_heavy_output_probability': mean_hop,
        'std_heavy_output_probability': std_hop,
        'hop_lower_bound_2sigma': hop_lower_bound,
        'num_passed': num_passed,
        'pass_rate': pass_rate,
        'qv_achieved': qv_achieved,
        'quantum_volume': 2 ** n_qubits if qv_achieved else 0,
        'log2_quantum_volume': n_qubits if qv_achieved else 0,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Clifford Quantum Volume Metric (Stim Stabilizer Simulation)")
    print("=" * 70)

    # Read configuration
    configs = read_config()
    n_qubits = get_num_qubits()

    # Limit qubits for reasonable runtime
    max_qubits = min(n_qubits, 12)
    if n_qubits > max_qubits:
        print(f"Note: Limiting to {max_qubits} qubits (config had {n_qubits})")
        n_qubits = max_qubits

    num_trials = 100
    shots = 1000

    device_name = "stim_sim"
    submitter = CircuitSubmitter(
        "clifford_quantum_volume",
        device_name
    )
    filepath = submitter.benchmark_path

    print(f"\n[Clifford QV Test: {n_qubits} qubits, depth {n_qubits}]")
    print(f"  Trials: {num_trials}, Shots per trial: {shots}")
    print(f"  Target: Heavy Output Probability > 2/3 = 0.667")

    import time
    start = time.time()

    result = run_clifford_qv_benchmark(
        submitter,
        n_qubits=n_qubits,
        num_trials=num_trials,
        shots_per_trial=shots
    )

    elapsed = time.time() - start
    result['elapsed_seconds'] = elapsed

    # Print results
    print(f"\n[Results]")
    print(f"  Mean Heavy Output Probability: {result['mean_heavy_output_probability']:.4f} +/- {result['std_heavy_output_probability']:.4f}")
    print(f"  HOP Lower Bound (2-sigma): {result['hop_lower_bound_2sigma']:.4f}")
    print(f"  Pass Rate: {result['pass_rate']:.2%} ({result['num_passed']}/{num_trials})")
    print(f"  QV Achieved: {result['qv_achieved']}")
    if result['qv_achieved']:
        print(f"  Quantum Volume: {result['quantum_volume']} (log2 = {result['log2_quantum_volume']})")
    print(f"  Time: {elapsed:.2f}s")

    # Set PERF_VALUE to mean heavy output probability
    perf_value = result['mean_heavy_output_probability']
    os.environ["PERF_VALUE"] = str(perf_value)

    # Power consumption summary
    print("\n[Power Consumption]")
    total, staggered = submitter.get_power_consumption()
    print(f"  Total gates: {dict(submitter.total_gates)}")

    # Summary
    print("\n" + "=" * 70)
    print(f"PERF_VALUE (Mean HOP at {n_qubits} qubits): {perf_value:.4f}")
    if result['qv_achieved']:
        print(f"Clifford Quantum Volume: {result['quantum_volume']}")
    else:
        print(f"Clifford QV NOT achieved (HOP lower bound {result['hop_lower_bound_2sigma']:.4f} <= 0.667)")
    print("=" * 70)

    # Save results
    import json
    results_path = f"{filepath}/results.json"
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {results_path}")
