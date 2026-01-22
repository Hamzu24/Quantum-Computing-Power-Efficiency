"""
Simple Stim Test Metric

A basic QEC-style metric to verify the Stim stabilizer simulation integration.
Tests GHZ state fidelity with and without noise, demonstrating Stim's scalability.

Compatible with metric_executor.py and reads configuration from configs.json.
"""

import os
import sys
from pathlib import Path
from _helpers.circuit_submitter import CircuitSubmitter
import numpy as np
import stim
from _helpers.helpers import read_config, get_num_qubits
from _helpers.nm_helper import craft_noise_model
import logging

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def create_ghz_circuit(n_qubits: int) -> stim.Circuit:
    """
    Create a GHZ state preparation and measurement circuit.

    |GHZ> = (|00...0> + |11...1>) / sqrt(2)
    """
    circuit = stim.Circuit()
    circuit.append("H", [0])
    for i in range(n_qubits - 1):
        circuit.append("CNOT", [i, i + 1])
    circuit.append("M", range(n_qubits))
    return circuit


def create_repetition_code_circuit(n_data_qubits: int, rounds: int = 1) -> stim.Circuit:
    """
    Create a simple repetition code circuit with syndrome measurements.

    This is a basic QEC circuit structure demonstrating Stim's capabilities.
    """
    circuit = stim.Circuit()

    # Data qubits: 0 to n_data_qubits-1
    # Ancilla qubits: n_data_qubits to 2*n_data_qubits-2
    n_ancilla = n_data_qubits - 1

    for _ in range(rounds):
        # Syndrome extraction: measure ZZ stabilizers
        for i in range(n_ancilla):
            ancilla = n_data_qubits + i
            circuit.append("CNOT", [i, ancilla])
            circuit.append("CNOT", [i + 1, ancilla])

        # Measure ancillas
        circuit.append("MR", range(n_data_qubits, n_data_qubits + n_ancilla))

    # Final data qubit measurement
    circuit.append("M", range(n_data_qubits))

    return circuit


def calculate_ghz_fidelity(counts: dict, n_qubits: int) -> float:
    """
    Calculate fidelity of measured state to ideal GHZ state.

    Ideal GHZ has 50% |00...0> and 50% |11...1>.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0

    all_zeros = '0' * n_qubits
    all_ones = '1' * n_qubits

    p_zeros = counts.get(all_zeros, 0) / total
    p_ones = counts.get(all_ones, 0) / total

    # Fidelity to ideal GHZ: should have only |00...0> and |11...1>
    # Perfect fidelity = 1.0 when p_zeros + p_ones = 1.0
    return p_zeros + p_ones


def run_ghz_fidelity_test(
    submitter: StimCircuitSubmitter,
    n_qubits: int,
    num_trials: int = 100,
    shots_per_trial: int = 1000
) -> dict:
    """
    Run GHZ fidelity test across multiple trials.
    """
    fidelities = []

    for _ in range(num_trials):
        circuit = create_ghz_circuit(n_qubits)
        result = submitter.submit_circuits(shots_per_trial, [circuit], print_summary=False)
        counts = result.tasks[0].result().measurement_counts
        fidelity = calculate_ghz_fidelity(counts, n_qubits)
        fidelities.append(fidelity)

    return {
        'n_qubits': n_qubits,
        'num_trials': num_trials,
        'shots_per_trial': shots_per_trial,
        'mean_fidelity': np.mean(fidelities),
        'std_fidelity': np.std(fidelities),
        'min_fidelity': np.min(fidelities),
        'max_fidelity': np.max(fidelities),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Stim Stabilizer Simulation - GHZ Fidelity Metric")
    print("=" * 70)

    # Read configuration
    configs = read_config()
    n_qubits = get_num_qubits()
    num_trials = 100
    shots = 250

    device_name="noisy_sim"
    # Create submitter
    submitter = CircuitSubmitter(
        "simple_stim_test",
        device_name
    )
    filepath = submitter.benchmark_path

    # Run tests at each qubit count
    results = {}
    for n in qubit_counts:
        print(f"\n[Testing {n} qubits]")
        print(f"  Trials: {num_trials}, Shots per trial: {shots}")

        import time
        start = time.time()
        result = run_ghz_fidelity_test(
            submitter,
            n_qubits=n,
            num_trials=num_trials,
            shots_per_trial=shots
        )
        elapsed = time.time() - start

        results[n] = result
        results[n]['elapsed_seconds'] = elapsed

        print(f"  Mean GHZ Fidelity: {result['mean_fidelity']:.4f} ± {result['std_fidelity']:.4f}")
        print(f"  Time: {elapsed:.2f}s")

    # Use the primary qubit count result for PERF_VALUE
    primary_result = results[qubit_counts[0]]
    perf_value = primary_result['mean_fidelity']
    os.environ["PERF_VALUE"] = str(perf_value)

    # Power consumption summary
    print("\n[Power Consumption]")
    total, staggered = submitter.get_power_consumption()
    print(f"  Total gates: {dict(submitter.total_gates)}")

    # Summary
    print("\n" + "=" * 70)
    print(f"PERF_VALUE (GHZ Fidelity at {qubit_counts[0]} qubits): {perf_value:.4f}")
    print("=" * 70)

    # Save results
    import json
    results_path = f"{filepath}/results.json"
    with open(results_path, 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to: {results_path}")
