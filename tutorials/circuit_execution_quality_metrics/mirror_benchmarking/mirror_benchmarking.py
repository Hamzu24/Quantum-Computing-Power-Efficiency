# This code is part of QCMet.
#
# (C) Copyright 2024 National Physical Laboratory and National Quantum Computing Centre
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""
Randomized Mirror Benchmarking

Measures circuit execution quality by building random Clifford circuits at a
single depth, appending the inverse, and measuring the probability of
returning to |00...0>.

Each mirror circuit at depth d:
1. Build forward circuit U with d layers, where each layer consists of:
   - One random single-qubit Clifford per qubit
   - A random maximal matching of 2-qubit gates from the backend connectivity
2. Append U† (the inverse of U)
3. Measure all qubits
4. Performance = P(|00...0>) averaged over trials
"""

import os
import sys
import pathlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford
from qiskit.synthesis import synth_clifford_full
from tqdm.auto import tqdm
import logging

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))
from _helpers.circuit_submitter import CircuitSubmitter
from _helpers.helpers import read_config, get_num_qubits
from _helpers.json_manager import JsonManager
from _helpers.constants import TWO_QUBIT_GATES


def get_connectivity_graph(backend_name):
    """Extract 2-qubit connectivity edges from the backend props JSON.

    Loads props_{backend_name}.json via JsonManager, iterates the gates section,
    and collects all entries where len(qubits) == 2. Returns deduplicated edges.
    """
    backend_folder = os.environ.get("BACKEND_CONFIGS_FOLDER", "qiskit_backend_configs/")
    props_path = os.path.join(backend_folder, backend_name, f"props_{backend_name}.json")

    jm = JsonManager(props_path)
    gates = jm.resolve("gates.")

    edges = set()
    for gate_entry in gates:
        qubits = gate_entry["qubits"]
        if len(qubits) == 2:
            edge = tuple(sorted(qubits))
            edges.add(edge)

    return list(edges)


def random_maximal_matching(edges, n_qubits, rng):
    """Find a random maximal matching from the connectivity edges.

    Shuffles the edge list randomly, then greedily picks edges whose qubits
    are not yet used. Returns a list of (q0, q1) pairs.
    """
    shuffled = list(edges)
    rng.shuffle(shuffled)

    used_qubits = set()
    matching = []
    for q0, q1 in shuffled:
        if q0 not in used_qubits and q1 not in used_qubits:
            matching.append((q0, q1))
            used_qubits.add(q0)
            used_qubits.add(q1)

    return matching


def build_mirror_circuit(n_qubits, n_layers, connectivity_edges, two_qb_gate_name, rng):
    """Build a mirror circuit: U followed by U†.

    For each layer of the forward circuit U:
      - Apply a random single-qubit Clifford per qubit (from the 24-element group)
      - Apply a random maximal matching of the native 2-qubit gate
      - Add a barrier

    Then compose with the inverse to form U · U†, and measure all qubits.
    """
    qc = QuantumCircuit(n_qubits)

    for _ in range(n_layers):
        for qubit in range(n_qubits):
            cliff = random_clifford(1)
            cliff_circ = synth_clifford_full(cliff, method="AG")
            qc.compose(cliff_circ, qubits=[qubit], inplace=True)

        matching = random_maximal_matching(connectivity_edges, n_qubits, rng)
        for q0, q1 in matching:
            if q0 < n_qubits and q1 < n_qubits:
                if two_qb_gate_name == "cx":
                    qc.cx(q0, q1)
                elif two_qb_gate_name == "cz":
                    qc.cz(q0, q1)
                elif two_qb_gate_name == "ecr":
                    qc.ecr(q0, q1)
                else:
                    qc.cx(q0, q1)

        qc.barrier()

    qc.compose(qc.inverse(), inplace=True)
    qc.measure_all()

    return qc


def run_mirror_benchmark(
    num_trials=10,
    num_shots=1000,
    circuit_submitter=None,
    optimization_level=3,
):
    """Run mirror benchmarking at a single circuit depth.

    The depth is read from the "mirror_depth" key in configs.json.

    Args:
        num_trials: Number of random mirror circuits to generate.
        num_shots: Number of measurement shots per circuit.
        circuit_submitter: CircuitSubmitter for noisy simulation.
        optimization_level: Qiskit transpilation optimization level.

    Returns:
        Dictionary with results including survival probabilities.
    """
    if circuit_submitter is None:
        circuit_submitter = CircuitSubmitter("mirror_benchmarking", "noisy_sim")

    num_qubits = get_num_qubits()

    # Get config
    config = read_config()
    depth = int(config.get("mirror_depth", 4))
    selected_nm = config.get("selected_noise_model", "experiment")
    backend_name = config["noise_models"][selected_nm]["name"]

    connectivity_edges = get_connectivity_graph(backend_name)
    logging.info(f"Backend {backend_name} connectivity: {connectivity_edges}")

    # Determine native 2-qubit gate from basis gates
    basis_gates = circuit_submitter.get_basis_gates()
    two_qb_gates = [g for g in basis_gates if g in TWO_QUBIT_GATES]
    two_qb_gate_name = two_qb_gates[0] if two_qb_gates else "cx"
    logging.info(f"Using 2-qubit gate: {two_qb_gate_name}")

    submitter_noiseless = CircuitSubmitter("mirror_benchmarking", "noiseless_sim")

    rng = np.random.default_rng()

    # Build circuits for all trials
    all_circuits_qasm = []

    for t_idx in tqdm(range(num_trials), desc="Building circuits"):
        qc = build_mirror_circuit(
            num_qubits, depth, connectivity_edges, two_qb_gate_name, rng
        )

        qc_no_measure = qc.copy()
        qc_no_measure.remove_final_measurements()

        transpiled = qiskit.transpile(
            qc_no_measure,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
        )
        transpiled.measure_all()
        all_circuits_qasm.append(transpiled.qasm())

    logging.info(f"Built {len(all_circuits_qasm)} circuits total")

    # Run noiseless simulation
    submitter_noiseless.submit_circuits(
        shots=num_shots,
        qasm_strs=all_circuits_qasm,
        skip_asking=True,
        print_summary=False,
        skip_transpilation=True,
    )
    ideal_counts_list = submitter_noiseless.retrieve_counts(
        wait=True, print_timestamp_when_done=False
    )

    # Run noisy simulation
    circuit_submitter.submit_circuits(
        shots=num_shots,
        qasm_strs=all_circuits_qasm,
        skip_asking=True,
        print_summary=False,
        skip_transpilation=True,
    )
    noisy_counts_list = circuit_submitter.retrieve_counts(
        wait=True, print_timestamp_when_done=False
    )

    # Process results: compute P(|00...0>) per circuit
    zero_state = "0" * num_qubits
    ideal_probs = np.zeros(num_trials)
    noisy_probs = np.zeros(num_trials)

    for i in range(num_trials):
        ideal_total = sum(ideal_counts_list[i].values())
        noisy_total = sum(noisy_counts_list[i].values())

        ideal_probs[i] = ideal_counts_list[i].get(zero_state, 0) / ideal_total
        noisy_probs[i] = noisy_counts_list[i].get(zero_state, 0) / noisy_total

    submitter_noiseless.tasks = []

    results = {
        "depth": depth,
        "num_trials": num_trials,
        "num_qubits": num_qubits,
        "ideal_probs": ideal_probs,
        "noisy_probs": noisy_probs,
        "two_qb_gate": two_qb_gate_name,
        "backend_name": backend_name,
        "connectivity": connectivity_edges,
    }

    return results


def plot_mirror_results(results, save=False, path=None, show_plot=True):
    """Plot survival probability distribution across trials."""
    noisy_probs = results["noisy_probs"]
    n_qubits = results["num_qubits"]
    depth = results["depth"]
    mean_survival = np.mean(noisy_probs)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(noisy_probs, bins=20, color="#1f77b4", alpha=0.7, edgecolor="black")
    ax.axvline(mean_survival, color="red", linestyle="--",
               label=f"Mean: {mean_survival:.4f}")

    ax.set_xlabel(r"$p_{\mathrm{survival}}$ (P(|0...0>))")
    ax.set_ylabel("Count")
    ax.set_title(f"Mirror Benchmarking ({n_qubits} qubits, depth {depth})")
    ax.legend()

    plt.tight_layout()

    if save and path:
        fig.savefig(
            os.path.join(path, "mirror_benchmarking.png"), dpi=300, bbox_inches="tight"
        )

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    device_name = "noisy_sim"
    submitter = CircuitSubmitter(
        benchmark_name="mirror_benchmarking", device_name=device_name
    )
    filepath = submitter.benchmark_path

    num_qubits = get_num_qubits()
    optimization_level = int(os.environ.get("CIRCUIT_OPTIMIZATION", 3))

    num_trials = 10
    num_shots = 100

    sim_name = submitter.device_name

    logging.info(f"Running mirror benchmarking on {sim_name}")
    logging.info(f"Qubits: {num_qubits}, Trials: {num_trials}")

    results = run_mirror_benchmark(
        num_trials=num_trials,
        num_shots=num_shots,
        circuit_submitter=submitter,
        optimization_level=optimization_level,
    )

    mean_survival = np.mean(results["noisy_probs"])

    # Set PERF_VALUE to mean survival probability
    os.environ["PERF_VALUE"] = str(mean_survival)

    # Save results
    with open(os.path.join(filepath, "mirror_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    # Plot
    plot_mirror_results(results, save=True, path=filepath, show_plot=True)

    print(f"\n{'='*50}")
    print(f"MIRROR BENCHMARKING RESULTS:")
    print(f"  Backend: {results['backend_name']}")
    print(f"  Qubits: {results['num_qubits']}")
    print(f"  Depth: {results['depth']}")
    print(f"  2QB gate: {results['two_qb_gate']}")
    print(f"  Mean survival probability: {mean_survival:.4f}")
    print(f"  Std survival probability: {np.std(results['noisy_probs']):.4f}")
    print(f"{'='*50}")
