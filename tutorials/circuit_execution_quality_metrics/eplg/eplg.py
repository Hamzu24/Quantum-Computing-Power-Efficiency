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
Error Per Layered Gate (EPLG)

Measures the average error per two-qubit gate when gates run simultaneously
across a qubit chain using Simultaneous Direct Randomized Benchmarking.

For each layer of parallel 2-qubit pairs along a chain:
1. Build RB circuits at multiple depths, each depth consisting of:
   - Random single-qubit Cliffords on every chain qubit
   - Parallel native 2-qubit gates on the layer's pairs
   - A barrier
2. Append the circuit inverse and measure all qubits
3. Extract per-pair survival probability P(|00>) at each depth
4. Fit exponential decay per pair: p(d) = A * alpha^d + B
5. Process fidelity per pair: F_j = (1 + 3*alpha_j) / 4
6. Layer fidelity: LF = product(F_j) across all pairs in both layers
7. EPLG = 1 - LF^(1/n_2q) where n_2q = num_qubits - 1
"""

import os
import sys
import pathlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford
from qiskit.synthesis import synth_clifford_full
from scipy.optimize import curve_fit
from tqdm.auto import tqdm
import logging
import qiskit

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))
from _helpers.circuit_submitter import CircuitSubmitter
from _helpers.helpers import read_config, get_num_qubits
from _helpers.json_manager import JsonManager
from _helpers.constants import TWO_QUBIT_GATES

try:
    import networkx as nx
except ImportError:
    nx = None


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


def find_qubit_chain(edges, chain_length):
    """Find a simple path of chain_length nodes in the connectivity graph via DFS.

    Args:
        edges: List of (q0, q1) connectivity edges.
        chain_length: Number of qubits in the desired chain.

    Returns:
        List of qubit indices forming the chain.

    Raises:
        ValueError: If no chain of the requested length exists.
    """
    if nx is None:
        raise ImportError("networkx is required for EPLG. Install with: pip install networkx")

    G = nx.Graph()
    G.add_edges_from(edges)

    # DFS to find a simple path of the required length
    for start_node in sorted(G.nodes()):
        stack = [(start_node, [start_node])]
        while stack:
            node, path = stack.pop()
            if len(path) == chain_length:
                return path
            for neighbor in sorted(G.neighbors(node)):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))

    raise ValueError(
        f"No chain of length {chain_length} found in connectivity graph "
        f"with {len(G.nodes())} nodes and {len(G.edges())} edges"
    )


def partition_chain_into_layers(chain):
    """Split chain edges into two layers of non-overlapping pairs.

    Even layer: edges at indices 0, 2, 4, ... -> pairs (q0,q1), (q2,q3), ...
    Odd layer:  edges at indices 1, 3, 5, ... -> pairs (q1,q2), (q3,q4), ...

    Args:
        chain: List of qubit indices forming the chain.

    Returns:
        List of two layers, each layer being a list of (q0, q1) pairs.
    """
    all_edges = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
    even_layer = [all_edges[i] for i in range(0, len(all_edges), 2)]
    odd_layer = [all_edges[i] for i in range(1, len(all_edges), 2)]
    return [even_layer, odd_layer]


def build_eplg_rb_circuit(chain, layer_pairs, depth, two_qb_gate_name, rng):
    """Build an RB circuit for one layer of the EPLG benchmark.

    Each of `depth` blocks applies:
      - A random single-qubit Clifford per chain qubit
      - Parallel 2-qubit gates on the layer's pairs
      - A barrier

    Then appends the circuit inverse and measures all qubits.

    Args:
        chain: List of physical qubit indices forming the chain.
        layer_pairs: List of (q0, q1) pairs for this layer.
        depth: Number of RB blocks.
        two_qb_gate_name: Name of the native 2-qubit gate.
        rng: numpy random generator.

    Returns:
        QuantumCircuit for this RB sequence.
    """
    n_chain = len(chain)
    qc = QuantumCircuit(n_chain)

    # Map physical qubit index -> circuit qubit index
    qubit_map = {phys: idx for idx, phys in enumerate(chain)}

    for _ in range(depth):
        # Random 1q Clifford per chain qubit
        for phys_q in chain:
            cliff = random_clifford(1)
            cliff_circ = synth_clifford_full(cliff, method="AG")
            qc.compose(cliff_circ, qubits=[qubit_map[phys_q]], inplace=True)

        # Parallel 2q gates on the layer pairs
        for q0, q1 in layer_pairs:
            cq0, cq1 = qubit_map[q0], qubit_map[q1]
            if two_qb_gate_name == "cx":
                qc.cx(cq0, cq1)
            elif two_qb_gate_name == "cz":
                qc.cz(cq0, cq1)
            elif two_qb_gate_name == "ecr":
                qc.ecr(cq0, cq1)
            else:
                qc.cx(cq0, cq1)

        qc.barrier()

    # Append inverse and measure
    qc.compose(qc.inverse(), inplace=True)
    qc.measure_all()

    return qc


def extract_pair_survival(counts, chain, pair):
    """Extract P(|00>) for a specific qubit pair from full measurement counts.

    Args:
        counts: Dictionary of bitstring -> count.
        pair: Tuple (phys_q0, phys_q1) physical qubit indices.
        chain: List of physical qubit indices forming the chain.

    Returns:
        Survival probability P(|00>) for the pair.
    """
    qubit_map = {phys: idx for idx, phys in enumerate(chain)}
    n_chain = len(chain)
    idx0 = qubit_map[pair[0]]
    idx1 = qubit_map[pair[1]]

    survival_counts = 0
    total_counts = 0

    for bitstring, count in counts.items():
        total_counts += count
        # Qiskit bitstrings are little-endian: bit 0 is rightmost
        bit0 = bitstring[n_chain - 1 - idx0]
        bit1 = bitstring[n_chain - 1 - idx1]
        if bit0 == '0' and bit1 == '0':
            survival_counts += count

    if total_counts == 0:
        return 0.0
    return survival_counts / total_counts


def rb_decay(d, A, alpha, B):
    """RB exponential decay model: p(d) = A * alpha^d + B."""
    return A * np.power(alpha, d) + B


def fit_rb_decay(depths, survival_probs):
    """Fit the RB exponential decay model to survival probability data.

    Fits p(d) = A * alpha^d + B using scipy curve_fit.

    Args:
        depths: Array of circuit depths.
        survival_probs: Array of survival probabilities at each depth.

    Returns:
        Tuple (A, alpha, B) of fitted parameters.
    """
    try:
        popt, _ = curve_fit(
            rb_decay,
            np.array(depths, dtype=float),
            np.array(survival_probs, dtype=float),
            p0=[0.5, 0.95, 0.25],
            bounds=([0, 0, 0], [1, 1, 1]),
            maxfev=10000,
        )
        return tuple(popt)
    except RuntimeError:
        logging.warning("RB decay fit failed, returning default (0.5, 0.9, 0.25)")
        return (0.5, 0.9, 0.25)


def run_eplg(
    depths=None,
    num_trials=10,
    num_shots=1000,
    circuit_submitter=None,
    optimization_level=3,
):
    """Run EPLG (Error Per Layered Gate) benchmark.

    Args:
        depths: List of RB circuit depths to sample.
        num_trials: Number of random circuits per depth per layer.
        num_shots: Number of measurement shots per circuit.
        circuit_submitter: CircuitSubmitter for noisy simulation.
        optimization_level: Qiskit transpilation optimization level.

    Returns:
        Dictionary with EPLG value and all intermediate results.
    """
    if depths is None:
        depths = [2, 4, 8, 16, 32, 64]

    if circuit_submitter is None:
        circuit_submitter = CircuitSubmitter("eplg", "noisy_sim")

    num_qubits = get_num_qubits()

    config = read_config()
    selected_nm = config.get("selected_noise_model", "experiment")
    backend_name = config["noise_models"][selected_nm]["name"]

    connectivity_edges = get_connectivity_graph(backend_name)
    logging.info(f"Backend {backend_name} connectivity: {connectivity_edges}")

    # Find a qubit chain of the requested length
    chain = find_qubit_chain(connectivity_edges, num_qubits)
    logging.info(f"Found chain: {chain}")

    # Partition chain into two layers
    layers = partition_chain_into_layers(chain)
    logging.info(f"Layer 0 (even): {layers[0]}")
    logging.info(f"Layer 1 (odd):  {layers[1]}")

    # Determine native 2-qubit gate
    basis_gates = circuit_submitter.get_basis_gates()
    two_qb_gates = [g for g in basis_gates if g in TWO_QUBIT_GATES]
    two_qb_gate_name = two_qb_gates[0] if two_qb_gates else "cx"
    logging.info(f"Using 2-qubit gate: {two_qb_gate_name}")

    rng = np.random.default_rng()

    # Storage: per_pair_survival[layer_idx][(q0,q1)][depth_idx] = list of probs
    per_pair_survival = {}
    for layer_idx, layer_pairs in enumerate(layers):
        per_pair_survival[layer_idx] = {}
        for pair in layer_pairs:
            per_pair_survival[layer_idx][pair] = {d: [] for d in depths}

    # Build and run circuits for each layer and depth
    for layer_idx, layer_pairs in enumerate(layers):
        if not layer_pairs:
            logging.info(f"Layer {layer_idx} has no pairs, skipping")
            continue

        for depth in tqdm(depths, desc=f"Layer {layer_idx} depths"):
            circuits_qasm = []
            for _ in range(num_trials):
                qc = build_eplg_rb_circuit(
                    chain, layer_pairs, depth, two_qb_gate_name, rng
                )
                qc_no_measure = qc.copy()
                qc_no_measure.remove_final_measurements()
                transpiled = qiskit.transpile(
                    qc_no_measure,
                    basis_gates=basis_gates,
                    optimization_level=optimization_level,
                )
                transpiled.measure_all()
                circuits_qasm.append(transpiled.qasm())

            # Submit batch
            circuit_submitter.submit_circuits(
                shots=num_shots,
                qasm_strs=circuits_qasm,
                skip_asking=True,
                print_summary=False,
                skip_transpilation=True,
            )
            counts_list = circuit_submitter.retrieve_counts(
                wait=True, print_timestamp_when_done=False
            )

            # Extract per-pair survival probabilities
            for trial_idx, counts in enumerate(counts_list):
                for pair in layer_pairs:
                    p_surv = extract_pair_survival(counts, chain, pair)
                    per_pair_survival[layer_idx][pair][depth].append(p_surv)

    # Fit exponential decay per pair and compute fidelities
    pair_results = {}
    all_fidelities = []

    for layer_idx, layer_pairs in enumerate(layers):
        for pair in layer_pairs:
            mean_survivals = []
            for d in depths:
                probs = per_pair_survival[layer_idx][pair][d]
                mean_survivals.append(np.mean(probs) if probs else 0.0)

            A, alpha, B = fit_rb_decay(depths, mean_survivals)
            F_j = (1 + 3 * alpha) / 4  # process fidelity for 2q gate

            pair_results[pair] = {
                "layer": layer_idx,
                "depths": depths,
                "mean_survivals": mean_survivals,
                "fit_A": A,
                "fit_alpha": alpha,
                "fit_B": B,
                "fidelity": F_j,
            }
            all_fidelities.append(F_j)
            logging.info(
                f"Pair {pair}: alpha={alpha:.4f}, F={F_j:.6f}"
            )

    # Compute layer fidelity and EPLG
    n_2q = num_qubits - 1
    LF = np.prod(all_fidelities)
    eplg = 1 - LF ** (1 / n_2q) if n_2q > 0 else 0.0

    results = {
        "eplg": eplg,
        "layer_fidelity": LF,
        "pair_results": pair_results,
        "chain": chain,
        "layers": layers,
        "depths": depths,
        "num_trials": num_trials,
        "num_qubits": num_qubits,
        "two_qb_gate": two_qb_gate_name,
        "backend_name": backend_name,
        "n_2q_gates": n_2q,
    }

    return results


def plot_eplg_results(results, save=False, path=None, show_plot=True):
    """Plot survival probability vs depth for each qubit pair with fitted curve.

    Creates a grid of subplots, one per qubit pair. Title shows EPLG and LF.
    """
    pair_results = results["pair_results"]
    depths = results["depths"]
    eplg = results["eplg"]
    LF = results["layer_fidelity"]
    n_pairs = len(pair_results)

    if n_pairs == 0:
        logging.warning("No pairs to plot")
        return

    ncols = min(n_pairs, 3)
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    d_fit = np.linspace(min(depths), max(depths), 200)

    for idx, (pair, pr) in enumerate(sorted(pair_results.items())):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        ax.plot(depths, pr["mean_survivals"], "o", color="#1f77b4", markersize=6)

        y_fit = rb_decay(d_fit, pr["fit_A"], pr["fit_alpha"], pr["fit_B"])
        ax.plot(d_fit, y_fit, "-", color="#ff7f0e", linewidth=1.5)

        ax.set_xlabel("Depth")
        ax.set_ylabel("Survival P(|00>)")
        ax.set_title(
            f"Pair {pair} (L{pr['layer']})\n"
            f"α={pr['fit_alpha']:.4f}, F={pr['fidelity']:.4f}"
        )
        ax.set_ylim(-0.05, 1.05)

    # Hide unused subplots
    for idx in range(n_pairs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"EPLG = {eplg:.6f} | Layer Fidelity = {LF:.6f}\n"
        f"{results['num_qubits']} qubits, {results['backend_name']}",
        fontsize=13,
    )
    plt.tight_layout()

    if save and path:
        fig.savefig(
            os.path.join(path, "eplg_results.png"), dpi=300, bbox_inches="tight"
        )

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    device_name = "noisy_sim"
    submitter = CircuitSubmitter(
        benchmark_name="eplg", device_name=device_name
    )
    filepath = submitter.benchmark_path

    num_qubits = get_num_qubits()
    optimization_level = int(os.environ.get("CIRCUIT_OPTIMIZATION", 3))

    depths = [2, 4, 8, 16, 32, 64]
    num_trials = 10
    num_shots = 1000

    sim_name = submitter.device_name

    logging.info(f"Running EPLG benchmark on {sim_name}")
    logging.info(f"Qubits: {num_qubits}, Depths: {depths}, Trials: {num_trials}")

    results = run_eplg(
        depths=depths,
        num_trials=num_trials,
        num_shots=num_shots,
        circuit_submitter=submitter,
        optimization_level=optimization_level,
    )

    eplg_value = results["eplg"]

    # Set PERF_VALUE to EPLG (error rate, lower is better)
    os.environ["PERF_VALUE"] = str(eplg_value)

    # Save results
    with open(os.path.join(filepath, "eplg_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    # Plot
    plot_eplg_results(results, save=True, path=filepath, show_plot=True)

    print(f"\n{'='*50}")
    print(f"EPLG BENCHMARK RESULTS:")
    print(f"  Backend: {results['backend_name']}")
    print(f"  Qubits: {results['num_qubits']}")
    print(f"  Chain: {results['chain']}")
    print(f"  2QB gate: {results['two_qb_gate']}")
    print(f"  Layer fidelity: {results['layer_fidelity']:.6f}")
    print(f"  EPLG: {eplg_value:.6f}")
    print(f"  Pair fidelities:")
    for pair, pr in sorted(results["pair_results"].items()):
        print(f"    {pair}: alpha={pr['fit_alpha']:.4f}, F={pr['fidelity']:.6f}")
    print(f"{'='*50}")
