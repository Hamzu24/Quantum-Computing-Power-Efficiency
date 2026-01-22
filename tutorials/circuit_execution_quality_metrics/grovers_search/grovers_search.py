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
Grover's Search Algorithm Benchmark

This benchmark measures the success probability of Grover's search algorithm
under noisy conditions. The performance metric is the probability of measuring
the marked (target) state after applying the optimal number of Grover iterations.

For an n-qubit search space with a single marked item:
- Search space size: N = 2^n
- Optimal iterations: floor(pi/4 * sqrt(N))
- Ideal success probability: ~1 (approaches 1 as N increases)

The benchmark:
1. Generates random target states for multiple trials
2. Constructs Grover circuits with optimal iteration count
3. Compares noisy vs ideal success probabilities
4. Reports mean success probability as PERF_VALUE
"""

import os
import sys
import pathlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from tqdm.auto import tqdm
import logging

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))
from _helpers.circuit_submitter import CircuitSubmitter


def create_oracle(num_qubits: int, marked_state: int) -> QuantumCircuit:
    """
    Create an oracle that marks the target state with a phase flip.

    The oracle applies a phase of -1 to the marked state |m> while leaving
    all other states unchanged: O|x> = -|x> if x=m, else |x>

    :param num_qubits: Number of qubits in the search space
    :param marked_state: Integer representation of the target state (0 to 2^n - 1)
    :return: QuantumCircuit implementing the oracle
    """
    oracle = QuantumCircuit(num_qubits, name="Oracle")

    # Convert marked state to binary and determine which qubits need X gates
    marked_binary = format(marked_state, f'0{num_qubits}b')

    # Apply X gates to qubits that are 0 in the marked state
    # This transforms |marked> to |11...1>
    for i, bit in enumerate(reversed(marked_binary)):
        if bit == '0':
            oracle.x(i)

    # Apply multi-controlled Z gate (phase flip on |11...1>)
    # MCZ = H on last qubit, MCX, H on last qubit
    if num_qubits == 1:
        oracle.z(0)
    else:
        oracle.h(num_qubits - 1)
        oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        oracle.h(num_qubits - 1)

    # Undo the X gates
    for i, bit in enumerate(reversed(marked_binary)):
        if bit == '0':
            oracle.x(i)

    return oracle


def create_diffusion_operator(num_qubits: int) -> QuantumCircuit:
    """
    Create the Grover diffusion operator (amplitude amplification).

    The diffusion operator reflects amplitudes about their mean:
    D = 2|s><s| - I, where |s> is the uniform superposition state.

    Implementation: H^n (2|0><0| - I) H^n = H^n Z_0 H^n
    where Z_0 applies -1 phase to |0> state.

    :param num_qubits: Number of qubits
    :return: QuantumCircuit implementing the diffusion operator
    """
    diffusion = QuantumCircuit(num_qubits, name="Diffusion")

    # Apply Hadamard to all qubits
    diffusion.h(range(num_qubits))

    # Apply X to all qubits (transforms |0> to |11...1>)
    diffusion.x(range(num_qubits))

    # Multi-controlled Z gate (phase flip on |11...1>)
    if num_qubits == 1:
        diffusion.z(0)
    else:
        diffusion.h(num_qubits - 1)
        diffusion.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        diffusion.h(num_qubits - 1)

    # Undo X gates
    diffusion.x(range(num_qubits))

    # Apply Hadamard to all qubits
    diffusion.h(range(num_qubits))

    return diffusion


def optimal_num_iterations(num_qubits: int, num_marked: int = 1) -> int:
    """
    Calculate the optimal number of Grover iterations.

    For N items with M marked items:
    optimal_iterations = floor(pi/4 * sqrt(N/M))

    :param num_qubits: Number of qubits (N = 2^num_qubits)
    :param num_marked: Number of marked items (default 1)
    :return: Optimal number of iterations
    """
    N = 2 ** num_qubits
    return max(1, int(np.floor(np.pi / 4 * np.sqrt(N / num_marked))))


def create_grover_circuit(num_qubits: int, marked_state: int, num_iterations: int = None) -> QuantumCircuit:
    """
    Create a complete Grover search circuit.

    :param num_qubits: Number of qubits in the search space
    :param marked_state: Integer representation of the target state
    :param num_iterations: Number of Grover iterations (uses optimal if None)
    :return: QuantumCircuit implementing Grover's algorithm
    """
    if num_iterations is None:
        num_iterations = optimal_num_iterations(num_qubits)

    qc = QuantumCircuit(num_qubits)

    # Initialize uniform superposition
    qc.h(range(num_qubits))
    qc.barrier()

    # Create oracle and diffusion operator
    oracle = create_oracle(num_qubits, marked_state)
    diffusion = create_diffusion_operator(num_qubits)

    # Apply Grover iterations
    for _ in range(num_iterations):
        qc.compose(oracle, inplace=True)
        qc.barrier()
        qc.compose(diffusion, inplace=True)
        qc.barrier()

    # Measure all qubits
    qc.measure_all()

    return qc


def calculate_success_probability(counts: dict, marked_state: int, num_qubits: int) -> float:
    """
    Calculate the probability of measuring the marked state.

    :param counts: Dictionary of measurement outcomes {bitstring: count}
    :param marked_state: Integer representation of the target state
    :param num_qubits: Number of qubits
    :return: Probability of measuring the marked state
    """
    total_counts = sum(counts.values())
    # Reverse the bitstring to match Braket format (the wrappers reverse Qiskit's bit order)
    marked_bitstring = format(marked_state, f'0{num_qubits}b')[::-1]
    marked_counts = counts.get(marked_bitstring, 0)
    return marked_counts / total_counts


def plot_success_probability_distribution(
    success_probs: np.ndarray,
    sim_name: str,
    title: str,
    num_trials: int,
    save: bool = False,
    path: str = None,
    show_plot: bool = False
):
    """Plot histogram of success probabilities across trials."""
    mean_prob = np.mean(success_probs)
    std_prob = np.std(success_probs)

    plt.figure(figsize=(8, 6))
    plt.hist(success_probs, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(mean_prob, c='b', linewidth=2, label=f'Mean: {mean_prob:.3f}')
    plt.axvline(mean_prob - 2*std_prob, c='r', linestyle='--', label=r'2$\sigma$')
    plt.axvline(mean_prob + 2*std_prob, c='r', linestyle='--')

    plt.xlabel('Success Probability')
    plt.ylabel('Occurrences')
    plt.title(title)
    plt.legend()
    plt.xlim(0, 1.05)

    if save and path:
        os.makedirs(path, exist_ok=True)
        plt.savefig(
            os.path.join(path, f"{title.replace(' ', '_')}.png"),
            format='png', dpi=300, bbox_inches='tight'
        )
    if show_plot:
        plt.show()
    plt.close('all')


def plot_success_vs_qubits(
    num_qubits_list: list,
    ideal_means: list,
    noisy_means: list,
    noisy_stds: list,
    sim_name: str,
    num_trials: int,
    save: bool = False,
    path: str = None,
    show_plot: bool = False
):
    """Plot success probability vs number of qubits for ideal and noisy simulations."""
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})

    plt.scatter(num_qubits_list, ideal_means, label='Ideal', marker='^', s=100)
    plt.scatter(num_qubits_list, noisy_means, label='Noisy', s=100)
    plt.errorbar(
        num_qubits_list, noisy_means, yerr=2*np.array(noisy_stds),
        ls='none', ecolor='#ff7f0e', capsize=3, alpha=0.6, elinewidth=2
    )

    # Theoretical success probability line
    theoretical_probs = []
    for n in num_qubits_list:
        N = 2**n
        k = optimal_num_iterations(n)
        theta = np.arcsin(1/np.sqrt(N))
        prob = np.sin((2*k + 1) * theta)**2
        theoretical_probs.append(prob)
    plt.plot(num_qubits_list, theoretical_probs, 'g--', label='Theoretical', linewidth=2)

    plt.xlabel('Number of Qubits')
    plt.ylabel('Success Probability')
    plt.title(f"Grover's Search: Success Probability vs Qubits ({sim_name})")
    plt.legend()
    plt.xticks(num_qubits_list)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)

    if save and path:
        os.makedirs(path, exist_ok=True)
        plt.savefig(
            os.path.join(path, f"grovers_success_vs_qubits_{sim_name}.png"),
            format='png', dpi=300, bbox_inches='tight'
        )
    if show_plot:
        plt.show()
    plt.close('all')


def run_grover_benchmark(
    num_qubits_list: list = [2, 3, 4],
    num_trials: int = 100,
    optimization_level: int = 3,
    circuit_submitter: CircuitSubmitter = None,
    num_shots: int = 1000
) -> dict:
    """
    Run Grover's search benchmark across multiple qubit counts.

    :param num_qubits_list: List of qubit counts to test
    :param num_trials: Number of random target states to test per qubit count
    :param optimization_level: Qiskit transpilation optimization level
    :param circuit_submitter: CircuitSubmitter for noisy simulation
    :param num_shots: Number of measurement shots per circuit
    :return: Dictionary with results for each qubit count
    """
    results = {}

    if circuit_submitter is None:
        circuit_submitter = CircuitSubmitter("grovers_search", "noisy_sim")

    basis_gates = circuit_submitter.get_basis_gates()
    submitter_noiseless = CircuitSubmitter("grovers_search", "noiseless_sim")

    for num_qubits in num_qubits_list:
        logging.info(f"Running Grover benchmark with {num_qubits} qubits")

        N = 2 ** num_qubits
        num_iterations = optimal_num_iterations(num_qubits)

        results[num_qubits] = {
            'trials': [],
            'num_iterations': num_iterations,
            'search_space_size': N
        }

        # Generate random target states for each trial
        marked_states = np.random.randint(0, N, size=num_trials)

        # Build circuits
        circuits_qasm = []
        for marked_state in marked_states:
            qc = create_grover_circuit(num_qubits, marked_state, num_iterations)

            # Transpile for the target backend
            qc_no_measure = qc.copy()
            qc_no_measure.remove_final_measurements()

            transpiled = qiskit.transpile(
                qc_no_measure,
                basis_gates=basis_gates,
                optimization_level=optimization_level
            )
            transpiled.measure_all()

            circuits_qasm.append(transpiled.qasm())
            results[num_qubits]['trials'].append({
                'marked_state': int(marked_state)
            })

        # Run noiseless simulation
        submitter_noiseless.submit_circuits(
            shots=num_shots, qasm_strs=circuits_qasm,
            skip_asking=True, print_summary=False, skip_transpilation=True
        )
        ideal_counts_list = submitter_noiseless.retrieve_counts(
            wait=True, print_timestamp_when_done=False
        )

        # Run noisy simulation
        circuit_submitter.submit_circuits(
            shots=num_shots, qasm_strs=circuits_qasm,
            skip_asking=True, print_summary=False, skip_transpilation=True
        )
        noisy_counts_list = circuit_submitter.retrieve_counts(
            wait=True, print_timestamp_when_done=False
        )

        # Process results
        for i, (trial, ideal_counts, noisy_counts) in enumerate(
            zip(results[num_qubits]['trials'], ideal_counts_list, noisy_counts_list)
        ):
            marked_state = trial['marked_state']

            trial['ideal_success_prob'] = calculate_success_probability(
                ideal_counts, marked_state, num_qubits
            )
            trial['noisy_success_prob'] = calculate_success_probability(
                noisy_counts, marked_state, num_qubits
            )

        # Clear memory
        del circuits_qasm
        del ideal_counts_list
        del noisy_counts_list

    submitter_noiseless.tasks = []

    return results


if __name__ == "__main__":
    device_name = "noisy_sim"
    submitter = CircuitSubmitter(benchmark_name="grovers_search", device_name=device_name)
    filepath = submitter.benchmark_path

    # Get configuration
    from _helpers.helpers import get_num_qubits
    num_qubits_list = [get_num_qubits()]

    num_trials = 100
    num_shots = 100
    optimization_level = int(os.environ.get("CIRCUIT_OPTIMIZATION", 3))

    sim_name = submitter.device_name

    logging.info(f"Running Grover's search benchmark on {sim_name}")
    logging.info(f"Qubit counts: {num_qubits_list}, Trials: {num_trials}, Shots: {num_shots}")

    # Run benchmark
    results = run_grover_benchmark(
        num_qubits_list=num_qubits_list,
        num_trials=num_trials,
        optimization_level=optimization_level,
        circuit_submitter=submitter,
        num_shots=num_shots
    )

    # Analyze and plot results
    ideal_means = []
    noisy_means = []
    noisy_stds = []

    for num_qubits in num_qubits_list:
        trial_data = results[num_qubits]['trials']

        ideal_probs = np.array([t['ideal_success_prob'] for t in trial_data])
        noisy_probs = np.array([t['noisy_success_prob'] for t in trial_data])

        ideal_means.append(np.mean(ideal_probs))
        noisy_means.append(np.mean(noisy_probs))
        noisy_stds.append(np.std(noisy_probs))

        print(f"\n{num_qubits} qubits:")
        print(f"  Optimal iterations: {results[num_qubits]['num_iterations']}")
        print(f"  Ideal mean success: {ideal_means[-1]:.4f}")
        print(f"  Noisy mean success: {noisy_means[-1]:.4f} +/- {noisy_stds[-1]:.4f}")

        # Plot distribution for this qubit count
        plot_success_probability_distribution(
            noisy_probs, sim_name,
            f"Grover Noisy Success Prob n={num_qubits}",
            num_trials, save=True, path=filepath, show_plot=True
        )
        plot_success_probability_distribution(
            ideal_probs, sim_name,
            f"Grover Ideal Success Prob n={num_qubits}",
            num_trials, save=True, path=filepath, show_plot=True
        )

    # Plot success vs qubits
    plot_success_vs_qubits(
        num_qubits_list, ideal_means, noisy_means, noisy_stds,
        sim_name, num_trials, save=True, path=filepath, show_plot=True
    )

    # Set performance metric (mean noisy success probability)
    mean_ideal_success = np.mean(ideal_means)
    mean_noisy_success = np.mean(noisy_means)
    os.environ['PERF_VALUE'] = str(mean_noisy_success)

    print(f"\n{'='*50}")
    print(f"OVERALL RESULTS:")
    print(f"  Ideal performance:  {mean_ideal_success:.4f}")
    print(f"  Noisy performance:  {mean_noisy_success:.4f}")
    print(f"{'='*50}")

    # Save results
    with open(os.path.join(filepath, 'grovers_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
