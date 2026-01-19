import os
import json
import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit.circuit.library import RZXGate
from qiskit.circuit.measure import Measure
from qiskit_aer import AerSimulator, AerJob
from qiskit_aer.noise import NoiseModel
import numpy as np
from _helpers.noise_model import custom_noise_model
from _helpers.constants import SIMULATION_METHOD
from _helpers.helpers import read_config, get_max_parallel_from_config

class QiskitTaskResultWrapper:
    def __init__(self, result: AerJob, shots: int) -> None:
        result = result.result()
        num_qubits = result.results[0].header.n_qubits
        output_state = np.abs(
            np.asarray(result.data()["before_measurement"], dtype=np.complex128)
        )
        eps = np.finfo(float).eps
        output_state[np.abs(output_state) < eps] = 0
        output_state = np.rint(output_state * shots)

        counts = {
            format(b, f"0{num_qubits}b"): int(v)
            for b, v in enumerate(np.diag(output_state))
        }

        # Adjust the highest valued count to align with the shots
        highest_count_key = max(counts, key=counts.get)
        sum_counts = sum(counts.values())
        counts[highest_count_key] -= sum_counts - shots

        # Convert qisit order to braket order, so this is consistent with other wrappers
        counts = {k[::-1]: v for k, v in counts.items()}
        self.measurement_counts = counts


class ExactProbabilityResultWrapper:
    """
    Result wrapper that returns exact probabilities from the density matrix
    without discretization artifacts. Uses a large effective shot count to
    preserve probability resolution for all 2^n states.

    This fixes the quantum volume bug where discretization (rounding probs * shots)
    causes information loss when shots < 2^n_qubits, leading to incorrect heavy
    output calculations for larger qubit counts.
    """
    EFFECTIVE_SHOTS = 10_000_000

    def __init__(self, result: AerJob) -> None:
        result = result.result()
        num_qubits = result.results[0].header.n_qubits
        output_state = np.abs(
            np.asarray(result.data()["before_measurement"], dtype=np.complex128)
        )
        eps = np.finfo(float).eps
        output_state[np.abs(output_state) < eps] = 0

        effective_shots = self.EFFECTIVE_SHOTS
        output_state = np.rint(output_state * effective_shots)

        counts = {
            format(b, f"0{num_qubits}b"): int(v)
            for b, v in enumerate(np.diag(output_state))
        }

        highest_count_key = max(counts, key=counts.get)
        sum_counts = sum(counts.values())
        counts[highest_count_key] -= sum_counts - effective_shots

        counts = {k[::-1]: v for k, v in counts.items()}
        self.measurement_counts = counts


class QiskitTaskResultWrapperWithShots:
    def __init__(self, result: AerJob, shots: int) -> None:
        result = result.result()
        num_qubits = result.results[0].header.n_qubits
        # print(dir(result))
        self.measurement_counts = result.get_counts(0)


class QiskitTaskWrapper:
    def __init__(self, task: AerJob, shots: int, shot_noise: bool, exact_probabilities: bool = False) -> None:
        self.task = task
        self.id = task.job_id()
        self.shots = shots
        self.shot_noise = shot_noise
        self.exact_probabilities = exact_probabilities
        self._cached_result = None

    def result(self):
        if self._cached_result is not None:
            return self._cached_result

        if self.shot_noise:
            self._cached_result = QiskitTaskResultWrapperWithShots(self.task, self.shots)
        elif self.exact_probabilities:
            self._cached_result = ExactProbabilityResultWrapper(self.task)
        else:
            self._cached_result = QiskitTaskResultWrapper(self.task, self.shots)

        # Clear job reference to free memory (contains large density matrices)
        self.task = None

        return self._cached_result

    def state(self):
        return "COMPLETED"


class QiskitTaskBatchWrapper:
    def __init__(self, tasks: list[QiskitTaskWrapper]) -> None:
        self.tasks = tasks


class SimWrapper:
    def __init__(
        self, backend: Backend = None, noise_model: NoiseModel = None, shot_noise=False, exact_probabilities=False
    ):
        self.shot_noise = shot_noise
        self.exact_probabilities = exact_probabilities
        if backend is None:
            self.backend = AerSimulator
        else:
            self.backend = backend
        if noise_model is not None:
            self.noise_model = noise_model
            self.sim = self.backend(
                method=SIMULATION_METHOD, noise_model=self.noise_model, device="GPU", blocking_qubits=11
            )
        else:
            self.noise_model = None
            self.sim = self.backend(method='statevector', device="GPU")

    def _remove_measurement_and_add_dm_save(self, circ):
        index_to_delete = []
        for index, instruction in enumerate(circ.data):
            if isinstance(instruction.operation, Measure):
                index_to_delete.append(index)
        for index in reversed(index_to_delete):
            del circ.data[index]
        circ.save_density_matrix(label="before_measurement")
        return circ

    def run_batch(
        self, circuits: list[QuantumCircuit], shots=1000, max_parallel=None, **kwargs
    ):
        """
        Run a batch of circuits with memory-aware batching.

        Args:
            circuits: List of quantum circuits to execute
            shots: Number of shots per circuit
            max_parallel: Maximum circuits to run concurrently. If None, reads from
                         config "max_parallel_circuits" (default 10 to avoid GPU OOM)

        Returns:
            QiskitTaskBatchWrapper containing all task results
        """
        self.shots = shots
        # Read from config to ensure consistent behavior regardless of caller
        max_parallel = get_max_parallel_from_config(default=10)
        all_tasks = []

        if self.shot_noise:
            # Process in batches to avoid GPU memory exhaustion
            for i in range(0, len(circuits), max_parallel):
                batch = circuits[i:i + max_parallel]
                batch_tasks = []
                for c in batch:
                    task = QiskitTaskWrapper(
                        self.sim.run(c.reverse_bits(), shots=shots),
                        shots=shots,
                        shot_noise=True,
                    )
                    batch_tasks.append(task)

                # Force completion of this batch before starting next
                # This releases GPU memory for the next batch
                for task in batch_tasks:
                    task.result()  # Forces execution and caches result
                all_tasks.extend(batch_tasks)

            return QiskitTaskBatchWrapper(all_tasks)
        else:
            # Preprocess all circuits (lightweight, doesn't use GPU memory)
            circuits2 = []
            for c in circuits:
                c = self._remove_measurement_and_add_dm_save(c)
                circuits2.append(c)

            # Process in batches to avoid GPU memory exhaustion
            for i in range(0, len(circuits2), max_parallel):
                batch = circuits2[i:i + max_parallel]
                batch_tasks = []
                for c in batch:
                    task = QiskitTaskWrapper(
                        self.sim.run(c, shots=shots),
                        shots=shots,
                        shot_noise=False,
                        exact_probabilities=self.exact_probabilities
                    )
                    batch_tasks.append(task)

                # Force completion of this batch before starting next
                # This releases GPU memory for the next batch
                for task in batch_tasks:
                    task.result()  # Forces execution and caches result
                all_tasks.extend(batch_tasks)

            return QiskitTaskBatchWrapper(all_tasks)



class NoisySimWrapper(SimWrapper):
    def __init__(self, backend: Backend = None, noise_model: NoiseModel = None):

        self.noise_model = noise_model
        self.backend = backend

        if self.backend is None:
            self.backend = AerSimulator

        if self.noise_model is None:
            self.noise_model = custom_noise_model()

        # Use exact probabilities since this is deterministic density matrix simulation
        # (no shot noise). This ensures consistent comparison with noiseless_sim.
        super().__init__(backend=self.backend, noise_model=self.noise_model, exact_probabilities=True)

    def set_noise_model(self, noise_model):
        super().__init__(self.backend, noise_model=noise_model, exact_probabilities=True)


class NoisySimWrapperWithShots(SimWrapper):
    def __init__(self, backend: Backend = None, noise_model: NoiseModel = None):

        self.noise_model = custom_noise_model()
        self.backend = backend

        if self.backend is None:
            self.backend = AerSimulator

        if noise_model is not None:
            self.noise_model = noise_model

        super().__init__(backend=self.backend, noise_model=self.noise_model,  shot_noise=True)

    def set_noise_model(self, noise_model):
        super().__init__(self.backend, noise_model=noise_model, shot_noise=True)


class NoiselessDensityMatrixSimWrapper(SimWrapper):
    def __init__(self):
        super().__init__(exact_probabilities=True)
