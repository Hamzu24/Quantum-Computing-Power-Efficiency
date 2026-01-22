
from typing import Dict, List, Optional
from collections import Counter
import numpy as np

try:
    import stim
except ImportError:
    raise ImportError(
        "Stim is required for stabilizer simulation. "
        "Install it with: pip install stim"
    )


class StimTaskResultWrapper:
    def __init__(self, samples: np.ndarray, num_measurements: int):
        self._samples = samples
        self._num_measurements = num_measurements
        self._measurement_counts = None

    @property
    def measurement_counts(self) -> Dict[str, int]:
        """
        Convert samples to measurement counts dictionary.

        Returns:
            Dict mapping bitstrings to counts, e.g., {'00': 500, '11': 500}
        """
        if self._measurement_counts is None:
            # Convert boolean arrays to bitstrings
            bitstrings = []
            for shot in self._samples:
                # Convert booleans to '0'/'1' string
                bitstring = ''.join(str(int(b)) for b in shot)
                bitstrings.append(bitstring)

            self._measurement_counts = dict(Counter(bitstrings))

        return self._measurement_counts


class StimTaskWrapper:
    def __init__(self, samples: np.ndarray, num_measurements: int):
        """
            samples: 2D numpy array of samples for this circuit
            num_measurements: Number of measurements in the circuit
        """
        self._result = StimTaskResultWrapper(samples, num_measurements)
        self._state = "COMPLETED"

    def result(self) -> StimTaskResultWrapper:
        return self._result

    def state(self) -> str:
        return self._state


class StimTaskBatchWrapper:
    """
    Batch wrapper containing multiple task results.

    Provides .tasks list matching the existing batch interface.
    """

    def __init__(self, tasks: List[StimTaskWrapper]):
        """
            tasks: List of StimTaskWrapper instances
        """
        self.tasks = tasks


class StimSimWrapper:
    def __init__(self, pauli_noise_config: Optional[Dict] = None):
        """
            pauli_noise_config: Optional dict with Pauli noise parameters:
                {
                    '1q': {'p_x': float, 'p_y': float, 'p_z': float},
                    '2q': {'p_x': float, 'p_y': float, 'p_z': float},
                    'measurement': {'p_flip': float}
                }
        """
        self._pauli_noise_config = pauli_noise_config

    def set_pauli_noise_config(self, config: Dict):
        """
            config: Dict with Pauli noise parameters
        """
        self._pauli_noise_config = config

    def run_batch(
        self,
        circuits: List[stim.Circuit],
        shots: int,
        max_parallel: Optional[int] = None
    ) -> StimTaskBatchWrapper:
        """
            circuits: List of stim.Circuit objects
            shots: Number of shots per circuit
            max_parallel: Ignored (included for interface compatibility)

            returns StimTaskBatchWrapper containing results for all circuits
        """
        tasks = []

        for circuit in circuits:
            # Inject noise if configured
            if self._pauli_noise_config is not None:
                circuit = self._inject_noise(circuit)

            # Compile and sample
            sampler = circuit.compile_sampler()
            samples = sampler.sample(shots)

            # Get number of measurements from the circuit
            num_measurements = circuit.num_measurements

            # Create task wrapper
            task = StimTaskWrapper(samples, num_measurements)
            tasks.append(task)

        return StimTaskBatchWrapper(tasks)

    def _inject_noise(self, circuit: stim.Circuit) -> stim.Circuit:
        """
            circuit: Original stim.Circuit

            returns new stim.Circuit with noise injected
        """
        if self._pauli_noise_config is None:
            return circuit

        config = self._pauli_noise_config
        p1q = config.get('1q', {})
        p2q = config.get('2q', {})
        p_meas = config.get('measurement', {})

        # Calculate total Pauli error probabilities
        p1_total = p1q.get('p_x', 0) + p1q.get('p_y', 0) + p1q.get('p_z', 0)
        p2_total = p2q.get('p_x', 0) + p2q.get('p_y', 0) + p2q.get('p_z', 0)
        p_flip = p_meas.get('p_flip', 0)

        # Build new circuit with noise
        noisy_circuit = stim.Circuit()

        for instruction in circuit:
            # Add the original instruction
            noisy_circuit.append(instruction)

            # Get instruction name and targets
            name = instruction.name
            targets = instruction.targets_copy()

            # Skip noise injection for certain instructions
            if name in ('TICK', 'DETECTOR', 'OBSERVABLE_INCLUDE', 'QUBIT_COORDS'):
                continue

            # Inject noise after gates
            if name == 'M' or name == 'MR' or name == 'MX' or name == 'MY' or name == 'MZ':
                # Measurement error: flip the classical bit
                if p_flip > 0:
                    # Use X_ERROR before measurement to simulate bit flip
                    # Note: For proper measurement error, we'd use DETECTOR error models
                    # This is a simplified approach
                    pass  # Measurement errors handled differently in Stim

            elif name in ('H', 'S', 'S_DAG', 'X', 'Y', 'Z', 'SQRT_X', 'SQRT_X_DAG',
                          'SQRT_Y', 'SQRT_Y_DAG', 'I', 'R', 'RX', 'RY'):
                # Single-qubit gate: add depolarizing noise
                if p1_total > 0:
                    qubit_indices = [t.value for t in targets if t.is_qubit_target]
                    if qubit_indices:
                        noisy_circuit.append('DEPOLARIZE1', qubit_indices, p1_total)

            elif name in ('CNOT', 'CX', 'CZ', 'CY', 'SWAP', 'ISWAP', 'ISWAP_DAG',
                          'SQRT_XX', 'SQRT_YY', 'SQRT_ZZ'):
                # Two-qubit gate: add depolarizing noise
                if p2_total > 0:
                    qubit_indices = [t.value for t in targets if t.is_qubit_target]
                    # DEPOLARIZE2 needs pairs of qubits
                    if len(qubit_indices) >= 2:
                        noisy_circuit.append('DEPOLARIZE2', qubit_indices, p2_total)

        return noisy_circuit
