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

DEPOL_2q_P = 0.01 # Hardcode for now, build into pauli noise model later!


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
            pauli_noise_config: Dict with per-qubit Pauli noise parameters:
                {
                    'qubits': [
                        {'p_x': float, 'p_y': float, 'p_z': float},  # qubit 0
                        {'p_x': float, 'p_y': float, 'p_z': float},  # qubit 1
                        ...
                    ],
                    'general': {'p_x': float, 'p_y': float, 'p_z': float}  # fallback for qubits not in 'qubits' list
                }

            Missing keys (p_x, p_y, p_z) default to 0.
            For 2-qubit gates, 1-qubit Pauli errors are applied to each qubit independently.
        """
        self._pauli_noise_config = pauli_noise_config

    def set_pauli_noise_config(self, config: Dict):
        """
            config: Dict with per-qubit Pauli noise parameters (see __init__ for format)
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

    def _get_qubit_noise_config(self, qubit_index: int) -> Dict:
        """
        Get the noise config for a specific qubit.

        Args:
            qubit_index: Index of the qubit

        Returns:
            Dict with p_x, p_y, p_z (missing keys default to 0)
        """
        config = self._pauli_noise_config
        qubits_list = config.get('qubits', [])
        general_config = config.get('general', {})

        if qubit_index < len(qubits_list):
            qubit_config = qubits_list[qubit_index]
        else:
            qubit_config = general_config

        return {
            'p_x': qubit_config.get('p_x', 0),
            'p_y': qubit_config.get('p_y', 0),
            'p_z': qubit_config.get('p_z', 0)
        }

    def _inject_noise(self, circuit: stim.Circuit) -> stim.Circuit:
        """
            circuit: Original stim.Circuit

            returns new stim.Circuit with noise injected after gates.
            Uses per-qubit Pauli noise from config.
        """
        if self._pauli_noise_config is None:
            return circuit

        # Build new circuit with noise
        noisy_circuit = stim.Circuit()

        for instruction in circuit:
            # Add the original instruction
            noisy_circuit.append(instruction)

            # Get instruction name and targets
            name = instruction.name
            targets = instruction.targets_copy()

            # Skip noise injection for certain instructions
            if name in ('TICK', 'DETECTOR', 'OBSERVABLE_INCLUDE', 'QUBIT_COORDS',
                        'M', 'MR', 'MX', 'MY', 'MZ', 'R', 'RX', 'RY', 'RZ'):
                continue

            # Get qubit indices from targets
            qubit_indices = [t.value for t in targets if t.is_qubit_target]

            if not qubit_indices:
                continue

            # Inject per-qubit Pauli noise after gates
            if name in ('H', 'S', 'S_DAG', 'X', 'Y', 'Z', 'SQRT_X', 'SQRT_X_DAG',
                        'SQRT_Y', 'SQRT_Y_DAG', 'I'):
                # Single-qubit gate: add Pauli channel noise per qubit
                for qubit_idx in qubit_indices:
                    noise = self._get_qubit_noise_config(qubit_idx)
                    p_x, p_y, p_z = noise['p_x'], noise['p_y'], noise['p_z']
                    if p_x > 0 or p_y > 0 or p_z > 0:
                        noisy_circuit.append('PAULI_CHANNEL_1', [qubit_idx], [p_x, p_y, p_z])

            elif name in ('CNOT', 'CX', 'CZ', 'CY', 'SWAP', 'ISWAP', 'ISWAP_DAG',
                          'SQRT_XX', 'SQRT_YY', 'SQRT_ZZ'):
                # Two-qubit gate: first apply 1-qubit Pauli error to each qubit independently
                for qubit_idx in qubit_indices:
                    noise = self._get_qubit_noise_config(qubit_idx)
                    p_x, p_y, p_z = noise['p_x'], noise['p_y'], noise['p_z']
                    if p_x > 0 or p_y > 0 or p_z > 0:
                        noisy_circuit.append('PAULI_CHANNEL_1', [qubit_idx], [p_x, p_y, p_z])
                # Then apply depolarising channel to each two qubit pair
                for i in range(0, len(qubit_indices), 2):
                    qubit_a = qubit_indices[i]
                    qubit_b = qubit_indices[i+1]
                    noisy_circuit.append(f'DEPOLARIZE2({DEPOL_2q_P}) {qubit_a} {qubit_b}')

        return noisy_circuit
