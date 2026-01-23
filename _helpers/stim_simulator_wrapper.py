from typing import Dict, List, Optional, Set
from collections import Counter
from _helpers.constants import STIM_TO_BASIC
import numpy as np

try:
    import stim
except ImportError:
    raise ImportError(
        "Stim is required for stabilizer simulation. "
        "Install it with: pip install stim"
    )

DEPOL_2q_P = 0.01 # Hardcode for now, build into pauli noise model later!

# Gate decompositions for transpiling to basis gates
# Maps stim gate names to sequences of basis gates
# Format: {gate_name: [(basis_gate, qubits_selector), ...]}
# qubits_selector: 'all' applies to all qubits, 'pairs' for 2-qubit gates
STIM_GATE_DECOMPOSITIONS = {
    # H = S · SQRT_X · S (up to global phase)
    'H': [('S', 'all'), ('SQRT_X', 'all'), ('S', 'all')],
    # Y = S · S · X = Z · X (up to global phase)
    'Y': [('S', 'all'), ('S', 'all'), ('X', 'all')],
    # Z = S · S
    'Z': [('S', 'all'), ('S', 'all')],
    # SQRT_Y = S · SQRT_X · S_DAG (up to global phase)
    'SQRT_Y': [('S', 'all'), ('SQRT_X', 'all'), ('S_DAG', 'all')],
    # SQRT_Y_DAG = S_DAG · SQRT_X · S (up to global phase)
    'SQRT_Y_DAG': [('S_DAG', 'all'), ('SQRT_X', 'all'), ('S', 'all')],
}

# Qiskit gate names to stim gate names mapping
QISKIT_TO_STIM = {
    'sx': 'SQRT_X',
    'sxdg': 'SQRT_X_DAG',
    'x': 'X',
    'y': 'Y',
    'z': 'Z',
    'h': 'H',
    's': 'S',
    'sdg': 'S_DAG',
    't': 'T',
    'tdg': 'T_DAG',
    'cx': 'CX',
    'cz': 'CZ',
    'cy': 'CY',
    'swap': 'SWAP',
    'iswap': 'ISWAP',
    'id': 'I',
    'rz': 'RZ',
}


def transpile_stim_circuit(circuit: stim.Circuit, basis_gates: List[str]) -> stim.Circuit:
    """
    Transpile a stim circuit to use only the specified basis gates.

    Args:
        circuit: The stim circuit to transpile
        basis_gates: List of allowed gate names (Qiskit-style, e.g., ['sx', 'x', 's', 'cx'])

    Returns:
        A new stim circuit using only the basis gates
    """
    if basis_gates is None:
        return circuit

    # Convert qiskit gate names to stim gate names
    stim_basis_gates: Set[str] = set()
    for gate in basis_gates:
        gate_lower = gate.lower()
        if gate_lower in QISKIT_TO_STIM:
            stim_basis_gates.add(QISKIT_TO_STIM[gate_lower])
        else:
            # Try uppercase directly (might already be stim format)
            stim_basis_gates.add(gate.upper())

    # Always allow measurement and reset operations
    stim_basis_gates.update({'M', 'MR', 'MX', 'MY', 'MZ', 'R', 'RX', 'RY', 'RZ',
                            'TICK', 'DETECTOR', 'OBSERVABLE_INCLUDE', 'QUBIT_COORDS',
                            'PAULI_CHANNEL_1', 'PAULI_CHANNEL_2', 'DEPOLARIZE1', 'DEPOLARIZE2'})

    transpiled = stim.Circuit()

    for instruction in circuit:
        name = instruction.name
        targets = instruction.targets_copy()
        args = instruction.gate_args_copy()

        # Check if gate is already in basis set
        if name in stim_basis_gates:
            transpiled.append(instruction)
            continue

        # Check if we have a decomposition
        if name in STIM_GATE_DECOMPOSITIONS:
            qubit_indices = [t.value for t in targets if t.is_qubit_target]
            decomposition = STIM_GATE_DECOMPOSITIONS[name]

            for basis_gate, selector in decomposition:
                if selector == 'all':
                    for q in qubit_indices:
                        transpiled.append(basis_gate, [q])
                elif selector == 'pairs':
                    # For 2-qubit gates, apply to pairs
                    for i in range(0, len(qubit_indices), 2):
                        transpiled.append(basis_gate, [qubit_indices[i], qubit_indices[i+1]])
        else:
            # No decomposition available, keep original gate
            # (will work if stim supports it natively)
            transpiled.append(instruction)

    return transpiled


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

    def _get_qubit_noise_config(self, qubit_index: int, basic_name: str) -> Dict:
        """
        Get the noise config for a specific qubit.

        Args:
            qubit_index: Index of the qubit
            basic_name: The name of the instruction being run

        Returns:
            Dict with p_x, p_y, p_z (missing keys default to 0)
        """
        config = self._pauli_noise_config
        qubits_list = config.get('qubits', [])
        general_config = config.get('general', {})

        if qubit_index < len(qubits_list):
            qubit_settings = qubits_list[qubit_index]
        else:
            qubit_settings = general_config

        noise_settings = qubit_settings.get(basic_name)
        if noise_settings is None:
            noise_settings = general_config.get('default')

        return noise_settings

    def _inject_noise(self, circuit: stim.Circuit) -> stim.Circuit:
        """
            circuit: Original stim.Circuit

            returns new stim.Circuit with noise injected after gates.
            Uses per-qubit Pauli noise from config.
        """
        if self._pauli_noise_config is None:
            return circuit

        noisy_circuit = stim.Circuit()

        for instruction in circuit:
            noisy_circuit.append(instruction)

            name = instruction.name
            targets = instruction.targets_copy()

            # IN THE FUTURE:
            #  Potentially add noise to M, MR, MX, MY and MZ, the measure operations
            #  Also potentially add noise to R, RX, RY and RZ, the reset operations
            if name in ('TICK', 'DETECTOR', 'OBSERVABLE_INCLUDE', 'QUBIT_COORDS',
                        'M', 'MR', 'MX', 'MY', 'MZ', 'R', 'RX', 'RY', 'RZ'):
                continue

            qubit_indices = [t.value for t in targets if t.is_qubit_target]

            if not qubit_indices:
                continue

            basic_name = STIM_TO_BASIC.get(name)

            if name in ('H', 'S', 'S_DAG', 'X', 'Y', 'Z', 'SQRT_X', 'SQRT_X_DAG',
                        'SQRT_Y', 'SQRT_Y_DAG', 'I'):
                # Single-qubit gate: add Pauli channel noise per qubit
                for qubit_idx in qubit_indices:
                    noise = self._get_qubit_noise_config(qubit_idx, basic_name)
                    if noise is None:
                        # No noise config for this gate, skip noise injection
                        continue
                    p_x, p_y, p_z = noise.get('p_x', 0), noise.get('p_y', 0), noise.get('p_z', 0)
                    if p_x > 0 or p_y > 0 or p_z > 0:
                        noisy_circuit.append('PAULI_CHANNEL_1', [qubit_idx], [p_x, p_y, p_z])

            elif name in ('CNOT', 'CX', 'CZ', 'CY', 'SWAP', 'ISWAP', 'ISWAP_DAG',
                          'SQRT_XX', 'SQRT_YY', 'SQRT_ZZ'):
                # Two-qubit gate: first apply 1-qubit Pauli error to each qubit independently
                for qubit_idx in qubit_indices:
                    noise = self._get_qubit_noise_config(qubit_idx, basic_name)
                    if noise is None:
                        continue
                    p_x, p_y, p_z = noise.get('p_x', 0), noise.get('p_y', 0), noise.get('p_z', 0)
                    if p_x > 0 or p_y > 0 or p_z > 0:
                        noisy_circuit.append('PAULI_CHANNEL_1', [qubit_idx], [p_x, p_y, p_z])
                # Then apply depolarising channel to each two qubit pair
                for i in range(0, len(qubit_indices), 2):
                    qubit_a = qubit_indices[i]
                    qubit_b = qubit_indices[i+1]
                    noisy_circuit.append('DEPOLARIZE2', [qubit_a, qubit_b], [DEPOL_2q_P])

        return noisy_circuit
