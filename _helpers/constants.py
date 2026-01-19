from qiskit_ibm_runtime.fake_provider import (
    FakeAuckland, FakeGeneva, FakeKolkataV2, FakeManilaV2,
    FakeMontrealV2, FakeOslo, FakePerth, FakePrague,
    FakeSherbrooke, FakeTokyo, FakeWashingtonV2, FakeBrooklynV2,
    FakeManhattanV2
)

DEFAULT_PATH = "tutorials/circuit_execution_quality_metrics/quantum_volume/quantum_volume.py"

# Algorithm name to file path mapping
# Use short names like "quantum_volume" or "qv" instead of full paths
ALGORITHM_PATHS = {
    # Circuit execution quality metrics
    "quantum_volume": "tutorials/circuit_execution_quality_metrics/quantum_volume/quantum_volume.py",
    "qv": "tutorials/circuit_execution_quality_metrics/quantum_volume/quantum_volume.py",

    # Algorithmic qubits - Qiskit implementations (default)
    "amplitude_estimation": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/amplitude-estimation/qiskit/ae_benchmark.py",
    "ae": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/amplitude-estimation/qiskit/ae_benchmark.py",

    "bernstein_vazirani": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/bernstein-vazirani/qiskit/bv_benchmark.py",
    "bv": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/bernstein-vazirani/qiskit/bv_benchmark.py",

    "deutsch_jozsa": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/deutsch-jozsa/qiskit/dj_benchmark.py",
    "dj": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/deutsch-jozsa/qiskit/dj_benchmark.py",

    "grovers": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/grovers/qiskit/grovers_benchmark.py",
    "grover": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/grovers/qiskit/grovers_benchmark.py",

    "hamiltonian_simulation": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/hamiltonian-simulation/qiskit/hamiltonian_simulation_benchmark.py",
    "ham_sim": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/hamiltonian-simulation/qiskit/hamiltonian_simulation_benchmark.py",

    "hidden_shift": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/hidden-shift/qiskit/hs_benchmark.py",
    "hs": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/hidden-shift/qiskit/hs_benchmark.py",

    "monte_carlo": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/monte-carlo/qiskit/mc_benchmark.py",
    "mc": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/monte-carlo/qiskit/mc_benchmark.py",

    "phase_estimation": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/phase-estimation/qiskit/pe_benchmark.py",
    "pe": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/phase-estimation/qiskit/pe_benchmark.py",

    "quantum_fourier_transform": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/quantum-fourier-transform/qiskit/qft_benchmark.py",
    "qft": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/quantum-fourier-transform/qiskit/qft_benchmark.py",

    "shors": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/shors/qiskit/shors_benchmark.py",
    "shor": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/shors/qiskit/shors_benchmark.py",

    "vqe": "tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/vqe/qiskit/vqe_benchmark.py",

    # Gate execution quality metrics
    "cycle_benchmarking": "tutorials/gate_execution_quality_metrics/cycle_benchmarking_composite_process_fidelity/cycle_benchmarking_composite_process_fidelity.py",
    "cb": "tutorials/gate_execution_quality_metrics/cycle_benchmarking_composite_process_fidelity/cycle_benchmarking_composite_process_fidelity.py",

    "randomized_benchmarking": "tutorials/gate_execution_quality_metrics/randomized_benchmarking/rb.py",
    "rb": "tutorials/gate_execution_quality_metrics/randomized_benchmarking/rb.py",

    "gst": "tutorials/gate_execution_quality_metrics/gst_based_gate_execution_quality_metrics/run_gst_circuits.py",

    # Qubit quality metrics
    "t1": "tutorials/qubit_quality_metrics/t1/braket_device_t1.py",

    # Well-studied task metrics
    "qft_task": "tutorials/well_studied_task_execution_quality_metrics/qft/qft.py",
    "vqe_task": "tutorials/well_studied_task_execution_quality_metrics/vqe/vqe.py",
}


def resolve_metric_path(path_or_name: str) -> str:
    """
    Resolve a metric path from either an algorithm name or a file path.

    Args:
        path_or_name: Either a short algorithm name (e.g., 'quantum_volume', 'qv')
                      or a full file path

    Returns:
        The resolved file path

    Raises:
        ValueError: If the algorithm name is not found and it's not a valid path
    """
    # Check if it's a known algorithm name
    if path_or_name in ALGORITHM_PATHS:
        return ALGORITHM_PATHS[path_or_name]

    # Otherwise, assume it's a file path and return as-is
    return path_or_name

SIMULATION_METHOD = "density_matrix"

HARDWARE_CONFIG_GROUPS = {
    "modern": ("default", "sherbrooke"),
    "intermediate": (),
    "legacy": ("oslo")
}

EXISTING_MODELS = {
    "fakeAuckland": FakeAuckland,
    "fakeGeneva": FakeGeneva,
    "fakeKolkataV2": FakeKolkataV2,
    "fakeManilaV2": FakeManilaV2,
    "fakeMontrealV2": FakeMontrealV2,
    "fakeOslo": FakeOslo,
    "fakePerth": FakePerth,
    "fakePrague": FakePrague,
    "fakeSherbrooke": FakeSherbrooke,
    "fakeTokyo": FakeTokyo,
    "fakeWashingtonV2": FakeWashingtonV2,
    "fakeBrooklynV2": FakeBrooklynV2,
    "fakeManhattanV2": FakeManhattanV2
}

DEFAULT_INSTRUCTION_TIMES = {
    "time_rz": 0,
    "time_sx": 50,
    "time_x": 100,
    "time_cx": 300,
    "time_reset": 1000,
    "time_measure": 100
}

PROPERTY_UNITS = {
    "T1": "us",
    "T2": "us", 
    "frequency": "GHz",
    "anharmonicity": "GHz",
    "readout_error": "",
    "prob_meas0_prep1": "",
    "prob_meas1_prep0": "",
    "readout_length": "ns",
    "gate_length": "ns",
    "gate_error": ""
}


SI_PREFIXES = {
    'P': 1e15,   # peta
    'T': 1e12,   # tera
    'G': 1e9,    # giga
    'M': 1e6,    # mega
    'k': 1e3,    # kilo
    '': 1,       # base unit
    'm': 1e-3,   # milli
    'μ': 1e-6,   # micro (Greek mu)
    'u': 1e-6,   # micro (alternative 'u' for systems that don't support μ)
    'n': 1e-9,   # nano
}

NoiselessSimBasisGates = ['u1', 'u2', 'u3', 'u', 'p', 'r', 'rx', 'ry', 'rz', 'id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'swap', 'cx', 'cy', 'cz', 'ch', 'cp', 'cu', 'cu1', 'cu2', 'cu3', 'crx', 'cry', 'crz', 'ccx', 'cswap', 'mcx', 'mcy', 'mcz', 'mcp', 'mcu1', 'mcu2', 'mcu3', 'mcswap', 'unitary', 'diagonal', 'multiplexer', 'initialize', 'delay', 'pauli']

DefaultBasisGatesNoiseless = ['rz']
DefaultBasisGates1qb = ['id', 'sx', 'x']
DefaultBasisGates2qb = ['cx']

# Comprehensive gate sets

# Virtual/noiseless gates - no physical operation, just software frame changes
NOISELESS_GATES = {
    'rz',       # Z rotation (virtual)
    'p',        # Phase gate (virtual)
    'u1',       # U1 gate (equivalent to p, virtual)
    's',        # S gate = sqrt(Z) (virtual)
    'sdg',      # S-dagger (virtual)
    't',        # T gate = sqrt(S) (virtual)
    'tdg',      # T-dagger (virtual)
    'z',        # Pauli Z (virtual)
    'id',       # Identity (no operation)
    'barrier',  # Barrier (compiler directive, not a gate)
    'delay',    # Delay (no active operation, but has decoherence)
}

# All single-qubit gates
SINGLE_QUBIT_GATES = {
    # Pauli gates
    'x', 'y', 'z',
    # Hadamard
    'h',
    # Phase gates
    's', 'sdg', 't', 'tdg', 'p', 'u1',
    # Rotation gates
    'rx', 'ry', 'rz', 'r',
    # General unitary gates
    'u', 'u2', 'u3',
    # Square root gates
    'sx', 'sxdg',
    # Identity
    'id',
    # Measurement and reset (single-qubit operations)
    'reset', 'measure',
}

# All two-qubit gates
TWO_QUBIT_GATES = {
    # Controlled Paulis
    'cx', 'cy', 'cz',
    # Controlled phase gates
    'cp', 'cu1', 'cs', 'csdg', 'csx',
    # Controlled rotations
    'crx', 'cry', 'crz',
    # Controlled unitaries
    'cu', 'cu2', 'cu3', 'ch',
    # SWAP family
    'swap', 'iswap',
    # Two-qubit rotations (Ising-type)
    'rxx', 'ryy', 'rzz', 'rzx',
    # Hardware-native gates
    'ecr',      # Echoed cross-resonance (IBM)
    'cz',       # Controlled-Z (common native gate)
    'dcx',      # Double CNOT
}
