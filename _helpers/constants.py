import logging
from qiskit_ibm_runtime.fake_provider import (
    FakeAlmadenV2, FakeArmonkV2, FakeAthensV2, FakeAuckland, FakeBelemV2, FakeBoeblingenV2, FakeBogotaV2, FakeBrooklynV2, FakeBurlingtonV2, FakeCairoV2, FakeCambridgeV2, FakeCasablancaV2, FakeEssexV2, FakeGeneva, FakeGuadalupeV2, FakeHanoiV2, FakeJakartaV2, FakeJohannesburgV2, FakeKolkataV2, FakeLagosV2, FakeLimaV2, FakeLondonV2, FakeManhattanV2, FakeManilaV2, FakeMelbourneV2, FakeMontrealV2, FakeMumbaiV2, FakeNairobiV2, FakeOslo, FakeOurenseV2, FakeParisV2, FakePerth, FakePoughkeepsieV2, FakePrague, FakeQuitoV2, FakeRochesterV2, FakeRomeV2, FakeRueschlikon, FakeSantiagoV2, FakeSherbrooke, FakeSingaporeV2, FakeSydneyV2, FakeTenerife, FakeTokyo, FakeTorontoV2, FakeValenciaV2, FakeVigoV2, FakeWashingtonV2, FakeYorktownV2,
)

Pauli_nm_FD = 1 # Frequency dependence of the pauli nm

DEFAULT_PATH = "tutorials/circuit_execution_quality_metrics/quantum_volume/quantum_volume.py"

# Algorithm name to file path mapping
# Use short names like "quantum_volume" or "qv" instead of full paths
ALGORITHM_PATHS = {
    # Circuit execution quality metrics
    "quantum_volume": "tutorials/circuit_execution_quality_metrics/quantum_volume/quantum_volume.py",
    "qv": "tutorials/circuit_execution_quality_metrics/quantum_volume/quantum_volume.py",
    "grovers_search": "tutorials/circuit_execution_quality_metrics/grovers_search/grovers_search.py",
    "grover": "tutorials/circuit_execution_quality_metrics/grovers_search/grovers_search.py",
    "vqe": "tutorials/well_studied_task_execution_quality_metrics/vqe/vqe.py",
    "stim_test": "tutorials/qec_metrics/simple_stim_test/simple_stim_test.py",
    "clifford_qv": "tutorials/qec_metrics/clifford_quantum_volume/clifford_quantum_volume.py"
}


def resolve_metric_path(path_or_name: str) -> str:
    if path_or_name in ALGORITHM_PATHS:
        return ALGORITHM_PATHS[path_or_name]
    logging.warning("Didn't find the path name in ALGORITHM_PATHS, assume the input was a path already")

    # Assume it's a file path and return as is
    return path_or_name

SIMULATION_METHOD = "density_matrix"

HARDWARE_CONFIG_GROUPS = {
    "modern": ("default", "sherbrooke"),
    "intermediate": (),
    "legacy": ()
}

EXISTING_MODELS = {
    "fakeAlmadenV2": FakeAlmadenV2,
    "fakeArmonkV2": FakeArmonkV2,
    "fakeAthensV2": FakeAthensV2,
    "fakeAuckland": FakeAuckland,
    "fakeBelemV2": FakeBelemV2,
    "fakeBoeblingenV2": FakeBoeblingenV2,
    "fakeBogotaV2": FakeBogotaV2,
    "fakeBrooklynV2": FakeBrooklynV2,
    "fakeBurlingtonV2": FakeBurlingtonV2,
    "fakeCairoV2": FakeCairoV2,
    "fakeCambridgeV2": FakeCambridgeV2,
    "fakeCasablancaV2": FakeCasablancaV2,
    "fakeEssexV2": FakeEssexV2,
    "fakeGeneva": FakeGeneva,
    "fakeGuadalupeV2": FakeGuadalupeV2,
    "fakeHanoiV2": FakeHanoiV2,
    "fakeJakartaV2": FakeJakartaV2,
    "fakeJohannesburgV2": FakeJohannesburgV2,
    "fakeKolkataV2": FakeKolkataV2,
    "fakeLagosV2": FakeLagosV2,
    "fakeLimaV2": FakeLimaV2,
    "fakeLondonV2": FakeLondonV2,
    "fakeManhattanV2": FakeManhattanV2,
    "fakeManilaV2": FakeManilaV2,
    "fakeMelbourneV2": FakeMelbourneV2,
    "fakeMontrealV2": FakeMontrealV2,
    "fakeMumbaiV2": FakeMumbaiV2,
    "fakeNairobiV2": FakeNairobiV2,
    "fakeOslo": FakeOslo,
    "fakeOurenseV2": FakeOurenseV2,
    "fakeParisV2": FakeParisV2,
    "fakePerth": FakePerth,
    "fakePoughkeepsieV2": FakePoughkeepsieV2,
    "fakePrague": FakePrague,
    "fakeQuitoV2": FakeQuitoV2,
    "fakeRochesterV2": FakeRochesterV2,
    "fakeRomeV2": FakeRomeV2,
    "fakeRueschlikon": FakeRueschlikon,
    "fakeSantiagoV2": FakeSantiagoV2,
    "fakeSherbrooke": FakeSherbrooke,
    "fakeSingaporeV2": FakeSingaporeV2,
    "fakeSydneyV2": FakeSydneyV2,
    "fakeTenerife": FakeTenerife,
    "fakeTokyo": FakeTokyo,
    "fakeTorontoV2": FakeTorontoV2,
    "fakeValenciaV2": FakeValenciaV2,
    "fakeVigoV2": FakeVigoV2,
    "fakeWashingtonV2": FakeWashingtonV2,
    "fakeYorktownV2": FakeYorktownV2,
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
    'rz',
    'p',
    'u1'
    's'
    'sdg'
    't',
    'tdg'
    'z',
    'id'
    'barrier'
    'delay',    # Delay (no active operation,
}

# All single-qubit gates
SINGLE_QUBIT_GATES = {
    'x', 'y', 'z',
    'h',
    's', 'sdg', 't', 'tdg', 'p', 'u1',
    'rx', 'ry', 'rz', 'r',
    'u', 'u2', 'u3',
    'sx', 'sxdg',
    'id',
    'reset', 'measure',
}

# All two-qubit gates
TWO_QUBIT_GATES = {
    'cx', 'cy', 'cz',
    'cp', 'cu1', 'cs', 'csdg', 'csx',
    'crx', 'cry', 'crz',
    'cu', 'cu2', 'cu3', 'ch',
    'swap', 'iswap',
    'rxx', 'ryy', 'rzz', 'rzx',
    'ecr',
    'cz',
    'dcx',
}

# =============================================================================
# Stim Stabilizer Simulation (QEC Metrics)
# =============================================================================

# Metrics that use Stim stabilizer simulator for QEC benchmarks.
# QEC metrics are written directly in Stim and use StimCircuitSubmitter.
# Add metric names here as QEC metrics are created.
CLIFFORD_METRICS = [
    "simple_stim_test",
    "clifford_quantum_volume"
]

# Basis gates for Stim stabilizer simulation, matching superconducting hardware.
# Uses the same native gate set as real IBM devices (sx, x, rz, cx).
# All Clifford gates decompose into these (e.g., H = RZ·SX·RZ, S = RZ(π/2)).
STIM_TO_BASIC = {
    # Single-qubit gates
    'H': 'h',
    'S': 's',
    'S_DAG': 'sdg',
    'X': 'x',
    'Y': 'y',
    'Z': 'z',
    'SQRT_X': 'sx',
    'SQRT_X_DAG': 'sxdg',
    'SQRT_Y': 'sy',
    'SQRT_Y_DAG': 'sydg',
    'SQRT_Z': 's',
    'SQRT_Z_DAG': 'sdg',
    'I': 'id',
    'H_XY': 'h_xy',
    'H_XZ': 'h',
    'H_YZ': 'h_yz',
    'C_XYZ': 'c_xyz',
    'C_ZYX': 'c_zyx',
    
    # Two-qubit gates
    'CNOT': 'cx',
    'CX': 'cx',
    'CZ': 'cz',
    'CY': 'cy',
    'SWAP': 'swap',
    'ISWAP': 'iswap',
    'ISWAP_DAG': 'iswap_dag',
    'SQRT_XX': 'rxx',
    'SQRT_YY': 'ryy',
    'SQRT_ZZ': 'rzz',
    'SQRT_XX_DAG': 'rxx_dag',
    'SQRT_YY_DAG': 'ryy_dag',
    'SQRT_ZZ_DAG': 'rzz_dag',
    'XCX': 'xcx',
    'XCY': 'xcy',
    'XCZ': 'xcz',
    'YCX': 'ycx',
    'YCY': 'ycy',
    'YCZ': 'ycz',
    'CXSWAP': 'cxswap',
    'SWAPCX': 'swapcx',
    'CZSWAP': 'czswap',
}

RequiredStimGates = ['h', 's', 'sdg', 'x', 'y', 'z', 'sx', 'sxdg', 'cx', 'cz']

StimBasisGates = ['H', 'S', 'S_DAG', 'X', 'Y', 'Z', 'SQRT_X', 'SQRT_X_DAG', 'CX', 'CZ']
