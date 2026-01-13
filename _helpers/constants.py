from qiskit_ibm_runtime.fake_provider import (
    FakeAuckland, FakeGeneva, FakeKolkataV2, FakeManilaV2,
    FakeMontrealV2, FakeOslo, FakePerth, FakePrague,
    FakeSherbrooke, FakeTokyo, FakeWashingtonV2, FakeBrooklynV2,
    FakeManhattanV2
)

DEFAULT_PATH = "tutorials/circuit_execution_quality_metrics/quantum_volume/quantum_volume.py"

SIMULATION_METHOD = "density_matrix"

HARDWARE_CONFIG_GROUPS = {
    "modern": ("perth", "default", "sherbrooke"),
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
