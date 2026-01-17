"""
Quantum circuit fixtures for testing

This module provides quantum circuit fixtures used across the test suite,
including sample circuits in various formats (Qiskit, OpenQASM, Braket).
"""

import pytest
from unittest.mock import Mock


@pytest.fixture
def sample_qiskit_circuit():
    """
    Sample Qiskit QuantumCircuit for testing

    Returns:
        Mock: Mocked Qiskit circuit with basic properties
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    return qc


@pytest.fixture
def sample_qasm_string():
    """
    Sample OpenQASM 2.0 string for testing

    Returns:
        str: QASM representation of a Bell state circuit
    """
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""


@pytest.fixture
def sample_qasm_no_measurements():
    """
    Sample OpenQASM string without measurements

    Returns:
        str: QASM representation without measurement operations
    """
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
"""


@pytest.fixture
def sample_multi_qubit_circuit():
    """
    Sample multi-qubit circuit (5 qubits) for testing

    Returns:
        QuantumCircuit: 5-qubit circuit with multiple gates
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(5, 5)
    # Create GHZ state
    qc.h(0)
    for i in range(4):
        qc.cx(i, i+1)
    qc.measure(range(5), range(5))

    return qc


@pytest.fixture
def sample_circuit_list():
    """
    List of sample circuits for batch testing

    Returns:
        list: List of QuantumCircuit objects
    """
    from qiskit import QuantumCircuit

    circuits = []

    # Bell state
    qc1 = QuantumCircuit(2, 2)
    qc1.h(0)
    qc1.cx(0, 1)
    qc1.measure([0, 1], [0, 1])
    circuits.append(qc1)

    # Single qubit X gate
    qc2 = QuantumCircuit(1, 1)
    qc2.x(0)
    qc2.measure(0, 0)
    circuits.append(qc2)

    # Superposition
    qc3 = QuantumCircuit(1, 1)
    qc3.h(0)
    qc3.measure(0, 0)
    circuits.append(qc3)

    return circuits


@pytest.fixture
def sample_qasm_list():
    """
    List of QASM strings for batch testing

    Returns:
        list: List of QASM string representations
    """
    qasm_list = [
        """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
""",
        """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
x q[0];
measure q -> c;
""",
        """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
h q[0];
measure q -> c;
"""
    ]
    return qasm_list


@pytest.fixture
def sample_braket_circuit():
    """
    Sample Braket Circuit for testing

    Returns:
        Mock: Mocked Braket circuit
    """
    # Create a mock since we may not have Braket installed
    circuit = Mock()
    circuit.qubit_count = 2
    circuit.to_ir.return_value = Mock()
    return circuit


@pytest.fixture
def sample_counts_dict():
    """
    Sample measurement counts dictionary

    Returns:
        dict: Measurement counts in Qiskit format
    """
    return {
        "00": 512,
        "01": 128,
        "10": 256,
        "11": 128
    }


@pytest.fixture
def sample_braket_counts():
    """
    Sample measurement counts in Braket format

    Returns:
        dict: Measurement counts with reversed bit order
    """
    return {
        "00": 512,  # Braket uses opposite bit order
        "10": 128,
        "01": 256,
        "11": 128
    }


@pytest.fixture
def empty_circuit():
    """
    Empty quantum circuit for edge case testing

    Returns:
        QuantumCircuit: Circuit with no gates
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2)
    # No gates added
    return qc


@pytest.fixture
def circuit_with_custom_gates():
    """
    Circuit with custom/uncommon gate types

    Returns:
        QuantumCircuit: Circuit with various gate types
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.rx(0.5, 1)
    qc.ry(0.7, 2)
    qc.rz(0.3, 0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.ccx(0, 1, 2)  # Toffoli
    qc.measure(range(3), range(3))

    return qc


@pytest.fixture
def quantum_volume_circuit():
    """
    Sample quantum volume circuit structure

    Returns:
        QuantumCircuit: Simple QV-like circuit
    """
    from qiskit import QuantumCircuit
    import numpy as np

    qc = QuantumCircuit(4, 4)

    # Simple QV-like layer
    qc.h(0)
    qc.h(1)
    qc.cx(0, 1)
    qc.cx(2, 3)

    # Random rotations
    qc.rz(np.pi/4, 0)
    qc.rz(np.pi/3, 1)

    # Measurement
    qc.measure(range(4), range(4))

    return qc
