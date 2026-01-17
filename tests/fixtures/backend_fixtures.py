"""
Backend fixtures for testing

This module provides mocked quantum backend fixtures used across the test suite,
including Qiskit simulators and AWS Braket devices.
"""

import pytest
from unittest.mock import Mock, MagicMock
import json


@pytest.fixture
def mock_qiskit_backend():
    """
    Mock Qiskit AerSimulator backend

    Returns:
        Mock: Mocked Qiskit backend with standard configuration
    """
    backend = Mock()

    # Mock configuration
    config = Mock()
    config.basis_gates = ['sx', 'rz', 'cx', 'id']
    config.n_qubits = 5
    config.backend_name = "aer_simulator"
    backend.configuration.return_value = config
    backend.name.return_value = "aer_simulator"

    # Mock properties
    props = Mock()
    props_dict = {
        "backend_name": "aer_simulator",
        "qubits": [
            [{"name": "T1", "value": 50e-6, "unit": "us"}]
        ],
        "gates": []
    }
    props.to_dict.return_value = props_dict
    backend.properties.return_value = props

    return backend


@pytest.fixture
def mock_noisy_qiskit_backend():
    """
    Mock noisy Qiskit simulator with noise model

    Returns:
        Mock: Mocked noisy Qiskit backend
    """
    backend = Mock()

    # Configuration
    config = Mock()
    config.basis_gates = ['sx', 'rz', 'cx', 'id', 'reset', 'measure']
    config.n_qubits = 5
    config.backend_name = "noisy_simulator"
    backend.configuration.return_value = config
    backend.name.return_value = "noisy_simulator"

    # Properties with noise
    props = Mock()
    props_dict = {
        "backend_name": "noisy_simulator",
        "qubits": [
            [
                {"name": "T1", "value": 30e-6, "unit": "us"},
                {"name": "T2", "value": 45e-6, "unit": "us"},
                {"name": "frequency", "value": 5.0, "unit": "GHz"}
            ]
        ],
        "gates": [
            {
                "name": "cx",
                "qubits": [0, 1],
                "parameters": [
                    {"name": "gate_error", "value": 0.02, "unit": ""},
                    {"name": "gate_length", "value": 50, "unit": "ns"}
                ]
            }
        ]
    }
    props.to_dict.return_value = props_dict
    backend.properties.return_value = props

    return backend


@pytest.fixture
def mock_braket_device():
    """
    Mock AWS Braket LocalSimulator device

    Returns:
        Mock: Mocked Braket device
    """
    device = Mock()

    # Device properties as JSON
    device.properties.json.return_value = json.dumps({
        "provider": {
            "name": "Amazon Braket",
            "version": "1.0"
        },
        "deviceType": "SIMULATOR",
        "deviceCapabilities": {
            "paradigm": {
                "qubitCount": 25
            }
        }
    })

    # Mock run method returning a task
    task = Mock()
    task.id = "mock-task-id-12345"
    device.run.return_value = task

    return device


@pytest.fixture
def mock_braket_sv1_device():
    """
    Mock AWS Braket SV1 (state vector) simulator

    Returns:
        Mock: Mocked SV1 device
    """
    device = Mock()

    device.properties.json.return_value = json.dumps({
        "provider": {
            "name": "Amazon Braket",
            "version": "1.0"
        },
        "deviceType": "SIMULATOR",
        "deviceName": "SV1",
        "deviceCapabilities": {
            "paradigm": {
                "qubitCount": 34
            }
        }
    })

    task = Mock()
    task.id = "mock-sv1-task-id"
    device.run.return_value = task

    return device


@pytest.fixture
def mock_noise_model():
    """
    Mock Qiskit NoiseModel

    Returns:
        Mock: Mocked noise model
    """
    noise_model = Mock()
    noise_model.name = "test_noise_model"
    noise_model.basis_gates = ['sx', 'rz', 'cx', 'id']
    noise_model.noise_qubits = [0, 1, 2, 3, 4]

    # Mock the to_dict method
    noise_model.to_dict.return_value = {
        "errors": [],
        "basis_gates": ['sx', 'rz', 'cx', 'id']
    }

    return noise_model


@pytest.fixture
def mock_fake_backend_class():
    """
    Mock Qiskit fake backend class (e.g., FakeTokyo)

    Returns:
        Mock: Mocked fake backend class that can be instantiated
    """
    backend_instance = Mock()

    # Configuration
    config = Mock()
    config.basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    config.n_qubits = 5
    config.backend_name = "fake_tokyo"
    backend_instance.configuration.return_value = config

    # Properties
    props = Mock()
    props_dict = {
        "backend_name": "fake_tokyo",
        "qubits": [
            [
                {"name": "T1", "value": 50e-6, "unit": "us"},
                {"name": "T2", "value": 70e-6, "unit": "us"},
                {"name": "frequency", "value": 5.0, "unit": "GHz"}
            ]
        ],
        "gates": [
            {
                "name": "cx",
                "qubits": [0, 1],
                "parameters": [
                    {"name": "gate_error", "value": 0.01, "unit": ""},
                    {"name": "gate_length", "value": 40, "unit": "ns"}
                ]
            }
        ]
    }
    props.to_dict.return_value = props_dict
    backend_instance.properties.return_value = props

    # Create a mock class that returns the instance when called
    backend_class = Mock(return_value=backend_instance)

    return backend_class


@pytest.fixture
def mock_github_api(monkeypatch):
    """
    Mock GitHub API calls for backend config fetching

    Args:
        monkeypatch: pytest fixture for modifying imports

    Returns:
        dict: Dictionary of mock functions for GitHub API
    """
    def mock_get_commit_sha(owner, repo, branch):
        """Mock getting commit SHA"""
        return "abc123def456789" if branch == "stable/0.46" else None

    def mock_download_files(backend_name, commit_sha):
        """Mock downloading backend files from GitHub"""
        return {
            f"conf_{backend_name}.json": json.dumps({
                "backend_name": backend_name,
                "n_qubits": 5,
                "basis_gates": ["cx", "id", "rz", "sx", "x"]
            }),
            f"defs_{backend_name}.json": json.dumps({
                "gates": []
            }),
            f"props_{backend_name}.json": json.dumps({
                "backend_name": backend_name,
                "qubits": [[
                    {"name": "T1", "value": 50e-6, "unit": "us"},
                    {"name": "T2", "value": 70e-6, "unit": "us"}
                ]],
                "gates": []
            })
        }

    return {
        "get_commit_sha": mock_get_commit_sha,
        "download_files": mock_download_files
    }


@pytest.fixture
def mock_qiskit_task():
    """
    Mock Qiskit task (job) for circuit execution

    Returns:
        Mock: Mocked Qiskit job
    """
    task = Mock()
    task.job_id.return_value = "mock-qiskit-job-123"

    # Mock result
    result = Mock()
    result.get_counts.return_value = {"00": 512, "11": 512}
    task.result.return_value = result

    return task


@pytest.fixture
def mock_braket_task():
    """
    Mock Braket task for circuit execution

    Returns:
        Mock: Mocked Braket task
    """
    task = Mock()
    task.id = "mock-braket-task-456"

    # Mock result
    result = Mock()
    result.measurement_counts = {"00": 512, "11": 512}
    result.measurements = None
    task.result.return_value = result

    return task
