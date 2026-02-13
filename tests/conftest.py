"""
Shared pytest fixtures and configuration for QCMet test suite

This module provides fixtures that are automatically available to all tests,
including registry resets and environment variable management.
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))


@pytest.fixture(autouse=True)
def reset_registries():
    """
    Reset all global registries before each test

    This fixture automatically runs before every test to ensure test isolation.
    It clears all registry state (submitters, optimisations, control parameters).
    """
    from _helpers.registry import submitter_registry, control_parameter_registry
    from _helpers.builders.base import builder_registry

    # Store original state
    original_submitters = submitter_registry._submitters.copy()
    original_control_params = control_parameter_registry.control_parameters.copy()
    original_builders = builder_registry.builders.copy()

    # Clear registries
    submitter_registry._submitters = {}
    control_parameter_registry.control_parameters = {}
    builder_registry.builders = {}

    yield

    # Restore original state (though typically tests should not depend on this)
    submitter_registry._submitters = original_submitters
    control_parameter_registry.control_parameters = original_control_params
    builder_registry.builders = original_builders


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """
    Reset environment variables before each test

    This fixture automatically runs before every test to ensure clean environment.
    It removes all QCMet-specific environment variables.
    """
    env_vars = [
        "CONFIG_PATH",
        "HARDWARE_CONFIG_PATH",
        "BACKEND_CONFIGS_FOLDER",
        "SINGLE_RUN",
        "iteration",
        "PERF_VALUE",
        "NUM_QUBITS",
        "CIRCUIT_OPTIMIZATION"
    ]

    for var in env_vars:
        monkeypatch.delenv(var, raising=False)

    yield


@pytest.fixture
def mock_filesystem(tmp_path, monkeypatch):
    """
    Create temporary directory structure for file system testing

    Sets up a minimal file system with:
    - configs.json location
    - hardware_constants.json location
    - qiskit_backend_configs/ folder

    Args:
        tmp_path: pytest fixture providing temporary directory
        monkeypatch: pytest fixture for modifying environment

    Returns:
        Path object pointing to the temporary directory root
    """
    # Create backend configs directory
    config_dir = tmp_path / "qiskit_backend_configs"
    config_dir.mkdir()

    # Set environment variables to point to temp locations
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "configs.json"))
    monkeypatch.setenv("HARDWARE_CONFIG_PATH", str(config_dir / "hardware_constants.json"))
    monkeypatch.setenv("BACKEND_CONFIGS_FOLDER", str(config_dir) + "/")

    return tmp_path


@pytest.fixture
def sample_config_path(tmp_path, monkeypatch):
    """
    Create a sample configs.json file for testing

    Returns:
        Path to the created config file
    """
    import json

    config = {
        "num_qubits": 5,
        "circuit_optimisation_level": 1,
        "optimisation_iterations": 10,
        "device_tracking": {
            "noiseless_sim": 0,
            "noisy_sim": 1
        },
        "power_configs": {
            "default_power_config": {
                "cx": 2,
                "rz": 1,
                "sx": 1,
                "x": 1
            }
        },
        "noise_models": {
            "default": {
                "type": "fake_backend",
                "name": "tokyo",
                "control_parameters": {
                    "temperature": [[40, "mK"], [20, 10, "mK"]]
                },
                "init_control_parameters": {
                    "temperature": [13, "mK"]
                }
            }
        },
        "control_parameters": {
            "temperature": [[40, "mK"], [20, 10, "mK"]]
        }
    }

    config_file = tmp_path / "configs.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    monkeypatch.setenv("CONFIG_PATH", str(config_file))

    return config_file
