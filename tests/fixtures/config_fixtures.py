"""
Configuration fixtures for testing

This module provides various configuration fixtures used across the test suite,
including minimal configs, multi-device configs, hardware constants, and backend properties.
"""

import pytest
import json
from pathlib import Path


@pytest.fixture
def minimal_config():
    """
    Minimal valid configuration for testing

    Returns:
        dict: Configuration with all required fields
    """
    return {
        "num_qubits": 4,
        "circuit_optimisation_level": 1,
        "optimisation_iterations": 1,
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


@pytest.fixture
def multi_device_config(minimal_config):
    """
    Configuration with multiple device-specific settings

    Returns:
        dict: Configuration with device-specific power configs and noise models
    """
    config = minimal_config.copy()
    config["power_configs"]["noisy_sim"] = {
        "cx": 3,
        "rz": 2,
        "sx": 1.5,
        "x": 1.5
    }
    config["noise_models"]["noisy_sim"] = {
        "type": "simple_nm",
        "num_qubits": 4,
        "T1s": 50e-6,
        "T2s": 70e-6
    }
    return config


@pytest.fixture
def simple_noise_model_config():
    """
    Configuration for simple noise model (not fake backend)

    Returns:
        dict: Noise model config with T1/T2 values
    """
    return {
        "num_qubits": 4,
        "circuit_optimisation_level": 1,
        "optimisation_iterations": 1,
        "device_tracking": {"noisy_sim": 0},
        "power_configs": {
            "default_power_config": {"cx": 2, "rz": 1, "sx": 1, "x": 1}
        },
        "noise_models": {
            "default": {
                "type": "simple_nm",
                "num_qubits": 4,
                "T1s": 50e-6,
                "T2s": 70e-6
            }
        },
        "control_parameters": {}
    }


@pytest.fixture
def hardware_constants():
    """
    Hardware constants for builder testing

    Returns:
        dict: Hardware constants grouped by era (modern/intermediate/legacy)
    """
    return {
        "modern": {
            "builder_class": "DefaultBuilder",
            "y0": 1e5,
            "ymxc_y0": 0.8,
            "T_env": 0.050,
            "gamma_psi_base": 1e3,
            "delta": 1.764e-23,
            "E_c": 3.5e-24,
            "E_J": 2.1e-22,
            "R_n": 5000,
            "subgap_transparency": 0.01
        },
        "intermediate": {
            "builder_class": "DefaultBuilder",
            "y0": 8e4,
            "ymxc_y0": 0.75,
            "T_env": 0.060,
            "gamma_psi_base": 1.5e3,
            "delta": 1.764e-23,
            "E_c": 3.0e-24,
            "E_J": 2.0e-22,
            "R_n": 4500,
            "subgap_transparency": 0.015
        },
        "legacy": {
            "builder_class": "DefaultBuilder",
            "y0": 5e4,
            "ymxc_y0": 0.7,
            "T_env": 0.070,
            "gamma_psi_base": 2e3,
            "delta": 1.764e-23,
            "E_c": 2.5e-24,
            "E_J": 1.8e-22,
            "R_n": 4000,
            "subgap_transparency": 0.02
        }
    }


@pytest.fixture
def backend_props_tokyo():
    """
    Minimal Tokyo backend properties structure for testing

    Returns:
        dict: Backend properties matching Qiskit BackendProperties format
    """
    return {
        "backend_name": "tokyo",
        "backend_version": "1.0.0",
        "last_update_date": "2023-01-01T00:00:00Z",
        "qubits": [
            [
                {"name": "T1", "value": 50e-6, "unit": "us"},
                {"name": "T2", "value": 70e-6, "unit": "us"},
                {"name": "frequency", "value": 5.0, "unit": "GHz"},
                {"name": "anharmonicity", "value": -0.33, "unit": "GHz"},
                {"name": "readout_error", "value": 0.02, "unit": ""}
            ],
            [
                {"name": "T1", "value": 48e-6, "unit": "us"},
                {"name": "T2", "value": 65e-6, "unit": "us"},
                {"name": "frequency", "value": 5.1, "unit": "GHz"},
                {"name": "anharmonicity", "value": -0.32, "unit": "GHz"},
                {"name": "readout_error", "value": 0.025, "unit": ""}
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
            },
            {
                "name": "sx",
                "qubits": [0],
                "parameters": [
                    {"name": "gate_error", "value": 0.0005, "unit": ""},
                    {"name": "gate_length", "value": 35, "unit": "ns"}
                ]
            }
        ],
        "general": []
    }


@pytest.fixture
def backend_props_perth():
    """
    Perth backend properties structure for testing modern hardware

    Returns:
        dict: Backend properties for a modern quantum processor
    """
    return {
        "backend_name": "perth",
        "backend_version": "2.0.0",
        "last_update_date": "2024-01-01T00:00:00Z",
        "qubits": [
            [
                {"name": "T1", "value": 80e-6, "unit": "us"},
                {"name": "T2", "value": 120e-6, "unit": "us"},
                {"name": "frequency", "value": 5.2, "unit": "GHz"},
                {"name": "anharmonicity", "value": -0.35, "unit": "GHz"},
                {"name": "readout_error", "value": 0.015, "unit": ""}
            ]
        ],
        "gates": [
            {
                "name": "sx",
                "qubits": [0],
                "parameters": [
                    {"name": "gate_error", "value": 0.0003, "unit": ""},
                    {"name": "gate_length", "value": 30, "unit": "ns"}
                ]
            }
        ],
        "general": []
    }


@pytest.fixture
def real_config_setup(tmp_path, minimal_config, hardware_constants, backend_props_tokyo):
    """
    Setup for integration tests using real config files

    Creates actual config files on disk for integration testing.
    Must call fetch_config_files() before use to reset backend configs.

    Args:
        tmp_path: pytest fixture providing temporary directory
        minimal_config: fixture providing minimal config
        hardware_constants: fixture providing hardware constants
        backend_props_tokyo: fixture providing Tokyo backend props

    Returns:
        dict: Paths to created config files
    """
    # Write configs.json
    config_file = tmp_path / "configs.json"
    with open(config_file, 'w') as f:
        json.dump(minimal_config, f, indent=2)

    # Write hardware_constants.json
    backend_dir = tmp_path / "qiskit_backend_configs"
    backend_dir.mkdir()
    hw_file = backend_dir / "hardware_constants.json"
    with open(hw_file, 'w') as f:
        json.dump(hardware_constants, f, indent=2)

    # Write Tokyo backend props
    tokyo_dir = backend_dir / "tokyo"
    tokyo_dir.mkdir()

    # Create props file
    props_file = tokyo_dir / "props_tokyo.json"
    with open(props_file, 'w') as f:
        json.dump(backend_props_tokyo, f, indent=2)

    # Create original copy for reset functionality
    original_props = tokyo_dir / "props_tokyo_original.json"
    with open(original_props, 'w') as f:
        json.dump(backend_props_tokyo, f, indent=2)

    # Create minimal conf and defs files (required by some functions)
    conf_file = tokyo_dir / "conf_tokyo.json"
    with open(conf_file, 'w') as f:
        json.dump({
            "backend_name": "tokyo",
            "backend_version": "1.0.0",
            "n_qubits": 2,
            "basis_gates": ["cx", "id", "rz", "sx", "x"],
            "coupling_map": [[0, 1]],
            "simulator": False
        }, f, indent=2)

    defs_file = tokyo_dir / "defs_tokyo.json"
    with open(defs_file, 'w') as f:
        json.dump({"gates": []}, f, indent=2)

    return {
        "config_path": config_file,
        "hardware_path": hw_file,
        "backend_folder": backend_dir,
        "tokyo_dir": tokyo_dir,
        "tokyo_props": props_file,
        "tokyo_props_original": original_props,
        "tokyo_conf": conf_file,
        "tokyo_defs": defs_file
    }
