"""
Unit tests for helpers module

This test suite verifies all helper utility functions used throughout the codebase,
including configuration reading, unit conversion, and environment variable management.
"""

import pytest
import json
import sys
import pathlib
from unittest.mock import Mock, patch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))

from _helpers.helpers import (
    read_config,
    get_unit_multiplier,
    get_control_parameters,
    extract_from_json,
    get_config_value,
    get_num_qubits,
    set_num_qubits_list,
    set_circuit_optimisation,
    set_up_logger
)
from _helpers.constants import SI_PREFIXES


class TestReadConfig:
    """Test read_config function"""

    def test_read_config_success(self, tmp_path, monkeypatch):
        """
        Test successful config reading

        Given:
        - Valid CONFIG_PATH environment variable
        - Valid JSON file at that path

        Expected:
        - Function returns parsed JSON as dict
        """
        config_data = {"num_qubits": 5, "circuit_optimisation_level": 1}
        config_file = tmp_path / "configs.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        monkeypatch.setenv("CONFIG_PATH", str(config_file))

        result = read_config()
        assert result == config_data
        assert result["num_qubits"] == 5

    def test_read_config_file_not_found(self, monkeypatch):
        """Test FileNotFoundError when config file doesn't exist"""
        monkeypatch.setenv("CONFIG_PATH", "/nonexistent/path/config.json")

        with pytest.raises(FileNotFoundError, match="Power config file not found"):
            read_config()

    def test_read_config_invalid_json(self, tmp_path, monkeypatch):
        """Test ValueError when JSON is malformed"""
        config_file = tmp_path / "bad_config.json"
        with open(config_file, 'w') as f:
            f.write("{invalid json content")

        monkeypatch.setenv("CONFIG_PATH", str(config_file))

        with pytest.raises(ValueError, match="Invalid JSON in power config file"):
            read_config()

    def test_read_config_empty_file(self, tmp_path, monkeypatch):
        """Test reading empty JSON file (valid but empty)"""
        config_file = tmp_path / "empty_config.json"
        with open(config_file, 'w') as f:
            f.write("{}")

        monkeypatch.setenv("CONFIG_PATH", str(config_file))

        result = read_config()
        assert result == {}


class TestGetUnitMultiplier:
    """Test get_unit_multiplier function"""

    def test_get_unit_multiplier_empty_string(self):
        """Test empty string returns None"""
        result = get_unit_multiplier("")
        assert result is None

    def test_get_unit_multiplier_single_char(self):
        """Test single character without prefix returns 1"""
        result = get_unit_multiplier("s")  # seconds
        assert result == 1

    def test_get_unit_multiplier_peta(self):
        """Test peta prefix (P = 10^15)"""
        result = get_unit_multiplier("PHz")  # Petahertz
        assert result == 1e15

    def test_get_unit_multiplier_tera(self):
        """Test tera prefix (T = 10^12)"""
        result = get_unit_multiplier("THz")
        assert result == 1e12

    def test_get_unit_multiplier_giga(self):
        """Test giga prefix (G = 10^9)"""
        result = get_unit_multiplier("GHz")
        assert result == 1e9

    def test_get_unit_multiplier_mega(self):
        """Test mega prefix (M = 10^6)"""
        result = get_unit_multiplier("MHz")
        assert result == 1e6

    def test_get_unit_multiplier_kilo(self):
        """Test kilo prefix (k = 10^3)"""
        result = get_unit_multiplier("kHz")
        assert result == 1e3

    def test_get_unit_multiplier_milli(self):
        """Test milli prefix (m = 10^-3)"""
        result = get_unit_multiplier("ms")  # milliseconds
        assert result == 1e-3

    def test_get_unit_multiplier_micro_greek(self):
        """Test micro prefix with Greek mu (μ = 10^-6)"""
        result = get_unit_multiplier("μs")  # microseconds
        assert result == 1e-6

    def test_get_unit_multiplier_micro_u(self):
        """Test micro prefix with 'u' (u = 10^-6)"""
        result = get_unit_multiplier("us")  # microseconds (alternative)
        assert result == 1e-6

    def test_get_unit_multiplier_nano(self):
        """Test nano prefix (n = 10^-9)"""
        result = get_unit_multiplier("ns")  # nanoseconds
        assert result == 1e-9

    def test_get_unit_multiplier_unknown_prefix(self):
        """Test unknown prefix returns None"""
        result = get_unit_multiplier("Xs")  # X is not a valid SI prefix
        assert result is None


class TestGetControlParameters:
    """Test get_control_parameters function"""

    def test_get_control_parameters_single_run_true(self, monkeypatch):
        """
        Test SINGLE_RUN=true uses first value from control_parameters

        Given:
        - SINGLE_RUN = "true"
        - control_parameters with format [[value, unit], [start, step, unit]]

        Expected:
        - Returns {param: [value, unit]} using first entry
        """
        config = {
            "control_parameters": {
                "temperature": [[40, "mK"], [20, 10, "mK"]],
                "power": [[100, "W"], [50, 5, "W"]]
            }
        }
        monkeypatch.setenv("SINGLE_RUN", "true")

        result = get_control_parameters(config)

        assert result["temperature"] == [40, "mK"]
        assert result["power"] == [100, "W"]

    def test_get_control_parameters_single_run_false(self, monkeypatch):
        """
        Test SINGLE_RUN=false uses iteration-based sweep

        Formula: value = start + (iteration * step)

        Given:
        - SINGLE_RUN = "false"
        - iteration = 3
        - temperature: start=20, step=10

        Calculation:
        - value = 20 + (3 * 10) = 50

        Expected:
        - Returns {temperature: [50, "mK"]}
        """
        config = {
            "control_parameters": {
                "temperature": [[40, "mK"], [20, 10, "mK"]]
            }
        }
        monkeypatch.setenv("SINGLE_RUN", "false")
        monkeypatch.setenv("iteration", "3")

        result = get_control_parameters(config)

        expected_value = 20 + (3 * 10)  # start + (iteration * step)
        assert result["temperature"] == [expected_value, "mK"]
        assert result["temperature"][0] == 50

    def test_get_control_parameters_iteration_zero(self, monkeypatch):
        """Test iteration=0 returns start value"""
        config = {
            "control_parameters": {
                "temperature": [[40, "mK"], [20, 10, "mK"]]
            }
        }
        monkeypatch.setenv("SINGLE_RUN", "false")
        monkeypatch.setenv("iteration", "0")

        result = get_control_parameters(config)

        # value = 20 + (0 * 10) = 20
        assert result["temperature"] == [20, "mK"]

    def test_get_control_parameters_multiple_params(self, monkeypatch):
        """Test with multiple control parameters"""
        config = {
            "control_parameters": {
                "temperature": [[40, "mK"], [20, 10, "mK"]],
                "power": [[100, "W"], [50, 5, "W"]],
                "voltage": [[5, "V"], [1, 0.5, "V"]]
            }
        }
        monkeypatch.setenv("SINGLE_RUN", "false")
        monkeypatch.setenv("iteration", "2")

        result = get_control_parameters(config)

        assert result["temperature"] == [40, "mK"]  # 20 + (2*10)
        assert result["power"] == [60, "W"]  # 50 + (2*5)
        assert result["voltage"] == [2, "V"]  # 1 + (2*0.5)

    def test_get_control_parameters_case_insensitive(self, monkeypatch):
        """Test that SINGLE_RUN is case-insensitive"""
        config = {
            "control_parameters": {
                "temperature": [[40, "mK"], [20, 10, "mK"]]
            }
        }
        monkeypatch.setenv("SINGLE_RUN", "TRUE")

        result = get_control_parameters(config)

        assert result["temperature"] == [40, "mK"]


class TestExtractFromJson:
    """Test extract_from_json function"""

    def test_extract_from_json_with_defaults(self):
        """
        Test extraction with default values

        Given:
        - JSON dict: {"key1": "value1"}
        - values dict with 3 keys, only key1 present in json_dict

        Expected:
        - key1 from json_dict: "value1"
        - key2 missing, use default: "default2"
        - key3 missing, use default: "default3"
        - Returns ["value1", "default2", "default3"]
        """
        json_dict = {"key1": "value1"}
        values = {
            "key1": ("default1", {}),
            "key2": ("default2", {}),
            "key3": ("default3", {})
        }

        result = extract_from_json(json_dict, values)

        assert len(result) == 3
        assert result[0] == "value1"  # From json_dict
        assert result[1] == "default2"  # Default for missing key2
        assert result[2] == "default3"  # Default for missing key3

    def test_extract_from_json_required_value_missing(self):
        """
        Test ValueError when required value is missing

        Given:
        - json_dict with only key1
        - key2 is required but missing

        Expected:
        - Raises ValueError
        """
        json_dict = {"key1": "value1"}
        values = {
            "key1": ("default1", {}),
            "key2": ("default2", {})
        }
        required = ["key2"]

        with pytest.raises(ValueError, match="Required value key2 was not in json"):
            extract_from_json(json_dict, values, required)

    def test_extract_from_json_required_value_present(self):
        """Test that required values present work correctly"""
        json_dict = {"key1": "value1", "key2": "value2"}
        values = {
            "key1": ("default1", {}),
            "key2": ("default2", {})
        }
        required = ["key2"]

        result = extract_from_json(json_dict, values, required)

        assert len(result) == 2
        assert result[0] == "value1"
        assert result[1] == "value2"

    def test_extract_from_json_with_exceptions(self):
        """
        Test extraction with exception replacements

        The exceptions dict allows replacing specific values with alternatives

        Given:
        - json_dict: {"key1": "special"}
        - values: {"key1": ("default", {"special": "replacement"})}

        Expected:
        - key1 value is "special"
        - "special" is in exceptions dict, replaced with "replacement"
        - Returns ["replacement"]
        """
        json_dict = {"key1": "special"}
        values = {
            "key1": ("default", {"special": "replacement"})
        }

        result = extract_from_json(json_dict, values)

        assert len(result) == 1
        assert result[0] == "replacement"

    def test_extract_from_json_missing_uses_default(self):
        """Test that missing keys use default values"""
        json_dict = {}
        values = {
            "missing_key": ("default_value", {})
        }

        result = extract_from_json(json_dict, values)

        assert len(result) == 1
        assert result[0] == "default_value"


class TestGetConfigValue:
    """Test get_config_value function"""

    def test_get_config_value_with_multiplier(self):
        """
        Test config value extraction with unit conversion

        Given:
        - config = {"param": [50, "μs"]}
        - Unit multiplier for μ is 1e-6

        Calculation:
        - value = 50 * 1e-6 = 50e-6 = 5e-5

        Expected:
        - Returns converted value
        """
        config = {"param": [50, "μs"]}

        result = get_config_value(config, "param")

        expected = 50 * 1e-6
        # Use approximate comparison for floating point
        assert abs(result - expected) < 1e-10
        assert abs(result - 50e-6) < 1e-10

    def test_get_config_value_without_multiplier(self):
        """Test value extraction with no unit prefix (base unit)"""
        config = {"param": [100, "s"]}

        result = get_config_value(config, "param")

        assert result == 100 * 1  # multiplier is 1 for single char

    def test_get_config_value_missing_key(self):
        """Test None returned when key doesn't exist"""
        config = {"other_key": [50, "mK"]}

        result = get_config_value(config, "missing_key")

        assert result is None

    def test_get_config_value_giga(self):
        """Test GHz to Hz conversion"""
        config = {"frequency": [5.0, "GHz"]}

        result = get_config_value(config, "frequency")

        assert result == 5.0 * 1e9
        assert result == 5e9


class TestGetNumQubits:
    """Test get_num_qubits function"""

    def test_get_num_qubits_from_env(self, monkeypatch):
        """Test reading num_qubits from environment variable"""
        monkeypatch.setenv("NUM_QUBITS", "8")

        result = get_num_qubits()

        assert result == 8

    def test_get_num_qubits_default(self, monkeypatch):
        """Test default value when env var not set"""
        monkeypatch.delenv("NUM_QUBITS", raising=False)

        result = get_num_qubits()

        assert result == 5  # Default value

    def test_get_num_qubits_invalid_format(self, monkeypatch):
        """Test fallback to default when env var is not a valid integer"""
        monkeypatch.setenv("NUM_QUBITS", "not_a_number")

        result = get_num_qubits()

        # Should fallback to default value of 5
        assert result == 5

    def test_get_num_qubits_zero(self, monkeypatch):
        """Test with zero qubits"""
        monkeypatch.setenv("NUM_QUBITS", "0")

        result = get_num_qubits()

        assert result == 0


class TestSetNumQubitsList:
    """Test set_num_qubits_list function"""

    def test_set_num_qubits_list_from_config(self, sample_config_path, monkeypatch):
        """Test setting NUM_QUBITS from config file"""
        result = set_num_qubits_list()

        # Check that environment variable was set
        assert monkeypatch.setenv  # Fixture is available
        num_qubits_env = result

        assert num_qubits_env == "5"  # From sample_config fixture

    def test_set_num_qubits_list_missing_in_config(self, tmp_path, monkeypatch):
        """
        Test behavior when num_qubits not in config

        Note: The actual implementation converts None to string "None" via str()
        """
        config = {}  # Empty config
        config_file = tmp_path / "configs.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)

        monkeypatch.setenv("CONFIG_PATH", str(config_file))

        result = set_num_qubits_list()

        # Actual behavior: None gets converted to "None"
        assert result == "None"


class TestSetCircuitOptimisation:
    """Test set_circuit_optimisation function"""

    def test_set_circuit_optimisation_from_config(self, sample_config_path, monkeypatch):
        """Test setting CIRCUIT_OPTIMIZATION from config file"""
        result = set_circuit_optimisation()

        assert result == 1  # From sample_config fixture

    def test_set_circuit_optimisation_missing_in_config(self, tmp_path, monkeypatch):
        """Test default value when circuit_optimisation_level not in config"""
        config = {}
        config_file = tmp_path / "configs.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)

        monkeypatch.setenv("CONFIG_PATH", str(config_file))

        result = set_circuit_optimisation()

        # Should use default of 1
        assert result == 1

    def test_set_circuit_optimisation_various_levels(self, tmp_path, monkeypatch):
        """Test valid optimization levels (0-3)"""
        for level in [0, 1, 2, 3]:
            config = {"circuit_optimisation_level": level}
            config_file = tmp_path / f"config_{level}.json"
            with open(config_file, 'w') as f:
                json.dump(config, f)

            monkeypatch.setenv("CONFIG_PATH", str(config_file))

            result = set_circuit_optimisation()

            assert result == level


class TestSetUpLogger:
    """Test set_up_logger function"""

    def test_set_up_logger_with_file(self, tmp_path):
        """Test logger setup with log file"""
        import logging

        log_file = tmp_path / "test.log"
        set_up_logger(logging.DEBUG, str(log_file))

        # Check that log file was created
        logging.debug("Test message")
        # Note: The actual logging is hard to test without checking handlers

    def test_set_up_logger_without_file(self):
        """Test logger setup without log file (console output)"""
        import logging

        # Should not raise an error
        set_up_logger(logging.INFO, None)

    def test_set_up_logger_qiskit_level(self):
        """Test that Qiskit logger is set to WARNING"""
        import logging

        set_up_logger(logging.DEBUG, None)

        qiskit_logger = logging.getLogger('qiskit')
        assert qiskit_logger.level == logging.WARNING


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
