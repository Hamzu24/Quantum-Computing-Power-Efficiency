"""
Unit tests for builders/builder_wrapper module

This test suite validates the BuilderWrapper class which orchestrates
hardware parameter loading, builder instantiation, and backend configuration.
"""

import pytest
import json
import sys
import pathlib
from unittest.mock import Mock, patch, MagicMock, mock_open
from io import StringIO

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))

from _helpers.builders.builder_wrapper import BuilderWrapper
from _helpers.builders.base import Builder, builder_registry
from _helpers.constants import HARDWARE_CONFIG_GROUPS


class TestBuilderWrapperInitialization:
    """Test BuilderWrapper initialization"""

    def test_init_loads_hardware_params(self, tmp_path, monkeypatch):
        """
        Test that initialization loads hardware parameters

        Expected: load_hardware_params() is called with correct path
        """
        # Setup
        hw_config = tmp_path / "hardware_constants.json"
        backend_dir = tmp_path / "backends"
        backend_dir.mkdir()
        perth_dir = backend_dir / "perth"
        perth_dir.mkdir()

        # Write hardware config
        hw_data = {
            "modern": {
                "builder_class": "DefaultBuilder",
                "y0": 1e5
            }
        }
        with open(hw_config, 'w') as f:
            json.dump(hw_data, f)

        # Write backend props
        props_data = {"qubits": [], "gates": []}
        props_file = perth_dir / "props_perth.json"
        with open(props_file, 'w') as f:
            json.dump(props_data, f)

        monkeypatch.setenv("HARDWARE_CONFIG_PATH", str(hw_config))
        monkeypatch.setenv("BACKEND_CONFIGS_FOLDER", str(backend_dir) + "/")

        # Register a mock builder
        class MockBuilder:
            registry_name = "DefaultBuilder"

            def __init__(self, name, config, jm, init_control_parameters=None):
                self.name = name
                self.config = config

            def initialize_per_qubit_params(self):
                pass

        builder_registry.register_builder(MockBuilder)

        # Initialize wrapper
        wrapper = BuilderWrapper("perth", {"temperature": [40, "mK"]})

        assert wrapper.name == "perth"
        assert wrapper.config is not None
        assert "builder_class" in wrapper.config

    def test_init_creates_json_manager(self, tmp_path, monkeypatch):
        """
        Test that initialization creates JsonManager for backend props file

        Expected: JsonManager is created with correct filename
        """
        hw_config = tmp_path / "hardware_constants.json"
        backend_dir = tmp_path / "backends"
        backend_dir.mkdir()
        perth_dir = backend_dir / "perth"
        perth_dir.mkdir()

        hw_data = {
            "modern": {"builder_class": "DefaultBuilder"}
        }
        with open(hw_config, 'w') as f:
            json.dump(hw_data, f)

        props_data = {"qubits": [], "gates": []}
        props_file = perth_dir / "props_perth.json"
        with open(props_file, 'w') as f:
            json.dump(props_data, f)

        monkeypatch.setenv("HARDWARE_CONFIG_PATH", str(hw_config))
        monkeypatch.setenv("BACKEND_CONFIGS_FOLDER", str(backend_dir) + "/")

        class MockBuilder:
            registry_name = "DefaultBuilder"
            def __init__(self, name, config, jm, init_control_parameters=None):
                self.name = name
                self.config = config
            def initialize_per_qubit_params(self):
                pass

        builder_registry.register_builder(MockBuilder)

        wrapper = BuilderWrapper("perth", {})

        assert wrapper.json_manager is not None
        assert wrapper.json_manager.filename == str(props_file)

    def test_init_calls_builder_initialize_per_qubit_params(self, tmp_path, monkeypatch):
        """
        Test that initialization calls initialize_per_qubit_params on builder

        Expected: Builder's initialize_per_qubit_params() method is called
        """
        hw_config = tmp_path / "hardware_constants.json"
        backend_dir = tmp_path / "backends"
        backend_dir.mkdir()
        perth_dir = backend_dir / "perth"
        perth_dir.mkdir()

        hw_data = {
            "modern": {"builder_class": "DefaultBuilder"}
        }
        with open(hw_config, 'w') as f:
            json.dump(hw_data, f)

        props_data = {"qubits": [], "gates": []}
        with open(perth_dir / "props_perth.json", 'w') as f:
            json.dump(props_data, f)

        monkeypatch.setenv("HARDWARE_CONFIG_PATH", str(hw_config))
        monkeypatch.setenv("BACKEND_CONFIGS_FOLDER", str(backend_dir) + "/")

        init_called = []

        class MockBuilder:
            registry_name = "DefaultBuilder"
            def __init__(self, name, config, jm, init_control_parameters=None):
                self.config = config
            def initialize_per_qubit_params(self):
                init_called.append(True)

        builder_registry.register_builder(MockBuilder)

        wrapper = BuilderWrapper("perth", {})

        assert len(init_called) == 1


class TestBuilderWrapperFindGroup:
    """Test find_group() method"""

    def test_find_group_exact_match(self, tmp_path, monkeypatch):
        """
        Test finding group when backend name matches exactly

        Given: Backend name "perth" which is in "modern" group
        Expected: Returns "modern"
        """
        hw_config = tmp_path / "hardware_constants.json"
        hw_data = {
            "modern": {"builder_class": "DefaultBuilder"},
            "legacy": {"builder_class": "DefaultBuilder"}
        }
        with open(hw_config, 'w') as f:
            json.dump(hw_data, f)

        # Create minimal setup to test find_group
        wrapper_partial = type('obj', (object,), {
            'name': 'perth',
            'find_group': BuilderWrapper.find_group
        })()

        result = wrapper_partial.find_group()

        assert result == "modern"

    def test_find_group_uses_default(self, tmp_path):
        """
        Test finding group using 'default' fallback

        Given: Backend name not in any group, but "default" exists in a group
        Expected: Returns the group containing "default"
        """
        wrapper_partial = type('obj', (object,), {
            'name': 'unknown_backend',
            'find_group': BuilderWrapper.find_group
        })()

        result = wrapper_partial.find_group()

        # Should find the group with "default"
        assert result in HARDWARE_CONFIG_GROUPS.keys()

    def test_find_group_no_match_raises(self):
        """
        Test that ValueError is raised when no group matches

        Expected: Raises ValueError with descriptive message
        """
        # Temporarily modify HARDWARE_CONFIG_GROUPS to have no default
        with patch('_helpers.builders.builder_wrapper.HARDWARE_CONFIG_GROUPS', {
            "modern": ("perth",),
            "legacy": ("oslo",)
        }):
            wrapper_partial = type('obj', (object,), {
                'name': 'unknown_backend',
                'find_group': BuilderWrapper.find_group
            })()

            with pytest.raises(ValueError, match="Neither the name nor default was found"):
                wrapper_partial.find_group()


class TestBuilderWrapperLoadHardwareParams:
    """Test load_hardware_params() method"""

    def test_load_hardware_params_success(self, tmp_path):
        """
        Test successful loading of hardware parameters

        Given: Valid hardware config file with group config
        Expected: self.config is set to the group's configuration
        """
        hw_config = tmp_path / "hardware_constants.json"
        hw_data = {
            "modern": {
                "builder_class": "DefaultBuilder",
                "y0": 1e5,
                "T_env": 0.050
            }
        }
        with open(hw_config, 'w') as f:
            json.dump(hw_data, f)

        wrapper_partial = type('obj', (object,), {
            'name': 'perth',
            'find_group': BuilderWrapper.find_group,
            'load_hardware_params': BuilderWrapper.load_hardware_params
        })()

        wrapper_partial.load_hardware_params(str(hw_config))

        assert wrapper_partial.config == hw_data["modern"]
        assert wrapper_partial.config["y0"] == 1e5

    def test_load_hardware_params_missing_group_raises(self, tmp_path):
        """
        Test that ValueError is raised when group config is missing

        Given: Hardware config file without the required group
        Expected: Raises ValueError
        """
        hw_config = tmp_path / "hardware_constants.json"
        hw_data = {
            "other_group": {"builder_class": "SomeBuilder"}
        }
        with open(hw_config, 'w') as f:
            json.dump(hw_data, f)

        with patch('_helpers.builders.builder_wrapper.HARDWARE_CONFIG_GROUPS', {
            "modern": ("perth",)
        }):
            wrapper_partial = type('obj', (object,), {
                'name': 'perth',
                'find_group': BuilderWrapper.find_group,
                'load_hardware_params': BuilderWrapper.load_hardware_params
            })()

            with pytest.raises(ValueError, match="does not have a corresponding configuration"):
                wrapper_partial.load_hardware_params(str(hw_config))

    def test_load_hardware_params_file_not_found(self):
        """
        Test that FileNotFoundError is raised for missing file

        Expected: Raises FileNotFoundError
        """
        wrapper_partial = type('obj', (object,), {
            'name': 'perth',
            'find_group': BuilderWrapper.find_group,
            'load_hardware_params': BuilderWrapper.load_hardware_params
        })()

        with pytest.raises(FileNotFoundError):
            wrapper_partial.load_hardware_params("/nonexistent/file.json")


class TestBuilderWrapperBuildBackend:
    """Test build_backend() method and helper methods"""

    def test_build_backend_orchestrates_building(self, tmp_path, monkeypatch):
        """
        Test that build_backend orchestrates qubit and gate building

        Expected: Calls _build_qubits, _build_gates, and json_manager.write()
        """
        # Setup
        hw_config = tmp_path / "hardware_constants.json"
        backend_dir = tmp_path / "backends"
        backend_dir.mkdir()
        perth_dir = backend_dir / "perth"
        perth_dir.mkdir()

        hw_data = {
            "modern": {"builder_class": "DefaultBuilder"}
        }
        with open(hw_config, 'w') as f:
            json.dump(hw_data, f)

        props_data = {
            "qubits": [[{"name": "T1", "value": 50, "unit": "us"}]],
            "gates": [{"parameters": [{"name": "gate_error", "value": 0.01, "unit": ""}]}]
        }
        props_file = perth_dir / "props_perth.json"
        with open(props_file, 'w') as f:
            json.dump(props_data, f)

        monkeypatch.setenv("HARDWARE_CONFIG_PATH", str(hw_config))
        monkeypatch.setenv("BACKEND_CONFIGS_FOLDER", str(backend_dir) + "/")

        # Mock builder
        mock_builder = Mock()
        mock_builder.calculate_qb_config = Mock(return_value={"T1": 50e-6})
        mock_builder.calculate_gate_config = Mock(return_value={"gate_error": 0.01})

        class MockBuilderClass:
            registry_name = "DefaultBuilder"
            def __init__(self, name, config, jm, init_control_parameters=None):
                self.config = config
                return_value = mock_builder
            def initialize_per_qubit_params(self):
                pass

        # Use a hacky way to return the mock
        def mock_builder_init(name, config, jm, init_control_parameters=None):
            return mock_builder

        with patch.object(builder_registry, 'get_builder', return_value=mock_builder_init):
            wrapper = BuilderWrapper("perth", {})

            control_params = {"temperature": [40, "mK"]}

            # Call build_backend
            wrapper.build_backend(control_params)

            # Should have called builder methods
            assert mock_builder.calculate_qb_config.called
            assert mock_builder.calculate_gate_config.called

    def test_build_qubits_updates_all_qubits(self, tmp_path, monkeypatch):
        """
        Test that _build_qubits updates all qubit configurations

        Given: Multiple qubits in backend
        Expected: calculate_qb_config called for each qubit
        """
        hw_config = tmp_path / "hardware_constants.json"
        backend_dir = tmp_path / "backends"
        backend_dir.mkdir()
        perth_dir = backend_dir / "perth"
        perth_dir.mkdir()

        hw_data = {
            "modern": {"builder_class": "DefaultBuilder"}
        }
        with open(hw_config, 'w') as f:
            json.dump(hw_data, f)

        # Multiple qubits
        props_data = {
            "qubits": [
                [{"name": "T1", "value": 50, "unit": "us"}],
                [{"name": "T1", "value": 60, "unit": "us"}],
                [{"name": "T1", "value": 70, "unit": "us"}]
            ],
            "gates": []
        }
        props_file = perth_dir / "props_perth.json"
        with open(props_file, 'w') as f:
            json.dump(props_data, f)

        monkeypatch.setenv("HARDWARE_CONFIG_PATH", str(hw_config))
        monkeypatch.setenv("BACKEND_CONFIGS_FOLDER", str(backend_dir) + "/")

        mock_builder = Mock()
        mock_builder.calculate_qb_config = Mock(return_value={"T1": 50e-6})

        def mock_builder_init(name, config, jm, init_control_parameters=None):
            return mock_builder

        with patch.object(builder_registry, 'get_builder', return_value=mock_builder_init):
            wrapper = BuilderWrapper("perth", {})
            wrapper._build_qubits({"temperature": [40, "mK"]})

            # Should have been called 3 times (one per qubit)
            assert mock_builder.calculate_qb_config.call_count == 3

    def test_build_gates_skips_none_values(self, tmp_path, monkeypatch):
        """
        Test that _build_gates skips None values

        Given: Gate config returns None for some properties
        Expected: Only non-None values are updated
        """
        hw_config = tmp_path / "hardware_constants.json"
        backend_dir = tmp_path / "backends"
        backend_dir.mkdir()
        perth_dir = backend_dir / "perth"
        perth_dir.mkdir()

        hw_data = {
            "modern": {"builder_class": "DefaultBuilder"}
        }
        with open(hw_config, 'w') as f:
            json.dump(hw_data, f)

        props_data = {
            "qubits": [],
            "gates": [{
                "parameters": [
                    {"name": "gate_error", "value": 0.01, "unit": ""}
                ]
            }]
        }
        props_file = perth_dir / "props_perth.json"
        with open(props_file, 'w') as f:
            json.dump(props_data, f)

        monkeypatch.setenv("HARDWARE_CONFIG_PATH", str(hw_config))
        monkeypatch.setenv("BACKEND_CONFIGS_FOLDER", str(backend_dir) + "/")

        mock_builder = Mock()
        # Return dict with None value
        mock_builder.calculate_gate_config = Mock(return_value={
            "gate_error": 0.01,
            "gate_length": None  # This should be skipped
        })

        def mock_builder_init(name, config, jm, init_control_parameters=None):
            return mock_builder

        with patch.object(builder_registry, 'get_builder', return_value=mock_builder_init):
            wrapper = BuilderWrapper("perth", {})

            # Mock update_with_units to track calls
            wrapper.json_manager.update_with_units = Mock()

            wrapper._build_gates({"temperature": [40, "mK"]})

            # Should only update gate_error (not gate_length which is None)
            assert wrapper.json_manager.update_with_units.call_count == 1
            call_args = wrapper.json_manager.update_with_units.call_args[0]
            assert call_args[0] == "gate_error"


class TestBuilderWrapperIntegration:
    """Test BuilderWrapper integration scenarios"""

    def test_full_workflow_perth_backend(self, tmp_path, monkeypatch):
        """
        Test complete workflow for Perth backend

        Expected: Loads modern config, builds backend successfully
        """
        hw_config = tmp_path / "hardware_constants.json"
        backend_dir = tmp_path / "backends"
        backend_dir.mkdir()
        perth_dir = backend_dir / "perth"
        perth_dir.mkdir()

        hw_data = {
            "modern": {
                "builder_class": "DefaultBuilder",
                "y0": 1e5,
                "T_env": 0.050
            }
        }
        with open(hw_config, 'w') as f:
            json.dump(hw_data, f)

        props_data = {
            "qubits": [[{"name": "T1", "value": 50, "unit": "us"}]],
            "gates": []
        }
        with open(perth_dir / "props_perth.json", 'w') as f:
            json.dump(props_data, f)

        monkeypatch.setenv("HARDWARE_CONFIG_PATH", str(hw_config))
        monkeypatch.setenv("BACKEND_CONFIGS_FOLDER", str(backend_dir) + "/")

        mock_builder = Mock()
        mock_builder.calculate_qb_config = Mock(return_value={"T1": 50e-6})
        mock_builder.calculate_gate_config = Mock(return_value={})
        mock_builder.config = hw_data["modern"]

        def mock_builder_init(name, config, jm, init_control_parameters=None):
            return mock_builder

        with patch.object(builder_registry, 'get_builder', return_value=mock_builder_init):
            wrapper = BuilderWrapper("perth", {"temperature": [40, "mK"]})

            assert wrapper.config == hw_data["modern"]
            assert wrapper.name == "perth"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
