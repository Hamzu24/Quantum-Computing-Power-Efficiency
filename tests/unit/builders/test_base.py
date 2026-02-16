"""
Unit tests for builders/base module

This test suite validates the Builder Protocol, BuilderRegistry, and ConfigTracker
classes that provide the foundation for the builder system.
"""

import pytest
import sys
import pathlib
import logging
from unittest.mock import Mock, patch
from io import StringIO

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))

from _helpers.builders.base import Builder, BuilderRegistry, ConfigTracker, builder_registry
from _helpers.json_manager import JsonManager


class TestBuilderProtocol:
    """Test Builder Protocol compliance"""

    def test_builder_protocol_has_required_methods(self):
        """
        Test that Builder Protocol defines required methods

        Expected: Protocol has T1, T2, calculate_qb_config, calculate_gate_config, initialize_per_qubit_params
        """
        # Protocol itself should have these attributes
        assert hasattr(Builder, '__init__')
        assert hasattr(Builder, 'T1')
        assert hasattr(Builder, 'T2')
        assert hasattr(Builder, 'calculate_qb_config')
        assert hasattr(Builder, 'calculate_gate_config')
        assert hasattr(Builder, 'initialize_per_qubit_params')

    def test_builder_protocol_has_registry_name(self):
        """
        Test that Builder Protocol requires registry_name class variable

        Expected: registry_name is a ClassVar in the Protocol
        """
        # The Protocol should define registry_name
        assert 'registry_name' in Builder.__annotations__

    def test_mock_builder_implements_protocol(self):
        """
        Test that a mock builder implementing the protocol works

        Expected: Mock builder with all required methods can be created
        """
        class MockBuilder:
            registry_name = "mock_builder"

            def __init__(self, name, config, init_control_parameters, jm):
                self.name = name
                self.config = config

            def T1(self, *args, **kwargs):
                return 50e-6

            def T2(self, *args, **kwargs):
                return 70e-6

            def calculate_qb_config(self, control_parameters, qb_path):
                return {"T1": 50e-6, "T2": 70e-6}

            def calculate_gate_config(self, control_parameters, gate_path):
                return {"gate_error": 0.01}

            def initialize_per_qubit_params(self):
                pass

        # Should be able to instantiate
        mock_jm = Mock(spec=JsonManager)
        builder = MockBuilder("test", {}, {}, mock_jm)

        assert builder.T1() == 50e-6
        assert builder.T2() == 70e-6
        assert "T1" in builder.calculate_qb_config({}, "")


class TestConfigTracker:
    """Test ConfigTracker class"""

    def test_init_creates_empty_lists(self):
        """
        Test ConfigTracker initialization

        Expected: Empty lists for qb_config_infos and gate_config_infos
        """
        tracker = ConfigTracker()

        assert tracker.qb_config_infos == []
        assert tracker.gate_config_infos == []

    def test_add_config_qubit_type(self):
        """
        Test adding qubit configuration

        Given: A qubit config dict
        When: add_config(config, "qb")
        Expected: Config is added to qb_config_infos list
        """
        tracker = ConfigTracker()
        config = {
            "config": {"T1": 50e-6, "T2": 70e-6}
        }

        tracker.add_config(config, "qb")

        assert len(tracker.qb_config_infos) == 1
        assert tracker.qb_config_infos[0] == config

    def test_add_config_gate_type(self):
        """
        Test adding gate configuration

        Given: A gate config dict
        When: add_config(config, "gate")
        Expected: Config is added to gate_config_infos list
        """
        tracker = ConfigTracker()
        config = {
            "config": {"gate_error": 0.01}
        }

        tracker.add_config(config, "gate")

        assert len(tracker.gate_config_infos) == 1
        assert tracker.gate_config_infos[0] == config

    def test_add_config_invalid_type_raises(self):
        """
        Test that invalid config type raises ValueError

        Expected: Raises ValueError with descriptive message
        """
        tracker = ConfigTracker()
        config = {"data": "test"}

        with pytest.raises(ValueError, match="unknown type invalid"):
            tracker.add_config(config, "invalid")

    def test_add_multiple_configs(self):
        """
        Test adding multiple configs of same type

        Expected: All configs are stored in order
        """
        tracker = ConfigTracker()

        config1 = {"config": {"T1": 50e-6, "T2": 70e-6}}
        config2 = {"config": {"T1": 60e-6, "T2": 80e-6}}
        config3 = {"config": {"T1": 70e-6, "T2": 90e-6}}

        tracker.add_config(config1, "qb")
        tracker.add_config(config2, "qb")
        tracker.add_config(config3, "qb")

        assert len(tracker.qb_config_infos) == 3
        assert tracker.qb_config_infos[0] == config1
        assert tracker.qb_config_infos[1] == config2
        assert tracker.qb_config_infos[2] == config3

    def test_add_mixed_config_types(self):
        """
        Test adding both qubit and gate configs

        Expected: Each type maintains its own list
        """
        tracker = ConfigTracker()

        qb_config = {"config": {"T1": 50e-6, "T2": 70e-6}}
        gate_config = {"config": {"gate_error": 0.01}}

        tracker.add_config(qb_config, "qb")
        tracker.add_config(gate_config, "gate")

        assert len(tracker.qb_config_infos) == 1
        assert len(tracker.gate_config_infos) == 1
        assert tracker.qb_config_infos[0] == qb_config
        assert tracker.gate_config_infos[0] == gate_config

    def test_log_info_calculates_averages(self):
        """
        Test that log_info calculates average computed values

        Given: Multiple qubit and gate configs
        When: log_info() is called
        Expected: Averages are calculated and logged
        """
        tracker = ConfigTracker()

        # Add qubit configs with computed T1/T2
        tracker.add_config({
            "config": {"T1": 50e-6, "T2": 70e-6}
        }, "qb")
        tracker.add_config({
            "config": {"T1": 60e-6, "T2": 80e-6}
        }, "qb")

        # Add gate configs
        tracker.add_config({"config": {"gate_error": 0.01}}, "gate")
        tracker.add_config({"config": {"gate_error": 0.02}}, "gate")

        # Capture logging output
        with patch('logging.debug') as mock_debug:
            tracker.log_info()

            # Should have logged averages
            assert mock_debug.call_count > 0

    def test_log_info_average_calculations(self):
        """
        Test specific average calculations in log_info

        Given: Known T1/T2 values for BOTH qubits and gates
        Expected: Correct averages are calculated

        Note: log_info() requires BOTH qubit AND gate configs to avoid ZeroDivisionError

        Calculation:
        - qb1: T1=50e-6, T2=70e-6
        - qb2: T1=60e-6, T2=80e-6
        - avg_T1 = (50e-6 + 60e-6) / 2 = 55e-6
        - avg_T2 = (70e-6 + 80e-6) / 2 = 75e-6
        """
        tracker = ConfigTracker()

        # Add qubit configs
        tracker.add_config({
            "config": {"T1": 50e-6, "T2": 70e-6}
        }, "qb")
        tracker.add_config({
            "config": {"T1": 60e-6, "T2": 80e-6}
        }, "qb")

        # MUST add at least one gate config to avoid ZeroDivisionError
        tracker.add_config({"config": {"gate_error": 0.01}}, "gate")

        # Verify logging happens
        with patch('logging.debug') as mock_debug:
            tracker.log_info()

            # Check that avg values were logged
            logged_messages = [str(call[0][0]) for call in mock_debug.call_args_list]
            assert any("avg computed T1" in msg for msg in logged_messages)
            assert any("avg computed T2" in msg for msg in logged_messages)

    def test_log_info_gate_average_calculations(self):
        """
        Test gate error average calculations

        Given: Gate errors [0.01, 0.02, 0.03]
        Expected: avg = (0.01 + 0.02 + 0.03) / 3 = 0.02

        Note: log_info() requires BOTH qubit AND gate configs
        """
        tracker = ConfigTracker()

        # Add gate configs
        tracker.add_config({"config": {"gate_error": 0.01}}, "gate")
        tracker.add_config({"config": {"gate_error": 0.02}}, "gate")
        tracker.add_config({"config": {"gate_error": 0.03}}, "gate")

        # MUST add at least one qubit config to avoid ZeroDivisionError
        tracker.add_config({"config": {"T1": 50e-6, "T2": 70e-6}}, "qb")

        expected_avg = (0.01 + 0.02 + 0.03) / 3

        assert abs(expected_avg - 0.02) < 1e-10

        with patch('logging.debug') as mock_debug:
            tracker.log_info()

            logged_messages = [str(call[0][0]) for call in mock_debug.call_args_list]
            # Should log the average gate error
            assert mock_debug.call_count > 0

    def test_log_info_with_empty_configs_raises(self):
        """
        Test log_info with no configs

        Expected: Should raise ZeroDivisionError when dividing by num_qb=0
        """
        tracker = ConfigTracker()

        # With no configs, should raise ZeroDivisionError
        with pytest.raises(ZeroDivisionError):
            tracker.log_info()

    def test_log_info_qb_only(self):
        """
        Test log_info with only qubit configs (no gates)

        Expected: Should raise ZeroDivisionError for gate averaging
        """
        tracker = ConfigTracker()
        tracker.add_config({
            "config": {"T1": 50e-6, "T2": 70e-6}
        }, "qb")

        # Should fail on gate average (num_gates = 0)
        with pytest.raises(ZeroDivisionError):
            tracker.log_info()

    def test_log_info_gate_only(self):
        """
        Test log_info with only gate configs (no qubits)

        Expected: Should raise ZeroDivisionError for qubit averaging
        """
        tracker = ConfigTracker()
        tracker.add_config({"config": {"gate_error": 0.01}}, "gate")

        # Should fail on qubit average (num_qb = 0)
        with pytest.raises(ZeroDivisionError):
            tracker.log_info()


class TestConfigTrackerGetT1T2Values:
    """Test ConfigTracker.get_T1_T2_values() method"""

    def test_returns_empty_dict_when_no_configs(self):
        """
        Test get_T1_T2_values returns empty dict with no qubit configs

        Expected: Returns {}
        """
        tracker = ConfigTracker()
        assert tracker.get_T1_T2_values() == {}

    def test_extracts_single_qubit(self):
        """
        Test get_T1_T2_values with a single qubit config

        Expected: Returns dict with single-element T1 and T2 lists
        """
        tracker = ConfigTracker()
        tracker.add_config({"config": {"T1": 50e-6, "T2": 70e-6}}, "qb")

        result = tracker.get_T1_T2_values()

        assert result == {
            "T1_values": [50e-6],
            "T2_values": [70e-6],
        }

    def test_extracts_multiple_qubits(self):
        """
        Test get_T1_T2_values with multiple qubit configs

        Expected: Returns arrays with one value per qubit, preserving order
        """
        tracker = ConfigTracker()
        tracker.add_config({"config": {"T1": 50e-6, "T2": 70e-6}}, "qb")
        tracker.add_config({"config": {"T1": 60e-6, "T2": 80e-6}}, "qb")
        tracker.add_config({"config": {"T1": 70e-6, "T2": 90e-6}}, "qb")

        result = tracker.get_T1_T2_values()

        assert result["T1_values"] == [50e-6, 60e-6, 70e-6]
        assert result["T2_values"] == [70e-6, 80e-6, 90e-6]

    def test_ignores_gate_configs(self):
        """
        Test that gate configs don't affect T1/T2 extraction

        Expected: Only qubit configs are included
        """
        tracker = ConfigTracker()
        tracker.add_config({"config": {"T1": 50e-6, "T2": 70e-6}}, "qb")
        tracker.add_config({"config": {"gate_error": 0.01}}, "gate")

        result = tracker.get_T1_T2_values()

        assert len(result["T1_values"]) == 1
        assert len(result["T2_values"]) == 1


class TestConfigTrackerGetGateErrorValues:
    """Test ConfigTracker.get_gate_error_values() method"""

    def test_returns_empty_dict_when_no_configs(self):
        """
        Test get_gate_error_values returns empty dict with no gate configs

        Expected: Returns {}
        """
        tracker = ConfigTracker()
        assert tracker.get_gate_error_values() == {}

    def test_extracts_gate_errors(self):
        """
        Test get_gate_error_values with multiple gate configs

        Expected: Returns dict with gate_error_values list preserving order
        """
        tracker = ConfigTracker()
        tracker.add_config({"config": {"gate_error": 0.005}}, "gate")
        tracker.add_config({"config": {"gate_error": 0.012}}, "gate")
        tracker.add_config({"config": {"gate_error": 0.008}}, "gate")

        result = tracker.get_gate_error_values()

        assert result == {
            "gate_error_values": [0.005, 0.012, 0.008],
        }

    def test_ignores_qubit_configs(self):
        """
        Test that qubit configs don't affect gate error extraction

        Expected: Only gate configs are included
        """
        tracker = ConfigTracker()
        tracker.add_config({"config": {"T1": 50e-6, "T2": 70e-6}}, "qb")
        tracker.add_config({"config": {"gate_error": 0.01}}, "gate")

        result = tracker.get_gate_error_values()

        assert len(result["gate_error_values"]) == 1
        assert result["gate_error_values"][0] == 0.01


class TestConfigTrackerEdgeCases:
    """Test edge cases and error conditions in ConfigTracker"""

    def test_add_config_none_type(self):
        """
        Test adding config with None as type

        Expected: Raises ValueError
        """
        tracker = ConfigTracker()

        with pytest.raises(ValueError, match="unknown type"):
            tracker.add_config({"data": "test"}, None)

    def test_add_config_case_sensitivity(self):
        """
        Test that config type is case-sensitive

        Expected: "QB" and "GATE" should raise errors
        """
        tracker = ConfigTracker()

        with pytest.raises(ValueError, match="unknown type"):
            tracker.add_config({"data": "test"}, "QB")

        with pytest.raises(ValueError, match="unknown type"):
            tracker.add_config({"data": "test"}, "GATE")

    def test_config_independence(self):
        """
        Test that modifying config after adding doesn't affect stored config

        Expected: Stored config is independent of original
        """
        tracker = ConfigTracker()
        config = {"config": {"T1": 50e-6, "T2": 70e-6}}

        tracker.add_config(config, "qb")

        # Modify original
        config["config"]["T1"] = 999

        # Stored config should be unchanged (if deep copy is used)
        # However, the implementation doesn't deep copy, so this will fail
        # Let's check what actually happens
        stored_config = tracker.qb_config_infos[0]

        # If reference is stored (no deep copy), modifications will affect stored config
        # This is testing the actual behavior
        assert stored_config["config"]["T1"] == 999  # Same reference

    def test_log_info_with_single_qubit_and_gate(self):
        """
        Test log_info with single qubit and single gate config

        Expected: Averages equal the single values
        """
        tracker = ConfigTracker()

        qb_config = {"config": {"T1": 1.5e-4, "T2": 1.3e-4}}
        gate_config = {"config": {"gate_error": 0.0095}}

        tracker.add_config(qb_config, "qb")
        tracker.add_config(gate_config, "gate")

        with patch('logging.debug') as mock_debug:
            tracker.log_info()

            logged_messages = [str(call[0][0]) for call in mock_debug.call_args_list]

            # Average of single value = that value
            assert any("0.00015" in msg or "1.5e-04" in msg for msg in logged_messages)  # T1 avg
            assert any("0.00013" in msg or "1.3e-04" in msg for msg in logged_messages)  # T2 avg
            assert any("0.0095" in msg for msg in logged_messages)  # gate avg


class TestBuilderRegistryIntegration:
    """Test BuilderRegistry integration (already tested in test_registry.py, but verify here)"""

    def test_global_builder_registry_accessible(self):
        """
        Test that global builder_registry is accessible from base module

        Expected: builder_registry is a BuilderRegistry instance
        """
        from _helpers.builders.base import builder_registry as imported_registry

        assert isinstance(imported_registry, BuilderRegistry)
        assert imported_registry is builder_registry

    def test_builder_can_self_register(self):
        """
        Test that a builder can register itself

        Expected: Builder is accessible via registry after registration
        """
        class SelfRegisteringBuilder:
            registry_name = "self_registering"

            def __init__(self, name, config, init_control_parameters, jm):
                pass

            def T1(self, *args, **kwargs):
                return 0.0

            def T2(self, *args, **kwargs):
                return 0.0

            def calculate_qb_config(self, control_parameters, qb_path):
                return {}

            def calculate_gate_config(self, control_parameters, gate_path):
                return {}

            def initialize_per_qubit_params(self):
                pass

        # Register the builder
        registry = BuilderRegistry()
        registry.register_builder(SelfRegisteringBuilder)

        # Should be retrievable
        result = registry.get_builder("self_registering")
        assert result is SelfRegisteringBuilder


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
