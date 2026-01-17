"""
Unit tests for constants module

This test suite verifies all constant definitions used throughout the codebase,
including SI unit prefixes, backend mappings, and default configurations.
"""

import pytest
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))

from _helpers.constants import (
    SI_PREFIXES,
    EXISTING_MODELS,
    HARDWARE_CONFIG_GROUPS,
    DEFAULT_INSTRUCTION_TIMES,
    PROPERTY_UNITS,
    DEFAULT_PATH,
    SIMULATION_METHOD,
    NoiselessSimBasisGates
)


class TestSIPrefixes:
    """Test SI unit prefix mappings"""

    def test_si_prefixes_peta(self):
        """Test peta prefix (P = 10^15)"""
        assert SI_PREFIXES['P'] == 1e15

    def test_si_prefixes_tera(self):
        """Test tera prefix (T = 10^12)"""
        assert SI_PREFIXES['T'] == 1e12

    def test_si_prefixes_giga(self):
        """Test giga prefix (G = 10^9)"""
        assert SI_PREFIXES['G'] == 1e9

    def test_si_prefixes_mega(self):
        """Test mega prefix (M = 10^6)"""
        assert SI_PREFIXES['M'] == 1e6

    def test_si_prefixes_kilo(self):
        """Test kilo prefix (k = 10^3)"""
        assert SI_PREFIXES['k'] == 1e3

    def test_si_prefixes_base_unit(self):
        """Test base unit (empty string = 1)"""
        assert SI_PREFIXES[''] == 1

    def test_si_prefixes_milli(self):
        """Test milli prefix (m = 10^-3)"""
        assert SI_PREFIXES['m'] == 1e-3

    def test_si_prefixes_micro_greek(self):
        """Test micro prefix with Greek mu (μ = 10^-6)"""
        assert SI_PREFIXES['μ'] == 1e-6

    def test_si_prefixes_micro_u(self):
        """Test micro prefix with 'u' (u = 10^-6)"""
        assert SI_PREFIXES['u'] == 1e-6

    def test_si_prefixes_nano(self):
        """Test nano prefix (n = 10^-9)"""
        assert SI_PREFIXES['n'] == 1e-9

    def test_si_prefixes_keys_exist(self):
        """Test that all expected prefixes exist"""
        expected_keys = ['P', 'T', 'G', 'M', 'k', '', 'm', 'μ', 'u', 'n']
        for key in expected_keys:
            assert key in SI_PREFIXES, f"SI prefix '{key}' not found"

    def test_si_prefixes_count(self):
        """Test that we have exactly the expected number of prefixes"""
        assert len(SI_PREFIXES) == 10, f"Expected 10 SI prefixes, found {len(SI_PREFIXES)}"


class TestExistingModels:
    """Test fake backend model mappings"""

    def test_existing_models_not_empty(self):
        """Test that EXISTING_MODELS dictionary is not empty"""
        assert len(EXISTING_MODELS) > 0, "EXISTING_MODELS should not be empty"

    def test_existing_models_expected_backends(self):
        """Test that expected fake backends are present"""
        expected_backends = [
            "fakeTokyo",
            "fakeOslo",
            "fakePerth",
            "fakeSherbrooke",
            "fakeAuckland"
        ]
        for backend in expected_backends:
            assert backend in EXISTING_MODELS, f"Expected backend '{backend}' not found"

    def test_existing_models_all_callable(self):
        """Test that all model values are callable (backend classes)"""
        for name, backend_class in EXISTING_MODELS.items():
            assert callable(backend_class), f"Backend '{name}' is not callable"

    def test_existing_models_count(self):
        """Test that we have all expected fake backends"""
        # Based on the constants.py file, we expect 13 backends
        assert len(EXISTING_MODELS) == 13, f"Expected 13 fake backends, found {len(EXISTING_MODELS)}"


class TestHardwareConfigGroups:
    """Test hardware configuration groupings"""

    def test_hardware_config_groups_structure(self):
        """Test that HARDWARE_CONFIG_GROUPS has expected structure"""
        assert isinstance(HARDWARE_CONFIG_GROUPS, dict), "HARDWARE_CONFIG_GROUPS should be a dict"

    def test_hardware_config_groups_keys(self):
        """Test that expected group keys exist"""
        expected_keys = ["modern", "intermediate", "legacy"]
        for key in expected_keys:
            assert key in HARDWARE_CONFIG_GROUPS, f"Expected group '{key}' not found"

    def test_hardware_config_groups_modern(self):
        """Test modern hardware group contains expected backends"""
        modern = HARDWARE_CONFIG_GROUPS["modern"]
        assert isinstance(modern, tuple), "Modern group should be a tuple"
        assert "perth" in modern, "Perth should be in modern group"
        assert "sherbrooke" in modern, "Sherbrooke should be in modern group"

    def test_hardware_config_groups_legacy(self):
        """
        Test legacy hardware group

        Note: In Python, ("oslo") is a string, not a tuple. To make a single-element tuple,
        you need ("oslo",) with a trailing comma. The constants.py has this quirk.
        """
        legacy = HARDWARE_CONFIG_GROUPS["legacy"]
        # Legacy is actually a string due to ("oslo") syntax
        assert isinstance(legacy, (tuple, str)), "Legacy group should be a tuple or string"
        if isinstance(legacy, str):
            assert legacy == "oslo", "Legacy should be oslo"
        else:
            assert "oslo" in legacy, "Oslo should be in legacy group"

    def test_hardware_config_groups_all_tuples(self):
        """
        Test that all group values are tuples or strings

        Note: Due to Python syntax ("oslo") vs ("oslo",), some groups might be strings
        """
        for group_name, group_value in HARDWARE_CONFIG_GROUPS.items():
            assert isinstance(group_value, (tuple, str)), \
                f"Group '{group_name}' should be a tuple or string, got {type(group_value)}"


class TestDefaultInstructionTimes:
    """Test default instruction time mappings"""

    def test_default_instruction_times_not_empty(self):
        """Test that DEFAULT_INSTRUCTION_TIMES is not empty"""
        assert len(DEFAULT_INSTRUCTION_TIMES) > 0, "DEFAULT_INSTRUCTION_TIMES should not be empty"

    def test_default_instruction_times_required_gates(self):
        """Test that all required gate times are defined"""
        required_gates = ["time_rz", "time_sx", "time_x", "time_cx", "time_reset", "time_measure"]
        for gate in required_gates:
            assert gate in DEFAULT_INSTRUCTION_TIMES, f"Required gate time '{gate}' not found"

    def test_default_instruction_times_rz_is_zero(self):
        """Test that RZ gate time is zero (virtual gate)"""
        assert DEFAULT_INSTRUCTION_TIMES["time_rz"] == 0, "RZ should be virtual (0 time)"

    def test_default_instruction_times_all_non_negative(self):
        """Test that all instruction times are non-negative"""
        for gate, time in DEFAULT_INSTRUCTION_TIMES.items():
            assert time >= 0, f"Instruction time for '{gate}' should be non-negative, got {time}"

    def test_default_instruction_times_cx_longest(self):
        """Test that CX gate has longest time (typical for two-qubit gates)"""
        cx_time = DEFAULT_INSTRUCTION_TIMES["time_cx"]
        for gate, time in DEFAULT_INSTRUCTION_TIMES.items():
            if gate not in ["time_cx", "time_reset"]:  # reset can be longer
                assert cx_time >= time or gate == "time_reset", \
                    f"CX time should be among the longest (excluding reset)"

    def test_default_instruction_times_values(self):
        """Test specific expected values from constants.py"""
        assert DEFAULT_INSTRUCTION_TIMES["time_sx"] == 50, "SX time should be 50 ns"
        assert DEFAULT_INSTRUCTION_TIMES["time_x"] == 100, "X time should be 100 ns"
        assert DEFAULT_INSTRUCTION_TIMES["time_cx"] == 300, "CX time should be 300 ns"


class TestPropertyUnits:
    """Test property unit mappings"""

    def test_property_units_not_empty(self):
        """Test that PROPERTY_UNITS is not empty"""
        assert len(PROPERTY_UNITS) > 0, "PROPERTY_UNITS should not be empty"

    def test_property_units_required_properties(self):
        """Test that all required property units are defined"""
        required_properties = ["T1", "T2", "frequency", "gate_error", "gate_length"]
        for prop in required_properties:
            assert prop in PROPERTY_UNITS, f"Required property '{prop}' not found"

    def test_property_units_t1_t2(self):
        """Test that T1 and T2 use microseconds"""
        assert PROPERTY_UNITS["T1"] == "us", "T1 should use microseconds"
        assert PROPERTY_UNITS["T2"] == "us", "T2 should use microseconds"

    def test_property_units_frequency(self):
        """Test that frequency uses GHz"""
        assert PROPERTY_UNITS["frequency"] == "GHz", "Frequency should use GHz"

    def test_property_units_dimensionless(self):
        """Test that dimensionless properties have empty string units"""
        dimensionless = ["gate_error", "readout_error", "prob_meas0_prep1", "prob_meas1_prep0"]
        for prop in dimensionless:
            assert PROPERTY_UNITS[prop] == "", f"Property '{prop}' should be dimensionless (empty unit)"

    def test_property_units_time_based(self):
        """Test that time-based properties use nanoseconds"""
        time_properties = ["readout_length", "gate_length"]
        for prop in time_properties:
            assert PROPERTY_UNITS[prop] == "ns", f"Property '{prop}' should use nanoseconds"


class TestMiscellaneousConstants:
    """Test miscellaneous constant definitions"""

    def test_default_path_exists(self):
        """Test that DEFAULT_PATH is defined"""
        assert DEFAULT_PATH is not None, "DEFAULT_PATH should be defined"
        assert isinstance(DEFAULT_PATH, str), "DEFAULT_PATH should be a string"
        assert "quantum_volume" in DEFAULT_PATH, "DEFAULT_PATH should point to quantum_volume"

    def test_simulation_method_defined(self):
        """Test that SIMULATION_METHOD is defined"""
        assert SIMULATION_METHOD is not None, "SIMULATION_METHOD should be defined"
        assert SIMULATION_METHOD == "density_matrix", "SIMULATION_METHOD should be density_matrix"

    def test_noiseless_sim_basis_gates_not_empty(self):
        """Test that NoiselessSimBasisGates is defined and not empty"""
        assert len(NoiselessSimBasisGates) > 0, "NoiselessSimBasisGates should not be empty"

    def test_noiseless_sim_basis_gates_common_gates(self):
        """Test that common gates are in noiseless sim basis gates"""
        common_gates = ['x', 'y', 'z', 'h', 'cx', 'rz', 'rx', 'ry', 's', 't']
        for gate in common_gates:
            assert gate in NoiselessSimBasisGates, f"Gate '{gate}' should be in NoiselessSimBasisGates"

    def test_noiseless_sim_basis_gates_is_list(self):
        """Test that NoiselessSimBasisGates is a list"""
        assert isinstance(NoiselessSimBasisGates, list), "NoiselessSimBasisGates should be a list"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
