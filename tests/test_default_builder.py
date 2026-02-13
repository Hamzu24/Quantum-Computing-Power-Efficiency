"""
Comprehensive unit tests for DefaultBuilder class

This test suite verifies the physics calculations for superconducting qubit
parameters including T1, T2, gate fidelities, and related quantities.

Manual calculations are documented for each test to ensure correctness.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from math import sqrt, exp, cosh
from scipy.constants import pi, k, hbar, e, h
from scipy.special import k0

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve()))

from _helpers.builders.default_builder import DefaultBuilder
from _helpers.json_manager import JsonManager


def make_qubit_params(frequency_hz=5e9, E_c=3.5e-24, E_J=2.1e-22,
                      R_n=5000, subgap_transparency=0.01,
                      y0=1e5, gamma_psi_base=1e3, T_env=0.050):
    """Helper to build a qubit_params dict for physics method tests."""
    w_ge = 2 * pi * frequency_hz
    w_p = sqrt(8 * E_J * E_c) / hbar
    g_k = e**2 / h
    N_e = (1 / subgap_transparency) * 1 / (2 * R_n * g_k)
    return {
        "E_c": E_c,
        "E_J": E_J,
        "w_p": w_p,
        "R_n": R_n,
        "N_e": N_e,
        "y0": y0,
        "gamma_psi_base": gamma_psi_base,
        "T_env": T_env,
        "w_ge": w_ge,
    }


class TestDefaultBuilderInitialization:
    """Test initialization and validation of DefaultBuilder"""

    def test_init_with_valid_config(self):
        """Test initialization with all required parameters"""
        config = {
            "delta": 1.764e-23,
            "ymxc_y0": 0.8,
            "subgap_transparency": 0.01,
            "E_c": 3.5e-24,
            "E_J": 2.1e-22,
            "R_n": 5000,
        }

        jm = Mock(spec=JsonManager)
        init_control_params = {"temperature": [0.013, "K"]}

        builder = DefaultBuilder("test_backend", config, jm, init_control_params)

        assert builder.name == "test_backend"
        assert builder.config == config
        assert builder.init_T == 0.013
        assert isinstance(builder.config_tracker, object)

    def test_init_missing_temperature(self):
        """Test that initialization fails without temperature in control parameters"""
        config = {
            "delta": 1.764e-23,
            "ymxc_y0": 0.8,
            "subgap_transparency": 0.01,
        }

        jm = Mock(spec=JsonManager)
        init_control_params = {"power": [100, "W"]}  # Wrong parameter

        with pytest.raises(ValueError, match="temperature.*not present"):
            DefaultBuilder("test_backend", config, jm, init_control_params)

    def test_init_missing_required_param(self):
        """Test that initialization fails with missing required hardware parameters"""
        config = {
            "delta": 1.764e-23,
            # Missing ymxc_y0
            "subgap_transparency": 0.01,
        }

        jm = Mock(spec=JsonManager)
        init_control_params = {"temperature": [0.013, "K"]}

        with pytest.raises(ValueError, match="Missing the required param: 'ymxc_y0'"):
            DefaultBuilder("test_backend", config, jm, init_control_params)


class TestQuasiparticleDensity:
    """Test x_qp calculation"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "delta": 1.764e-23,
            "ymxc_y0": 0.8,
            "subgap_transparency": 0.01,
        }
        self.jm = Mock(spec=JsonManager)
        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

    def test_x_qp_manual_calculation(self):
        """
        Test x_qp with manual calculation

        Formula: x_qp = sqrt(2*pi*k*T/delta) * exp(-delta/(k*T))

        Given:
        - T = 0.020 K = 20 mK
        - delta = 1.764e-23 J (typical superconducting gap for Al)
        - k = 1.380649e-23 J/K (Boltzmann constant)

        Calculation:
        1. k*T = 1.380649e-23 * 0.020 = 2.761298e-25 J
        2. delta/(k*T) = 1.764e-23 / 2.761298e-25 = 63.883
        3. 2*pi*k*T/delta = 2*pi*2.761298e-25 / 1.764e-23 = 0.098428
        4. sqrt(0.098428) = 0.31373
        5. exp(-63.883) = 1.467e-28
        6. x_qp = 0.31373 * 1.467e-28 = 4.603e-29
        """
        T = 0.020
        delta = 1.764e-23

        # Manual calculation
        arg_exp = delta / (k * T)
        arg_sqrt = 2 * pi * k * T / delta
        expected = sqrt(arg_sqrt) * exp(-arg_exp)

        result = self.builder.x_qp(T)

        # Verify intermediate values for transparency
        assert abs(arg_exp - 63.883) < 0.01, f"Expected arg_exp ~ 63.883, got {arg_exp}"
        assert abs(arg_sqrt - 0.098428) < 0.001, f"Expected arg_sqrt ~ 0.098428, got {arg_sqrt}"

        # Verify final result
        assert abs(result - expected) < 1e-32
        assert result < 1e-25  # Quasiparticle density should be very small at low T

    def test_x_qp_temperature_dependence(self):
        """Test that x_qp increases with temperature (physical expectation)"""
        T_low = 0.010  # 10 mK
        T_high = 0.030  # 30 mK

        x_qp_low = self.builder.x_qp(T_low)
        x_qp_high = self.builder.x_qp(T_high)

        assert x_qp_high > x_qp_low, "Quasiparticle density should increase with temperature"
        # At higher T, exponential term dominates, so ratio should be roughly exp(delta/k * (1/T_low - 1/T_high))
        expected_ratio_approx = exp(self.config["delta"]/k * (1/T_low - 1/T_high))
        actual_ratio = x_qp_high / x_qp_low
        # Ratios should be within same order of magnitude
        assert abs(np.log10(actual_ratio) - np.log10(expected_ratio_approx)) < 2


class TestBoseEinstein:
    """Test Bose-Einstein distribution"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "delta": 1.764e-23,
            "ymxc_y0": 0.8,
            "subgap_transparency": 0.01,
        }
        self.jm = Mock(spec=JsonManager)
        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

    def test_bose_einstein_manual_calculation(self):
        """
        Test Bose-Einstein distribution with manual calculation

        Formula: n(w, T) = 1 / (exp(hbar*w/(k*T)) - 1)

        Given:
        - w = 5e10 rad/s (typical qubit frequency ~ 8 GHz / 2pi)
        - T = 0.025 K = 25 mK
        - hbar = 1.054571817e-34 J*s
        - k = 1.380649e-23 J/K

        Calculation:
        1. hbar*w = 1.054571817e-34 * 5e10 = 5.273e-24 J
        2. k*T = 1.380649e-23 * 0.025 = 3.452e-25 J
        3. x = hbar*w/(k*T) = 5.273e-24 / 3.452e-25 = 15.28
        4. exp(15.28) = 4.316e6
        5. n = 1 / (4.316e6 - 1) ~ 2.317e-7
        """
        w = 5e10  # rad/s
        T = 0.025  # K

        # Manual calculation
        x = (hbar * w) / (k * T)
        expected = 1.0 / (exp(x) - 1.0)

        result = self.builder.bose_einstein(w, T)

        # Verify intermediate value
        assert abs(x - 15.28) < 0.01, f"Expected x ~ 15.28, got {x}"

        # Verify result
        assert abs(result - expected) / expected < 1e-10
        assert result < 1e-6, "At low T and high frequency, occupation should be very low"

    def test_bose_einstein_overflow_protection(self):
        """Test overflow protection for very large arguments"""
        w = 1e15  # Very high frequency
        T = 0.010  # Low temperature

        # This would cause exp(x) to overflow without protection
        x = (hbar * w) / (k * T)
        assert x > 700, "Test should use parameters that would cause overflow"

        # Should return 0 instead of raising OverflowError
        result = self.builder.bose_einstein(w, T)
        assert result == 0.0

    def test_bose_einstein_high_temperature_limit(self):
        """Test classical limit at high temperature: n ~ kT/(hbar*w)"""
        w = 1e9  # rad/s
        T = 1.0  # 1 K (high for quantum regime)

        result = self.builder.bose_einstein(w, T)

        # In classical limit: n ~ kT/(hbar*w) when hbar*w << kT
        classical_approx = (k * T) / (hbar * w)

        # They should be close when kT >> hbar*w
        if (k * T) > 10 * (hbar * w):
            assert abs(result - classical_approx) / classical_approx < 0.1


class TestEffectivePhotonNumber:
    """Test n_eff calculation (two-bath model)"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "ymxc_y0": 0.8,  # 80% from MXC bath, 20% from environment
            "delta": 1.764e-23,
            "subgap_transparency": 0.01,
        }
        self.jm = Mock(spec=JsonManager)
        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

    def test_n_eff_manual_calculation(self):
        """
        Test effective photon number with manual calculation

        Formula: n_eff = gamma_MXC_ratio * n_MXC + (1 - gamma_MXC_ratio) * n_env

        Given:
        - T = 0.020 K (MXC bath temperature)
        - T_env = 0.050 K (environment temperature)
        - gamma_MXC_ratio = 0.8
        - w_ge = 5e10 rad/s

        Calculation:
        1. n_MXC = bose_einstein(w_ge, T)
        2. n_env = bose_einstein(w_ge, T_env)
        3. n_eff = 0.8 * n_MXC + 0.2 * n_env
        """
        T = 0.020
        qp = make_qubit_params(T_env=0.050)

        n_MXC = self.builder.bose_einstein(qp["w_ge"], T)
        n_env = self.builder.bose_einstein(qp["w_ge"], qp["T_env"])
        gamma_ratio = self.config["ymxc_y0"]

        expected = gamma_ratio * n_MXC + (1 - gamma_ratio) * n_env

        result = self.builder.n_eff(T, qp)

        assert abs(result - expected) / max(expected, 1e-30) < 1e-10
        # n_env should be higher since T_env > T
        assert n_env > n_MXC
        # n_eff should be between n_MXC and n_env
        assert n_MXC <= result <= n_env


class TestT1Calculation:
    """Test T1 (energy relaxation time) calculation"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "delta": 1.764e-23,
            "ymxc_y0": 0.8,
            "subgap_transparency": 0.01,
        }
        self.jm = Mock(spec=JsonManager)
        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

    def test_T1_manual_calculation(self):
        """
        Test T1 with manual calculation

        Formula: T1 = 1 / [gamma_qp(T) + y0*(2*n_eff + 1)]

        Given:
        - T = 0.020 K
        - qp with y0 = 1e5

        The calculation involves:
        1. Calculate gamma_qp(T, qp)
        2. Calculate n_eff(T, qp)
        3. T1 = 1 / [gamma_qp + y0*(2*n_eff + 1)]

        At low T, gamma_qp should be very small, so T1 ~ 1/(y0*1) = 1/1e5 = 10 us
        """
        T = 0.020
        qp = make_qubit_params()

        gamma_qp = self.builder.gamma_qp(T, qp)
        n_eff = self.builder.n_eff(T, qp)
        y0 = qp["y0"]

        expected = 1 / (gamma_qp + y0 * (2*n_eff + 1))

        result = self.builder.T1(T, qp)

        assert abs(result - expected) / expected < 1e-10

        # Physical sanity checks
        assert result > 0, "T1 must be positive"
        assert result < 1e-3, "T1 should be on order of microseconds to milliseconds"

    def test_T1_temperature_dependence(self):
        """Test that T1 decreases with temperature (more decoherence at higher T)"""
        qp = make_qubit_params()
        T_low = 0.010
        T_high = 0.030

        T1_low = self.builder.T1(T_low, qp)
        T1_high = self.builder.T1(T_high, qp)

        # T1 should decrease at higher temperature due to increased thermal excitations
        assert T1_high < T1_low, "T1 should decrease with increasing temperature"


class TestT2Calculation:
    """Test T2 (dephasing time) calculation"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "delta": 1.764e-23,
            "ymxc_y0": 0.8,
            "subgap_transparency": 0.01,
        }
        self.jm = Mock(spec=JsonManager)
        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

    def test_T2_manual_calculation(self):
        """
        Test T2 with manual calculation

        Formula: T2 = 1 / (1/(2*T1) + 1/T_psi)
        where T_psi = 1 / (gamma_phi_base + gamma_phi_qp)

        The relationship T2 <= 2*T1 must always hold (physical constraint)
        """
        T = 0.020
        qp = make_qubit_params()

        T1 = self.builder.T1(T, qp)
        T_psi = self.builder.T_psi(T, qp)

        expected = 1 / (1/(2*T1) + 1/T_psi)

        result = self.builder.T2(T, qp)

        assert abs(result - expected) / expected < 1e-10

        # Physical constraint: T2 <= 2*T1
        assert result <= 2 * T1, "T2 cannot exceed 2*T1 (physical constraint)"
        assert result > 0, "T2 must be positive"

    def test_T2_vs_T1_relationship(self):
        """Test the fundamental relationship between T2 and T1"""
        T = 0.015
        qp = make_qubit_params()

        T1 = self.builder.T1(T, qp)
        T2 = self.builder.T2(T, qp)

        # T2 should always be <= 2*T1
        assert T2 <= 2 * T1 + 1e-15  # Small tolerance for floating point


class TestGateFidelity:
    """Test F_N (gate fidelity) calculation"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "delta": 1.764e-23,
            "ymxc_y0": 0.8,
            "subgap_transparency": 0.01,
        }
        self.jm = Mock(spec=JsonManager)
        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

    def test_F_N_manual_calculation(self):
        """
        Test gate fidelity with manual calculation

        Formula: F_N = 1 - (N*gate_length)/(2*(d+1)) * sum_i(1/T1_i + 1/T_psi_i)
        where d = 2^N (Hilbert space dimension)

        Given:
        - N = 2 (two-qubit gate)
        - gate_length = 40e-9 s (40 ns)
        - T = 0.015 K
        """
        N = 2  # Two-qubit gate
        gate_length = 40e-9  # 40 ns
        T = 0.015

        qp1 = make_qubit_params(frequency_hz=5e9)
        qp2 = make_qubit_params(frequency_hz=5.2e9)
        qubit_params_list = [qp1, qp2]

        d = 2 ** N
        rate_sum = 0
        for qp in qubit_params_list:
            T1_i = self.builder.T1(T, qp)
            T_psi_i = self.builder.T_psi(T, qp)
            rate_sum += 1/T1_i + 1/T_psi_i

        expected = 1 - (d * gate_length) / (2 * (d + 1)) * rate_sum

        result = self.builder.F_N(T, N, gate_length, qubit_params_list)

        assert abs(result - expected) / max(abs(expected), 1e-10) < 1e-10
        assert d == 4, f"Expected d=4 for N=2, got {d}"
        assert result <= 1.0, "Fidelity cannot exceed 1"

    def test_F_N_single_vs_two_qubit(self):
        """Test that two-qubit gates have lower fidelity than single-qubit gates"""
        gate_length = 40e-9
        T = 0.015
        qp = make_qubit_params()

        F_1 = self.builder.F_N(T, 1, gate_length, [qp])
        F_2 = self.builder.F_N(T, 2, gate_length, [qp, qp])

        # Two-qubit gates should have lower fidelity
        assert F_2 < F_1, "Two-qubit gates should have lower fidelity than single-qubit gates"

    def test_F_N_gate_length_dependence(self):
        """Test that longer gates have lower fidelity"""
        N = 1
        T = 0.015
        qp = make_qubit_params()

        F_short = self.builder.F_N(T, N, 20e-9, [qp])  # 20 ns
        F_long = self.builder.F_N(T, N, 60e-9, [qp])   # 60 ns

        # Longer gates accumulate more error
        assert F_long < F_short, "Longer gates should have lower fidelity"


class TestQubitConfigCalculation:
    """Test calculate_qb_config method"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "delta": 1.764e-23,
            "ymxc_y0": 0.8,
            "subgap_transparency": 0.01,
            "E_c": 3.5e-24,
            "E_J": 2.1e-22,
            "R_n": 5000,
        }
        self.jm = Mock(spec=JsonManager)
        self.jm.find_value_with_units = Mock(side_effect=self._mock_find_value)
        self.jm.get_qubit_paths = Mock(return_value=["qubits.[0]."])

        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

        # Initialize per-qubit params so the cache is populated
        self.builder.initialize_per_qubit_params()

    def _mock_find_value(self, param, path):
        """Mock return values for JsonManager.find_value_with_units"""
        if param == "frequency":
            return 5e9  # 5 GHz
        elif param == "anharmonicity":
            return -0.33e9  # -330 MHz
        elif param == "T1":
            return 50e-6  # 50 us
        elif param == "T2":
            return 70e-6  # 70 us
        elif param == "prob_meas1_prep0":
            return 0.02
        return None

    def test_calculate_qb_config_structure(self):
        """Test that calculate_qb_config returns correct structure"""
        control_params = {"temperature": [0.020, "K"]}
        qb_path = "qubits.[0]."

        qb_config = self.builder.calculate_qb_config(control_params, qb_path)

        assert "T1" in qb_config
        assert "T2" in qb_config
        assert qb_config["T1"] > 0
        assert qb_config["T2"] > 0
        assert qb_config["T2"] <= 2 * qb_config["T1"]  # Physical constraint

    def test_calculate_qb_config_tracks_configs(self):
        """Test that calculation adds entry to config_tracker"""
        control_params = {"temperature": [0.020, "K"]}
        qb_path = "qubits.[0]."

        initial_count = len(self.builder.config_tracker.qb_config_infos)
        self.builder.calculate_qb_config(control_params, qb_path)

        assert len(self.builder.config_tracker.qb_config_infos) == initial_count + 1


class TestGateConfigCalculation:
    """Test calculate_gate_config method"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "delta": 1.764e-23,
            "ymxc_y0": 0.8,
            "subgap_transparency": 0.01,
            "E_c": 3.5e-24,
            "E_J": 2.1e-22,
            "R_n": 5000,
        }
        self.jm = Mock(spec=JsonManager)
        self.jm.resolve = Mock(return_value=[0, 1])  # Two-qubit gate
        self.jm.find_value_with_units = Mock(side_effect=self._mock_find_value)
        self.jm.get_qubit_paths = Mock(return_value=["qubits.[0].", "qubits.[1]."])

        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

        # Initialize per-qubit params so the cache is populated
        self.builder.initialize_per_qubit_params()

    def _mock_find_value(self, param, path):
        """Mock return values for JsonManager.find_value_with_units"""
        if param == "frequency":
            if "[0]" in path:
                return 5.0e9
            elif "[1]" in path:
                return 5.2e9
            return 5.1e9
        elif param == "anharmonicity":
            return -0.33e9
        elif param == "T1":
            return 50e-6
        elif param == "T2":
            return 70e-6
        elif param == "prob_meas1_prep0":
            return 0.02
        elif param == "gate_error":
            return 0.01  # 1% gate error
        elif param == "gate_length":
            return 40e-9  # 40 ns
        return None

    def test_calculate_gate_config_structure(self):
        """Test that calculate_gate_config returns correct structure"""
        control_params = {"temperature": [0.020, "K"]}
        gate_path = "gates.[0]."

        gate_config = self.builder.calculate_gate_config(control_params, gate_path)

        assert "gate_error" in gate_config
        assert gate_config["gate_error"] is not None
        assert 0 <= gate_config["gate_error"] <= 1

    def test_calculate_gate_config_clamps_error(self):
        """Test that gate error is clamped to [0, 1] range"""
        control_params = {"temperature": [0.001, "K"]}  # Very low T -> low error
        gate_path = "gates.[0]."

        gate_config = self.builder.calculate_gate_config(control_params, gate_path)

        # Error should be clamped
        assert gate_config["gate_error"] >= 0
        assert gate_config["gate_error"] <= 1


class TestPerQubitInitialization:
    """Test per-qubit parameter initialization"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "delta": 2.88e-23,
            "ymxc_y0": 0.9,
            "subgap_transparency": 0.00005,
            "E_c": 1.524e-25,
            "E_J": 9.944e-24,
            "R_n": 5000,
        }
        self.jm = Mock(spec=JsonManager)
        self.jm.get_qubit_paths = Mock(return_value=["qubits.[0].", "qubits.[1]."])
        self.jm.find_value_with_units = Mock(side_effect=self._mock_find_value)

        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

    def _mock_find_value(self, param, path):
        """Mock return values for JsonManager.find_value_with_units"""
        if param == "frequency":
            if "[0]" in path:
                return 4.636e9  # ~4.636 GHz
            return 4.736e9
        elif param == "anharmonicity":
            return -0.313e9  # ~-313 MHz
        elif param == "T1":
            if "[0]" in path:
                return 571e-6  # 571 us
            return 337e-6
        elif param == "T2":
            if "[0]" in path:
                return 283e-6
            return 191e-6
        elif param == "prob_meas1_prep0":
            return 0.02
        return None

    def test_initialize_populates_cache(self):
        """Test that initialize_per_qubit_params populates the cache for all qubits"""
        self.builder.initialize_per_qubit_params()

        assert len(self.builder._qubit_params_cache) == 2
        assert "qubits.[0]." in self.builder._qubit_params_cache
        assert "qubits.[1]." in self.builder._qubit_params_cache

    def test_derived_params_have_correct_keys(self):
        """Test that derived qubit params have all expected keys"""
        self.builder.initialize_per_qubit_params()
        qp = self.builder.get_qubit_params("qubits.[0].")

        expected_keys = {"E_c", "E_J", "w_p", "R_n", "N_e", "y0", "gamma_psi_base", "T_env", "w_ge"}
        assert set(qp.keys()) == expected_keys

    def test_E_c_from_anharmonicity(self):
        """Test that E_c is derived from anharmonicity data"""
        self.builder.initialize_per_qubit_params()
        qp = self.builder.get_qubit_params("qubits.[0].")

        # E_c = h * |anharmonicity|
        expected_E_c = h * abs(-0.313e9)
        assert abs(qp["E_c"] - expected_E_c) / expected_E_c < 1e-10

    def test_E_J_E_c_ratio_sanity(self):
        """Test that E_J/E_c ratio is in typical transmon range (30-80)"""
        self.builder.initialize_per_qubit_params()
        qp = self.builder.get_qubit_params("qubits.[0].")

        ratio = qp["E_J"] / qp["E_c"]
        assert 10 < ratio < 200, f"E_J/E_c ratio {ratio:.1f} outside expected range"

    def test_R_n_in_expected_range(self):
        """Test that derived R_n is in typical range (kOhm)"""
        self.builder.initialize_per_qubit_params()
        qp = self.builder.get_qubit_params("qubits.[0].")

        # R_n should be in 1-50 kOhm range for typical transmons
        assert 100 < qp["R_n"] < 100000, f"R_n = {qp['R_n']:.0f} Ohm outside expected range"

    def test_gamma_psi_base_non_negative(self):
        """Test that gamma_psi_base is non-negative (physical constraint)"""
        self.builder.initialize_per_qubit_params()
        qp = self.builder.get_qubit_params("qubits.[0].")

        assert qp["gamma_psi_base"] >= 0, "gamma_psi_base must be non-negative"

    def test_fallback_E_c_when_no_anharmonicity(self):
        """Test that global E_c is used as fallback when anharmonicity is missing"""
        def mock_find_no_anharmonicity(param, path):
            if param == "anharmonicity":
                return None
            return self._mock_find_value(param, path)

        self.jm.find_value_with_units = Mock(side_effect=mock_find_no_anharmonicity)

        self.builder.initialize_per_qubit_params()
        qp = self.builder.get_qubit_params("qubits.[0].")

        # Should use global E_c from config
        assert qp["E_c"] == self.config["E_c"]

    def test_median_T_env_computed(self):
        """Test that median_T_env is computed during initialization"""
        self.builder.initialize_per_qubit_params()

        assert hasattr(self.builder, 'median_T_env')
        assert self.builder.median_T_env > 0

    def test_different_qubits_get_different_params(self):
        """Test that qubits with different frequencies get different derived params"""
        self.builder.initialize_per_qubit_params()
        qp0 = self.builder.get_qubit_params("qubits.[0].")
        qp1 = self.builder.get_qubit_params("qubits.[1].")

        # Different frequencies should give different E_J
        assert qp0["w_ge"] != qp1["w_ge"]
        assert qp0["E_J"] != qp1["E_J"]
        # Different T1_meas should give different y0
        assert qp0["y0"] != qp1["y0"]


class TestPhysicalConsistency:
    """Test physical consistency across calculations"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "delta": 1.764e-23,
            "ymxc_y0": 0.8,
            "subgap_transparency": 0.01,
        }
        self.jm = Mock(spec=JsonManager)
        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

    def test_T2_always_less_than_2T1(self):
        """Test fundamental constraint T2 <= 2*T1"""
        temperatures = [0.010, 0.015, 0.020, 0.030, 0.050]
        frequencies = [3e9, 5e9, 7e9]

        for T in temperatures:
            for freq in frequencies:
                qp = make_qubit_params(frequency_hz=freq)
                T1 = self.builder.T1(T, qp)
                T2 = self.builder.T2(T, qp)

                assert T2 <= 2 * T1 + 1e-15, \
                    f"T2 ({T2}) > 2*T1 ({2*T1}) at T={T}, freq={freq}"

    def test_gate_fidelity_decreases_with_temperature(self):
        """Test that gate fidelity decreases with increasing temperature"""
        N = 2
        gate_length = 40e-9
        qp1 = make_qubit_params(frequency_hz=5e9)
        qp2 = make_qubit_params(frequency_hz=5.2e9)
        qubit_params_list = [qp1, qp2]

        temperatures = [0.010, 0.020, 0.030]
        fidelities = [self.builder.F_N(T, N, gate_length, qubit_params_list) for T in temperatures]

        for i in range(len(fidelities) - 1):
            assert fidelities[i] > fidelities[i+1], \
                f"Fidelity should decrease with temperature: F({temperatures[i]})={fidelities[i]}, F({temperatures[i+1]})={fidelities[i+1]}"

    def test_all_times_positive(self):
        """Test that all time constants are positive"""
        T = 0.020
        qp = make_qubit_params()

        T1 = self.builder.T1(T, qp)
        T2 = self.builder.T2(T, qp)
        T_psi = self.builder.T_psi(T, qp)

        assert T1 > 0, "T1 must be positive"
        assert T2 > 0, "T2 must be positive"
        assert T_psi > 0, "T_psi must be positive"

    def test_quasiparticle_density_very_small_at_low_T(self):
        """Test that quasiparticle density is exponentially suppressed at low T"""
        T_low = 0.010
        T_high = 0.040

        x_qp_low = self.builder.x_qp(T_low)
        x_qp_high = self.builder.x_qp(T_high)

        # Should be exponentially smaller
        assert x_qp_low < 1e-20, "Quasiparticle density should be very small at low T"
        assert x_qp_high > x_qp_low * 1e5, "Should increase exponentially with temperature"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
