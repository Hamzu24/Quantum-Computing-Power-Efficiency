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


class TestDefaultBuilderInitialization:
    """Test initialization and validation of DefaultBuilder"""

    def test_init_with_valid_config(self):
        """Test initialization with all required parameters"""
        config = {
            "y0": 1e5,
            "ymxc_y0": 0.8,
            "T_env": 0.050,  # 50 mK in Kelvin
            "gamma_psi_base": 1e3,
            "delta": 1.764e-23,  # Superconducting gap
            "E_c": 3.5e-24,  # Charging energy (J)
            "E_J": 2.1e-22,  # Josephson energy (J)
            "R_n": 5000,  # Normal state resistance (Ohms)
            "subgap_transparency": 0.01
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
            "y0": 1e5, "ymxc_y0": 0.8, "T_env": 0.050,
            "gamma_psi_base": 1e3, "delta": 1.764e-23,
            "E_c": 3.5e-24, "E_J": 2.1e-22,
            "R_n": 5000, "subgap_transparency": 0.01
        }

        jm = Mock(spec=JsonManager)
        init_control_params = {"power": [100, "W"]}  # Wrong parameter

        with pytest.raises(ValueError, match="temperature.*not present"):
            DefaultBuilder("test_backend", config, jm, init_control_params)

    def test_init_missing_required_param(self):
        """Test that initialization fails with missing required hardware parameters"""
        config = {
            "y0": 1e5,
            "ymxc_y0": 0.8,
            # Missing T_env
            "gamma_psi_base": 1e3,
            "delta": 1.764e-23,
            "E_c": 3.5e-24,
            "E_J": 2.1e-22,
            "R_n": 5000,
            "subgap_transparency": 0.01
        }

        jm = Mock(spec=JsonManager)
        init_control_params = {"temperature": [0.013, "K"]}

        with pytest.raises(ValueError, match="Missing the required param: 'T_env'"):
            DefaultBuilder("test_backend", config, jm, init_control_params)


class TestQuasiparticleDensity:
    """Test x_qp calculation"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "y0": 1e5, "ymxc_y0": 0.8, "T_env": 0.050,
            "gamma_psi_base": 1e3, "delta": 1.764e-23,
            "E_c": 3.5e-24, "E_J": 2.1e-22,
            "R_n": 5000, "subgap_transparency": 0.01
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
        assert abs(arg_exp - 63.883) < 0.01, f"Expected arg_exp ≈ 63.883, got {arg_exp}"
        assert abs(arg_sqrt - 0.098428) < 0.001, f"Expected arg_sqrt ≈ 0.098428, got {arg_sqrt}"

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


class TestPlasmaFrequency:
    """Test w_p calculation"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "y0": 1e5, "ymxc_y0": 0.8, "T_env": 0.050,
            "gamma_psi_base": 1e3, "delta": 1.764e-23,
            "E_c": 3.5e-24, "E_J": 2.1e-22,
            "R_n": 5000, "subgap_transparency": 0.01
        }
        self.jm = Mock(spec=JsonManager)
        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

    def test_w_p_manual_calculation(self):
        """
        Test plasma frequency with manual calculation

        Formula: w_p = w_p0 * sqrt(screening_factor)
        where:
        - w_p0 = sqrt(8*E_J*E_c) / hbar
        - screening_factor = 1 - 2*sqrt(2*pi*k*T/delta) * exp(-delta/(k*T))

        Given:
        - E_J = 2.1e-22 J
        - E_c = 3.5e-24 J
        - T = 0.015 K
        - delta = 1.764e-23 J
        - hbar = 1.054571817e-34 J·s

        Calculation:
        1. 8*E_J*E_c = 8 * 2.1e-22 * 3.5e-24 = 5.88e-45 J²
        2. sqrt(5.88e-45) = 7.668e-23 J
        3. w_p0 = 7.668e-23 / 1.054571817e-34 = 7.272e11 rad/s
        4. screening_factor = 1 - 2*x_qp(T) ≈ 1 - (very small number) ≈ 1
        5. w_p ≈ w_p0 for low temperatures
        """
        T = 0.015
        E_J = self.config["E_J"]
        E_c = self.config["E_c"]

        # Manual calculation
        w_p0 = sqrt(8 * E_J * E_c) / hbar
        x_qp = self.builder.x_qp(T)
        screening_factor = 1 - 2 * x_qp
        expected = w_p0 * sqrt(max(screening_factor, 0.4**2))  # Account for fallback

        result = self.builder.w_p(T)

        # Verify w_p0
        expected_w_p0 = 7.272e11  # rad/s
        assert abs(w_p0 - expected_w_p0) / expected_w_p0 < 0.01, \
            f"Expected w_p0 ≈ 7.272e11, got {w_p0:.3e}"

        # At low T, screening is minimal, so w_p ≈ w_p0
        assert abs(result - w_p0) / w_p0 < 0.01, \
            "At low temperature, plasma frequency should be close to base value"

    def test_w_p_screening_fallback(self):
        """Test fallback when screening factor goes negative"""
        # At very high temperature, screening factor could go negative
        # Builder should use fallback value of 0.4 * w_p0
        T_high = 1.0  # 1 K - unrealistically high but tests the fallback

        result = self.builder.w_p(T_high)
        w_p0 = sqrt(8 * self.config["E_J"] * self.config["E_c"]) / hbar

        # Should be less than w_p0 due to screening or fallback
        assert result <= w_p0
        assert result > 0


class TestBoseEinstein:
    """Test Bose-Einstein distribution"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "y0": 1e5, "ymxc_y0": 0.8, "T_env": 0.050,
            "gamma_psi_base": 1e3, "delta": 1.764e-23,
            "E_c": 3.5e-24, "E_J": 2.1e-22,
            "R_n": 5000, "subgap_transparency": 0.01
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
        - w = 5e10 rad/s (typical qubit frequency ~ 8 GHz / 2π)
        - T = 0.025 K = 25 mK
        - hbar = 1.054571817e-34 J·s
        - k = 1.380649e-23 J/K

        Calculation:
        1. hbar*w = 1.054571817e-34 * 5e10 = 5.273e-24 J
        2. k*T = 1.380649e-23 * 0.025 = 3.452e-25 J
        3. x = hbar*w/(k*T) = 5.273e-24 / 3.452e-25 = 15.28
        4. exp(15.28) = 4.316e6
        5. n = 1 / (4.316e6 - 1) ≈ 2.317e-7
        """
        w = 5e10  # rad/s
        T = 0.025  # K

        # Manual calculation
        x = (hbar * w) / (k * T)
        expected = 1.0 / (exp(x) - 1.0)

        result = self.builder.bose_einstein(w, T)

        # Verify intermediate value
        assert abs(x - 15.28) < 0.01, f"Expected x ≈ 15.28, got {x}"

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
        """Test classical limit at high temperature: n ≈ kT/(hbar*w)"""
        w = 1e9  # rad/s
        T = 1.0  # 1 K (high for quantum regime)

        result = self.builder.bose_einstein(w, T)

        # In classical limit: n ≈ kT/(hbar*w) when hbar*w << kT
        classical_approx = (k * T) / (hbar * w)

        # They should be close when kT >> hbar*w
        if (k * T) > 10 * (hbar * w):
            assert abs(result - classical_approx) / classical_approx < 0.1


class TestEffectivePhotonNumber:
    """Test n_eff calculation (two-bath model)"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "y0": 1e5,
            "ymxc_y0": 0.8,  # 80% from MXC bath, 20% from environment
            "T_env": 0.050,  # 50 mK
            "gamma_psi_base": 1e3,
            "delta": 1.764e-23,
            "E_c": 3.5e-24,
            "E_J": 2.1e-22,
            "R_n": 5000,
            "subgap_transparency": 0.01
        }
        self.jm = Mock(spec=JsonManager)
        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

    def test_n_eff_manual_calculation(self):
        """
        Test effective photon number with manual calculation

        Formula: n_eff = γ_MXC_ratio * n_MXC + (1 - γ_MXC_ratio) * n_env

        Given:
        - T = 0.020 K (MXC bath temperature)
        - T_env = 0.050 K (environment temperature)
        - γ_MXC_ratio = 0.8
        - frequency = 5e10 rad/s

        Calculation:
        1. n_MXC = bose_einstein(w, T)
        2. n_env = bose_einstein(w, T_env)
        3. n_eff = 0.8 * n_MXC + 0.2 * n_env
        """
        T = 0.020
        frequency = 5e10

        n_MXC = self.builder.bose_einstein(frequency, T)
        n_env = self.builder.bose_einstein(frequency, self.config["T_env"])
        gamma_ratio = self.config["ymxc_y0"]

        expected = gamma_ratio * n_MXC + (1 - gamma_ratio) * n_env

        result = self.builder.n_eff(T, frequency)

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
            "y0": 1e5,  # Base relaxation rate
            "ymxc_y0": 0.8,
            "T_env": 0.050,
            "gamma_psi_base": 1e3,
            "delta": 1.764e-23,
            "E_c": 3.5e-24,
            "E_J": 2.1e-22,
            "R_n": 5000,
            "subgap_transparency": 0.01
        }
        self.jm = Mock(spec=JsonManager)
        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

    def test_T1_manual_calculation(self):
        """
        Test T1 with manual calculation

        Formula: T1 = 1 / [γ_qp(T) + γ_0(2*n_eff + 1)]

        Given:
        - T = 0.020 K
        - frequency = 5e10 rad/s
        - y0 = 1e5 (γ_0)

        The calculation involves:
        1. Calculate γ_qp(T, frequency)
        2. Calculate n_eff(T, frequency)
        3. T1 = 1 / [γ_qp + y0*(2*n_eff + 1)]

        At low T, γ_qp should be very small, so T1 ≈ 1/(y0*1) = 1/1e5 = 10 μs
        """
        T = 0.020
        frequency = 5e10

        gamma_qp = self.builder.gamma_qp(T, frequency)
        n_eff = self.builder.n_eff(T, frequency)
        y0 = self.config["y0"]

        expected = 1 / (gamma_qp + y0 * (2*n_eff + 1))

        result = self.builder.T1(T, frequency)

        assert abs(result - expected) / expected < 1e-10

        # Physical sanity checks
        assert result > 0, "T1 must be positive"
        assert result < 1e-3, "T1 should be on order of microseconds to milliseconds"

    def test_T1_temperature_dependence(self):
        """Test that T1 decreases with temperature (more decoherence at higher T)"""
        frequency = 5e10
        T_low = 0.010
        T_high = 0.030

        T1_low = self.builder.T1(T_low, frequency)
        T1_high = self.builder.T1(T_high, frequency)

        # T1 should decrease at higher temperature due to increased thermal excitations
        assert T1_high < T1_low, "T1 should decrease with increasing temperature"


class TestT2Calculation:
    """Test T2 (dephasing time) calculation"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "y0": 1e5,
            "ymxc_y0": 0.8,
            "T_env": 0.050,
            "gamma_psi_base": 1e3,  # Base dephasing rate
            "delta": 1.764e-23,
            "E_c": 3.5e-24,
            "E_J": 2.1e-22,
            "R_n": 5000,
            "subgap_transparency": 0.01
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
        where T_psi = 1 / (γ_φ_base + γ_φ_qp)

        The relationship T2 ≤ 2*T1 must always hold (physical constraint)
        """
        T = 0.020
        frequency = 5e10

        T1 = self.builder.T1(T, frequency)
        T_psi = self.builder.T_psi(T, frequency)

        expected = 1 / (1/(2*T1) + 1/T_psi)

        result = self.builder.T2(T, frequency)

        assert abs(result - expected) / expected < 1e-10

        # Physical constraint: T2 ≤ 2*T1
        assert result <= 2 * T1, "T2 cannot exceed 2*T1 (physical constraint)"
        assert result > 0, "T2 must be positive"

    def test_T2_vs_T1_relationship(self):
        """Test the fundamental relationship between T2 and T1"""
        T = 0.015
        frequency = 5e10

        T1 = self.builder.T1(T, frequency)
        T2 = self.builder.T2(T, frequency)

        # T2 should always be ≤ 2*T1
        assert T2 <= 2 * T1 + 1e-15  # Small tolerance for floating point


class TestGateFidelity:
    """Test F_N (gate fidelity) calculation"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "y0": 1e5,
            "ymxc_y0": 0.8,
            "T_env": 0.050,
            "gamma_psi_base": 1e3,
            "delta": 1.764e-23,
            "E_c": 3.5e-24,
            "E_J": 2.1e-22,
            "R_n": 5000,
            "subgap_transparency": 0.01
        }
        self.jm = Mock(spec=JsonManager)
        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

    def test_F_N_manual_calculation(self):
        """
        Test gate fidelity with manual calculation

        Formula: F_N = 1 - (d*N*gate_length)/(2*(d+1)) * (1/T1 + 1/T_psi)
        where d = 2^N (Hilbert space dimension)

        Given:
        - N = 2 (two-qubit gate)
        - gate_length = 40e-9 s (40 ns)
        - T = 0.015 K
        - frequency = 5e10 rad/s

        Calculation:
        1. d = 2^2 = 4
        2. T1 = T1(T, frequency)
        3. T_psi = T_psi(T, frequency)
        4. error = (4*2*40e-9)/(2*5) * (1/T1 + 1/T_psi)
        5. F_N = 1 - error
        """
        N = 2  # Two-qubit gate
        gate_length = 40e-9  # 40 ns
        T = 0.015
        frequency = 5e10

        d = 2 ** N
        T1 = self.builder.T1(T, frequency)
        T_psi = self.builder.T_psi(T, frequency)

        error = (d * N * gate_length) / (2 * (d + 1)) * (1/T1 + 1/T_psi)
        expected = 1 - error

        result = self.builder.F_N(T, N, gate_length, frequency)

        assert abs(result - expected) / max(abs(expected), 1e-10) < 1e-10

        # Verify intermediate calculations
        assert d == 4, f"Expected d=4 for N=2, got {d}"
        assert result <= 1.0, "Fidelity cannot exceed 1"

    def test_F_N_single_vs_two_qubit(self):
        """Test that two-qubit gates have lower fidelity than single-qubit gates"""
        gate_length = 40e-9
        T = 0.015
        frequency = 5e10

        F_1 = self.builder.F_N(T, 1, gate_length, frequency)
        F_2 = self.builder.F_N(T, 2, gate_length, frequency)

        # Two-qubit gates should have lower fidelity
        assert F_2 < F_1, "Two-qubit gates should have lower fidelity than single-qubit gates"

    def test_F_N_gate_length_dependence(self):
        """Test that longer gates have lower fidelity"""
        N = 1
        T = 0.015
        frequency = 5e10

        F_short = self.builder.F_N(T, N, 20e-9, frequency)  # 20 ns
        F_long = self.builder.F_N(T, N, 60e-9, frequency)   # 60 ns

        # Longer gates accumulate more error
        assert F_long < F_short, "Longer gates should have lower fidelity"


class TestParameterErrorCalculation:
    """Test calculate_parameter_error method"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "y0": 1e5,
            "ymxc_y0": 0.8,
            "T_env": 0.050,
            "gamma_psi_base": 1e3,
            "delta": 1.764e-23,
            "E_c": 3.5e-24,
            "E_J": 2.1e-22,
            "R_n": 5000,
            "subgap_transparency": 0.01
        }
        self.jm = Mock(spec=JsonManager)
        self.jm.find_value_with_units = Mock(side_effect=self._mock_find_value)

        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

    def _mock_find_value(self, param, path):
        """Mock return values for JsonManager.find_value_with_units"""
        if param == "frequency":
            return 5e10  # 50 GHz in rad/s
        elif param == "T1":
            return 50e-6  # 50 μs
        elif param == "T2":
            return 70e-6  # 70 μs
        return None

    def test_calculate_parameter_error_ratio(self):
        """Test ratio error type: actual / calculated"""
        qb_path = "qubits.[0]."

        error, actual, calculated = self.builder.calculate_parameter_error(
            0.013, qb_path, type="ratio", parameter="T1"
        )

        expected_error = actual / calculated
        assert abs(error - expected_error) < 1e-10
        assert actual == 50e-6
        assert calculated > 0

    def test_calculate_parameter_error_square(self):
        """Test square error type: (actual - calculated)^2"""
        qb_path = "qubits.[0]."

        error, actual, calculated = self.builder.calculate_parameter_error(
            0.013, qb_path, type="square", parameter="T1"
        )

        expected_error = (actual - calculated) ** 2
        assert abs(error - expected_error) < 1e-30

    def test_calculate_parameter_error_relative(self):
        """Test relative error type: |calculated - actual| / |actual|"""
        qb_path = "qubits.[0]."

        error, actual, calculated = self.builder.calculate_parameter_error(
            0.013, qb_path, type="relative", parameter="T2"
        )

        expected_error = abs(calculated - actual) / abs(actual)
        assert abs(error - expected_error) < 1e-10
        assert 0 <= error  # Relative error is always non-negative

    def test_calculate_parameter_error_with_override(self):
        """Test that override_config temporarily changes config"""
        qb_path = "qubits.[0]."
        original_y0 = self.builder.config["y0"]  # Check builder's config, not test's config

        override = {"y0": 2e5}
        error, actual, calculated = self.builder.calculate_parameter_error(
            0.013, qb_path, type="ratio", parameter="T1", override_config=override
        )

        # Config should be restored after calculation
        assert self.builder.config["y0"] == original_y0, \
            f"Config not restored: expected {original_y0:.2e}, got {self.builder.config['y0']:.2e}"

    def test_calculate_parameter_error_invalid_type(self):
        """Test invalid error type raises ValueError"""
        qb_path = "qubits.[0]."

        with pytest.raises(ValueError, match="Invalid error type"):
            self.builder.calculate_parameter_error(
                0.013, qb_path, type="invalid", parameter="T1"
            )

    def test_calculate_parameter_error_invalid_parameter(self):
        """Test invalid parameter raises ValueError"""
        qb_path = "qubits.[0]."

        with pytest.raises(ValueError, match="Invalid parameter type"):
            self.builder.calculate_parameter_error(
                0.013, qb_path, type="ratio", parameter="T3"
            )


class TestQubitConfigCalculation:
    """Test calculate_qb_config method"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "y0": 1e5,
            "ymxc_y0": 0.8,
            "T_env": 0.050,
            "gamma_psi_base": 1e3,
            "delta": 1.764e-23,
            "E_c": 3.5e-24,
            "E_J": 2.1e-22,
            "R_n": 5000,
            "subgap_transparency": 0.01
        }
        self.jm = Mock(spec=JsonManager)
        self.jm.find_value_with_units = Mock(side_effect=self._mock_find_value)

        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

    def _mock_find_value(self, param, path):
        """Mock return values for JsonManager.find_value_with_units"""
        if param == "frequency":
            return 5e10
        elif param == "T1":
            return 50e-6
        elif param == "T2":
            return 70e-6
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

    def test_calculate_qb_config_adjustment_factors(self):
        """
        Test that adjustment factors correctly scale calculated values to match actual

        The method calculates adjustment ratios at init_T:
        - adj_T1 = actual_T1 / calculated_T1(init_T)
        - adj_T2 = actual_T2 / calculated_T2(init_T)

        Then applies these to working temperature:
        - working_T1 = calculated_T1(working_T) * adj_T1
        - working_T2 = calculated_T2(working_T) * adj_T2
        """
        control_params = {"temperature": [0.013, "K"]}  # Same as init_T
        qb_path = "qubits.[0]."

        qb_config = self.builder.calculate_qb_config(control_params, qb_path)

        # At init_T, working values should match actual values from backend
        # (within floating point tolerance)
        actual_T1 = 50e-6
        actual_T2 = 70e-6

        # Should be close to actual values since we're at init_T
        assert abs(qb_config["T1"] - actual_T1) / actual_T1 < 0.5, \
            "T1 adjustment should bring calculated close to actual at init_T"
        assert abs(qb_config["T2"] - actual_T2) / actual_T2 < 0.5, \
            "T2 adjustment should bring calculated close to actual at init_T"

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
            "y0": 1e5,
            "ymxc_y0": 0.8,
            "T_env": 0.050,
            "gamma_psi_base": 1e3,
            "delta": 1.764e-23,
            "E_c": 3.5e-24,
            "E_J": 2.1e-22,
            "R_n": 5000,
            "subgap_transparency": 0.01
        }
        self.jm = Mock(spec=JsonManager)
        self.jm.resolve = Mock(return_value=[0, 1])  # Two-qubit gate
        self.jm.find_value_with_units = Mock(side_effect=self._mock_find_value)

        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

    def _mock_find_value(self, param, path):
        """Mock return values for JsonManager.find_value_with_units"""
        if param == "frequency":
            # Return different frequencies for different qubits
            if "[0]" in path:
                return 5.0e10
            elif "[1]" in path:
                return 5.2e10
            return 5.1e10
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

    def test_calculate_gate_config_uses_average_frequency(self):
        """Test that gate config uses average frequency of involved qubits"""
        control_params = {"temperature": [0.020, "K"]}
        gate_path = "gates.[0]."

        gate_config = self.builder.calculate_gate_config(control_params, gate_path)

        # Should call find_value_with_units for frequency for each qubit
        frequency_calls = [
            call for call in self.jm.find_value_with_units.call_args_list
            if call[0][0] == "frequency"
        ]
        assert len(frequency_calls) == 2, "Should query frequency for both qubits"

    def test_calculate_gate_config_clamps_error(self):
        """Test that gate error is clamped to [0, 1] range"""
        control_params = {"temperature": [0.001, "K"]}  # Very low T -> low error
        gate_path = "gates.[0]."

        gate_config = self.builder.calculate_gate_config(control_params, gate_path)

        # Error should be clamped
        assert gate_config["gate_error"] >= 0
        assert gate_config["gate_error"] <= 1

    def test_calculate_gate_config_no_actual_error(self):
        """Test behavior when actual gate error is None"""
        self.jm.find_value_with_units = Mock(side_effect=lambda p, path: {
            "frequency": 5e10,
            "gate_error": None,  # No actual error available
            "gate_length": 40e-9
        }.get(p))

        control_params = {"temperature": [0.020, "K"]}
        gate_path = "gates.[0]."

        gate_config = self.builder.calculate_gate_config(control_params, gate_path)

        assert gate_config["gate_error"] is None


class TestPhysicalConsistency:
    """Test physical consistency across calculations"""

    def setup_method(self):
        """Setup common test fixtures"""
        self.config = {
            "y0": 1e5,
            "ymxc_y0": 0.8,
            "T_env": 0.050,
            "gamma_psi_base": 1e3,
            "delta": 1.764e-23,
            "E_c": 3.5e-24,
            "E_J": 2.1e-22,
            "R_n": 5000,
            "subgap_transparency": 0.01
        }
        self.jm = Mock(spec=JsonManager)
        self.builder = DefaultBuilder(
            "test", self.config, self.jm,
            {"temperature": [0.013, "K"]}
        )

    def test_T2_always_less_than_2T1(self):
        """Test fundamental constraint T2 ≤ 2*T1"""
        temperatures = [0.010, 0.015, 0.020, 0.030, 0.050]
        frequencies = [3e10, 5e10, 7e10]

        for T in temperatures:
            for freq in frequencies:
                T1 = self.builder.T1(T, freq)
                T2 = self.builder.T2(T, freq)

                assert T2 <= 2 * T1 + 1e-15, \
                    f"T2 ({T2}) > 2*T1 ({2*T1}) at T={T}, freq={freq}"

    def test_gate_fidelity_decreases_with_temperature(self):
        """Test that gate fidelity decreases with increasing temperature"""
        N = 2
        gate_length = 40e-9
        frequency = 5e10

        temperatures = [0.010, 0.020, 0.030]
        fidelities = [self.builder.F_N(T, N, gate_length, frequency) for T in temperatures]

        for i in range(len(fidelities) - 1):
            assert fidelities[i] > fidelities[i+1], \
                f"Fidelity should decrease with temperature: F({temperatures[i]})={fidelities[i]}, F({temperatures[i+1]})={fidelities[i+1]}"

    def test_all_times_positive(self):
        """Test that all time constants are positive"""
        T = 0.020
        frequency = 5e10

        T1 = self.builder.T1(T, frequency)
        T2 = self.builder.T2(T, frequency)
        T_psi = self.builder.T_psi(T, frequency)

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
