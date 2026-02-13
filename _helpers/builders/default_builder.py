from _helpers.builders.base import builder_registry, ConfigTracker
import logging
from _helpers.json_manager import JsonManager
from math import sqrt, exp, cosh, log
from statistics import median
from scipy.constants import pi, k, hbar, e, h
from scipy.special import k0
from _helpers.helpers import get_config_value

@builder_registry.register_builder
class DefaultBuilder:
    registry_name = "DefaultBuilder"
    def __init__(self, name: str, config: dict, jm: JsonManager, init_control_parameters: dict={"temperature": [13, "mK"]}):
        self.name = name
        self.config = config
        self.init_T=get_config_value(init_control_parameters, "temperature")
        if self.init_T is None:
            raise ValueError("The expected key 'temperature' is not present in the control parameters dictionary!")
        self.is_valid()
        self.json_manager = jm
        self.config_tracker = ConfigTracker()
        self._qubit_params_cache = {}

    def is_valid(self):
        required_params = ["delta", "ymxc_y0", "subgap_transparency"]

        for param in required_params:
            if self.config.get(param) is None:
                raise ValueError(f"Missing the required param: '{param}' in the hardware constants group that contains the backend {self.name}")

    # ── Per-qubit initialization ──────────────────────────────────────

    def _compute_median_T_env(self):
        """Compute T_env per qubit from readout data, return the median."""
        qubit_paths = self.json_manager.get_qubit_paths()
        T_env_estimates = []
        gamma_env_ratio = 1 - self.config.get("ymxc_y0")

        for qb_path in qubit_paths:
            p_e = self.json_manager.find_value_with_units("prob_meas1_prep0", qb_path)
            frequency = self.json_manager.find_value_with_units("frequency", qb_path)
            w_ge = 2 * pi * frequency

            if p_e is None or frequency is None or p_e >= 0.5 or p_e <= 0:
                continue

            n_eff = p_e / (1 - 2 * p_e)

            arg = 1 + gamma_env_ratio / n_eff
            if arg <= 1:
                continue

            T_env_i = hbar * w_ge / (k * log(arg))
            T_env_estimates.append(T_env_i)

        if not T_env_estimates:
            logging.warning("Could not estimate T_env from any qubit. Using init_T as fallback.")
            return self.init_T

        return median(T_env_estimates)

    def _derive_qubit_params(self, qb_path: str) -> dict:
        """Derive all per-qubit physics parameters (Steps 1-9 from parameters.md)."""
        frequency = self.json_manager.find_value_with_units("frequency", qb_path)
        w_ge = 2 * pi * frequency

        # Step 1: E_c from anharmonicity (fallback to global config)
        anharmonicity = self.json_manager.find_value_with_units("anharmonicity", qb_path)
        if anharmonicity is not None:
            E_c = h * abs(anharmonicity)
        else:
            E_c = self.config.get("E_c")
            if E_c is None:
                raise ValueError(f"No anharmonicity data for {qb_path} and no fallback E_c in hardware constants")
            logging.debug(f"Using global E_c fallback for {qb_path}")

        # Step 2: E_J from frequency and E_c
        E_J = (h * frequency + E_c)**2 / (8 * E_c)

        if E_J / E_c < 20:
            logging.warning(f"E_J/E_c = {E_J/E_c:.1f} for {qb_path} — transmon approximation may be inaccurate (expected >= ~30)")

        # Step 3: w_p (plasma frequency)
        w_p = sqrt(8 * E_J * E_c) / hbar

        # Step 4: R_n (normal-state junction resistance)
        delta = self.config.get("delta")
        R_n = pi * hbar * delta / (4 * e**2 * E_J)

        # Step 5: N_e (effective number of junction channels)
        subgap_transparency = self.config.get("subgap_transparency")
        g_k = e**2 / h
        N_e = (1 / subgap_transparency) * 1 / (2 * R_n * g_k)

        # Step 7: T_env (precomputed median)
        T_env = self.median_T_env

        # Step 8: y0 (gamma_1_0) — closed-form from T1_meas
        T1_meas = self.json_manager.find_value_with_units("T1", qb_path)
        ymxc_y0 = self.config.get("ymxc_y0")
        n_BE_init = self.bose_einstein(w_ge, self.init_T)
        n_BE_env = self.bose_einstein(w_ge, T_env)
        n_eff_init = ymxc_y0 * n_BE_init + (1 - ymxc_y0) * n_BE_env
        y0 = 1 / (T1_meas * (2 * n_eff_init + 1))

        # Step 9: gamma_psi_base — closed-form from T1 and T2
        T2_meas = self.json_manager.find_value_with_units("T2", qb_path)
        gamma_psi_base = max(0, 1 / T2_meas - 1 / (2 * T1_meas))

        qp = {
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

        logging.debug(f"Derived params for {qb_path}: E_J/E_c={E_J/E_c:.1f}, R_n={R_n:.0f} Ohm, y0={y0:.1f}, gamma_psi_base={gamma_psi_base:.1f}")
        return qp

    def initialize_per_qubit_params(self):
        """Compute median T_env, then derive all per-qubit parameters."""
        self.median_T_env = self._compute_median_T_env()
        logging.debug(f"Median T_env = {self.median_T_env}")

        self._qubit_params_cache = {}
        for qb_path in self.json_manager.get_qubit_paths():
            self._qubit_params_cache[qb_path] = self._derive_qubit_params(qb_path)

    def get_qubit_params(self, qb_path: str) -> dict:
        return self._qubit_params_cache[qb_path]

    # ── Physics functions (temperature-dependent) ─────────────────────

    def x_qp(self, T: float):
        """Equilibrium quasiparticle density"""
        delta = self.config.get("delta")
        x_qp = sqrt(2*pi*k*T/delta) * exp(-delta/(k*T))
        return x_qp

    def bose_einstein(self, w: float, T: float):
        """Bose-Einstein distribution"""
        x = (hbar*w) / (k * T)
        if x > 700:  # Prevent overflow
            return 0.0
        return 1.0 / (exp(x) - 1.0)

    def n_eff(self, T: float, qp: dict):
        """Effective photon number from two-bath model (Paper 1, Eq. 14)"""
        w_ge = qp["w_ge"]
        gamma_MXC_ratio = self.config.get("ymxc_y0")
        T_env = qp["T_env"]

        n_MXC = self.bose_einstein(w_ge, T)
        n_env = self.bose_einstein(w_ge, T_env)

        n_eff = gamma_MXC_ratio * n_MXC + (1 - gamma_MXC_ratio) * n_env
        return n_eff

    def gamma_qp(self, T: float, qp: dict):
        """Quasiparticle-induced relaxation rate (Paper 1, Eq. 26)"""
        delta = self.config.get("delta")
        w_ge = qp["w_ge"]
        w_p = qp["w_p"]

        x_qp = self.x_qp(T)

        # First term: x_qp * sqrt(2Δ/ℏω_ge)
        term1 = x_qp * sqrt(2*delta/(hbar*w_ge))

        # Second term: 4*exp(-Δ/k_B T) * cosh(ℏω_ge/(2k_B T)) * K_0(ℏω_ge/(2k_B T))
        arg = hbar*w_ge/(2*k*T)
        if arg > 700:  # Prevent overflow
            term2 = 0.0
        else:
            term2 = 4*exp(-delta/(k*T)) * cosh(arg) * k0(arg)

        gamma_qp = (w_p**2)/(pi*w_ge) * (term1 + term2)
        return gamma_qp

    def T1(self, T: float, qp: dict) -> float:
        """Energy relaxation time (Paper 1, Eq. 27)"""
        gamma_qp = self.gamma_qp(T, qp)
        n_eff = self.n_eff(T, qp)
        y0 = qp["y0"]

        # T1 = 1/[γ_qp(T) + γ_0(2n_eff + 1)]
        T1 = 1/(gamma_qp + y0*(2*n_eff + 1))
        return T1

    def gamma_psi_qp(self, T: float, qp: dict):
        """Quasiparticle-induced dephasing rate (Paper 1, Eq. 29)"""
        w_p = qp["w_p"]
        w_ge = qp["w_ge"]
        N_e = qp["N_e"]
        delta = self.config.get("delta")

        # x_A_qp = exp(-Δ/k_B T) (Andreev state occupation)
        if delta/(k*T) > 700:  # Prevent overflow
            x_A_qp = 0.0
        else:
            x_A_qp = exp(-delta/(k*T))

        gamma_psi_qp = 4*pi*(w_p**2/w_ge) * sqrt(x_A_qp/N_e)
        return gamma_psi_qp

    def T_psi(self, T: float, qp: dict):
        """Pure Dephasing time"""
        gamma_psi_qp = self.gamma_psi_qp(T, qp)
        gamma_psi_base = qp["gamma_psi_base"]
        T_psi = 1/(gamma_psi_base + gamma_psi_qp)
        return T_psi

    def T2(self, T: float, qp: dict) -> float:
        """Total Dephasing time, or T2*"""
        T1 = self.T1(T, qp)
        T_psi = self.T_psi(T, qp)
        T2 = 1/(1/(2*T1) + 1/(T_psi))
        return T2

    def F_N(self, T: float, N: int, gate_length: float, qubit_params_list: list) -> float:
        """Gate fidelity with per-qubit decoherence rates (Simbierowicz et al., PRX Quantum 5, 030302, 2024, Eq. 6)

        F_N = 1 - d*t_gate/(2(d+1)) * sum_i(1/T1_i + 1/T_phi_i), where d = 2^N.
        The N from the homogeneous formula is absorbed by the per-qubit summation.
        """
        d = 2 ** N
        rate_sum = 0
        for qp in qubit_params_list:
            T1_i = self.T1(T, qp)
            T_psi_i = self.T_psi(T, qp)
            rate_sum += 1/T1_i + 1/T_psi_i
        return 1 - (d * gate_length) / (2 * (d + 1)) * rate_sum

    # ── Backend building ──────────────────────────────────────────────

    def calculate_qb_config(self, control_parameters: dict, qb_path: str):
        logging.debug(f"Calculating a qb config now\n")
        T = get_config_value(control_parameters, "temperature")
        qp = self.get_qubit_params(qb_path)

        T1 = self.T1(T, qp)
        logging.debug(f"working_T1: {T1} at temperature {T}")
        T2 = self.T2(T, qp)
        logging.debug(f"working_T2: {T2} at temperature {T}")

        qb_config = {"T1": T1, "T2": T2}
        logging.debug(f"ending qb calculations with the following config: {qb_config}")

        self.config_tracker.add_config({"config": qb_config}, "qb")
        return qb_config

    def calculate_gate_config(self, control_parameters: dict, gate_path: str):
        logging.debug(f"Calculating a gate config now\n")
        gate_param_path = gate_path + "parameters."

        relevant_qubits = self.json_manager.resolve(gate_path + "qubits")
        N = len(relevant_qubits)
        T = get_config_value(control_parameters, "temperature")
        gate_length = self.json_manager.find_value_with_units("gate_length", gate_param_path)
        logging.debug(f"gate_length: {gate_length}")

        qubit_params_list = [self.get_qubit_params(f"qubits.[{qb}].") for qb in relevant_qubits]

        gate_error = max(0, min(1, 1 - self.F_N(T, N, gate_length, qubit_params_list)))
        logging.debug(f"gate_error: {gate_error} at temperature {T}")

        gate_config = {"gate_error": gate_error}
        self.config_tracker.add_config({"config": gate_config}, "gate")
        return gate_config
