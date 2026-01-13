from _helpers.builders.base import builder_registry, ConfigTracker
import logging
from copy import deepcopy
from _helpers.json_manager import JsonManager
from math import sqrt, exp, cosh
from scipy.constants import pi, k, hbar, e, h
from scipy.special import k0
from _helpers.helpers import get_config_value
from _helpers.registry import optimiser_registry
from scipy.optimize import minimize_scalar

class DefaultBuilder:
    registry_name = "DefaultBuilder"
    def __init__(self, name: str, config: dict, jm: JsonManager, init_control_parameters: dict={"temperature": [13, "mK"]}):
        self.name = name
        self.config = config
        self.init_T=get_config_value(init_control_parameters, "temperature")
        if self.init_T is None:
            raise ValueError("The expected key 'temperature' is not present in the control parameters dictionary!")
        self.calculated_values = {}
        self.is_valid()
        self.json_manager = jm
        self.config_tracker = ConfigTracker()

    def is_valid(self):
        required_params = [
            "y0", "ymxc_y0", "T_env", "gamma_psi_base", 
            "delta", "E_c", "E_J", "R_n", "subgap_transparency"
        ]
        
        for param in required_params:
            if self.config.get(param) is None:
                raise ValueError(f"Missing the required param: '{param}' in the hardware constants group that contains the backend {self.name}")

    def x_qp(self, T: float):
        """Equilibrium quasiparticle density"""

        delta = self.config.get("delta")
        x_qp = sqrt(2*pi*k*T/delta) * exp(-delta/(k*T))

        return x_qp
    
    def w_p(self, T: float):
        """Plasma frequency with temperature-dependent screening"""
        E_J = self.config.get("E_J")
        E_c = self.config.get("E_c")
        
        # Base plasma frequency: ω_p = √(8E_J E_c)/ℏ
        w_p0 = sqrt(8 * E_J * E_c) / hbar
        
        # Temperature-dependent screening factor
        delta = self.config.get("delta")
        screening_factor = 1 - 2*sqrt(2*pi*k*T/delta) * exp(-delta/(k*T))
        
        if screening_factor <= 0:
            w_p = w_p0 * 0.4  # Fallback value
        else:
            w_p = w_p0 * sqrt(screening_factor)


        return w_p

    def bose_einstein(self, w: float, T: float):
        """Bose-Einstein distribution"""
        x = (hbar*w) / (k * T)
        if x > 700:  # Prevent overflow
            return 0.0
        return 1.0 / (exp(x) - 1.0)
    
    def n_eff(self, T: float, frequency: float):
        """Effective photon number from two-bath model (Paper 1, Eq. 14)"""
        w_ge = frequency
        gamma_MXC_ratio = self.config.get("ymxc_y0")  # γ_MXC/γ_0
        T_env = self.config.get("T_env")
        
        n_MXC = self.bose_einstein(w_ge, T)
        n_env = self.bose_einstein(w_ge, T_env)
        
        n_eff = gamma_MXC_ratio * n_MXC + (1 - gamma_MXC_ratio) * n_env

        return n_eff
    
    def gamma_qp(self, T: float, frequency: float):
        """Quasiparticle-induced relaxation rate (Paper 1, Eq. 26)"""
        delta = self.config.get("delta")
        w_ge = frequency
        
        x_qp = self.x_qp(T)
        w_p = self.w_p(T)

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
    
    def T1(self, T: float, frequency: float) -> float:
        """Energy relaxation time (Paper 1, Eq. 27)"""
        gamma_qp = self.gamma_qp(T, frequency)
        n_eff = self.n_eff(T, frequency)
        y0 = self.config.get("y0")
        
        # T1 = 1/[γ_qp(T) + γ_0(2n_eff + 1)]
        T1 = 1/(gamma_qp + y0*(2*n_eff + 1))
        
        return T1
    
    def gamma_psi_qp(self, T: float, frequency: float):
        """Quasiparticle-induced dephasing rate (Paper 1, Eq. 29)"""
        w_p = self.w_p(T)
        w_ge = frequency
        delta = self.config.get("delta")
        R_n = self.config.get("R_n")
        subgap_transparency = self.config.get("subgap_transparency")
        g_t = 1/R_n
        g_k = (e**2) / h
        N_e = 1/(subgap_transparency) * (g_t/(2*g_k))
        
        # x_A_qp = exp(-Δ/k_B T) (Andreev state occupation)
        if delta/(k*T) > 700:  # Prevent overflow
            x_A_qp = 0.0
        else:
            x_A_qp = exp(-delta/(k*T))
        
        gamma_psi_qp = 4*pi*(w_p**2/w_ge) * sqrt(x_A_qp/N_e)
        
        return gamma_psi_qp
    
    def T2(self, T: float, frequency: float) -> float:
        """Total Dephasing time, or T2*"""
        T1 = self.T1(T, frequency)
        gamma_psi_qp = self.gamma_psi_qp(T, frequency)
        gamma_psi_base = self.config.get("gamma_psi_base")
        
        # T2 = 1/(1/(2T1) + γ_φ_base + γ_φ_qp)
        T_psi = self.T_psi(T, frequency)
        T2 = 1/(1/(2*T1) + 1/(T_psi))
        
        return T2
    
    def T_psi(self, T: float, frequency: float):
        """Pure Dephasing time"""
        gamma_psi_qp = self.gamma_psi_qp(T, frequency)
        gamma_psi_base = self.config.get("gamma_psi_base")
        T_psi = 1/(gamma_psi_base + gamma_psi_qp)

        return T_psi
    
    def F_N(self, T: float, N: int, gate_length: float, frequency: float) -> float:
        """Gate fidelity when only accounting for thermal sources of error (Paper 2, Eq. 6)"""
        T1 = self.T1(T, frequency)
        T_psi = self.T_psi(T, frequency)
        d = 2 ** N  # Dimension of Hilbert space

        F_N = 1 - (d*N*gate_length)/(2*(d+1)) * (1/T1 + 1/T_psi)
        #return min(max(0, F_N), 1)

        return F_N
    
    def calculate_parameter_error(self, T: float, qb_path: str, type="ratio", parameter="T1", override_config: dict = None):
        if override_config is not None:
            original_config = deepcopy(self.config)
            for key, val in override_config.items():
                self.config[key] = val
            self.is_valid()
        
        frequency = self.json_manager.find_value_with_units("frequency", qb_path)

        actual_parameter = self.json_manager.find_value_with_units(parameter, qb_path)
        if parameter == "T1":
            calculated_parameter = self.T1(T, frequency)
        elif parameter == "T2":
            calculated_parameter = self.T2(T, frequency)
        else:
            raise ValueError(f"Invalid parameter type of {parameter} when trying to calculate a qubit parameter error")

        logging.debug(f"calculated_{parameter}: {calculated_parameter} at temperature {T}")
        if type == "ratio":
            error =     actual_parameter / calculated_parameter
        elif type == "square":
            error = (actual_parameter - calculated_parameter) ** 2
        elif type == "relative":
            error = abs(calculated_parameter - actual_parameter) / abs(actual_parameter)
        else:
            raise ValueError(f"Invalid error type of {type} when trying to calculate a qubit error for parameter {parameter}")
        logging.debug(f"error (of type {type}): {error}")

        if override_config is not None:
            self.config = original_config
        
        return error, actual_parameter, calculated_parameter

    def calculate_qb_config(self, control_parameters: dict, qb_path: str):
        logging.debug(f"Calculating a qb config now\n")
        adj_T1, actual_T1, calculated_T1 = self.calculate_parameter_error(self.init_T, qb_path, "ratio", "T1")
        adj_T2, actual_T2, calculated_T2 = self.calculate_parameter_error(self.init_T, qb_path, "ratio", "T2")

        T = get_config_value(control_parameters, "temperature")
        frequency = self.json_manager.find_value_with_units("frequency", qb_path)

        T1 = self.T1(T, frequency) * adj_T1
        logging.debug(f"working_T1: {T1} at temperature {T}")
        T2 = self.T2(T, frequency) * adj_T2
        logging.debug(f"working_T2: {T2} at temperature {T}")

        qb_config = {"T1": T1, "T2": T2}
        logging.debug(f"ending qb calculations with the following config: {qb_config}")

        self.config_tracker.add_config({"config": qb_config, "actual": {"T1": actual_T1, "T2": actual_T2}, "calculated": {"T1": calculated_T1, "T2": calculated_T2}, "adjs": {"T1": adj_T1, "T2": adj_T2}}, "qb")
        return qb_config
    
    def calculate_gate_config(self, control_parameters: dict, gate_path: str):
        logging.debug(f"Calculating a gate config now\n")
        gate_param_path = gate_path + "parameters."

        relevant_qubits = self.json_manager.resolve(gate_path + "qubits")
        N = len(relevant_qubits)
        freq_sum = 0
        for qb_num in relevant_qubits:
            qb_path = f"qubits.[{qb_num}]."
            freq = self.json_manager.find_value_with_units("frequency", qb_path)
            freq_sum += freq
        
        avg_frequency = freq_sum / N

        actual_gate_error = self.json_manager.find_value_with_units("gate_error", gate_param_path)
        gate_length = self.json_manager.find_value_with_units("gate_length", gate_param_path)
        logging.debug(f"gate_length: {gate_length}")

        calculated_error = 1 - self.F_N(self.init_T, N, gate_length, avg_frequency)
        if actual_gate_error:
            adjustement = calculated_error - actual_gate_error
            logging.debug(f"for initial temperature of {self.init_T}:\n actual_error: {actual_gate_error}, calculated_error: {calculated_error}, gate adjustement: {adjustement}")

        working_T = get_config_value(control_parameters, "temperature")
            
        if actual_gate_error:
            working_gate_error = max(0, min(1, (1-self.F_N(working_T, N, gate_length, avg_frequency)) - adjustement))
        else:
            working_gate_error = None
        logging.debug(f"working_gate_error: {working_gate_error} at temperature {working_T}")
        logging.debug(f"ending gate calculations with the following config: gate_error: {working_gate_error}")

        gate_config = {"gate_error": working_gate_error}
        if actual_gate_error:
            self.config_tracker.add_config({"config": gate_config, "actual": actual_gate_error, "calculated": calculated_error, "adjs": adjustement}, "gate")
        return gate_config

    def total_init_parameter_error(self, override_config: dict, parameter: str): 
        qubit_paths = self.json_manager.get_qubit_paths()

        sum = 0
        num_qb = 0
        for qb_path in qubit_paths:
            error, _, _ = self.calculate_parameter_error(self.init_T, qb_path, "relative", parameter, override_config)
            sum += error
            num_qb += 1
        
        sum /= num_qb
        return sum
    
    def optimise_parameters(self):
        existing_optimisation = optimiser_registry.get_optimisation(self.name, self.__class__)
        if existing_optimisation is not None:
            self.config = deepcopy(existing_optimisation)
            return
            
        logging.debug("\nNow optimising y0")
        optimiser_y0 = lambda cur_y0: logging.debug(f"now trying y: {cur_y0}") or self.total_init_parameter_error({"y0": cur_y0}, "T1")
        res = minimize_scalar(optimiser_y0, method='brent', options={'maxiter': 100, 'xtol': 0.0001})
        optimal_y0 = res.x
        logging.debug(f"optimal y0 was found to be {optimal_y0}")
        self.config["y0"] = optimal_y0

        logging.debug("\nNow optimising y_phi_b")
        optimiser_y_phi_b = lambda cur_y_phi_b: logging.debug(f"now trying y_phi_b: {cur_y_phi_b}") or self.total_init_parameter_error({"gamma_psi_base": cur_y_phi_b}, "T2")
        res = minimize_scalar(optimiser_y_phi_b, options={'maxiter': 100, 'xtol': 0.0001})
        optimal_y_phi_b = res.x
        logging.debug(f"optimal y_phi_b was found to be {optimal_y_phi_b}")
        self.config["gamma_psi_base"] = optimal_y_phi_b
        logging.debug("\nSucessfully optimised both parameters\n")

        optimiser_registry.store_optimisation(self.name, self.__class__, self.config)

builder_registry.register_builder(DefaultBuilder)
