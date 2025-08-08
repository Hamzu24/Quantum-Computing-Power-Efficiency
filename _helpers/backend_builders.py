from typing import Protocol, Any
from pathlib import *
import json
from math import sqrt, exp, cosh
from scipy.constants import pi, k, hbar
from scipy.special import k0
import argparse
import pprint
import os
from _helpers.json_manager import json_manager, get_unit_multiplier, SI_PREFIXES, PROPERTY_UNITS
import logging
import time

Papers = ["""Lvov, D. S., Lemziakov, S. A., Ankerhold, E., Peltonen, J. T., & Pekola, J. P. (2025).
          Thermometry based on a superconducting qubit. Physical Review Applied, 23(5). https://doi.org/10.1103/physrevapplied.23.054079""",
          """Simbierowicz, S., Borrelli, M., Monarkha, V., Nuutinen, V., & Lake, R. E. (2024).
          Inherent Thermal-Noise problem in addressing Qubits. PRX Quantum, 5(3). https://doi.org/10.1103/prxquantum.5.030302"""]

HARDWARE_CONFIG_PATH = Path("qiskit_backend_configs/hardware_constants.json")

hardware_config_groups = {
    "modern": ("perth", "default"),
    "intermediate": (),
    "legacy": ("oslo")
}

def get_unit_multiplier(unit: str):
    assert (len(unit) != 0)
    return SI_PREFIXES.get(unit[0])

def get_config_value(config: dict, name: str):
    val, unit = config.get(name)
    multiplier = get_unit_multiplier(unit)
    return val * multiplier

class builder(Protocol):
    def T1(self, *args: Any, **kwargs: Any) -> float:
        "calculate T1 relaxation time from the hardware constants and temperature passed in"

    def T2(self, *args: Any, **kwargs: Any) -> float:
        "calculate T2 dephasing time from the hardware constants and temperature passed in"

    def calculate_qb_config(self, control_parameters: dict, qb_path: str, jm: json_manager):
        "build the backend configuration from the hardware constants and temperature passed in"

    def calculate_gate_config(self, control_parameters: dict, qb_path: str, jm: json_manager):
        "Build a gate configuration from the control parameters and a specific gate dictionary"
    

class builder_wrapper:
    def __init__(self, name: str):
        self.name = name
        self.load_hardware_params(HARDWARE_CONFIG_PATH)
        
        builder_name = self.config["builder_class"]
        self.builder = globals()[builder_name](self.name, self.config)
        filename = f"qiskit_backend_configs/{name}/props_{name}.json"
        self.json_manager = json_manager(filename)
    
    def load_hardware_params(self, path: str):
        with open(path, "r") as f:
            configs = json.load(f)

        items = hardware_config_groups.items()
        group = None

        # Finding the group of the name or of the default
        for group_name, group_contents in items:
            if self.name in group_contents:
                group = group_name
                break
            elif "default" in group_contents:
                group = group_name 
        
        if group is None:
            raise ValueError("Neither the name nor default was found in a group! Please check the hardware_config_groups variable in the _helpers/backend_builders file.")
            
        self.config = configs.get(group)
        if self.config is None:
            raise ValueError(f"Error: the group {group} does not have a corresponding configuration in the {str(HARDWARE_CONFIG_PATH)} file.")
    
    def build_backend(self, control_parameters):
        self._build_qubits(control_parameters)
        self._build_gates(control_parameters)
        self.json_manager.write()

    def _build_qubits(self, control_parameters):
        qubit_paths = self.json_manager.get_qubit_paths()

        for qb_path in qubit_paths:
            qb_config = self.builder.calculate_qb_config(control_parameters, qb_path, self.json_manager)
            for prop, val in qb_config.items():
                self.json_manager.update_with_units(prop, val, qb_path)
    
    def _build_gates(self, control_parameters):
        gate_paths = self.json_manager.get_gate_paths()

        for gate_path in gate_paths:
            gate_config = self.builder.calculate_gate_config(control_parameters, gate_path, self.json_manager)
            gate_param_path = gate_path + "parameters."
            for prop, val in gate_config.items():
                if val is not None:
                    self.json_manager.update_with_units(prop, val, gate_param_path)

class default_builder:
    def __init__(self, name: str, config: dict, init_T=0.013):
        self.name = name
        self.config = config
        self.init_T=init_T
        self.calculated_values = {}
        self.is_valid()

    def is_valid(self):
        required_params = [
            "y0", "ymxc_y0", "T_env", "gamma_psi_base", 
            "N_e", "delta", "E_c", "E_J"
        ]
        
        for param in required_params:
            if self.config.get(param) is None:
                raise ValueError(f"Missing the required param: '{param}' in the hardware constants group that contains the backend {self.name}")

    def x_qp(self, T: float):
        """Equilibrium quasiparticle density"""

        delta = self.config.get("delta")
        x_qp = sqrt(2*pi*k*T/delta) * exp(-delta/(k*T))
        logging.info(f"Calculated value of x_qp at ({T}): {x_qp}")

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

        logging.info(f"Calculated value of w_p at ({T}): {w_p}")

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
        logging.info(f"Calculated value of n_eff at ({T}, {frequency}): {n_eff}")

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
        logging.info(f"Calculated value of gamma_qp at ({T}, {frequency}): {gamma_qp}")
        
        return gamma_qp
    
    def T1(self, T: float, frequency: float) -> float:
        """Energy relaxation time (Paper 1, Eq. 27)"""
        gamma_qp = self.gamma_qp(T, frequency)
        n_eff = self.n_eff(T, frequency)
        y0 = self.config.get("y0")
        
        # T1 = 1/[γ_qp(T) + γ_0(2n_eff + 1)]
        T1 = 1/(gamma_qp + y0*(2*n_eff + 1))
        logging.info(f"Calculated value of T1 at ({T}, {frequency}): {T1}")
        
        return T1
    
    def gamma_psi_qp(self, T: float, frequency: float):
        """Quasiparticle-induced dephasing rate (Paper 1, Eq. 29)"""
        w_p = self.w_p(T)
        w_ge = frequency
        delta = self.config.get("delta")
        N_e = self.config.get("N_e")
        
        # x_A_qp = exp(-Δ/k_B T) (Andreev state occupation)
        if delta/(k*T) > 700:  # Prevent overflow
            x_A_qp = 0.0
        else:
            x_A_qp = exp(-delta/(k*T))
        
        gamma_psi_qp = 4*pi*(w_p**2/w_ge) * sqrt(x_A_qp/N_e)
        logging.info(f"Calculated value of gamma_psi_qp at ({T}, {frequency}): {gamma_psi_qp}")
        
        return gamma_psi_qp
    
    def T2(self, T: float, frequency: float) -> float:
        """Total Dephasing time, or T2*"""
        T1 = self.T1(T, frequency)
        gamma_psi_qp = self.gamma_psi_qp(T, frequency)
        gamma_psi_base = self.config.get("gamma_psi_base")
        
        # T2 = 1/(1/(2T1) + γ_φ_base + γ_φ_qp)
        T_psi = self.T_psi(T, frequency)
        T2 = 1/(1/(2*T1) + T_psi)
        logging.info(f"Calculated value of T2 at ({T}, {frequency}): {T2}")
        
        return T2
    
    def T_psi(self, T: float, frequency: float):
        """Pure Dephasing time"""
        gamma_psi_qp = self.gamma_psi_qp(T, frequency)
        gamma_psi_base = self.config.get("gamma_psi_base")
        T_psi = gamma_psi_base + gamma_psi_qp
        logging.info(f"Calculated value of T_psi at ({T}, {frequency}): {T_psi}")

        return T_psi
    
    def F_N(self, T: float, N: int, gate_length: float, frequency: float) -> float:
        """Gate fidelity when only accounting for thermal sources of error (Paper 2, Eq. 6)"""
        T1 = self.T1(T, frequency)
        T_psi = self.T_psi(T, frequency)
        d = 2 ** N  # Dimension of Hilbert space

        print(f"All vals: T1:{T1},T_psi:{T_psi},d:{d},N:{N},gate_length:{gate_length}")
        F_N = 1 - (d*N*gate_length)/(2*(d+1)) * (1/T1 + 1/T_psi)
        #return min(max(0, F_N), 1)
        logging.info(f"Calculated value of F_N at ({T}, {frequency}): {F_N}")

        return F_N

    def calculate_qb_config(self, control_parameters: dict, qb_path: str, jm: json_manager):
        logging.info(f"Calculating a qubit config now\n")
        T = get_config_value(control_parameters, "temperature")

        frequency = jm.find_value_with_units("frequency", qb_path)
        logging.info(f"frequency: {frequency}")

        actual_T1 = jm.find_value_with_units("T1", qb_path)
        logging.info(f"actual_T1: {actual_T1}")
        logging.info(f"calculating T1 for the initial temperature of {self.init_T} now")
        calculated_T1 = self.T1(self.init_T, frequency)
        adj_T1 = actual_T1 / calculated_T1
        logging.info(f"calculated_T1: {calculated_T1}")
        logging.info(f"adj_T1: {adj_T1}")

        actual_T2 = jm.find_value_with_units("T2", qb_path)
        calculated_T2 = self.T2(self.init_T, frequency)
        logging.info(f"actual_T2: {actual_T2}")
        logging.info(f"calculating T2 for the initial temperature of {self.init_T} now")
        adj_T2 = actual_T2 / calculated_T2
        logging.info(f"calculated_T2: {calculated_T2}")
        logging.info(f"adj_T2: {adj_T2}")

        logging.info(f"calculating T2 for the working temperature of {T} now")
        T1 = self.T1(T, frequency) * adj_T1
        logging.info(f"working_T1: {T1}")
        logging.info(f"calculating T2 for the working temperature of {T} now")
        T2 = self.T2(T, frequency) * adj_T2
        logging.info(f"working_T2: {T2}")

        qb_config = {"T1": T1, "T2": T2}
        logging.info(f"ending qb calculations with the following config: working_T1: {T1}, working_T2: {T2}")
        return qb_config
    
    def calculate_gate_config(self, control_parameters: dict, gate_path: str, jm: json_manager):
        logging.info(f"Calculating a gate config now\n")
        gate_param_path = gate_path + "parameters."

        relevant_qubits = jm.resolve(gate_path + "qubits")
        N = len(relevant_qubits)
        freq_sum = 0
        for qb_num in relevant_qubits:
            qb_path = f"qubits.[{qb_num}]."
            freq = jm.find_value_with_units("frequency", qb_path)
            freq_sum += freq
        
        avg_frequency = freq_sum / N
        logging.info(f"avg_frequency: {avg_frequency}")

        logging.info(f"Now looking for gate error from {gate_param_path}")
        actual_gate_error = jm.find_value_with_units("gate_error", gate_param_path)
        gate_length = jm.find_value_with_units("gate_length", gate_param_path)
        logging.info(f"actual gate error: {actual_gate_error}")
        logging.info(f"gate_length: {gate_length}")

        logging.info(f"calculating gate error for the initial temperature of {self.init_T} now")
        calculated_error = 1 - self.F_N(self.init_T, N, gate_length, avg_frequency)
        if actual_gate_error:
            adjustement = calculated_error - actual_gate_error
            logging.info(f"calculated_error: {calculated_error}")
            logging.info(f"adjustement: {adjustement}")

        working_T = get_config_value(control_parameters, "temperature")
        logging.info(f"calculating working gate error for the working temperature of {working_T} now")
            
        if actual_gate_error:
            working_gate_error = (1-self.F_N(working_T, N, gate_length, avg_frequency)) - adjustement
        else:
            working_gate_error = None
        logging.info(f"working_gate_error: {working_gate_error}")
        logging.info(f"ending gate calculations with the following config: gate_error: {working_gate_error}")

        return {"gate_error": working_gate_error}


backend_name = "perth"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('temperature', type=float, help='Temperature value')
    args = parser.parse_args()
    
    builder = builder_wrapper(backend_name)
    builder.build_backend({"temperature": args.temperature})

    #backend_config = {"T1": 100, "T2": 150}
    #output_config_path = f"qiskit_backend_configs/{backend_name}/props_{backend_name}.json"
    #for property, value in backend_config.items():
    #    paths = get_qubit_paths(output_config_path, property, "value")
    #    for path in paths:
    #        update_nested_json(output_config_path, path, value)
