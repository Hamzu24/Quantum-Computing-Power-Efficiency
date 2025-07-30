from typing import Protocol
from pathlib import *
import json
from math import sqrt, exp, cosh
from scipy.constants import pi, k, hbar
from scipy.special import k0
import argparse
import pprint
import os

HARDWARE_CONFIG_PATH = Path("qiskit_backend_configs/hardware_constants.json")

hardware_config_groups = {
    "modern": ("perth", "default"),
    "intermediate": (),
    "legacy": ("oslo")
}

SI_PREFIXES = {
    'P': 1e15,   # peta
    'T': 1e12,   # tera
    'G': 1e9,    # giga
    'M': 1e6,    # mega
    'k': 1e3,    # kilo
    '': 1,       # base unit
    'm': 1e-3,   # milli
    'μ': 1e-6,   # micro (Greek mu)
    'u': 1e-6,   # micro (alternative 'u' for systems that don't support μ)
    'n': 1e-9,   # nano
}

PROPERTY_UNITS = {
    "T1": "us",
    "T2": "us", 
    "frequency": "GHz",
    "anharmonicity": "GHz",
    "readout_error": "",
    "prob_meas0_prep1": "",
    "prob_meas1_prep0": "",
    "readout_length": "ns"
}

def retrieve_value_with_units(config: dict, name: str):
    val, units = config.get(name)
    multiplier = SI_PREFIXES.get(units[0])
    return val * multiplier

def update_nested_json(filename, path, value, create_path=True):
    """Update nested JSON values using dot notation"""
    with open(filename, 'r') as file:
        data = json.load(file)
    
    # Navigate to nested location
    keys = path.split('.')
    current = data
    for key in keys[:-1]:
        if isinstance(current, list):
            idx = int(key[1:-1])
            current = current[idx]
            continue
                
        if key not in current:
            if create_path:
                current[key] = {}
            else:
                raise ValueError("Error: specified path does not exist, and the create_path flag is set to False")
        current = current[key]
    
    # Set the value
    current[keys[-1]] = value
    
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def get_qubit_paths(filename: str, prop: str, target_str: str, all_qubits=True):
    with open(filename, 'r') as file:
        data = json.load(file)
    q_list = data.get("qubits")
    if not q_list:
        raise ValueError("configuration file does not match expected structure: qubits list not found")

    paths = []
    cur_path = "qubits."

    for i, qb in enumerate(q_list):
        cur_path = cur_path + f"[{i}]."
        found = False

        for j, qb_prop in enumerate(qb):
            cur_path = cur_path + f"[{j}]."

            cur_prop = qb_prop.get("name")
            if cur_prop == prop:
                paths.append(cur_path + target_str)
                found = True
                break

            cur_path = ".".join(cur_path.split(".")[:-2]) + "."

        if not found and all_qubits:
            raise ValueError(f"configuration file does not match expected structure: one qubit didn't have the property {prop}")
        found = False
        cur_path = ".".join(cur_path.split(".")[:-3]) + "."

    return paths


class builder(Protocol):
    def T1(self, T: float) -> float:
        "calculate T1 relaxation time from the hardware constants and temperature passed in"

    def T2(self, T: float) -> float:
        "calculate T2 dephasing time from the hardware constants and temperature passed in"

    def calculate_config(self, control_parameters: dict):
        "build the backend configuration from the hardware constants and temperature passed in"
    

class builder_wrapper:
    def __init__(self, name: str):
        self.name = name
        self.load_hardware_params(HARDWARE_CONFIG_PATH)
        
        builder_name = self.config["builder_class"]
        self.builder = globals()[builder_name](self.name, self.config)
    
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
        backend_config = self.builder.calculate_config(control_parameters)

        output_config_path = f"qiskit_backend_configs/{backend_name}/props_{backend_name}.json"
        for property, value in backend_config.items():

            property_unit = PROPERTY_UNITS.get(property)
            if property_unit is None or property_unit == "":
                # Leave units unchanged in this case
                return
            
            unit_paths = get_qubit_paths(output_config_path, property, "unit")
            for unit_path in unit_paths:
                update_nested_json(output_config_path, property, property_unit)

            unit_multiplier = SI_PREFIXES.get(property_unit[0])
            scaled_value = value / unit_multiplier
            paths = get_qubit_paths(output_config_path, property, "value")
            for path in paths:
                update_nested_json(output_config_path, path, scaled_value)


class default_builder:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.calculated_values = {}
        self.is_valid()

    def is_valid(self):
        required_params = [
            "y0", "ymxc_y0", "T_env", "gamma_psi_base", 
            "N_e", "delta", "w_ge", "E_c", "E_J"
        ]
        
        for param in required_params:
            if self.config.get(param) is None:
                raise ValueError(f"Missing the required param: '{param}' in the hardware constants group that contains the backend {self.name}")

    def x_qp(self, T: float):
        """Equilibrium quasiparticle density"""

        delta = self.config.get("delta")
        x_qp = sqrt(2*pi*k*T/delta) * exp(-delta/(k*T))
        self.calculated_values["x_qp"] = x_qp
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

        self.calculated_values["w_p"] = w_p
        return w_p

    def bose_einstein(self, w: float, T: float):
        """Bose-Einstein distribution"""
        x = (hbar*w) / (k * T)
        if x > 700:  # Prevent overflow
            return 0.0
        return 1.0 / (exp(x) - 1.0)
    
    def n_eff(self, T: float):
        """Effective photon number from two-bath model (Eq. 14)"""
        w_ge = self.config.get("w_ge")
        gamma_MXC_ratio = self.config.get("ymxc_y0")  # γ_MXC/γ_0
        T_env = self.config.get("T_env")
        
        n_MXC = self.bose_einstein(w_ge, T)
        n_env = self.bose_einstein(w_ge, T_env)
        
        n_eff = gamma_MXC_ratio * n_MXC + (1 - gamma_MXC_ratio) * n_env
        
        self.calculated_values["n_eff"] = n_eff
        return n_eff
    
    def gamma_qp(self, T: float):
        """Quasiparticle-induced relaxation rate (Eq. 26)"""
        delta = self.config.get("delta")
        w_ge = self.config.get("w_ge")
        
        x_qp = self.calculated_values.get("x_qp") if self.calculated_values.get("x_qp") else self.x_qp(T)
        w_p = self.calculated_values.get("w_p") if self.calculated_values.get("w_p") else self.w_p(T)

        # First term: x_qp * sqrt(2Δ/ℏω_ge)
        term1 = x_qp * sqrt(2*delta/(hbar*w_ge))
        
        # Second term: 4*exp(-Δ/k_B T) * cosh(ℏω_ge/(2k_B T)) * K_0(ℏω_ge/(2k_B T))
        arg = hbar*w_ge/(2*k*T)
        if arg > 700:  # Prevent overflow
            term2 = 0.0
        else:
            term2 = 4*exp(-delta/(k*T)) * cosh(arg) * k0(arg)
        
        gamma_qp = (w_p**2)/(pi*w_ge) * (term1 + term2)
        
        self.calculated_values["gamma_qp"] = gamma_qp
        return gamma_qp
    
    def T1(self, T: float) -> float:
        """Energy relaxation time (Eq. 27)"""
        gamma_qp = self.calculated_values.get("gamma_qp") if self.calculated_values.get("gamma_qp") else self.gamma_qp(T)
        n_eff = self.calculated_values.get("n_eff") if self.calculated_values.get("n_eff") else self.n_eff(T)
        y0 = self.config.get("y0")
        
        # T1 = 1/[γ_qp(T) + γ_0(2n_eff + 1)]
        T1 = 1/(gamma_qp + y0*(2*n_eff + 1))
        
        self.calculated_values["T1"] = T1
        return T1
    
    def gamma_psi_qp(self, T: float):
        """Quasiparticle-induced dephasing rate (Eq. 29)"""
        w_p = self.calculated_values.get("w_p") if self.calculated_values.get("w_p") else self.w_p(T)
        w_ge = self.config.get("w_ge")
        delta = self.config.get("delta")
        N_e = self.config.get("N_e")
        
        # x_A_qp = exp(-Δ/k_B T) (Andreev state occupation)
        if delta/(k*T) > 700:  # Prevent overflow
            x_A_qp = 0.0
        else:
            x_A_qp = exp(-delta/(k*T))
        
        gamma_psi_qp = 4*pi*(w_p**2/w_ge) * sqrt(x_A_qp/N_e)
        
        self.calculated_values["gamma_psi_qp"] = gamma_psi_qp
        return gamma_psi_qp
    
    def T2(self, T: float) -> float:
        """Dephasing time"""
        T1 = self.calculated_values.get("T1") if self.calculated_values.get("T1") else self.T1(T)
        gamma_psi_qp = self.calculated_values.get("gamma_psi_qp") if self.calculated_values.get("gamma_psi_qp") else self.gamma_psi_qp(T)
        gamma_psi_base = self.config.get("gamma_psi_base")
        
        # T2 = 1/(1/(2T1) + γ_φ_base + γ_φ_qp)
        T2 = 1/(1/(2*T1) + gamma_psi_base + gamma_psi_qp)
        
        self.calculated_values["T2"] = T2
        return T2

    def calculate_config(self, control_parameters: dict):
        DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'
        T = retrieve_value_with_units(control_parameters, "temperature")
        T1 = self.calculated_values.get("T1") if self.calculated_values.get("T1") else self.T1(T)
        T2 = self.calculated_values.get("T2") if self.calculated_values.get("T2") else self.T2(T)
        if DEBUG:
            print(f"At temperature {T}\nT1: {T1}, T2: {T2}")
            print(f"debug output:\n")
            pprint.pprint(self.calculated_values)

        return {"T1": T1, "T2": T2}
        


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