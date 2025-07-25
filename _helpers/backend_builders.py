from typing import Protocol
from pathlib import *
import json
from math import sqrt, exp, cosh
from scipy.constants import pi, k, hbar
from scipy.special import k0

HARDWARE_CONFIG_PATH = Path("../qiskit_backend_configs/hardware_constants.json")
hardware_config_groups = {
    "modern": ("perth", "default"),
    "intermediate": (),
    "legacy": ("oslo")
}

class builder(Protocol):
    def T1(self, T: float) -> float:
        "calculate T1 relaxation time from the hardware constants and temperature passed in"

    def T2(self, T: float) -> float:
        "calculate T2 dephasing time from the hardware constants and temperature passed in"


class builer_wrapper:
    def __init__(self, name: str):
        self.name = name
        self.load_hardware_params(HARDWARE_CONFIG_PATH)
        
        builder_name = self.config["builder_class"]
        self.builder = globals()[builder_name]()
    
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
            raise ValueError("Neither the name nor default was found in a group!")
            
        self.config = configs.get(group)
        if self.config is None:
            raise ValueError("A group without a corresponding configuration was found!")

class default_builder:
    def __init__(self, group_name: str, config: dict):
        self.group_name = group_name
        self.config = config
        self.calculated_values = {}

        if self.get("y0") is None or self.get("ymcx_y0") is None or self.get("t_env") is None or self.get("gamma_psi_base") is None or self.get("N_e") is None or self.get("delta") is None:
            raise ValueError(f"Incomplete configuration for the group {self.group_name}")

    def x_qp(self, T: float):
        x_qp = sqrt(2*pi*k*T*(1/self.config.get("delta"))) * exp(-1*self.config.get("delta")*(1/(k*T)))
        self.calculated_values["x_qp"] = x_qp
        return x_qp
    
    def w_p(self, T: float):
        w_p0 = 0 # Look into how accurate this is in the future!
        w_p = w_p0 * sqrt(1 - 2*sqrt(2*pi*k*T/self.config.get("delta")) * exp(-self.config.get("delta")/(k*T)))
        self.calculated_values["w_p"] = w_p
        return w_p

    def bose_einstein(w: float, T: float):
        x = (hbar*w) / (k * T)
        return 1.0 / (exp(x) - 1.0)
    
    def n_eff(self, T: float):
        n_eff =  self.config.get("ymxc_y0") * self.bose_einstein(self.config.get("w_ge"), T)  +  \
            (1-self.config.get("ymxc_y0")) * self.bose_einstein(self.config.get("w_ge"), T)

        self.calculated_values["n_eff"] = n_eff
        return n_eff
    
    def gamma_qp(self, T: float):
        delta = self.config.get("delta")
        w_ge = self.config.get("w_ge")
        x_qp = self.calculated_values.get("x_qp") if self.calculated_values.get("x_qp") else self.x_qp(T)
        w_p = self.calculated_values.get("w_p") if self.calculated_values.get("w_p") else self.w_p(T)

        gamma_qp = ((w_p**2)/(pi*delta)) * ( x_qp * sqrt(2*delta/(hbar*w_ge)) * 4*exp(-1*delta/(k*T)) * cosh(hbar*w_ge/(2*k*T)) * k0(hbar*w_ge/(2*k*T)))
        self.calculated_values["gamma_qp"] = gamma_qp
        return gamma_qp
    
    def T1(self, T: float) -> float:
        gamma_qp = self.calculated_values.get("gamma_qp") if self.calculated_values.get("gamma_qp") else self.gamma_qp(T)
        n_eff = self.calculated_values.get("n_eff") if self.calculated_values.get("n_eff") else self.n_eff(T)

        T1 =  1/(gamma_qp + self.config.get("y_0")*(2*n_eff + 1))
        self.calculated_values["T1"] = T1
        return T1
    
    def gamma_psi_qp(self, T: float):
        w_p = self.calculated_values.get("w_p") if self.calculated_values.get("w_p") else self.w_p(T)

        gamma_psi_qp =  (4*pi*(w_p**2)/(self.config.get("w_ge"))) * sqrt(exp(-1*self.config.get("delta")/(k*T))/(self.config.get("N_e")))
        self.calculated_values["gamma_psi_qp"] = gamma_psi_qp
        return gamma_psi_qp
    
    def T2(self, T: float) -> float:
        T1 = self.calculated_values.get("T1") if self.calculated_values.get("T1") else self.T1(T)
        gamma_psi_qp = self.calculated_values.get("gamma_psi_qp") if self.calculated_values.get("gamma_psi_qp ") else self.gamma_psi_qp(T)

        T2 = 1/(1/(2*T1) + self.config.get("gamma_psi_base") + gamma_psi_qp)
        self.calculated_values["T2"] = T2
        return T2