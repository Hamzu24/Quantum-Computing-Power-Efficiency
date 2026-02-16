from _helpers.json_manager import JsonManager
from typing import Protocol, Any, Type, ClassVar
import logging

class Builder(Protocol):
    registry_name: ClassVar[str]
    
    def __init__(self, name: str, config: dict, init_control_parameters: dict, jm: JsonManager):
        "initialise the builder"
    
    def T1(self, *args: Any, **kwargs: Any) -> float:
        "calculate T1 relaxation time from the hardware constants and temperature passed in"

    def T2(self, *args: Any, **kwargs: Any) -> float:
        "calculate T2 dephasing time from the hardware constants and temperature passed in"

    def calculate_qb_config(self, control_parameters: dict, qb_path: str):
        "build the backend configuration from the hardware constants and temperature passed in"

    def calculate_gate_config(self, control_parameters: dict, qb_path: str):
        "build a gate configuration from the control parameters and a specific gate dictionary"

    def initialize_per_qubit_params(self):
        "OPTIONAL: derive per-qubit parameters from calibration data"

    def initialize_per_gate_params(self):
        "OPTIONAL: derive per-gate parameters from calibration data"

class BuilderRegistry:
    def __init__(self):
        self.builders = {}

    def get_builder(self, name: str):
        return self.builders.get(name)
    
    def register_builder(self, cls: Type[Builder]):
        self.builders[cls.registry_name] = cls
        return cls
    
    def list_builders(self):
        return list(self.builders.keys())

builder_registry = BuilderRegistry()

class ConfigTracker:
    def __init__(self):
        self.qb_config_infos = []
        self.gate_config_infos = []

    def add_config(self, config, type):
        if type == "qb":
            self.qb_config_infos.append(config)
        elif type == "gate":
            self.gate_config_infos.append(config)
        else:
            raise ValueError(f"Unable to add config to the tracker: unknown type {type}")
    
    def get_T1_T2_values(self) -> dict:
        if not self.qb_config_infos:
            return {}
        return {
            "T1_values": [info["config"]["T1"] for info in self.qb_config_infos],
            "T2_values": [info["config"]["T2"] for info in self.qb_config_infos],
        }

    def get_gate_error_values(self) -> dict:
        if not self.gate_config_infos:
            return {}
        return {
            "gate_error_values": [info["config"]["gate_error"] for info in self.gate_config_infos],
        }

    def log_info(self):
        avg_T1 = 0
        avg_T2 = 0
        num_qb = 0
        for info in self.qb_config_infos:
            avg_T1 += info["config"]["T1"]
            avg_T2 += info["config"]["T2"]
            num_qb += 1

        avg_T1 /= num_qb
        avg_T2 /= num_qb
        logging.debug(f"avg computed T1: {avg_T1}")
        logging.debug(f"avg computed T2: {avg_T2}")

        logging.debug("\n\n")
        avg_ge = 0
        num_gates = 0
        for info in self.gate_config_infos:
            avg_ge += info["config"]["gate_error"]
            num_gates += 1

        avg_ge /= num_gates
        logging.debug(f"avg gate error: {avg_ge}")
