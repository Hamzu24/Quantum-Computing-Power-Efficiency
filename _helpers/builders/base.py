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
    
    def optimise_parameters(self):
        "optimise certain hardware parameters to fit the system"

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
    
    def log_info(self):
        avg_T1 = 0
        avg_T2 = 0
        num_qb = 0
        for info in self.qb_config_infos:
            logging.debug(info["adjs"])
            avg_T1 += info["adjs"]["T1"]
            avg_T2 += info["adjs"]["T2"]
            num_qb += 1

        avg_T1 /= num_qb
        avg_T2 /= num_qb
        logging.debug(f"avg T1 adj: {avg_T1}")
        logging.debug(f"avg T2 adj: {avg_T2}")

        logging.debug("\n\n")
        avg_ge = 0
        num_gates = 0
        for info in self.gate_config_infos:
            logging.debug(info["adjs"])
            avg_ge += info["adjs"]
            num_gates += 1

        avg_ge /= num_gates
        logging.debug(f"avg ge adj: {avg_ge}")
