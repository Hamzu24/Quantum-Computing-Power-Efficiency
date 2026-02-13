import json
import os
from _helpers.json_manager import JsonManager
import logging
from _helpers.constants import HARDWARE_CONFIG_GROUPS
from _helpers.builders.base import builder_registry
from _helpers.registry import control_parameter_registry
from _helpers.builders.default_builder import DefaultBuilder

class BuilderWrapper:
    def __init__(self, name: str, init_control_parameters: dict):
        self.name = name
        HARDWARE_CONFIG_PATH = os.environ.get("HARDWARE_CONFIG_PATH")
        self.load_hardware_params(HARDWARE_CONFIG_PATH)
        logging.info(f"hardware config for backend: {self.config}")
        
        builder_name = self.config["builder_class"]
        backend_config_folder = os.environ.get("BACKEND_CONFIGS_FOLDER")
        filename = backend_config_folder + f"{name}/props_{name}.json"
        self.json_manager = JsonManager(filename)
        logging.info(f"All available builders: {builder_registry.list_builders()}")
        self.builder = builder_registry.get_builder(builder_name)(name, self.config, self.json_manager, init_control_parameters)

        self.builder.initialize_per_qubit_params()
        logging.info(f"Initialized per-qubit parameters for {self.name}")
    
    def find_group(self):
        group = None
        for group_name, group_contents in HARDWARE_CONFIG_GROUPS.items():
            if self.name in group_contents:
                group = group_name
                break
            elif "default" in group_contents:
                group = group_name 
        
        if group is None:
            raise ValueError("Neither the name nor default was found in a group! Please check the HARDWARE_CONFIG_GROUPS variable in the _helpers/constants file.")
        
        return group
            

    def load_hardware_params(self, path: str):
        with open(path, "r") as f:
            configs = json.load(f)

        group = self.find_group()
        self.config = configs.get(group)
        if self.config is None:
            raise ValueError(f"Error: the group {group} does not have a corresponding configuration in the {str(path)} file.")


    
    def build_backend(self, control_parameters):
        control_parameter_registry.set_control_parameters(control_parameters)
        self._build_qubits(control_parameters)
        self._build_gates(control_parameters)
        self.json_manager.write()
        
        # Add a config_tracker if you need to log the final configs
        if (self.builder, "config_tracker"):
            self.builder.config_tracker.log_info()

    def _build_qubits(self, control_parameters):
        qubit_paths = self.json_manager.get_qubit_paths()

        for qb_path in qubit_paths:
            qb_config = self.builder.calculate_qb_config(control_parameters, qb_path)
            for prop, val in qb_config.items():
                self.json_manager.update_with_units(prop, val, qb_path)
    
    def _build_gates(self, control_parameters):
        gate_paths = self.json_manager.get_gate_paths()

        for gate_path in gate_paths:
            gate_config = self.builder.calculate_gate_config(control_parameters, gate_path)
            gate_param_path = gate_path + "parameters."
            for prop, val in gate_config.items():
                if val is not None:
                    self.json_manager.update_with_units(prop, val, gate_param_path)
