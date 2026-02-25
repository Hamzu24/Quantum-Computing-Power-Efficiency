import json
import os
import datetime
from _helpers.json_manager import JsonManager
import logging
from _helpers.constants import HARDWARE_CONFIG_GROUPS
from _helpers.builders.base import builder_registry
from _helpers.registry import control_parameter_registry
from _helpers.builders.default_builder import DefaultBuilder
from _helpers.helpers import get_config_value

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

        if hasattr(self.builder, 'initialize_per_qubit_params'):
            self.builder.initialize_per_qubit_params()
            logging.info(f"Initialized per-qubit parameters for {self.name}")

        if hasattr(self.builder, 'initialize_per_gate_params'):
            self.builder.initialize_per_gate_params()
            logging.info(f"Initialized per-gate parameters for {self.name}")
    
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


    
    def get_noise_model_metadata(self) -> dict:
        if hasattr(self.builder, 'config_tracker'):
            metadata = self.builder.config_tracker.get_T1_T2_values()
            metadata.update(self.builder.config_tracker.get_gate_error_values())
            return metadata
        return {}

    def log_qubit_params(self, control_parameters: dict):
        """Log all per-qubit physics parameters to a JSON file in logs/."""
        if not hasattr(self.builder, 'get_all_qubit_params_at_T'):
            return

        T = get_config_value(control_parameters, "temperature")
        if T is None:
            return

        all_params = self.builder.get_all_qubit_params_at_T(T)

        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        T_mK = T * 1e3
        filename = f"qubit_params_{self.name}_{T_mK:.1f}mK_{timestamp}.json"
        filepath = os.path.join(logs_dir, filename)

        log_data = {
            "backend": self.name,
            "temperature_K": T,
            "temperature_mK": T_mK,
            "timestamp": timestamp,
            "num_qubits": len(all_params),
            "qubits": {str(k): v for k, v in all_params.items()},
        }

        with open(filepath, "w") as f:
            json.dump(log_data, f, indent=2)

        logging.info(f"Qubit parameters logged to {filepath}")

    def build_backend(self, control_parameters) -> dict:
        control_parameter_registry.set_control_parameters(control_parameters)
        self._build_qubits(control_parameters)
        self._build_gates(control_parameters)
        self.json_manager.write()

        if hasattr(self.builder, "config_tracker"):
            self.builder.config_tracker.log_info()

        self.log_qubit_params(control_parameters)

        return self.get_noise_model_metadata()

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
                    if self.json_manager.find_path(prop, "value", gate_param_path) is None:
                        logging.debug(f"Skipping property '{prop}' — not found in gate at {gate_path}")
                        continue
                    self.json_manager.update_with_units(prop, val, gate_param_path)
