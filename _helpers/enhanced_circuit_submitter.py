import _helpers.circuit_submitter
from collections import Counter
from pathlib import Path
import json
from typing import Iterable, Union, Tuple
from braket.tasks.local_quantum_task import LocalQuantumTask
from qiskit.providers.aer import AerJob
from qiskit.circuit.quantumcircuit import QuantumCircuit
from _helpers.noisy_simulator_wrappers import QiskitTaskWrapper
from braket.circuits import Circuit as BraketCircuit
from braket.aws import AwsQuantumTask
from pprint import pprint
from copy import deepcopy
from _helpers.nm_helper import craft_noise_model
from _helpers.helpers import get_basis_gates_from_backend, read_config, display_noise_model
from _helpers.registry import submitter_registry
import os
import logging
from _helpers.constants import NoiselessSimBasisGates, SIMULATION_METHOD

class CircuitSubmitter(_helpers.circuit_submitter.CircuitSubmitter):

    def __init__(self, benchmark_name: str, device_name: str = "noisy_sim"):
        super().__init__(benchmark_name, device_name)
        
        configs = read_config()
        self._validate_configs(configs)

        self._setup_power_config(configs, device_name)
        self._setup_noise_model(configs, device_name)
        self._setup_device_tracking(configs, device_name)
        
        self.total_gates = Counter()
        self.gate_history = []
        submitter_registry.store_submitter(self, device_name)
        
    def _validate_configs(self, configs):
        required_keys = ["power_configs", "noise_models", "device_tracking"]
        for key in required_keys:
            if configs.get(key) is None:
                raise ValueError("Invalid config file! Must include power_configs, noise_models and device_tracking config options!")

    def _setup_power_config(self, configs, device_name):
        power_configs = configs.get("power_configs")
        if power_configs.get(device_name) is not None:
            self.power_config = power_configs.get(device_name)
        else:
            self.power_config = power_configs.get("default_power_config")

    def _apply_noise_model(self, noise_model_instance):
        self.backend.noise_model = noise_model_instance
        self.backend.device.noise_model = noise_model_instance
        self.backend.device.sim = self.backend.device.backend(
            method=SIMULATION_METHOD, noise_model=noise_model_instance, device='GPU'
        )

    def _setup_noise_model(self, configs, device_name):
        self.nm_backend = None

        selected_noise_model = configs.get("selected_noise_model")
        noise_models = configs.get("noise_models")
        noisy_devices = ["noisy_sim", "noisy_sim_with_shots"]
        if device_name not in noisy_devices:
            logging.debug(f"You are not using a noisy device simulator. The backend being used is {self.backend}")
            return

        if noise_models.get(selected_noise_model) is not None:
            noise_model_specs = noise_models.get(selected_noise_model)
        elif noise_models.get("default") is not None:
            noise_model_specs = noise_models.get("default")
        print(f"\n\nnoise model specs: {noise_model_specs}")

        if noise_model_specs:
            noise_model_instance, nm_backend = craft_noise_model(noise_model_specs)
            self.nm_backend = nm_backend
            self._apply_noise_model(noise_model_instance)
            return
        
        logging.warning(f"No noise model spec set for the device {device_name} or for default in the noise_models configuration dictionary. Using the program's default noise model")
        
        
        logging.debug(f"You are using a noisy simulator. The backend being used is {self.backend}, noise model is {self.backend.noise_model.name}, device noise model is {self.backend.device.noise_model.name}")
        logging.debug(f"basis gates are {noise_model_instance.basis_gates}")

    def _setup_device_tracking(self, configs, device_name):
        device_tracking = configs.get("device_tracking")
        if device_tracking.get(device_name) is not None:
            self.tracking_number = device_tracking.get(device_name)
        else:
            self.tracking_number = 0
        
    def _qasm_string_has_measurement(self, qasm_string):
        measurement_keywords = ['measure', 'reset']
        
        qasm_lower = qasm_string.lower()
        return any(keyword in qasm_lower for keyword in measurement_keywords)
    
    def _has_a_measurement(self, circuits, circuit_type: str = "qasm_strs"):
        if isinstance(circuits, Iterable) and not isinstance(circuits, str):
            return any(self._has_a_measurement(circuit, circuit_type) for circuit in circuits)
        
        if circuit_type == "qasm_strs":
            return self._qasm_string_has_measurement(circuits)
        elif circuit_type == "qasm_paths":
            qasm_strs = QuantumCircuit.from_qasm_file(circuits).qasm()
            return self._qasm_string_has_measurement(qasm_strs)
        elif circuit_type == "braket_circuits":
            raise NotImplementedError("measurement checking not implemented for braket circuits!")
        else:
            raise ValueError(f"Unsupported circuit type: {circuit_type}")
        
        return False

    def submit_circuits(self, shots: int, verbatim: bool = True, skip_asking: bool = False, skip_transpilation: bool = False, print_summary: bool = True, braket_circuits: list = None, qasm_strs: list[str] = None, qasm_paths: list[str] = None, inputs: dict[str, float] = None) -> Union[list[AwsQuantumTask], list[LocalQuantumTask]]:
        tasks = super().submit_circuits(shots, verbatim, skip_asking, skip_transpilation, print_summary, braket_circuits, qasm_strs, qasm_paths, inputs)

        # Gate counting disabled to prevent memory leaks
        # To re-enable, uncomment the code below
        return tasks

        # circuits = self._get_circuits_from_tasks(tasks)
        #
        # if self.tracking_number <= 0:
        #     return tasks
        #
        # self._populate_gate_counter(self.total_gates, circuits)
        # if self.tracking_number <= 1:
        #     return tasks
        #
        # if qasm_strs is not None:
        #     has_a_measurement = self._has_a_measurement(qasm_strs, "qasm_strs")
        # elif qasm_paths is not None:
        #     has_a_measurement = self._has_a_measurement(qasm_paths, "qasm_paths")
        # else:
        #     has_a_measurement = self._has_a_measurement(braket_circuits, "braket_circuits")
        #
        # if has_a_measurement:
        #     self.gate_history.append(deepcopy(self.total_gates))
        #
        # return tasks

    def get_power_consumption(self) -> Tuple[Counter, list[Counter]]:
        total_consumption = self._calculate_power_consumption(self.total_gates)

        staggered_consumptions = []
        prev_consumption = Counter()
        for gate_count in self.gate_history:
            staggered_consumption = self._calculate_power_consumption(gate_count)

            new_prev_consumption = staggered_consumption
            staggered_consumption = staggered_consumption - prev_consumption
            prev_consumption = new_prev_consumption

            staggered_consumptions.append(staggered_consumption)

        return total_consumption, staggered_consumptions

    def _get_circuits_from_aer_job(self, job: AerJob):
        if hasattr(job, '_circuits'):
            return job._circuits
        elif hasattr(job, 'circuits'):
            return job.circuits
    
        raise ValueError(f"Unable to extract the circuit from the AerJob: {job}")

    def _get_circuits_from_tasks(self, tasks: Union[list[LocalQuantumTask], list[QiskitTaskWrapper]]):
        circuits = []

        if not tasks:
            return None
        elif isinstance(tasks[0], QiskitTaskWrapper):
            for task in tasks:
                job = task.task
                circuit = self._get_circuits_from_aer_job(job)
                circuits.append(circuit)
        elif isinstance(tasks[0], LocalQuantumTask):
            for task in tasks:
                result = task.result()
                metadata = result.task_metadata
                additional_metadata = result.additional_metadata
                circuit = result.task_metadata.braketSchemaHeader
                circuits.append(circuit)

        return circuits


    def _populate_gate_counter(self, counter: Counter, circuits):
        def update_single_circuit(circuit):
            if isinstance(circuit, QuantumCircuit):
                counter.update(circuit.count_ops())
            elif isinstance(circuit, BraketCircuit):
                instructions = circuit.instructions
                for instruction in instructions:
                    counter[instruction.operator.name] += 1

        if not isinstance(circuits, Iterable):
            update_single_circuit(circuits)
        else:
            for circuit in circuits:
                self._populate_gate_counter(counter, circuit)

    def _calculate_power_consumption(self, gates: Counter, error_if_incomplete: bool = True):
        consumption = Counter()
        operations_to_ignore = ["save_density_matrix", "barrier"]
        
        for operation, count in gates.items():
            if operation in operations_to_ignore:
                continue

            operation_cost = self.power_config.get(operation)
            if operation_cost is None:
                if error_if_incomplete:
                    print(f"The operation {operation} was not found in the config! Exiting now.")
                    raise KeyError(f"The key {operation} is required to be in the config!")

                print(f"Operation not found in config: {operation}. Skipping...")
                continue

            if consumption.get(operation) is None:
                consumption[operation] = 0
            consumption[operation] += count*operation_cost

        return consumption

    def get_basis_gates(self):
        if self.nm_backend is None:
            return NoiselessSimBasisGates

        if self.nm_backend.version == -1: # -1 represents a custom noise model, not from a fake backend
            return self.nm_backend.basis_gates

        return get_basis_gates_from_backend(self.nm_backend)

_helpers.circuit_submitter.CircuitSubmitter = CircuitSubmitter
