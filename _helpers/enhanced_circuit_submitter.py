import _helpers.circuit_submitter
from collections import Counter
from pathlib import Path
import json
from typing import Iterable, Union
from braket.tasks.local_quantum_task import LocalQuantumTask
from qiskit.providers.aer import AerJob
from qiskit.circuit.quantumcircuit import QuantumCircuit
from _helpers.noisy_simulator_wrappers import QiskitTaskWrapper
from braket.circuits import Circuit as BraketCircuit
from braket.aws import AwsQuantumTask
from pprint import pprint
from copy import deepcopy
from _helpers.nm_helper import craft_noise_model

power_configs = None
noise_models = None
device_tracking = None
def load_config_file(config_path):
    try:
        with open(config_path, 'r') as f:
            configs = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Power config file not found")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in power config file: {e}")

    power_configs = configs.get("power_configs")
    noise_models = configs.get("noise_models")
    device_tracking = configs.get("device_tracking")

    if power_configs is None or noise_models is None or device_tracking is None:
        raise ValueError("Invalid config file! Must include power_configs, noise_models and device_tracking config options!")

class CircuitSubmitter(_helpers.circuit_submitter.CircuitSubmitter):
    def __init__(self, benchmark_name: str, device_name: str = "noisy_sim"):
        super().__init__(benchmark_name, device_name)
        self.total_gates = Counter()

        if power_configs.get(device_name) is not None:
            self.power_config = power_configs.get(device_name)
        else:
            self.power_config = power_configs.get("default_power_config")
        
        if device_name in ["noisy_sim", "noisy_sim_with_shots"]:
            if noise_models.get(device_name) is not None:
                noise_model_specs = noise_models.get(device_name)
                self.backend.noise_model = craft_noise_model(noise_model_specs)
            elif noise_models.get("default_noise_model") is not None:
                noise_model_specs = noise_models.get("default_noise_model")
                self.backend.noise_model = craft_noise_model(noise_model_specs)
        
        if device_tracking.get(device_name) is not None:
            self.tracking_number = device_tracking.get(device_name)
        else:
            self.tracking_number = 0
        
        self.gate_history = []

        if self.device_name in ["noisy_sim", "noisy_sim_with_shots"]:
            print(f"the backend being used is {self.backend}, noise model is {self.backend.noise_model.__class__.__name__}")
        else:
            print(f"the backend being used is {self.backend}")
    
    def _has_a_measurement(self, circuits, circuit_type: str = "qasm_strs"):
        def qasm_string_has_measurement(qasm_string):
            measurement_keywords = ['measure', 'reset']
            
            for keyword in measurement_keywords:
                if keyword in qasm_string.lower():
                    return True
            
            return False
        
        if isinstance(circuits, Iterable) and not isinstance(circuits, str):
            for circuit in circuits:
                if self._has_a_measurement(circuit, circuit_type):
                    return True
            return False
        
        if circuit_type == "qasm_strs":
            if qasm_string_has_measurement(circuits):
                return True
        elif circuit_type == "qasm_paths":
            qasm_strs = QuantumCircuit.from_qasm_file(circuits).qasm()
            if qasm_string_has_measurement(qasm_strs):
                return True
        elif circuit_type == "braket_circuits":
            raise ValueError("measurement checking not implemented for braket circuits!")
        
        return False

    def submit_circuits(self, shots: int, verbatim: bool = True, skip_asking: bool = False, skip_transpilation: bool = False, print_summary: bool = True, braket_circuits: list = None, qasm_strs: list[str] = None, qasm_paths: list[str] = None, inputs: dict[str, float] = None) -> Union[list[AwsQuantumTask], list[LocalQuantumTask]]:
        tasks = super().submit_circuits(shots, verbatim, skip_asking, skip_transpilation, print_summary, braket_circuits, qasm_strs, qasm_paths, inputs)

        circuits = self._get_circuits_from_tasks(tasks)

        if self.tracking_number <= 0:
            return tasks
        self._populate_gate_counter(self.total_gates, circuits)


        if self.tracking_number <= 1:
            return tasks
        if qasm_strs is not None:
            has_a_measurement = self._has_a_measurement(qasm_strs, "qasm_strs")
        elif qasm_paths is not None:
            has_a_measurement = self._has_a_measurement(qasm_paths, "qasm_paths")
        else:
            has_a_measurement = self._has_a_measurement(braket_circuits, "braket_circuits")
        
        if has_a_measurement:
            self.gate_history.append(deepcopy(self.total_gates))

        return tasks

    def get_power_consumption(self):
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

    def _get_circuits_from_tasks(self, tasks: Union[list[LocalQuantumTask], list[QiskitTaskWrapper]]):
        circuits = []

        def get_circuits_from_aer_job(job: AerJob):
            if hasattr(job, '_circuits'):
                return job._circuits
            elif hasattr(job, 'circuits'):
                return job.circuits
        
            raise ValueError(f"Unable to extract the circuit from the AerJob: {job}")

        if not tasks:
            return None
        elif isinstance(tasks[0], QiskitTaskWrapper):
            for task in tasks:
                job = task.task
                circuit = get_circuits_from_aer_job(job)
                circuits.append(circuit)
        elif isinstance(tasks[0], LocalQuantumTask):
            for task in tasks:
                result = task.result()
                metadata = result.task_metadata
                additional_metadata = result.additional_metadata
                #print(f"QuantumCircuit from Braket has this metadata: {metadata}\n{additional_metadata}")
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

    def _calculate_power_consumption(self, gates: Counter, silent: bool = False, error_if_incomplete: bool = True):
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

CONFIG_PATH = "configs.json"
load_config_file(CONFIG_PATH)
_helpers.circuit_submitter.CircuitSubmitter = CircuitSubmitter