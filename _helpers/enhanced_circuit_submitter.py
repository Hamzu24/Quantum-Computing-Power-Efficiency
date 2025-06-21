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

with open("configs/power_configs.json", 'r') as f:
    power_configs = json.load(f)

try:
    with open("configs/power_configs.json", "r") as f:
        power_configs = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("Power config file not found")
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in power config file: {e}")

try:
    with open("configs/noise_models.json", "r") as f:
        noise_models = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("noise model file not found")
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in power config file: {e}")

class CircuitSubmitter(_helpers.circuit_submitter.CircuitSubmitter):
    def __init__(self, benchmark_name: str, device_name: str = "noisy_sim"):
        super().__init__(benchmark_name, device_name)
        self.total_gates = Counter()
        self.global_consumption = Counter()

        if power_configs.get(device_name) is not None:
            self.power_config = power_configs.get(device_name)
        else:
            self.power_config = power_configs.get("default_power_config")
        
        if power_configs.get(device_name) is not None:
            self.backend.noise_model = noise_models.get(device_name)
    
    def _has_a_measurement(circuits, circuit_type: str = "qasm_strs"):
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
        if qasm_strs is not None:
            has_a_measurement = self._has_a_measurement(qasm_strs, "qasm_strs")
        elif qasm_paths is not None:
            has_a_measurement = self._has_a_measurement(qasm_paths, "qasm_paths")
        else:
            has_a_measurement = self._has_a_measurement(braket_circuits, "braket_circuits")
        
        if has_a_measurement:
            pass
        # Save it!

        circuits = self._get_circuits_from_tasks(tasks)
        self._populate_gate_counter(self.total_gates, circuits)

        return tasks

    def get_power_consumption(self):
        consumption = self._calculate_power_consumption(self.total_gates)
        return consumption

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
                print(f"QuantumCircuit from Braket has this metadata: {metadata}\n{additional_metadata}")
                circuit = result.task_metadata.braketSchemaHeader
                circuits.append(circuit)

        return circuits


    def _populate_gate_counter(self, counter: Counter, circuits):
        def update_single_circuit(circuit):
            if isinstance(circuit, QuantumCircuit):
                counter.update(circuit.count_ops())
            elif isinstance(circuit, BraketCircuit):
                print(dir(circuit))
                instructions = circuit.instructions
                for instruction in instructions:
                    counter[instruction.operator.name] += 1

        if not isinstance(circuits, Iterable):
            update_single_circuit(circuits)
        else:
            for circuit in circuits:
                self._populate_gate_counter(counter, circuit)
        print(counter)

    def _calculate_power_consumption(self, gates: Counter, silent: bool = False, error_if_incomplete: bool = True):
        consumption = Counter()
        for operation, count in gates.items():

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

        self.global_consumption = self.global_consumption + consumption
        return consumption

_helpers.circuit_submitter.CircuitSubmitter = CircuitSubmitter