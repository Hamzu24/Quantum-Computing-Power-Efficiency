from circuit_submitter import CircuitSubmitter as OriginalCircuitSubmitter
from collections import Counter
from pathlib import Path
import json
from typing import Union
from braket.tasks.local_quantum_task import LocalQuantumTask
from qiskit.providers.aer import AerJob
from qiskit.circuit.quantumcircuit import QuantumCircuit
from noisy_simulator_wrappers import QiskitTaskWrapper
from typing import Iterable
from braket.circuit import BraketCircuit

class CircuitSubmitter(OriginalCircuitSubmitter):
    def __init__(self, benchmark_name: str, power_config_path: Path = None, power_config_dict: dict = None, device_name: str = "noisy_sim"):
        """
        Args:
            benchmark_name: the name of the benchmark.
            device_name: the name of the device.
                Choose among "simulator", "noisy_sim", "noiseless_sim", "Aria".
        """
        super().__init__()
        self.total_gates = Counter()
        self.global_consumption = Counter()

        if power_config_path and power_config_dict:
            raise ValueError("Please provide either power_config_path OR power_config_dict, not both!")
        elif power_config_path:
            try:
                with open(power_config_path, "r") as f:
                    self.power_config = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Power config file not found: {power_config_path}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in power config file: {e}")
        elif power_config_dict:
            self.power_config = power_config_dict
        else:
            raise ValueError("Please provide either power_config_path or power_config_dict!")
    
    def submit_circuits(self, shots: int, verbatim: bool = True, skip_asking: bool = False, skip_transpilation: bool = False, print_summary: bool = True, braket_circuits: list = None, qasm_strs: list[str] = None, qasm_paths: list[str] = None, inputs: dict[str, float] = None) -> Union[list[AwsQuantumTask], list[LocalQuantumTask]]:
        tasks = super().submit_circuits(shots, verbatim, skip_asking, skip_transpilation, print_summary, braket_circuits, qasm_strs, qasm_paths, inputs)

        circuits = _get_circuits_from_tasks(tasks)
        self._populate_gate_counter(self.total_gates, circuits)

        return tasks

    def get_power_consumption(self):
        consumption = self.power_module._calculate_power_consumption(self.total_gates)
        with open(self.benchmark_path, "w") as f:
            json.dumps(dict(consumption), f, indent=4)
    
    def _get_circuits_from_tasks(tasks: Union[list[LocalQuantumTask], list[QiskitTaskWrapper]]):
        circuits = []

        if not tasks:
            return None #EMPTY CIRCUIT!
        elif isinstance(tasks[0], QiskitTaskWrapper):
            for task in tasks:
                job = task.task
                circuit = _get_circuits_from_aer_job(job)
                circuits.append(circuit)
        elif isinstance(tasks[0], LocalQuantumTask):
            for circuit in circuits:
                circuit = task.quantum_task_arn
                circuits.append(circuit)

        return circuits

    def _get_circuits_from_aer_job(job: AerJob):
        if hasattr(job, '_circuits'):
            return job._circuits
        elif hasattr(job, 'circuits'):
            return job.circuits
        
        raise ValueError(f"Unable to extract the circuit from the AerJob: {job}")

    def _populate_gate_counter(self, counter: Counter, circuits):
        def update_single_circuit(circuit):
            if isinstance(circuits, QuantumCircuit):
                counter.update(circuit.count_ops())
            elif isinstance(circuits, BraketCircuit):
                print(dir(circuit))
                instructions = circuit.instructions
                for instruction in instructions:
                    counter[instruction.operator.name] += 1

        if not isinstance(circuits, Iterable):
            update_single_circuit(circuits)
        else:
            for circuit in circuits:
                update_single_circuit(circuit)

    def _calculate_power_consumption(self, gates: Counter, silent: bool = False, error_if_incomplete: bool = True):
        consumption = Counter()
        for operation, count in gates.items():

            operation_cost = self.power_config.get(operation)
            if operation_cost is None:
                if error_if_incomplete:
                    print(f"The operation {operation} was not found in the config! Exiting now.")
                    raise KeyError(f"The key {operation} is required to be in the config!")
                print(f"Operation not found in config: {operation}. Skipping...")

            if consumption.get(operation) is None:
                consumption[operation] = 0
            consumption[operation] += count*operation_cost

        self.global_consumption = self.global_consumption + consumption
        return consumption

import circuit_submitter
circuit_submitter.CircuitSubmitter = CircuitSubmitter