import circuit_submitter
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
from braket.circuits import Circuit as BraketCircuit
from braket.aws import AwsQuantumTask

class CircuitSubmitter(OriginalCircuitSubmitter):
    def __init__(self, benchmark_name: str, power_config_path: Path = None, power_config_dict: dict = None, device_name: str = "noisy_sim"):
        print("patch me monkey")
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
                print(f"QiskitTaskWrapper has this circuit: {circuit}")
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
                continue

            if consumption.get(operation) is None:
                consumption[operation] = 0
            consumption[operation] += count*operation_cost

        self.global_consumption = self.global_consumption + consumption
        return consumption

circuit_submitter.CircuitSubmitter = CircuitSubmitter