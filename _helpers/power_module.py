from cirq.contrib.qasm_import import circuit_from_qasm
from cirq_ionq.ionq_native_gates import GPIGate, GPI2Gate, MSGate
import cirq_ionq
import numpy as np
import cirq
import qiskit
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import RemoveBarriers
from _helpers.backend_helpers import *
from braket.aws import AwsQuantumTask
from braket.tasks.local_quantum_task import LocalQuantumTask
from braket.ir.openqasm import Program
import os
import sys
from pathlib import Path
from datetime import datetime
import time
import json
from collections import Counter

class PowerModule():
    def __init__(self, power_config_path: Path = None, power_config_dict: dict = None):
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
        
    def calculate_power_consumption(self, gates: Counter, silent: bool = False, error_if_incomplete: bool = True):
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