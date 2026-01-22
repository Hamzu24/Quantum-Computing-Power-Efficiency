import json
import logging
import os
from typing import Any
from _helpers.constants import SI_PREFIXES
from qiskit_aer.noise import NoiseModel
import numpy as np

def read_config():
    CONFIG_PATH = os.environ.get("CONFIG_PATH")

    try:
        with open(CONFIG_PATH, 'r') as f:
            config_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Config file not found!")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in power config file: {e}")
    
    return config_data 

def write_config(config_data):
    CONFIG_PATH = os.environ.get("CONFIG_PATH")
    try:
        json_string = json.dumps(config_data, indent=2)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Config is not valid JSON: {e}")
    with open(CONFIG_PATH, 'w') as f:
        f.write(json_string)

def set_up_logger(log_level: int, log_file: str):
    logging_fmt='%(asctime)s | %(funcName)s:%(lineno)d | %(levelname)s | %(message)s'

    if log_file is not None:
        logging.basicConfig(
            filename=log_file,
            level=log_level,
            format=logging_fmt
        )
    else:
        logging.basicConfig(
            level=log_level,
            format=logging_fmt
        )

    logging.getLogger('qiskit').setLevel(logging.WARNING)

def get_control_parameters(config: dict):
    total_control_parameters = config.get("control_parameters")
    single_run = os.environ.get("SINGLE_RUN").lower() == "true"
    if not single_run:
        iteration = int(os.environ.get("iteration"))
    control_parameters = {}

    for cur_param, param_vals in total_control_parameters.items():
        if single_run:
            control_parameters[cur_param] = param_vals[0]
        else:
            start, step, unit = param_vals[1]

            val = start + (iteration*step)
            control_parameters[cur_param] = [val, unit]
    
    return control_parameters

"""
A function to easily and configurably extract many values from a dictionary

json_dict: The dictionary to extract the data from

values: A dictionary containing the keys to extract from the dictionary as well as the configuration to manipulate them
The values in this dictionary are tuples, T.
T[0]: the default value when the key is not found
T[1]: A dictionary of replacement values. When json_dict.get(key) is in the keys of this dictionary, it is replaced with its corresponding value.

required_values: the keys required to be in json_dict. If a key in this list is not found, a ValueError is raised


returns a list of all the extracted and configured values from values.keys()
"""
def extract_from_json(json_dict, values: dict[str, tuple[Any, dict[Any, Any]]], required_values: list = []):
    return_values = []

    for key, data in values.items():
        exceptions = data[1]
        cur_val = json_dict.get(key)
        if cur_val is None:
            if key in required_values:
                raise ValueError(f"Required value {key} was not in json!")

            cur_val = data[0]

        elif cur_val in exceptions.keys():
            cur_val = exceptions.get(cur_val)
        
        return_values.append(cur_val)

    return return_values

def get_unit_multiplier(unit: str):
    if len(unit) == 0:
        return None
    if len(unit) == 1:
        return 1

    return SI_PREFIXES.get(unit[0])

def get_config_value(config: dict, name: str):
    result = config.get(name)
    if result is None:
        return None

    val, unit = result
    multiplier = get_unit_multiplier(unit)
    return val * multiplier

def get_num_qubits():
    if os.environ.get("NUM_QUBITS") is not None:
        try:
            num_qubits = int(os.environ.get("NUM_QUBITS"))
        except ValueError as e:
            logging.error("Unable to parse a list of qubits from the environment variable. Falling back on the default of [5]")
            num_qubits = 5
    else:
        num_qubits = 5

    return num_qubits

def set_num_qubits_list():
    config_data = read_config()
    num_qubits_list = str(config_data.get("num_qubits"))
    if num_qubits_list is None:
        logging.error("No number of qubits specified for the simulation. Running with a default of 5")
        num_qubits_list = "5"
    
    os.environ["NUM_QUBITS"] = num_qubits_list
    logging.info(f"\n\n\nSet environ variable NUM_QUBITS to: {num_qubits_list}")
    return num_qubits_list

def set_circuit_optimisation():
    config_data = read_config()
    circuit_optimisation_level = config_data.get("circuit_optimisation_level")
    if circuit_optimisation_level is None:
        logging.error("No circuit optimisaiton level set. Running with a default of 1")
        circuit_optimisation_level = 1

    os.environ["CIRCUIT_OPTIMIZATION"] = str(circuit_optimisation_level)
    return circuit_optimisation_level

def get_basis_gates_from_backend(backend):
        if backend.version == 2:
            return backend.target.operation_names

        # V1 backends use configuration().basis_gates
        return backend.configuration().basis_gates

def get_max_parallel_from_config(default=10):
    config_data = read_config()
    return config_data.get("max_parallel_circuits", default)


def extract_metric_name(metric_path: str) -> str:
    return os.path.splitext(os.path.basename(metric_path))[0]

# NOISE MODEL DISPLAY FUNCTIONS

def display_noise_model(noise_model: NoiseModel, verbose: bool = True) -> dict:
    """
    Fully display all details of a Qiskit NoiseModel instance.
    
    Args:
        noise_model: The NoiseModel instance to inspect
        verbose: If True, print detailed output. If False, just return the dict.
    
    Returns:
        Dictionary containing all extracted noise model information
    """
    info = {
        'basis_gates': noise_model.basis_gates,
        'noise_instructions': noise_model.noise_instructions,
        'noise_qubits': noise_model.noise_qubits,
        'quantum_errors': [],
        'readout_errors': []
    }
    
    if verbose:
        print("=" * 80)
        print("NOISE MODEL DETAILED REPORT")
        print("=" * 80)
        
        print(f"\nBasis Gates: {noise_model.basis_gates}")
        print(f"Instructions with noise: {noise_model.noise_instructions}")
        print(f"Qubits with noise: {noise_model.noise_qubits}")
    
    if verbose:
        print("\n" + "-" * 80)
        print("QUANTUM ERRORS (Gate Errors)")
        print("-" * 80)
    
    for instruction in noise_model.noise_instructions:
        if instruction == 'measure': # Handle readout errors separately
            continue
            
        default_error = noise_model._default_quantum_errors.get(instruction)
        if default_error is not None:
            error_info = _extract_quantum_error_info(default_error, instruction, 'all')
            info['quantum_errors'].append(error_info)
            if verbose:
                _print_quantum_error(error_info)
        
        if instruction in noise_model._local_quantum_errors:
            for qubits, error in noise_model._local_quantum_errors[instruction].items():
                error_info = _extract_quantum_error_info(error, instruction, qubits)
                info['quantum_errors'].append(error_info)
                if verbose:
                    _print_quantum_error(error_info)
    
    # Now handling readout errors
    if verbose:
        print("\n" + "-" * 80)
        print("READOUT ERRORS (Measurement Errors)")
        print("-" * 80)
    
    if noise_model._default_readout_error is not None:
        error_info = _extract_readout_error_info(noise_model._default_readout_error, 'all')
        info['readout_errors'].append(error_info)
        if verbose:
            _print_readout_error(error_info)
    
    for qubits, error in noise_model._local_readout_errors.items():
        error_info = _extract_readout_error_info(error, qubits)
        info['readout_errors'].append(error_info)
        if verbose:
            _print_readout_error(error_info)
    
    # Summary statistics
    if verbose:
        print("\n" + "-" * 80)
        print("SUMMARY STATISTICS")
        print("-" * 80)
        _print_summary(info)
    
    return info


def _extract_quantum_error_info(error, instruction: str, qubits) -> dict:
    """Extract detailed information from a QuantumError object."""
    error_info = {
        'instruction': instruction,
        'qubits': qubits,
        'num_qubits': error.num_qubits,
        'size': error.size,  # Number of Kraus operators or circuits
        'probabilities': error.probabilities,
        'circuits': [],
        'error_type': _identify_error_type(error)
    }
    
    # Extract circuit/operator information
    for i, (prob, circ) in enumerate(zip(error.probabilities, error.circuits)):
        circ_info = {
            'probability': prob,
            'num_ops': len(circ.data) if hasattr(circ, 'data') else 0,
            'operations': []
        }
        
        if hasattr(circ, 'data'):
            for op in circ.data:
                op_info = {
                    'name': op.operation.name,
                    'qubits': [q._index for q in op.qubits] if hasattr(op.qubits[0], '_index') else list(range(len(op.qubits))),
                }
                # Try to get parameters if available
                if hasattr(op.operation, 'params') and op.operation.params:
                    op_info['params'] = [float(p) if isinstance(p, (int, float, np.floating)) else str(p) 
                                         for p in op.operation.params]
                circ_info['operations'].append(op_info)
        
        error_info['circuits'].append(circ_info)
    
    return error_info


def _extract_readout_error_info(error, qubits) -> dict:
    """Extract detailed information from a ReadoutError object."""
    probs = error.probabilities
    
    error_info = {
        'qubits': qubits,
        'num_qubits': error.number_of_qubits,
        'probabilities_matrix': probs.tolist() if hasattr(probs, 'tolist') else probs,
    }
    
    # For single-qubit readout errors, extract p(0|1) and p(1|0)
    if error.number_of_qubits == 1:
        error_info['p0_given_0'] = probs[0][0]
        error_info['p1_given_0'] = probs[0][1]
        error_info['p0_given_1'] = probs[1][0]
        error_info['p1_given_1'] = probs[1][1]
        error_info['avg_readout_error'] = (probs[0][1] + probs[1][0]) / 2
    
    return error_info


def _identify_error_type(error) -> str:
    """Try to identify the type of quantum error."""
    probs = error.probabilities
    circuits = error.circuits
    
    # Check if it's close to a depolarizing channel
    if len(probs) == 2 and len(circuits) == 2:
        # Could be a simple bit-flip or phase-flip
        return "simple_pauli"
    
    if error.num_qubits == 1:
        if len(probs) == 4:
            # Check for depolarizing (equal Pauli probabilities)
            pauli_probs = probs[1:]  # Exclude identity
            if len(pauli_probs) == 3 and np.allclose(pauli_probs, pauli_probs[0], rtol=0.1):
                return "depolarizing"
            return "general_pauli_1q"
        return "general_1q"
    
    if error.num_qubits == 2:
        if len(probs) == 16:
            return "general_pauli_2q"
        return "general_2q"
    
    return "unknown"


def _print_quantum_error(error_info: dict):
    """Print formatted quantum error information."""
    print(f"\n  {error_info['instruction'].upper()} on qubit(s) {error_info['qubits']}:")
    print(f"    Error type: {error_info['error_type']}")
    print(f"    Number of Kraus terms: {error_info['size']}")
    
    # Calculate and display error rate (1 - probability of identity)
    identity_prob = error_info['probabilities'][0] if error_info['probabilities'] else 0
    error_rate = 1 - identity_prob
    print(f"    Total error rate: {error_rate:.6e} (identity prob: {identity_prob:.6f})")
    
    # Show breakdown of error probabilities
    print(f"    Probability distribution:")
    for i, circ_info in enumerate(error_info['circuits']):
        prob = circ_info['probability']
        if prob > 1e-10:  # Only show non-negligible probabilities
            ops_str = ", ".join([op['name'] for op in circ_info['operations']]) if circ_info['operations'] else "identity"
            if prob > 0.001:
                print(f"      [{i}] p={prob:.6f}: {ops_str}")
            else:
                print(f"      [{i}] p={prob:.2e}: {ops_str}")


def _print_readout_error(error_info: dict):
    """Print formatted readout error information."""
    print(f"\n  Qubit(s) {error_info['qubits']}:")
    
    if error_info['num_qubits'] == 1:
        print(f"    P(measure 0 | prepared 0): {error_info['p0_given_0']:.6f}")
        print(f"    P(measure 1 | prepared 0): {error_info['p1_given_0']:.6e}")
        print(f"    P(measure 0 | prepared 1): {error_info['p0_given_1']:.6e}")
        print(f"    P(measure 1 | prepared 1): {error_info['p1_given_1']:.6f}")
        print(f"    Average readout error: {error_info['avg_readout_error']:.6e}")
    else:
        print(f"    Probability matrix ({2**error_info['num_qubits']}x{2**error_info['num_qubits']}):")
        matrix = error_info['probabilities_matrix']
        for i, row in enumerate(matrix):
            print(f"      |{i:0{error_info['num_qubits']}b}⟩ → {[f'{p:.4f}' for p in row]}")


def _print_summary(info: dict):
    """Print summary statistics of the noise model."""
    # Gate error statistics
    gate_errors = {}
    for err in info['quantum_errors']:
        instr = err['instruction']
        error_rate = 1 - err['probabilities'][0]
        if instr not in gate_errors:
            gate_errors[instr] = []
        gate_errors[instr].append(error_rate)
    
    print("\nGate Error Rates:")
    for gate, rates in sorted(gate_errors.items()):
        rates = np.array(rates)
        print(f"  {gate}:")
        print(f"    Min:  {rates.min():.6e}")
        print(f"    Max:  {rates.max():.6e}")
        print(f"    Mean: {rates.mean():.6e}")
        print(f"    Std:  {rates.std():.6e}")
    
    # Readout error statistics
    readout_rates = []
    for err in info['readout_errors']:
        if 'avg_readout_error' in err:
            readout_rates.append(err['avg_readout_error'])
    
    if readout_rates:
        rates = np.array(readout_rates)
        print("\nReadout Error Rates:")
        print(f"  Min:  {rates.min():.6e}")
        print(f"  Max:  {rates.max():.6e}")
        print(f"  Mean: {rates.mean():.6e}")
        print(f"  Std:  {rates.std():.6e}")

def _infidelity_to_angle_1q(infidelity):
    """
    Convert infidelity to rotation angle for single-qubit rotation.
    
    For U = exp(-i θ/2 σ) where σ is a Pauli:
    F = (1 + cos(θ/2)²) / 2
    infidelity = sin²(θ/2)
    Therefore: θ = 2 * arcsin(sqrt(infidelity))
    """
    infidelity = np.clip(infidelity, 0, 1)
    return 2 * np.arcsin(np.sqrt(infidelity))


def _infidelity_to_angle_2q(infidelity):
    """
    Convert infidelity to rotation angle for two-qubit ZZ rotation.
    
    For U = exp(-i θ/2 ZZ) on d=4 dimensional system:
    infidelity ≈ θ²/15 for small θ
    Therefore: θ ≈ sqrt(15 * infidelity)
    """
    infidelity = np.clip(infidelity, 0, 1)
    return np.sqrt(15 * infidelity)


def _single_qubit_coherent_unitary(theta):
    """
    Create single-qubit coherent error unitary.
    
    Uses a combined rotation that models both amplitude (X) and phase (Z) errors.
    Rotation is about an axis tilted 45° between X and Z.
    """
    theta_x = theta / np.sqrt(2)
    theta_z = theta / np.sqrt(2)
    
    Rx = RXGate(theta_x).to_matrix()
    Rz = RZGate(theta_z).to_matrix()
    
    return Rz @ Rx


def _two_qubit_coherent_unitary(theta, model='zz'):
    """
    Create two-qubit coherent error unitary.

    Parameters
    ----------
    theta : float
        Rotation angle
    model : str
        Error model to use:
        - 'zz': ZZ rotation (default, appropriate for CR gates on IBM hardware)
        - 'zx': ZX rotation (CR drive error)
        - 'xx': XX rotation (ion trap Mølmer-Sørensen gates)

    Returns
    -------
    np.ndarray
        4x4 unitary matrix
    """
    if model == 'zz':
        return RZZGate(theta).to_matrix()
    elif model == 'zx':
        return RZXGate(theta).to_matrix()
    elif model == 'xx':
        return RXXGate(theta).to_matrix()
    else:
        raise ValueError(f"Unknown two-qubit coherent error model: {model}")


class BackendPropertiesAdapter:
    """
    Adapter class to provide a unified interface for accessing backend properties
    from both V1 backends (with .properties()) and V2 backends (with .target).
    """

    def __init__(self, backend):
        self.backend = backend
        self._is_v2 = hasattr(backend, 'target') and backend.target is not None

        if self._is_v2:
            self._target = backend.target
            self._props = None
        else:
            self._target = None
            self._props = backend.properties()

    def t1(self, qubit: int) -> float:
        if self._is_v2:
            qp = self._target.qubit_properties
            if qp is None or qp[qubit] is None:
                raise ValueError(f"No qubit properties for qubit {qubit}")
            return qp[qubit].t1
        else:
            return self._props.t1(qubit)

    def t2(self, qubit: int) -> float:
        if self._is_v2:
            qp = self._target.qubit_properties
            if qp is None or qp[qubit] is None:
                raise ValueError(f"No qubit properties for qubit {qubit}")
            return qp[qubit].t2
        else:
            return self._props.t2(qubit)

    def gate_length(self, gate: str, qubits) -> float:
        if self._is_v2:
            if isinstance(qubits, int):
                qubits = (qubits,)
            else:
                qubits = tuple(qubits)

            if gate not in self._target.operation_names:
                raise ValueError(f"Gate {gate} not in target")

            inst_props = self._target[gate].get(qubits)
            if inst_props is None:
                raise ValueError(f"No properties for {gate} on qubits {qubits}")
            return inst_props.duration
        else:
            return self._props.gate_length(gate, qubits)

    def gate_error(self, gate: str, qubits) -> float:
        if self._is_v2:
            if isinstance(qubits, int):
                qubits = (qubits,)
            else:
                qubits = tuple(qubits)

            if gate not in self._target.operation_names:
                raise ValueError(f"Gate {gate} not in target")

            inst_props = self._target[gate].get(qubits)
            if inst_props is None:
                raise ValueError(f"No properties for {gate} on qubits {qubits}")
            return inst_props.error
        else:
            return self._props.gate_error(gate, qubits)

    def readout_error(self, qubit: int) -> float:
        if self._is_v2:
            # V2 backends store readout error in the measure instruction
            if 'measure' in self._target.operation_names:
                inst_props = self._target['measure'].get((qubit,))
                if inst_props is not None and inst_props.error is not None:
                    return inst_props.error
            raise ValueError(f"No readout error for qubit {qubit}")
        else:
            return self._props.readout_error(qubit)
