import numpy as np
import os
import subprocess
import requests
import json
from pathlib import Path
import inspect
import copy
from typing import Any
from qiskit.circuit.library import RZXGate, RZGate, RXGate, RZZGate
from qiskit.quantum_info import Operator, average_gate_fidelity
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    pauli_error,
    depolarizing_error,
    thermal_relaxation_error,
    coherent_unitary_error
)
from qiskit.providers.models import (
    BackendProperties,
)
from _helpers.builders.builder_wrapper import BuilderWrapper
import logging
from _helpers.helpers import get_basis_gates_from_backend, get_control_parameters, extract_from_json, get_config_value
from _helpers.constants import (
    EXISTING_MODELS, DEFAULT_INSTRUCTION_TIMES,
    DefaultBasisGatesNoiseless, DefaultBasisGates1qb, DefaultBasisGates2qb,
    NOISELESS_GATES, SINGLE_QUBIT_GATES, TWO_QUBIT_GATES
)

# Main function
def craft_noise_model(config: dict):
    config_type = config.get("type")
    if config_type is None:
        raise ValueError("A noise model must have a type field to be valid!")

    elif config_type == "fake_backend":
        pauli_twirling = config.get("pauli_twirling", True)
        if pauli_twirling:
            return nm_from_fake_backend(config)
        else:
            return nm_from_fake_backend_no_twirl(config)

    elif config_type == "arrhenius":
        return nm_from_arrhenius(config)

    elif config_type == "simple_nm":
        num_qubits, T1s, T2s, instruction_times, overrotation_amount, detuning_amount = \
            extract_from_json(config, {
                "num_qubits": (4, {}),
                "T1s": (50e3, {}),
                "T2s": (70e3, {}),
                "instruction_times": (DEFAULT_INSTRUCTION_TIMES, {}),
            })
        return simple_custom_nm(num_qubits, T1s, T2s, instruction_times, overrotation_amount, detuning_amount), None

    elif config_type == "random_simple_nm":
        num_qubits, seed = extract_from_json(config, {"num_qubits": (4, {}), "seed": (0, {"random": None})})
        return random_noise_model(num_qubits, seed), None
    
    return None
        
def nm_from_fake_backend(config):
    backend_name = config.get("name")

    exists = config_exists(backend_name)
    fetch_config_files(backend_name, exit_if_unavailable=exists)

    # Note that below I build the backend from the config files and get the backend class separately
    matching_class = get_backend_class(config, backend_name)
    create_backend_symlinks(config, matching_class)
    build_backend(config, backend_name)
    control_parameters = get_control_parameters(config)

    # Temperature in from_backend determines the target excitation of the qubits asymptotic drift. This only has a minor effect.
    # 0 is the default value, where the target is just |0>

    temperature = get_config_value(config, "temperature")

    # Now, I translate the backend class into a noise model with the correct configuration from the config files
    # The config files implicitly affThis only has a minor effect.
    backend = matching_class()
    noise_model = NoiseModel.from_backend(
        backend,
        gate_error=True,
        readout_error=True,
        temperature=temperature
    )
    noise_model.name = config["name"]
    return noise_model, backend

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


def nm_from_fake_backend_no_twirl(config):
    """
    Creates a noise model from a fake backend WITHOUT Pauli twirling approximation.

    Instead of using depolarizing errors (which are Pauli channels), this function
    uses coherent unitary errors to model gate miscalibration. This provides a more
    physically accurate model for structured circuits where coherent errors can
    accumulate systematically rather than averaging out.

    The noise model includes:
    - Thermal relaxation errors (full Kraus operators, not twirled)
    - Coherent over-rotation errors (unitary, not Pauli)
    - Readout errors (classical bit-flip)

    Use this when:
    - Running structured circuits (VQE, QAOA) where coherent errors accumulate
    - You need more accurate absolute error predictions
    - Studying error accumulation in variational algorithms

    Use the standard nm_from_fake_backend when:
    - Running random circuits (Quantum Volume, RB)
    - You want faster simulation
    - Relative comparisons are sufficient

    Config options:
    - "pauli_twirling": false to enable this function
    - "two_qubit_error_model": 'zz' (default), 'zx', or 'xx'

    Limitations:
    - Two-qubit coherent error assumes ZZ model (appropriate for IBM CR gates)
    - For other architectures (ion traps, flux-tunable), change two_qubit_error_model
    - Single error axis per gate type (real errors vary per qubit pair)
    - Does not model leakage or crosstalk
    """
    # ===========================================
    # BACKEND SETUP
    # ===========================================
    backend_name = config.get("name")
    two_qubit_error_model = config.get("two_qubit_error_model", "zz")

    exists = config_exists(backend_name)
    fetch_config_files(backend_name, exit_if_unavailable=exists)

    matching_class = get_backend_class(config, backend_name)
    create_backend_symlinks(config, matching_class)
    build_backend(config, backend_name)
    control_parameters = get_control_parameters(config)

    logging.info(f"Building noise model without twirling for backend: {backend_name}")
    logging.info(f"Two-qubit coherent error model: {two_qubit_error_model}")

    backend = matching_class()
    props = backend.properties()

    if hasattr(backend, 'num_qubits'):
        n_qubits = backend.num_qubits
    else:
        n_qubits = backend.configuration().n_qubits

    noise_model = NoiseModel()

    basis_gates = set(get_basis_gates_from_backend(backend))
    logging.info(f"Basis gates for {backend_name}: {basis_gates}")

    single_qubit_gates = (basis_gates & SINGLE_QUBIT_GATES) - NOISELESS_GATES
    two_qubit_gates = basis_gates & TWO_QUBIT_GATES

    logging.info(f"1-qubit gates with noise: {single_qubit_gates}")
    logging.info(f"2-qubit gates: {two_qubit_gates}")

    # ===========================================
    # ONE-QUBIT GATE ERRORS
    # ===========================================
    for qubit in range(n_qubits):
        try:
            t1 = props.t1(qubit)
            t2 = min(props.t2(qubit), 2 * t1)
        except Exception:
            raise ValueError(f"Could not find T1/T2 times for qubit {qubit}")

        for gate in single_qubit_gates:
            try:
                gate_time = props.gate_length(gate, qubit)
            except Exception:
                raise ValueError(f"No gate length for {gate} on qubit {qubit}!")

            try:
                gate_error = props.gate_error(gate, qubit)
            except Exception:
                raise ValueError(f"No gate error for {gate} on qubit {qubit}!")

            errors_to_compose = []

            thermal_err = thermal_relaxation_error(t1, t2, gate_time)
            errors_to_compose.append(thermal_err)
            relax_infidelity = 1 - average_gate_fidelity(thermal_err)

            # Coherent error (REPLACES Qiskit's depolarizing_error)
            remaining_error = max(0, gate_error - relax_infidelity)
            if remaining_error > 1e-10 and gate not in ['measure', 'reset']:
                theta = _infidelity_to_angle_1q(remaining_error)
                U_err = _single_qubit_coherent_unitary(theta)
                errors_to_compose.append(coherent_unitary_error(U_err))

            combined = errors_to_compose[0]
            for err in errors_to_compose[1:]:
                combined = combined.compose(err)
            noise_model.add_quantum_error(combined, gate, [qubit])

        # Readout error
        try:
            p = props.readout_error(qubit)
            probs = [[1 - p, p], [p, 1 - p]]
            noise_model.add_readout_error(ReadoutError(probs), [qubit])
        except Exception:
            raise ValueError(f"Qubit {qubit} has no defined readout error!")

    # ===========================================
    # TWO-QUBIT GATE ERRORS
    # ===========================================
    if hasattr(backend, 'coupling_map'):
        coupling_map = backend.coupling_map
    else:
        coupling_map = backend.configuration().coupling_map

    if coupling_map is not None:
        for qubits in coupling_map:
            q0, q1 = qubits[0], qubits[1]

            try:
                t1_0 = props.t1(q0)
                t2_0 = min(props.t2(q0), 2 * t1_0)
                t1_1 = props.t1(q1)
                t2_1 = min(props.t2(q1), 2 * t1_1)
            except Exception:
                raise ValueError(f"Could not find T1/T2 times for qubits {qubits}")

            for gate in two_qubit_gates:
                try:
                    gate_time = props.gate_length(gate, qubits)
                except Exception:
                    raise ValueError(f"No gate length for {gate} on qubits {qubits}!")

                try:
                    gate_error = props.gate_error(gate, qubits)
                except Exception:
                    raise ValueError(f"No gate error for {gate} on qubits {qubits}!")

                errors_to_compose = []

                thermal_err_0 = thermal_relaxation_error(t1_0, t2_0, gate_time)
                thermal_err_1 = thermal_relaxation_error(t1_1, t2_1, gate_time)
                thermal_err_2q = thermal_err_0.expand(thermal_err_1)
                errors_to_compose.append(thermal_err_2q)

                relax_infidelity = 1 - average_gate_fidelity(thermal_err_2q)

                remaining_error = max(0, gate_error - relax_infidelity)
                if remaining_error > 1e-10:
                    theta = _infidelity_to_angle_2q(remaining_error)
                    U_err = _two_qubit_coherent_unitary(theta, model=two_qubit_error_model)
                    errors_to_compose.append(coherent_unitary_error(U_err))

                combined = errors_to_compose[0]
                for err in errors_to_compose[1:]:
                    combined = combined.compose(err)
                noise_model.add_quantum_error(combined, gate, qubits)

    # ===========================================
    # FINALIZATION
    # ===========================================
    noise_model.name = config["name"] + "_no_twirl"
    logging.info(f"Created no-twirl noise model with {n_qubits} qubits")

    return noise_model, backend


def fetch_config_files(backend_name, exit_if_unavailable=True):
    save_location = os.environ.get("BACKEND_CONFIGS_FOLDER") + backend_name
    dir_path = Path(save_location)
    needed_keywords = get_needed_files(dir_path)

    # Get files from GitHub API
    if needed_keywords:
        CURRENT_BRANCH = "stable/0.46"
        latest_commit_sha = get_commit_sha_for_branch("Qiskit", "qiskit", CURRENT_BRANCH)
        if latest_commit_sha is None:
            if exit_if_unavailable:
                raise Exception("The commit SHA was not found from the branch. Exiting because exit_if_unavailable is True")
            else:
                logging.error("Returning from download_config without downloading anything")
                return None
    
        files = download_github_backend_files(backend_name, latest_commit_sha)
        write_needed_files(files, dir_path, needed_keywords)
    
    # The props file is currently the only one that is configured
    reset_props_file(dir_path)

def config_exists(backend_name):
    config_folder = os.environ.get("BACKEND_CONFIGS_FOLDER") + backend_name
    res = subprocess.run(["ls", config_folder], capture_output=True)
    if res.stderr:
        return False
    filenames = res.stdout.decode().split("\n")
    if "props" not in filenames:
        return False
    return True

def create_backend_symlinks(config, matching_class):
    backend_name = config.get("name")
    target_dir = Path(inspect.getsourcefile(matching_class)).parent
    source_dir = Path(os.environ.get("BACKEND_CONFIGS_FOLDER") + backend_name + "/")

    for item in source_dir.iterdir():
        if "original" in item.name:
            # No need to create symlinks to the original/copy files
            continue

        symlink_path = target_dir / item.name
        
        if symlink_path.exists():
            continue
        
        symlink_path.symlink_to(item.absolute())

def get_backend_class(config, backend_name):
    match = next((item for item in EXISTING_MODELS if backend_name in item.lower()), None)
    if match is None:
        raise ValueError("The specified fake backend cannot be found in the supported models!")

    matching_class = EXISTING_MODELS[match]
    return matching_class

# This builds the backend using the config files
def build_backend(config, backend_name):
    init_control_parameters = config.get("init_control_parameters")
    builder = BuilderWrapper(backend_name, init_control_parameters)
    control_parameters = get_control_parameters(config)
    builder.build_backend(control_parameters)

# GPT function
def get_commit_sha_for_branch(owner, repo, branch):
    """Get the commit SHA for a branch with slashes in the name."""
    encoded_branch = branch.replace('/', '%2F')
    url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{encoded_branch}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['object']['sha']
    else:
        logging.error(f"Branch not found: {response.status_code}")
        return None

def get_needed_files(dir_path: Path):
    needed_keywords = {'conf', 'defs', 'props', 'original'}
    if dir_path.exists():
        found_keywords = {keyword for item in dir_path.iterdir() if item.is_file()
                          for keyword in needed_keywords if keyword in item.name.lower()}

        # temp below
        # id = dir_path.iterdir()
        # for i, item in enumerate(id):
            # print(f"{i} item is: {item}")
        #
        needed_keywords -= found_keywords

    # Handle the edge case of props existing but not original
    if 'original' in needed_keywords:
        needed_keywords.add("props")
        needed_keywords.remove("original")

    return needed_keywords

def download_github_backend_files(backend_name, commit_sha):
    url = f"https://api.github.com/repos/Qiskit/qiskit/contents/qiskit/providers/fake_provider/backends/{backend_name}?ref={commit_sha}"
    response = requests.get(url)

    if response.status_code != 200:
        logging.error(f"Unable to find files for the backend {backend_name} from the url {url}.")
        # Note: exit_if_unavailable is not available in this scope, just return None
        return None
    else:
        files = response.json()
        return files

def create_original_props(dir_path, props_filename):
    try:
        props_path = dir_path / Path(props_filename)
        original_props_filename = dir_path / f"{props_path.stem}_original.json"
        subprocess.run(["cp", str(props_path), str(original_props_filename)])
    except subprocess.CalledProcessError as e:
        logging.critical(f"The error {e} was thrown when creating the original props file!")

def write_needed_files(files, dir_path: Path, needed_keywords: list):
    # Create output folder
    os.makedirs(str(dir_path), exist_ok=True)
    
    # Download files containing keywords
    for file_info in files:
        filename = file_info['name']
        if any(keyword in filename.lower() for keyword in needed_keywords):
            # Download file
            file_response = requests.get(file_info['download_url'])
            file_path = str(dir_path / filename)
            with open(file_path, 'wb') as f:
                f.write(file_response.content)
            logging.info(f"Downloaded: {filename}")

            if "props" in filename:
                create_original_props(dir_path, filename)

def get_props_filename(dir_path):
    props_files = {item for item in dir_path.iterdir() if item.is_file()
                   and "props" in item.name.lower() and "original" not in item.name.lower()}
    if len(props_files) != 1:
        raise FileNotFoundError("Could not isolate the props file in the directory. Please manually clean")
    
    props_filename = next(iter(props_files))
    return props_filename

def reset_props_file(dir_path):
    props_filename = get_props_filename(dir_path)
    
    # Copy the props file to reset any changes
    if props_filename:
        try:
            props_path = Path(props_filename)
            original_props_filename = dir_path / f"{props_path.stem}_original.json"
            subprocess.run(["cp", str(original_props_filename), str(props_path)])
        except subprocess.CalledProcessError as e:
            logging.critical(f"The error {e} was thrown when copying the original props file!")
    else:
        logging.critical(f"The props file which is necessary to configuration of fake backends was not found from the qiskit repository")

# Noise models not from backend functions are below

class CustomNmClass():
    basis_gates = DefaultBasisGates2qb + DefaultBasisGates1qb + DefaultBasisGatesNoiseless
    version = -1

def nm_from_arrhenius(config):
    """
    Returns a NoiseModel where T1 is derived from the Arrhenius error probability.
    This accurately models the 'decay' to |0> rather than just scrambling.
    """
    k_b = 1.380649e-23
    h   = 6.626070e-34

    control_parameters = get_control_parameters(config)

    T = get_config_value(control_parameters, "temperature")
    gate_length = config.get("gate_length", 50e-9) # Default 50ns gate
    
    if "qubit_frequency_hz" in config:
        E = h * config["qubit_frequency_hz"]
    else:
        E = h * 5e9 

    p_phys = np.exp(-E / (k_b * T)) # Use simple arrhenius probability
    
    if p_phys <= 0:
        T1 = np.inf
        T2 = np.inf
    else:
        # Assume T1 ≈ gate_length / p_phys
        T1 = gate_length / p_phys
        
        # Assume T2 = 2*T1
        T2 = 2 * T1

    noise_model = NoiseModel()
    
    error_1q = thermal_relaxation_error(T1, T2, gate_length)

    gate_length_2q = 4 * gate_length 
    error_2q_single = thermal_relaxation_error(T1, T2, gate_length_2q)
    error_2q = error_2q_single.tensor(error_2q_single)

    noise_model.add_all_qubit_quantum_error(error_1q, DefaultBasisGates1qb)
    noise_model.add_all_qubit_quantum_error(error_2q, DefaultBasisGates2qb)

    return noise_model, CustomNmClass

def simple_custom_nm(num_qubits = 4, T1s = 50e3, T2s = 70e3, instruction_times: dict = None, overrotation_amount = np.pi/100, detuning_amount = np.pi/120):

    # Convert scalars to arrays if needed
    if isinstance(T1s, (int, float)):
        T1s = np.full(num_qubits, T1s)
    if isinstance(T2s, (int, float)):
        T2s = np.full(num_qubits, T2s)

    # Truncate T2s <= 2*T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(num_qubits)])

    # Use DEFAULT_INSTRUCTION_TIMES if not provided
    if instruction_times is None:
        instruction_times = DEFAULT_INSTRUCTION_TIMES

    # Instruction times (in nanoseconds)
    time_rz = instruction_times.get("time_rz")
    time_sx = instruction_times.get("time_sx")
    time_x = instruction_times.get("time_x")
    time_cx = instruction_times.get("time_cx")
    time_reset = instruction_times.get("time_reset")
    time_measure = instruction_times.get("time_measure")

    if time_rz is None or time_sx is None or time_x is None or time_cx is None or time_reset is None or time_measure is None:
        raise ValueError("The instruction times extracted from the configuration did not include all the necessary fields to create a custom noise model!")

    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                    for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                    for t1, t2 in zip(T1s, T2s)]
    errors_u1  = [thermal_relaxation_error(t1, t2, time_rz)
                for t1, t2 in zip(T1s, T2s)]
    errors_u2  = [thermal_relaxation_error(t1, t2, time_sx)
                for t1, t2 in zip(T1s, T2s)]
    errors_u3  = [thermal_relaxation_error(t1, t2, time_x)
                for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
                thermal_relaxation_error(t1b, t2b, time_cx))
                for t1a, t2a in zip(T1s, T2s)]
                for t1b, t2b in zip(T1s, T2s)]

    noise_model = NoiseModel()

    overrotation_amount = np.pi/100
    detuning_amount = np.pi/120
    overrotation_unitary_1q = RXGate(overrotation_amount).to_matrix()
    detuning_unitary_1q = RZGate(detuning_amount).to_matrix()
    sx_gate_overrotation_error = coherent_unitary_error(overrotation_unitary_1q)
    sx_gate_detuning_error = coherent_unitary_error(detuning_unitary_1q)

    coherent_unitary_2q = RZXGate(overrotation_amount).to_matrix()
    zz_unitary_2q = RZZGate(overrotation_amount).to_matrix()
    coherent_unitary_2q_error = coherent_unitary_error(coherent_unitary_2q)
    zz_2q_error = coherent_unitary_error(zz_unitary_2q)


    # Add errors to noise model
    noise_model = NoiseModel()
    for j in range(num_qubits):
        noise_model.add_quantum_error(errors_reset[j], "reset", [j]) # maybe specify amplitude damping and dephasing separately
        noise_model.add_quantum_error(errors_measure[j], "measure", [j])
        noise_model.add_quantum_error(errors_u1[j], "rz", [j])
        noise_model.add_quantum_error(errors_u2[j], "sx", [j])
        # noise_model.add_quantum_error(errors_u3[j], "x", [j])
        noise_model.add_quantum_error(sx_gate_overrotation_error, ['sx'], [j], warnings=False)
        noise_model.add_quantum_error(sx_gate_detuning_error, ['sx'], [j], warnings=False)
        noise_model.add_quantum_error(depolarizing_error(0.0005,1), ['sx'], [j], warnings=False)
        noise_model.add_quantum_error(errors_u2[j], "id", [j])
        #add detuning error maybe tlak to abhishek
        for k in range(num_qubits):
            noise_model.add_quantum_error(errors_cx[j][k], "cx", [j, k])
            noise_model.add_quantum_error(depolarizing_error(0.005,2), ["cx"], [j, k], warnings=False)
            noise_model.add_quantum_error(coherent_unitary_2q_error, ["cx"], [j, k], warnings=False)
            noise_model.add_quantum_error(zz_2q_error, ["cx"], [j, k], warnings=False)

    return noise_model, CustomNmClass

def random_noise_model(num_qubits = 4, seed = 0):

    np.random.seed(seed)

    T1s = np.random.normal(50e3, 1e3, num_qubits) # Sampled from normal distribution mean 50 microsec
    T2s = np.random.normal(70e3, 1e3, num_qubits)  # Sampled from normal distribution mean 50 microsec

    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(num_qubits)])


    # Instruction times (in nanoseconds)
    time_rz = 0   # virtual gate
    time_sx = 50  # (single X90 pulse)
    time_x = 100 # (two X90 pulses)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000 # 1 microsecond


    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                    for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                    for t1, t2 in zip(T1s, T2s)]
    errors_u1  = [thermal_relaxation_error(t1, t2, time_rz)
                for t1, t2 in zip(T1s, T2s)]
    errors_u2  = [thermal_relaxation_error(t1, t2, time_sx)
                for t1, t2 in zip(T1s, T2s)]
    errors_u3  = [thermal_relaxation_error(t1, t2, time_x)
                for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
                thermal_relaxation_error(t1b, t2b, time_cx))
                for t1a, t2a in zip(T1s, T2s)]
                for t1b, t2b in zip(T1s, T2s)]

    noise_model = NoiseModel()

    detuning_mean = 0.0
    detuning_std = np.pi/150          # ~1.2 degrees std dev
    overrotation_mean = 0.0           
    overrotation_std = np.pi/100      # ~1.8 degrees std dev

    overrotation_amount = np.random.normal(overrotation_mean, overrotation_std)
    detuning_amount = np.random.normal(detuning_mean, detuning_std)
    overrotation_unitary_1q = RXGate(overrotation_amount).to_matrix()
    detuning_unitary_1q = RZGate(detuning_amount).to_matrix()
    sx_gate_overrotation_error = coherent_unitary_error(overrotation_unitary_1q)
    sx_gate_detuning_error = coherent_unitary_error(detuning_unitary_1q)

    coherent_unitary_2q = RZXGate(overrotation_amount).to_matrix()
    zz_unitary_2q = RZZGate(overrotation_amount).to_matrix()
    coherent_unitary_2q_error = coherent_unitary_error(coherent_unitary_2q)
    zz_2q_error = coherent_unitary_error(zz_unitary_2q)


    # Add errors to noise model
    noise_model = NoiseModel()
    for j in range(num_qubits):
        noise_model.add_quantum_error(errors_reset[j], "reset", [j]) # maybe specify amplitude damping and dephasing separately
        noise_model.add_quantum_error(errors_measure[j], "measure", [j])
        noise_model.add_quantum_error(errors_u1[j], "rz", [j])
        noise_model.add_quantum_error(errors_u2[j], "sx", [j])
        # noise_model.add_quantum_error(errors_u3[j], "x", [j])
        noise_model.add_quantum_error(sx_gate_overrotation_error, ['sx'], [j], warnings=False)
        noise_model.add_quantum_error(sx_gate_detuning_error, ['sx'], [j], warnings=False)
        noise_model.add_quantum_error(depolarizing_error(0.0005,1), ['sx'], [j], warnings=False)
        noise_model.add_quantum_error(errors_u2[j], "id", [j])
        #add detuning error maybe tlak to abhishek
        for k in range(num_qubits):
            noise_model.add_quantum_error(errors_cx[j][k], "cx", [j, k])
            noise_model.add_quantum_error(depolarizing_error(0.005,2), ["cx"], [j, k], warnings=False)
            noise_model.add_quantum_error(coherent_unitary_2q_error, ["cx"], [j, k], warnings=False)
            noise_model.add_quantum_error(zz_2q_error, ["cx"], [j, k], warnings=False)

    return noise_model, CustomNmClass

