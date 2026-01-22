import numpy as np
import os
import subprocess
import requests
import json
from pathlib import Path
import inspect
import copy
from typing import Any
from qiskit.circuit.library import RZXGate, RZGate, RXGate, RZZGate, RXXGate
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
from _helpers.helpers import get_basis_gates_from_backend, get_control_parameters, extract_from_json, get_config_value, _infidelity_to_angle_1q, _infidelity_to_angle_2q, _single_qubit_coherent_unitary, _two_qubit_coherent_unitary, BackendPropertiesAdapter
from _helpers.constants import (
    EXISTING_MODELS, DEFAULT_INSTRUCTION_TIMES,
    DefaultBasisGatesNoiseless, DefaultBasisGates1qb, DefaultBasisGates2qb,
    NOISELESS_GATES, SINGLE_QUBIT_GATES, TWO_QUBIT_GATES,
    StimBasisGates
)
from _helpers.noise_models.base import CustomNoiseModelBackend
from _helpers.noise_models import noise_model_registry, NoiseModelWrapper
from typing import Type, Tuple
from json_manager import JsonManager

# Main function
def craft_noise_model(config: dict, pauli_mode: bool = False):
    config_type = config.get("type")
    if config_type is None:
        raise ValueError("A noise model must have a type field to be valid!")

    # Fake backend noise models are handled outside the registry!
    if config_type == "fake_backend":
        if pauli_mode:
            return pauli_nm_from_fake_backend(config)

        pauli_twirling = config.get("pauli_twirling", True)
        if pauli_twirling:
            return nm_from_fake_backend(config)
        else:
            return nm_from_fake_backend_no_twirl(config)

    if config_type in noise_model_registry:
        wrapper = NoiseModelWrapper(config)
        if pauli_mode:
            return wrapper.build_pauli()
        return wrapper.build()

    raise ValueError(f"Unknown noise model type: {config_type}. "
                     f"Available types: fake_backend, {noise_model_registry.list_factories()}")


def _prepare_fake_backend(config):
    backend_name = config.get("name")

    exists = config_exists(backend_name)
    fetch_config_files(backend_name, exit_if_unavailable=exists)

    matching_class = get_backend_class(config, backend_name)
    create_backend_symlinks(config, matching_class)
    build_backend(config, backend_name)

    backend = matching_class()
    return backend, backend_name


def nm_from_fake_backend(config):
    backend, backend_name = _prepare_fake_backend(config)

    # Temperature determines the target excitation of qubits' asymptotic drift.
    # 0 is the default value, where the target is just |0>
    temperature = get_config_value(config, "temperature")

    noise_model = NoiseModel.from_backend(
        backend,
        gate_error=True,
        readout_error=True,
        temperature=temperature
    )
    noise_model.name = backend_name
    return noise_model, backend

def pauli_nm_from_fake_backend(config) -> Tuple[dict, CustomNoiseModelBackend]:
    """
        Tuple of (pauli_config, CustomNoiseModelBackend) where pauli_config has structure:
        [
            {'1q': {'p_x': float, 'p_y': float, 'p_z': float},
            }
            ...
            with one dictionary for each qubit
        ]
            {
                '1q': {'p_x': float, 'p_y': float, 'p_z': float},
                '2q': {'p_x': float, 'p_y': float, 'p_z': float},
                'measurement': {'p_flip': float}
            }
    """
    backend, backend_name = _prepare_fake_backend(config)

    pauli_config = {
        '1q': {'p_x': 0.0, 'p_y': 0.0, 'p_z': 0.0},
        '2q': {'p_x': 0.0, 'p_y': 0.0, 'p_z': 0.0},
        'measurement': {'p_flip': 0.0}
    }
    return pauli_config, CustomNoiseModelBackend(StimBasisGates)

def nm_from_fake_backend_no_twirl(config):
    """
    Create noise model from fake backend WITHOUT Pauli twirling approximation.

    Uses coherent unitary errors instead of depolarizing errors for more
    physically accurate modeling of structured circuits.

    Config options:
    - "pauli_twirling": false to enable this function
    - "two_qubit_error_model": 'zz' (default), 'zx', or 'xx'

    Limitations:
    - Two-qubit coherent error assumes ZZ model (appropriate for IBM CR gates)
    - For other architectures (ion traps, flux-tunable), change two_qubit_error_model
    - Single error axis per gate type (real errors vary per qubit pair)
    - Does not model leakage or crosstalk
    """
    backend, backend_name = _prepare_fake_backend(config)
    two_qubit_error_model = config.get("two_qubit_error_model", "zz")

    logging.info(f"Building noise model without twirling for backend: {backend_name}")
    logging.info(f"Two-qubit coherent error model: {two_qubit_error_model}")

    props = BackendPropertiesAdapter(backend)

    if hasattr(backend, 'num_qubits'):
        n_qubits = backend.num_qubits
    elif hasattr(backend, 'target') and backend.target is not None:
        n_qubits = backend.target.num_qubits
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

# Custom noise model implementations are now in _helpers/noise_models/
# Available types: arrhenius, simple_nm, random_simple_nm
