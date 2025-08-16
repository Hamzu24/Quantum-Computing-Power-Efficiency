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
from _helpers.helpers import get_control_parameters, extract_from_json
from _helpers.constants import EXISTING_MODELS, DEFAULT_INSTRUCTION_TIMES
from _helpers.registry import download_registry

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

def nm_from_fake_backend(config):
    backend_name = config.get("name")
    already_downloaded = download_registry.has_downloaded(backend_name)

    if not already_downloaded:
        exists = config_exists(backend_name)
        fetch_config_files(backend_name, exit_if_unavailable=exists)

    # Note that below I build the backend from the config files and get the backend class separately
    matching_class = get_backend_class(config, backend_name)
    create_backend_symlinks(config, matching_class)
    build_backend(config, backend_name)

    # Now, I translate the backend class into a noise model with the correct configuration from the config files
    # The config files implicitly affect the backend
    backend = matching_class()
    noise_model = NoiseModel.from_backend(backend)
    noise_model.name = config["name"]
    return noise_model
                
# Main function
def craft_noise_model(config: dict):
    config_type = config.get("type")
    if config_type is None:
        raise ValueError("A noise model must have a type field to be valid!")

    elif config_type == "fake_backend":
        return nm_from_fake_backend(config)

    elif config_type == "simple_nm":
        num_qubits, T1s, T2s, instruction_times, overrotation_amount, detuning_amount = \
            extract_from_json(config, {
                "num_qubits": (4, {}),
                "T1s": (50e3, {}),
                "T2s": (70e3, {}),
                "instruction_times": (DEFAULT_INSTRUCTION_TIMES, {}),
            })
        return custom_noise_model(num_qubits, T1s, T2s, instruction_times, overrotation_amount, detuning_amount)

    elif config_type == "random_simple_nm":
        num_qubits, seed = extract_from_json(config, {"num_qubits": (4, {}), "seed": (0, {"random": None})})
        return random_noise_model(num_qubits, seed)
    
    return None
        
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
    needed_keywords = {'conf', 'defs', 'props'}
    if dir_path.exists():
        found_keywords = {keyword for item in dir_path.iterdir() if item.is_file()
                          for keyword in needed_keywords if keyword in item.name.lower()}
        needed_keywords -= found_keywords

    return needed_keywords

def download_config_files(backend_name, commit_sha):
    url = f"https://api.github.com/repos/Qiskit/qiskit/contents/qiskit/providers/fake_provider/backends/{backend_name}?ref={commit_sha}"
    response = requests.get(url)

    if response.status_code != 200:
        logging.error(f"Unable to find files for the backend {backend_name} from the url {url}.")
        if exit_if_unavailable:
            raise requests.exceptions.HTTPError(
                f"HTTP: {response.status_code} for {url}"
            )
        return None
    else:
        files = response.json()
        return files

def create_original_props(dir_path, props_filename):
    try:
        props_path = Path(props_filename)
        original_props_filename = dir_path / f"{props_path.stem}_original.json"
        subprocess.run(["cp", props_path, original_props_filename])
    except subprocess.CalledProcessError as e:
        logging.critical(f"The error {e} was thrown when creating the original props file!")

def write_needed_files(files, dir_path: Path, needed_keywords: list):
    # Create output folder
    os.makedirs(str(dir_path), exist_ok=True)
    
    # Download files containing keywords
    props_filename = None
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
                props_filename = filename
                
                # When downloading the props file, make sure to copy it onto an original_props!
                create_original_props(dir_path, props_filename)

    # Make sure to still get props_filename if no files are needed
    if not needed_keywords:
        props_filename
        props_files = {item for item in dir_path.iterdir() if item.is_file()
                       and "props" in item.name.lower() and "original" not in item.name.lower()}
        assert len(props_files) == 1
        props_filename = next(iter(props_files))
        print(f"props filename found! {props_filename}")

    return props_filename

def reset_props_file(props_filename, dir_path):
    # Copy the props file to reset any changes
    if props_filename:
        try:
            props_path = Path(props_filename)
            original_props_filename = dir_path / f"{props_path.stem}_original.json"
            subprocess.run(["cp", original_props_filename, props_path])
        except subprocess.CalledProcessError as e:
            logging.critical(f"The error {e} was thrown when copying the original props file!")
    else:
        logging.critical(f"The props file which is necessary to configuration of fake backends was not found from the qiskit repository")

def fetch_config_files(backend_name, exit_if_unavailable=True):
    save_location = os.environ.get("BACKEND_CONFIGS_FOLDER") + backend_name
    dir_path = Path(save_location)
    needed_keywords = get_needed_files(dir_path)

    # Get files from GitHub API
    CURRENT_BRANCH = "stable/0.46"
    latest_commit_sha = get_commit_sha_for_branch("Qiskit", "qiskit", CURRENT_BRANCH)
    if latest_commit_sha is None:
        if exit_if_unavailable:
            raise Exception("The commit SHA was not found from the branch. Exiting because exit_if_unavailable is True")
        else:
            logging.error("Returning from download_config without downloading anything")
            return None
    
    files = download_config_files(backend_name, latest_commit_sha)
    if files:
        props_filename = write_needed_files(files, dir_path, needed_keywords)
    
    # The props file is currently the only one that is configured
    reset_props_file(props_filename, dir_path)
    

# Noise models not from backend functions below
        
def custom_noise_model(num_qubits = 4, T1s = 50e3, T2s = 70e3, instruction_times: dict = None, overrotation_amount = np.pi/100, detuning_amount = np.pi/120):

    # Truncate T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(num_qubits)])

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

    return noise_model

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

    return noise_model

