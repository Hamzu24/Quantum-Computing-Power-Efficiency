import numpy as np
import os
import requests
import json
from qiskit_ibm_runtime.fake_provider import (
    FakeAuckland, FakeGeneva, FakeKolkataV2, FakeManilaV2,
    FakeMontrealV2, FakeOslo, FakePerth, FakePrague,
    FakeSherbrooke, FakeTokyo
)
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

EXISTING_MODELS = {
    "fakeAuckland": FakeAuckland,
    "fakeGeneva": FakeGeneva,
    "fakeKolkataV2": FakeKolkataV2,
    "fakeManilaV2": FakeManilaV2,
    "fakeMontrealV2": FakeMontrealV2,
    "fakeOslo": FakeOslo,
    "fakePerth": FakePerth,
    "fakePrague": FakePrague,
    "fakeSherbrooke": FakeSherbrooke,
    "fakeTokyo": FakeTokyo
}

DEFAULT_INSTRUCTION_TIMES = {
    "time_rz": 0,
    "time_sx": 50,
    "time_x": 100,
    "time_cx": 300,
    "time_reset": 1000,
    "time_measure": 100
}

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
        cur_val = json_dict.get("key")
        if cur_val is None:
            if key in required_values:
                raise ValueError(f"Required value {key} was not in json!")

            cur_val = data[0]
            continue

        elif cur_val in exceptions.keys():
            cur_val = exceptions.get(cur_val)
        
        return_values.append(cur_val)

    return return_values

def get_dynamic_backend(backend_name, parent_class):

    class DynamicFakeBackend(parent_class):
        conf_path_base = f"qiskit_backend_configs/{backend_name}/"
        conf_filename = conf_path_base + f"conf_{backend_name}.json"
        props_filename = conf_path_base + f"props_{backend_name}.json"
        defs_filename = conf_path_base + f"defs_{backend_name}.json"

        def _load_json(self, filename):
            with open(filename) as f_json:
                the_json = json.load(f_json)
            return the_json
    
    return DynamicFakeBackend

# Main function
def craft_noise_model(config: dict):
    config_type = config.get("type")
    if config_type is None:
        return None

    elif config_type == "fake_backend":
        backend_name = config["name"]
        download_config(backend_name, True)

        match = next((item for item in EXISTING_MODELS if backend_name in item.lower()), None)
        if match is None:
            raise ValueError("The specified fake backend cannot be found in the supported models!")

        matching_class = EXISTING_MODELS[match]

        backend = get_dynamic_backend(backend_name, matching_class)()
        noise_model = NoiseModel.from_backend(backend)
        noise_model.name = config["name"]
        return noise_model

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
        raise ValueError("instruction times did not include all of the necessary fields to create a noise model!")

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

# GPT function
def get_commit_sha_for_branch(owner, repo, branch):
    """Get the commit SHA for a branch with slashes in the name."""
    encoded_branch = branch.replace('/', '%2F')
    url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{encoded_branch}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['object']['sha']
    else:
        raise Exception(f"Branch not found: {response.status_code}")

def download_config(backend_name, silent):

    # Get files from GitHub API
    latest_commit_sha = get_commit_sha_for_branch("Qiskit", "qiskit", "stable/0.46")
    url = f"https://api.github.com/repos/Qiskit/qiskit/contents/qiskit/providers/fake_provider/backends/{backend_name}?ref={latest_commit_sha}"
    response = requests.get(url)
    files = response.json()

    save_location = f"qiskit_backend_configs/{backend_name}"
    
    # Create output folder
    os.makedirs(save_location, exist_ok=True)
    
    # Download files containing keywords
    keywords = ['conf', 'defs', 'props']
    for file_info in files:
        filename = file_info['name']
        if any(keyword in filename.lower() for keyword in keywords):
            # Download file
            file_response = requests.get(file_info['download_url'])
            with open(save_location + f"/{filename}", 'wb') as f:
                f.write(file_response.content)
            if not silent:
                print(f"Downloaded: {filename}")
