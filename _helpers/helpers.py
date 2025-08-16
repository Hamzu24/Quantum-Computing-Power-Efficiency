import json
import logging
import os
from typing import Any
from _helpers.constants import SI_PREFIXES

def read_config():
    CONFIG_PATH = os.environ.get("CONFIG_PATH")

    try:
        with open(CONFIG_PATH, 'r') as f:
            config_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Power config file not found")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in power config file: {e}")
    
    return config_data 

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
    
    print(f"control parameters: {control_parameters}")
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

def get_unit_multiplier(unit: str):
    if len(unit) == 0:
        return None

    return SI_PREFIXES.get(unit[0])

def get_config_value(config: dict, name: str):
    val, unit = config.get(name)
    multiplier = get_unit_multiplier(unit)
    return val * multiplier

def get_num_qubits_list():
    if os.environ.get("NUM_QUBITS") is not None:
        try:
            num_qubits_list = [int(n) for n in os.environ.get("NUM_QUBITS").split(",")]
        except ValueError as e:
            logging.error("Unable to parse a list of qubits from the environment variable. Falling back on the default of [5]")
            num_qubits_list = [5]
    else:
        num_qubits_list = [5]

    return num_qubits_list

def set_num_qubits_list():
    config_data = read_config()
    num_qubits_list = config_data.get("num_qubits")
    if num_qubits_list is None:
        logging.error("No number of qubits specified for the simulation. Running with a default of 5")
        num_qubits_list = 5
    
    os.environ["NUM_QUBITS"] = num_qubits_list
    return num_qubits_list
