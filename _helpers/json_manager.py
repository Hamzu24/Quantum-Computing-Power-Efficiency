import json
import logging
from _helpers.helpers import get_unit_multiplier
from _helpers.constants import PROPERTY_UNITS

class JsonManager():
    def __init__(self, filename: str):
        self.filename = filename
        with open(filename, 'r') as file:
            self.file_data = json.load(file)

    def write(self):
        with open(self.filename, 'w') as file:
            json.dump(self.file_data, file, indent=4)
        
    def update(self, path, value, create_path=False):
        """ Update the stored file data using dot notation paths """
        
        # Navigate to nested location
        keys = path.split('.')
        current = self.file_data
        for key in keys[:-1]:
            if isinstance(current, list):
                idx = int(key[1:-1])
                current = current[idx]
                continue
                    
            if key not in current:
                if create_path:
                    current[key] = {}
                else:
                    raise ValueError("Error: specified path does not exist, and the create_path flag is set to False")

            current = current[key]

        # Set the value
        final_key = keys[-1]
        if isinstance(current, list):
            idx = int(final_key[1:-1])
            current[idx] = value
        else:
            current[final_key] = value

    def resolve(self, path: str, create_path=False):
        """ Resolve a path to the contents in the file data """
        cur_data = self.file_data
        keys = path.split('.')
        if keys[-1] == "":
            keys = keys[:-1]

        for key in keys:
            if isinstance(cur_data, list):
                idx = int(key[1:-1])
                cur_data = cur_data[idx]
                continue
                    
            if key not in cur_data:
                if create_path:
                    cur_data[key] = {}
                else:
                    raise ValueError("Error: specified path does not exist, and the create_path flag is set to False")

            cur_data = cur_data[key]

        return cur_data

    def find_path(self, target_prop_name: str, target_str: str, cur_path: str):
        """ Find a path from a list of dictionaries from a target property and target string """
        prop_list = self.resolve(cur_path)

        for j, prop in enumerate(prop_list):
            cur_path = cur_path + f"[{j}]."

            cur_prop_name = prop.get("name")
            if cur_prop_name == target_prop_name:
                return (cur_path + target_str)

            cur_path = ".".join(cur_path.split(".")[:-2]) + "."

        return None

    def find_value(self, target_prop_name: str, target_str: str, cur_path: str):
        """ Find a value from a list of dictionaries from a target property and target string """
        prop_list = self.resolve(cur_path)

        for prop in prop_list:
            cur_prop_name = prop.get("name")
            if cur_prop_name == target_prop_name:
                return prop.get(target_str)

        return None

    def get_gate_paths(self):
        path = "gates."
        gates = self.resolve(path)
        return [path + f"[{i}]." for i in range(0, len(gates))]

    def get_qubit_paths(self):
        path = "qubits."
        qubits = self.resolve(path)
        return [path + f"[{i}]." for i in range(0, len(qubits))]

    def find_value_with_units(self, target: str, path: str):
        value = self.find_value(target, "value", path)
        if value is None:
            return None

        unit = self.find_value(target, "unit", path)
        unit_multiplier = get_unit_multiplier(unit)
        if unit_multiplier is not None:
            value = value * unit_multiplier

        return value

    def update_with_units(self, prop: str, value, path: str):
        property_unit = PROPERTY_UNITS.get(prop)

        if property_unit is not None and property_unit != "":
            unit_path = self.find_path(prop, "unit", path)
            self.update(unit_path, property_unit)

            unit_multiplier = get_unit_multiplier(property_unit)
            value = value / unit_multiplier

        prop_path = self.find_path(prop, "value", path)
        self.update(prop_path, value)
