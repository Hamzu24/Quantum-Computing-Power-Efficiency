import gc
import _helpers.circuit_submitter
import _helpers.enhanced_circuit_submitter
from copy import copy
import runpy
import json

PATH = "tutorials/circuit_execution_quality_metrics/quantum_volume/quantum_volume.py"
power_config_path = "power_config.json"
devices_needed = []

def get_submitters(objects):
    submitters = []
    devices_still_needed = copy(devices_needed)

    for obj in objects:
        if isinstance(obj, _helpers.circuit_submitter.CircuitSubmitter):
            if obj.device_name in devices_needed:
                submitters.append(obj)
                if obj.device_name in devices_still_needed:
                    devices_still_needed.remove(obj.device_name)

    return submitters

module_globals = runpy.run_path(PATH, run_name="__main__")

gc.collect()
objects = gc.get_objects()

submitters = get_submitters(objects)