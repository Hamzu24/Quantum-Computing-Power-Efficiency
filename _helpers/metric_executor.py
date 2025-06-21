from importlib.util import spec_from_file_location, module_from_spec
import pathlib
import gc
import _helpers.enhanced_circuit_submitter
from _helpers.circuit_submitter import CircuitSubmitter
from copy import copy

PATH = "../tutorials/circuit_execution_quality_metrics/quantum_volume/quantum_volume.py"
metric_module = None
devices_needed = []

def get_submitters(objects):
    submitters = []
    devices_still_needed = copy(devices_needed)

    for obj in object:
        if isinstance(obj, CircuitSubmitter):
            if object.device_name in devices_needed:
                submitters.append(obj)
                if object.device_name in devices_still_needed:
                    devices_still_needed.remove(obj.device_name)

    return submitters


spec = spec_from_file_location("metric_module", PATH)
metric_module = module_from_spec(spec)
spec.loader.exec_module(metric_module)

gc.collect()
objects = gc.get_objects()

submitters = get_submitters(objects)