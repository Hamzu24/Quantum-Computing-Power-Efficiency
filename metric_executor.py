import gc
import _helpers.circuit_submitter
import _helpers.enhanced_circuit_submitter
from copy import copy
import runpy
import json
import matplotlib
import argparse

DEFAULT_PATH = "tutorials/circuit_execution_quality_metrics/quantum_volume/quantum_volume.py"
devices_needed = ["noisy_sim"]

def get_submitters(objects):
    submitters = {}
    devices_still_needed = copy(devices_needed)

    for obj in objects:
        if isinstance(obj, _helpers.circuit_submitter.CircuitSubmitter):
            if obj.device_name in devices_needed:
                submitters[obj.device_name] = obj
                if obj.device_name in devices_still_needed:
                    devices_still_needed.remove(obj.device_name)

    return submitters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', default=DEFAULT_PATH, help='Optional path')
    parser.add_argument('-v', '--visual', action='store_true', help='Visual flag')
    args = parser.parse_args()

    if not args.visual:
        matplotlib.use('Agg')  # Use non-interactive backend

    module_globals = runpy.run_path(DEFAULT_PATH, run_name="__main__")

    gc.collect()
    objects = gc.get_objects()

    submitters = get_submitters(objects)
    total_consumption, staggered_consumptions = submitters['noisy_sim'].get_power_consumption()
    print(f"total power usage: {sum(total_consumption.values())}\n\n")
    print(f"2 qubit power consumption: {sum(staggered_consumptions[0].values())}")
    print(f"3 qubit power consumption: {sum(staggered_consumptions[1].values())}")
