import gc
import _helpers.circuit_submitter
import _helpers.enhanced_circuit_submitter
from copy import copy
import runpy
import json
import matplotlib
import argparse
import os
import datetime
import logging

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
    DEFAULT_PATH = "tutorials/circuit_execution_quality_metrics/quantum_volume/quantum_volume.py"
    devices_needed = ["noisy_sim"]

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', default=DEFAULT_PATH, help='Optional path')
    parser.add_argument('-v', '--visual', action='store_true', help='Visual flag')
    parser.add_argument('--log', 
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'debug', 'info', 'error', 'critical'],
                    default='INFO',
                    help='Set the logging level')
    parser.add_argument('--log-file', 
                    help='Log to file instead of console')
    args = parser.parse_args()

    log_level = getattr(logging, args.log.upper())
    logging_fmt='%(asctime)s | %(funcName)s:%(lineno)d | %(levelname)s | %(message)s'

    if args.log_file:
        logging.basicConfig(
            filename=args.log_file,
            level=log_level,
            format=logging_fmt
        )
    else:
        logging.basicConfig(
            level=log_level,
            format=logging_fmt
        )

    logging.getLogger('qiskit').setLevel(logging.WARNING)

    metric_name = args.path.split('/')[-1].split('.')[0]

    if not args.visual:
        matplotlib.use('Agg')  # Use non-interactive backend

    logging.info(f"""Now running the metric with the following settings:
        metric: {metric_name}
        visual mode: {args.visual}
        logging level: {log_level}
         """) 
    module_globals = runpy.run_path(args.path, run_name="__main__")

    gc.collect()
    objects = gc.get_objects()

    submitters = get_submitters(objects)
    total_consumption, staggered_consumptions = submitters['noisy_sim'].get_power_consumption()
    print(f"perfomance: {os.environ.get('PERF_VALUE')}\n")
    print(f"total power usage: {sum(total_consumption.values())}\n\n")
    print(f"2 qubit power consumption: {sum(staggered_consumptions[0].values())}")
    print(f"3 qubit power consumption: {sum(staggered_consumptions[1].values())}")
