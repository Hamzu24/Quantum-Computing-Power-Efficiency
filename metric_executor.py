import _helpers.circuit_submitter
import _helpers.enhanced_circuit_submitter
from copy import copy
import runpy
import json
import matplotlib
import argparse
import datetime
import logging
import os
from _helpers.helpers import set_up_logger, read_config, get_num_qubits, set_num_qubits_list, set_circuit_optimisation, extract_metric_name
from _helpers.constants import DEFAULT_PATH, resolve_metric_path
from _helpers.registry import submitter_registry

def run_metric(metric_path=DEFAULT_PATH, calculate_consumption=False):
    runpy.run_path(metric_path, run_name="__main__")

    performance = float(os.environ.get('PERF_VALUE'))
    print(f"Performance value: {performance}")

    submitter = submitter_registry.get_submitter("noisy_sim")

    nm_metadata = {}
    if submitter is not None and hasattr(submitter, 'get_noise_model_metadata'):
        nm_metadata = submitter.get_noise_model_metadata()

    circuit_fig = None
    if submitter is not None and hasattr(submitter, 'get_sample_circuit_figure'):
        circuit_fig = submitter.get_sample_circuit_figure()

    if calculate_consumption:
        total_consumption, staggered_consumptions = submitter.get_power_consumption()

        total_consumption = sum(total_consumption.values())
        consumption_dict = {"total_consumption": total_consumption, "staggered_consumptions": []}
        print(f"total_consumption: {total_consumption}")
        num_qubits_list = get_num_qubits()
        # NEED TO FIX BELOW, INVALID CODE
        for i, (cons, num_qb) in enumerate(zip(staggered_consumptions, num_qubits_list)):
            print(f"{num_qb} qubit power consumption: {sum(cons.values())}")
            consumption_dict["staggered_consumptions"].append(total_consumption)

        return {"performance": performance, "power_consumption": consumption_dict, "nm_metadata": nm_metadata, "circuit_figure": circuit_fig}
    else:
        return {"performance": performance, "nm_metadata": nm_metadata, "circuit_figure": circuit_fig}

if __name__ == "__main__":
    os.environ["CONFIG_PATH"] = "configs.json"
    os.environ["HARDWARE_CONFIG_PATH"] = "qiskit_backend_configs/hardware_constants.json"
    os.environ["BACKEND_CONFIGS_FOLDER"] = "qiskit_backend_configs/"
    os.environ["SINGLE_RUN"] = "true"
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', default=DEFAULT_PATH, help='Optional path')
    parser.add_argument('-v', '--visual', action='store_true', help='Visual flag')
    parser.add_argument('--log', 
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'debug', 'info', 'error', 'critical'],
                    default='WARNING',
                    help='Set the logging level')
    parser.add_argument('--log-file', 
                    default=None,
                    help='Log to file instead of console')
    args = parser.parse_args()

    # Resolve algorithm name to file path if needed
    metric_path = resolve_metric_path(args.path)
    print(f"metric_path: {metric_path}")

    log_level = getattr(logging, args.log.upper())
    set_up_logger(log_level, args.log_file)

    if not args.visual:
        matplotlib.use('Agg')  # Use non-interactive backend

    metric_name = extract_metric_name(metric_path)

    num_qubits_list = set_num_qubits_list()
    set_circuit_optimisation()

    print(f"""Now running the metric with the following settings:
        metric: {metric_name}
        visual mode: {args.visual}
        logging level: {log_level}
        logging to file: {args.log_file}
        number of qubits: {num_qubits_list}
         """) 

    run_metric(metric_path)
