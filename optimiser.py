from metric_executor import run_metric
import argparse
import logging
import os
import json
from _helpers.helpers import read_config, set_up_logger, set_num_qubits_list, get_control_parameters, get_config_value
from _helpers.constants import DEFAULT_PATH
import matplotlib
import matplotlib.pyplot as plt
from _helpers.registry import control_parameter_registry

def optimise(metric_path: str):
    os.environ["SINGLE_RUN"] = "false"
    os.environ["iteration"] = "0"
    
    config_data = read_config()
    iters = config_data.get("optimisation_iterations")
    if iters is None:
        iters = 50

    perfs = []
    cons = []
    temps = []
    for i in range(0, iters):
        print(f"Runnning metric for iteration {i}")
        output = run_metric(metric_path)
        control_parameters = control_parameter_registry.get_control_parameters()
        print(f"cp: {control_parameters}")
        temp = get_config_value(control_parameters, "temperature")
        print(f"temp: {temp}")
        temps.append(temp)

        os.environ["iteration"] = str(int(os.environ.get("iteration")) + 1)
        perfs.append(output.get("performance"))
        cons.append(output.get("power_consumption").get("total_consumption"))
        logging.info("-------------------------------------------------------------\n\n")
    
    print(f"perfs: {perfs}, cons: {cons}")
    matplotlib.use('qtagg')
    fig, ax = plt.subplots()
    assert len(temps) == len(perfs)
    ax.plot(temps, perfs)
    plt.show()

if __name__ == "__main__":
    os.environ["CONFIG_PATH"] = "configs.json"
    os.environ["HARDWARE_CONFIG_PATH"] = "qiskit_backend_configs/hardware_constants.json"
    os.environ["BACKEND_CONFIGS_FOLDER"] = "qiskit_backend_configs/"

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', help='Optional path', default=DEFAULT_PATH)
    parser.add_argument('-v', '--visual', action='store_true', help='Visual flag. Not recommended for the optimiser')
    parser.add_argument('--log', 
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'debug', 'info', 'error', 'critical'],
                    default='WARNING',
                    help='Set the logging level')
    parser.add_argument('--log-file', 
                    default=None,
                    help='Log to file instead of console')

    args = parser.parse_args()
    log_level = getattr(logging, args.log.upper())

    set_up_logger(log_level, args.log_file)
    if not args.visual:
        matplotlib.use('Agg')  # Use non-interactive backend

    set_num_qubits_list()
    num_qubits_list = set_num_qubits_list()

    metric_name = args.path.split('/')[-1].split('.')[0]
    print(f"""Now optimising the metric with the following settings:
        metric: {metric_name}
        visual mode: False
        logging level: {log_level}
        logging to file: {args.log_file}
        number of qubits: {num_qubits_list}
        """) 
    
    optimise(DEFAULT_PATH)
    
