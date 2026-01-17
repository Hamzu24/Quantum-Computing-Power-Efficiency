from metric_executor import run_metric
import argparse
import logging
import os
from _helpers.helpers import read_config, set_up_logger, set_num_qubits_list, get_control_parameters, get_config_value, get_num_qubits, set_circuit_optimisation
from _helpers.constants import DEFAULT_PATH
import matplotlib
import matplotlib.pyplot as plt
from _helpers.registry import control_parameter_registry

def get_nm_name(name: str, config_data):
    noise_models = config_data.get("noise_models")
    specific_nm = noise_models.get("name")
    return specific_nm

# Params is used as a temporary variable to pass data in if sweeping
def optimise(metric_path: str, save_image=False, params=None):
    os.environ["SINGLE_RUN"] = "false"
    os.environ["iteration"] = "0"
    
    config_data = read_config()
    iters = config_data.get("optimisation_iterations")
    set_circuit_optimisation()

    if iters is None:
        iters = 50

    perfs = []
    cons = []
    temps = []
    for i in range(0, iters):
        print(f"Runnning metric for iteration {i}")
        calculate_consumption = False
        output = run_metric(metric_path, calculate_consumption)
        control_parameters = control_parameter_registry.get_control_parameters()
        temp = get_config_value(control_parameters, "temperature")
        temps.append(temp)

        os.environ["iteration"] = str(int(os.environ.get("iteration")) + 1)
        perfs.append(output.get("performance"))
        if calculate_consumption:
            cons.append(output.get("power_consumption").get("total_consumption"))
        logging.info("-------------------------------------------------------------\n\n")
    
    print(f"perfs: {perfs}, cons: {cons}")
    fig, ax = plt.subplots()
    assert len(temps) == len(perfs)
    ax.plot(temps, perfs)

    if save_image:
        if not params:
            nm_name = get_nm_name("default", config_data)
            num_qb = get_num_qubits()
            metric_name = DEFAULT_PATH
            metric_name = DEFAULT_PATH.split('/')[-1].split('.')[0]
        else:
            nm_name = params[1]
            num_qb = params[0]
            metric_name = params[2]

        fig.savefig(f"images/{num_qb}_{metric_name}_{nm_name}.png")
        plt.close()
    else:
        matplotlib.use('qtagg')
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

    num_qubits_list = set_num_qubits_list()

    metric_name = args.path.split('/')[-1].split('.')[0]
    print(f"""Now optimising the metric with the following settings:
        metric: {metric_name}
        visual mode: False
        logging level: {log_level}
        logging to file: {args.log_file}
        number of qubits: {num_qubits_list}
        """) 
    
    optimise(DEFAULT_PATH, save_image=True)
