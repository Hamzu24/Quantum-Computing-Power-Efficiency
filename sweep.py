import os
import argparse
from _helpers.helpers import read_config, set_up_logger, set_num_qubits_list, get_control_parameters, get_config_value
from _helpers.constants import DEFAULT_PATH
import logging
import matplotlib
from optimiser import optimise
import json

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

    with open("sweep_config.json", 'r') as f:
        config = json.load(f)
        qubits_list = config.get("qubits")
        backends_list = config.get("backends")
        all_configs = [(qb, b) for qb in qubits_list for b in backends_list]

    cur_nm = "default"
    for sweep_config in all_configs:
        num_qb = sweep_config[0]
        backend = sweep_config[1]
        print(f"""\n\n\nNow running another iteration with:
            num_qb: {num_qb}
            backend: {backend}
        """)

        with open("configs.json", 'r') as f:
            config = json.load(f)

        config['num_qubits'] = num_qb
        config['noise_models'][cur_nm]["name"] = backend

        with open("configs.json", 'w') as f:
            json.dump(config, f, indent=2)

        num_qubits_list = set_num_qubits_list()

        metric_name = args.path.split('/')[-1].split('.')[0]
        print(f"""Now optimising the metric with the following settings:
            metric: {metric_name}
            visual mode: False
            logging level: {log_level}
            logging to file: {args.log_file}
            number of qubits: {num_qubits_list}
            """) 
        
        sweep_config = (sweep_config[0], sweep_config[1], metric_name)
        optimise(DEFAULT_PATH, save_image=True, params=sweep_config)
        print("one optimisation round completed successfully")
