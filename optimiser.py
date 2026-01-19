from metric_executor import run_metric
import argparse
import logging
import os
from _helpers.helpers import read_config, set_up_logger, set_num_qubits_list, get_control_parameters, get_config_value, get_num_qubits, set_circuit_optimisation, extract_metric_name
from _helpers.constants import DEFAULT_PATH, resolve_metric_path
import matplotlib
import matplotlib.pyplot as plt
from _helpers.registry import control_parameter_registry

IMAGES_DIR = "images"


def get_nm_name(config_data):
    noise_models = config_data.get("noise_models", {})
    selected_nm_name = config_data.get("selected_noise_model", "default")
    selected_nm = noise_models.get(selected_nm_name)
    nm_name = selected_nm.get("name")

    if nm_name is None:
        nm_name = selected_nm.get("type", "Unknown")

    return nm_name


def save_plot(fig, metric_path: str, config_data, params=None):
    """Save plot to images directory with descriptive filename."""
    os.makedirs(IMAGES_DIR, exist_ok=True)

    if params:
        num_qb, nm_name, metric_name = params[0], params[1], params[2]
    else:
        nm_name = get_nm_name(config_data)
        num_qb = get_num_qubits()
        metric_name = extract_metric_name(metric_path)

    filename = f"{num_qb}qb_{metric_name}_{nm_name}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {filepath}")


def create_performance_plot(temps, perfs, metric_name: str):
    """Create a formatted performance vs temperature plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(temps, perfs, marker='o', linewidth=2, markersize=4)
    ax.set_xlabel("Temperature (mK)")
    ax.set_ylabel("Performance")
    ax.set_title(f"{metric_name}: Performance vs Temperature")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def optimise(metric_path: str, save_image=False, show_plot=False, params=None):
    """Run metric optimization across control parameter sweep."""
    os.environ["SINGLE_RUN"] = "false"
    os.environ["iteration"] = "0"

    config_data = read_config()
    iters = config_data.get("optimisation_iterations", 10)
    set_circuit_optimisation()

    perfs = []
    cons = []
    temps = []

    # Always use non-interactive backend during metric runs to prevent blocking
    original_backend = matplotlib.get_backend()
    matplotlib.use('Agg')
    plt.switch_backend('Agg')

    for i in range(iters):
        print(f"Running metric for iteration {i}")
        calculate_consumption = False
        output = run_metric(metric_path, calculate_consumption)
        control_parameters = control_parameter_registry.get_control_parameters()
        temp = get_config_value(control_parameters, "temperature")
        temps.append(temp)

        os.environ["iteration"] = str(int(os.environ.get("iteration")) + 1)
        perfs.append(output.get("performance"))
        if calculate_consumption:
            cons.append(output.get("power_consumption").get("total_consumption"))

        # Close any figures created by the metric
        plt.close('all')
        logging.info("-------------------------------------------------------------\n\n")

    if show_plot:
        plt.switch_backend(original_backend)

    print(f"perfs: {perfs}, cons: {cons}")

    assert len(temps) == len(perfs), "Temperature and performance arrays must have same length"

    metric_name = params[2] if params else extract_metric_name(metric_path)
    fig = create_performance_plot(temps, perfs, metric_name)

    if save_image:
        save_plot(fig, metric_path, config_data, params)
    if show_plot:
        plt.show()

    return {
        "performances": perfs,
        "temperatures": temps,
        "figure": fig
    }

if __name__ == "__main__":
    os.environ["CONFIG_PATH"] = "configs.json"
    os.environ["HARDWARE_CONFIG_PATH"] = "qiskit_backend_configs/hardware_constants.json"
    os.environ["BACKEND_CONFIGS_FOLDER"] = "qiskit_backend_configs/"

    parser = argparse.ArgumentParser(description="Run metric optimization across control parameter sweeps")
    parser.add_argument('path', nargs='?', help='Path to metric script', default=DEFAULT_PATH)
    parser.add_argument('-v', '--visual', action='store_true',
                        help='Show plot interactively')
    parser.add_argument('-s', '--save', action='store_true',
                        help='Save plot to images directory')
    parser.add_argument('--log',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'debug', 'info', 'error', 'critical'],
                        default='WARNING',
                        help='Set the logging level')
    parser.add_argument('--log-file',
                        default=None,
                        help='Log to file instead of console')

    args = parser.parse_args()

    # Default to saving if neither flag is specified
    if not args.visual and not args.save:
        args.save = True

    # Set matplotlib backend before any plotting
    if not args.visual:
        matplotlib.use('Agg')

    log_level = getattr(logging, args.log.upper())
    set_up_logger(log_level, args.log_file)

    # Resolve algorithm name to file path if needed
    metric_path = resolve_metric_path(args.path)

    num_qubits_list = set_num_qubits_list()
    metric_name = extract_metric_name(metric_path)

    print(f"""Now optimising the metric with the following settings:
        metric: {metric_name}
        show plot: {args.visual}
        save plot: {args.save}
        logging level: {log_level}
        logging to file: {args.log_file}
        number of qubits: {num_qubits_list}
        """)

    optimise(metric_path, save_image=args.save, show_plot=args.visual)
