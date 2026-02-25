from metric_executor import run_metric
import argparse
import logging
import os
import json
from _helpers.helpers import read_config, set_up_logger, set_num_qubits_list, get_num_qubits, set_circuit_optimisation, extract_metric_name
from _helpers.constants import resolve_metric_path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

IMAGES_DIR = "images"
MIRROR_BENCHMARK_PATH = "tutorials/circuit_execution_quality_metrics/mirror_benchmarking/mirror_benchmarking.py"


def rb_fit_func(m, alpha, a_0, b_0):
    """Exponential decay: P(m) = a_0 * alpha^m + b_0"""
    return a_0 * np.power(alpha, m) + b_0


def fit_decay(depths, survival_probs, n_qubits):
    """Fit exponential decay to mean survival probabilities vs depth.

    Returns (fit_params, pcov) where fit_params = [alpha, a_0, b_0].
    """
    depths_arr = np.array(depths, dtype=float)
    probs_arr = np.array(survival_probs, dtype=float)

    try:
        fit, pcov = curve_fit(
            rb_fit_func,
            depths_arr,
            probs_arr,
            p0=(0.99, 0.8, 0),
            bounds=((0.0, 0, 0), (1, 1, 1 / (2 ** n_qubits))),
        )
    except RuntimeError:
        logging.warning("Curve fit failed, returning default parameters")
        fit = np.array([0.99, 0.8, 0.0])
        pcov = np.zeros((3, 3))

    return fit, pcov


def get_nm_name(config_data):
    noise_models = config_data.get("noise_models", {})
    selected_nm_name = config_data.get("selected_noise_model", "default")
    selected_nm = noise_models.get(selected_nm_name)
    nm_name = selected_nm.get("name")

    if nm_name is None:
        nm_name = selected_nm.get("type", "Unknown")

    return nm_name


def save_plot(fig, config_data, suffix="", params=None):
    """Save plot to images directory with descriptive filename."""
    os.makedirs(IMAGES_DIR, exist_ok=True)

    if params:
        num_qb, nm_name = params[0], params[1]
    else:
        nm_name = get_nm_name(config_data)
        num_qb = get_num_qubits()

    filename = f"{num_qb}qb_mirror_depth_sweep_{nm_name}{suffix}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {filepath}")


def create_depth_sweep_plot(depths, perfs, fit_params, num_qubits):
    """Create performance vs depth plot with exponential decay fit."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(depths, perfs, marker='o', s=40, zorder=5, label="Mean survival")

    alpha, a_0, b_0 = fit_params
    xxs = np.linspace(0, max(depths), 500)
    ax.plot(
        xxs,
        rb_fit_func(xxs, *fit_params),
        ls="--",
        color="#1f77b4",
        label=f"Fit: $\\alpha$={alpha:.4f}",
    )

    ax.set_xlabel("Depth (number of layers)")
    ax.set_ylabel(r"$p_{\mathrm{survival}}$ (P(|0...0>))")
    ax.set_title(f"Mirror Benchmarking Depth Sweep ({num_qubits} qubits)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    y_min = max(0, min(perfs) - 0.02)
    ax.set_ylim(y_min, 1.02)

    fig.tight_layout()
    return fig


def set_mirror_depth(depth):
    """Update mirror_depth in configs.json."""
    config_path = os.environ.get("CONFIG_PATH", "configs.json")
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    config_data["mirror_depth"] = depth
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)


def depth_sweep(metric_path: str, depths=None, save_image=False, show_plot=False, params=None):
    """Run mirror benchmarking across a range of depths at fixed temperature.

    Args:
        metric_path: Path to the mirror benchmarking metric script.
        depths: List of depths to sweep. Defaults to [1, 2, 4, 8, 16, 32].
        save_image: Whether to save the plot to disk.
        show_plot: Whether to display the plot interactively.
        params: Optional (num_qb, nm_name) tuple for filename generation.

    Returns:
        Dictionary with depths, performances, fit parameters, and figure.
    """
    if depths is None:
        depths = [1, 2, 4, 8, 16, 32]

    os.environ["SINGLE_RUN"] = "true"
    os.environ["iteration"] = "0"

    config_data = read_config()
    set_circuit_optimisation()
    num_qubits = get_num_qubits()

    perfs = []
    circuit_fig = None

    # Use non-interactive backend during metric runs
    original_backend = matplotlib.get_backend()
    matplotlib.use('Agg')
    plt.switch_backend('Agg')

    for depth in depths:
        print(f"Running mirror benchmarking at depth {depth}")
        set_mirror_depth(depth)

        output = run_metric(metric_path, calculate_consumption=False)
        if circuit_fig is None:
            circuit_fig = output.get("circuit_figure")
        perfs.append(output.get("performance"))

        plt.close('all')
        logging.info("-------------------------------------------------------------\n\n")

    if show_plot:
        plt.switch_backend(original_backend)

    print(f"depths: {depths}, perfs: {perfs}")

    assert len(depths) == len(perfs), "Depth and performance arrays must have same length"

    # Fit exponential decay
    fit_params, pcov = fit_decay(depths, perfs, num_qubits)
    alpha, a_0, b_0 = fit_params

    fig = create_depth_sweep_plot(depths, perfs, fit_params, num_qubits)

    if save_image:
        save_plot(fig, config_data, params=params)

    if show_plot:
        plt.show()

    return {
        "depths": depths,
        "performances": perfs,
        "fit_params": fit_params,
        "pcov": pcov,
        "alpha": alpha,
        "num_qubits": num_qubits,
        "figure": fig,
        "circuit_figure": circuit_fig,
    }


if __name__ == "__main__":
    os.environ["CONFIG_PATH"] = "configs.json"
    os.environ["HARDWARE_CONFIG_PATH"] = "qiskit_backend_configs/hardware_constants.json"
    os.environ["BACKEND_CONFIGS_FOLDER"] = "qiskit_backend_configs/"

    parser = argparse.ArgumentParser(description="Run mirror benchmarking across a sweep of circuit depths")
    parser.add_argument('path', nargs='?', help='Path to mirror benchmarking metric script',
                        default=MIRROR_BENCHMARK_PATH)
    parser.add_argument('-v', '--visual', action='store_true',
                        help='Show plot interactively')
    parser.add_argument('-s', '--save', action='store_true',
                        help='Save plot to images directory')
    parser.add_argument('--depths', nargs='+', type=int, default=None,
                        help='List of depths to sweep (e.g. --depths 1 2 4 8 16 32)')
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

    if not args.visual:
        matplotlib.use('Agg')

    log_level = getattr(logging, args.log.upper())
    set_up_logger(log_level, args.log_file)

    metric_path = resolve_metric_path(args.path)

    num_qubits_list = set_num_qubits_list()

    print(f"""Now running mirror benchmarking depth sweep with the following settings:
        metric: {metric_path}
        depths: {args.depths or [1, 2, 4, 8, 16, 32]}
        show plot: {args.visual}
        save plot: {args.save}
        logging level: {log_level}
        logging to file: {args.log_file}
        number of qubits: {num_qubits_list}
        """)

    results = depth_sweep(
        metric_path,
        depths=args.depths,
        save_image=args.save,
        show_plot=args.visual,
    )

    alpha = results["alpha"]
    print(f"\n{'='*50}")
    print(f"DEPTH SWEEP RESULTS:")
    print(f"  Qubits: {results['num_qubits']}")
    print(f"  Depths: {results['depths']}")
    print(f"  Decay parameter alpha: {alpha:.4f}")
    print(f"  Fit: {results['fit_params'][1]:.4f} * {alpha:.4f}^d + {results['fit_params'][2]:.6f}")
    print(f"{'='*50}")
