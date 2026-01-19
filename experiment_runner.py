from collections.abc import Iterable
import os
import json
import argparse
import logging
import traceback
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from itertools import product
from typing import Any
import matplotlib
import matplotlib.pyplot as plt

from _helpers.constants import DEFAULT_PATH, resolve_metric_path
from _helpers.helpers import set_up_logger, set_num_qubits_list, set_circuit_optimisation, read_config, write_config
from optimiser import optimise, create_performance_plot
from metric_executor import run_metric

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENT_PARAMETERS = ["num_qubits", "optimisation_iterations", "noise_model_name", "max_parallel_circuits"]
# Relevant noise model parameters are noise_model_additional_parameters, control_parameters, noise_model_type and noise_model_name
# Other parameters are run_type, use_nm_base and metric

experiments = [
    # Quantum Volume on different backends
    [
        {
            "run_type": "optimiser",
            "metric": "quantum_volume",
            "num_qubits": 6,
            "optimisation_iterations": 10,
            "noise_model_type": "fake_backend",
            "noise_model_name": "fakeSherbrooke",
            "noise_model_additional_params": {"pauli_twirling": True},
            "control_parameters": {"temperature": [[40, "mK"], [15, 5, "mK"]]},
            "init_control_parameters": {"temperature": [15, "mK"]}
        },
        {
            "run_type": "optimiser",
            "metric": "quantum_volume",
            "num_qubits": 6,
            "optimisation_iterations": 10,
            "noise_model_type": "fake_backend",
            "noise_model_name": "fakeTokyo",
            "noise_model_additional_params": {"pauli_twirling": True},
            "control_parameters": {"temperature": [[40, "mK"], [15, 5, "mK"]]},
            "init_control_parameters": {"temperature": [15, "mK"]}
        }
    ],
    # Different qubit numbers
    [
        {
            "run_type": "optimiser",
            "metric": "quantum_volume",
            "num_qubits": 4,
            "optimisation_iterations": 10,
            "noise_model_type": "fake_backend",
            "noise_model_name": "fakeSherbrooke",
            "control_parameters": {"temperature": [[40, "mK"], [15, 5, "mK"]]},
            "init_control_parameters": {"temperature": [15, "mK"]}
        },
        {
            "run_type": "optimiser",
            "metric": "quantum_volume",
            "num_qubits": 6,
            "optimisation_iterations": 10,
            "noise_model_type": "fake_backend",
            "noise_model_name": "fakeSherbrooke",
            "control_parameters": {"temperature": [[40, "mK"], [15, 5, "mK"]]},
            "init_control_parameters": {"temperature": [15, "mK"]}
        },
        {
            "run_type": "optimiser",
            "metric": "quantum_volume",
            "num_qubits": 8,
            "optimisation_iterations": 10,
            "noise_model_type": "fake_backend",
            "noise_model_name": "fakeSherbrooke",
            "control_parameters": {"temperature": [[40, "mK"], [15, 5, "mK"]]},
            "init_control_parameters": {"temperature": [15, "mK"]}
        },
        {
            "run_type": "optimiser",
            "metric": "quantum_volume",
            "num_qubits": 10,
            "optimisation_iterations": 10,
            "noise_model_type": "fake_backend",
            "noise_model_name": "fakeSherbrooke",
            "control_parameters": {"temperature": [[40, "mK"], [15, 5, "mK"]]},
            "init_control_parameters": {"temperature": [15, "mK"]}
        }
    ],
    # Single run example (no optimization sweep)
    [
        {
            "run_type": "single",
            "metric": "vqe",
            "num_qubits": 5,
            "noise_model_type": "fake_backend",
            "noise_model_name": "fakeSherbrooke",
            "control_parameters": {"temperature": [[50, "mK"], [15, 5, "mK"]]},
            "init_control_parameters": {"temperature": [13, "mK"]}
        }
    ]
]

DEFAULT_OUTPUT_DIR = "experiment_results"

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.output_dir / f"results_{self.timestamp}.json"
        if not Path(self.results_file).exists():
            with open(self.results_file, "w") as f:
                json.dump([], f)

        self.results = []
        self.original_config = read_config()

    def _save_config(self, config_data: dict):
        write_config(config_data)

    def _restore_config(self):
        self._save_config(self.original_config)

    def _build_noise_model_config(
        self,
        nm_base: dict | None,
        nm_type: dict | None,
        nm_name: str | None,
        control_parameters: dict | None,
        additional_parameters: dict | None
    ) -> dict:
        """Build a complete noise model configuration."""

        if nm_name is not None:
            nm_base["name"] = nm_name
        if additional_parameters is not None:
            for k, v in additional_parameters.items():
                nm_base[k] = v
        nm_base["control_parameters"] = control_parameters
        nm_base["type"] = nm_type
        nm_base["name"] = nm_name

        return nm_base

    def _build_full_config(self, params: dict) -> dict:
        config = deepcopy(self.original_config)

        base_noise_models = config.get("noise_models", {})
        base_selected_nm = config.get("selected_noise_model", "default")
        use_nm_base = config.get("use_nm_base", True)
        if use_nm_base:
            nm_base = base_noise_models.get(base_selected_nm)
            if nm_base is None:
                raise ValueError("The default config file is invalid!")
        else:
            nm_base = {}
        logging.info(f"Using base noise model: {nm_base}")

        for param in EXPERIMENT_PARAMETERS:
            if params.get(param) is not None:
                config[param] = params[param]
                logging.info(f"Assigned param {param}: {params[param]} in config")

        nm_config = self._build_noise_model_config(
            nm_base,
            params.get("noise_model_type"),
            params.get("noise_model_name"),
            params.get("control_parameters"),
            params.get("noise_model_additional_params")
        )

        config["noise_models"] = {"experiment": nm_config}
        config["selected_noise_model"] = "experiment"

        run_type = params.get("run_type", "optimiser")
        metric = params.get("metric", "quantum_volume")

        return config, run_type, metric

    def _generate_trial_label(self, params):
        nm_type = params.get('noise_model_type')
        nm_name = params.get('noise_model_name')
        num_qubits = params.get('num_qubits')
        nm_info = f"{num_qubits}qb"
        nm_info += f", {nm_type}"
        if nm_name:
            nm_info += f" ({nm_name})"

        return nm_info

    def display_trial_info(self, config, trial_number, run_type, metric):
        nm_info = self._generate_trial_label(config)
        print(f"Running experiment {trial_number}: metric={metric}, run_type={run_type}, {nm_info}")

    def save_plot(self, fig, fig_name, output_dir=None):
        if fig is None:
            return None

        target_dir = output_dir if output_dir else self.output_dir
        plot_path = target_dir / fig_name
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to {plot_path}")

        return plot_path

    def run_trial(self, trial_number: int, params: dict, output_dir: Path = None) -> dict:
        config, run_type, metric = self._build_full_config(params)
        self._save_config(config)
        self.display_trial_info(config, trial_number, run_type, metric)
        metric_path = resolve_metric_path(metric)

        start_time = datetime.now()
        if run_type == "single":
            os.environ["SINGLE_RUN"] = "true"
            set_circuit_optimisation()
            matplotlib.use('Agg')
            output = run_metric(metric_path, False)
            performances = [output.get("performance")]
            temperatures = []
            fig = None
        else:
            num_qubits_list = set_num_qubits_list()
            output = optimise(metric_path, False, False)
            performances = output["performances"]
            temperatures = output["temperatures"]
            fig = output["figure"]
        end_time = datetime.now()

        plot_filename = self._generate_plot_filename(f"trial{trial_number}", params, metric)
        plot_path = self.save_plot(fig, plot_filename, output_dir)

        result = {
            "trial_number": trial_number,
            "params": params,
            "metric": metric,
            "performances": performances,
            "temperatures": temperatures,
            "plot_path": str(plot_path) if plot_path else None,
            "duration_seconds": (end_time - start_time).total_seconds(),
            "timestamp": start_time.isoformat()
        }

        return result

    def _generate_plot_filename(self, prefix: str, params: dict, metric: str) -> str:
        trial_label = self._generate_trial_label(params)
        return f"{prefix}_{metric}_{trial_label}.png"

    def create_combined_plot(self, trial_results: list[dict], experiment_name: str):
        """Create a plot with all trials overlaid."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for result in trial_results:
            if result["temperatures"] and result["performances"]:
                label = self._generate_trial_label(result["params"])
                ax.plot(result["temperatures"], result["performances"],
                       marker='o', linewidth=2, markersize=4, label=label)

        ax.set_xlabel("Temperature (mK)")
        ax.set_ylabel("Performance")
        ax.set_title(f"{experiment_name}: Performance vs Temperature")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def run_single_experiment(self, experiment_number: int, trials: list[dict]) -> dict:
        experiment_dir = self.output_dir / f"experiment_{experiment_number}"
        experiment_dir.mkdir(exist_ok=True)
        trials_dir = experiment_dir / "trials"
        trials_dir.mkdir(exist_ok=True)

        total = len(trials)
        metric = trials[0].get("metric", "quantum_volume")

        print(f"\n{'='*60}")
        print(f"EXPERIMENT {experiment_number}")
        print(f"{'='*60}")
        print(f"Metric: {metric}")
        print(f"Total trials: {total}")
        print(f"Output directory: {experiment_dir}")
        print(f"{'='*60}\n")

        trial_results = {}
        trial_results_list = []  # For combined plot
        failed = 0
        for i, params in enumerate(trials):
            print(f"[{i+1}/{total}] ", end="")

            try:
                result = self.run_trial(i, params, trials_dir)
                trial_key = self._generate_trial_label(params)
                trial_results[trial_key] = result
                trial_results_list.append(result)
                perfs = result['performances']
                final_perf = perfs[-1] if perfs else 0
                print(f"  -> Final performance: {final_perf:.4f} "
                      f"(took {result['duration_seconds']:.1f}s)")
            except Exception as e:
                print(f"Trial {i} threw an error: {e}")
                failed += 1

        # Create combined plot
        experiment_name = f"Experiment {experiment_number}: {metric}"
        combined_fig = self.create_combined_plot(trial_results_list, experiment_name)
        fig_name = experiment_name + ".png"
        combined_plot_path = self.save_plot(combined_fig, fig_name, experiment_dir)

        # Save experiment data to JSON
        experiment_data = {
            "experiment_number": experiment_number,
            "metric": metric,
            "total_trials": total,
            "failed_trials": failed,
            "combined_plot_path": str(combined_plot_path) if combined_plot_path else None,
            "trials": trial_results,
            "timestamp": datetime.now().isoformat()
        }
        data_file = experiment_dir / "experiment_data.json"
        with open(data_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        print(f"Experiment data saved to {data_file}")

        # Print summary
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {experiment_number} SUMMARY")
        print(f"{'='*60}")
        print(f"Total trials: {total}")
        print(f"Completed: {total - failed}")
        print(f"Failed: {failed}")
        print(f"Combined plot: {combined_plot_path}")
        print(f"Data file: {data_file}")
        print(f"{'='*60}\n")

        return experiment_data

    def run_experiments(self, experiments: list[list[dict]]):
        total_experiments = len(experiments)

        print(f"\n{'='*60}")
        print(f"RUNNING {total_experiments} EXPERIMENTS")
        print(f"{'='*60}\n")

        all_results = []
        for i, experiment in enumerate(experiments):
            # Handle both single trial (dict) and multiple trials (list)
            if isinstance(experiment, dict):
                trials = [experiment]
            else:
                trials = experiment

            result = self.run_single_experiment(i, trials)
            all_results.append(result)
            self._save_result(result)

        self._restore_config()

        print(f"\n{'='*60}")
        print(f"ALL EXPERIMENTS COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {self.results_file}")
        print(f"{'='*60}\n")

        return all_results

    def _save_result(self, result):
        with open(self.results_file, 'r') as f:
            result_data = json.load(f)

        if not isinstance(result_data, Iterable):
            print(f"Unable to save the result because the target file is not iterable")
            return

        result_data.append(result)
        with open(self.results_file, 'w') as f:
            json.dump(result_data, f)

def print_experiment_preview(experiments: list[list[dict]]):
    total_trials = sum(len(exp) for exp in experiments)
    print(f"\n{'='*95}")
    print("EXPERIMENT PREVIEW")
    print(f"{'='*95}")
    print(f"Total experiments: {len(experiments)}, Total trials: {total_trials}\n")

    headers = ["Exp", "Trial", "Metric", "Run Type", "Qubits", "Iters", "Noise Model", "Backend", "Twirling"]
    col_widths = [4, 6, 16, 10, 7, 6, 14, 18, 8]

    header_row = " ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * (sum(col_widths) + len(col_widths) - 1))

    for exp_i, trials in enumerate(experiments):
        for trial_i, trial in enumerate(trials):
            metric = trial.get("metric", "N/A")
            run_type = trial.get("run_type", "N/A")
            num_qubits = str(trial.get("num_qubits", "N/A"))
            iterations = str(trial.get("optimisation_iterations", "N/A"))
            nm_type = trial.get("noise_model_type", "N/A")
            nm_name = trial.get("noise_model_name", "N/A")
            additional = trial.get("noise_model_additional_params", {})
            if additional is None:
                additional = {}
            if additional.get("pauli_twirling") is None or additional.get("pauli_twirling"):
                twirling = "Yes"
            else:
                twirling = "No"

            row = [
                str(exp_i),
                str(trial_i),
                metric[:15],
                run_type[:9],
                num_qubits,
                iterations,
                nm_type[:13],
                nm_name[:17],
                twirling
            ]
            print(" ".join(val.ljust(w) for val, w in zip(row, col_widths)))

    print(f"{'='*95}\n")

def main():
    os.environ["CONFIG_PATH"] = "configs.json"
    os.environ["HARDWARE_CONFIG_PATH"] = "qiskit_backend_configs/hardware_constants.json"
    os.environ["BACKEND_CONFIGS_FOLDER"] = "qiskit_backend_configs/"

    parser = argparse.ArgumentParser(
        description="Run experiments across many configuration combinations"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Only show preview of experiments, don't run"
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory to save experiment results"
    )
    parser.add_argument(
        "--log", type=str, default="WARNING",
        help="Log level (DEBUG, INFO, WARNING, ERROR)"
    )

    args = parser.parse_args()

    log_level = getattr(logging, args.log.upper())
    set_up_logger(log_level, None)

    runner = ExperimentRunner(output_dir=args.output_dir)

    print_experiment_preview(experiments)

    if args.preview:
        print("Preview mode - not running experiments")
        return

    response = input(f"Run {len(experiments)} experiments? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted")
        return

    runner.run_experiments(experiments)

if __name__ == "__main__":
    main()
