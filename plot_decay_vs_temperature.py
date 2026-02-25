"""One-time script to plot exponential decay constant (alpha) vs temperature
from mirror benchmarking results in experiment_results_4/."""

import json
import re
import matplotlib.pyplot as plt
import numpy as np


def plot_decay_vs_temperature():
    with open("experiment_results_4/results_20260220_172906.json") as f:
        data = json.load(f)

    experiment = data[0]
    trials = experiment["trials"]

    temperatures = []
    alphas = []

    for trial_key, trial_data in trials.items():
        # Extract temperature from the control_parameters
        temp_value = trial_data["params"]["control_parameters"]["temperature"][0][0]
        temp_unit = trial_data["params"]["control_parameters"]["temperature"][0][1]
        temperatures.append(temp_value)
        alphas.append(trial_data["alpha"])

    # Sort by temperature
    sorted_pairs = sorted(zip(temperatures, alphas))
    temperatures, alphas = zip(*sorted_pairs)
    temperatures = np.array(temperatures)
    alphas = np.array(alphas)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(temperatures, alphas, "o-", color="tab:blue", markersize=7, linewidth=1.5)
    ax.set_xlabel(f"Temperature ({temp_unit})", fontsize=13)
    ax.set_ylabel("Exponential Decay Constant (α)", fontsize=13)
    ax.set_title(
        f"Mirror Benchmarking Decay vs Temperature\n"
        f"({experiment['trials'][trial_key]['params']['num_qubits']}qb, "
        f"fake_backend ({experiment['trials'][trial_key]['params']['noise_model_name']}))",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out_path = "experiment_results_4/decay_vs_temperature.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_decay_vs_temperature()
