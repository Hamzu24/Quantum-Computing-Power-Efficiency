"""
Combine all trial plots from an experiment's trials/ directory into a single figure.

Arranges plots in a grid: rows = trials (backends), columns = plot types (quantum_volume, T1T2).
Circuit diagrams in the circuits/ subdirectory are excluded by default.

Usage:
    python combine_trial_plots.py <trials_dir> [--output combined.png] [--include-circuits]
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def parse_trial_filename(filename):
    """Extract trial number and plot type from a filename like:
    'trial0_quantum_volume_11qb, fake_backend (tokyo), 40 mK.png'
    Returns (trial_num, plot_type, label) or None if not parseable.
    """
    match = re.match(
        r"trial(\d+)_(.+?)_(\d+qb, .+)\.png",
        filename,
    )
    if not match:
        return None
    trial_num = int(match.group(1))
    plot_type = match.group(2)
    label = match.group(3)
    return trial_num, plot_type, label


def collect_images(trials_dir, include_circuits=False):
    """Collect and organize images from the trials directory.
    Returns dict: {plot_type: [(trial_num, label, path), ...]} sorted by trial_num.
    """
    trials_path = Path(trials_dir)
    images = {}

    for img_path in sorted(trials_path.glob("*.png")):
        parsed = parse_trial_filename(img_path.name)
        if parsed is None:
            continue
        trial_num, plot_type, label = parsed
        images.setdefault(plot_type, []).append((trial_num, label, img_path))

    if include_circuits:
        circuits_dir = trials_path / "circuits"
        if circuits_dir.exists():
            for img_path in sorted(circuits_dir.glob("*.png")):
                parsed = parse_trial_filename(img_path.name)
                if parsed is None:
                    continue
                trial_num, plot_type, label = parsed
                images.setdefault(plot_type, []).append((trial_num, label, img_path))

    # Sort each group by trial number
    for plot_type in images:
        images[plot_type].sort(key=lambda x: x[0])

    return images


def combine_plots(trials_dir, output_path, include_circuits=False, dpi=150):
    images = collect_images(trials_dir, include_circuits)

    if not images:
        print(f"No trial images found in {trials_dir}")
        return

    # Columns = plot types (sorted so quantum_volume comes before T1T2 alphabetically,
    # but let's put quantum_volume first explicitly)
    plot_types = sorted(images.keys())
    # Move quantum_volume to front if present
    if "quantum_volume" in plot_types:
        plot_types.remove("quantum_volume")
        plot_types.insert(0, "quantum_volume")

    # Rows = trials (use the max trial count across plot types)
    n_trials = max(len(v) for v in images.values())
    n_types = len(plot_types)

    # Read one image per type to determine aspect ratios
    sample_aspects = {}
    for pt in plot_types:
        if images[pt]:
            sample = mpimg.imread(images[pt][0][2])
            h, w = sample.shape[:2]
            sample_aspects[pt] = w / h

    # Calculate figure size: give each subplot a base height, width proportional to aspect
    base_height = 4
    col_widths = [sample_aspects.get(pt, 1.5) * base_height for pt in plot_types]
    fig_width = sum(col_widths) + 1  # padding
    fig_height = base_height * n_trials + 1.5  # padding for titles

    fig, axes = plt.subplots(
        n_trials,
        n_types,
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": [sample_aspects.get(pt, 1.5) for pt in plot_types]},
        squeeze=False,
    )

    # Column headers
    type_labels = {
        "quantum_volume": "Quantum Volume",
        "T1T2": "T1 / T2 Distribution",
        "circuit": "Circuit",
    }

    for col_idx, pt in enumerate(plot_types):
        axes[0, col_idx].set_title(
            type_labels.get(pt, pt), fontsize=14, fontweight="bold", pad=12
        )

    for col_idx, pt in enumerate(plot_types):
        entries = images.get(pt, [])
        for row_idx in range(n_trials):
            ax = axes[row_idx, col_idx]
            ax.axis("off")

            if row_idx < len(entries):
                trial_num, label, path = entries[row_idx]
                img = mpimg.imread(path)
                ax.imshow(img)

                # Extract backend name for row label
                backend_match = re.search(r"fake_backend \((\w+)\)", label)
                backend_name = backend_match.group(1) if backend_match else label

                if col_idx == 0:
                    ax.set_ylabel(
                        f"Trial {trial_num}\n({backend_name})",
                        fontsize=11,
                        fontweight="bold",
                        rotation=0,
                        labelpad=80,
                        va="center",
                    )
                    ax.yaxis.set_visible(True)

    fig.suptitle(
        f"Combined Trial Results — {Path(trials_dir).parent.name}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0.08, 0, 1, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved combined plot to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Combine trial plots into a single figure")
    parser.add_argument(
        "trials_dir",
        nargs="?",
        default="experiment_results_5/experiment_0/trials",
        help="Path to trials/ directory (default: experiment_results_5/experiment_0/trials)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path (default: <experiment_dir>/combined_trials.png)",
    )
    parser.add_argument(
        "--include-circuits",
        action="store_true",
        help="Include circuit diagrams from circuits/ subdirectory",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI (default: 150)",
    )
    args = parser.parse_args()

    trials_path = Path(args.trials_dir)
    if not trials_path.is_dir():
        print(f"Error: {trials_path} is not a directory")
        return

    if args.output is None:
        output_path = trials_path.parent / "combined_trials.png"
    else:
        output_path = Path(args.output)

    combine_plots(args.trials_dir, output_path, args.include_circuits, args.dpi)


if __name__ == "__main__":
    main()
