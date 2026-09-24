"""Plot training logs, including JSONL logs from the original experiments."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import ROOT


def plot_histories(runs, path, log_scale=False):
    metrics = ["train_mse"]
    if any("test_mse" in row for history in runs.values() for row in history):
        metrics.append("test_mse")
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 4 * len(metrics)),
                             dpi=110, sharex=True, squeeze=False)
    for ax, metric in zip(axes[:, 0], metrics):
        for label, history in runs.items():
            rows = [row for row in history if metric in row]
            if rows:
                values = [max(row[metric], 1e-12) if log_scale else row[metric] for row in rows]
                ax.plot([row["step"] for row in rows], values, label=label)
        ax.set_ylabel(metric.replace("_", " "))
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1, 0].set_xlabel("step")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="path/to/log.jsonl[:label]; repeatable")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/plots/comparison.png")
    parser.add_argument("--log-scale", action="store_true")
    args = parser.parse_args()
    runs = {}
    for spec in args.run:
        filename, _, label = spec.partition(":")
        path = Path(filename)
        try:
            history = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError) as error:
            parser.error(str(error))
        if not history:
            parser.error(f"empty log: {path}")
        runs[label or path.stem] = history
    plot_histories(runs, args.out, args.log_scale)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
