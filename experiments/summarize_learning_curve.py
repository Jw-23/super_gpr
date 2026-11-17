"""Validate fixed-test nested-sample experiments and summarize their learning curves."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

MODELS = ("independent", "shared_ard", "plain_matern25")
LABELS = {
    "independent": "Additive, independent",
    "shared_ard": "Additive, shared ARD",
    "plain_matern25": "Plain Matérn 2.5",
}


def load_records(paths):
    rows = []
    for path in paths:
        with path.open(newline="") as source:
            rows.extend(csv.DictReader(source))
    assert rows, "No learning-curve records"
    unique = set()
    metadata = defaultdict(set)
    grids = defaultdict(set)
    for row in rows:
        dimension, count, seed = (int(row[key]) for key in ("dimension", "train_count", "seed"))
        model = row["model"]
        assert model in MODELS
        key = (dimension, seed, count, model)
        assert key not in unique, f"Duplicate fit: {key}"
        unique.add(key)
        metadata[(dimension, seed)].add((row["test_count"], row["training_pool_checksum"], row["test_checksum"]))
        grids[(dimension, seed, model)].add(count)
        expected = {"independent": dimension // 2 + 1, "shared_ard": 6, "plain_matern25": 3}[model]
        assert int(row["parameters"]) == expected
        values = [float(value) for value in row["physical_parameters"].split(";")]
        assert len(values) == expected and all(math.isfinite(value) and value > 0 for value in values)
        for metric in ("rmse", "nrmse", "r2", "mae", "p95_absolute_error", "max_absolute_error", "train_rmse", "noise_std", "lml"):
            assert math.isfinite(float(row[metric])), (key, metric)
        assert math.isclose(float(row["nrmse"]) ** 2, 1.0 - float(row["r2"]), abs_tol=1e-8)
        assert float(row["mae"]) <= float(row["rmse"]) + 1e-8
        assert 0 <= float(row["p95_absolute_error"]) <= float(row["max_absolute_error"]) + 1e-8
    for key, values in metadata.items():
        assert len(values) == 1, f"Dataset changed across sample counts or models: {key}"
    present = {row["model"] for row in rows}
    for dimension, seed in metadata:
        assert all((dimension, seed, model) in grids for model in present), "Missing model for a seed"
    all_grids = list(grids.values())
    assert all(grid == all_grids[0] for grid in all_grids), "Seeds use different sample counts"
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    trajectories = defaultdict(list)
    for row in rows:
        grouped[(int(row["train_count"]), row["model"])].append(row)
        trajectories[(row["seed"], row["model"])].append(row)
    dimensions = sorted({int(row["dimension"]) for row in rows})
    assert len(dimensions) == 1, "Summarize each dimension separately"
    seeds = sorted({int(row["seed"]) for row in rows})
    lines = ["# Training sample learning curve", "",
             f"Dimension: {dimensions[0]}. Seeds: {', '.join(map(str, seeds))}. Test points per seed: {rows[0]['test_count']}.", "",
             "Training sets are nested prefixes of one noisy pool. Test inputs and noiseless targets stay fixed for each seed.",
             "Every fit uses the same cold-start settings; test labels do not select hyperparameters or model structure.", "",
             "Values are mean ± sample standard deviation across seeds.", "",
             "| Training samples | Model | Test R² | Test RMSE | MAE | 95th percentile absolute error | Fit seconds | Converged |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, runs in sorted(grouped.items()):
        count, model = key
        cells = []
        for field in ("r2", "rmse"):
            values = [float(row[field]) for row in runs]
            deviation = stdev(values) if len(values) > 1 else 0.0
            cells.append(f"{mean(values):.4f} ± {deviation:.4f}")
        cells += [f"{mean(float(row[field]) for row in runs):.4f}" for field in ("mae", "p95_absolute_error", "fit_seconds")]
        lines.append(f"| {count} | {LABELS[model]} | " + " | ".join(cells) +
                     f" | {sum(int(row['converged']) for row in runs)}/{len(runs)} |")
    lines += ["", "## Each seed", "", "| Seed | Model | Training samples | R² | RMSE | Noise std | Iterations | Evaluations |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    decreases = 0
    transitions = 0
    for (seed, model), runs in sorted(trajectories.items()):
        runs.sort(key=lambda row: int(row["train_count"]))
        for row in runs:
            lines.append(f"| {seed} | {LABELS[model]} | {row['train_count']} | {float(row['r2']):.5f} | "
                         f"{float(row['rmse']):.5f} | {float(row['noise_std']):.5f} | {row['iterations']} | {row['evaluations']} |")
        for first, second in zip(runs, runs[1:]):
            transitions += 1
            decreases += float(second["rmse"]) < float(first["rmse"])
    lines += ["", f"Test RMSE decreases in {decreases}/{transitions} consecutive within-seed transitions.",
              f"Optimizer convergence: {sum(int(row['converged']) for row in rows)}/{len(rows)} fits.",
              "All results are retained. More data does not impose monotonic test accuracy after hyperparameter reoptimization.", ""]
    return "\n".join(lines)


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = sorted({int(row["train_count"]) for row in rows})
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    present = [model for model in MODELS if any(row["model"] == model for row in rows)]
    for model, color in zip(present, ("#4169a1", "#d17432", "#389269")):
        for axis, metric, ylabel in zip(axes, ("r2", "rmse"), ("Test R² (higher is better)", "Test RMSE (lower is better)")):
            values = [[float(row[metric]) for row in rows if row["model"] == model and int(row["train_count"]) == count]
                      for count in counts]
            axis.errorbar(counts, [mean(samples) for samples in values],
                          yerr=[stdev(samples) if len(samples) > 1 else 0 for samples in values],
                          color=color, marker="o", linewidth=2, capsize=3, label=LABELS[model])
            axis.set_xscale("log", base=2)
            axis.set_xticks(counts, [str(count) for count in counts])
            axis.set_xlabel("Training samples")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2)
            axis.legend(frameon=False)
    seed_count = len({row["seed"] for row in rows})
    figure.suptitle(
        f"{rows[0]['dimension']}-dimensional additive GP: fixed test sets, nested training sets\n"
        f"Mean ± sample SD over {seed_count} seeds; {rows[0]['test_count']} fixed test points per seed",
        fontsize=12)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    rows = load_records(args.csv)
    report = summarize(rows)
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")
    if args.plot:
        plot(rows, args.plot)


if __name__ == "__main__":
    main()
