"""Validate and summarize the paired synthetic-function suite without model selection."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from summarize_learning_curve import LABELS, MODELS, load_records, summarize
from generate_bbob_data import FUNCTIONS as BBOB_FUNCTIONS

FUNCTIONS = ("polynomial", "periodic", "nonsmooth", "multiscale", "cross_pair", "linear", "dense_fourier")


def report(rows, functions=FUNCTIONS):
    assert {row["function"] for row in rows} == set(functions), "Incomplete function suite"
    assert {row["model"] for row in rows} == set(MODELS)
    seeds = {row["seed"] for row in rows}
    for function in functions:
        assert {row["seed"] for row in rows if row["function"] == function} == seeds
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["function"], int(row["train_count"]), row["model"])].append(row)
    counts = sorted({int(row["train_count"]) for row in rows})
    replication = "instance/data pairs" if any(int(row.get("instance") or 0) > 0 for row in rows) else "seeds"
    lines = ["# Paired function results", "",
             f"Values are test R², mean ± sample SD across {replication}. Higher is better; negative values are retained.", "",
             "| Function | Samples | Additive, independent | Additive, shared ARD | Plain Matérn 2.5 |",
             "| --- | --- | --- | --- | --- |"]
    for function in functions:
        for count in counts:
            cells = []
            for model in MODELS:
                values = [float(row["r2"]) for row in grouped[(function, count, model)]]
                cells.append(f"{mean(values):.4f} ± {stdev(values) if len(values) > 1 else 0:.4f}")
            lines.append(f"| {function} | {count} | " + " | ".join(cells) + " |")
    lines += ["", f"Converged: {sum(int(row['converged']) for row in rows)}/{len(rows)} fits. All fits are retained.", ""]
    for function in functions:
        lines += [f"## {function}", "", summarize([row for row in rows if row["function"] == function]).replace(
            "# Training sample learning curve\n", "", 1).replace("## Each seed", "### Each seed").replace(
                "one noisy pool", "one observation pool").replace("across seeds", f"across {replication}")]
    return "\n".join(lines)


def plot(rows, path, functions=FUNCTIONS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = sorted({int(row["train_count"]) for row in rows})
    height = (len(functions) + 3) // 4
    figure, axes = plt.subplots(height, 4, figsize=(14, 3.4 * height), layout="constrained")
    for axis, function in zip(axes.flat, functions):
        for model, color in zip(MODELS, ("#4169a1", "#d17432", "#389269")):
            values = [[float(row["r2"]) for row in rows if row["function"] == function
                       and row["model"] == model and int(row["train_count"]) == count] for count in counts]
            axis.errorbar(counts, [mean(samples) for samples in values],
                          yerr=[stdev(samples) if len(samples) > 1 else 0 for samples in values],
                          marker="o", capsize=3, color=color, label=LABELS[model])
        axis.set_title(function.replace("_", " "))
        axis.set_xscale("log", base=2)
        axis.set_xticks(counts, [str(count) for count in counts])
        axis.set_xlabel("Training samples")
        axis.set_ylabel("Test R²")
        axis.axhline(0, linewidth=0.8, color="gray")
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    for axis in list(axes.flat)[len(functions):]:
        axis.axis("off")
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.suptitle(f"{rows[0]['dimension']}D paired functions: mean ± sample SD over "
                   f"{len({row['seed'] for row in rows})} seeds\n"
                   f"Fixed {rows[0]['test_count']}-point test sets; nested training sets; all fits retained")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def validate_local(rows):
    import math

    metadata = defaultdict(set)
    baselines = defaultdict(set)
    for row in rows:
        assert int(row["instance"]) > 0
        for field in ("local_rmse", "local_nrmse", "local_r2", "constant_local_nrmse"):
            assert math.isfinite(float(row[field])), (row["function"], field)
        metadata[(row["function"], row["seed"])].add((row["instance"], row["local_checksum"]))
        baselines[(row["function"], row["seed"], row["train_count"])].add(row["constant_local_nrmse"])
        global_sd = float(row["rmse"]) / float(row["nrmse"])
        assert math.isclose(float(row["local_rmse"]) / global_sd, float(row["local_nrmse"]), rel_tol=1e-8)
    assert all(len(values) == 1 for values in metadata.values()), "Local test data changed"
    assert all(len(values) == 1 for values in baselines.values()), "Training mean baseline changed across models"


def local_report(rows, functions):
    count = max(int(row["train_count"]) for row in rows)
    lines = ["", "## Near-optimum diagnostics", "",
             f"{count} training samples. Error is RMSE / global test target SD; lower is better.",
             "The training set contains only global uniform samples; optimum information only constructs the held-out local test set.",
             "Values are means across instance/data pairs. The constant baseline predicts the training target mean.", "",
             "| Function | Additive, independent | Additive, shared ARD | Plain Matérn 2.5 | Constant baseline |",
             "| --- | --- | --- | --- | --- |"]
    for function in functions:
        subset = [row for row in rows if row["function"] == function and int(row["train_count"]) == count]
        cells = [f"{mean(float(row['local_nrmse']) for row in subset if row['model'] == model):.4g}" for model in MODELS]
        baseline = mean(float(row["constant_local_nrmse"]) for row in subset if row["model"] == MODELS[0])
        lines.append(f"| {function} | " + " | ".join(cells) + f" | {baseline:.4g} |")
    lines += ["", "### Each local test", "",
              "| Function | Instance | Seed | Samples | Model | RMSE | RMSE/global SD | Local R² |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['function']} | {row['instance']} | {row['seed']} | {row['train_count']} | "
                     f"{LABELS[row['model']]} | {float(row['local_rmse']):.6g} | "
                     f"{float(row['local_nrmse']):.6g} | {float(row['local_r2']):.6g} |")
    return "\n".join(lines) + "\n"


def local_plot(rows, path, functions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    count = max(int(row["train_count"]) for row in rows)
    values = []
    for function in functions:
        subset = [row for row in rows if row["function"] == function and int(row["train_count"]) == count]
        values.append([mean(float(row["local_nrmse"]) for row in subset if row["model"] == model)
                       for model in MODELS] + [mean(float(row["constant_local_nrmse"]) for row in subset
                                                  if row["model"] == MODELS[0])])
    figure, axis = plt.subplots(figsize=(9, 7), layout="constrained")
    heatmap = axis.imshow(np.log10(np.maximum(values, 1e-12)), cmap="YlOrRd", aspect="auto")
    axis.set_yticks(range(len(functions)), functions)
    axis.set_xticks(range(4), ("Additive\nindependent", "Additive\nshared ARD", "Plain\nMatérn 2.5", "Training\nmean"))
    for row, samples in enumerate(values):
        for column, value in enumerate(samples):
            axis.text(column, row, f"{value:.2g}", ha="center", va="center", color="black")
    figure.colorbar(heatmap, ax=axis, label="log10(RMSE / global test SD), lower is better")
    axis.set_title(f"Near-optimum error after {count} global training samples\n"
                   f"Mean over {len({row['seed'] for row in rows})} instance/data pairs; no local training")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--local-plot", type=Path)
    parser.add_argument("--bbob", action="store_true")
    args = parser.parse_args()
    rows = load_records(args.csv)
    functions = tuple(name for _, name in BBOB_FUNCTIONS) if args.bbob else FUNCTIONS
    extra = ""
    if args.bbob:
        validate_local(rows)
        extra = local_report(rows, functions)
    args.output.write_text(report(rows, functions) + extra)
    if args.plot:
        plot(rows, args.plot, functions)
    if args.local_plot:
        assert args.bbob, "Local diagnostics require BBOB records"
        local_plot(rows, args.local_plot, functions)


if __name__ == "__main__":
    main()
