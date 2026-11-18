"""Validate and summarize the paired synthetic-function suite without model selection."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from summarize_learning_curve import LABELS, MODELS, load_records, summarize

FUNCTIONS = ("polynomial", "periodic", "nonsmooth", "multiscale", "cross_pair", "linear", "dense_fourier")


def report(rows):
    functions = {row["function"] for row in rows}
    assert functions == set(FUNCTIONS), "Incomplete function suite"
    assert {row["model"] for row in rows} == set(MODELS)
    seeds = {row["seed"] for row in rows}
    for function in FUNCTIONS:
        assert {row["seed"] for row in rows if row["function"] == function} == seeds
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["function"], int(row["train_count"]), row["model"])].append(row)
    counts = sorted({int(row["train_count"]) for row in rows})
    lines = ["# Paired synthetic-function results", "",
             "Values are test R², mean ± sample SD across seeds. Higher is better; negative values are retained.", "",
             "| Function | Samples | Additive, independent | Additive, shared ARD | Plain Matérn 2.5 |",
             "| --- | --- | --- | --- | --- |"]
    for function in FUNCTIONS:
        for count in counts:
            cells = []
            for model in MODELS:
                values = [float(row["r2"]) for row in grouped[(function, count, model)]]
                cells.append(f"{mean(values):.4f} ± {stdev(values) if len(values) > 1 else 0:.4f}")
            lines.append(f"| {function} | {count} | " + " | ".join(cells) + " |")
    lines += ["", f"Converged: {sum(int(row['converged']) for row in rows)}/{len(rows)} fits. All fits are retained.", ""]
    for function in FUNCTIONS:
        lines += [f"## {function}", "", summarize([row for row in rows if row["function"] == function]).replace(
            "# Training sample learning curve\n", "", 1).replace("## Each seed", "### Each seed")]
    return "\n".join(lines)


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = sorted({int(row["train_count"]) for row in rows})
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    for axis, function in zip(axes.flat, FUNCTIONS):
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
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    figure.suptitle(f"{rows[0]['dimension']}D synthetic functions: mean ± sample SD over "
                   f"{len({row['seed'] for row in rows})} seeds\n"
                   f"Fixed {rows[0]['test_count']}-point test sets; nested training sets; all fits retained")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    rows = load_records(args.csv)
    args.output.write_text(report(rows))
    if args.plot:
        plot(rows, args.plot)


if __name__ == "__main__":
    main()
