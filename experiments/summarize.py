"""Validate experiment records and summarize paired runs without external packages."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

MODELS = ("additive", "full_rbf", "full_ard")


def summarize(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    grouped = defaultdict(list)
    paired = defaultdict(dict)
    for row in rows:
        dimension = int(row["dimension"])
        model = row["model"]
        assert model in MODELS
        key = (row["function"], dimension, int(row["train_count"]))
        pair = (*key, int(row["test_count"]), int(row["seed"]))
        assert model not in paired[pair], f"Duplicate run: {pair}, {model}"
        paired[pair][model] = row
        expected_count = {"additive": dimension // 2 + 1, "full_rbf": 3, "full_ard": dimension + 2}[model]
        assert int(row["parameters"]) == expected_count
        parameters = [float(value) for value in row["physical_parameters"].split(";")]
        assert len(parameters) == expected_count and all(math.isfinite(value) and value > 0 for value in parameters)
        assert float(row["lml"]) >= float(row["initial_lml"]) - 1e-7
        assert math.isclose(float(row["nrmse"]) ** 2, 1.0 - float(row["r2"]), abs_tol=1e-8)
        grouped[(*key, model)].append(row)
    for pair, models in paired.items():
        assert set(models) == set(MODELS), f"Unpaired dataset: {pair}"

    lines = ["# Prediction results", "", f"{len(rows)} fitted models on {len(paired)} paired datasets.", "",
             "Values are mean ± sample standard deviation across seeds; higher R² is better.", ""]
    for function in ("additive", "interaction"):
        lines += [f"## {function}", "", "| Dimension | Training samples | Additive GP R² | Full RBF GP R² | Full ARD GP R² |",
                  "| --- | --- | --- | --- | --- |"]
        keys = sorted({key[:3] for key in grouped if key[0] == function})
        for _, dimension, count in keys:
            cells = []
            for model in MODELS:
                values = [float(row["r2"]) for row in grouped[(function, dimension, count, model)]]
                deviation = stdev(values) if len(values) > 1 else 0.0
                cells.append(f"{mean(values):.3f} ± {deviation:.3f}")
            lines.append(f"| {dimension} | {count} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## NRMSE and optimizer diagnostics", "",
              "NRMSE = RMSE / test-target population standard deviation; lower is better.", "",
              "| Function | Dimension | Samples | Model | NRMSE | Parameters | Converged | Mean fit seconds |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, runs in sorted(grouped.items()):
        function, dimension, count, model = key
        converged = sum(int(row["converged"]) for row in runs)
        lines.append(f"| {function} | {dimension} | {count} | {model} | "
                     f"{mean(float(row['nrmse']) for row in runs):.3f} | {runs[0]['parameters']} | "
                     f"{converged}/{len(runs)} | {mean(float(row['fit_seconds']) for row in runs):.3f} |")
    lines += ["", "These results use known variable groups and a single optimizer start per model.",
              "A finite iteration budget and non-convex LML optimization do not guarantee a global optimum.",
              "Fits that did not converge are retained and reported rather than omitted.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?", default=Path(__file__).with_name("additive_results.csv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.csv)
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
