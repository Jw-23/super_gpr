"""Validate experiment records and summarize paired kernels without external packages."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

MODELS = ("additive", "full_iso", "full_ard")
MODEL_LABELS = {"additive": "Additive", "full_iso": "Full isotropic", "full_ard": "Full ARD"}
KERNEL_LABELS = {"rbf": "RBF", "matern15": "Matérn 1.5", "matern25": "Matérn 2.5"}


def summarize(paths):
    rows = []
    for path in paths:
        with path.open(newline="") as source:
            rows.extend(csv.DictReader(source))
    assert rows, "No experiment records"
    grouped = defaultdict(list)
    paired = defaultdict(dict)
    family_datasets = defaultdict(set)
    for row in rows:
        dimension = int(row["dimension"])
        family = row.get("kernel", "rbf")
        model = "full_iso" if row["model"] == "full_rbf" else row["model"]
        assert model in MODELS and family in KERNEL_LABELS
        key = (row["function"], family, dimension, int(row["train_count"]))
        dataset = (row["function"], dimension, int(row["train_count"]), int(row["test_count"]), int(row["seed"]))
        pair = (family, *dataset)
        assert model not in paired[pair], f"Duplicate run: {pair}, {model}"
        paired[pair][model] = row
        family_datasets[family].add(dataset)
        expected_count = {"additive": dimension // 2 + 1, "full_iso": 3, "full_ard": dimension + 2}[model]
        assert int(row["parameters"]) == expected_count
        parameters = [float(value) for value in row["physical_parameters"].split(";")]
        assert len(parameters) == expected_count and all(math.isfinite(value) and value > 0 for value in parameters)
        assert float(row["lml"]) >= float(row["initial_lml"]) - 1e-7
        assert math.isclose(float(row["nrmse"]) ** 2, 1.0 - float(row["r2"]), abs_tol=1e-8)
        grouped[(*key, model)].append(row)
    for pair, models in paired.items():
        assert set(models) == set(MODELS), f"Unpaired dataset: {pair}"
    datasets = next(iter(family_datasets.values()))
    assert all(cases == datasets for cases in family_datasets.values()), "Kernel families use different dataset keys"
    families = [family for family in KERNEL_LABELS if family in family_datasets]

    def cells_for(function, dimension, count, model, metric):
        cells = []
        for family in families:
            values = [float(row[metric]) for row in grouped[(function, family, dimension, count, model)]]
            deviation = stdev(values) if len(values) > 1 else 0.0
            cells.append(f"{mean(values):.4f} ± {deviation:.4f}")
        return cells

    lines = ["# Prediction results", "",
             f"{len(rows)} fitted models on {len(datasets)} paired datasets across {len(families)} kernel families.", "",
             "Values are mean ± sample standard deviation across seeds. Kernel families share identical dataset keys.", ""]
    for metric, direction in (("r2", "higher is better"), ("rmse", "lower is better")):
        for function in ("additive", "interaction"):
            keys = sorted({(key[2], key[3]) for key in grouped if key[0] == function})
            if not keys:
                continue
            lines += [f"## {function}: {metric.upper()} ({direction})", "",
                      "| Dimension | Training samples | Model | " + " | ".join(KERNEL_LABELS[family] for family in families) + " |",
                      "| --- | --- | --- | " + " | ".join("---" for _ in families) + " |"]
            for dimension, count in keys:
                for model in MODELS:
                    cells = cells_for(function, dimension, count, model, metric)
                    lines.append(f"| {dimension} | {count} | {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")
            lines.append("")
    lines += ["## NRMSE and optimizer diagnostics", "",
              "NRMSE = RMSE / test-target population standard deviation; lower is better.", "",
              "| Function | Kernel | Dimension | Samples | Model | NRMSE | Parameters | Converged | Mean fit seconds |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, runs in sorted(grouped.items()):
        function, family, dimension, count, model = key
        converged = sum(int(row["converged"]) for row in runs)
        lines.append(f"| {function} | {KERNEL_LABELS[family]} | {dimension} | {count} | {MODEL_LABELS[model]} | "
                     f"{mean(float(row['nrmse']) for row in runs):.4f} | {runs[0]['parameters']} | "
                     f"{converged}/{len(runs)} | {mean(float(row['fit_seconds']) for row in runs):.3f} |")
    lines += ["", "These results use known variable groups and a single optimizer start per model.",
              "Matérn nu is fixed; amplitudes, length scales, and noise are optimized jointly.",
              "A finite iteration budget and non-convex LML optimization do not guarantee a global optimum.",
              "Fits that did not converge are retained and reported rather than omitted.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="*", default=[Path(__file__).with_name("additive_results.csv")])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.csv)
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
