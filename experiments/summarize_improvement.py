"""Validate the nested train/validation/test records and summarize additive GP ablations."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

CANDIDATES = ("independent", "shared_iso", "shared_ard", "map_0.5", "map_1.0", "map_2.0")


def validate(rows):
    cases = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        case = (row["function"], int(row["dimension"]), int(row["train_count"]), int(row["seed"]))
        candidate = row["candidate"]
        assert candidate in CANDIDATES
        phase, fold = row["phase"], int(row["fold"])
        assert phase in ("cv", "test")
        key = (phase, fold)
        assert key not in cases[case][candidate], f"Duplicate record: {case}, {candidate}, {key}"
        cases[case][candidate][key] = row
        expected = 3 if candidate == "shared_iso" else 6 if candidate == "shared_ard" else case[1] // 2 + 1
        values = [float(value) for value in row["physical_parameters"].split(";")]
        assert len(values) == int(row["parameters"]) == expected
        assert all(math.isfinite(value) and value > 0 for value in values)
        for metric in ("rmse", "nrmse", "r2", "train_rmse", "noise_std", "effective_df", "lml", "objective"):
            assert math.isfinite(float(row[metric])), (case, candidate, metric)
        assert math.isclose(float(row["nrmse"]) ** 2, 1.0 - float(row["r2"]), abs_tol=1e-8)
        assert -1e-3 <= float(row["effective_df"]) <= int(row["fit_count"]) + 1e-3
        penalty = 0.0
        if candidate.startswith("map_"):
            width = float(candidate.split("_")[1])
            initial = [value for _ in range(case[1] // 4) for value in (4.0 / case[1], 0.7)] + [0.001]
            deviations = [width] * (expected - 1) + [1.5]
            penalty = 0.5 * sum((math.log(value / center) / deviation) ** 2
                                for value, center, deviation in zip(values, initial, deviations))
        assert math.isclose(float(row["lml"]) - penalty, float(row["objective"]), abs_tol=1e-7)
        if phase == "cv":
            assert fold in (0, 1, 2) and row["selected"] == "0"
            assert int(row["fit_count"]) + int(row["query_count"]) == case[2]
        else:
            assert fold == -1 and int(row["fit_count"]) == case[2]
    for case, candidates in cases.items():
        assert set(candidates) == set(CANDIDATES)
        selected = []
        scores = []
        for candidate, records in candidates.items():
            assert set(records) == {("cv", 0), ("cv", 1), ("cv", 2), ("test", -1)}
            folds = [records[("cv", index)] for index in range(3)]
            assert sum(int(row["query_count"]) for row in folds) == case[2]
            cv_rmse = math.sqrt(sum(float(row["rmse"]) ** 2 * int(row["query_count"])
                                    for row in folds) / case[2])
            test = records[("test", -1)]
            assert math.isclose(cv_rmse, float(test["validation_rmse"]), abs_tol=1e-8)
            scores.append(cv_rmse)
            if test["selected"] == "1":
                selected.append(cv_rmse)
        assert len(selected) == 1 and selected[0] <= min(scores) + 1e-8
    return cases


def summarize(rows):
    cases = validate(rows)
    grouped = defaultdict(list)
    for row in rows:
        if row["phase"] != "test":
            continue
        key = (row["function"], int(row["dimension"]), int(row["train_count"]))
        grouped[(*key, row["candidate"])].append(row)
        if row["selected"] == "1":
            grouped[(*key, "cv_selected")].append(row)
    labels = (*CANDIDATES, "cv_selected")
    lines = ["# Additive GP improvement results", "",
             f"{len(rows)} fits on {len(cases)} fresh datasets: three training folds and one full refit per candidate.", "",
             "Model selection uses only training-fold RMSE. Test labels are used after the winner is locked.", "",
             "R² values are mean ± sample standard deviation across seeds; test targets are noiseless.", "",
             "| Function | Dimension | Samples | " + " | ".join(labels) + " |",
             "| --- | --- | --- | " + " | ".join("---" for _ in labels) + " |"]
    settings = sorted({key[:3] for key in grouped})
    for setting in settings:
        cells = []
        for label in labels:
            values = [float(row["r2"]) for row in grouped[(*setting, label)]]
            deviation = stdev(values) if len(values) > 1 else 0.0
            cells.append(f"{mean(values):.3f} ± {deviation:.3f}")
        lines.append("| " + " | ".join(map(str, setting)) + " | " + " | ".join(cells) + " |")
    lines += ["", "## Fit and generalization diagnostics", "",
              "Effective degrees of freedom: trace(K_signal (K_signal + noise I)^-1).",
              "Noise standard deviations are restored to the original target scale; injected noise is 0.03.", "",
              "| Function | D | N | Candidate | Test RMSE | Train RMSE | Noise std | Effective df | Refit seconds | Converged |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, runs in sorted(grouped.items()):
        function, dimension, count, candidate = key
        values = [mean(float(row[field]) for row in runs)
                  for field in ("rmse", "train_rmse", "noise_std", "effective_df", "fit_seconds")]
        lines.append(f"| {function} | {dimension} | {count} | {candidate} | " +
                     " | ".join(f"{value:.4f}" for value in values) +
                     f" | {sum(int(row['converged']) for row in runs)}/{len(runs)} |")
    lines += ["", "## Training-only selections", "",
              "| Function | D | N | Seed | Selected | CV RMSE | CV fitting seconds |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for case, candidates in sorted(cases.items()):
        cv_seconds = sum(float(record["fit_seconds"]) for records in candidates.values()
                         for key, record in records.items() if key[0] == "cv")
        chosen = next(records[("test", -1)] for records in candidates.values()
                      if records[("test", -1)]["selected"] == "1")
        lines.append("| " + " | ".join(map(str, case)) + f" | {chosen['candidate']} | "
                     f"{float(chosen['validation_rmse']):.4f} | {cv_seconds:.3f} |")
    failed = [row for row in rows if row["converged"] != "1"]
    lines += ["", f"Optimizer convergence: {len(rows) - len(failed)}/{len(rows)} fits.",
              "All finite results are retained, including optimizer failures or iteration limits.",
              "Refit timings exclude the training-only CV selection cost listed separately.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.csv.open(newline="") as source:
        report = summarize(list(csv.DictReader(source)))
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
