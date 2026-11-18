"""Generate transformed BBOB observations for the C++ regression experiment."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

FUNCTIONS = (
    (7, "f07_step_ellipsoid"), (9, "f09_rosenbrock_rotated"),
    (10, "f10_ellipsoid_rotated"), (11, "f11_discus"),
    (12, "f12_bent_cigar"), (13, "f13_sharp_ridge"),
    (15, "f15_rastrigin_rotated"), (16, "f16_weierstrass"),
    (18, "f18_schaffers1000"), (21, "f21_gallagher101"),
    (23, "f23_katsuura"), (24, "f24_lunacek"),
)
RUNS = ((1, 21001), (2, 22003), (3, 23003))


def generate(root, instance, seed, quick):
    import ioh
    import numpy as np

    assert importlib.metadata.version("ioh") == "0.3.22"
    dimension, training, testing, local = (20, 64, 128, 64) if quick else (80, 512, 2048, 512)
    train_x = np.random.default_rng(seed).uniform(-5, 5, (training, dimension))
    test_x = np.random.default_rng(seed + 1000000).uniform(-5, 5, (testing, dimension))
    local_offsets = np.random.default_rng(seed + 2000000).normal(0, 0.1, (local, dimension))
    folder = root / f"run_{instance}"
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {"ioh_version": importlib.metadata.version("ioh"), "numpy_version": np.__version__,
                "dimension": dimension, "instance": instance, "seed": seed,
                "training_count": training, "global_test_count": testing, "local_test_count": local,
                "input_mapping": "GP input = BBOB input / 5", "noise": 0,
                "local_distribution": "clip(xopt + Normal(0, 0.1^2 I), -5, 5)", "cases": []}
    manifest = [f"{seed} {instance}"]
    for fid, name in FUNCTIONS:
        problem = ioh.get_problem(fid, instance, dimension, ioh.ProblemClass.BBOB)
        assert problem.meta_data.n_variables == dimension
        optimum = np.asarray(problem.optimum.x)
        assert np.isclose(problem(optimum.tolist()), problem.optimum.y, rtol=1e-9, atol=1e-7)
        local_x = np.clip(optimum + local_offsets, -5, 5)
        blocks = []
        for x in (train_x, test_x, local_x):
            y = np.array([problem(row.tolist()) for row in x])
            assert np.isfinite(y).all() and y.std() > 0
            blocks.append(np.column_stack((x / 5, y)))
        path = folder / f"{name}.txt"
        with path.open("w") as target:
            target.write(f"{dimension} {training} {testing} {local}\n")
            for block in blocks:
                np.savetxt(target, block, fmt="%.17g")
        manifest.append(f"{name} {path.name}")
        metadata["cases"].append({"function_id": fid, "name": name, "implementation_name": problem.meta_data.name,
                                  "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "train_input_sha256": hashlib.sha256(train_x.tobytes()).hexdigest(),
                                  "global_input_sha256": hashlib.sha256(test_x.tobytes()).hexdigest()})
    (folder / "cases.txt").write_text("\n".join(manifest) + "\n")
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Generated {len(FUNCTIONS)} cases: instance={instance}, seed={seed}, dimension={dimension}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=int, choices=(1, 2, 3))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Use preregistered improvement instances 4/5/6")
    args = parser.parse_args()
    runs = ((1, 101),) if args.quick else RUNS
    if args.fresh and not args.quick:
        runs = ((4, 31013), (5, 32003), (6, 33013))
    for run, (instance, seed) in enumerate(runs, 1):
        if args.run is None or run == args.run:
            generate(args.output, instance, seed, args.quick)


if __name__ == "__main__":
    main()
