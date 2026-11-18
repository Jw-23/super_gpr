"""Audit train-only candidate selection and summarize fresh BBOB GP improvements."""

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

from generate_bbob_data import FUNCTIONS

CANDIDATES = ("baseline", "multistart", "quadratic", "asinh_matern", "asinh_quadratic")


def load(paths):
    rows = []
    for path in paths:
        with path.open(newline="") as source:
            rows.extend(csv.DictReader(source))
    tasks = defaultdict(list)
    for row in rows:
        tasks[(row['function'], row['instance'], row['seed'], row['train_count'])].append(row)
        for field in ('rmse', 'nrmse', 'r2', 'train_rmse', 'cv_nrmse', 'lml', 'warp_center', 'warp_scale'):
            assert math.isfinite(float(row[field])), (field, row)
        assert math.isclose(float(row['nrmse']) ** 2, 1 - float(row['r2']), abs_tol=1e-8)
        assert int(row['converged_starts']) <= int(row['starts'])
        assert int(row['starts']) == (1 if row['candidate'] == 'baseline' else 3)
        parameters = list(map(float, row['physical_parameters'].split(';')))
        assert len(parameters) == (4 if 'quadratic' in row['candidate'] else 3)
        assert all(math.isfinite(value) and value > 0 for value in parameters)
    for key, records in tasks.items():
        assert len(records) == 20, (key, len(records))
        indexed = {(row['candidate'], row['phase'], row['fold']): row for row in records}
        assert len(indexed) == 20
        checksums = defaultdict(set)
        for row in records:
            checksums[(row['phase'], row['fold'])].add((row['training_checksum'], row['query_checksum'], row['local_checksum']))
        assert all(len(values) == 1 for values in checksums.values()), key
        scores = []
        for candidate in CANDIDATES:
            folds = [indexed[(candidate, 'cv', str(fold))] for fold in range(3)]
            assert sum(int(row['query_count']) for row in folds) == int(key[-1])
            test = indexed[(candidate, 'test', '-1')]
            for field in ('local_rmse', 'local_nrmse', 'local_r2'):
                assert math.isfinite(float(test[field])), (key, field)
            sd = float(test['rmse']) / float(test['nrmse'])
            assert math.isclose(float(test['local_rmse']) / sd, float(test['local_nrmse']), rel_tol=1e-8)
            pooled_mse = sum(float(row['rmse']) ** 2 * int(row['query_count']) for row in folds) / int(key[-1])
            scores.append(pooled_mse)
            assert float(test['cv_nrmse']) >= 0
        selected = [row for row in records if row['selected'] == '1']
        assert len(selected) == 1 and selected[0]['phase'] == 'test'
        assert selected[0]['candidate'] == CANDIDATES[min(range(5), key=lambda i: scores[i])]
        for phase, fold in (('test', '-1'), ('cv', '0'), ('cv', '1'), ('cv', '2')):
            assert float(indexed[('multistart', phase, fold)]['lml']) >= float(indexed[('baseline', phase, fold)]['lml']) - 1e-7
    assert {key[0] for key in tasks} == {name for _, name in FUNCTIONS}
    instance_sets = [{key[1:3] for key in tasks if key[0] == name} for _, name in FUNCTIONS]
    assert all(value == instance_sets[0] for value in instance_sets)
    return rows


def report(rows):
    tests = [row for row in rows if row['phase'] == 'test']
    selected = [row for row in tests if row['selected'] == '1']
    labels = ['baseline', 'multistart', 'quadratic', 'asinh_matern', 'asinh_quadratic', 'cv_selected']
    groups = defaultdict(list)
    for row in tests:
        groups[(row['function'], row['candidate'])].append(row)
    for row in selected:
        groups[(row['function'], 'cv_selected')].append(row)
    lines = ['# Full-dimensional GP improvement results', '',
             'Mean test R² ± sample SD across fresh instance/data pairs. Higher is better.',
             'CV selection uses pooled original-output squared error in three training folds. All candidate results are retained.', '',
             '| Function | ' + ' | '.join(labels) + ' |',
             '| --- | ' + ' | '.join(['---'] * len(labels)) + ' |']
    for _, function in FUNCTIONS:
        cells = []
        for label in labels:
            values = [float(row['r2']) for row in groups[(function, label)]]
            cells.append(f'{mean(values):.4f} ± {stdev(values) if len(values)>1 else 0:.4f}')
        lines.append(f'| {function} | ' + ' | '.join(cells) + ' |')
    baseline = {(row['function'], row['instance'], row['seed']): row for row in tests if row['candidate'] == 'baseline'}
    wins = sum(float(row['rmse']) < float(baseline[(row['function'],row['instance'],row['seed'])]['rmse']) for row in selected)
    ties = sum(float(row['rmse']) == float(baseline[(row['function'],row['instance'],row['seed'])]['rmse']) for row in selected)
    lines += ['', f'Selected versus baseline: {wins} lower RMSE, {ties} equal, {len(selected)-wins-ties} higher.',
              f'Selected candidates: {dict(Counter(row["candidate"] for row in selected))}.',
              f'Fits: {len(rows)}. Optimizer runs: {sum(int(row["starts"]) for row in rows)}; '
              f'converged: {sum(int(row["converged_starts"]) for row in rows)}. '
              f'Chosen LML run converged in {sum(int(row["best_converged"]) for row in rows)}/{len(rows)} fits.', '',
              '## Paired selected results', '',
              '| Function | Instance | Selected | Baseline R² | Selected R² | Baseline RMSE | Selected RMSE | Baseline local NRMSE | Selected local NRMSE | CV fit/predict seconds | Selected fit/predict seconds |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in selected:
        reference = baseline[(row['function'],row['instance'],row['seed'])]
        cv_seconds = sum(float(r['fit_predict_seconds']) for r in rows if r['phase']=='cv' and r['function']==row['function'] and r['instance']==row['instance'])
        lines.append(f"| {row['function']} | {row['instance']} | {row['candidate']} | " +
                     ' | '.join(f'{float(r[field]):.6g}' for field in ('r2','rmse','local_nrmse') for r in (reference,row)) +
                     f" | {cv_seconds:.3f} | {float(row['fit_predict_seconds']):.3f} |")
    lines += ['', 'Local NRMSE uses the global test target SD. Local targets are never used for candidate selection.',
              'LML values are only compared among starts of the same candidate and training fold, not across output transforms.', '']
    return '\n'.join(lines)


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tests = [row for row in rows if row['phase']=='test']
    figure, axes = plt.subplots(1,2,figsize=(12,6.5), layout='constrained')
    for candidate, label, color, offset in (('baseline','Ordinary Matérn 2.5','#4169a1',-0.13),
                                             ('selected','Training-CV selected','#d17432',0.13)):
        for axis, metric in zip(axes, ('r2','local_nrmse')):
            samples = [[float(row[metric]) for row in tests if row['function']==name
                        and (row['selected']=='1' if candidate=='selected' else row['candidate']==candidate)] for _,name in FUNCTIONS]
            means = [mean(values) for values in samples]
            positions = [index + offset for index in range(len(FUNCTIONS))]
            if metric == 'r2':
                axis.errorbar(means, positions, xerr=[stdev(values) if len(values)>1 else 0 for values in samples],
                              fmt='o', capsize=3, color=color, label=label)
            else:
                axis.plot(means, positions, 'o', color=color, label=label)
                for position, values in zip(positions, samples):
                    axis.scatter(values, [position] * len(values), marker='|', color=color, alpha=0.5)
            axis.set_yticks(range(len(FUNCTIONS)),[name for _,name in FUNCTIONS])
            axis.grid(alpha=.2)
    axes[0].set_xlabel('Global test R² (higher is better)')
    axes[0].axvline(0,color='gray',linewidth=.8)
    axes[1].set_xscale('log')
    local_values = [float(row['local_nrmse']) for row in tests if row['candidate']=='baseline' or row['selected']=='1']
    axes[1].set_xlim(min(local_values)*0.7, max(local_values)*1.5)
    axes[1].set_xlabel('Local RMSE / global SD (lower is better; log scale)')
    for axis in axes:
        axis.invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    figure.suptitle('Full-dimensional GP: 512 training samples, fresh BBOB instances\n'
                   'Global: mean ± sample SD; local: mean and individual instances; training-only selection')
    figure.savefig(path,dpi=180)
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plot',type=Path)
    args=parser.parse_args()
    rows=load(args.csv)
    args.output.write_text(report(rows))
    if args.plot:
        plot(rows,args.plot)


if __name__=='__main__':
    main()
