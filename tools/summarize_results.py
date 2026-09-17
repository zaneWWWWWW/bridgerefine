"""Recompute tables from local exports; never overwrite the source evidence."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(pvalues):
    """Step-down Holm adjustment with the required cumulative maximum."""
    order = np.argsort(pvalues, kind='stable')
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True,
                        help='Folder containing patient_metrics_final.csv and significance.json')
    parser.add_argument('--seeds', type=Path, required=True, help='Archived seed_results.csv')
    parser.add_argument('--efficiency', type=Path, required=True, help='efficiency_final_10x2.csv')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory already exists. Choose a new directory to preserve prior results.')

    rows = read_csv(args.metrics / 'patient_metrics_final.csv')
    grouped = defaultdict(dict)
    for row in rows:
        group = grouped[(row['method'], row['seed'])]
        if row['case_id'] in group:
            raise ValueError('Duplicate patient within method/seed')
        group[row['case_id']] = row
    base = grouped[('CT-only', '42')]
    patients = sorted(base)
    if len(patients) != 18 or sum(int(r['n_slices']) for r in base.values()) != 3415:
        raise ValueError('Expected 18 test patients and 3415 paired slices')
    summaries = []
    for (method, seed), group in grouped.items():
        if set(group) != set(patients):
            raise ValueError(f'Patient set mismatch: {method}/{seed}')
        summary = dict(method=method, seed=seed, n_patients=len(group))
        for metric in ['ssim', 'psnr_db', 'lpips', 'mask_ssim', 'mask_psnr_db', 'mad_ratio']:
            values = np.array([float(group[pid][metric]) for pid in patients])
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite metric: {method}/{metric}')
            summary[metric + '_mean'] = float(values.mean())
            summary[metric + '_sample_sd'] = float(values.std(ddof=1))
        summaries.append(summary)

    bridge = grouped[('BridgeRefine-L1', '42')]
    comparisons = []
    for method in ['SelfRDB', 'SynDiff', 'MG-CycleGAN', 'BridgeGAN', 'coarse-only', 'CT-only']:
        seed = '42' if method in ['coarse-only', 'CT-only'] else ''
        other = grouped[(method, seed)]
        # Exported SSIM values have four decimal places. Rounding differences
        # avoids introducing spurious rank differences through float subtraction.
        delta = np.round([float(bridge[p]['ssim']) - float(other[p]['ssim']) for p in patients], 4)
        raw = float(wilcoxon(delta, alternative='two-sided', method='auto').pvalue)
        comparisons.append(dict(comparison=f'BridgeRefine-L1 vs {method}', raw_p=raw,
                                mean_delta_ssim=float(delta.mean())))
    for row, adjusted in zip(comparisons, holm_adjust([r['raw_p'] for r in comparisons])):
        row['holm_p'] = adjusted

    delta = np.round([float(bridge[p]['ssim']) - float(base[p]['ssim']) for p in patients], 4)
    rng = np.random.default_rng(42)
    boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    ci = np.percentile(boot, [2.5, 97.5]).tolist()
    archived = json.loads((args.metrics / 'significance.json').read_text(encoding='utf-8-sig'))
    archived_by_name = {r['comparison']: r for r in archived['comparisons']}
    comparison_audit = []
    for row in comparisons:
        old = archived_by_name[row['comparison']]
        comparison_audit.append(dict(comparison=row['comparison'],
                                    archived_raw_p=old['raw_p'], recomputed_raw_p=row['raw_p'],
                                    archived_holm_p=old['holm_p'], recomputed_holm_p=row['holm_p']))

    seed_groups = defaultdict(list)
    for row in read_csv(args.seeds):
        seed_groups[row['model']].append(row)
    seed_summary = []
    for method, group in seed_groups.items():
        seeds = [r['seed'] for r in group]
        if len(seeds) != len(set(seeds)) or len(seeds) < 2:
            raise ValueError('Duplicate or insufficient seed records')
        values = np.array([float(r['ssim']) for r in group])
        seed_summary.append(dict(model=method, n_runs=len(values), seeds=','.join(seeds),
                                 ssim_mean=float(values.mean()), sample_sd=float(values.std(ddof=1)),
                                 population_sd=float(values.std(ddof=0)),
                                 precision_note='Computed from archived rounded per-run SSIM, not full raw runs'))
    efficiency = read_csv(args.efficiency)
    methods = {r['method']: r for r in efficiency}
    if methods['BridgeRefine']['sampling_steps'] != '10' or methods['BridgeRefine']['recursive_estimates'] != '2':
        raise ValueError('Expected final 10-step, 2-recursion efficiency protocol')
    ratio = float(methods['BridgeRefine']['time_ms_mean']) / float(methods['CT-only']['time_ms_mean'])

    report = dict(n_patients=len(patients), n_slices=3415, mean_delta=float(delta.mean()),
                  median_delta=float(np.median(delta)), improved_patients=int((delta > 0).sum()),
                  bootstrap_ci95=ci, bootstrap='NumPy default_rng(42), 10000 percentile resamples',
                  wilcoxon='two-sided, rounded paired differences, scipy method=auto',
                  comparisons=comparisons, archived_comparison_audit=comparison_audit,
                  latency_ratio=ratio,
                  caveats=['Outputs are recomputed from exported, rounded metrics.',
                           'Archived p-values and CI remain preserved in the source directory.',
                           'Differences require source-record review, not silent manuscript replacement.'])
    args.output.mkdir(parents=True)
    write_csv(args.output / 'patient_summary_recomputed.csv', summaries)
    write_csv(args.output / 'seed_summary_recomputed.csv', seed_summary)
    write_csv(args.output / 'efficiency_10x2.csv', efficiency)
    write_csv(args.output / 'significance_comparison.csv', comparison_audit)
    (args.output / 'statistics_recomputed.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), patients=len(patients), improved=report['improved_patients'],
                          mean_delta=report['mean_delta'], latency_ratio=ratio), indent=2))


if __name__ == '__main__':
    main()
