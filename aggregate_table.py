"""Aggregate 7-method x 7-dataset sweep results into tables."""
import pandas as pd
import glob
import re
import numpy as np
import collections
import os

LOG_DIR = '/work1/jianxu/dbvi/logs/sweep'
METHOD_ORDER = ['dsvi', 'fbvi', 'fbvi-noisy', 'fbvi-bridge', 'ddvi', 'dbvi-s', 'ipvi']
DATASET_ORDER = ['yacht', 'boston', 'energy', 'qsar', 'concrete', 'power', 'protein']


def parse_all():
    buckets = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(LOG_DIR, '*.csv'))):
        base = os.path.basename(f)
        m = re.match(r'^(?P<meth>[\w-]+?)__(?P<ds>\w+?)__seed(?P<sd>\d+)\.csv$', base)
        if not m:
            continue
        method = m.group('meth')
        dataset = m.group('ds')
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if 'rmse' not in df.columns or len(df) == 0:
            continue
        rmse = float(df['rmse'].iloc[-1])
        nll = float(df['nll'].iloc[-1]) if 'nll' in df.columns else float('nan')
        # filter exploded runs
        if not np.isfinite(rmse):
            continue
        if abs(rmse) > 10000:
            continue
        buckets[(method, dataset)].append((rmse, nll))
    return buckets


def fmt_cell(arr):
    if not arr:
        return '   --   '
    m, s = np.mean(arr), np.std(arr)
    return f"{m:.3f}+/-{s:.3f}"


def best_per_dataset(buckets, idx, lower_better=True):
    res = {}
    for ds in DATASET_ORDER:
        best = None
        best_val = float('inf') if lower_better else -float('inf')
        for me in METHOD_ORDER:
            vals = [v[idx] for v in buckets.get((me, ds), []) if np.isfinite(v[idx])]
            if not vals:
                continue
            mean_v = float(np.mean(vals))
            if (lower_better and mean_v < best_val) or ((not lower_better) and mean_v > best_val):
                best_val = mean_v
                best = me
        res[ds] = best
    return res


def print_table(buckets, idx, title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    header = "| {:<13} |".format("method") + " | ".join(f"{ds:^13}" for ds in DATASET_ORDER) + " |"
    print(header)
    print("|" + "-" * 15 + "|" + ("-" * 15 + "|") * len(DATASET_ORDER))
    winners = best_per_dataset(buckets, idx)
    for me in METHOD_ORDER:
        row = "| {:<13} |".format(me)
        for ds in DATASET_ORDER:
            vals = [v[idx] for v in buckets.get((me, ds), []) if np.isfinite(v[idx])]
            cell = fmt_cell(vals)
            if winners.get(ds) == me and vals:
                cell = "**" + cell + "**"
            row += f" {cell:<13} |"
        print(row)


def main():
    buckets = parse_all()
    print_table(buckets, 0, "Test RMSE (lower is better; best in bold)")
    print_table(buckets, 1, "Test NLL (lower is better; best in bold)")
    print("\nMethod counts:")
    for me in METHOD_ORDER:
        per_ds = [len(buckets.get((me, ds), [])) for ds in DATASET_ORDER]
        total = sum(per_ds)
        print(f"  {me:<13} total={total:3d}  per_ds={per_ds}")


if __name__ == '__main__':
    main()
