"""
Compare pretraining runs on the training-FLOPs axis, from the metrics.jsonl that base_train writes into each
checkpoint directory: a table of val bpb at the same compute, and optionally a plot of val bpb and train loss
vs FLOPs. The FLOPs count everything an arm spends per predicted token (see GPT.estimate_flops), so this is the
fair axis between the arms. With --eval-every-flops (as the runs/*.sh scripts set) all runs are evaluated at
the same FLOPs; otherwise the other runs' val bpb is linearly interpolated to the first run's eval points.

python -m scripts.compare_runs baseline_d16 cheat_sheet_d14          # model tags (base_checkpoints/<tag>), or paths
uv run --with matplotlib python -m scripts.compare_runs baseline_d16 cheat_sheet_d14 --plot compare.png
"""

import os
import json
import argparse

import numpy as np

from nanochat.common import get_base_dir


def load_run(arg):
    path = arg
    if not os.path.exists(path):
        path = os.path.join(get_base_dir(), "base_checkpoints", arg)
    if os.path.isdir(path):
        path = os.path.join(path, "metrics.jsonl")
    assert os.path.exists(path), f"no metrics for {arg} at {path}"
    config, val, train = None, {}, {}
    with open(path) as f:
        for record in map(json.loads, f):
            if "config" in record:
                config = record["config"]
            if "val/bpb" in record:
                val[record["step"]] = (record["total_training_flops"], record["val/bpb"])
            if "train/loss" in record:
                train[record["step"]] = (record["total_training_flops"], record["train/loss"])
    name = os.path.basename(os.path.dirname(os.path.abspath(path)))
    val = np.array([val[s] for s in sorted(val)]).reshape(-1, 2)
    train = np.array([train[s] for s in sorted(train)]).reshape(-1, 2)
    return name, config, val, train


parser = argparse.ArgumentParser(description="Compare pretraining runs on the FLOPs axis")
parser.add_argument("runs", nargs="+", help="model tags (base_checkpoints/<tag>/metrics.jsonl) or paths to metrics.jsonl files / their directories")
parser.add_argument("--plot", type=str, default=None, help="save a plot of val bpb and train loss vs FLOPs to this path (needs matplotlib)")
args = parser.parse_args()
runs = [load_run(r) for r in args.runs]

# Per-run summary
print(f"{'run':32s} {'ref_mode':>8s} {'depth':>5s} {'params':>13s} {'flops/tok':>10s} {'steps':>13s} {'FLOPs':>9s} {'final bpb':>9s} {'min bpb':>9s}")
for name, config, val, train in runs:
    uc = config["user_config"]
    flops = max(val[-1, 0] if len(val) else 0, train[-1, 0] if len(train) else 0)
    steps = f"{len(train)}/{config['num_iterations']}"
    final_bpb = f"{val[-1, 1]:.4f}" if len(val) else "-"
    min_bpb = f"{val[:, 1].min():.4f}" if len(val) else "-"
    print(f"{name:32s} {uc['ref_mode']:>8s} {uc['depth']:5d} {config['param_counts']['total']:13,d} {config['flops_per_token']:10.3e} "
          f"{steps:>13s} {flops:9.2e} {final_bpb:>9s} {min_bpb:>9s}")
print()

# Val bpb at the same compute: the first run's eval points, the other runs interpolated (exact when the evals line up)
ref_flops = runs[0][2][:, 0]
header = f"{'FLOPs':>9s} " + " ".join(f"{name[-24:]:>24s}" for name, *_ in runs)
if len(runs) == 2:
    header += f" {'diff':>8s}"
print(header)
for x in ref_flops:
    row = []
    for _, _, val, _ in runs:
        # (runs of the same budget end up to a step apart in FLOPs, so their final evals still count as the same compute)
        inside = len(val) and val[0, 0] <= x <= val[-1, 0] * 1.02
        row.append(np.interp(x, val[:, 0], val[:, 1]) if inside else None)
    line = f"{x:9.2e} " + " ".join(f"{v:24.4f}" if v is not None else f"{'-':>24s}" for v in row)
    if len(runs) == 2 and None not in row:
        line += f" {row[1] - row[0]:+8.4f}"
    print(line)

if args.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, _, val, train in runs:
        axes[0].plot(val[:, 0], val[:, 1], marker=".", label=name)
        axes[1].plot(train[:, 0], train[:, 1], label=name)
    for ax, ylabel, curves in [(axes[0], "val bpb", [r[2] for r in runs]), (axes[1], "train loss (EMA)", [r[3] for r in runs])]:
        # zoom past the initial drop: the top of the y range is the worst run's value at 10% of the compute
        tops = [c[np.searchsorted(c[:, 0], 0.1 * c[-1, 0]), 1] for c in curves if len(c)]
        bottoms = [c[:, 1].min() for c in curves if len(c)]
        if tops:
            ax.set_ylim(min(bottoms) - 0.02 * (max(tops) - min(bottoms)), max(tops))
        ax.set_xlabel("training FLOPs")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.plot, dpi=120)
    print(f"\nSaved plot to {args.plot}")
