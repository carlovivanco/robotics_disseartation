"""Export TensorBoard curves to portfolio_export/curves.json ({"name": [[step, reward], ...]}).

Usage: python scripts/export_curves.py [--run tensorboard/BoltSD_OSC/TD3/TD3_16] [--points 200]
"""
import argparse
import glob
import json
import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(run_dir, tag):
    files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"no event files in {run_dir}")
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    ev = acc.Scalars(tag)
    return np.array([e.step for e in ev], dtype=np.int64), np.array([e.value for e in ev], dtype=np.float64)


def window_average(steps, vals, n_points):
    """Average over contiguous windows; step = last step of the window. No-op if already <= n_points."""
    if len(steps) <= n_points:
        return steps, vals
    idx = np.array_split(np.arange(len(steps)), n_points)
    return (np.array([steps[i[-1]] for i in idx]), np.array([vals[i].mean() for i in idx]))


def to_series(steps, vals):
    return [[int(s), round(float(v), 1)] for s, v in zip(steps, vals)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="tensorboard/BoltSD_OSC/TD3/TD3_16")
    p.add_argument("--points", type=int, default=200)
    p.add_argument("--out", default="portfolio_export/curves.json")
    a = p.parse_args()

    series = {
        "TD3": ("eval/mean_reward", False),
        "TD3 (train)": ("rollout/ep_rew_mean", True),
    }
    out = {}
    print(f"run: {a.run}")
    print(f"{'series':<14}{'tag':<22}{'raw':>5}{'pts':>5}{'max_step':>10}{'final':>9}")
    for name, (tag, reduce) in series.items():
        s, v = load_scalars(a.run, tag)
        raw = len(s)
        if reduce:
            s, v = window_average(s, v, a.points)
        out[name] = to_series(s, v)
        print(f"{name:<14}{tag:<22}{raw:>5}{len(out[name]):>5}{out[name][-1][0]:>10}{out[name][-1][1]:>9}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
