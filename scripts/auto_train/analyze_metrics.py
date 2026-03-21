# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extract TensorBoard metrics from a training run and output as JSON.

Usage:
    python scripts/auto_train/analyze_metrics.py --log-dir <path> [--output <path>/metrics.json]
"""

import argparse
import json
import os
import sys

import numpy as np
from tbparse import SummaryReader


def compute_trend(values: list[float], fraction: float = 0.2) -> str:
    """Compute trend from the last `fraction` of values using linear regression slope."""
    if len(values) < 10:
        return "insufficient_data"
    n = max(int(len(values) * fraction), 5)
    tail = values[-n:]
    x = np.arange(len(tail), dtype=np.float64)
    y = np.array(tail, dtype=np.float64)
    # Linear regression: slope = cov(x,y) / var(x)
    slope = np.cov(x, y)[0, 1] / (np.var(x) + 1e-12)
    # Normalize slope relative to the mean magnitude
    mean_abs = np.mean(np.abs(y)) + 1e-12
    normalized_slope = slope / mean_abs
    if normalized_slope > 0.01:
        return "improving"
    elif normalized_slope < -0.01:
        return "degrading"
    return "stable"


def analyze_scalar(values: list[float]) -> dict:
    """Compute summary statistics for a scalar time series."""
    if not values:
        return {"final": None, "mean_last_100": None, "std_last_100": None, "max": None, "trend": "no_data"}

    arr = np.array(values, dtype=np.float64)
    last_100 = arr[-100:] if len(arr) >= 100 else arr

    return {
        "final": float(arr[-1]),
        "mean_last_100": float(np.mean(last_100)),
        "std_last_100": float(np.std(last_100)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "trend": compute_trend(values),
        "num_points": len(values),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract TensorBoard metrics to JSON.")
    parser.add_argument("--log-dir", type=str, required=True, help="Path to training run log directory.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path. Defaults to <log-dir>/metrics.json.")
    args = parser.parse_args()

    log_dir = os.path.abspath(args.log_dir)
    output_path = args.output or os.path.join(log_dir, "metrics.json")

    if not os.path.isdir(log_dir):
        print(f"[ERROR] Log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    # Read all scalar events
    try:
        reader = SummaryReader(log_dir)
        df = reader.scalars
    except Exception as e:
        print(f"[ERROR] Failed to read TensorBoard events: {e}", file=sys.stderr)
        # Write empty report
        report = {"log_dir": log_dir, "error": str(e), "scalars": {}, "reward_terms": {}}
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        sys.exit(1)

    if df.empty:
        print("[WARNING] No scalar data found in TensorBoard events.", file=sys.stderr)
        report = {"log_dir": log_dir, "total_iterations": 0, "scalars": {}, "reward_terms": {}}
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        return

    # Get all unique tags
    tags = df["tag"].unique().tolist()

    # Separate main scalars from reward terms
    scalars = {}
    reward_terms = {}

    for tag in tags:
        tag_df = df[df["tag"] == tag].sort_values("step")
        values = tag_df["value"].tolist()
        summary = analyze_scalar(values)

        if tag.startswith("Episode_Reward/") or tag.startswith("Reward/"):
            # Individual reward term (RSL-RL logs as Episode_Reward/<name>)
            term_name = tag.split("/", 1)[1]
            reward_terms[term_name] = summary
        else:
            scalars[tag] = summary

    # Compute total iterations from max step
    total_iterations = int(df["step"].max()) if not df.empty else 0

    # Compute wall time
    wall_time_seconds = None
    if "wall_time" in df.columns and not df["wall_time"].isna().all():
        wall_time_seconds = float(df["wall_time"].max() - df["wall_time"].min())

    report = {
        "log_dir": log_dir,
        "total_iterations": total_iterations,
        "wall_time_seconds": wall_time_seconds,
        "scalars": scalars,
        "reward_terms": reward_terms,
        "all_tags": tags,
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[INFO] Metrics report written to: {output_path}")
    print(f"[INFO] Total iterations: {total_iterations}, Tags found: {len(tags)}")
    if reward_terms:
        print(f"[INFO] Reward terms: {', '.join(sorted(reward_terms.keys()))}")


if __name__ == "__main__":
    main()
