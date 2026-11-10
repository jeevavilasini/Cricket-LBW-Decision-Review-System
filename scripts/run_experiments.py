"""
Run the full experimental programme and write CSV results, figures and a
markdown summary.

    python scripts/run_experiments.py            # the full suite
    python scripts/run_experiments.py --quick    # a fast subset, for checking
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from drs import experiments as X  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="outputs/results")
    p.add_argument("--quick", action="store_true", help="small ensembles; for a smoke test")
    p.add_argument("--seed", type=int, default=20240)
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=("noise", "frames", "missing", "models", "camera", "uncertainty"),
        help="run only the named experiments",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if args.quick:
        n = 6
        n_rep = 1
    else:
        n = 20
        n_rep = 3

    want = set(args.only) if args.only else {"noise", "frames", "missing", "models", "camera", "uncertainty"}
    results: dict[str, object] = {}

    pd.set_option("display.width", 200)

    if "noise" in want:
        print("[noise] measurement noise sweep ...")
        trials, summary = X.experiment_noise(
            n_deliveries=n, n_repeats=n_rep, seed=args.seed, out_dir=out
        )
        results["noise_trials"] = trials
        print(_greeting(summary, "median error at wicket plane (mm)", "noise_mm"))

    if "frames" in want:
        print("\n[frames] frame budget ...")
        trials, summary = X.experiment_frames(
            n_deliveries=n, n_repeats=n_rep, seed=args.seed, out_dir=out
        )
        results["frames_trials"] = trials
        print(_greeting(summary, "median error at wicket plane (mm)", "n_frames"))

    if "missing" in want:
        print("\n[missing] dropped detections ...")
        trials, summary = X.experiment_missing(
            n_deliveries=n, n_repeats=n_rep, seed=args.seed, out_dir=out
        )
        results["missing_trials"] = trials
        print(_greeting(summary, "median error at wicket plane (mm)", "dropout"))

    if "models" in want:
        print("\n[models] estimator comparison ...")
        if args.quick:
            trials, summary = X.experiment_models(
                n_deliveries=n, models=("polynomial", "ballistic", "post_bounce", "piecewise", "kalman"),
                seed=args.seed, out_dir=out,
            )
        else:
            trials, summary = X.experiment_models(n_deliveries=n, seed=args.seed, out_dir=out)
        results["models_trials"] = trials
        print(summary.pivot_table(index="model", columns="noise_mm", values="median").round(2).to_string())

    if "camera" in want:
        print("\n[camera] geometry and calibration ...")
        kw = dict(n_deliveries=4 if args.quick else 12, seed=args.seed, out_dir=out)
        if args.quick:
            kw.update(baselines=(1.0, 4.0), pixel_noises=(0.5,), focal_errors=(0.0, 0.05))
        b, p, f = X.experiment_camera(**kw)
        results["camera_baseline"], results["camera_pixel"], results["camera_focal"] = b, p, f
        print(b.groupby("baseline_m")["reconstruction_rms_mm"].median().round(2).to_string())

    if "uncertainty" in want:
        print("\n[uncertainty] Monte-Carlo convergence and coverage ...")
        if args.quick:
            conv, cov = X.experiment_uncertainty(sample_sizes=(100, 1000), n_trials=4, seed=args.seed, out_dir=out)
        else:
            conv, cov = X.experiment_uncertainty(seed=args.seed, out_dir=out)
        results["uncertainty_convergence"], results["uncertainty_coverage"] = conv, cov
        if len(cov):
            print(
                f"  95% interval coverage: y {cov['covers_y'].mean():.0%}, "
                f"z {cov['covers_z'].mean():.0%}, both {cov['covers_both'].mean():.0%} "
                f"(nominal 95%, n={len(cov)})"
            )

    summary_path = X.write_summary(results, out / "SUMMARY.md")
    print(f"\nWrote {summary_path}")
    print(f"Total wall time: {time.perf_counter() - started:.1f} s")
    return 0


def _greeting(summary: pd.DataFrame, metric: str, column: str) -> str:
    table = summary.pivot_table(index=column, columns="model", values="median")
    return f"  {metric}:\n{table.round(2).to_string()}"


if __name__ == "__main__":
    raise SystemExit(main())
