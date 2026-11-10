"""
End-to-end demonstration of the DRS model.

Runs one delivery through the whole pipeline and writes the figures that the
report uses:

    simulate  ->  observe  ->  estimate  ->  decide  ->  quantify uncertainty

Two observation paths are available:

* the **ideal** path (default) samples the true trajectory directly, which is
  the right way to study the *estimators*;
* the **vision** path (``--video``) renders synthetic camera frames, runs the
  colour detector, and triangulates two views, which is the right way to study
  the *pipeline*.

Example
-------
    python scripts/run_demo.py --preset leg_spin_marginal
    python scripts/run_demo.py --video --preset outside_off
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from drs import physics, viz  # noqa: E402
from drs.decision import review_lbw  # noqa: E402
from drs.estimate import ESTIMATORS, fit, state_at_plane  # noqa: E402
from drs.geometry import Wicket, default_stereo_rig  # noqa: E402
from drs.simulate import build_ground_truth, make_observations  # noqa: E402
from drs.uncertainty import bootstrap_wicket, monte_carlo_wicket  # noqa: E402

FIG_DIR = Path("outputs/figures")
RESULTS_DIR = Path("outputs/results")

MODELS = ("polynomial", "ballistic", "post_bounce", "piecewise", "kalman")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="good_length_middle", choices=sorted(physics.PRESETS))
    p.add_argument("--fps", type=float, default=50.0, help="observation frame rate (Hz)")
    p.add_argument("--noise-mm", type=float, default=5.0, help="per-coordinate measurement noise (mm)")
    p.add_argument("--dropout", type=float, default=0.0, help="fraction of frames not detected")
    p.add_argument("--monte-carlo", type=int, default=20000, help="Monte-Carlo sample count")
    p.add_argument("--on-field", default="not_out", choices=("out", "not_out"))
    p.add_argument("--out-dir", default="outputs", help="root output directory")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--video", action="store_true", help="run the synthetic-video vision pipeline")
    p.add_argument("--render-fps", type=float, default=150.0, help="frame rate for the rendered clip")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    fig_dir = Path(args.out_dir) / "figures"
    res_dir = Path(args.out_dir) / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    wicket = Wicket()
    truth = build_ground_truth(physics.preset(args.preset), label=args.preset)
    t_true, r_true, v_true = truth.state_at_wicket()

    print("=" * 74)
    print(f"DELIVERY: {args.preset}")
    print("=" * 74)
    print(f"  release position   {np.round(truth.release, 3)} m")
    print(f"  release velocity   {np.round(truth.release_velocity, 3)} m/s "
          f"({np.linalg.norm(truth.release_velocity) * 3.6:.1f} km/h)")
    print(f"  bounce             t = {truth.bounce_time:.4f} s at "
          f"{np.round(truth.trajectory.pitch_point(), 3)}")
    print(f"  at the stumps      {np.round(r_true, 4)} m  "
          f"({np.linalg.norm(v_true) * 3.6:.1f} km/h)")
    print(f"  true clearance     {float(wicket.clearance(r_true.reshape(1, 3))[0]) * 1000:+.1f} mm "
          f"-> {'HITS' if wicket.hits(r_true.reshape(1, 3))[0] else 'MISSES'}")

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    if args.video:
        obs = _observations_from_video(truth, args, fig_dir)
    else:
        obs = make_observations(
            truth,
            fps=args.fps,
            noise_std=args.noise_mm / 1000.0,
            dropout=args.dropout,
            rng=args.seed,
        )
    print(f"\nObservations: {obs.n_visible}/{len(obs)} frames used, "
          f"sigma = {obs.sigma * 1000:.2f} mm  (source: {obs.source})")

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------
    print("\n" + "-" * 74)
    print(f"{'model':<16}{'fit RMSE':>10}{'arrival y':>12}{'arrival z':>12}{'3-D err':>11}{'time':>10}")
    print("-" * 74)
    import time

    fits = []
    for name in MODELS:
        t0 = time.perf_counter()
        try:
            f = fit(obs, name)
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
            print(f"{name:<16}  failed: {exc}")
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        fits.append(f)
        p = state_at_plane(f)
        err = float(np.linalg.norm(p - r_true)) * 1000
        print(f"{name:<16}{f.residual_rmse * 1000:>9.2f}mm{p[1]:>12.4f}{p[2]:>12.4f}"
              f"{err:>10.1f}mm{elapsed_ms:>8.1f}ms")
    print("-" * 74)

    # ------------------------------------------------------------------
    # Decision and uncertainty
    # ------------------------------------------------------------------
    best = next((f for f in fits if f.model == "piecewise"), fits[-1] if fits else None)
    unc = None
    if best is not None:
        verdict = review_lbw(best, wicket=wicket, on_field=args.on_field)
        print("\nDRS-STYLE REVIEW (piecewise trajectory)")
        print(verdict.describe())
        for reason in verdict.reasons:
            print(f"   - {reason}")

        unc = monte_carlo_wicket(best, wicket=wicket, n=args.monte_carlo, rng=args.seed)
        print(f"\nUNCERTAINTY (parameter-covariance Monte Carlo, N = {unc.n_samples})")
        print(f"   P(hit)             {unc.hit_probability:.4f}   ({unc.n_hits}/{unc.n_samples})")
        print(f"   arrival y          {unc.mean_point[1]:+.4f} "
              f"[{unc.interval(1)[0]:+.4f}, {unc.interval(1)[1]:+.4f}]  (95%)")
        print(f"   arrival z          {unc.mean_point[2]:.4f} "
              f"[{unc.interval(2)[0]:.4f}, {unc.interval(2)[1]:.4f}]  (95%)")
        print(f"   clearance          {unc.clearances.mean() * 1000:+.2f} mm "
              f"+/- {unc.clearances.std() * 1000:.2f} mm")
        if args.dropout > 0:
            with np.errstate(all="ignore"):
                boot = bootstrap_wicket(obs, model="piecewise", n=100, rng=args.seed)
            print(f"   bootstrap P(hit)   {boot.hit_probability:.4f}   (N = {boot.n_samples})")

        _write_report(res_dir / "demo_report.md", args, truth, obs, fits, verdict, unc, wicket)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    paths = _make_figures(truth, obs, fits, wicket, unc, fig_dir, args.preset)
    print("\nFigures written:")
    for p in paths:
        print(f"   {p}")
    return 0


def _observations_from_video(truth, args, fig_dir: Path):
    """Render a synthetic stereo clip, detect, and triangulate it."""
    from drs import synthvideo
    from drs.tracking import pixel_rmse, track_pixels, triangulate_tracks
    from drs.video import ImageSequence

    cams = default_stereo_rig()
    root = Path(args.out_dir) / "synth_frames"
    print(f"\nRendering synthetic stereo footage to {root} ...")
    seqs = synthvideo.render_stereo_rig(
        truth.trajectory, cams, root, fps=args.render_fps, blur_subframes=3, seed=args.seed
    )

    tracks = []
    for cam in cams:
        seq = seqs[cam.name]
        stream = ImageSequence(seq.directory, fps=seq.fps)
        track, _ = track_pixels(stream.frames(), fps=seq.fps, gate_px=500.0, min_area=4)
        print(f"   camera {cam.name}: detected {track.n_visible}/{len(track)} frames, "
              f"pixel RMSE {pixel_rmse(track, seq.true_pixels):.3f} px")
        tracks.append(track)

        import matplotlib.pyplot as plt

        from drs.detection import detect_ball

        mid = int(len(track) * 0.6)
        _, frame = list(stream.frames())[mid]
        blob = detect_ball(frame, expected=track.uv[mid], min_area=4)
        ax = viz.plot_detection_frame(
            frame, blob=blob, true_uv=seq.true_pixels[mid],
            title=f"Camera {cam.name}, frame {mid}: detected vs true centre",
        )
        viz.save_figure(ax.figure, f"detection_{cam.name}", fig_dir)
        plt.close("all")

    obs = triangulate_tracks(tracks, cams)
    # A short track plus a bounce is a demanding input; keep the vision path
    # honest by reporting the triangulation residual spread.
    print(f"   triangulated {obs.n_visible} frames, estimated sigma {obs.sigma * 1000:.2f} mm")
    return obs


def _make_figures(truth, obs, fits, wicket, unc, fig_dir: Path, name: str) -> list[Path]:
    import matplotlib.pyplot as plt

    written: list[Path] = []

    fig = viz.plot_trajectory_3d(truth, fits[-1] if fits else None, wicket,
                                 title=f"Trajectory in 3-D: {name}")
    written.append(viz.save_figure(fig, f"demo_3d_{name}", fig_dir))
    plt.close("all")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    viz.plot_side_view(truth, fits=fits, obs=obs, ax=axes[0])
    viz.plot_plan_view(truth, fits=fits, obs=obs, wicket=wicket, ax=axes[1])
    written.append(viz.save_figure(fig, f"demo_views_{name}", fig_dir))
    plt.close("all")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    viz.plot_fit_comparison(obs, fits, truth=truth, ax=axes[0])
    viz.plot_residuals(obs, fits, ax=axes[1])
    written.append(viz.save_figure(fig, f"demo_fit_diagnostics_{name}", fig_dir))
    plt.close("all")

    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    viz.plot_wicket_plane(truth, fits=fits, uncertainty=unc, wicket=wicket, ax=ax,
                          title=f"Wicket plane: {name}")
    written.append(viz.save_figure(fig, f"demo_wicket_plane_{name}", fig_dir))
    plt.close("all")

    return written


def _write_report(path: Path, args, truth, obs, fits, verdict, unc, wicket) -> None:
    t_true, r_true, v_true = truth.state_at_wicket()
    lines = [
        f"# Demo report: `{args.preset}`",
        "",
        f"- Frame rate: {args.fps:.0f} Hz, measurement noise {args.noise_mm:.1f} mm, "
        f"dropout {args.dropout:.0%}, seed {args.seed}",
        f"- Observations used: {obs.n_visible}/{len(obs)}",
        f"- True arrival: y = {r_true[1]:+.4f} m, z = {r_true[2]:.4f} m, "
        f"speed {np.linalg.norm(v_true) * 3.6:.1f} km/h",
        "",
        "## Estimator comparison",
        "",
        "| model | fit RMSE (mm) | arrival y (m) | arrival z (m) | 3-D error (mm) |",
        "|---|---|---|---|---|",
    ]
    for f in fits:
        p = state_at_plane(f)
        err = float(np.linalg.norm(p - r_true)) * 1000
        lines.append(f"| {f.model} | {f.residual_rmse * 1000:.2f} | {p[1]:+.4f} | {p[2]:.4f} | {err:.1f} |")

    lines += [
        "",
        "## Decision",
        "",
        "```",
        verdict.describe(),
        "```",
        "",
        "Reasons: " + "; ".join(verdict.reasons),
        "",
        "## Uncertainty",
        "",
        f"- P(hit) = **{unc.hit_probability:.4f}** from {unc.n_samples} simulated trajectories",
        f"- 95% interval for arrival y: [{unc.interval(1)[0]:+.4f}, {unc.interval(1)[1]:+.4f}] m",
        f"- 95% interval for arrival z: [{unc.interval(2)[0]:.4f}, {unc.interval(2)[1]:.4f}] m",
        f"- Mean clearance {unc.clearances.mean() * 1000:+.2f} mm "
        f"(sd {unc.clearances.std() * 1000:.2f} mm)",
        "",
        "This is a statistical statement about *our* model, not the official DRS",
        "probability and not the umpire's-call rule.",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
