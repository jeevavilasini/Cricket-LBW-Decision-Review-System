"""
The experimental programme.

The proposal asks six questions, and each gets a function that returns a tidy
:class:`pandas.DataFrame` plus figures:

======================  ====================================================
``experiment_noise``    How does measurement noise translate into error at
                        the wicket plane?
``experiment_frames``   How many frames are needed for a reliable estimate?
``experiment_missing``  How much does a failed detector hurt?
``experiment_models``   Which estimator wins, and why?
``experiment_camera``   What do camera geometry and calibration error cost?
``experiment_uncertainty``  Is the hit probability stable, and are the
                        intervals honest?
======================  ====================================================

Every experiment is a Monte-Carlo over an ensemble of physically plausible
deliveries drawn by :func:`drs.simulate.sample_deliveries`; ground truth is
exact, so each metric is *measured* rather than asserted.  Results are written
to ``outputs/results`` as CSV, with figures in ``outputs/figures`` and a
markdown summary alongside.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from drs import viz  # importing viz selects the head-less Agg backend

import matplotlib.pyplot as plt  # noqa: E402  (after the backend is set)

from drs.config import BALL_RADIUS, DEFAULT_UMPIRE_CALL_MARGIN  # noqa: E402
from drs.decision import review_lbw
from drs.estimate import (
    ESTIMATORS,
    FitResult,
    fit,
    fit_ballistic,
    state_at_plane,
)
from drs.geometry import Camera, Wicket, default_stereo_rig, triangulate_sequence
from drs.kalman import fit_kalman
from drs.physics import simulate_delivery
from drs.simulate import (
    GroundTruth,
    Observations,
    build_ground_truth,
    make_observations,
    sample_deliveries,
)
from drs.uncertainty import bootstrap_wicket, monte_carlo_wicket

RESULTS_DIR = Path("outputs/results")
FIGURES_DIR = Path("outputs/figures")

# Models used in most comparisons.  ``ballistic_drag`` is excluded from the
# broad sweeps because it re-integrates an ODE for every trial.
FAST_MODELS = ("polynomial", "ballistic", "post_bounce", "piecewise")


# --------------------------------------------------------------------------
# Single-trial evaluation
# --------------------------------------------------------------------------


@dataclass
class TrialConfig:
    """Everything that defines one synthetic experiment."""

    fps: float = 50.0
    noise_std: float = 0.010
    dropout: float = 0.0
    cluster_dropout: bool = False
    n_frames: int | None = None
    keep: str = "first"
    onset: float = 0.0


def evaluate_trial(
    truth: GroundTruth,
    model: str,
    config: TrialConfig,
    rng: np.random.Generator | int | None = None,
    fit_kwargs: dict | None = None,
    wicket: Wicket | None = None,
    monte_carlo: int = 0,
) -> dict:
    """Generate observations, fit, and score one delivery.

    Returns
    -------
    dict
        One row of the result table.  Always includes ``ok`` so that failed
        fits can be counted rather than silently dropped.
    """
    wicket = wicket or Wicket()
    row: dict = {
        "model": model,
        "noise_std_m": config.noise_std,
        "dropout": config.dropout,
        "fps": config.fps,
        "ok": False,
    }
    try:
        obs = make_observations(
            truth,
            fps=config.fps,
            noise_std=config.noise_std,
            dropout=config.dropout,
            t_start=config.onset,
            rng=rng,
            cluster_dropout=config.cluster_dropout,
        )
        if config.n_frames is not None:
            obs = obs.last_n(config.n_frames) if config.keep == "last" else obs.first_n(config.n_frames)
        row["n_obs"] = obs.n_visible

        t0 = time.perf_counter()
        fit = _run_model(obs, model, fit_kwargs)
        row["fit_time_ms"] = (time.perf_counter() - t0) * 1e3

        t_true, r_true, _ = truth.state_at_wicket()
        r_pred = state_at_plane(fit)
        err = r_pred - r_true
        row.update(
            ok=True,
            plane_err_mm=float(np.linalg.norm(err)) * 1e3,
            lateral_err_mm=float(abs(err[1])) * 1e3,
            height_err_mm=float(abs(err[2])) * 1e3,
            depth_err_mm=float(abs(err[0])) * 1e3,
            arrival_y=float(r_pred[1]),
            arrival_z=float(r_pred[2]),
            fit_rmse_mm=float(fit.residual_rmse) * 1e3,
        )

        bounce_true = truth.bounce_time
        bounce_hat = fit.metadata.get("bounce_time")
        row["bounce_err_ms"] = (
            float(abs(bounce_hat - bounce_true)) * 1e3
            if bounce_hat is not None and bounce_true is not None
            else np.nan
        )

        hit_true = bool(wicket.hits(r_true.reshape(1, 3))[0])
        hit_pred = bool(wicket.hits(r_pred.reshape(1, 3))[0])
        row["hit_true"] = hit_true
        row["hit_pred"] = hit_pred
        row["hit_correct"] = bool(hit_true == hit_pred)
        row["clearance_true_mm"] = float(wicket.clearance(r_true.reshape(1, 3))[0]) * 1e3
        row["clearance_pred_mm"] = float(wicket.clearance(r_pred.reshape(1, 3))[0]) * 1e3

        verdict = review_lbw(fit, wicket=wicket, on_field="not_out", margin=DEFAULT_UMPIRE_CALL_MARGIN)
        row["decision"] = verdict.decision
        row["umpires_call"] = bool(verdict.umpires_call)

        if monte_carlo > 0:
            unc = monte_carlo_wicket(fit, wicket=wicket, n=monte_carlo, rng=rng)
            row["mc_hit_probability"] = float(unc.hit_probability)

    except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def _run_model(obs: Observations, model: str, fit_kwargs: dict | None) -> FitResult:
    """Dispatch to the right estimator, including the Kalman filter."""
    return fit(obs, model, **(fit_kwargs or {}))


def run_ensemble(
    truths: list[GroundTruth],
    models,
    config: TrialConfig,
    rng: np.random.Generator | int | None = None,
    n_repeats: int = 1,
    fit_kwargs: dict | None = None,
    monte_carlo: int = 0,
) -> pd.DataFrame:
    """Score every (truth, model, repeat) combination into one table."""
    rng = np.random.default_rng(rng)
    rows: list[dict] = []
    for rep in range(n_repeats):
        for truth in truths:
            for model in models:
                row = evaluate_trial(
                    truth, model, config, rng=rng, fit_kwargs=fit_kwargs, monte_carlo=monte_carlo
                )
                row["repeat"] = rep
                row["delivery"] = truth.label
                rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, by, metric: str = "plane_err_mm") -> pd.DataFrame:
    """Median/mean/percentile summary of ``metric`` grouped by ``by``."""
    def agg(g):
        v = g[metric].dropna()
        return pd.Series(
            {
                "n": len(v),
                "median": float(np.median(v)) if len(v) else np.nan,
                "mean": float(np.mean(v)) if len(v) else np.nan,
                "p90": float(np.quantile(v, 0.90)) if len(v) else np.nan,
                "max": float(np.max(v)) if len(v) else np.nan,
            }
        )

    return df.dropna(subset=[metric]).groupby(by, dropna=False).apply(agg, include_groups=False).reset_index()


# --------------------------------------------------------------------------
# Experiment 1: measurement noise
# --------------------------------------------------------------------------


def experiment_noise(
    noise_levels_mm=(0.0, 2.0, 5.0, 10.0, 20.0, 40.0),
    n_deliveries: int = 20,
    models=FAST_MODELS,
    n_repeats: int = 3,
    fps: float = 50.0,
    seed: int = 20240,
    out_dir: Path = RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep the measurement noise and record error at the wicket plane.

    The prediction of interest is at ``x = 0``, i.e. several metres *beyond*
    the last observation.  This sweep isolates how measurement error alone
    propagates through that extrapolation.
    """
    truths = sample_deliveries(n_deliveries, rng=seed)
    rows = []
    for noise_mm in noise_levels_mm:
        cfg = TrialConfig(fps=fps, noise_std=noise_mm / 1000.0)
        df = run_ensemble(truths, models, cfg, rng=seed, n_repeats=n_repeats)
        df["noise_mm"] = noise_mm
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)
    table = summarise(all_df, ["noise_mm", "model"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_dir / "noise_trials.csv", index=False)
    table.to_csv(out_dir / "noise_summary.csv", index=False)

    ax = viz.plot_metric_vs_param(
        table, "noise_mm", "median", hue="model",
        xlabel="measurement noise, per coordinate (mm)",
        ylabel="median error at wicket plane (mm)",
        title="Effect of measurement noise on the wicket-plane prediction",
        logx=False,
    )
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("log")
    viz.save_figure(ax.figure, "exp1_noise_vs_error", FIGURES_DIR)
    return all_df, table


# --------------------------------------------------------------------------
# Experiment 2: number of frames
# --------------------------------------------------------------------------


def experiment_frames(
    frame_counts=(4, 6, 8, 10, 15, 20, 25, 30, 40),
    n_deliveries: int = 20,
    models=("ballistic", "post_bounce", "piecewise"),
    n_repeats: int = 5,
    noise_mm: float = 5.0,
    fps: float = 50.0,
    seed: int = 7841,
    out_dir: Path = RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How many frames does the estimator actually need?

    Two variants make the answer honest.  Keeping the **first** ``k`` frames
    models a tracker that loses the ball late; keeping the **last** ``k`` models
    one that acquires it late.  The comparison shows that *which* frames are
    kept matters more than how many: the post-bounce arc is what determines the
    wicket-plane prediction, so a tracker that misses the bounce is helpless no
    matter how many pre-bounce frames it retains.
    """
    truths = sample_deliveries(n_deliveries, rng=seed)
    rows = []
    for keep in ("first", "last"):
        for k in frame_counts:
            cfg = TrialConfig(fps=fps, noise_std=noise_mm / 1000.0, n_frames=int(k), keep=keep)
            df = run_ensemble(truths, models, cfg, rng=seed, n_repeats=n_repeats)
            df["n_frames"] = int(k)
            df["keep"] = keep
            rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)
    table = summarise(all_df, ["n_frames", "keep", "model"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_dir / "frames_trials.csv", index=False)
    table.to_csv(out_dir / "frames_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True)
    for ax, keep, title in (
        (axes[0], "first", "Tracker loses the ball late"),
        (axes[1], "last", "Tracker acquires the ball late"),
    ):
        sub = table[table["keep"] == keep]
        viz.plot_metric_vs_param(
            sub, "n_frames", "median", hue="model", ax=ax,
            xlabel="number of observations retained",
            ylabel="median error at wicket plane (mm)",
            title=title, yscale="log",
        )
    viz.save_figure(fig, "exp2_frames_vs_error", FIGURES_DIR)
    return all_df, table


# --------------------------------------------------------------------------
# Experiment 3: missing detections
# --------------------------------------------------------------------------


def experiment_missing(
    dropout_levels=(0.0, 0.10, 0.25, 0.40),
    n_deliveries: int = 20,
    models=("ballistic", "post_bounce", "piecewise"),
    n_repeats: int = 5,
    noise_mm: float = 5.0,
    fps: float = 50.0,
    clustered: bool = True,
    seed: int = 5150,
    out_dir: Path = RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Effect of a failed detector, modelled as isolated and as clustered gaps."""
    truths = sample_deliveries(n_deliveries, rng=seed)
    rows = []
    for dropout in dropout_levels:
        cfg = TrialConfig(fps=fps, noise_std=noise_mm / 1000.0, dropout=float(dropout),
                          cluster_dropout=clustered)
        df = run_ensemble(truths, models, cfg, rng=seed, n_repeats=n_repeats)
        df["dropout"] = float(dropout)
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)
    table = summarise(all_df, ["dropout", "model"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_dir / "missing_trials.csv", index=False)
    table.to_csv(out_dir / "missing_summary.csv", index=False)

    ax = viz.plot_metric_vs_param(
        table, "dropout", "median", hue="model",
        xlabel="fraction of frames not detected",
        ylabel="median error at wicket plane (mm)",
        title="Robustness to missing detections",
        yscale="log",
    )
    viz.save_figure(ax.figure, "exp3_missing_vs_error", FIGURES_DIR)
    return all_df, table


# --------------------------------------------------------------------------
# Experiment 4: model comparison
# --------------------------------------------------------------------------


def heldout_rmse(
    truth: GroundTruth,
    obs: Observations,
    model: str,
    mode: str = "interleaved",
    frac: float = 0.6,
    fit_kwargs: dict | None = None,
) -> float:
    """RMSE on observations that were withheld from the fit.

    In-sample residual says how well a model can *describe* data it has seen;
    only held-out error says how well it can *predict*.  Two splits are offered
    because they answer different questions:

    ``interleaved`` (default)
        Hold out every other observation and predict it from its neighbours.
        This is the standard generalisation test, and the fair way to compare
        functional forms.
    ``temporal``
        Train on the first ``frac`` of the frames and predict the remainder.
        This is an *extrapolation* test, and every model fails it once the
        withheld frames lie beyond a bounce -- a result worth seeing rather
        than engineering away.
    """
    visible = np.flatnonzero(obs.visible)
    if visible.size < 4:
        return float("nan")

    hold = np.zeros_like(obs.visible)
    if mode == "interleaved":
        hold[visible[1::2]] = True
    elif mode == "temporal":
        k = max(3, int(frac * visible.size))
        hold[visible[k:]] = True
    else:
        raise ValueError(f"unknown split mode {mode!r}")

    train_mask = obs.visible & ~hold
    if train_mask.sum() < 3 or hold.sum() == 0:
        return float("nan")

    train = obs.select(train_mask)
    test = obs.select(hold)
    try:
        fit_result = _run_model(train, model, fit_kwargs)
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        return float("nan")

    pred = fit_result.predict(test.times)
    true = truth.trajectory.position(test.times)
    return float(np.sqrt(np.mean((pred - true) ** 2))) * 1e3


def experiment_models(
    n_deliveries: int = 20,
    models=("polynomial", "ballistic", "post_bounce", "piecewise", "ballistic_drag", "kalman"),
    noise_mm: float = 5.0,
    fps: float = 50.0,
    seed: int = 31337,
    out_dir: Path = RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Head-to-head comparison, including held-out error and a noiseless control."""
    truths = sample_deliveries(n_deliveries, rng=seed)
    rows = []
    for noise in (0.0, noise_mm):
        cfg = TrialConfig(fps=fps, noise_std=noise / 1000.0)
        for truth in truths:
            obs = make_observations(truth, fps=fps, noise_std=noise / 1000.0, rng=seed)
            for model in models:
                fit_kwargs = None
                row = evaluate_trial(truth, model, cfg, rng=seed, fit_kwargs=fit_kwargs)
                row["noise_mm"] = noise
                row["delivery"] = truth.label
                try:
                    row["heldout_rmse_mm"] = heldout_rmse(truth, obs, model)
                except (ValueError, np.linalg.LinAlgError, RuntimeError):
                    row["heldout_rmse_mm"] = np.nan
                rows.append(row)
    all_df = pd.DataFrame(rows)
    table = summarise(all_df, ["model", "noise_mm"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_dir / "models_trials.csv", index=False)
    table.to_csv(out_dir / "models_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    pivot = all_df.dropna(subset=["plane_err_mm"]).groupby(["model", "noise_mm"])["plane_err_mm"].median().unstack()
    for model in pivot.index:
        axes[0].plot(pivot.columns, pivot.loc[model], "o-", label=model, color=viz.model_color(model))
    axes[0].set_yscale("log")
    axes[0].set_xlabel("measurement noise (mm)")
    axes[0].set_ylabel("median error at wicket plane (mm)")
    axes[0].set_title("Prediction error vs noise")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)

    ho = all_df.dropna(subset=["heldout_rmse_mm"]).groupby("model")["heldout_rmse_mm"].median().sort_values()
    axes[1].barh(ho.index, ho.values, color=[viz.model_color(m) for m in ho.index])
    axes[1].set_xlabel("median held-out RMSE (mm)")
    axes[1].set_title("Error on unseen frames")
    axes[1].grid(alpha=0.25, axis="x")
    viz.save_figure(fig, "exp4_model_comparison", FIGURES_DIR)
    return all_df, table


# --------------------------------------------------------------------------
# Experiment 5: camera geometry and calibration
# --------------------------------------------------------------------------


def experiment_camera(
    baselines=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
    pixel_noises=(0.1, 0.25, 0.5, 1.0, 2.0),
    focal_errors=(-0.05, -0.02, 0.0, 0.02, 0.05),
    n_deliveries: int = 12,
    seed: int = 8080,
    out_dir: Path = RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stereo baseline, pixel noise and calibration error.

    Pixel noise is converted to 3-D error by *actually projecting and
    triangulating*, so the anisotropy of the reconstruction (depth much worse
    than lateral and vertical) appears in the numbers rather than being assumed.
    """
    truths = sample_deliveries(n_deliveries, rng=seed)
    rng = np.random.default_rng(seed)

    baseline_rows: list[dict] = []
    for baseline in baselines:
        cams = default_stereo_rig(baseline=baseline)
        for truth in truths:
            row = _camera_trial(truth, cams, pixel_noise_px=0.5, rng=rng, focal_scale=1.0)
            row.update(baseline_m=baseline, pixel_noise_px=0.5, focal_error=0.0)
            baseline_rows.append(row)
    baseline_df = pd.DataFrame(baseline_rows)

    pixel_rows: list[dict] = []
    cams = default_stereo_rig()
    for px in pixel_noises:
        for truth in truths:
            row = _camera_trial(truth, cams, pixel_noise_px=px, rng=rng, focal_scale=1.0)
            row.update(baseline_m=4.0, pixel_noise_px=px, focal_error=0.0)
            pixel_rows.append(row)
    pixel_df = pd.DataFrame(pixel_rows)

    focal_rows: list[dict] = []
    for err in focal_errors:
        cams = default_stereo_rig()
        for truth in truths:
            row = _camera_trial(truth, cams, pixel_noise_px=0.5, rng=rng, focal_scale=1.0 + err)
            row.update(baseline_m=4.0, pixel_noise_px=0.5, focal_error=err)
            focal_rows.append(row)
    focal_df = pd.DataFrame(focal_rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_df.to_csv(out_dir / "camera_baseline.csv", index=False)
    pixel_df.to_csv(out_dir / "camera_pixelnoise.csv", index=False)
    focal_df.to_csv(out_dir / "camera_focalerror.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    for ax, df, col, xlabel, title in (
        (axes[0], baseline_df, "baseline_m", "stereo baseline (m)", "Depth error vs baseline"),
        (axes[1], pixel_df, "pixel_noise_px", "centroid noise (px)", "Error vs detector accuracy"),
        (axes[2], focal_df, "focal_error", "relative focal-length error", "Effect of mis-calibration"),
    ):
        g = df.groupby(col)[["reconstruction_rms_mm", "plane_err_mm"]].median().reset_index()
        ax.plot(g[col], g["reconstruction_rms_mm"], "o-", label="3-D reconstruction", color="#3d5a80")
        ax.plot(g[col], g["plane_err_mm"], "s-", label="at wicket plane", color="#2a9d8f")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("median error (mm)")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    if baselines[0] < 1.0:
        axes[0].set_xscale("log")
    viz.save_figure(fig, "exp5_camera_geometry", FIGURES_DIR)
    return baseline_df, pixel_df, focal_df


def _camera_trial(
    truth: GroundTruth,
    cameras: list[Camera],
    pixel_noise_px: float,
    rng: np.random.Generator,
    focal_scale: float = 1.0,
) -> dict:
    """Project, perturb in pixels, triangulate, fit, and score."""
    if focal_scale != 1.0:
        cameras = [_rescale_focal(c, focal_scale) for c in cameras]

    t = np.linspace(0.0, truth.trajectory.duration, 40)
    X = truth.trajectory.position(t)
    pixels = np.stack([cam.project(X)[0] for cam in cameras], axis=0)  # (n_cam, n, 2)
    if pixel_noise_px > 0:
        pixels = pixels + rng.normal(0.0, pixel_noise_px, size=pixels.shape)

    finite = np.ones(pixels.shape[1], dtype=bool)
    for c in range(pixels.shape[0]):
        finite &= cameras[c].in_frame(pixels[c], margin=0.0)

    Xr = triangulate_sequence(cameras, pixels)
    recon_rms = float(np.sqrt(np.mean((Xr[finite] - X[finite]) ** 2))) * 1e3

    obs = Observations(t=t, positions=Xr, sigma=0.01, visible=finite)
    try:
        fit = fit_ballistic(obs, t_ref=float(t[finite][0]))
        t_true, r_true, _ = truth.state_at_wicket()
        plane_err = float(np.linalg.norm(state_at_plane(fit) - r_true)) * 1e3
    except (ValueError, np.linalg.LinAlgError):
        plane_err = np.nan

    return {
        "delivery": truth.label,
        "reconstruction_rms_mm": recon_rms,
        "plane_err_mm": plane_err,
        "n_in_frame": int(finite.sum()),
    }


def _rescale_focal(cam: Camera, scale: float) -> Camera:
    K = cam.K.copy()
    K[0, 0] *= scale
    K[1, 1] *= scale
    return Camera(K=K, R=cam.R, t=cam.t, width=cam.width, height=cam.height, name=cam.name + "-miscal")


# --------------------------------------------------------------------------
# Experiment 6: uncertainty quantification
# --------------------------------------------------------------------------


def experiment_uncertainty(
    delivery_name: str = "leg_spin_marginal",
    sample_sizes=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000),
    n_trials: int = 40,
    noise_mm: float = 5.0,
    fps: float = 50.0,
    model: str = "piecewise",
    seed: int = 24680,
    out_dir: Path = RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Is the hit probability stable, and do the intervals cover the truth?

    Two questions, two tables:

    * *convergence* -- how the estimate varies with the number of simulated
      trajectories (the answer to "how many do I need?");
    * *coverage* -- across many independent noisy realisations, how often the
      95% interval of the predicted arrival point actually contains the truth.
      A nominal 95% interval that covers far fewer than 95% of cases is
      over-confident, and reporting that is more valuable than a pretty plot.
    """
    from drs.physics import preset

    truth = build_ground_truth(preset(delivery_name), label=delivery_name)
    rng = np.random.default_rng(seed)

    obs = make_observations(truth, fps=fps, noise_std=noise_mm / 1000.0, rng=seed)
    fit = _run_model(obs, model, None)

    sizes, probs = [], []
    for n in sample_sizes:
        unc = monte_carlo_wicket(fit, n=int(n), rng=seed)
        sizes.append(int(n))
        probs.append(float(unc.hit_probability))
    convergence = pd.DataFrame({"n_samples": sizes, "hit_probability": probs})

    coverage_rows: list[dict] = []
    _t_true, r_true, _ = truth.state_at_wicket()
    for i in range(n_trials):
        obs_i = make_observations(truth, fps=fps, noise_std=noise_mm / 1000.0, rng=int(rng.integers(1 << 30)))
        try:
            fit_i = _run_model(obs_i, model, None)
            unc_i = monte_carlo_wicket(fit_i, n=4000, rng=int(rng.integers(1 << 30)))
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            continue
        y_lo, y_hi = unc_i.interval(1)
        z_lo, z_hi = unc_i.interval(2)
        coverage_rows.append(
            {
                "trial": i,
                "y_lo": y_lo,
                "y_hi": y_hi,
                "z_lo": z_lo,
                "z_hi": z_hi,
                "y_true": float(r_true[1]),
                "z_true": float(r_true[2]),
                "covers_y": bool(y_lo <= r_true[1] <= y_hi),
                "covers_z": bool(z_lo <= r_true[2] <= z_hi),
                "covers_both": bool(y_lo <= r_true[1] <= y_hi and z_lo <= r_true[2] <= z_hi),
                "hit_probability": float(unc_i.hit_probability),
            }
        )
    coverage = pd.DataFrame(coverage_rows)

    # Bootstrap intervals on the same delivery, for comparison.
    try:
        boot = bootstrap_wicket(obs, model=model, n=100, rng=seed)
        by_lo, by_hi = boot.interval(1)
        bz_lo, bz_hi = boot.interval(2)
        coverage.attrs["bootstrap"] = {
            "hit_probability": boot.hit_probability,
            "covers_truth": bool(by_lo <= r_true[1] <= by_hi and bz_lo <= r_true[2] <= bz_hi),
            "y_interval": (by_lo, by_hi),
            "z_interval": (bz_lo, bz_hi),
        }
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        pass

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    convergence.to_csv(out_dir / "uncertainty_convergence.csv", index=False)
    coverage.to_csv(out_dir / "uncertainty_coverage.csv", index=False)

    ax = viz.plot_uncertainty_convergence(convergence["n_samples"], convergence["hit_probability"])
    viz.save_figure(ax.figure, "exp6_mc_convergence", FIGURES_DIR)
    return convergence, coverage


# --------------------------------------------------------------------------
# Everything
# --------------------------------------------------------------------------


def run_all(out_dir: Path = RESULTS_DIR) -> dict:
    """Run the whole programme and write a markdown summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {}
    print("[1/7] noise sweep ...")
    results["noise_trials"], noise_summary = experiment_noise(out_dir=out_dir)
    print("[2/7] frame budget ...")
    results["frames_trials"], frames_summary = experiment_frames(out_dir=out_dir)
    print("[3/7] missing detections ...")
    results["missing_trials"], missing_summary = experiment_missing(out_dir=out_dir)
    print("[4/7] model comparison ...")
    results["models_trials"], models_summary = experiment_models(out_dir=out_dir)
    print("[5/7] camera geometry ...")
    results["camera_baseline"], results["camera_pixel"], results["camera_focal"] = experiment_camera(
        out_dir=out_dir
    )
    print("[6/7] uncertainty ...")
    results["uncertainty_convergence"], results["uncertainty_coverage"] = experiment_uncertainty(
        out_dir=out_dir
    )
    print("[7/7] summary ...")
    write_summary(results, out_dir / "SUMMARY.md")
    return results


def _render_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Markdown table, falling back to plain text when ``tabulate`` is absent.

    A named index (a grouped median, say) is promoted to a column so that the
    table is self-describing rather than relying on row order.
    """
    frame = df
    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()
    try:
        return frame.to_markdown(index=False, floatfmt=floatfmt)
    except ImportError:  # pragma: no cover - tabulate is optional
        return "```\n" + frame.to_string(index=False) + "\n```"


def write_summary(results: dict, path: Path) -> Path:
    """Write a readable markdown digest of every experiment."""
    lines: list[str] = ["# Experimental results\n", "Generated by `drs.experiments.run_all()`.\n"]

    def table_block(title: str, df: pd.DataFrame, cols: list[str] | None = None) -> None:
        lines.append(f"## {title}\n")
        if cols:
            df = df[cols]
        lines.append(_render_table(df))
        lines.append("")

    def summarise_if_possible(df: pd.DataFrame, by: list[str]) -> pd.DataFrame | None:
        """Summarise only when every grouping column is actually present."""
        if not all(column in df.columns for column in by):
            return None
        return summarise(df, by)

    if "noise_trials" in results:
        df = results["noise_trials"]
        s = summarise_if_possible(df, ["noise_mm", "model"])
        if s is not None:
            table_block("Measurement noise vs wicket-plane error", s)
        else:
            fallback = summarise_if_possible(df, ["model"])
            if fallback is not None:
                table_block("Wicket-plane error by model", fallback)

    if "frames_trials" in results:
        df = results["frames_trials"]
        by = ["n_frames", "keep", "model"] if "keep" in df.columns else ["n_frames", "model"]
        s = summarise_if_possible(df, by)
        if s is not None:
            table_block("Frame budget vs wicket-plane error", s)

    if "missing_trials" in results:
        s = summarise_if_possible(results["missing_trials"], ["dropout", "model"])
        if s is not None:
            table_block("Missing detections vs wicket-plane error", s)

    if "models_trials" in results:
        df = results["models_trials"]
        s = summarise_if_possible(df, ["model", "noise_mm"])
        if s is not None:
            table_block("Estimator comparison", s)
        grouped = summarise_if_possible(df, ["model"]) if "noise_mm" not in df.columns else None
        if grouped is not None:
            table_block("Estimator comparison (all noise levels pooled)", grouped)
        if "heldout_rmse_mm" in df.columns:
            ho = df.dropna(subset=["heldout_rmse_mm"]).groupby("model")["heldout_rmse_mm"].median()
            lines.append("### Held-out RMSE (median, mm)\n")
            lines.append(_render_table(ho.to_frame("heldout_rmse_mm")))
            lines.append("")

    if "uncertainty_coverage" in results:
        cov = results["uncertainty_coverage"]
        if len(cov):
            lines.append("## Uncertainty coverage\n")
            lines.append(
                f"- 95% intervals covering the true arrival y: **{cov['covers_y'].mean():.1%}**\n"
                f"- covering the true arrival z: **{cov['covers_z'].mean():.1%}**\n"
                f"- covering both: **{cov['covers_both'].mean():.1%}**\n"
                f"- trials: {len(cov)}\n"
            )
            boot = cov.attrs.get("bootstrap")
            if boot:
                lines.append(
                    f"- a single-delivery bootstrap interval "
                    f"({'covered' if boot['covers_truth'] else 'did NOT cover'} the truth), "
                    f"P(hit) = {boot['hit_probability']:.3f}\n"
                )
            lines.append("")

    path = Path(path)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
