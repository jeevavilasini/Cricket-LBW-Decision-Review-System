"""
Figures.

Everything here is written to disk with the non-interactive Agg backend, so the
scripts run head-less on a laptop.  The plots are chosen to answer the
questions the report asks, not to decorate it:

* the *side* view (x-z) is the view a broadcast DRS graphic shows;
* the *plan* view (x-y) is the only way to see the pitching and impact lines;
* the *wicket-plane* view (y-z) is where the decision is actually made, and is
  where the Monte-Carlo cloud belongs.

Plotting the predicted trajectory without the last of these is what makes a
project look like a curve-fitting exercise rather than a decision model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # head-less: must be set before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from drs.config import BALL_RADIUS, STUMP_HEIGHT  # noqa: E402
from drs.geometry import Wicket  # noqa: E402

DEFAULT_FIG_DIR = Path("outputs/figures")

_COLORS = {
    "truth": "#111111",
    "polynomial": "#d1495b",
    "ballistic": "#edae49",
    "post_bounce": "#00798c",
    "piecewise": "#2a9d8f",
    "ballistic_drag": "#8e7dbe",
    "kalman": "#3d5a80",
    "detection": "#e63946",
}


def save_figure(fig, name: str, out_dir: str | Path = DEFAULT_FIG_DIR, dpi: int = 150) -> Path:
    """Save a figure and close it, returning the path written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def model_color(model: str) -> str:
    return _COLORS.get(model, "#666666")


# --------------------------------------------------------------------------
# Geometry helpers for drawing
# --------------------------------------------------------------------------


def _draw_ground(ax, x_range=(-20.5, 0.5), y_range=(-1.5, 1.5)) -> None:
    xs = np.linspace(*x_range, 2)
    ys = np.linspace(*y_range, 2)
    X, Y = np.meshgrid(xs, ys)
    ax.plot_surface(X, Y, np.zeros_like(X), color="#6f8f4a", alpha=0.25, linewidth=0)


def _draw_wicket_3d(ax, wicket: Wicket) -> None:
    for (cx, cy), radius, z0, z1 in wicket.as_cylinders():
        t = np.linspace(0, 2 * np.pi, 24)
        xs = cx + radius * np.cos(t)
        ys = cy + radius * np.sin(t)
        ax.plot(xs, ys, np.full_like(xs, z1), color="#e8e6d8", linewidth=2)
        ax.plot([cx, cx], [cy, cy], [z0, z1], color="#e8e6d8", linewidth=2)
    # Bails, drawn as the two horizontal bars across the top.
    for y0, y1 in zip(wicket.stump_offsets[:-1], wicket.stump_offsets[1:]):
        ax.plot([0, 0], [y0, y1], [STUMP_HEIGHT + 0.012, STUMP_HEIGHT + 0.012], color="#e8e6d8", linewidth=2)


def _draw_stumps_yz(ax, wicket: Wicket) -> None:
    """Stumps as they appear in the plane of the wicket: the decision view."""
    for off in wicket.stump_offsets:
        ax.add_patch(
            plt.Rectangle(
                (off - wicket.stump_radius, 0.0),
                2 * wicket.stump_radius,
                wicket.stump_height,
                color="#dcd9c8",
                zorder=2,
            )
        )
    ax.plot(
        [-wicket.half_width, wicket.half_width],
        [STUMP_HEIGHT + 0.012, STUMP_HEIGHT + 0.012],
        color="#b9b6a4",
        linewidth=4,
        solid_capstyle="butt",
        zorder=2,
    )
    ax.axhline(0.0, color="#6f8f4a", linewidth=3, zorder=1)
    ax.axvline(-wicket.half_width, color="#999999", linestyle=":", linewidth=1)
    ax.axvline(+wicket.half_width, color="#999999", linestyle=":", linewidth=1)


# --------------------------------------------------------------------------
# Trajectory plots
# --------------------------------------------------------------------------


def plot_trajectory_3d(truth, fit_result=None, wicket: Wicket | None = None, title: str | None = None):
    """Three-dimensional view of the true path and, optionally, a fitted one."""
    wicket = wicket or Wicket()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    t, r = truth.trajectory.sample(n=300)
    ax.plot(r[:, 0], r[:, 1], r[:, 2], color=_COLORS["truth"], linewidth=2, label="true path")
    _draw_ground(ax)
    _draw_wicket_3d(ax, wicket)

    if fit_result is not None:
        tf = np.linspace(0.0, ground_crossing_time(fit_result), 300)
        rf = fit_result.predict(tf)
        ax.plot(
            rf[:, 0],
            rf[:, 1],
            rf[:, 2],
            "--",
            color=model_color(fit_result.model),
            linewidth=2,
            label=f"{fit_result.model} fit",
        )

    pitch = truth.trajectory.pitch_point()
    if np.isfinite(pitch).all():
        ax.scatter(*pitch, color="#b5651d", s=45, marker="o", label="bounce", depthshade=False)

    ax.set_xlabel("x along pitch (m)")
    ax.set_ylabel("y across pitch (m)")
    ax.set_zlabel("z height (m)")
    ax.set_title(title or "Ball trajectory in three dimensions")
    ax.legend(loc="upper left", fontsize=8)
    return fig


def plot_side_view(truth, fits=None, obs=None, ax=None, title: str = "Side view (broadcast plane)"):
    """x-z view: the classic side-on picture used on television."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.2))

    t, r = truth.trajectory.sample(n=400)
    ax.plot(r[:, 0], r[:, 2], color=_COLORS["truth"], linewidth=2.2, label="truth", zorder=4)
    if obs is not None:
        ax.scatter(obs.points[:, 0], obs.points[:, 2], s=18, color="#444444", alpha=0.7,
                   label="observations", zorder=5)

    for fit_result in fits or []:
        tf = np.linspace(0.0, ground_crossing_time(fit_result), 300)
        rf = fit_result.predict(tf)
        ax.plot(rf[:, 0], rf[:, 2], "--", color=model_color(fit_result.model), linewidth=1.8, label=fit_result.model)

    ax.axhline(0.0, color="#6f8f4a", linewidth=2.5)
    ax.axvline(0.0, color="#e8e6d8", linewidth=6, alpha=0.9)
    ax.plot([0, 0], [0, STUMP_HEIGHT], color="#b9b6a4", linewidth=3)
    ax.annotate(
        "stumps", xy=(0.05, STUMP_HEIGHT / 2), xytext=(1.2, STUMP_HEIGHT * 0.75), fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8),
    )
    ax.set_xlabel("x along pitch (m)")
    ax.set_ylabel("height z (m)")
    ax.set_xlim(-20.5, 2.0)
    ax.set_ylim(-0.15, 2.6)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return ax


def plot_plan_view(truth, fits=None, obs=None, wicket: Wicket | None = None, ax=None,
                   title: str = "Plan view (pitching and impact lines)"):
    """x-y view: the only view in which "outside leg" is visible."""
    wicket = wicket or Wicket()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.2))

    t, r = truth.trajectory.sample(n=400)
    ax.plot(r[:, 0], r[:, 1], color=_COLORS["truth"], linewidth=2.2, label="truth", zorder=4)
    if obs is not None:
        ax.scatter(obs.points[:, 0], obs.points[:, 1], s=18, color="#444444", alpha=0.7,
                   label="observations", zorder=5)

    for fit_result in fits or []:
        tf = np.linspace(0.0, ground_crossing_time(fit_result), 300)
        rf = fit_result.predict(tf)
        ax.plot(rf[:, 0], rf[:, 1], "--", color=model_color(fit_result.model), linewidth=1.8, label=fit_result.model)

    ax.axhline(0.0, color="#3d5a80", linewidth=1, linestyle="--", label="line of stumps")
    for sign in (-1, 1):
        ax.axhline(sign * wicket.half_width, color="#999999", linewidth=1, linestyle=":")
    ax.axvline(0.0, color="#cccccc", linewidth=5)
    ax.axvline(-1.22, color="#dddddd", linewidth=1.5, linestyle="-.")
    ax.annotate("off side", xy=(-2.0, 0.42), fontsize=8, color="#3d5a80")
    ax.annotate("leg side", xy=(-2.0, -0.55), fontsize=8, color="#3d5a80")
    ax.set_xlabel("x along pitch (m)")
    ax.set_ylabel("y across pitch (m)")
    ax.set_xlim(-20.5, 2.0)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    return ax


def plot_wicket_plane(
    truth=None,
    fits=None,
    uncertainty=None,
    wicket: Wicket | None = None,
    ax=None,
    title: str = "Wicket plane (the decision view)",
    show_cloud: bool = True,
    inset: bool = True,
):
    """y-z view at the plane of the stumps, with the Monte-Carlo cloud.

    The cloud is usually only a few millimetres across while the full view spans
    most of a metre, so it is invisible at the scale at which the *stump* is
    visible.  A zoomed inset therefore carries the uncertainty, and the main
    axes carry the context.
    """
    wicket = wicket or Wicket()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.8, 5.6))

    if show_cloud and uncertainty is not None:
        pts = uncertainty.arrival_points
        hit = uncertainty.hit_mask
        ax.scatter(pts[~hit, 1], pts[~hit, 2], s=4, color="#9aa5b1", alpha=0.35,
                   label="miss (samples)", zorder=1)
        ax.scatter(pts[hit, 1], pts[hit, 2], s=4, color="#2a9d8f", alpha=0.35,
                   label="hit (samples)", zorder=1)

    if truth is not None:
        _, r_w, _ = truth.state_at_wicket()
        ax.scatter([r_w[1]], [r_w[2]], marker="*", s=200, color="#111111", label="truth", zorder=6)

    for fit_result in fits or []:
        from drs.estimate import state_at_plane
        try:
            p = state_at_plane(fit_result)
        except ValueError:
            continue
        ax.scatter([p[1]], [p[2]], marker="X", s=90, color=model_color(fit_result.model),
                   label=fit_result.model, zorder=7, edgecolor="white", linewidth=0.6)

    _draw_stumps_yz(ax, wicket)
    ax.set_xlabel("y across pitch (m)")
    ax.set_ylabel("height z (m)")
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.2, 0.95)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="upper right")

    if show_cloud and inset and uncertainty is not None:
        _add_zoom_inset(ax, uncertainty, truth, fits or [], wicket)
    return ax


def _add_zoom_inset(ax, uncertainty, truth, fits, wicket: Wicket) -> None:
    """A magnified view of the arrival cluster, where the decision is decided."""
    pts = uncertainty.arrival_points
    mean = pts.mean(axis=0)
    half_y = max(3.0 * float(pts[:, 1].std()), 0.008)
    half_z = max(3.0 * float(pts[:, 2].std()), 0.008)

    axins = ax.inset_axes([0.06, 0.06, 0.40, 0.40])
    if truth is not None:
        _, r_w, _ = truth.state_at_wicket()
        axins.scatter([r_w[1]], [r_w[2]], marker="*", s=90, color="#111111", zorder=6)
    for fit_result in fits:
        try:
            from drs.estimate import state_at_plane

            p = state_at_plane(fit_result)
        except ValueError:
            continue
        axins.scatter([p[1]], [p[2]], marker="X", s=40, color=model_color(fit_result.model),
                      zorder=7, edgecolor="white", linewidth=0.5)

    hit = uncertainty.hit_mask
    axins.scatter(pts[~hit, 1], pts[~hit, 2], s=2, color="#9aa5b1", alpha=0.5, zorder=1)
    axins.scatter(pts[hit, 1], pts[hit, 2], s=2, color="#2a9d8f", alpha=0.5, zorder=1)
    _draw_stumps_yz(axins, wicket)

    axins.set_xlim(mean[1] - half_y, mean[1] + half_y)
    axins.set_ylim(mean[2] - half_z, mean[2] + half_z)
    # A zoom this tight needs few, short tick labels or they collide.
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    axins.xaxis.set_major_locator(MaxNLocator(3))
    axins.yaxis.set_major_locator(MaxNLocator(3))
    axins.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axins.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axins.set_title("zoom on arrival", fontsize=7)
    axins.tick_params(labelsize=6, length=2, pad=1)
    axins.grid(alpha=0.2)
    ax.indicate_inset_zoom(axins, edgecolor="#888888", alpha=0.6)


def ground_crossing_time(fit_result, x_plane: float = 0.0, fallback: float = 0.7) -> float:
    """Horizon for plotting a fitted trajectory (time it reaches the stumps)."""
    from drs.estimate import time_at_plane

    try:
        return time_at_plane(fit_result, x_plane=x_plane)
    except ValueError:  # pragma: no cover
        return fallback


# --------------------------------------------------------------------------
# Fit diagnostics
# --------------------------------------------------------------------------


def plot_fit_comparison(obs, fits, truth=None, ax=None, title: str = "Height against time"):
    """Measured heights, the truth, and every fitted model, versus time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.2))

    ax.scatter(obs.times, obs.points[:, 2], s=22, color="#444444", label="observations", zorder=5)
    if truth is not None:
        t, r = truth.trajectory.sample(n=400)
        ax.plot(t, r[:, 2], color=_COLORS["truth"], linewidth=2, label="truth")

    for fit_result in fits:
        tf = np.linspace(0.0, ground_crossing_time(fit_result), 300)
        ax.plot(tf, fit_result.predict(tf)[:, 2], "--", color=model_color(fit_result.model),
                linewidth=1.7, label=fit_result.model)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("height z (m)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return ax


def plot_residuals(obs, fits, ax=None, title: str = "Fit residuals"):
    """Residual vectors, so that model error is visible rather than summarised."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.6))
    for fit_result in fits:
        res = fit_result.predict(obs.times) - obs.points
        ax.plot(obs.times, np.linalg.norm(res, axis=1) * 1000.0, "-o", markersize=3,
                color=model_color(fit_result.model), label=fit_result.model)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("residual norm (mm)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return ax


# --------------------------------------------------------------------------
# Metric plots used by the experiments
# --------------------------------------------------------------------------


def plot_metric_vs_param(df, x: str, y: str, hue: str | None = None, ax=None,
                         xlabel: str | None = None, ylabel: str | None = None,
                         title: str | None = None, yscale: str | None = None, logx: bool = False):
    """Generic line plot: ``y`` against ``x``, one line per value of ``hue``."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.4))
    if hue is None:
        ax.plot(df[x], df[y], "o-", color="#3d5a80")
    else:
        for name, grp in df.groupby(hue):
            grp = grp.sort_values(x)
            ax.plot(grp[x], grp[y], "o-", label=str(name), color=model_color(str(name)))
        ax.legend(fontsize=8, title=hue)
    if logx:
        ax.set_xscale("log")
    if yscale:
        ax.set_yscale(yscale)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title or f"{y} vs {x}")
    ax.grid(alpha=0.25)
    return ax


def plot_uncertainty_convergence(sizes, probs, ax=None, title: str = "Monte-Carlo convergence"):
    """Hit probability against the number of simulated trajectories."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.2))
    sizes_arr = np.asarray(sizes, dtype=float)
    probs_arr = np.asarray(probs, dtype=float)
    ax.semilogx(sizes_arr, probs_arr, "o-", color="#2a9d8f")
    final = float(probs_arr[-1])
    ax.axhline(final, color="#888888", linestyle="--", linewidth=1)
    ax.fill_between(sizes_arr, final - 0.02, final + 0.02, color="#2a9d8f", alpha=0.12,
                    label="+/- 2 percentage points")
    ax.set_xlabel("number of simulated trajectories N")
    ax.set_ylabel(r"estimated $P(\mathrm{hit})$")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return ax


def plot_detection_frame(frame, blob=None, true_uv=None, ax=None, title: str = "Detection"):
    """A single frame with the detected and true ball centres marked."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.05))
    ax.imshow(np.clip(frame, 0, 255).astype(np.uint8))
    if true_uv is not None:
        ax.scatter([true_uv[0]], [true_uv[1]], s=140, facecolors="none", edgecolors="#00ff88",
                   linewidths=1.4, label="true centre")
    if blob is not None:
        ax.scatter([blob.centroid[0]], [blob.centroid[1]], s=60, marker="x", color="#ff2d55",
                   linewidths=1.6, label="detected centre")
        x0, y0, x1, y1 = blob.bbox
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#ff2d55", linewidth=1))
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_axis_off()
    return ax


def plot_error_histogram(errors_mm, ax=None, title: str = "Wicket-plane prediction error",
                         bins: int = 30, label: str | None = None):
    """Histogram of a failure-mode metric."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(np.asarray(errors_mm), bins=bins, color="#3d5a80", alpha=0.8, label=label)
    med = float(np.median(errors_mm))
    ax.axvline(med, color="#d1495b", linestyle="--", linewidth=1.5, label=f"median {med:.1f} mm")
    ax.set_xlabel("error at the wicket plane (mm)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return ax
