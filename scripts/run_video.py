"""
Video pipeline runner.

Three ways to use it:

    # 1. render a synthetic stereo clip, then detect, triangulate and review it
    python scripts/run_video.py --synthetic good_length_middle

    # 2. run on a real clip (single camera) with a specified calibration
    python scripts/run_video.py --source clip.mp4 --focal-px 2400 \
        --cam 0 -25 3.2 --target 0 0 0.7

    # 3. run on a directory of frames
    python scripts/run_video.py --source frames/ --fps 200

Case 2 and 3 exercise the honest single-camera path: a pixel only defines a
ray, so the reconstruction needs an assumption, and the script states which one
it made and how much error it costs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from drs import viz  # noqa: E402
from drs.config import BALL_RADIUS  # noqa: E402
from drs.decision import review_lbw  # noqa: E402
from drs.detection import detect_ball  # noqa: E402
from drs.estimate import fit, state_at_plane  # noqa: E402
from drs.geometry import Camera, Wicket, default_stereo_rig  # noqa: E402
from drs.physics import preset  # noqa: E402
from drs.simulate import build_ground_truth  # noqa: E402
from drs.tracking import (
    pixel_rmse,
    reconstruct_on_plane,
    reject_outliers,
    track_pixels,
    triangulate_tracks,
)
from drs.uncertainty import monte_carlo_wicket  # noqa: E402
from drs.video import open_source  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", metavar="PRESET", help="render a synthetic clip from a preset delivery")
    src.add_argument("--source", help="a video file or a directory of image frames")

    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--fps", type=float, default=200.0, help="frame rate when reading a frame directory")
    p.add_argument("--render-fps", type=float, default=150.0, help="frame rate for a rendered clip")
    p.add_argument("--model", default="piecewise")
    p.add_argument("--min-area", type=int, default=4, help="minimum blob area in pixels")
    p.add_argument("--gate-px", type=float, default=500.0, help="max pixel displacement per frame")
    p.add_argument("--monte-carlo", type=int, default=10000)

    # Calibration for the single-camera path.
    p.add_argument("--camera", choices=("broadcast", "back-left", "back-right"), default="broadcast")
    p.add_argument("--cam", nargs=3, type=float, metavar=("X", "Y", "Z"), default=None)
    p.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"), default=(0.0, 0.0, 0.7))
    p.add_argument("--focal-px", type=float, default=None)
    p.add_argument("--size", nargs=2, type=int, metavar=("W", "H"), default=(960, 540))
    p.add_argument("--on-field", choices=("out", "not_out"), default="not_out")
    return p.parse_args(argv)


def build_camera(args) -> Camera:
    """Single-camera calibration from the command line."""
    default_positions = {
        "broadcast": (-25.0, 0.0, 3.2),
        "back-left": (-25.0, -2.0, 3.2),
        "back-right": (-25.0, 2.0, 3.2),
    }
    pos = tuple(args.cam) if args.cam else default_positions[args.camera]
    w, h = args.size
    if args.focal_px is None:
        return Camera.look_at(pos, tuple(args.target), fov_deg=32.0, width=w, height=h, name=args.camera)
    K = np.array([[args.focal_px, 0.0, w / 2.0], [0.0, args.focal_px, h / 2.0], [0.0, 0.0, 1.0]])
    cam = Camera.look_at(pos, tuple(args.target), fov_deg=32.0, width=w, height=h, name=args.camera)
    return Camera(K=K, R=cam.R, t=cam.t, width=w, height=h, name=args.camera)


def main(argv=None) -> int:
    args = parse_args(argv)
    fig_dir = Path(args.out_dir) / "figures"
    wicket = Wicket()

    if args.synthetic:
        return _run_synthetic(args, wicket, fig_dir)
    return _run_real(args, wicket, fig_dir)


# --------------------------------------------------------------------------
# Synthetic stereo
# --------------------------------------------------------------------------


def _run_synthetic(args, wicket: Wicket, fig_dir: Path) -> int:
    from drs.synthvideo import render_stereo_rig

    truth = build_ground_truth(preset(args.synthetic), label=args.synthetic)
    cameras = default_stereo_rig()
    root = Path(args.out_dir) / "synth_frames"

    print(f"Rendering synthetic stereo footage for '{args.synthetic}' -> {root}")
    seqs = render_stereo_rig(truth.trajectory, cameras, root, fps=args.render_fps, blur_subframes=3, seed=0)

    tracks = []
    for cam in cameras:
        seq = seqs[cam.name]
        stream = open_source(seq.directory, fps=seq.fps)
        track, detections = track_pixels(
            stream.frames(), fps=stream.fps, gate_px=args.gate_px, min_area=args.min_area
        )
        gated = reject_outliers(track)
        print(
            f"  camera {cam.name}: {seq.n_frames} frames, detected {track.n_visible}, "
            f"kept {gated.n_visible}, pixel RMSE {pixel_rmse(track, seq.true_pixels):.3f} px"
        )
        tracks.append(gated)

        _detection_figure(seq, stream, gated, cam, fig_dir)

    obs = triangulate_tracks(tracks, cameras)
    true_positions = seqs[cameras[0].name].true_positions
    err = obs.points[obs.visible] - true_positions[obs.visible]
    print(f"  triangulated {obs.n_visible} frames; 3-D RMS error "
          f"{np.round(np.sqrt((err ** 2).mean(0)) * 1000, 2)} mm (x, y, z)")

    return _finish(obs, truth, wicket, args, fig_dir, "synthetic")


# --------------------------------------------------------------------------
# Real footage (single camera)
# --------------------------------------------------------------------------


def _run_real(args, wicket: Wicket, fig_dir: Path) -> int:
    camera = build_camera(args)
    stream = open_source(args.source, fps=args.fps)
    print(f"Opened {stream.source}: {stream.n_frames or '?'} frames at {stream.fps:.1f} fps")
    if stream.width:
        print(f"  resolution {stream.width}x{stream.height}; camera {camera}")

    track, detections = track_pixels(
        stream.frames(), fps=stream.fps, gate_px=args.gate_px, min_area=args.min_area
    )
    gated = reject_outliers(track)
    print(f"  detected {track.n_visible}/{len(track)} frames, kept {gated.n_visible} after gating")

    _detection_figure_simple(gated, fig_dir)

    # A single view cannot recover depth on its own.  The assumption made here
    # is that the ball's centre lies on the ground plane, which is correct for a
    # rolling ball and increasingly wrong for one in flight -- so the residual
    # reported below is itself the measurement of how bad that assumption is.
    obs = reconstruct_on_plane(gated, camera, plane_point=(0.0, 0.0, BALL_RADIUS), plane_normal=(0.0, 0.0, 1.0))
    print(
        "  single-camera reconstruction: assumed the ball lies on the ground plane "
        f"(residual sigma {obs.sigma * 1000:.1f} mm -- large values mean the assumption fails)"
    )
    return _finish(obs, None, wicket, args, fig_dir, "real", camera=camera)


# --------------------------------------------------------------------------
# Shared tail
# --------------------------------------------------------------------------


def _finish(obs, truth, wicket: Wicket, args, fig_dir: Path, mode: str, camera=None) -> int:
    print(f"\nFitting '{args.model}' to {obs.n_visible} 3-D observations ...")
    try:
        f = fit(obs, args.model)
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
        print(f"  fit failed: {exc}")
        return 1
    p = state_at_plane(f)
    print(f"  residual RMSE {f.residual_rmse * 1000:.2f} mm; arrival y {p[1]:+.4f} m, z {p[2]:.4f} m")

    if truth is not None:
        _, r_true, _ = truth.state_at_wicket()
        print(f"  truth         arrival y {r_true[1]:+.4f} m, z {r_true[2]:.4f} m "
              f"(error {np.linalg.norm(p - r_true) * 1000:.1f} mm)")

    verdict = review_lbw(f, wicket=wicket, on_field=args.on_field)
    print("\nREVIEW")
    print(verdict.describe())
    for reason in verdict.reasons:
        print(f"   - {reason}")

    unc = monte_carlo_wicket(f, wicket=wicket, n=args.monte_carlo, rng=0)
    print(f"\n  P(hit) = {unc.hit_probability:.4f} from {unc.n_samples} trajectories")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    if truth is not None:
        viz.plot_side_view(truth, fits=[f], obs=obs, ax=axes[0])
        viz.plot_plan_view(truth, fits=[f], obs=obs, wicket=wicket, ax=axes[1])
    else:
        # A real clip has no ground truth to plot, so show the observations.
        axes[0].plot(obs.points[:, 0], obs.points[:, 2], "o", ms=3, color="#3d5a80", label="observations")
        axes[0].set_xlabel("x (m)")
        axes[0].set_ylabel("z (m)")
        axes[0].set_title("Reconstructed observations (x-z)")
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        axes[1].plot(obs.points[:, 0], obs.points[:, 1], "o", ms=3, color="#2a9d8f", label="observations")
        axes[1].axhline(0.0, color="#3d5a80", linestyle="--", linewidth=1, label="line of stumps")
        axes[1].set_xlabel("x (m)")
        axes[1].set_ylabel("y (m)")
        axes[1].set_title("Reconstructed observations (x-y)")
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=8)
    viz.save_figure(fig, f"video_views_{mode}", fig_dir)
    plt.close("all")

    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    viz.plot_wicket_plane(truth, fits=[f], uncertainty=unc, wicket=wicket, ax=ax,
                          title=f"Wicket plane ({mode})")
    viz.save_figure(fig, f"video_wicket_plane_{mode}", fig_dir)
    plt.close("all")

    print(f"\nFigures written to {fig_dir}")
    return 0


def _detection_figure(seq, stream, gated, cam: Camera, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from drs.detection import detect_ball

    idx = int(seq.n_frames * 0.6)
    frames = dict(stream.frames())
    frame = frames.get(idx)
    if frame is None:
        return
    blob = detect_ball(frame, expected=gated.uv[idx] if np.isfinite(gated.uv[idx]).all() else None,
                       min_area=4)
    ax = viz.plot_detection_frame(
        frame, blob=blob, true_uv=seq.true_pixels[idx],
        title=f"Camera {cam.name}, frame {idx}",
    )
    viz.save_figure(ax.figure, f"video_detection_{cam.name}", fig_dir)
    plt.close("all")


def _detection_figure_simple(track, fig_dir: Path) -> None:
    """Frame-by-frame pixel track when the frames themselves are not retained."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    m = track.visible
    axes[0].plot(track.t[m], track.uv[m, 0], "o-", ms=3, label="u")
    axes[0].plot(track.t[m], track.uv[m, 1], "s-", ms=3, label="v")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("pixel coordinate")
    axes[0].set_title("Detected pixel track")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(track.uv[m, 0], track.uv[m, 1], "o-", ms=3, color="#e63946")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("u (px)")
    axes[1].set_ylabel("v (px)")
    axes[1].set_title("Track in image space")
    axes[1].grid(alpha=0.25)
    viz.save_figure(fig, "video_pixel_track", fig_dir)
    plt.close("all")


if __name__ == "__main__":
    raise SystemExit(main())
