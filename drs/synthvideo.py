"""
Synthetic video rendering, so that the image-processing pipeline can be tested
end to end without a real broadcast clip.

The renderer ray-projects a known 3-D trajectory through a known camera to
obtain the *exact* pixel position of the ball in every frame, paints a ball
there, and writes the frames to disk.  Because the ground truth is retained,
the detector's error can be measured in pixels and, after triangulation, in
millimetres -- which is precisely the comparison the project needs.

Motion blur is rendered by averaging several sub-exposures across the frame
interval.  That matters: a ball at 140 km/h observed at 200 fps moves tens of
pixels per frame, so blur is not a cosmetic detail but the dominant image
artifact.  The centroid of a symmetric streak still estimates the mid-exposure
position, which is why a centroid detector remains usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drs.config import BALL_RADIUS
from drs.geometry import Camera
from drs.physics import Trajectory

# Default palette.  The ball is a saturated red; the pitch is green; the stumps
# are near-white.  Any detector thresholds should be chosen in HSV space so
# that brightness changes (cloud, floodlights) do not move the colour cluster.
BALL_RGB = (200, 40, 35)
PITCH_RGB = (96, 118, 62)
PITCH_STRIP_RGB = (138, 150, 96)
STUMP_RGB = (232, 230, 216)
SKY_RGB = (60, 70, 84)


@dataclass
class RenderedSequence:
    """Where a rendered sequence was written, and what it should contain."""

    directory: Path
    camera: Camera
    fps: float
    n_frames: int
    times: np.ndarray
    true_pixels: np.ndarray  # (n_frames, 2) exact projections
    true_positions: np.ndarray  # (n_frames, 3) exact world positions

    @property
    def in_frame(self) -> np.ndarray:
        uv = self.true_pixels
        return (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < self.camera.width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < self.camera.height)
        )


def _background(camera: Camera, trail_len: float) -> np.ndarray:
    """Paint a crude but unambiguous scene: sky, pitch wedge, crease, stumps."""
    h, w = camera.height, camera.width
    img = np.zeros((h, w, 3), dtype=np.float64)
    img[:, :] = SKY_RGB

    # The pitch is the image of a long rectangle on the ground plane; sampling
    # the projection of a grid of points is the simplest way to fill it.
    xs = np.linspace(-21.0, 0.5, 40)
    ys = np.linspace(0.0, 0.0, 1)
    left = np.array([[x, -1.45, 0.0] for x in xs])
    right = np.array([[x, +1.45, 0.0] for x in xs])
    uv_l, _ = camera.project(left)
    uv_r, _ = camera.project(right)
    horizon_y = int(np.clip(min(uv_l[:, 1].min(), uv_r[:, 1].min()), 0, h - 1))
    img[horizon_y:, :] = PITCH_RGB

    # The worn strip between the creases, drawn as a polygon in image space.
    top_l = camera.project(np.array([[0.0, -1.05, 0.0]]))[0][0]
    top_r = camera.project(np.array([[0.0, +1.05, 0.0]]))[0][0]
    bot_l = camera.project(np.array([[-20.12, -1.05, 0.0]]))[0][0]
    bot_r = camera.project(np.array([[-20.12, +1.05, 0.0]]))[0][0]
    _fill_polygon(img, np.array([bot_l, bot_r, top_r, top_l]), PITCH_STRIP_RGB)

    # Popping crease: a white line 1.22 m in front of the stumps.
    for y in np.linspace(-1.32, 1.32, 24):
        p = camera.project(np.array([[-1.22, y, 0.0]]))[0][0]
        _disc(img, p, 1.4, (240, 240, 235))

    # Stumps: vertical bars at the projected base positions.
    for off in (-0.09525, 0.0, 0.09525):
        base = camera.project(np.array([[0.0, off, 0.0]]))[0][0]
        top = camera.project(np.array([[0.0, off, 0.7112]]))[0][0]
        _bar(img, base, top, 2.0, STUMP_RGB)

    return img


def _disc(img: np.ndarray, center, radius: float, color) -> None:
    """Draw a filled, anti-aliased disc in place."""
    if not np.isfinite(center).all() or radius <= 0:
        return
    h, w = img.shape[:2]
    r = max(radius, 0.6)
    x0, x1 = int(max(0, np.floor(center[0] - r - 1))), int(min(w, np.ceil(center[0] + r + 2)))
    y0, y1 = int(max(0, np.floor(center[1] - r - 1))), int(min(h, np.ceil(center[1] + r + 2)))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - center[0], yy - center[1])
    alpha = np.clip(r + 0.5 - d, 0.0, 1.0)[..., None]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - alpha) + np.asarray(color, dtype=float) * alpha


def _bar(img: np.ndarray, base, top, half_width: float, color) -> None:
    n = max(2, int(np.hypot(*(np.asarray(top) - np.asarray(base))) / 2) + 2)
    for s in np.linspace(0.0, 1.0, n):
        p = np.asarray(base) + s * (np.asarray(top) - np.asarray(base))
        _disc(img, p, half_width, color)


def _fill_polygon(img: np.ndarray, vertices: np.ndarray, color) -> None:
    """Scanline fill of a convex polygon given in image coordinates."""
    h, w = img.shape[:2]
    ys = vertices[:, 1]
    y0, y1 = int(max(0, np.floor(ys.min()))), int(min(h, np.ceil(ys.max()) + 1))
    if y1 <= y0:
        return
    xs_all = []
    for i in range(len(vertices)):
        p, q = vertices[i], vertices[(i + 1) % len(vertices)]
        if p[1] == q[1]:
            continue
        t = (np.arange(y0, y1) - p[1]) / (q[1] - p[1])
        m = (t >= 0) & (t <= 1)
        xs_all.append((p[0] + t * (q[0] - p[0]), m))
    if not xs_all:
        return
    acc = np.full(y1 - y0, np.nan)
    acc2 = np.full(y1 - y0, np.nan)
    for xv, m in xs_all:
        lo = np.where(m & (np.isnan(acc) | (xv < acc)), xv, acc)
        hi = np.where(m & (np.isnan(acc2) | (xv > acc2)), xv, acc2)
        acc, acc2 = lo, hi
    for i in range(y1 - y0):
        if np.isnan(acc[i]) or np.isnan(acc2[i]):
            continue
        a = int(max(0, np.floor(acc[i])))
        b = int(min(w, np.ceil(acc2[i]) + 1))
        if b > a:
            img[y0 + i, a:b] = color


def render_delivery(
    trajectory: Trajectory,
    camera: Camera,
    out_dir: str | Path,
    fps: float = 200.0,
    blur_subframes: int = 4,
    noise_sigma: float = 2.0,
    ball_radius: float = BALL_RADIUS,
    seed: int | None = 0,
    prefix: str = "frame",
) -> RenderedSequence:
    """Render one delivery as a sequence of PNG frames.

    Parameters
    ----------
    trajectory:
        The true continuous trajectory.
    camera:
        The calibrated view to render.
    out_dir:
        Directory for the PNG frames (created if necessary).
    fps:
        Frame rate.  The default 200 Hz keeps the blur manageable.
    blur_subframes:
        Number of sub-exposures averaged per frame; 1 disables motion blur.
    noise_sigma:
        Standard deviation of additive Gaussian sensor noise (0-255 scale).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Remove any frames from a previous render at a different frame rate, which
    # would otherwise be picked up as extra frames when the directory is read
    # back as an image sequence.
    for stale in out.glob(f"{prefix}_*.png"):
        stale.unlink()
    rng = np.random.default_rng(seed)

    duration = float(trajectory.duration)
    n_frames = max(2, int(np.floor(duration * fps)) + 1)
    times = np.arange(n_frames) / fps

    background = _background(camera, duration)
    true_pixels = np.zeros((n_frames, 2))
    true_positions = np.zeros((n_frames, 3))

    for i, t in enumerate(times):
        r_true = np.asarray(trajectory.position(float(t)))
        true_positions[i] = r_true
        uv, depth = camera.project(r_true.reshape(1, 3))
        true_pixels[i] = uv[0]

        frame = background.copy()
        if blur_subframes > 1:
            offsets = np.linspace(-0.5, 0.5, blur_subframes) * (1.0 / fps)
        else:
            offsets = np.array([0.0])
        for off in offsets:
            r_sub = np.asarray(trajectory.position(float(np.clip(t + off, 0.0, duration))))
            uv_sub, depth_sub = camera.project(r_sub.reshape(1, 3))
            r_px = max(1.0, camera.focal_length * ball_radius / max(float(depth_sub[0]), 1e-3))
            _disc(frame, uv_sub[0], r_px, BALL_RGB)

        if noise_sigma > 0:
            # float32 halves the cost of the dominant term in rendering.
            noise = rng.standard_normal(frame.shape, dtype=np.float32) * np.float32(noise_sigma)
            frame = frame + noise
        _write_png(out / f"{prefix}_{i:04d}.png", frame)

    return RenderedSequence(
        directory=out,
        camera=camera,
        fps=float(fps),
        n_frames=n_frames,
        times=times,
        true_pixels=true_pixels,
        true_positions=true_positions,
    )


def _write_png(path: Path, frame: np.ndarray) -> None:
    """Write a float [0, 255] RGB array as an 8-bit PNG."""
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - Pillow is a hard dep here
        raise RuntimeError("Pillow is required to write rendered frames") from exc
    arr = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def render_stereo_rig(
    trajectory: Trajectory,
    cameras: list[Camera],
    root: str | Path,
    **kwargs,
) -> dict[str, RenderedSequence]:
    """Render the same delivery from every camera of a rig into sub-directories."""
    root = Path(root)
    out: dict[str, RenderedSequence] = {}
    for cam in cameras:
        out[cam.name] = render_delivery(trajectory, cam, root / f"cam_{cam.name}", **kwargs)
    return out
