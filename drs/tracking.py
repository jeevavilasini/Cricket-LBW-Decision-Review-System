"""
From pixels to a track, and from tracks to 3-D observations.

Two jobs:

**Temporal association.**  A detector that returns "the most ball-like blob"
will occasionally return the wrong blob.  Because a cricket ball cannot
accelerate without bound, implausible detections can be rejected by a simple
physical gate: compute the implied velocity and acceleration and discard frames
that require motion the ball is not capable of.  This is a cheap alternative to
a full multi-hypothesis tracker and it is stated as an assumption rather than
hidden.

**Reconstruction.**  With two calibrated views, each frame's pair of pixels is
triangulated by DLT into a 3-D point, producing the :class:`~drs.simulate.Observations`
that the estimators consume.  With only one view, no 3-D point is available
without an extra assumption -- see :func:`reconstruct_on_plane`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drs.detection import Blob, detect_ball
from drs.geometry import Camera, triangulate
from drs.simulate import Observations


@dataclass
class PixelTrack:
    """Detected ball centres in one camera, over time."""

    t: np.ndarray
    uv: np.ndarray
    visible: np.ndarray
    source: str = ""
    units: str = "pixels"

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.uv = np.asarray(self.uv, dtype=float)
        self.visible = np.asarray(self.visible, dtype=bool)
        if self.uv.shape != (self.t.size, 2):
            raise ValueError("uv must have shape (n, 2)")

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def times(self) -> np.ndarray:
        return self.t[self.visible]

    @property
    def points(self) -> np.ndarray:
        return self.uv[self.visible]

    @property
    def n_visible(self) -> int:
        return int(self.visible.sum())

    def subset(self, mask: np.ndarray) -> "PixelTrack":
        return PixelTrack(self.t[mask], self.uv[mask], self.visible[mask], self.source)


def track_pixels(
    frames,
    fps: float = 200.0,
    gate_px: float = 120.0,
    detector=None,
    **detect_kwargs,
) -> tuple[PixelTrack, list[Blob | None]]:
    """Detect and associate the ball across a sequence of frames.

    Parameters
    ----------
    frames:
        Iterable of ``(index, rgb_frame)`` pairs, or a plain iterable of frames.
    fps:
        Frame rate, used to convert frame indices into timestamps.
    gate_px:
        Maximum plausible pixel displacement between consecutive frames.  A ball
        at 150 km/h in a typical broadcast view moves far less than this; the
        gate exists to reject a detector locking onto something else.
    detector:
        Optional callable ``(frame, expected=..., gate_px=...) -> Blob | None``,
        so a trained detector could replace the default without touching this
        function.
    """
    detect = detector or (lambda frame, expected=None, gate_px=None: detect_ball(frame, expected=expected, gate_px=gate_px, **detect_kwargs))

    detections: list[Blob | None] = []
    uv = []
    visible = []
    last: np.ndarray | None = None
    velocity = np.zeros(2)

    for item in frames:
        if isinstance(item, tuple):
            _, frame = item
        else:
            frame = item
        expected = None if last is None else last + velocity
        blob = detect(frame, expected=expected, gate_px=gate_px)
        detections.append(blob)
        if blob is None:
            uv.append([np.nan, np.nan])
            visible.append(False)
        else:
            uv.append(blob.centroid)
            visible.append(True)
            if last is not None:
                velocity = 0.5 * velocity + 0.5 * (blob.centroid - last)
            last = blob.centroid

    n = len(detections)
    t = np.arange(n) / float(fps)
    return PixelTrack(t, np.array(uv), np.array(visible), source="detected"), detections


def reject_outliers(
    track: PixelTrack,
    max_speed_px_s: float = 6000.0,
    max_accel_px_s2: float = 300000.0,
    max_gap_s: float = 0.05,
) -> PixelTrack:
    """Discard detections that imply physically impossible motion.

    The gate is deliberately simple and its constants are exposed: a real
    tracker would use the Kalman innovation covariance instead.  Frames are also
    rejected when they are isolated by a long gap in which the ball could not
    plausibly have been followed.
    """
    t, uv, vis = track.t, track.uv, track.visible.copy()
    idx = np.flatnonzero(vis)
    if idx.size < 3:
        return track

    ti, ui = t[idx], uv[idx]
    dt = np.diff(ti)
    dv = np.diff(ui, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.linalg.norm(dv, axis=1) / np.where(dt > 0, dt, np.nan)

    # A point is suspicious when both its incoming and outgoing speeds are
    # excessive, which avoids punishing a single noisy sample on one side.
    v_in = np.concatenate([[0.0], speed])   # length n
    v_out = np.concatenate([speed, [0.0]])  # length n
    bad = (v_in > max_speed_px_s) & (v_out > max_speed_px_s)

    # A gap longer than the tracking window invalidates both of its endpoints.
    gap = dt > max_gap_s
    if gap.any():
        bad[:-1] |= gap
        bad[1:] |= gap

    if bad.any():
        vis[idx[bad]] = False
    return PixelTrack(t, uv, vis, source=track.source + "+gated", units=track.units)


def pixel_rmse(track: PixelTrack, true_pixels: np.ndarray) -> float:
    """Detection error in pixels, where ground truth is available."""
    m = track.visible
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((track.uv[m] - true_pixels[m]) ** 2)))


# --------------------------------------------------------------------------
# 3-D reconstruction
# --------------------------------------------------------------------------


def triangulate_tracks(
    tracks: list[PixelTrack],
    cameras: list[Camera],
    sigma: float | None = None,
) -> Observations:
    """Turn one pixel track per camera into 3-D observations.

    Parameters
    ----------
    tracks:
        One :class:`PixelTrack` per camera, all sharing the same time base.
    cameras:
        The calibrated rig, same order as ``tracks``.
    sigma:
        Assumed 3-D measurement standard deviation (m).  When omitted it is
        estimated from the residuals of the triangulated points about their own
        best-fit ballistic arc -- an empirical, honest value rather than a
        guess.
    """
    if len(tracks) != len(cameras):
        raise ValueError("one pixel track per camera is required")
    if len(cameras) < 2:
        raise ValueError("triangulation needs at least two cameras; see reconstruct_on_plane")

    n = len(tracks[0])
    for tr in tracks:
        if len(tr) != n:
            raise ValueError("all tracks must share the same number of frames")

    visible = np.ones(n, dtype=bool)
    for tr in tracks:
        visible &= tr.visible

    points = np.zeros((n, 3))
    for i in range(n):
        if not visible[i]:
            continue
        pixels = np.array([tr.uv[i] for tr in tracks])
        points[i] = triangulate(cameras, pixels)

    # Missing frames keep a placeholder: the midpoint of the viewing rays, so
    # that nothing downstream sees a wild value even before masking.
    from drs.geometry import intersect_rays

    for i in np.flatnonzero(~visible):
        origins, directions = [], []
        for tr, cam in zip(tracks, cameras):
            uv = tr.uv[i]
            if not np.isfinite(uv).all():
                uv = np.array([cam.width / 2.0, cam.height / 2.0])
            o, d = cam.ray(uv)
            origins.append(o[0])
            directions.append(d[0])
        points[i] = intersect_rays(np.array(origins), np.array(directions))[0]

    if sigma is None:
        sigma = _estimate_sigma(tracks[0].t[visible], points[visible])

    return Observations(
        t=tracks[0].t,
        positions=points,
        sigma=float(sigma),
        visible=visible,
        source="triangulated",
    )


def _estimate_sigma(t: np.ndarray, points: np.ndarray) -> float:
    """Estimate the 3-D measurement noise of a reconstructed track.

    The obvious approach -- roughness, via the second differences -- is wrong
    here, and instructively so.  Second differences cancel any error that
    changes *slowly* across frames, and the dominant reconstruction error is
    exactly that kind: the motion-blur streak biases the centroid in the same
    direction for many consecutive frames, so the resulting depth error is
    correlated rather than white.  A roughness estimator therefore reports a
    fraction of a millimetre for a track whose real error is several
    millimetres.

    The honest measure is the residual about a model that actually fits: the
    piecewise ballistic fit absorbs the bounce, so what remains is the
    reconstruction error itself, systematic part included.  The roughness
    estimate is kept only as a fallback for short tracks.
    """
    if t.size < 8:
        return _roughness_sigma(points)
    from drs.estimate import fit_piecewise_ballistic
    from drs.simulate import Observations

    obs = Observations(
        t=t, positions=points, sigma=0.01, visible=np.ones(t.size, dtype=bool), source="reconstruction"
    )
    try:
        fitted = fit_piecewise_ballistic(obs)
    except (ValueError, np.linalg.LinAlgError):
        return _roughness_sigma(points)

    residual = fitted.predict(t) - points
    return max(float(np.sqrt(np.mean(residual**2))), 1e-4)


def _roughness_sigma(points: np.ndarray) -> float:
    """Robust per-coordinate noise from the second differences (fallback only)."""
    if points.shape[0] < 5:
        return 0.02
    d2 = np.diff(points, n=2, axis=0)
    mad = np.median(np.abs(d2 - np.median(d2, axis=0)), axis=0) * 1.4826
    return max(float(np.sqrt(np.mean((mad / np.sqrt(6.0)) ** 2))), 1e-4)


def reconstruct_on_plane(
    track: PixelTrack,
    camera: Camera,
    plane_point=(0.0, 0.0, 0.0),
    plane_normal=(0.0, 0.0, 1.0),
) -> Observations:
    """Single-camera reconstruction using an assumed plane of motion.

    A single image point defines a *ray*, not a point, so 3-D recovery needs an
    extra assumption.  Here the assumption is that the ball always lies on a
    known plane -- useful for a ball rolling or skidding along the ground, and
    a clear demonstration of how much work that assumption is doing.  The
    returned ``sigma`` reflects the residual disagreement with a ballistic arc,
    which will be large because the assumption is wrong for a ball in flight.
    """
    plane_point = np.asarray(plane_point, dtype=float)
    plane_normal = np.asarray(plane_normal, dtype=float)
    normal = plane_normal / np.linalg.norm(plane_normal)

    n = len(track)
    points = np.zeros((n, 3))
    for i in range(n):
        if not track.visible[i]:
            continue
        origin, direction = camera.ray(track.uv[i])
        denom = float(direction[0] @ normal)
        if abs(denom) < 1e-12:  # pragma: no cover
            points[i] = np.full(3, np.nan)
            continue
        s = float((plane_point - origin[0]) @ normal) / denom
        points[i] = origin[0] + s * direction[0]

    return Observations(
        t=track.t,
        positions=points,
        sigma=_estimate_sigma(track.t[track.visible], points[track.visible]),
        visible=track.visible,
        source="single-camera+plane",
    )
