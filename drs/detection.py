"""
Ball detection in a single frame.

The detector is deliberately classical: a colour threshold in HSV followed by
connected-component analysis and a sub-pixel centroid.  It is not what a
commercial system uses -- those rely on trained detectors and high-speed
cameras -- but it is fully transparent, needs no training data, and its failure
modes are exactly the ones the project wants to discuss (motion blur, a
similarly coloured background, partial occlusion).

Two design choices are worth stating:

*Colour in HSV, not RGB.*  A red ball stays in the same hue band under
different exposure, while its RGB triple moves a long way.  Thresholding in HSV
makes the detector robust to global brightness changes.

*A redness weighting for the centroid.*  The binary centroid of an
anti-aliased blob is biased by whatever pixels happen to cross the threshold.
Weighting each pixel by ``R - max(G, B)`` -- which peaks at the ball centre and
decays at the blurred rim -- recovers a meaningful fraction of a pixel of
sub-pixel accuracy.

OpenCV conventions are used for the threshold values so that ranges are
portable: hue in ``[0, 180)``, saturation and value in ``[0, 255]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # scipy is already a core dependency
    from scipy import ndimage as _ndimage
except ImportError:  # pragma: no cover
    _ndimage = None


# A red cricket ball against grass and stumps.  Hue wraps around 180 for reds,
# hence the two ranges.
DEFAULT_HUE_RANGES: tuple[tuple[float, float], ...] = ((0.0, 13.0), (167.0, 180.0))
DEFAULT_SAT_MIN: float = 80.0
DEFAULT_VAL_MIN: float = 50.0


@dataclass
class Blob:
    """A candidate ball detection in one frame."""

    centroid: np.ndarray  # (x, y) pixels, sub-pixel
    area: int
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (x1/y1 exclusive)
    mean_value: float = 0.0

    @property
    def radius_px(self) -> float:
        return float(np.sqrt(self.area / np.pi))

    def circularity(self) -> float:
        """1.0 for a perfect disc; low for streaks and fragmented blobs."""
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        if max(w, h) == 0:
            return 0.0
        return float(self.area / (np.pi * (0.5 * max(w, h)) ** 2))


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB -> HSV using OpenCV's ranges.

    Parameters
    ----------
    rgb:
        ``(..., 3)`` array in ``[0, 255]``.

    Returns
    -------
    ndarray
        Same shape; ``H`` in ``[0, 180)``, ``S`` and ``V`` in ``[0, 255]``.
    """
    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.max(arr, axis=-1)
    minc = np.min(arr, axis=-1)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    safe = delta > 1e-12
    # Hue, in degrees, computed branch-wise to stay vectorised.
    rc = safe & (maxc == r)
    gc = safe & (maxc == g) & ~rc
    bc = safe & ~rc & ~gc
    h = np.where(rc, 60.0 * (((g - b) / np.where(safe, delta, 1.0)) % 6.0), h)
    h = np.where(gc, 60.0 * (((b - r) / np.where(safe, delta, 1.0)) + 2.0), h)
    h = np.where(bc, 60.0 * (((r - g) / np.where(safe, delta, 1.0)) + 4.0), h)

    s = np.where(maxc > 1e-12, delta / np.where(maxc > 1e-12, maxc, 1.0), 0.0)
    return np.stack([h / 2.0, s * 255.0, maxc * 255.0], axis=-1)


def redness(rgb: np.ndarray) -> np.ndarray:
    """``R - max(G, B)``: a cheap, illumination-robust "how red" score."""
    arr = np.asarray(rgb, dtype=np.float64)
    return arr[..., 0] - np.maximum(arr[..., 1], arr[..., 2])


def ball_mask(
    frame_rgb: np.ndarray,
    hue_ranges: tuple[tuple[float, float], ...] = DEFAULT_HUE_RANGES,
    sat_min: float = DEFAULT_SAT_MIN,
    val_min: float = DEFAULT_VAL_MIN,
    redness_min: float = 20.0,
) -> np.ndarray:
    """Boolean mask of ball-coloured pixels."""
    hsv = rgb_to_hsv(frame_rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_ok = np.zeros(h.shape, dtype=bool)
    for lo, hi in hue_ranges:
        hue_ok |= (h >= lo) & (h <= hi)
    mask = hue_ok & (s >= sat_min) & (v >= val_min)
    mask &= redness(frame_rgb) >= redness_min
    return mask


def find_blobs(mask: np.ndarray) -> list[Blob]:
    """Connected components of a boolean mask, as candidate blobs."""
    if _ndimage is None:  # pragma: no cover
        raise RuntimeError("scipy is required for connected-component labelling")
    labels, n = _ndimage.label(mask)
    if n == 0:
        return []
    slices = _ndimage.find_objects(labels)
    out: list[Blob] = []
    for i, sl in enumerate(slices, start=1):
        ys, xs = sl
        area = int(np.sum(labels[sl] == i))
        cy = 0.5 * (ys.start + ys.stop - 1)
        cx = 0.5 * (xs.start + xs.stop - 1)
        out.append(Blob(centroid=np.array([cx, cy]), area=area, bbox=(xs.start, ys.start, xs.stop, ys.stop)))
    return out


def refine_centroid(frame_rgb: np.ndarray, blob: Blob, weight: str = "redness") -> np.ndarray:
    """Sub-pixel centroid of a blob, weighted by its own colour evidence."""
    x0, y0, x1, y1 = blob.bbox
    # Pad by one pixel so the anti-aliased rim is included.
    frame_rgb = np.asarray(frame_rgb)
    pad = 2
    ya, yb = max(0, y0 - pad), min(frame_rgb.shape[0], y1 + pad)
    xa, xb = max(0, x0 - pad), min(frame_rgb.shape[1], x1 + pad)
    patch = frame_rgb[ya:yb, xa:xb]

    if weight == "redness":
        w = np.clip(redness(patch), 0.0, None)
    elif weight == "value":
        w = rgb_to_hsv(patch)[..., 2]
    else:
        raise ValueError(f"unknown weight {weight!r}")
    total = float(w.sum())
    if total <= 0:
        return blob.centroid.copy()

    yy, xx = np.mgrid[ya:yb, xa:xb]
    return np.array([float((w * xx).sum() / total), float((w * yy).sum() / total)])


def detect_blobs(
    frame_rgb: np.ndarray,
    min_area: int = 5,
    max_area: int | None = None,
    **mask_kwargs,
) -> list[Blob]:
    """All plausible ball blobs in a frame, largest first."""
    mask = ball_mask(frame_rgb, **mask_kwargs)
    blobs = [b for b in find_blobs(mask) if b.area >= min_area]
    if max_area is not None:
        blobs = [b for b in blobs if b.area <= max_area]
    for b in blobs:
        b.centroid = refine_centroid(frame_rgb, b)
    blobs.sort(key=lambda b: b.area, reverse=True)
    return blobs


def detect_ball(
    frame_rgb: np.ndarray,
    min_area: int = 5,
    max_area: int | None = None,
    expected: np.ndarray | None = None,
    gate_px: float | None = None,
    **mask_kwargs,
) -> Blob | None:
    """Pick the single most likely ball in a frame.

    When ``expected`` (the predicted pixel position, e.g. from the tracker's
    constant-velocity model) is supplied, candidates are ranked by distance to
    it and one further away than ``gate_px`` is rejected.  That gate is what
    stops the detector from locking onto, say, a red shirt in the crowd.
    """
    blobs = detect_blobs(frame_rgb, min_area=min_area, max_area=max_area, **mask_kwargs)
    if not blobs:
        return None
    if expected is None:
        return max(blobs, key=lambda b: b.area * max(b.circularity(), 0.05))

    expected = np.asarray(expected, dtype=float)
    scored = []
    for b in blobs:
        d = float(np.linalg.norm(b.centroid - expected))
        # Prefer near, large and round -- a product keeps all three in play
        # while remaining monotone in each.
        score = b.area * max(b.circularity(), 0.05) / (1.0 + d)
        scored.append((score, d, b))
    score, d, best = max(scored, key=lambda item: item[0])
    if gate_px is not None and d > gate_px:
        return None
    return best


def detect_sequence(frames, **kwargs) -> list[Blob | None]:
    """Run :func:`detect_ball` over an iterable of frames, with a naive gate."""
    out: list[Blob | None] = []
    last: np.ndarray | None = None
    velocity = np.zeros(2)
    for frame in frames:
        expected = None if last is None else last + velocity
        blob = detect_ball(frame, expected=expected, **kwargs)
        if blob is not None:
            if last is not None:
                velocity = blob.centroid - last
            last = blob.centroid
        out.append(blob)
    return out
