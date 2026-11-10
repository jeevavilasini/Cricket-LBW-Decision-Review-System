"""
Computational geometry for DRS.

Three things live here, and keeping them apart matters:

1.  :class:`Camera` -- the pinhole projection model

        lambda [u, v, 1]^T = K [R | t] [x, y, z, 1]^T

    plus its exact inverse *as a ray*, because a single pixel does not
    determine a 3-D point.  :func:`triangulate` closes the loop when two or
    more views are available.

2.  :class:`Wicket` -- the stumps and bails as solid geometry, with an exact
    point-to-solid Euclidean distance so that the ball (a sphere) can be
    tested for intersection properly rather than by comparing coordinates.

3.  The ICC zone predicates -- *pitching*, *impact* and *wicket* -- expressed
    on the pitch coordinate system, with an explicit margin parameter for the
    umpire's call band.

The distinction between (1) and (2) is the heart of the project's honesty:
image space is 2-D, the decision is 3-D, and only a camera model plus either a
second view or a physical assumption bridges them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drs.config import (
    BAIL_CENTER_Z,
    BAIL_HEIGHT,
    BAIL_RADIUS,
    BALL_RADIUS,
    STUMP_HEIGHT,
    STUMP_OFFSETS,
    STUMP_RADIUS,
    WICKET_HALF_WIDTH,
)

# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------


@dataclass
class Camera:
    """A calibrated pinhole camera.

    Attributes
    ----------
    K:
        ``(3, 3)`` intrinsic matrix.
    R:
        ``(3, 3)`` rotation taking world coordinates into camera coordinates.
    t:
        ``(3,)`` translation taking world coordinates into camera coordinates.
    width, height:
        Image size in pixels, used only to report whether a projection falls
        inside the frame.
    name:
        Label for plots and tables.
    """

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int = 1920
    height: int = 1080
    name: str = "camera"

    # -- construction -----------------------------------------------------

    @classmethod
    def look_at(
        cls,
        position,
        target,
        up=(0.0, 0.0, 1.0),
        fov_deg: float = 35.0,
        width: int = 1920,
        height: int = 1080,
        name: str = "camera",
        focal_scale: float = 1.0,
    ) -> "Camera":
        """Build a camera from a viewpoint, a look-at target and a horizontal FOV.

        The camera frame follows the OpenCV convention: +x to the image right,
        +y to the image bottom, +z along the viewing direction.  ``focal_scale``
        lets a caller inject a *calibration error* by scaling the focal length
        away from its nominal value.
        """
        position = np.asarray(position, dtype=float)
        target = np.asarray(target, dtype=float)
        up = np.asarray(up, dtype=float)

        forward = target - position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        norm = np.linalg.norm(right)
        if norm < 1e-12:
            raise ValueError("view direction is parallel to the up vector")
        right = right / norm
        down = np.cross(forward, right)

        R = np.vstack([right, down, forward])
        t = -R @ position

        f = focal_scale * 0.5 * width / np.tan(np.deg2rad(fov_deg) / 2.0)
        K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
        return cls(K=K, R=R, t=t, width=width, height=height, name=name)

    # -- properties -------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.R.T @ self.t

    @property
    def projection_matrix(self) -> np.ndarray:
        """The ``(3, 4)`` matrix ``K [R | t]``."""
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    @property
    def focal_length(self) -> float:
        return float(0.5 * (self.K[0, 0] + self.K[1, 1]))

    # -- projection -------------------------------------------------------

    def to_camera_frame(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return X @ self.R.T + self.t

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to pixels.

        Returns
        -------
        uv:
            ``(n, 2)`` pixel coordinates.
        depth:
            ``(n,)`` coordinate along the viewing axis (metres); negative means
            the point is behind the camera.
        """
        Xc = self.to_camera_frame(X)
        depth = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.K[0, 0] * Xc[:, 0] / depth + self.K[0, 2]
            v = self.K[1, 1] * Xc[:, 1] / depth + self.K[1, 2]
        return np.column_stack([u, v]), depth

    def ray(self, uv) -> tuple[np.ndarray, np.ndarray]:
        """Back-project a pixel to a world-space ray.

        Returns
        -------
        origin:
            The camera centre.
        direction:
            A unit vector in world coordinates.  *Any* point along
            ``origin + s * direction`` (s > 0) projects to the same pixel:
            this is the depth ambiguity at the centre of the project.
        """
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        x = (uv[:, 0] - self.K[0, 2]) / self.K[0, 0]
        y = (uv[:, 1] - self.K[1, 2]) / self.K[1, 1]
        dirs_cam = np.column_stack([x, y, np.ones(uv.shape[0])])
        dirs_world = dirs_cam @ self.R  # R^T applied on the right
        dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True)
        origins = np.tile(self.position, (uv.shape[0], 1))
        return origins, dirs_world

    def in_frame(self, uv: np.ndarray, margin: float = 0.0) -> np.ndarray:
        uv = np.atleast_2d(uv)
        return (
            (uv[:, 0] >= -margin)
            & (uv[:, 0] <= self.width + margin)
            & (uv[:, 1] >= -margin)
            & (uv[:, 1] <= self.height + margin)
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        p = self.position
        return f"<Camera {self.name} at ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) f={self.focal_length:.0f}px>"


def triangulate(cameras: list[Camera], pixels: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation of one point seen by two or more cameras.

    For each camera, the pixel ``(u, v)`` gives two independent linear
    constraints on the homogeneous 3-D point ``X``:

        u (p3 . X) - (p1 . X) = 0
        v (p3 . X) - (p2 . X) = 0

    where ``p_i`` are the rows of the projection matrix.  Stacking the
    constraints from every view gives a homogeneous system solved by SVD; the
    right singular vector for the smallest singular value is the solution.
    With exactly two views this is the classical midpoint-free stereo
    reconstruction, and its accuracy degrades as the baseline shrinks -- which
    is precisely the behaviour the single-camera experiment demonstrates.
    """
    pixels = np.atleast_2d(np.asarray(pixels, dtype=float))
    if len(cameras) != pixels.shape[0]:
        raise ValueError("one pixel per camera is required")

    A = []
    for cam, (u, v) in zip(cameras, pixels):
        P = cam.projection_matrix
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A)

    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        raise np.linalg.LinAlgError("degenerate triangulation (point at infinity)")
    return X[:3] / X[3]


def triangulate_sequence(cameras: list[Camera], pixel_sequences: np.ndarray) -> np.ndarray:
    """Triangulate a whole track.

    Parameters
    ----------
    cameras:
        The camera rig.
    pixel_sequences:
        ``(n_cameras, n_frames, 2)`` pixel observations.

    Returns
    -------
    ndarray, shape (n_frames, 3)
    """
    pixel_sequences = np.asarray(pixel_sequences, dtype=float)
    _, n_frames, _ = pixel_sequences.shape
    return np.array([triangulate(cameras, pixel_sequences[:, i, :]) for i in range(n_frames)])


# --------------------------------------------------------------------------
# Camera rigs
# --------------------------------------------------------------------------


def default_stereo_rig(
    baseline: float = 4.0,
    distance: float = 25.0,
    height: float = 3.2,
    target=(0.0, 0.0, 0.7),
    fov_deg: float = 32.0,
    width: int = 960,
    height_px: int = 540,
) -> list[Camera]:
    """A realistic toed-in stereo pair behind the bowler's end.

    A second camera at the *other* end of the pitch would have a baseline
    almost parallel to the flight direction, so the stereo problem becomes
    ill-conditioned and depth is barely observable.  Separating the two
    cameras laterally (here by ``baseline`` metres at ``distance`` metres)
    puts the baseline perpendicular to the dominant motion, which is what makes
    the depth coordinate recoverable.  Both cameras see the whole flight.

    The default 960x540 resolution keeps CPU rendering fast; a real system
    would use full HD or better, which only improves the centroid accuracy.
    """
    return [
        Camera.look_at(
            (-distance, -baseline / 2.0, height),
            target,
            fov_deg=fov_deg,
            width=width,
            height=height_px,
            name="left",
        ),
        Camera.look_at(
            (-distance, +baseline / 2.0, height),
            target,
            fov_deg=fov_deg,
            width=width,
            height=height_px,
            name="right",
        ),
    ]


def broadcast_single_camera(
    position=(-25.0, 0.0, 3.2),
    target=(0.0, 0.0, 0.7),
    fov_deg: float = 45.0,
    width: int = 1280,
    height: int = 720,
) -> Camera:
    """The realistic-but-hard case: one camera behind the bowler's end."""
    return Camera.look_at(position, target, fov_deg=fov_deg, width=width, height=height, name="broadcast")


# --------------------------------------------------------------------------
# Wicket geometry
# --------------------------------------------------------------------------


def _distance_to_cylinder(points: np.ndarray, center_xy, radius: float, z0: float, z1: float) -> np.ndarray:
    """Exact Euclidean distance from points to a finite solid cylinder."""
    d_radial = np.hypot(points[:, 0] - center_xy[0], points[:, 1] - center_xy[1]) - radius
    d_axial = np.maximum(np.maximum(z0 - points[:, 2], points[:, 2] - z1), 0.0)
    inside_radial = d_radial <= 0.0
    out = np.where(inside_radial, d_axial, np.where(d_axial <= 0.0, d_radial, np.hypot(d_radial, d_axial)))
    return out


def _distance_to_box(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance from points to an axis-aligned box."""
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    d = np.abs(points - center) - half
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    return outside


@dataclass
class Wicket:
    """The striker's wicket as three cylinders plus two bails.

    The stump *height* (1.4 m in the laws, 0.71 m here) and the stump *radius*
    are the parameters that matter for a hit test; the bails extend the solid
    a further :data:`~drs.config.BAIL_HEIGHT` upwards across the full width.
    """

    x: float = 0.0
    stump_offsets: tuple[float, ...] = STUMP_OFFSETS
    stump_radius: float = STUMP_RADIUS
    stump_height: float = STUMP_HEIGHT
    bail_radius: float = BAIL_RADIUS
    bail_height: float = BAIL_HEIGHT
    half_width: float = WICKET_HALF_WIDTH

    # -- distance ---------------------------------------------------------

    def distance(self, points) -> np.ndarray:
        """Distance from the ball *centre* to the nearest point of the wicket.

        ``distance <= ball_radius`` is the exact condition for a spherical ball
        to touch the wicket, which is why the ball radius is passed in rather
        than baked in here.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        d = np.full(pts.shape[0], np.inf)

        for off in self.stump_offsets:
            d = np.minimum(
                d,
                _distance_to_cylinder(
                    pts,
                    (self.x, off),
                    self.stump_radius,
                    0.0,
                    self.stump_height,
                ),
            )

        # Bails: two horizontal cylinders lying across the top of the stumps.
        for y_a, y_b in zip(self.stump_offsets[:-1], self.stump_offsets[1:]):
            lo = np.array([self.x - self.bail_radius, min(y_a, y_b), BAIL_CENTER_Z - self.bail_radius])
            hi = np.array([self.x + self.bail_radius, max(y_a, y_b), BAIL_CENTER_Z + self.bail_radius])
            d = np.minimum(d, _distance_to_box(pts, lo, hi))

        return d

    def hits(self, points, ball_radius: float = BALL_RADIUS) -> np.ndarray:
        """Boolean hit test for a sphere of the given radius."""
        return self.distance(points) <= ball_radius

    def clearance(self, points, ball_radius: float = BALL_RADIUS) -> np.ndarray:
        """Signed clearance: positive means the ball passes clear of the wicket."""
        return self.distance(points) - ball_radius

    # -- zones ------------------------------------------------------------

    def in_line(self, y) -> np.ndarray:
        """True where a point's lateral coordinate is between the outer stumps."""
        y = np.atleast_1d(np.asarray(y, dtype=float))
        return np.abs(y) <= self.half_width

    def outside_leg(self, y) -> np.ndarray:
        y = np.atleast_1d(np.asarray(y, dtype=float))
        return y < -self.half_width

    def outside_off(self, y) -> np.ndarray:
        y = np.atleast_1d(np.asarray(y, dtype=float))
        return y > self.half_width

    def band_distance(self, y) -> np.ndarray:
        """Signed distance from the nearest stump line (positive = outside)."""
        y = np.atleast_1d(np.asarray(y, dtype=float))
        return np.abs(y) - self.half_width

    # -- reporting --------------------------------------------------------

    def stump_name(self, y: float) -> str:
        """Nearest stump name for a given lateral coordinate."""
        offs = np.array(self.stump_offsets)
        i = int(np.argmin(np.abs(offs - y)))
        from drs.config import STUMP_NAMES

        return STUMP_NAMES[i]

    def as_cylinders(self) -> list[tuple[np.ndarray, float, float, float]]:
        """Geometry as ``(centre_xy, radius, z0, z1)`` tuples, for plotting."""
        return [((self.x, off), self.stump_radius, 0.0, self.stump_height) for off in self.stump_offsets]


# --------------------------------------------------------------------------
# Plane intersections
# --------------------------------------------------------------------------


def intersect_ray_plane(origin: np.ndarray, direction: np.ndarray, point: np.ndarray, normal: np.ndarray):
    """Intersect a ray with a plane; returns the parameter ``s``."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    denom = float(np.dot(direction, normal))
    if abs(denom) < 1e-12:
        return None
    return float(np.dot(np.asarray(point) - origin, normal) / denom)


def intersect_rays(origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares closest point to a set of rays (the midpoint solution).

    This is the alternative to :func:`triangulate` and is useful for showing
    that the two agree when the rays genuinely intersect, and how they diverge
    as noise is added.
    """
    origins = np.atleast_2d(np.asarray(origins, dtype=float))
    directions = np.atleast_2d(np.asarray(directions, dtype=float))
    unit = np.array([d / np.linalg.norm(d) for d in directions])

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, unit):
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ o
    X = np.linalg.solve(A, b)

    residual = float(
        np.mean([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (X - o)) for o, d in zip(origins, unit)])
    )
    return X, residual
