"""
Observation model: turning a continuous trajectory into the discrete, noisy,
occasionally missing measurements that a tracker would actually deliver.

The estimators in :mod:`drs.estimate` never see the true trajectory.  They see

    p_i = r(t_i) + eps_i ,      eps_i ~ N(0, sigma^2 I),

sampled at the camera frame rate, with a configurable fraction of frames
dropped (occlusion, motion blur, a failed detector).  Separating the
*generative* model (here) from the *inverse* model (``estimate``) is what makes
the error analysis in :mod:`drs.experiments` meaningful: ground truth is known
exactly, so every error metric is measured rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drs.config import BALL_RADIUS, G
from drs.physics import BounceModel, Delivery, Trajectory, simulate_delivery

# Typical broadcast tracking systems run at 50-100 Hz; 50 fps is a good default
# and leaves ~25 frames on a 0.5 s flight.
DEFAULT_FPS: float = 50.0

# A tracking system that quotes millimetre-to-centimetre accuracy in the plane
# of the stumps is realistic for a monocular rig.  1 cm is a deliberately
# pessimistic-but-fair default for a student prototype.
DEFAULT_NOISE_STD: float = 0.010


@dataclass
class Observations:
    """A set of measured ball positions at known times.

    Attributes
    ----------
    t:
        Frame times (s), measured from release, strictly increasing.
    positions:
        Observed ball centres, shape ``(n, 3)``, in metres.
    sigma:
        Standard deviation of the per-coordinate measurement error (m).  This
        is the number the uncertainty analysis must be honest about.
    visible:
        Boolean mask; ``False`` marks a frame where the ball was not detected.
        Missing frames are *retained in the record* (with a placeholder
        position) and filtered at fit time, which mirrors how a real pipeline
        keeps a fixed frame timeline.
    """

    t: np.ndarray
    positions: np.ndarray
    sigma: float = DEFAULT_NOISE_STD
    visible: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    source: str = "synthetic"

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.positions = np.asarray(self.positions, dtype=float)
        if self.visible.size == 0:
            self.visible = np.ones(self.t.shape, dtype=bool)
        else:
            self.visible = np.asarray(self.visible, dtype=bool)
        if self.positions.shape != (self.t.size, 3):
            raise ValueError("positions must have shape (n, 3) matching t")

    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def n_visible(self) -> int:
        return int(self.visible.sum())

    @property
    def times(self) -> np.ndarray:
        return self.t[self.visible]

    @property
    def points(self) -> np.ndarray:
        return self.positions[self.visible]

    @property
    def missing_fraction(self) -> float:
        return 1.0 - self.n_visible / max(1, len(self))

    def window(self, t_min: float | None = None, t_max: float | None = None) -> "Observations":
        """Return the subset of observations inside a time window."""
        lo = -np.inf if t_min is None else t_min
        hi = np.inf if t_max is None else t_max
        m = (self.t >= lo) & (self.t <= hi)
        return Observations(
            t=self.t[m],
            positions=self.positions[m],
            sigma=self.sigma,
            visible=self.visible[m],
            source=self.source,
        )

    def first_n(self, n: int) -> "Observations":
        """Keep only the first ``n`` visible observations (frame-budget study)."""
        idx = np.flatnonzero(self.visible)[:n]
        m = np.zeros_like(self.visible)
        m[idx] = True
        return self.select(m)

    def last_n(self, n: int) -> "Observations":
        """Keep only the last ``n`` visible observations.

        The mirror image of :meth:`first_n`, and the interesting comparison: a
        tracker whose *early* frames survive can still fail badly if it loses
        the ball before it pitches.
        """
        idx = np.flatnonzero(self.visible)[-n:]
        m = np.zeros_like(self.visible)
        m[idx] = True
        return self.select(m)

    def select(self, mask: np.ndarray) -> "Observations":
        """Keep the frames selected by a boolean mask.

        Unlike :meth:`window`, the mask may be arbitrary -- which is what a
        held-out validation split needs.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.t.shape:
            raise ValueError("mask must have one entry per frame")
        return Observations(
            t=self.t[mask],
            positions=self.positions[mask],
            sigma=self.sigma,
            visible=self.visible[mask],
            source=self.source,
        )


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


@dataclass
class GroundTruth:
    """A simulated delivery together with the trajectory it actually followed."""

    delivery: Delivery
    trajectory: Trajectory
    label: str = ""

    @property
    def release(self) -> np.ndarray:
        return self.delivery.release

    @property
    def release_velocity(self) -> np.ndarray:
        return self.delivery.release_velocity

    @property
    def bounce_time(self) -> float | None:
        return float(self.trajectory.bounce_times[0]) if len(self.trajectory.bounce_times) else None

    def state_at_wicket(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Time, position and velocity as the ball reaches the plane of the stumps."""
        t_hit, r_hit = self.trajectory.at_plane(0.0)
        return t_hit, r_hit, np.asarray(self.trajectory.velocity(t_hit))

    def parameters(self) -> np.ndarray:
        """The 6-vector ``[r0, v0]`` that defines the pre-bounce arc."""
        return np.concatenate([self.delivery.release, self.delivery.release_velocity])


def build_ground_truth(delivery: Delivery, label: str = "") -> GroundTruth:
    """Simulate a delivery once and wrap it as ground truth."""
    return GroundTruth(delivery=delivery, trajectory=simulate_delivery(delivery), label=label)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def frame_times(
    duration: float,
    fps: float = DEFAULT_FPS,
    t_start: float = 0.0,
) -> np.ndarray:
    """Uniform frame times covering ``[t_start, duration]`` at ``fps``."""
    n = int(np.floor((duration - t_start) * fps)) + 1
    n = max(n, 1)
    return t_start + np.arange(n) / fps


def make_observations(
    truth: GroundTruth | Trajectory,
    fps: float = DEFAULT_FPS,
    noise_std: float = DEFAULT_NOISE_STD,
    dropout: float = 0.0,
    t_start: float = 0.0,
    rng: np.random.Generator | int | None = None,
    cluster_dropout: bool = False,
) -> Observations:
    """Sample a trajectory on a video frame grid and corrupt it with noise.

    Parameters
    ----------
    truth:
        The trajectory (or :class:`GroundTruth` wrapping one) to observe.
    fps:
        Camera frame rate (Hz).
    noise_std:
        Per-coordinate standard deviation of the measurement noise (m).
        ``0`` reproduces the noiseless idealisation.
    dropout:
        Fraction of frames to mark as *not detected*.  When
        ``cluster_dropout`` is true the dropped frames are contiguous, which
        models an occlusion (the batter's body) rather than isolated failures.
    t_start:
        Delay before the first frame, useful for trimming the release.
    rng:
        Seed or ``numpy`` generator, so every experiment is reproducible.
    """
    traj = truth.trajectory if isinstance(truth, GroundTruth) else truth
    rng = np.random.default_rng(rng)

    t = frame_times(traj.duration, fps=fps, t_start=t_start)
    r_true = traj.position(t)

    if noise_std > 0:
        observed = r_true + rng.normal(0.0, noise_std, size=r_true.shape)
    else:
        observed = r_true.copy()

    n = t.size
    visible = np.ones(n, dtype=bool)
    n_drop = int(round(dropout * n))
    if n_drop > 0:
        if cluster_dropout:
            # One contiguous gap, kept away from the first and last frames so
            # that the trajectory remains bracketed by observations.
            lo = max(1, n // 4)
            hi = max(lo + n_drop, n - 1)
            hi = min(hi, n - 1)
            start = int(rng.integers(lo, max(lo + 1, hi - n_drop + 1)))
            visible[start : start + n_drop] = False
        else:
            drop_idx = rng.choice(n, size=min(n_drop, n), replace=False)
            visible[drop_idx] = False

    return Observations(t=t, positions=observed, sigma=float(noise_std), visible=visible, source="synthetic")


def noiseless_truth_positions(truth: GroundTruth | Trajectory, obs: Observations) -> np.ndarray:
    """The exact positions at the observation times (the hidden ground truth)."""
    traj = truth.trajectory if isinstance(truth, GroundTruth) else truth
    return traj.position(obs.t)


def bounce_observation_index(traj: Trajectory, t: np.ndarray) -> int | None:
    """Index of the first observation at or after the first bounce.

    Dividing observations this way defines the two sub-problems of the
    piecewise fit.  A real pipeline would have to *estimate* this index; the
    estimator module supports both the oracle and the searched variant.
    """
    if len(traj.bounce_times) == 0:
        return None
    tb = float(traj.bounce_times[0])
    idx = int(np.searchsorted(t, tb))
    if idx >= t.size:
        return None
    return idx


# --------------------------------------------------------------------------
# Batched truth generation for the experiments
# --------------------------------------------------------------------------


def sample_deliveries(
    n: int = 50,
    rng: np.random.Generator | int | None = None,
    speed_kmh_range: tuple[float, float] = (118.0, 148.0),
    length_x_range: tuple[float, float] = (-7.5, -4.0),
    line_y_range: tuple[float, float] = (-0.15, 0.15),
    drag: bool = False,
) -> list[GroundTruth]:
    """Draw a randomised-but-physical ensemble of deliveries.

    Rather than jittering the release state blindly, the sampler picks
    *cricketing* quantities -- a release speed, a length (the x at which the
    ball pitches) and a line (the y at which it pitches) -- and solves for the
    release velocity that delivers them.  The resulting ensemble looks like a
    realistic over instead of a cloud of arbitrary parabolas.
    """
    rng = np.random.default_rng(rng)
    out: list[GroundTruth] = []
    for i in range(n):
        speed = rng.uniform(*speed_kmh_range) / 3.6
        x_pitch = rng.uniform(*length_x_range)
        y_pitch = rng.uniform(*line_y_range)

        release_x = rng.uniform(-19.0, -18.0)
        release_y = y_pitch + rng.normal(0.0, 0.05)
        release_z = 2.15

        bounce = _default_bounce(rng)
        # Velocity that lands the ball at (x_pitch, y_pitch): the vertical
        # component follows from the time of flight, which itself depends on
        # the vertical drop, so solve the quadratic directly.
        dx = x_pitch - release_x
        # Choose the horizontal speed, then the flight time, then v_z.
        vx = speed
        t_flight = dx / vx
        vz = (BALL_RADIUS - release_z + 0.5 * G * t_flight**2) / t_flight
        vy = (y_pitch - release_y) / t_flight

        # Scale so that the release *speed* matches the requested speed.
        scale = speed / float(np.hypot(vx, vz))
        vx, vz = vx * scale, vz * scale
        vy *= scale

        delivery = Delivery(
            release=np.array([release_x, release_y, release_z]),
            release_velocity=np.array([vx, vy, vz]),
            bounce=bounce,
            drag_enabled=drag,
        )
        out.append(build_ground_truth(delivery, label=f"draw-{i:03d}"))
    return out


def _default_bounce(rng: np.random.Generator) -> BounceModel:
    return BounceModel(
        restitution=float(rng.uniform(0.45, 0.65)),
        retain_x=float(rng.uniform(0.80, 0.92)),
        retain_y=float(rng.uniform(0.80, 0.95)),
    )
