"""
Forward dynamics of a cricket delivery.

This module answers the "physics" half of the project: given initial
conditions and a bounce model, where does the ball go?  It deliberately
exposes *two* levels of model so that the error introduced by each
simplification can be measured later:

1.  **Ballistic (drag-free).**  With the only force being gravity the motion
    is analytic and piecewise-quadratic.  Between bounces

        r(t) = r0 + v0 (t - t0) + 1/2 g (t - t0)^2 ,   g = (0, 0, -9.81),

    so the trajectory is a sequence of parabolas joined at the bounce times.
    This is the model the estimators invert, and it is exact in the sense that
    a noiseless sample of it is recovered with zero error.

2.  **Quadratic drag.**  Adding aerodynamic drag

        m dv/dt = m g - 1/2 rho C_D A ||v - w|| (v - w)

    destroys the closed form, so the trajectory is integrated numerically with
    ``scipy.integrate.solve_ivp``.  Comparing (1) and (2) on the same delivery
    quantifies how much the drag-free assumption costs.

The bounce itself is an impulsive event modelled by the velocity map

        v_z^+ = -e v_z^- ,      v_x^+ = k_x v_x^- ,      v_y^+ = k_y v_y^- ,

with restitution ``e`` and horizontal retention ``k_x, k_y``.  These are
*not* universal constants: they depend on the ball, the surface and the
delivery.  The project therefore treats them as parameters to be estimated or
assumed, and studies the sensitivity of the final decision to them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from drs.config import (
    BALL_AREA,
    BALL_MASS,
    BALL_RADIUS,
    DRAG_COEFFICIENT,
    G,
    G_VEC,
    RHO_AIR,
)

_EPS = 1e-12


# --------------------------------------------------------------------------
# Delivery specification
# --------------------------------------------------------------------------


@dataclass
class BounceModel:
    """Impulsive bounce: velocity immediately after the ball hits the pitch.

    Parameters
    ----------
    restitution:
        Coefficient of restitution ``e`` for the vertical component.
        ``v_z^+ = -e * v_z^-``.  A cricket ball on a turf pitch typically
        rebounds with 0.4 < e < 0.65.
    retain_x, retain_y:
        Fraction of the horizontal velocity retained through the impact.
        Losses come from friction and from the ball's spin; typical values are
        0.75-0.90.  Deviations between the pre- and post-bounce *direction*
        caused by spin are captured by ``spin_kick``.
    spin_kick:
        Additional lateral velocity (m/s) imparted by spin, applied along
        ``spin_direction`` (a unit vector in the (x, y) plane).
    """

    restitution: float = 0.55
    retain_x: float = 0.88
    retain_y: float = 0.88
    spin_kick: float = 0.0
    spin_direction: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must lie in [0, 1]")
        if not 0.0 < self.retain_x <= 1.0 or not 0.0 < self.retain_y <= 1.0:
            raise ValueError("horizontal retention must lie in (0, 1]")

    def apply(self, v_minus: np.ndarray) -> np.ndarray:
        """Map the incoming velocity to the outgoing velocity."""
        v_out = np.array(
            [
                self.retain_x * v_minus[0],
                self.retain_y * v_minus[1],
                -self.restitution * v_minus[2],
            ],
            dtype=float,
        )
        if self.spin_kick:
            dx, dy = self.spin_direction
            norm = float(np.hypot(dx, dy)) or 1.0
            v_out[0] += self.spin_kick * dx / norm
            v_out[1] += self.spin_kick * dy / norm
        return v_out


@dataclass
class Delivery:
    """A single delivery to be simulated.

    Parameters
    ----------
    release:
        Position of the ball centre at release (m), in the package frame.
    release_velocity:
        Velocity of the ball centre at release (m/s).
    bounce:
        The :class:`BounceModel` used at every ground contact.
    bounce_height:
        Height of the ball *centre* at which contact with the pitch is
        declared (m).  Defaults to the ball radius, i.e. the moment the ball
        touches the ground.
    x_end:
        The simulation stops once the ball reaches this x plane.  The default
        is the plane of the stumps.
    max_time:
        Safety cap on the simulated flight time (s).
    drag_enabled:
        Whether the aerodynamic drag term is included.
    wind:
        Air velocity (m/s) in the package frame.  Zero means still air.
    """

    release: np.ndarray
    release_velocity: np.ndarray
    bounce: BounceModel = field(default_factory=BounceModel)
    bounce_height: float = BALL_RADIUS
    x_end: float = 0.0
    max_time: float = 2.0
    drag_enabled: bool = False
    wind: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.release = np.asarray(self.release, dtype=float)
        self.release_velocity = np.asarray(self.release_velocity, dtype=float)
        self.wind = np.asarray(self.wind, dtype=float)

    # -- convenient constructors ------------------------------------------

    @classmethod
    def from_speed_angle(
        cls,
        speed_kmh: float,
        release: tuple[float, float, float],
        direction_xz_deg: float,
        y_vel: float = 0.0,
        **kwargs,
    ) -> "Delivery":
        """Build a delivery from a release speed and a vertical launch angle.

        Parameters
        ----------
        speed_kmh:
            Release speed in km/h (broadcast "speed gun" convention: the
            magnitude of the release velocity).
        release:
            Release point.
        direction_xz_deg:
            Angle of the velocity vector below the horizontal, in the x-z
            plane (positive = downwards).  Typical fast bowling is 5-10 deg.
        y_vel:
            Lateral velocity component (m/s).
        """
        speed = speed_kmh / 3.6
        phi = np.deg2rad(direction_xz_deg)
        # Component magnitudes such that sqrt(vx^2 + vz^2) == speed.
        vx = speed * np.cos(phi)
        vz = -speed * np.sin(phi)
        return cls(
            release=np.asarray(release, dtype=float),
            release_velocity=np.array([vx, y_vel, vz]),
            **kwargs,
        )


# --------------------------------------------------------------------------
# Trajectory containers
# --------------------------------------------------------------------------


class Trajectory:
    """Common interface for the simulated trajectories of this module."""

    bounce_times: np.ndarray
    duration: float

    def position(self, t: float | np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def velocity(self, t: float | np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def sample(self, n: int | None = None, t: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(times, positions)`` on a uniform (or supplied) time grid."""
        if t is None:
            if n is None:
                n = 200
            t = np.linspace(0.0, self.duration, int(n))
        return t, self.position(t)

    def at_plane(self, x_target: float = 0.0) -> tuple[float, np.ndarray]:
        """Earliest time and position at which the ball crosses ``x = x_target``.

        This is the quantity the LBW decision depends on: the ball's state in
        the plane of the stumps.
        """
        raise NotImplementedError

    def pitch_point(self) -> np.ndarray:
        """Position of the first bounce (``nan`` vector if the ball never bounced)."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} duration={self.duration:.3f}s bounces={len(self.bounce_times)}>"


@dataclass
class _Segment:
    t0: float
    r0: np.ndarray
    v0: np.ndarray
    g: np.ndarray
    t_span: float = 0.0


class PiecewiseBallistic(Trajectory):
    """Sequence of analytic parabolas joined at bounce times.

    Between two consecutive bounces the acceleration is exactly ``-g z_hat``,
    so the position and velocity have closed forms.  This is the object that
    :mod:`drs.estimate` inverts.
    """

    def __init__(self, segments: list[_Segment], bounce_times: np.ndarray, bounce_points: np.ndarray):
        if not segments:
            raise ValueError("a trajectory needs at least one segment")
        self.segments = segments
        self.bounce_times = np.asarray(bounce_times, dtype=float)
        self.bounce_points = np.asarray(bounce_points, dtype=float)
        self._t0 = np.array([s.t0 for s in segments], dtype=float)
        self.duration = segments[-1].t0 + segments[-1].t_span

    def _locate(self, t: np.ndarray) -> np.ndarray:
        """Index of the segment containing each t (clipped into range)."""
        idx = np.searchsorted(self._t0, t, side="right") - 1
        return np.clip(idx, 0, len(self.segments) - 1)

    def position(self, t: float | np.ndarray) -> np.ndarray:
        scalar = np.ndim(t) == 0
        t = np.atleast_1d(np.asarray(t, dtype=float))
        idx = self._locate(t)
        out = np.empty((t.size, 3))
        for k, seg in enumerate(self.segments):
            m = idx == k
            if not np.any(m):
                continue
            tau = (t[m] - seg.t0)[:, None]
            out[m] = seg.r0 + seg.v0 * tau + 0.5 * seg.g * tau**2
        return out[0] if scalar else out

    def velocity(self, t: float | np.ndarray) -> np.ndarray:
        scalar = np.ndim(t) == 0
        t = np.atleast_1d(np.asarray(t, dtype=float))
        idx = self._locate(t)
        out = np.empty((t.size, 3))
        for k, seg in enumerate(self.segments):
            m = idx == k
            if not np.any(m):
                continue
            tau = (t[m] - seg.t0)[:, None]
            out[m] = seg.v0 + seg.g * tau
        return out[0] if scalar else out

    def at_plane(self, x_target: float = 0.0) -> tuple[float, np.ndarray]:
        for seg in self.segments:
            vx = seg.v0[0]
            if abs(vx) < _EPS:
                continue
            tau = (x_target - seg.r0[0]) / vx
            if -1e-9 <= tau <= seg.t_span + 1e-9:
                t = seg.t0 + tau
                return t, seg.r0 + seg.v0 * tau + 0.5 * seg.g * tau**2
        raise ValueError(f"trajectory never reaches x = {x_target}")

    def pitch_point(self) -> np.ndarray:
        if len(self.bounce_points) == 0:
            return np.full(3, np.nan)
        return self.bounce_points[0]


@dataclass
class _SampledSegment:
    t: np.ndarray
    pos: np.ndarray
    vel: np.ndarray


class SampledTrajectory(Trajectory):
    """Numerically integrated trajectory (used when drag is enabled).

    Positions and velocities are stored on a fine time grid and interpolated
    with a cubic spline, so the class presents the same interface as
    :class:`PiecewiseBallistic`.
    """

    def __init__(self, segments: list[_SampledSegment], bounce_times: np.ndarray, bounce_points: np.ndarray):
        t = np.concatenate([s.t for s in segments])
        pos = np.vstack([s.pos for s in segments])
        vel = np.vstack([s.vel for s in segments])
        keep = np.concatenate([[True], np.diff(t) > 0])
        t, pos, vel = t[keep], pos[keep], vel[keep]
        self._t = t
        self._x = pos[:, 0]
        self._pos_spline = CubicSpline(t, pos, axis=0, extrapolate=True)
        self._vel_spline = CubicSpline(t, vel, axis=0, extrapolate=True)
        self.bounce_times = np.asarray(bounce_times, dtype=float)
        self.bounce_points = np.asarray(bounce_points, dtype=float)
        self.duration = float(t[-1])

    def position(self, t: float | np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        out = self._pos_spline(np.clip(t, self._t[0], self._t[-1]))
        return out if np.ndim(t) else out.reshape(3)

    def velocity(self, t: float | np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        out = self._vel_spline(np.clip(t, self._t[0], self._t[-1]))
        return out if np.ndim(t) else out.reshape(3)

    def at_plane(self, x_target: float = 0.0) -> tuple[float, np.ndarray]:
        from scipy.optimize import brentq

        x = self._x
        if x_target < x[0] - 1e-9 or x_target > x[-1] + 1e-9:
            raise ValueError(f"trajectory never reaches x = {x_target}")
        # The integrator halts a hair short of the terminal event, so a target
        # at the very end of the record is answered with the final state.
        if x_target >= x[-1]:
            return float(self._t[-1]), np.asarray(self._pos_spline(self._t[-1]))
        idx = int(np.searchsorted(x, x_target, side="left"))
        if idx <= 0:
            return float(self._t[0]), np.asarray(self._pos_spline(self._t[0]))
        f = lambda s: float(self._pos_spline(s)[0] - x_target)  # noqa: E731
        if f(self._t[idx - 1]) * f(self._t[idx]) > 0:
            # Numerically a touch rather than a crossing.
            return float(self._t[idx]), np.asarray(self._pos_spline(self._t[idx]))
        t_hit = brentq(f, self._t[idx - 1], self._t[idx], xtol=1e-12)
        return t_hit, np.asarray(self._pos_spline(t_hit))

    def pitch_point(self) -> np.ndarray:
        if len(self.bounce_points) == 0:
            return np.full(3, np.nan)
        return self.bounce_points[0]


# --------------------------------------------------------------------------
# Trajectory construction
# --------------------------------------------------------------------------


def _quadratic_roots(a: float, b: float, c: float) -> list[float]:
    """Real roots of ``a t^2 + b t + c = 0`` (``a`` may be zero)."""
    if abs(a) < _EPS:
        if abs(b) < _EPS:
            return []
        return [-c / b]
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    root = np.sqrt(disc)
    return [(-b - root) / (2 * a), (-b + root) / (2 * a)]


def _time_to_ground(r0: np.ndarray, v0: np.ndarray, z_ground: float) -> float | None:
    """Smallest strictly positive time at which a ballistic arc meets the pitch."""
    # 1/2 g tau^2 - v0z tau + (z_ground - z0) = 0
    roots = _quadratic_roots(0.5 * G, -v0[2], z_ground - r0[2])
    positive = [tau for tau in roots if tau > 1e-9]
    return min(positive) if positive else None


def _time_to_plane(r0: np.ndarray, v0: np.ndarray, x_target: float) -> float | None:
    """Smallest strictly positive time at which a ballistic arc crosses an x plane."""
    if abs(v0[0]) < _EPS:
        return None
    tau = (x_target - r0[0]) / v0[0]
    return tau if tau > 1e-9 else None


def simulate_delivery(delivery: Delivery) -> Trajectory:
    """Simulate a delivery and return its continuous trajectory.

    With ``delivery.drag_enabled`` false the result is a
    :class:`PiecewiseBallistic` (exact, analytic).  With drag enabled it is a
    :class:`SampledTrajectory` produced by numerical integration.
    """
    if delivery.drag_enabled:
        return _simulate_with_drag(delivery)
    return _simulate_ballistic(delivery)


def _simulate_ballistic(delivery: Delivery) -> PiecewiseBallistic:
    segments: list[_Segment] = []
    bounce_times: list[float] = []
    bounce_points: list[np.ndarray] = []

    t = 0.0
    r = delivery.release.copy()
    v = delivery.release_velocity.copy()
    g = G_VEC.copy()

    while t < delivery.max_time:
        t_bounce = _time_to_ground(r, v, delivery.bounce_height)
        t_plane = _time_to_plane(r, v, delivery.x_end)
        candidates = [c for c in (t_bounce, t_plane) if c is not None]
        if not candidates:
            break
        t_next = min(candidates)
        t_next = min(t_next, delivery.max_time - t)
        if t_next <= 1e-9:
            break

        segments.append(_Segment(t0=t, r0=r.copy(), v0=v.copy(), g=g, t_span=t_next))

        r_next = r + v * t_next + 0.5 * g * t_next**2
        v_next = v + g * t_next
        t += t_next

        if t_plane is not None and t_next >= t_plane - 1e-9:
            break  # reached the wicket plane

        # Ground contact: apply the impulsive bounce and continue.
        bounce_times.append(t)
        bounce_points.append(r_next.copy())
        r = r_next
        r[2] = delivery.bounce_height
        v = delivery.bounce.apply(v_next)

    return PiecewiseBallistic(segments, np.array(bounce_times), np.array(bounce_points).reshape(-1, 3))


def _drag_accel(v: np.ndarray, delivery: Delivery) -> np.ndarray:
    rel = v - delivery.wind
    speed = float(np.linalg.norm(rel))
    k = 0.5 * RHO_AIR * DRAG_COEFFICIENT * BALL_AREA / BALL_MASS
    return G_VEC - k * speed * rel


def _simulate_with_drag(delivery: Delivery, n_per_segment: int = 400) -> SampledTrajectory:
    segments: list[_SampledSegment] = []
    bounce_times: list[float] = []
    bounce_points: list[np.ndarray] = []

    t = 0.0
    state = np.concatenate([delivery.release.copy(), delivery.release_velocity.copy()])

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        return np.concatenate([y[3:], _drag_accel(y[3:], delivery)])

    def ground_event(_t: float, y: np.ndarray) -> float:
        return float(y[2] - delivery.bounce_height)

    ground_event.terminal = True  # type: ignore[attr-defined]
    ground_event.direction = -1  # type: ignore[attr-defined]

    def plane_event(_t: float, y: np.ndarray) -> float:
        return float(y[0] - delivery.x_end)

    plane_event.terminal = True  # type: ignore[attr-defined]
    plane_event.direction = 1  # type: ignore[attr-defined]

    max_segments = 8
    for _ in range(max_segments):
        if t >= delivery.max_time:
            break
        sol = solve_ivp(
            rhs,
            (t, delivery.max_time),
            state,
            events=[ground_event, plane_event],
            dense_output=True,
            rtol=1e-9,
            atol=1e-11,
            max_step=2e-3,
        )
        if sol.t.size == 0:
            break
        t_stop = float(sol.t[-1])
        grid = np.linspace(t, t_stop, max(2, n_per_segment))
        pos = sol.sol(grid).T[:, :3]
        vel = sol.sol(grid).T[:, 3:]
        segments.append(_SampledSegment(t=grid, pos=pos, vel=vel))

        state = sol.y[:, -1].copy()
        t = t_stop

        hit_plane = sol.t_events[1].size > 0 and abs(float(sol.t_events[1][0]) - t_stop) < 1e-6
        if hit_plane:
            break
        hit_ground = sol.t_events[0].size > 0 and abs(float(sol.t_events[0][0]) - t_stop) < 1e-6
        if not hit_ground:
            break  # integration ran out of time without a terminal event
        bounce_times.append(t)
        bounce_points.append(state[:3].copy())
        v_out = delivery.bounce.apply(state[3:])
        state = np.concatenate([state[:3], v_out])
        state[2] = delivery.bounce_height

    if not segments:  # pragma: no cover - only for pathological inputs
        raise RuntimeError("drag integration produced no trajectory")
    return SampledTrajectory(segments, np.array(bounce_times), np.array(bounce_points).reshape(-1, 3))


# --------------------------------------------------------------------------
# Preset deliveries used by the demos and experiments
# --------------------------------------------------------------------------


def _preset(
    x_release: float,
    speed_kmh: float,
    angle_deg: float,
    release_y: float,
    y_vel: float,
    **bounce_kwargs,
) -> Delivery:
    return Delivery.from_speed_angle(
        speed_kmh=speed_kmh,
        release=(x_release, release_y, 2.15),
        direction_xz_deg=angle_deg,
        y_vel=y_vel,
        bounce=BounceModel(**bounce_kwargs),
    )


PRESETS: dict[str, Delivery] = {
    # Good length, on the stumps: bounces ~6 m out and hits the wicket.
    "good_length_middle": _preset(-18.5, 140.0, 7.0, 0.10, -0.35),
    # Fuller and angling in: reaches the stumps lower down.
    "full_yorkerish": _preset(-18.5, 142.0, 5.5, 0.08, -0.15, restitution=0.50),
    # Short: bounces ~7 m out and passes over the top of the stumps.
    "short_over_stumps": _preset(-18.2, 138.0, 8.5, 0.05, 0.05, restitution=0.62, retain_x=0.90),
    # Lands just outside off stump and holds its line: a near miss off side.
    "outside_off": _preset(-18.5, 136.0, 7.0, 0.14, 0.18, retain_y=0.95),
    # Pitches outside leg stump: not out under Law 36 regardless of the rest.
    "pitches_outside_leg": _preset(-18.5, 138.0, 7.0, -0.10, -0.30, retain_y=0.95),
    # Marginal: pitches just in line with leg stump and clips the outside of
    # it -- the classic umpire's call.
    "umpires_call_leg": _preset(-18.5, 138.0, 7.0, -0.06, -0.14, retain_y=0.90),
    # A big off-break: pitches on middle stump and turns sharply across the
    # batter so that the ball grazes the outer edge of leg stump.  The spin is
    # chosen so that the *true* clearance is exactly zero, which makes this the
    # genuinely 50/50 review -- the case where uncertainty changes the answer.
    "leg_spin_marginal": _preset(-18.5, 132.0, 7.0, 0.02, -0.05, retain_y=0.90, spin_kick=-0.7741),
    # A leg-break from over the wicket: pitches well outside leg stump and turns
    # back to hit middle.  It would demolish the stumps, but Law 36 makes it
    # not out because of where it pitched -- the clearest example of ball
    # tracking alone not being the decision.
    "leg_break_into_stumps": _preset(-18.5, 130.0, 7.0, -0.10, -0.35, retain_y=0.90, spin_kick=1.10),
    # Slower ball: lower release speed, arrives at stump height.
    "slower_ball": _preset(-18.5, 118.0, 6.5, 0.10, -0.25),
}


def preset(name: str) -> Delivery:
    """Return an independent copy of a named preset delivery.

    A copy is returned so that callers may mutate the result (e.g. set
    ``drag_enabled``) without corrupting the module-level registry.
    """
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return copy.deepcopy(PRESETS[name])
