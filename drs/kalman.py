"""
A ballistic Kalman filter for ball tracking.

The estimators in :mod:`drs.estimate` are *batch* methods: they consume a
complete set of observations and return a complete trajectory.  A tracking
system, by contrast, must update its belief frame by frame and cope with
frames in which the ball is not detected at all.  The Kalman filter does both,
and it makes the uncertainty propagation explicit.

State and dynamics
------------------
With ``x = [p, v]`` (position and velocity, six components) the continuous
model is

    dp/dt = v ,        dv/dt = g ,

so between observations at times ``t`` and ``t + dt`` the discrete transition
is exactly

    F = [[I, dt I], [0, I]] ,     x_{k+1} = F x_k + [ dt^2/2 I ; dt I ] g + w .

When gravity is known the filter is a *constrained* estimator: it never
attributes to the ball an acceleration other than ``g``.  Everything the ball
does that gravity does not explain -- drag, swing, the bounce, spin -- enters
through the process noise ``w``.  Choosing the process-noise intensity ``q``
is therefore choosing how much to trust the physical model, and ``q`` is
exposed as a parameter so that its effect on the final decision can be
measured rather than hidden.

The measurement is the ball centre in three dimensions, ``z = H x`` with
``H = [I, 0]`` and ``R = sigma_meas^2 I``.

Missing detections are handled by simply propagating without an update, which
is what grows the covariance during an occlusion -- the quantitative statement
of "a tracker becomes less sure when it cannot see the ball".
"""

from __future__ import annotations

import numpy as np

from drs.config import G_VEC
from drs.estimate import FitResult
from drs.simulate import Observations


def _F(dt: float) -> np.ndarray:
    F = np.eye(6)
    F[:3, 3:] = dt * np.eye(3)
    return F


def _B(dt: float) -> np.ndarray:
    B = np.zeros((6, 3))
    B[:3] = 0.5 * dt**2 * np.eye(3)
    B[3:] = dt * np.eye(3)
    return B


def _Q(dt: float, q: float) -> np.ndarray:
    """Process noise from a continuous white-acceleration model."""
    Q = np.zeros((6, 6))
    Q[:3, :3] = (dt**3 / 3.0) * np.eye(3)
    Q[:3, 3:] = (dt**2 / 2.0) * np.eye(3)
    Q[3:, :3] = (dt**2 / 2.0) * np.eye(3)
    Q[3:, 3:] = dt * np.eye(3)
    return q * Q


class BallisticKalmanFilter:
    """Constant-gravity Kalman filter over ``[position, velocity]``.

    Parameters
    ----------
    state:
        Initial 6-vector guess ``[p, v]``.
    covariance:
        Initial 6x6 state covariance.
    sigma_meas:
        Per-coordinate measurement standard deviation (m).
    sigma_accel:
        Process-noise intensity (m/s^2 per sqrt(Hz)).  Zero means "trust the
        ballistic model completely"; large values let the data override it.
    g:
        Gravity vector.
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        sigma_meas: float = 0.01,
        sigma_accel: float = 1.0,
        g: np.ndarray = G_VEC,
    ) -> None:
        self.x = np.asarray(state, dtype=float).copy()
        self.P = np.asarray(covariance, dtype=float).copy()
        self.R = (sigma_meas**2) * np.eye(3)
        self.q = float(sigma_accel) ** 2
        self.g = np.asarray(g, dtype=float)
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])

    # -- primitives -------------------------------------------------------

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F, B = _F(dt), _B(dt)
        self.x = F @ self.x + B @ self.g
        self.P = F @ self.P @ F.T + _Q(dt, self.q)

    def update(self, z: np.ndarray) -> None:
        y = np.asarray(z, dtype=float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        # Joseph form: numerically stable and keeps P symmetric.
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def predict_only_state(self, dt: float) -> np.ndarray:
        """State after propagating ``dt`` with no measurement (used for forecasts)."""
        return _F(dt) @ self.x + _B(dt) @ self.g

    # -- convenience ------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        return self.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:]


def fit_kalman(
    obs: Observations,
    sigma_meas: float | None = None,
    sigma_accel: float = 2.0,
    g: np.ndarray = G_VEC,
    initial_position: np.ndarray | None = None,
    initial_velocity: np.ndarray | None = None,
    initial_position_std: float = 0.5,
    initial_velocity_std: float = 10.0,
) -> FitResult:
    """Run the ballistic Kalman filter over a set of observations.

    The filter is initialised from the first few observations (or from explicit
    guesses if supplied) and then stepped through the whole record.  Missing
    frames propagate the state without an update.

    Returns
    -------
    FitResult
        ``model == "kalman"``, ``params == [p_last, v_last]`` at the last
        observed time, and a ``predict`` that continues the ballistic motion
        beyond the record -- which is exactly what the wicket-plane prediction
        needs.
    """
    t = obs.t
    Z = obs.positions
    vis = obs.visible
    if vis.sum() < 2:
        raise ValueError("the Kalman filter needs at least 2 visible observations")

    sigma = float(obs.sigma if sigma_meas is None else sigma_meas)
    if sigma <= 0:
        sigma = 1e-4

    first = np.flatnonzero(vis)[:2]
    if initial_position is None:
        p0 = Z[first[0]]
    else:
        p0 = np.asarray(initial_position, dtype=float)
    if initial_velocity is None:
        dt = t[first[1]] - t[first[0]]
        v0 = (Z[first[1]] - Z[first[0]]) / dt
    else:
        v0 = np.asarray(initial_velocity, dtype=float)

    state0 = np.concatenate([p0, v0])
    P0 = np.diag(np.concatenate([np.full(3, initial_position_std**2), np.full(3, initial_velocity_std**2)]))

    kf = BallisticKalmanFilter(state0, P0, sigma_meas=sigma, sigma_accel=sigma_accel, g=g)

    states = np.zeros((t.size, 6))
    covs = np.zeros((t.size, 6, 6))
    for i in range(t.size):
        if i > 0:
            kf.predict(float(t[i] - t[i - 1]))
        if vis[i]:
            kf.update(Z[i])
        states[i] = kf.x
        covs[i] = kf.P

    t_last = float(t[-1])
    p_last, v_last = kf.x[:3].copy(), kf.x[3:].copy()
    gvec = np.asarray(g, dtype=float)

    def predict(tq) -> np.ndarray:
        tq = np.atleast_1d(np.asarray(tq, dtype=float))
        tau = tq - t_last
        return p_last + np.outer(tau, v_last) + 0.5 * np.outer(tau**2, gvec)

    # Residuals of the filtered track against the observations it was shown.
    res = states[vis, :3] - Z[vis]
    rmse = float(np.sqrt(np.mean(res**2)))

    return FitResult(
        model="kalman",
        params=np.concatenate([p_last, v_last]),
        param_names=["x", "y", "z", "vx", "vy", "vz"],
        covariance=kf.P.copy(),
        residual_rmse=rmse,
        n_obs=int(vis.sum()),
        dof=int(3 * vis.sum() - 6),
        _predict=predict,
        metadata={
            "t_last": t_last,
            "t_ref": t_last,
            "g": gvec.tolist(),
            "dt": float(t[1] - t[0]) if t.size > 1 else 0.0,
            "times": t,
            "sigma_meas": sigma,
            "sigma_accel": float(sigma_accel),
            "states": states,
            "covariances": covs,
            "filtered_positions": states[:, :3],
            "filtered_velocities": states[:, 3:],
            "position_std": np.sqrt(np.maximum(covs[:, np.arange(3), np.arange(3)], 0.0)),
        },
    )


def rts_smooth(fit_result: FitResult) -> np.ndarray:
    """Rauch-Tung-Striebel smoother over a :func:`fit_kalman` result.

    Optional refinement: the filtered estimate at each frame uses only past
    data, whereas DRS-style analysis is offline and may use the whole clip.
    Smoothing therefore gives a (slightly) better track and a lower variance --
    a useful upper bound against which to compare the causal filter.
    """
    states = fit_result.metadata.get("states")
    covs = fit_result.metadata.get("covariances")
    if states is None or covs is None:
        raise ValueError("RTS smoothing needs a fit produced by fit_kalman")
    n = states.shape[0]
    dt = fit_result.metadata.get("dt")
    if dt is None:
        raise ValueError("RTS smoothing requires the frame interval in metadata")

    xs = states.copy()
    Ps = covs.copy()
    for k in range(n - 2, -1, -1):
        F = _F(dt)
        Pp = F @ covs[k] @ F.T + _Q(dt, fit_result.metadata.get("sigma_accel", 1.0) ** 2)
        C = covs[k] @ F.T @ np.linalg.pinv(Pp)
        xs[k] = states[k] + C @ (xs[k + 1] - F @ states[k])
        Ps[k] = covs[k] + C @ (Ps[k + 1] - Pp) @ C.T
    return xs
