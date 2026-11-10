"""
Trajectory estimation: the inverse problem.

Given noisy observations ``p_i = r(t_i) + eps_i`` we must recover a
*continuous* trajectory and, from it, the ball's state in the plane of the
stumps.  Four estimators are implemented so that they can be compared rather
than merely asserted to work:

``polynomial``
    Per-axis least squares fit of ``a + b t + c t^2``.  Nine free parameters,
    no physics.  The fitted ``c_z`` yields a free estimate of gravity,
    ``g_hat = -2 c_z`` -- which is the point of the first experiment: with
    enough noise, a freely estimated gravity drifts away from 9.81.

``ballistic``
    The same functional form but with gravity *fixed*.  Writing

        r(t) = r0 + v0 t + 1/2 g t^2

    shows the model is **linear** in the six unknowns ``(r0, v0)``: subtract
    the known ``1/2 g t^2`` term and solve a 6-parameter linear least-squares
    problem.  Because it is linear, the parameter covariance is available in
    closed form, which is exactly what the Monte-Carlo layer needs.  Valid on
    one arc only.

``piecewise``
    A delivery genuinely bounces, so a single parabola is the wrong model.  The
    trajectory is a *piecewise* dynamical system with two arcs and an unknown
    switch index.  The switch index is recovered by scanning candidate
    break points and taking the one that minimises the total residual -- a
    one-dimensional discrete inverse problem.

``ballistic_drag``
    Gravity plus quadratic drag, integrated numerically, with the drag
    coefficient treated as a third unknown.  Used to answer "how much does
    ignoring air resistance cost?" rather than as the production model.

Every estimator returns a :class:`FitResult` carrying its parameters, an
optional covariance, and a ``predict`` callable, so downstream code
(uncertainty, decision, plots) is model-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares

from drs.config import BALL_AREA, BALL_MASS, G, G_VEC, RHO_AIR
from drs.simulate import Observations

MIN_POINTS_PER_ARC = 3


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class FitResult:
    """A fitted trajectory plus everything needed to reason about it.

    Attributes
    ----------
    model:
        Name of the estimator that produced the fit.
    params:
        Flat parameter vector, ordered as documented per model.
    param_names:
        Human-readable names, same order as ``params``.
    covariance:
        ``(p, p)`` parameter covariance, or ``None`` when unavailable.
    residual_rmse:
        Root-mean-square residual on the *fitted* observations (metres).
    n_obs:
        Number of observations used.
    dof:
        Degrees of freedom of the fit.
    metadata:
        Model-specific extras (estimated gravity, break time, arc split, ...).
    """

    model: str
    params: np.ndarray
    param_names: list[str]
    covariance: np.ndarray | None
    residual_rmse: float
    n_obs: int
    dof: int
    _predict: Callable[[np.ndarray], np.ndarray] = field(repr=False, default=None)  # type: ignore[assignment]
    metadata: dict = field(default_factory=dict)

    def predict(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return self._predict(t)

    def parameter_std(self) -> np.ndarray:
        if self.covariance is None:
            return np.full(self.params.shape, np.nan)
        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))

    def as_dict(self) -> dict:
        d = {
            "model": self.model,
            "residual_rmse_m": float(self.residual_rmse),
            "n_obs": int(self.n_obs),
            "dof": int(self.dof),
        }
        for name, value in zip(self.param_names, self.params):
            d[name] = float(value)
        d.update(self.metadata)
        return d


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _pooled_residual_variance(residuals: np.ndarray, n_params: int) -> float:
    """Unbiased estimate of the per-coordinate measurement variance."""
    n = residuals.size
    dof = n - n_params
    if dof <= 0:
        return float(np.mean(residuals**2))
    return float(np.sum(residuals**2) / dof)


def _rmse(residuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean(residuals**2)))


def linear_least_squares(A: np.ndarray, Y: np.ndarray, n_params: int):
    """Solve ``A Theta = Y`` with ``Theta`` (k, 3) and return fit + covariance.

    The three coordinates are treated as independent observations of the same
    design matrix with a common noise variance, so the parameter covariance is
    the Kronecker product ``Kron(inv(A^T A), sigma^2 I_3)`` for the parameter
    vector flattened *coefficient-major*
    ``[theta_1(x, y, z), theta_2(x, y, z), ...]``.
    """
    Theta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    residuals = Y - A @ Theta
    sigma2 = _pooled_residual_variance(residuals, n_params)
    AtA_inv = np.linalg.pinv(A.T @ A)
    cov = sigma2 * np.kron(AtA_inv, np.eye(3))
    params = Theta.reshape(-1)  # coefficient-major flatten
    return params, cov, residuals, sigma2


# --------------------------------------------------------------------------
# Model 1: unconstrained polynomial
# --------------------------------------------------------------------------


def fit_polynomial(obs: Observations, degree: int = 2, t_ref: float | None = None) -> FitResult:
    """Least-squares polynomial fit, one polynomial per coordinate.

    Parameters
    ----------
    obs:
        Observations (missing frames are skipped).
    degree:
        Polynomial degree.  ``2`` matches the drag-free ballistic model exactly.
    t_ref:
        Time origin for the polynomial.  Defaults to the first observation, so
        the constant term is the position at the first frame and the linear
        term is the velocity there.  This keeps the parameters well-scaled.
    """
    t_all, P = obs.times, obs.points
    if t_all.size < degree + 1:
        raise ValueError(f"need at least {degree + 1} observations, got {t_all.size}")

    t0 = float(t_all[0]) if t_ref is None else float(t_ref)
    tau = t_all - t0

    A = np.column_stack([tau**k for k in range(degree + 1)])
    params, cov, residuals, _ = linear_least_squares(A, P, degree + 1)

    names: list[str] = []
    for k in range(degree + 1):
        names += [f"c{k}_x", f"c{k}_y", f"c{k}_z"]

    coeffs = params.reshape(degree + 1, 3)

    def predict(t: np.ndarray) -> np.ndarray:
        tau = np.asarray(t, dtype=float) - t0
        basis = np.stack([tau**k for k in range(degree + 1)], axis=1)
        return basis @ coeffs

    meta = {"t_ref": t0, "degree": degree}
    if degree == 2:
        # a + b t + c t^2  ==>  r0 = a, v0 = b, accel = 2c, so g = -2 c_z
        meta["estimated_gravity"] = float(-2.0 * coeffs[2, 2])

    return FitResult(
        model="polynomial",
        params=params,
        param_names=names,
        covariance=cov,
        residual_rmse=_rmse(residuals),
        n_obs=int(t_all.size),
        dof=int(3 * t_all.size - 3 * (degree + 1)),
        _predict=predict,
        metadata=meta,
    )


# --------------------------------------------------------------------------
# Model 2: gravity-constrained ballistic arc
# --------------------------------------------------------------------------


def _ballistic_design(t: np.ndarray, t_ref: float) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix and the gravity offset for the linear ballistic model."""
    tau = t - t_ref
    A = np.column_stack([np.ones_like(tau), tau])
    offset = 0.5 * np.outer(tau**2, G_VEC)
    return A, offset


def fit_ballistic(
    obs: Observations,
    g: np.ndarray | float = G_VEC,
    t_ref: float | None = None,
    index: slice | None = None,
) -> FitResult:
    """Fit ``r(t) = r0 + v0 (t - t_ref) + 1/2 g (t - t_ref)^2`` by linear LS.

    Gravity is *imposed*, not fitted: that is the whole difference between this
    model and :func:`fit_polynomial`, and it removes three free parameters.
    With gravity fixed the residual is linear in ``(r0, v0)``, so the solution
    is a single least-squares solve and the covariance is exact.

    Parameters
    ----------
    index:
        Optionally restrict the fit to a slice of the observations (used by the
        piecewise estimator to fit one arc at a time).
    """
    gvec = np.full(3, float(g)) * np.array([0.0, 0.0, -1.0]) if np.isscalar(g) else np.asarray(g, dtype=float)

    t_all, P = obs.times, obs.points
    if index is not None:
        # ``index`` slices the *visible* observation sequence, which is the
        # indexing the piecewise estimator works in.
        t_all, P = t_all[index], P[index]

    if t_all.size < 2:
        raise ValueError("need at least 2 observations to fit a ballistic arc")

    t0 = float(t_all[0]) if t_ref is None else float(t_ref)
    A, offset = _ballistic_design(t_all, t0)
    Y = P - offset
    params, cov, residuals, _ = linear_least_squares(A, Y, 2)

    r0 = params[0:3]
    v0 = params[3:6]

    def predict(t: np.ndarray) -> np.ndarray:
        tau = np.asarray(t, dtype=float) - t0
        return r0 + np.outer(tau, v0) + 0.5 * np.outer(tau**2, gvec)

    return FitResult(
        model="ballistic",
        params=params,
        param_names=["x0", "y0", "z0", "vx", "vy", "vz"],
        covariance=cov,
        residual_rmse=_rmse(residuals),
        n_obs=int(t_all.size),
        dof=int(3 * t_all.size - 6),
        _predict=predict,
        metadata={"t_ref": t0, "g": gvec.tolist(), "g_fixed": True},
    )


# --------------------------------------------------------------------------
# Model 3: piecewise ballistic (pre- and post-bounce arcs)
# --------------------------------------------------------------------------


def fit_piecewise_ballistic(
    obs: Observations,
    break_index: int | None = None,
    g: np.ndarray | float = G_VEC,
    min_arc: int = MIN_POINTS_PER_ARC,
) -> FitResult:
    """Fit two ballistic arcs joined at an unknown bounce.

    The trajectory of a real delivery is a piecewise dynamical system: the
    ball obeys one parabola up to the bounce and a *different* one afterwards,
    because the bounce maps the velocity through

        v_z^+ = -e v_z^-,   v_x^+ = k_x v_x^-,   v_y^+ = k_y v_y^- .

    Fitting a single arc across the bounce is therefore a model error, however
    small the measurement noise.  Treating the switch index as unknown turns
    the problem into a one-dimensional discrete search: fit both arcs for every
    admissible split and keep the split with the smallest total residual.

    Parameters
    ----------
    break_index:
        Position (in the visible-observation indexing) of the first observation
        belonging to the second arc.  ``None`` triggers the search.
    """
    t_all, P = obs.times, obs.points
    n = t_all.size
    if n < 2 * min_arc:
        raise ValueError(f"piecewise fit needs at least {2 * min_arc} observations, got {n}")

    def build(split: int) -> tuple[FitResult, FitResult, float]:
        pre = fit_ballistic(obs, g=g, t_ref=float(t_all[0]), index=slice(0, split))
        post = fit_ballistic(obs, g=g, t_ref=float(t_all[split]), index=slice(split, n))
        # Compare on a common scale: the sum of squared residuals, so that
        # splits of different sizes are still comparable.
        sse = pre.residual_rmse**2 * 3 * pre.n_obs + post.residual_rmse**2 * 3 * post.n_obs
        return pre, post, float(sse)

    if break_index is None:
        candidates = range(min_arc, n - min_arc + 1)
        best = None
        for split in candidates:
            try:
                pre, post, sse = build(split)
            except (ValueError, np.linalg.LinAlgError):
                continue
            # Tie-break towards the earliest split: with a smooth arc the SSE
            # surface is flat, and the physically meaningful bounce is the first
            # point where the second arc genuinely fits better.
            if best is None or sse < best[2] - 1e-12:
                best = (split, (pre, post), sse)
        if best is None:
            raise ValueError("no admissible split found for the piecewise fit")
        split, (pre, post), sse = best
    else:
        split = int(break_index)
        pre, post, sse = build(split)

    t_break = float(t_all[split])
    visible_idx = np.flatnonzero(obs.visible)
    params = np.concatenate([pre.params, post.params])
    names = [f"pre_{n_}" for n_ in pre.param_names] + [f"post_{n_}" for n_ in post.param_names]

    cov = None
    if pre.covariance is not None and post.covariance is not None:
        cov = np.zeros((12, 12))
        cov[:6, :6] = pre.covariance
        cov[6:, 6:] = post.covariance

    def predict(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        out = np.empty((t.size, 3))
        m = t < t_break
        if np.any(m):
            out[m] = pre.predict(t[m])
        if np.any(~m):
            out[~m] = post.predict(t[~m])
        return out[0] if scalar else out

    # Residuals across both arcs, in metres.
    res = np.concatenate(
        [pre.predict(t_all[:split]) - P[:split], post.predict(t_all[split:]) - P[split:]]
    )

    return FitResult(
        model="piecewise",
        params=params,
        param_names=names,
        covariance=cov,
        residual_rmse=_rmse(res),
        n_obs=int(n),
        dof=int(3 * n - 12),
        _predict=predict,
        metadata={
            "break_index": int(split),
            "break_index_original": int(visible_idx[split]),
            "bounce_time": t_break,
            "sse": sse,
            "pre": pre,
            "post": post,
        },
    )


# --------------------------------------------------------------------------
# Model 4: ballistic with drag
# --------------------------------------------------------------------------

DRAG_K = 0.5 * RHO_AIR * BALL_AREA / BALL_MASS  # 1/m, multiplies C_D


def _drag_predictor(r0: np.ndarray, v0: np.ndarray, cd: float, t0: float = 0.0):
    """Return a callable evaluating the drag model at arbitrary times.

    The integration always starts at the model origin ``t0`` (the release), so
    that ``(r0, v0)`` retain their physical meaning as the release state.  A
    query is answered by integrating forward to the latest requested time and
    reading off the earlier ones with ``t_eval``.
    """
    y0 = np.concatenate([np.asarray(r0, dtype=float), np.asarray(v0, dtype=float)])

    def predict(t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        t_eval = np.clip(t, t0, None)
        t_end = float(t_eval.max())
        if t_end <= t0:
            t_end = t0 + 1e-6

        def rhs(_t, y):
            v = y[3:]
            speed = float(np.linalg.norm(v))
            return np.concatenate([v, G_VEC - cd * DRAG_K * speed * v])

        order = np.argsort(t_eval)
        sol = solve_ivp(
            rhs,
            (t0, t_end),
            y0,
            t_eval=t_eval[order],
            rtol=1e-8,
            atol=1e-10,
            max_step=5e-3,
        )
        y = np.asarray(sol.y, dtype=float)
        if y.ndim != 2 or y.shape[1] != t_eval.size:
            y = np.atleast_2d(np.asarray(sol.sol(t_eval[order]), dtype=float))
            if y.shape[0] != 6:  # dense output is (state, time)
                y = y.T
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        return y[:3].T[inv]

    return predict


def fit_ballistic_drag(
    obs: Observations,
    p0: np.ndarray | None = None,
    cd_bounds: tuple[float, float] = (0.0, 1.5),
    max_nfev: int = 200,
) -> FitResult:
    """Nonlinear least squares on ``(r0, v0, C_D)`` with drag integrated.

    The inverse problem is now genuinely nonlinear because the model is a
    boundary-value problem for an ODE rather than a closed form.  The Jacobian
    is estimated numerically, so this estimator is materially more expensive
    than the linear ones and is included mainly to *quantify* the drag-free
    approximation error.

    Only the pre-bounce arc is modelled; the bounce introduces a non-smooth
    event that would need to be handled explicitly.
    """
    t_all, P = obs.times, obs.points
    # The drag model's origin is the release, t = 0, so that (r0, v0) are the
    # physical release state.  The observations begin at the release time too.
    t0 = 0.0
    if p0 is None:
        base = fit_ballistic(obs, t_ref=t0)
        p0 = np.concatenate([base.params, [0.5]])

    def residual(p: np.ndarray) -> np.ndarray:
        r0, v0, cd = p[:3], p[3:6], float(p[6])
        try:
            pred = _drag_predictor(r0, v0, cd, t0=t0)(t_all)
        except (ValueError, RuntimeError):
            return np.full(P.size, 1e3)
        return (pred - P).reshape(-1)

    lb = np.concatenate([[-np.inf] * 6, [cd_bounds[0]]])
    ub = np.concatenate([[np.inf] * 6, [cd_bounds[1]]])
    result = least_squares(residual, p0, bounds=(lb, ub), max_nfev=max_nfev)

    r0, v0, cd = result.x[:3], result.x[3:6], float(result.x[6])
    predict = _drag_predictor(r0, v0, cd, t0=t0)

    # Numerical covariance from the Jacobian: cov ~ sigma^2 (J^T J)^-1.
    J = result.jac
    dof = max(1, J.shape[0] - J.shape[1])
    sigma2 = float(np.sum(result.fun**2) / dof)
    try:
        cov = sigma2 * np.linalg.pinv(J.T @ J)
    except np.linalg.LinAlgError:  # pragma: no cover
        cov = None

    return FitResult(
        model="ballistic_drag",
        params=result.x,
        param_names=["x0", "y0", "z0", "vx", "vy", "vz", "Cd"],
        covariance=cov,
        residual_rmse=_rmse(result.fun.reshape(-1, 3)),
        n_obs=int(t_all.size),
        dof=int(3 * t_all.size - 7),
        _predict=predict,
        metadata={"t_ref": t0, "drag_coefficient": cd, "cost": float(result.cost)},
    )


# --------------------------------------------------------------------------
# Model 5: single arc fitted to the post-bounce observations only
# --------------------------------------------------------------------------


def fit_post_bounce(obs: Observations, g: np.ndarray | float = G_VEC, **kwargs) -> FitResult:
    """Fit one ballistic arc to the post-bounce observations alone.

    This estimator exists to separate two very different errors.  A single-arc
    model fitted across the bounce fails badly (the bounce is a discontinuity in
    velocity, not a smooth feature), but the arc that actually determines the
    wicket-plane prediction is the *post-bounce* one.  Once the bounce has been
    located -- here using the piecewise search -- a plain gravity-constrained
    ballistic fit on the remaining frames is a fair, cheap and accurate
    alternative to the full two-arc model.

    The returned fit is a :class:`FitResult` whose ``predict`` extrapolates the
    post-bounce arc, so it can be compared directly against ``piecewise``.
    """
    pw = fit_piecewise_ballistic(obs, g=g, **kwargs)
    split = int(pw.metadata["break_index"])
    t_all = obs.times
    t_ref = float(t_all[split])

    post = fit_ballistic(obs, g=g, t_ref=t_ref, index=slice(split, None))
    post.model = "post_bounce"
    post.metadata.update(
        {
            "break_index": split,
            "break_index_original": int(pw.metadata["break_index_original"]),
            "bounce_time": t_ref,
            "arc_points": post.n_obs,
        }
    )
    return post


# --------------------------------------------------------------------------
# Dispatcher + evaluation utilities
# --------------------------------------------------------------------------

ESTIMATORS = {
    "polynomial": fit_polynomial,
    "ballistic": fit_ballistic,
    "post_bounce": fit_post_bounce,
    "piecewise": fit_piecewise_ballistic,
    "ballistic_drag": fit_ballistic_drag,
}


def fit(obs: Observations, model: str = "piecewise", **kwargs) -> FitResult:
    """Fit ``obs`` with a named estimator.

    ``"kalman"`` is handled here by a lazy import: the Kalman filter lives in
    :mod:`drs.kalman`, which imports this module, so registering it eagerly
    would create a cycle.  Every other name is a direct estimator.
    """
    if model == "kalman":
        from drs.kalman import fit_kalman  # local import: breaks the cycle

        kwargs.setdefault("sigma_meas", float(obs.sigma))
        return fit_kalman(obs, **kwargs)
    if model not in ESTIMATORS:
        raise KeyError(f"unknown model {model!r}; available: {sorted(ESTIMATORS) + ['kalman']}")
    return ESTIMATORS[model](obs, **kwargs)


# --------------------------------------------------------------------------
# Rebuilding a predictor from a parameter vector
# --------------------------------------------------------------------------


def rebuild_predictor(fit_result: FitResult, params: np.ndarray):
    """Return a ``predict(t)`` callable for a *new* parameter vector.

    The estimators in this module capture their parameters in a closure, which
    is convenient for a single fit but useless for uncertainty propagation: the
    Monte-Carlo layer needs to evaluate the model at thousands of perturbed
    parameter vectors.  This function re-creates the appropriate predictor from
    a raw vector, so that

        ``p_hat + delta``  ->  a trajectory

    is available for every model, and the meaning of the parameters stays in one
    place.
    """
    params = np.asarray(params, dtype=float)
    model = fit_result.model
    meta = fit_result.metadata

    if model == "polynomial":
        degree = int(meta.get("degree", 2))
        t0 = float(meta.get("t_ref", 0.0))
        coeffs = params.reshape(degree + 1, 3)

        def predict(t):
            tau = np.atleast_1d(np.asarray(t, dtype=float)) - t0
            basis = np.stack([tau**k for k in range(degree + 1)], axis=1)
            return basis @ coeffs

        return predict

    if model in ("ballistic", "post_bounce", "kalman"):
        t0 = float(meta.get("t_ref", 0.0))
        gvec = np.asarray(meta.get("g", G_VEC), dtype=float)
        r0, v0 = params[:3], params[3:6]

        def predict(t):
            tau = np.atleast_1d(np.asarray(t, dtype=float)) - t0
            return r0 + np.outer(tau, v0) + 0.5 * np.outer(tau**2, gvec)

        return predict

    if model == "piecewise":
        t_break = float(meta["bounce_time"])
        gvec = np.asarray(meta.get("g", G_VEC), dtype=float)
        pre_r0, pre_v0 = params[:3], params[3:6]
        post_r0, post_v0 = params[6:9], params[9:12]
        t_pre0 = float(meta["pre"].metadata.get("t_ref", 0.0))
        t_post0 = float(meta["post"].metadata.get("t_ref", t_break))

        def predict(t):
            t = np.atleast_1d(np.asarray(t, dtype=float))
            out = np.empty((t.size, 3))
            m = t < t_break
            if np.any(m):
                tau = t[m] - t_pre0
                out[m] = pre_r0 + np.outer(tau, pre_v0) + 0.5 * np.outer(tau**2, gvec)
            if np.any(~m):
                tau = t[~m] - t_post0
                out[~m] = post_r0 + np.outer(tau, post_v0) + 0.5 * np.outer(tau**2, gvec)
            return out

        return predict

    if model == "ballistic_drag":
        t0 = float(meta.get("t_ref", 0.0))
        return _drag_predictor(params[:3], params[3:6], float(params[6]), t0=t0)

    raise KeyError(f"cannot rebuild a predictor for model {model!r}")


# --------------------------------------------------------------------------
# Vectorised evaluation over a batch of parameter vectors
# --------------------------------------------------------------------------


def vectorized_predictor(fit_result: FitResult, params_batch: np.ndarray):
    """Return ``f(t) -> (n_samples, 3)`` evaluating every sample.

    ``t`` may be a scalar (every sample evaluated at the same time) or an array
    of shape ``(n_samples,)`` (each sample at its own time).  The second form is
    what makes the batched bisection in :func:`plane_crossing_times` work.

    Uncertainty quantification needs the *model* evaluated for thousands of
    parameter vectors.  Doing that one sample at a time is O(n) Python overhead
    per evaluation; evaluating all samples in one NumPy expression turns the
    same work into a single array operation.  Returns ``None`` for models whose
    predictor cannot be batched (the drag model integrates an ODE per sample).
    """
    P = np.asarray(params_batch, dtype=float)
    if P.ndim != 2:
        raise ValueError("params_batch must have shape (n_samples, n_params)")
    n = P.shape[0]
    model = fit_result.model
    meta = fit_result.metadata

    def tau_of(t, t0: float) -> np.ndarray:
        """Broadcast a time argument to shape ``(n, 1)``."""
        arr = np.asarray(t, dtype=float)
        arr = np.atleast_1d(arr)
        return (arr - t0)[:, None]

    if model == "polynomial":
        degree = int(meta.get("degree", 2))
        t0 = float(meta.get("t_ref", 0.0))
        coeffs = P.reshape(n, degree + 1, 3)

        def f(t) -> np.ndarray:
            tau = tau_of(t, t0)
            out = np.zeros((n, 3))
            for k in range(degree + 1):
                out += tau**k * coeffs[:, k, :]
            return out

        return f

    if model in ("ballistic", "post_bounce", "kalman"):
        t0 = float(meta.get("t_ref", 0.0))
        gvec = np.asarray(meta.get("g", G_VEC), dtype=float)
        r0, v0 = P[:, :3], P[:, 3:6]

        def f(t) -> np.ndarray:
            tau = tau_of(t, t0)
            return r0 + tau * v0 + 0.5 * tau**2 * gvec

        return f

    if model == "piecewise":
        t_break = float(meta["bounce_time"])
        gvec = np.asarray(meta.get("g", G_VEC), dtype=float)
        t_pre0 = float(meta["pre"].metadata.get("t_ref", 0.0))
        t_post0 = float(meta["post"].metadata.get("t_ref", t_break))
        pre_r0, pre_v0 = P[:, :3], P[:, 3:6]
        post_r0, post_v0 = P[:, 6:9], P[:, 9:12]

        def f(t) -> np.ndarray:
            arr = np.atleast_1d(np.asarray(t, dtype=float))[:, None]
            tau_pre = arr - t_pre0
            tau_post = arr - t_post0
            pre = pre_r0 + tau_pre * pre_v0 + 0.5 * tau_pre**2 * gvec
            post = post_r0 + tau_post * post_v0 + 0.5 * tau_post**2 * gvec
            return np.where(arr < t_break, pre, post)

        return f

    return None


def plane_crossing_times(
    fit_result: FitResult,
    params_batch: np.ndarray,
    x_plane: float = 0.0,
    t_lo: float = 0.0,
    t_hi: float | None = None,
    n_iter: int = 60,
) -> np.ndarray:
    """Vectorised solution of "when does each sampled trajectory reach x_plane?".

    A bracketed, fully batched bisection: every iteration evaluates the whole
    ensemble at one time value, so the cost is ``n_iter`` array operations
    rather than ``n_samples`` root finds.  Sixty iterations resolve the bracket
    to machine precision.
    """
    f = vectorized_predictor(fit_result, params_batch)
    if f is None:
        raise ValueError("this model has no batched predictor")

    lo = np.full(params_batch.shape[0], float(t_lo))
    if t_hi is None:
        t_hi = _default_horizon(fit_result)
    hi = np.full(params_batch.shape[0], float(t_hi))

    # Expand the upper bracket until every sample has passed the plane.
    for _ in range(40):
        x_hi = f(hi)[:, 0]
        behind = x_hi < x_plane
        if not np.any(behind):
            break
        hi[behind] *= 1.5
        hi = np.maximum(hi, lo + 1e-6)
    else:  # pragma: no cover
        raise ValueError("could not bracket the wicket plane for every sample")

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        x_mid = f(mid)[:, 0]
        before = x_mid < x_plane
        lo = np.where(before, mid, lo)
        hi = np.where(before, hi, mid)
    return 0.5 * (lo + hi)


def _default_horizon(fit_result: FitResult) -> float:
    """A generous upper bound on the time at which the ball reaches x = 0."""
    meta = fit_result.metadata
    t_last = meta.get("t_last")
    if t_last is None:
        t_ref = float(meta.get("t_ref", 0.0))
        t_last = t_ref + 0.5
    return float(t_last) * 2.0 + 0.5


def predict_rmse(fit_result: FitResult, t: np.ndarray, truth: np.ndarray) -> float:
    """RMSE of a fit against ground truth at the given times."""
    return float(np.sqrt(np.mean((fit_result.predict(t) - truth) ** 2)))


def time_at_plane(fit_result: FitResult, x_plane: float = 0.0, t_lo: float = 0.0, t_hi: float = 1.2) -> float:
    """First time at which the fitted trajectory reaches the plane ``x = x_plane``.

    A generic root find on the model's own ``predict``, so it works for the
    analytic and the numerically integrated models alike.
    """
    f = lambda s: float(fit_result.predict(np.array([s]))[0, 0] - x_plane)  # noqa: E731
    a, b = float(t_lo), float(t_hi)
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fa * fb > 0:
        # Expand the bracket once before giving up: extrapolated fits can need
        # a slightly longer window than the observed record.
        b = a + 2.0 * (b - a)
        fb = f(b)
        if fa * fb > 0:
            raise ValueError("fitted trajectory does not cross the wicket plane in the search window")
    return float(brentq(f, a, b, xtol=1e-12))


def state_at_plane(fit_result: FitResult, x_plane: float = 0.0, **kwargs) -> np.ndarray:
    """Predicted ball centre in the plane of the stumps."""
    t = time_at_plane(fit_result, x_plane=x_plane, **kwargs)
    return fit_result.predict(np.array([t]))[0]


def extrapolation_distance(fit_result: FitResult, obs: Observations) -> float:
    """How far beyond the last observation the prediction is extrapolated (m).

    The wicket-plane prediction is always an *extrapolation*: the tracking data
    stop when the ball reaches the batter.  Reporting this distance keeps the
    uncertainty discussion honest.
    """
    t_last = float(obs.times[-1])
    r_last = fit_result.predict(np.array([t_last]))[0]
    r_w = state_at_plane(fit_result)
    return float(np.linalg.norm(r_w - r_last))
