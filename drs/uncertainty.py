"""
Uncertainty quantification for the wicket prediction.

A single trajectory and a single yes/no answer is not a defensible research
output.  The measurement is imperfect, the model is imperfect, and the
prediction is an extrapolation well beyond the last observed frame.  This
module turns that into a distribution.

Two propagation strategies are implemented, and the difference between them
matters:

**Parameter-covariance Monte Carlo.**  Draw

    theta_j ~ N(theta_hat, Sigma_hat),   j = 1 .. N

from the estimator's own parameter covariance, simulate each draw, and count
the fraction that would hit the wicket:

    P_hat(hit) = N_hits / N .

This is cheap and, for the *linear* estimators, the covariance is exact.

**Bootstrap Monte Carlo.**  Resample the observations (or the residuals) and
re-run the whole estimator.  This costs a refit per sample but captures
something the first method cannot: **model error**.  Parameter covariance
describes how well the chosen functional form fits the data; it says nothing
about whether that form is right.  Comparing the two intervals is one of the
project's most informative experiments.

Neither number is the official DRS probability and neither implements the
umpire's-call rule.  They are statements of uncertainty *in our own model*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drs.config import BALL_RADIUS
from drs.estimate import (
    FitResult,
    plane_crossing_times,
    vectorized_predictor,
)
from drs.geometry import Wicket
from drs.simulate import Observations

DEFAULT_N_SAMPLES = 5000


@dataclass
class UncertaintyResult:
    """Distribution of predicted wicket-plane outcomes."""

    method: str
    n_samples: int
    hit_probability: float
    hit_mask: np.ndarray
    arrival_points: np.ndarray
    clearances: np.ndarray
    times: np.ndarray
    weight_interval: tuple[float, float] = (0.025, 0.975)

    @property
    def n_hits(self) -> int:
        return int(self.hit_mask.sum())

    @property
    def mean_point(self) -> np.ndarray:
        return self.arrival_points.mean(axis=0)

    @property
    def std_point(self) -> np.ndarray:
        return self.arrival_points.std(axis=0)

    def interval(self, axis: int = 1) -> tuple[float, float]:
        lo, hi = self.weight_interval
        return (
            float(np.quantile(self.arrival_points[:, axis], lo)),
            float(np.quantile(self.arrival_points[:, axis], hi)),
        )

    def summary(self) -> dict:
        y_lo, y_hi = self.interval(1)
        z_lo, z_hi = self.interval(2)
        return {
            "method": self.method,
            "n_samples": int(self.n_samples),
            "hit_probability": float(self.hit_probability),
            "n_hits": self.n_hits,
            "clearance_mean_m": float(self.clearances.mean()),
            "clearance_std_m": float(self.clearances.std()),
            "arrival_y_mean": float(self.mean_point[1]),
            "arrival_z_mean": float(self.mean_point[2]),
            "arrival_y_ci_lo": y_lo,
            "arrival_y_ci_hi": y_hi,
            "arrival_z_ci_lo": z_lo,
            "arrival_z_ci_hi": z_hi,
        }


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def sample_parameters(
    fit_result: FitResult,
    n: int = DEFAULT_N_SAMPLES,
    rng: np.random.Generator | int | None = None,
    scale: float = 1.0,
    std_floor: np.ndarray | float | None = None,
) -> np.ndarray:
    """Draw parameter vectors from the estimator's Gaussian approximation.

    Parameters
    ----------
    scale:
        Inflate or deflate the whole covariance.  A convenient way to ask
        "what if the tracker were twice as accurate?".
    std_floor:
        Minimum per-parameter standard deviation.  A singular covariance (which
        happens for short tracks, where the data barely constrain the fit)
        would otherwise produce a falsely confident ensemble.

    Returns
    -------
    ndarray, shape ``(n, n_params)``
    """
    rng = np.random.default_rng(rng)
    theta = np.asarray(fit_result.params, dtype=float)
    p = theta.size

    if fit_result.covariance is None:
        cov = np.diag(np.full(p, 1e-3**2))
    else:
        cov = np.asarray(fit_result.covariance, dtype=float)
        # Symmetrise, then clip negative eigenvalues so that a numerically
        # indefinite covariance still yields a usable factor.
        cov = 0.5 * (cov + cov.T)
        w, V = np.linalg.eigh(cov)
        w = np.clip(w * scale**2, 0.0, None)
        cov = (V * w) @ V.T
        cov = 0.5 * (cov + cov.T)

    if std_floor is not None:
        floor = np.asarray(std_floor, dtype=float)
        if floor.ndim == 0:
            floor = np.full(p, float(floor))
        cov = cov + np.diag(np.maximum(floor, 0.0) ** 2)

    return rng.multivariate_normal(theta, cov, size=n, check_valid="ignore")


def arrival_ensemble(
    fit_result: FitResult,
    params_batch: np.ndarray,
    x_plane: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Wicket-plane arrival time and position for every sample.

    Uses the batched bisection from :mod:`drs.estimate` when the model allows
    it, and falls back to a per-sample root find for the ODE-based drag model.
    """
    f = vectorized_predictor(fit_result, params_batch)
    if f is not None:
        times = plane_crossing_times(fit_result, params_batch, x_plane=x_plane)
        points = f(times)
        return times, points

    from drs.estimate import rebuild_predictor, time_at_plane

    times = np.empty(params_batch.shape[0])
    points = np.empty((params_batch.shape[0], 3))
    for i, theta in enumerate(params_batch):
        pred = rebuild_predictor(fit_result, theta)

        def predict_scalar(t, _pred=pred):
            return _pred(np.array([t]))

        proxy = FitResult(
            model=fit_result.model,
            params=theta,
            param_names=fit_result.param_names,
            covariance=None,
            residual_rmse=fit_result.residual_rmse,
            n_obs=fit_result.n_obs,
            dof=fit_result.dof,
            _predict=predict_scalar,
            metadata=fit_result.metadata,
        )
        times[i] = time_at_plane(proxy, x_plane=x_plane)
        points[i] = proxy.predict(np.array([times[i]]))[0]
    return times, points


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------


def monte_carlo_wicket(
    fit_result: FitResult,
    wicket: Wicket | None = None,
    n: int = DEFAULT_N_SAMPLES,
    x_plane: float = 0.0,
    rng: np.random.Generator | int | None = None,
    scale: float = 1.0,
    std_floor: np.ndarray | float | None = None,
    ball_radius: float = BALL_RADIUS,
) -> UncertaintyResult:
    """Hit probability from the estimator's parameter covariance."""
    wicket = wicket or Wicket()
    params = sample_parameters(fit_result, n=n, rng=rng, scale=scale, std_floor=std_floor)
    times, points = arrival_ensemble(fit_result, params, x_plane=x_plane)
    clearances = wicket.clearance(points, ball_radius=ball_radius)
    hit = clearances <= 0.0
    return UncertaintyResult(
        method="parameter-covariance",
        n_samples=int(n),
        hit_probability=float(hit.mean()),
        hit_mask=hit,
        arrival_points=points,
        clearances=clearances,
        times=times,
    )


def bootstrap_wicket(
    obs: Observations,
    model: str = "piecewise",
    wicket: Wicket | None = None,
    n: int = 200,
    x_plane: float = 0.0,
    rng: np.random.Generator | int | None = None,
    noise_std: float | None = None,
    residual_bootstrap: bool = False,
    fit_kwargs: dict | None = None,
    ball_radius: float = BALL_RADIUS,
) -> UncertaintyResult:
    """Hit probability by refitting on resampled data.

    Unlike :func:`monte_carlo_wicket`, every sample here goes through the whole
    estimation pipeline.  That makes the ensemble sensitive to **model error**
    as well as measurement error: if the functional form cannot represent the
    true trajectory, the refits will disagree with one another more than the
    parameter covariance suggests.

    Parameters
    ----------
    noise_std:
        Standard deviation of the perturbation added to the observed positions.
        Defaults to the observation set's own ``sigma``.
    residual_bootstrap:
        When true, resample the *residuals* of an initial fit with replacement
        instead of adding fresh Gaussian noise.  This makes no assumption that
        the errors are Gaussian.
    """
    from drs.estimate import ESTIMATORS

    wicket = wicket or Wicket()
    fit_kwargs = dict(fit_kwargs or {})
    rng = np.random.default_rng(rng)
    sigma = float(obs.sigma if noise_std is None else noise_std)

    base_fit = ESTIMATORS[model](obs, **fit_kwargs)

    if residual_bootstrap:
        fitted = base_fit.predict(obs.times)
        residuals = (obs.points - fitted).reshape(-1, 3)

    points = np.empty((n, 3))
    times = np.empty(n)
    kept = 0
    attempts = 0
    max_attempts = 5 * n

    while kept < n and attempts < max_attempts:
        attempts += 1
        if residual_bootstrap:
            idx = rng.integers(0, residuals.shape[0], size=obs.times.shape[0])
            sample_points = base_fit.predict(obs.times) + residuals[idx]
        else:
            sample_points = obs.points + rng.normal(0.0, sigma, size=obs.points.shape)

        resampled = Observations(
            t=obs.t,
            positions=sample_points,
            sigma=sigma,
            visible=obs.visible,
            source=f"{obs.source}+bootstrap",
        )
        try:
            fit_i = ESTIMATORS[model](resampled, **fit_kwargs)
            t_i = _time_at_plane_safe(fit_i, x_plane)
            if t_i is None:
                continue
            points[kept] = fit_i.predict(np.array([t_i]))[0]
            times[kept] = t_i
            kept += 1
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            continue

    if kept == 0:  # pragma: no cover - only for pathological inputs
        raise RuntimeError("every bootstrap resample failed to fit")

    points, times = points[:kept], times[:kept]
    clearances = wicket.clearance(points, ball_radius=ball_radius)
    hit = clearances <= 0.0
    return UncertaintyResult(
        method="bootstrap",
        n_samples=int(kept),
        hit_probability=float(hit.mean()),
        hit_mask=hit,
        arrival_points=points,
        clearances=clearances,
        times=times,
    )


def _time_at_plane_safe(fit_result: FitResult, x_plane: float) -> float | None:
    from drs.estimate import time_at_plane

    try:
        return time_at_plane(fit_result, x_plane=x_plane)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Convergence study
# --------------------------------------------------------------------------


def hit_probability_convergence(
    fit_result: FitResult,
    wicket: Wicket | None = None,
    sizes: tuple[int, ...] = (10, 50, 100, 500, 1000, 5000, 10000, 50000),
    rng: np.random.Generator | int | None = None,
    **kwargs,
) -> list[tuple[int, float]]:
    """Hit probability as a function of the Monte-Carlo sample size.

    Answers the question "how many simulated trajectories are enough?" by
    running the same estimator at increasing ``n`` and watching the estimate
    settle.  The variability of this sequence is itself the answer.
    """
    rng = np.random.default_rng(rng)
    max_n = max(sizes)
    params = sample_parameters(fit_result, n=max_n, rng=rng, **kwargs)
    wicket = wicket or Wicket()
    times, points = arrival_ensemble(fit_result, params)
    hit = wicket.clearance(points) <= 0.0
    return [(int(k), float(hit[:k].mean())) for k in sizes]
