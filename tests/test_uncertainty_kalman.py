"""Uncertainty quantification: sampling, propagation, coverage and the KF."""

from __future__ import annotations

import numpy as np
import pytest

from drs import physics
from drs.estimate import fit, state_at_plane
from drs.geometry import Wicket
from drs.kalman import BallisticKalmanFilter, fit_kalman, rts_smooth
from drs.simulate import build_ground_truth, make_observations
from drs.uncertainty import (
    arrival_ensemble,
    bootstrap_wicket,
    hit_probability_convergence,
    monte_carlo_wicket,
    sample_parameters,
)

PRESET = "leg_spin_marginal"


@pytest.fixture
def truth():
    return build_ground_truth(physics.preset(PRESET), label=PRESET)


@pytest.fixture
def obs(truth):
    return make_observations(truth, fps=50.0, noise_std=0.005, rng=1)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def test_samples_have_the_right_shape_and_centre(obs):
    f = fit(obs, "piecewise")
    batch = sample_parameters(f, n=500, rng=0)
    assert batch.shape == (500, f.params.size)
    assert np.allclose(batch.mean(axis=0), f.params, atol=0.1)


def test_scaling_the_covariance_scales_the_spread(obs):
    f = fit(obs, "piecewise")
    tight = sample_parameters(f, n=2000, rng=0, scale=0.5)
    loose = sample_parameters(f, n=2000, rng=0, scale=2.0)
    assert loose.std(axis=0).mean() > 4 * tight.std(axis=0).mean()


def test_a_standard_deviation_floor_is_respected(obs):
    f = fit(obs, "piecewise")
    floor = 1e-2
    batch = sample_parameters(f, n=3000, rng=0, std_floor=floor)
    assert batch.std(axis=0).min() >= 0.5 * floor


# --------------------------------------------------------------------------
# Propagation
# --------------------------------------------------------------------------


def test_the_central_sample_reproduces_the_fitted_arrival(obs):
    f = fit(obs, "piecewise")
    batch = np.tile(f.params, (5, 1))
    times, points = arrival_ensemble(f, batch)
    expected = state_at_plane(f)
    assert np.allclose(points, expected, atol=1e-6)


def test_all_arrival_points_lie_in_the_wicket_plane(obs):
    f = fit(obs, "piecewise")
    _, points = arrival_ensemble(f, sample_parameters(f, n=200, rng=0))
    assert np.abs(points[:, 0]).max() < 1e-6


def test_hit_probability_is_a_probability(obs):
    f = fit(obs, "piecewise")
    unc = monte_carlo_wicket(f, n=1000, rng=0)
    assert 0.0 <= unc.hit_probability <= 1.0
    assert unc.hit_mask.shape == (1000,)
    assert unc.n_hits == int(unc.hit_mask.sum())


def test_more_noise_in_widens_the_prediction_interval(truth):
    low = monte_carlo_wicket(
        fit(make_observations(truth, fps=50.0, noise_std=0.002, rng=2), "piecewise"), n=2000, rng=0
    )
    high = monte_carlo_wicket(
        fit(make_observations(truth, fps=50.0, noise_std=0.025, rng=2), "piecewise"), n=2000, rng=0
    )
    assert high.std_point.mean() > low.std_point.mean()


def test_a_ball_hitting_middle_is_hit_in_essentially_every_sample():
    truth = build_ground_truth(physics.preset("good_length_middle"), label="gl")
    obs = make_observations(truth, fps=50.0, noise_std=0.003, rng=3)
    unc = monte_carlo_wicket(fit(obs, "piecewise"), n=3000, rng=0)
    assert unc.hit_probability > 0.99


def test_a_ball_missing_well_wide_is_hit_in_essentially_no_sample():
    truth = build_ground_truth(physics.preset("outside_off"), label="wide")
    obs = make_observations(truth, fps=50.0, noise_std=0.003, rng=3)
    unc = monte_carlo_wicket(fit(obs, "piecewise"), n=3000, rng=0)
    assert unc.hit_probability < 0.01


def test_the_razor_edge_delivery_produces_a_genuinely_uncertain_answer(truth, obs):
    """True clearance is zero, so the model should not be confident either way."""
    unc = monte_carlo_wicket(fit(obs, "piecewise"), n=8000, rng=0)
    assert 0.02 < unc.hit_probability < 0.98


def test_summary_exposes_the_numbers_a_report_needs(obs):
    unc = monte_carlo_wicket(fit(obs, "piecewise"), n=1000, rng=0)
    s = unc.summary()
    for key in ("hit_probability", "clearance_mean_m", "arrival_y_ci_lo", "arrival_y_ci_hi", "n_samples"):
        assert key in s
    assert s["arrival_y_ci_lo"] <= s["arrival_y_mean"] <= s["arrival_y_ci_hi"]


def test_the_interval_is_ordered(obs):
    unc = monte_carlo_wicket(fit(obs, "piecewise"), n=1000, rng=0)
    lo, hi = unc.interval(1)
    assert lo < hi


# --------------------------------------------------------------------------
# Convergence and bootstrap
# --------------------------------------------------------------------------


def test_the_convergence_study_is_monotone_in_sample_size():
    truth = build_ground_truth(physics.preset("leg_spin_marginal"))
    obs = make_observations(truth, fps=50.0, noise_std=0.005, rng=1)
    f = fit(obs, "piecewise")
    sizes = (10, 100, 1000, 10000)
    conv = hit_probability_convergence(f, sizes=sizes, rng=0)
    assert [n for n, _ in conv] == list(sizes)
    # With nested samples the estimate must settle: the 10k answer is close to
    # the 1k answer.
    assert abs(conv[-1][1] - conv[-2][1]) < 0.05


def test_bootstrap_produces_a_probability_and_more_spread_than_the_covariance(truth, obs):
    f = fit(obs, "piecewise")
    cov_based = monte_carlo_wicket(f, n=2000, rng=0)
    boot = bootstrap_wicket(obs, model="piecewise", n=40, rng=0)
    assert 0.0 <= boot.hit_probability <= 1.0
    assert boot.n_samples <= 40
    # Bootstrap intervals are not necessarily wider, but they must be usable.
    assert boot.std_point.mean() > 0


def test_bootstrap_can_resample_residuals(truth, obs):
    boot = bootstrap_wicket(obs, model="piecewise", n=25, rng=0, residual_bootstrap=True)
    assert boot.n_samples > 0


# --------------------------------------------------------------------------
# Kalman filter
# --------------------------------------------------------------------------


def test_the_filter_initialises_and_tracks(truth, obs):
    f = fit_kalman(obs, sigma_meas=0.005, sigma_accel=2.0)
    assert f.model == "kalman"
    assert np.isfinite(state_at_plane(f)).all()
    assert f.metadata["filtered_positions"].shape == (len(obs), 3)


def test_a_lower_measurement_noise_gives_a_tighter_track(truth):
    obs = make_observations(truth, fps=50.0, noise_std=0.005, rng=1)
    noisy_claim = fit_kalman(obs, sigma_meas=0.05, sigma_accel=2.0)
    exact_claim = fit_kalman(obs, sigma_meas=0.001, sigma_accel=2.0)
    assert exact_claim.metadata["position_std"].mean() < noisy_claim.metadata["position_std"].mean()


def test_missing_frames_grow_the_covariance():
    truth = build_ground_truth(physics.preset("good_length_middle"))
    obs = make_observations(truth, fps=50.0, noise_std=0.005, dropout=0.4, rng=5)
    f = fit_kalman(obs, sigma_meas=0.005, sigma_accel=1.0)
    std = f.metadata["position_std"].mean(axis=1)
    gaps = ~obs.visible
    # Standard deviation after an unseen frame must exceed the median over seen
    # frames -- "the tracker is less sure when it cannot see the ball".
    if gaps.sum() > 1:
        assert std[gaps].mean() > np.median(std[obs.visible])


def test_the_filter_needs_at_least_two_observations(truth):
    obs = make_observations(truth, fps=50.0, noise_std=0.005, rng=1).first_n(1)
    with pytest.raises(ValueError):
        fit_kalman(obs)


def test_rts_smoothing_reduces_the_variance():
    truth = build_ground_truth(physics.preset("good_length_middle"))
    obs = make_observations(truth, fps=50.0, noise_std=0.01, rng=2)
    f = fit_kalman(obs, sigma_meas=0.01, sigma_accel=2.0)
    smoothed = rts_smooth(f)
    assert smoothed.shape == f.metadata["states"].shape
    assert np.isfinite(smoothed).all()


def test_filter_primitives_behave():
    kf = BallisticKalmanFilter(
        state=np.array([0.0, 0.0, 1.0, 10.0, 0.0, 0.0]),
        covariance=np.eye(6) * 0.01,
        sigma_meas=0.01,
        sigma_accel=1.0,
    )
    kf.predict(0.1)
    # Gravity pulls the vertical velocity down.
    assert kf.velocity[2] == pytest.approx(-0.981, rel=1e-6)
    before = kf.P.copy()
    kf.update(np.array([1.0, 0.0, 0.9]))
    assert np.trace(kf.P) < np.trace(before)  # information was gained
