"""Estimator correctness, model error, covariance and the vectorised helpers."""

from __future__ import annotations

import numpy as np
import pytest

from drs import physics
from drs.config import G_VEC
from drs.estimate import (
    ESTIMATORS,
    fit,
    fit_ballistic,
    fit_piecewise_ballistic,
    fit_polynomial,
    plane_crossing_times,
    rebuild_predictor,
    state_at_plane,
    time_at_plane,
    vectorized_predictor,
)
from drs.kalman import fit_kalman
from drs.simulate import build_ground_truth, make_observations

PRESET = "good_length_middle"


@pytest.fixture
def truth():
    return build_ground_truth(physics.preset(PRESET))


@pytest.fixture
def obs(truth):
    return make_observations(truth, fps=50.0, noise_std=0.0, rng=0)


# --------------------------------------------------------------------------
# Noiseless recovery
# --------------------------------------------------------------------------


def test_piecewise_recovers_a_noiseless_trajectory_exactly(truth, obs):
    f = fit_piecewise_ballistic(obs)
    assert f.residual_rmse < 1e-9
    t_true, r_true, _ = truth.state_at_wicket()
    assert np.linalg.norm(state_at_plane(f) - r_true) < 1e-6


def test_piecewise_finds_the_bounce_to_within_one_frame(truth, obs):
    f = fit_piecewise_ballistic(obs)
    bounce_true = truth.bounce_time
    assert abs(f.metadata["bounce_time"] - bounce_true) <= (1.0 / 50.0) + 1e-9


def test_single_bounce_models_cannot_represent_a_bounced_delivery(truth, obs):
    """The central structural point: one parabola is the wrong model."""
    piecewise = fit_piecewise_ballistic(obs).residual_rmse
    single = fit_ballistic(obs).residual_rmse
    assert single > 50 * piecewise
    assert single > 0.05  # tens of millimetres of pure model error


def test_polynomial_estimates_gravity_when_it_is_genuinely_free(truth):
    """A single arc, no bounce: a free quadratic must return g = 9.81."""
    delivery = physics.Delivery(
        release=np.array([-18.5, 0.05, 2.1]),
        release_velocity=np.array([38.0, -0.3, -5.0]),
        x_end=0.0,
    )
    # Restrict to the pre-bounce window by ending the delivery at the bounce.
    traj = physics.simulate_delivery(delivery)
    t_b = float(traj.bounce_times[0])
    t = np.linspace(0.0, t_b * 0.95, 12)
    from drs.simulate import Observations

    obs = Observations(t=t, positions=traj.position(t), sigma=0.0)
    f = fit_polynomial(obs)
    assert f.metadata["estimated_gravity"] == pytest.approx(9.81, abs=1e-6)


def test_fixing_gravity_removes_three_free_parameters():
    """The ballistic model is the same curve with three fewer unknowns."""
    from drs.simulate import Observations

    delivery = physics.Delivery(
        release=np.array([-18.5, 0.05, 2.1]),
        release_velocity=np.array([38.0, -0.3, -5.0]),
    )
    traj = physics.simulate_delivery(delivery)
    t_b = float(traj.bounce_times[0])
    tt = np.linspace(0.0, t_b * 0.95, 12)
    obs = Observations(t=tt, positions=traj.position(tt), sigma=0.0)

    free = fit_polynomial(obs)
    fixed = fit_ballistic(obs)

    assert free.params.size == 9
    assert fixed.params.size == 6
    assert fixed.covariance.shape == (6, 6)
    # On a genuinely quadratic arc both fit perfectly.
    assert free.residual_rmse < 1e-9
    assert fixed.residual_rmse < 1e-9


# --------------------------------------------------------------------------
# Covariance
# --------------------------------------------------------------------------


def test_ballistic_covariance_is_symmetric_positive_semidefinite(truth):
    obs = make_observations(truth, fps=50.0, noise_std=0.005, rng=3)
    f = fit_ballistic(obs)
    C = f.covariance
    assert np.allclose(C, C.T)
    assert np.all(np.linalg.eigvalsh(C) > -1e-18)


def test_piecewise_covariance_is_block_diagonal_12_by_12(truth):
    obs = make_observations(truth, fps=50.0, noise_std=0.005, rng=3)
    f = fit_piecewise_ballistic(obs)
    assert f.covariance.shape == (12, 12)
    assert np.allclose(f.covariance[:6, 6:], 0.0)


def test_more_noise_produces_a_larger_parameter_uncertainty(truth):
    low = fit_ballistic(make_observations(truth, fps=50.0, noise_std=0.002, rng=1))
    high = fit_ballistic(make_observations(truth, fps=50.0, noise_std=0.020, rng=1))
    assert high.parameter_std()[3:].mean() > low.parameter_std()[3:].mean()


def test_more_frames_produce_a_smaller_uncertainty(truth):
    # Both budgets must stay within the pre-bounce arc: a single ballistic fit
    # that straddles the bounce is a different (misspecified) problem.
    all_obs = make_observations(truth, fps=50.0, noise_std=0.01, rng=1)
    few = fit_ballistic(all_obs.first_n(4))
    many = fit_ballistic(all_obs.first_n(15))
    assert many.parameter_std().mean() < few.parameter_std().mean()


# --------------------------------------------------------------------------
# Rebuilding, vectorisation and root finding
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["polynomial", "ballistic", "post_bounce", "piecewise"])
def test_rebuilt_predictor_matches_the_original(truth, model):
    obs = make_observations(truth, fps=50.0, noise_std=0.004, rng=11)
    f = fit(obs, model)
    predict = rebuild_predictor(f, f.params)
    t = np.linspace(0.0, 0.6, 25)
    assert np.allclose(predict(t), f.predict(t), atol=1e-9)


def test_rebuilding_from_the_parameter_and_covariance_recipe_is_self_consistent(truth):
    obs = make_observations(truth, fps=50.0, noise_std=0.004, rng=11)
    f = fit(obs, "ballistic")
    rebuilt = rebuild_predictor(f, f.params)
    assert np.allclose(rebuilt(np.array([0.2])), f.predict(np.array([0.2])), atol=1e-12)


@pytest.mark.parametrize("model", ["polynomial", "ballistic", "piecewise"])
def test_vectorised_predictor_agrees_with_the_scalar_predictor(truth, model):
    obs = make_observations(truth, fps=50.0, noise_std=0.004, rng=5)
    f = fit(obs, model)
    batch = np.vstack([f.params, f.params + 1e-6, f.params - 1e-6])
    fvec = vectorized_predictor(f, batch)
    for t in (0.05, 0.3, 0.55):
        expected = np.array([rebuild_predictor(f, theta)(np.array([t]))[0] for theta in batch])
        assert np.allclose(fvec(t), expected, atol=1e-9)


def test_vectorised_bisection_matches_the_scalar_root_finder(truth):
    obs = make_observations(truth, fps=50.0, noise_std=0.004, rng=5)
    f = fit(obs, "piecewise")
    batch = np.vstack([f.params, f.params + 1e-5])
    times = plane_crossing_times(f, batch, x_plane=0.0)
    for i, theta in enumerate(batch):
        scalar = time_at_plane(f, x_plane=0.0)
        assert times[i] == pytest.approx(scalar, abs=1e-6)


def test_state_at_plane_reproduces_the_plane_coordinate(truth):
    f = fit(make_observations(truth, fps=50.0, noise_std=0.005, rng=2), "piecewise")
    p = state_at_plane(f, x_plane=0.0)
    assert p[0] == pytest.approx(0.0, abs=1e-9)


def test_time_at_plane_raises_when_the_ball_never_arrives():
    delivery = physics.Delivery(
        release=np.array([-18.5, 0.0, 2.0]),
        release_velocity=np.array([-38.0, 0.0, 0.0]),  # travelling the wrong way
        x_end=-40.0,
    )
    traj = physics.simulate_delivery(delivery)
    from drs.simulate import Observations

    obs = Observations(t=np.linspace(0, 0.1, 8), positions=traj.position(np.linspace(0, 0.1, 8)))
    f = fit_ballistic(obs)
    with pytest.raises(ValueError):
        time_at_plane(f, x_plane=0.0)


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["polynomial", "ballistic", "post_bounce", "piecewise", "kalman"])
def test_every_model_survives_a_realistic_noisy_track(truth, model):
    obs = make_observations(truth, fps=50.0, noise_std=0.006, dropout=0.2, rng=4)
    f = fit(obs, model)
    assert np.all(np.isfinite(f.params))
    assert np.isfinite(state_at_plane(f)).all()


def test_piecewise_handles_missing_frames(truth):
    obs = make_observations(truth, fps=50.0, noise_std=0.004, dropout=0.25, rng=9)
    f = fit_piecewise_ballistic(obs)
    assert f.metadata["break_index_original"] < len(obs)


def test_too_few_points_is_reported_not_silently_wrong(truth):
    obs = make_observations(truth, fps=50.0, noise_std=0.004, rng=9).first_n(3)
    with pytest.raises(ValueError):
        fit_piecewise_ballistic(obs)


def test_estimator_registry_matches_the_documented_names():
    assert set(ESTIMATORS) == {"polynomial", "ballistic", "post_bounce", "piecewise", "ballistic_drag"}


def test_unknown_model_raises():
    from drs.simulate import Observations

    obs = Observations(t=np.linspace(0, 0.1, 8), positions=np.zeros((8, 3)))
    with pytest.raises(KeyError):
        fit(obs, "no_such_model")
