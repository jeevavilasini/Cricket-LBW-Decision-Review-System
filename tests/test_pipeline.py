"""Observation model, experiment plumbing and end-to-end pipeline checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drs import experiments as X
from drs import physics
from drs.config import BALL_RADIUS
from drs.decision import review_lbw
from drs.estimate import fit, state_at_plane
from drs.geometry import Wicket, default_stereo_rig
from drs.simulate import (
    Observations,
    build_ground_truth,
    make_observations,
    sample_deliveries,
)


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------


def test_observations_have_one_row_per_frame():
    truth = build_ground_truth(physics.preset("good_length_middle"))
    obs = make_observations(truth, fps=50.0, noise_std=0.0)
    assert obs.positions.shape == (len(obs), 3)
    assert np.all(np.diff(obs.t) > 0)


def test_zero_noise_returns_the_exact_trajectory():
    truth = build_ground_truth(physics.preset("good_length_middle"))
    obs = make_observations(truth, fps=50.0, noise_std=0.0)
    assert np.allclose(obs.positions, truth.trajectory.position(obs.t))


def test_noise_is_calibrated():
    truth = build_ground_truth(physics.preset("good_length_middle"))
    sigma = 0.01
    obs = make_observations(truth, fps=50.0, noise_std=sigma, rng=0)
    residual = obs.positions - truth.trajectory.position(obs.t)
    assert residual.std() == pytest.approx(sigma, rel=0.15)


def test_noise_is_reproducible_for_a_given_seed():
    truth = build_ground_truth(physics.preset("good_length_middle"))
    a = make_observations(truth, noise_std=0.01, rng=42)
    b = make_observations(truth, noise_std=0.01, rng=42)
    assert np.array_equal(a.positions, b.positions)


def test_dropout_marks_the_requested_fraction():
    truth = build_ground_truth(physics.preset("good_length_middle"))
    obs = make_observations(truth, fps=50.0, noise_std=0.0, dropout=0.25, rng=0)
    assert obs.missing_fraction == pytest.approx(0.25, abs=0.05)
    assert obs.n_visible == int(obs.visible.sum())


def test_clustered_dropout_makes_one_contiguous_gap():
    truth = build_ground_truth(physics.preset("good_length_middle"))
    obs = make_observations(truth, fps=50.0, noise_std=0.0, dropout=0.3, rng=0, cluster_dropout=True)
    missing = np.flatnonzero(~obs.visible)
    assert missing.size > 0
    assert np.all(np.diff(missing) == 1)  # contiguous


def test_first_n_keeps_only_the_leading_frames():
    truth = build_ground_truth(physics.preset("good_length_middle"))
    obs = make_observations(truth, fps=50.0, noise_std=0.0)
    few = obs.first_n(6)
    assert few.n_visible == 6
    assert few.times[-1] < obs.times[-1]


def test_window_selects_a_time_interval():
    truth = build_ground_truth(physics.preset("good_length_middle"))
    obs = make_observations(truth, fps=50.0, noise_std=0.0)
    part = obs.window(0.1, 0.3)
    assert np.all((part.t >= 0.1) & (part.t <= 0.3))
    assert len(part) < len(obs)


def test_observations_reject_a_shape_mismatch():
    with pytest.raises(ValueError):
        Observations(t=np.zeros(5), positions=np.zeros((4, 3)))


# --------------------------------------------------------------------------
# Delivery sampling
# --------------------------------------------------------------------------


def test_sampled_deliveries_are_physically_plausible():
    truths = sample_deliveries(15, rng=1)
    assert len(truths) == 15
    for truth in truths:
        speed = float(np.linalg.norm(truth.release_velocity)) * 3.6
        assert 110.0 < speed < 160.0
        pitch = truth.trajectory.pitch_point()
        assert np.isfinite(pitch).all()
        assert -9.0 < pitch[0] < -3.0  # a plausible length
        t_hit, r_hit = truth.trajectory.at_plane(0.0)
        assert t_hit > 0
        assert r_hit[2] > 0.0


def test_sampling_is_reproducible():
    a = sample_deliveries(5, rng=7)
    b = sample_deliveries(5, rng=7)
    for x, y in zip(a, b):
        assert np.allclose(x.release, y.release)
        assert np.allclose(x.release_velocity, y.release_velocity)


# --------------------------------------------------------------------------
# Experiment plumbing
# --------------------------------------------------------------------------


def test_a_trial_row_contains_every_documented_metric():
    truth = build_ground_truth(physics.preset("good_length_middle"), label="t")
    row = X.evaluate_trial(truth, "piecewise", X.TrialConfig(noise_std=0.005), rng=0, monte_carlo=200)
    assert row["ok"] is True
    for key in ("plane_err_mm", "lateral_err_mm", "height_err_mm", "fit_rmse_mm", "hit_correct",
                "decision", "mc_hit_probability", "bounce_err_ms"):
        assert key in row
    assert row["plane_err_mm"] < 50.0  # the piecewise model on clean data


def test_the_piecewise_model_beats_the_single_arc_model_on_a_sweep():
    truths = sample_deliveries(8, rng=3)
    cfg = X.TrialConfig(noise_std=0.004)
    df = X.run_ensemble(truths, ("ballistic", "piecewise"), cfg, rng=5)
    med = df.dropna(subset=["plane_err_mm"]).groupby("model")["plane_err_mm"].median()
    assert med["piecewise"] < 0.1 * med["ballistic"]


def test_a_failed_fit_is_recorded_rather_than_raised():
    truth = build_ground_truth(physics.preset("good_length_middle"), label="t")
    cfg = X.TrialConfig(noise_std=0.005, n_frames=2)  # far too few for a piecewise fit
    row = X.evaluate_trial(truth, "piecewise", cfg, rng=0)
    assert row["ok"] is False
    assert "error" in row


def test_summarise_reports_percentiles():
    truths = sample_deliveries(6, rng=2)
    df = X.run_ensemble(truths, ("piecewise",), X.TrialConfig(noise_std=0.004), rng=1, n_repeats=2)
    summary = X.summarise(df, ["model"])
    assert set(summary.columns) >= {"model", "n", "median", "mean", "p90"}
    assert summary["median"].iloc[0] <= summary["p90"].iloc[0]


def test_heldout_error_separates_a_right_model_from_a_wrong_one():
    """The piecewise model generalises; a single parabola across the bounce does not."""
    truth = build_ground_truth(physics.preset("good_length_middle"), label="t")
    obs = make_observations(truth, fps=50.0, noise_std=0.003, rng=0)

    heldout_bad = X.heldout_rmse(truth, obs, "ballistic")
    heldout_good = X.heldout_rmse(truth, obs, "piecewise")

    assert heldout_good < 30.0, "the piecewise model should reconstruct unseen frames"
    assert heldout_bad > 100.0, "the single-arc model should fail on unseen frames"
    assert heldout_bad > 5 * heldout_good


def test_temporal_extrapolation_fails_across_a_bounce_for_every_model():
    """A finding, not a bug: no model can extrapolate a parabola past a bounce."""
    truth = build_ground_truth(physics.preset("good_length_middle"), label="t")
    obs = make_observations(truth, fps=50.0, noise_std=0.003, rng=0)
    err = X.heldout_rmse(truth, obs, "piecewise", mode="temporal", frac=0.5)
    assert err > 50.0


def test_experiment_outputs_are_written(tmp_path: Path):
    _, summary = X.experiment_noise(
        noise_levels_mm=(0.0, 5.0), n_deliveries=3, models=("piecewise",), n_repeats=1,
        seed=1, out_dir=tmp_path,
    )
    assert (tmp_path / "noise_trials.csv").exists()
    assert (tmp_path / "noise_summary.csv").exists()
    assert {"noise_mm", "model", "median"} <= set(summary.columns)


def test_the_summary_markdown_is_written(tmp_path: Path):
    df = X.run_ensemble(
        sample_deliveries(3, rng=1), ("piecewise",), X.TrialConfig(noise_std=0.004), rng=0
    )
    path = X.write_summary({"noise_trials": df}, tmp_path / "SUMMARY.md")
    text = path.read_text(encoding="utf-8")
    assert "wicket-plane error" in text.lower()
    assert "piecewise" in text


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["good_length_middle", "short_over_stumps", "outside_off",
                                    "leg_break_into_stumps", "leg_spin_marginal"])
def test_the_whole_pipeline_runs_for_every_preset(preset: str):
    truth = build_ground_truth(physics.preset(preset), label=preset)
    wicket = Wicket()
    obs = make_observations(truth, fps=50.0, noise_std=0.005, dropout=0.1, rng=0)

    f = fit(obs, "piecewise")
    p = state_at_plane(f)
    _, r_true, _ = truth.state_at_wicket()

    # The prediction must be accurate to the order of a couple of centimetres.
    assert np.linalg.norm(p - r_true) < 0.03
    # And the verdict must be a legal one.
    verdict = review_lbw(f)
    assert verdict.decision in ("out", "not_out")
    assert verdict.clearance == pytest.approx(float(wicket.clearance(p.reshape(1, 3))[0]))


def test_stereo_pipeline_recovers_the_wicket_plane():
    """Project -> triangulate -> fit -> decide, with no noise anywhere."""
    cams = default_stereo_rig()
    truth = build_ground_truth(physics.preset("good_length_middle"), label="gl")
    t = np.linspace(0.0, truth.trajectory.duration, 40)
    X_true = truth.trajectory.position(t)
    pixels = np.stack([c.project(X_true)[0] for c in cams], axis=0)

    from drs.geometry import triangulate_sequence

    recovered = triangulate_sequence(cams, pixels)
    obs = Observations(t=t, positions=recovered, sigma=1e-6, visible=np.ones(t.size, dtype=bool))
    f = fit(obs, "piecewise")

    _, r_true, _ = truth.state_at_wicket()
    assert np.linalg.norm(state_at_plane(f) - r_true) < 1e-6


def test_ball_radius_constant_matches_the_laws():
    # Law 4: diameter 2.80-2.86 in, i.e. a radius of 35.6-36.3 mm.
    assert 0.035 < BALL_RADIUS < 0.037
