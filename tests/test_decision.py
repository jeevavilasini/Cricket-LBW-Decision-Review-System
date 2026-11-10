"""The LBW decision layer: the Laws, the zones and umpire's call."""

from __future__ import annotations

import numpy as np
import pytest

from drs import physics
from drs.config import DEFAULT_UMPIRE_CALL_MARGIN
from drs.decision import (
    IN_LINE,
    MISSING,
    OUTSIDE_LEG,
    OUTSIDE_OFF,
    classify_lateral,
    fitted_pitch_point,
    review_lbw,
)
from drs.estimate import fit
from drs.geometry import Wicket
from drs.physics import BounceModel, Delivery
from drs.simulate import build_ground_truth, make_observations


def _review(preset_name: str, on_field: str = "not_out", **kwargs):
    truth = build_ground_truth(physics.preset(preset_name), label=preset_name)
    obs = make_observations(truth, fps=50.0, noise_std=0.0, rng=0)
    f = fit(obs, "piecewise")
    return review_lbw(f, on_field=on_field, **kwargs), truth


def test_a_straight_hitting_delivery_is_out():
    verdict, _ = _review("good_length_middle")
    assert verdict.wicket.zone != MISSING
    assert verdict.pitching.zone == IN_LINE
    assert verdict.impact.zone == IN_LINE
    assert verdict.decision == "out"
    assert not verdict.umpires_call


def test_pitching_outside_leg_is_never_out():
    verdict, truth = _review("pitches_outside_leg")
    assert verdict.pitching.zone == OUTSIDE_LEG
    assert verdict.decision == "not_out"
    assert any("leg stump" in r for r in verdict.reasons)


def test_pitching_outside_leg_overrides_a_wicket_hit():
    """The ball would demolish middle stump, but it pitched outside leg."""
    verdict, truth = _review("leg_break_into_stumps")
    _, r_true, _ = truth.state_at_wicket()

    assert verdict.pitching.zone == OUTSIDE_LEG
    assert verdict.wicket.zone != MISSING, "this delivery is supposed to hit the stumps"
    assert Wicket().hits(r_true.reshape(1, 3))[0]
    assert verdict.decision == "not_out"
    assert any("pitched outside" in r for r in verdict.reasons)


def test_a_ball_passing_over_the_stumps_is_not_out():
    verdict, _ = _review("short_over_stumps")
    assert verdict.wicket.zone == MISSING
    assert verdict.decision == "not_out"


def test_impact_outside_leg_is_not_out_even_when_the_ball_would_hit():
    """The real ball grazes the leg stump, but it struck the batter outside leg."""
    truth = build_ground_truth(physics.preset("leg_spin_marginal"), label="spin")
    _, r_true, _ = truth.state_at_wicket()
    true_clearance = float(Wicket().clearance(r_true.reshape(1, 3))[0])
    assert abs(true_clearance) < 1e-3, "true clearance should be essentially zero"

    verdict, _ = _review("leg_spin_marginal")
    assert verdict.impact.zone == OUTSIDE_LEG
    assert verdict.decision == "not_out"
    assert any("outside the line of leg" in r for r in verdict.reasons)


def test_impact_outside_off_with_no_shot_is_not_out():
    truth = build_ground_truth(physics.preset("outside_off"), label="outside_off")
    obs = make_observations(truth, fps=50.0, noise_std=0.0, rng=0)
    f = fit(obs, "piecewise")

    with_shot = review_lbw(f, offered_shot=True)
    without_shot = review_lbw(f, offered_shot=False)
    assert without_shot.decision == "not_out"
    assert without_shot.impact.zone == OUTSIDE_OFF
    assert any("no shot" in r for r in without_shot.reasons)


def test_umpires_call_keeps_the_on_field_decision():
    """A marginal decision is not overruled in either direction."""
    from drs.estimate import state_at_plane

    truth = build_ground_truth(physics.preset("good_length_middle"), label="gl")
    obs = make_observations(truth, fps=50.0, noise_std=0.0, rng=0)
    f = fit(obs, "piecewise")

    # Force a marginal verdict by choosing a margin wide enough to swallow the
    # genuine clearance, then check that both on-field calls survive.
    clearance = float(Wicket().clearance(state_at_plane(f).reshape(1, 3))[0])
    margin = abs(clearance) + 0.005

    as_out = review_lbw(f, on_field="out", margin=margin)
    as_not_out = review_lbw(f, on_field="not_out", margin=margin)
    assert as_out.umpires_call and as_not_out.umpires_call
    assert as_out.decision == "out"
    assert as_not_out.decision == "not_out"


def test_a_wide_margin_makes_more_calls_marginal():
    """A comfortable hit becomes an umpire's call once the tolerance grows."""
    truth = build_ground_truth(physics.preset("good_length_middle"), label="gl")
    obs = make_observations(truth, fps=50.0, noise_std=0.0, rng=0)
    f = fit(obs, "piecewise")

    tight = review_lbw(f, on_field="not_out", margin=0.001)
    loose = review_lbw(f, on_field="not_out", margin=0.5)
    assert not tight.umpires_call
    assert loose.umpires_call


def test_classify_lateral_boundaries():
    w = Wicket()
    zone, marginal = classify_lateral(0.0, w, margin=0.03)
    assert zone == IN_LINE and not marginal

    zone, marginal = classify_lateral(w.half_width + 0.01, w, margin=0.03)
    assert zone == OUTSIDE_OFF and marginal

    zone, marginal = classify_lateral(-(w.half_width + 0.10), w, margin=0.03)
    assert zone == OUTSIDE_LEG and not marginal

    zone, marginal = classify_lateral(w.half_width - 0.01, w, margin=0.03)
    assert zone == IN_LINE and marginal


def test_pitch_point_lies_on_the_ground_and_before_the_stumps():
    truth = build_ground_truth(physics.preset("good_length_middle"))
    obs = make_observations(truth, fps=50.0, noise_std=0.0, rng=0)
    f = fit(obs, "piecewise")
    p = fitted_pitch_point(f)
    # Ground contact is defined as the ball centre reaching one ball-radius up.
    assert abs(p[2] - physics.BALL_RADIUS) < 2e-3
    assert p[0] < 0
    assert np.linalg.norm(p - truth.trajectory.pitch_point()) < 0.05


@pytest.mark.parametrize("fps", [50.0, 120.0, 150.0, 200.0])
@pytest.mark.parametrize("preset", ["good_length_middle", "leg_spin_marginal", "slower_ball",
                                    "short_over_stumps", "outside_off"])
def test_the_pitch_point_is_found_at_every_frame_rate(fps: float, preset: str):
    """Regression: a fixed sampling grid can straddle the bounce and miss it.

    At the bounce the fitted height touches the ground and turns, so a grid that
    happens to step over the turning point sees no sign change and wrongly
    concludes the ball never pitched.  Solving on the descending branch must
    find it regardless of where the frame boundaries fall.
    """
    truth = build_ground_truth(physics.preset(preset))
    obs = make_observations(truth, fps=fps, noise_std=0.005, rng=0)
    p = fitted_pitch_point(fit(obs, "piecewise"))
    true_pitch = truth.trajectory.pitch_point()

    assert not np.isnan(p).any(), f"no pitch point found at {fps} fps"
    # Ground contact: the ball centre is exactly one ball-radius above the pitch.
    assert abs(p[2] - physics.BALL_RADIUS) < 2e-3
    # The lateral coordinate is what the pitching-line decision uses.
    assert abs(p[1] - true_pitch[1]) < 0.02
    # The along-pitch coordinate is quantised by the frame interval.
    assert abs(p[0] - true_pitch[0]) < 2.0 / fps * 35.0
    assert p[0] < 0.0


def test_impact_plane_must_be_in_front_of_the_stumps():
    truth = build_ground_truth(physics.preset("good_length_middle"))
    obs = make_observations(truth, fps=50.0, noise_std=0.0, rng=0)
    f = fit(obs, "piecewise")
    with pytest.raises(ValueError):
        review_lbw(f, impact_x=0.1)


def test_verdict_serialises_for_reporting():
    verdict, _ = _review("good_length_middle")
    d = verdict.as_dict()
    for key in ("pitching_zone", "impact_zone", "wicket_zone", "decision", "umpires_call", "clearance_m"):
        assert key in d
    assert "Decision" in verdict.describe()


def test_a_yorker_that_cannot_bounce_is_indeterminate_but_does_not_crash():
    """A delivery already at the stumps with no bounce: no pitch point exists.

    Fitted and extrapolated far enough, the arc *would* meet the ground -- but
    behind the striker, which is not a pitching point at all.
    """
    delivery = Delivery(
        release=np.array([-2.0, 0.0, 0.6]),
        release_velocity=np.array([38.0, 0.0, -1.0]),
        bounce=BounceModel(restitution=0.5),
        x_end=0.0,
    )
    truth = build_ground_truth(delivery, label="no-bounce")
    assert truth.bounce_time is None

    obs = make_observations(truth, fps=50.0, noise_std=0.002, rng=0)
    f = fit(obs, "ballistic")
    assert np.isnan(fitted_pitch_point(f)).all()
    assert fitted_pitch_point(f, x_max=1e9)[0] > 0.0  # would "pitch" past the stumps

    verdict = review_lbw(f)
    assert verdict.decision in ("out", "not_out")
    assert verdict.pitching.marginal  # flagged as indeterminate rather than guessed
