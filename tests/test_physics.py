"""Forward-model tests: the physics must be self-consistent and physical."""

from __future__ import annotations

import numpy as np
import pytest

from drs import physics
from drs.config import BALL_RADIUS, G, STUMP_HEIGHT
from drs.simulate import build_ground_truth


def test_release_speed_preserved_through_ballistic_flight():
    delivery = physics.preset("good_length_middle")
    traj = physics.simulate_delivery(delivery)
    v_release = np.linalg.norm(delivery.release_velocity)
    v_after = np.linalg.norm(traj.velocity(1e-9))
    assert v_after == pytest.approx(v_release, rel=1e-6)


def test_gravity_acts_only_on_the_vertical_component():
    delivery = physics.preset("good_length_middle")
    traj = physics.simulate_delivery(delivery)
    t_b = float(traj.bounce_times[0])
    # Sample just *before* the bounce: at exactly t_b the segment lookup already
    # returns the post-bounce velocity, because that is when the jump happens.
    v0 = np.asarray(traj.velocity(0.0))
    v1 = np.asarray(traj.velocity(t_b - 1e-9))
    # Horizontal velocity is untouched before the bounce...
    assert v1[0] == pytest.approx(v0[0], rel=1e-6)
    assert v1[1] == pytest.approx(v0[1], rel=1e-6)
    # ...and the vertical component changes by exactly -g t.
    assert v1[2] == pytest.approx(v0[2] - G * t_b, rel=1e-6)


def test_bounce_vertical_velocity_is_reversed_and_scaled():
    delivery = physics.preset("good_length_middle")
    traj = physics.simulate_delivery(delivery)
    t_b = float(traj.bounce_times[0])
    eps = 1e-6
    v_minus = np.asarray(traj.velocity(t_b - eps))
    v_plus = np.asarray(traj.velocity(t_b + eps))
    e = delivery.bounce.restitution
    assert v_plus[2] > 0 > v_minus[2]
    assert v_plus[2] == pytest.approx(-e * v_minus[2], rel=1e-3)


def test_position_is_continuous_across_the_bounce():
    """Only the velocity jumps; the ball does not teleport."""
    traj = physics.simulate_delivery(physics.preset("good_length_middle"))
    t_b = float(traj.bounce_times[0])
    eps = 1e-7
    assert np.allclose(traj.position(t_b - eps), traj.position(t_b + eps), atol=1e-6)


def test_all_presets_reach_the_wicket_plane_in_frame():
    for name in physics.PRESETS:
        traj = physics.simulate_delivery(physics.preset(name))
        t_hit, r_hit = traj.at_plane(0.0)
        assert t_hit > 0.0
        assert -0.6 < r_hit[1] < 0.6, f"{name} drifted implausibly far laterally"
        assert r_hit[2] > 0.0, f"{name} arrived below ground level"


def test_each_preset_behaves_as_its_name_promises():
    traj = physics.simulate_delivery(physics.preset("good_length_middle"))
    assert np.isfinite(traj.pitch_point()).all()
    assert traj.pitch_point()[0] == pytest.approx(-6.0, abs=1.5)

    short = physics.simulate_delivery(physics.preset("short_over_stumps"))
    assert short.at_plane(0.0)[1][2] > STUMP_HEIGHT  # passes over the top

    wide = physics.simulate_delivery(physics.preset("outside_off"))
    assert wide.at_plane(0.0)[1][1] > 0.15  # misses off side

    leg = physics.simulate_delivery(physics.preset("pitches_outside_leg"))
    assert leg.pitch_point()[1] < 0.0 and leg.at_plane(0.0)[1][1] < 0.0


def test_drag_shortens_the_length():
    """Air resistance removes speed, so the ball covers less ground before pitching."""
    ballistic = physics.simulate_delivery(physics.preset("good_length_middle"))

    dragged = physics.preset("good_length_middle")
    dragged.drag_enabled = True
    with_drag = physics.simulate_delivery(dragged)

    # The ball travels in +x from -18.5, so a *less* advanced pitch point (a more
    # negative x) means a shorter delivery.
    assert with_drag.pitch_point()[0] < ballistic.pitch_point()[0]

    # And it is still travelling more slowly when it reaches the stumps.
    v_ballistic = float(np.linalg.norm(ballistic.velocity(ballistic.at_plane(0.0)[0])))
    v_dragged = float(np.linalg.norm(with_drag.velocity(with_drag.at_plane(0.0)[0])))
    assert v_dragged < v_ballistic


def test_drag_model_still_arrives_above_the_ground():
    dragged = physics.preset("full_yorkerish")
    dragged.drag_enabled = True
    traj = physics.simulate_delivery(dragged)
    assert traj.at_plane(0.0)[1][2] > 0.0


def test_preset_returns_an_independent_copy():
    a = physics.preset("good_length_middle")
    a.drag_enabled = True
    a.release[0] = 123.0
    b = physics.preset("good_length_middle")
    assert b.drag_enabled is False
    assert b.release[0] != 123.0


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        physics.preset("no_such_delivery")


def test_bounce_model_rejects_unphysical_coefficients():
    with pytest.raises(ValueError):
        physics.BounceModel(restitution=1.5)
    with pytest.raises(ValueError):
        physics.BounceModel(retain_x=0.0)


def test_sampled_trajectory_matches_piecewise_for_the_same_ball():
    """The drag-free sampler and the analytic model agree with drag disabled."""
    delivery = physics.preset("good_length_middle")
    traj = physics.simulate_delivery(delivery)
    t = np.linspace(0, float(traj.bounce_times[0]) * 0.9, 20)
    r = traj.position(t)
    # Closed form: r0 + v0 t + 1/2 g t^2
    expected = delivery.release + np.outer(t, delivery.release_velocity) + 0.5 * np.outer(t**2, physics.G_VEC)
    assert np.allclose(r, expected, atol=1e-9)


def test_noiseless_ground_truth_bounce_matches_the_analytic_root():
    truth = build_ground_truth(physics.preset("good_length_middle"))
    t_b = truth.bounce_time
    z = float(truth.trajectory.position(t_b)[2])
    assert z == pytest.approx(BALL_RADIUS, abs=1e-6)
